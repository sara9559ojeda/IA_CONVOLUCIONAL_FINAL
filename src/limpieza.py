"""Limpieza y preparación del dataset D-Fire antes de entrenar.

Problemas que resuelve y por qué importan:

  1. Imágenes corruptas, vacías, diminutas, con relación de aspecto extrema o
     casi uniformes (fotogramas negros).
     -> Una imagen corrupta detiene `tf.data` a mitad del entrenamiento; las
        demás no aportan información útil.
  2. Etiquetas defectuosas: líneas mal formadas, clase inexistente,
     coordenadas fuera de 0..1, cajas diminutas (menos de 3 px a 320x320) y
     cajas repetidas.
     -> Ensucian la función de pérdida: la red aprende ruido.
  3. Imágenes duplicadas (D-Fire tiene muchos fotogramas consecutivos de video).
     -> Repetir la misma imagen no aporta y sesga el entrenamiento.
  4. Agrupa globalmente imágenes similares, incluidas conexiones transitivas.
  5. Divide grupos completos entre train/val/test con objetivo 70/15/15.
  6. Conserva negativos sin recortes posteriores que cambien las proporciones.
  7. Rendimiento: todas las imágenes se re-codifican una sola vez a 384x384.
     -> En Colab la CPU deja de ser el cuello de botella y cada época es
        mucho más rápida que decodificando las imágenes originales.

La similitud entre imágenes se mide con un "hash de diferencias" (dHash) de
256 bits: dos imágenes con hashes que difieren en pocos bits son casi iguales.

Todo queda documentado en un reporte (tabla + imágenes de ejemplo) para la
sección "Dataset" del informe.
"""
import json
import os
import random
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from . import config

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **_):
        return x

LADO_HASH = 16                # dHash de 16x16 = 256 bits
MIN_LADO_IMAGEN = 64          # px
DESV_MIN_UMBRAL = 3.0         # desviación estándar mínima del gris (imagen no uniforme)

DESCRIPCIONES = [
    ("imagenes_iniciales", "Imágenes encontradas en el dataset original"),
    ("img_corrupta", "Imágenes corruptas o ilegibles (eliminadas)"),
    ("img_pequena", f"Imágenes con lado menor a {MIN_LADO_IMAGEN} px (eliminadas)"),
    ("img_aspecto", "Imágenes con relación de aspecto extrema (eliminadas)"),
    ("img_uniforme", "Imágenes casi uniformes, p. ej. fotogramas negros (eliminadas)"),
    ("sin_archivo_etiquetas", "Imágenes sin archivo .txt (se tratan como sin objetos)"),
    ("cajas_iniciales", "Cajas (líneas) en las etiquetas originales"),
    ("lineas_mal_formadas", "Líneas de etiqueta mal formadas (eliminadas)"),
    ("cajas_clase_invalida", "Cajas con clase inexistente (eliminadas)"),
    ("cajas_recortadas", "Cajas que se salían de la imagen (recortadas a 0..1)"),
    ("cajas_diminutas", "Cajas demasiado pequeñas para la cuadrícula (eliminadas)"),
    ("cajas_duplicadas", "Cajas repetidas en la misma imagen (eliminadas)"),
    ("img_duplicadas", "Imágenes duplicadas o casi idénticas (eliminadas)"),
    ("cross_split_pairs", "Pares similares entre particiones (debe ser cero)"),
    ("grupos_similares", "Grupos de imágenes similares usados para dividir train/val/test"),
    ("grupo_mas_grande", "Tamaño del grupo de imágenes similares más grande"),
    ("negativas_descartadas", "Imágenes sin objetos descartadas para balancear el entrenamiento"),
    ("final_train", "Imágenes finales de entrenamiento"),
    ("final_val", "Imágenes finales de validación"),
    ("final_test", "Imágenes finales de prueba"),
    ("cajas_finales", "Cajas finales (todas las particiones)"),
]


# ==========================================================================
# Imágenes
# ==========================================================================
def _dhash(gris):
    r = cv2.resize(gris, (LADO_HASH + 1, LADO_HASH), interpolation=cv2.INTER_AREA)
    return (r[:, 1:] > r[:, :-1]).ravel()


def _procesar_imagen(tarea):
    """Valida, calcula el hash y guarda la imagen redimensionada.
    Retorna (estado, bits_hash)."""
    origen, destino, tam, max_aspecto = tarea
    try:
        crudo = np.fromfile(origen, np.uint8)
        img = cv2.imdecode(crudo, cv2.IMREAD_COLOR) if crudo.size else None
    except Exception:  # noqa: BLE001
        img = None
    if img is None:
        return "corrupta", None
    alto, ancho = img.shape[:2]
    if min(alto, ancho) < MIN_LADO_IMAGEN:
        return "pequena", None
    if max(alto, ancho) / min(alto, ancho) > max_aspecto:
        return "aspecto", None
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if float(gris.std()) < DESV_MIN_UMBRAL:
        return "uniforme", None
    reducida = cv2.resize(img, (tam, tam), interpolation=cv2.INTER_AREA)
    if not cv2.imwrite(destino, reducida, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        return "corrupta", None
    return "ok", _dhash(gris)


# ==========================================================================
# Etiquetas
# ==========================================================================
def _iou_xyxy(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def limpiar_etiquetas(ruta, num_clases, min_lado, conteo):
    """Lee y depura un .txt YOLO. Retorna array [n, 5] (clase, cx, cy, w, h)."""
    if not ruta or not os.path.exists(ruta):
        conteo["sin_archivo_etiquetas"] += 1
        return np.zeros((0, 5), np.float32)
    cajas = []          # (clase, x1, y1, x2, y2)
    with open(ruta, encoding="utf-8", errors="ignore") as f:
        for linea in f:
            if not linea.strip():
                continue
            conteo["cajas_iniciales"] += 1
            p = linea.split()
            try:
                c, cx, cy, w, h = (float(v) for v in p[:5])
                assert len(p) >= 5 and all(np.isfinite([c, cx, cy, w, h]))
            except (ValueError, AssertionError):
                conteo["lineas_mal_formadas"] += 1
                continue
            if c != int(c) or not 0 <= int(c) < num_clases:
                conteo["cajas_clase_invalida"] += 1
                continue
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            xc1, yc1, xc2, yc2 = (min(max(v, 0.0), 1.0) for v in (x1, y1, x2, y2))
            if (xc1, yc1, xc2, yc2) != (x1, y1, x2, y2):
                conteo["cajas_recortadas"] += 1
            if xc2 - xc1 < min_lado or yc2 - yc1 < min_lado:
                conteo["cajas_diminutas"] += 1
                continue
            nueva = (int(c), xc1, yc1, xc2, yc2)
            if any(o[0] == nueva[0] and _iou_xyxy(o[1:], nueva[1:]) > 0.9 for o in cajas):
                conteo["cajas_duplicadas"] += 1
                continue
            cajas.append(nueva)
    salida = [[c, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1] for c, x1, y1, x2, y2 in cajas]
    return np.asarray(salida, np.float32).reshape(-1, 5)


# ==========================================================================
# Similitud entre imágenes (distancia de Hamming entre hashes)
# ==========================================================================
def _pares(bits_a, bits_b, umbral, misma=False, bloque=2048):
    """Pares (i, j) cuya distancia de Hamming es <= umbral.
    Se calcula con productos de matrices por bloques (rápido y con poca RAM)."""
    if len(bits_a) == 0 or len(bits_b) == 0:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0, int)
    a_f = bits_a.astype(np.float32)
    b_t = bits_b.astype(np.float32).T
    nb_t = 1.0 - b_t
    n_bits = a_f.shape[1]
    ii, jj, dd = [], [], []
    for i0 in range(0, len(a_f), bloque):
        a = a_f[i0:i0 + bloque]
        dist = n_bits - (a @ b_t + (1.0 - a) @ nb_t)
        x, y = np.nonzero(dist <= umbral + 0.5)
        d = dist[x, y]
        x = x + i0
        if misma:
            m = y > x
            x, y, d = x[m], y[m], d[m]
        ii.append(x)
        jj.append(y)
        dd.append(d.astype(int))
    return np.concatenate(ii), np.concatenate(jj), np.concatenate(dd)


def _grupos(n, ii, jj):
    """Union-find: agrupa índices conectados por pares similares."""
    padre = list(range(n))

    def raiz(x):
        while padre[x] != x:
            padre[x] = padre[padre[x]]
            x = padre[x]
        return x

    for a, b in zip(ii.tolist(), jj.tolist()):
        ra, rb = raiz(a), raiz(b)
        if ra != rb:
            padre[max(ra, rb)] = min(ra, rb)
    grupos = defaultdict(list)
    for k in range(n):
        grupos[raiz(k)].append(k)
    return list(grupos.values())


def split_groups(groups, val_fraction=0.15, test_fraction=0.15, seed=42):
    """Allocate whole groups to image quotas; never drop samples to force ratios."""
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1
            and val_fraction + test_fraction < 1):
        raise ValueError("Invalid split fractions")
    if len(groups) < 3 or any(not group for group in groups):
        raise ValueError("Se necesitan al menos tres grupos independientes no vacíos.")
    groups = [list(group) for group in groups]
    random.Random(seed).shuffle(groups)
    groups.sort(key=len, reverse=True)
    total = sum(map(len, groups))
    targets = dict(zip(("train", "val", "test"),
                       (total*(1-val_fraction-test_fraction), total*val_fraction, total*test_fraction)))
    splits = {name: [] for name in targets}
    for index, group in enumerate(groups):
        empty = [name for name in splits if not splits[name]]
        candidates = empty if len(groups)-index == len(empty) else list(splits)
        name = max(candidates, key=lambda key: targets[key]-len(splits[key]))
        splits[name].extend(group)
    return {name: sorted(items) for name, items in splits.items()}


def _montaje(pares, titulo, ruta, max_pares=6, lado=320):
    """Guarda una imagen con pares lado a lado (para verificar los umbrales)."""
    filas = []
    for ruta_a, ruta_b, dist in pares[:max_pares]:
        a, b = cv2.imread(ruta_a), cv2.imread(ruta_b)
        if a is None or b is None:
            continue
        fila = np.hstack([cv2.resize(a, (lado, lado)), np.full((lado, 6, 3), 255, np.uint8),
                          cv2.resize(b, (lado, lado))])
        cv2.putText(fila, f"distancia {dist}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 255, 255), 2, cv2.LINE_AA)
        filas.append(fila)
    if not filas:
        return
    encabezado = np.zeros((34, filas[0].shape[1], 3), np.uint8)
    cv2.putText(encabezado, titulo, (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(ruta), np.vstack([encabezado] + filas))


# ==========================================================================
# Proceso completo
# ==========================================================================
def limpiar_dataset(datos, clases, dir_salida, tam_guardado=None, umbral_duplicado=None,
                    umbral_similar=None, frac_val=0.15, frac_negativas=None,
                    min_lado_caja_px=3, max_aspecto=4.0, hilos=8, semilla=42, frac_test=0.15):
    """Limpia el dataset que entrega `entrenamiento.buscar_dataset`.

    Retorna (datos_limpios, reporte) donde datos_limpios tiene la misma forma
    que `datos` ({split: {"rutas", "cajas"}}) pero apunta a las imágenes ya
    redimensionadas en `dir_salida/imagenes/`.
    """
    tam_guardado = tam_guardado or config.TAM_GUARDADO
    umbral_duplicado = config.UMBRAL_HASH_DUPLICADO if umbral_duplicado is None else umbral_duplicado
    umbral_similar = config.UMBRAL_HASH_SIMILAR if umbral_similar is None else umbral_similar
    if frac_negativas not in (None, 1.0):
        raise ValueError("No se recortan negativos: alteraría la partición 70/15/15.")
    if not (0 < frac_val < 1 and 0 < frac_test < 1 and frac_val + frac_test < 1):
        raise ValueError("Fracciones inválidas.")
    rng = random.Random(semilla)
    dir_salida = Path(dir_salida)
    dir_img = dir_salida / "imagenes"
    dir_img.mkdir(parents=True, exist_ok=True)
    conteo = Counter()

    # ---- 1. Registro único de todas las imágenes ----
    registros = []
    for split in ("train", "val", "test"):
        etiquetas = datos[split].get("etiquetas", [None] * len(datos[split]["rutas"]))
        for ruta, lbl in zip(datos[split]["rutas"], etiquetas):
            registros.append({"split": split, "origen": ruta, "etiqueta": lbl})
    registros.sort(key=lambda record: record["origen"])
    conteo["imagenes_iniciales"] = len(registros)
    print(f"1/6 Validando y redimensionando {len(registros)} imágenes a {tam_guardado}x{tam_guardado}...")

    # ---- 2. Validar imágenes, redimensionar y calcular hash (en paralelo) ----
    tareas = [(r["origen"], str(dir_img / f"{i:06d}.jpg"), tam_guardado, max_aspecto)
              for i, r in enumerate(registros)]
    with ThreadPoolExecutor(hilos) as ex:
        resultados = list(tqdm(ex.map(_procesar_imagen, tareas), total=len(tareas)))

    print("2/6 Depurando etiquetas...")
    validos = []
    min_lado = min_lado_caja_px / config.TAM_IMAGEN
    for r, tarea, (estado, bits) in zip(registros, tareas, resultados):
        if estado != "ok":
            conteo[f"img_{estado}"] += 1
            continue
        r.update(destino=tarea[1], bits=bits,
                 cajas=limpiar_etiquetas(r["etiqueta"], len(clases), min_lado, conteo))
        validos.append(r)
    if not validos:
        raise ValueError("No quedaron imágenes válidas.")
    bits = np.stack([r["bits"] for r in validos])

    # ---- 3. Duplicados casi exactos ----
    print("3/6 Buscando imágenes duplicadas...")
    ii, jj, _ = _pares(bits, bits, umbral_duplicado, misma=True)
    eliminar, ej_dup = set(), []
    for miembros in _grupos(len(validos), ii, jj):
        if len(miembros) < 2:
            continue
        # Se reconstruyen todas las particiones; conservar la mejor etiquetada.
        conservar = max(miembros, key=lambda k: len(validos[k]["cajas"]))
        for k in miembros:
            if k != conservar:
                eliminar.add(k)
                d = int(np.sum(bits[k] != bits[conservar]))
                ej_dup.append((validos[conservar]["destino"], validos[k]["destino"], d))
    conteo["img_duplicadas"] = len(eliminar)

    # Agrupar antes de retirar duplicados conserva los puentes A~B~C,
    # incluso cuando B se elimina por duplicación.
    print("4/6 Agrupando similitud global y conexiones transitivas...")
    ii, jj, distances = _pares(bits, bits, max(umbral_similar, umbral_duplicado), misma=True)
    grupos = [[k for k in group if k not in eliminar]
              for group in _grupos(len(validos), ii, jj)]
    grupos = [group for group in grupos if group]
    conteo["grupos_similares"] = len(grupos)
    conteo["grupo_mas_grande"] = max(map(len, grupos), default=0)
    group_ids = {k: group_id for group_id, group in enumerate(grupos) for k in group}
    print("5/6 Dividiendo grupos completos con objetivo 70/15/15...")
    particiones = split_groups(grupos, frac_val, frac_test, semilla)
    owners = {k: split for split, indices in particiones.items() for k in indices}
    cross_pairs = sum(owners[a] != owners[b] for a, b in zip(ii, jj)
                      if a in owners and b in owners)
    if cross_pairs:
        raise RuntimeError("Se detectaron pares similares entre particiones.")
    conteo["cross_split_pairs"] = cross_pairs
    ej_fuga = [(validos[a]["destino"], validos[b]["destino"], int(d))
               for a, b, d in zip(ii[:50], jj[:50], distances[:50])]
    print("6/6 Guardando sin recortar imágenes después de dividir...")

    # ---- Imágenes de ejemplo para el reporte (antes de borrar archivos) ----
    rng.shuffle(ej_dup)
    rng.shuffle(ej_fuga)
    _montaje(ej_dup, f"Duplicados eliminados (umbral {umbral_duplicado} de 256 bits)",
             dir_salida / "ejemplos_duplicados.jpg")
    _montaje(ej_fuga, f"Similitud agrupada (no cruza particiones), umbral {umbral_similar}",
             dir_salida / "ejemplos_fuga.jpg")

    # Borra del disco las imágenes descartadas (el zip queda más liviano)
    for k in eliminar:
        try:
            os.remove(validos[k]["destino"])
        except OSError:
            pass

    # ---- Manifiesto + reporte ----
    manifiesto = {"tam_guardado": tam_guardado, "clases": list(clases), "splits": {},
                  "partition_version": "grouped_70_15_15_v1", "seed": semilla,
                  "target_fractions": {"train": 1-frac_val-frac_test, "val": frac_val, "test": frac_test},
                  "actual_fractions": {name: len(items)/sum(map(len, particiones.values()))
                                       for name, items in particiones.items()},
                  "cross_split_similar_pairs": cross_pairs}
    for split, indices in particiones.items():
        manifiesto["splits"][split] = [
            {"img": os.path.relpath(validos[k]["destino"], dir_salida),
             "origen": validos[k]["origen"], "group_id": group_ids[k], "cajas": validos[k]["cajas"].round(6).tolist()}
            for k in indices]
        conteo[f"final_{split}"] = len(indices)
    conteo["cajas_finales"] = sum(len(validos[k]["cajas"]) for ind in particiones.values() for k in ind)
    reporte = pd.DataFrame([{"paso": d, "cantidad": int(conteo.get(k, 0))} for k, d in DESCRIPCIONES])
    with open(dir_salida / "manifiesto.json", "w", encoding="utf-8") as f:
        json.dump(manifiesto, f)
    reporte.to_csv(dir_salida / "reporte_limpieza.csv", index=False)
    for name, fraction in manifiesto["actual_fractions"].items():
        print(f"{name}: {len(particiones[name])} imágenes ({fraction:.2%})")
    print("Pares similares entre particiones: 0. Listo.")
    return cargar_manifiesto(dir_salida), reporte


# ==========================================================================
# Guardar / restaurar (para no repetir la limpieza si Colab se desconecta)
# ==========================================================================
def cargar_manifiesto(dir_salida):
    dir_salida = Path(dir_salida)
    with open(dir_salida / "manifiesto.json", encoding="utf-8") as f:
        manifiesto = json.load(f)
    datos = {}
    for split, items in manifiesto["splits"].items():
        datos[split] = {"rutas": [str(dir_salida / it["img"]) for it in items],
                        "cajas": [np.asarray(it["cajas"], np.float32).reshape(-1, 5) for it in items]}
    return datos


def cargar_reporte(dir_salida):
    return pd.read_csv(Path(dir_salida) / "reporte_limpieza.csv")


def empaquetar(dir_salida, ruta_zip):
    """Comprime el dataset limpio y lo copia a Drive."""
    base = "/content/_dfire_limpio" if os.path.isdir("/content") else str(Path(dir_salida).parent / "_dfire_limpio")
    archivo = shutil.make_archive(base, "zip", root_dir=dir_salida)
    Path(ruta_zip).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(archivo, ruta_zip)
    print("Dataset limpio guardado en", ruta_zip)


def restaurar(ruta_zip, dir_salida):
    """Descomprime el dataset limpio guardado en Drive (si aún no está)."""
    dir_salida = Path(dir_salida)
    if not (dir_salida / "manifiesto.json").exists():
        shutil.unpack_archive(str(ruta_zip), str(dir_salida))
    return cargar_manifiesto(dir_salida)


# ==========================================================================
# Estadísticas para el reporte
# ==========================================================================
def graficar_estadisticas(datos, clases):
    import matplotlib.pyplot as plt

    splits = list(datos)
    fig, ejes = plt.subplots(2, 2, figsize=(14, 9))

    con = [sum(len(c) > 0 for c in datos[s]["cajas"]) for s in splits]
    sin = [sum(len(c) == 0 for c in datos[s]["cajas"]) for s in splits]
    ejes[0, 0].bar(splits, con, label="con fuego/humo", color="#ff7043")
    ejes[0, 0].bar(splits, sin, bottom=con, label="sin objetos", color="#90a4ae")
    ejes[0, 0].set_title("Imágenes por partición")
    ejes[0, 0].legend()

    ancho = 0.8 / len(clases)
    for k, c in enumerate(clases):
        valores = [sum(int((cj[:, 0] == k).sum()) for cj in datos[s]["cajas"]) for s in splits]
        ejes[0, 1].bar(np.arange(len(splits)) + k * ancho, valores, ancho, label=c)
    ejes[0, 1].set_xticks(np.arange(len(splits)) + ancho * (len(clases) - 1) / 2, splits)
    ejes[0, 1].set_title("Cajas por clase")
    ejes[0, 1].legend()

    todas = np.concatenate([c for c in datos["train"]["cajas"] if len(c)])
    for k, c in enumerate(clases):
        areas = todas[todas[:, 0] == k][:, 3] * todas[todas[:, 0] == k][:, 4]
        ejes[1, 0].hist(np.log10(areas + 1e-6), bins=40, alpha=0.6, label=c)
    ejes[1, 0].set_title("Tamaño de las cajas (log10 del área relativa) · train")
    ejes[1, 0].set_xlabel("-4 = diminuta · 0 = toda la imagen")
    ejes[1, 0].legend()

    por_img = [len(c) for c in datos["train"]["cajas"]]
    ejes[1, 1].hist(por_img, bins=range(0, min(max(por_img) + 2, 30)), color="#5c6bc0")
    ejes[1, 1].set_title("Cajas por imagen · train")
    plt.tight_layout()
    plt.show()


def mostrar_imagen_reporte(ruta):
    import matplotlib.pyplot as plt

    img = cv2.imread(str(ruta))
    if img is None:
        print("No hay ejemplos en", ruta)
        return
    plt.figure(figsize=(8, img.shape[0] / img.shape[1] * 8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()
