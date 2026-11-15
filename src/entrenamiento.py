"""Funciones para preparar D-Fire, entrenar, evaluar y exportar el detector.

Se usan desde `notebooks/entrenamiento_detector_colab.ipynb`.
"""
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from .modelo import (codificar_objetivos, construir_detector, decodificar,
                     iou_matriz, metricas_perdida, perdida_detector)

AUTOTUNE = tf.data.AUTOTUNE
EXT_IMG = {".jpg", ".jpeg", ".png", ".bmp"}
NOMBRES_SPLIT = {"train": "train", "training": "train", "val": "val", "valid": "val",
                 "validation": "val", "test": "test"}
COLORES_CLASE = ["#9e9e9e", "#ff5722"]   # humo gris, fuego naranja


# ==========================================================================
# 1. Lectura del dataset (formato YOLO: images/xxx.jpg + labels/xxx.txt)
# ==========================================================================
def leer_etiquetas(ruta):
    """Lee un .txt YOLO: 'clase cx cy w h' por línea. Sin archivo = imagen sin objetos."""
    cajas = []
    if ruta and os.path.exists(ruta):
        with open(ruta) as f:
            for linea in f:
                p = linea.split()
                if len(p) < 5:
                    continue
                try:
                    cajas.append([float(v) for v in p[:5]])
                except ValueError:
                    continue
    return np.asarray(cajas, np.float32).reshape(-1, 5)


def buscar_dataset(raiz, frac_val=0.1, frac_test=0.1, semilla=42):
    """Recorre la carpeta del dataset y arma los splits train / val / test.

    Funciona con estructuras 'train/images' y 'images/train'. Si no existe
    split de validación o prueba, lo separa del entrenamiento.
    """
    raiz = Path(raiz)
    pares = {"train": [], "val": [], "test": []}
    for img in raiz.rglob("*"):
        if img.suffix.lower() not in EXT_IMG or "__MACOSX" in img.parts:
            continue
        partes = list(img.relative_to(raiz).parts)
        idx = [i for i, p in enumerate(partes[:-1]) if p.lower() == "images"]
        if not idx:
            continue
        partes_lbl = partes.copy()
        partes_lbl[idx[-1]] = "labels"
        partes_lbl[-1] = img.stem + ".txt"
        split = next((NOMBRES_SPLIT[p.lower()] for p in partes[:-1]
                      if p.lower() in NOMBRES_SPLIT), "train")
        pares[split].append((str(img), str(raiz.joinpath(*partes_lbl))))

    rng = random.Random(semilla)
    rng.shuffle(pares["train"])
    for nombre, frac in (("val", frac_val), ("test", frac_test)):
        if not pares[nombre]:
            n = int(len(pares["train"]) * frac)
            pares[nombre], pares["train"] = pares["train"][:n], pares["train"][n:]

    datos = {}
    for nombre, lista in pares.items():
        lista.sort()
        datos[nombre] = {"rutas": [p[0] for p in lista],
                         "etiquetas": [p[1] for p in lista],
                         "cajas": [leer_etiquetas(p[1]) for p in lista]}
    return datos


def resumen_dataset(datos, clases):
    filas = []
    for nombre, split in datos.items():
        todas = np.concatenate(split["cajas"]) if split["cajas"] else np.zeros((0, 5))
        fila = {"split": nombre, "imagenes": len(split["rutas"]),
                "sin_objetos": sum(len(c) == 0 for c in split["cajas"])}
        for i, c in enumerate(clases):
            fila[f"cajas_{c}"] = int((todas[:, 0] == i).sum())
        filas.append(fila)
    return pd.DataFrame(filas)


def mostrar_ejemplos(split, clases, n=6, semilla=0):
    """Dibuja imágenes con sus cajas reales. Sirve para verificar que el orden
    de las clases es correcto (humo = 0, fuego = 1 en D-Fire)."""
    con_obj = [i for i, c in enumerate(split["cajas"]) if len(c)]
    random.Random(semilla).shuffle(con_obj)
    fig, ejes = plt.subplots(2, (n + 1) // 2, figsize=(16, 7))
    for eje, i in zip(ejes.ravel(), con_obj[:n]):
        img = plt.imread(split["rutas"][i])
        alto, ancho = img.shape[:2]
        eje.imshow(img)
        for c, cx, cy, w, h in split["cajas"][i]:
            c = int(c)
            eje.add_patch(plt.Rectangle(((cx - w / 2) * ancho, (cy - h / 2) * alto), w * ancho, h * alto,
                                        fill=False, color=COLORES_CLASE[c % 2], lw=2))
            eje.text((cx - w / 2) * ancho, (cy - h / 2) * alto, clases[c], color="white",
                     backgroundcolor=COLORES_CLASE[c % 2], fontsize=8)
        eje.axis("off")
    plt.tight_layout()
    plt.show()


# ==========================================================================
# 2. Anclas por k-means (tamaños típicos de las cajas del dataset)
# ==========================================================================
def calcular_anclas(split, k=3, iteraciones=100, semilla=0):
    wh = np.concatenate([c[:, 3:5] for c in split["cajas"] if len(c)])
    wh = wh[(wh[:, 0] > 0) & (wh[:, 1] > 0)]
    rng = np.random.default_rng(semilla)
    centros = wh[rng.choice(len(wh), k, replace=False)]

    def iou(wh, centros):
        inter = (np.minimum(wh[:, None, 0], centros[None, :, 0]) *
                 np.minimum(wh[:, None, 1], centros[None, :, 1]))
        return inter / (wh[:, :1] * wh[:, 1:2] + (centros[:, 0] * centros[:, 1])[None] - inter)

    for _ in range(iteraciones):
        asignacion = iou(wh, centros).argmax(1)
        nuevos = np.array([np.median(wh[asignacion == j], 0) if (asignacion == j).any()
                           else centros[j] for j in range(k)])
        if np.allclose(nuevos, centros):
            break
        centros = nuevos
    centros = centros[np.argsort(centros[:, 0] * centros[:, 1])]
    print(f"IoU medio caja-ancla: {iou(wh, centros).max(1).mean():.3f}")
    return centros.astype(np.float32)


# ==========================================================================
# 3. Pipeline tf.data con aumento de datos
# ==========================================================================
def leer_imagen(ruta, tam):
    img = tf.io.decode_image(tf.io.read_file(ruta), channels=3, expand_animations=False)
    img = tf.image.resize(img, (tam, tam))           # float32 0..255
    img.set_shape([tam, tam, 3])
    return img


def _recorte_aleatorio(img, cajas, tam):
    """Zoom aleatorio (recorta 70-100 % de la imagen) y ajusta las cajas."""
    s = tf.random.uniform((), 0.7, 1.0)
    ox = tf.random.uniform((), 0.0, 1.0 - s)
    oy = tf.random.uniform((), 0.0, 1.0 - s)
    img = tf.image.crop_and_resize(img[None], [[oy, ox, oy + s, ox + s]], [0], [tam, tam])[0]
    cls, cx, cy, w, h = [cajas[:, k] for k in range(5)]
    x1 = tf.clip_by_value((cx - w / 2 - ox) / s, 0.0, 1.0)
    x2 = tf.clip_by_value((cx + w / 2 - ox) / s, 0.0, 1.0)
    y1 = tf.clip_by_value((cy - h / 2 - oy) / s, 0.0, 1.0)
    y2 = tf.clip_by_value((cy + h / 2 - oy) / s, 0.0, 1.0)
    nw, nh = x2 - x1, y2 - y1
    valida = (cls >= 0) & (nw > 0.01) & (nh > 0.01)
    cajas = tf.stack([tf.where(valida, cls, -1.0), (x1 + x2) / 2, (y1 + y2) / 2, nw, nh], axis=1)
    return img, cajas


def aumentar_ejemplo(img, cajas, tam):
    img, cajas = tf.cond(tf.random.uniform(()) < 0.5,
                         lambda: _recorte_aleatorio(img, cajas, tam),
                         lambda: (img, cajas))
    voltear = tf.random.uniform(()) < 0.5
    img = tf.cond(voltear, lambda: tf.image.flip_left_right(img), lambda: img)
    cx = tf.where(voltear, 1.0 - cajas[:, 1], cajas[:, 1])
    cajas = tf.concat([cajas[:, :1], cx[:, None], cajas[:, 2:]], axis=1)

    img = img / 255.0
    img = tf.image.random_brightness(img, 0.15)
    img = tf.image.random_contrast(img, 0.8, 1.2)
    img = tf.image.random_saturation(img, 0.7, 1.3)
    img = tf.image.random_hue(img, 0.02)
    return tf.clip_by_value(img, 0.0, 1.0) * 255.0, cajas


def crear_dataset(split, anclas, tam, num_clases, lote=16, aumentar=False,
                  barajar=False, max_cajas=100):
    grid = tam // 16
    n = len(split["rutas"])
    cajas = np.full((n, max_cajas, 5), -1.0, np.float32)
    for i, c in enumerate(split["cajas"]):
        c = c[:max_cajas]
        cajas[i, :len(c)] = c
    anclas = np.asarray(anclas, np.float32)

    def preparar(ruta, cj):
        img = leer_imagen(ruta, tam)
        if aumentar:
            img, cj = aumentar_ejemplo(img, cj, tam)
        return img, codificar_objetivos(cj, anclas, grid, num_clases)

    ds = tf.data.Dataset.from_tensor_slices((split["rutas"], cajas))
    if barajar:
        ds = ds.shuffle(n, reshuffle_each_iteration=True)
    ds = ds.map(preparar, num_parallel_calls=AUTOTUNE)
    return ds.batch(lote, drop_remainder=aumentar).prefetch(AUTOTUNE)


# ==========================================================================
# 4. Entrenamiento (con reanudación si Colab se desconecta)
# ==========================================================================
class MetricasValidacion(keras.callbacks.Callback):
    """Al final de cada época mide la detección real en validación (mAP50, AP50
    por clase, precisión y recall por imagen) y la añade a los registros de la
    época con el prefijo `val_`. Así el mejor modelo se puede elegir por lo que
    de verdad importa y no por `val_loss`, que casi no cambia entre épocas.

    Debe ir ANTES de ModelCheckpoint, CSVLogger y EarlyStopping en la lista."""

    def __init__(self, split, anclas, clases, tam, umbral=0.35, lote=32):
        super().__init__()
        self.anclas, self.clases = np.asarray(anclas, np.float32), list(clases)
        self.umbrales = np.broadcast_to(np.asarray(umbral, np.float32), (len(self.clases),))
        self.ds = _dataset_imagenes(split, tam, lote)
        self.gts = _cajas_reales(split)

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.ds, verbose=0)
        todas = [decodificar(p, self.anclas, umbral=0.01) for p in preds]
        resultados, _ = calcular_metricas(todas, self.gts, self.clases, self.umbrales)
        if logs is not None:
            logs.update({f"val_{k}": v for k, v in resultados.items()})


def entrenar(ds_train, ds_val, anclas, num_clases, tam, dir_salida, epocas=60,
             lr_max=1e-3, epocas_calentamiento=3, paciencia=12,
             split_val=None, clases=None, umbral=0.35, monitor=None):
    """Entrena el detector y devuelve la ruta del mejor modelo.

    Si se pasan `split_val` (p. ej. datos['val']) y `clases`, cada época se mide
    la detección en validación y el mejor modelo y la parada temprana usan
    `monitor` (por defecto `val_mAP50`; también sirven `val_AP50_humo`,
    `val_recall_imagen_humo`, ...). `umbral` es el de las métricas por imagen.
    Sin ellos se usa `val_loss`, como antes.
    """
    dir_salida = Path(dir_salida)
    dir_salida.mkdir(parents=True, exist_ok=True)
    ruta_ultimo = dir_salida / "ultimo.keras"
    ruta_mejor = dir_salida / "mejor.keras"
    ruta_csv = dir_salida / "historial.csv"

    medir = split_val is not None and clases is not None
    monitor = monitor or ("val_mAP50" if medir else "val_loss")
    modo = "min" if monitor.endswith("loss") or "perd_" in monitor else "max"

    epoca_inicial, mejor_val = 0, None
    if ruta_ultimo.exists() and ruta_csv.exists():
        hist = pd.read_csv(ruta_csv)
        faltan = {monitor, "perd_noobj"} - set(hist.columns)
        if faltan:
            raise ValueError(
                f"'{dir_salida}' contiene un entrenamiento anterior hecho con otra "
                f"pérdida o sin estas métricas: {sorted(faltan)}. No se puede reanudar "
                "desde ahí: usa otra carpeta de salida, o borra ultimo.keras, "
                "mejor.keras e historial.csv para empezar de cero.")
        modelo = keras.models.load_model(ruta_ultimo, compile=False)
        epoca_inicial = len(hist)
        mejor_val = float(hist[monitor].min() if modo == "min" else hist[monitor].max())
        print(f"Reanudando desde la época {epoca_inicial} (mejor {monitor} = {mejor_val:.3f})")
    else:
        modelo = construir_detector(tam, len(anclas), num_clases)
        if ruta_csv.exists():
            ruta_csv.unlink()

    def tasa_aprendizaje(epoca, lr=None):
        if epoca < epocas_calentamiento:                       # calentamiento lineal
            return float(lr_max * (epoca + 1) / epocas_calentamiento)
        progreso = (epoca - epocas_calentamiento) / max(1, epocas - epocas_calentamiento)
        return float(lr_max * (0.02 + 0.98 * 0.5 * (1 + np.cos(np.pi * progreso))))  # coseno

    # Las métricas perd_* son los términos de la pérdida por separado (suman `loss`)
    modelo.compile(optimizer=keras.optimizers.Adam(lr_max, global_clipnorm=10.0),
                   loss=perdida_detector(anclas), metrics=metricas_perdida(anclas))
    callbacks = [keras.callbacks.LearningRateScheduler(tasa_aprendizaje)]
    if medir:   # antes de los callbacks que leen sus métricas
        callbacks.append(MetricasValidacion(split_val, anclas, clases, tam, umbral))
    callbacks += [
        keras.callbacks.ModelCheckpoint(str(ruta_ultimo)),
        keras.callbacks.ModelCheckpoint(str(ruta_mejor), monitor=monitor, mode=modo,
                                        save_best_only=True, initial_value_threshold=mejor_val),
        keras.callbacks.CSVLogger(str(ruta_csv), append=True),
        keras.callbacks.EarlyStopping(monitor=monitor, mode=modo, patience=paciencia,
                                      start_from_epoch=epocas_calentamiento),
    ]
    print(f"El mejor modelo se elige por {monitor} ({'menor' if modo == 'min' else 'mayor'} es mejor)")
    modelo.fit(ds_train, validation_data=ds_val, epochs=epocas,
               initial_epoch=epoca_inicial, callbacks=callbacks)
    return ruta_mejor


def graficar_historial(ruta_csv):
    """Pérdida, sus términos, la detección en validación y la tasa de aprendizaje.
    Solo dibuja los paneles cuyas columnas existan en el historial."""
    h = pd.read_csv(ruta_csv)
    col_lr = "learning_rate" if "learning_rate" in h else "lr"
    paneles = [
        ("Pérdida", {"entrenamiento": "loss", "validación": "val_loss"}),
        ("Términos de la pérdida (validación)",
         {c[len("val_perd_"):]: c for c in h.columns if c.startswith("val_perd_")}),
        ("Detección en validación",
         {c[len("val_"):]: c for c in h.columns
          if c == "val_mAP50" or c.startswith(("val_AP50_", "val_recall_imagen_"))}),
        ("Tasa de aprendizaje", {None: col_lr}),
    ]
    paneles = [(t, {e: c for e, c in cols.items() if c in h}) for t, cols in paneles]
    paneles = [(t, cols) for t, cols in paneles if cols]
    filas = (len(paneles) + 1) // 2
    fig, ejes = plt.subplots(filas, 2, figsize=(13, 4 * filas), squeeze=False)
    for eje, (titulo, cols) in zip(ejes.ravel(), paneles):
        for etiqueta, col in cols.items():
            eje.plot(h[col], label=etiqueta)
        eje.set_title(titulo)
        eje.set_xlabel("época")
        if None not in cols:
            eje.legend()
    for eje in ejes.ravel()[len(paneles):]:
        eje.axis("off")
    plt.tight_layout()
    plt.show()


# ==========================================================================
# 5. Exportar para la app (pesos en float32 + configuración)
# ==========================================================================
def exportar_modelo(ruta_mejor, anclas, clases, tam, dir_salida):
    """Guarda el modelo en float32 (el entrenamiento usa precisión mixta, que
    es lenta en CPU) junto con detector_config.json."""
    dir_salida = Path(dir_salida)
    dir_salida.mkdir(parents=True, exist_ok=True)
    keras.mixed_precision.set_global_policy("float32")
    entrenado = keras.models.load_model(ruta_mejor, compile=False)
    modelo = construir_detector(tam, len(anclas), len(clases))
    modelo.set_weights(entrenado.get_weights())
    ruta_modelo = dir_salida / "detector_fuego_humo.keras"
    modelo.save(ruta_modelo)
    cfg = {"tam_imagen": tam, "grid": tam // 16, "anclas": np.asarray(anclas).tolist(),
           "clases": list(clases), "tensorflow": tf.__version__, "keras": keras.__version__}
    with open(dir_salida / "detector_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print("Guardado:", ruta_modelo, "y", dir_salida / "detector_config.json")
    return modelo


# ==========================================================================
# 6. Evaluación
# ==========================================================================
def _dataset_imagenes(split, tam, lote=32):
    return (tf.data.Dataset.from_tensor_slices(split["rutas"])
            .map(lambda r: leer_imagen(r, tam), num_parallel_calls=AUTOTUNE)
            .batch(lote).prefetch(AUTOTUNE))


def predecir_split(modelo, split, tam, lote=32):
    return modelo.predict(_dataset_imagenes(split, tam, lote), verbose=1)


def _cxcywh_a_xyxy(c):
    return np.stack([c[:, 1] - c[:, 3] / 2, c[:, 2] - c[:, 4] / 2,
                     c[:, 1] + c[:, 3] / 2, c[:, 2] + c[:, 4] / 2], axis=1)


def _cajas_reales(split):
    """Por imagen: (clases, cajas x1 y1 x2 y2) de las etiquetas."""
    return [(c[:, 0].astype(int), _cxcywh_a_xyxy(c)) for c in split["cajas"]]


def _ap(recall, precision):
    """Average Precision (interpolación de todos los puntos, como en VOC)."""
    r = np.concatenate([[0.0], recall, [1.0]])
    p = np.concatenate([[0.0], precision, [0.0]])
    for i in range(len(p) - 2, -1, -1):
        p[i] = max(p[i], p[i + 1])
    idx = np.where(r[1:] != r[:-1])[0]
    return float(np.sum((r[idx + 1] - r[idx]) * p[idx + 1]))


NOMBRES_EVENTO = ["normal", "solo humo", "solo fuego", "fuego y humo"]


def calcular_metricas(todas, gts, clases, umbrales, iou_umbral=0.5):
    """AP@0.5 por clase (mAP) y métricas a nivel de imagen.

    todas   : por imagen, las detecciones de `decodificar` con umbral bajo (0.01).
    gts     : por imagen, (clases, cajas x1 y1 x2 y2) reales (`_cajas_reales`).
    umbrales: confianza mínima por clase para decir "esta imagen tiene humo/fuego".
    Retorna (resultados, matriz de confusión 4x4 de eventos por imagen)."""
    resultados = {}
    for k, nombre in enumerate(clases):
        dets = [(i, s, caja) for i, ds in enumerate(todas) for (c, s, caja) in ds if c == k]
        dets.sort(key=lambda d: d[1], reverse=True)
        gt_k = {i: g[1][g[0] == k] for i, g in enumerate(gts)}
        usados = {i: np.zeros(len(v), bool) for i, v in gt_k.items()}
        total_gt = sum(len(v) for v in gt_k.values())
        tp = np.zeros(len(dets))
        for n, (i, s, caja) in enumerate(dets):
            if len(gt_k[i]):
                ious = iou_matriz(caja[None], gt_k[i])[0]
                j = ious.argmax()
                if ious[j] >= iou_umbral and not usados[i][j]:
                    tp[n], usados[i][j] = 1, True
        acum_tp = np.cumsum(tp)
        recall = acum_tp / max(total_gt, 1)
        precision = acum_tp / np.arange(1, len(dets) + 1)
        resultados[f"AP50_{nombre}"] = _ap(recall, precision) if len(dets) else 0.0
    resultados["mAP50"] = float(np.mean([resultados[f"AP50_{c}"] for c in clases]))

    # ---- Nivel imagen: clasificación del evento ----
    i_humo, i_fuego = clases.index("humo"), clases.index("fuego")
    def categoria(tiene_humo, tiene_fuego):
        return 2 * int(tiene_fuego) + int(tiene_humo)
    y_real, y_pred = [], []
    for g, d in zip(gts, todas):
        y_real.append(categoria((g[0] == i_humo).any(), (g[0] == i_fuego).any()))
        conf = [c for c, s, _ in d if s >= umbrales[c]]
        y_pred.append(categoria(i_humo in conf, i_fuego in conf))
    y_real, y_pred = np.array(y_real), np.array(y_pred)
    matriz = np.zeros((4, 4), int)
    for a, b in zip(y_real, y_pred):
        matriz[a, b] += 1
    resultados["exactitud_evento_imagen"] = float((y_real == y_pred).mean())
    for nombre, bit in (("fuego", 2), ("humo", 1)):
        real = (y_real & bit) > 0
        pred = (y_pred & bit) > 0
        vp = (real & pred).sum()
        resultados[f"precision_imagen_{nombre}"] = float(vp / max(pred.sum(), 1))
        resultados[f"recall_imagen_{nombre}"] = float(vp / max(real.sum(), 1))
    return resultados, matriz


def evaluar(modelo, split, anclas, clases, tam, umbral=0.35, iou_umbral=0.5):
    """Calcula AP@0.5 por clase (mAP) y métricas a nivel de imagen
    (¿la imagen tiene fuego? ¿tiene humo?) con matriz de confusión de eventos.

    umbral: un número o una lista con un umbral por clase (config.umbrales_por_clase)."""
    umbrales = np.broadcast_to(np.asarray(umbral, np.float32), (len(clases),))
    preds = predecir_split(modelo, split, tam)
    todas = [decodificar(p, anclas, umbral=0.01) for p in preds]
    resultados, matriz = calcular_metricas(todas, _cajas_reales(split), clases, umbrales,
                                           iou_umbral)

    fig, eje = plt.subplots(figsize=(6, 5))
    eje.imshow(matriz, cmap="Oranges")
    for a in range(4):
        for b in range(4):
            eje.text(b, a, matriz[a, b], ha="center", va="center")
    eje.set_xticks(range(4), NOMBRES_EVENTO, rotation=20)
    eje.set_yticks(range(4), NOMBRES_EVENTO)
    eje.set_xlabel("predicción")
    eje.set_ylabel("real")
    texto_umbral = ", ".join(f"{c} {u:.2f}" for c, u in zip(clases, umbrales))
    eje.set_title(f"Eventos por imagen (umbral: {texto_umbral})")
    plt.tight_layout()
    plt.show()

    for k, v in resultados.items():
        print(f"{k:28s} {v:.3f}")
    return resultados, preds


def barrer_umbrales(todas, gts, clases, umbrales=None):
    """Qué pasaría a nivel de imagen con cada umbral de confianza, clase por clase.

    todas, gts: como en `calcular_metricas`. Retorna una tabla con una fila por
    (clase, umbral) y estas columnas:
      recall_imagen     de las imágenes que tienen la clase, en cuántas se detecta
      precision_imagen  de las imágenes donde se anuncia la clase, cuántas la tienen
                        (NaN si con ese umbral no se anuncia en ninguna)
      falsas_normales   fracción de las imágenes SIN ningún objeto donde se anuncia

    El umbral de una clase no afecta a las métricas de la otra, así que cada uno
    se puede elegir por separado."""
    if umbrales is None:
        umbrales = np.arange(1, 13) * 0.05                 # 0.05, 0.10, ..., 0.60
    maxima = np.zeros((len(todas), len(clases)))           # mayor confianza por imagen y clase
    for i, dets in enumerate(todas):
        for c, s, _ in dets:
            maxima[i, c] = max(maxima[i, c], s)
    tiene = np.array([[(g[0] == k).any() for k in range(len(clases))] for g in gts],
                     bool).reshape(len(gts), len(clases))
    normal = ~tiene.any(axis=1)
    filas = []
    for k, nombre in enumerate(clases):
        for u in umbrales:
            pred = maxima[:, k] >= u
            vp = (pred & tiene[:, k]).sum()
            filas.append({
                "clase": nombre, "umbral": round(float(u), 3),
                "recall_imagen": float(vp / max(tiene[:, k].sum(), 1)),
                "precision_imagen": float(vp / pred.sum()) if pred.any() else float("nan"),
                "falsas_normales": float((pred & normal).sum() / max(normal.sum(), 1)),
            })
    return pd.DataFrame(filas)


def explorar_umbrales(modelo, split, anclas, clases, tam, actuales=None, umbrales=None):
    """Dibuja e imprime `barrer_umbrales` sobre un conjunto (usa VALIDACIÓN: si
    los umbrales se eligen mirando el conjunto de prueba, sus métricas dejan de
    ser una medida limpia).

    actuales: umbrales en uso (config.umbrales_por_clase), se marcan en la gráfica.
    Retorna (tabla, predicciones)."""
    preds = predecir_split(modelo, split, tam)
    todas = [decodificar(p, anclas, umbral=0.01) for p in preds]
    tabla = barrer_umbrales(todas, _cajas_reales(split), clases, umbrales)

    series = {"recall_imagen": "recall (imágenes con la clase que se detectan)",
              "precision_imagen": "precisión (avisos que son ciertos)",
              "falsas_normales": "falsas alarmas en imágenes normales"}
    fig, ejes = plt.subplots(1, len(clases), figsize=(6.5 * len(clases), 4.5), squeeze=False)
    for k, (eje, nombre) in enumerate(zip(ejes[0], clases)):
        t = tabla[tabla["clase"] == nombre]
        for col, etiqueta in series.items():
            eje.plot(t["umbral"], t[col], marker="o", ms=3, label=etiqueta)
        if actuales is not None:
            eje.axvline(actuales[k], color="gray", ls="--", label=f"actual ({actuales[k]:.2f})")
        eje.set_title(f"{nombre}: efecto del umbral de confianza")
        eje.set_xlabel("umbral")
        eje.set_ylim(0, 1.02)
        eje.grid(alpha=0.3)
        eje.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    for nombre in clases:
        t = tabla[tabla["clase"] == nombre].drop(columns="clase")
        print(f"\n{nombre}")
        print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return tabla, preds


def mostrar_predicciones(preds, split, anclas, clases, umbral=0.35, n=8, semilla=1):
    indices = list(range(len(split["rutas"])))
    random.Random(semilla).shuffle(indices)
    fig, ejes = plt.subplots(2, (n + 1) // 2, figsize=(18, 8))
    for eje, i in zip(ejes.ravel(), indices[:n]):
        img = plt.imread(split["rutas"][i])
        alto, ancho = img.shape[:2]
        eje.imshow(img)
        for c, s, (x1, y1, x2, y2) in decodificar(preds[i], anclas, umbral):
            eje.add_patch(plt.Rectangle((x1 * ancho, y1 * alto), (x2 - x1) * ancho, (y2 - y1) * alto,
                                        fill=False, color="#7ce63c", lw=2))
            eje.text(x1 * ancho, y1 * alto, f"{clases[c]} ({s:.2f})", color="white",
                     backgroundcolor="#7ce63c", fontsize=8)
        eje.axis("off")
    plt.tight_layout()
    plt.show()
