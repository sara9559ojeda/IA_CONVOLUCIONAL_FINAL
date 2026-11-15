"""Pruebas de las métricas de detección (src/entrenamiento.py).

Con estas métricas se elige el mejor modelo durante el entrenamiento, así que
un error aquí guardaría el modelo equivocado sin dar ningún mensaje.
"""
import numpy as np
import pytest

pytest.importorskip("tensorflow")

from src.entrenamiento import barrer_umbrales, calcular_metricas  # noqa: E402

CLASES = ["humo", "fuego"]
HUMO, FUEGO = 0, 1
CAJA_A = np.array([0.10, 0.10, 0.50, 0.50])
CAJA_B = np.array([0.60, 0.60, 0.90, 0.90])
SIN_CAJAS = (np.zeros(0, int), np.zeros((0, 4)))


def reales(*cajas):
    """cajas: pares (clase, caja)."""
    return np.array([c for c, _ in cajas], int), np.array([b for _, b in cajas], float)


def test_detecciones_perfectas():
    gts = [reales((HUMO, CAJA_A)), reales((FUEGO, CAJA_B)),
           reales((HUMO, CAJA_A), (FUEGO, CAJA_B)), SIN_CAJAS]
    todas = [[(HUMO, 0.9, CAJA_A)], [(FUEGO, 0.8, CAJA_B)],
             [(HUMO, 0.7, CAJA_A), (FUEGO, 0.6, CAJA_B)], []]
    resultados, matriz = calcular_metricas(todas, gts, CLASES, [0.20, 0.35])

    assert resultados["AP50_humo"] == pytest.approx(1.0)
    assert resultados["AP50_fuego"] == pytest.approx(1.0)
    assert resultados["mAP50"] == pytest.approx(1.0)
    assert resultados["exactitud_evento_imagen"] == 1.0
    # filas/columnas: normal, solo humo, solo fuego, fuego y humo
    np.testing.assert_array_equal(matriz, np.eye(4, dtype=int))


def test_una_caja_sin_detectar_baja_el_ap_y_el_recall():
    gts = [reales((HUMO, CAJA_A)), reales((HUMO, CAJA_B))]
    todas = [[(HUMO, 0.9, CAJA_A)], []]                   # el segundo humo no se ve
    resultados, matriz = calcular_metricas(todas, gts, CLASES, [0.20, 0.35])

    assert resultados["AP50_humo"] == pytest.approx(0.5)
    assert resultados["recall_imagen_humo"] == pytest.approx(0.5)
    assert resultados["precision_imagen_humo"] == pytest.approx(1.0)
    assert matriz[1, 0] == 1                              # real "solo humo", predicho "normal"


def test_una_caja_mal_ubicada_no_cuenta_para_el_ap_pero_si_para_la_imagen():
    gts = [reales((FUEGO, CAJA_A))]
    todas = [[(FUEGO, 0.9, CAJA_B)]]                      # dice fuego, pero en otro sitio
    resultados, _ = calcular_metricas(todas, gts, CLASES, [0.20, 0.35])

    assert resultados["AP50_fuego"] == 0.0
    assert resultados["recall_imagen_fuego"] == 1.0


def test_las_metricas_por_imagen_usan_el_umbral_de_cada_clase():
    gts = [reales((HUMO, CAJA_A), (FUEGO, CAJA_B))]
    todas = [[(HUMO, 0.25, CAJA_A), (FUEGO, 0.25, CAJA_B)]]

    resultados, _ = calcular_metricas(todas, gts, CLASES, [0.20, 0.35])
    assert resultados["recall_imagen_humo"] == 1.0       # 0.25 >= 0.20
    assert resultados["recall_imagen_fuego"] == 0.0      # 0.25 <  0.35
    # El AP no depende del umbral: usa todas las detecciones
    assert resultados["AP50_fuego"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Barrido de umbrales (con él se eligen los umbrales de la app)
# --------------------------------------------------------------------------
def fila(tabla, clase, umbral):
    f = tabla[(tabla["clase"] == clase) & np.isclose(tabla["umbral"], umbral)]
    assert len(f) == 1
    return f.iloc[0]


def test_el_barrido_muestra_el_costo_de_bajar_el_umbral():
    # 2 imágenes con humo (confianzas 0.6 y 0.15) y 2 normales (una con un humo falso de 0.3)
    gts = [reales((HUMO, CAJA_A)), reales((HUMO, CAJA_B)), SIN_CAJAS, SIN_CAJAS]
    todas = [[(HUMO, 0.6, CAJA_A)], [(HUMO, 0.15, CAJA_B)], [(HUMO, 0.3, CAJA_A)], []]
    tabla = barrer_umbrales(todas, gts, CLASES, umbrales=[0.1, 0.2, 0.5, 0.9])

    bajo = fila(tabla, "humo", 0.1)             # se ve todo el humo, con una falsa alarma
    assert bajo["recall_imagen"] == pytest.approx(1.0)
    assert bajo["precision_imagen"] == pytest.approx(2 / 3)
    assert bajo["falsas_normales"] == pytest.approx(0.5)

    medio = fila(tabla, "humo", 0.2)            # se pierde el humo de 0.15
    assert medio["recall_imagen"] == pytest.approx(0.5)
    assert medio["precision_imagen"] == pytest.approx(0.5)
    assert medio["falsas_normales"] == pytest.approx(0.5)

    alto = fila(tabla, "humo", 0.5)             # sin falsas alarmas, pero medio humo perdido
    assert alto["recall_imagen"] == pytest.approx(0.5)
    assert alto["precision_imagen"] == pytest.approx(1.0)
    assert alto["falsas_normales"] == 0.0

    nada = fila(tabla, "humo", 0.9)             # no avisa nunca: la precisión no está definida
    assert nada["recall_imagen"] == 0.0
    assert np.isnan(nada["precision_imagen"])


def test_el_barrido_usa_la_caja_mas_segura_de_cada_imagen_y_separa_las_clases():
    gts = [reales((HUMO, CAJA_A), (FUEGO, CAJA_B))]
    todas = [[(HUMO, 0.1, CAJA_B), (HUMO, 0.4, CAJA_A), (FUEGO, 0.25, CAJA_B)]]
    tabla = barrer_umbrales(todas, gts, CLASES, umbrales=[0.3])

    assert fila(tabla, "humo", 0.3)["recall_imagen"] == 1.0      # cuenta la de 0.4
    assert fila(tabla, "fuego", 0.3)["recall_imagen"] == 0.0     # el humo no cuenta como fuego


def test_el_barrido_coincide_con_las_metricas_de_evaluar():
    rng = np.random.default_rng(0)
    gts, todas = [], []
    for _ in range(200):
        gts.append(reales(*[(c, CAJA_A) for c in (HUMO, FUEGO) if rng.random() < 0.4])
                   if rng.random() < 0.7 else SIN_CAJAS)
        todas.append([(int(rng.integers(2)), float(rng.random()), CAJA_A)
                      for _ in range(rng.integers(0, 4))])
    gts = [g if len(g[0]) else SIN_CAJAS for g in gts]
    tabla = barrer_umbrales(todas, gts, CLASES, umbrales=[0.2, 0.35])
    resultados, matriz = calcular_metricas(todas, gts, CLASES, [0.20, 0.35])

    for clase, umbral in (("humo", 0.2), ("fuego", 0.35)):
        f = fila(tabla, clase, umbral)
        assert f["recall_imagen"] == pytest.approx(resultados[f"recall_imagen_{clase}"])
        assert f["precision_imagen"] == pytest.approx(resultados[f"precision_imagen_{clase}"])
    # Falsas alarmas de humo en imágenes normales: columnas "solo humo" y "fuego y humo"
    assert fila(tabla, "humo", 0.2)["falsas_normales"] == pytest.approx(
        (matriz[0, 1] + matriz[0, 3]) / matriz[0].sum())


def test_el_barrido_por_defecto_va_de_005_a_060():
    tabla = barrer_umbrales([[]], [SIN_CAJAS], CLASES)
    umbrales = tabla[tabla["clase"] == "humo"]["umbral"].to_numpy()
    np.testing.assert_allclose(umbrales, np.arange(1, 13) * 0.05)


# --------------------------------------------------------------------------
# Anclas
# --------------------------------------------------------------------------
from src.entrenamiento import anclas_del_experimento, calcular_anclas  # noqa: E402
from src.modelo import (codificar_objetivos, construir_detector, decodificar,  # noqa: E402
                        perdida_detector)

TAMANOS = [(0.03, 0.04), (0.08, 0.20), (0.20, 0.10), (0.40, 0.45), (0.80, 0.70)]


def split_con_cajas(tamanos=TAMANOS, por_tamano=40, semilla=0):
    """Cajas de unos pocos tamaños bien separados, con un poco de variación."""
    rng = np.random.default_rng(semilla)
    cajas = []
    for w, h in tamanos:
        for _ in range(por_tamano):
            f = rng.uniform(0.95, 1.05, 2)
            cajas.append(np.array([[0, 0.5, 0.5, w * f[0], h * f[1]]], np.float32))
    return {"cajas": cajas}


def test_calcular_anclas_encuentra_los_tamanos_tipicos(capsys):
    anclas = calcular_anclas(split_con_cajas(), k=5)

    assert anclas.shape == (5, 2)
    areas = anclas[:, 0] * anclas[:, 1]
    assert (np.diff(areas) > 0).all()                      # ordenadas de pequeña a grande
    for w, h in TAMANOS:                                   # cada tamaño tiene un ancla cerca
        assert np.abs(anclas - [w, h]).max(axis=1).min() < 0.03
    assert "IoU medio caja-ancla: 0.9" in capsys.readouterr().out


def test_mas_anclas_no_empeoran_el_iou_medio(capsys):
    split = split_con_cajas()

    def iou_medio(k):
        calcular_anclas(split, k=k)
        return float(capsys.readouterr().out.split("caja-ancla: ")[1].split()[0])

    assert iou_medio(5) > iou_medio(3)


def test_varios_intentos_nunca_dan_peor_resultado_que_uno(capsys):
    split = split_con_cajas()

    def iou_medio(intentos):
        calcular_anclas(split, k=5, intentos=intentos)
        return float(capsys.readouterr().out.split("caja-ancla: ")[1].split()[0])

    assert iou_medio(10) >= iou_medio(1)


def test_las_anclas_se_guardan_y_se_reutilizan_por_experimento(tmp_path):
    primeras = np.array([[0.1, 0.1], [0.5, 0.5]], np.float32)
    otras = primeras * 2                                   # el k-means de otra sesión

    np.testing.assert_array_equal(anclas_del_experimento(tmp_path / "exp", primeras), primeras)
    np.testing.assert_array_equal(anclas_del_experimento(tmp_path / "exp", otras), primeras)


def test_un_experimento_anterior_recupera_las_anclas_compartidas(tmp_path):
    antiguas = np.array([[0.1, 0.1], [0.3, 0.3], [0.6, 0.6]], np.float32)
    np.save(tmp_path / "anclas.npy", antiguas)
    exp = tmp_path / "v2"
    exp.mkdir()
    (exp / "mejor.keras").write_bytes(b"")                 # ya se entrenó, sin anclas propias

    anclas = anclas_del_experimento(exp, antiguas * 1.1, ruta_antigua=tmp_path / "anclas.npy")
    np.testing.assert_array_equal(anclas, antiguas)
    assert (exp / "anclas.npy").exists()

    sin_anclas = tmp_path / "perdido"
    sin_anclas.mkdir()
    (sin_anclas / "ultimo.keras").write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        anclas_del_experimento(sin_anclas, antiguas)


def test_cambiar_el_numero_de_anclas_exige_otra_carpeta(tmp_path):
    tres = np.array([[0.1, 0.1], [0.3, 0.3], [0.6, 0.6]], np.float32)
    cinco = np.linspace(0.1, 0.8, 10, dtype=np.float32).reshape(5, 2)
    anclas_del_experimento(tmp_path, tres)

    with pytest.raises(ValueError, match="3 anclas"):
        anclas_del_experimento(tmp_path, cinco)


def test_el_detector_completo_funciona_con_cinco_anclas():
    anclas = calcular_anclas(split_con_cajas(), k=5)
    tam, grid = 64, 4
    modelo = construir_detector(tam, len(anclas), len(CLASES))
    assert modelo.output_shape == (None, grid, grid, 5, 5 + len(CLASES))

    cajas = np.array([[HUMO, 0.5, 0.5, 0.8, 0.7], [FUEGO, 0.2, 0.2, 0.03, 0.04]], np.float32)
    objetivo = codificar_objetivos(cajas, anclas, grid, len(CLASES)).numpy()
    assert objetivo[..., 4].sum() == 2
    assert objetivo[2, 2, 4, 4] == 1                       # el humo grande va al ancla mayor
    assert objetivo[0, 0, 0, 4] == 1                       # el fuego diminuto, a la menor

    pred = modelo(np.zeros((1, tam, tam, 3), np.float32), training=False).numpy()
    assert np.isfinite(float(perdida_detector(anclas)(objetivo[None], pred)))
    assert isinstance(decodificar(pred[0], anclas, umbral=0.01), list)
