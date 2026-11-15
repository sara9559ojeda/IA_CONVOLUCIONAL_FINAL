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
