"""Pruebas de la codificación de cajas, la decodificación y el NMS (src/modelo.py).

Un error aquí (por ejemplo, intercambiar filas y columnas) no da ningún
mensaje: la red simplemente aprende mal y se pierde un entrenamiento de horas.
"""
import numpy as np
import pytest

pytest.importorskip("tensorflow")

from src.modelo import (codificar_objetivos, decodificar, iou_matriz,  # noqa: E402
                        metricas_perdida, nms, perdida_detector)

GRID, NUM_CLASES = 20, 2
ANCLAS = np.array([[0.08, 0.10], [0.25, 0.30], [0.60, 0.55]], np.float32)
HUMO, FUEGO = 0, 1
RELLENO = [-1.0, 0.0, 0.0, 0.0, 0.0]


def codificar(cajas):
    return codificar_objetivos(np.asarray(cajas, np.float32), ANCLAS, GRID, NUM_CLASES).numpy()


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def salida_perfecta(objetivo):
    """Salida cruda que daría una red que acierta exactamente el objetivo."""
    hay = objetivo[..., 4] > 0
    pred = np.full(objetivo.shape, -12.0, np.float32)
    pred[..., 0:2] = logit(objetivo[..., 0:2])
    pred[..., 2:4] = np.where(hay[..., None],
                              np.log(np.maximum(objetivo[..., 2:4], 1e-6) / ANCLAS), 0.0)
    pred[..., 4] = np.where(hay, 12.0, -12.0)
    pred[..., 5:] = np.where(objetivo[..., 5:] > 0, 12.0, -12.0)
    return pred


def pred_vacia():
    return np.full((GRID, GRID, len(ANCLAS), 5 + NUM_CLASES), -12.0, np.float32)


def poner(pred, fila, col, ancla, clase, logit_obj=12.0):
    """Una caja del tamaño del ancla, centrada en la celda (fila, col)."""
    pred[fila, col, ancla, 0:4] = 0.0
    pred[fila, col, ancla, 4] = logit_obj
    pred[fila, col, ancla, 5 + clase] = 12.0


def xyxy(cx, cy, w, h):
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])


# ==========================================================================
# codificar_objetivos
# ==========================================================================
def test_codificar_ubica_la_caja_en_su_celda_y_su_ancla():
    objetivo = codificar([[FUEGO, 0.83, 0.22, 0.24, 0.31], RELLENO])

    assert objetivo.shape == (GRID, GRID, len(ANCLAS), 5 + NUM_CLASES)
    assert objetivo[..., 4].sum() == 1                  # una sola caja
    # cy = 0.22 -> fila 4 ; cx = 0.83 -> columna 16 ; forma parecida al ancla 1
    celda = objetivo[4, 16, 1]
    np.testing.assert_allclose(celda[0:2], [0.6, 0.4], atol=1e-5)     # desplazamiento en la celda
    np.testing.assert_allclose(celda[2:4], [0.24, 0.31], atol=1e-6)   # tamaño
    np.testing.assert_array_equal(celda[4:], [1, 0, 1])                # objeto, no humo, fuego


@pytest.mark.parametrize("w, h, ancla", [(0.07, 0.11, 0), (0.24, 0.31, 1), (0.62, 0.50, 2)])
def test_codificar_elige_el_ancla_de_forma_mas_parecida(w, h, ancla):
    objetivo = codificar([[HUMO, 0.52, 0.52, w, h]])
    assert objetivo[10, 10, ancla, 4] == 1
    assert objetivo[..., 4].sum() == 1


def test_codificar_ignora_el_relleno_y_las_cajas_sin_tamano():
    objetivo = codificar([RELLENO, [FUEGO, 0.5, 0.5, 0.0, 0.2], [HUMO, 0.5, 0.5, 0.2, 0.0]])
    assert not objetivo.any()


# ==========================================================================
# codificar -> decodificar
# ==========================================================================
def test_ida_y_vuelta_recupera_las_cajas_originales():
    cajas = [[HUMO, 0.52, 0.40, 0.62, 0.50],
             [FUEGO, 0.83, 0.22, 0.07, 0.11],
             [FUEGO, 0.21, 0.77, 0.24, 0.31],
             RELLENO, RELLENO]
    detecciones = decodificar(salida_perfecta(codificar(cajas)), ANCLAS, umbral=0.5)

    assert len(detecciones) == 3
    for clase, cx, cy, w, h in cajas[:3]:
        iguales = [d for d in detecciones
                   if d[0] == clase and np.allclose(d[2], xyxy(cx, cy, w, h), atol=1e-4)]
        assert len(iguales) == 1, f"no se recuperó la caja {(clase, cx, cy, w, h)}"
        assert iguales[0][1] > 0.99


# ==========================================================================
# decodificar
# ==========================================================================
def test_decodificar_sin_objetos_no_devuelve_nada():
    assert decodificar(pred_vacia(), ANCLAS) == []


def test_decodificar_respeta_el_umbral_de_confianza():
    pred = pred_vacia()
    poner(pred, 5, 5, 1, FUEGO, logit_obj=0.0)          # confianza = 0.5 * ~1
    assert decodificar(pred, ANCLAS, umbral=0.6) == []
    (clase, confianza, _), = decodificar(pred, ANCLAS, umbral=0.4)
    assert clase == FUEGO
    assert confianza == pytest.approx(0.5, abs=1e-3)


def test_decodificar_admite_un_umbral_por_clase():
    pred = pred_vacia()
    poner(pred, 3, 3, 1, HUMO, logit_obj=-1.0)           # confianza ~0.27 en las dos
    poner(pred, 15, 15, 1, FUEGO, logit_obj=-1.0)

    def clases(umbral):
        return sorted(d[0] for d in decodificar(pred, ANCLAS, umbral=umbral))

    assert clases(0.35) == []
    assert clases(0.20) == [HUMO, FUEGO]
    assert clases([0.20, 0.35]) == [HUMO]                # orden de las clases: humo, fuego
    assert clases([0.35, 0.20]) == [FUEGO]


def test_decodificar_convierte_celda_y_ancla_en_coordenadas():
    pred = pred_vacia()
    poner(pred, 4, 16, 1, FUEGO)                         # fila 4, columna 16
    (_, _, caja), = decodificar(pred, ANCLAS)
    np.testing.assert_allclose(caja, xyxy(16.5 / GRID, 4.5 / GRID, 0.25, 0.30), atol=1e-5)


def test_decodificar_ordena_por_confianza():
    pred = pred_vacia()
    poner(pred, 2, 2, 0, FUEGO, logit_obj=1.0)
    poner(pred, 15, 15, 0, HUMO, logit_obj=3.0)
    confianzas = [d[1] for d in decodificar(pred, ANCLAS, umbral=0.1)]
    assert confianzas == sorted(confianzas, reverse=True)
    assert len(confianzas) == 2


def test_decodificar_elimina_la_caja_repetida_de_la_misma_clase():
    pred = pred_vacia()
    poner(pred, 10, 10, 2, HUMO, logit_obj=12.0)         # dos celdas vecinas predicen
    poner(pred, 10, 11, 2, HUMO, logit_obj=2.0)          # casi la misma caja grande
    (_, confianza, _), = decodificar(pred, ANCLAS, umbral=0.3)
    assert confianza > 0.99                              # queda la más confiable


def test_decodificar_no_mezcla_clases_en_el_nms():
    pred = pred_vacia()
    poner(pred, 10, 10, 2, HUMO)
    poner(pred, 10, 11, 2, FUEGO)                        # se solapan, pero son clases distintas
    assert sorted(d[0] for d in decodificar(pred, ANCLAS)) == [HUMO, FUEGO]


# ==========================================================================
# nms / iou_matriz
# ==========================================================================
CAJAS_NMS = np.array([[0.00, 0.00, 0.50, 0.50],
                      [0.02, 0.02, 0.52, 0.52],          # casi igual a la primera
                      [0.60, 0.60, 0.90, 0.90]])
PUNTAJES_NMS = np.array([0.8, 0.9, 0.7])


def test_nms_elimina_la_caja_solapada_de_menor_puntaje():
    assert [int(k) for k in nms(CAJAS_NMS, PUNTAJES_NMS, 0.45)] == [1, 2]


def test_nms_conserva_todas_si_el_umbral_es_alto():
    assert [int(k) for k in nms(CAJAS_NMS, PUNTAJES_NMS, 0.95)] == [1, 0, 2]


def test_iou_matriz():
    a = np.array([[0.0, 0.0, 1.0, 1.0]])
    b = np.array([[0.0, 0.0, 1.0, 1.0],     # idéntica
                  [0.5, 0.0, 1.5, 1.0],     # se solapa la mitad -> 0.5 / 1.5
                  [2.0, 2.0, 3.0, 3.0]])    # separada
    np.testing.assert_allclose(iou_matriz(a, b), [[1.0, 1 / 3, 0.0]], atol=1e-6)


def test_iou_matriz_sin_cajas():
    assert iou_matriz(np.zeros((0, 4)), CAJAS_NMS).shape == (0, 3)
    assert iou_matriz(CAJAS_NMS, np.zeros((0, 4))).shape == (3, 0)


# ==========================================================================
# pérdida: debe usar la misma parametrización que decodificar
# ==========================================================================
def test_la_perdida_es_menor_con_la_salida_correcta():
    objetivo = codificar([[HUMO, 0.52, 0.40, 0.62, 0.50], [FUEGO, 0.83, 0.22, 0.07, 0.11]])
    perdida = perdida_detector(ANCLAS)

    def valor(pred):
        return float(perdida(objetivo[None], pred[None]))

    correcta = salida_perfecta(objetivo)
    desplazada = np.roll(correcta, 3, axis=1)            # mismas cajas, 3 celdas a la derecha
    tamano_malo = correcta.copy()
    tamano_malo[..., 2:4] += 1.0                         # cajas e = 2.7 veces más grandes

    assert valor(correcta) < valor(tamano_malo)
    assert valor(correcta) < valor(desplazada)
    assert valor(correcta) < valor(pred_vacia())


# ==========================================================================
# pérdida: zona de "ignorar" y términos por separado
# ==========================================================================
CAJA_HUMO = [HUMO, 0.52, 0.40, 0.62, 0.50]               # celda (fila 8, col 10), ancla 2


def valor_perdida(objetivo, pred, **opciones):
    return float(perdida_detector(ANCLAS, **opciones)(objetivo[None], pred[None]))


def con_caja_segura(pred, fila, col, ancla, cx, cy, w, h):
    """Añade una predicción muy segura (objetividad ~1) de la caja dada."""
    pred = pred.copy()
    pred[fila, col, ancla, 0:2] = logit(np.array([cx * GRID - col, cy * GRID - fila]))
    pred[fila, col, ancla, 2:4] = np.log(np.array([w, h]) / ANCLAS[ancla])
    pred[fila, col, ancla, 4] = 12.0
    return pred


def test_la_perdida_no_castiga_a_la_celda_vecina_que_predice_la_misma_caja():
    objetivo = codificar([CAJA_HUMO])
    correcta = salida_perfecta(objetivo)
    # La celda de al lado predice casi el mismo humo (IoU ~0.8 con la caja real)
    vecina = con_caja_segura(correcta, 8, 11, 2, 0.57, 0.42, 0.62, 0.50)

    assert valor_perdida(objetivo, vecina) == pytest.approx(valor_perdida(objetivo, correcta),
                                                            abs=1e-3)
    # Sin la zona de ignorar (comportamiento anterior) sí se castigaba: 0.5 * ~12
    sin_ignorar = valor_perdida(objetivo, vecina, umbral_ignorar=None)
    assert sin_ignorar > valor_perdida(objetivo, correcta, umbral_ignorar=None) + 5


def test_la_perdida_sigue_castigando_las_falsas_alarmas():
    objetivo = codificar([CAJA_HUMO])
    correcta = salida_perfecta(objetivo)
    lejos = con_caja_segura(correcta, 2, 2, 0, 0.12, 0.12, 0.08, 0.10)      # no toca el humo
    diminuta = con_caja_segura(correcta, 8, 11, 0, 0.57, 0.42, 0.08, 0.10)  # dentro, pero IoU ~0.03
    assert valor_perdida(objetivo, lejos) > valor_perdida(objetivo, correcta) + 5
    assert valor_perdida(objetivo, diminuta) > valor_perdida(objetivo, correcta) + 5

    # En una imagen sin objetos no hay nada que ignorar
    vacio = codificar([RELLENO])
    falsa = con_caja_segura(pred_vacia(), 8, 10, 2, 0.52, 0.40, 0.62, 0.50)
    assert valor_perdida(vacio, falsa) > valor_perdida(vacio, pred_vacia()) + 5


def test_los_terminos_de_la_perdida_suman_la_perdida_total():
    objetivo = codificar([CAJA_HUMO, [FUEGO, 0.83, 0.22, 0.07, 0.11]])
    rng = np.random.default_rng(0)
    y_true = np.stack([objetivo, codificar([RELLENO])])                    # lote de 2
    y_pred = rng.normal(0, 2, y_true.shape).astype(np.float32)

    metricas = metricas_perdida(ANCLAS)
    assert [m.__name__ for m in metricas] == ["perd_caja", "perd_obj", "perd_noobj", "perd_clase"]
    terminos = [float(m(y_true, y_pred)) for m in metricas]
    assert all(t > 0 for t in terminos)
    assert sum(terminos) == pytest.approx(float(perdida_detector(ANCLAS)(y_true, y_pred)), rel=1e-5)
