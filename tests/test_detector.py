"""Pruebas de la inferencia (src/detector.py) con un modelo SIN entrenar.

No miden la calidad de las detecciones: comprueban que el modelo se guarda y
se carga en formato .keras, que la llamada compilada da lo mismo que la
directa y que las cajas salen en píxeles dentro del fotograma.
"""
import json

import numpy as np
import pytest

pytest.importorskip("tensorflow")
cv2 = pytest.importorskip("cv2")

from src.detector import DetectorFuegoHumo, clasificar_color_humo  # noqa: E402
from src.modelo import construir_detector  # noqa: E402

TAM = 320
ANCLAS = [[0.08, 0.10], [0.25, 0.30], [0.60, 0.55]]
CLASES = ["humo", "fuego"]


@pytest.fixture(scope="module")
def detector(tmp_path_factory):
    carpeta = tmp_path_factory.mktemp("modelos")
    construir_detector(TAM, len(ANCLAS), len(CLASES)).save(carpeta / "detector.keras")
    cfg = {"tam_imagen": TAM, "grid": TAM // 16, "anclas": ANCLAS, "clases": CLASES}
    (carpeta / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    return DetectorFuegoHumo(carpeta / "detector.keras", carpeta / "config.json")


def test_sin_modelo_el_error_explica_que_hacer(tmp_path):
    with pytest.raises(FileNotFoundError, match="notebook de Colab"):
        DetectorFuegoHumo(tmp_path / "no_existe.keras", tmp_path / "no_existe.json")


def test_la_llamada_compilada_da_lo_mismo_que_la_directa(detector):
    x = np.random.default_rng(0).uniform(0, 255, (1, TAM, TAM, 3)).astype(np.float32)
    directa = np.asarray(detector.modelo(x, training=False))
    compilada = np.asarray(detector._inferir(x))
    assert compilada.shape == (1, TAM // 16, TAM // 16, len(ANCLAS), 5 + len(CLASES))
    np.testing.assert_allclose(compilada, directa, rtol=1e-4, atol=1e-4)


def test_detectar_devuelve_cajas_en_pixeles_dentro_del_fotograma(detector):
    alto, ancho = 240, 426                               # cualquier tamaño de video
    frame = np.random.default_rng(1).integers(0, 256, (alto, ancho, 3), dtype=np.uint8)
    detecciones = detector.detectar(frame, umbral=0.0)   # umbral 0: el modelo sin entrenar "detecta" de todo

    assert 0 < len(detecciones) <= 50
    for d in detecciones:
        x1, y1, x2, y2 = d.caja
        assert 0 <= x1 <= x2 <= ancho and 0 <= y1 <= y2 <= alto
        assert d.clase in CLASES
        assert d.subtipo in (("negro", "blanco") if d.clase == "humo" else ("llama",))
        assert 0.0 <= d.confianza <= 1.0 and 0.0 <= d.area_relativa <= 1.0


def test_con_un_umbral_alto_el_modelo_sin_entrenar_no_detecta_nada(detector):
    frame = np.zeros((240, 320, 3), np.uint8)
    assert detector.detectar(frame, umbral=0.9) == []


def test_detectar_aplica_el_umbral_de_cada_clase(detector):
    frame = np.random.default_rng(2).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    solo_humo = detector.detectar(frame, umbral={"humo": 0.0, "fuego": 0.99})
    solo_fuego = detector.detectar(frame, umbral=[0.99, 0.0])     # lista: orden de CLASES
    assert solo_humo and {d.clase for d in solo_humo} == {"humo"}
    assert solo_fuego and {d.clase for d in solo_fuego} == {"fuego"}


def test_sin_umbral_se_usan_los_de_config(detector, monkeypatch):
    from src import config
    frame = np.random.default_rng(3).integers(0, 256, (240, 320, 3), dtype=np.uint8)
    monkeypatch.setattr(config, "UMBRALES_CLASE", {"humo": 0.0, "fuego": 0.99})
    assert {d.clase for d in detector.detectar(frame)} == {"humo"}
    assert config.umbrales_por_clase(["humo", "fuego", "otra"]) == [0.0, 0.99, config.UMBRAL_CONFIANZA]


@pytest.mark.parametrize("gris, esperado", [(30, "negro"), (220, "blanco")])
def test_color_del_humo(gris, esperado):
    frame = np.full((100, 100, 3), gris, np.uint8)
    assert clasificar_color_humo(frame, (10, 10, 90, 90)) == esperado


def test_color_del_humo_con_caja_vacia():
    frame = np.zeros((100, 100, 3), np.uint8)
    assert clasificar_color_humo(frame, (50, 50, 50, 50)) == "blanco"
