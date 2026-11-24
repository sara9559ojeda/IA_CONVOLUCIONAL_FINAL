"""Integration tests for the detector and its assistant panel."""
import base64
import platform
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from src import asistente, config, video


def test_switch_between_results_and_detector(monkeypatch, tmp_path):
    st.cache_resource.clear()
    monkeypatch.setitem(sys.modules, "src.detector", SimpleNamespace(DetectorFuegoHumo=Mock))
    monkeypatch.setattr(asistente, "AsistenteIncendios", lambda: Mock(nombre_modelo="test"))
    monkeypatch.setattr(config, "DIR_ALERTAS", tmp_path)
    monkeypatch.setattr(video, "limpiar_videos_antiguos", lambda: None)
    try:
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=20)
        for section in ("Resultados", "Probar detector", "Resultados"):
            app.radio(key="project_section").set_value(section).run(timeout=20)
            assert not app.exception
            assert any(m.label == "mAP50" for m in app.metric) == (section == "Resultados")
            assert any(s.value == "Observación" for s in app.subheader) == (section == "Probar detector")
    finally:
        st.cache_resource.clear()


def test_confirmar_borrado_actualiza_el_panel(monkeypatch, tmp_path):
    st.cache_resource.clear()
    monkeypatch.setitem(sys.modules, "src.detector", SimpleNamespace(DetectorFuegoHumo=Mock))
    monkeypatch.setattr(asistente, "AsistenteIncendios", lambda: Mock(nombre_modelo="test"))
    monkeypatch.setattr(config, "DIR_ALERTAS", tmp_path)
    monkeypatch.setattr(video, "limpiar_videos_antiguos", lambda: None)
    (tmp_path / "alertas.csv").write_text("sesion,id,captura\nprueba,1,\n", encoding="utf-8")
    try:
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=20)
        assert any(e.label == "Alertas guardadas en disco (1)" for e in app.expander)
        app.button(key="confirmar_borrado_inicial").click().run(timeout=20)
        assert not app.exception
        assert any(e.label == "Alertas guardadas en disco (0)" for e in app.expander)
        assert not (tmp_path / "alertas.csv").exists()
    finally:
        st.cache_resource.clear()


@pytest.mark.parametrize("failure", ["load", "recommendations", "chat"])
def test_assistant_failure_does_not_stop_detection(monkeypatch, tmp_path, failure):
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    st.cache_resource.clear()
    detector = Mock()
    detector.detectar.return_value = []
    monkeypatch.setitem(sys.modules, "src.detector", SimpleNamespace(
        DetectorFuegoHumo=lambda: detector,
        dibujar=lambda frame, detections, state: frame,
    ))
    assistant = Mock(nombre_modelo="test-model")
    assistant.recomendaciones.return_value = []
    assistant.responder.side_effect = RuntimeError("inference failed")
    factory = Mock(return_value=assistant)
    if failure == "load":
        factory.side_effect = RuntimeError("model unavailable")
    elif failure == "recommendations":
        assistant.recomendaciones.side_effect = RuntimeError("inference failed")
    monkeypatch.setattr(asistente, "AsistenteIncendios", factory)
    monkeypatch.setattr(config, "DIR_ALERTAS", tmp_path)
    monkeypatch.setattr(video, "limpiar_videos_antiguos", lambda: None)
    capture = Mock()
    capture.isOpened.return_value = True
    capture.get.return_value = 30.0
    frame = np.full((32, 32, 3), (10, 65, 215), dtype=np.uint8)
    from streamlit.delta_generator import DeltaGenerator
    display = DeltaGenerator.image
    displayed_colors = []

    def record_image(container, image, **kwargs):
        if isinstance(image, np.ndarray):
            displayed_colors.append((tuple(image[0, 0]), kwargs["channels"]))
        return display(container, image, **kwargs)

    monkeypatch.setattr(DeltaGenerator, "image", record_image)
    reads = iter([(False, None), (True, frame)])
    capture.read.side_effect = lambda: next(reads, (False, None))
    monkeypatch.setattr(cv2, "VideoCapture", Mock(return_value=capture))
    try:
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
        app.run(timeout=20)
        assert not app.exception
        assert app.text_input[0].disabled == (failure == "load")
        if failure == "chat":
            app.text_input[0].set_value("¿Qué se detectó?")
            next(button for button in app.button if button.label == "Preguntar").click()
            app.run(timeout=20)
            assert not app.exception
            assert any("No se pudo responder" in warning.value for warning in app.warning)
        elif failure == "load":
            assert app.warning
        else:
            assistant.recomendaciones.assert_not_called()
        app.checkbox(key="local_camera").set_value(True).run(timeout=20)
        app.toggle(key="analizando").set_value(True).run(timeout=20)
        assert not app.exception
        detector.detectar.assert_called_once()
        capture.release.assert_called_once()
        assert displayed_colors == [((215, 65, 10), "RGB")]
        assert any("dejó de enviar imágenes" in warning.value for warning in app.warning)
        assert app.session_state["ultimo_frame"] is None
        if failure == "recommendations":
            assistant.recomendaciones.assert_called_once_with("sin_incidentes")
            assert any("No se pudieron obtener" in warning.value for warning in app.warning)
        if failure == "chat":
            app.session_state["source_chat"] = [("Consulta previa", {"accepted": False, "respuesta": "Anterior"})]
            reads = iter([(True, frame)])
            next(s for s in app.slider if "humo" in s.label).set_value(0.55).run(timeout=20)
            assert not app.exception
            assert app.toggle(key="analizando").value
            assert not app.session_state["source_chat"]
            assert detector.detectar.call_args.args[1]["humo"] == 0.55
            assert capture.release.call_count == 2
            capture.isOpened.return_value = False
            app.number_input[0].set_value(4).run(timeout=20)
            assert not app.exception
            assert any("Sin señal de la cámara 4" in w.value for w in app.warning)
            assert any("índice 0" in e.value for e in app.error)
            assert not any("Sin fuego ni humo" in m.value for m in app.markdown)
    finally:
        st.cache_resource.clear()


def test_browser_camera_analyzes_each_received_frame_once(monkeypatch, tmp_path):
    st.cache_resource.clear()
    detector = Mock()
    detector.detectar.return_value = []
    monkeypatch.setitem(sys.modules, "src.detector", SimpleNamespace(
        DetectorFuegoHumo=lambda: detector, dibujar=lambda frame, *_: frame))
    assistant = Mock(nombre_modelo="test")
    assistant.recomendaciones.return_value = []
    monkeypatch.setattr(asistente, "AsistenteIncendios", lambda: assistant)
    monkeypatch.setattr(config, "DIR_ALERTAS", tmp_path)
    fake_camera = Mock()
    monkeypatch.setitem(sys.modules, "src.browser_camera", SimpleNamespace(capture=fake_camera))
    raw = np.full((24, 32, 3), 120, dtype=np.uint8)
    ok, jpeg = cv2.imencode(".jpg", raw)
    assert ok
    packet = {"id": "test-frame", "jpeg": base64.b64encode(jpeg).decode()}
    fake_camera.return_value = SimpleNamespace(frame=packet)
    try:
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=20)
        app.toggle(key="analizando").set_value(True).run(timeout=20)
        assert not app.exception
        detector.detectar.assert_called_once()
        assert np.array_equal(detector.detectar.call_args.args[0], raw)
        assert fake_camera.call_args.kwargs["data"]["ack"] == packet["id"]
        assert app.session_state["frame_idx"] == 1
    finally:
        st.cache_resource.clear()


def test_chat_keeps_uploaded_video_as_capture_source(monkeypatch, tmp_path):
    st.cache_resource.clear()
    monkeypatch.setitem(sys.modules, "src.detector", SimpleNamespace(DetectorFuegoHumo=Mock))
    assistant = Mock(nombre_modelo="test-model")
    assistant.recomendaciones.return_value = []
    assistant.responder.return_value = {
        "respuesta": "Respuesta", "fuente": "Guía", "fragmento": "Texto", "score": 0.9,
    }
    monkeypatch.setattr(asistente, "AsistenteIncendios", lambda: assistant)
    monkeypatch.setattr(config, "DIR_ALERTAS", tmp_path)
    monkeypatch.setattr(video, "limpiar_videos_antiguos", lambda: None)
    path = tmp_path / "escena.mp4"
    path.touch()
    monkeypatch.setattr(video, "guardar_video_temporal", lambda data, suffix: str(path))
    capture = Mock()
    capture.read.return_value = (False, None)
    open_capture = Mock(return_value=capture)
    monkeypatch.setattr(cv2, "VideoCapture", open_capture)
    try:
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
        next(r for r in app.radio if r.label == "Fuente").set_value("Archivo de video").run()
        app.file_uploader[0].upload("escena.mp4", b"test", "video/mp4")
        app.toggle(key="analizando").set_value(True).run()
        open_capture.assert_called_once_with(str(path))
        open_capture.reset_mock()
        app.text_input[0].set_value("¿Qué debo hacer?")
        next(b for b in app.button if b.label == "Preguntar").click().run()
        assert not app.exception
        assistant.responder.assert_called_once()
        open_capture.assert_called_once_with(str(path))
    finally:
        st.cache_resource.clear()


def test_new_image_starts_with_empty_chat(monkeypatch, tmp_path):
    st.cache_resource.clear()
    detector = Mock()
    detector.detectar.return_value = []
    monkeypatch.setitem(sys.modules, "src.detector", SimpleNamespace(
        DetectorFuegoHumo=lambda: detector,
        dibujar=lambda frame, detections, state: frame,
    ))
    assistant = Mock(nombre_modelo="test-model")
    assistant.recomendaciones.return_value = []
    assistant.responder.return_value = {"respuesta": "Respuesta anterior", "accepted": False}
    monkeypatch.setattr(asistente, "AsistenteIncendios", lambda: assistant)
    monkeypatch.setattr(config, "DIR_ALERTAS", tmp_path)
    monkeypatch.setattr(video, "limpiar_videos_antiguos", lambda: None)
    try:
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run()
        next(r for r in app.radio if r.label == "Fuente").set_value("Imagen").run()
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        uploader = app.file_uploader[0]
        uploader.upload("escena.png", cv2.imencode(".png", image)[1].tobytes()).run()
        app.text_input[0].set_value("¿Qué debo hacer?")
        next(b for b in app.button if b.label == "Preguntar").click().run()
        assert any("Respuesta anterior" in m.value for m in app.markdown)
        app.file_uploader[0].clear().upload(
            "escena.png", cv2.imencode(".png", 255 - image)[1].tobytes()
        ).run()
        assert not app.exception
        assert not app.session_state["source_chat"]
        assert not any("Respuesta anterior" in m.value for m in app.markdown)
    finally:
        st.cache_resource.clear()
