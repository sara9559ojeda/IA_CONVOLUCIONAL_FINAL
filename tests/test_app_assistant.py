"""The detector must remain usable when the assistant is unavailable."""
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


@pytest.mark.parametrize("failure", ["load", "recommendations", "chat"])
def test_assistant_failure_does_not_stop_detection(monkeypatch, tmp_path, failure):
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
    capture.read.side_effect = [(True, np.zeros((32, 32, 3), dtype=np.uint8)), (False, None)]
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
        app.toggle(key="analizando").set_value(True).run(timeout=20)
        assert not app.exception
        detector.detectar.assert_called_once()
        capture.release.assert_called_once()
        if failure == "recommendations":
            assistant.recomendaciones.assert_called_once_with("sin_incidentes")
            assert any("No se pudieron obtener" in warning.value for warning in app.warning)
    finally:
        st.cache_resource.clear()
