from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from evaluar_asistente import calcular_metricas


def test_metricas_incluyen_errores_abstenciones_y_respuesta_del_monitor():
    casos = [
        ("estado", "", "sin fotogramas", {"accepted": False}, True),
        ("protocolo", "", "correcta", {"accepted": True}, True),
        ("chat", "", "otra respuesta", {"accepted": True}, False),
        ("chat", "", "dato omitido", {"accepted": False}, False),
        ("sin evidencia", "", None, {"accepted": True}, False),
        ("sin evidencia", "", None, {"accepted": False}, True),
        ("sin evidencia", "", None, {"accepted": False}, True),
    ]
    m = calcular_metricas(casos)
    assert m["accuracy"] == pytest.approx(4 / 7)
    assert m["precision"] == m["recall"] == m["f1"] == 0.5
    assert calcular_metricas(casos[-1:])["precision"] is None


def test_resultados_del_transformer_en_la_app(monkeypatch):
    from src import project_sections

    informe = dict(modelo="modelo de prueba", umbral=0.01, casos=5, respuestas_correctas=2,
                   abstenciones_correctas=1, contestadas=3, con_respuesta=4,
                   accuracy=3/5, precision=2/3, recall=1/2, f1=4/7)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    app.session_state["project_section"] = "Resultados"
    app.run(timeout=20)
    monkeypatch.setattr(project_sections, "read_json", lambda filename: informe)
    next(r for r in app.radio if r.label == "Evaluación").set_value("Asistente (transformer)").run()
    assert not app.exception
    assert {m.label: m.value for m in app.metric} == {
        "Accuracy": "60.00%", "Precision": "66.67%", "Recall": "50.00%", "F1": "57.14%",
    }
