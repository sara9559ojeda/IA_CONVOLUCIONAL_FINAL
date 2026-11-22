"""Regression tests for extractive QA, without downloading models."""
from unittest.mock import Mock

import pytest

from src import config
from src.asistente import AsistenteIncendios, NO_ANSWER
from src.eventos import EstadoAlerta
from src.protocolos import PROTOCOLOS


def make_assistant(result):
    assistant = AsistenteIncendios.__new__(AsistenteIncendios)
    assistant.qa = Mock(return_value=result)
    assistant._cache_recomendaciones = {}
    return assistant


@pytest.mark.parametrize("score", [0.0, config.UMBRAL_QA / 2, float("nan")])
def test_low_confidence_recommendations_are_rejected(score):
    context = PROTOCOLOS["incendio_declarado"]
    assistant = make_assistant(dict(answer=context[:9], start=0, end=9, score=score))
    recommendations = assistant.recomendaciones("incendio_declarado")
    assert len(recommendations) == 5
    assert all(item["respuesta"] == NO_ANSWER for item in recommendations)
    assert all(not item["accepted"] and not item["fragmento"] for item in recommendations)
    assert assistant.recomendaciones("incendio_declarado") == recommendations
    assert assistant.qa.call_count == 5


@pytest.mark.parametrize("result", [
    dict(answer="", start=0, end=0, score=0.99),
    dict(answer=" ", start=0, end=1, score=0.99),
    dict(answer="incorrecto", start=0, end=9, score=0.99),
    dict(answer="texto", start=-1, end=4, score=0.99),
    [],
])
def test_unanswerable_chat_does_not_expand_first_sentence(result):
    assistant = make_assistant(result)
    response = assistant.responder("¿Cuál es la capital de Francia?", EstadoAlerta(), [])
    assert response["respuesta"] == NO_ANSWER
    assert response["fragmento"] == ""
    assert not response["accepted"]
    assert assistant.qa.call_args.kwargs["handle_impossible_answer"] is True


def test_valid_answer_preserves_whole_sentence_and_list_support():
    context = "Primera oración. La recomendación es evacuar ahora. Última oración."
    start = context.index("evacuar")
    assistant = make_assistant([dict(answer="evacuar", start=start, end=start + 7,
                                    score=config.UMBRAL_QA)])
    response = assistant._preguntar("¿Qué hacer?", context)
    assert response["accepted"]
    assert response["respuesta"] == "La recomendación es evacuar ahora."
    assert response["fragmento"] == "evacuar"


def test_contexto_respeta_fuente_y_estado():
    assistant = make_assistant({})
    for estado in (EstadoAlerta(), EstadoAlerta(ventana=1, frames_humo=1)):
        context = assistant.construir_contexto(estado, [])
        assert PROTOCOLOS["sin_incidentes"] not in context
    image = EstadoAlerta(ventana=1, imagen_individual=True)
    context = assistant.construir_contexto(image, [])
    assert "un solo fotograma" in context and "últimos 8" not in context
    assert PROTOCOLOS["sin_incidentes"] in context


def test_chat_selecciona_y_cita_el_evento_solicitado():
    assistant = make_assistant({})
    eventos = [dict(id=n, tipo="humo", nivel="ALERTA", inicio="t0", duracion_s=n * 10,
                    estado="en curso" if n == 9 else "finalizado", _abierto=n == 9)
               for n in range(1, 10)]

    def extract(question, context, **kwargs):
        fragment = context.split()[-2]
        start = context.rindex(fragment)
        return dict(answer=fragment, score=0.99, start=start, end=start + len(fragment))

    assistant.qa.side_effect = extract
    for pregunta, n in (("¿Cuánto duró el evento anterior?", 8), ("¿Cuánto duró el evento 1?", 1),
                        ("¿Cuánto lleva el evento 9?", 9)):
        r = assistant.responder(pregunta, EstadoAlerta(), eventos)
        assert r["fragmento"] == str(n * 10)
        assert r["fuente"] == f"Evento {n} · {eventos[n - 1]['estado']}"
    assert not assistant.responder("¿Cuánto duró el evento 12?", EstadoAlerta(), eventos)["accepted"]
    assert not assistant.responder("¿Cuánto duró el evento anterior?", EstadoAlerta(), [])["accepted"]
    assert assistant.qa.call_count == 3
