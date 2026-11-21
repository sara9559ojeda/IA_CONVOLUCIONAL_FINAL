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
