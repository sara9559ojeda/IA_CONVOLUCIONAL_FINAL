"""Asistente que explica las detecciones y da recomendaciones.

Usa un TRANSFORMER de HuggingFace para pregunta-respuesta, como en el ejemplo
del profesor:

    qa = pipeline("question-answering", model=...)
    respuesta = qa(question=pregunta, context=contexto)

Cómo se conecta con la CNN:
  1. La CNN detecta fuego/humo -> `MonitorEventos` clasifica el evento.
  2. El evento se convierte en texto (resumen_para_asistente) y se une con el
     protocolo de ese tipo de evento y la base de conocimiento -> CONTEXTO.
  3. El transformer recibe una pregunta y busca la respuesta en el contexto.
     - Recomendaciones automáticas: se le hacen 5 preguntas fijas sobre el
       protocolo del evento detectado (riesgo, acción, llamada, qué evitar, después).
     - Chat libre: el usuario pregunta lo que quiera.
  4. La respuesta del transformer se completa hasta el final de la oración
     para que se lea como una recomendación completa.
"""
import re

from . import config
from .eventos import resumen_para_asistente
from .protocolos import CONOCIMIENTO_GENERAL, PREGUNTAS_RECOMENDACION, PROTOCOLOS

_FIN_ORACION = re.compile(r"[.!?](\s|$)")


def completar_oracion(contexto, inicio, fin):
    """Expande el fragmento [inicio, fin) a la oración completa que lo contiene."""
    previos = [m.end() for m in _FIN_ORACION.finditer(contexto, 0, max(inicio, 0))]
    ini = previos[-1] if previos else 0
    m = _FIN_ORACION.search(contexto, max(fin - 1, inicio))
    final = m.start() + 1 if m else len(contexto)
    texto = contexto[ini:final].strip()
    return texto[:1].upper() + texto[1:]


class AsistenteIncendios:
    def __init__(self, modelos=None):
        from transformers import pipeline

        dispositivo = -1
        try:
            import torch
            dispositivo = 0 if torch.cuda.is_available() else -1
        except ImportError:
            pass

        errores = []
        for nombre in modelos or config.MODELOS_QA:
            try:
                self.qa = pipeline("question-answering", model=nombre, tokenizer=nombre,
                                   device=dispositivo)
                self.nombre_modelo = nombre
                break
            except Exception as e:  # noqa: BLE001 - se prueba el siguiente modelo
                errores.append(f"{nombre}: {e}")
        else:
            raise RuntimeError("No se pudo cargar ningún modelo de pregunta-respuesta:\n"
                               + "\n".join(errores))
        self._cache_recomendaciones = {}

    def _preguntar(self, pregunta, contexto):
        r = self.qa(question=pregunta, context=contexto)
        if isinstance(r, list):
            r = r[0]
        return {
            "respuesta": completar_oracion(contexto, r["start"], r["end"]),
            "fragmento": r["answer"],          # lo que extrajo el transformer
            "score": float(r["score"]),
        }

    def recomendaciones(self, tipo_evento):
        """Recomendaciones para el tipo de evento que detectó la CNN (con caché)."""
        if tipo_evento not in self._cache_recomendaciones:
            contexto = PROTOCOLOS[tipo_evento]
            recs = []
            for titulo, pregunta in PREGUNTAS_RECOMENDACION:
                r = self._preguntar(pregunta, contexto)
                r.update(titulo=titulo, pregunta=pregunta)
                recs.append(r)
            self._cache_recomendaciones[tipo_evento] = recs
        return self._cache_recomendaciones[tipo_evento]

    def construir_contexto(self, estado, eventos):
        return " ".join([resumen_para_asistente(estado, eventos),
                         PROTOCOLOS[estado.tipo], CONOCIMIENTO_GENERAL])

    def responder(self, pregunta, estado, eventos):
        """Responde una pregunta libre usando la detección actual como contexto."""
        contexto = self.construir_contexto(estado, eventos)
        r = self._preguntar(pregunta, contexto)
        if r["score"] < config.UMBRAL_QA:
            r["respuesta"] = ("No encontré una respuesta clara en la información disponible. "
                              "Prueba preguntando qué se detectó, por qué se generó la alerta, "
                              "qué hacer o a quién llamar.")
        return r
