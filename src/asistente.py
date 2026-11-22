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
import math
import re

from . import config
from .eventos import bloques_sesion, texto_evento
from .protocolos import CONOCIMIENTO_GENERAL, PREGUNTAS_RECOMENDACION, PROTOCOLOS

_FIN_ORACION = re.compile(r"[.!?](\s|$)")
NO_ANSWER = ("No encontré una respuesta clara en la información disponible. "
             "Prueba preguntando qué se detectó, por qué se generó la alerta, "
             "qué hacer o a quién llamar.")


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

    def _preguntar(self, pregunta, contexto, bloques=()):
        result = self.qa(question=pregunta, context=contexto,
                         handle_impossible_answer=True)
        if isinstance(result, list):
            result = result[0] if result else {}
        fragment = result.get("answer", "")
        score = float(result.get("score", 0.0))
        start, end = result.get("start", 0), result.get("end", 0)
        accepted = (bool(fragment.strip()) and math.isfinite(score)
                    and score >= config.UMBRAL_QA
                    and 0 <= start < end <= len(contexto)
                    and contexto[start:end] == fragment)
        fuente = None
        if accepted and bloques:
            posicion = 0
            for etiqueta, texto in bloques:
                if posicion <= start < end <= posicion + len(texto):
                    fuente = etiqueta
                    break
                posicion += len(texto) + 1
            accepted = fuente is not None
        return {
            "respuesta": completar_oracion(contexto, start, end) if accepted else NO_ANSWER,
            "fragmento": fragment if accepted else "",
            "score": score,
            "accepted": accepted,
            "fuente": fuente,
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

    def _bloques(self, estado, eventos, incluir_cerrados=True):
        bloques = bloques_sesion(estado, eventos, incluir_cerrados)
        if estado.estado_asistente in PROTOCOLOS:
            bloques.append((f"Protocolo · {estado.nombre_corto}", PROTOCOLOS[estado.tipo]))
        if estado.ventana:
            regla = ("En una imagen individual la detección se confirma con un solo fotograma."
                     if estado.imagen_individual else
                     f"En esta fuente se requieren {estado.minimo_confirmacion} fotogramas con detección "
                     f"dentro de los últimos {estado.ventana_configurada} fotogramas analizados.")
            bloques.append(("Regla de confirmación", regla))
        bloques.append(("Conocimiento general", CONOCIMIENTO_GENERAL))
        return bloques

    def construir_contexto(self, estado, eventos):
        return " ".join(texto for _, texto in self._bloques(estado, eventos))

    def responder(self, pregunta, estado, eventos):
        """Responde una pregunta libre usando la detección actual como contexto."""
        numero = re.search(r"\bevento\s+(?:(?:n[uú]mero|n[º°])\s*|#\s*)?(\d+)\b", pregunta, re.I)
        anterior = re.search(r"\b(?:evento|alerta|incidente)\s+(?:anterior|pasad[oa]|finalizad[oa]|cerrad[oa])\b"
                             r"|\b(?:anterior|[uú]ltim[oa]\s+(?:cerrad[oa]|finalizad[oa]))\s+"
                             r"(?:evento|alerta|incidente)\b", pregunta, re.I)
        ultimo = re.search(r"\b[uú]ltim[oa] (?:evento|alerta|incidente)\b", pregunta, re.I)
        if numero or anterior or ultimo:
            if numero:
                evento = next((e for e in eventos if str(e["id"]) == numero[1]), None)
                sin_registro = f"No se ha registrado el evento {numero[1]} en esta sesión."
            elif anterior:
                evento = next((e for e in reversed(eventos) if not e.get("_abierto")), None)
                sin_registro = "Todavía no hay un evento finalizado en esta sesión."
            else:
                evento = eventos[-1] if eventos else None
                sin_registro = "Todavía no se han registrado eventos en esta sesión."
            if evento is None:
                return dict(respuesta=sin_registro, fragmento="", accepted=False, score=0.0)
            fuente = f"Evento {evento['id']} · {evento['estado']}"
            bloques = [(fuente, texto_evento(evento))]
        else:
            historial = re.search(r"\b(?:historial|eventos|alertas|incidentes|anterior(?:es)?|"
                                  r"pasad[oa]s?|finalizad[oa]s?|primer[oa]?|duró|ocurrió)\b",
                                  pregunta, re.I)
            bloques = self._bloques(estado, eventos, incluir_cerrados=bool(historial))
            consulta_actual = re.fullmatch(
                r"¿?(?:por qu[ée] (?:no hay|hay|se (?:generó|activó)) (?:una |la |esta )?alerta|"
                r"qu[ée] (?:se (?:ha detectado|detect[óa])|detect[óa] el sistema)(?: ahora)?|"
                r"cu[aá]l es (?:el nivel de alerta|la tendencia) actual)\??", pregunta.strip(), re.I)
            if consulta_actual:
                bloques = bloques[:1]
                if estado.estado_asistente == "sin_analizar":
                    return dict(respuesta=bloques[0][1], fragmento="", accepted=False, score=0.0)
        contexto = " ".join(texto for _, texto in bloques)
        return self._preguntar(pregunta, contexto, bloques)
