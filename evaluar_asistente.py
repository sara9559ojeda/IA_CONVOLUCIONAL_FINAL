"""Evalúa las respuestas que muestra el asistente con el modelo real de Hugging Face."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from tempfile import NamedTemporaryFile

from src import config
from src.asistente import AsistenteIncendios
from src.eventos import EstadoAlerta
from src.protocolos import PROTOCOLOS


# La respuesta esperada de cada tarjeta es la oración completa de ese apartado
# del protocolo, incluidas sus condiciones. El modelo recibe todo el protocolo.
INICIOS_ESPERADOS = {
    "Nivel de riesgo": "El riesgo de esta situación",
    "Acción inmediata": "La recomendación principal",
    "Llamada de emergencia": "Se debe llamar",
    "Qué evitar": "No se debe",
    "Después del evento": "Después se recomienda",
}


def casos_chat():
    anterior = dict(id=1, tipo="Humo blanco", nivel="PRECAUCION", inicio="10:00",
                    duracion_s=37, estado="finalizado", _abierto=False)
    actual = dict(id=2, tipo="Fuego incipiente", nivel="ALERTA", inicio="10:04",
                  duracion_s=12, estado="en curso", _abierto=True)
    estado = EstadoAlerta(tipo="fuego_incipiente", ventana=8, frames_fuego=5,
                          conf_fuego=0.82, area_fuego=0.02, evento=actual,
                          razon="se detectó fuego en 5 de los últimos 8 fotogramas")
    pendiente = EstadoAlerta(ventana=5, frames_humo=3)
    # La cadena esperada debe aparecer en la oración presentada al usuario.
    # None indica que la pregunta carece de evidencia.
    return [
        ("estado", EstadoAlerta(), [], "¿Qué se ha detectado?", "no se ha analizado ningún fotograma"),
        ("estado", pendiente, [], "¿Por qué no hay alerta?", "humo en 3 fotogramas y se requieren 4"),
        ("reformulación", estado, [anterior, actual], "¿Qué tan seguro está el detector de que hay fuego?", "82 por ciento"),
        ("reformulación", estado, [anterior, actual], "¿Qué parte de la imagen ocupan las llamas?", "2 por ciento"),
        ("reformulación", estado, [anterior, actual], "¿Cuánto tiempo lleva el evento actual?", "12 segundos"),
        ("historial", estado, [anterior, actual], "¿Cuánto duró el evento anterior?", "37 segundos"),
        ("historial", estado, [anterior, actual], "¿A qué hora empezó el evento 1?", "10:00"),
        ("sin evidencia", estado, [anterior, actual], "¿En qué habitación ocurrió?", None),
        ("sin evidencia", estado, [anterior, actual], "¿Qué temperatura exacta tiene la habitación?", None),
        ("sin evidencia", estado, [anterior, actual], "¿Quién llamó a los bomberos?", None),
        ("sin evidencia", estado, [anterior, actual], "¿Cuántas personas había en el edificio?", None),
    ]


def evaluar(asistente):
    for tipo, protocolo in PROTOCOLOS.items():
        oraciones = protocolo.split(". ")
        for r in asistente.recomendaciones(tipo):
            esperada, = (s.rstrip(".") + "." for s in oraciones
                         if s.startswith(INICIOS_ESPERADOS[r["titulo"]]))
            yield tipo, r["pregunta"], esperada, r, r["accepted"] and r["respuesta"] == esperada
    for grupo, estado, eventos, pregunta, esperada in casos_chat():
        r = asistente.responder(pregunta, estado, eventos)
        if esperada is None:
            correcta = not r["accepted"]
        else:
            coincide = re.search(r"(?<!\w)" + re.escape(esperada) + r"(?!\w)", r["respuesta"], re.I)
            correcta = coincide is not None and (r["accepted"] or estado.ventana == 0)
        yield grupo, pregunta, esperada, r, correcta


def calcular_metricas(casos):
    total = con_respuesta = contestadas = correctas = abstenciones_correctas = 0
    for _, _, esperada, respuesta, acierto in casos:
        total += 1
        if esperada is None:
            abstenciones_correctas += int(acierto)
            contestadas += int(respuesta["accepted"])
        else:
            con_respuesta += 1
            correctas += int(acierto)
            # Un aviso de estado correcto también responde a la pregunta del usuario.
            contestadas += int(respuesta["accepted"] or acierto)

    return {
        "casos": total, "con_respuesta": con_respuesta, "contestadas": contestadas,
        "respuestas_correctas": correctas, "abstenciones_correctas": abstenciones_correctas,
        "accuracy": (correctas + abstenciones_correctas) / total if total else None,
        "precision": correctas / contestadas if contestadas else None,
        "recall": correctas / con_respuesta if con_respuesta else None,
        "f1": 2 * correctas / (contestadas + con_respuesta) if contestadas + con_respuesta else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modelo", choices=config.MODELOS_QA, default=config.MODELOS_QA[0])
    args = parser.parse_args()
    asistente = AsistenteIncendios([args.modelo])
    print(f"Modelo: {asistente.nombre_modelo} | Umbral: {config.UMBRAL_QA}")
    totales, aciertos = Counter(), Counter()
    casos = list(evaluar(asistente))
    for grupo, pregunta, esperada, respuesta, correcta in casos:
        totales[grupo] += 1
        aciertos[grupo] += int(correcta)
        if not correcta:
            print(f"\n[{grupo}] {pregunta}\n  Esperada: {esperada or 'abstención'}"
                  f"\n  Obtenida: {respuesta['respuesta']} (puntaje {respuesta['score']:.4f})")
    for grupo, total in totales.items():
        print(f"{grupo}: {aciertos[grupo]}/{total} aciertos")
    print(f"Total: {aciertos.total()}/{totales.total()} aciertos")
    print(f"Respuestas sin evidencia: {totales['sin evidencia'] - aciertos['sin evidencia']}"
          f"/{totales['sin evidencia']}")
    metricas = calcular_metricas(casos)
    for nombre in ("accuracy", "precision", "recall", "f1"):
        valor = metricas[nombre]
        print(f"{nombre.upper()}: {valor:.2%}" if valor is not None else f"{nombre.upper()}: sin datos")
    informe = {"modelo": asistente.nombre_modelo, "umbral": config.UMBRAL_QA, **metricas}
    destino = config.DIR_MODELOS / "metricas_asistente.json"
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=destino.parent, delete=False) as temporal:
        json.dump(informe, temporal, ensure_ascii=False, indent=2, allow_nan=False)
        temporal.write("\n")
    Path(temporal.name).replace(destino)
    print(f"Informe guardado en {destino}")


if __name__ == "__main__":
    main()
