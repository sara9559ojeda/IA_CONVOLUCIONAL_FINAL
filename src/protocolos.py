"""Base de conocimiento que usa el asistente (transformer de pregunta-respuesta).

El modelo de pregunta-respuesta es EXTRACTIVO: no inventa texto, busca dentro
del contexto el fragmento que responde la pregunta. Por eso cada protocolo se
escribe con oraciones que responden directamente a las preguntas de
PREGUNTAS_RECOMENDACION ("El riesgo de esta situación es...",
"La recomendación principal es...", etc.).

Los protocolos son orientativos y se basan en recomendaciones generales de
seguridad contra incendios. Ajústalos a la normativa de tu institución.
"""
from .config import NUMERO_BOMBEROS, NUMERO_EMERGENCIAS

# (título que se muestra, pregunta que se le hace al transformer)
PREGUNTAS_RECOMENDACION = [
    ("Nivel de riesgo", "¿Cuál es el riesgo de esta situación?"),
    ("Acción inmediata", "¿Cuál es la recomendación principal?"),
    ("Llamada de emergencia", "¿A qué número se debe llamar?"),
    ("Qué evitar", "¿Qué no se debe hacer?"),
    ("Después del evento", "¿Qué se recomienda hacer después?"),
]

_LLAMAR = (f"la línea de emergencias {NUMERO_EMERGENCIAS} o a los bomberos "
           f"al {NUMERO_BOMBEROS}")

PROTOCOLOS = {
    "sin_incidentes": (
        "Situación actual: no se detecta fuego ni humo. "
        "El riesgo de esta situación es bajo y el área se encuentra en condiciones normales. "
        "La recomendación principal es mantener la vigilancia y conservar despejadas las rutas de evacuación. "
        f"Se debe llamar a {_LLAMAR} únicamente si se observa humo, fuego u olor a quemado. "
        "No se debe obstruir los extintores ni acumular material inflamable cerca de fuentes de calor. "
        "Después se recomienda revisar periódicamente los extintores, las alarmas y las instalaciones eléctricas."
    ),
    "humo_blanco": (
        "Situación actual: se detecta humo blanco sin llamas visibles. "
        "El riesgo de esta situación es moderado porque el humo blanco suele indicar la fase inicial de un incendio, "
        "la combustión de vegetación húmeda o la presencia de vapor. "
        "La recomendación principal es verificar el origen del humo a distancia segura y preparar la evacuación del área. "
        f"Se debe llamar a {_LLAMAR} si el humo aumenta o si no se logra identificar su origen. "
        "No se debe ignorar el humo ni ingresar a zonas donde el humo sea denso. "
        "Después se recomienda revisar las instalaciones eléctricas y las fuentes de calor cercanas antes de retomar las actividades. "
        "Si el humo proviene de un equipo eléctrico se debe cortar la energía desde el tablero principal."
    ),
    "humo_negro": (
        "Situación actual: se detecta humo negro sin llamas visibles. "
        "El riesgo de esta situación es alto porque el humo negro indica la combustión de plásticos, caucho o combustibles "
        "y contiene gases tóxicos como el monóxido de carbono. "
        "La recomendación principal es evacuar el área de inmediato desplazándose agachado y cubriendo nariz y boca con un paño húmedo. "
        f"Se debe llamar a {_LLAMAR} apenas se esté en un lugar seguro. "
        "No se debe respirar el humo, usar ascensores ni regresar por objetos personales. "
        "Después se recomienda reunirse en el punto de encuentro y buscar atención médica si hay mareo, tos o dificultad para respirar."
    ),
    "fuego_incipiente": (
        "Situación actual: se detecta un fuego pequeño en su etapa inicial. "
        "El riesgo de esta situación es alto porque un fuego incipiente puede propagarse en pocos minutos. "
        "La recomendación principal es activar la alarma y, solo si el fuego es pequeño y se tiene una salida libre a la espalda, "
        "usar un extintor de tipo ABC apuntando a la base de las llamas. "
        f"Se debe llamar a {_LLAMAR} aunque el fuego parezca controlado. "
        "No se debe usar agua sobre fuegos eléctricos o de aceite ni intentar apagar el fuego si se propaga. "
        "Después se recomienda vigilar la zona porque el fuego puede reavivarse y esperar la revisión de los bomberos."
    ),
    "incendio_declarado": (
        "Situación actual: se detecta un incendio declarado. "
        "El riesgo de esta situación es muy alto porque hay fuego extenso o en crecimiento y riesgo para la vida de las personas. "
        "La recomendación principal es evacuar de inmediato por la ruta de evacuación más cercana, sin intentar apagar el fuego. "
        f"Se debe llamar a {_LLAMAR} de inmediato e informar la ubicación exacta del incendio. "
        "No se debe usar ascensores, abrir puertas calientes ni regresar al lugar del incendio. "
        "Después se recomienda reunirse en el punto de encuentro, verificar que todas las personas hayan salido "
        "y seguir las instrucciones de los bomberos. "
        "En espacios abiertos se debe alejar en dirección contraria al viento y hacia zonas sin vegetación."
    ),
}

CONOCIMIENTO_GENERAL = (
    "El humo negro indica la combustión de plásticos, caucho, combustibles o materiales sintéticos y es más tóxico que el humo blanco. "
    "El humo blanco indica la combustión de vegetación húmeda, la presencia de vapor o la fase inicial de un incendio. "
    "El nivel PRECAUCION significa que hay humo blanco y se debe verificar su origen. "
    "El nivel ALERTA significa que hay humo negro o un fuego incipiente y se debe actuar de inmediato. "
    "El nivel EMERGENCIA significa que hay un incendio declarado y se debe evacuar. "
    "Un extintor de tipo ABC sirve para fuegos de sólidos, líquidos inflamables y equipos eléctricos. "
    "Para protegerse del humo se debe caminar agachado porque el aire más limpio está cerca del piso. "
    "El sistema usa una red neuronal convolucional entrenada desde cero que detecta fuego y humo en cada fotograma. "
    "El color del humo se determina analizando el brillo y la saturación de los píxeles dentro de la caja detectada. "
    "Las falsas alarmas pueden deberse a atardeceres, luces rojas o anaranjadas, niebla o nubes."
)
