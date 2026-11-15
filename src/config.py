"""Configuración central del proyecto de detección de fuego y humo.

Todos los umbrales y rutas se ajustan aquí para no tener "números mágicos"
repartidos por el código.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
DIR_MODELOS = RAIZ / "modelos"
RUTA_MODELO = DIR_MODELOS / "detector_fuego_humo.keras"     # se genera en Colab
RUTA_CONFIG_MODELO = DIR_MODELOS / "detector_config.json"   # anclas, clases, tamaño
DIR_RESULTADOS = RAIZ / "resultados"
DIR_ALERTAS = DIR_RESULTADOS / "alertas"                    # capturas + alertas.csv (src/registro.py)

# --------------------------------------------------------------------------
# Detector (CNN tipo YOLO entrenada desde cero)
# --------------------------------------------------------------------------
TAM_IMAGEN = 320          # la imagen se redimensiona a 320x320
STRIDE = 16               # la red reduce la resolución 16 veces -> cuadrícula 20x20
NUM_ANCLAS = 3            # cajas de referencia por celda (se calculan con k-means)
CLASES = ["humo", "fuego"]  # mismo orden que D-Fire: 0 = smoke, 1 = fire

# Limpieza del dataset (src/limpieza.py)
TAM_GUARDADO = 384            # las imágenes limpias se guardan a 384x384 (margen para el zoom)
UMBRAL_HASH_DUPLICADO = 8     # bits distintos (de 256) para considerar dos imágenes duplicadas
UMBRAL_HASH_SIMILAR = 24      # bits distintos para considerarlas "muy parecidas" (fuga / grupos)
FRAC_NEGATIVAS = 0.30         # proporción máxima de imágenes sin objetos en entrenamiento

UMBRAL_CONFIANZA = 0.35  # confianza mínima para mostrar una caja (clases sin umbral propio)
# Umbral por clase: el humo sale con menos confianza que el fuego. Estos valores
# se eligieron con el primer modelo; después de reentrenar conviene revisarlos
# con el barrido de umbrales en validación (notebook, sección 6.1) y escribir
# aquí los elegidos, porque la app y evaluar_escenarios.py los leen de aquí.
UMBRALES_CLASE = {"humo": 0.20, "fuego": 0.35}
UMBRAL_NMS = 0.45         # IoU para eliminar cajas repetidas (Non-Max Suppression)


def umbrales_por_clase(clases):
    """Lista de umbrales en el mismo orden que `clases` (p. ej. [0.20, 0.35])."""
    return [UMBRALES_CLASE.get(c, UMBRAL_CONFIANZA) for c in clases]


# Subtipo de humo (negro / blanco) por análisis de color dentro de la caja.
# Un píxel es "oscuro" si tiene poco brillo (V) y poca saturación (S) en HSV.
HUMO_V_OSCURO = 0.35
HUMO_S_MAX = 0.40
HUMO_FRACCION_OSCURA = 0.20   # si >= 20 % de la caja es oscura -> humo negro

# --------------------------------------------------------------------------
# Alertas y clasificación de eventos
# --------------------------------------------------------------------------
VENTANA_FRAMES = 8        # se analizan los últimos 8 fotogramas
MIN_FRAMES_CONFIRMAR = 4  # una detección se confirma si aparece en 4 de 8 fotogramas
AREA_INCENDIO = 0.10      # fuego que ocupa >= 10 % de la imagen -> incendio declarado
AREA_FUEGO_CRECIENDO = 0.04

# Colores (formato BGR de OpenCV)
COLOR_CAJA = (60, 230, 124)           # verde como en la imagen de referencia
COLOR_TEXTO = (255, 255, 255)
COLORES_NIVEL = {
    "NORMAL": (60, 180, 60),
    "PRECAUCION": (0, 215, 255),      # amarillo
    "ALERTA": (0, 140, 255),          # naranja
    "EMERGENCIA": (0, 0, 255),        # rojo
}

# --------------------------------------------------------------------------
# Asistente (transformer de HuggingFace, pregunta-respuesta)
# --------------------------------------------------------------------------
# Se intenta cargar el primero; si falla se usa el siguiente.
MODELOS_QA = [
    "mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es",
    "deepset/xlm-roberta-base-squad2",
]
UMBRAL_QA = 0.01   # por debajo de este puntaje la respuesta se considera poco fiable

# Números de emergencia (Colombia). Cámbialos si el proyecto es de otro país.
NUMERO_EMERGENCIAS = "123"
NUMERO_BOMBEROS = "119"
