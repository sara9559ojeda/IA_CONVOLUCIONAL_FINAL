"""Inferencia del detector: carga la CNN entrenada, detecta fuego/humo en un
fotograma y dibuja las cajas con el estilo de la imagen de referencia:

    ┌───────────────────────────┐
    │ Humo(negro)  (0.76)       │  <- etiqueta con fondo verde
    ├───────────────────────────┤
    │                           │  <- caja verde
"""
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from . import config
from .modelo import decodificar

FUENTE = cv2.FONT_HERSHEY_SIMPLEX


@dataclass
class Deteccion:
    clase: str            # "fuego" | "humo"
    subtipo: str          # "llama" | "negro" | "blanco"
    confianza: float      # 0..1
    caja: tuple           # (x1, y1, x2, y2) en píxeles
    area_relativa: float  # fracción de la imagen que ocupa la caja (0..1)

    @property
    def etiqueta(self):
        # OpenCV no dibuja tildes, por eso las etiquetas van sin acentos
        return f"{self.clase.capitalize()}({self.subtipo})  ({self.confianza:.2f})"


def clasificar_color_humo(frame_bgr, caja):
    """Decide si el humo es negro o blanco según la fracción de píxeles oscuros.

    Humo negro = combustión de plásticos, caucho o combustibles (más tóxico).
    Humo blanco = vegetación húmeda, vapor o fase inicial.
    D-Fire no trae esta distinción, así que se resuelve con análisis de color
    (espacio HSV) dentro de la caja que entregó la CNN.
    """
    x1, y1, x2, y2 = caja
    recorte = frame_bgr[y1:y2, x1:x2]
    if recorte.size == 0:
        return "blanco"
    hsv = cv2.cvtColor(recorte, cv2.COLOR_BGR2HSV)
    s = hsv[..., 1] / 255.0
    v = hsv[..., 2] / 255.0
    oscuros = (v < config.HUMO_V_OSCURO) & (s < config.HUMO_S_MAX)
    return "negro" if oscuros.mean() >= config.HUMO_FRACCION_OSCURA else "blanco"


class DetectorFuegoHumo:
    """Envuelve el modelo Keras y su configuración (anclas, clases, tamaño)."""

    def __init__(self, ruta_modelo=None, ruta_config=None):
        import tensorflow as tf
        from tensorflow import keras

        ruta_modelo = Path(ruta_modelo or config.RUTA_MODELO)
        ruta_config = Path(ruta_config or config.RUTA_CONFIG_MODELO)
        if not ruta_modelo.exists() or not ruta_config.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo entrenado en '{ruta_modelo}' o su "
                f"configuración en '{ruta_config}'. Entrena el detector con el "
                "notebook de Colab y copia ambos archivos a la carpeta 'modelos/'."
            )
        self.modelo = keras.models.load_model(ruta_modelo, compile=False)
        with open(ruta_config, encoding="utf-8") as f:
            cfg = json.load(f)
        self.tam = int(cfg["tam_imagen"])
        self.anclas = np.asarray(cfg["anclas"], np.float32)
        self.clases = list(cfg["clases"])
        # Llamada compilada (grafo): evita ejecutar la red capa por capa en modo
        # "eager". En CPU cada fotograma tarda cerca de un tercio (medido: 147 -> 50 ms).
        self._inferir = tf.function(
            lambda x: self.modelo(x, training=False),
            input_signature=[tf.TensorSpec([1, self.tam, self.tam, 3], tf.float32)])

    def detectar(self, frame_bgr, umbral=None, umbral_nms=None):
        """Retorna la lista de `Deteccion` encontradas en el fotograma.

        umbral: None usa los umbrales por clase de config; también admite un
        número (igual para todas), una lista en el orden de las clases o un
        diccionario {"humo": 0.20, "fuego": 0.35}.
        """
        if umbral is None:
            umbral = config.umbrales_por_clase(self.clases)
        elif isinstance(umbral, dict):
            umbral = [umbral.get(c, config.UMBRAL_CONFIANZA) for c in self.clases]
        umbral_nms = config.UMBRAL_NMS if umbral_nms is None else umbral_nms
        alto, ancho = frame_bgr.shape[:2]

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        x = cv2.resize(rgb, (self.tam, self.tam)).astype(np.float32)[None]
        pred = np.asarray(self._inferir(x))[0]

        detecciones = []
        for idx, conf, (x1, y1, x2, y2) in decodificar(pred, self.anclas, umbral, umbral_nms):
            caja = (int(x1 * ancho), int(y1 * alto), int(x2 * ancho), int(y2 * alto))
            clase = self.clases[idx]
            subtipo = clasificar_color_humo(frame_bgr, caja) if clase == "humo" else "llama"
            area = float((x2 - x1) * (y2 - y1))
            detecciones.append(Deteccion(clase, subtipo, conf, caja, area))
        return detecciones


def dibujar(frame_bgr, detecciones, estado=None):
    """Dibuja cajas, etiquetas y un marco de alerta; el aviso va en la interfaz."""
    img = frame_bgr.copy()
    alto, ancho = img.shape[:2]
    base = min(alto, ancho)
    grosor = max(2, round(base / 200))
    escala = max(0.5, base / 550)
    grosor_txt = max(1, round(base / 350))

    if estado is not None and estado.nivel != "NORMAL":
        color = config.COLORES_NIVEL[estado.nivel]
        borde = max(6, base // 60)
        cv2.rectangle(img, (0, 0), (ancho - 1, alto - 1), color, borde)

    for d in detecciones:
        x1, y1, x2, y2 = d.caja
        cv2.rectangle(img, (x1, y1), (x2, y2), config.COLOR_CAJA, grosor)
        (tw, th), linea_base = cv2.getTextSize(d.etiqueta, FUENTE, escala, grosor_txt)
        alto_etq = th + linea_base + 8
        y_etq = y1 - alto_etq if y1 - alto_etq >= 0 else y1   # si no cabe arriba, va adentro
        cv2.rectangle(img, (x1, y_etq), (x1 + tw + 10, y_etq + alto_etq),
                      config.COLOR_CAJA, -1)
        cv2.putText(img, d.etiqueta, (x1 + 5, y_etq + th + 4), FUENTE, escala,
                    config.COLOR_TEXTO, grosor_txt, cv2.LINE_AA)

    return img
