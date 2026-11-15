"""Red convolucional detectora de fuego y humo (estilo YOLO), entrenada DESDE CERO.

Idea general (igual que YOLOv2, pero más pequeña):
  1. La imagen (320x320) pasa por capas Conv2D + BatchNorm + LeakyReLU que van
     reduciendo la resolución hasta una cuadrícula de 20x20 celdas.
  2. Cada celda predice NUM_ANCLAS cajas. Para cada caja la red entrega:
        [tx, ty, tw, th, objetividad, p(humo), p(fuego)]
     - tx, ty : desplazamiento del centro dentro de la celda (sigmoide -> 0..1)
     - tw, th : tamaño relativo a la caja ancla (exponencial)
     - objetividad : probabilidad de que haya un objeto en esa caja
     - p(clase): probabilidad de cada clase
  3. Las cajas con confianza = objetividad * p(clase) mayor al umbral se filtran
     con Non-Max Suppression (NMS) para quitar duplicados.

Al entrenar, la pérdida no castiga como "sin objeto" a las cajas predichas que
ya coinciden con una caja real (zona de "ignorar", como en YOLOv2/v3).

Solo se usan capas estándar de Keras, así el modelo se guarda y carga con
`keras.models.load_model(..., compile=False)` sin código adicional.
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# ==========================================================================
# Arquitectura
# ==========================================================================
def conv_bn(x, filtros, k=3, s=1):
    """Convolución + normalización por lotes + LeakyReLU."""
    x = layers.Conv2D(filtros, k, strides=s, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.LeakyReLU(0.1)(x)


def bloque_residual(x, filtros):
    """Bloque residual: 1x1 (reduce canales) -> 3x3, y se suma la entrada."""
    y = conv_bn(x, filtros // 2, 1)
    y = conv_bn(y, filtros, 3)
    return layers.Add()([x, y])


def construir_detector(tam=320, num_anclas=3, num_clases=2):
    """Construye la CNN detectora. Salida: (grid, grid, num_anclas, 5 + num_clases)."""
    grid = tam // 16
    entrada = keras.Input((tam, tam, 3), name="imagen")       # RGB 0..255
    x = layers.Rescaling(1.0 / 255)(entrada)                   # normaliza a 0..1

    x = conv_bn(x, 32, 3)                 # 320x320
    x = conv_bn(x, 64, 3, s=2)            # 160x160
    x = bloque_residual(x, 64)
    x = conv_bn(x, 128, 3, s=2)           # 80x80
    x = bloque_residual(x, 128)
    x = bloque_residual(x, 128)
    x = conv_bn(x, 256, 3, s=2)           # 40x40
    x = bloque_residual(x, 256)
    x = bloque_residual(x, 256)
    detalle_40 = x                        # se guarda para el "atajo" (passthrough)
    x = conv_bn(x, 512, 3, s=2)           # 20x20
    x = bloque_residual(x, 512)
    x = bloque_residual(x, 512)

    # Passthrough: trae detalles finos de 40x40 para detectar objetos pequeños
    p = conv_bn(detalle_40, 128, 1)
    p = conv_bn(p, 128, 3, s=2)           # 40x40 -> 20x20
    x = layers.Concatenate()([x, p])

    # Cabeza de detección
    x = conv_bn(x, 512, 3)
    x = conv_bn(x, 256, 1)
    x = conv_bn(x, 512, 3)
    x = layers.Conv2D(num_anclas * (5 + num_clases), 1, dtype="float32",
                      name="salida_conv")(x)
    salida = layers.Reshape((grid, grid, num_anclas, 5 + num_clases),
                            dtype="float32", name="salida")(x)
    modelo = keras.Model(entrada, salida, name="detector_fuego_humo")

    # Inicializa el sesgo de "objetividad" en un valor bajo (prob ~1 %).
    # Al inicio casi todas las celdas están vacías; esto estabiliza el
    # entrenamiento desde cero.
    capa = modelo.get_layer("salida_conv")
    pesos, sesgo = capa.get_weights()
    sesgo = sesgo.reshape(num_anclas, 5 + num_clases)
    sesgo[:, 4] = -4.6
    sesgo[:, 5:] = -2.0
    capa.set_weights([pesos, sesgo.reshape(-1)])
    return modelo


# ==========================================================================
# Codificación de etiquetas (cajas -> tensor objetivo de la cuadrícula)
# ==========================================================================
def codificar_objetivos(cajas, anclas, grid, num_clases):
    """Convierte las cajas de una imagen en el tensor que la red debe predecir.

    cajas : tensor [N, 5] con (clase, cx, cy, w, h) normalizados 0..1.
            Las filas de relleno tienen clase = -1.
    anclas: array numpy [A, 2] con (w, h) normalizados.
    Retorna tensor [grid, grid, A, 5 + num_clases] con
            (tx, ty, w, h, objeto, one-hot de clase).
    """
    anclas = np.asarray(anclas, np.float32)
    num_anclas = anclas.shape[0]
    validas = (cajas[:, 0] >= 0) & (cajas[:, 3] > 0) & (cajas[:, 4] > 0)
    cajas = tf.boolean_mask(cajas, validas)

    clase = tf.cast(cajas[:, 0], tf.int32)
    cx, cy, w, h = cajas[:, 1], cajas[:, 2], cajas[:, 3], cajas[:, 4]
    gx, gy = cx * grid, cy * grid
    i = tf.clip_by_value(tf.cast(tf.floor(gx), tf.int32), 0, grid - 1)   # columna
    j = tf.clip_by_value(tf.cast(tf.floor(gy), tf.int32), 0, grid - 1)   # fila
    tx = gx - tf.cast(i, tf.float32)
    ty = gy - tf.cast(j, tf.float32)

    # Se asigna cada caja al ancla con la forma más parecida (mayor IoU)
    a_t = tf.constant(anclas)
    inter = (tf.minimum(w[:, None], a_t[None, :, 0]) *
             tf.minimum(h[:, None], a_t[None, :, 1]))
    union = (w * h)[:, None] + (a_t[:, 0] * a_t[:, 1])[None, :] - inter
    a = tf.cast(tf.argmax(inter / union, axis=1), tf.int32)

    indices = tf.stack([j, i, a], axis=1)
    valores = tf.concat([tf.stack([tx, ty, w, h, tf.ones_like(w)], axis=1),
                         tf.one_hot(clase, num_clases)], axis=1)
    objetivo = tf.zeros([grid, grid, num_anclas, 5 + num_clases], tf.float32)
    return tf.tensor_scatter_nd_update(objetivo, indices, valores)


# ==========================================================================
# Función de pérdida
# ==========================================================================
def mascara_ignorar(y_true, y_pred, anclas_t, umbral_iou=0.5, max_cajas=50):
    """Marca con 1 las cajas predichas que ya coinciden con alguna caja real.

    Cada caja real se asigna a UNA sola celda y ancla. Las celdas vecinas que
    predicen casi la misma caja no están equivocadas, así que no se les debe
    exigir "aquí no hay objeto". Esto pesa sobre todo en el humo: sus cajas son
    grandes y difusas, y su centro puede caer en varias celdas.

    Retorna un tensor [lote, grid, grid, A] con 1.0 donde la caja predicha tiene
    IoU > umbral_iou con alguna caja real de la imagen (de cualquier clase).
    Se comparan a lo sumo `max_cajas` cajas reales por imagen.
    """
    forma = tf.shape(y_true)
    lote, grid = forma[0], forma[1]
    g = tf.cast(grid, tf.float32)
    indice = tf.range(g)
    col = tf.reshape(indice, [1, 1, -1, 1])
    fila = tf.reshape(indice, [1, -1, 1, 1])

    def esquinas(cx, cy, wh):
        return tf.stack([cx - wh[..., 0] / 2, cy - wh[..., 1] / 2,
                         cx + wh[..., 0] / 2, cy + wh[..., 1] / 2], axis=-1)

    # Misma parametrización que `decodificar`
    pred = esquinas((col + tf.sigmoid(y_pred[..., 0])) / g,
                    (fila + tf.sigmoid(y_pred[..., 1])) / g,
                    anclas_t * tf.exp(tf.clip_by_value(y_pred[..., 2:4], -6.0, 4.0)))
    real = esquinas((col + y_true[..., 0]) / g, (fila + y_true[..., 1]) / g, y_true[..., 2:4])
    pred = tf.reshape(pred, [lote, -1, 1, 4])                       # [B, N, 1, 4]
    real = tf.reshape(real, [lote, -1, 4])                          # [B, N, 4]

    # Las cajas reales son las posiciones del objetivo con objeto = 1
    obj = tf.reshape(y_true[..., 4], [lote, -1])
    hay, idx = tf.math.top_k(obj, k=tf.minimum(max_cajas, tf.shape(obj)[1]))
    real = tf.gather(real, idx, batch_dims=1)[:, None]              # [B, 1, K, 4]

    lados = tf.maximum(tf.minimum(pred[..., 2:], real[..., 2:]) -
                       tf.maximum(pred[..., :2], real[..., :2]), 0.0)
    inter = lados[..., 0] * lados[..., 1]
    area_pred = (pred[..., 2] - pred[..., 0]) * (pred[..., 3] - pred[..., 1])
    area_real = (real[..., 2] - real[..., 0]) * (real[..., 3] - real[..., 1])
    iou = inter / (area_pred + area_real - inter + 1e-9) * hay[:, None, :]   # relleno -> 0
    coincide = tf.reduce_max(iou, axis=-1) > umbral_iou             # [B, N]
    return tf.reshape(tf.cast(coincide, tf.float32), forma[:-1])


def _terminos_perdida(anclas, l_caja=5.0, l_obj=1.0, l_noobj=0.5, l_clase=1.0,
                      umbral_ignorar=0.5):
    """Los cuatro términos de la pérdida, cada uno ya multiplicado por su peso."""
    anclas_t = tf.constant(np.asarray(anclas, np.float32))

    def bce(etiqueta, logits):
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=etiqueta, logits=logits)

    def por_imagen(suma, y_true):
        return suma / tf.cast(tf.shape(y_true)[0], tf.float32)

    def caja(y_true, y_pred):
        y_pred = tf.cast(y_pred, tf.float32)
        obj = y_true[..., 4]                                  # 1 donde hay objeto
        escala = 2.0 - y_true[..., 2] * y_true[..., 3]        # más peso a cajas pequeñas
        # Centro (tx, ty)
        p_xy = tf.reduce_sum(bce(y_true[..., 0:2], y_pred[..., 0:2]), axis=-1)
        # Tamaño en escala logarítmica respecto al ancla
        wh_obj = tf.math.log(tf.maximum(y_true[..., 2:4], 1e-6) / anclas_t)
        p_wh = tf.reduce_sum(tf.square(wh_obj - y_pred[..., 2:4]), axis=-1)
        return por_imagen(l_caja * tf.reduce_sum(obj * escala * (p_xy + p_wh)), y_true)

    def obj(y_true, y_pred):
        """Objetividad en las celdas que SÍ tienen objeto."""
        y_pred = tf.cast(y_pred, tf.float32)
        hay = y_true[..., 4]
        return por_imagen(l_obj * tf.reduce_sum(hay * bce(hay, y_pred[..., 4])), y_true)

    def noobj(y_true, y_pred):
        """Objetividad en las celdas sin objeto (lo que frena las falsas alarmas)."""
        y_pred = tf.cast(y_pred, tf.float32)
        hay = y_true[..., 4]
        vacia = 1.0 - hay
        if umbral_ignorar is not None:
            vacia *= 1.0 - tf.stop_gradient(
                mascara_ignorar(y_true, y_pred, anclas_t, umbral_ignorar))
        return por_imagen(l_noobj * tf.reduce_sum(vacia * bce(hay, y_pred[..., 4])), y_true)

    def clase(y_true, y_pred):
        """Clase (sigmoide por clase), solo donde hay objeto."""
        y_pred = tf.cast(y_pred, tf.float32)
        hay = y_true[..., 4:5]
        return por_imagen(
            l_clase * tf.reduce_sum(hay * bce(y_true[..., 5:], y_pred[..., 5:])), y_true)

    return {"caja": caja, "obj": obj, "noobj": noobj, "clase": clase}


def perdida_detector(anclas, **opciones):
    """Pérdida tipo YOLO = posición + tamaño + objetividad + clase.

    opciones: l_caja, l_obj, l_noobj, l_clase (pesos de cada término) y
    umbral_ignorar (IoU de `mascara_ignorar`; None la desactiva).
    """
    terminos = _terminos_perdida(anclas, **opciones)

    def perdida(y_true, y_pred):
        return tf.add_n([f(y_true, y_pred) for f in terminos.values()])

    return perdida


def metricas_perdida(anclas, **opciones):
    """Cada término de la pérdida como métrica de Keras: perd_caja, perd_obj,
    perd_noobj y perd_clase. Sumados dan `loss`; sirven para ver cuál se estanca.
    Deben recibir las mismas opciones que `perdida_detector`."""
    metricas = []
    for nombre, f in _terminos_perdida(anclas, **opciones).items():
        f.__name__ = f"perd_{nombre}"          # nombre de la columna en historial.csv
        metricas.append(f)
    return metricas


# ==========================================================================
# Decodificación de la salida (tensor -> lista de cajas)
# ==========================================================================
def _sigmoide(z):
    return 1.0 / (1.0 + np.exp(-z))


def nms(cajas, puntajes, umbral_iou):
    """Non-Max Suppression: conserva la caja más confiable y elimina las que se solapan."""
    x1, y1, x2, y2 = cajas[:, 0], cajas[:, 1], cajas[:, 2], cajas[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    orden = puntajes.argsort()[::-1]
    conservar = []
    while orden.size > 0:
        k = orden[0]
        conservar.append(k)
        resto = orden[1:]
        xx1 = np.maximum(x1[k], x1[resto])
        yy1 = np.maximum(y1[k], y1[resto])
        xx2 = np.minimum(x2[k], x2[resto])
        yy2 = np.minimum(y2[k], y2[resto])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[k] + areas[resto] - inter + 1e-9)
        orden = resto[iou < umbral_iou]
    return conservar


def decodificar(pred, anclas, umbral=0.35, umbral_nms=0.45, max_det=50, max_candidatos=300):
    """Convierte la salida cruda de UNA imagen en detecciones.

    pred: array [grid, grid, A, 5 + C]
    umbral: un número (igual para todas las clases) o una lista con un umbral
            por clase, en el orden de las clases.
    Retorna lista de (indice_clase, confianza, [x1, y1, x2, y2] normalizados 0..1).
    """
    anclas = np.asarray(anclas, np.float32)
    grid = pred.shape[0]
    num_clases = pred.shape[-1] - 5
    umbrales = np.broadcast_to(np.asarray(umbral, np.float32), (num_clases,))
    fila, col = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")

    cx = (col[..., None] + _sigmoide(pred[..., 0])) / grid
    cy = (fila[..., None] + _sigmoide(pred[..., 1])) / grid
    w = anclas[:, 0] * np.exp(np.clip(pred[..., 2], -6, 4))
    h = anclas[:, 1] * np.exp(np.clip(pred[..., 3], -6, 4))
    puntajes = _sigmoide(pred[..., 4])[..., None] * _sigmoide(pred[..., 5:])

    detecciones = []
    for c in range(num_clases):
        p = puntajes[..., c]
        m = p >= umbrales[c]
        if not m.any():
            continue
        cajas = np.stack([cx[m] - w[m] / 2, cy[m] - h[m] / 2,
                          cx[m] + w[m] / 2, cy[m] + h[m] / 2], axis=1).clip(0, 1)
        s = p[m]
        if len(s) > max_candidatos:
            top = s.argsort()[::-1][:max_candidatos]
            cajas, s = cajas[top], s[top]
        for k in nms(cajas, s, umbral_nms):
            detecciones.append((c, float(s[k]), cajas[k]))
    detecciones.sort(key=lambda d: d[1], reverse=True)
    return detecciones[:max_det]


def iou_matriz(a, b):
    """IoU entre dos conjuntos de cajas [N,4] y [M,4] en formato x1,y1,x2,y2."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    xx1 = np.maximum(a[:, None, 0], b[None, :, 0])
    yy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    xx2 = np.minimum(a[:, None, 2], b[None, :, 2])
    yy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)
