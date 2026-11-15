# Proyecto 5 · Detección de incendios y presencia de humo

> 👉 **¿Primera vez? Sigue la guía [PASO_A_PASO.md](PASO_A_PASO.md)**, que va desde descargar el dataset hasta la presentación.

Un sistema de visión artificial **detecta fuego y humo en video, imágenes o cámara web**, **clasifica el evento**, **genera alertas visuales** e incluye un **asistente con transformers** que explica las detecciones y da recomendaciones.

```
 Video / cámara ─► CNN detectora (desde cero) ─► cajas: Fuego(llama) 0.82 · Humo(negro) 0.76
                                                   │
                                                   ▼
                                MonitorEventos: confirma en 4 de 8 fotogramas,
                                clasifica el evento y asigna el nivel de alerta
                                                   │
                     ┌─────────────────────────────┴───────────────────────────┐
                     ▼                                                         ▼
     Alerta visual (marco + aviso)                      Asistente (transformer HuggingFace, QA)
     Historial de eventos                               contexto = detección + protocolo
                                                        → recomendaciones y chat
```

## Estructura

| Archivo | Qué hace |
|---|---|
| `notebooks/entrenamiento_detector_colab.ipynb` | Entrena la CNN en Colab, la evalúa y la exporta |
| `src/modelo.py` | Arquitectura de la CNN tipo YOLO, la función de pérdida, la codificación de cajas y el NMS |
| `src/limpieza.py` | Limpieza del dataset: imágenes corruptas, etiquetas defectuosas, duplicados, fuga train→test, división por grupos, balance y re-codificación a 384×384 |
| `src/entrenamiento.py` | Lectura de D-Fire, anclas por k-means, aumento de datos, entrenamiento, mAP y matriz de confusión |
| `src/detector.py` | Inferencia: detecta en un fotograma, clasifica el humo como negro o blanco y dibuja las cajas |
| `src/eventos.py` | Suavizado temporal, clasificación del evento, nivel de alerta e historial |
| `src/registro.py` | Guarda en `resultados/alertas/` una captura y una fila de `alertas.csv` por cada alerta |
| `src/video.py` | Salto de fotogramas para ir en tiempo real y manejo de los videos temporales de la app |
| `src/protocolos.py` | Base de conocimiento: protocolos por tipo de evento |
| `src/asistente.py` | Asistente con `pipeline("question-answering")` de HuggingFace |
| `app.py` | Demostración en Streamlit |
| `evaluar_escenarios.py` | Evaluación en varios videos y tablas para el reporte |
| `src/config.py` | Umbrales, rutas y números de emergencia |
| `tests/` | Pruebas unitarias (`pytest`) de la codificación de cajas, el NMS, los eventos, el registro y el video |

## Cumplimiento de los requisitos

| Requisito | Cómo se cumple |
|---|---|
| Detectar fuego y humo en video | CNN detectora sobre cada fotograma (cámara, archivo de video o imagen) |
| Clasificar los eventos | Sin incidentes · Humo blanco · Humo negro · Fuego incipiente · Incendio declarado |
| Alertas visuales | Marco y aviso de color en el video, más un banner en la app (PRECAUCION / ALERTA / EMERGENCIA) |
| Evaluar escenarios | `evaluar_escenarios.py` genera métricas por video: tiempo hasta la alerta, % de fotogramas en alerta y falsas alarmas |
| Asistente que explica | Un transformer de pregunta-respuesta usa como contexto la predicción de la CNN |

Sobre los modelos preentrenados: **la CNN no usa ninguno**. Se construye y entrena desde cero con Keras, igual que en el notebook de clase. Solo el asistente usa un transformer de HuggingFace, como en el ejemplo del profesor.

---

## Cómo entrenar la CNN (Colab)

1. **Descarga D-Fire** desde https://github.com/gaiasd/DFireDataset (los enlaces de descarga están en su README). Son unas 21 000 imágenes con etiquetas YOLO: 0 = humo y 1 = fuego.
2. **Sube a Google Drive:**
   - esta carpeta completa → `MyDrive/deteccion_incendios/`
   - el zip del dataset → `MyDrive/datasets/D-Fire.zip`
3. Abre `notebooks/entrenamiento_detector_colab.ipynb` en Colab y elige **Entorno de ejecución → GPU T4**.
4. Ejecuta las celdas en orden. Revisa la celda de *verificación visual*: las cajas de **fuego** deben estar sobre llamas.
5. Entrenamiento: 60 épocas como máximo, con parada temprana. Cada época se guarda en `MyDrive/deteccion_incendios/entrenamiento/v3_cinco_anclas/`. **Si Colab se desconecta, vuelve a ejecutar las celdas y el entrenamiento continúa** desde la última época.
6. Después de entrenar, el notebook muestra en **validación** cómo cambian el recall, la precisión y las falsas alarmas por imagen con cada umbral de confianza (`explorar_umbrales`). Con eso se eligen los umbrales de humo y fuego, se escriben en `UMBRALES_CLASE` de `src/config.py` y se evalúa en el conjunto de prueba.
7. Al final quedan en `MyDrive/deteccion_incendios/modelos/` los archivos `detector_fuego_humo.keras`, `detector_config.json` y `metricas_test.json`. Descárgalos a la carpeta `modelos/` local.

### ¿Cuánto tarda? (estimaciones, sin medir)

| Hardware | Por época (~17 000 imágenes, 320×320) | 60 épocas |
|---|---|---|
| Colab GPU T4 (gratis) | ~1,5–3 min | **~2–3 h** (suele parar antes por parada temprana) |
| Kaggle Notebooks (T4/P100, 30 h/semana gratis) | parecido | ~2–3 h, con sesiones de hasta 12 h |
| RTX 4050 6 GB | ~1–2 min | ~1–2 h, **pero solo con WSL2**: TensorFlow no usa la GPU en Windows nativo |
| Solo CPU | 20+ min | no recomendado |

- **Si sale un error de memoria (OOM):** baja `LOTE` a 8.
- **Si queda poco tiempo de Colab:** baja `EPOCAS` a 30. El resultado será un poco peor, pero sirve para la demo.
- **Qué esperar:** un detector entrenado desde cero rinde menos que un YOLO preentrenado. Un mAP50 de ~0,35–0,55 es razonable para este proyecto. Lo importante es explicar el porqué en el reporte.

---

## Cómo ejecutar la demostración (local)

Las versiones de `requirements.txt` están **fijas** y requieren **Python 3.12**. `tensorflow` y `keras` deben ser los mismos del entrenamiento (están en `modelos/detector_config.json`); el detalle está en la Fase 6 de [PASO_A_PASO.md](PASO_A_PASO.md).

```bash
py -3.12 -m pip install -r requirements.txt
```

```bash
py -3.12 -m streamlit run app.py
```

- En la barra lateral elige **Cámara web**, **Archivo de video** o **Imagen**, y activa **▶ Analizar**.
- Ajusta los **umbrales de confianza** (uno para fuego y otro para humo) para ver cómo cambian las detecciones. Los valores iniciales salen de `UMBRALES_CLASE` en `src/config.py` (humo 0,20 y fuego 0,35, porque el humo sale con menos confianza); se eligen con el barrido de umbrales del notebook.
- **Analizar 1 de cada N fotogramas:** en *Auto* se descartan los fotogramas que el PC no alcanza a analizar, para que el video avance a su velocidad real. Con *1* se analizan todos.
- Cada alerta guarda una captura y una fila en `resultados/alertas/alertas.csv`, que no se borran al reiniciar la sesión.
- La primera vez se descarga el transformer de HuggingFace (unos cientos de MB).

Evaluación en varios escenarios (para la sección de resultados del reporte):

```bash
py -3.12 evaluar_escenarios.py videos/
```

Pruebas unitarias:

```bash
py -3.12 -m pip install -r requirements-dev.txt
```

```bash
py -3.12 -m pytest
```

Escenarios sugeridos: interior (cocina, bodega), exterior (bosque, pastizal), de noche y con humo sin llama. Incluye también **videos sin fuego que puedan confundir** (atardecer, luces rojas, niebla, chimenea encendida) para medir falsas alarmas.

---

## Explicación técnica (para la sustentación)

### Limpieza del dataset (`src/limpieza.py`)
| Paso | Qué se hace | Por qué |
|---|---|---|
| Imágenes | Se eliminan las corruptas, las de menos de 64 px, las de aspecto mayor a 4:1 y las casi uniformes | Una imagen corrupta detiene `tf.data` a mitad de una época |
| Etiquetas | Se eliminan las líneas mal formadas, las clases inexistentes, las cajas de menos de 3 px a 320×320 y las cajas repetidas (IoU > 0,9); las cajas que salen del borde se recortan | Las etiquetas erróneas meten ruido en la pérdida |
| Duplicados | Se calcula un hash perceptual (dHash de 256 bits) y se agrupan las imágenes con 8 bits distintos o menos; se conserva una por grupo | D-Fire tiene muchos fotogramas consecutivos de video |
| Fuga | Se eliminan de train/val las imágenes a 24 bits o menos de alguna imagen de prueba | Sin esto el mAP de prueba sale inflado |
| División | Train y val se separan por **grupos** de imágenes similares | La validación mide generalización real, no memoria |
| Balance | Las imágenes sin objetos se limitan al 30 % del entrenamiento (val y test no se tocan) | Siguen sirviendo contra falsas alarmas, pero las épocas son más cortas |
| Rendimiento | Todo se re-codifica a 384×384 una sola vez y se guarda en Drive | La CPU de Colab deja de frenar a la GPU |

Los umbrales están en `src/config.py`. El notebook muestra pares de ejemplo para verificarlos y deja un `reporte_limpieza.csv` con los conteos de cada paso.

### CNN detectora (estilo YOLO, desde cero)
- **Entrada:** imagen 320×320 RGB. Bloques `Conv2D → BatchNorm → LeakyReLU` con conexiones residuales reducen la imagen hasta una **cuadrícula de 20×20**.
- **Salida:** cada celda predice **3 cajas**, cada una con `[tx, ty, tw, th, objetividad, p(humo), p(fuego)]`.
- **Anclas:** son los 5 tamaños típicos de las cajas, obtenidos con **k-means** sobre el dataset (10 inicios distintos; se conserva el de mayor IoU medio). Se pasó de 3 a 5 porque con 3 el ancla más parecida solo se solapaba un 50 % en promedio con la caja real: el fuego va de unos pocos píxeles a media imagen y el humo ocupa casi toda. Cada experimento guarda sus anclas junto a sus checkpoints.
- **Pérdida:** error de posición y de tamaño, más entropía cruzada de objetividad y de clase. Cada caja real se asigna a una sola celda y ancla; las demás cajas predichas que ya coinciden con una caja real (IoU > 0,5) **no se castigan** como "sin objeto" (zona de *ignorar*, como en YOLOv2/v3). Sin esto, el humo, que es grande y de centro ambiguo, salía con muy poca confianza.
- **Selección del modelo:** al final de cada época se mide el mAP50 en validación y se guarda la época con el mayor valor (antes se usaba `val_loss`, que casi no cambia entre épocas).
- **NMS (Non-Max Suppression):** elimina las cajas repetidas que se solapan.
- **Aumento de datos:** zoom aleatorio, volteo horizontal, brillo, contraste, saturación y tono.
- **Precisión mixta (float16):** acelera el entrenamiento en la GPU. El modelo se exporta en float32 para la app.

### Humo negro / blanco
D-Fire solo etiqueta "humo". El subtipo se obtiene **analizando el color (HSV) dentro de la caja** que entrega la CNN. Si al menos el 20 % de los píxeles son oscuros y poco saturados, el humo se clasifica como negro. Importa porque el humo negro indica plásticos o combustibles y gases tóxicos.

### Clasificación de eventos y alertas
| Evento | Condición | Nivel |
|---|---|---|
| Sin incidentes | nada confirmado | NORMAL |
| Humo blanco | humo confirmado, sin fuego | PRECAUCION |
| Humo negro | humo oscuro confirmado, sin fuego | ALERTA |
| Fuego incipiente | fuego confirmado y pequeño | ALERTA |
| Incendio declarado | fuego ≥ 10 % de la imagen, en crecimiento o con humo negro | EMERGENCIA |

"Confirmado" significa que aparece en **al menos 4 de los últimos 8 fotogramas**. Así se reducen las falsas alarmas causadas por un fotograma aislado.

### Asistente (transformer)
Igual que el ejemplo de clase, `pipeline("question-answering")`, pero con un modelo en español (`config.MODELOS_QA`):
1. La detección de la CNN se convierte en texto. Ejemplo: *"La alerta se generó porque se detectó fuego en 6 de los últimos 8 fotogramas con una confianza media de 81 por ciento..."*.
2. Ese texto se une con el **protocolo del evento detectado** (`src/protocolos.py`) → **contexto**.
3. **Recomendaciones automáticas:** al cambiar el evento, el transformer responde 5 preguntas: riesgo, acción inmediata, a quién llamar, qué evitar y qué hacer después.
4. **Chat:** el usuario pregunta libremente (*"¿por qué se generó la alerta?"*, *"¿qué significa humo negro?"*).

El modelo es **extractivo**: elige el fragmento del contexto que responde la pregunta, y el sistema lo completa hasta el final de la oración. Por eso nunca inventa información que no esté en la detección o en los protocolos.

### Limitaciones conocidas
- La CNN aprende desde cero con ~17 000 imágenes, así que falla más que un modelo preentrenado con millones de imágenes.
- Puede confundir el fuego con atardeceres, luces anaranjadas o reflejos, y el humo con niebla o nubes.
- La regla de color para el humo negro o blanco puede fallar de noche o a contraluz.
- El asistente solo responde con lo que hay en su contexto. No razona fuera de los protocolos.
