# Guía paso a paso · Detección de incendios y humo

Esta guía cubre todo el proceso, de principio a fin: descargar los datos, entrenar la CNN en Google Colab, ejecutar la demostración en tu computador, evaluar escenarios y preparar el reporte y la presentación.

> **Resumen en una línea:** entrenas la CNN en Colab (unas 2–3 h), descargas 2 archivos a la carpeta `modelos/` y ejecutas `streamlit run app.py` en tu PC.

---

## Mapa general

| Fase | Dónde | Tiempo aprox. |
|---|---|---|
| 0. Preparar cuentas y herramientas | Navegador / PC | 15 min |
| 1. Descargar el dataset D-Fire | Navegador | 20–40 min (depende del internet) |
| 2. Subir el proyecto y el dataset a Google Drive | Navegador | 20–40 min |
| 3. Entrenar la CNN en Colab | Colab | 2,5–3,5 h, incluida la limpieza (casi todo es espera) |
| 4. Evaluar el modelo y guardar resultados | Colab | 10 min |
| 5. Descargar el modelo a tu PC | Navegador | 5 min |
| 6. Preparar Python en tu PC | PowerShell | 20 min |
| 7. Ejecutar la demostración (Streamlit) | PC | 10 min |
| 8. Evaluar distintos escenarios | PC | 30–60 min |
| 9. Reporte y presentación | — | — |

---

## Fase 0 · Lo que necesitas antes de empezar

- [ ] **Una cuenta de Google**, para usar Drive y Colab.
- [ ] **Unos 6 GB libres en Google Drive**: el dataset comprimido más los modelos.
- [ ] **Python 3.12 en tu PC** (descárgalo de https://www.python.org/downloads/). Usa 3.12 aunque ya tengas 3.13, porque TensorFlow a veces tarda en ser compatible con la versión más nueva. En el instalador marca **"Add python.exe to PATH"**.
- [ ] **Una cámara web**, si quieren hacer la demo en vivo. Es opcional: también sirve un video.

---

## Fase 1 · Descargar el dataset D-Fire

D-Fire tiene unas **21 000 imágenes** con cajas dibujadas alrededor del **humo** (clase 0) y del **fuego** (clase 1).

1. Entra a **https://github.com/gaiasd/DFireDataset**.
2. En el README del repositorio busca la sección de descarga del dataset completo (imágenes + etiquetas). Los enlaces llevan a un almacenamiento en la nube.
3. Descarga el `.zip`. Pesa varios GB, así que deja que termine sin cerrar el navegador.
4. **No lo descomprimas.** Lo vas a subir tal cual a Drive.

Por dentro, el zip debería tener carpetas parecidas a estas:
```
train/images/*.jpg     train/labels/*.txt
test/images/*.jpg      test/labels/*.txt
```
Cada `.txt` tiene una línea por objeto: `clase cx cy ancho alto`. No necesitas abrirlos: el notebook los lee automáticamente.

---

## Fase 2 · Subir todo a Google Drive

1. Abre https://drive.google.com.
2. En **Mi unidad** crea la carpeta **`datasets`** y sube ahí el zip con el nombre **`D-Fire.zip`**. Si tiene otro nombre, renómbralo o cambia la ruta en el notebook (Fase 3, paso 5).
3. Sube **la carpeta completa del proyecto** `deteccion_incendios` a **Mi unidad**. Arrástrala desde el Explorador de Windows.
   - Si ya creaste la carpeta `.venv` (Fase 6), **no la subas**: pesa mucho y no se usa en Colab.

Al final, tu Drive debe quedar así:
```
Mi unidad/
├── datasets/
│   └── D-Fire.zip
└── deteccion_incendios/
    ├── app.py
    ├── notebooks/entrenamiento_detector_colab.ipynb
    ├── src/ ...
    └── modelos/
```

---

## Fase 3 · Entrenar la CNN en Google Colab

### 3.1 Abrir el notebook con GPU
1. En Drive entra a `deteccion_incendios/notebooks/`.
2. Haz clic derecho en `entrenamiento_detector_colab.ipynb` → **Abrir con → Google Colaboratory**.
   - Si no aparece Colaboratory: **Abrir con → Conectar más aplicaciones**, busca "Colaboratory" e instálalo.
3. En el menú: **Entorno de ejecución → Cambiar tipo de entorno de ejecución → Acelerador de hardware: GPU T4 → Guardar**.

### 3.2 Ejecutar las celdas en orden
Ejecuta cada celda con **Shift + Enter**, una por una, y revisa que cada paso dé el resultado esperado antes de seguir:

| # | Celda | Qué debe pasar |
|---|---|---|
| 1 | `nvidia-smi` / versión de TensorFlow | Debe decir **Tesla T4** y mostrar una GPU en la lista. Si dice que no hay GPU, revisa el paso 3.1. |
| 2 | Montar Drive | Aparece una ventana para autorizar el acceso a tu Drive. Acepta con tu cuenta. |
| 3 | `RUTA_PROYECTO` | No debe dar error. Si aparece `ModuleNotFoundError: src`, la carpeta del proyecto no está en `Mi unidad/deteccion_incendios`; corrige `RUTA_PROYECTO`. |
| 4 | Copiar y descomprimir el zip | Tarda 3–10 min. Al final lista las carpetas del dataset. Si dice *No such file*, revisa `ZIP_DFIRE`. |
| 5 | **Limpieza del dataset** | Muestra primero la tabla del dataset **original** y luego 6 etapas (`1/6 … 6/6`) con barra de progreso. La primera vez tarda **5–15 min**. Al final aparecen el **reporte de limpieza** y la tabla del dataset **limpio**. 📸 **Toma captura de ambas tablas para el reporte.** Si el dataset original sale con 0 imágenes, la estructura del zip es distinta (ver *Problemas frecuentes*). |
| 6 | Ejemplos de duplicados y de fuga | Muestra pares de imágenes. **Revisa que cada par sea casi la misma imagen.** Si ves pares que no se parecen, mira el paso 3.3. |
| 7 | **Verificación visual** | ⚠️ **Muy importante:** mira las imágenes. Las cajas que dicen **fuego** deben estar sobre **llamas** y las de **humo** sobre **humo**. Si salen al revés, abre `src/config.py` en Drive, cambia `CLASES = ["humo", "fuego"]` por `["fuego", "humo"]`, guarda, borra `entrenamiento/dfire_limpio.zip` y reinicia el entorno (**Entorno de ejecución → Reiniciar sesión**). Las imágenes se ven cuadradas porque se guardaron a 384×384; es normal. |
| 8 | Estadísticas | 4 gráficas: imágenes por partición, cajas por clase, tamaño de las cajas y cajas por imagen. 📸 **Captura para el reporte.** |
| 9 | Anclas (k-means) | Muestra 3 pares de números (ancho, alto) y un "IoU medio". |
| 10 | Modelo y `summary()` | Muestra la tabla de capas de la CNN. 📸 **Toma una captura para el reporte.** |
| 11 | **Entrenamiento** | Empieza a mostrar `Epoch 1/60 ... loss: ... val_loss: ...`. Esta celda tarda **2–3 horas**. |

### 3.3 Qué hace la limpieza y cómo leer su reporte

La limpieza se ejecuta **una sola vez**. El resultado se guarda en Drive como `deteccion_incendios/entrenamiento/dfire_limpio.zip`. Las siguientes veces que abras el notebook, el paso 5 solo lo descomprime (~1 min) y ni siquiera hace falta volver a copiar el zip original.

| Fila del reporte | Qué significa | Valor esperado |
|---|---|---|
| Imágenes corruptas, diminutas, aspecto extremo, uniformes | Imágenes que no sirven o que harían fallar el entrenamiento | Pocas o ninguna |
| Líneas mal formadas, clase inexistente | Errores en los `.txt` de etiquetas | Pocas o ninguna |
| Cajas recortadas | Cajas que se salían del borde y se ajustaron | Algunas |
| Cajas diminutas | Cajas de menos de 3 px a 320×320, que la red no puede aprender | Algunas decenas o cientos |
| Imágenes duplicadas | Copias casi exactas, por ejemplo fotogramas repetidos de un video | Puede ser bastante: D-Fire tiene muchos fotogramas |
| Fuga con test | Imágenes de entrenamiento casi iguales a una de prueba. Se eliminan para que el mAP sea honesto | Variable. Si supera el 25 % sale una advertencia |
| Grupos similares / grupo más grande | Cómo se agruparon las imágenes para separar entrenamiento y validación | Un grupo grande es normal (un mismo video) |
| Negativas descartadas | Imágenes sin fuego ni humo que se quitaron para que sean como máximo el 30 % del entrenamiento | Varios miles |

**Si en los ejemplos aparecen pares que NO son casi la misma imagen**, los umbrales están demasiado altos:
1. Abre `src/config.py` en Drive y baja `UMBRAL_HASH_DUPLICADO` (por ejemplo de 8 a 5) y/o `UMBRAL_HASH_SIMILAR` (de 24 a 16).
2. Borra `entrenamiento/dfire_limpio.zip` en Drive.
3. **Entorno de ejecución → Reiniciar sesión** y ejecuta todo otra vez.

Al revés: si se elimina muy poco y sabes que hay muchos fotogramas repetidos, puedes subir los umbrales.

### 3.4 Mientras entrena
- **Deja la pestaña de Colab abierta.** Si la cierras o el computador se suspende mucho tiempo, Colab puede desconectarse.
- Lo normal es que `loss` y `val_loss` bajen época a época. Al principio bajan rápido y después más despacio.
- `perd_caja`, `perd_obj`, `perd_noobj` y `perd_clase` son los cuatro términos de la pérdida por separado (sumados dan `loss`). Sirven para ver cuál se queda atascado.
- Al final de cada época se mide la **detección real en validación**: `val_mAP50`, `val_AP50_humo`, `val_AP50_fuego` y la precisión y el recall por imagen. En las primeras épocas es normal que salgan en 0.
- Cada época se guarda sola en `Mi unidad/deteccion_incendios/entrenamiento/v2_zona_ignorar/` (la carpeta `DIR_EXPERIMENTO` de la sección 1 del notebook), con los archivos `ultimo.keras`, `mejor.keras` y `historial.csv`. `mejor.keras` es la época con mayor `val_mAP50`.
- Puede terminar antes de las 60 épocas: si `val_mAP50` no mejora durante 12 épocas seguidas, la parada temprana detiene el entrenamiento. Eso es normal.

### 3.5 Si Colab se desconecta
1. Vuelve a conectar (botón **Conectar**, arriba a la derecha) y verifica que siga en GPU T4.
2. Ejecuta **otra vez todas las celdas desde la primera**.
3. En la celda de entrenamiento verás `Reanudando desde la época N`. No se pierde lo ya entrenado.

> Para empezar el entrenamiento desde cero, borra en `deteccion_incendios/entrenamiento/v2_zona_ignorar/` los archivos `ultimo.keras`, `mejor.keras` e `historial.csv`, o cambia el nombre de `DIR_EXPERIMENTO` en la sección 1 del notebook para conservar el entrenamiento anterior y poder compararlos. Borra además `entrenamiento/anclas.npy` solo si quieres recalcular las anclas. **No borres `dfire_limpio.zip`**, para no repetir la limpieza.
>
> Un entrenamiento hecho con una versión anterior del código (otra pérdida) no se puede reanudar: la celda avisa con un error en vez de mezclar los dos.

### 3.6 Si aparece un error de memoria (`ResourceExhaustedError` / OOM)
En la celda 10 cambia `LOTE` de `16` a `8` y vuelve a ejecutar las celdas 10 y 11.

### 3.7 Si te quedas sin tiempo de GPU en Colab
Colab gratis tiene un límite diario. Tienes tres opciones:
- Esperar unas horas y reanudar, como en el paso 3.5.
- Bajar `EPOCAS` a 30.
- Usar **Kaggle Notebooks** (https://www.kaggle.com/code), que da 30 h de GPU por semana. Subes el dataset y el proyecto como "Datasets" de Kaggle y ajustas las rutas.

---

## Fase 4 · Evaluar el modelo (en el mismo notebook)

Cuando termine el entrenamiento, ejecuta las celdas que siguen:

| Celda | Qué hace | Qué guardar para el reporte |
|---|---|---|
| `graficar_historial` | Gráficas de la pérdida, de sus cuatro términos, de la detección en validación (mAP50, recall) y de la tasa de aprendizaje | 📸 Captura de las gráficas |
| `exportar_modelo` | Exporta el mejor modelo a `modelos/` | — |
| `explorar_umbrales` | Muestra, en **validación**, qué pasa con cada umbral de confianza: recall, precisión y falsas alarmas por imagen | 📸 Las dos gráficas (humo y fuego) |
| `evaluar` | Calcula métricas en el conjunto de **prueba** con los umbrales elegidos | 📸 Matriz de confusión y la tabla de métricas |
| `mostrar_predicciones` | Muestra imágenes de prueba con las cajas predichas | 📸 2 o 3 capturas con aciertos y errores |
| Guardar métricas | Crea `modelos/metricas_test.json` y `modelos/barrido_umbrales_val.csv`, y copia a `modelos/` el reporte y los ejemplos de limpieza | Los archivos |

**Cómo elegir los umbrales:** bajar el umbral de una clase hace que se detecte en más imágenes (sube el recall) pero también que avise donde no hay nada (suben las falsas alarmas). En la tabla de `explorar_umbrales` busca, para el humo, el umbral más bajo cuyas falsas alarmas todavía te parezcan aceptables: en este proyecto es peor no ver el humo que dar una falsa alarma. Se elige en validación y no en prueba para que las métricas de prueba sigan siendo una medida limpia. Para probar otros valores, cambia `UMBRALES` en la celda de `evaluar`; cuando te decidas, escríbelos en `UMBRALES_CLASE` de `src/config.py` (en Drive y en tu PC), que es de donde los lee la app.

> Si el entrenamiento ya terminó y solo quieres repetir esta fase, ejecuta todas las celdas **menos la de `ent.entrenar`**: si el entrenamiento se detuvo antes de la última época por la parada temprana, esa celda seguiría entrenando.

**Cómo leer las métricas:**
- **AP50_fuego / AP50_humo:** qué tan bien ubica las cajas de cada clase. Una caja cuenta como correcta si se solapa al menos un 50 % con la real. Va de 0 a 1.
- **mAP50:** el promedio de las dos anteriores. Para una CNN entrenada **desde cero**, **0,35–0,55** es un resultado razonable.
- **exactitud_evento_imagen:** el porcentaje de imágenes en las que el sistema acertó la categoría: normal, solo humo, solo fuego, o fuego y humo.
- **precision / recall por imagen:**
  - La **precisión** responde: de las veces que dijo "hay fuego", ¿cuántas veces sí había fuego?
  - El **recall** responde: de las imágenes con fuego, ¿en cuántas lo detectó?

---

## Fase 5 · Descargar el modelo a tu PC

1. En Drive entra a `deteccion_incendios/modelos/`.
2. Descarga estos archivos:
   - `detector_fuego_humo.keras`
   - `detector_config.json`
   - `metricas_test.json`, `reporte_limpieza.csv`, `ejemplos_duplicados.jpg` y `ejemplos_fuga.jpg` (para el reporte)
3. Cópialos a la carpeta **`modelos/`** dentro de la carpeta del proyecto en tu PC.
4. Abre `detector_config.json` con el Bloc de notas y **anota los valores de `"tensorflow"` y `"keras"`**, por ejemplo `"2.21.0"` y `"3.15.1"`. Los necesitas en la Fase 6.

---

## Fase 6 · Preparar Python en tu PC (una sola vez)

Abre **PowerShell** en la carpeta del proyecto. Una forma de hacerlo: en el Explorador, entra a la carpeta, escribe `powershell` en la barra de direcciones y presiona Enter.

**6.1 Crear un entorno virtual con Python 3.12:**
```powershell
py -3.12 -m venv .venv
```

**6.2 Comprobar que las versiones coinciden con las de Colab.** `requirements.txt` tiene todas las versiones fijas. Ábrelo con el Bloc de notas y compara con lo que anotaste en la Fase 5:
- `tensorflow==...` y `keras==...` deben ser **los mismos números** de `detector_config.json`. Si son distintos, cámbialos en `requirements.txt`.
- `tf-keras==...` debe empezar igual que `tensorflow` (para `tensorflow==2.19.0`, pon `tf-keras==2.19.*`).
- No cambies `transformers`: la versión 5 eliminó la función de pregunta-respuesta que usa el asistente.

```powershell
.venv\Scripts\python -m pip install --upgrade pip
```

**6.3 Instalar las librerías:**
```powershell
.venv\Scripts\python -m pip install -r requirements.txt
```
Tarda varios minutos, porque `torch` y `tensorflow` son grandes.

> Todos los comandos usan `.venv\Scripts\python` directamente para no tener que "activar" el entorno. Así se evita el error de permisos de PowerShell con `Activate.ps1`.

**6.4 Comprobar que todo quedó bien instalado:**
```powershell
.venv\Scripts\python -c "import tensorflow, transformers, cv2, streamlit; print('OK')"
```
Debe imprimir `OK`.

---

## Fase 7 · Ejecutar la demostración

```powershell
.venv\Scripts\python -m streamlit run app.py
```

Se abre el navegador en `http://localhost:8501`.

- **La primera vez** la app descarga el transformer de HuggingFace (unos cientos de MB), así que tarda 1–3 minutos. Después carga rápido.
- Si aparece *"No se encontró el modelo entrenado…"*, revisa la Fase 5: los dos archivos deben estar en `modelos/`.

### Cómo usarla
1. **Barra lateral → Fuente:**
   - **Imagen:** sube una foto de un incendio. Es lo más rápido para probar.
   - **Archivo de video:** sube un `.mp4` y activa **▶ Analizar**.
   - **Cámara web:** activa **▶ Analizar**. Si no abre, cambia el *Índice de cámara* a 1.
2. **Umbrales de confianza (fuego y humo):** hay uno por clase, porque el humo sale con menos confianza que el fuego. Empiezan en 0,35 (fuego) y 0,20 (humo). Si salen muchas cajas falsas de una clase, sube su umbral; si no la detecta, bájalo.
   **Analizar 1 de cada N fotogramas:** déjalo en **Auto** para que el video avance a su velocidad real aunque el PC sea lento (se descartan los fotogramas que no alcanza a analizar). Con **1** se analizan todos, pero el video se ve en cámara lenta.
3. **Panel derecho:**
   - **Banner de alerta:** verde para NORMAL, amarillo para PRECAUCION, naranja para ALERTA y rojo para EMERGENCIA.
   - **Recomendaciones:** el transformer las genera automáticamente cuando cambia el tipo de evento.
   - **Chat:** escribe una pregunta y presiona **Preguntar**.
   - **Historial de eventos:** muestra cada evento con su hora de inicio, duración y confianza máxima.
   - **💾 Alertas guardadas en disco:** cada alerta deja una captura (`.jpg`) y una fila en `resultados/alertas/alertas.csv`. Sirven como evidencia para el reporte.
4. **🗑 Reiniciar sesión** borra el historial y el chat de la pantalla. Las alertas guardadas en `resultados/alertas/` **no** se borran.

### Preguntas de ejemplo para el asistente
- ¿Qué se detectó?
- ¿Por qué se generó la alerta?
- ¿Cuál es el nivel de alerta?
- ¿Qué debo hacer?
- ¿A qué número debo llamar?
- ¿Qué significa el humo negro?
- ¿Qué no se debe hacer?
- ¿Cuánto lleva el evento?
- ¿Qué extintor sirve?

### Dónde conseguir videos de prueba (gratis)
- https://www.pexels.com/search/videos/fire/ y https://www.pexels.com/search/videos/smoke/
- https://pixabay.com/videos/search/fire/
- Para probar **falsas alarmas**, busca videos *sin* fuego: "sunset" (atardecer), "red lights", "fog" (niebla), "fireplace" (chimenea), "candle" (vela).

---

## Fase 8 · Evaluar distintos escenarios (requisito del proyecto)

1. Crea la carpeta `videos/` dentro del proyecto y guarda ahí **entre 6 y 10 videos cortos** (10–30 s). Ponles nombres que describan el escenario:
   ```
   videos/
   ├── 01_interior_cocina_fuego.mp4
   ├── 02_bosque_fuego_humo.mp4
   ├── 03_humo_blanco_sin_llama.mp4
   ├── 04_noche_fuego.mp4
   ├── 05_falsa_alarma_atardecer.mp4
   └── 06_falsa_alarma_luces_rojas.mp4
   ```
2. Ejecuta:
   ```powershell
   .venv\Scripts\python evaluar_escenarios.py videos
   ```
3. Revisa la carpeta `resultados/`:
   - `resumen_escenarios.csv`: **la tabla principal para el reporte**. Ábrela en Excel.
   - `*_anotado.mp4`: videos con las cajas y las alertas. Sirven para la presentación.
   - `*_eventos.csv`: los eventos detectados en cada video.
4. **Cómo interpretar `resumen_escenarios.csv`:**
   - `primera_alerta_s`: cuántos segundos tardó en alertar, o sea, el tiempo de reacción.
   - `%_frames_en_alerta`: debe ser alto en los videos con fuego y **cercano a 0 en los de falsa alarma**.
   - `nivel_maximo` y `tipos_evento`: si la clasificación del evento tiene sentido para ese escenario.

---

## Fase 9 · Reporte y presentación

### Reporte (documento)
| Sección | Qué poner y de dónde sacarlo |
|---|---|
| Problema | Incendios y humo en espacios abiertos o cerrados; la detección temprana salva vidas |
| Objetivos | Los requisitos del enunciado: detectar, clasificar, alertar, evaluar y explicar |
| Metodología | Diagrama del README: CNN → monitor de eventos → alertas y asistente |
| Dataset y limpieza | D-Fire original vs. limpio (tablas de la celda 5), `reporte_limpieza.csv`, `ejemplos_duplicados.jpg`, `ejemplos_fuga.jpg`, gráficas de estadísticas, verificación visual y aumento de datos. Explica **por qué** cada paso de limpieza (sección 3.3) |
| Modelo CNN | Arquitectura (captura del `summary()`), anclas, función de pérdida, NMS y por qué se entrenó desde cero |
| Asistente | Cómo la predicción se convierte en contexto para el transformer de pregunta-respuesta (sección técnica del README) |
| Implementación | Estructura de archivos (README) y capturas de la app |
| Resultados | Gráfica de pérdida, mAP, matriz de confusión, tabla `resumen_escenarios.csv` y capturas de aciertos y errores |
| Conclusiones | Qué funcionó, qué falla (falsas alarmas, noche, humo vs. niebla) y cómo mejorarlo |

### Presentación (10 minutos)
| Minuto | Contenido |
|---|---|
| 0–1 | Problema y objetivo |
| 1–3 | Cómo funciona: CNN tipo YOLO desde cero, cuadrícula, cajas y anclas |
| 3–4 | Clasificación de eventos y niveles de alerta |
| 4–5 | Asistente con transformer: la detección se convierte en contexto y de ahí salen las recomendaciones |
| 5–8 | **Demo en vivo:** imagen → video con fuego → video de falsa alarma → preguntas al asistente |
| 8–9 | Resultados: mAP, matriz de confusión y tabla de escenarios |
| 9–10 | Limitaciones y conclusiones |

> **Consejo para la demo:** lleva los videos ya descargados y los `*_anotado.mp4` como respaldo, por si la cámara o el internet fallan ese día.

### Preguntas probables en la sustentación
- **¿Por qué una cuadrícula y anclas?** Cada celda se encarga de los objetos cuyo centro cae en ella, y las anclas le dan a la red tamaños de referencia para que solo tenga que ajustar.
- **¿Qué hace NMS?** Cuando varias cajas marcan el mismo objeto, se queda con la de mayor confianza.
- **¿Por qué 4 de 8 fotogramas?** Para que un fotograma aislado mal clasificado no dispare una alarma.
- **¿Cómo se sabe si el humo es negro?** Por análisis de color en HSV dentro de la caja, porque el dataset no etiqueta el color.
- **¿El transformer inventa respuestas?** No. Es extractivo: solo selecciona texto del contexto, que está formado por la detección y los protocolos.
- **¿Por qué no se usó YOLO preentrenado?** Por la regla del proyecto: la CNN debía entrenarse desde cero.
- **¿Cómo limpiaron el dataset?** Se eliminaron imágenes corruptas y etiquetas defectuosas, se quitaron duplicados con un hash perceptual y se eliminó la fuga entre entrenamiento y prueba. Además, train y val se separaron por grupos de imágenes similares y se limitaron las imágenes sin objetos al 30 %.
- **¿Qué es la fuga de información?** Que el modelo vea en entrenamiento imágenes casi iguales a las de prueba. Eso infla las métricas, porque el modelo estaría "recordando" en vez de generalizar.

---

## Problemas frecuentes

| Problema | Solución |
|---|---|
| Colab dice que no hay GPU | **Entorno de ejecución → Cambiar tipo → GPU T4.** Si se acabó la cuota, espera unas horas o usa Kaggle. |
| `ModuleNotFoundError: No module named 'src'` en Colab | `RUTA_PROYECTO` no apunta a la carpeta del proyecto en Drive. |
| La tabla del dataset sale con 0 imágenes | Revisa la salida de `!ls /content/dfire`. Las imágenes deben estar dentro de carpetas llamadas `images` y las etiquetas en carpetas `labels`. |
| La limpieza se queda sin RAM (Colab se reinicia) | En la celda 5 llama `limpieza.limpiar_dataset(..., hilos=4)` y vuelve a ejecutar. |
| La advertencia "más del 25 % por fuga" | Mira `ejemplos_fuga.jpg`. Si los pares se parecen de verdad, la advertencia es correcta y puedes seguir. Si no se parecen, baja `UMBRAL_HASH_SIMILAR` (sección 3.3). |
| Cambié algo en `config.py` y no se aplica | Colab guarda los módulos importados en memoria: usa **Entorno de ejecución → Reiniciar sesión** y ejecuta todo otra vez. |
| Error de memoria (OOM) | Baja `LOTE` a 8. |
| `loss` sale `nan` | Borra `ultimo.keras`, `mejor.keras` e `historial.csv` de `entrenamiento/v2_zona_ignorar/` y baja la tasa de aprendizaje: `ent.entrenar(..., lr_max=5e-4)`. |
| La app dice que no encuentra el modelo | Copia `detector_fuego_humo.keras` y `detector_config.json` a `modelos/`. |
| Error al cargar el `.keras` en tu PC | La versión de TensorFlow local no coincide con la de Colab. Repite el paso 6.2 con la versión de `detector_config.json`. |
| `pip install tensorflow` falla | Asegúrate de haber creado el entorno con **Python 3.12** (`py -3.12`). |
| La cámara no abre | Cierra Zoom, Teams u otras apps que usen la cámara y prueba con el índice 1. |
| Muchas detecciones falsas | Sube el umbral en la barra lateral, o entrena más épocas. |
| El asistente responde algo raro | Reformula la pregunta de forma directa, por ejemplo "¿Cuál es la recomendación principal?". Con muy baja confianza responde que no encontró la respuesta. |
| Los números de emergencia no son los de tu país | Cámbialos en `src/config.py` (`NUMERO_EMERGENCIAS`, `NUMERO_BOMBEROS`). |
