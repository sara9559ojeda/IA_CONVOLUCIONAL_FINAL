"""Aplicación de demostración (Streamlit).

    streamlit run app.py

Izquierda : video / cámara / imagen con cajas de fuego y humo y alerta visual.
Derecha   : estado de la alerta, recomendaciones del asistente (transformer),
            chat para preguntar sobre las detecciones e historial de eventos.
"""
import os
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src import config
from src.eventos import EstadoAlerta, MonitorEventos, pct
from src.registro import RegistroAlertas
from src.video import (borrar_video_temporal, descartar_frames, frames_a_saltar,
                       guardar_video_temporal, limpiar_videos_antiguos)

st.set_page_config(page_title="Detección de fuego y humo", page_icon="🔥", layout="wide")


# --------------------------------------------------------------------------
# Carga de modelos (una sola vez)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando la red convolucional...")
def cargar_detector():
    from src.detector import DetectorFuegoHumo
    return DetectorFuegoHumo()


@st.cache_resource(show_spinner="Cargando el asistente (transformer)...")
def cargar_asistente():
    from src.asistente import AsistenteIncendios
    return AsistenteIncendios()


@st.cache_resource(show_spinner=False)
def limpiar_temporales():
    """Una sola vez por ejecución del servidor: borra los videos temporales
    que dejaron sesiones anteriores (por ejemplo, al cerrar el navegador)."""
    return limpiar_videos_antiguos()


def mostrar_imagen(contenedor, imagen_rgb):
    """st.image cambió de parámetros entre versiones; se prueban ambos."""
    try:
        contenedor.image(imagen_rgb, channels="RGB", width="stretch")
    except Exception:  # noqa: BLE001 - versiones antiguas de Streamlit
        contenedor.image(imagen_rgb, channels="RGB", use_container_width=True)


# --------------------------------------------------------------------------
# Estado de la sesión
# --------------------------------------------------------------------------
ss = st.session_state
ss.setdefault("monitor", MonitorEventos())
ss.setdefault("registro", RegistroAlertas())    # captura + fila de CSV por alerta
ss.setdefault("estado", EstadoAlerta())
ss.setdefault("chat", [])
ss.setdefault("frame_idx", 0)
ss.setdefault("video_id", None)
ss.setdefault("video_ruta", None)
ss.setdefault("imagen_id", None)
ss.setdefault("ultimo_frame", None)
limpiar_temporales()

try:
    detector = cargar_detector()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()
try:
    asistente = cargar_asistente()
except Exception as e:  # noqa: BLE001
    st.error(f"No se pudo cargar el asistente: {e}")
    st.stop()

# --------------------------------------------------------------------------
# Barra lateral
# --------------------------------------------------------------------------
st.sidebar.title("🔥 Detección de fuego y humo")
fuente = st.sidebar.radio("Fuente", ["Cámara web", "Archivo de video", "Imagen"])
umbral = {   # un umbral por clase: el humo sale con menos confianza que el fuego
    "fuego": st.sidebar.slider("Umbral de confianza · fuego", 0.10, 0.90,
                               config.UMBRALES_CLASE["fuego"], 0.05),
    "humo": st.sidebar.slider("Umbral de confianza · humo", 0.10, 0.90,
                              config.UMBRALES_CLASE["humo"], 0.05),
}

archivo = None
indice_camara = 0
if fuente == "Cámara web":
    indice_camara = int(st.sidebar.number_input("Índice de cámara", 0, 5, 0))
elif fuente == "Archivo de video":
    archivo = st.sidebar.file_uploader("Video", type=["mp4", "avi", "mov", "mkv"])
    if st.sidebar.button("⏮ Reiniciar video"):
        ss.frame_idx = 0
else:
    archivo = st.sidebar.file_uploader("Imagen", type=["jpg", "jpeg", "png", "bmp", "webp"])

salto = "Auto"
if fuente != "Imagen":
    salto = st.sidebar.select_slider(
        "Analizar 1 de cada N fotogramas", options=["Auto", 1, 2, 3, 5, 10], value="Auto",
        help="Auto descarta los fotogramas necesarios para que el video avance a su "
             "velocidad real. Con 1 se analizan todos (más lento en CPU).")
analizar = fuente != "Imagen" and st.sidebar.toggle("▶ Analizar", key="analizando")
nombre_fuente = archivo.name if archivo is not None else fuente

# El video subido se copia a un archivo temporal: se borra cuando deja de usarse.
if ss.video_ruta and (fuente != "Archivo de video" or archivo is None):
    borrar_video_temporal(ss.video_ruta)
    ss.video_ruta = ss.video_id = None

if st.sidebar.button("🗑 Reiniciar sesión"):
    borrar_video_temporal(ss.video_ruta)
    ss.video_ruta = ss.video_id = None
    ss.monitor.reiniciar()
    ss.registro = RegistroAlertas()     # lo ya guardado en disco se conserva
    ss.estado = EstadoAlerta()
    ss.chat = []
    ss.frame_idx = 0
    ss.ultimo_frame = None
    ss.imagen_id = None
    st.rerun()
st.sidebar.caption(f"Asistente: `{asistente.nombre_modelo}`")

# --------------------------------------------------------------------------
# Diseño
# --------------------------------------------------------------------------
col_video, col_panel = st.columns([3, 2])
with col_video:
    ph_alerta = st.empty()
    ph_frame = st.empty()
    ph_info = st.empty()
with col_panel:
    st.subheader("🤖 Asistente")
    ph_estado = st.empty()
    st.markdown("**Recomendaciones según la detección**")
    ph_recom = st.empty()
    with st.form("form_chat", clear_on_submit=True):
        pregunta = st.text_input("Pregunta al asistente",
                                 placeholder="¿Por qué se generó la alerta? ¿Qué debo hacer?")
        enviada = st.form_submit_button("Preguntar")
    contenedor_chat = st.container()
    st.markdown("**Historial de eventos**")
    ph_historial = st.empty()
    ph_guardadas = st.empty()


def render_alerta(estado):
    estilos = {"PRECAUCION": "#f9c513", "ALERTA": "#ff8c00", "EMERGENCIA": "#e02424"}
    if estado.nivel == "NORMAL":
        ph_alerta.markdown(
            "<div style='padding:10px;border-radius:8px;background:#1f7a3a;color:white;"
            "font-weight:600;text-align:center'>✅ NORMAL · Sin fuego ni humo</div>",
            unsafe_allow_html=True)
    else:
        ph_alerta.markdown(
            f"<div style='padding:14px;border-radius:8px;background:{estilos[estado.nivel]};"
            f"color:white;font-size:1.3rem;font-weight:700;text-align:center'>"
            f"🚨 {estado.nivel} · {estado.nombre}</div>", unsafe_allow_html=True)


def render_estado(estado):
    with ph_estado.container():
        c1, c2, c3 = st.columns(3)
        c1.metric("Fuego", f"{pct(estado.conf_fuego)} %" if estado.frames_fuego else "—")
        c2.metric("Humo", f"{pct(estado.conf_humo)} % {estado.color_humo}" if estado.color_humo else "—")
        c3.metric("Tendencia", estado.tendencia)
        if estado.tipo != "sin_incidentes":
            st.caption(f"Motivo de la alerta: {estado.razon}.")


def render_recomendaciones(tipo):
    with ph_recom.container():
        for r in asistente.recomendaciones(tipo):
            st.markdown(f"**{r['titulo']}:** {r['respuesta']}")
        st.caption("Respuestas extraídas por el transformer a partir del evento detectado por la CNN.")


def render_historial():
    tabla = ss.monitor.tabla_eventos()
    if tabla:
        ph_historial.dataframe(pd.DataFrame(tabla[::-1]), hide_index=True)
    else:
        ph_historial.caption("Aún no hay eventos registrados.")


def render_guardadas():
    """Alertas que quedaron en disco (de esta sesión y de las anteriores)."""
    filas = ss.registro.leer()
    with ph_guardadas.container():
        with st.expander(f"💾 Alertas guardadas en disco ({len(filas)})"):
            if filas:
                st.dataframe(pd.DataFrame(filas[::-1]).drop(columns="sesion", errors="ignore"),
                             hide_index=True)
            st.caption(f"Cada alerta guarda una captura y una fila de `alertas.csv` en "
                       f"`{ss.registro.dir}`. No se borran al reiniciar la sesión.")


# --------------------------------------------------------------------------
# Chat (se atiende antes del bucle de video)
# --------------------------------------------------------------------------
if enviada and pregunta.strip():
    with st.spinner("El asistente está pensando..."):
        r = asistente.responder(pregunta.strip(), ss.estado, ss.monitor.eventos)
    ss.chat.append((pregunta.strip(), r))
with contenedor_chat:
    for p, r in reversed(ss.chat[-6:]):
        st.markdown(f"🧑 **{p}**")
        st.markdown(f"🤖 {r['respuesta']}")
        st.caption(f"Fragmento extraído: “{r['fragmento']}” · confianza del modelo: {r['score']:.2f}")

render_alerta(ss.estado)
render_estado(ss.estado)
render_recomendaciones(ss.estado.tipo)
render_historial()
render_guardadas()
if ss.ultimo_frame is not None:
    mostrar_imagen(ph_frame, ss.ultimo_frame)


def procesar_frame(frame, t, marca, inmediato=False):
    detecciones = detector.detectar(frame, umbral)
    estado = ss.monitor.actualizar(detecciones, t, marca, inmediato=inmediato)
    from src.detector import dibujar
    anotado_bgr = dibujar(frame, detecciones, estado)
    ss.registro.actualizar(ss.monitor.eventos, anotado_bgr, nombre_fuente)
    anotado = cv2.cvtColor(anotado_bgr, cv2.COLOR_BGR2RGB)
    ss.estado, ss.ultimo_frame = estado, anotado
    return estado, anotado, detecciones


# --------------------------------------------------------------------------
# Imagen suelta
# --------------------------------------------------------------------------
if fuente == "Imagen":
    if archivo is not None:
        id_archivo = (archivo.name, archivo.size)
        if ss.imagen_id != id_archivo:
            frame = cv2.imdecode(np.frombuffer(archivo.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            estado, anotado, dets = procesar_frame(frame, time.time(),
                                                   datetime.now().strftime("%H:%M:%S"), inmediato=True)
            ss.imagen_id = id_archivo
            st.rerun()   # vuelve a dibujar todo el panel con el nuevo estado
    else:
        ph_info.info("Sube una imagen en la barra lateral.")

# --------------------------------------------------------------------------
# Video o cámara
# --------------------------------------------------------------------------
elif analizar:
    if fuente == "Archivo de video":
        if archivo is None:
            ph_info.info("Sube un video en la barra lateral.")
            st.stop()
        id_archivo = (archivo.name, archivo.size)
        video_nuevo = ss.video_id != id_archivo
        if video_nuevo or not (ss.video_ruta and os.path.exists(ss.video_ruta)):
            borrar_video_temporal(ss.video_ruta)        # el del video anterior
            ss.video_ruta = guardar_video_temporal(archivo.getvalue(), Path(archivo.name).suffix)
            ss.video_id = id_archivo
            if video_nuevo:
                ss.frame_idx = 0
        cap = cv2.VideoCapture(ss.video_ruta)
        cap.set(cv2.CAP_PROP_POS_FRAMES, ss.frame_idx)
        fps_video = cap.get(cv2.CAP_PROP_FPS) or 25.0
        fps_fuente, limite_descarte = fps_video, None
    else:
        backend = cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else cv2.CAP_ANY
        cap = cv2.VideoCapture(indice_camara, backend)
        fps_video = None
        fps_camara = cap.get(cv2.CAP_PROP_FPS)
        fps_fuente = fps_camara if fps_camara and fps_camara > 0 else 30.0
        limite_descarte = 0.5 / fps_fuente      # en vivo solo se vacía el búfer atrasado

    if not cap.isOpened():
        ph_info.error("No se pudo abrir la fuente de video.")
        st.stop()

    tipo_recomendado = ss.estado.tipo
    n, t_ini = 0, time.time()
    descartados, t_ciclo = 0, time.perf_counter()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                ph_info.info("Fin del video. Usa '⏮ Reiniciar video' para verlo de nuevo.")
                break
            if fps_video:                                   # archivo de video
                ss.frame_idx += 1
                t = ss.frame_idx / fps_video
                marca = f"el minuto {int(t // 60):02d}:{int(t % 60):02d} del video"
            else:                                           # cámara
                t = time.time()
                marca = "las " + datetime.now().strftime("%H:%M:%S")

            estado, anotado, dets = procesar_frame(frame, t, marca)
            mostrar_imagen(ph_frame, anotado)
            n += 1
            if n % 3 == 0:
                render_alerta(estado)
                render_estado(estado)
                render_historial()
                ph_info.caption(f"{n / (time.time() - t_ini):.1f} fotogramas analizados por segundo "
                                f"(1 de cada {descartados + 1}) · "
                                f"{len(dets)} detecciones en este fotograma")
            if estado.tipo != tipo_recomendado:     # el transformer solo se consulta al cambiar el evento
                render_alerta(estado)
                render_recomendaciones(estado.tipo)
                tipo_recomendado = estado.tipo

            # Salto de fotogramas: en CPU analizar cada uno deja el video en cámara lenta.
            ahora = time.perf_counter()
            a_saltar = (frames_a_saltar(ahora - t_ciclo, fps_fuente) if salto == "Auto"
                        else int(salto) - 1)
            t_ciclo = ahora
            descartados = descartar_frames(cap, a_saltar, limite_descarte)
            if fps_video:
                ss.frame_idx += descartados
    finally:
        cap.release()
        # deja en disco la duración final de la alerta que siga abierta
        ss.registro.actualizar(ss.monitor.eventos, fuente=nombre_fuente, forzar=True)
    render_alerta(ss.estado)
    render_estado(ss.estado)
    render_historial()
    render_guardadas()
elif ss.ultimo_frame is None:
    ph_info.info("Elige una fuente en la barra lateral y activa '▶ Analizar'.")
