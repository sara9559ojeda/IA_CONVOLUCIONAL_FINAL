"""Presentación visual de la aplicación, independiente de la detección."""
from html import escape
from pathlib import Path

import streamlit as st


def apply_theme():
    stylesheet = Path(__file__).resolve().parent.parent / "assets" / "interface.css"
    st.markdown(f"<style>{stylesheet.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def render_header(section="Probar detector"):
    descriptions = {
        "Probar detector": ("Prueba el detector", "Carga una imagen, un video o conecta tu cámara para analizar la escena."),
        "Datos": ("Los datos del proyecto", "Origen, preparación y revisión visual del conjunto de imágenes."),
        "Modelado": ("Cómo aprende el detector", "De los píxeles a las cajas: arquitectura, entrenamiento y predicción."),
        "Resultados": ("Qué dicen los resultados", "Métricas del modelo exportado y comportamiento de sus umbrales."),
    }
    title, description = descriptions[section]
    st.markdown(
        '<header class="workspace-header">'
        '<div><p class="eyebrow">DETECCIÓN DE FUEGO Y HUMO</p>'
        f'<h1>{escape(title)}</h1>'
        f'<p class="header-description">{escape(description)}</p></div></header>',
        unsafe_allow_html=True,
    )


def sidebar_brand():
    st.sidebar.markdown(
        '<div class="sidebar-brand">'
        '<div><div class="brand-name">Vigía</div>'
        '<div class="brand-caption">Fuego y humo</div></div></div>'
        '<p class="sidebar-intro">Detección visual de fuego y humo.</p>',
        unsafe_allow_html=True,
    )


def empty_viewer():
    return (
        '<div class="empty-viewer"><p class="eyebrow">VISOR DE DETECCIONES</p>'
        '<h3>Tu escena, aquí.</h3>'
        '<p>Selecciona una fuente en el panel lateral.<br>'
        'Aquí verás la imagen y sus detecciones.</p>'
        '<span class="viewer-formats">Cámara en vivo &nbsp; · &nbsp; Video &nbsp; · &nbsp; Imagen</span>'
        '</div>'
    )


def alert_markup(level, name):
    severity = {"NORMAL": "normal", "PRECAUCION": "caution", "ALERTA": "warning", "EMERGENCIA": "emergency"}
    description = "Sin fuego ni humo" if level == "NORMAL" else name
    return (
        f'<div class="status-banner status-{severity.get(level, "warning")}" role="status">'
        '<span class="status-dot" aria-hidden="true"></span>'
        f'<span class="status-level">{escape(level)}</span>'
        f'<span class="status-description">{escape(description)}</span></div>'
    )
