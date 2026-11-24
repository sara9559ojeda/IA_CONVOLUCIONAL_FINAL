from pathlib import Path

import streamlit as st


capture = st.components.v2.component(
    "vigia_camera",
    html="<vigia-browser-camera></vigia-browser-camera>",
    js=Path(__file__).with_suffix(".js").read_text(encoding="utf-8"),
)
