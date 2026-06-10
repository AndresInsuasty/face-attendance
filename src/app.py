"""Entrypoint principal de la aplicación Streamlit."""
from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="Reconocimiento Facial",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.database.connection import init_db  # noqa: E402

init_db()

st.title("🎓 Sistema de Reconocimiento Facial")
st.markdown(
    """
    Selecciona una opción en el menú lateral:

    - **Estudiantes** — Registrar estudiantes y cargar sus fotos de referencia.
    - **Identificar** — Sube una foto o usa la cámara para identificar a un estudiante.
    """
)
