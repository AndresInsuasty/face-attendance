"""Página para identificar a qué estudiante pertenece una foto."""
from __future__ import annotations

import logging

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.config import settings
from src.database.connection import get_db
from src.repositories import student_repo
from src.services.face_service import FaceNotDetectedError, FaceService

logger = logging.getLogger(__name__)
face_service = FaceService()

st.set_page_config(page_title="Identificar", page_icon="🔍", layout="wide")
st.title("🔍 Identificar estudiante")

# ---------------------------------------------------------------------------
# Verificar que hay estudiantes con embedding
# ---------------------------------------------------------------------------
with get_db() as db:
    estudiantes = student_repo.get_students_with_embeddings(db)

if not estudiantes:
    st.warning(
        "No hay estudiantes con fotos registradas. "
        "Ve a **Estudiantes** para registrar algunos primero."
    )
    st.stop()

st.info(f"Base de datos: **{len(estudiantes)}** estudiante(s) con foto registrada.", icon="ℹ️")

# Construir galería en memoria
gallery: list[tuple[int, str, np.ndarray]] = [
    (e.id, e.nombre, face_service.bytes_to_embedding(e.embedding))  # type: ignore[arg-type]
    for e in estudiantes
]

# ---------------------------------------------------------------------------
# Captura de imagen
# ---------------------------------------------------------------------------
col_entrada, col_resultado = st.columns([1, 1])

with col_entrada:
    st.subheader("Imagen a identificar")
    modo = st.radio("Modo", ["Subir foto", "Cámara"], horizontal=True)

    imagen_bgr: np.ndarray | None = None

    if modo == "Subir foto":
        archivo = st.file_uploader("Seleccionar imagen", type=["jpg", "jpeg", "png"])
        if archivo:
            pil = Image.open(archivo).convert("RGB")
            st.image(pil, use_column_width=True)
            imagen_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    else:
        captura = st.camera_input("Tomar foto")
        if captura:
            pil = Image.open(captura).convert("RGB")
            imagen_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)

    identificar = st.button(
        "Identificar",
        disabled=imagen_bgr is None,
        use_container_width=True,
        type="primary",
    )

# ---------------------------------------------------------------------------
# Identificación
# ---------------------------------------------------------------------------
with col_resultado:
    st.subheader("Resultado")

    if identificar and imagen_bgr is not None:
        with st.spinner("Analizando..."):
            try:
                gallery_emb = [(sid, emb) for sid, _, emb in gallery]
                match = face_service.identify(
                    probe_image=imagen_bgr,
                    gallery=gallery_emb,
                    threshold=settings.SIMILARITY_THRESHOLD,
                )

                if match is None:
                    st.error("No se reconoció a ningún estudiante registrado.")
                else:
                    student_id, confianza = match
                    nombre_match = next(n for sid, n, _ in gallery if sid == student_id)

                    st.success("Estudiante identificado")
                    st.markdown(f"## {nombre_match}")
                    st.metric("Confianza", f"{confianza * 100:.1f}%")

                    # Mostrar foto de referencia si existe
                    foto_ref = settings.FACES_DIR / str(student_id) / "principal.jpg"
                    if foto_ref.exists():
                        st.image(str(foto_ref), caption="Foto de referencia", width=200)

            except FaceNotDetectedError:
                st.error("No se detectó un rostro claro en la imagen.")
            except Exception as exc:
                logger.exception("Error inesperado en identificación")
                st.error(f"Error inesperado: {exc}")
    elif not identificar:
        st.markdown("Carga una imagen y presiona **Identificar**.")
