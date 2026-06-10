"""Página para gestionar estudiantes: registro, agregar fotos y listado."""
from __future__ import annotations

import contextlib
import io
import logging

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.config import settings
from src.database.connection import get_db
from src.repositories import student_repo
from src.schemas.student import StudentCreate
from src.services.face_service import (
    FaceService,
    FaceServiceError,
    InsufficientPhotosError,
)

logger = logging.getLogger(__name__)
face_service = FaceService()


@st.cache_data(show_spinner=False)
def _preview_deteccion(image_bytes: bytes) -> tuple[bytes, bool, str]:
    """Detecta rostros y retorna imagen anotada con bounding boxes.

    Cacheada por bytes de imagen para evitar re-procesar la misma foto.
    Verde = rostro principal, amarillo = rostros secundarios ignorados.
    """
    from deepface import DeepFace

    pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img_np = np.array(pil)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_h, img_w = img_bgr.shape[:2]

    try:
        faces = DeepFace.extract_faces(
            img_path=img_np,
            detector_backend="retinaface",
            enforce_detection=True,
            align=False,
        )
        if not faces:
            raise ValueError("Sin rostros")

        # Ordenar de mayor a menor área
        faces_sorted = sorted(
            faces,
            key=lambda f: f.get("facial_area", {}).get("w", 0)
            * f.get("facial_area", {}).get("h", 0),
            reverse=True,
        )

        for i, face in enumerate(faces_sorted):
            area = face.get("facial_area", {})
            x = max(0, area.get("x", 0))
            y = max(0, area.get("y", 0))
            w = max(1, area.get("w", 0))
            h = max(1, area.get("h", 0))
            x2 = min(x + w, img_w - 1)
            y2 = min(y + h, img_h - 1)
            color = (34, 197, 94) if i == 0 else (251, 191, 36)  # verde / amarillo
            cv2.rectangle(img_bgr, (x, y), (x2, y2), color, 3)

        n = len(faces_sorted)
        msg = "✅ Rostro detectado" if n == 1 else f"⚠️ {n} rostros — se usa el principal"

    except Exception:
        # Borde rojo: sin rostro
        cv2.rectangle(img_bgr, (0, 0), (img_w - 1, img_h - 1), (239, 68, 68), 5)
        msg = "❌ Sin rostro detectado"
        annotated = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        buf = io.BytesIO()
        annotated.save(buf, format="JPEG", quality=85)
        return buf.getvalue(), False, msg

    annotated = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    buf = io.BytesIO()
    annotated.save(buf, format="JPEG", quality=85)
    return buf.getvalue(), True, msg


def _mostrar_previews(fotos: list) -> int:  # type: ignore[type-arg]
    """Muestra grid de fotos con bounding boxes. Retorna cantidad de fotos con rostro."""
    cols = st.columns(min(len(fotos), 4))
    validas = 0
    for i, foto in enumerate(fotos):
        annotated_bytes, found, msg = _preview_deteccion(foto.getvalue())
        with cols[i % 4]:
            st.image(annotated_bytes, use_column_width=True)
            if found:
                st.success(msg)
                validas += 1
            else:
                st.error(msg)
    return validas


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Estudiantes", page_icon="👤", layout="wide")
st.title("👤 Estudiantes")

tab_nuevo, tab_agregar, tab_lista = st.tabs(
    ["Nuevo estudiante", "Agregar fotos", "Lista"]
)

# ---------------------------------------------------------------------------
# Tab 1: Registrar nuevo estudiante
# ---------------------------------------------------------------------------
with tab_nuevo:
    st.subheader("Registrar nuevo estudiante")

    nombre = st.text_input(
        "Nombre completo *", placeholder="Ej: Ana García", key="nombre_nuevo"
    )
    fotos_nuevas = st.file_uploader(
        f"Fotos de referencia (mínimo {settings.ENROLLMENT_PHOTOS_MIN})",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        key="fotos_nuevo",
    )

    fotos_validas_nuevo = 0
    if fotos_nuevas:
        with st.spinner("Detectando rostros..."):
            fotos_validas_nuevo = _mostrar_previews(fotos_nuevas)
        st.caption(
            f"**{fotos_validas_nuevo} de {len(fotos_nuevas)}** fotos con rostro detectado."
        )
        if fotos_validas_nuevo < settings.ENROLLMENT_PHOTOS_MIN:
            st.warning(
                f"Necesitas al menos **{settings.ENROLLMENT_PHOTOS_MIN}** fotos válidas."
            )

    puede_registrar = (
        bool(nombre.strip()) and fotos_validas_nuevo >= settings.ENROLLMENT_PHOTOS_MIN
    )

    if st.button(
        "Registrar estudiante",
        disabled=not puede_registrar,
        type="primary",
        use_container_width=True,
        key="btn_registrar",
    ):
        with st.spinner("Generando embedding facial..."):
            try:
                imagenes = [
                    cv2.cvtColor(
                        np.array(Image.open(f).convert("RGB")), cv2.COLOR_RGB2BGR
                    )
                    for f in fotos_nuevas
                ]
                embedding = face_service.enroll_student(
                    imagenes, min_photos=settings.ENROLLMENT_PHOTOS_MIN
                )
                emb_bytes = face_service.embedding_to_bytes(embedding)

                with get_db() as db:
                    est = student_repo.create_student(
                        db, StudentCreate(nombre=nombre.strip())
                    )
                    foto_dir = settings.FACES_DIR / str(est.id)
                    foto_dir.mkdir(parents=True, exist_ok=True)
                    Image.open(fotos_nuevas[0]).convert("RGB").save(
                        foto_dir / "principal.jpg"
                    )
                    student_repo.update_student_embedding(
                        db, est.id, emb_bytes, f"faces/{est.id}/principal.jpg"
                    )

                st.success(f"**{nombre.strip()}** registrado exitosamente.")
                st.balloons()
            except InsufficientPhotosError as exc:
                st.error(f"Fotos insuficientes: {exc}")
            except FaceServiceError as exc:
                st.error(f"Error de reconocimiento facial: {exc}")
            except Exception as exc:
                logger.exception("Error al registrar estudiante")
                st.error(f"Error inesperado: {exc}")

# ---------------------------------------------------------------------------
# Tab 2: Agregar fotos a estudiante existente
# ---------------------------------------------------------------------------
with tab_agregar:
    st.subheader("Agregar fotos a estudiante existente")
    st.info(
        "Las fotos nuevas se promedian con el embedding actual para mejorar el reconocimiento.",
        icon="ℹ️",
    )

    with get_db() as db:
        todos = student_repo.list_students(db)

    if not todos:
        st.warning("No hay estudiantes registrados aún.")
    else:
        opciones_agr = {e.nombre: e for e in todos}
        seleccion_agr = st.selectbox(
            "Seleccionar estudiante", list(opciones_agr.keys()), key="sel_agregar"
        )
        est_sel = opciones_agr[seleccion_agr]

        # Foto de referencia actual
        foto_ref = settings.FACES_DIR / str(est_sel.id) / "principal.jpg"
        col_ref, col_estado = st.columns([1, 2])
        with col_ref:
            if foto_ref.exists():
                st.image(str(foto_ref), caption="Foto actual", width=140)
            else:
                st.caption("Sin foto de referencia")
        with col_estado:
            if est_sel.embedding is not None:
                st.success("Este estudiante tiene embedding registrado.")
            else:
                st.warning("Sin embedding — sube al menos 1 foto para generarlo.")

        fotos_agr = st.file_uploader(
            "Fotos adicionales (mínimo 1)",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            key="fotos_agregar",
        )

        fotos_validas_agr = 0
        if fotos_agr:
            with st.spinner("Detectando rostros..."):
                fotos_validas_agr = _mostrar_previews(fotos_agr)
            st.caption(
                f"**{fotos_validas_agr} de {len(fotos_agr)}** fotos con rostro detectado."
            )

        if st.button(
            "Actualizar embedding",
            disabled=fotos_validas_agr == 0,
            type="primary",
            use_container_width=True,
            key="btn_agregar",
        ):
            with st.spinner("Actualizando embedding..."):
                try:
                    imagenes = [
                        cv2.cvtColor(
                            np.array(Image.open(f).convert("RGB")), cv2.COLOR_RGB2BGR
                        )
                        for f in fotos_agr
                    ]
                    nuevos_embs: list[np.ndarray] = []
                    for img in imagenes:
                        with contextlib.suppress(FaceServiceError):
                            nuevos_embs.append(face_service.generate_embedding(img))

                    if not nuevos_embs:
                        st.error("Ninguna foto nueva tenía rostro detectable.")
                    else:
                        with get_db() as db:
                            est_actual = student_repo.get_student_by_id(
                                db, est_sel.id
                            )
                            if est_actual and est_actual.embedding:
                                viejo = face_service.bytes_to_embedding(
                                    est_actual.embedding
                                )
                                merged = face_service.merge_embeddings(viejo, nuevos_embs)
                            else:
                                # Sin embedding previo: promediar solo los nuevos
                                merged = face_service.merge_embeddings(
                                    nuevos_embs[0], nuevos_embs[1:]
                                )

                            emb_bytes = face_service.embedding_to_bytes(merged)
                            foto_dir = settings.FACES_DIR / str(est_sel.id)
                            foto_dir.mkdir(parents=True, exist_ok=True)
                            Image.open(fotos_agr[0]).convert("RGB").save(
                                foto_dir / "principal.jpg"
                            )
                            student_repo.update_student_embedding(
                                db,
                                est_sel.id,
                                emb_bytes,
                                f"faces/{est_sel.id}/principal.jpg",
                            )

                        st.success(
                            f"Embedding de **{seleccion_agr}** actualizado "
                            f"con {len(nuevos_embs)} foto(s) nueva(s)."
                        )
                except Exception as exc:
                    logger.exception("Error al actualizar embedding")
                    st.error(f"Error inesperado: {exc}")

# ---------------------------------------------------------------------------
# Tab 3: Lista de estudiantes
# ---------------------------------------------------------------------------
with tab_lista:
    st.subheader("Estudiantes registrados")

    with get_db() as db:
        estudiantes = student_repo.list_students(db)

    if not estudiantes:
        st.info("No hay estudiantes registrados todavía.")
    else:
        datos = [
            {
                "ID": e.id,
                "Nombre": e.nombre,
                "Embedding": "✅" if e.embedding is not None else "❌",
                "Registrado": e.created_at.strftime("%d/%m/%Y"),
            }
            for e in estudiantes
        ]
        st.dataframe(datos, use_container_width=True, hide_index=True)
        st.caption(f"Total: {len(datos)} estudiante(s)")

        st.markdown("---")
        st.subheader("Eliminar estudiante")
        opciones_del = {e.nombre: e.id for e in estudiantes}
        sel_del = st.selectbox(
            "Seleccionar", list(opciones_del.keys()), key="sel_eliminar"
        )
        if st.button("Eliminar", type="secondary", key="btn_eliminar"):
            with get_db() as db:
                student_repo.delete_student(db, opciones_del[sel_del])
            st.success(f"**{sel_del}** eliminado.")
            st.rerun()
