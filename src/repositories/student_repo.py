"""Repositorio CRUD para estudiantes."""
from __future__ import annotations

import logging

from sqlalchemy.orm import Session

from src.database.models import Estudiante
from src.schemas.student import StudentCreate

logger = logging.getLogger(__name__)


def create_student(db: Session, data: StudentCreate) -> Estudiante:
    """Crea un nuevo estudiante en la base de datos."""
    estudiante = Estudiante(nombre=data.nombre)
    db.add(estudiante)
    db.flush()
    logger.info("Estudiante creado: %s", data.nombre)
    return estudiante


def get_student_by_id(db: Session, student_id: int) -> Estudiante | None:
    """Recupera un estudiante por su ID."""
    return db.get(Estudiante, student_id)


def list_students(db: Session, only_active: bool = True) -> list[Estudiante]:
    """Lista estudiantes, opcionalmente solo los activos."""
    q = db.query(Estudiante)
    if only_active:
        q = q.filter(Estudiante.activo == True)  # noqa: E712
    return q.order_by(Estudiante.nombre).all()


def update_student_embedding(
    db: Session, student_id: int, embedding: bytes, foto_path: str
) -> Estudiante:
    """Actualiza el embedding y ruta de foto de un estudiante."""
    estudiante = db.get(Estudiante, student_id)
    if estudiante is None:
        raise ValueError(f"Estudiante id={student_id} no encontrado.")
    estudiante.embedding = embedding
    estudiante.foto_path = foto_path
    db.flush()
    logger.info("Embedding actualizado para estudiante id=%d", student_id)
    return estudiante


def get_students_with_embeddings(db: Session) -> list[Estudiante]:
    """Recupera estudiantes activos que tienen embedding registrado."""
    return (
        db.query(Estudiante)
        .filter(Estudiante.activo == True, Estudiante.embedding.isnot(None))  # noqa: E712
        .all()
    )


def delete_student(db: Session, student_id: int) -> None:
    """Elimina un estudiante de la base de datos."""
    estudiante = db.get(Estudiante, student_id)
    if estudiante is None:
        raise ValueError(f"Estudiante id={student_id} no encontrado.")
    db.delete(estudiante)
    db.flush()
    logger.info("Estudiante id=%d eliminado.", student_id)
