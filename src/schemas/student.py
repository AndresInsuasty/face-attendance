"""Schemas Pydantic para estudiantes."""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_validator


class StudentCreate(BaseModel):
    """Datos para crear un estudiante."""

    nombre: str

    @field_validator("nombre")
    @classmethod
    def no_vacio(cls, v: str) -> str:
        """Valida que el nombre no sea vacío."""
        if not v.strip():
            raise ValueError("El nombre no puede estar vacío.")
        return v.strip()


class StudentRead(BaseModel):
    """Representación de un estudiante para lectura."""

    id: int
    nombre: str
    tiene_embedding: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

    @classmethod
    def from_orm_model(cls, obj: object) -> StudentRead:
        """Construye un StudentRead desde un modelo ORM."""
        from src.database.models import Estudiante

        est: Estudiante = obj  # type: ignore[assignment]
        return cls(
            id=est.id,
            nombre=est.nombre,
            tiene_embedding=est.embedding is not None,
            created_at=est.created_at,
        )
