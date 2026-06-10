"""Modelos SQLAlchemy 2.x para el sistema de reconocimiento facial."""
from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, LargeBinary, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def _utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    """Base declarativa para todos los modelos."""


class Estudiante(Base):
    """Representa un estudiante con su embedding facial."""

    __tablename__ = "estudiantes"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    nombre: Mapped[str] = mapped_column(String(200), nullable=False)
    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    foto_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    activo: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=_utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return f"<Estudiante id={self.id} nombre={self.nombre!r}>"
