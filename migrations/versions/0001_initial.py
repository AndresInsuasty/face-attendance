"""Migración inicial: crea tablas cursos, estudiantes, estudiante_curso, asistencias.

Revision ID: 0001
Revises:
Create Date: 2024-01-01 00:00:00.000000
"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Crea todas las tablas del sistema."""
    op.create_table(
        "cursos",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("nombre", sa.String(200), nullable=False),
        sa.Column("codigo", sa.String(50), nullable=False),
        sa.Column("descripcion", sa.String(500), nullable=True),
        sa.Column("activo", sa.Boolean(), nullable=False, default=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("codigo"),
    )

    op.create_table(
        "estudiantes",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("nombre", sa.String(100), nullable=False),
        sa.Column("apellido", sa.String(100), nullable=False),
        sa.Column("codigo", sa.String(50), nullable=False),
        sa.Column("email", sa.String(200), nullable=True),
        sa.Column("embedding", sa.LargeBinary(), nullable=True),
        sa.Column("foto_path", sa.String(500), nullable=True),
        sa.Column("activo", sa.Boolean(), nullable=False, default=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("codigo"),
    )

    op.create_table(
        "estudiante_curso",
        sa.Column("estudiante_id", sa.Integer(), nullable=False),
        sa.Column("curso_id", sa.Integer(), nullable=False),
        sa.Column("inscrito_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["curso_id"], ["cursos.id"]),
        sa.ForeignKeyConstraint(["estudiante_id"], ["estudiantes.id"]),
        sa.PrimaryKeyConstraint("estudiante_id", "curso_id"),
    )

    op.create_table(
        "asistencias",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("estudiante_id", sa.Integer(), nullable=False),
        sa.Column("curso_id", sa.Integer(), nullable=False),
        sa.Column("fecha", sa.Date(), nullable=False),
        sa.Column("hora", sa.Time(), nullable=False),
        sa.Column("confianza", sa.Float(), nullable=False),
        sa.Column("metodo", sa.String(20), nullable=False, default="facial"),
        sa.Column("registrado_por", sa.String(200), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.ForeignKeyConstraint(["curso_id"], ["cursos.id"]),
        sa.ForeignKeyConstraint(["estudiante_id"], ["estudiantes.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "estudiante_id", "curso_id", "fecha", name="uq_asistencia_dia"
        ),
    )


def downgrade() -> None:
    """Elimina todas las tablas en orden inverso."""
    op.drop_table("asistencias")
    op.drop_table("estudiante_curso")
    op.drop_table("estudiantes")
    op.drop_table("cursos")
