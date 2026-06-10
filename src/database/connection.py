"""Conexión y sesión SQLAlchemy para la aplicación."""
from __future__ import annotations

import logging
from collections.abc import Generator
from contextlib import contextmanager

import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import settings
from src.database.models import Base

logger = logging.getLogger(__name__)


@st.cache_resource
def _get_engine():  # type: ignore[no-untyped-def]
    """Crea el engine SQLAlchemy una sola vez usando cache de Streamlit."""
    return create_engine(
        settings.DATABASE_URL,
        connect_args={"check_same_thread": False},
        echo=settings.DEBUG,
    )


def _get_session_factory():  # type: ignore[no-untyped-def]
    """Retorna la fábrica de sesiones ligada al engine cacheado."""
    return sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=_get_engine(),
        # evita DetachedInstanceError cuando se usan objetos fuera del with
        expire_on_commit=False,
    )


def init_db() -> None:
    """Crea todas las tablas si no existen."""
    Base.metadata.create_all(bind=_get_engine())
    logger.info("Base de datos inicializada correctamente.")


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Context manager que provee una sesión de BD con manejo de transacciones."""
    session_factory = _get_session_factory()
    db: Session = session_factory()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()
