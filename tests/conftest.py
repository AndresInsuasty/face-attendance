"""Fixtures compartidas para los tests."""
from __future__ import annotations

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.database.models import Base


@pytest.fixture(scope="function")
def db_session() -> Session:
    """Provee una sesión de BD en memoria para cada test (aislada)."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)
    session: Session = session_factory()
    yield session  # type: ignore[misc]
    session.close()
    Base.metadata.drop_all(engine)
