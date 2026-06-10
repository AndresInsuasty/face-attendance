"""Configuración central de la aplicación usando pydantic-settings."""
from __future__ import annotations

import logging
from pathlib import Path

from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Configuración de la aplicación cargada desde variables de entorno o .env."""

    # General
    APP_NAME: str = "Sistema de Asistencia Facial"
    DEBUG: bool = False

    # Rutas base
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    DB_PATH: Path = DATA_DIR / "db" / "attendance.db"
    FACES_DIR: Path = DATA_DIR / "faces"

    # Base de datos
    DATABASE_URL: str = ""

    # Reconocimiento facial
    DEEPFACE_MODEL: str = "ArcFace"
    DEEPFACE_DETECTOR: str = "retinaface"
    SIMILARITY_THRESHOLD: float = 0.40
    ENROLLMENT_PHOTOS_MIN: int = 3

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    def model_post_init(self, __context: object) -> None:
        """Crea directorios necesarios y construye DATABASE_URL si está vacío."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.FACES_DIR.mkdir(parents=True, exist_ok=True)
        (self.DATA_DIR / "db").mkdir(parents=True, exist_ok=True)
        if not self.DATABASE_URL:
            object.__setattr__(
                self, "DATABASE_URL", f"sqlite:///{self.DB_PATH}"
            )
        log_level = logging.DEBUG if self.DEBUG else logging.INFO
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )


settings = Settings()
