"""Tests para FaceService — operaciones sin imágenes reales."""
from __future__ import annotations

import numpy as np
import pytest

from src.services.face_service import FaceService, InsufficientPhotosError


@pytest.fixture()
def service() -> FaceService:
    """Instancia de FaceService para los tests."""
    return FaceService()


@pytest.fixture()
def random_embedding() -> np.ndarray:
    """Embedding aleatorio de 512 dimensiones."""
    rng = np.random.default_rng(42)
    vec = rng.standard_normal(512).astype(np.float32)
    return vec / np.linalg.norm(vec)


class TestEmbeddingSerialization:
    """Tests de serialización/deserialización de embeddings."""

    def test_round_trip(self, service: FaceService, random_embedding: np.ndarray) -> None:
        """embedding_to_bytes y bytes_to_embedding son inversos exactos."""
        serialized = service.embedding_to_bytes(random_embedding)
        restored = service.bytes_to_embedding(serialized)
        np.testing.assert_array_equal(random_embedding, restored)

    def test_bytes_type(self, service: FaceService, random_embedding: np.ndarray) -> None:
        """embedding_to_bytes retorna bytes."""
        result = service.embedding_to_bytes(random_embedding)
        assert isinstance(result, bytes)


class TestComputeSimilarity:
    """Tests para el cálculo de similitud coseno."""

    def test_identical_vectors(self, service: FaceService) -> None:
        """Vectores idénticos tienen similitud 1.0."""
        rng = np.random.default_rng(0)
        vec = rng.standard_normal(512).astype(np.float32)
        sim = service.compute_similarity(vec, vec)
        assert abs(sim - 1.0) < 1e-5

    def test_orthogonal_vectors(self, service: FaceService) -> None:
        """Vectores ortogonales tienen similitud aproximadamente 0.5 (mapeado de [-1,1] a [0,1])."""
        vec1 = np.zeros(512, dtype=np.float32)
        vec2 = np.zeros(512, dtype=np.float32)
        vec1[0] = 1.0
        vec2[1] = 1.0
        sim = service.compute_similarity(vec1, vec2)
        # cos(90°) = 0, mapeado a (0+1)/2 = 0.5
        assert abs(sim - 0.5) < 1e-5

    def test_opposite_vectors(self, service: FaceService) -> None:
        """Vectores opuestos tienen similitud 0.0."""
        vec = np.ones(512, dtype=np.float32)
        sim = service.compute_similarity(vec, -vec)
        assert abs(sim - 0.0) < 1e-5

    def test_zero_vector_returns_zero(self, service: FaceService) -> None:
        """Un vector cero retorna similitud 0.0 sin error."""
        zero = np.zeros(512, dtype=np.float32)
        vec = np.ones(512, dtype=np.float32)
        assert service.compute_similarity(zero, vec) == 0.0
        assert service.compute_similarity(vec, zero) == 0.0

    def test_similarity_range(self, service: FaceService) -> None:
        """La similitud siempre está entre 0.0 y 1.0."""
        rng = np.random.default_rng(99)
        for _ in range(20):
            a = rng.standard_normal(512).astype(np.float32)
            b = rng.standard_normal(512).astype(np.float32)
            sim = service.compute_similarity(a, b)
            assert 0.0 <= sim <= 1.0


class TestEnrollStudent:
    """Tests para el proceso de enrollment."""

    def test_insufficient_photos_raises(self, service: FaceService) -> None:
        """enroll_student lanza InsufficientPhotosError con menos fotos que el mínimo."""
        # Pasamos imágenes vacías que no tienen rostro detectable
        # Mockeamos generate_embedding para simular falla de detección
        dummy_images = [np.zeros((100, 100, 3), dtype=np.uint8)]

        with pytest.raises(InsufficientPhotosError):
            # min_photos=3, solo 1 imagen que fallará la detección
            service.enroll_student(dummy_images, min_photos=3)

    def test_insufficient_photos_custom_min(self, service: FaceService) -> None:
        """InsufficientPhotosError respeta el parámetro min_photos."""
        dummy_images = [np.zeros((100, 100, 3), dtype=np.uint8)] * 2

        with pytest.raises(InsufficientPhotosError):
            service.enroll_student(dummy_images, min_photos=5)
