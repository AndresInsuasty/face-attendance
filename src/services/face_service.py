"""Servicio de reconocimiento facial usando DeepFace + ArcFace."""
from __future__ import annotations

import logging
import pickle

import numpy as np
from deepface import DeepFace

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Excepciones del dominio facial
# ---------------------------------------------------------------------------

class FaceServiceError(Exception):
    """Error base del servicio facial."""


class FaceNotDetectedError(FaceServiceError):
    """No se detectó ningún rostro en la imagen."""


class MultipleFacesError(FaceServiceError):
    """Se detectaron múltiples rostros cuando se esperaba uno solo."""


class InsufficientPhotosError(FaceServiceError):
    """No hay suficientes fotos con rostros detectados para el enrollment."""


def _face_area(representation: dict) -> float:  # type: ignore[type-arg]
    """Retorna el área del bounding box de un resultado de DeepFace.represent."""
    region = representation.get("facial_area", {})
    w = region.get("w", 0)
    h = region.get("h", 0)
    return float(w * h)


# ---------------------------------------------------------------------------
# Servicio principal
# ---------------------------------------------------------------------------

class FaceService:
    """Maneja generación de embeddings y matching con DeepFace + ArcFace.

    No entrena modelos: compara embeddings en espacio vectorial usando
    similitud coseno. Los embeddings ArcFace tienen 512 dimensiones.
    """

    def generate_embedding(self, image: np.ndarray) -> np.ndarray:
        """Genera un embedding ArcFace de 512 dims a partir de una imagen BGR.

        Cuando se detectan múltiples rostros, usa el de mayor área (el más
        prominente en la foto). Detector: RetinaFace, mucho más preciso que
        los Haar Cascades de opencv.

        Args:
            image: Imagen en formato BGR (como la provee OpenCV).

        Returns:
            np.ndarray de shape (512,) normalizado.

        Raises:
            FaceNotDetectedError: Si no se detecta ningún rostro.
        """
        # DeepFace espera RGB
        image_rgb = image[:, :, ::-1] if image.ndim == 3 else image

        try:
            result = DeepFace.represent(
                img_path=image_rgb,
                model_name="ArcFace",
                detector_backend="retinaface",
                enforce_detection=True,
                align=True,
            )
        except ValueError as exc:
            raise FaceNotDetectedError(
                "No se detectó ningún rostro en la imagen."
            ) from exc
        except Exception as exc:
            logger.exception("Error inesperado en DeepFace.represent")
            raise FaceNotDetectedError(
                f"Error al procesar la imagen: {exc}"
            ) from exc

        if not result:
            raise FaceNotDetectedError("DeepFace no retornó embeddings.")

        if len(result) > 1:
            # Tomar el rostro con mayor área de región facial
            logger.warning(
                "%d rostros detectados, usando el de mayor área.", len(result)
            )
            result = [max(result, key=lambda r: _face_area(r))]

        embedding = np.array(result[0]["embedding"], dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding

    def embedding_to_bytes(self, embedding: np.ndarray) -> bytes:
        """Serializa un embedding numpy a bytes con pickle.

        Args:
            embedding: Array numpy a serializar.

        Returns:
            Bytes serializados.
        """
        return pickle.dumps(embedding)

    def bytes_to_embedding(self, data: bytes) -> np.ndarray:
        """Deserializa un embedding desde bytes pickle.

        Args:
            data: Bytes serializados con pickle.

        Returns:
            Array numpy restaurado.
        """
        result: np.ndarray = pickle.loads(data)  # noqa: S301 — datos internos de confianza
        return result

    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calcula similitud coseno entre dos embeddings normalizados.

        Args:
            emb1: Primer embedding (shape 512,).
            emb2: Segundo embedding (shape 512,).

        Returns:
            Float entre 0.0 y 1.0 donde 1.0 significa vectores idénticos.
        """
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        similarity = float(np.dot(emb1, emb2) / (norm1 * norm2))
        # Clampear al rango [0, 1] para evitar valores negativos por ruido numérico
        return max(0.0, min(1.0, (similarity + 1.0) / 2.0))

    def enroll_student(
        self,
        images: list[np.ndarray],
        min_photos: int = 3,
    ) -> np.ndarray:
        """Genera un embedding promedio a partir de múltiples fotos del estudiante.

        Args:
            images: Lista de imágenes BGR (OpenCV).
            min_photos: Número mínimo de fotos con rostro detectado requeridas.

        Returns:
            Embedding promedio normalizado de shape (512,).

        Raises:
            InsufficientPhotosError: Si hay menos fotos válidas que min_photos.
        """
        embeddings: list[np.ndarray] = []
        errores: list[str] = []

        for i, img in enumerate(images):
            try:
                emb = self.generate_embedding(img)
                embeddings.append(emb)
            except FaceServiceError as exc:
                errores.append(f"Imagen {i + 1}: {exc}")
                logger.warning("Imagen %d descartada en enrollment: %s", i + 1, exc)

        if len(embeddings) < min_photos:
            raise InsufficientPhotosError(
                f"Se necesitan al menos {min_photos} fotos con rostros detectados, "
                f"pero solo se obtuvieron {len(embeddings)}. "
                f"Problemas: {'; '.join(errores)}"
            )

        mean_embedding = np.mean(np.stack(embeddings), axis=0).astype(np.float32)
        norm = np.linalg.norm(mean_embedding)
        if norm > 0:
            mean_embedding = mean_embedding / norm
        logger.info(
            "Enrollment completado: %d fotos válidas de %d subidas.",
            len(embeddings),
            len(images),
        )
        return mean_embedding

    def merge_embeddings(
        self, existing: np.ndarray, new_ones: list[np.ndarray]
    ) -> np.ndarray:
        """Promedia el embedding existente con los nuevos para refinar el perfil facial.

        Args:
            existing: Embedding actual del estudiante.
            new_ones: Lista de embeddings nuevos a incorporar.

        Returns:
            Embedding promedio normalizado de shape (512,).
        """
        if not new_ones:
            return existing
        all_embs = [existing, *new_ones]
        merged = np.mean(np.stack(all_embs), axis=0).astype(np.float32)
        norm = np.linalg.norm(merged)
        if norm > 0:
            merged = merged / norm
        return merged

    def identify(
        self,
        probe_image: np.ndarray,
        gallery: list[tuple[int, np.ndarray]],
        threshold: float,
    ) -> tuple[int, float] | None:
        """Identifica a qué estudiante pertenece la imagen de prueba.

        Args:
            probe_image: Imagen BGR del estudiante a identificar.
            gallery: Lista de tuplas (student_id, embedding) contra las que comparar.
            threshold: Similitud mínima (0.0–1.0) para considerar un match válido.

        Returns:
            Tupla (student_id, confianza) del mejor match si supera el umbral,
            o None si no hay match suficiente.
        """
        if not gallery:
            return None

        try:
            probe_emb = self.generate_embedding(probe_image)
        except FaceServiceError:
            raise

        best_id: int | None = None
        best_similarity = -1.0

        for student_id, ref_emb in gallery:
            sim = self.compute_similarity(probe_emb, ref_emb)
            if sim > best_similarity:
                best_similarity = sim
                best_id = student_id

        if best_id is not None and best_similarity >= threshold:
            logger.info(
                "Match encontrado: student_id=%d similitud=%.4f",
                best_id,
                best_similarity,
            )
            return best_id, best_similarity

        logger.info(
            "Sin match: mejor similitud=%.4f < umbral=%.4f", best_similarity, threshold
        )
        return None
