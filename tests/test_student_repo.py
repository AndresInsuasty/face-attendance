"""Tests para el repositorio de estudiantes."""
from __future__ import annotations

from sqlalchemy.orm import Session

from src.repositories import student_repo
from src.schemas.student import StudentCreate


class TestStudentRepo:
    """Tests de CRUD para estudiantes."""

    def test_create_and_retrieve_by_id(self, db_session: Session) -> None:
        """Crear un estudiante y recuperarlo por ID."""
        est = student_repo.create_student(db_session, StudentCreate(nombre="Ana García"))
        db_session.commit()

        found = student_repo.get_student_by_id(db_session, est.id)
        assert found is not None
        assert found.nombre == "Ana García"

    def test_create_student_no_embedding_by_default(self, db_session: Session) -> None:
        """El estudiante recién creado no tiene embedding."""
        est = student_repo.create_student(db_session, StudentCreate(nombre="Carlos"))
        db_session.commit()
        assert est.embedding is None
        assert est.activo is True

    def test_get_nonexistent_student(self, db_session: Session) -> None:
        """Buscar un ID inexistente retorna None."""
        assert student_repo.get_student_by_id(db_session, 9999) is None

    def test_list_students_only_active(self, db_session: Session) -> None:
        """list_students solo retorna activos por defecto."""
        s1 = student_repo.create_student(db_session, StudentCreate(nombre="Activo"))
        s2 = student_repo.create_student(db_session, StudentCreate(nombre="Inactivo"))
        db_session.commit()
        s2.activo = False
        db_session.commit()

        activos = student_repo.list_students(db_session, only_active=True)
        ids = [s.id for s in activos]
        assert s1.id in ids
        assert s2.id not in ids

    def test_update_embedding(self, db_session: Session) -> None:
        """update_student_embedding almacena los bytes correctamente."""
        est = student_repo.create_student(db_session, StudentCreate(nombre="Pedro"))
        db_session.commit()

        fake_bytes = b"fake_embedding_data"
        student_repo.update_student_embedding(db_session, est.id, fake_bytes, "faces/1/principal.jpg")
        db_session.commit()

        actualizado = student_repo.get_student_by_id(db_session, est.id)
        assert actualizado is not None
        assert actualizado.embedding == fake_bytes

    def test_delete_student(self, db_session: Session) -> None:
        """delete_student elimina el estudiante de la BD."""
        est = student_repo.create_student(db_session, StudentCreate(nombre="Temporal"))
        db_session.commit()

        student_repo.delete_student(db_session, est.id)
        db_session.commit()

        assert student_repo.get_student_by_id(db_session, est.id) is None

    def test_get_students_with_embeddings(self, db_session: Session) -> None:
        """get_students_with_embeddings solo retorna los que tienen embedding."""
        s1 = student_repo.create_student(db_session, StudentCreate(nombre="Con foto"))
        s2 = student_repo.create_student(db_session, StudentCreate(nombre="Sin foto"))
        db_session.commit()

        student_repo.update_student_embedding(db_session, s1.id, b"emb", "path")
        db_session.commit()

        con_emb = student_repo.get_students_with_embeddings(db_session)
        ids = [e.id for e in con_emb]
        assert s1.id in ids
        assert s2.id not in ids
