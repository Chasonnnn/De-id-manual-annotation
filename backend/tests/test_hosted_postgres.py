from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

import pytest
from hosted_app.database import create_schema
from hosted_app.domain import DocumentImport, RevisionConflict, Role
from hosted_app.repository import HostedRepository
from sqlalchemy import text
from sqlmodel import Session, create_engine


def test_postgresql_serializes_same_revision_saves() -> None:
    database_url = os.environ.get("TEST_POSTGRES_URL")
    if not database_url:
        pytest.skip("TEST_POSTGRES_URL is not configured")
    if os.environ.get("TEST_POSTGRES_ALLOW_EPHEMERAL_SCHEMA") != "true":
        pytest.skip("temporary PostgreSQL schema creation is not explicitly enabled")

    schema_name = f"annotation_test_{uuid4().hex}"
    admin_engine = create_engine(database_url, pool_pre_ping=True)
    with admin_engine.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema_name}"'))
    engine = create_engine(
        database_url,
        pool_pre_ping=True,
        execution_options={"schema_translate_map": {None: schema_name}},
    )
    create_schema(engine)
    repository = HostedRepository(lambda: Session(engine))
    admin = repository.create_user(
        email="admin@example.edu", password_hash="hash", role=Role.ADMIN
    )
    annotator = repository.create_user(
        email="annotator@example.edu", password_hash="hash", role=Role.ANNOTATOR
    )
    repository.import_batch(
        name="Concurrency",
        created_by=admin.id,
        documents=[
            DocumentImport(
                external_id="session-1",
                filename="session-1.json",
                raw_text="Alice",
                label_set=["NAME"],
                reference_spans=None,
            )
        ],
    )
    document = repository.list_visible_documents(admin.id)[0]
    repository.assign_document(
        document_id=document.id,
        assignee_id=annotator.id,
        assigned_by_id=admin.id,
    )

    def save(mutation_id: str):
        try:
            return repository.save_annotations(
                document_id=document.id,
                user_id=annotator.id,
                spans=[],
                expected_revision=0,
                mutation_id=mutation_id,
            )
        except RevisionConflict as error:
            return error

    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(save, ["mutation-a", "mutation-b"]))

        assert sum(not isinstance(result, RevisionConflict) for result in results) == 1
        conflicts = [
            result for result in results if isinstance(result, RevisionConflict)
        ]
        assert len(conflicts) == 1
        assert conflicts[0].current_revision == 1
        assert (
            repository.get_document(
                document.id,
                user_id=annotator.id,
            ).revision
            == 1
        )
    finally:
        engine.dispose()
        with admin_engine.begin() as connection:
            connection.execute(text(f'DROP SCHEMA "{schema_name}" CASCADE'))
        admin_engine.dispose()
