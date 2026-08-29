from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import UTC, datetime, timedelta

import pytest
from hosted_app.database import Assignment, create_schema
from hosted_app.domain import (
    AssignmentState,
    CompletedLocked,
    DocumentImport,
    DuplicateExternalId,
    Forbidden,
    InvalidAssignee,
    InvalidReference,
    LoginSessionRecord,
    RevisionConflict,
    Role,
)
from hosted_app.repository import HostedRepository
from sqlalchemy.dialects import postgresql
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine


@pytest.fixture
def repository() -> HostedRepository:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    create_schema(engine)
    return HostedRepository(lambda: Session(engine))


def import_documents(
    repository: HostedRepository,
    *,
    admin_id: str,
    documents: list[DocumentImport],
):
    repository.import_batch(
        name="Test sessions",
        created_by=admin_id,
        documents=documents,
    )
    imported = {
        document.external_id: document
        for document in repository.list_visible_documents(admin_id)
    }
    return [imported[document.external_id] for document in documents]


def test_annotators_see_only_assigned_documents_while_admins_see_all(
    repository: HostedRepository,
) -> None:
    admin = repository.create_user(
        email="admin@example.edu", password_hash="hash-admin", role=Role.ADMIN
    )
    ada = repository.create_user(
        email="ada@example.edu", password_hash="hash-ada", role=Role.ANNOTATOR
    )
    grace = repository.create_user(
        email="grace@example.edu", password_hash="hash-grace", role=Role.ANNOTATOR
    )
    ada_document, grace_document = import_documents(
        repository,
        admin_id=admin.id,
        documents=[
            DocumentImport(
                external_id="session-001",
                filename="session-001.json",
                raw_text="immutable raw transcript one",
                label_set=["NAME", "LOCATION"],
                reference_spans=None,
            ),
            DocumentImport(
                external_id="session-002",
                filename="session-002.json",
                raw_text="immutable raw transcript two",
                label_set=["NAME", "LOCATION"],
                reference_spans=[
                    {
                        "start": 0,
                        "end": 9,
                        "label": "NAME",
                        "text": "immutable",
                    }
                ],
            ),
        ],
    )
    repository.assign_document(
        document_id=ada_document.id, assignee_id=ada.id, assigned_by_id=admin.id
    )
    repository.assign_document(
        document_id=grace_document.id, assignee_id=grace.id, assigned_by_id=admin.id
    )

    assert [item.external_id for item in repository.list_visible_documents(ada.id)] == [
        "session-001"
    ]
    assert [
        item.external_id for item in repository.list_visible_documents(grace.id)
    ] == ["session-002"]
    assert {
        item.external_id for item in repository.list_visible_documents(admin.id)
    } == {"session-001", "session-002"}


def test_assignment_is_idempotent_and_reassigns_the_existing_record(
    repository: HostedRepository,
) -> None:
    admin = repository.create_user(
        email="admin@example.edu", password_hash="hash-admin", role=Role.ADMIN
    )
    ada = repository.create_user(
        email="ada@example.edu", password_hash="hash-ada", role=Role.ANNOTATOR
    )
    grace = repository.create_user(
        email="grace@example.edu", password_hash="hash-grace", role=Role.ANNOTATOR
    )
    document = import_documents(
        repository,
        admin_id=admin.id,
        documents=[
            DocumentImport(
                external_id="session-001",
                filename="session-001.json",
                raw_text="raw",
                label_set=["NAME"],
                reference_spans=None,
            )
        ],
    )[0]

    first = repository.assign_document(
        document_id=document.id, assignee_id=ada.id, assigned_by_id=admin.id
    )
    repeated = repository.assign_document(
        document_id=document.id, assignee_id=ada.id, assigned_by_id=admin.id
    )
    reassigned = repository.assign_document(
        document_id=document.id, assignee_id=grace.id, assigned_by_id=admin.id
    )

    assert repeated.id == first.id
    assert reassigned.id == first.id
    assert reassigned.assignee_id == grace.id
    assert len(repository.list_assignments(admin_id=admin.id)) == 1
    with pytest.raises(Forbidden):
        repository.assign_document(
            document_id=document.id,
            assignee_id=ada.id,
            assigned_by_id=grace.id,
        )
    assert repository.list_assignments(admin_id=admin.id)[0].assignee_id == grace.id


def test_admin_can_self_assign_but_cannot_assign_another_admin(
    repository: HostedRepository,
) -> None:
    admin = repository.create_user(
        email="admin@example.edu", password_hash="hash-admin", role=Role.ADMIN
    )
    other_admin = repository.create_user(
        email="other-admin@example.edu",
        password_hash="hash-other-admin",
        role=Role.ADMIN,
    )
    document = import_documents(
        repository,
        admin_id=admin.id,
        documents=[
            DocumentImport(
                external_id="session-001",
                filename="session-001.json",
                raw_text="raw",
                label_set=["NAME"],
                reference_spans=None,
            )
        ],
    )[0]

    assignment = repository.assign_document(
        document_id=document.id,
        assignee_id=admin.id,
        assigned_by_id=admin.id,
    )

    assert assignment.assignee_id == admin.id
    with pytest.raises(
        InvalidAssignee, match="active annotator or self-assigning admin"
    ):
        repository.assign_document(
            document_id=document.id,
            assignee_id=other_admin.id,
            assigned_by_id=admin.id,
        )
    saved = repository.save_annotations(
        document_id=document.id,
        user_id=admin.id,
        spans=[],
        expected_revision=0,
        mutation_id="admin-self-save",
    )
    assert saved.revision == 1
    assert repository.progress(admin_id=admin.id).by_annotator == [
        {
            "user_id": admin.id,
            "email": "admin@example.edu",
            "display_name": "admin",
            "assigned": 0,
            "in_progress": 1,
            "completed": 0,
            "total": 1,
        }
    ]


def test_saves_are_revisioned_idempotent_and_preserve_imported_content(
    repository: HostedRepository,
) -> None:
    admin = repository.create_user(
        email="admin@example.edu", password_hash="hash-admin", role=Role.ADMIN
    )
    annotator = repository.create_user(
        email="annotator@example.edu",
        password_hash="hash-annotator",
        role=Role.ANNOTATOR,
    )
    reference = [{"start": 0, "end": 5, "label": "NAME", "text": "Alice"}]
    document = import_documents(
        repository,
        admin_id=admin.id,
        documents=[
            DocumentImport(
                external_id="session-001",
                filename="session-001.json",
                raw_text="Alice spoke",
                label_set=["NAME"],
                reference_spans=reference,
            )
        ],
    )[0]
    assignment = repository.assign_document(
        document_id=document.id,
        assignee_id=annotator.id,
        assigned_by_id=admin.id,
    )
    spans = [{"start": 0, "end": 5, "label": "NAME"}]

    saved = repository.save_annotations(
        document_id=document.id,
        user_id=annotator.id,
        spans=spans,
        expected_revision=0,
        mutation_id="mutation-001",
    )
    repeated = repository.save_annotations(
        document_id=document.id,
        user_id=annotator.id,
        spans=[{"start": 6, "end": 11, "label": "NAME"}],
        expected_revision=0,
        mutation_id="mutation-001",
    )

    assert saved.revision == 1
    assert saved.spans == spans
    assert saved.assignment_state == AssignmentState.IN_PROGRESS
    assert repeated == saved

    with pytest.raises(RevisionConflict) as conflict:
        repository.save_annotations(
            document_id=document.id,
            user_id=annotator.id,
            spans=[],
            expected_revision=0,
            mutation_id="mutation-002",
        )

    assert conflict.value.current_revision == 1
    detail = repository.get_document(document.id, user_id=annotator.id)
    assert detail.raw_text == "Alice spoke"
    assert detail.reference_spans == reference
    assert detail.manual_spans == spans
    assert detail.revision == 1
    assert detail.assignment_id == assignment.id


def test_assignment_mutations_use_a_postgresql_row_lock() -> None:
    statement = HostedRepository._assignment_for_update(document_id="document-1")

    sql = str(statement.compile(dialect=postgresql.dialect()))

    assert "FROM assignments" in sql
    assert "FOR UPDATE" in sql
    assert statement.column_descriptions[0]["entity"] is Assignment


def test_completed_work_is_locked_until_an_admin_reopens_it(
    repository: HostedRepository,
) -> None:
    admin = repository.create_user(
        email="admin@example.edu", password_hash="hash-admin", role=Role.ADMIN
    )
    annotator = repository.create_user(
        email="annotator@example.edu", password_hash="hash", role=Role.ANNOTATOR
    )
    other_annotator = repository.create_user(
        email="other@example.edu", password_hash="hash", role=Role.ANNOTATOR
    )
    document = import_documents(
        repository,
        admin_id=admin.id,
        documents=[
            DocumentImport(
                external_id="session-001",
                filename="session-001.json",
                raw_text="raw",
                label_set=["NAME"],
                reference_spans=None,
            )
        ],
    )[0]
    assignment = repository.assign_document(
        document_id=document.id,
        assignee_id=annotator.id,
        assigned_by_id=admin.id,
    )
    repository.save_annotations(
        document_id=document.id,
        user_id=annotator.id,
        spans=[],
        expected_revision=0,
        mutation_id="mutation-001",
    )

    completed = repository.complete_assignment(assignment.id, user_id=annotator.id)
    assert completed.state == AssignmentState.COMPLETED
    assert completed.completed_at is not None

    with pytest.raises(CompletedLocked):
        repository.save_annotations(
            document_id=document.id,
            user_id=annotator.id,
            spans=[],
            expected_revision=1,
            mutation_id="mutation-002",
        )
    with pytest.raises(Forbidden):
        repository.reopen_assignment(assignment.id, admin_id=other_annotator.id)

    reopened = repository.reopen_assignment(assignment.id, admin_id=admin.id)
    assert reopened.state == AssignmentState.IN_PROGRESS
    assert reopened.completed_at is None
    saved = repository.save_annotations(
        document_id=document.id,
        user_id=annotator.id,
        spans=[],
        expected_revision=1,
        mutation_id="mutation-003",
    )
    assert saved.revision == 2


def test_admin_progress_counts_assignment_states_and_unassigned_documents(
    repository: HostedRepository,
) -> None:
    admin = repository.create_user(
        email="admin@example.edu", password_hash="hash-admin", role=Role.ADMIN
    )
    annotator = repository.create_user(
        email="annotator@example.edu", password_hash="hash", role=Role.ANNOTATOR
    )
    documents = import_documents(
        repository,
        admin_id=admin.id,
        documents=[
            DocumentImport(
                external_id=f"session-{index}",
                filename=f"session-{index}.json",
                raw_text="raw",
                label_set=["NAME"],
                reference_spans=None,
            )
            for index in range(4)
        ],
    )
    assigned = repository.assign_document(
        document_id=documents[0].id,
        assignee_id=annotator.id,
        assigned_by_id=admin.id,
    )
    repository.assign_document(
        document_id=documents[1].id,
        assignee_id=annotator.id,
        assigned_by_id=admin.id,
    )
    completed = repository.assign_document(
        document_id=documents[2].id,
        assignee_id=annotator.id,
        assigned_by_id=admin.id,
    )
    repository.save_annotations(
        document_id=documents[1].id,
        user_id=annotator.id,
        spans=[],
        expected_revision=0,
        mutation_id="mutation-in-progress",
    )
    repository.save_annotations(
        document_id=documents[2].id,
        user_id=annotator.id,
        spans=[],
        expected_revision=0,
        mutation_id="mutation-complete",
    )
    repository.complete_assignment(completed.id, user_id=annotator.id)

    progress = repository.progress(admin_id=admin.id)

    assert progress.total == 4
    assert progress.unassigned == 1
    assert progress.assigned == 1
    assert progress.in_progress == 1
    assert progress.completed == 1
    assert progress.by_annotator == [
        {
            "user_id": annotator.id,
            "email": "annotator@example.edu",
            "display_name": "annotator",
            "assigned": 1,
            "in_progress": 1,
            "completed": 1,
            "total": 3,
        }
    ]
    assert assigned.state == AssignmentState.ASSIGNED

    with pytest.raises(Forbidden):
        repository.progress(admin_id=annotator.id)
    assert [user.email for user in repository.list_users(admin_id=admin.id)] == [
        "admin@example.edu",
        "annotator@example.edu",
    ]


def test_admin_progress_includes_annotators_with_no_assignments(
    repository: HostedRepository,
) -> None:
    admin = repository.create_user(
        email="admin@example.edu", password_hash="hash-admin", role=Role.ADMIN
    )
    repository.create_user(
        email="new@example.edu",
        display_name="New Annotator",
        password_hash="hash-new",
        role=Role.ANNOTATOR,
    )

    progress = repository.progress(admin_id=admin.id)

    assert progress.by_annotator == [
        {
            "user_id": progress.by_annotator[0]["user_id"],
            "email": "new@example.edu",
            "display_name": "New Annotator",
            "assigned": 0,
            "in_progress": 0,
            "completed": 0,
            "total": 0,
        }
    ]


def test_auth_records_are_persisted_without_plaintext_tokens_and_password_change_revokes_sessions(
    repository: HostedRepository,
) -> None:
    admin = repository.bootstrap_admin(
        email="ADMIN@EXAMPLE.EDU", display_name="Admin", password_hash="argon-old"
    )
    expiry = datetime.now(UTC) + timedelta(hours=8)
    repository.create_login_session(
        LoginSessionRecord(
            token_hash="sha256-opaque-token",
            user_id=admin.id,
            expires_at=expiry,
        )
    )

    assert repository.get_user_by_email("admin@example.edu") == admin
    assert repository.get_user_by_id(admin.id) == admin
    stored_session = repository.get_login_session("sha256-opaque-token")
    assert stored_session is not None
    assert stored_session.token_hash == "sha256-opaque-token"
    assert stored_session.user_id == admin.id
    assert stored_session.expires_at == expiry

    assert (
        repository.bootstrap_admin(
            email="second-admin@example.edu",
            display_name="Second Admin",
            password_hash="argon-second",
        )
        is None
    )

    repository.replace_password_and_delete_sessions(
        user_id=admin.id, password_hash="argon-new"
    )

    assert repository.get_user_by_id(admin.id).password_hash == "argon-new"
    assert repository.get_login_session("sha256-opaque-token") is None


def test_admin_bootstrap_allows_only_one_concurrent_winner(tmp_path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'bootstrap.db'}")
    create_schema(engine)
    repository = HostedRepository(lambda: Session(engine))

    def bootstrap(index: int):
        return repository.bootstrap_admin(
            email=f"admin-{index}@example.edu",
            display_name=f"Admin {index}",
            password_hash=f"hash-{index}",
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(bootstrap, range(16)))

    winners = [result for result in results if result is not None]
    assert len(winners) == 1
    assert repository.get_user_by_id(winners[0].id) == winners[0]


def test_import_batch_rejects_invalid_reference_without_partial_writes(
    repository: HostedRepository,
) -> None:
    admin = repository.create_user(
        email="admin@example.edu", password_hash="hash", role=Role.ADMIN
    )
    documents = [
        DocumentImport(
            external_id="session-valid",
            filename="valid.json",
            raw_text="Alice spoke",
            label_set=["NAME"],
            reference_spans=[{"start": 0, "end": 5, "label": "NAME", "text": "Alice"}],
        ),
        DocumentImport(
            external_id="session-invalid",
            filename="invalid.json",
            raw_text="Bob spoke",
            label_set=["NAME"],
            reference_spans=[{"start": 0, "end": 3, "label": "NAME", "text": "Alice"}],
        ),
    ]

    with pytest.raises(InvalidReference, match="reference text does not match"):
        repository.import_batch(
            name="Rejected batch", created_by=admin.id, documents=documents
        )

    assert repository.list_visible_documents(admin.id) == []


def test_import_batch_rejects_duplicate_ids_without_partial_writes(
    repository: HostedRepository,
) -> None:
    admin = repository.create_user(
        email="admin@example.edu", password_hash="hash", role=Role.ADMIN
    )
    duplicate = DocumentImport(
        external_id="session-duplicate",
        filename="session.json",
        raw_text="No reference",
        label_set=["NAME"],
        reference_spans=None,
    )

    with pytest.raises(DuplicateExternalId, match="session-duplicate"):
        repository.import_batch(
            name="Rejected batch",
            created_by=admin.id,
            documents=[duplicate, duplicate],
        )

    assert repository.list_visible_documents(admin.id) == []


def test_imported_sessions_can_be_exported_with_current_manual_work(
    repository: HostedRepository,
) -> None:
    admin = repository.create_user(
        email="admin@example.edu", password_hash="secret-hash", role=Role.ADMIN
    )
    annotator = repository.create_user(
        email="annotator@example.edu", password_hash="other-secret", role=Role.ANNOTATOR
    )
    imported = repository.import_batch(
        name="Accepted batch",
        created_by=admin.id,
        documents=[
            DocumentImport(
                external_id="session-001",
                filename="one.json",
                raw_text="Alice spoke",
                label_set=["NAME"],
                reference_spans=[
                    {"start": 0, "end": 5, "label": "NAME", "text": "Alice"}
                ],
            ),
            DocumentImport(
                external_id="session-002",
                filename="two.json",
                raw_text="No names",
                label_set=["NAME"],
                reference_spans=None,
            ),
        ],
    )
    documents = repository.list_visible_documents(admin.id)
    annotated_document = next(
        document for document in documents if document.external_id == "session-001"
    )
    repository.assign_document(
        document_id=annotated_document.id,
        assignee_id=annotator.id,
        assigned_by_id=admin.id,
    )
    manual_spans = [{"start": 0, "end": 5, "label": "NAME", "text": "Alice"}]
    repository.save_annotations(
        document_id=annotated_document.id,
        user_id=annotator.id,
        spans=manual_spans,
        expected_revision=0,
        mutation_id="exported-mutation",
    )

    exported = repository.export_manual_annotations(admin_id=admin.id)

    assert imported.imported_count == 2
    assert all(document.batch_id == imported.batch_id for document in documents)
    assert [item.external_id for item in exported] == ["session-001", "session-002"]
    annotated, unannotated = exported
    assert annotated.filename == "one.json"
    assert annotated.manual_annotations == manual_spans
    assert annotated.annotation_revision == 1
    assert annotated.assignee_id == annotator.id
    assert annotated.assignment_state == AssignmentState.IN_PROGRESS
    assert annotated.updated_at is not None
    assert unannotated.manual_annotations == []
    assert unannotated.annotation_revision == 0
    assert unannotated.assignee_id is None
    assert unannotated.assignment_state is None
    assert unannotated.updated_at is None
    assert set(asdict(annotated)) == {
        "document_id",
        "external_id",
        "filename",
        "manual_annotations",
        "annotation_revision",
        "assignee_id",
        "assignment_state",
        "updated_at",
    }
    with pytest.raises(Forbidden):
        repository.export_manual_annotations(admin_id=annotator.id)
