from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.dialects import postgresql
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine

from hosted_app.api import create_hosted_app
from hosted_app.auth import AuthenticatedPrincipal, AuthManager
from hosted_app.bulk_audit_api import create_bulk_audit_router
from hosted_app.database import create_schema
from hosted_app.domain import (
    ActivationTokenRecord,
    BulkMutationConflict,
    BulkPlanStale,
    DocumentImport,
    DuplicateSelection,
    Forbidden,
    InvalidAssignee,
    Role,
)
from hosted_app.repository import HostedRepository


def make_repository() -> HostedRepository:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    create_schema(engine)
    return HostedRepository(lambda: Session(engine))


def seed_bulk_case(
    repository: HostedRepository,
):
    admin = repository.create_user(
        email="admin@example.edu", password_hash="hash-admin", role=Role.ADMIN
    )
    annotators = [
        repository.create_user(
            email=f"annotator-{index}@example.edu",
            password_hash=f"hash-{index}",
            role=Role.ANNOTATOR,
        )
        for index in range(3)
    ]
    repository.import_batch(
        name="Bulk sessions",
        created_by=admin.id,
        documents=[
            DocumentImport(
                external_id=f"session-{index}",
                filename=f"session-{index}.json",
                raw_text=f"Transcript {index}",
                label_set=["NAME"],
                reference_spans=None,
            )
            for index in range(7)
        ],
    )
    documents = repository.list_visible_documents(admin.id)
    return admin, annotators, documents


def test_bulk_preview_is_deterministic_balanced_and_read_only() -> None:
    repository = make_repository()
    admin, annotators, documents = seed_bulk_case(repository)
    audit_before = repository.list_audit_events(admin_id=admin.id)

    first = repository.preview_balanced_assignment(
        admin_id=admin.id,
        document_ids=[document.id for document in reversed(documents)],
        annotator_ids=[annotator.id for annotator in reversed(annotators)],
    )
    second = repository.preview_balanced_assignment(
        admin_id=admin.id,
        document_ids=[document.id for document in documents],
        annotator_ids=[annotator.id for annotator in annotators],
    )

    assert first == second
    assert len(first.plan_digest) == 64
    counts = {
        annotator.id: sum(
            item.assignee_id == annotator.id for item in first.assignments
        )
        for annotator in annotators
    }
    assert sorted(counts.values()) == [2, 2, 3]
    assert all(item.assignment_id is None for item in first.document_preconditions)
    assert all(item.state is None for item in first.document_preconditions)
    assert all(item.revision == 0 for item in first.document_preconditions)
    assert {item.state for item in first.annotator_preconditions} == {"active"}
    assert repository.list_assignments(admin_id=admin.id) == []
    assert repository.list_audit_events(admin_id=admin.id) == audit_before


def test_bulk_preview_rejects_duplicates_missing_documents_and_inactive_users() -> None:
    repository = make_repository()
    admin, annotators, documents = seed_bulk_case(repository)

    for document_ids, annotator_ids in (
        ([documents[0].id, documents[0].id], [annotators[0].id]),
        ([documents[0].id], [annotators[0].id, annotators[0].id]),
    ):
        try:
            repository.preview_balanced_assignment(
                admin_id=admin.id,
                document_ids=document_ids,
                annotator_ids=annotator_ids,
            )
        except DuplicateSelection:
            pass
        else:
            raise AssertionError("duplicate selection must be rejected")

    repository.deactivate_user(
        user_id=annotators[0].id,
        admin_id=admin.id,
        incomplete_action="unassign",
        reassign_to_id=None,
    )
    try:
        repository.preview_balanced_assignment(
            admin_id=admin.id,
            document_ids=[documents[0].id],
            annotator_ids=[annotators[0].id],
        )
    except InvalidAssignee:
        pass
    else:
        raise AssertionError("inactive annotator must be rejected")


def test_bulk_apply_is_atomic_idempotent_and_audited_without_content() -> None:
    repository = make_repository()
    admin, annotators, documents = seed_bulk_case(repository)
    plan = repository.preview_balanced_assignment(
        admin_id=admin.id,
        document_ids=[document.id for document in documents],
        annotator_ids=[annotator.id for annotator in annotators],
    )

    applied = repository.apply_balanced_assignment(
        admin_id=admin.id,
        document_ids=[document.id for document in documents],
        annotator_ids=[annotator.id for annotator in annotators],
        plan_digest=plan.plan_digest,
        mutation_id="bulk-001",
    )
    retried = repository.apply_balanced_assignment(
        admin_id=admin.id,
        document_ids=[document.id for document in reversed(documents)],
        annotator_ids=[annotator.id for annotator in reversed(annotators)],
        plan_digest=plan.plan_digest,
        mutation_id="bulk-001",
    )

    assert retried == applied
    assert len(applied.assignment_ids) == 7
    assignments = repository.list_assignments(admin_id=admin.id)
    assert len(assignments) == 7
    events = repository.list_audit_events(
        admin_id=admin.id,
        action="assignment.bulk_applied",
        mutation_id="bulk-001",
    )
    assert len(events) == 1
    event = events[0]
    assert event.actor_id == admin.id
    assert event.target_type == "assignment_plan"
    assert event.target_id == plan.plan_digest
    assert event.result == "success"
    assert event.reason is None
    assert event.before_metadata == {
        "assignment_ids": [],
        "state": "previewed",
    }
    assert event.after_metadata == {
        "assignment_ids": sorted(applied.assignment_ids),
        "state": "applied",
    }
    serialized = repr(event.before_metadata) + repr(event.after_metadata)
    assert "Transcript" not in serialized
    assert "password" not in serialized


def test_bulk_apply_rejects_stale_state_and_mutation_reuse_without_partial_changes() -> (
    None
):
    repository = make_repository()
    admin, annotators, documents = seed_bulk_case(repository)
    plan = repository.preview_balanced_assignment(
        admin_id=admin.id,
        document_ids=[document.id for document in documents],
        annotator_ids=[annotator.id for annotator in annotators],
    )
    repository.assign_document(
        document_id=documents[0].id,
        assignee_id=annotators[0].id,
        assigned_by_id=admin.id,
    )

    try:
        repository.apply_balanced_assignment(
            admin_id=admin.id,
            document_ids=[document.id for document in documents],
            annotator_ids=[annotator.id for annotator in annotators],
            plan_digest=plan.plan_digest,
            mutation_id="bulk-stale",
        )
    except BulkPlanStale:
        pass
    else:
        raise AssertionError("stale plan must be rejected")
    assert len(repository.list_assignments(admin_id=admin.id)) == 1

    fresh = repository.preview_balanced_assignment(
        admin_id=admin.id,
        document_ids=[document.id for document in documents],
        annotator_ids=[annotator.id for annotator in annotators],
    )
    repository.apply_balanced_assignment(
        admin_id=admin.id,
        document_ids=[document.id for document in documents],
        annotator_ids=[annotator.id for annotator in annotators],
        plan_digest=fresh.plan_digest,
        mutation_id="bulk-reused",
    )
    other_plan = repository.preview_balanced_assignment(
        admin_id=admin.id,
        document_ids=[documents[0].id],
        annotator_ids=[annotators[0].id],
    )
    try:
        repository.apply_balanced_assignment(
            admin_id=admin.id,
            document_ids=[documents[0].id],
            annotator_ids=[annotators[0].id],
            plan_digest=other_plan.plan_digest,
            mutation_id="bulk-reused",
        )
    except BulkMutationConflict:
        pass
    else:
        raise AssertionError("mutation reuse with another plan must be rejected")
    assert len(repository.list_assignments(admin_id=admin.id)) == 7


def test_bulk_apply_rejects_annotation_revision_changed_after_preview() -> None:
    repository = make_repository()
    admin, annotators, documents = seed_bulk_case(repository)
    assignment = repository.assign_document(
        document_id=documents[0].id,
        assignee_id=annotators[0].id,
        assigned_by_id=admin.id,
    )
    repository.save_annotations(
        document_id=documents[0].id,
        user_id=annotators[0].id,
        spans=[],
        expected_revision=0,
        mutation_id="save-before-preview",
    )
    plan = repository.preview_balanced_assignment(
        admin_id=admin.id,
        document_ids=[documents[0].id],
        annotator_ids=[annotators[1].id],
    )
    assert plan.document_preconditions[0].revision == 1
    repository.save_annotations(
        document_id=documents[0].id,
        user_id=annotators[0].id,
        spans=[],
        expected_revision=1,
        mutation_id="save-after-preview",
    )

    try:
        repository.apply_balanced_assignment(
            admin_id=admin.id,
            document_ids=[documents[0].id],
            annotator_ids=[annotators[1].id],
            plan_digest=plan.plan_digest,
            mutation_id="bulk-after-edit",
        )
    except BulkPlanStale:
        pass
    else:
        raise AssertionError("annotation revision change must make the plan stale")
    current = repository.list_assignments(admin_id=admin.id)[0]
    assert current.id == assignment.id
    assert current.assignee_id == annotators[0].id


def test_bulk_apply_assignment_query_uses_a_postgresql_row_lock() -> None:
    statement = HostedRepository._assignments_for_documents(
        ["document-1", "document-2"],
        lock=True,
    )

    compiled = str(statement.compile(dialect=postgresql.dialect()))

    assert "assignments.document_id IN" in compiled
    assert compiled.endswith("FOR UPDATE")


def test_successful_lifecycle_import_assignment_completion_and_reopen_are_audited() -> (
    None
):
    repository = make_repository()
    admin = repository.create_user(
        email="admin@example.edu", password_hash="hash-admin", role=Role.ADMIN
    )
    invited = repository.create_pending_user_with_activation(
        email="new@example.edu",
        display_name="New Annotator",
        role="annotator",
        activation=ActivationTokenRecord(
            token_hash="activation-1",
            expires_at=datetime.now(UTC) + timedelta(hours=24),
        ),
        admin_id=admin.id,
    )
    activated = repository.activate_user(
        token_hash="activation-1",
        password_hash="hash-new",
        now=datetime.now(UTC),
    )
    assert activated is not None
    imported = repository.import_batch(
        name="Audited sessions",
        created_by=admin.id,
        mutation_id="import-001",
        manifest_digest="manifest-001",
        documents=[
            DocumentImport(
                external_id="audited-session",
                filename="audited-session.json",
                raw_text="Private transcript text",
                label_set=["NAME"],
                reference_spans=None,
            )
        ],
    )
    document = repository.list_visible_documents(admin.id)[0]
    assignment = repository.assign_document(
        document_id=document.id,
        assignee_id=invited.id,
        assigned_by_id=admin.id,
    )
    repository.complete_assignment(assignment.id, user_id=invited.id)
    repository.reopen_assignment(assignment.id, admin_id=admin.id)
    repository.reset_user_password(
        user_id=invited.id,
        activation=ActivationTokenRecord(
            token_hash="activation-2",
            expires_at=datetime.now(UTC) + timedelta(hours=24),
        ),
        admin_id=admin.id,
    )
    repository.activate_user(
        token_hash="activation-2",
        password_hash="hash-replacement",
        now=datetime.now(UTC),
    )
    repository.deactivate_user(
        user_id=invited.id,
        admin_id=admin.id,
        incomplete_action="unassign",
        reassign_to_id=None,
    )
    repository.reactivate_user(user_id=invited.id, admin_id=admin.id)
    repository.export_manual_annotations(admin_id=admin.id)

    events = repository.list_audit_events(admin_id=admin.id, limit=100)
    assert {event.action for event in events} == {
        "account.activated",
        "account.created",
        "account.deactivated",
        "account.password_reset",
        "account.reactivated",
        "assignment.assigned",
        "assignment.completed",
        "assignment.reopened",
        "assignment.unassigned",
        "annotations.exported",
        "batch.imported",
    }
    imported_event = next(event for event in events if event.action == "batch.imported")
    assert imported_event.target_id == imported.batch_id
    assert imported_event.mutation_id == "import-001"
    assert imported_event.after_metadata == {
        "batch_id": imported.batch_id,
        "imported_count": 1,
        "manifest_digest": "manifest-001",
        "state": "imported",
    }
    serialized = repr(events)
    assert "Private transcript text" not in serialized
    assert "activation-1" not in serialized
    assert "hash-replacement" not in serialized


def test_audit_list_is_admin_only_and_filters_by_actor_target_and_result() -> None:
    repository = make_repository()
    admin, annotators, documents = seed_bulk_case(repository)
    assignment = repository.assign_document(
        document_id=documents[0].id,
        assignee_id=annotators[0].id,
        assigned_by_id=admin.id,
    )

    assert (
        len(
            repository.list_audit_events(
                admin_id=admin.id,
                actor_id=admin.id,
                action="assignment.assigned",
                target_type="assignment",
                target_id=assignment.id,
                result="success",
            )
        )
        == 1
    )
    assert (
        repository.list_audit_events(
            admin_id=admin.id,
            action="assignment.reassigned",
        )
        == []
    )
    try:
        repository.list_audit_events(admin_id=annotators[0].id)
    except Forbidden:
        pass
    else:
        raise AssertionError("annotators must not read audit events")


def test_bulk_and_audit_router_enforces_admin_and_exposes_filtered_events() -> None:
    repository = make_repository()
    admin, annotators, documents = seed_bulk_case(repository)
    current = {
        "principal": AuthenticatedPrincipal(
            id=admin.id,
            email=admin.email,
            display_name=admin.display_name,
            role=admin.role,
            state=admin.state,
        )
    }

    def principal() -> AuthenticatedPrincipal:
        return current["principal"]

    def csrf() -> None:
        return None

    app = FastAPI()
    app.include_router(
        create_bulk_audit_router(
            repository=repository,
            current_principal=principal,
            require_csrf=csrf,
        )
    )
    client = TestClient(app)
    payload = {
        "document_ids": [document.id for document in reversed(documents)],
        "annotator_ids": [annotator.id for annotator in reversed(annotators)],
    }

    preview = client.post("/api/admin/assignments/bulk/preview", json=payload)
    assert preview.status_code == 200, preview.text
    assert len(preview.json()["assignments"]) == 7
    assert repository.list_assignments(admin_id=admin.id) == []
    applied = client.post(
        "/api/admin/assignments/bulk/apply",
        json={
            **payload,
            "plan_digest": preview.json()["plan_digest"],
            "mutation_id": "api-bulk-001",
        },
    )
    assert applied.status_code == 200
    assert len(applied.json()["assignment_ids"]) == 7
    audit = client.get(
        "/api/admin/audit",
        params={
            "actor_id": admin.id,
            "action": "assignment.bulk_applied",
            "target_type": "assignment_plan",
            "mutation_id": "api-bulk-001",
            "result": "success",
        },
    )
    assert audit.status_code == 200
    assert len(audit.json()) == 1
    assert audit.json()[0]["target_id"] == preview.json()["plan_digest"]

    annotator = annotators[0]
    current["principal"] = AuthenticatedPrincipal(
        id=annotator.id,
        email=annotator.email,
        display_name=annotator.display_name,
        role=annotator.role,
        state=annotator.state,
    )
    assert client.get("/api/admin/audit").status_code == 403
    assert (
        client.post("/api/admin/assignments/bulk/preview", json=payload).status_code
        == 403
    )


def test_integrated_bulk_routes_require_login_csrf_and_admin_role() -> None:
    repository = make_repository()
    auth = AuthManager(repository)
    admin = auth.bootstrap_admin(
        "admin@example.edu",
        "correct horse battery staple",
        display_name="Admin",
    )
    annotator = repository.create_user(
        email="annotator@example.edu",
        password_hash=auth.create_password_hash("annotator secure password"),
        role=Role.ANNOTATOR,
    )
    repository.import_batch(
        name="Integrated bulk",
        created_by=admin.id,
        documents=[
            DocumentImport(
                external_id="integrated-session",
                filename="integrated-session.json",
                raw_text="Sensitive session",
                label_set=["NAME"],
                reference_spans=None,
            )
        ],
    )
    document = repository.list_visible_documents(admin.id)[0]
    client = TestClient(
        create_hosted_app(repository=repository, auth=auth, cookie_secure=False)
    )
    payload = {
        "document_ids": [document.id],
        "annotator_ids": [annotator.id],
    }

    assert client.get("/api/admin/audit").status_code == 401
    assert (
        client.post(
            "/api/auth/login",
            json={
                "email": "admin@example.edu",
                "password": "correct horse battery staple",
            },
        ).status_code
        == 200
    )
    assert (
        client.post("/api/admin/assignments/bulk/preview", json=payload).status_code
        == 403
    )
    csrf = {"X-CSRF-Token": client.cookies["annotation_csrf"]}
    preview = client.post(
        "/api/admin/assignments/bulk/preview",
        json=payload,
        headers=csrf,
    )
    assert preview.status_code == 200
    assert (
        client.post(
            "/api/admin/assignments/bulk/apply",
            json={
                **payload,
                "plan_digest": preview.json()["plan_digest"],
                "mutation_id": "integrated-bulk-001",
            },
            headers=csrf,
        ).status_code
        == 200
    )
    assert (
        client.get(
            "/api/admin/audit",
            params={
                "action": "assignment.bulk_applied",
                "mutation_id": "integrated-bulk-001",
            },
        ).status_code
        == 200
    )
    assert client.post("/api/auth/logout", headers=csrf).status_code == 204
    assert (
        client.post(
            "/api/auth/login",
            json={
                "email": "annotator@example.edu",
                "password": "annotator secure password",
            },
        ).status_code
        == 200
    )
    assert client.get("/api/admin/audit").status_code == 403
    assert (
        client.post(
            "/api/admin/assignments/bulk/preview",
            json=payload,
            headers={"X-CSRF-Token": client.cookies["annotation_csrf"]},
        ).status_code
        == 403
    )
