from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from hosted_app.api import create_hosted_app
from hosted_app.auth import AuthManager
from hosted_app.database import create_schema
from hosted_app.domain import DocumentImport, Role
from hosted_app.repository import HostedRepository
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine


def csrf_headers(client: TestClient) -> dict[str, str]:
    return {"X-CSRF-Token": client.cookies["annotation_csrf"]}


@pytest.fixture
def hosted_client() -> tuple[TestClient, HostedRepository, AuthManager]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    create_schema(engine)
    repository = HostedRepository(lambda: Session(engine))
    auth = AuthManager(repository)
    admin = auth.bootstrap_admin(
        "admin@example.edu",
        "correct horse battery staple",
        display_name="Admin",
    )
    ada = repository.create_user(
        email="ada@example.edu",
        display_name="Ada Annotator",
        password_hash=auth.create_password_hash("ada secure password"),
        role=Role.ANNOTATOR,
    )
    grace = repository.create_user(
        email="grace@example.edu",
        display_name="Grace Annotator",
        password_hash=auth.create_password_hash("grace secure password"),
        role=Role.ANNOTATOR,
    )
    repository.import_batch(
        name="New sessions",
        created_by=admin.id,
        documents=[
            DocumentImport(
                external_id="session-ada",
                filename="session-ada.json",
                raw_text="Ada met Alice.",
                label_set=["NAME"],
                reference_spans=None,
            ),
            DocumentImport(
                external_id="session-grace",
                filename="session-grace.json",
                raw_text="Grace met Bob.",
                label_set=["NAME"],
                reference_spans=[],
            ),
            DocumentImport(
                external_id="session-unassigned",
                filename="session-unassigned.json",
                raw_text="An unassigned transcript.",
                label_set=["NAME"],
                reference_spans=None,
            ),
        ],
    )
    documents = {
        document.external_id: document
        for document in repository.list_visible_documents(admin.id)
    }
    ada_document = documents["session-ada"]
    grace_document = documents["session-grace"]
    repository.assign_document(
        document_id=ada_document.id,
        assignee_id=ada.id,
        assigned_by_id=admin.id,
    )
    repository.assign_document(
        document_id=grace_document.id,
        assignee_id=grace.id,
        assigned_by_id=admin.id,
    )
    app = create_hosted_app(
        repository=repository,
        auth=auth,
        cookie_secure=False,
    )
    return TestClient(app), repository, auth


def test_login_cookie_authenticates_me_without_exposing_password_data(
    hosted_client: tuple[TestClient, HostedRepository, AuthManager],
) -> None:
    client, _, _ = hosted_client

    response = client.post(
        "/api/auth/login",
        json={
            "email": " ADMIN@example.edu ",
            "password": "correct horse battery staple",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "id": response.json()["id"],
        "email": "admin@example.edu",
        "display_name": "Admin",
        "role": "admin",
        "state": "active",
    }
    cookie = response.headers["set-cookie"]
    assert "annotation_session=" in cookie
    assert "HttpOnly" in cookie
    assert "SameSite=lax" in cookie
    assert client.cookies.get("annotation_csrf")
    assert "correct horse battery staple" not in response.text

    me = client.get("/api/auth/me")

    assert me.status_code == 200
    assert me.json() == response.json()


def test_login_rejects_an_unknown_email_from_any_domain(
    hosted_client: tuple[TestClient, HostedRepository, AuthManager],
) -> None:
    client, _, _ = hosted_client

    response = client.post(
        "/api/auth/login",
        json={"email": "outsider@gmail.com", "password": "any password at all"},
    )

    assert response.status_code == 401
    assert response.json() == {"detail": "Invalid email or password"}


def test_admin_rejects_a_malformed_email_address(
    hosted_client: tuple[TestClient, HostedRepository, AuthManager],
) -> None:
    client, _, _ = hosted_client
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

    response = client.post(
        "/api/admin/users",
        json={"email": "not-an-email", "role": "annotator"},
        headers=csrf_headers(client),
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "valid email is required"}


def test_request_bodies_reject_unknown_fields(
    hosted_client: tuple[TestClient, HostedRepository, AuthManager],
) -> None:
    client, _, _ = hosted_client

    response = client.post(
        "/api/auth/login",
        json={
            "email": "admin@example.edu",
            "password": "correct horse battery staple",
            "unexpected": "must not be silently ignored",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"][0]["type"] == "extra_forbidden"


def test_authenticated_writes_require_the_session_bound_csrf_token(
    hosted_client: tuple[TestClient, HostedRepository, AuthManager],
) -> None:
    client, _, _ = hosted_client
    assert (
        client.post(
            "/api/auth/login",
            json={"email": "ada@example.edu", "password": "ada secure password"},
        ).status_code
        == 200
    )
    document_id = client.get("/api/workspace").json()["sessions"][0]["id"]
    payload = {
        "spans": [],
        "expected_revision": 0,
        "mutation_id": "csrf-check",
    }

    missing = client.put(
        f"/api/documents/{document_id}/annotations",
        json=payload,
    )
    wrong = client.put(
        f"/api/documents/{document_id}/annotations",
        json=payload,
        headers={"X-CSRF-Token": "wrong-token"},
    )
    accepted = client.put(
        f"/api/documents/{document_id}/annotations",
        json=payload,
        headers={"X-CSRF-Token": client.cookies["annotation_csrf"]},
    )

    assert missing.status_code == 403
    assert missing.json() == {"detail": "CSRF validation failed"}
    assert wrong.status_code == 403
    assert accepted.status_code == 200


def test_authenticated_client_can_restore_a_missing_csrf_cookie(
    hosted_client: tuple[TestClient, HostedRepository, AuthManager],
) -> None:
    client, _, _ = hosted_client
    assert client.get("/api/auth/csrf").status_code == 401
    assert (
        client.post(
            "/api/auth/login",
            json={"email": "ada@example.edu", "password": "ada secure password"},
        ).status_code
        == 200
    )
    document_id = client.get("/api/workspace").json()["sessions"][0]["id"]
    session_cookie = client.cookies["annotation_session"]
    client.cookies.delete("annotation_csrf")

    failed = client.put(
        f"/api/documents/{document_id}/annotations",
        json={
            "spans": [],
            "expected_revision": 0,
            "mutation_id": "csrf-recovery",
        },
    )
    refreshed = client.get("/api/auth/csrf")
    recovered = client.put(
        f"/api/documents/{document_id}/annotations",
        json={
            "spans": [],
            "expected_revision": 0,
            "mutation_id": "csrf-recovery",
        },
        headers=csrf_headers(client),
    )

    assert failed.status_code == 403
    assert failed.json() == {"detail": "CSRF validation failed"}
    assert refreshed.status_code == 204
    assert client.cookies["annotation_session"] == session_cookie
    assert client.cookies["annotation_csrf"]
    assert recovered.status_code == 200


def test_health_probe_does_not_require_authentication(
    hosted_client: tuple[TestClient, HostedRepository, AuthManager],
) -> None:
    client, _, _ = hosted_client

    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["cache-control"] == "no-store"
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert client.get("/docs").status_code == 404


def test_legacy_experiment_and_evaluation_routes_do_not_exist(
    hosted_client: tuple[TestClient, HostedRepository, AuthManager],
) -> None:
    client, _, _ = hosted_client

    for path in (
        "/api/experiments/limits",
        "/api/metrics/dashboard",
        "/api/models/presets",
        "/api/evaluations",
    ):
        assert client.get(path).status_code == 404


def test_logout_invalidates_the_server_session_and_clears_the_cookie(
    hosted_client: tuple[TestClient, HostedRepository, AuthManager],
) -> None:
    client, _, _ = hosted_client
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

    logout = client.post("/api/auth/logout", headers=csrf_headers(client))

    assert logout.status_code == 204
    assert "annotation_session=" in logout.headers["set-cookie"]
    assert client.get("/api/auth/me").status_code == 401


def test_annotator_workspace_contains_only_assigned_sessions(
    hosted_client: tuple[TestClient, HostedRepository, AuthManager],
) -> None:
    client, _, _ = hosted_client
    login = client.post(
        "/api/auth/login",
        json={"email": "ada@example.edu", "password": "ada secure password"},
    )
    assert login.status_code == 200

    response = client.get("/api/workspace")

    assert response.status_code == 200
    assert response.json() == {
        "sessions": [
            {
                "id": response.json()["sessions"][0]["id"],
                "external_id": "session-ada",
                "filename": "session-ada.json",
                "folder_id": None,
                "folder_name": None,
                "assignment_id": response.json()["sessions"][0]["assignment_id"],
                "assignment_state": "assigned",
                "manual_annotation_count": 0,
                "assignee_id": login.json()["id"],
                "assignee_name": "ada@example.edu",
            }
        ]
    }


def test_annotator_cannot_read_edit_or_complete_another_annotators_session(
    hosted_client: tuple[TestClient, HostedRepository, AuthManager],
) -> None:
    client, repository, _ = hosted_client
    ada = repository.get_user_by_email("ada@example.edu")
    grace = repository.get_user_by_email("grace@example.edu")
    assert ada is not None
    assert grace is not None
    grace_document = repository.list_visible_documents(grace.id)[0]
    grace_assignment_id = repository.get_document(
        grace_document.id,
        user_id=grace.id,
    ).assignment_id
    assert grace_assignment_id is not None
    assert (
        client.post(
            "/api/auth/login",
            json={"email": "ada@example.edu", "password": "ada secure password"},
        ).status_code
        == 200
    )

    read = client.get(f"/api/documents/{grace_document.id}")
    edit = client.put(
        f"/api/documents/{grace_document.id}/annotations",
        json={
            "spans": [],
            "expected_revision": 0,
            "mutation_id": "cross-user-write",
        },
        headers=csrf_headers(client),
    )
    complete = client.post(
        f"/api/assignments/{grace_assignment_id}/complete",
        headers=csrf_headers(client),
    )

    assert read.status_code == 404
    assert edit.status_code == 404
    assert complete.status_code == 404


def test_annotation_save_is_revisioned_and_survives_document_reload(
    hosted_client: tuple[TestClient, HostedRepository, AuthManager],
) -> None:
    client, _, _ = hosted_client
    assert (
        client.post(
            "/api/auth/login",
            json={"email": "ada@example.edu", "password": "ada secure password"},
        ).status_code
        == 200
    )
    workspace = client.get("/api/workspace").json()
    document_id = workspace["sessions"][0]["id"]

    initial = client.get(f"/api/documents/{document_id}")
    assert initial.status_code == 200
    assert initial.json()["raw_text"] == "Ada met Alice."
    assert initial.json()["reference_annotations"] is None
    assert initial.json()["manual_annotations"] == []
    assert initial.json()["annotation_revision"] == 0

    saved = client.put(
        f"/api/documents/{document_id}/annotations",
        json={
            "spans": [{"start": 8, "end": 13, "label": "NAME", "text": "Alice"}],
            "expected_revision": 0,
            "mutation_id": "browser-mutation-1",
        },
        headers=csrf_headers(client),
    )

    assert saved.status_code == 200
    assert saved.json() == {
        "revision": 1,
        "spans": [{"start": 8, "end": 13, "label": "NAME", "text": "Alice"}],
        "assignment_state": "in_progress",
    }
    reloaded = client.get(f"/api/documents/{document_id}")
    assert reloaded.status_code == 200
    assert reloaded.json()["manual_annotations"] == saved.json()["spans"]
    assert reloaded.json()["annotation_revision"] == 1
    assert reloaded.json()["assignment"]["state"] == "in_progress"
    assert (
        client.get("/api/workspace").json()["sessions"][0]["manual_annotation_count"]
        == 1
    )

    stale = client.put(
        f"/api/documents/{document_id}/annotations",
        json={
            "spans": [],
            "expected_revision": 0,
            "mutation_id": "browser-mutation-2",
        },
        headers=csrf_headers(client),
    )
    assert stale.status_code == 409
    assert stale.json() == {
        "detail": "expected revision is stale; current revision is 1"
    }


def test_platform_entity_types_are_available_for_every_session(
    hosted_client: tuple[TestClient, HostedRepository, AuthManager],
) -> None:
    client, _, _ = hosted_client
    assert (
        client.post(
            "/api/auth/login",
            json={"email": "ada@example.edu", "password": "ada secure password"},
        ).status_code
        == 200
    )
    document_id = client.get("/api/workspace").json()["sessions"][0]["id"]

    document = client.get(f"/api/documents/{document_id}")

    assert document.status_code == 200
    assert document.json()["label_set"] == [
        "NAME",
        "ADDRESS",
        "DATE",
        "PHONE_NUMBER",
        "FAX_NUMBER",
        "EMAIL",
        "SSN",
        "ACCOUNT_NUMBER",
        "DEVICE_IDENTIFIER",
        "URL",
        "IP_ADDRESS",
        "BIOMETRIC_IDENTIFIER",
        "IMAGE",
        "IDENTIFYING_NUMBER",
        "AGE",
        "SCHOOL",
        "TUTOR_PROVIDER",
        "CUSTOMIZED_FIELD",
        "OTHER_LOCATIONS_IDENTIFIED",
    ]


def test_editing_a_completed_assignment_reopens_it_for_review(
    hosted_client: tuple[TestClient, HostedRepository, AuthManager],
) -> None:
    client, _, _ = hosted_client
    assert (
        client.post(
            "/api/auth/login",
            json={"email": "ada@example.edu", "password": "ada secure password"},
        ).status_code
        == 200
    )
    document_id = client.get("/api/workspace").json()["sessions"][0]["id"]
    assignment_id = client.get(f"/api/documents/{document_id}").json()["assignment"][
        "id"
    ]

    completed = client.post(
        f"/api/assignments/{assignment_id}/complete",
        headers=csrf_headers(client),
    )

    assert completed.status_code == 200
    assert completed.json() == {
        "assignment_id": assignment_id,
        "state": "completed",
    }
    saved = client.put(
        f"/api/documents/{document_id}/annotations",
        json={
            "spans": [{"start": 0, "end": 3, "label": "NAME", "text": "Ada"}],
            "expected_revision": 0,
            "mutation_id": "after-completion",
        },
        headers=csrf_headers(client),
    )
    assert saved.status_code == 200
    assert saved.json() == {
        "revision": 1,
        "spans": [{"start": 0, "end": 3, "label": "NAME", "text": "Ada"}],
        "assignment_state": "in_progress",
    }
    detail = client.get(f"/api/documents/{document_id}").json()
    assert detail["assignment"]["state"] == "in_progress"
    assert detail["manual_annotations"] == saved.json()["spans"]
    completed_again = client.post(
        f"/api/assignments/{assignment_id}/complete",
        headers=csrf_headers(client),
    )
    assert completed_again.status_code == 200
    assert completed_again.json() == {
        "assignment_id": assignment_id,
        "state": "completed",
    }

    assert (
        client.post(
            "/api/auth/logout",
            headers=csrf_headers(client),
        ).status_code
        == 204
    )
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
    reopened = client.post(
        f"/api/admin/assignments/{assignment_id}/reopen",
        headers=csrf_headers(client),
    )
    assert reopened.status_code == 200
    assert reopened.json() == {
        "assignment_id": assignment_id,
        "state": "in_progress",
    }


def test_admin_can_track_assign_and_reassign_sessions(
    hosted_client: tuple[TestClient, HostedRepository, AuthManager],
) -> None:
    client, _, _ = hosted_client
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

    users = client.get("/api/admin/users")
    assert users.status_code == 200
    assert [user["email"] for user in users.json()] == [
        "ada@example.edu",
        "admin@example.edu",
        "grace@example.edu",
    ]
    progress = client.get("/api/admin/progress")
    assert progress.status_code == 200
    assert progress.json()["totals"] == {
        "unassigned": 1,
        "assigned": 2,
        "in_progress": 0,
        "completed": 0,
        "total": 3,
    }
    workspace = client.get("/api/workspace").json()["sessions"]
    unassigned = next(item for item in workspace if item["assignment_id"] is None)
    ada_id = next(
        user["id"] for user in users.json() if user["email"] == "ada@example.edu"
    )
    grace_id = next(
        user["id"] for user in users.json() if user["email"] == "grace@example.edu"
    )

    assigned = client.put(
        f"/api/admin/documents/{unassigned['id']}/assignment",
        json={"assignee_id": ada_id},
        headers=csrf_headers(client),
    )

    assert assigned.status_code == 200
    assignment_id = assigned.json()["assignment_id"]
    repeated = client.put(
        f"/api/admin/documents/{unassigned['id']}/assignment",
        json={"assignee_id": ada_id},
        headers=csrf_headers(client),
    )
    reassigned = client.put(
        f"/api/admin/documents/{unassigned['id']}/assignment",
        json={"assignee_id": grace_id},
        headers=csrf_headers(client),
    )
    assert repeated.json() == {"assignment_id": assignment_id}
    assert reassigned.status_code == 200
    assert reassigned.json() == {"assignment_id": assignment_id}
    assert client.get("/api/admin/progress").json()["totals"]["assigned"] == 3


def test_admin_can_create_and_list_session_folders(
    hosted_client: tuple[TestClient, HostedRepository, AuthManager],
) -> None:
    client, _, _ = hosted_client
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

    created = client.post(
        "/api/admin/folders",
        json={"name": "  August intake  "},
        headers=csrf_headers(client),
    )

    assert created.status_code == 201
    assert created.json() == {
        "id": created.json()["id"],
        "name": "August intake",
        "session_count": 0,
        "unassigned": 0,
        "assigned": 0,
        "in_progress": 0,
        "completed": 0,
    }
    assert client.get("/api/admin/folders").json() == [created.json()]


def test_admin_can_move_sessions_into_one_folder_and_workspace_exposes_membership(
    hosted_client: tuple[TestClient, HostedRepository, AuthManager],
) -> None:
    client, _, _ = hosted_client
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
    folder = client.post(
        "/api/admin/folders",
        json={"name": "August intake"},
        headers=csrf_headers(client),
    ).json()
    sessions = client.get("/api/workspace").json()["sessions"]
    assigned_ids = [
        item["id"] for item in sessions if item["assignment_state"] == "assigned"
    ]

    moved = client.put(
        f"/api/admin/folders/{folder['id']}/sessions",
        json={"document_ids": assigned_ids},
        headers=csrf_headers(client),
    )

    assert moved.status_code == 200
    assert moved.json() == {
        **folder,
        "session_count": 2,
        "assigned": 2,
    }
    assert client.get("/api/admin/progress").json()["folders"] == [moved.json()]
    reloaded = client.get("/api/workspace").json()["sessions"]
    assert {
        (item["external_id"], item["folder_id"], item["folder_name"])
        for item in reloaded
    } == {
        ("session-ada", folder["id"], "August intake"),
        ("session-grace", folder["id"], "August intake"),
        ("session-unassigned", None, None),
    }


def test_admin_can_assign_every_session_in_a_folder_atomically(
    hosted_client: tuple[TestClient, HostedRepository, AuthManager],
) -> None:
    client, _, _ = hosted_client
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
    users = client.get("/api/admin/users").json()
    grace_id = next(
        user["id"] for user in users if user["email"] == "grace@example.edu"
    )
    folder = client.post(
        "/api/admin/folders",
        json={"name": "Priority review"},
        headers=csrf_headers(client),
    ).json()
    document_ids = [
        item["id"] for item in client.get("/api/workspace").json()["sessions"]
    ]
    assert (
        client.put(
            f"/api/admin/folders/{folder['id']}/sessions",
            json={"document_ids": document_ids},
            headers=csrf_headers(client),
        ).status_code
        == 200
    )

    assigned = client.put(
        f"/api/admin/folders/{folder['id']}/assignment",
        json={"assignee_id": grace_id},
        headers=csrf_headers(client),
    )

    assert assigned.status_code == 200
    assert assigned.json()["folder_id"] == folder["id"]
    assert len(assigned.json()["assignment_ids"]) == 3
    workspace = client.get("/api/workspace").json()["sessions"]
    assert {item["assignee_id"] for item in workspace} == {grace_id}
    tracked = client.get("/api/admin/folders").json()[0]
    assert tracked["assigned"] == 3
    assert tracked["unassigned"] == 0


def test_annotator_cannot_access_admin_management(
    hosted_client: tuple[TestClient, HostedRepository, AuthManager],
) -> None:
    client, _, _ = hosted_client
    assert (
        client.post(
            "/api/auth/login",
            json={"email": "ada@example.edu", "password": "ada secure password"},
        ).status_code
        == 200
    )

    assert client.get("/api/admin/users").status_code == 403
    assert client.get("/api/admin/progress").status_code == 403


def test_admin_can_create_and_activate_an_invite_only_annotator_account(
    hosted_client: tuple[TestClient, HostedRepository, AuthManager],
) -> None:
    client, _, _ = hosted_client
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

    invitation_started_at = datetime.now(UTC)
    created = client.post(
        "/api/admin/users",
        json={
            "email": " New.Annotator@example.edu ",
            "role": "annotator",
        },
        headers=csrf_headers(client),
    )

    assert created.status_code == 201
    assert created.json()["user"] == {
        "id": created.json()["user"]["id"],
        "email": "new.annotator@example.edu",
        "display_name": "new.annotator@example.edu",
        "role": "annotator",
        "state": "pending_activation",
    }
    assert created.json()["activation_url"].startswith("/activate#token=")
    activation_expires_at = datetime.fromisoformat(
        created.json()["activation_expires_at"]
    )
    assert timedelta(hours=24) <= activation_expires_at - invitation_started_at
    assert activation_expires_at - invitation_started_at < timedelta(
        hours=24,
        seconds=2,
    )
    assert "password" not in created.text
    activation_token = created.json()["activation_url"].split("#token=", 1)[1]
    assert (
        client.post(
            "/api/auth/logout",
            headers=csrf_headers(client),
        ).status_code
        == 204
    )
    assert (
        client.post(
            "/api/auth/login",
            json={
                "email": "new.annotator@example.edu",
                "password": "temporary secure password",
            },
        ).status_code
        == 401
    )

    activated = client.post(
        "/api/auth/activate",
        json={
            "token": activation_token,
            "password": "new annotator secure password",
        },
    )

    assert activated.status_code == 200
    assert activated.json() == {
        **created.json()["user"],
        "state": "active",
    }
    assert (
        client.post(
            "/api/auth/activate",
            json={
                "token": activation_token,
                "password": "another secure password",
            },
        ).status_code
        == 400
    )
    assert (
        client.post(
            "/api/auth/login",
            json={
                "email": "new.annotator@example.edu",
                "password": "new annotator secure password",
            },
        ).status_code
        == 200
    )


def test_admin_can_invite_an_account_from_any_email_domain(
    hosted_client: tuple[TestClient, HostedRepository, AuthManager],
) -> None:
    client, _, _ = hosted_client
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

    response = client.post(
        "/api/admin/users",
        json={
            "email": "annotator@freshcognate.com",
            "role": "annotator",
        },
        headers=csrf_headers(client),
    )

    assert response.status_code == 201
    assert response.json()["user"] == {
        "id": response.json()["user"]["id"],
        "email": "annotator@freshcognate.com",
        "display_name": "annotator@freshcognate.com",
        "role": "annotator",
        "state": "pending_activation",
    }


def test_admin_password_reset_revokes_sessions_and_issues_single_use_activation(
    hosted_client: tuple[TestClient, HostedRepository, AuthManager],
) -> None:
    admin_client, repository, _ = hosted_client
    annotator_client = TestClient(admin_client.app)
    assert (
        annotator_client.post(
            "/api/auth/login",
            json={"email": "ada@example.edu", "password": "ada secure password"},
        ).status_code
        == 200
    )
    ada = repository.get_user_by_email("ada@example.edu")
    assert ada is not None
    assert (
        admin_client.post(
            "/api/auth/login",
            json={
                "email": "admin@example.edu",
                "password": "correct horse battery staple",
            },
        ).status_code
        == 200
    )

    reset = admin_client.post(
        f"/api/admin/users/{ada.id}/reset-password",
        json={},
        headers=csrf_headers(admin_client),
    )

    assert reset.status_code == 200
    assert reset.json()["user"]["state"] == "pending_activation"
    token = reset.json()["activation_url"].split("#token=", 1)[1]
    assert annotator_client.get("/api/auth/me").status_code == 401
    assert (
        annotator_client.post(
            "/api/auth/login",
            json={"email": "ada@example.edu", "password": "ada secure password"},
        ).status_code
        == 401
    )
    assert (
        annotator_client.post(
            "/api/auth/activate",
            json={"token": token, "password": "ada replacement password"},
        ).status_code
        == 200
    )
    assert (
        annotator_client.post(
            "/api/auth/login",
            json={"email": "ada@example.edu", "password": "ada replacement password"},
        ).status_code
        == 200
    )


def test_admin_deactivation_requires_and_applies_explicit_unassignment_then_reactivates(
    hosted_client: tuple[TestClient, HostedRepository, AuthManager],
) -> None:
    admin_client, repository, _ = hosted_client
    annotator_client = TestClient(admin_client.app)
    assert (
        annotator_client.post(
            "/api/auth/login",
            json={"email": "ada@example.edu", "password": "ada secure password"},
        ).status_code
        == 200
    )
    ada = repository.get_user_by_email("ada@example.edu")
    assert ada is not None
    assert (
        admin_client.post(
            "/api/auth/login",
            json={
                "email": "admin@example.edu",
                "password": "correct horse battery staple",
            },
        ).status_code
        == 200
    )

    missing_choice = admin_client.post(
        f"/api/admin/users/{ada.id}/deactivate",
        json={},
        headers=csrf_headers(admin_client),
    )
    deactivated = admin_client.post(
        f"/api/admin/users/{ada.id}/deactivate",
        json={"incomplete_assignments": {"action": "unassign"}},
        headers=csrf_headers(admin_client),
    )

    assert missing_choice.status_code == 422
    assert deactivated.status_code == 200
    assert deactivated.json()["state"] == "deactivated"
    assert annotator_client.get("/api/auth/me").status_code == 401
    assert admin_client.get("/api/admin/progress").json()["totals"] == {
        "unassigned": 2,
        "assigned": 1,
        "in_progress": 0,
        "completed": 0,
        "total": 3,
    }
    assert (
        annotator_client.post(
            "/api/auth/login",
            json={"email": "ada@example.edu", "password": "ada secure password"},
        ).status_code
        == 401
    )

    reactivated = admin_client.post(
        f"/api/admin/users/{ada.id}/reactivate",
        json={},
        headers=csrf_headers(admin_client),
    )

    assert reactivated.status_code == 200
    assert reactivated.json()["state"] == "active"
    assert (
        annotator_client.post(
            "/api/auth/login",
            json={"email": "ada@example.edu", "password": "ada secure password"},
        ).status_code
        == 200
    )


def test_admin_deactivation_can_reassign_incomplete_work_atomically(
    hosted_client: tuple[TestClient, HostedRepository, AuthManager],
) -> None:
    client, repository, _ = hosted_client
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
    ada = repository.get_user_by_email("ada@example.edu")
    grace = repository.get_user_by_email("grace@example.edu")
    assert ada is not None
    assert grace is not None

    invalid = client.post(
        f"/api/admin/users/{ada.id}/deactivate",
        json={
            "incomplete_assignments": {
                "action": "reassign",
                "assignee_id": ada.id,
            }
        },
        headers=csrf_headers(client),
    )
    assert invalid.status_code == 422
    assert client.get("/api/admin/progress").json()["totals"]["assigned"] == 2

    deactivated = client.post(
        f"/api/admin/users/{ada.id}/deactivate",
        json={
            "incomplete_assignments": {
                "action": "reassign",
                "assignee_id": grace.id,
            }
        },
        headers=csrf_headers(client),
    )

    assert deactivated.status_code == 200
    assert deactivated.json()["state"] == "deactivated"
    progress = client.get("/api/admin/progress").json()
    assert progress["totals"] == {
        "unassigned": 1,
        "assigned": 2,
        "in_progress": 0,
        "completed": 0,
        "total": 3,
    }
    grace_progress = next(
        item for item in progress["annotators"] if item["user_id"] == grace.id
    )
    assert grace_progress["assigned"] == 2


def test_deactivation_leaves_completed_assignments_attached_for_provenance(
    hosted_client: tuple[TestClient, HostedRepository, AuthManager],
) -> None:
    client, repository, _ = hosted_client
    login = client.post(
        "/api/auth/login",
        json={"email": "ada@example.edu", "password": "ada secure password"},
    )
    ada_id = login.json()["id"]
    session = client.get("/api/workspace").json()["sessions"][0]
    assert (
        client.post(
            f"/api/assignments/{session['assignment_id']}/complete",
            headers=csrf_headers(client),
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/auth/logout",
            headers=csrf_headers(client),
        ).status_code
        == 204
    )
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

    response = client.post(
        f"/api/admin/users/{ada_id}/deactivate",
        json={"incomplete_assignments": {"action": "unassign"}},
        headers=csrf_headers(client),
    )

    assert response.status_code == 200
    assert client.get("/api/admin/progress").json()["totals"] == {
        "unassigned": 1,
        "assigned": 1,
        "in_progress": 0,
        "completed": 1,
        "total": 3,
    }
    completed = client.get(f"/api/documents/{session['id']}").json()["assignment"]
    assert completed["assignee_id"] == ada_id
    assert completed["state"] == "completed"
    assert repository.get_user_by_id(ada_id).state == "deactivated"


def test_admin_can_reset_own_password_but_not_another_admin(
    hosted_client: tuple[TestClient, HostedRepository, AuthManager],
) -> None:
    client, repository, auth = hosted_client
    other_admin = repository.create_user(
        email="other-admin@example.edu",
        display_name="Other Admin",
        password_hash=auth.create_password_hash("other admin password"),
        role=Role.ADMIN,
    )
    login = client.post(
        "/api/auth/login",
        json={
            "email": "admin@example.edu",
            "password": "correct horse battery staple",
        },
    )
    admin_id = login.json()["id"]

    other_reset = client.post(
        f"/api/admin/users/{other_admin.id}/reset-password",
        json={},
        headers=csrf_headers(client),
    )
    own_reset = client.post(
        f"/api/admin/users/{admin_id}/reset-password",
        json={},
        headers=csrf_headers(client),
    )

    assert other_reset.status_code == 404
    assert other_reset.json() == {"detail": "account not found"}
    assert own_reset.status_code == 200
    assert own_reset.json()["user"]["role"] == "admin"
    assert own_reset.json()["user"]["state"] == "pending_activation"
    assert client.get("/api/auth/me").status_code == 401

    token = own_reset.json()["activation_url"].split("#token=", 1)[1]
    activation_client = TestClient(client.app)
    activation = activation_client.post(
        "/api/auth/activate",
        json={"token": token, "password": "replacement admin password"},
    )
    assert activation.status_code == 200
    assert activation.json()["role"] == "admin"
    assert activation.json()["state"] == "active"
    assert (
        activation_client.post(
            "/api/auth/login",
            json={
                "email": "admin@example.edu",
                "password": "replacement admin password",
            },
        ).status_code
        == 200
    )


def test_annotator_management_endpoints_reject_the_admin_account(
    hosted_client: tuple[TestClient, HostedRepository, AuthManager],
) -> None:
    client, _, _ = hosted_client
    login = client.post(
        "/api/auth/login",
        json={
            "email": "admin@example.edu",
            "password": "correct horse battery staple",
        },
    )
    admin_id = login.json()["id"]

    responses = [
        client.post(
            f"/api/admin/users/{admin_id}/deactivate",
            json={"incomplete_assignments": {"action": "unassign"}},
            headers=csrf_headers(client),
        ),
        client.post(
            f"/api/admin/users/{admin_id}/reactivate",
            json={},
            headers=csrf_headers(client),
        ),
    ]

    assert [response.status_code for response in responses] == [404, 404]
    assert {response.json()["detail"] for response in responses} == {
        "annotator not found"
    }


def test_admin_can_import_new_sessions_without_local_files(
    hosted_client: tuple[TestClient, HostedRepository, AuthManager],
) -> None:
    client, _, _ = hosted_client
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

    imported = client.post(
        "/api/admin/batches/import",
        json={
            "name": "September sessions",
            "sessions": [
                {
                    "external_id": "new-session-001",
                    "filename": "new-session-001.json",
                    "raw_text": "Meet Eve.",
                    "label_set": ["NAME"],
                    "reference_annotations": [
                        {"start": 5, "end": 8, "label": "NAME", "text": "Eve"}
                    ],
                }
            ],
        },
        headers=csrf_headers(client),
    )

    assert imported.status_code == 201
    assert imported.json() == {
        "batch_id": imported.json()["batch_id"],
        "imported_count": 1,
    }
    sessions = client.get("/api/workspace").json()["sessions"]
    assert any(item["external_id"] == "new-session-001" for item in sessions)


def test_admin_export_contains_manual_state_but_no_credentials(
    hosted_client: tuple[TestClient, HostedRepository, AuthManager],
) -> None:
    client, _, _ = hosted_client
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

    exported = client.get("/api/admin/export")

    assert exported.status_code == 200
    assert "sessions" in exported.json()
    assert len(exported.json()["sessions"]) == 3
    assert "password_hash" not in exported.text
    assert "token_hash" not in exported.text
