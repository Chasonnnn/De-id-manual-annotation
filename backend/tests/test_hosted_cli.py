import json
from io import StringIO

import httpx2
from hosted_app.cli import run_cli
from hosted_app.cli_credentials import Credential, InMemoryCredentialStore

BASE_URL = "https://annotation.example.com"
EMAIL = "admin@cornell.edu"
SESSION = "opaque-login-session"
CSRF = "opaque-csrf-value"


def invoke(
    args: list[str],
    handler,
    *,
    store: InMemoryCredentialStore | None = None,
    password: str = "correct horse battery staple",
    stdin_text: str = "",
):
    stdout = StringIO()
    stderr = StringIO()
    requests: list[httpx2.Request] = []

    def recording_handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(request)
        return handler(request)

    credential_store = store or InMemoryCredentialStore()
    code = run_cli(
        args,
        stdout=stdout,
        stderr=stderr,
        credential_store=credential_store,
        password_prompt=lambda _: password,
        stdin=StringIO(stdin_text),
        transport=httpx2.MockTransport(recording_handler),
    )
    return code, stdout.getvalue(), stderr.getvalue(), credential_store, requests


def logged_in_store() -> InMemoryCredentialStore:
    store = InMemoryCredentialStore()
    store.save(
        Credential(
            base_url=BASE_URL,
            email=EMAIL,
            session_token=SESSION,
            csrf_token=CSRF,
        )
    )
    return store


def assert_authenticated(request: httpx2.Request, *, csrf: bool = False) -> None:
    assert request.headers["cookie"] in {
        f"annotation_session={SESSION}; annotation_csrf={CSRF}",
        f"annotation_csrf={CSRF}; annotation_session={SESSION}",
    }
    if csrf:
        assert request.headers["x-csrf-token"] == CSRF
    assert "x-mutation-id" not in request.headers


def test_login_prompts_for_password_and_saves_only_successful_session() -> None:
    password = "not-for-arguments-or-output"

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.method == "POST"
        assert request.url == f"{BASE_URL}/api/auth/login"
        assert json.loads(request.content) == {"email": EMAIL, "password": password}
        return httpx2.Response(
            200,
            json={
                "id": "admin-1",
                "email": EMAIL,
                "display_name": "Admin",
                "role": "admin",
                "state": "active",
            },
            headers=[
                ("set-cookie", f"annotation_session={SESSION}; Path=/; HttpOnly"),
                ("set-cookie", f"annotation_csrf={CSRF}; Path=/"),
            ],
        )

    code, stdout, stderr, store, _ = invoke(
        ["login", "--url", f"{BASE_URL}/", "--email", EMAIL],
        handler,
        password=password,
    )

    assert code == 0
    assert stdout == f"Logged in as {EMAIL} (admin).\n"
    assert stderr == ""
    assert password not in stdout + stderr
    assert SESSION not in stdout + stderr
    assert CSRF not in stdout + stderr
    assert store.load() == Credential(BASE_URL, EMAIL, SESSION, CSRF)


def test_login_can_read_password_from_stdin_without_prompting_or_printing_it() -> None:
    password = "stdin-only-bootstrap-password"

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert json.loads(request.content) == {"email": EMAIL, "password": password}
        return httpx2.Response(
            200,
            json={
                "id": "admin-1",
                "email": EMAIL,
                "display_name": "Admin",
                "role": "admin",
                "state": "active",
            },
            headers=[
                ("set-cookie", f"annotation_session={SESSION}; Path=/; HttpOnly"),
                ("set-cookie", f"annotation_csrf={CSRF}; Path=/"),
            ],
        )

    code, stdout, stderr, store, requests = invoke(
        [
            "login",
            "--url",
            BASE_URL,
            "--email",
            EMAIL,
            "--password-stdin",
        ],
        handler,
        stdin_text=password + "\n",
        password="prompt-must-not-be-used",
    )

    assert code == 0
    assert len(requests) == 1
    assert password not in stdout + stderr
    assert store.load() == Credential(BASE_URL, EMAIL, SESSION, CSRF)


def test_failed_login_does_not_write_keychain_and_preserves_server_detail() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(401, json={"detail": "invalid email or password"})

    code, stdout, stderr, store, _ = invoke(
        ["login", "--url", BASE_URL, "--email", EMAIL],
        handler,
    )

    assert code == 1
    assert stdout == ""
    assert stderr == "error: HTTP 401: invalid email or password\n"
    assert store.load() is None


def test_whoami_json_uses_the_saved_human_session() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.method == "GET"
        assert request.url == f"{BASE_URL}/api/auth/me"
        assert_authenticated(request)
        return httpx2.Response(
            200,
            json={
                "id": "admin-1",
                "email": EMAIL,
                "display_name": "Admin",
                "role": "admin",
                "state": "active",
            },
        )

    code, stdout, stderr, _, _ = invoke(
        ["whoami", "--json"], handler, store=logged_in_store()
    )

    assert code == 0
    assert json.loads(stdout) == {
        "display_name": "Admin",
        "email": EMAIL,
        "id": "admin-1",
        "role": "admin",
        "state": "active",
    }
    assert stderr == ""
    assert SESSION not in stdout
    assert CSRF not in stdout


def test_read_commands_support_json_without_transcript_content() -> None:
    responses = {
        "/api/admin/users": [
            {
                "id": "annotator-1",
                "email": "ada@cornell.edu",
                "display_name": "Ada",
                "role": "annotator",
                "state": "active",
            }
        ],
        "/api/workspace": {
            "sessions": [
                {
                    "id": "document-1",
                    "external_id": "saga-001",
                    "filename": "session.json",
                    "assignment_id": None,
                    "assignment_state": None,
                    "assignee_id": None,
                    "assignee_name": None,
                }
            ]
        },
        "/api/admin/progress": {
            "totals": {
                "unassigned": 1,
                "assigned": 0,
                "in_progress": 0,
                "completed": 0,
                "total": 1,
            },
            "annotators": [],
        },
    }

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert_authenticated(request)
        return httpx2.Response(200, json=responses[request.url.path])

    for args, expected in (
        (["users", "list", "--json"], responses["/api/admin/users"]),
        (["sessions", "list", "--json"], responses["/api/workspace"]["sessions"]),
        (["status", "--json"], responses["/api/admin/progress"]),
    ):
        code, stdout, stderr, _, _ = invoke(args, handler, store=logged_in_store())
        assert code == 0
        assert json.loads(stdout) == expected
        assert stderr == ""
        assert "raw_text" not in stdout
        assert SESSION not in stdout


def test_assignment_set_and_reopen_send_csrf_and_report_server_results() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        assert_authenticated(request, csrf=True)
        if request.url.path == "/api/admin/documents/document-1/assignment":
            assert request.method == "PUT"
            assert json.loads(request.content) == {"assignee_id": "annotator-1"}
            return httpx2.Response(200, json={"assignment_id": "assignment-1"})
        assert request.method == "POST"
        assert request.url.path == "/api/admin/assignments/assignment-1/reopen"
        return httpx2.Response(
            200,
            json={"assignment_id": "assignment-1", "state": "in_progress"},
        )

    store = logged_in_store()
    set_result = invoke(
        [
            "assignments",
            "set",
            "--document-id",
            "document-1",
            "--assignee-id",
            "annotator-1",
        ],
        handler,
        store=store,
    )
    reopen_result = invoke(
        [
            "assignments",
            "reopen",
            "--assignment-id",
            "assignment-1",
        ],
        handler,
        store=store,
    )

    assert set_result[:3] == (
        0,
        "Assignment assignment-1 updated.\n",
        "",
    )
    assert reopen_result[:3] == (
        0,
        "Assignment assignment-1 reopened.\n",
        "",
    )


def test_bulk_assignment_preview_and_apply_use_the_server_plan_contract() -> None:
    plan_digest = "a" * 64

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert_authenticated(request, csrf=True)
        assert request.method == "POST"
        body = json.loads(request.content)
        assert body["document_ids"] == ["document-2", "document-1"]
        assert body["annotator_ids"] == ["annotator-2", "annotator-1"]
        if request.url.path.endswith("/preview"):
            assert set(body) == {"document_ids", "annotator_ids"}
            return httpx2.Response(
                200,
                json={
                    "plan_digest": plan_digest,
                    "assignments": [
                        {"document_id": "document-1", "assignee_id": "annotator-1"},
                        {"document_id": "document-2", "assignee_id": "annotator-2"},
                    ],
                    "document_preconditions": [],
                    "annotator_preconditions": [],
                },
            )
        assert request.url.path.endswith("/apply")
        assert body["plan_digest"] == plan_digest
        assert body["mutation_id"] == "bulk-001"
        return httpx2.Response(
            200,
            json={
                "plan_digest": plan_digest,
                "mutation_id": "bulk-001",
                "assignment_ids": ["assignment-1", "assignment-2"],
            },
        )

    common = [
        "--document-id",
        "document-2",
        "--document-id",
        "document-1",
        "--annotator-id",
        "annotator-2",
        "--annotator-id",
        "annotator-1",
    ]
    preview = invoke(
        ["assignments", "preview", *common, "--json"],
        handler,
        store=logged_in_store(),
    )
    applied = invoke(
        [
            "assignments",
            "apply",
            *common,
            "--plan-digest",
            plan_digest,
            "--mutation-id",
            "bulk-001",
            "--json",
        ],
        handler,
        store=logged_in_store(),
    )

    assert preview[0] == applied[0] == 0
    assert json.loads(preview[1])["plan_digest"] == plan_digest
    assert json.loads(applied[1]) == {
        "assignment_ids": ["assignment-1", "assignment-2"],
        "mutation_id": "bulk-001",
        "plan_digest": plan_digest,
    }
    assert preview[2] == applied[2] == ""


def test_s3_batch_plan_and_apply_use_manifest_and_exact_plan_files(tmp_path) -> None:
    manifest = {
        "name": "Saga pilot",
        "documents": [
            {
                "external_id": "session-1",
                "filename": "session-1.txt",
                "label_set": ["NAME"],
                "raw_format": "saga_zip_transcript",
                "reference_format": None,
                "raw": {
                    "bucket": "governed-data",
                    "key": "raw/batch.zip",
                    "version_id": "raw-v1",
                    "archive_member": "batch/session-1/transcript.json",
                },
                "reference": None,
            }
        ],
    }
    plan = {
        "manifest_digest": "b" * 64,
        "sources": [
            {
                "external_id": "session-1",
                "kind": "raw",
                "format": "saga_zip_transcript",
                "archive_member": "batch/session-1/transcript.json",
                "bucket": "governed-data",
                "key": "raw/batch.zip",
                "version_id": "raw-v1",
                "content_length": 100,
                "etag": '"etag"',
                "last_modified": "2026-08-29T02:00:00Z",
                "server_checksum_algorithm": None,
                "server_checksum_value": None,
                "checksum_type": None,
            }
        ],
    }
    manifest_path = tmp_path / "manifest.json"
    plan_path = tmp_path / "plan.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    plan_path.write_text(json.dumps(plan), encoding="utf-8")

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert_authenticated(request, csrf=True)
        assert request.method == "POST"
        body = json.loads(request.content)
        assert body["manifest"] == manifest
        if request.url.path.endswith("/plan"):
            assert set(body) == {"manifest"}
            return httpx2.Response(200, json=plan)
        assert request.url.path.endswith("/apply")
        assert body == {
            "manifest": manifest,
            "expected_manifest_digest": plan["manifest_digest"],
            "expected_sources": plan["sources"],
            "mutation_id": "import-001",
        }
        return httpx2.Response(
            201,
            json={"batch_id": "batch-1", "imported_count": 1},
        )

    planned = invoke(
        ["batches", "plan-import", "--manifest", str(manifest_path), "--json"],
        handler,
        store=logged_in_store(),
    )
    applied = invoke(
        [
            "batches",
            "apply-import",
            "--manifest",
            str(manifest_path),
            "--plan",
            str(plan_path),
            "--mutation-id",
            "import-001",
            "--json",
        ],
        handler,
        store=logged_in_store(),
    )

    assert planned[0] == applied[0] == 0
    assert json.loads(planned[1]) == plan
    assert json.loads(applied[1]) == {"batch_id": "batch-1", "imported_count": 1}
    assert planned[2] == applied[2] == ""


def test_audit_list_passes_filters_and_supports_stable_json() -> None:
    event = {
        "id": "event-1",
        "actor_id": "admin-1",
        "action": "assignment.bulk_applied",
        "target_type": "assignment_plan",
        "target_id": "digest-1",
        "before_metadata": {"state": "previewed"},
        "after_metadata": {"state": "applied", "assignment_ids": ["a-1"]},
        "mutation_id": "bulk-001",
        "occurred_at": "2026-08-29T02:00:00Z",
        "result": "success",
        "reason": None,
    }

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert_authenticated(request)
        assert request.method == "GET"
        assert request.url.path == "/api/admin/audit"
        assert dict(request.url.params) == {
            "actor_id": "admin-1",
            "action": "assignment.bulk_applied",
            "target_type": "assignment_plan",
            "target_id": "digest-1",
            "mutation_id": "bulk-001",
            "result": "success",
            "limit": "25",
        }
        return httpx2.Response(200, json=[event])

    code, stdout, stderr, _, _ = invoke(
        [
            "audit",
            "list",
            "--actor-id",
            "admin-1",
            "--action",
            "assignment.bulk_applied",
            "--target-type",
            "assignment_plan",
            "--target-id",
            "digest-1",
            "--mutation-id",
            "bulk-001",
            "--result",
            "success",
            "--limit",
            "25",
            "--json",
        ],
        handler,
        store=logged_in_store(),
    )

    assert code == 0
    assert json.loads(stdout) == [event]
    assert stderr == ""


def test_logout_revokes_server_session_then_clears_keychain() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.method == "POST"
        assert request.url.path == "/api/auth/logout"
        assert_authenticated(request, csrf=True)
        return httpx2.Response(204)

    store = logged_in_store()
    code, stdout, stderr, _, _ = invoke(["logout"], handler, store=store)

    assert code == 0
    assert stdout == "Logged out.\n"
    assert stderr == ""
    assert store.load() is None


def test_user_lifecycle_uses_fixed_backend_contract_and_outputs_activation_once() -> (
    None
):
    def handler(request: httpx2.Request) -> httpx2.Response:
        assert_authenticated(request, csrf=True)
        path = request.url.path
        if path == "/api/admin/users":
            assert request.method == "POST"
            assert json.loads(request.content) == {
                "email": "ada@cornell.edu",
                "display_name": "Ada Lovelace",
                "role": "annotator",
            }
            return httpx2.Response(
                201,
                json={
                    "user": {
                        "id": "annotator-1",
                        "email": "ada@cornell.edu",
                        "display_name": "Ada Lovelace",
                        "role": "annotator",
                        "state": "pending_activation",
                    },
                    "activation_url": "/activate#token=one-time",
                    "activation_expires_at": "2026-08-29T20:00:00Z",
                },
            )
        if path.endswith("/deactivate"):
            assert json.loads(request.content) == {
                "incomplete_assignments": {
                    "action": "reassign",
                    "assignee_id": "annotator-2",
                }
            }
            return httpx2.Response(
                200, json={"id": "annotator-1", "state": "deactivated"}
            )
        if path.endswith("/reactivate"):
            assert json.loads(request.content) == {}
            return httpx2.Response(200, json={"id": "annotator-1", "state": "active"})
        assert path.endswith("/reset-password")
        assert json.loads(request.content) == {}
        return httpx2.Response(
            200,
            json={
                "user": {"id": "annotator-1", "state": "pending_activation"},
                "activation_url": "/activate#token=reset-once",
                "activation_expires_at": "2026-08-29T21:00:00Z",
            },
        )

    store = logged_in_store()
    create = invoke(
        [
            "users",
            "create",
            "--email",
            "ada@cornell.edu",
            "--display-name",
            "Ada Lovelace",
        ],
        handler,
        store=store,
    )
    deactivate = invoke(
        [
            "users",
            "deactivate",
            "--user-id",
            "annotator-1",
            "--reassign-to",
            "annotator-2",
        ],
        handler,
        store=store,
    )
    reactivate = invoke(
        [
            "users",
            "reactivate",
            "--user-id",
            "annotator-1",
        ],
        handler,
        store=store,
    )
    reset = invoke(
        [
            "users",
            "reset-password",
            "--user-id",
            "annotator-1",
        ],
        handler,
        store=store,
    )

    assert create[0] == deactivate[0] == reactivate[0] == reset[0] == 0
    assert create[1] == (
        "User ada@cornell.edu created.\n"
        "Single-use activation URL (expires 2026-08-29T20:00:00Z): "
        "https://annotation.example.com/activate#token=one-time\n"
    )
    assert deactivate[1] == "User annotator-1 deactivated.\n"
    assert reactivate[1] == "User annotator-1 reactivated.\n"
    assert reset[1] == (
        "Password reset started for annotator-1.\n"
        "Single-use activation URL (expires 2026-08-29T21:00:00Z): "
        "https://annotation.example.com/activate#token=reset-once\n"
    )


def test_activation_output_accepts_only_the_fragment_contract() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        assert_authenticated(request, csrf=True)
        return httpx2.Response(
            201,
            json={
                "user": {"email": "ada@cornell.edu"},
                "activation_url": "/activate?token=must-not-print",
                "activation_expires_at": "2026-08-29T20:00:00Z",
            },
        )

    code, stdout, stderr, _, _ = invoke(
        [
            "users",
            "create",
            "--email",
            "ada@cornell.edu",
            "--display-name",
            "Ada Lovelace",
        ],
        handler,
        store=logged_in_store(),
    )

    assert code == 1
    assert stdout == ""
    assert stderr == "error: server returned an invalid response\n"
    assert "must-not-print" not in stderr


def test_missing_or_expired_credentials_fail_explicitly() -> None:
    def no_request(request: httpx2.Request) -> httpx2.Response:
        raise AssertionError("no HTTP request expected")

    code, stdout, stderr, _, requests = invoke(["whoami"], no_request)
    assert code == 1
    assert stdout == ""
    assert stderr == "error: not logged in; run annotationctl login\n"
    assert requests == []

    def expired(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(401, json={"detail": "Authentication required"})

    code, stdout, stderr, _, _ = invoke(["whoami"], expired, store=logged_in_store())
    assert code == 1
    assert stdout == ""
    assert stderr == "error: login expired; run annotationctl login again\n"
