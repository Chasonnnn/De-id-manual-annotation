from __future__ import annotations

import argparse
import getpass
import json
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TextIO
from urllib.parse import urljoin, urlsplit
from uuid import uuid4

import httpx2

from .cli_credentials import (
    Credential,
    CredentialStore,
    CredentialStoreError,
    MacOSKeychainCredentialStore,
)


class CliError(RuntimeError):
    pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="annotationctl")
    commands = parser.add_subparsers(dest="command", required=True)

    login = commands.add_parser("login")
    login.add_argument("--url", required=True)
    login.add_argument("--email", required=True)
    login.add_argument("--password-stdin", action="store_true")

    whoami = commands.add_parser("whoami")
    whoami.add_argument("--json", action="store_true")

    commands.add_parser("logout")

    users = commands.add_parser("users").add_subparsers(
        dest="users_command", required=True
    )
    users_list = users.add_parser("list")
    users_list.add_argument("--json", action="store_true")
    users_create = users.add_parser("create")
    users_create.add_argument("--email", required=True)
    users_deactivate = users.add_parser("deactivate")
    users_deactivate.add_argument("--user-id", required=True)
    incomplete = users_deactivate.add_mutually_exclusive_group(required=True)
    incomplete.add_argument("--unassign", action="store_true")
    incomplete.add_argument("--reassign-to")
    users_reactivate = users.add_parser("reactivate")
    users_reactivate.add_argument("--user-id", required=True)
    users_reset = users.add_parser("reset-password")
    users_reset.add_argument("--user-id", required=True)

    sessions = commands.add_parser("sessions").add_subparsers(
        dest="sessions_command", required=True
    )
    sessions_list = sessions.add_parser("list")
    sessions_list.add_argument("--json", action="store_true")

    status = commands.add_parser("status")
    status.add_argument("--json", action="store_true")

    folders = commands.add_parser("folders").add_subparsers(
        dest="folders_command", required=True
    )
    folders_list = folders.add_parser("list")
    folders_list.add_argument("--json", action="store_true")
    folders_create = folders.add_parser("create")
    folders_create.add_argument("--name", required=True)
    folders_create.add_argument("--json", action="store_true")
    folders_move = folders.add_parser("move")
    folders_move.add_argument("--folder-id", required=True)
    folders_move.add_argument("--document-id", action="append", required=True)
    folders_move.add_argument("--json", action="store_true")
    folders_assign = folders.add_parser("assign")
    folders_assign.add_argument("--folder-id", required=True)
    folders_assign.add_argument("--assignee-id", required=True)
    folders_assign.add_argument("--json", action="store_true")

    assignments = commands.add_parser("assignments").add_subparsers(
        dest="assignments_command", required=True
    )
    assignment_set = assignments.add_parser("set")
    assignment_set.add_argument("--document-id", required=True)
    assignment_set.add_argument("--assignee-id", required=True)
    assignment_reopen = assignments.add_parser("reopen")
    assignment_reopen.add_argument("--assignment-id", required=True)
    for bulk_command in ("preview", "apply"):
        bulk = assignments.add_parser(bulk_command)
        bulk.add_argument("--document-id", action="append", required=True)
        bulk.add_argument("--annotator-id", action="append", required=True)
        bulk.add_argument("--json", action="store_true")
        if bulk_command == "apply":
            bulk.add_argument("--plan-digest", required=True)
            bulk.add_argument("--mutation-id")

    audit = commands.add_parser("audit").add_subparsers(
        dest="audit_command", required=True
    )
    audit_list = audit.add_parser("list")
    audit_list.add_argument("--actor-id")
    audit_list.add_argument("--action")
    audit_list.add_argument("--target-type")
    audit_list.add_argument("--target-id")
    audit_list.add_argument("--mutation-id")
    audit_list.add_argument("--result", choices=("success",))
    audit_list.add_argument("--limit", type=int, default=100)
    audit_list.add_argument("--json", action="store_true")

    batches = commands.add_parser("batches").add_subparsers(
        dest="batches_command", required=True
    )
    plan_import = batches.add_parser("plan-import")
    plan_import.add_argument("--manifest", required=True)
    plan_import.add_argument("--json", action="store_true")
    apply_import = batches.add_parser("apply-import")
    apply_import.add_argument("--manifest", required=True)
    apply_import.add_argument("--plan", required=True)
    apply_import.add_argument("--mutation-id")
    apply_import.add_argument("--json", action="store_true")

    return parser


class ApiClient:
    def __init__(
        self,
        credential: Credential,
        *,
        transport: httpx2.BaseTransport | None,
    ) -> None:
        self.credential = credential
        self._client = httpx2.Client(
            base_url=credential.base_url,
            cookies={
                "annotation_session": credential.session_token,
                "annotation_csrf": credential.csrf_token,
            },
            transport=transport,
            timeout=15,
        )

    def request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        csrf: bool = False,
    ) -> Any:
        headers: dict[str, str] = {}
        if csrf:
            headers["X-CSRF-Token"] = self.credential.csrf_token
        response = self._client.request(
            method,
            path,
            json=body,
            params=params,
            headers=headers,
        )
        _raise_for_status(response, authenticated=True)
        if response.status_code == 204:
            return None
        return response.json()


def _raise_for_status(response: httpx2.Response, *, authenticated: bool) -> None:
    if response.status_code < 400:
        return
    if authenticated and response.status_code == 401:
        raise CliError("login expired; run annotationctl login again")
    detail = None
    try:
        body = response.json()
        if isinstance(body, dict) and isinstance(body.get("detail"), str):
            detail = body["detail"]
    except json.JSONDecodeError, UnicodeDecodeError:
        pass
    message = f"HTTP {response.status_code}"
    if detail:
        message += f": {detail}"
    raise CliError(message)


def _load_client(
    store: CredentialStore,
    *,
    transport: httpx2.BaseTransport | None,
) -> ApiClient:
    credential = store.load()
    if credential is None:
        raise CliError("not logged in; run annotationctl login")
    return ApiClient(credential, transport=transport)


def _json_output(stdout: TextIO, value: Any) -> None:
    stdout.write(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _activation_url(client: ApiClient, value: object) -> str:
    if not isinstance(value, str):
        raise CliError("server returned an invalid response")
    parsed = urlsplit(value)
    if (
        parsed.scheme
        or parsed.netloc
        or parsed.query
        or parsed.path != "/activate"
        or not parsed.fragment.startswith("token=")
        or not parsed.fragment.removeprefix("token=")
    ):
        raise CliError("server returned an invalid response")
    return urljoin(client.credential.base_url + "/", value)


def _table(
    stdout: TextIO, headers: Sequence[str], rows: Sequence[Sequence[Any]]
) -> None:
    text_rows = [["" if cell is None else str(cell) for cell in row] for row in rows]
    widths = [len(header) for header in headers]
    for row in text_rows:
        widths = [
            max(width, len(cell)) for width, cell in zip(widths, row, strict=True)
        ]
    stdout.write(
        "  ".join(
            header.ljust(width) for header, width in zip(headers, widths, strict=True)
        ).rstrip()
        + "\n"
    )
    for row in text_rows:
        stdout.write(
            "  ".join(
                cell.ljust(width) for cell, width in zip(row, widths, strict=True)
            ).rstrip()
            + "\n"
        )


def _run_login(
    args: argparse.Namespace,
    *,
    stdout: TextIO,
    stdin: TextIO,
    store: CredentialStore,
    password_prompt: Callable[[str], str],
    transport: httpx2.BaseTransport | None,
) -> None:
    base_url = args.url.rstrip("/")
    password = (
        stdin.readline().removesuffix("\n").removesuffix("\r")
        if args.password_stdin
        else password_prompt("Password: ")
    )
    if not password:
        raise CliError("password is required")
    with httpx2.Client(base_url=base_url, transport=transport, timeout=15) as client:
        response = client.post(
            "/api/auth/login",
            json={"email": args.email, "password": password},
        )
        _raise_for_status(response, authenticated=False)
        session_token = client.cookies.get("annotation_session")
        csrf_token = client.cookies.get("annotation_csrf")
    if not session_token or not csrf_token:
        raise CliError("login response did not include required session cookies")
    user = response.json()
    store.save(
        Credential(
            base_url=base_url,
            email=user["email"],
            session_token=session_token,
            csrf_token=csrf_token,
        )
    )
    stdout.write(f"Logged in as {user['email']} ({user['role']}).\n")


def _run_users(
    args: argparse.Namespace,
    *,
    stdout: TextIO,
    client: ApiClient,
) -> None:
    command = args.users_command
    if command == "list":
        users = client.request("GET", "/api/admin/users")
        if args.json:
            _json_output(stdout, users)
        else:
            _table(
                stdout,
                ("ID", "EMAIL", "ROLE", "STATE"),
                [
                    (
                        user["id"],
                        user["email"],
                        user["role"],
                        user["state"],
                    )
                    for user in users
                ],
            )
        return

    if command == "create":
        result = client.request(
            "POST",
            "/api/admin/users",
            body={
                "email": args.email,
                "role": "annotator",
            },
            csrf=True,
        )
        activation_url = _activation_url(client, result["activation_url"])
        stdout.write(
            f"User {result['user']['email']} created.\n"
            f"Single-use activation URL (expires {result['activation_expires_at']}): "
            f"{activation_url}\n"
        )
        return
    if command == "deactivate":
        incomplete = (
            {"action": "unassign"}
            if args.unassign
            else {"action": "reassign", "assignee_id": args.reassign_to}
        )
        client.request(
            "POST",
            f"/api/admin/users/{args.user_id}/deactivate",
            body={"incomplete_assignments": incomplete},
            csrf=True,
        )
        stdout.write(f"User {args.user_id} deactivated.\n")
        return
    if command == "reactivate":
        client.request(
            "POST",
            f"/api/admin/users/{args.user_id}/reactivate",
            body={},
            csrf=True,
        )
        stdout.write(f"User {args.user_id} reactivated.\n")
        return
    result = client.request(
        "POST",
        f"/api/admin/users/{args.user_id}/reset-password",
        body={},
        csrf=True,
    )
    activation_url = _activation_url(client, result["activation_url"])
    stdout.write(
        f"Password reset started for {args.user_id}.\n"
        f"Single-use activation URL (expires {result['activation_expires_at']}): "
        f"{activation_url}\n"
    )


def _run_assignments(
    args: argparse.Namespace,
    *,
    stdout: TextIO,
    client: ApiClient,
) -> None:
    command = args.assignments_command
    if command == "set":
        result = client.request(
            "PUT",
            f"/api/admin/documents/{args.document_id}/assignment",
            body={"assignee_id": args.assignee_id},
            csrf=True,
        )
        stdout.write(f"Assignment {result['assignment_id']} updated.\n")
        return
    if command == "reopen":
        result = client.request(
            "POST",
            f"/api/admin/assignments/{args.assignment_id}/reopen",
            body={},
            csrf=True,
        )
        stdout.write(f"Assignment {result['assignment_id']} reopened.\n")
        return

    body: dict[str, Any] = {
        "document_ids": args.document_id,
        "annotator_ids": args.annotator_id,
    }
    if command == "apply":
        body["plan_digest"] = args.plan_digest
        body["mutation_id"] = args.mutation_id or str(uuid4())
    result = client.request(
        "POST",
        f"/api/admin/assignments/bulk/{command}",
        body=body,
        csrf=True,
    )
    if args.json:
        _json_output(stdout, result)
    elif command == "preview":
        _table(
            stdout,
            ("DOCUMENT", "ASSIGNEE"),
            [
                (item["document_id"], item["assignee_id"])
                for item in result["assignments"]
            ],
        )
        stdout.write(f"Plan digest: {result['plan_digest']}\n")
    else:
        stdout.write(
            f"Applied {len(result['assignment_ids'])} assignments.\n"
            f"Mutation ID: {result['mutation_id']}\n"
        )


def _run_folders(
    args: argparse.Namespace,
    *,
    stdout: TextIO,
    client: ApiClient,
) -> None:
    command = args.folders_command
    if command == "list":
        result = client.request("GET", "/api/admin/folders")
    elif command == "create":
        result = client.request(
            "POST",
            "/api/admin/folders",
            body={"name": args.name},
            csrf=True,
        )
    elif command == "move":
        result = client.request(
            "PUT",
            f"/api/admin/folders/{args.folder_id}/sessions",
            body={"document_ids": args.document_id},
            csrf=True,
        )
    else:
        result = client.request(
            "PUT",
            f"/api/admin/folders/{args.folder_id}/assignment",
            body={"assignee_id": args.assignee_id},
            csrf=True,
        )
    if args.json:
        _json_output(stdout, result)
        return
    if command == "list":
        _table(
            stdout,
            ("ID", "NAME", "SESSIONS", "UNASSIGNED", "IN PROGRESS", "COMPLETED"),
            [
                (
                    folder["id"],
                    folder["name"],
                    folder["session_count"],
                    folder["unassigned"],
                    folder["in_progress"],
                    folder["completed"],
                )
                for folder in result
            ],
        )
    elif command == "create":
        stdout.write(f"Created folder {result['id']} ({result['name']}).\n")
    elif command == "move":
        stdout.write(
            f"Moved {len(args.document_id)} sessions to folder {args.folder_id}.\n"
        )
    else:
        stdout.write(
            f"Assigned {len(result['assignment_ids'])} sessions in folder "
            f"{args.folder_id}.\n"
        )


def _run_audit(
    args: argparse.Namespace,
    *,
    stdout: TextIO,
    client: ApiClient,
) -> None:
    params = {
        key: value
        for key, value in {
            "actor_id": args.actor_id,
            "action": args.action,
            "target_type": args.target_type,
            "target_id": args.target_id,
            "mutation_id": args.mutation_id,
            "result": args.result,
            "limit": args.limit,
        }.items()
        if value is not None
    }
    events = client.request("GET", "/api/admin/audit", params=params)
    if args.json:
        _json_output(stdout, events)
        return
    _table(
        stdout,
        ("TIME", "ACTOR", "ACTION", "TARGET", "MUTATION"),
        [
            (
                event["occurred_at"],
                event["actor_id"],
                event["action"],
                f"{event['target_type']}:{event['target_id']}",
                event["mutation_id"],
            )
            for event in events
        ],
    )


def _read_json_object(path: str, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CliError(f"could not read valid {description} JSON") from error
    if not isinstance(value, dict):
        raise CliError(f"{description} JSON must be an object")
    return value


def _run_batches(
    args: argparse.Namespace,
    *,
    stdout: TextIO,
    client: ApiClient,
) -> None:
    manifest = _read_json_object(args.manifest, description="manifest")
    if args.batches_command == "plan-import":
        result = client.request(
            "POST",
            "/api/admin/s3-imports/plan",
            body={"manifest": manifest},
            csrf=True,
        )
        if args.json:
            _json_output(stdout, result)
        else:
            stdout.write(
                f"Planned {len(result['sources'])} source bindings.\n"
                f"Manifest digest: {result['manifest_digest']}\n"
            )
        return

    plan = _read_json_object(args.plan, description="plan")
    digest = plan.get("manifest_digest")
    sources = plan.get("sources")
    if not isinstance(digest, str) or not isinstance(sources, list):
        raise CliError("plan JSON must contain manifest_digest and sources")
    mutation_id = args.mutation_id or str(uuid4())
    result = client.request(
        "POST",
        "/api/admin/s3-imports/apply",
        body={
            "manifest": manifest,
            "expected_manifest_digest": digest,
            "expected_sources": sources,
            "mutation_id": mutation_id,
        },
        csrf=True,
    )
    if args.json:
        _json_output(stdout, result)
    else:
        stdout.write(
            f"Imported {result['imported_count']} sessions into batch "
            f"{result['batch_id']}.\nMutation ID: {mutation_id}\n"
        )


def run_cli(
    argv: Sequence[str],
    *,
    stdout: TextIO,
    stderr: TextIO,
    stdin: TextIO,
    credential_store: CredentialStore,
    password_prompt: Callable[[str], str] = getpass.getpass,
    transport: httpx2.BaseTransport | None = None,
) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "login":
            _run_login(
                args,
                stdout=stdout,
                stdin=stdin,
                store=credential_store,
                password_prompt=password_prompt,
                transport=transport,
            )
            return 0

        client = _load_client(credential_store, transport=transport)
        if args.command == "whoami":
            user = client.request("GET", "/api/auth/me")
            if args.json:
                _json_output(stdout, user)
            else:
                stdout.write(f"{user['email']} ({user['role']}, {user['state']})\n")
        elif args.command == "logout":
            client.request("POST", "/api/auth/logout", body={}, csrf=True)
            credential_store.delete()
            stdout.write("Logged out.\n")
        elif args.command == "users":
            _run_users(args, stdout=stdout, client=client)
        elif args.command == "sessions":
            sessions = client.request("GET", "/api/workspace")["sessions"]
            if args.json:
                _json_output(stdout, sessions)
            else:
                _table(
                    stdout,
                    ("ID", "EXTERNAL ID", "FOLDER", "FILE", "STATE", "ASSIGNEE"),
                    [
                        (
                            item["id"],
                            item["external_id"],
                            item.get("folder_name") or "Unfiled",
                            item["filename"],
                            item["assignment_state"] or "unassigned",
                            item["assignee_name"],
                        )
                        for item in sessions
                    ],
                )
        elif args.command == "status":
            status = client.request("GET", "/api/admin/progress")
            if args.json:
                _json_output(stdout, status)
            else:
                totals = status["totals"]
                _table(
                    stdout,
                    ("UNASSIGNED", "ASSIGNED", "IN PROGRESS", "COMPLETED", "TOTAL"),
                    [
                        (
                            totals["unassigned"],
                            totals["assigned"],
                            totals["in_progress"],
                            totals["completed"],
                            totals["total"],
                        )
                    ],
                )
                if status.get("folders"):
                    _table(
                        stdout,
                        (
                            "FOLDER",
                            "SESSIONS",
                            "UNASSIGNED",
                            "ASSIGNED",
                            "IN PROGRESS",
                            "COMPLETED",
                        ),
                        [
                            (
                                folder["name"],
                                folder["session_count"],
                                folder["unassigned"],
                                folder["assigned"],
                                folder["in_progress"],
                                folder["completed"],
                            )
                            for folder in status["folders"]
                        ],
                    )
        elif args.command == "folders":
            _run_folders(args, stdout=stdout, client=client)
        elif args.command == "audit":
            _run_audit(args, stdout=stdout, client=client)
        elif args.command == "batches":
            _run_batches(args, stdout=stdout, client=client)
        else:
            _run_assignments(args, stdout=stdout, client=client)
        return 0
    except (
        CliError,
        CredentialStoreError,
        httpx2.RequestError,
        KeyError,
        ValueError,
    ) as error:
        if isinstance(error, httpx2.RequestError):
            message = "request failed; check the application URL and network connection"
        elif isinstance(error, (KeyError, ValueError)):
            message = "server returned an invalid response"
        else:
            message = str(error)
        stderr.write(f"error: {message}\n")
        return 1


def main() -> None:
    raise SystemExit(
        run_cli(
            sys.argv[1:],
            stdout=sys.stdout,
            stderr=sys.stderr,
            stdin=sys.stdin,
            credential_store=MacOSKeychainCredentialStore(),
        )
    )
