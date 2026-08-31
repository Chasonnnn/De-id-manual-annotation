from __future__ import annotations

import io
import json
import zipfile
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine

from hosted_app.api import create_hosted_app
from hosted_app.auth import AuthManager
from hosted_app.database import create_schema
from hosted_app.domain import Role
from hosted_app.repository import HostedRepository
from hosted_app.s3_import import (
    DownloadedS3Object,
    S3CatalogConfig,
    S3ObjectIdentity,
)


class InMemoryS3Reader:
    def __init__(self) -> None:
        self.objects: dict[str, tuple[S3ObjectIdentity, bytes]] = {}
        self.read_count = 0

    def seed(
        self,
        *,
        key: str,
        body: bytes,
        version_id: str | None,
        etag: str,
    ) -> None:
        self.objects[key] = (
            S3ObjectIdentity(
                bucket="governed-data",
                key=key,
                version_id=version_id,
                content_length=len(body),
                etag=etag,
                last_modified=datetime(2026, 8, 28, tzinfo=UTC),
                server_checksum_algorithm=None,
                server_checksum_value=None,
                checksum_type=None,
            ),
            body,
        )

    def list_keys(self, *, bucket: str, prefix: str) -> list[str]:
        assert bucket == "governed-data"
        return sorted(key for key in self.objects if key.startswith(prefix))

    def head(
        self, *, bucket: str, key: str, version_id: str | None
    ) -> S3ObjectIdentity:
        identity, _ = self.objects[key]
        assert bucket == identity.bucket
        if version_id is not None:
            assert version_id == identity.version_id
        return identity

    def read(
        self, *, bucket: str, key: str, version_id: str | None
    ) -> DownloadedS3Object:
        self.read_count += 1
        identity = self.head(bucket=bucket, key=key, version_id=version_id)
        return DownloadedS3Object(identity=identity, body=self.objects[key][1])


@pytest.fixture
def s3_api() -> tuple[TestClient, HostedRepository, AuthManager, InMemoryS3Reader]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    create_schema(engine)
    repository = HostedRepository(lambda: Session(engine))
    auth = AuthManager(repository)
    auth.bootstrap_admin(
        "admin@example.edu",
        "correct horse battery staple",
        display_name="Admin",
    )
    repository.create_user(
        email="ada@example.edu",
        display_name="Ada",
        password_hash=auth.create_password_hash("ada secure password"),
        role=Role.ANNOTATOR,
    )
    reader = InMemoryS3Reader()
    reader.seed(
        key="raw/session-001.txt",
        body=b"Alice spoke",
        version_id="raw-v1",
        etag='"raw-etag"',
    )
    reader.seed(
        key="reference/session-001.json",
        body=b"[]",
        version_id="reference-v1",
        etag='"reference-etag"',
    )
    app = create_hosted_app(
        repository=repository,
        auth=auth,
        cookie_secure=False,
        s3_reader=reader,
        s3_catalog_config=S3CatalogConfig(
            bucket="governed-data",
            raw_prefixes=("raw/",),
            reference_prefixes=("reference/",),
        ),
    )
    client = TestClient(app)
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
    return client, repository, auth, reader


def csrf_headers(client: TestClient) -> dict[str, str]:
    return {"X-CSRF-Token": client.cookies["annotation_csrf"]}


def manifest_payload() -> dict[str, object]:
    return {
        "name": "Governed batch",
        "documents": [
            {
                "external_id": "session-001",
                "filename": "session-001.txt",
                "label_set": ["NAME"],
                "raw_format": "canonical_text",
                "reference_format": "canonical_spans_json",
                "raw": {
                    "bucket": "governed-data",
                    "key": "raw/session-001.txt",
                    "version_id": "raw-v1",
                },
                "reference": {
                    "bucket": "governed-data",
                    "key": "reference/session-001.json",
                    "version_id": "reference-v1",
                },
            }
        ],
    }


def test_admin_can_plan_an_exact_s3_import_without_reading_bodies_or_writing_db(
    s3_api: tuple[TestClient, HostedRepository, AuthManager, InMemoryS3Reader],
) -> None:
    client, _, _, reader = s3_api
    before = client.get("/api/workspace").json()

    response = client.post(
        "/api/admin/s3-imports/plan",
        json={"manifest": manifest_payload()},
        headers=csrf_headers(client),
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["manifest_digest"] == (
        "8f366475e3a8ea4e7690f6e3dd859ee10c6192a97d833744fa7cc252bc080408"
    )
    assert payload["sources"] == [
        {
            "external_id": "session-001",
            "kind": "raw",
            "format": "canonical_text",
            "archive_member": None,
            "bucket": "governed-data",
            "key": "raw/session-001.txt",
            "version_id": "raw-v1",
            "content_length": 11,
            "etag": '"raw-etag"',
            "last_modified": "2026-08-28T00:00:00Z",
            "server_checksum_algorithm": None,
            "server_checksum_value": None,
            "checksum_type": None,
        },
        {
            "external_id": "session-001",
            "kind": "reference",
            "format": "canonical_spans_json",
            "archive_member": None,
            "bucket": "governed-data",
            "key": "reference/session-001.json",
            "version_id": "reference-v1",
            "content_length": 2,
            "etag": '"reference-etag"',
            "last_modified": "2026-08-28T00:00:00Z",
            "server_checksum_algorithm": None,
            "server_checksum_value": None,
            "checksum_type": None,
        },
    ]
    assert "Alice spoke" not in response.text
    assert reader.read_count == 0
    assert client.get("/api/workspace").json() == before


def test_admin_can_apply_the_exact_plan_idempotently(
    s3_api: tuple[TestClient, HostedRepository, AuthManager, InMemoryS3Reader],
) -> None:
    client, _, _, reader = s3_api
    manifest = manifest_payload()
    plan = client.post(
        "/api/admin/s3-imports/plan",
        json={"manifest": manifest},
        headers=csrf_headers(client),
    ).json()
    request = {
        "manifest": manifest,
        "expected_manifest_digest": plan["manifest_digest"],
        "expected_sources": plan["sources"],
        "mutation_id": "s3-import-001",
    }

    first = client.post(
        "/api/admin/s3-imports/apply",
        json=request,
        headers=csrf_headers(client),
    )

    assert first.status_code == 201
    assert first.json() == {
        "batch_id": first.json()["batch_id"],
        "imported_count": 1,
    }
    assert reader.read_count == 2
    workspace = client.get("/api/workspace").json()
    assert [item["external_id"] for item in workspace["sessions"]] == ["session-001"]

    reader.objects.clear()
    repeated = client.post(
        "/api/admin/s3-imports/apply",
        json=request,
        headers=csrf_headers(client),
    )

    assert repeated.status_code == 201
    assert repeated.json() == first.json()
    assert reader.read_count == 2


def test_admin_can_plan_and_apply_a_raw_only_session(
    s3_api: tuple[TestClient, HostedRepository, AuthManager, InMemoryS3Reader],
) -> None:
    client, _, _, reader = s3_api
    manifest = manifest_payload()
    manifest["documents"][0]["reference"] = None  # type: ignore[index]
    manifest["documents"][0]["reference_format"] = None  # type: ignore[index]

    plan_response = client.post(
        "/api/admin/s3-imports/plan",
        json={"manifest": manifest},
        headers=csrf_headers(client),
    )

    assert plan_response.status_code == 200
    plan = plan_response.json()
    assert [source["kind"] for source in plan["sources"]] == ["raw"]
    applied = client.post(
        "/api/admin/s3-imports/apply",
        json={
            "manifest": manifest,
            "expected_manifest_digest": plan["manifest_digest"],
            "expected_sources": plan["sources"],
            "mutation_id": "s3-import-raw-only",
        },
        headers=csrf_headers(client),
    )

    assert applied.status_code == 201
    assert applied.json()["imported_count"] == 1
    assert reader.read_count == 1
    document_id = client.get("/api/workspace").json()["sessions"][0]["id"]
    assert (
        client.get(f"/api/documents/{document_id}").json()["reference_annotations"]
        is None
    )


def test_s3_import_endpoints_require_configuration_and_admin_role(
    s3_api: tuple[TestClient, HostedRepository, AuthManager, InMemoryS3Reader],
) -> None:
    admin_client, repository, auth, _ = s3_api
    unconfigured = TestClient(
        create_hosted_app(
            repository=repository,
            auth=auth,
            cookie_secure=False,
        )
    )
    assert (
        unconfigured.post(
            "/api/auth/login",
            json={
                "email": "admin@example.edu",
                "password": "correct horse battery staple",
            },
        ).status_code
        == 200
    )
    unavailable = unconfigured.post(
        "/api/admin/s3-imports/plan",
        json={"manifest": manifest_payload()},
        headers=csrf_headers(unconfigured),
    )
    assert unavailable.status_code == 503
    assert unavailable.json() == {"detail": "governed S3 import is not configured"}

    annotator = TestClient(admin_client.app)
    assert (
        annotator.post(
            "/api/auth/login",
            json={"email": "ada@example.edu", "password": "ada secure password"},
        ).status_code
        == 200
    )
    forbidden = annotator.post(
        "/api/admin/s3-imports/plan",
        json={"manifest": manifest_payload()},
        headers=csrf_headers(annotator),
    )
    assert forbidden.status_code == 403
    assert forbidden.json() == {"detail": "admin role required"}


@pytest.mark.parametrize(
    ("field", "value", "detail"),
    [
        ("bucket", "other-bucket", "configured bucket"),
        ("key", "reference/session-001.txt", "raw prefix"),
    ],
)
def test_s3_plan_rejects_cross_bucket_and_cross_prefix_references(
    s3_api: tuple[TestClient, HostedRepository, AuthManager, InMemoryS3Reader],
    field: str,
    value: str,
    detail: str,
) -> None:
    client, _, _, _ = s3_api
    manifest = manifest_payload()
    manifest["documents"][0]["raw"][field] = value  # type: ignore[index]

    response = client.post(
        "/api/admin/s3-imports/plan",
        json={"manifest": manifest},
        headers=csrf_headers(client),
    )

    assert response.status_code == 422
    assert detail in response.json()["detail"]


def test_s3_apply_rejects_a_source_changed_after_plan_without_writing_db(
    s3_api: tuple[TestClient, HostedRepository, AuthManager, InMemoryS3Reader],
) -> None:
    client, _, _, reader = s3_api
    manifest = manifest_payload()
    plan = client.post(
        "/api/admin/s3-imports/plan",
        json={"manifest": manifest},
        headers=csrf_headers(client),
    ).json()
    reader.seed(
        key="raw/session-001.txt",
        body=b"Mallory spoke",
        version_id="raw-v1",
        etag='"changed-etag"',
    )

    response = client.post(
        "/api/admin/s3-imports/apply",
        json={
            "manifest": manifest,
            "expected_manifest_digest": plan["manifest_digest"],
            "expected_sources": plan["sources"],
            "mutation_id": "stale-plan",
        },
        headers=csrf_headers(client),
    )

    assert response.status_code == 409
    assert response.json() == {"detail": "S3 catalog changed after import planning"}
    assert reader.read_count == 0
    assert client.get("/api/workspace").json() == {"sessions": []}


def test_s3_plan_rejects_unsupported_raw_format_before_body_reads(
    s3_api: tuple[TestClient, HostedRepository, AuthManager, InMemoryS3Reader],
) -> None:
    client, _, _, reader = s3_api
    reader.seed(
        key="raw/batch.zip",
        body=b"not interpreted by the canonical text adapter",
        version_id="zip-v1",
        etag='"zip-etag"',
    )
    manifest = manifest_payload()
    manifest["documents"][0]["raw"] = {  # type: ignore[index]
        "bucket": "governed-data",
        "key": "raw/batch.zip",
        "version_id": "zip-v1",
    }

    response = client.post(
        "/api/admin/s3-imports/plan",
        json={"manifest": manifest},
        headers=csrf_headers(client),
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": (
            "canonical raw source must be a UTF-8 .txt object without an archive member"
        )
    }
    assert reader.read_count == 0


def test_saga_zip_and_cascade_reference_use_the_discovered_exact_schema(
    s3_api: tuple[TestClient, HostedRepository, AuthManager, InMemoryS3Reader],
) -> None:
    client, _, _, reader = s3_api
    transcript_json = json.dumps(
        {
            "sls_id": "session-zip",
            "segments": [
                {"speaker_type": "tutor", "text": " Alice\nspoke "},
                {"speaker_type": "student", "text": "Hello"},
            ],
            "metadata": {},
        }
    ).encode()
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("batch-01/session-zip/transcript.json", transcript_json)
        bundle.writestr(
            "batch-01/session-raw-only/transcript.json",
            json.dumps(
                {
                    "sls_id": "session-raw-only",
                    "segments": [
                        {"speaker_type": "student", "text": "No reference yet"}
                    ],
                }
            ).encode(),
        )
    raw_text = "Tutor: Alice spoke\nStudent: Hello"
    reference = json.dumps(
        {
            "id": "session-zip",
            "transcript": raw_text,
            "predicted_pii_occurrences": [{"start": 7, "end": 12, "pii_type": "NAME"}],
            "cascade_prediction_export": {"span_field": "predicted_pii_occurrences"},
        }
    ).encode()
    reader.seed(
        key="raw/batch-01.zip",
        body=archive.getvalue(),
        version_id="batch-v1",
        etag='"batch-etag"',
    )
    reader.seed(
        key="reference/session-zip.json",
        body=reference,
        version_id="reference-zip-v1",
        etag='"reference-zip-etag"',
    )
    manifest = {
        "name": "Saga batch",
        "documents": [
            {
                "external_id": "session-zip",
                "filename": "session-zip.txt",
                "label_set": ["NAME"],
                "raw_format": "saga_zip_transcript",
                "reference_format": "cascade_prediction_export",
                "raw": {
                    "bucket": "governed-data",
                    "key": "raw/batch-01.zip",
                    "version_id": "batch-v1",
                    "archive_member": "batch-01/session-zip/transcript.json",
                },
                "reference": {
                    "bucket": "governed-data",
                    "key": "reference/session-zip.json",
                    "version_id": "reference-zip-v1",
                },
            },
            {
                "external_id": "session-raw-only",
                "filename": "session-raw-only.txt",
                "label_set": ["NAME"],
                "raw_format": "saga_zip_transcript",
                "reference_format": None,
                "raw": {
                    "bucket": "governed-data",
                    "key": "raw/batch-01.zip",
                    "version_id": "batch-v1",
                    "archive_member": ("batch-01/session-raw-only/transcript.json"),
                },
                "reference": None,
            },
        ],
    }

    plan_response = client.post(
        "/api/admin/s3-imports/plan",
        json={"manifest": manifest},
        headers=csrf_headers(client),
    )

    assert plan_response.status_code == 200
    plan = plan_response.json()
    assert [source["archive_member"] for source in plan["sources"]] == [
        "batch-01/session-zip/transcript.json",
        None,
        "batch-01/session-raw-only/transcript.json",
    ]
    applied = client.post(
        "/api/admin/s3-imports/apply",
        json={
            "manifest": manifest,
            "expected_manifest_digest": plan["manifest_digest"],
            "expected_sources": plan["sources"],
            "mutation_id": "saga-zip-import",
        },
        headers=csrf_headers(client),
    )

    assert applied.status_code == 201
    assert applied.json()["imported_count"] == 2
    workspace = client.get("/api/workspace").json()["sessions"]
    document_id = next(
        item["id"] for item in workspace if item["external_id"] == "session-zip"
    )
    document = client.get(f"/api/documents/{document_id}").json()
    assert document["raw_text"] == raw_text
    assert document["reference_annotations"] == [
        {"start": 7, "end": 12, "label": "NAME", "text": "Alice"}
    ]


def test_saga_multimodel_transcript_normalizes_verified_group_roles(
    s3_api: tuple[TestClient, HostedRepository, AuthManager, InMemoryS3Reader],
) -> None:
    client, _, _, reader = s3_api
    reader.seed(
        key="raw/session01/transcript.json",
        body=json.dumps(
            [
                {"role": "Tutor 1", "content": " Welcome\nback "},
                {"role": "student 2", "content": "Thanks"},
            ]
        ).encode(),
        version_id="mm-v1",
        etag='"mm-etag"',
    )
    manifest = {
        "name": "Saga MultiModel",
        "documents": [
            {
                "external_id": "saga-mm-session01",
                "filename": "saga-mm-session01.txt",
                "label_set": ["NAME"],
                "raw_format": "saga_multimodel_transcript",
                "reference_format": None,
                "raw": {
                    "bucket": "governed-data",
                    "key": "raw/session01/transcript.json",
                    "version_id": "mm-v1",
                },
                "reference": None,
            }
        ],
    }
    plan = client.post(
        "/api/admin/s3-imports/plan",
        json={"manifest": manifest},
        headers=csrf_headers(client),
    ).json()

    applied = client.post(
        "/api/admin/s3-imports/apply",
        json={
            "manifest": manifest,
            "expected_manifest_digest": plan["manifest_digest"],
            "expected_sources": plan["sources"],
            "mutation_id": "saga-mm-import",
        },
        headers=csrf_headers(client),
    )

    assert applied.status_code == 201
    document_id = client.get("/api/workspace").json()["sessions"][0]["id"]
    assert client.get(f"/api/documents/{document_id}").json()["raw_text"] == (
        "Tutor: Welcome back\nStudent: Thanks"
    )


def test_cascade_reference_with_mismatched_transcript_is_rejected_atomically(
    s3_api: tuple[TestClient, HostedRepository, AuthManager, InMemoryS3Reader],
) -> None:
    client, _, _, reader = s3_api
    reader.seed(
        key="reference/mismatched.json",
        body=json.dumps(
            {
                "transcript": "different text",
                "predicted_pii_occurrences": [],
                "cascade_prediction_export": {
                    "span_field": "predicted_pii_occurrences"
                },
            }
        ).encode(),
        version_id="mismatch-v1",
        etag='"mismatch-etag"',
    )
    manifest = manifest_payload()
    document = manifest["documents"][0]  # type: ignore[index]
    document["reference_format"] = "cascade_prediction_export"
    document["reference"] = {
        "bucket": "governed-data",
        "key": "reference/mismatched.json",
        "version_id": "mismatch-v1",
    }
    plan = client.post(
        "/api/admin/s3-imports/plan",
        json={"manifest": manifest},
        headers=csrf_headers(client),
    ).json()

    response = client.post(
        "/api/admin/s3-imports/apply",
        json={
            "manifest": manifest,
            "expected_manifest_digest": plan["manifest_digest"],
            "expected_sources": plan["sources"],
            "mutation_id": "mismatched-reference",
        },
        headers=csrf_headers(client),
    )

    assert response.status_code == 422
    assert response.json() == {
        "detail": "cascade reference transcript must exactly match decoded raw text"
    }
    assert client.get("/api/workspace").json() == {"sessions": []}
