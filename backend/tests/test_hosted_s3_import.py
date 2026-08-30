from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from io import BytesIO
from typing import Any

import pytest
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, create_engine

from hosted_app.database import create_schema
from hosted_app.domain import ImportMutationConflict, Role
from hosted_app.repository import HostedRepository
from hosted_app.s3_import import (
    AmbiguousSourcePair,
    Boto3S3ReadAdapter,
    CanonicalReferenceJsonDecoder,
    DisallowedSource,
    GovernedS3Catalog,
    GovernedS3Importer,
    InvalidSourceFormat,
    ManifestDigestMismatch,
    MissingSourcePair,
    S3CatalogConfig,
    S3CatalogError,
    S3ImportManifest,
    S3ManifestDocument,
    S3ObjectRef,
    SourceIntegrityError,
    Utf8TranscriptDecoder,
    canonical_manifest_digest,
)


class InMemoryS3Client:
    """Small read-only Boto3 S3 client fake used by governed import tests."""

    def __init__(self, *, page_size: int = 1) -> None:
        self.page_size = page_size
        self.objects: dict[tuple[str, str], dict[str, Any]] = {}
        self.list_calls: list[dict[str, Any]] = []
        self.head_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []
        self.get_body_overrides: dict[tuple[str, str], bytes] = {}

    def seed(
        self,
        *,
        bucket: str,
        key: str,
        body: bytes,
        version_id: str | None = None,
        etag: str | None = None,
        checksum_sha256: bool = True,
    ) -> None:
        self.objects[(bucket, key)] = {
            "body": body,
            "VersionId": version_id,
            "ETag": etag or f'"opaque-{len(body)}"',
            "LastModified": datetime(2026, 8, 28, tzinfo=UTC),
            "ChecksumSHA256": (
                base64.b64encode(hashlib.sha256(body).digest()).decode()
                if checksum_sha256
                else None
            ),
            "ChecksumType": "FULL_OBJECT" if checksum_sha256 else None,
        }

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        self.list_calls.append(kwargs)
        bucket = kwargs["Bucket"]
        prefix = kwargs["Prefix"]
        keys = sorted(
            key
            for object_bucket, key in self.objects
            if object_bucket == bucket and key.startswith(prefix)
        )
        offset = int(kwargs.get("ContinuationToken", "0"))
        page = keys[offset : offset + self.page_size]
        next_offset = offset + len(page)
        return {
            "Contents": [{"Key": key} for key in page],
            "IsTruncated": next_offset < len(keys),
            "NextContinuationToken": str(next_offset),
        }

    def head_object(self, **kwargs: Any) -> dict[str, Any]:
        self.head_calls.append(kwargs)
        return self._response(kwargs, include_body=False)

    def get_object(self, **kwargs: Any) -> dict[str, Any]:
        self.get_calls.append(kwargs)
        return self._response(kwargs, include_body=True)

    def _response(
        self, request: dict[str, Any], *, include_body: bool
    ) -> dict[str, Any]:
        stored = self.objects[(request["Bucket"], request["Key"])]
        response = {
            "ContentLength": len(stored["body"]),
            "ETag": stored["ETag"],
            "LastModified": stored["LastModified"],
        }
        for field in ("VersionId", "ChecksumSHA256", "ChecksumType"):
            if stored[field] is not None:
                response[field] = stored[field]
        if include_body:
            body = self.get_body_overrides.get(
                (request["Bucket"], request["Key"]), stored["body"]
            )
            response["Body"] = BytesIO(body)
        return response


@pytest.fixture
def repository() -> HostedRepository:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    create_schema(engine)
    return HostedRepository(lambda: Session(engine))


def test_catalog_lists_only_allowlisted_source_prefixes_and_keeps_versions() -> None:
    client = InMemoryS3Client(page_size=1)
    client.seed(
        bucket="governed-data",
        key="raw/session-001.txt",
        body=b"Alice spoke",
        version_id="raw-v1",
    )
    client.seed(
        bucket="governed-data",
        key="reference/session-001.json",
        body=b"[]",
        version_id="reference-v1",
    )
    client.seed(
        bucket="governed-data",
        key="raw/annotation-output/session-001.json",
        body=b"must not be cataloged",
    )
    client.seed(
        bucket="governed-data",
        key="ground-truth/session-001.json",
        body=b"must not be listed",
    )
    catalog = GovernedS3Catalog(
        Boto3S3ReadAdapter(client),
        S3CatalogConfig(
            bucket="governed-data",
            raw_prefixes=("raw/",),
            reference_prefixes=("reference/",),
        ),
    )

    snapshot = catalog.catalog()

    assert [(item.kind, item.key, item.version_id) for item in snapshot.objects] == [
        ("raw", "raw/session-001.txt", "raw-v1"),
        ("reference", "reference/session-001.json", "reference-v1"),
    ]
    assert {call["Prefix"] for call in client.list_calls} == {"raw/", "reference/"}
    assert all(call["Prefix"] != "ground-truth/" for call in client.list_calls)
    assert all(
        "annotation-output" not in call["Key"] and "ground-truth" not in call["Key"]
        for call in client.head_calls
    )
    assert client.get_calls == []


@pytest.mark.parametrize(
    "excluded_prefix",
    ["annotation-output/", "project/ground-truth/"],
)
def test_catalog_configuration_rejects_output_prefixes(
    excluded_prefix: str,
) -> None:
    with pytest.raises(S3CatalogError, match="must not include"):
        S3CatalogConfig(
            bucket="governed-data",
            raw_prefixes=(excluded_prefix,),
            reference_prefixes=("reference/",),
        )


def test_manifest_import_downloads_exact_versions_and_records_sha256_provenance(
    repository: HostedRepository,
) -> None:
    raw_body = b"Alice spoke"
    reference_spans = [{"start": 0, "end": 5, "label": "NAME", "text": "Alice"}]
    reference_body = json.dumps(reference_spans).encode()
    client = InMemoryS3Client()
    client.seed(
        bucket="governed-data",
        key="raw/session-001.txt",
        body=raw_body,
        version_id="raw-v1",
        etag='"multipart-etag-2"',
    )
    client.seed(
        bucket="governed-data",
        key="reference/session-001.json",
        body=reference_body,
        version_id="reference-v1",
    )
    reader = Boto3S3ReadAdapter(client)
    config = S3CatalogConfig(
        bucket="governed-data",
        raw_prefixes=("raw/",),
        reference_prefixes=("reference/",),
    )
    snapshot = GovernedS3Catalog(reader, config).catalog()
    admin = repository.create_user(
        email="admin@example.edu", password_hash="hash", role=Role.ADMIN
    )
    importer = GovernedS3Importer(
        reader=reader,
        config=config,
        repository=repository,
        raw_decoder=Utf8TranscriptDecoder(),
        reference_decoder=CanonicalReferenceJsonDecoder(),
    )
    manifest = S3ImportManifest(
        name="Governed batch",
        documents=(
            S3ManifestDocument(
                external_id="session-001",
                filename="session-001.txt",
                label_set=("NAME",),
                raw=S3ObjectRef(
                    bucket="governed-data",
                    key="raw/session-001.txt",
                    version_id="raw-v1",
                ),
                reference=S3ObjectRef(
                    bucket="governed-data",
                    key="reference/session-001.json",
                    version_id="reference-v1",
                ),
            ),
        ),
    )

    imported = importer.import_manifest(
        manifest=manifest,
        catalog=snapshot,
        created_by=admin.id,
        expected_manifest_digest=canonical_manifest_digest(manifest),
        mutation_id="import-mutation-success",
    )

    assert imported.imported_count == 1
    document = repository.list_visible_documents(admin.id)[0]
    detail = repository.get_document(document.id, user_id=admin.id)
    assert detail.raw_text == "Alice spoke"
    assert detail.reference_spans == reference_spans
    provenance = repository.get_document_provenance(document.id, admin_id=admin.id)
    assert provenance.raw_source["sha256"] == hashlib.sha256(raw_body).hexdigest()
    assert provenance.raw_source["etag"] == '"multipart-etag-2"'
    assert provenance.raw_source["sha256"] != "multipart-etag-2"
    assert provenance.raw_source["version_id"] == "raw-v1"
    assert (
        provenance.reference_source["sha256"]
        == hashlib.sha256(reference_body).hexdigest()
    )
    assert all(call["ChecksumMode"] == "ENABLED" for call in client.get_calls)
    assert all("VersionId" in call for call in client.get_calls)
    assert not hasattr(provenance, "raw_text")


def test_manifest_rejects_disallowed_duplicate_and_ambiguous_sources_before_reads(
    repository: HostedRepository,
) -> None:
    client = InMemoryS3Client()
    client.seed(
        bucket="governed-data",
        key="raw/session-001.txt",
        body=b"Alice spoke",
        version_id="raw-v1",
    )
    client.seed(
        bucket="governed-data",
        key="reference/session-001.json",
        body=b"[]",
        version_id="reference-v1",
    )
    reader = Boto3S3ReadAdapter(client)
    config = S3CatalogConfig(
        bucket="governed-data",
        raw_prefixes=("raw/",),
        reference_prefixes=("reference/",),
    )
    snapshot = GovernedS3Catalog(reader, config).catalog()
    admin = repository.create_user(
        email="admin@example.edu", password_hash="hash", role=Role.ADMIN
    )
    importer = GovernedS3Importer(
        reader=reader,
        config=config,
        repository=repository,
        raw_decoder=Utf8TranscriptDecoder(),
        reference_decoder=CanonicalReferenceJsonDecoder(),
    )
    valid = S3ManifestDocument(
        external_id="session-001",
        filename="session-001.txt",
        label_set=("NAME",),
        raw=S3ObjectRef("governed-data", "raw/session-001.txt", "raw-v1"),
        reference=S3ObjectRef(
            "governed-data", "reference/session-001.json", "reference-v1"
        ),
    )

    cases = [
        (
            (replace(valid, raw=replace(valid.raw, bucket="other-bucket")),),
            DisallowedSource,
            "configured bucket",
        ),
        (
            (replace(valid, raw=replace(valid.raw, key="reference/session-001.txt")),),
            DisallowedSource,
            "raw prefix",
        ),
        (
            (valid, valid),
            AmbiguousSourcePair,
            "duplicate document id",
        ),
        (
            (valid, replace(valid, external_id="session-002")),
            AmbiguousSourcePair,
            "reuses an S3 source",
        ),
        (
            (
                replace(
                    valid,
                    reference=replace(valid.reference, version_id="missing-version"),
                ),
            ),
            MissingSourcePair,
            "absent from catalog",
        ),
    ]
    for documents, error_type, message in cases:
        client.get_calls.clear()
        rejected_manifest = S3ImportManifest(name="Rejected", documents=documents)
        with pytest.raises(error_type, match=message):
            importer.import_manifest(
                manifest=rejected_manifest,
                catalog=snapshot,
                created_by=admin.id,
                expected_manifest_digest=canonical_manifest_digest(rejected_manifest),
                mutation_id=f"rejected-{message}",
            )
        assert client.get_calls == []

    assert repository.export_manual_annotations(admin_id=admin.id) == []


def test_manifest_digest_and_mutation_id_make_apply_idempotent(
    repository: HostedRepository,
) -> None:
    client = InMemoryS3Client()
    client.seed(
        bucket="governed-data",
        key="raw/session-001.txt",
        body=b"Alice spoke",
        version_id="raw-v1",
    )
    client.seed(
        bucket="governed-data",
        key="reference/session-001.json",
        body=b"[]",
        version_id="reference-v1",
    )
    reader = Boto3S3ReadAdapter(client)
    config = S3CatalogConfig(
        bucket="governed-data",
        raw_prefixes=("raw/",),
        reference_prefixes=("reference/",),
    )
    snapshot = GovernedS3Catalog(reader, config).catalog()
    admin = repository.create_user(
        email="admin@example.edu", password_hash="hash", role=Role.ADMIN
    )
    importer = GovernedS3Importer(
        reader=reader,
        config=config,
        repository=repository,
        raw_decoder=Utf8TranscriptDecoder(),
        reference_decoder=CanonicalReferenceJsonDecoder(),
    )
    manifest = S3ImportManifest(
        name="Governed batch",
        documents=(
            S3ManifestDocument(
                external_id="session-001",
                filename="session.txt",
                label_set=("NAME",),
                raw=S3ObjectRef("governed-data", "raw/session-001.txt", "raw-v1"),
                reference=S3ObjectRef(
                    "governed-data",
                    "reference/session-001.json",
                    "reference-v1",
                ),
            ),
        ),
    )
    digest = canonical_manifest_digest(manifest)

    first = importer.import_manifest(
        manifest=manifest,
        catalog=snapshot,
        created_by=admin.id,
        expected_manifest_digest=digest,
        mutation_id="import-mutation-001",
    )
    client.get_calls.clear()
    client.objects.clear()
    repeated = importer.import_manifest(
        manifest=manifest,
        catalog=snapshot,
        created_by=admin.id,
        expected_manifest_digest=digest,
        mutation_id="import-mutation-001",
    )

    assert repeated == first
    assert client.get_calls == []
    assert len(repository.list_visible_documents(admin.id)) == 1

    changed_manifest = replace(manifest, name="Different content")
    changed_digest = canonical_manifest_digest(changed_manifest)
    with pytest.raises(ImportMutationConflict, match="mutation id"):
        importer.import_manifest(
            manifest=changed_manifest,
            catalog=snapshot,
            created_by=admin.id,
            expected_manifest_digest=changed_digest,
            mutation_id="import-mutation-001",
        )
    with pytest.raises(ManifestDigestMismatch, match="does not match"):
        importer.import_manifest(
            manifest=changed_manifest,
            catalog=snapshot,
            created_by=admin.id,
            expected_manifest_digest=digest,
            mutation_id="import-mutation-002",
        )


def test_import_rejects_source_mutation_after_catalog_without_database_writes(
    repository: HostedRepository,
) -> None:
    client, _, _, importer, snapshot, admin, manifest = _unversioned_import_setup(
        repository
    )
    client.seed(
        bucket="governed-data",
        key="raw/session-001.txt",
        body=b"Mallory said",
        etag='"changed-object"',
    )

    with pytest.raises(SourceIntegrityError, match="changed after catalog"):
        importer.import_manifest(
            manifest=manifest,
            catalog=snapshot,
            created_by=admin.id,
            expected_manifest_digest=canonical_manifest_digest(manifest),
            mutation_id="mutated-source",
        )

    assert repository.export_manual_annotations(admin_id=admin.id) == []


def test_import_verifies_server_sha256_and_rejects_reference_json_wrappers(
    repository: HostedRepository,
) -> None:
    client, reader, config, importer, snapshot, admin, manifest = (
        _unversioned_import_setup(repository)
    )
    client.get_body_overrides[("governed-data", "raw/session-001.txt")] = (
        b"Mallory said"
    )
    with pytest.raises(SourceIntegrityError, match="SHA256 checksum mismatch"):
        importer.import_manifest(
            manifest=manifest,
            catalog=snapshot,
            created_by=admin.id,
            expected_manifest_digest=canonical_manifest_digest(manifest),
            mutation_id="bad-checksum",
        )

    client.get_body_overrides.clear()
    client.seed(
        bucket="governed-data",
        key="reference/session-001.json",
        body=b'{"spans": []}',
    )
    snapshot = GovernedS3Catalog(reader, config).catalog()
    with pytest.raises(InvalidSourceFormat, match="JSON array"):
        importer.import_manifest(
            manifest=manifest,
            catalog=snapshot,
            created_by=admin.id,
            expected_manifest_digest=canonical_manifest_digest(manifest),
            mutation_id="unsupported-wrapper",
        )
    assert repository.export_manual_annotations(admin_id=admin.id) == []


def _unversioned_import_setup(repository: HostedRepository):
    client = InMemoryS3Client()
    client.seed(
        bucket="governed-data",
        key="raw/session-001.txt",
        body=b"Alice spoke",
        etag='"raw-original"',
    )
    client.seed(
        bucket="governed-data",
        key="reference/session-001.json",
        body=b"[]",
        etag='"reference-original"',
    )
    reader = Boto3S3ReadAdapter(client)
    config = S3CatalogConfig(
        bucket="governed-data",
        raw_prefixes=("raw/",),
        reference_prefixes=("reference/",),
    )
    snapshot = GovernedS3Catalog(reader, config).catalog()
    admin = repository.create_user(
        email="admin@example.edu", password_hash="hash", role=Role.ADMIN
    )
    importer = GovernedS3Importer(
        reader=reader,
        config=config,
        repository=repository,
        raw_decoder=Utf8TranscriptDecoder(),
        reference_decoder=CanonicalReferenceJsonDecoder(),
    )
    manifest = S3ImportManifest(
        name="Governed batch",
        documents=(
            S3ManifestDocument(
                external_id="session-001",
                filename="session.txt",
                label_set=("NAME",),
                raw=S3ObjectRef("governed-data", "raw/session-001.txt"),
                reference=S3ObjectRef("governed-data", "reference/session-001.json"),
            ),
        ),
    )
    return client, reader, config, importer, snapshot, admin, manifest
