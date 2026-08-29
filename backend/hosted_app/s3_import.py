from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePosixPath
from typing import Any, Literal, Protocol

from botocore.exceptions import BotoCoreError, ClientError

from .domain import DocumentImport, ImportedBatch

SourceKind = Literal["raw", "reference"]
RawSourceFormat = Literal[
    "canonical_text",
    "saga_zip_transcript",
    "saga_multimodel_transcript",
]
ReferenceSourceFormat = Literal[
    "canonical_spans_json",
    "cascade_prediction_export",
]
_EXCLUDED_PATH_SEGMENTS = frozenset({"annotation-output", "ground-truth", "gt"})
_LINE_BREAK_RE = re.compile(r"[\n\r\v\f\x1c\x1d\x1e\x85  ]+")
_MULTIMODEL_ROLE_RE = re.compile(
    r"^(tutor|student)(?:\s+[0-9]+)?$",
    re.IGNORECASE | re.ASCII,
)


class S3CatalogError(RuntimeError):
    pass


class S3ImportError(RuntimeError):
    pass


class SourceIntegrityError(S3ImportError):
    pass


class InvalidSourceFormat(S3ImportError):
    pass


class MissingSourcePair(S3ImportError):
    pass


class AmbiguousSourcePair(S3ImportError):
    pass


class DisallowedSource(S3ImportError):
    pass


class ManifestDigestMismatch(S3ImportError):
    pass


@dataclass(frozen=True)
class S3CatalogConfig:
    bucket: str
    raw_prefixes: tuple[str, ...]
    reference_prefixes: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.bucket.strip():
            raise S3CatalogError("S3 bucket is required")
        prefixes = self.raw_prefixes + self.reference_prefixes
        if not prefixes or any(not prefix for prefix in prefixes):
            raise S3CatalogError("explicit non-root source prefixes are required")
        if any(_is_excluded_key(prefix) for prefix in prefixes):
            raise S3CatalogError(
                "source prefixes must not include annotation output or ground truth"
            )
        if len(set(prefixes)) != len(prefixes):
            raise S3CatalogError("source prefixes must be unique")
        if any(
            left.startswith(right) or right.startswith(left)
            for index, left in enumerate(prefixes)
            for right in prefixes[index + 1 :]
        ):
            raise S3CatalogError("source prefixes must not overlap")


@dataclass(frozen=True)
class S3ObjectIdentity:
    bucket: str
    key: str
    version_id: str | None
    content_length: int
    etag: str | None
    last_modified: datetime | None
    server_checksum_algorithm: str | None
    server_checksum_value: str | None
    checksum_type: str | None


@dataclass(frozen=True)
class CatalogObject(S3ObjectIdentity):
    kind: SourceKind


@dataclass(frozen=True)
class S3CatalogSnapshot:
    bucket: str
    objects: tuple[CatalogObject, ...]


@dataclass(frozen=True)
class S3ObjectRef:
    bucket: str
    key: str
    version_id: str | None = None
    archive_member: str | None = None


@dataclass(frozen=True)
class S3ManifestDocument:
    external_id: str
    filename: str
    label_set: tuple[str, ...]
    raw: S3ObjectRef
    reference: S3ObjectRef | None
    raw_format: RawSourceFormat = "canonical_text"
    reference_format: ReferenceSourceFormat | None = "canonical_spans_json"


@dataclass(frozen=True)
class S3ImportManifest:
    name: str
    documents: tuple[S3ManifestDocument, ...]


@dataclass(frozen=True)
class DownloadedS3Object:
    identity: S3ObjectIdentity
    body: bytes


class S3ReadAdapter(Protocol):
    def list_keys(self, *, bucket: str, prefix: str) -> list[str]: ...

    def head(
        self, *, bucket: str, key: str, version_id: str | None
    ) -> S3ObjectIdentity: ...

    def read(
        self, *, bucket: str, key: str, version_id: str | None
    ) -> DownloadedS3Object: ...


class RawTranscriptDecoder(Protocol):
    def decode(self, body: bytes) -> str: ...


class ReferenceDecoder(Protocol):
    def decode(self, body: bytes) -> list[dict[str, Any]]: ...


class ImportRepository(Protocol):
    def import_batch(
        self,
        *,
        name: str,
        created_by: str,
        documents: list[DocumentImport],
        manifest_digest: str,
        mutation_id: str,
    ) -> ImportedBatch: ...

    def resolve_import_retry(
        self,
        *,
        admin_id: str,
        mutation_id: str,
        manifest_digest: str,
    ) -> ImportedBatch | None: ...


class Utf8TranscriptDecoder:
    """Contract: an S3 raw object is exactly UTF-8 transcript text."""

    def decode(self, body: bytes) -> str:
        try:
            return body.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise InvalidSourceFormat("raw transcript must be UTF-8 text") from exc


class CanonicalReferenceJsonDecoder:
    """Contract: a reference object is a JSON array of canonical span objects."""

    def decode(self, body: bytes) -> list[dict[str, Any]]:
        try:
            decoded = json.loads(body)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InvalidSourceFormat("reference must be valid UTF-8 JSON") from exc
        if not isinstance(decoded, list) or not all(
            isinstance(span, dict) for span in decoded
        ):
            raise InvalidSourceFormat(
                "reference schema must be a JSON array of span objects"
            )
        return decoded


def _decode_raw_source(
    *,
    entry: S3ManifestDocument,
    body: bytes,
    canonical_decoder: RawTranscriptDecoder,
) -> str:
    if entry.raw_format == "canonical_text":
        return canonical_decoder.decode(body)
    if entry.raw_format == "saga_zip_transcript":
        return _decode_saga_zip_transcript(
            body=body,
            external_id=entry.external_id,
            archive_member=entry.raw.archive_member or "",
        )
    if entry.raw_format == "saga_multimodel_transcript":
        return _decode_saga_multimodel_transcript(body)
    raise InvalidSourceFormat("unsupported raw source format")


def _decode_saga_zip_transcript(
    *, body: bytes, external_id: str, archive_member: str
) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(body)) as archive:
            info = archive.getinfo(archive_member)
            if info.is_dir() or info.flag_bits & 0x1:
                raise InvalidSourceFormat(
                    "Saga transcript archive member must be a readable file"
                )
            transcript_body = archive.read(info)
    except (KeyError, zipfile.BadZipFile, RuntimeError) as exc:
        raise InvalidSourceFormat(
            "Saga raw source must contain the declared transcript member"
        ) from exc
    payload = _decode_json(transcript_body, description="Saga transcript")
    if not isinstance(payload, dict):
        raise InvalidSourceFormat("Saga transcript root must be an object")
    if str(payload.get("sls_id") or "") != external_id:
        raise InvalidSourceFormat("Saga sls_id must match manifest external_id")
    segments = payload.get("segments")
    if not isinstance(segments, list) or not segments:
        raise InvalidSourceFormat("Saga transcript must contain non-empty segments")
    lines: list[str] = []
    role_by_speaker = {"tutor": "Tutor", "student": "Student"}
    for segment in segments:
        if not isinstance(segment, dict):
            raise InvalidSourceFormat("Saga transcript segment must be an object")
        speaker_type = str(segment.get("speaker_type") or "")
        role = role_by_speaker.get(speaker_type)
        if role is None:
            raise InvalidSourceFormat(f"unknown Saga speaker_type {speaker_type!r}")
        text = str(segment.get("text") or "")
        flattened = _LINE_BREAK_RE.sub(" ", text).strip()
        if flattened:
            lines.append(f"{role}: {flattened}")
    if not lines:
        raise InvalidSourceFormat("Saga transcript has no non-empty segments")
    return "\n".join(lines)


def _decode_saga_multimodel_transcript(body: bytes) -> str:
    turns = _decode_json(body, description="Saga MultiModel transcript")
    if not isinstance(turns, list) or not turns:
        raise InvalidSourceFormat(
            "Saga MultiModel transcript root must be a non-empty array"
        )
    lines: list[str] = []
    for turn in turns:
        if not isinstance(turn, dict):
            raise InvalidSourceFormat("Saga MultiModel turn must be an object")
        original_role = str(turn.get("role") or "").strip()
        role_match = _MULTIMODEL_ROLE_RE.fullmatch(original_role)
        if role_match is None:
            raise InvalidSourceFormat(
                f"unrecognized Saga MultiModel speaker role {original_role!r}"
            )
        content = str(turn.get("content") or "")
        flattened = _LINE_BREAK_RE.sub(" ", content).strip()
        lines.append(f"{role_match.group(1).capitalize()}: {flattened}")
    return "\n".join(lines)


def _decode_reference_source(
    *,
    format: ReferenceSourceFormat | None,
    body: bytes,
    raw_text: str,
    canonical_decoder: ReferenceDecoder,
) -> list[dict[str, Any]]:
    if format == "canonical_spans_json":
        return canonical_decoder.decode(body)
    if format != "cascade_prediction_export":
        raise InvalidSourceFormat("unsupported reference source format")
    payload = _decode_json(body, description="cascade reference export")
    if not isinstance(payload, dict):
        raise InvalidSourceFormat("cascade reference export root must be an object")
    if payload.get("transcript") != raw_text:
        raise InvalidSourceFormat(
            "cascade reference transcript must exactly match decoded raw text"
        )
    metadata = payload.get("cascade_prediction_export")
    if not isinstance(metadata, dict) or metadata.get("span_field") != (
        "predicted_pii_occurrences"
    ):
        raise InvalidSourceFormat(
            "cascade reference span_field must be predicted_pii_occurrences"
        )
    predictions = payload.get("predicted_pii_occurrences")
    if not isinstance(predictions, list):
        raise InvalidSourceFormat(
            "cascade reference predicted_pii_occurrences must be an array"
        )
    spans: list[dict[str, Any]] = []
    for prediction in predictions:
        if not isinstance(prediction, dict):
            raise InvalidSourceFormat("cascade reference span must be an object")
        start = prediction.get("start")
        end = prediction.get("end")
        label = prediction.get("pii_type")
        if (
            not isinstance(start, int)
            or isinstance(start, bool)
            or not isinstance(end, int)
            or isinstance(end, bool)
            or start < 0
            or end <= start
            or end > len(raw_text)
            or not isinstance(label, str)
            or not label
        ):
            raise InvalidSourceFormat("cascade reference span is invalid")
        spans.append(
            {
                "start": start,
                "end": end,
                "label": label,
                "text": raw_text[start:end],
            }
        )
    return spans


def _decode_json(body: bytes, *, description: str) -> Any:
    try:
        return json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidSourceFormat(f"{description} must be valid UTF-8 JSON") from exc


class Boto3S3ReadAdapter:
    """Read-only boundary around the small Boto3 S3 client surface we use."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def list_keys(self, *, bucket: str, prefix: str) -> list[str]:
        keys: list[str] = []
        continuation_token: str | None = None
        while True:
            request: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
            if continuation_token is not None:
                request["ContinuationToken"] = continuation_token
            try:
                response = self._client.list_objects_v2(**request)
            except (BotoCoreError, ClientError) as exc:
                raise S3CatalogError("governed S3 listing failed") from exc
            keys.extend(item["Key"] for item in response.get("Contents", []))
            if not response.get("IsTruncated"):
                return keys
            continuation_token = response.get("NextContinuationToken")
            if continuation_token is None:
                raise S3CatalogError("truncated S3 listing omitted continuation token")

    def head(
        self, *, bucket: str, key: str, version_id: str | None
    ) -> S3ObjectIdentity:
        request: dict[str, Any] = {
            "Bucket": bucket,
            "Key": key,
            "ChecksumMode": "ENABLED",
        }
        if version_id is not None:
            request["VersionId"] = version_id
        try:
            response = self._client.head_object(**request)
        except (BotoCoreError, ClientError) as exc:
            raise S3CatalogError("governed S3 metadata read failed") from exc
        return _identity_from_response(bucket=bucket, key=key, response=response)

    def read(
        self, *, bucket: str, key: str, version_id: str | None
    ) -> DownloadedS3Object:
        request: dict[str, Any] = {
            "Bucket": bucket,
            "Key": key,
            "ChecksumMode": "ENABLED",
        }
        if version_id is not None:
            request["VersionId"] = version_id
        try:
            response = self._client.get_object(**request)
        except (BotoCoreError, ClientError) as exc:
            raise S3ImportError("governed S3 object read failed") from exc
        stream = response["Body"]
        try:
            try:
                body = stream.read()
            except (BotoCoreError, ClientError) as exc:
                raise S3ImportError("governed S3 object read failed") from exc
        finally:
            stream.close()
        return DownloadedS3Object(
            identity=_identity_from_response(
                bucket=bucket,
                key=key,
                response=response,
            ),
            body=body,
        )


class GovernedS3Catalog:
    def __init__(self, reader: S3ReadAdapter, config: S3CatalogConfig) -> None:
        self._reader = reader
        self._config = config

    def catalog(self) -> S3CatalogSnapshot:
        objects: list[CatalogObject] = []
        for kind, prefixes in (
            ("raw", self._config.raw_prefixes),
            ("reference", self._config.reference_prefixes),
        ):
            for prefix in prefixes:
                for key in self._reader.list_keys(
                    bucket=self._config.bucket, prefix=prefix
                ):
                    if _is_excluded_key(key):
                        continue
                    identity = self._reader.head(
                        bucket=self._config.bucket,
                        key=key,
                        version_id=None,
                    )
                    objects.append(CatalogObject(kind=kind, **identity.__dict__))
        return S3CatalogSnapshot(
            bucket=self._config.bucket,
            objects=tuple(sorted(objects, key=lambda item: (item.kind, item.key))),
        )


class GovernedS3Importer:
    def __init__(
        self,
        *,
        reader: S3ReadAdapter,
        config: S3CatalogConfig,
        repository: ImportRepository,
        raw_decoder: RawTranscriptDecoder,
        reference_decoder: ReferenceDecoder,
    ) -> None:
        self._reader = reader
        self._config = config
        self._repository = repository
        self._raw_decoder = raw_decoder
        self._reference_decoder = reference_decoder

    def import_manifest(
        self,
        *,
        manifest: S3ImportManifest,
        catalog: S3CatalogSnapshot,
        created_by: str,
        expected_manifest_digest: str,
        mutation_id: str,
    ) -> ImportedBatch:
        actual_manifest_digest = canonical_manifest_digest(manifest)
        if actual_manifest_digest != expected_manifest_digest:
            raise ManifestDigestMismatch(
                "expected manifest digest does not match canonical manifest"
            )
        retry = self._repository.resolve_import_retry(
            admin_id=created_by,
            mutation_id=mutation_id,
            manifest_digest=actual_manifest_digest,
        )
        if retry is not None:
            return retry
        if catalog.bucket != self._config.bucket:
            raise DisallowedSource("catalog bucket differs from configured bucket")
        resolved = resolve_manifest_sources(
            manifest=manifest,
            catalog=catalog,
            config=self._config,
        )

        documents: list[DocumentImport] = []
        for entry, raw_catalog, reference_catalog in resolved:
            raw_download, raw_provenance = self._download(raw_catalog)
            raw_text = _decode_raw_source(
                entry=entry,
                body=raw_download,
                canonical_decoder=self._raw_decoder,
            )
            raw_provenance["format"] = entry.raw_format
            raw_provenance["archive_member"] = entry.raw.archive_member
            reference_download = None
            reference_provenance = None
            if reference_catalog is not None:
                reference_download, reference_provenance = self._download(
                    reference_catalog
                )
                reference_provenance["format"] = entry.reference_format
            documents.append(
                DocumentImport(
                    external_id=entry.external_id,
                    filename=entry.filename,
                    raw_text=raw_text,
                    label_set=list(entry.label_set),
                    reference_spans=(
                        _decode_reference_source(
                            format=entry.reference_format,
                            body=reference_download,
                            raw_text=raw_text,
                            canonical_decoder=self._reference_decoder,
                        )
                        if reference_download is not None
                        else None
                    ),
                    raw_source=raw_provenance,
                    reference_source=reference_provenance,
                )
            )
        return self._repository.import_batch(
            name=manifest.name,
            created_by=created_by,
            documents=documents,
            manifest_digest=actual_manifest_digest,
            mutation_id=mutation_id,
        )

    def _download(self, expected: CatalogObject) -> tuple[bytes, dict[str, Any]]:
        before = self._reader.head(
            bucket=expected.bucket,
            key=expected.key,
            version_id=expected.version_id,
        )
        _require_same_identity(expected, before)
        downloaded = self._reader.read(
            bucket=expected.bucket,
            key=expected.key,
            version_id=expected.version_id,
        )
        _require_same_identity(expected, downloaded.identity)
        after = self._reader.head(
            bucket=expected.bucket,
            key=expected.key,
            version_id=expected.version_id,
        )
        _require_same_identity(expected, after)
        sha256_hex = hashlib.sha256(downloaded.body).hexdigest()
        _verify_strong_checksum(downloaded.identity, downloaded.body)
        return downloaded.body, {
            "bucket": expected.bucket,
            "key": expected.key,
            "version_id": expected.version_id,
            "content_length": expected.content_length,
            "etag": expected.etag,
            "last_modified": (
                expected.last_modified.isoformat()
                if expected.last_modified is not None
                else None
            ),
            "server_checksum_algorithm": expected.server_checksum_algorithm,
            "server_checksum_value": expected.server_checksum_value,
            "checksum_type": expected.checksum_type,
            "sha256": sha256_hex,
        }


def resolve_manifest_sources(
    *,
    manifest: S3ImportManifest,
    catalog: S3CatalogSnapshot,
    config: S3CatalogConfig,
) -> tuple[tuple[S3ManifestDocument, CatalogObject, CatalogObject | None], ...]:
    if catalog.bucket != config.bucket:
        raise DisallowedSource("catalog bucket differs from configured bucket")
    catalog_index = {
        (item.kind, item.bucket, item.key, item.version_id): item
        for item in catalog.objects
    }
    resolved: list[tuple[S3ManifestDocument, CatalogObject, CatalogObject | None]] = []
    document_ids: set[str] = set()
    used_sources: set[tuple[str, str, str | None, str | None]] = set()
    for entry in manifest.documents:
        if entry.external_id in document_ids:
            raise AmbiguousSourcePair(
                f"duplicate document id {entry.external_id!r} in manifest"
            )
        document_ids.add(entry.external_id)
        _require_adapter_contract(entry)
        _require_allowed_ref(entry.raw, kind="raw", config=config)
        if entry.reference is not None:
            _require_allowed_ref(entry.reference, kind="reference", config=config)
        for source in filter(None, (entry.raw, entry.reference)):
            identity = (
                source.bucket,
                source.key,
                source.version_id,
                source.archive_member,
            )
            if identity in used_sources:
                raise AmbiguousSourcePair(
                    f"document {entry.external_id!r} reuses an S3 source"
                )
            used_sources.add(identity)
        raw_catalog = catalog_index.get(
            ("raw", entry.raw.bucket, entry.raw.key, entry.raw.version_id)
        )
        reference_catalog = None
        if entry.reference is not None:
            reference_catalog = catalog_index.get(
                (
                    "reference",
                    entry.reference.bucket,
                    entry.reference.key,
                    entry.reference.version_id,
                )
            )
        if raw_catalog is None or (
            entry.reference is not None and reference_catalog is None
        ):
            raise MissingSourcePair(
                f"document {entry.external_id!r} source pair is absent from catalog"
            )
        resolved.append((entry, raw_catalog, reference_catalog))
    return tuple(resolved)


def _require_adapter_contract(entry: S3ManifestDocument) -> None:
    raw = entry.raw
    if entry.raw_format == "canonical_text":
        if not raw.key.endswith(".txt") or raw.archive_member is not None:
            raise InvalidSourceFormat(
                "canonical raw source must be a UTF-8 .txt object without an archive member"
            )
    elif entry.raw_format == "saga_zip_transcript":
        if not raw.key.endswith(".zip") or raw.archive_member is None:
            raise InvalidSourceFormat(
                "Saga raw source must be a .zip object with an explicit transcript member"
            )
        member = PurePosixPath(raw.archive_member)
        if (
            member.is_absolute()
            or ".." in member.parts
            or member.name != "transcript.json"
            or member.parent.name != entry.external_id
        ):
            raise InvalidSourceFormat(
                "Saga archive member must end with <external_id>/transcript.json"
            )
    elif entry.raw_format == "saga_multimodel_transcript":
        source_session = PurePosixPath(raw.key).parent.name
        if (
            PurePosixPath(raw.key).name != "transcript.json"
            or raw.archive_member is not None
            or entry.external_id != f"saga-mm-{source_session}"
        ):
            raise InvalidSourceFormat(
                "Saga MultiModel source must be <session>/transcript.json with "
                "external_id saga-mm-<session>"
            )
    else:
        raise InvalidSourceFormat("unsupported raw source format")

    if entry.reference is None:
        if entry.reference_format is not None:
            raise InvalidSourceFormat(
                "reference format must be null when reference source is null"
            )
        return
    if entry.reference.archive_member is not None:
        raise InvalidSourceFormat("reference source must not use an archive member")
    if entry.reference_format is None:
        raise InvalidSourceFormat(
            "reference format is required when a reference source is present"
        )
    if not entry.reference.key.endswith(".json"):
        raise InvalidSourceFormat("reference source must be a .json object")


def _require_allowed_ref(
    source: S3ObjectRef,
    *,
    kind: SourceKind,
    config: S3CatalogConfig,
) -> None:
    if source.bucket != config.bucket:
        raise DisallowedSource(f"source must use configured bucket {config.bucket!r}")
    prefixes = config.raw_prefixes if kind == "raw" else config.reference_prefixes
    if not any(source.key.startswith(prefix) for prefix in prefixes):
        raise DisallowedSource(f"source key is outside the {kind} prefix allowlist")
    if _is_excluded_key(source.key):
        raise DisallowedSource(
            "annotation output and ground truth sources are excluded"
        )


def _is_excluded_key(key: str) -> bool:
    return bool(
        _EXCLUDED_PATH_SEGMENTS.intersection(
            segment.casefold() for segment in key.split("/")
        )
    )


def canonical_manifest_digest(manifest: S3ImportManifest) -> str:
    def source_payload(source: S3ObjectRef | None) -> dict[str, Any] | None:
        if source is None:
            return None
        return {
            "bucket": source.bucket,
            "key": source.key,
            "version_id": source.version_id,
            "archive_member": source.archive_member,
        }

    payload = {
        "name": manifest.name,
        "documents": [
            {
                "external_id": document.external_id,
                "filename": document.filename,
                "label_set": sorted(document.label_set),
                "raw_format": document.raw_format,
                "reference_format": document.reference_format,
                "raw": source_payload(document.raw),
                "reference": source_payload(document.reference),
            }
            for document in sorted(
                manifest.documents,
                key=lambda item: (
                    item.external_id,
                    item.filename,
                    item.raw.bucket,
                    item.raw.key,
                    item.raw.version_id or "",
                    item.raw.archive_member or "",
                ),
            )
        ],
    }
    canonical_json = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical_json).hexdigest()


def _require_same_identity(
    expected: S3ObjectIdentity, actual: S3ObjectIdentity
) -> None:
    expected_identity = (
        expected.bucket,
        expected.key,
        expected.version_id,
        expected.content_length,
        expected.etag,
        expected.last_modified,
        expected.server_checksum_algorithm,
        expected.server_checksum_value,
        expected.checksum_type,
    )
    actual_identity = (
        actual.bucket,
        actual.key,
        actual.version_id,
        actual.content_length,
        actual.etag,
        actual.last_modified,
        actual.server_checksum_algorithm,
        actual.server_checksum_value,
        actual.checksum_type,
    )
    if actual_identity != expected_identity:
        raise SourceIntegrityError(
            f"S3 source changed after catalog: s3://{expected.bucket}/{expected.key}"
        )


def _verify_strong_checksum(identity: S3ObjectIdentity, body: bytes) -> None:
    algorithm = identity.server_checksum_algorithm
    value = identity.server_checksum_value
    if algorithm is None or value is None or identity.checksum_type == "COMPOSITE":
        return
    if algorithm == "SHA256":
        digest = hashlib.sha256(body).digest()
    elif algorithm == "SHA1":
        digest = hashlib.sha1(body).digest()
    else:
        return
    if base64.b64encode(digest).decode() != value:
        raise SourceIntegrityError(
            f"S3 {algorithm} checksum mismatch for s3://{identity.bucket}/{identity.key}"
        )


def _identity_from_response(
    *, bucket: str, key: str, response: dict[str, Any]
) -> S3ObjectIdentity:
    checksum_algorithm = None
    checksum_value = None
    for algorithm in ("SHA256", "SHA1"):
        value = response.get(f"Checksum{algorithm}")
        if value is not None:
            checksum_algorithm = algorithm
            checksum_value = value
            break
    return S3ObjectIdentity(
        bucket=bucket,
        key=key,
        version_id=response.get("VersionId"),
        content_length=response["ContentLength"],
        etag=response.get("ETag"),
        last_modified=response.get("LastModified"),
        server_checksum_algorithm=checksum_algorithm,
        server_checksum_value=checksum_value,
        checksum_type=response.get("ChecksumType"),
    )
