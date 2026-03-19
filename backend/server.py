from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import random
import re
import shutil
import threading
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any, Callable, Literal, cast
from urllib.parse import urlparse

import httpx
from dotenv import dotenv_values
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from models import (
    AgentChunkDiagnostic,
    AgentOutputs,
    AgentRunMetrics,
    CanonicalDocument,
    CanonicalSpan,
    FolderRecord,
    LLMConfidenceMetric,
    ResolutionEvent,
    SavedRunMetadata,
)
from normalizer import parse_file, parse_jsonl_file
from agent import (
    METHOD_DEFINITION_BY_ID,
    _bundle_preserves_native_labels,
    MODEL_PRESETS,
    SYSTEM_PROMPT,
    _bundle_uses_detected_value_post_process,
    _expand_detected_value_occurrences,
    _normalize_method_bundle,
    _drop_implausible_name_spans,
    build_extraction_system_prompt,
    get_method_definition_by_id,
    list_agent_methods,
    merge_method_spans,
    normalize_method_spans,
    run_llm_with_metadata,
    run_method_with_metadata,
    run_regex,
)
from metrics import compute_metrics
from span_resolution import RESOLUTION_POLICY_VERSION, resolve_spans, summarize_resolution_events

app = FastAPI(title="Annotation Tool")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BACKEND_DIR = Path(__file__).resolve().parent
REPO_ROOT = BACKEND_DIR.parent
ROOT_ENV_PATH = REPO_ROOT / ".env.local"
BASE_DIR = BACKEND_DIR / ".annotation_tool"
LEGACY_BASE_DIR = REPO_ROOT / ".annotation_tool"
SESSIONS_DIR = BASE_DIR / "sessions"
CONFIG_PATH = BASE_DIR / "config.json"

BUNDLE_FORMAT = "annotation_tool_session"
BUNDLE_VERSION = 5
SUPPORTED_BUNDLE_VERSIONS = {1, 2, 3, 4, 5}
TOOL_VERSION = "2026.03.03"
PROMPT_LAB_DIR_NAME = "prompt_lab"
METHODS_LAB_DIR_NAME = "methods_lab"
PROMPT_LAB_MAX_VARIANTS = 6
PROMPT_LAB_DEFAULT_CONCURRENCY = 10
PROMPT_LAB_DEFAULT_MAX_CONCURRENCY = 16
METHODS_LAB_MAX_METHOD_VARIANTS = max(12, len(METHOD_DEFINITION_BY_ID))
METHODS_LAB_DEFAULT_CONCURRENCY = 10
METHODS_LAB_DEFAULT_MAX_CONCURRENCY = 16
EXPERIMENT_HARD_MAX_CONCURRENCY = 32
DEFAULT_CHUNK_SIZE_CHARS = 10_000
MIN_CHUNK_SIZE_CHARS = 2_000
MAX_CHUNK_SIZE_CHARS = 30_000
FALLBACK_CHUNK_OVERLAP_CHARS = 200
DEFAULT_CHUNK_PARALLEL_WORKERS = 4
MAX_CHUNK_PARALLEL_WORKERS = 8
PROMPT_LAB_ALLOWED_PRESET_METHODS = set(METHOD_DEFINITION_BY_ID.keys())
DEIDENTIFY_V2_LEGACY_CHUNK_SIZE = 50_000
DEIDENTIFY_V2_LABEL_ALIASES: dict[str, str] = {
    "NAME": "PERSON",
    "NAME_STUDENT": "PERSON",
    "PHONE": "PHONE_NUMBER",
    "DRIVER_LICENSE": "US_DRIVER_LICENSE",
    "IP": "IP_ADDRESS",
    "ADDRESS": "LOCATION",
    "SCHOOL": "LOCATION",
}

COARSE_SIMPLE_LABEL_MAP: dict[str, str] = {
    # Simple labels remain unchanged.
    "AGE": "AGE",
    "DATE": "DATE",
    "EMAIL": "EMAIL",
    "LOCATION": "LOCATION",
    "MISC_ID": "MISC_ID",
    "NAME": "NAME",
    "PHONE": "PHONE",
    "SCHOOL": "SCHOOL",
    "URL": "URL",
    # Advanced -> simple projection.
    "COURSE": "MISC_ID",
    "EMAIL_ADDRESS": "EMAIL",
    "GRADE_LEVEL": "MISC_ID",
    "IP_ADDRESS": "MISC_ID",
    "NRP": "MISC_ID",
    "PERSON": "NAME",
    "PHONE_NUMBER": "PHONE",
    "SOCIAL_HANDLE": "MISC_ID",
    "US_BANK_NUMBER": "MISC_ID",
    "US_DRIVER_LICENSE": "MISC_ID",
    "US_PASSPORT": "MISC_ID",
    "US_SSN": "MISC_ID",
}

DEFAULT_CONFIG = {
    "system_prompt": SYSTEM_PROMPT,
    "model": "openai.gpt-5.3-codex",
    "temperature": 0.0,
    "api_base": "",
    "reasoning_effort": "xhigh",
    "anthropic_thinking": False,
    "anthropic_thinking_budget_tokens": None,
    "prompt_lab_max_concurrency": PROMPT_LAB_DEFAULT_MAX_CONCURRENCY,
    "methods_lab_max_concurrency": METHODS_LAB_DEFAULT_MAX_CONCURRENCY,
}

MANUAL_RUNS_SIDECAR_KIND = "manual.runs"
MANUAL_RUNS_METADATA_SIDECAR_KIND = "manual.runs.meta"
LLM_RUNS_SIDECAR_KIND = "agent.llm.runs"
METHOD_RUNS_SIDECAR_KIND = "agent.method.runs"
LLM_RUNS_METADATA_SIDECAR_KIND = "agent.llm.runs.meta"
METHOD_RUNS_METADATA_SIDECAR_KIND = "agent.method.runs.meta"


def _ensure_dirs():
    _migrate_legacy_storage_if_needed()
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def _migrate_legacy_storage_if_needed():
    if LEGACY_BASE_DIR == BASE_DIR:
        return
    if not LEGACY_BASE_DIR.is_dir():
        return
    if BASE_DIR.exists() and any(BASE_DIR.iterdir()):
        return

    BASE_DIR.parent.mkdir(parents=True, exist_ok=True)
    if not BASE_DIR.exists():
        shutil.copytree(LEGACY_BASE_DIR, BASE_DIR)
        return

    for legacy_child in LEGACY_BASE_DIR.iterdir():
        target = BASE_DIR / legacy_child.name
        if target.exists():
            continue
        if legacy_child.is_dir():
            shutil.copytree(legacy_child, target)
        else:
            shutil.copy2(legacy_child, target)


def _session_dir(session_id: str = "default") -> Path:
    _migrate_legacy_storage_if_needed()
    return SESSIONS_DIR / session_id


def _folders_dir(session_id: str = "default") -> Path:
    return _session_dir(session_id) / "folders"


def _load_folder_index(session_id: str = "default") -> list[str]:
    idx_path = _folders_dir(session_id) / "_index.json"
    if not idx_path.exists():
        return []
    raw = json.loads(idx_path.read_text())
    if not isinstance(raw, list):
        return []
    return [str(value) for value in raw if str(value).strip()]


def _save_folder_index(folder_ids: list[str], session_id: str = "default") -> None:
    folder_dir = _folders_dir(session_id)
    folder_dir.mkdir(parents=True, exist_ok=True)
    (folder_dir / "_index.json").write_text(json.dumps(folder_ids, indent=2))


def _load_folder(folder_id: str, session_id: str = "default") -> FolderRecord | None:
    path = _folders_dir(session_id) / f"{folder_id}.json"
    if not path.exists():
        return None
    return FolderRecord.model_validate_json(path.read_text())


def _save_folder(folder: FolderRecord, session_id: str = "default") -> None:
    folder_dir = _folders_dir(session_id)
    folder_dir.mkdir(parents=True, exist_ok=True)
    (folder_dir / f"{folder.id}.json").write_text(folder.model_dump_json(indent=2))


def _delete_folder_file(folder_id: str, session_id: str = "default") -> bool:
    path = _folders_dir(session_id) / f"{folder_id}.json"
    if not path.exists():
        return False
    path.unlink()
    return True


def _load_all_folders(session_id: str = "default") -> list[FolderRecord]:
    folders: list[FolderRecord] = []
    for folder_id in _load_folder_index(session_id):
        folder = _load_folder(folder_id, session_id)
        if folder is not None:
            folders.append(folder)
    return folders


def _build_document_summary(
    doc_id: str,
    session_id: str = "default",
    *,
    display_name: str | None = None,
) -> dict[str, str]:
    doc = _load_doc(doc_id, session_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
    manual = _load_sidecar(doc_id, "manual", session_id)
    status = "in_progress" if manual else "pending"
    return {
        "id": doc.id,
        "filename": doc.filename,
        "display_name": display_name or doc.filename,
        "status": status,
    }


def _folder_to_summary(folder: FolderRecord, session_id: str = "default") -> dict[str, object]:
    return {
        "id": folder.id,
        "name": folder.name,
        "kind": folder.kind,
        "parent_folder_id": folder.parent_folder_id,
        "merged_doc_id": folder.merged_doc_id,
        "doc_count": len(folder.doc_ids),
        "child_folder_count": len(folder.child_folder_ids),
        "source_filename": folder.source_filename,
        "source_folder_id": folder.source_folder_id,
        "sample_size": folder.sample_size,
        "sample_seed": folder.sample_seed,
        "created_at": folder.created_at,
    }


def _folder_to_detail(folder: FolderRecord, session_id: str = "default") -> dict[str, object]:
    documents: list[dict[str, str]] = []
    for doc_id in folder.doc_ids:
        doc = _load_doc(doc_id, session_id)
        if doc is None:
            continue
        documents.append(
            _build_document_summary(
                doc_id,
                session_id,
                display_name=folder.doc_display_names.get(doc_id),
            )
        )
    child_folders: list[dict[str, object]] = []
    for child_folder_id in folder.child_folder_ids:
        child = _load_folder(child_folder_id, session_id)
        if child is None:
            continue
        child_folders.append(_folder_to_summary(child, session_id))
    return {
        **_folder_to_summary(folder, session_id),
        "doc_ids": [item["id"] for item in documents],
        "child_folder_ids": [item["id"] for item in child_folders],
        "documents": documents,
        "child_folders": child_folders,
    }


def _resolve_ground_truth_export_doc_ids(
    scope: Literal["top_level", "folder"],
    folder_id: str | None,
    session_id: str = "default",
) -> list[str]:
    if scope == "top_level":
        return _load_session_index(session_id)

    normalized_folder_id = folder_id.strip() if folder_id else ""
    if not normalized_folder_id:
        raise HTTPException(status_code=400, detail="folder_id is required when scope=folder")

    root_folder = _load_folder(normalized_folder_id, session_id)
    if root_folder is None:
        raise HTTPException(status_code=404, detail="Folder not found")

    doc_ids: list[str] = []
    seen_doc_ids: set[str] = set()
    visited_folder_ids: set[str] = set()

    def append_doc_id(doc_id: str | None) -> None:
        if not doc_id:
            return
        normalized = str(doc_id).strip()
        if not normalized or normalized in seen_doc_ids:
            return
        seen_doc_ids.add(normalized)
        doc_ids.append(normalized)

    def visit_folder(folder: FolderRecord) -> None:
        if folder.id in visited_folder_ids:
            return
        visited_folder_ids.add(folder.id)
        for doc_id in folder.doc_ids:
            append_doc_id(doc_id)
        for child_folder_id in folder.child_folder_ids:
            child_folder = _load_folder(child_folder_id, session_id)
            if child_folder is not None:
                visit_folder(child_folder)

    visit_folder(root_folder)
    return doc_ids


def _find_folders_for_doc(doc_id: str, session_id: str = "default") -> list[FolderRecord]:
    return [folder for folder in _load_all_folders(session_id) if doc_id in folder.doc_ids]


def _doc_has_pre_or_manual_annotations(doc_id: str, session_id: str = "default") -> bool:
    doc = _load_doc(doc_id, session_id)
    if doc is None:
        return False
    if doc.pre_annotations:
        return True
    return bool(_load_sidecar(doc_id, "manual", session_id) or [])


def _remove_doc_ids_from_folder_records(
    doc_ids: set[str],
    session_id: str = "default",
) -> list[str]:
    if not doc_ids:
        return []

    updated_folder_ids: list[str] = []
    for folder in _load_all_folders(session_id):
        next_doc_ids = [doc_id for doc_id in folder.doc_ids if doc_id not in doc_ids]
        next_doc_display_names = {
            doc_id: display_name
            for doc_id, display_name in folder.doc_display_names.items()
            if doc_id in next_doc_ids
        }
        next_merged_doc_id = None if folder.merged_doc_id in doc_ids else folder.merged_doc_id
        if (
            next_doc_ids == folder.doc_ids
            and next_doc_display_names == folder.doc_display_names
            and next_merged_doc_id == folder.merged_doc_id
        ):
            continue

        _save_folder(
            folder.model_copy(
                update={
                    "doc_ids": next_doc_ids,
                    "doc_display_names": next_doc_display_names,
                    "merged_doc_id": next_merged_doc_id,
                }
            ),
            session_id,
        )
        updated_folder_ids.append(folder.id)

    return updated_folder_ids


def _remove_doc_ids_from_session_index(doc_ids: set[str], session_id: str = "default") -> None:
    if not doc_ids:
        return
    ids = _load_session_index(session_id)
    next_ids = [doc_id for doc_id in ids if doc_id not in doc_ids]
    if next_ids == ids:
        return
    _session_docs[session_id] = next_ids
    _save_session_index(session_id)


def _prune_unannotated_folder_docs(
    folder: FolderRecord,
    session_id: str = "default",
) -> tuple[list[str], list[str]]:
    removed_doc_ids = [
        doc_id
        for doc_id in folder.doc_ids
        if not _doc_has_pre_or_manual_annotations(doc_id, session_id)
    ]
    if not removed_doc_ids:
        return [], []

    removed_doc_id_set = set(removed_doc_ids)
    updated_folder_ids = _remove_doc_ids_from_folder_records(removed_doc_id_set, session_id)
    _remove_doc_ids_from_session_index(removed_doc_id_set, session_id)
    for doc_id in removed_doc_ids:
        _delete_doc_files(doc_id, session_id)
    return removed_doc_ids, updated_folder_ids


def _save_hidden_doc(doc: CanonicalDocument, session_id: str = "default") -> None:
    _save_doc(doc, session_id)


def _save_doc(doc: CanonicalDocument, session_id: str = "default"):
    d = _session_dir(session_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{doc.id}.source.json").write_text(doc.model_dump_json(indent=2))


def _load_doc(doc_id: str, session_id: str = "default") -> CanonicalDocument | None:
    p = _session_dir(session_id) / f"{doc_id}.source.json"
    if not p.exists():
        return None
    return CanonicalDocument.model_validate_json(p.read_text())


def _save_sidecar(
    doc_id: str, kind: str, spans: list[CanonicalSpan], session_id: str = "default"
):
    d = _session_dir(session_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{doc_id}.{kind}.json").write_text(
        json.dumps([s.model_dump() for s in spans], indent=2)
    )


def _load_sidecar(
    doc_id: str, kind: str, session_id: str = "default"
) -> list[CanonicalSpan] | None:
    p = _session_dir(session_id) / f"{doc_id}.{kind}.json"
    if not p.exists():
        return None
    raw = json.loads(p.read_text())
    return [CanonicalSpan(**s) for s in raw]


def _delete_sidecar(doc_id: str, kind: str, session_id: str = "default") -> bool:
    path = _session_dir(session_id) / f"{doc_id}.{kind}.json"
    if not path.exists():
        return False
    path.unlink()
    return True


def _persist_manual_annotations(
    doc_id: str,
    spans: list[CanonicalSpan],
    session_id: str = "default",
    *,
    updated_at: str | None = None,
) -> None:
    _save_sidecar(doc_id, "manual", spans, session_id)
    _upsert_span_map_entry(
        doc_id,
        MANUAL_RUNS_SIDECAR_KIND,
        "manual",
        spans,
        session_id,
    )
    _upsert_run_metadata(
        doc_id,
        MANUAL_RUNS_METADATA_SIDECAR_KIND,
        "manual",
        SavedRunMetadata(
            mode="manual",
            updated_at=updated_at or _now_iso(),
        ),
        session_id,
    )


def _load_method_sidecars(
    doc_id: str,
    session_id: str = "default",
) -> dict[str, list[CanonicalSpan]]:
    base = _session_dir(session_id)
    sidecars: dict[str, list[CanonicalSpan]] = {}
    prefix = f"{doc_id}.agent.method."
    for path in sorted(base.glob(f"{prefix}*.json")):
        name = path.name
        if not name.startswith(prefix) or not name.endswith(".json"):
            continue
        method_id = name[len(prefix) : -len(".json")]
        if not method_id:
            continue
        try:
            raw = json.loads(path.read_text())
            if not isinstance(raw, list):
                continue
            sidecars[method_id] = [CanonicalSpan(**item) for item in raw]
        except Exception:
            continue
    return sidecars


def _save_json_sidecar(
    doc_id: str,
    kind: str,
    payload: dict,
    session_id: str = "default",
):
    d = _session_dir(session_id)
    d.mkdir(parents=True, exist_ok=True)
    target = d / f"{doc_id}.{kind}.json"
    temp = d / f".{doc_id}.{kind}.{uuid.uuid4().hex}.tmp"
    try:
        temp.write_text(json.dumps(payload, indent=2))
        temp.replace(target)
    finally:
        if temp.exists():
            temp.unlink()


def _load_json_sidecar(
    doc_id: str,
    kind: str,
    session_id: str = "default",
) -> dict | None:
    p = _session_dir(session_id) / f"{doc_id}.{kind}.json"
    if not p.exists():
        return None
    raw = json.loads(p.read_text())
    if not isinstance(raw, dict):
        return None
    return raw


def _load_span_map_sidecar(
    doc_id: str,
    kind: str,
    session_id: str = "default",
) -> dict[str, list[CanonicalSpan]]:
    payload = _load_json_sidecar(doc_id, kind, session_id)
    if not isinstance(payload, dict):
        return {}
    result: dict[str, list[CanonicalSpan]] = {}
    for raw_key, raw_spans in payload.items():
        if not isinstance(raw_key, str) or not isinstance(raw_spans, list):
            continue
        spans: list[CanonicalSpan] = []
        malformed = False
        for item in raw_spans:
            try:
                spans.append(CanonicalSpan.model_validate(item))
            except Exception:
                malformed = True
                break
        if not malformed:
            result[raw_key] = spans
    return result


def _save_span_map_sidecar(
    doc_id: str,
    kind: str,
    payload: dict[str, list[CanonicalSpan]],
    session_id: str = "default",
):
    serialized = {
        key: [span.model_dump() for span in spans]
        for key, spans in payload.items()
        if isinstance(key, str) and key.strip() != ""
    }
    _save_json_sidecar(doc_id, kind, serialized, session_id)


def _upsert_span_map_entry(
    doc_id: str,
    kind: str,
    key: str,
    spans: list[CanonicalSpan],
    session_id: str = "default",
):
    normalized_key = key.strip()
    if not normalized_key:
        return
    existing = _load_span_map_sidecar(doc_id, kind, session_id)
    existing[normalized_key] = spans
    _save_span_map_sidecar(doc_id, kind, existing, session_id)


def _load_run_metadata_map_sidecar(
    doc_id: str,
    kind: str,
    session_id: str = "default",
) -> dict[str, SavedRunMetadata]:
    payload = _load_json_sidecar(doc_id, kind, session_id)
    if not isinstance(payload, dict):
        return {}
    result: dict[str, SavedRunMetadata] = {}
    for raw_key, raw_value in payload.items():
        if not isinstance(raw_key, str):
            continue
        try:
            result[raw_key] = SavedRunMetadata.model_validate(raw_value)
        except Exception:
            continue
    return result


def _save_run_metadata_map_sidecar(
    doc_id: str,
    kind: str,
    payload: dict[str, SavedRunMetadata],
    session_id: str = "default",
):
    serialized = {
        key: metadata.model_dump()
        for key, metadata in payload.items()
        if isinstance(key, str) and key.strip() != ""
    }
    _save_json_sidecar(doc_id, kind, serialized, session_id)


def _upsert_run_metadata(
    doc_id: str,
    kind: str,
    key: str,
    metadata: SavedRunMetadata,
    session_id: str = "default",
):
    normalized_key = key.strip()
    if not normalized_key:
        return
    existing = _load_run_metadata_map_sidecar(doc_id, kind, session_id)
    existing[normalized_key] = metadata
    _save_run_metadata_map_sidecar(doc_id, kind, existing, session_id)


def _normalize_optional_llm_confidence(raw: object) -> LLMConfidenceMetric | None:
    if raw is None:
        return None
    try:
        return LLMConfidenceMetric.model_validate(raw)
    except Exception as exc:
        raise ValueError(f"Invalid llm_confidence metric: {exc}") from exc


# Session index management
_session_docs: dict[str, list[str]] = {}
_prompt_lab_runs: dict[str, list[str]] = {}
_prompt_lab_lock = threading.Lock()
_prompt_lab_cancel_events: dict[tuple[str, str], threading.Event] = {}
_methods_lab_runs: dict[str, list[str]] = {}
_methods_lab_lock = threading.Lock()
_methods_lab_cancel_events: dict[tuple[str, str], threading.Event] = {}
_agent_progress: dict[str, dict[str, Any]] = {}
_agent_progress_lock = threading.Lock()


def _load_session_index(session_id: str = "default") -> list[str]:
    if session_id in _session_docs:
        return _session_docs[session_id]
    idx_path = _session_dir(session_id) / "_index.json"
    if idx_path.exists():
        ids = json.loads(idx_path.read_text())
        _session_docs[session_id] = ids
        return ids
    _session_docs[session_id] = []
    return _session_docs[session_id]


def _save_session_index(session_id: str = "default"):
    d = _session_dir(session_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "_index.json").write_text(json.dumps(_session_docs.get(session_id, [])))


ImportConflictPolicy = Literal["replace", "add_new", "keep_current"]


@dataclass(frozen=True)
class ImportedDocumentCommitResult:
    doc_id: str
    created: bool
    conflict_action: ImportConflictPolicy | None = None


def _empty_import_conflict_counts() -> dict[ImportConflictPolicy, int]:
    return {"replace": 0, "add_new": 0, "keep_current": 0}


def _record_import_conflict(
    counts: dict[ImportConflictPolicy, int],
    result: ImportedDocumentCommitResult,
) -> None:
    if result.conflict_action is None:
        return
    counts[result.conflict_action] += 1


def _persist_imported_document(
    *,
    doc_id: str,
    doc: CanonicalDocument,
    session_id: str,
    manual_spans: list[CanonicalSpan] | None = None,
    rule_spans: list[CanonicalSpan] | None = None,
    llm_spans: list[CanonicalSpan] | None = None,
    method_spans: dict[str, list[CanonicalSpan]] | None = None,
    llm_runs: dict[str, list[CanonicalSpan]] | None = None,
    method_runs: dict[str, list[CanonicalSpan]] | None = None,
    llm_run_metadata: dict[str, SavedRunMetadata] | None = None,
    method_run_metadata: dict[str, SavedRunMetadata] | None = None,
    llm_confidence: LLMConfidenceMetric | None = None,
    imported_label_profile: Literal["simple", "advanced"] | None = None,
) -> None:
    persisted_doc = doc if doc.id == doc_id else doc.model_copy(update={"id": doc_id})
    _save_doc(persisted_doc, session_id)
    if manual_spans:
        _save_sidecar(doc_id, "manual", manual_spans, session_id)
    if rule_spans:
        _save_sidecar(doc_id, "agent.rule", rule_spans, session_id)
    if llm_spans:
        _save_sidecar(doc_id, "agent.llm", llm_spans, session_id)
    for method_id, spans in (method_spans or {}).items():
        _save_sidecar(doc_id, f"agent.method.{method_id}", spans, session_id)
    if llm_runs:
        _save_span_map_sidecar(doc_id, LLM_RUNS_SIDECAR_KIND, llm_runs, session_id)
    if method_runs:
        _save_span_map_sidecar(doc_id, METHOD_RUNS_SIDECAR_KIND, method_runs, session_id)
    if llm_run_metadata:
        _save_run_metadata_map_sidecar(
            doc_id,
            LLM_RUNS_METADATA_SIDECAR_KIND,
            llm_run_metadata,
            session_id,
        )
    if method_run_metadata:
        _save_run_metadata_map_sidecar(
            doc_id,
            METHOD_RUNS_METADATA_SIDECAR_KIND,
            method_run_metadata,
            session_id,
        )
    if llm_confidence is not None:
        _save_json_sidecar(
            doc_id,
            "agent.llm.metrics",
            llm_confidence.model_dump(),
            session_id,
        )
    if imported_label_profile is not None:
        _save_json_sidecar(
            doc_id,
            "agent.last_run",
            {
                "mode": "imported",
                "label_profile": imported_label_profile,
                "updated_at": _now_iso(),
            },
            session_id,
        )


def _create_imported_document(
    *,
    doc: CanonicalDocument,
    session_id: str,
    ids: list[str],
    existing_ids: set[str],
    add_to_session_index: bool = True,
    manual_spans: list[CanonicalSpan] | None = None,
    rule_spans: list[CanonicalSpan] | None = None,
    llm_spans: list[CanonicalSpan] | None = None,
    method_spans: dict[str, list[CanonicalSpan]] | None = None,
    llm_runs: dict[str, list[CanonicalSpan]] | None = None,
    method_runs: dict[str, list[CanonicalSpan]] | None = None,
    llm_run_metadata: dict[str, SavedRunMetadata] | None = None,
    method_run_metadata: dict[str, SavedRunMetadata] | None = None,
    llm_confidence: LLMConfidenceMetric | None = None,
    imported_label_profile: Literal["simple", "advanced"] | None = None,
) -> str:
    new_id = doc.id
    while new_id in existing_ids or _load_doc(new_id, session_id) is not None:
        new_id = str(uuid.uuid4())[:8]

    _persist_imported_document(
        doc_id=new_id,
        doc=doc,
        session_id=session_id,
        manual_spans=manual_spans,
        rule_spans=rule_spans,
        llm_spans=llm_spans,
        method_spans=method_spans,
        llm_runs=llm_runs,
        method_runs=method_runs,
        llm_run_metadata=llm_run_metadata,
        method_run_metadata=method_run_metadata,
        llm_confidence=llm_confidence,
        imported_label_profile=imported_label_profile,
    )

    if add_to_session_index:
        ids.append(new_id)
    existing_ids.add(new_id)
    return new_id


def _commit_imported_document(
    *,
    doc: CanonicalDocument,
    session_id: str,
    ids: list[str],
    existing_ids: set[str],
    add_to_session_index: bool = True,
    manual_spans: list[CanonicalSpan] | None = None,
    rule_spans: list[CanonicalSpan] | None = None,
    llm_spans: list[CanonicalSpan] | None = None,
    method_spans: dict[str, list[CanonicalSpan]] | None = None,
    llm_runs: dict[str, list[CanonicalSpan]] | None = None,
    method_runs: dict[str, list[CanonicalSpan]] | None = None,
    llm_run_metadata: dict[str, SavedRunMetadata] | None = None,
    method_run_metadata: dict[str, SavedRunMetadata] | None = None,
    llm_confidence: LLMConfidenceMetric | None = None,
    imported_label_profile: Literal["simple", "advanced"] | None = None,
    conflict_policy: ImportConflictPolicy = "replace",
) -> ImportedDocumentCommitResult:
    matched_doc_id = _find_existing_import_match(doc=doc, session_id=session_id, ids=ids)
    if matched_doc_id is not None:
        existing_ids.add(matched_doc_id)
        if add_to_session_index and matched_doc_id not in ids:
            ids.append(matched_doc_id)
        if conflict_policy == "keep_current":
            return ImportedDocumentCommitResult(
                doc_id=matched_doc_id,
                created=False,
                conflict_action="keep_current",
            )
        if conflict_policy == "replace":
            _delete_doc_files(matched_doc_id, session_id)
            _persist_imported_document(
                doc_id=matched_doc_id,
                doc=doc,
                session_id=session_id,
                manual_spans=manual_spans,
                rule_spans=rule_spans,
                llm_spans=llm_spans,
                method_spans=method_spans,
                llm_runs=llm_runs,
                method_runs=method_runs,
                llm_run_metadata=llm_run_metadata,
                method_run_metadata=method_run_metadata,
                llm_confidence=llm_confidence,
                imported_label_profile=imported_label_profile,
            )
            return ImportedDocumentCommitResult(
                doc_id=matched_doc_id,
                created=False,
                conflict_action="replace",
            )

    new_id = _create_imported_document(
        doc=doc,
        session_id=session_id,
        ids=ids,
        existing_ids=existing_ids,
        add_to_session_index=add_to_session_index,
        manual_spans=manual_spans,
        rule_spans=rule_spans,
        llm_spans=llm_spans,
        method_spans=method_spans,
        llm_runs=llm_runs,
        method_runs=method_runs,
        llm_run_metadata=llm_run_metadata,
        method_run_metadata=method_run_metadata,
        llm_confidence=llm_confidence,
        imported_label_profile=imported_label_profile,
    )
    return ImportedDocumentCommitResult(
        doc_id=new_id,
        created=True,
        conflict_action="add_new" if matched_doc_id is not None else None,
    )


def _document_import_fingerprint(doc: CanonicalDocument) -> tuple[str, str]:
    return (
        doc.filename,
        hashlib.sha256(doc.raw_text.encode("utf-8")).hexdigest(),
    )


def _find_existing_import_match(
    *,
    doc: CanonicalDocument,
    session_id: str,
    ids: list[str],
) -> str | None:
    fingerprint = _document_import_fingerprint(doc)
    exact = _load_doc(doc.id, session_id)
    if exact is not None and _document_import_fingerprint(exact) == fingerprint:
        return exact.id

    best_match_id: str | None = None
    best_manual_count = -1
    for existing_id in ids:
        existing_doc = _load_doc(existing_id, session_id)
        if existing_doc is None:
            continue
        if _document_import_fingerprint(existing_doc) != fingerprint:
            continue
        manual_count = len(_load_sidecar(existing_id, "manual", session_id) or [])
        if manual_count > best_manual_count:
            best_match_id = existing_id
            best_manual_count = manual_count
    return best_match_id


def _span_signature(spans: list[CanonicalSpan]) -> list[tuple[int, int, str, str]]:
    return sorted((span.start, span.end, span.label, span.text) for span in spans)


def _prefer_imported_spans(
    existing_spans: list[CanonicalSpan] | None,
    imported_spans: list[CanonicalSpan] | None,
) -> tuple[list[CanonicalSpan] | None, bool]:
    if not imported_spans:
        return existing_spans, False
    if not existing_spans:
        return imported_spans, True
    if _span_signature(existing_spans) == _span_signature(imported_spans):
        return existing_spans, False
    if len(imported_spans) > len(existing_spans):
        return imported_spans, True
    return existing_spans, False


def _merge_span_sidecar(
    *,
    doc_id: str,
    kind: str,
    imported_spans: list[CanonicalSpan] | None,
    session_id: str,
):
    existing_spans = _load_sidecar(doc_id, kind, session_id)
    chosen_spans, should_save = _prefer_imported_spans(existing_spans, imported_spans)
    if should_save and chosen_spans:
        _save_sidecar(doc_id, kind, chosen_spans, session_id)


def _total_span_map_count(payload: dict[str, list[CanonicalSpan]]) -> int:
    return sum(len(spans) for spans in payload.values())


def _merge_span_map_sidecar(
    *,
    doc_id: str,
    kind: str,
    imported_payload: dict[str, list[CanonicalSpan]] | None,
    session_id: str,
):
    if not imported_payload:
        return
    existing_payload = _load_span_map_sidecar(doc_id, kind, session_id)
    if not existing_payload or _total_span_map_count(imported_payload) > _total_span_map_count(
        existing_payload
    ):
        _save_span_map_sidecar(doc_id, kind, imported_payload, session_id)


def _merge_run_metadata_map_sidecar(
    *,
    doc_id: str,
    kind: str,
    imported_payload: dict[str, SavedRunMetadata] | None,
    session_id: str,
):
    if not imported_payload:
        return
    existing_payload = _load_run_metadata_map_sidecar(doc_id, kind, session_id)
    if not existing_payload or len(imported_payload) > len(existing_payload):
        _save_run_metadata_map_sidecar(doc_id, kind, imported_payload, session_id)


def _merge_imported_sidecars_into_existing(
    *,
    doc_id: str,
    session_id: str,
    manual_spans: list[CanonicalSpan] | None = None,
    rule_spans: list[CanonicalSpan] | None = None,
    llm_spans: list[CanonicalSpan] | None = None,
    method_spans: dict[str, list[CanonicalSpan]] | None = None,
    llm_runs: dict[str, list[CanonicalSpan]] | None = None,
    method_runs: dict[str, list[CanonicalSpan]] | None = None,
    llm_run_metadata: dict[str, SavedRunMetadata] | None = None,
    method_run_metadata: dict[str, SavedRunMetadata] | None = None,
    llm_confidence: LLMConfidenceMetric | None = None,
    imported_label_profile: Literal["simple", "advanced"] | None = None,
):
    _merge_span_sidecar(
        doc_id=doc_id,
        kind="manual",
        imported_spans=manual_spans,
        session_id=session_id,
    )
    _merge_span_sidecar(
        doc_id=doc_id,
        kind="agent.rule",
        imported_spans=rule_spans,
        session_id=session_id,
    )
    _merge_span_sidecar(
        doc_id=doc_id,
        kind="agent.llm",
        imported_spans=llm_spans,
        session_id=session_id,
    )
    for method_id, spans in (method_spans or {}).items():
        _merge_span_sidecar(
            doc_id=doc_id,
            kind=f"agent.method.{method_id}",
            imported_spans=spans,
            session_id=session_id,
        )

    _merge_span_map_sidecar(
        doc_id=doc_id,
        kind=LLM_RUNS_SIDECAR_KIND,
        imported_payload=llm_runs,
        session_id=session_id,
    )
    _merge_span_map_sidecar(
        doc_id=doc_id,
        kind=METHOD_RUNS_SIDECAR_KIND,
        imported_payload=method_runs,
        session_id=session_id,
    )
    _merge_run_metadata_map_sidecar(
        doc_id=doc_id,
        kind=LLM_RUNS_METADATA_SIDECAR_KIND,
        imported_payload=llm_run_metadata,
        session_id=session_id,
    )
    _merge_run_metadata_map_sidecar(
        doc_id=doc_id,
        kind=METHOD_RUNS_METADATA_SIDECAR_KIND,
        imported_payload=method_run_metadata,
        session_id=session_id,
    )

    if llm_confidence is not None:
        existing_metric = _load_json_sidecar(doc_id, "agent.llm.metrics", session_id)
        if not existing_metric:
            _save_json_sidecar(
                doc_id,
                "agent.llm.metrics",
                llm_confidence.model_dump(),
                session_id,
            )
    if imported_label_profile is not None:
        existing_last_run = _load_json_sidecar(doc_id, "agent.last_run", session_id)
        if not existing_last_run:
            _save_json_sidecar(
                doc_id,
                "agent.last_run",
                {
                    "mode": "imported",
                    "label_profile": imported_label_profile,
                    "updated_at": _now_iso(),
                },
                session_id,
            )


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_agent_llm_run_key(model: str) -> str:
    normalized_model = model.strip() or "llm"
    return f"{normalized_model}::{uuid.uuid4().hex[:10]}"


def _agent_progress_key(doc_id: str, session_id: str = "default") -> str:
    return f"{session_id}:{doc_id}"


def _start_agent_progress(
    doc_id: str,
    *,
    mode: str,
    total_chunks: int,
    session_id: str = "default",
):
    now = _now_iso()
    key = _agent_progress_key(doc_id, session_id)
    with _agent_progress_lock:
        _agent_progress[key] = {
            "doc_id": doc_id,
            "mode": mode,
            "status": "running",
            "completed_chunks": 0,
            "total_chunks": max(1, int(total_chunks)),
            "progress": 0.0,
            "started_at": now,
            "updated_at": now,
            "message": None,
        }


def _update_agent_progress(
    doc_id: str,
    *,
    completed_chunks: int,
    total_chunks: int,
    session_id: str = "default",
):
    key = _agent_progress_key(doc_id, session_id)
    now = _now_iso()
    with _agent_progress_lock:
        current = _agent_progress.get(key)
        if not current:
            return
        total = max(1, int(total_chunks))
        completed = max(0, min(int(completed_chunks), total))
        current["status"] = "running"
        current["completed_chunks"] = completed
        current["total_chunks"] = total
        current["progress"] = completed / total
        current["updated_at"] = now
        _agent_progress[key] = current


def _finish_agent_progress(
    doc_id: str,
    *,
    status: Literal["completed", "failed"],
    session_id: str = "default",
    message: str | None = None,
):
    key = _agent_progress_key(doc_id, session_id)
    now = _now_iso()
    with _agent_progress_lock:
        current = _agent_progress.get(key)
        if not current:
            current = {
                "doc_id": doc_id,
                "mode": "unknown",
                "status": status,
                "completed_chunks": 1 if status == "completed" else 0,
                "total_chunks": 1,
                "progress": 1.0 if status == "completed" else 0.0,
                "started_at": now,
                "updated_at": now,
                "message": message,
            }
        else:
            total = max(1, int(current.get("total_chunks") or 1))
            if status == "completed":
                completed = total
            else:
                completed = max(0, min(int(current.get("completed_chunks") or 0), total))
            current["status"] = status
            current["completed_chunks"] = completed
            current["total_chunks"] = total
            current["progress"] = completed / total
            current["updated_at"] = now
            current["message"] = message
        _agent_progress[key] = current


def _get_agent_progress(doc_id: str, session_id: str = "default") -> dict[str, Any]:
    key = _agent_progress_key(doc_id, session_id)
    with _agent_progress_lock:
        payload = _agent_progress.get(key)
        if payload:
            return dict(payload)
    return {
        "doc_id": doc_id,
        "mode": None,
        "status": "idle",
        "completed_chunks": 0,
        "total_chunks": 0,
        "progress": 0.0,
        "started_at": None,
        "updated_at": _now_iso(),
        "message": None,
    }


def _prompt_lab_dir(session_id: str = "default") -> Path:
    return _session_dir(session_id) / PROMPT_LAB_DIR_NAME


def _prompt_lab_index_path(session_id: str = "default") -> Path:
    return _prompt_lab_dir(session_id) / "_index.json"


def _prompt_lab_run_path(run_id: str, session_id: str = "default") -> Path:
    return _prompt_lab_dir(session_id) / f"{run_id}.json"


def _prompt_lab_cancel_key(run_id: str, session_id: str = "default") -> tuple[str, str]:
    return session_id, run_id


def _register_prompt_lab_cancel_event(run_id: str, session_id: str = "default") -> threading.Event:
    event = threading.Event()
    _prompt_lab_cancel_events[_prompt_lab_cancel_key(run_id, session_id)] = event
    return event


def _get_prompt_lab_cancel_event(
    run_id: str, session_id: str = "default"
) -> threading.Event | None:
    return _prompt_lab_cancel_events.get(_prompt_lab_cancel_key(run_id, session_id))


def _clear_prompt_lab_cancel_event(run_id: str, session_id: str = "default"):
    _prompt_lab_cancel_events.pop(_prompt_lab_cancel_key(run_id, session_id), None)


def _load_prompt_lab_index(session_id: str = "default") -> list[str]:
    index_path = _prompt_lab_index_path(session_id)
    if index_path.exists():
        ids = json.loads(index_path.read_text())
        if isinstance(ids, list):
            _prompt_lab_runs[session_id] = [str(item) for item in ids]
            return _prompt_lab_runs[session_id]
    _prompt_lab_runs[session_id] = []
    return _prompt_lab_runs[session_id]


def _save_prompt_lab_index(session_id: str = "default"):
    d = _prompt_lab_dir(session_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "_index.json").write_text(json.dumps(_prompt_lab_runs.get(session_id, []), indent=2))


def _load_prompt_lab_run(run_id: str, session_id: str = "default") -> dict | None:
    path = _prompt_lab_run_path(run_id, session_id)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        return None
    return payload


def _save_prompt_lab_run(run: dict, session_id: str = "default"):
    run_id = str(run.get("id", "")).strip()
    if not run_id:
        raise ValueError("Prompt Lab run payload missing id")
    d = _prompt_lab_dir(session_id)
    d.mkdir(parents=True, exist_ok=True)
    _prompt_lab_run_path(run_id, session_id).write_text(json.dumps(run, indent=2))


def _delete_prompt_lab_run(run_id: str, session_id: str = "default") -> bool:
    ids = _load_prompt_lab_index(session_id)
    path = _prompt_lab_run_path(run_id, session_id)
    existed = path.exists()
    if not existed and run_id not in ids:
        return False
    if existed:
        path.unlink()
    _prompt_lab_runs[session_id] = [item for item in ids if item != run_id]
    _clear_prompt_lab_cancel_event(run_id, session_id)
    _save_prompt_lab_index(session_id)
    return True


def _is_terminal_prompt_lab_status(status: str) -> bool:
    return status in {"completed", "completed_with_errors", "failed", "cancelled"}


def _methods_lab_dir(session_id: str = "default") -> Path:
    return _session_dir(session_id) / METHODS_LAB_DIR_NAME


def _methods_lab_index_path(session_id: str = "default") -> Path:
    return _methods_lab_dir(session_id) / "_index.json"


def _methods_lab_run_path(run_id: str, session_id: str = "default") -> Path:
    return _methods_lab_dir(session_id) / f"{run_id}.json"


def _methods_lab_cancel_key(run_id: str, session_id: str = "default") -> tuple[str, str]:
    return session_id, run_id


def _register_methods_lab_cancel_event(
    run_id: str, session_id: str = "default"
) -> threading.Event:
    event = threading.Event()
    _methods_lab_cancel_events[_methods_lab_cancel_key(run_id, session_id)] = event
    return event


def _get_methods_lab_cancel_event(
    run_id: str, session_id: str = "default"
) -> threading.Event | None:
    return _methods_lab_cancel_events.get(_methods_lab_cancel_key(run_id, session_id))


def _clear_methods_lab_cancel_event(run_id: str, session_id: str = "default"):
    _methods_lab_cancel_events.pop(_methods_lab_cancel_key(run_id, session_id), None)


def _load_methods_lab_index(session_id: str = "default") -> list[str]:
    index_path = _methods_lab_index_path(session_id)
    if index_path.exists():
        ids = json.loads(index_path.read_text())
        if isinstance(ids, list):
            _methods_lab_runs[session_id] = [str(item) for item in ids]
            return _methods_lab_runs[session_id]
    _methods_lab_runs[session_id] = []
    return _methods_lab_runs[session_id]


def _save_methods_lab_index(session_id: str = "default"):
    d = _methods_lab_dir(session_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "_index.json").write_text(json.dumps(_methods_lab_runs.get(session_id, []), indent=2))


def _load_methods_lab_run(run_id: str, session_id: str = "default") -> dict | None:
    path = _methods_lab_run_path(run_id, session_id)
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        return None
    return payload


def _save_methods_lab_run(run: dict, session_id: str = "default"):
    run_id = str(run.get("id", "")).strip()
    if not run_id:
        raise ValueError("Methods Lab run payload missing id")
    d = _methods_lab_dir(session_id)
    d.mkdir(parents=True, exist_ok=True)
    _methods_lab_run_path(run_id, session_id).write_text(json.dumps(run, indent=2))


def _delete_methods_lab_run(run_id: str, session_id: str = "default") -> bool:
    ids = _load_methods_lab_index(session_id)
    path = _methods_lab_run_path(run_id, session_id)
    existed = path.exists()
    if not existed and run_id not in ids:
        return False
    if existed:
        path.unlink()
    _methods_lab_runs[session_id] = [item for item in ids if item != run_id]
    _clear_methods_lab_cancel_event(run_id, session_id)
    _save_methods_lab_index(session_id)
    return True


def _is_terminal_methods_lab_status(status: str) -> bool:
    return status in {"completed", "completed_with_errors", "failed", "cancelled"}


def _request_prompt_lab_cancel(run_id: str, session_id: str = "default") -> dict[str, object]:
    run = _load_prompt_lab_run(run_id, session_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Prompt Lab run not found")
    status = str(run.get("status", ""))
    if _is_terminal_prompt_lab_status(status):
        raise HTTPException(
            status_code=409,
            detail=f"Prompt Lab run is not cancellable in status '{status}'",
        )
    cancel_event = _get_prompt_lab_cancel_event(run_id, session_id)
    if cancel_event is None:
        raise HTTPException(
            status_code=409,
            detail="Prompt Lab run is not active in this server process and cannot be cancelled",
        )
    cancel_event.set()
    if status != "cancelling":
        run["status"] = "cancelling"
        warnings = run.get("warnings", [])
        if not isinstance(warnings, list):
            warnings = []
        warnings.append(f"Cancellation requested at {_now_iso()}.")
        run["warnings"] = warnings
        _save_prompt_lab_run(run, session_id)
    return {"ok": True, "id": run_id, "status": "cancelling"}


def _request_methods_lab_cancel(run_id: str, session_id: str = "default") -> dict[str, object]:
    run = _load_methods_lab_run(run_id, session_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Methods Lab run not found")
    status = str(run.get("status", ""))
    if _is_terminal_methods_lab_status(status):
        raise HTTPException(
            status_code=409,
            detail=f"Methods Lab run is not cancellable in status '{status}'",
        )
    cancel_event = _get_methods_lab_cancel_event(run_id, session_id)
    if cancel_event is None:
        raise HTTPException(
            status_code=409,
            detail="Methods Lab run is not active in this server process and cannot be cancelled",
        )
    cancel_event.set()
    if status != "cancelling":
        run["status"] = "cancelling"
        warnings = run.get("warnings", [])
        if not isinstance(warnings, list):
            warnings = []
        warnings.append(f"Cancellation requested at {_now_iso()}.")
        run["warnings"] = warnings
        _save_methods_lab_run(run, session_id)
    return {"ok": True, "id": run_id, "status": "cancelling"}


def _safe_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _normalize_error_family(error: str | None) -> str | None:
    message = (error or "").strip().lower()
    if not message:
        return None
    if "finish_reason=length" in message or (
        "empty output content" in message and "length" in message
    ):
        return "empty_output_finish_reason_length"
    if "connection error" in message or "connecterror" in message:
        return "connection_error"
    if "timeout" in message or "timed out" in message:
        return "timeout"
    if (
        "requires temperature=1" in message
        or "invalid" in message
        or "unsupported" in message
        or "api key" in message
        or "missing api key" in message
        or "system_prompt is required" in message
        or "method_verify_override" in message
    ):
        return "config_error"
    return "unknown_error"


def _serialize_metrics_value(value: object) -> object:
    if isinstance(value, CanonicalSpan):
        return value.model_dump()
    if isinstance(value, dict):
        return {
            str(key): _serialize_metrics_value(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_serialize_metrics_value(item) for item in value]
    return value


def _serialize_metrics_payload(metrics: dict) -> dict:
    payload = copy.deepcopy(metrics)
    serialized = _serialize_metrics_value(payload)
    return serialized if isinstance(serialized, dict) else {}


def _build_prompt_lab_cell_summary(cell: dict, total_docs: int, run_status: str) -> dict:
    from experiment_service import _build_prompt_lab_cell_summary as service_impl

    return service_impl(cell, total_docs, run_status)


def _build_prompt_lab_matrix(run: dict) -> dict:
    from experiment_service import build_prompt_lab_matrix

    return build_prompt_lab_matrix(run)


def _build_prompt_lab_run_summary(run: dict) -> dict:
    from experiment_service import build_prompt_lab_run_summary

    return build_prompt_lab_run_summary(run)


def _build_prompt_lab_run_detail(run: dict) -> dict:
    from experiment_service import build_prompt_lab_run_detail

    return build_prompt_lab_run_detail(run)


def _export_prompt_lab_runs(session_id: str = "default") -> list[dict]:
    with _prompt_lab_lock:
        runs: list[dict] = []
        for run_id in _load_prompt_lab_index(session_id):
            payload = _load_prompt_lab_run(run_id, session_id)
            if payload is None:
                continue
            runs.append(copy.deepcopy(payload))
        return runs


def _remap_prompt_lab_run_doc_ids(
    run: dict,
    doc_id_remap: dict[str, str],
    folder_id_remap: dict[str, str] | None = None,
) -> tuple[dict, list[str]]:
    remapped = copy.deepcopy(run)
    warnings: list[str] = []

    original_doc_ids = remapped.get("doc_ids", [])
    if not isinstance(original_doc_ids, list):
        original_doc_ids = []
    mapped_doc_ids: list[str] = []
    seen_doc_ids: set[str] = set()
    for value in original_doc_ids:
        old_id = str(value)
        new_id = doc_id_remap.get(old_id, old_id)
        if new_id not in seen_doc_ids:
            mapped_doc_ids.append(new_id)
            seen_doc_ids.add(new_id)
        if old_id not in doc_id_remap:
            warnings.append(
                f"Referenced document '{old_id}' was not imported; marked as unavailable."
            )
    remapped["doc_ids"] = mapped_doc_ids

    original_folder_ids = remapped.get("folder_ids", [])
    if not isinstance(original_folder_ids, list):
        original_folder_ids = []
    mapped_folder_ids: list[str] = []
    seen_folder_ids: set[str] = set()
    for value in original_folder_ids:
        old_id = str(value)
        new_id = (folder_id_remap or {}).get(old_id, old_id)
        if new_id not in seen_folder_ids:
            mapped_folder_ids.append(new_id)
            seen_folder_ids.add(new_id)
    remapped["folder_ids"] = mapped_folder_ids

    cells = remapped.get("cells", {})
    if isinstance(cells, dict):
        for cell in cells.values():
            if not isinstance(cell, dict):
                continue
            documents = cell.get("documents", {})
            if not isinstance(documents, dict):
                cell["documents"] = {}
                continue
            updated_documents: dict[str, dict] = {}
            for old_id, result in documents.items():
                mapped_id = doc_id_remap.get(str(old_id), str(old_id))
                if isinstance(result, dict):
                    item = copy.deepcopy(result)
                else:
                    item = {"status": "failed", "error": "Invalid imported result payload"}
                if str(old_id) not in doc_id_remap:
                    item["status"] = "unavailable"
                    item["error"] = "Referenced document was not imported."
                updated_documents[mapped_id] = item
            cell["documents"] = updated_documents

    run_warnings = remapped.get("warnings", [])
    if not isinstance(run_warnings, list):
        run_warnings = []
    run_warnings.extend(warnings)
    remapped["warnings"] = run_warnings
    if warnings:
        remapped["status"] = "completed_with_errors"
    return remapped, warnings


def _build_methods_lab_cell_summary(cell: dict, total_docs: int, run_status: str) -> dict:
    from experiment_service import _build_methods_lab_cell_summary as service_impl

    return service_impl(cell, total_docs, run_status)


def _build_methods_lab_matrix(run: dict) -> dict:
    from experiment_service import build_methods_lab_matrix

    return build_methods_lab_matrix(run)


def _build_methods_lab_run_summary(run: dict) -> dict:
    from experiment_service import build_methods_lab_run_summary

    return build_methods_lab_run_summary(run)


def _build_methods_lab_run_detail(run: dict) -> dict:
    from experiment_service import build_methods_lab_run_detail

    return build_methods_lab_run_detail(run)


def _export_methods_lab_runs(session_id: str = "default") -> list[dict]:
    with _methods_lab_lock:
        runs: list[dict] = []
        for run_id in _load_methods_lab_index(session_id):
            payload = _load_methods_lab_run(run_id, session_id)
            if payload is None:
                continue
            runs.append(copy.deepcopy(payload))
        return runs


def _remap_methods_lab_run_doc_ids(
    run: dict,
    doc_id_remap: dict[str, str],
    folder_id_remap: dict[str, str] | None = None,
) -> tuple[dict, list[str]]:
    remapped = copy.deepcopy(run)
    warnings: list[str] = []

    original_doc_ids = remapped.get("doc_ids", [])
    if not isinstance(original_doc_ids, list):
        original_doc_ids = []
    mapped_doc_ids: list[str] = []
    seen_doc_ids: set[str] = set()
    for value in original_doc_ids:
        old_id = str(value)
        new_id = doc_id_remap.get(old_id, old_id)
        if new_id not in seen_doc_ids:
            mapped_doc_ids.append(new_id)
            seen_doc_ids.add(new_id)
        if old_id not in doc_id_remap:
            warnings.append(
                f"Referenced document '{old_id}' was not imported; marked as unavailable."
            )
    remapped["doc_ids"] = mapped_doc_ids

    original_folder_ids = remapped.get("folder_ids", [])
    if not isinstance(original_folder_ids, list):
        original_folder_ids = []
    mapped_folder_ids: list[str] = []
    seen_folder_ids: set[str] = set()
    for value in original_folder_ids:
        old_id = str(value)
        new_id = (folder_id_remap or {}).get(old_id, old_id)
        if new_id not in seen_folder_ids:
            mapped_folder_ids.append(new_id)
            seen_folder_ids.add(new_id)
    remapped["folder_ids"] = mapped_folder_ids

    cells = remapped.get("cells", {})
    if isinstance(cells, dict):
        for cell in cells.values():
            if not isinstance(cell, dict):
                continue
            documents = cell.get("documents", {})
            if not isinstance(documents, dict):
                continue
            remapped_documents: dict[str, object] = {}
            for doc_id, result in documents.items():
                old_id = str(doc_id)
                new_id = doc_id_remap.get(old_id, old_id)
                remapped_documents[new_id] = result
            cell["documents"] = remapped_documents

    return remapped, warnings


def _enrich_doc(
    doc: CanonicalDocument, session_id: str = "default"
) -> CanonicalDocument:
    """Load sidecars and merge into document."""
    manual = _load_sidecar(doc.id, "manual", session_id)
    agent_rule = _load_sidecar(doc.id, "agent.rule", session_id)
    agent_llm = _load_sidecar(doc.id, "agent.llm", session_id)
    agent_llm_runs = _load_span_map_sidecar(doc.id, LLM_RUNS_SIDECAR_KIND, session_id)
    agent_llm_run_metadata = _load_run_metadata_map_sidecar(
        doc.id,
        LLM_RUNS_METADATA_SIDECAR_KIND,
        session_id,
    )
    agent_methods = _load_method_sidecars(doc.id, session_id)
    method_runs = _load_span_map_sidecar(doc.id, METHOD_RUNS_SIDECAR_KIND, session_id)
    method_run_metadata = _load_run_metadata_map_sidecar(
        doc.id,
        METHOD_RUNS_METADATA_SIDECAR_KIND,
        session_id,
    )
    for run_key, run_spans in method_runs.items():
        if run_key not in agent_methods:
            agent_methods[run_key] = run_spans
    # Backward compat: also check old sidecar name
    agent_openai = _load_sidecar(doc.id, "agent.openai", session_id)
    llm_metric_raw = _load_json_sidecar(doc.id, "agent.llm.metrics", session_id)
    last_run_raw = _load_json_sidecar(doc.id, "agent.last_run", session_id)

    # Prefer canonical sidecar. Only fall back to legacy sidecar when canonical is absent.
    if agent_llm is not None:
        llm_spans = agent_llm
    elif agent_openai is not None:
        llm_spans = agent_openai
    else:
        llm_spans = []
    llm_confidence: LLMConfidenceMetric | None = None
    if isinstance(llm_metric_raw, dict):
        try:
            llm_confidence = LLMConfidenceMetric.model_validate(llm_metric_raw)
        except Exception:
            llm_confidence = None
    label_profile: Literal["simple", "advanced"] | None = None
    latest_run_key: str | None = None
    latest_run_mode: str | None = None
    if isinstance(last_run_raw, dict):
        raw_label_profile = str(last_run_raw.get("label_profile", "")).strip().lower()
        if raw_label_profile in {"simple", "advanced"}:
            label_profile = raw_label_profile  # type: ignore[assignment]
        raw_run_key = str(last_run_raw.get("run_key", "")).strip()
        latest_run_key = raw_run_key or None
        raw_mode = str(last_run_raw.get("mode", "")).strip().lower()
        latest_run_mode = raw_mode or None

    latest_run_metadata: SavedRunMetadata | None = None
    if latest_run_mode == "llm" and latest_run_key:
        latest_run_metadata = agent_llm_run_metadata.get(latest_run_key)
    elif latest_run_mode == "method" and latest_run_key:
        latest_run_metadata = method_run_metadata.get(latest_run_key)

    if latest_run_metadata is not None and latest_run_metadata.llm_confidence is not None:
        llm_confidence = latest_run_metadata.llm_confidence
    chunk_diagnostics = (
        list(latest_run_metadata.chunk_diagnostics)
        if latest_run_metadata is not None
        else []
    )
    raw_hypothesis_spans = (
        list(latest_run_metadata.raw_hypothesis_spans)
        if latest_run_metadata is not None
        else []
    )
    resolution_events = (
        list(latest_run_metadata.resolution_events)
        if latest_run_metadata is not None
        else []
    )
    resolution_policy_version = (
        latest_run_metadata.resolution_policy_version
        if latest_run_metadata is not None
        else None
    )

    doc.manual_annotations = manual or []
    doc.agent_outputs = AgentOutputs(
        rule=agent_rule or [],
        llm=llm_spans,
        llm_runs=agent_llm_runs,
        llm_run_metadata=agent_llm_run_metadata,
        methods=agent_methods,
        method_run_metadata=method_run_metadata,
    )
    # "Agent (combined)" must remain deterministic and limited to agent-native outputs.
    # Method outputs are exposed separately under agent_outputs.methods.
    doc.agent_annotations = doc.agent_outputs.rule + doc.agent_outputs.llm
    doc.agent_run_warnings = []
    doc.agent_run_metrics = AgentRunMetrics(
        llm_confidence=llm_confidence,
        label_profile=label_profile,
        chunk_diagnostics=chunk_diagnostics,
        raw_hypothesis_spans=raw_hypothesis_spans,
        resolution_events=resolution_events,
        resolution_policy_version=resolution_policy_version,
    )

    if manual:
        doc.status = "in_progress"

    return doc


def _dedup_spans(spans: list[CanonicalSpan]) -> list[CanonicalSpan]:
    seen: set[tuple[int, int, str]] = set()
    result: list[CanonicalSpan] = []
    for span in sorted(spans, key=lambda s: (s.start, s.end, s.label)):
        key = (span.start, span.end, span.label)
        if key in seen:
            continue
        seen.add(key)
        result.append(span)
    return result


def _normalize_and_validate_spans(
    spans: list[CanonicalSpan], raw_text: str
) -> list[CanonicalSpan]:
    normalized: list[CanonicalSpan] = []
    n = len(raw_text)
    for span in spans:
        if span.start < 0 or span.end > n or span.start >= span.end:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid span offsets [{span.start}:{span.end}] for transcript length {n}."
                ),
            )
        expected = raw_text[span.start : span.end]
        normalized.append(
            CanonicalSpan(
                start=span.start,
                end=span.end,
                label=span.label.upper(),
                text=expected,
            )
        )
    return _dedup_spans(normalized)


def _spans_from_source(
    doc: CanonicalDocument,
    source: str,
) -> list[CanonicalSpan]:
    if source == "pre":
        return doc.pre_annotations
    if source == "manual":
        return doc.manual_annotations
    if source == "agent":
        return doc.agent_annotations
    if source == "agent.rule":
        return doc.agent_outputs.rule
    if source == "agent.llm":
        return doc.agent_outputs.llm
    if source.startswith("agent.llm_run."):
        run_key = source[len("agent.llm_run.") :]
        if not run_key:
            raise HTTPException(status_code=400, detail=f"Unknown source: {source}")
        return doc.agent_outputs.llm_runs.get(run_key, [])
    if source.startswith("agent.method."):
        method_id = source[len("agent.method.") :]
        if not method_id:
            raise HTTPException(status_code=400, detail=f"Unknown source: {source}")
        return doc.agent_outputs.methods.get(method_id, [])
    raise HTTPException(status_code=400, detail=f"Unknown source: {source}")


def _spans_to_pii_occurrences(spans: list[CanonicalSpan]) -> list[dict[str, object]]:
    return [
        {
            "start": span.start,
            "end": span.end,
            "text": span.text,
            "pii_type": span.label,
        }
        for span in spans
    ]


def _import_ground_truth_payload(
    *,
    payload: dict[str, object],
    raw: bytes,
    upload_filename: str,
    session_id: str,
    ids: list[str],
    existing_ids: set[str],
    conflict_policy: ImportConflictPolicy,
) -> ImportedDocumentCommitResult:
    doc_id_raw = payload.get("id")
    doc_id = (
        doc_id_raw.strip()
        if isinstance(doc_id_raw, str) and doc_id_raw.strip()
        else str(uuid.uuid4())[:8]
    )
    filename_raw = payload.get("filename")
    filename = (
        filename_raw.strip()
        if isinstance(filename_raw, str) and filename_raw.strip()
        else upload_filename
    )

    docs = parse_file(raw, filename, doc_id)
    if len(docs) != 1:
        raise ValueError("expected exactly one document")

    parsed_doc = docs[0]
    spans_raw = payload.get("spans")
    manual_spans = (
        _normalize_optional_spans(spans_raw, parsed_doc.raw_text)
        if spans_raw is not None
        else parsed_doc.pre_annotations
    )

    imported_doc = parsed_doc.model_copy(update={"pre_annotations": []})
    matched_doc_id = _find_existing_import_match(doc=imported_doc, session_id=session_id, ids=ids)
    if matched_doc_id is not None:
        existing_ids.add(matched_doc_id)
        if conflict_policy == "keep_current":
            if matched_doc_id not in ids:
                ids.append(matched_doc_id)
            return ImportedDocumentCommitResult(
                doc_id=matched_doc_id,
                created=False,
                conflict_action="keep_current",
            )
        if conflict_policy == "replace":
            if matched_doc_id not in ids:
                ids.append(matched_doc_id)
            if manual_spans:
                _save_sidecar(matched_doc_id, "manual", manual_spans, session_id)
            else:
                _delete_sidecar(matched_doc_id, "manual", session_id)
            return ImportedDocumentCommitResult(
                doc_id=matched_doc_id,
                created=False,
                conflict_action="replace",
            )

    new_id = _create_imported_document(
        doc=imported_doc,
        session_id=session_id,
        ids=ids,
        existing_ids=existing_ids,
        manual_spans=manual_spans,
    )
    return ImportedDocumentCommitResult(
        doc_id=new_id,
        created=True,
        conflict_action="add_new" if matched_doc_id is not None else None,
    )


def _looks_like_ground_truth_payload(payload: dict[str, object]) -> bool:
    if "format" in payload:
        return False
    return (
        isinstance(payload.get("id"), str)
        and isinstance(payload.get("filename"), str)
        and isinstance(payload.get("transcript"), str)
        and ("spans" in payload or "ground_truth_source" in payload)
    )


def _import_ground_truth_archive(
    raw: bytes,
    session_id: str = "default",
    *,
    conflict_policy: ImportConflictPolicy = "replace",
) -> dict[str, object]:
    archive_buffer = io.BytesIO(raw)
    try:
        with zipfile.ZipFile(archive_buffer, mode="r") as archive:
            ids = _load_session_index(session_id)
            existing_ids = set(ids)
            imported_ids: list[str] = []
            created_ids: list[str] = []
            skipped: list[dict[str, str | int]] = []
            conflict_counts = _empty_import_conflict_counts()

            members = [name for name in archive.namelist() if not name.endswith("/")]
            if not members:
                raise HTTPException(status_code=400, detail="Ground-truth archive is empty")

            for idx, member_name in enumerate(members):
                if not member_name.lower().endswith(".json"):
                    skipped.append(
                        {"index": idx, "reason": f"Unsupported archive member: {member_name}"}
                    )
                    continue

                try:
                    member_raw = archive.read(member_name)
                except Exception as exc:
                    skipped.append({"index": idx, "reason": f"Could not read {member_name}: {exc}"})
                    continue

                try:
                    member_payload = json.loads(member_raw.decode("utf-8"))
                except Exception as exc:
                    skipped.append(
                        {"index": idx, "reason": f"{member_name} must be valid UTF-8 JSON: {exc}"}
                    )
                    continue

                if not isinstance(member_payload, dict):
                    skipped.append({"index": idx, "reason": f"{member_name} is not a JSON object"})
                    continue

                try:
                    commit_result = _import_ground_truth_payload(
                        payload=member_payload,
                        raw=member_raw,
                        upload_filename=Path(member_name).name,
                        session_id=session_id,
                        ids=ids,
                        existing_ids=existing_ids,
                        conflict_policy=conflict_policy,
                    )
                except ValueError as exc:
                    skipped.append({"index": idx, "reason": f"{member_name}: {exc}"})
                    continue
                except Exception as exc:
                    skipped.append(
                        {"index": idx, "reason": f"{member_name}: {exc}"}
                    )
                    continue
                imported_ids.append(commit_result.doc_id)
                if commit_result.created:
                    created_ids.append(commit_result.doc_id)
                _record_import_conflict(conflict_counts, commit_result)
    except zipfile.BadZipFile as exc:
        raise HTTPException(
            status_code=400,
            detail="Import file must be a valid ground-truth ZIP archive",
        ) from exc

    _save_session_index(session_id)

    return {
        "bundle_version": None,
        "imported_count": len(imported_ids),
        "imported_ids": imported_ids,
        "created_count": len(created_ids),
        "created_ids": created_ids,
        "conflict_policy": conflict_policy,
        "conflict_count": sum(conflict_counts.values()),
        "replaced_count": conflict_counts["replace"],
        "kept_current_count": conflict_counts["keep_current"],
        "added_as_new_count": conflict_counts["add_new"],
        "skipped_count": len(skipped),
        "skipped": skipped,
        "warnings": [],
        "imported_prompt_lab_runs": 0,
        "imported_methods_lab_runs": 0,
        "total_in_bundle": len(members),
    }


def _parse_iso_datetime(value: str | None) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(value)
    except Exception:
        return datetime.min.replace(tzinfo=timezone.utc)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _resolve_source_llm_confidence(
    doc: CanonicalDocument,
    source: str,
) -> LLMConfidenceMetric | None:
    if source == "agent.llm":
        return doc.agent_run_metrics.llm_confidence

    if source.startswith("agent.llm_run."):
        run_key = source[len("agent.llm_run.") :]
        if not run_key:
            return None
        run_meta = doc.agent_outputs.llm_run_metadata.get(run_key)
        if run_meta is not None and run_meta.llm_confidence is not None:
            return run_meta.llm_confidence

        # Backward compatibility for sessions that predate per-run confidence metadata.
        latest = doc.agent_run_metrics.llm_confidence
        if latest is not None and latest.model == run_key:
            return latest
        return None

    if source.startswith("agent.method."):
        method_key = source[len("agent.method.") :]
        if not method_key:
            return None

        run_meta = doc.agent_outputs.method_run_metadata.get(method_key)
        if run_meta is not None and run_meta.llm_confidence is not None:
            return run_meta.llm_confidence

        # For a base method source, use the latest run for that method if available.
        if "::" not in method_key:
            latest_metric: LLMConfidenceMetric | None = None
            latest_updated_at = datetime.min.replace(tzinfo=timezone.utc)
            for metadata in doc.agent_outputs.method_run_metadata.values():
                if metadata.method_id != method_key or metadata.llm_confidence is None:
                    continue
                updated_at = _parse_iso_datetime(metadata.updated_at)
                if updated_at >= latest_updated_at:
                    latest_updated_at = updated_at
                    latest_metric = metadata.llm_confidence
            return latest_metric

    return None


def _resolve_metrics_llm_confidence(
    doc: CanonicalDocument,
    reference: str,
    hypothesis: str,
) -> LLMConfidenceMetric | None:
    hypothesis_metric = _resolve_source_llm_confidence(doc, hypothesis)
    if hypothesis_metric is not None:
        return hypothesis_metric
    return _resolve_source_llm_confidence(doc, reference)


def _prf_from_counts(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _normalize_label_profile(raw: object) -> Literal["simple", "advanced"]:
    value = str(raw or "simple").strip().lower()
    if value not in {"simple", "advanced"}:
        raise HTTPException(
            status_code=400,
            detail="label_profile must be one of: simple, advanced",
        )
    return value  # type: ignore[return-value]


def _normalize_label_projection(raw: object) -> Literal["native", "coarse_simple"]:
    value = str(raw or "native").strip().lower()
    if value not in {"native", "coarse_simple"}:
        raise HTTPException(
            status_code=400,
            detail="label_projection must be one of: native, coarse_simple",
        )
    return value  # type: ignore[return-value]


def _normalize_match_mode(raw: object) -> Literal["exact", "boundary", "overlap", "substring"]:
    value = str(raw or "overlap").strip().lower()
    if value not in {"exact", "boundary", "overlap", "substring"}:
        raise HTTPException(
            status_code=400,
            detail="match_mode must be one of: exact, boundary, overlap, substring",
        )
    return cast(Literal["exact", "boundary", "overlap", "substring"], value)


def _project_spans_to_coarse_simple(spans: list[CanonicalSpan]) -> list[CanonicalSpan]:
    projected: list[CanonicalSpan] = []
    for span in spans:
        mapped = COARSE_SIMPLE_LABEL_MAP.get(span.label.upper(), "MISC_ID")
        projected.append(
            CanonicalSpan(
                start=span.start,
                end=span.end,
                label=mapped,
                text=span.text,
            )
        )
    return projected


def _apply_label_projection(
    reference_spans: list[CanonicalSpan],
    hypothesis_spans: list[CanonicalSpan],
    *,
    label_projection: Literal["native", "coarse_simple"],
) -> tuple[list[CanonicalSpan], list[CanonicalSpan]]:
    if label_projection == "coarse_simple":
        return (
            _project_spans_to_coarse_simple(reference_spans),
            _project_spans_to_coarse_simple(hypothesis_spans),
        )
    return reference_spans, hypothesis_spans


def _get_prompt_lab_allowed_preset_methods(*, method_bundle: str) -> set[str]:
    return PROMPT_LAB_ALLOWED_PRESET_METHODS | {
        str(method["id"])
        for method in list_agent_methods(
            method_bundle=_normalize_method_bundle(method_bundle)
        )
    }


def _build_deidentify_v2_label_mapping(
    pred_types: set[str],
    gold_types: set[str],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for pred_type in pred_types:
        if pred_type in gold_types:
            mapping[pred_type] = pred_type
            continue
        mapping[pred_type] = DEIDENTIFY_V2_LABEL_ALIASES.get(pred_type, pred_type)
    return mapping


def _map_span_labels(
    spans: list[CanonicalSpan],
    mapping: dict[str, str],
) -> list[CanonicalSpan]:
    return [
        CanonicalSpan(
            start=span.start,
            end=span.end,
            label=mapping.get(span.label, span.label),
            text=span.text,
        )
        for span in spans
    ]


def _prepare_experiment_scoring_spans(
    reference_spans: list[CanonicalSpan],
    hypothesis_spans: list[CanonicalSpan],
    *,
    label_projection: Literal["native", "coarse_simple"],
    method_bundle: str,
    reference_label_set: set[str] | None = None,
) -> tuple[list[CanonicalSpan], list[CanonicalSpan]]:
    scored_hypothesis = hypothesis_spans
    if _normalize_method_bundle(method_bundle) == "deidentify-v2":
        scored_hypothesis = _map_span_labels(
            hypothesis_spans,
            _build_deidentify_v2_label_mapping(
                {span.label for span in hypothesis_spans},
                reference_label_set
                if reference_label_set is not None
                else {span.label for span in reference_spans},
            ),
        )
    return _apply_label_projection(
        reference_spans,
        scored_hypothesis,
        label_projection=label_projection,
    )


def _normalize_metrics_mode_for_method_bundle(
    match_mode: str,
    *,
    method_bundle: str,
) -> str:
    normalized_bundle = _normalize_method_bundle(method_bundle)
    if normalized_bundle == "deidentify-v2" and match_mode == "substring":
        return "legacy_substring"
    return match_mode


def _normalize_chunk_mode(raw: object) -> Literal["auto", "off", "force"]:
    value = str(raw or "off").strip().lower()
    if value not in {"auto", "off", "force"}:
        raise HTTPException(status_code=400, detail="chunk_mode must be one of: auto, off, force")
    return value  # type: ignore[return-value]


def _normalize_chunk_size(raw: object) -> int:
    if raw is None:
        return DEFAULT_CHUNK_SIZE_CHARS
    try:
        value = int(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="chunk_size_chars must be an integer") from exc
    if value < MIN_CHUNK_SIZE_CHARS or value > MAX_CHUNK_SIZE_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"chunk_size_chars must be between {MIN_CHUNK_SIZE_CHARS} and {MAX_CHUNK_SIZE_CHARS}",
        )
    return value


def _build_text_chunks(doc: CanonicalDocument, chunk_size: int) -> list[tuple[int, int]]:
    if not doc.raw_text:
        return [(0, 0)]

    utterances = sorted(
        [u for u in doc.utterances if 0 <= u.global_start < u.global_end <= len(doc.raw_text)],
        key=lambda u: (u.global_start, u.global_end),
    )
    if utterances:
        chunks: list[tuple[int, int]] = []
        start = utterances[0].global_start
        end = utterances[0].global_end
        for utt in utterances[1:]:
            if end - start >= chunk_size or utt.global_end - start > chunk_size:
                chunks.append((start, end))
                start = utt.global_start
                end = utt.global_end
                continue
            end = max(end, utt.global_end)
        chunks.append((start, end))
        return chunks

    chunks = []
    n = len(doc.raw_text)
    overlap = min(FALLBACK_CHUNK_OVERLAP_CHARS, max(chunk_size // 5, 0))
    step = max(chunk_size - overlap, 1)
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        chunks.append((start, end))
        if end >= n:
            break
        start += step
    return chunks


@dataclass(frozen=True)
class _DeidentifyV2LegacyMessage:
    id: int
    text: str
    prefix: str
    global_start: int
    global_end: int


@dataclass(frozen=True)
class _DeidentifyV2LegacyChunkEntry:
    message_id: int
    offset: int
    prefix_len: int
    text_len: int


def _build_deidentify_v2_legacy_messages(
    doc: CanonicalDocument,
) -> list[_DeidentifyV2LegacyMessage]:
    utterances = sorted(
        [u for u in doc.utterances if 0 <= u.global_start <= u.global_end <= len(doc.raw_text)],
        key=lambda u: (u.global_start, u.global_end),
    )
    if utterances:
        return [
            _DeidentifyV2LegacyMessage(
                id=index,
                text=utterance.text,
                prefix=f"[MSG-{index + 1}:{utterance.speaker}] " if utterance.speaker else "",
                global_start=utterance.global_start,
                global_end=utterance.global_end,
            )
            for index, utterance in enumerate(utterances)
        ]

    return [
        _DeidentifyV2LegacyMessage(
            id=0,
            text=doc.raw_text,
            prefix="",
            global_start=0,
            global_end=len(doc.raw_text),
        )
    ]


def _build_deidentify_v2_legacy_chunks(
    messages: list[_DeidentifyV2LegacyMessage],
) -> list[tuple[list[str], list[_DeidentifyV2LegacyChunkEntry], int, int]]:
    chunks: list[tuple[list[str], list[_DeidentifyV2LegacyChunkEntry], int, int]] = []
    parts: list[str] = []
    entries: list[_DeidentifyV2LegacyChunkEntry] = []
    size = 0
    chunk_start = 0
    chunk_end = 0

    for message in messages:
        part = f"{message.prefix}{message.text}"
        added = len(part) + (1 if parts else 0)
        if parts and size + added > DEIDENTIFY_V2_LEGACY_CHUNK_SIZE:
            chunks.append((parts, entries, chunk_start, chunk_end))
            parts, entries, size = [], [], 0

        if not parts:
            chunk_start = message.global_start
            chunk_end = message.global_end

        offset = size + (1 if parts else 0)
        entries.append(
            _DeidentifyV2LegacyChunkEntry(
                message_id=message.id,
                offset=offset,
                prefix_len=len(message.prefix),
                text_len=len(message.text),
            )
        )
        parts.append(part)
        size = offset + len(part)
        chunk_end = message.global_end

    if parts:
        chunks.append((parts, entries, chunk_start, chunk_end))

    return chunks


def _find_deidentify_v2_chunk_entry(
    entries: list[_DeidentifyV2LegacyChunkEntry],
    char_offset: int,
) -> _DeidentifyV2LegacyChunkEntry | None:
    for entry in entries:
        total = entry.prefix_len + entry.text_len
        if entry.offset <= char_offset < entry.offset + total:
            return entry
    return None


def _extract_deidentify_v2_chunk_local_spans(
    *,
    chunk_spans: list[CanonicalSpan],
    entries: list[_DeidentifyV2LegacyChunkEntry],
) -> dict[int, list[CanonicalSpan]]:
    predicted: dict[int, list[CanonicalSpan]] = {}
    for span in chunk_spans:
        entry = _find_deidentify_v2_chunk_entry(entries, span.start)
        if entry is None:
            continue
        local_start = span.start - entry.offset - entry.prefix_len
        local_end = span.end - entry.offset - entry.prefix_len
        if local_start < 0:
            continue
        local_end = min(local_end, entry.text_len)
        if local_end <= local_start:
            continue
        text = span.text[: local_end - local_start]
        predicted.setdefault(entry.message_id, []).append(
            CanonicalSpan(
                start=local_start,
                end=local_end,
                label=span.label,
                text=text,
            )
        )
    return predicted


def _propagate_deidentify_v2_entities(
    messages: list[_DeidentifyV2LegacyMessage],
    predicted: dict[int, list[CanonicalSpan]],
) -> None:
    entities = {
        (span.text, span.label)
        for message in messages
        for span in predicted.get(message.id, [])
        if span.text
    }

    for text, label in sorted(entities, key=lambda item: -len(item[0])):
        pattern = re.compile(re.escape(text))
        for message in messages:
            existing = {
                (span.start, span.end)
                for span in predicted.setdefault(message.id, [])
            }
            for match in pattern.finditer(message.text):
                start, end = match.start(), match.end()
                if any(
                    existing_start <= start < existing_end
                    or existing_start < end <= existing_end
                    for existing_start, existing_end in existing
                ):
                    continue
                predicted[message.id].append(
                    CanonicalSpan(
                        start=start,
                        end=end,
                        label=label,
                        text=text,
                    )
                )
                existing.add((start, end))


def _project_deidentify_v2_predictions_to_global(
    *,
    messages: list[_DeidentifyV2LegacyMessage],
    predicted: dict[int, list[CanonicalSpan]],
) -> list[CanonicalSpan]:
    projected: list[CanonicalSpan] = []
    for message in messages:
        for span in predicted.get(message.id, []):
            projected.append(
                CanonicalSpan(
                    start=message.global_start + span.start,
                    end=message.global_start + span.end,
                    label=span.label,
                    text=span.text,
                )
            )
    projected.sort(key=lambda span: (span.start, span.end, span.label, span.text))
    return projected


def _shift_spans(spans: list[CanonicalSpan], offset: int) -> list[CanonicalSpan]:
    shifted: list[CanonicalSpan] = []
    for span in spans:
        shifted.append(
            CanonicalSpan(
                start=span.start + offset,
                end=span.end + offset,
                label=span.label,
                text=span.text,
            )
        )
    return shifted


def _aggregate_llm_confidence(metrics: list[LLMConfidenceMetric]) -> LLMConfidenceMetric:
    if not metrics:
        return LLMConfidenceMetric(
            available=False,
            provider="unknown",
            model="unknown",
            reason="unsupported_provider",
            token_count=0,
            band="na",
        )
    if len(metrics) == 1:
        return metrics[0]

    usable = [
        item
        for item in metrics
        if item.available and item.mean_logprob is not None and item.token_count > 0
    ]
    if not usable:
        return metrics[0]

    total_tokens = sum(item.token_count for item in usable)
    if total_tokens <= 0:
        return metrics[0]

    weighted_mean_logprob = (
        sum(float(item.mean_logprob or 0.0) * item.token_count for item in usable) / total_tokens
    )
    confidence = math.exp(weighted_mean_logprob)
    perplexity = math.exp(-weighted_mean_logprob)
    if confidence >= usable[0].high_threshold:
        band: Literal["high", "medium", "low", "na"] = "high"
    elif confidence >= usable[0].medium_threshold:
        band = "medium"
    else:
        band = "low"

    return LLMConfidenceMetric(
        available=True,
        provider=usable[0].provider,
        model=usable[0].model,
        reason="ok",
        token_count=total_tokens,
        mean_logprob=weighted_mean_logprob,
        confidence=confidence,
        perplexity=perplexity,
        band=band,
        high_threshold=usable[0].high_threshold,
        medium_threshold=usable[0].medium_threshold,
    )


def _normalize_optional_spans(raw: object, raw_text: str) -> list[CanonicalSpan]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("Span list must be a JSON array")
    spans: list[CanonicalSpan] = []
    for index, item in enumerate(raw):
        try:
            spans.append(CanonicalSpan.model_validate(item))
        except Exception as exc:  # pragma: no cover - pydantic message can vary
            raise ValueError(f"Invalid span at index {index}: {exc}") from exc
    try:
        return _normalize_and_validate_spans(spans, raw_text)
    except HTTPException as exc:
        raise ValueError(str(exc.detail)) from exc


def _delete_doc_files(doc_id: str, session_id: str = "default") -> bool:
    base = _session_dir(session_id)
    targets = [
        base / f"{doc_id}.source.json",
        base / f"{doc_id}.manual.json",
        base / f"{doc_id}.agent.rule.json",
        base / f"{doc_id}.agent.llm.json",
        base / f"{doc_id}.agent.llm.metrics.json",
        base / f"{doc_id}.agent.last_run.json",
        base / f"{doc_id}.{LLM_RUNS_SIDECAR_KIND}.json",
        base / f"{doc_id}.{LLM_RUNS_METADATA_SIDECAR_KIND}.json",
        base / f"{doc_id}.{METHOD_RUNS_SIDECAR_KIND}.json",
        base / f"{doc_id}.{METHOD_RUNS_METADATA_SIDECAR_KIND}.json",
        base / f"{doc_id}.{MANUAL_RUNS_SIDECAR_KIND}.json",
        base / f"{doc_id}.{MANUAL_RUNS_METADATA_SIDECAR_KIND}.json",
        base / f"{doc_id}.agent.openai.json",
    ]
    deleted = False
    for path in targets:
        if path.exists():
            path.unlink()
            deleted = True
    for path in base.glob(f"{doc_id}.agent.method.*.json"):
        path.unlink()
        deleted = True
    return deleted


# --- Config ---


def _load_config() -> dict:
    _ensure_dirs()
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return dict(DEFAULT_CONFIG)


def _save_config(cfg: dict):
    _ensure_dirs()
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


def _normalize_experiment_concurrency_cap(
    value: object,
    *,
    default: int,
    minimum: int,
) -> int:
    candidate = default
    if isinstance(value, bool):
        candidate = default
    elif isinstance(value, int):
        candidate = value
    elif isinstance(value, float) and value.is_integer():
        candidate = int(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped:
            try:
                candidate = int(stripped)
            except ValueError:
                candidate = default
    return max(minimum, min(candidate, EXPERIMENT_HARD_MAX_CONCURRENCY))


def _get_prompt_lab_max_concurrency(cfg: dict | None = None) -> int:
    config = _load_config() if cfg is None else cfg
    return _normalize_experiment_concurrency_cap(
        config.get("prompt_lab_max_concurrency"),
        default=PROMPT_LAB_DEFAULT_MAX_CONCURRENCY,
        minimum=PROMPT_LAB_DEFAULT_CONCURRENCY,
    )


def _get_methods_lab_max_concurrency(cfg: dict | None = None) -> int:
    config = _load_config() if cfg is None else cfg
    return _normalize_experiment_concurrency_cap(
        config.get("methods_lab_max_concurrency"),
        default=METHODS_LAB_DEFAULT_MAX_CONCURRENCY,
        minimum=METHODS_LAB_DEFAULT_CONCURRENCY,
    )


def _get_experiment_limits(cfg: dict | None = None) -> dict[str, int]:
    config = _load_config() if cfg is None else cfg
    return {
        "prompt_lab_default_concurrency": PROMPT_LAB_DEFAULT_CONCURRENCY,
        "prompt_lab_max_concurrency": _get_prompt_lab_max_concurrency(config),
        "methods_lab_default_concurrency": METHODS_LAB_DEFAULT_CONCURRENCY,
        "methods_lab_max_concurrency": _get_methods_lab_max_concurrency(config),
    }


def _extract_api_base_host(api_base: object) -> str | None:
    raw = str(api_base or "").strip()
    if not raw:
        return None
    parsed = urlparse(raw)
    if parsed.netloc:
        return parsed.netloc
    return parsed.path or None


def _validate_experiment_concurrency(value: int, *, max_allowed: int):
    if value < 1 or value > max_allowed:
        raise HTTPException(
            status_code=400,
            detail=f"concurrency must be between 1 and {max_allowed}",
        )


def _resolve_experiment_worker_count(
    requested: int,
    *,
    total_tasks: int,
    max_allowed: int,
) -> int:
    bounded_total = max(1, total_tasks)
    return max(1, min(int(requested), bounded_total, max_allowed))


def _count_prompt_lab_tasks(run: dict) -> int:
    return (
        len(run.get("doc_ids", []))
        * len(run.get("models", []))
        * len(run.get("prompts", []))
    )


def _count_methods_lab_tasks(run: dict) -> int:
    doc_count = len(run.get("doc_ids", []))
    model_count = len(run.get("models", []))
    total = 0
    for method in run.get("methods", []):
        if not isinstance(method, dict):
            continue
        definition = METHOD_DEFINITION_BY_ID.get(str(method.get("method_id", "")))
        if definition is None:
            continue
        total += doc_count * (model_count if bool(definition.get("uses_llm")) else 1)
    return total


def _build_experiment_run_diagnostics(
    run: dict,
    *,
    kind: Literal["prompt_lab", "methods_lab"],
) -> dict[str, object]:
    runtime_raw = run.get("runtime", {})
    runtime = runtime_raw if isinstance(runtime_raw, dict) else {}
    if kind == "prompt_lab":
        requested = int(run.get("concurrency", PROMPT_LAB_DEFAULT_CONCURRENCY))
        max_allowed = _get_prompt_lab_max_concurrency()
        total_tasks = _count_prompt_lab_tasks(run)
    else:
        requested = int(run.get("concurrency", METHODS_LAB_DEFAULT_CONCURRENCY))
        max_allowed = _get_methods_lab_max_concurrency()
        total_tasks = _count_methods_lab_tasks(run)
    effective = _resolve_experiment_worker_count(
        requested,
        total_tasks=total_tasks,
        max_allowed=max_allowed,
    )
    return {
        "requested_concurrency": requested,
        "effective_worker_count": effective,
        "max_allowed_concurrency": max_allowed,
        "total_tasks": total_tasks,
        "clamped_by_task_count": requested > max(1, total_tasks),
        "clamped_by_server_cap": requested > max_allowed,
        "api_base_host": _extract_api_base_host(runtime.get("api_base")),
    }


def _build_experiment_diagnostics_response() -> dict[str, object]:
    cfg = _load_config()
    resolved_api_base = str(cfg.get("api_base", "") or "") or _resolve_env_api_base()
    api_key = _resolve_env_api_key()
    checked_at = _now_iso()
    gateway_catalog: dict[str, object] = {
        "reachable": False,
        "model_count": None,
        "error": None,
        "checked_at": checked_at,
    }
    if not resolved_api_base:
        gateway_catalog["error"] = "No api_base configured."
    elif not api_key:
        gateway_catalog["error"] = "No API key available for gateway catalog check."
    else:
        try:
            models = _fetch_gateway_model_ids(resolved_api_base, api_key)
            gateway_catalog["reachable"] = True
            gateway_catalog["model_count"] = len(models)
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else json.dumps(exc.detail)
            gateway_catalog["error"] = detail
        except Exception as exc:  # pragma: no cover - defensive fallback
            gateway_catalog["error"] = str(exc) or exc.__class__.__name__
    return {
        "resolved_api_base": resolved_api_base or None,
        "api_base_host": _extract_api_base_host(resolved_api_base),
        "prompt_lab_max_concurrency": _get_prompt_lab_max_concurrency(cfg),
        "methods_lab_max_concurrency": _get_methods_lab_max_concurrency(cfg),
        "gateway_catalog": gateway_catalog,
    }


def _resolve_bundle_version(payload: dict) -> int:
    bundle_format = payload.get("format")
    raw_version = payload.get("version")

    if bundle_format == "annotation_tool_session_v1":
        return 1
    if bundle_format == "annotation_tool_session_v2":
        return 2
    if bundle_format == "annotation_tool_session_v3":
        return 3
    if bundle_format == BUNDLE_FORMAT:
        if raw_version is None:
            return BUNDLE_VERSION
        if isinstance(raw_version, int):
            return raw_version
        raise HTTPException(status_code=400, detail="Bundle version must be an integer")
    raise HTTPException(
        status_code=400,
        detail=(
            "Unsupported bundle format. Expected one of: "
            "'annotation_tool_session', 'annotation_tool_session_v1', "
            "'annotation_tool_session_v2', 'annotation_tool_session_v3'."
        ),
    )


def _normalize_gateway_base(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def _fetch_gateway_model_ids(api_base: str, api_key: str) -> list[str]:
    base = _normalize_gateway_base(api_base)
    url = f"{base}/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "x-litellm-api-key": api_key,
    }
    try:
        resp = httpx.get(url, headers=headers, timeout=15.0)
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to query gateway models endpoint ({url}): {exc}",
        ) from exc
    if resp.status_code != 200:
        snippet = resp.text[:300]
        raise HTTPException(
            status_code=502,
            detail=f"Gateway model list request failed ({resp.status_code}) at {url}: {snippet}",
        )
    try:
        payload = resp.json()
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Gateway returned non-JSON response for model list at {url}.",
        ) from exc
    data = payload.get("data", [])
    if not isinstance(data, list):
        raise HTTPException(
            status_code=502,
            detail=f"Gateway model list format unexpected at {url}.",
        )
    model_ids: list[str] = []
    for item in data:
        if isinstance(item, dict):
            value = item.get("id")
            if isinstance(value, str) and value:
                model_ids.append(value)
    return model_ids


# Credential status helpers (no secret exposure)
ENV_API_KEY_VARS = [
    "LITELLM_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
]
ENV_API_BASE_VARS = ["LITELLM_BASE_URL"]


def _load_repo_env_file(env_path: Path | None = None) -> dict[str, str]:
    resolved_env_path = ROOT_ENV_PATH if env_path is None else env_path
    if not resolved_env_path.is_file():
        return {}
    loaded: dict[str, str] = {}
    for key, value in dotenv_values(resolved_env_path).items():
        if not key or value is None or key in os.environ:
            continue
        loaded[key] = value
    return loaded


def _present_env_vars(names: list[str]) -> list[str]:
    repo_env = _load_repo_env_file()
    return [name for name in names if bool(os.environ.get(name) or repo_env.get(name))]


def _resolve_env_api_key() -> str:
    return (
        os.environ.get("LITELLM_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
        or os.environ.get("ANTHROPIC_API_KEY", "")
        or os.environ.get("GEMINI_API_KEY", "")
        or os.environ.get("GOOGLE_API_KEY", "")
    )


def _resolve_env_api_base() -> str:
    return os.environ.get("LITELLM_BASE_URL", "")


def _resolve_llm_runtime_config(body: "AgentRunBody") -> dict[str, object]:
    cfg = _load_config()
    api_key = (
        body.api_key
        or _resolve_env_api_key()
    )
    api_base = (
        body.api_base
        or str(cfg.get("api_base", "") or "")
        or _resolve_env_api_base()
    )
    model = body.model or cfg.get("model", DEFAULT_CONFIG["model"])
    requested_system_prompt = body.system_prompt or cfg.get(
        "system_prompt", DEFAULT_CONFIG["system_prompt"]
    )
    if (
        isinstance(requested_system_prompt, str)
        and "confidence" in requested_system_prompt
        and '"start"' not in requested_system_prompt
    ):
        requested_system_prompt = SYSTEM_PROMPT

    temperature = (
        body.temperature
        if body.temperature is not None
        else float(cfg.get("temperature", 0.0))
    )
    reasoning_effort = body.reasoning_effort or cfg.get("reasoning_effort", "none")
    anthropic_thinking = (
        body.anthropic_thinking
        if body.anthropic_thinking is not None
        else bool(cfg.get("anthropic_thinking", False))
    )
    anthropic_thinking_budget_tokens = (
        body.anthropic_thinking_budget_tokens
        if body.anthropic_thinking_budget_tokens is not None
        else cfg.get("anthropic_thinking_budget_tokens")
    )
    chunk_mode = _normalize_chunk_mode(body.chunk_mode or "off")
    chunk_size_chars = _normalize_chunk_size(body.chunk_size_chars)
    label_profile = _normalize_label_profile(body.label_profile or "simple")
    return {
        "api_key": api_key,
        "api_base": api_base,
        "model": model,
        "requested_system_prompt": requested_system_prompt,
        "temperature": temperature,
        "reasoning_effort": reasoning_effort,
        "anthropic_thinking": anthropic_thinking,
        "anthropic_thinking_budget_tokens": anthropic_thinking_budget_tokens,
        "chunk_mode": chunk_mode,
        "chunk_size_chars": chunk_size_chars,
        "label_profile": label_profile,
    }


def _build_llm_prompt_snapshot(
    requested_system_prompt: str,
    *,
    label_profile: Literal["simple", "advanced"],
    method_bundle: str = "audited",
) -> dict[str, Any]:
    requested = str(requested_system_prompt or "")
    return {
        "requested_system_prompt": requested,
        "format_guardrail_appended": True,
        "effective_system_prompt": build_extraction_system_prompt(
            requested,
            label_profile=label_profile,
            method_bundle=_normalize_method_bundle(method_bundle),
        ),
    }


def _build_method_prompt_snapshot(
    *,
    method_id: str,
    additional_constraints: str,
    method_verify: bool | None,
    label_profile: Literal["simple", "advanced"],
    method_bundle: str = "audited",
) -> dict[str, Any]:
    normalized_method_bundle = _normalize_method_bundle(method_bundle)
    method_definition = get_method_definition_by_id(
        method_id,
        method_bundle=normalized_method_bundle,
    )
    if method_definition is None:
        return {
            "method_id": method_id,
            "additional_constraints": additional_constraints,
            "passes": [],
        }

    constraints = str(additional_constraints or "").strip()
    verify_enabled = (
        bool(method_definition.get("default_verify", False))
        if method_verify is None
        else bool(method_verify)
    )
    passes: list[dict[str, Any]] = []
    for idx, method_pass in enumerate(method_definition.get("passes", []), start=1):
        pass_kind = str(method_pass.get("kind", ""))
        if pass_kind != "llm":
            continue
        base_prompt = str(method_pass.get("prompt") or SYSTEM_PROMPT)
        resolved_prompt = (
            f"{base_prompt}\n\nAdditional constraints:\n{constraints}"
            if constraints
            else base_prompt
        )
        passes.append(
            {
                "pass_index": idx,
                "entity_types": method_pass.get("entity_types"),
                "base_system_prompt": base_prompt,
                "resolved_system_prompt": resolved_prompt,
                "effective_system_prompt": build_extraction_system_prompt(
                    resolved_prompt,
                    label_profile=label_profile,
                    method_bundle=normalized_method_bundle,
                ),
            }
        )

    return {
        "method_id": method_id,
        "additional_constraints": constraints,
        "verify_enabled": verify_enabled,
        "passes": passes,
    }


def _validate_gateway_model_access(*, model: str, api_base: str, api_key: str):
    if "." in model and "/" not in model and not api_base:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{model}' is a gateway model ID. Set api_base (or LITELLM_BASE_URL) "
                "to use gateway-routed models."
            ),
        )
    if api_base:
        gateway_models = _fetch_gateway_model_ids(api_base, api_key)
        if model not in gateway_models:
            preview = ", ".join(gateway_models[:15])
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Model '{model}' is not available for this API key on {api_base}. "
                    f"Pick a model from /v1/models. Examples: {preview}"
                ),
            )


def _should_use_chunking(text_len: int, chunk_mode: str, chunk_size: int) -> bool:
    mode = _normalize_chunk_mode(chunk_mode)
    if mode == "off":
        return False
    if mode == "force":
        return True
    return text_len > chunk_size


def _estimate_chunk_total(
    doc: CanonicalDocument,
    *,
    chunk_mode: str,
    chunk_size_chars: int,
) -> int:
    if not _should_use_chunking(len(doc.raw_text), chunk_mode, chunk_size_chars):
        return 1
    return max(1, len(_build_text_chunks(doc, chunk_size_chars)))


def _resolve_chunk_parallel_workers(chunk_count: int) -> int:
    if chunk_count <= 1:
        return 1
    raw_value = os.environ.get("ANNOTATION_CHUNK_WORKERS", str(DEFAULT_CHUNK_PARALLEL_WORKERS))
    try:
        requested = int(raw_value)
    except Exception:
        requested = DEFAULT_CHUNK_PARALLEL_WORKERS
    bounded = max(1, min(requested, MAX_CHUNK_PARALLEL_WORKERS))
    return min(chunk_count, bounded)


@dataclass
class _ChunkExecutionResult:
    idx: int
    start: int
    end: int
    shifted_spans: list[CanonicalSpan]
    raw_shifted_spans: list[CanonicalSpan]
    warnings: list[str]
    response_debug: list[str]
    llm_confidence: LLMConfidenceMetric | None
    finish_reason: str | None = None


def _format_chunk_warnings(
    idx: int,
    chunk_count: int,
    warnings: list[str],
) -> list[str]:
    return [f"Chunk {idx + 1}/{chunk_count}: {item}" for item in warnings]


def _format_chunk_response_debug(
    idx: int,
    chunk_count: int,
    entries: list[str],
) -> list[str]:
    return [f"Chunk {idx + 1}/{chunk_count}: {item}" for item in entries]


def _build_chunk_diagnostic(
    *,
    result: _ChunkExecutionResult,
    attempt_count: int = 1,
    retry_used: bool = False,
    suspicious_empty: bool = False,
    status: Literal["completed", "failed"] = "completed",
    warnings: list[str] | None = None,
) -> AgentChunkDiagnostic:
    chunk_warnings = list(result.warnings if warnings is None else warnings)
    return AgentChunkDiagnostic(
        chunk_index=result.idx,
        start=result.start,
        end=result.end,
        char_count=max(0, result.end - result.start),
        span_count=len(result.shifted_spans),
        attempt_count=attempt_count,
        retry_used=retry_used,
        suspicious_empty=suspicious_empty,
        status=status,
        finish_reason=result.finish_reason,
        warnings=chunk_warnings,
    )


def _run_deidentify_v2_method_for_document(
    *,
    doc: CanonicalDocument,
    method_id: str,
    api_key: str | None,
    api_base: str | None,
    model: str,
    system_prompt: str,
    temperature: float,
    reasoning_effort: str,
    anthropic_thinking: bool,
    anthropic_thinking_budget_tokens: int | None,
    method_verify: bool | None,
    timeout_seconds: float | None,
    progress_callback: Callable[[int, int], None] | None = None,
    runtime_progress_callback: Callable[[dict[str, object]], None] | None = None,
) -> tuple[
    list[CanonicalSpan],
    list[str],
    LLMConfidenceMetric | None,
    list[AgentChunkDiagnostic],
    list[CanonicalSpan],
    list[ResolutionEvent],
    str | None,
    list[str],
]:
    def _emit_runtime_progress(
        *,
        chunk_index: int,
        total_chunks: int,
        start: int,
        end: int,
        pass_index: int | None = None,
        pass_label: str | None = None,
    ) -> None:
        if runtime_progress_callback is None:
            return
        runtime_progress_callback(
            {
                "current_chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "chunk_start": start,
                "chunk_end": end,
                "current_pass_index": pass_index,
                "current_pass_label": pass_label,
            }
        )

    messages = _build_deidentify_v2_legacy_messages(doc)
    chunks = _build_deidentify_v2_legacy_chunks(messages)
    message_by_id = {message.id: message for message in messages}
    total_chunks = max(len(chunks), 1)
    predicted: dict[int, list[CanonicalSpan]] = {message.id: [] for message in messages}
    raw_predicted: dict[int, list[CanonicalSpan]] = {message.id: [] for message in messages}
    warnings: list[str] = []
    response_debug: list[str] = []
    llm_metrics: list[LLMConfidenceMetric] = []
    chunk_diagnostics: list[AgentChunkDiagnostic] = []

    for idx, (parts, entries, chunk_start, chunk_end) in enumerate(chunks):
        chunk_text = "\n".join(parts)
        _emit_runtime_progress(
            chunk_index=idx + 1,
            total_chunks=total_chunks,
            start=chunk_start,
            end=chunk_end,
        )
        method_result = run_method_with_metadata(
            text=chunk_text,
            method_id=method_id,
            api_key=api_key,
            api_base=api_base,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            reasoning_effort=reasoning_effort,  # type: ignore[arg-type]
            anthropic_thinking=anthropic_thinking,
            anthropic_thinking_budget_tokens=anthropic_thinking_budget_tokens,  # type: ignore[arg-type]
            method_verify=method_verify,
            timeout_seconds=timeout_seconds,
            method_bundle="deidentify-v2",
            progress_callback=(
                lambda pass_index, pass_label: _emit_runtime_progress(
                    chunk_index=idx + 1,
                    total_chunks=total_chunks,
                    start=chunk_start,
                    end=chunk_end,
                    pass_index=pass_index,
                    pass_label=pass_label,
                )
            ),
        )
        local_spans = _extract_deidentify_v2_chunk_local_spans(
            chunk_spans=list(method_result.spans),
            entries=entries,
        )
        local_raw_spans = _extract_deidentify_v2_chunk_local_spans(
            chunk_spans=list(getattr(method_result, "raw_spans", method_result.spans)),
            entries=entries,
        )
        for message_id, spans in local_spans.items():
            predicted.setdefault(message_id, []).extend(spans)
        for message_id, spans in local_raw_spans.items():
            raw_predicted.setdefault(message_id, []).extend(spans)

        projected_chunk_spans = _project_deidentify_v2_predictions_to_global(
            messages=[message_by_id[entry.message_id] for entry in entries],
            predicted=local_spans,
        )
        if method_result.warnings:
            warnings.extend(_format_chunk_warnings(idx, total_chunks, list(method_result.warnings)))
        if getattr(method_result, "response_debug", None):
            response_debug.extend(
                _format_chunk_response_debug(
                    idx,
                    total_chunks,
                    list(getattr(method_result, "response_debug", [])),
                )
            )
        if getattr(method_result, "llm_confidence", None) is not None:
            llm_metrics.append(method_result.llm_confidence)
        chunk_diagnostics.append(
            AgentChunkDiagnostic(
                chunk_index=idx,
                start=chunk_start,
                end=chunk_end,
                char_count=len(chunk_text),
                span_count=len(projected_chunk_spans),
                status="completed",
                finish_reason=None,
                warnings=list(method_result.warnings),
            )
        )
        if progress_callback is not None:
            progress_callback(idx + 1, total_chunks)

    _propagate_deidentify_v2_entities(messages, predicted)
    _propagate_deidentify_v2_entities(messages, raw_predicted)

    spans = _normalize_and_validate_spans(
        _project_deidentify_v2_predictions_to_global(messages=messages, predicted=predicted),
        doc.raw_text,
    )
    raw_spans = _normalize_and_validate_spans(
        _project_deidentify_v2_predictions_to_global(messages=messages, predicted=raw_predicted),
        doc.raw_text,
    )
    aggregated_confidence = (
        _aggregate_llm_confidence(llm_metrics) if llm_metrics else None
    )
    return (
        spans,
        warnings,
        aggregated_confidence,
        chunk_diagnostics,
        raw_spans,
        [],
        None,
        response_debug,
    )


def _summarize_suspicious_empty_retry(
    *,
    recovered_span_count: int,
) -> str:
    if recovered_span_count > 0:
        return (
            f"suspicious empty first pass returned 0 spans; retry recovered "
            f"{recovered_span_count} span(s)."
        )
    return "suspicious empty first pass returned 0 spans; retry also returned 0 spans."


def _run_llm_for_document(
    *,
    doc: CanonicalDocument,
    api_key: str,
    api_base: str | None,
    model: str,
    system_prompt: str,
    temperature: float,
    reasoning_effort: str,
    anthropic_thinking: bool,
    anthropic_thinking_budget_tokens: int | None,
    label_profile: Literal["simple", "advanced"],
    chunk_mode: str,
    chunk_size_chars: int,
    enable_suspicious_empty_retry: bool = True,
    progress_callback: Callable[[int, int], None] | None = None,
    method_bundle: str = "audited",
) -> tuple[
    list[CanonicalSpan],
    list[str],
    LLMConfidenceMetric,
    list[AgentChunkDiagnostic],
    list[CanonicalSpan],
    list[ResolutionEvent],
    str | None,
    list[str],
]:
    if not _should_use_chunking(len(doc.raw_text), chunk_mode, chunk_size_chars):
        llm_result = run_llm_with_metadata(
            text=doc.raw_text,
            api_key=api_key,
            api_base=api_base,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            reasoning_effort=reasoning_effort,  # type: ignore[arg-type]
            anthropic_thinking=anthropic_thinking,
            anthropic_thinking_budget_tokens=anthropic_thinking_budget_tokens,  # type: ignore[arg-type]
            label_profile=label_profile,
            method_bundle=_normalize_method_bundle(method_bundle),
        )
        spans = _normalize_and_validate_spans(
            normalize_method_spans(llm_result.spans, label_profile=label_profile),
            doc.raw_text,
        )
        raw_spans = _normalize_and_validate_spans(
            normalize_method_spans(
                getattr(llm_result, "raw_spans", llm_result.spans),
                label_profile=label_profile,
            ),
            doc.raw_text,
        )
        if progress_callback is not None:
            progress_callback(1, 1)
        return (
            spans,
            llm_result.warnings,
            llm_result.llm_confidence,
            [],
            raw_spans,
            list(getattr(llm_result, "resolution_events", [])),
            getattr(llm_result, "resolution_policy_version", None),
            list(getattr(llm_result, "response_debug", [])),
        )

    chunks = _build_text_chunks(doc, chunk_size_chars)
    warnings: list[str] = []
    response_debug: list[str] = []
    all_raw_spans: list[CanonicalSpan] = []
    chunk_metrics: list[LLMConfidenceMetric] = []
    chunk_count = len(chunks)
    chunk_workers = _resolve_chunk_parallel_workers(chunk_count)

    def _run_chunk(idx: int, start: int, end: int) -> _ChunkExecutionResult:
        chunk_text = doc.raw_text[start:end]
        llm_result = run_llm_with_metadata(
            text=chunk_text,
            api_key=api_key,
            api_base=api_base,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            reasoning_effort=reasoning_effort,  # type: ignore[arg-type]
            anthropic_thinking=anthropic_thinking,
            anthropic_thinking_budget_tokens=anthropic_thinking_budget_tokens,  # type: ignore[arg-type]
            label_profile=label_profile,
            method_bundle=_normalize_method_bundle(method_bundle),
        )
        shifted = _shift_spans(
            normalize_method_spans(llm_result.spans, label_profile=label_profile),
            start,
        )
        raw_shifted = _shift_spans(
            normalize_method_spans(
                getattr(llm_result, "raw_spans", llm_result.spans),
                label_profile=label_profile,
            ),
            start,
        )
        return _ChunkExecutionResult(
            idx=idx,
            start=start,
            end=end,
            shifted_spans=shifted,
            raw_shifted_spans=raw_shifted,
            warnings=list(llm_result.warnings),
            response_debug=list(getattr(llm_result, "response_debug", [])),
            llm_confidence=llm_result.llm_confidence,
            finish_reason=getattr(llm_result, "finish_reason", None),
        )

    chunk_results: dict[int, _ChunkExecutionResult] = {}
    if chunk_workers == 1:
        for idx, (start, end) in enumerate(chunks):
            chunk_results[idx] = _run_chunk(idx, start, end)
            if progress_callback is not None:
                progress_callback(idx + 1, chunk_count)
    else:
        completed_count = 0
        with ThreadPoolExecutor(max_workers=chunk_workers) as pool:
            future_map = {
                pool.submit(_run_chunk, idx, start, end): idx
                for idx, (start, end) in enumerate(chunks)
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    chunk_result = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"Chunk {idx + 1}/{chunk_count} failed during parallel LLM execution: {exc}"
                    ) from exc
                chunk_results[idx] = chunk_result
                completed_count += 1
                if progress_callback is not None:
                    progress_callback(completed_count, chunk_count)

    chunk_outcomes = dict(chunk_results)
    has_non_empty_chunk = any(len(result.shifted_spans) > 0 for result in chunk_outcomes.values())
    suspicious_empty_indices = [
        idx
        for idx, result in chunk_outcomes.items()
        if has_non_empty_chunk and len(result.shifted_spans) == 0
    ]

    for idx in suspicious_empty_indices:
        if not enable_suspicious_empty_retry:
            chunk_outcomes[idx] = _ChunkExecutionResult(
                idx=chunk_outcomes[idx].idx,
                start=chunk_outcomes[idx].start,
                end=chunk_outcomes[idx].end,
                shifted_spans=chunk_outcomes[idx].shifted_spans,
                raw_shifted_spans=chunk_outcomes[idx].raw_shifted_spans,
                warnings=[
                    "suspicious empty first pass returned 0 spans; retry disabled.",
                    *chunk_outcomes[idx].warnings,
                ],
                response_debug=list(chunk_outcomes[idx].response_debug),
                llm_confidence=chunk_outcomes[idx].llm_confidence,
                finish_reason=chunk_outcomes[idx].finish_reason,
            )
            continue

        start, end = chunks[idx]
        try:
            retry_result = _run_chunk(idx, start, end)
        except Exception as exc:
            raise RuntimeError(
                f"Chunk {idx + 1}/{chunk_count} failed during suspicious-empty retry: {exc}"
            ) from exc
        recovered_span_count = len(retry_result.shifted_spans)
        selected_result = retry_result if recovered_span_count > 0 else chunk_outcomes[idx]
        chunk_outcomes[idx] = _ChunkExecutionResult(
            idx=selected_result.idx,
            start=selected_result.start,
            end=selected_result.end,
            shifted_spans=selected_result.shifted_spans,
            raw_shifted_spans=selected_result.raw_shifted_spans,
            warnings=[
                _summarize_suspicious_empty_retry(
                    recovered_span_count=recovered_span_count,
                ),
                *selected_result.warnings,
            ],
            response_debug=list(selected_result.response_debug),
            llm_confidence=selected_result.llm_confidence,
            finish_reason=selected_result.finish_reason,
        )

    chunk_diagnostics: list[AgentChunkDiagnostic] = []
    for idx in range(chunk_count):
        result = chunk_outcomes[idx]
        suspicious_empty = idx in suspicious_empty_indices
        retry_used = (
            enable_suspicious_empty_retry and suspicious_empty and len(result.shifted_spans) > 0
        )
        all_raw_spans.extend(result.raw_shifted_spans)
        warnings.extend(_format_chunk_warnings(idx, chunk_count, result.warnings))
        response_debug.extend(
            _format_chunk_response_debug(idx, chunk_count, result.response_debug)
        )
        if result.llm_confidence is not None:
            chunk_metrics.append(result.llm_confidence)
        chunk_diagnostics.append(
            _build_chunk_diagnostic(
                result=result,
                attempt_count=2 if suspicious_empty and enable_suspicious_empty_retry else 1,
                retry_used=retry_used,
                suspicious_empty=suspicious_empty,
            )
        )

    raw_spans = _normalize_and_validate_spans(
        merge_method_spans(all_raw_spans),
        doc.raw_text,
    )
    resolved_spans, resolution_events = resolve_spans(
        doc.raw_text,
        raw_spans,
        label_profile=label_profile,
        enable_augmentation=True,
    )
    normalized = _normalize_and_validate_spans(
        merge_method_spans(resolved_spans),
        doc.raw_text,
    )
    normalized, dropped_name_spans = _drop_implausible_name_spans(
        doc.raw_text,
        normalized,
        label_profile=label_profile,
    )
    if dropped_name_spans > 0:
        target_label = "NAME" if label_profile == "simple" else "PERSON"
        warnings.append(
            f"Dropped {dropped_name_spans} implausible {target_label} span(s) after shared resolution."
        )
    warnings.insert(
        0,
        (
            f"Chunked LLM run used {chunk_count} chunk(s) at ~{chunk_size_chars} chars "
            f"(mode={chunk_mode}, workers={chunk_workers})."
        ),
    )
    return (
        normalized,
        warnings,
        _aggregate_llm_confidence(chunk_metrics),
        chunk_diagnostics,
        raw_spans,
        resolution_events,
        RESOLUTION_POLICY_VERSION,
        response_debug,
    )


def _run_method_for_document(
    *,
    doc: CanonicalDocument,
    method_id: str,
    api_key: str | None,
    api_base: str | None,
    model: str,
    system_prompt: str,
    temperature: float,
    reasoning_effort: str,
    anthropic_thinking: bool,
    anthropic_thinking_budget_tokens: int | None,
    method_verify: bool | None,
    label_profile: Literal["simple", "advanced"],
    chunk_mode: str,
    chunk_size_chars: int,
    progress_callback: Callable[[int, int], None] | None = None,
    runtime_progress_callback: Callable[[dict[str, object]], None] | None = None,
    timeout_seconds: float | None = None,
    method_bundle: str = "audited",
) -> tuple[
    list[CanonicalSpan],
    list[str],
    LLMConfidenceMetric | None,
    list[AgentChunkDiagnostic],
    list[CanonicalSpan],
    list[ResolutionEvent],
    str | None,
    list[str],
]:
    def _emit_runtime_progress(
        *,
        chunk_index: int,
        total_chunks: int,
        start: int,
        end: int,
        pass_index: int | None = None,
        pass_label: str | None = None,
    ) -> None:
        if runtime_progress_callback is None:
            return
        runtime_progress_callback(
            {
                "current_chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "chunk_start": start,
                "chunk_end": end,
                "current_pass_index": pass_index,
                "current_pass_label": pass_label,
            }
        )

    use_detected_value_post_process = _bundle_uses_detected_value_post_process(method_bundle)
    preserve_native_labels = _bundle_preserves_native_labels(method_bundle)

    if _normalize_method_bundle(method_bundle) == "deidentify-v2":
        return _run_deidentify_v2_method_for_document(
            doc=doc,
            method_id=method_id,
            api_key=api_key,
            api_base=api_base,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            anthropic_thinking=anthropic_thinking,
            anthropic_thinking_budget_tokens=anthropic_thinking_budget_tokens,
            method_verify=method_verify,
            timeout_seconds=timeout_seconds,
            progress_callback=progress_callback,
            runtime_progress_callback=runtime_progress_callback,
        )

    if not _should_use_chunking(len(doc.raw_text), chunk_mode, chunk_size_chars):
        _emit_runtime_progress(
            chunk_index=1,
            total_chunks=1,
            start=0,
            end=len(doc.raw_text),
        )
        method_result = run_method_with_metadata(
            text=doc.raw_text,
            method_id=method_id,
            api_key=api_key,
            api_base=api_base,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            reasoning_effort=reasoning_effort,  # type: ignore[arg-type]
            anthropic_thinking=anthropic_thinking,
            anthropic_thinking_budget_tokens=anthropic_thinking_budget_tokens,  # type: ignore[arg-type]
            method_verify=method_verify,
            label_profile=label_profile,
            timeout_seconds=timeout_seconds,
            method_bundle=method_bundle,  # type: ignore[arg-type]
            progress_callback=(
                lambda pass_index, pass_label: _emit_runtime_progress(
                    chunk_index=1,
                    total_chunks=1,
                    start=0,
                    end=len(doc.raw_text),
                    pass_index=pass_index,
                    pass_label=pass_label,
                )
            ),
        )
        normalized_spans = (
            list(method_result.spans)
            if preserve_native_labels
            else normalize_method_spans(method_result.spans, label_profile=label_profile)
        )
        normalized_raw_spans = (
            list(getattr(method_result, "raw_spans", method_result.spans))
            if preserve_native_labels
            else normalize_method_spans(
                getattr(method_result, "raw_spans", method_result.spans),
                label_profile=label_profile,
            )
        )
        spans = _normalize_and_validate_spans(
            normalized_spans,
            doc.raw_text,
        )
        raw_spans = _normalize_and_validate_spans(
            normalized_raw_spans,
            doc.raw_text,
        )
        if use_detected_value_post_process:
            raw_spans = _normalize_and_validate_spans(
                merge_method_spans(_expand_detected_value_occurrences(doc.raw_text, raw_spans)),
                doc.raw_text,
            )
            spans = _normalize_and_validate_spans(
                merge_method_spans(_expand_detected_value_occurrences(doc.raw_text, spans)),
                doc.raw_text,
            )
        if progress_callback is not None:
            progress_callback(1, 1)
        return (
            spans,
            method_result.warnings,
            getattr(method_result, "llm_confidence", None),
            [],
            raw_spans,
            list(getattr(method_result, "resolution_events", [])),
            getattr(method_result, "resolution_policy_version", None),
            list(getattr(method_result, "response_debug", [])),
        )

    chunks = _build_text_chunks(doc, chunk_size_chars)
    warnings: list[str] = []
    response_debug: list[str] = []
    all_spans: list[CanonicalSpan] = []
    all_raw_spans: list[CanonicalSpan] = []
    chunk_count = len(chunks)
    chunk_workers = _resolve_chunk_parallel_workers(chunk_count)

    def _run_chunk(
        idx: int,
        start: int,
        end: int,
    ) -> _ChunkExecutionResult:
        chunk_text = doc.raw_text[start:end]
        _emit_runtime_progress(
            chunk_index=idx + 1,
            total_chunks=chunk_count,
            start=start,
            end=end,
        )
        method_result = run_method_with_metadata(
            text=chunk_text,
            method_id=method_id,
            api_key=api_key,
            api_base=api_base,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            reasoning_effort=reasoning_effort,  # type: ignore[arg-type]
            anthropic_thinking=anthropic_thinking,
            anthropic_thinking_budget_tokens=anthropic_thinking_budget_tokens,  # type: ignore[arg-type]
            method_verify=method_verify,
            label_profile=label_profile,
            timeout_seconds=timeout_seconds,
            method_bundle=method_bundle,  # type: ignore[arg-type]
            progress_callback=(
                lambda pass_index, pass_label: _emit_runtime_progress(
                    chunk_index=idx + 1,
                    total_chunks=chunk_count,
                    start=start,
                    end=end,
                    pass_index=pass_index,
                    pass_label=pass_label,
                )
            ),
        )
        normalized_shifted_spans = (
            _shift_spans(list(method_result.spans), start)
            if preserve_native_labels
            else _shift_spans(
                normalize_method_spans(method_result.spans, label_profile=label_profile),
                start,
            )
        )
        normalized_shifted_raw_spans = (
            _shift_spans(list(getattr(method_result, "raw_spans", method_result.spans)), start)
            if preserve_native_labels
            else _shift_spans(
                normalize_method_spans(
                    getattr(method_result, "raw_spans", method_result.spans),
                    label_profile=label_profile,
                ),
                start,
            )
        )
        return _ChunkExecutionResult(
            idx=idx,
            start=start,
            end=end,
            shifted_spans=normalized_shifted_spans,
            raw_shifted_spans=normalized_shifted_raw_spans,
            warnings=list(method_result.warnings),
            response_debug=list(getattr(method_result, "response_debug", [])),
            llm_confidence=getattr(method_result, "llm_confidence", None),
            finish_reason=None,
        )

    chunk_results: dict[int, _ChunkExecutionResult] = {}
    if chunk_workers == 1:
        for idx, (start, end) in enumerate(chunks):
            chunk_results[idx] = _run_chunk(idx, start, end)
            if progress_callback is not None:
                progress_callback(idx + 1, chunk_count)
    else:
        completed_count = 0
        with ThreadPoolExecutor(max_workers=chunk_workers) as pool:
            future_map = {
                pool.submit(_run_chunk, idx, start, end): idx
                for idx, (start, end) in enumerate(chunks)
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    chunk_result = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"Chunk {idx + 1}/{chunk_count} failed during parallel method execution: {exc}"
                    ) from exc
                chunk_results[idx] = chunk_result
                completed_count += 1
                if progress_callback is not None:
                    progress_callback(completed_count, chunk_count)

    chunk_confidence_metrics: list[LLMConfidenceMetric] = []
    chunk_diagnostics: list[AgentChunkDiagnostic] = []
    for idx in range(chunk_count):
        result = chunk_results[idx]
        all_spans.extend(result.shifted_spans)
        all_raw_spans.extend(result.raw_shifted_spans)
        warnings.extend(_format_chunk_warnings(idx, chunk_count, result.warnings))
        response_debug.extend(
            _format_chunk_response_debug(idx, chunk_count, result.response_debug)
        )
        if result.llm_confidence is not None:
            chunk_confidence_metrics.append(result.llm_confidence)
        chunk_diagnostics.append(_build_chunk_diagnostic(result=result))

    if use_detected_value_post_process:
        all_raw_spans = _expand_detected_value_occurrences(doc.raw_text, all_raw_spans)
        all_spans = _expand_detected_value_occurrences(doc.raw_text, all_spans)

    raw_spans = _normalize_and_validate_spans(
        merge_method_spans(all_raw_spans),
        doc.raw_text,
    )
    resolved_spans, resolution_events = resolve_spans(
        doc.raw_text,
        merge_method_spans(all_spans),
        label_profile=label_profile,
        enable_augmentation=True,
    )
    normalized = _normalize_and_validate_spans(
        merge_method_spans(resolved_spans),
        doc.raw_text,
    )
    normalized, dropped_name_spans = _drop_implausible_name_spans(
        doc.raw_text,
        normalized,
        label_profile=label_profile,
    )
    if dropped_name_spans > 0:
        target_label = "NAME" if label_profile == "simple" else "PERSON"
        warnings.append(
            f"Dropped {dropped_name_spans} implausible {target_label} span(s) after shared resolution."
        )
    warnings.insert(
        0,
        (
            f"Chunked method run used {chunk_count} chunk(s) at ~{chunk_size_chars} chars "
            f"(mode={chunk_mode}, workers={chunk_workers})."
        ),
    )
    aggregated_confidence = (
        _aggregate_llm_confidence(chunk_confidence_metrics)
        if chunk_confidence_metrics
        else None
    )
    return (
        normalized,
        warnings,
        aggregated_confidence,
        chunk_diagnostics,
        raw_spans,
        resolution_events,
        RESOLUTION_POLICY_VERSION,
        response_debug,
    )


# --- Routes ---


class FolderSampleCreateBody(BaseModel):
    count: int = Field(gt=0)


class FolderCreateBody(BaseModel):
    name: str = Field(min_length=1)
    parent_folder_id: str | None = None


@app.get("/api/documents")
async def list_documents():
    session_id = "default"
    ids = _load_session_index(session_id)
    docs = []
    for did in ids:
        doc = _load_doc(did, session_id)
        if doc:
            docs.append(_build_document_summary(did, session_id))
    return docs


@app.get("/api/folders")
async def list_folders():
    session_id = "default"
    return [_folder_to_summary(folder, session_id) for folder in _load_all_folders(session_id)]


@app.get("/api/folders/{folder_id}")
async def get_folder(folder_id: str):
    session_id = "default"
    folder = _load_folder(folder_id, session_id)
    if folder is None:
        raise HTTPException(status_code=404, detail="Folder not found")
    return _folder_to_detail(folder, session_id)


@app.post("/api/folders")
async def create_folder(body: FolderCreateBody):
    session_id = "default"
    folder_name = body.name.strip()
    if not folder_name:
        raise HTTPException(status_code=400, detail="Folder name cannot be empty.")

    parent_folder: FolderRecord | None = None
    parent_folder_id = body.parent_folder_id.strip() if body.parent_folder_id else None
    if parent_folder_id:
        parent_folder = _load_folder(parent_folder_id, session_id)
        if parent_folder is None:
            raise HTTPException(status_code=404, detail="Parent folder not found")

    folder_id = str(uuid.uuid4())[:8]
    while _load_folder(folder_id, session_id) is not None:
        folder_id = str(uuid.uuid4())[:8]

    folder = FolderRecord(
        id=folder_id,
        name=folder_name,
        kind="manual",
        parent_folder_id=parent_folder.id if parent_folder else None,
        merged_doc_id=None,
        doc_ids=[],
        child_folder_ids=[],
        source_filename=None,
        source_folder_id=None,
        sample_size=None,
        sample_seed=None,
        created_at=_now_iso(),
    )
    folder_ids = _load_folder_index(session_id)
    folder_ids.append(folder.id)
    _save_folder(folder, session_id)
    if parent_folder is not None:
        updated_parent = parent_folder.model_copy(
            update={"child_folder_ids": [*parent_folder.child_folder_ids, folder.id]}
        )
        _save_folder(updated_parent, session_id)
    _save_folder_index(folder_ids, session_id)
    return _folder_to_detail(folder, session_id)


@app.get("/api/documents/{doc_id}")
async def get_document(doc_id: str):
    session_id = "default"
    doc = _load_doc(doc_id, session_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return _enrich_doc(doc, session_id)


def _upload_document_payload(raw: bytes, filename: str, session_id: str = "default") -> CanonicalDocument:
    doc_id = str(uuid.uuid4())[:8]

    try:
        if filename.endswith(".jsonl"):
            merged_doc, record_docs, record_display_names = parse_jsonl_file(
                raw,
                filename,
                doc_id,
            )
            docs = [merged_doc]
        else:
            docs = parse_file(raw, filename, doc_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    ids = _load_session_index(session_id)
    if filename.endswith(".jsonl"):
        merged_doc = docs[0]
        _save_doc(merged_doc, session_id)
        ids.append(merged_doc.id)
        _save_session_index(session_id)
        if len(record_docs) > 1:
            for child_doc in record_docs:
                _save_hidden_doc(child_doc, session_id)
            folder_id = str(uuid.uuid4())[:8]
            while _load_folder(folder_id, session_id) is not None:
                folder_id = str(uuid.uuid4())[:8]
            folder = FolderRecord(
                id=folder_id,
                name=Path(filename).stem or filename,
                kind="import",
                merged_doc_id=merged_doc.id,
                doc_ids=[doc.id for doc in record_docs],
                child_folder_ids=[],
                source_filename=filename,
                created_at=_now_iso(),
                doc_display_names={
                    doc.id: record_display_names[index]
                    for index, doc in enumerate(record_docs)
                },
            )
            folder_ids = _load_folder_index(session_id)
            folder_ids.append(folder.id)
            _save_folder(folder, session_id)
            _save_folder_index(folder_ids, session_id)
        return _enrich_doc(merged_doc, session_id)

    for doc in docs:
        _save_doc(doc, session_id)
        ids.append(doc.id)
    _save_session_index(session_id)

    # Return the first document (for single-file uploads)
    # For JSONL with multiple records, return the first one
    if docs:
        return _enrich_doc(docs[0], session_id)
    raise HTTPException(status_code=400, detail="No documents parsed from file")


@app.post("/api/documents/upload")
async def upload_file(file: UploadFile = File(...)):
    raw = await file.read()
    filename = file.filename or "unknown"
    return _upload_document_payload(raw, filename)


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    session_id = "default"
    if _find_folders_for_doc(doc_id, session_id):
        raise HTTPException(
            status_code=409,
            detail="Document is managed by a folder and cannot be deleted directly.",
        )
    ids = _load_session_index(session_id)
    in_index = doc_id in ids
    removed_files = _delete_doc_files(doc_id, session_id)
    if not in_index and not removed_files:
        raise HTTPException(status_code=404, detail="Document not found")

    if in_index:
        _session_docs[session_id] = [did for did in ids if did != doc_id]
        _save_session_index(session_id)

    for folder in _load_all_folders(session_id):
        if folder.merged_doc_id != doc_id:
            continue
        _save_folder(folder.model_copy(update={"merged_doc_id": None}), session_id)

    return {"deleted": True, "doc_id": doc_id}

def _delete_folder_tree(folder: FolderRecord, session_id: str) -> None:
    for child_folder_id in list(folder.child_folder_ids):
        child = _load_folder(child_folder_id, session_id)
        if child is not None:
            _delete_folder_tree(child, session_id)
    for source_child in _load_all_folders(session_id):
        if source_child.source_folder_id != folder.id:
            continue
        if source_child.id == folder.id:
            continue
        _delete_folder_tree(source_child, session_id)
    if folder.kind == "import":
        for doc_id in folder.doc_ids:
            _delete_doc_files(doc_id, session_id)
    if folder.parent_folder_id:
        parent = _load_folder(folder.parent_folder_id, session_id)
        if parent is not None:
            updated_parent = parent.model_copy(
                update={
                    "child_folder_ids": [
                        child_id for child_id in parent.child_folder_ids if child_id != folder.id
                    ]
                }
            )
            _save_folder(updated_parent, session_id)
    folder_ids = [item for item in _load_folder_index(session_id) if item != folder.id]
    _save_folder_index(folder_ids, session_id)
    _delete_folder_file(folder.id, session_id)


@app.post("/api/folders/{folder_id}/samples")
async def create_folder_sample(folder_id: str, body: FolderSampleCreateBody):
    session_id = "default"
    folder = _load_folder(folder_id, session_id)
    if folder is None:
        raise HTTPException(status_code=404, detail="Folder not found")
    if body.count > len(folder.doc_ids):
        raise HTTPException(
            status_code=400,
            detail="Sample count cannot exceed the number of direct documents in the folder.",
        )
    sample_seed = random.randint(0, 2**31 - 1)
    rng = random.Random(sample_seed)
    sampled_doc_ids = rng.sample(folder.doc_ids, body.count)
    sample_id = str(uuid.uuid4())[:8]
    while _load_folder(sample_id, session_id) is not None:
        sample_id = str(uuid.uuid4())[:8]
    sample_folder = FolderRecord(
        id=sample_id,
        name=f"Sample {body.count}",
        kind="sample",
        parent_folder_id=None,
        merged_doc_id=None,
        doc_ids=sampled_doc_ids,
        child_folder_ids=[],
        source_filename=folder.source_filename,
        source_folder_id=folder.id,
        sample_size=body.count,
        sample_seed=sample_seed,
        created_at=_now_iso(),
        doc_display_names={
            doc_id: folder.doc_display_names.get(doc_id, _load_doc(doc_id, session_id).filename)
            for doc_id in sampled_doc_ids
            if _load_doc(doc_id, session_id) is not None
        },
    )
    folder_ids = _load_folder_index(session_id)
    folder_ids.append(sample_folder.id)
    _save_folder(sample_folder, session_id)
    _save_folder_index(folder_ids, session_id)
    return _folder_to_detail(sample_folder, session_id)


@app.post("/api/folders/{folder_id}/prune-empty-docs")
async def prune_folder_empty_docs(folder_id: str):
    session_id = "default"
    folder = _load_folder(folder_id, session_id)
    if folder is None:
        raise HTTPException(status_code=404, detail="Folder not found")
    removed_doc_ids, updated_folder_ids = _prune_unannotated_folder_docs(folder, session_id)
    return {
        "folder_id": folder_id,
        "removed_count": len(removed_doc_ids),
        "removed_doc_ids": removed_doc_ids,
        "updated_folder_ids": updated_folder_ids,
    }


@app.delete("/api/folders/{folder_id}")
async def delete_folder(folder_id: str):
    session_id = "default"
    folder = _load_folder(folder_id, session_id)
    if folder is None:
        raise HTTPException(status_code=404, detail="Folder not found")
    _delete_folder_tree(folder, session_id)
    return {"deleted": True, "folder_id": folder_id}


@app.put("/api/documents/{doc_id}/manual-annotations")
async def save_manual_annotations(doc_id: str, spans: list[CanonicalSpan]):
    session_id = "default"
    doc = _load_doc(doc_id, session_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    normalized = _normalize_and_validate_spans(spans, doc.raw_text)
    _persist_manual_annotations(doc_id, normalized, session_id)
    return _enrich_doc(doc, session_id)


@app.post("/api/session/mirror-pre-to-manual")
async def mirror_pre_to_manual(
    scope: Annotated[Literal["top_level", "folder"], Query()] = "top_level",
    folder_id: Annotated[str | None, Query()] = None,
):
    session_id = "default"
    doc_ids = _resolve_ground_truth_export_doc_ids(scope, folder_id, session_id)
    updated_at = _now_iso()
    processed_doc_ids: list[str] = []
    copied_count = 0
    cleared_count = 0

    for doc_id in doc_ids:
        doc = _load_doc(doc_id, session_id)
        if doc is None:
            continue
        mirrored = _dedup_spans(list(doc.pre_annotations))
        _persist_manual_annotations(
            doc_id,
            mirrored,
            session_id,
            updated_at=updated_at,
        )
        processed_doc_ids.append(doc_id)
        if mirrored:
            copied_count += 1
        else:
            cleared_count += 1

    return {
        "scope": scope,
        "folder_id": folder_id if scope == "folder" else None,
        "processed_count": len(processed_doc_ids),
        "copied_count": copied_count,
        "cleared_count": cleared_count,
        "doc_ids": processed_doc_ids,
    }


class AgentRunBody(BaseModel):
    mode: Literal["rule", "llm", "method", "openai"] = "rule"
    system_prompt: str | None = None
    model: str | None = None
    temperature: float | None = None
    api_key: str | None = None
    api_base: str | None = None
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh"] | None = None
    anthropic_thinking: bool | None = None
    anthropic_thinking_budget_tokens: int | None = None
    label_profile: Literal["simple", "advanced"] | None = None
    chunk_mode: Literal["auto", "off", "force"] | None = None
    chunk_size_chars: int | None = None
    method_id: str | None = None
    method_verify: bool | None = None


class PromptLabPromptInput(BaseModel):
    id: str | None = None
    label: str
    system_prompt: str | None = None
    prompt_file: str | None = None
    variant_type: Literal["prompt", "preset"] = "prompt"
    preset_method_id: str | None = None
    method_verify_override: bool | None = None


class PromptLabModelInput(BaseModel):
    id: str | None = None
    label: str
    model: str
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh"] = "none"
    anthropic_thinking: bool = False
    anthropic_thinking_budget_tokens: int | None = None


class PromptLabRuntimeInput(BaseModel):
    api_key: str | None = None
    api_base: str | None = None
    temperature: float = 0.0
    match_mode: Literal["exact", "boundary", "overlap", "substring"] = "overlap"
    reference_source: Literal["manual", "pre"] = "manual"
    fallback_reference_source: Literal["manual", "pre"] = "pre"
    label_profile: Literal["simple", "advanced"] = "simple"
    label_projection: Literal["native", "coarse_simple"] = "native"
    method_bundle: Literal[
        "legacy", "audited", "test", "v2", "v2+post-process", "deidentify-v2"
    ] = "audited"
    chunk_mode: Literal["auto", "off", "force"] = "off"
    chunk_size_chars: int = DEFAULT_CHUNK_SIZE_CHARS


class PromptLabRunCreateBody(BaseModel):
    name: str | None = None
    doc_ids: list[str] = Field(default_factory=list)
    folder_ids: list[str] = Field(default_factory=list)
    prompts: list[PromptLabPromptInput]
    models: list[PromptLabModelInput]
    runtime: PromptLabRuntimeInput
    concurrency: int = PROMPT_LAB_DEFAULT_CONCURRENCY


class MethodsLabMethodInput(BaseModel):
    id: str | None = None
    label: str
    method_id: str
    method_verify_override: bool | None = None


class MethodsLabRuntimeInput(BaseModel):
    api_key: str | None = None
    api_base: str | None = None
    temperature: float = 0.0
    match_mode: Literal["exact", "boundary", "overlap", "substring"] = "overlap"
    reference_source: Literal["manual", "pre"] = "manual"
    fallback_reference_source: Literal["manual", "pre"] = "pre"
    label_profile: Literal["simple", "advanced"] = "simple"
    label_projection: Literal["native", "coarse_simple"] = "native"
    method_bundle: Literal[
        "legacy", "audited", "test", "v2", "v2+post-process", "deidentify-v2"
    ] = "audited"
    chunk_mode: Literal["auto", "off", "force"] = "off"
    chunk_size_chars: int = DEFAULT_CHUNK_SIZE_CHARS
    task_timeout_seconds: float | None = None


class MethodsLabRunCreateBody(BaseModel):
    name: str | None = None
    doc_ids: list[str] = Field(default_factory=list)
    folder_ids: list[str] = Field(default_factory=list)
    methods: list[MethodsLabMethodInput]
    models: list[PromptLabModelInput]
    runtime: MethodsLabRuntimeInput
    concurrency: int = METHODS_LAB_DEFAULT_CONCURRENCY


class ExperimentLimitsResponse(BaseModel):
    prompt_lab_default_concurrency: int
    prompt_lab_max_concurrency: int
    methods_lab_default_concurrency: int
    methods_lab_max_concurrency: int


class ExperimentRunDiagnosticsResponse(BaseModel):
    requested_concurrency: int
    effective_worker_count: int
    max_allowed_concurrency: int
    total_tasks: int
    clamped_by_task_count: bool
    clamped_by_server_cap: bool
    api_base_host: str | None = None


class GatewayCatalogDiagnosticsResponse(BaseModel):
    reachable: bool
    model_count: int | None = None
    error: str | None = None
    checked_at: str | None = None


class ExperimentDiagnosticsResponse(BaseModel):
    resolved_api_base: str | None = None
    api_base_host: str | None = None
    prompt_lab_max_concurrency: int
    methods_lab_max_concurrency: int
    gateway_catalog: GatewayCatalogDiagnosticsResponse


def _resolve_prompt_lab_runtime(runtime: PromptLabRuntimeInput) -> dict[str, object]:
    cfg = _load_config()
    api_key = (
        runtime.api_key
        or _resolve_env_api_key()
    )
    api_base = (
        runtime.api_base
        or str(cfg.get("api_base", "") or "")
        or _resolve_env_api_base()
    )
    match_mode = _normalize_match_mode(runtime.match_mode)
    chunk_mode = _normalize_chunk_mode(runtime.chunk_mode)
    chunk_size_chars = _normalize_chunk_size(runtime.chunk_size_chars)
    label_profile = _normalize_label_profile(runtime.label_profile)
    label_projection = _normalize_label_projection(runtime.label_projection)
    method_bundle = _normalize_method_bundle(runtime.method_bundle)
    return {
        "api_key": api_key,
        "api_base": api_base,
        "temperature": runtime.temperature,
        "match_mode": match_mode,
        "reference_source": runtime.reference_source,
        "fallback_reference_source": runtime.fallback_reference_source,
        "chunk_mode": chunk_mode,
        "chunk_size_chars": chunk_size_chars,
        "label_profile": label_profile,
        "label_projection": label_projection,
        "method_bundle": method_bundle,
    }


def _validate_prompt_lab_request(body: PromptLabRunCreateBody, session_id: str = "default"):
    cfg = _load_config()
    prompt_lab_max_concurrency = _get_prompt_lab_max_concurrency(cfg)
    if not body.doc_ids:
        raise HTTPException(status_code=400, detail="doc_ids is required")
    if len(body.prompts) < 1 or len(body.prompts) > PROMPT_LAB_MAX_VARIANTS:
        raise HTTPException(
            status_code=400,
            detail=f"Prompt variants must be between 1 and {PROMPT_LAB_MAX_VARIANTS}",
        )
    if len(body.models) < 1 or len(body.models) > PROMPT_LAB_MAX_VARIANTS:
        raise HTTPException(
            status_code=400,
            detail=f"Model variants must be between 1 and {PROMPT_LAB_MAX_VARIANTS}",
        )
    _validate_experiment_concurrency(
        body.concurrency,
        max_allowed=prompt_lab_max_concurrency,
    )
    if len(body.prompts) * len(body.models) > PROMPT_LAB_MAX_VARIANTS * PROMPT_LAB_MAX_VARIANTS:
        raise HTTPException(status_code=400, detail="Matrix limit exceeded (max 6x6)")

    known_ids = set(_load_session_index(session_id))
    for doc_id in body.doc_ids:
        if doc_id not in known_ids:
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

    method_bundle = _normalize_method_bundle(body.runtime.method_bundle)
    prompt_ids_seen: set[str] = set()
    for index, prompt in enumerate(body.prompts):
        if not prompt.label.strip():
            raise HTTPException(status_code=400, detail=f"Prompt label required at index {index}")
        if prompt.variant_type == "prompt":
            if prompt.method_verify_override is not None:
                raise HTTPException(
                    status_code=400,
                    detail=f"method_verify_override is only supported for preset variants (prompt index {index})",
                )
            if prompt.preset_method_id:
                raise HTTPException(
                    status_code=400,
                    detail=f"preset_method_id is only supported for preset variants (prompt index {index})",
                )
            prompt_text = (prompt.system_prompt or "").strip()
            if not prompt_text:
                raise HTTPException(
                    status_code=400, detail=f"system_prompt required at prompt index {index}"
                )
            if "{" in prompt_text or "}" in prompt_text:
                raise HTTPException(
                    status_code=400,
                    detail="Prompt variants must be plain system prompt text (templating is not supported)",
                )
        elif prompt.variant_type == "preset":
            method_id = (prompt.preset_method_id or "").strip()
            if not method_id:
                raise HTTPException(
                    status_code=400,
                    detail=f"preset_method_id required for preset variant at prompt index {index}",
                )
            allowed_preset_methods = _get_prompt_lab_allowed_preset_methods(
                method_bundle=method_bundle
            )
            if method_id not in allowed_preset_methods:
                allowed = ", ".join(sorted(allowed_preset_methods))
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"preset_method_id '{method_id}' is not allowed in Prompt Lab. "
                        f"Allowed presets: {allowed}"
                    ),
                )
            definition = get_method_definition_by_id(method_id, method_bundle=method_bundle)
            if definition is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown preset method: {method_id}",
                )
            if (
                prompt.method_verify_override is not None
                and not bool(definition.get("supports_verify_override"))
            ):
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"method_verify_override is not supported for preset method "
                        f"'{method_id}' (prompt index {index})"
                    ),
                )
        prompt_id = (prompt.id or "").strip() or f"prompt_{index + 1}"
        if prompt_id in prompt_ids_seen:
            raise HTTPException(status_code=400, detail=f"Duplicate prompt id: {prompt_id}")
        prompt_ids_seen.add(prompt_id)

    model_ids_seen: set[str] = set()
    for index, model in enumerate(body.models):
        if not model.label.strip():
            raise HTTPException(status_code=400, detail=f"Model label required at index {index}")
        if not model.model.strip():
            raise HTTPException(status_code=400, detail=f"model required at index {index}")
        model_id = (model.id or "").strip() or f"model_{index + 1}"
        if model_id in model_ids_seen:
            raise HTTPException(status_code=400, detail=f"Duplicate model id: {model_id}")
        model_ids_seen.add(model_id)


def _resolve_prompt_lab_reference(
    doc: CanonicalDocument,
    reference_source: Literal["manual", "pre"],
    fallback_reference_source: Literal["manual", "pre"],
) -> tuple[Literal["manual", "pre"], list[CanonicalSpan]]:
    primary = doc.manual_annotations if reference_source == "manual" else doc.pre_annotations
    if primary:
        return reference_source, primary
    fallback = (
        doc.manual_annotations if fallback_reference_source == "manual" else doc.pre_annotations
    )
    return fallback_reference_source, fallback


def _initialize_prompt_lab_run(
    body: PromptLabRunCreateBody,
    session_id: str,
    runtime: dict[str, object],
) -> dict:
    run_id = str(uuid.uuid4())[:8]
    now = _now_iso()
    prompts: list[dict] = []
    models: list[dict] = []
    for index, item in enumerate(body.prompts):
        prompt_id = (item.id or "").strip() or f"prompt_{index + 1}"
        variant_type = str(item.variant_type or "prompt")
        preset_method_id = (
            (item.preset_method_id or "").strip() if item.preset_method_id else None
        )
        prompts.append(
            {
                "id": prompt_id,
                "label": item.label.strip(),
                "variant_type": variant_type,
                "system_prompt": item.system_prompt if variant_type == "prompt" else None,
                "preset_method_id": preset_method_id if variant_type == "preset" else None,
                "method_verify_override": (
                    bool(item.method_verify_override)
                    if item.method_verify_override is not None
                    else None
                ),
            }
        )
    for index, item in enumerate(body.models):
        model_id = (item.id or "").strip() or f"model_{index + 1}"
        models.append(
            {
                "id": model_id,
                "label": item.label.strip(),
                "model": item.model.strip(),
                "reasoning_effort": item.reasoning_effort,
                "anthropic_thinking": bool(item.anthropic_thinking),
                "anthropic_thinking_budget_tokens": item.anthropic_thinking_budget_tokens,
            }
        )

    cells: dict[str, dict] = {}
    for model in models:
        for prompt in prompts:
            cell_id = f"{model['id']}__{prompt['id']}"
            cells[cell_id] = {
                "id": cell_id,
                "model_id": model["id"],
                "model_label": model["label"],
                "prompt_id": prompt["id"],
                "prompt_label": prompt["label"],
                "documents": {
                    doc_id: {"status": "pending", "updated_at": now} for doc_id in body.doc_ids
                },
            }

    run = {
        "id": run_id,
        "name": (body.name or "").strip() or f"Prompt Lab {now}",
        "status": "queued",
        "created_at": now,
        "started_at": None,
        "finished_at": None,
        "doc_ids": list(dict.fromkeys(body.doc_ids)),
        "prompts": prompts,
        "models": models,
        "runtime": {
            "temperature": runtime["temperature"],
            "match_mode": runtime["match_mode"],
            "reference_source": runtime["reference_source"],
            "fallback_reference_source": runtime["fallback_reference_source"],
            "label_profile": runtime["label_profile"],
            "label_projection": runtime["label_projection"],
            "method_bundle": runtime["method_bundle"],
            "api_base": runtime["api_base"],
            "chunk_mode": runtime["chunk_mode"],
            "chunk_size_chars": runtime["chunk_size_chars"],
        },
        "concurrency": body.concurrency,
        "warnings": [],
        "errors": [],
        "cells": cells,
        "session_id": session_id,
    }
    return run


def _run_prompt_lab_job(run_id: str, session_id: str, runtime: dict[str, object]):
    api_key = str(runtime["api_key"])
    api_base = str(runtime["api_base"])
    temperature = float(runtime["temperature"])
    match_mode = str(runtime["match_mode"])
    reference_source = runtime["reference_source"]
    fallback_reference_source = runtime["fallback_reference_source"]
    label_profile = str(runtime["label_profile"])
    label_projection = _normalize_label_projection(runtime.get("label_projection", "native"))
    method_bundle = _normalize_method_bundle(runtime.get("method_bundle", "audited"))
    chunk_mode = str(runtime["chunk_mode"])
    chunk_size_chars = int(runtime["chunk_size_chars"])

    with _prompt_lab_lock:
        run = _load_prompt_lab_run(run_id, session_id)
        if run is None:
            return
        run["status"] = "running"
        run["started_at"] = _now_iso()
        _save_prompt_lab_run(run, session_id)

    tasks: list[tuple[str, str, str]] = []
    for model in run.get("models", []):
        for prompt in run.get("prompts", []):
            cell_id = f"{model['id']}__{prompt['id']}"
            for doc_id in run.get("doc_ids", []):
                tasks.append((cell_id, str(doc_id), str(model["id"])))

    model_by_id = {str(model["id"]): model for model in run.get("models", [])}
    prompt_by_id = {str(prompt["id"]): prompt for prompt in run.get("prompts", [])}

    def _execute_task(cell_id: str, doc_id: str, model_id: str) -> tuple[str, str, dict]:
        cell = run.get("cells", {}).get(cell_id, {})
        if not isinstance(cell, dict):
            return cell_id, doc_id, {"status": "failed", "error": f"Unknown cell '{cell_id}'"}
        prompt_id = str(cell.get("prompt_id", ""))
        prompt = prompt_by_id.get(prompt_id)
        model = model_by_id.get(model_id)
        if prompt is None or model is None:
            return cell_id, doc_id, {"status": "failed", "error": "Invalid model/prompt mapping"}

        doc = _load_doc(doc_id, session_id)
        if doc is None:
            return cell_id, doc_id, {"status": "unavailable", "error": "Document no longer exists"}
        enriched = _enrich_doc(doc, session_id)
        try:
            variant_type = str(prompt.get("variant_type") or "prompt")
            if variant_type == "preset":
                preset_method_id = str(prompt.get("preset_method_id") or "").strip()
                if not preset_method_id:
                    raise ValueError("preset_method_id is required for preset variants.")
                method_verify_override = prompt.get("method_verify_override")
                if not isinstance(method_verify_override, bool):
                    method_verify_override = None

                (
                    hypothesis_spans,
                    warnings,
                    llm_confidence,
                    _chunk_diagnostics,
                    raw_hypothesis_spans,
                    resolution_events,
                    resolution_policy_version,
                    response_debug,
                ) = _run_method_for_document(
                    doc=enriched,
                    method_id=preset_method_id,
                    api_key=api_key or None,
                    api_base=api_base or None,
                    model=str(model["model"]),
                    system_prompt="",
                    temperature=temperature,
                    reasoning_effort=str(model["reasoning_effort"]),
                    anthropic_thinking=bool(model["anthropic_thinking"]),
                    anthropic_thinking_budget_tokens=model["anthropic_thinking_budget_tokens"],
                    method_verify=method_verify_override,
                    label_profile=label_profile,  # type: ignore[arg-type]
                    chunk_mode=chunk_mode,
                    chunk_size_chars=chunk_size_chars,
                    method_bundle=method_bundle,
                )
            else:
                requested_system_prompt = str(prompt.get("system_prompt") or "").strip()
                if not requested_system_prompt:
                    raise ValueError("system_prompt is required for prompt variants.")
                (
                    hypothesis_spans,
                    warnings,
                    llm_confidence,
                    _chunk_diagnostics,
                    raw_hypothesis_spans,
                    resolution_events,
                    resolution_policy_version,
                    response_debug,
                ) = _run_llm_for_document(
                    doc=enriched,
                    api_key=api_key,
                    api_base=api_base or None,
                    model=str(model["model"]),
                    system_prompt=requested_system_prompt,
                    temperature=temperature,
                    reasoning_effort=str(model["reasoning_effort"]),
                    anthropic_thinking=bool(model["anthropic_thinking"]),
                    anthropic_thinking_budget_tokens=model["anthropic_thinking_budget_tokens"],
                    label_profile=label_profile,  # type: ignore[arg-type]
                    chunk_mode=chunk_mode,
                    chunk_size_chars=chunk_size_chars,
                    method_bundle=method_bundle,
                )

            resolved_reference_source, reference_spans = _resolve_prompt_lab_reference(
                enriched,
                reference_source,  # type: ignore[arg-type]
                fallback_reference_source,  # type: ignore[arg-type]
            )
            projected_reference, projected_hypothesis = _prepare_experiment_scoring_spans(
                reference_spans,
                hypothesis_spans,
                label_projection=label_projection,  # type: ignore[arg-type]
                method_bundle=method_bundle,
            )
            projected_reference_raw, projected_hypothesis_raw = _prepare_experiment_scoring_spans(
                reference_spans,
                raw_hypothesis_spans,
                label_projection=label_projection,  # type: ignore[arg-type]
                method_bundle=method_bundle,
            )
            scoring_match_mode = _normalize_metrics_mode_for_method_bundle(
                match_mode,
                method_bundle=method_bundle,
            )
            metrics = _serialize_metrics_payload(
                compute_metrics(projected_reference, projected_hypothesis, scoring_match_mode)
            )
            raw_metrics = _serialize_metrics_payload(
                compute_metrics(
                    projected_reference_raw,
                    projected_hypothesis_raw,
                    scoring_match_mode,
                )
            )
            return cell_id, doc_id, {
                "status": "completed",
                "reference_source_used": resolved_reference_source,
                "reference_spans": [span.model_dump() for span in reference_spans],
                "raw_hypothesis_spans": [
                    span.model_dump() for span in raw_hypothesis_spans
                ],
                "hypothesis_spans": [span.model_dump() for span in hypothesis_spans],
                "raw_metrics": raw_metrics,
                "metrics": metrics,
                "warnings": warnings,
                "llm_confidence": llm_confidence.model_dump() if llm_confidence else None,
                "response_debug": response_debug,
                "resolution_events": [event.model_dump() for event in resolution_events],
                "resolution_policy_version": resolution_policy_version,
                "resolution_summary": summarize_resolution_events(resolution_events),
                "updated_at": _now_iso(),
                "filename": enriched.filename,
            }
        except Exception as exc:
            message = str(exc).strip() or exc.__class__.__name__
            if len(message) > 800:
                message = f"{message[:800]}..."
            return cell_id, doc_id, {
                "status": "failed",
                "error": message,
                "updated_at": _now_iso(),
                "filename": enriched.filename,
            }

    try:
        with ThreadPoolExecutor(max_workers=int(run.get("concurrency", 1))) as executor:
            future_map = {
                executor.submit(_execute_task, cell_id, doc_id, model_id): (cell_id, doc_id)
                for cell_id, doc_id, model_id in tasks
            }
            for future in as_completed(future_map):
                cell_id, doc_id = future_map[future]
                try:
                    _, _, result = future.result()
                except Exception as exc:  # pragma: no cover - defensive fallback
                    result = {
                        "status": "failed",
                        "error": str(exc),
                        "updated_at": _now_iso(),
                    }
                with _prompt_lab_lock:
                    latest = _load_prompt_lab_run(run_id, session_id)
                    if latest is None:
                        continue
                    cells = latest.get("cells", {})
                    if isinstance(cells, dict) and isinstance(cells.get(cell_id), dict):
                        cells[cell_id]["documents"][doc_id] = result
                    _save_prompt_lab_run(latest, session_id)
    except Exception as exc:
        with _prompt_lab_lock:
            latest = _load_prompt_lab_run(run_id, session_id)
            if latest is not None:
                latest["status"] = "failed"
                latest["finished_at"] = _now_iso()
                errors = latest.get("errors", [])
                if not isinstance(errors, list):
                    errors = []
                errors.append(f"Prompt Lab run failed: {exc}")
                latest["errors"] = errors
                _save_prompt_lab_run(latest, session_id)
        return

    with _prompt_lab_lock:
        latest = _load_prompt_lab_run(run_id, session_id)
        if latest is None:
            return
        summary = _build_prompt_lab_run_summary(latest)
        latest["status"] = (
            "completed_with_errors"
            if int(summary.get("failed_tasks", 0)) > 0
            else "completed"
        )
        latest["finished_at"] = _now_iso()
        _save_prompt_lab_run(latest, session_id)


def _resolve_methods_lab_runtime(runtime: MethodsLabRuntimeInput) -> dict[str, object]:
    cfg = _load_config()
    api_key = (
        runtime.api_key
        or _resolve_env_api_key()
    )
    api_base = (
        runtime.api_base
        or str(cfg.get("api_base", "") or "")
        or _resolve_env_api_base()
    )
    match_mode = _normalize_match_mode(runtime.match_mode)
    chunk_mode = _normalize_chunk_mode(runtime.chunk_mode)
    chunk_size_chars = _normalize_chunk_size(runtime.chunk_size_chars)
    label_profile = _normalize_label_profile(runtime.label_profile)
    label_projection = _normalize_label_projection(runtime.label_projection)
    method_bundle = _normalize_method_bundle(runtime.method_bundle)
    task_timeout_seconds = _safe_float(runtime.task_timeout_seconds)
    if task_timeout_seconds is not None and task_timeout_seconds <= 0:
        raise HTTPException(status_code=400, detail="task_timeout_seconds must be greater than 0")
    return {
        "api_key": api_key,
        "api_base": api_base,
        "temperature": runtime.temperature,
        "match_mode": match_mode,
        "chunk_mode": chunk_mode,
        "chunk_size_chars": chunk_size_chars,
        "label_profile": label_profile,
        "label_projection": label_projection,
        "method_bundle": method_bundle,
        "task_timeout_seconds": task_timeout_seconds,
    }


def _validate_methods_lab_request(body: MethodsLabRunCreateBody, session_id: str = "default"):
    cfg = _load_config()
    methods_lab_max_concurrency = _get_methods_lab_max_concurrency(cfg)
    if not body.doc_ids:
        raise HTTPException(status_code=400, detail="doc_ids is required")
    if len(body.methods) < 1 or len(body.methods) > METHODS_LAB_MAX_METHOD_VARIANTS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Method variants must be between 1 and {METHODS_LAB_MAX_METHOD_VARIANTS}"
            ),
        )
    if len(body.models) < 1 or len(body.models) > PROMPT_LAB_MAX_VARIANTS:
        raise HTTPException(
            status_code=400,
            detail=f"Model variants must be between 1 and {PROMPT_LAB_MAX_VARIANTS}",
        )
    _validate_experiment_concurrency(
        body.concurrency,
        max_allowed=methods_lab_max_concurrency,
    )

    known_ids = set(_load_session_index(session_id))
    for doc_id in body.doc_ids:
        if doc_id not in known_ids:
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

    method_bundle = _normalize_method_bundle(body.runtime.method_bundle)
    method_ids_seen: set[str] = set()
    for index, method in enumerate(body.methods):
        if not method.label.strip():
            raise HTTPException(status_code=400, detail=f"Method label required at index {index}")
        method_variant_id = (method.id or "").strip() or f"method_{index + 1}"
        if method_variant_id in method_ids_seen:
            raise HTTPException(status_code=400, detail=f"Duplicate method id: {method_variant_id}")
        method_ids_seen.add(method_variant_id)

        method_id = (method.method_id or "").strip()
        if not method_id:
            raise HTTPException(status_code=400, detail=f"method_id required at index {index}")
        definition = get_method_definition_by_id(method_id, method_bundle=method_bundle)
        if definition is None:
            raise HTTPException(status_code=400, detail=f"Unknown method: {method_id}")
        if (
            method.method_verify_override is not None
            and not bool(definition.get("supports_verify_override"))
        ):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"method_verify_override is not supported for method '{method_id}' "
                    f"(method index {index})"
                ),
            )

    model_ids_seen: set[str] = set()
    for index, model in enumerate(body.models):
        if not model.label.strip():
            raise HTTPException(status_code=400, detail=f"Model label required at index {index}")
        if not model.model.strip():
            raise HTTPException(status_code=400, detail=f"model required at index {index}")
        model_id = (model.id or "").strip() or f"model_{index + 1}"
        if model_id in model_ids_seen:
            raise HTTPException(status_code=400, detail=f"Duplicate model id: {model_id}")
        model_ids_seen.add(model_id)


def _initialize_methods_lab_run(
    body: MethodsLabRunCreateBody,
    session_id: str,
    runtime: dict[str, object],
) -> dict:
    run_id = str(uuid.uuid4())[:8]
    now = _now_iso()
    methods: list[dict] = []
    models: list[dict] = []
    for index, item in enumerate(body.methods):
        method_variant_id = (item.id or "").strip() or f"method_{index + 1}"
        methods.append(
            {
                "id": method_variant_id,
                "label": item.label.strip(),
                "method_id": item.method_id.strip(),
                "method_verify_override": (
                    bool(item.method_verify_override)
                    if item.method_verify_override is not None
                    else None
                ),
            }
        )
    for index, item in enumerate(body.models):
        model_id = (item.id or "").strip() or f"model_{index + 1}"
        models.append(
            {
                "id": model_id,
                "label": item.label.strip(),
                "model": item.model.strip(),
                "reasoning_effort": item.reasoning_effort,
                "anthropic_thinking": bool(item.anthropic_thinking),
                "anthropic_thinking_budget_tokens": item.anthropic_thinking_budget_tokens,
            }
        )

    cells: dict[str, dict] = {}
    for model in models:
        for method in methods:
            cell_id = f"{model['id']}__{method['id']}"
            cells[cell_id] = {
                "id": cell_id,
                "model_id": model["id"],
                "model_label": model["label"],
                "method_id": method["id"],
                "method_label": method["label"],
                "documents": {
                    doc_id: {"status": "pending", "updated_at": now} for doc_id in body.doc_ids
                },
            }

    return {
        "id": run_id,
        "name": (body.name or "").strip() or f"Methods Lab {now}",
        "status": "queued",
        "created_at": now,
        "started_at": None,
        "finished_at": None,
        "doc_ids": list(dict.fromkeys(body.doc_ids)),
        "methods": methods,
        "models": models,
        "runtime": {
            "temperature": runtime["temperature"],
            "match_mode": runtime["match_mode"],
            "reference_source": runtime["reference_source"],
            "fallback_reference_source": runtime["fallback_reference_source"],
            "label_profile": runtime["label_profile"],
            "label_projection": runtime["label_projection"],
            "method_bundle": runtime["method_bundle"],
            "api_base": runtime["api_base"],
            "chunk_mode": runtime["chunk_mode"],
            "chunk_size_chars": runtime["chunk_size_chars"],
            "task_timeout_seconds": runtime.get("task_timeout_seconds"),
        },
        "concurrency": body.concurrency,
        "warnings": [],
        "errors": [],
        "cells": cells,
        "session_id": session_id,
    }


def _run_methods_lab_job(run_id: str, session_id: str, runtime: dict[str, object]):
    api_key = str(runtime["api_key"])
    api_base = str(runtime["api_base"])
    temperature = float(runtime["temperature"])
    match_mode = str(runtime["match_mode"])
    label_profile = str(runtime["label_profile"])
    label_projection = _normalize_label_projection(runtime.get("label_projection", "native"))
    method_bundle = _normalize_method_bundle(runtime.get("method_bundle", "audited"))
    chunk_mode = str(runtime["chunk_mode"])
    chunk_size_chars = int(runtime["chunk_size_chars"])

    with _methods_lab_lock:
        run = _load_methods_lab_run(run_id, session_id)
        if run is None:
            return
        run["status"] = "running"
        run["started_at"] = _now_iso()
        _save_methods_lab_run(run, session_id)

    models = run.get("models", [])
    methods = run.get("methods", [])
    doc_ids = run.get("doc_ids", [])
    model_by_id = {str(model["id"]): model for model in models if isinstance(model, dict)}
    method_by_variant_id = {
        str(method["id"]): method for method in methods if isinstance(method, dict)
    }
    all_model_ids = [str(model.get("id", "")) for model in models if isinstance(model, dict)]

    tasks: list[tuple[str, str, str | None]] = []
    for method in methods:
        if not isinstance(method, dict):
            continue
        method_variant_id = str(method.get("id", ""))
        method_definition = get_method_definition_by_id(
            str(method.get("method_id", "")),
            method_bundle=method_bundle,
        )
        if method_definition is None:
            continue
        if bool(method_definition.get("uses_llm")):
            for model in models:
                if not isinstance(model, dict):
                    continue
                for doc_id in doc_ids:
                    tasks.append((method_variant_id, str(doc_id), str(model.get("id", ""))))
        else:
            for doc_id in doc_ids:
                tasks.append((method_variant_id, str(doc_id), None))

    def _execute_task(
        method_variant_id: str,
        doc_id: str,
        model_id: str | None,
    ) -> tuple[str, str, list[str], dict]:
        method_variant = method_by_variant_id.get(method_variant_id)
        target_model_ids = [model_id] if model_id else all_model_ids
        if method_variant is None:
            return (
                method_variant_id,
                doc_id,
                target_model_ids,
                {
                    "status": "failed",
                    "error": f"Unknown method variant '{method_variant_id}'",
                    "updated_at": _now_iso(),
                },
            )

        definition = get_method_definition_by_id(
            str(method_variant.get("method_id", "")),
            method_bundle=method_bundle,
        )
        if definition is None:
            return (
                method_variant_id,
                doc_id,
                target_model_ids,
                {
                    "status": "failed",
                    "error": f"Unknown method: {method_variant.get('method_id')}",
                    "updated_at": _now_iso(),
                },
            )

        doc = _load_doc(doc_id, session_id)
        if doc is None:
            return (
                method_variant_id,
                doc_id,
                target_model_ids,
                {
                    "status": "unavailable",
                    "error": "Document no longer exists",
                    "updated_at": _now_iso(),
                },
            )

        enriched = _enrich_doc(doc, session_id)
        if not enriched.manual_annotations:
            return (
                method_variant_id,
                doc_id,
                target_model_ids,
                {
                    "status": "unavailable",
                    "error": "Methods Lab requires manual annotations for scoring.",
                    "updated_at": _now_iso(),
                    "filename": enriched.filename,
                },
            )

        model = model_by_id.get(model_id) if model_id else None
        request_model = (
            str(model.get("model"))
            if isinstance(model, dict) and model.get("model")
            else "rule"
        )
        reasoning_effort = (
            str(model.get("reasoning_effort", "none"))
            if isinstance(model, dict)
            else "none"
        )
        anthropic_thinking = bool(model.get("anthropic_thinking")) if isinstance(model, dict) else False
        anthropic_budget = (
            model.get("anthropic_thinking_budget_tokens") if isinstance(model, dict) else None
        )

        try:
            (
                hypothesis_spans,
                warnings,
                llm_confidence,
                _chunk_diagnostics,
                raw_hypothesis_spans,
                resolution_events,
                resolution_policy_version,
                response_debug,
            ) = _run_method_for_document(
                doc=enriched,
                method_id=str(method_variant["method_id"]),
                api_key=api_key or None,
                api_base=api_base or None,
                model=request_model,
                system_prompt="",
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                anthropic_thinking=anthropic_thinking,
                anthropic_thinking_budget_tokens=anthropic_budget,
                method_verify=method_variant.get("method_verify_override"),
                label_profile=label_profile,  # type: ignore[arg-type]
                chunk_mode=chunk_mode,
                chunk_size_chars=chunk_size_chars,
                method_bundle=method_bundle,
            )
            projected_reference, projected_hypothesis = _prepare_experiment_scoring_spans(
                enriched.manual_annotations,
                hypothesis_spans,
                label_projection=label_projection,  # type: ignore[arg-type]
                method_bundle=method_bundle,
            )
            projected_reference_raw, projected_hypothesis_raw = _prepare_experiment_scoring_spans(
                enriched.manual_annotations,
                raw_hypothesis_spans,
                label_projection=label_projection,  # type: ignore[arg-type]
                method_bundle=method_bundle,
            )
            scoring_match_mode = _normalize_metrics_mode_for_method_bundle(
                match_mode,
                method_bundle=method_bundle,
            )
            metrics = _serialize_metrics_payload(
                compute_metrics(projected_reference, projected_hypothesis, scoring_match_mode)
            )
            raw_metrics = _serialize_metrics_payload(
                compute_metrics(
                    projected_reference_raw,
                    projected_hypothesis_raw,
                    scoring_match_mode,
                )
            )
            return (
                method_variant_id,
                doc_id,
                target_model_ids,
                {
                    "status": "completed",
                    "reference_source_used": "manual",
                    "reference_spans": [
                        span.model_dump() for span in enriched.manual_annotations
                    ],
                    "raw_hypothesis_spans": [
                        span.model_dump() for span in raw_hypothesis_spans
                    ],
                    "hypothesis_spans": [span.model_dump() for span in hypothesis_spans],
                    "raw_metrics": raw_metrics,
                    "metrics": metrics,
                    "warnings": warnings,
                    "llm_confidence": (
                        llm_confidence.model_dump() if llm_confidence is not None else None
                    ),
                    "response_debug": response_debug,
                    "resolution_events": [event.model_dump() for event in resolution_events],
                    "resolution_policy_version": resolution_policy_version,
                    "resolution_summary": summarize_resolution_events(resolution_events),
                    "updated_at": _now_iso(),
                    "filename": enriched.filename,
                },
            )
        except Exception as exc:
            message = str(exc).strip() or exc.__class__.__name__
            if len(message) > 800:
                message = f"{message[:800]}..."
            return (
                method_variant_id,
                doc_id,
                target_model_ids,
                {
                    "status": "failed",
                    "error": message,
                    "updated_at": _now_iso(),
                    "filename": enriched.filename,
                },
            )

    try:
        with ThreadPoolExecutor(max_workers=int(run.get("concurrency", 1))) as executor:
            future_map = {
                executor.submit(_execute_task, method_variant_id, doc_id, model_id): (
                    method_variant_id,
                    doc_id,
                    model_id,
                )
                for method_variant_id, doc_id, model_id in tasks
            }
            for future in as_completed(future_map):
                try:
                    method_variant_id, doc_id, target_model_ids, result = future.result()
                except Exception as exc:  # pragma: no cover - defensive fallback
                    method_variant_id, doc_id, model_id = future_map[future]
                    target_model_ids = [model_id] if model_id else all_model_ids
                    result = {
                        "status": "failed",
                        "error": str(exc),
                        "updated_at": _now_iso(),
                    }
                with _methods_lab_lock:
                    latest = _load_methods_lab_run(run_id, session_id)
                    if latest is None:
                        continue
                    cells = latest.get("cells", {})
                    if not isinstance(cells, dict):
                        continue
                    for target_model_id in target_model_ids:
                        cell_id = f"{target_model_id}__{method_variant_id}"
                        if isinstance(cells.get(cell_id), dict):
                            cells[cell_id]["documents"][doc_id] = copy.deepcopy(result)
                    _save_methods_lab_run(latest, session_id)
    except Exception as exc:
        with _methods_lab_lock:
            latest = _load_methods_lab_run(run_id, session_id)
            if latest is not None:
                latest["status"] = "failed"
                latest["finished_at"] = _now_iso()
                errors = latest.get("errors", [])
                if not isinstance(errors, list):
                    errors = []
                errors.append(f"Methods Lab run failed: {exc}")
                latest["errors"] = errors
                _save_methods_lab_run(latest, session_id)
        return

    with _methods_lab_lock:
        latest = _load_methods_lab_run(run_id, session_id)
        if latest is None:
            return
        summary = _build_methods_lab_run_summary(latest)
        latest["status"] = (
            "completed_with_errors"
            if int(summary.get("failed_tasks", 0)) > 0
            else "completed"
        )
        latest["finished_at"] = _now_iso()
        _save_methods_lab_run(latest, session_id)


@app.post("/api/documents/{doc_id}/agent")
def run_agent(doc_id: str, body: AgentRunBody):
    session_id = "default"
    doc = _load_doc(doc_id, session_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if body.mode == "rule":
        _start_agent_progress(doc_id, mode="rule", total_chunks=1, session_id=session_id)
        spans = _normalize_and_validate_spans(run_regex(doc.raw_text), doc.raw_text)
        _save_sidecar(doc_id, "agent.rule", spans, session_id)
        _save_json_sidecar(
            doc_id,
            "agent.last_run",
            {
                "mode": "rule",
                "label_profile": "simple",
                "updated_at": _now_iso(),
            },
            session_id,
        )
        _update_agent_progress(
            doc_id,
            completed_chunks=1,
            total_chunks=1,
            session_id=session_id,
        )
        _finish_agent_progress(doc_id, status="completed", session_id=session_id)
        enriched = _enrich_doc(doc, session_id)
        enriched.agent_run_warnings = []
        enriched.agent_run_metrics = AgentRunMetrics(
            llm_confidence=enriched.agent_run_metrics.llm_confidence,
            label_profile="simple",
        )
        return enriched

    if body.mode in ("llm", "openai"):
        llm_runtime = _resolve_llm_runtime_config(body)
        api_key = str(llm_runtime["api_key"])
        api_base = str(llm_runtime["api_base"])
        model = str(llm_runtime["model"])
        requested_system_prompt = str(llm_runtime["requested_system_prompt"])
        temperature = float(llm_runtime["temperature"])
        reasoning_effort = str(llm_runtime["reasoning_effort"])
        anthropic_thinking = bool(llm_runtime["anthropic_thinking"])
        anthropic_thinking_budget_tokens = llm_runtime["anthropic_thinking_budget_tokens"]
        label_profile = str(llm_runtime["label_profile"])
        chunk_mode = str(llm_runtime["chunk_mode"])
        chunk_size_chars = int(llm_runtime["chunk_size_chars"])
        total_chunks = _estimate_chunk_total(
            doc,
            chunk_mode=chunk_mode,
            chunk_size_chars=chunk_size_chars,
        )

        if not api_key:
            raise HTTPException(
                status_code=400,
                detail=(
                    "API key required. Set LITELLM_API_KEY (recommended for proxy/gateway) "
                    "or provider keys like OPENAI_API_KEY / ANTHROPIC_API_KEY, "
                    "or provide api_key in request."
                ),
            )
        _validate_gateway_model_access(model=model, api_base=api_base, api_key=api_key)

        _start_agent_progress(
            doc_id,
            mode="llm",
            total_chunks=total_chunks,
            session_id=session_id,
        )

        try:
            (
                spans,
                warnings,
                llm_confidence,
                chunk_diagnostics,
                raw_hypothesis_spans,
                resolution_events,
                resolution_policy_version,
                _response_debug,
            ) = _run_llm_for_document(
                doc=doc,
                api_key=api_key,
                api_base=api_base or None,
                model=model,
                system_prompt=requested_system_prompt,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                anthropic_thinking=anthropic_thinking,
                anthropic_thinking_budget_tokens=anthropic_thinking_budget_tokens,
                label_profile=label_profile,  # type: ignore[arg-type]
                chunk_mode=chunk_mode,
                chunk_size_chars=chunk_size_chars,
                progress_callback=lambda completed, total: _update_agent_progress(
                    doc_id,
                    completed_chunks=completed,
                    total_chunks=total,
                    session_id=session_id,
                ),
            )
        except Exception as exc:
            message = str(exc).strip() or exc.__class__.__name__
            if len(message) > 800:
                message = f"{message[:800]}..."
            detail = f"LLM request failed: {message}"
            if api_base:
                parsed = urlparse(api_base)
                path = parsed.path.rstrip("/")
                message_lower = message.lower()
                is_base_routing_issue = any(
                    marker in message_lower
                    for marker in ("not found", "404", "connection", "timeout", "refused")
                )
                if not path.endswith("/v1") and is_base_routing_issue:
                    detail += (
                        " Hint: this gateway may require an OpenAI-compatible '/v1' path; "
                        "the backend also retries with '/v1' automatically."
                    )
            _finish_agent_progress(
                doc_id,
                status="failed",
                session_id=session_id,
                message=detail,
            )
            raise HTTPException(status_code=502, detail=detail) from exc
        run_key = _build_agent_llm_run_key(model)
        _save_sidecar(doc_id, "agent.llm", spans, session_id)
        _delete_sidecar(doc_id, "agent.openai", session_id)
        _upsert_span_map_entry(
            doc_id,
            LLM_RUNS_SIDECAR_KIND,
            run_key,
            spans,
            session_id,
        )
        _upsert_run_metadata(
            doc_id,
            LLM_RUNS_METADATA_SIDECAR_KIND,
            run_key,
            SavedRunMetadata(
                mode="llm",
                updated_at=_now_iso(),
                model=model,
                label_profile=label_profile,  # type: ignore[arg-type]
                prompt_snapshot=_build_llm_prompt_snapshot(
                    requested_system_prompt,
                    label_profile=label_profile,  # type: ignore[arg-type]
                ),
                llm_confidence=llm_confidence,
                chunk_diagnostics=chunk_diagnostics,
                raw_hypothesis_spans=raw_hypothesis_spans,
                resolution_events=resolution_events,
                resolution_policy_version=resolution_policy_version,
            ),
            session_id,
        )
        _save_json_sidecar(
            doc_id,
            "agent.llm.metrics",
            llm_confidence.model_dump(),
            session_id,
        )
        _save_json_sidecar(
            doc_id,
            "agent.last_run",
            {
                "mode": "llm",
                "run_key": run_key,
                "model": model,
                "label_profile": label_profile,
                "updated_at": _now_iso(),
            },
            session_id,
        )
        _finish_agent_progress(doc_id, status="completed", session_id=session_id)
        enriched = _enrich_doc(doc, session_id)
        enriched.agent_run_warnings = warnings
        enriched.agent_run_metrics = AgentRunMetrics(
            llm_confidence=llm_confidence,
            label_profile=label_profile,  # type: ignore[arg-type]
            chunk_diagnostics=chunk_diagnostics,
            raw_hypothesis_spans=raw_hypothesis_spans,
            resolution_events=resolution_events,
            resolution_policy_version=resolution_policy_version,
        )
        return enriched

    if body.mode == "method":
        method_id = (body.method_id or "").strip()
        if not method_id:
            raise HTTPException(status_code=400, detail="method_id is required for mode='method'")

        method_definition = METHOD_DEFINITION_BY_ID.get(method_id)
        if method_definition is None:
            raise HTTPException(status_code=400, detail=f"Unknown method: {method_id}")

        llm_runtime = _resolve_llm_runtime_config(body)
        api_key = str(llm_runtime["api_key"])
        api_base = str(llm_runtime["api_base"])
        model = str(llm_runtime["model"])
        requested_system_prompt = str(llm_runtime["requested_system_prompt"])
        temperature = float(llm_runtime["temperature"])
        reasoning_effort = str(llm_runtime["reasoning_effort"])
        anthropic_thinking = bool(llm_runtime["anthropic_thinking"])
        anthropic_thinking_budget_tokens = llm_runtime["anthropic_thinking_budget_tokens"]
        label_profile = str(llm_runtime["label_profile"])
        chunk_mode = str(llm_runtime["chunk_mode"])
        chunk_size_chars = int(llm_runtime["chunk_size_chars"])
        total_chunks = _estimate_chunk_total(
            doc,
            chunk_mode=chunk_mode,
            chunk_size_chars=chunk_size_chars,
        )

        if method_definition["uses_llm"] and not api_key:
            raise HTTPException(
                status_code=400,
                detail=(
                    "API key required for this method. Set LITELLM_API_KEY "
                    "(recommended for proxy/gateway) or provide api_key in request."
                ),
            )
        if method_definition["uses_llm"]:
            _validate_gateway_model_access(model=model, api_base=api_base, api_key=api_key)
        _start_agent_progress(
            doc_id,
            mode="method",
            total_chunks=total_chunks,
            session_id=session_id,
        )

        try:
            (
                spans,
                warnings,
                llm_confidence,
                chunk_diagnostics,
                raw_hypothesis_spans,
                resolution_events,
                resolution_policy_version,
                _response_debug,
            ) = _run_method_for_document(
                doc=doc,
                method_id=method_id,
                api_key=api_key or None,
                api_base=api_base or None,
                model=model,
                system_prompt=requested_system_prompt,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                anthropic_thinking=anthropic_thinking,
                anthropic_thinking_budget_tokens=anthropic_thinking_budget_tokens,
                method_verify=body.method_verify,
                label_profile=label_profile,  # type: ignore[arg-type]
                chunk_mode=chunk_mode,
                chunk_size_chars=chunk_size_chars,
                progress_callback=lambda completed, total: _update_agent_progress(
                    doc_id,
                    completed_chunks=completed,
                    total_chunks=total,
                    session_id=session_id,
                ),
            )
        except Exception as exc:
            message = str(exc).strip() or exc.__class__.__name__
            if len(message) > 800:
                message = f"{message[:800]}..."
            status_code = 400
            if "LLM returned" in message or "request failed" in message.lower():
                status_code = 502
            _finish_agent_progress(
                doc_id,
                status="failed",
                session_id=session_id,
                message=message,
            )
            raise HTTPException(status_code=status_code, detail=message) from exc

        _save_sidecar(doc_id, f"agent.method.{method_id}", spans, session_id)
        method_run_key = f"{method_id}::{model if method_definition['uses_llm'] else 'rule'}"
        _upsert_span_map_entry(
            doc_id,
            METHOD_RUNS_SIDECAR_KIND,
            method_run_key,
            spans,
            session_id,
        )
        _upsert_run_metadata(
            doc_id,
            METHOD_RUNS_METADATA_SIDECAR_KIND,
            method_run_key,
            SavedRunMetadata(
                mode="method",
                updated_at=_now_iso(),
                method_id=method_id,
                model=model if method_definition["uses_llm"] else None,
                label_profile=label_profile,  # type: ignore[arg-type]
                prompt_snapshot=_build_method_prompt_snapshot(
                    method_id=method_id,
                    additional_constraints=requested_system_prompt,
                    method_verify=body.method_verify,
                    label_profile=label_profile,  # type: ignore[arg-type]
                ),
                llm_confidence=llm_confidence,
                chunk_diagnostics=chunk_diagnostics,
                raw_hypothesis_spans=raw_hypothesis_spans,
                resolution_events=resolution_events,
                resolution_policy_version=resolution_policy_version,
            ),
            session_id,
        )
        _save_json_sidecar(
            doc_id,
            "agent.last_run",
            {
                "mode": "method",
                "run_key": method_run_key,
                "method_id": method_id,
                "model": model if method_definition["uses_llm"] else None,
                "label_profile": label_profile,
                "updated_at": _now_iso(),
            },
            session_id,
        )
        _finish_agent_progress(doc_id, status="completed", session_id=session_id)
        enriched = _enrich_doc(doc, session_id)
        enriched.agent_run_warnings = warnings
        enriched.agent_run_metrics = AgentRunMetrics(
            llm_confidence=llm_confidence,
            label_profile=label_profile,  # type: ignore[arg-type]
            chunk_diagnostics=chunk_diagnostics,
            raw_hypothesis_spans=raw_hypothesis_spans,
            resolution_events=resolution_events,
            resolution_policy_version=resolution_policy_version,
        )
        return enriched

    raise HTTPException(status_code=400, detail=f"Unknown agent mode: {body.mode}")


@app.get("/api/documents/{doc_id}/agent/progress")
async def get_agent_run_progress(doc_id: str):
    session_id = "default"
    doc = _load_doc(doc_id, session_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return _get_agent_progress(doc_id, session_id)


@app.post("/api/prompt-lab/runs")
async def create_prompt_lab_run(body: PromptLabRunCreateBody):
    from experiment_service import create_prompt_lab_run as create_prompt_lab_run_service

    return create_prompt_lab_run_service(body, session_id="default", run_async=True)


@app.get("/api/prompt-lab/runs")
async def list_prompt_lab_runs():
    session_id = "default"
    with _prompt_lab_lock:
        ids = list(_load_prompt_lab_index(session_id))
        results: list[dict] = []
        for run_id in reversed(ids):
            run = _load_prompt_lab_run(run_id, session_id)
            if run is None:
                continue
            results.append(_build_prompt_lab_run_summary(run))
        return {"runs": results}


@app.get("/api/prompt-lab/runs/{run_id}")
async def get_prompt_lab_run(run_id: str):
    session_id = "default"
    with _prompt_lab_lock:
        run = _load_prompt_lab_run(run_id, session_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Prompt Lab run not found")
        return _build_prompt_lab_run_detail(run)


@app.get("/api/prompt-lab/runs/{run_id}/cells/{cell_id}/documents/{doc_id}")
async def get_prompt_lab_document_detail(run_id: str, cell_id: str, doc_id: str):
    session_id = "default"
    with _prompt_lab_lock:
        run = _load_prompt_lab_run(run_id, session_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Prompt Lab run not found")
        cells = run.get("cells", {})
        if not isinstance(cells, dict):
            raise HTTPException(status_code=404, detail="Prompt Lab cell not found")
        cell = cells.get(cell_id)
        if not isinstance(cell, dict):
            raise HTTPException(status_code=404, detail="Prompt Lab cell not found")
        documents = cell.get("documents", {})
        if not isinstance(documents, dict):
            raise HTTPException(status_code=404, detail="Prompt Lab document result not found")
        result = documents.get(doc_id)
        if not isinstance(result, dict):
            raise HTTPException(status_code=404, detail="Prompt Lab document result not found")
        model = next(
            (item for item in run.get("models", []) if str(item.get("id")) == str(cell.get("model_id"))),
            None,
        )
        prompt = next(
            (item for item in run.get("prompts", []) if str(item.get("id")) == str(cell.get("prompt_id"))),
            None,
        )
        stored = copy.deepcopy(result)

    doc = _load_doc(doc_id, session_id)
    enriched = _enrich_doc(doc, session_id) if doc is not None else None
    if enriched is None:
        warnings = stored.get("warnings", [])
        if not isinstance(warnings, list):
            warnings = []
        warnings.append("Referenced document is unavailable in current session.")
        stored["warnings"] = warnings

    return {
        "run_id": run_id,
        "cell_id": cell_id,
        "doc_id": doc_id,
        "status": stored.get("status", "pending"),
        "error": stored.get("error"),
        "error_family": stored.get("error_family")
        or _normalize_error_family(cast(str | None, stored.get("error"))),
        "warnings": stored.get("warnings", []),
        "runtime_diagnostics": stored.get("runtime_diagnostics"),
        "reference_source_used": stored.get("reference_source_used"),
        "reference_spans": stored.get("reference_spans", []),
        "raw_hypothesis_spans": stored.get("raw_hypothesis_spans", []),
        "hypothesis_spans": stored.get("hypothesis_spans", []),
        "raw_metrics": stored.get("raw_metrics"),
        "metrics": stored.get("metrics"),
        "llm_confidence": stored.get("llm_confidence"),
        "response_debug": stored.get("response_debug", []),
        "resolution_events": stored.get("resolution_events", []),
        "resolution_policy_version": stored.get("resolution_policy_version"),
        "resolution_summary": stored.get("resolution_summary"),
        "transcript_text": enriched.raw_text if enriched is not None else None,
        "document": {
            "id": doc_id,
            "filename": enriched.filename if enriched is not None else stored.get("filename"),
        },
        "model": model,
        "prompt": prompt,
    }


@app.post("/api/prompt-lab/runs/{run_id}/cancel")
async def cancel_prompt_lab_run(run_id: str) -> dict[str, object]:
    session_id = "default"
    with _prompt_lab_lock:
        return _request_prompt_lab_cancel(run_id, session_id)


@app.delete("/api/prompt-lab/runs/{run_id}")
async def delete_prompt_lab_run(run_id: str) -> dict[str, object]:
    session_id = "default"
    with _prompt_lab_lock:
        deleted = _delete_prompt_lab_run(run_id, session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"ok": True, "id": run_id}


@app.post("/api/methods-lab/runs")
async def create_methods_lab_run(body: MethodsLabRunCreateBody):
    from experiment_service import create_methods_lab_run as create_methods_lab_run_service

    return create_methods_lab_run_service(body, session_id="default", run_async=True)


@app.get("/api/methods-lab/runs")
async def list_methods_lab_runs():
    session_id = "default"
    with _methods_lab_lock:
        ids = list(_load_methods_lab_index(session_id))
        results: list[dict] = []
        for run_id in reversed(ids):
            run = _load_methods_lab_run(run_id, session_id)
            if run is None:
                continue
            results.append(_build_methods_lab_run_summary(run))
        return {"runs": results}


@app.get("/api/methods-lab/runs/{run_id}")
async def get_methods_lab_run(run_id: str):
    session_id = "default"
    with _methods_lab_lock:
        run = _load_methods_lab_run(run_id, session_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Methods Lab run not found")
        return _build_methods_lab_run_detail(run)


@app.get("/api/methods-lab/runs/{run_id}/cells/{cell_id}/documents/{doc_id}")
async def get_methods_lab_document_detail(run_id: str, cell_id: str, doc_id: str):
    session_id = "default"
    with _methods_lab_lock:
        run = _load_methods_lab_run(run_id, session_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Methods Lab run not found")
        cells = run.get("cells", {})
        if not isinstance(cells, dict):
            raise HTTPException(status_code=404, detail="Methods Lab cell not found")
        cell = cells.get(cell_id)
        if not isinstance(cell, dict):
            raise HTTPException(status_code=404, detail="Methods Lab cell not found")
        documents = cell.get("documents", {})
        if not isinstance(documents, dict):
            raise HTTPException(status_code=404, detail="Methods Lab document result not found")
        result = documents.get(doc_id)
        if not isinstance(result, dict):
            raise HTTPException(status_code=404, detail="Methods Lab document result not found")
        model = next(
            (item for item in run.get("models", []) if str(item.get("id")) == str(cell.get("model_id"))),
            None,
        )
        method = next(
            (item for item in run.get("methods", []) if str(item.get("id")) == str(cell.get("method_id"))),
            None,
        )
        stored = copy.deepcopy(result)

    doc = _load_doc(doc_id, session_id)
    enriched = _enrich_doc(doc, session_id) if doc is not None else None
    if enriched is None:
        warnings = stored.get("warnings", [])
        if not isinstance(warnings, list):
            warnings = []
        warnings.append("Referenced document is unavailable in current session.")
        stored["warnings"] = warnings

    return {
        "run_id": run_id,
        "cell_id": cell_id,
        "doc_id": doc_id,
        "status": stored.get("status", "pending"),
        "error": stored.get("error"),
        "error_family": stored.get("error_family")
        or _normalize_error_family(cast(str | None, stored.get("error"))),
        "warnings": stored.get("warnings", []),
        "runtime_diagnostics": stored.get("runtime_diagnostics"),
        "reference_source_used": stored.get("reference_source_used"),
        "reference_spans": stored.get("reference_spans", []),
        "raw_hypothesis_spans": stored.get("raw_hypothesis_spans", []),
        "hypothesis_spans": stored.get("hypothesis_spans", []),
        "raw_metrics": stored.get("raw_metrics"),
        "metrics": stored.get("metrics"),
        "llm_confidence": stored.get("llm_confidence"),
        "response_debug": stored.get("response_debug", []),
        "resolution_events": stored.get("resolution_events", []),
        "resolution_policy_version": stored.get("resolution_policy_version"),
        "resolution_summary": stored.get("resolution_summary"),
        "transcript_text": enriched.raw_text if enriched is not None else None,
        "document": {
            "id": doc_id,
            "filename": enriched.filename if enriched is not None else stored.get("filename"),
        },
        "model": model,
        "method": method,
    }


@app.post("/api/methods-lab/runs/{run_id}/cancel")
async def cancel_methods_lab_run(run_id: str) -> dict[str, object]:
    session_id = "default"
    with _methods_lab_lock:
        return _request_methods_lab_cancel(run_id, session_id)


@app.delete("/api/methods-lab/runs/{run_id}")
async def delete_methods_lab_run(run_id: str) -> dict[str, object]:
    session_id = "default"
    with _methods_lab_lock:
        deleted = _delete_methods_lab_run(run_id, session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Run not found")
    return {"ok": True, "id": run_id}


@app.get("/api/documents/{doc_id}/metrics")
async def get_metrics(
    doc_id: str,
    reference: str = Query(...),
    hypothesis: str = Query(...),
    match_mode: str = Query("overlap"),
    label_projection: str = Query("native"),
):
    session_id = "default"
    doc = _load_doc(doc_id, session_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    enriched = _enrich_doc(doc, session_id)
    ref_spans = _spans_from_source(enriched, reference)
    hyp_spans = _spans_from_source(enriched, hypothesis)
    normalized_match_mode = _normalize_match_mode(match_mode)
    normalized_projection = _normalize_label_projection(label_projection)
    eval_ref_spans, eval_hyp_spans = _apply_label_projection(
        ref_spans,
        hyp_spans,
        label_projection=normalized_projection,
    )

    result = _serialize_metrics_payload(
        compute_metrics(eval_ref_spans, eval_hyp_spans, normalized_match_mode)
    )
    llm_confidence = _resolve_metrics_llm_confidence(enriched, reference, hypothesis)
    result["llm_confidence"] = (
        llm_confidence.model_dump()
        if llm_confidence is not None
        else None
    )
    result["match_mode"] = normalized_match_mode
    result["label_projection"] = normalized_projection
    return result


@app.get("/api/session/export")
async def export_session_bundle():
    session_id = "default"
    ids = _load_session_index(session_id)
    folder_records = _load_all_folders(session_id)
    doc_ids_to_export: list[str] = []
    seen_doc_ids: set[str] = set()

    def _append_export_doc_id(doc_id: str | None) -> None:
        if not doc_id:
            return
        normalized = str(doc_id).strip()
        if not normalized or normalized in seen_doc_ids:
            return
        seen_doc_ids.add(normalized)
        doc_ids_to_export.append(normalized)

    for did in ids:
        _append_export_doc_id(did)
    for folder in folder_records:
        _append_export_doc_id(folder.merged_doc_id)
        for doc_id in folder.doc_ids:
            _append_export_doc_id(doc_id)

    documents: list[dict] = []
    for did in doc_ids_to_export:
        doc = _load_doc(did, session_id)
        if not doc:
            continue
        manual = _load_sidecar(did, "manual", session_id) or []
        agent_rule = _load_sidecar(did, "agent.rule", session_id) or []
        agent_llm = _load_sidecar(did, "agent.llm", session_id) or []
        agent_methods = _load_method_sidecars(did, session_id)
        agent_llm_runs = _load_span_map_sidecar(did, LLM_RUNS_SIDECAR_KIND, session_id)
        agent_method_runs = _load_span_map_sidecar(did, METHOD_RUNS_SIDECAR_KIND, session_id)
        agent_llm_run_meta = _load_run_metadata_map_sidecar(
            did,
            LLM_RUNS_METADATA_SIDECAR_KIND,
            session_id,
        )
        agent_method_run_meta = _load_run_metadata_map_sidecar(
            did,
            METHOD_RUNS_METADATA_SIDECAR_KIND,
            session_id,
        )
        llm_confidence = _load_json_sidecar(did, "agent.llm.metrics", session_id)
        last_run = _load_json_sidecar(did, "agent.last_run", session_id)
        label_profile: str | None = None
        if isinstance(last_run, dict):
            candidate = str(last_run.get("label_profile", "")).strip().lower()
            if candidate in {"simple", "advanced"}:
                label_profile = candidate
        export_item = {
            "source": doc.model_dump(),
            "manual_annotations": [span.model_dump() for span in manual],
            "agent_outputs": {
                "rule": [span.model_dump() for span in agent_rule],
                "llm": [span.model_dump() for span in agent_llm],
                "methods": {
                    method_id: [span.model_dump() for span in spans]
                    for method_id, spans in agent_methods.items()
                },
            },
        }
        if agent_llm_runs or agent_method_runs or agent_llm_run_meta or agent_method_run_meta:
            export_item["agent_saved_outputs"] = {
                "llm_runs": {
                    model_key: [span.model_dump() for span in spans]
                    for model_key, spans in agent_llm_runs.items()
                },
                "method_runs": {
                    run_key: [span.model_dump() for span in spans]
                    for run_key, spans in agent_method_runs.items()
                },
                "llm_run_metadata": {
                    model_key: metadata.model_dump()
                    for model_key, metadata in agent_llm_run_meta.items()
                },
                "method_run_metadata": {
                    run_key: metadata.model_dump()
                    for run_key, metadata in agent_method_run_meta.items()
                },
            }
        if llm_confidence is not None or label_profile is not None:
            metrics_payload: dict[str, object] = {}
            if llm_confidence is not None:
                metrics_payload["llm_confidence"] = llm_confidence
            if label_profile is not None:
                metrics_payload["label_profile"] = label_profile
            export_item["agent_metrics"] = metrics_payload
        documents.append(export_item)

    return {
        "format": BUNDLE_FORMAT,
        "version": BUNDLE_VERSION,
        "compatibility": {
            "tool_version": TOOL_VERSION,
            "import_supported_versions": sorted(SUPPORTED_BUNDLE_VERSIONS),
        },
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "documents": documents,
        "folders": [folder.model_dump() for folder in folder_records],
        "prompt_lab_runs": _export_prompt_lab_runs(session_id),
        "methods_lab_runs": _export_methods_lab_runs(session_id),
        "config": _load_config(),
    }


@app.get("/api/session/export-ground-truth")
async def export_ground_truth_only(
    source: Annotated[str, Query()] = "manual",
    scope: Annotated[Literal["top_level", "folder"], Query()] = "top_level",
    folder_id: Annotated[str | None, Query()] = None,
):
    session_id = "default"
    ids = _resolve_ground_truth_export_doc_ids(scope, folder_id, session_id)
    buffer = io.BytesIO()
    source_value = source.strip()
    if not source_value:
        raise HTTPException(status_code=400, detail="source is required")

    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for did in ids:
            doc = _load_doc(did, session_id)
            if doc is None:
                continue
            enriched = _enrich_doc(doc, session_id)
            spans = _spans_from_source(enriched, source_value)
            payload = {
                "id": enriched.id,
                "filename": enriched.filename,
                "transcript": enriched.raw_text,
                "ground_truth_source": source_value,
                "spans": [span.model_dump() for span in spans],
                "pii_occurrences": _spans_to_pii_occurrences(spans),
            }
            stem = Path(enriched.filename).stem or enriched.id
            safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or enriched.id
            member_name = f"{safe_stem}.{enriched.id}.ground_truth.json"
            archive.writestr(member_name, json.dumps(payload, indent=2))

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_source = re.sub(r"[^A-Za-z0-9._-]+", "_", source_value).strip("._-") or "source"
    filename = f"ground-truth-{safe_source}-{stamp}.zip"
    return Response(
        content=buffer.getvalue(),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )


def _decode_json_object_payload(raw: bytes, *, invalid_detail: str) -> dict[str, object]:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=invalid_detail) from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Import payload must be a JSON object")
    return cast(dict[str, object], payload)


def _looks_like_session_bundle_payload(payload: dict[str, object]) -> bool:
    if "format" in payload or "documents" in payload:
        return True
    return any(
        key in payload
        for key in ("folders", "prompt_lab_runs", "methods_lab_runs", "project", "compatibility")
    )


def _resolve_ingest_mode(
    raw: bytes,
    filename: str,
) -> tuple[Literal["upload", "import"], dict[str, object] | None]:
    lower_filename = filename.lower()
    if zipfile.is_zipfile(io.BytesIO(raw)):
        return "import", None
    if lower_filename.endswith(".jsonl") or lower_filename.endswith(".txt"):
        return "upload", None

    payload = _decode_json_object_payload(
        raw,
        invalid_detail="Ingest file must be valid UTF-8 JSON, JSONL, or a ground-truth ZIP archive",
    )
    if _looks_like_ground_truth_payload(payload) or _looks_like_session_bundle_payload(payload):
        return "import", payload
    return "upload", payload


def _import_session_payload(
    raw: bytes,
    upload_filename: str,
    session_id: str = "default",
    payload: dict[str, object] | None = None,
    *,
    conflict_policy: ImportConflictPolicy = "replace",
) -> dict[str, object]:
    if zipfile.is_zipfile(io.BytesIO(raw)):
        return _import_ground_truth_archive(raw, session_id, conflict_policy=conflict_policy)
    if payload is None:
        payload = _decode_json_object_payload(
            raw,
            invalid_detail="Import file must be valid UTF-8 JSON or a ground-truth ZIP archive",
        )

    if _looks_like_ground_truth_payload(payload):
        ids = _load_session_index(session_id)
        existing_ids = set(ids)
        try:
            commit_result = _import_ground_truth_payload(
                payload=payload,
                raw=raw,
                upload_filename=upload_filename,
                session_id=session_id,
                ids=ids,
                existing_ids=existing_ids,
                conflict_policy=conflict_policy,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        _save_session_index(session_id)
        return {
            "bundle_version": None,
            "imported_count": 1,
            "imported_ids": [commit_result.doc_id],
            "created_count": 1 if commit_result.created else 0,
            "created_ids": [commit_result.doc_id] if commit_result.created else [],
            "conflict_policy": conflict_policy,
            "conflict_count": 1 if commit_result.conflict_action is not None else 0,
            "replaced_count": 1 if commit_result.conflict_action == "replace" else 0,
            "kept_current_count": 1 if commit_result.conflict_action == "keep_current" else 0,
            "added_as_new_count": 1 if commit_result.conflict_action == "add_new" else 0,
            "skipped_count": 0,
            "skipped": [],
            "warnings": [],
            "imported_prompt_lab_runs": 0,
            "imported_methods_lab_runs": 0,
            "total_in_bundle": 1,
        }

    bundle_version = _resolve_bundle_version(payload)
    if bundle_version not in SUPPORTED_BUNDLE_VERSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported bundle version '{bundle_version}'. "
                f"Supported versions: {sorted(SUPPORTED_BUNDLE_VERSIONS)}"
            ),
        )
    if bundle_version > BUNDLE_VERSION:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Bundle version {bundle_version} is newer than this tool "
                f"(max supported version is {BUNDLE_VERSION})."
            ),
        )

    raw_documents = payload.get("documents")
    if not isinstance(raw_documents, list):
        raise HTTPException(
            status_code=400, detail="Import payload must contain a 'documents' array"
        )

    ids = _load_session_index(session_id)
    existing_ids = set(ids)
    imported_ids: list[str] = []
    created_ids: list[str] = []
    doc_id_remap: dict[str, str] = {}
    skipped: list[dict[str, str | int]] = []
    warnings: list[str] = []
    conflict_counts = _empty_import_conflict_counts()
    raw_folders = payload.get("folders", [])
    hidden_original_doc_ids: set[str] = set()
    if raw_folders is None:
        raw_folders = []
    if not isinstance(raw_folders, list):
        warnings.append("Ignored folders because it is not an array.")
        raw_folders = []
    else:
        for folder_item in raw_folders:
            if not isinstance(folder_item, dict):
                continue
            folder_doc_ids = folder_item.get("doc_ids", [])
            if not isinstance(folder_doc_ids, list):
                continue
            hidden_original_doc_ids.update(
                str(doc_id).strip() for doc_id in folder_doc_ids if str(doc_id).strip()
            )

    for idx, item in enumerate(raw_documents):
        if not isinstance(item, dict):
            skipped.append({"index": idx, "reason": "Item is not an object"})
            continue

        source_raw = item.get("source", item)
        if not isinstance(source_raw, dict):
            skipped.append({"index": idx, "reason": "Missing document source object"})
            continue

        try:
            doc = CanonicalDocument.model_validate(source_raw)
        except Exception as exc:  # pragma: no cover - message can vary
            skipped.append({"index": idx, "reason": f"Invalid source document: {exc}"})
            continue
        original_doc_id = doc.id

        manual_raw = item.get("manual_annotations", source_raw.get("manual_annotations"))
        agent_outputs = item.get("agent_outputs", source_raw.get("agent_outputs", {}))
        if not isinstance(agent_outputs, dict):
            skipped.append({"index": idx, "reason": "agent_outputs must be an object"})
            continue
        agent_saved_outputs = item.get(
            "agent_saved_outputs",
            source_raw.get("agent_saved_outputs", {}),
        )
        if agent_saved_outputs is not None and not isinstance(agent_saved_outputs, dict):
            skipped.append({"index": idx, "reason": "agent_saved_outputs must be an object"})
            continue
        agent_metrics = item.get("agent_metrics", source_raw.get("agent_run_metrics", {}))
        if agent_metrics is not None and not isinstance(agent_metrics, dict):
            skipped.append({"index": idx, "reason": "agent_metrics must be an object"})
            continue

        try:
            manual_spans = _normalize_optional_spans(manual_raw, doc.raw_text)
            rule_spans = _normalize_optional_spans(agent_outputs.get("rule"), doc.raw_text)
            llm_spans = _normalize_optional_spans(agent_outputs.get("llm"), doc.raw_text)
            methods_raw = agent_outputs.get("methods", {})
            if methods_raw is None:
                methods_raw = {}
            if not isinstance(methods_raw, dict):
                raise ValueError("agent_outputs.methods must be an object")
            method_spans: dict[str, list[CanonicalSpan]] = {}
            for method_id, method_items in methods_raw.items():
                if not isinstance(method_id, str) or method_id.strip() == "":
                    raise ValueError("Method output keys must be non-empty strings")
                normalized_method = _normalize_optional_spans(method_items, doc.raw_text)
                if normalized_method:
                    method_spans[method_id] = normalized_method
            llm_runs_raw = {}
            method_runs_raw = {}
            llm_run_meta_raw = {}
            method_run_meta_raw = {}
            if isinstance(agent_saved_outputs, dict):
                llm_runs_raw = agent_saved_outputs.get("llm_runs", {})
                method_runs_raw = agent_saved_outputs.get("method_runs", {})
                llm_run_meta_raw = agent_saved_outputs.get("llm_run_metadata", {})
                method_run_meta_raw = agent_saved_outputs.get("method_run_metadata", {})
            if llm_runs_raw is None:
                llm_runs_raw = {}
            if method_runs_raw is None:
                method_runs_raw = {}
            if llm_run_meta_raw is None:
                llm_run_meta_raw = {}
            if method_run_meta_raw is None:
                method_run_meta_raw = {}
            if not isinstance(llm_runs_raw, dict):
                raise ValueError("agent_saved_outputs.llm_runs must be an object")
            if not isinstance(method_runs_raw, dict):
                raise ValueError("agent_saved_outputs.method_runs must be an object")
            if not isinstance(llm_run_meta_raw, dict):
                raise ValueError("agent_saved_outputs.llm_run_metadata must be an object")
            if not isinstance(method_run_meta_raw, dict):
                raise ValueError("agent_saved_outputs.method_run_metadata must be an object")
            llm_runs: dict[str, list[CanonicalSpan]] = {}
            for run_key, run_items in llm_runs_raw.items():
                if not isinstance(run_key, str) or run_key.strip() == "":
                    raise ValueError("agent_saved_outputs.llm_runs keys must be non-empty strings")
                normalized_runs = _normalize_optional_spans(run_items, doc.raw_text)
                if normalized_runs:
                    llm_runs[run_key] = normalized_runs
            method_runs: dict[str, list[CanonicalSpan]] = {}
            for run_key, run_items in method_runs_raw.items():
                if not isinstance(run_key, str) or run_key.strip() == "":
                    raise ValueError(
                        "agent_saved_outputs.method_runs keys must be non-empty strings"
                    )
                normalized_runs = _normalize_optional_spans(run_items, doc.raw_text)
                if normalized_runs:
                    method_runs[run_key] = normalized_runs
            llm_run_metadata: dict[str, SavedRunMetadata] = {}
            for run_key, raw_meta in llm_run_meta_raw.items():
                if not isinstance(run_key, str) or run_key.strip() == "":
                    raise ValueError(
                        "agent_saved_outputs.llm_run_metadata keys must be non-empty strings"
                    )
                try:
                    llm_run_metadata[run_key] = SavedRunMetadata.model_validate(raw_meta)
                except Exception as exc:
                    raise ValueError(
                        f"Invalid metadata for agent_saved_outputs.llm_run_metadata['{run_key}']: {exc}"
                    ) from exc
            method_run_metadata: dict[str, SavedRunMetadata] = {}
            for run_key, raw_meta in method_run_meta_raw.items():
                if not isinstance(run_key, str) or run_key.strip() == "":
                    raise ValueError(
                        "agent_saved_outputs.method_run_metadata keys must be non-empty strings"
                    )
                try:
                    method_run_metadata[run_key] = SavedRunMetadata.model_validate(raw_meta)
                except Exception as exc:
                    raise ValueError(
                        f"Invalid metadata for agent_saved_outputs.method_run_metadata['{run_key}']: {exc}"
                    ) from exc
            llm_confidence = _normalize_optional_llm_confidence(
                agent_metrics.get("llm_confidence") if isinstance(agent_metrics, dict) else None
            )
            label_profile_raw = (
                str(agent_metrics.get("label_profile", "")).strip().lower()
                if isinstance(agent_metrics, dict)
                else ""
            )
            imported_label_profile: Literal["simple", "advanced"] | None = None
            if label_profile_raw in {"simple", "advanced"}:
                imported_label_profile = label_profile_raw  # type: ignore[assignment]
        except ValueError as exc:
            skipped.append({"index": idx, "reason": str(exc)})
            continue

        commit_result = _commit_imported_document(
            doc=doc,
            session_id=session_id,
            ids=ids,
            existing_ids=existing_ids,
            add_to_session_index=original_doc_id not in hidden_original_doc_ids,
            manual_spans=manual_spans,
            rule_spans=rule_spans,
            llm_spans=llm_spans,
            method_spans=method_spans,
            llm_runs=llm_runs,
            method_runs=method_runs,
            llm_run_metadata=llm_run_metadata,
            method_run_metadata=method_run_metadata,
            llm_confidence=llm_confidence,
            imported_label_profile=imported_label_profile,
            conflict_policy=conflict_policy,
        )
        imported_ids.append(commit_result.doc_id)
        if commit_result.created:
            created_ids.append(commit_result.doc_id)
        _record_import_conflict(conflict_counts, commit_result)
        doc_id_remap[original_doc_id] = commit_result.doc_id

    _save_session_index(session_id)

    imported_folder_ids: list[str] = []
    folder_id_remap: dict[str, str] = {}
    if raw_folders:
        existing_folder_ids = set(_load_folder_index(session_id))
        provisional_folders: list[tuple[str, dict]] = []
        for index, raw_folder in enumerate(raw_folders):
            if not isinstance(raw_folder, dict):
                warnings.append(f"Skipped malformed folders item at index {index} (not an object).")
                continue
            original_folder_id = str(raw_folder.get("id") or "").strip()
            if not original_folder_id:
                warnings.append(f"Skipped folders item at index {index} without an id.")
                continue
            new_folder_id = original_folder_id
            while new_folder_id in existing_folder_ids or new_folder_id in folder_id_remap.values():
                new_folder_id = str(uuid.uuid4())[:8]
            folder_id_remap[original_folder_id] = new_folder_id
            provisional_folders.append((original_folder_id, raw_folder))

        folder_index = _load_folder_index(session_id)
        for original_folder_id, raw_folder in provisional_folders:
            mapped_doc_ids: list[str] = []
            for raw_doc_id in raw_folder.get("doc_ids", []):
                mapped_doc_id = doc_id_remap.get(str(raw_doc_id).strip())
                if mapped_doc_id:
                    mapped_doc_ids.append(mapped_doc_id)
            mapped_child_folder_ids: list[str] = []
            for raw_child_id in raw_folder.get("child_folder_ids", []):
                mapped_child_id = folder_id_remap.get(str(raw_child_id).strip())
                if mapped_child_id:
                    mapped_child_folder_ids.append(mapped_child_id)
            raw_display_names = raw_folder.get("doc_display_names", {})
            doc_display_names: dict[str, str] = {}
            if isinstance(raw_display_names, dict):
                for raw_doc_id, raw_display_name in raw_display_names.items():
                    mapped_doc_id = doc_id_remap.get(str(raw_doc_id).strip())
                    if mapped_doc_id and isinstance(raw_display_name, str):
                        doc_display_names[mapped_doc_id] = raw_display_name

            folder = FolderRecord(
                id=folder_id_remap[original_folder_id],
                name=str(raw_folder.get("name") or folder_id_remap[original_folder_id]),
                kind=str(raw_folder.get("kind") or "import"),
                parent_folder_id=folder_id_remap.get(
                    str(raw_folder.get("parent_folder_id") or "").strip()
                )
                or None,
                merged_doc_id=doc_id_remap.get(str(raw_folder.get("merged_doc_id") or "").strip())
                or None,
                doc_ids=list(dict.fromkeys(mapped_doc_ids)),
                child_folder_ids=list(dict.fromkeys(mapped_child_folder_ids)),
                source_filename=(
                    str(raw_folder.get("source_filename")).strip()
                    if raw_folder.get("source_filename") is not None
                    else None
                ),
                source_folder_id=folder_id_remap.get(
                    str(raw_folder.get("source_folder_id") or "").strip()
                )
                or None,
                sample_size=(
                    int(raw_folder.get("sample_size"))
                    if isinstance(raw_folder.get("sample_size"), int)
                    else None
                ),
                sample_seed=(
                    int(raw_folder.get("sample_seed"))
                    if isinstance(raw_folder.get("sample_seed"), int)
                    else None
                ),
                created_at=str(raw_folder.get("created_at") or _now_iso()),
                doc_display_names=doc_display_names,
            )
            _save_folder(folder, session_id)
            folder_index.append(folder.id)
            imported_folder_ids.append(folder.id)
        _save_folder_index(folder_index, session_id)

    raw_prompt_lab_runs = payload.get("prompt_lab_runs", [])
    imported_prompt_lab_runs = 0
    if raw_prompt_lab_runs is not None:
        if not isinstance(raw_prompt_lab_runs, list):
            warnings.append("Ignored prompt_lab_runs because it is not an array.")
        else:
            with _prompt_lab_lock:
                prompt_run_ids = _load_prompt_lab_index(session_id)
                existing_run_ids = set(prompt_run_ids)
                for run_item in raw_prompt_lab_runs:
                    if not isinstance(run_item, dict):
                        warnings.append("Skipped malformed prompt_lab_runs item (not an object).")
                        continue
                    remapped_run, remap_warnings = _remap_prompt_lab_run_doc_ids(
                        run_item,
                        doc_id_remap,
                        folder_id_remap,
                    )
                    warnings.extend(remap_warnings)
                    run_id = str(remapped_run.get("id") or str(uuid.uuid4())[:8])
                    while run_id in existing_run_ids:
                        run_id = str(uuid.uuid4())[:8]
                    remapped_run["id"] = run_id
                    _save_prompt_lab_run(remapped_run, session_id)
                    prompt_run_ids.append(run_id)
                    existing_run_ids.add(run_id)
                    imported_prompt_lab_runs += 1
                _save_prompt_lab_index(session_id)

    raw_methods_lab_runs = payload.get("methods_lab_runs", [])
    imported_methods_lab_runs = 0
    if raw_methods_lab_runs is not None:
        if not isinstance(raw_methods_lab_runs, list):
            warnings.append("Ignored methods_lab_runs because it is not an array.")
        else:
            with _methods_lab_lock:
                methods_run_ids = _load_methods_lab_index(session_id)
                existing_run_ids = set(methods_run_ids)
                for run_item in raw_methods_lab_runs:
                    if not isinstance(run_item, dict):
                        warnings.append("Skipped malformed methods_lab_runs item (not an object).")
                        continue
                    remapped_run, remap_warnings = _remap_methods_lab_run_doc_ids(
                        run_item,
                        doc_id_remap,
                        folder_id_remap,
                    )
                    warnings.extend(remap_warnings)
                    run_id = str(remapped_run.get("id") or str(uuid.uuid4())[:8])
                    while run_id in existing_run_ids:
                        run_id = str(uuid.uuid4())[:8]
                    remapped_run["id"] = run_id
                    _save_methods_lab_run(remapped_run, session_id)
                    methods_run_ids.append(run_id)
                    existing_run_ids.add(run_id)
                    imported_methods_lab_runs += 1
                _save_methods_lab_index(session_id)

    return {
        "bundle_version": bundle_version,
        "imported_count": len(imported_ids),
        "imported_ids": imported_ids,
        "created_count": len(created_ids),
        "created_ids": created_ids,
        "conflict_policy": conflict_policy,
        "conflict_count": sum(conflict_counts.values()),
        "replaced_count": conflict_counts["replace"],
        "kept_current_count": conflict_counts["keep_current"],
        "added_as_new_count": conflict_counts["add_new"],
        "skipped_count": len(skipped),
        "skipped": skipped,
        "warnings": warnings,
        "imported_prompt_lab_runs": imported_prompt_lab_runs,
        "imported_methods_lab_runs": imported_methods_lab_runs,
        "total_in_bundle": len(raw_documents),
    }


@app.post("/api/session/ingest")
async def ingest_session_file(
    file: Annotated[UploadFile, File(...)],
    conflict_policy: Annotated[ImportConflictPolicy, Form()] = "replace",
):
    session_id = "default"
    raw = await file.read()
    filename = file.filename or "unknown"
    mode, payload = _resolve_ingest_mode(raw, filename)
    if mode == "import":
        result = _import_session_payload(
            raw,
            filename,
            session_id,
            payload,
            conflict_policy=conflict_policy,
        )
        return {
            "mode": "import",
            "uploaded_count": 0,
            **result,
        }

    doc = _upload_document_payload(raw, filename, session_id)
    return {
        "mode": "upload",
        "created_count": 1,
        "created_ids": [doc.id],
        "uploaded_count": 1,
        "imported_count": 0,
        "imported_ids": [],
        "skipped_count": 0,
        "skipped": [],
        "warnings": [],
        "imported_prompt_lab_runs": 0,
        "imported_methods_lab_runs": 0,
        "total_in_bundle": 1,
    }


@app.post("/api/session/import")
async def import_session_bundle(
    file: Annotated[UploadFile, File(...)],
    conflict_policy: Annotated[ImportConflictPolicy, Form()] = "replace",
):
    session_id = "default"
    raw = await file.read()
    filename = file.filename or "ground_truth.json"
    return _import_session_payload(raw, filename, session_id, conflict_policy=conflict_policy)


@app.get("/api/metrics/dashboard")
async def get_dashboard_metrics(
    reference: str = Query(...),
    hypothesis: str = Query(...),
    match_mode: str = Query("overlap"),
    label_projection: str = Query("native"),
):
    session_id = "default"
    ids = _load_session_index(session_id)
    normalized_match_mode = _normalize_match_mode(match_mode)
    normalized_projection = _normalize_label_projection(label_projection)
    documents: list[dict] = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    sum_micro_p = 0.0
    sum_micro_r = 0.0
    sum_micro_f1 = 0.0
    sum_macro_p = 0.0
    sum_macro_r = 0.0
    sum_macro_f1 = 0.0
    co_primary_aggregates: dict[str, dict[str, float | int]] = {}
    confidence_sum = 0.0
    documents_with_confidence = 0
    band_counts = {"high": 0, "medium": 0, "low": 0, "na": 0}

    for did in ids:
        doc = _load_doc(did, session_id)
        if not doc:
            continue
        enriched = _enrich_doc(doc, session_id)
        ref_spans = _spans_from_source(enriched, reference)
        hyp_spans = _spans_from_source(enriched, hypothesis)
        eval_ref_spans, eval_hyp_spans = _apply_label_projection(
            ref_spans,
            hyp_spans,
            label_projection=normalized_projection,
        )
        result = compute_metrics(eval_ref_spans, eval_hyp_spans, normalized_match_mode)
        micro = result["micro"]
        macro = result["macro"]
        co_primary = result.get("co_primary_metrics", {})

        tp = int(micro.get("tp", 0))
        fp = int(micro.get("fp", 0))
        fn = int(micro.get("fn", 0))
        total_tp += tp
        total_fp += fp
        total_fn += fn

        sum_micro_p += float(micro.get("precision", 0.0))
        sum_micro_r += float(micro.get("recall", 0.0))
        sum_micro_f1 += float(micro.get("f1", 0.0))
        sum_macro_p += float(macro.get("precision", 0.0))
        sum_macro_r += float(macro.get("recall", 0.0))
        sum_macro_f1 += float(macro.get("f1", 0.0))
        if isinstance(co_primary, dict):
            for metric_name, metric_payload in co_primary.items():
                if not isinstance(metric_payload, dict):
                    continue
                aggregate = co_primary_aggregates.setdefault(
                    str(metric_name),
                    {
                        "tp": 0,
                        "fp": 0,
                        "fn": 0,
                        "sum_micro_p": 0.0,
                        "sum_micro_r": 0.0,
                        "sum_micro_f1": 0.0,
                        "sum_macro_p": 0.0,
                        "sum_macro_r": 0.0,
                        "sum_macro_f1": 0.0,
                        "doc_count": 0,
                    },
                )
                nested_micro = metric_payload.get("micro", {})
                nested_macro = metric_payload.get("macro", {})
                if isinstance(nested_micro, dict):
                    aggregate["tp"] += int(nested_micro.get("tp", 0))
                    aggregate["fp"] += int(nested_micro.get("fp", 0))
                    aggregate["fn"] += int(nested_micro.get("fn", 0))
                    aggregate["sum_micro_p"] += float(nested_micro.get("precision", 0.0))
                    aggregate["sum_micro_r"] += float(nested_micro.get("recall", 0.0))
                    aggregate["sum_micro_f1"] += float(nested_micro.get("f1", 0.0))
                if isinstance(nested_macro, dict):
                    aggregate["sum_macro_p"] += float(nested_macro.get("precision", 0.0))
                    aggregate["sum_macro_r"] += float(nested_macro.get("recall", 0.0))
                    aggregate["sum_macro_f1"] += float(nested_macro.get("f1", 0.0))
                aggregate["doc_count"] += 1
        llm_confidence = _resolve_metrics_llm_confidence(enriched, reference, hypothesis)
        if llm_confidence is not None:
            band_counts[llm_confidence.band] += 1
            if llm_confidence.available and llm_confidence.confidence is not None:
                confidence_sum += float(llm_confidence.confidence)
                documents_with_confidence += 1
        else:
            band_counts["na"] += 1

        documents.append(
            {
                "id": enriched.id,
                "filename": enriched.filename,
                "reference_count": len(eval_ref_spans),
                "hypothesis_count": len(eval_hyp_spans),
                "micro": {
                    "precision": float(micro.get("precision", 0.0)),
                    "recall": float(micro.get("recall", 0.0)),
                    "f1": float(micro.get("f1", 0.0)),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                },
                "macro": {
                    "precision": float(macro.get("precision", 0.0)),
                    "recall": float(macro.get("recall", 0.0)),
                    "f1": float(macro.get("f1", 0.0)),
                },
                "co_primary_metrics": (
                    _serialize_metrics_value(copy.deepcopy(co_primary))
                    if isinstance(co_primary, dict)
                    else {}
                ),
                "cohens_kappa": float(result.get("cohens_kappa", 0.0)),
                "mean_iou": float(result.get("mean_iou", 0.0)),
                "llm_confidence": (
                    llm_confidence.model_dump() if llm_confidence is not None else None
                ),
            }
        )

    compared = len(documents)
    avg_doc_micro = {
        "precision": (sum_micro_p / compared) if compared else 0.0,
        "recall": (sum_micro_r / compared) if compared else 0.0,
        "f1": (sum_micro_f1 / compared) if compared else 0.0,
    }
    avg_doc_macro = {
        "precision": (sum_macro_p / compared) if compared else 0.0,
        "recall": (sum_macro_r / compared) if compared else 0.0,
        "f1": (sum_macro_f1 / compared) if compared else 0.0,
    }
    co_primary_summary: dict[str, dict[str, Any]] = {}
    for metric_name, aggregate in co_primary_aggregates.items():
        metric_doc_count = int(aggregate["doc_count"])
        metric_tp = int(aggregate["tp"])
        metric_fp = int(aggregate["fp"])
        metric_fn = int(aggregate["fn"])
        metric_prf = _prf_from_counts(metric_tp, metric_fp, metric_fn)
        co_primary_summary[metric_name] = {
            "micro": {
                "precision": float(metric_prf["precision"]),
                "recall": float(metric_prf["recall"]),
                "f1": float(metric_prf["f1"]),
                "tp": metric_tp,
                "fp": metric_fp,
                "fn": metric_fn,
            },
            "avg_document_micro": {
                "precision": float(aggregate["sum_micro_p"]) / metric_doc_count if metric_doc_count else 0.0,
                "recall": float(aggregate["sum_micro_r"]) / metric_doc_count if metric_doc_count else 0.0,
                "f1": float(aggregate["sum_micro_f1"]) / metric_doc_count if metric_doc_count else 0.0,
            },
            "avg_document_macro": {
                "precision": float(aggregate["sum_macro_p"]) / metric_doc_count if metric_doc_count else 0.0,
                "recall": float(aggregate["sum_macro_r"]) / metric_doc_count if metric_doc_count else 0.0,
                "f1": float(aggregate["sum_macro_f1"]) / metric_doc_count if metric_doc_count else 0.0,
            },
        }

    documents.sort(
        key=lambda item: item.get("co_primary_metrics", {})
        .get("overlap", {})
        .get("micro", {})
        .get("f1", item["micro"]["f1"])
    )
    return {
        "reference": reference,
        "hypothesis": hypothesis,
        "match_mode": normalized_match_mode,
        "label_projection": normalized_projection,
        "total_documents": len(ids),
        "documents_compared": compared,
        "micro": _prf_from_counts(total_tp, total_fp, total_fn),
        "avg_document_micro": avg_doc_micro,
        "avg_document_macro": avg_doc_macro,
        "co_primary_metrics": co_primary_summary,
        "llm_confidence_summary": {
            "documents_with_confidence": documents_with_confidence,
            "mean_confidence": (
                (confidence_sum / documents_with_confidence)
                if documents_with_confidence > 0
                else None
            ),
            "band_counts": band_counts,
        },
        "documents": documents,
    }


@app.get("/api/config")
async def get_config():
    return _load_config()


@app.put("/api/config")
async def update_config(body: dict):
    cfg = _load_config()
    cfg.update(body)
    _save_config(cfg)
    return cfg


@app.get("/api/experiments/limits")
async def get_experiment_limits() -> ExperimentLimitsResponse:
    return ExperimentLimitsResponse(**_get_experiment_limits())


@app.get("/api/experiments/diagnostics")
async def get_experiment_diagnostics() -> ExperimentDiagnosticsResponse:
    return ExperimentDiagnosticsResponse(**_build_experiment_diagnostics_response())


@app.get("/api/models/presets")
async def list_model_presets():
    return {"presets": MODEL_PRESETS}


@app.get("/api/agent/methods")
async def list_agent_method_catalog(method_bundle: str = Query("audited")):
    return {"methods": list_agent_methods(method_bundle=_normalize_method_bundle(method_bundle))}


@app.get("/api/agent/credentials/status")
async def get_agent_credentials_status():
    api_key_sources = _present_env_vars(ENV_API_KEY_VARS)
    api_base_sources = _present_env_vars(ENV_API_BASE_VARS)
    return {
        "has_api_key": bool(api_key_sources),
        "api_key_sources": api_key_sources,
        "has_api_base": bool(api_base_sources),
        "api_base_sources": api_base_sources,
    }
