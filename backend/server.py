from __future__ import annotations

import copy
import io
import json
import math
import os
import re
import threading
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Literal
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException, UploadFile, File, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import (
    AgentOutputs,
    AgentRunMetrics,
    CanonicalDocument,
    CanonicalSpan,
    LLMConfidenceMetric,
    SavedRunMetadata,
)
from normalizer import parse_file
from agent import (
    FORMAT_GUARDRAIL,
    METHOD_DEFINITION_BY_ID,
    MODEL_PRESETS,
    SYSTEM_PROMPT,
    list_agent_methods,
    normalize_method_spans,
    run_llm_with_metadata,
    run_method_with_metadata,
    run_regex,
)
from metrics import compute_metrics

app = FastAPI(title="Annotation Tool")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(".annotation_tool")
SESSIONS_DIR = BASE_DIR / "sessions"
CONFIG_PATH = BASE_DIR / "config.json"
PROFILE_PATH = BASE_DIR / "session_profile.json"

BUNDLE_FORMAT = "annotation_tool_session"
BUNDLE_VERSION = 3
SUPPORTED_BUNDLE_VERSIONS = {1, 2, 3}
TOOL_VERSION = "2026.03.03"
PROMPT_LAB_DIR_NAME = "prompt_lab"
PROMPT_LAB_MAX_VARIANTS = 6
PROMPT_LAB_DEFAULT_CONCURRENCY = 4
DEFAULT_CHUNK_SIZE_CHARS = 10_000
MIN_CHUNK_SIZE_CHARS = 2_000
MAX_CHUNK_SIZE_CHARS = 30_000
FALLBACK_CHUNK_OVERLAP_CHARS = 200
DEFAULT_CHUNK_PARALLEL_WORKERS = 4
MAX_CHUNK_PARALLEL_WORKERS = 8
PROMPT_LAB_ALLOWED_PRESET_METHODS = set(METHOD_DEFINITION_BY_ID.keys())

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
}

DEFAULT_SESSION_PROFILE = {
    "project_name": "",
    "author": "",
}

MANUAL_RUNS_SIDECAR_KIND = "manual.runs"
MANUAL_RUNS_METADATA_SIDECAR_KIND = "manual.runs.meta"
LLM_RUNS_SIDECAR_KIND = "agent.llm.runs"
METHOD_RUNS_SIDECAR_KIND = "agent.method.runs"
LLM_RUNS_METADATA_SIDECAR_KIND = "agent.llm.runs.meta"
METHOD_RUNS_METADATA_SIDECAR_KIND = "agent.method.runs.meta"


def _ensure_dirs():
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def _session_dir(session_id: str = "default") -> Path:
    return SESSIONS_DIR / session_id


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
    (d / f"{doc_id}.{kind}.json").write_text(json.dumps(payload, indent=2))


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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _load_prompt_lab_index(session_id: str = "default") -> list[str]:
    if session_id in _prompt_lab_runs:
        return _prompt_lab_runs[session_id]
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


def _is_terminal_prompt_lab_status(status: str) -> bool:
    return status in {"completed", "completed_with_errors", "failed"}


def _safe_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _serialize_metrics_payload(metrics: dict) -> dict:
    payload = copy.deepcopy(metrics)
    false_positives = payload.get("false_positives", [])
    if isinstance(false_positives, list):
        payload["false_positives"] = [
            item.model_dump() if isinstance(item, CanonicalSpan) else item
            for item in false_positives
        ]
    false_negatives = payload.get("false_negatives", [])
    if isinstance(false_negatives, list):
        payload["false_negatives"] = [
            item.model_dump() if isinstance(item, CanonicalSpan) else item
            for item in false_negatives
        ]
    return payload


def _build_prompt_lab_cell_summary(cell: dict, total_docs: int) -> dict:
    docs_raw = cell.get("documents", {})
    documents = docs_raw if isinstance(docs_raw, dict) else {}
    completed_docs = 0
    failed_docs = 0
    tp = 0
    fp = 0
    fn = 0
    confidence_values: list[float] = []
    per_label_counts: dict[str, dict[str, int]] = {}

    for result in documents.values():
        if not isinstance(result, dict):
            continue
        status = str(result.get("status", "pending"))
        if status == "completed":
            completed_docs += 1
            metrics = result.get("metrics", {})
            if isinstance(metrics, dict):
                micro = metrics.get("micro", {})
                if isinstance(micro, dict):
                    tp += int(micro.get("tp", 0))
                    fp += int(micro.get("fp", 0))
                    fn += int(micro.get("fn", 0))
                per_label = metrics.get("per_label", {})
                if isinstance(per_label, dict):
                    for label, label_metrics in per_label.items():
                        if not isinstance(label_metrics, dict):
                            continue
                        aggregate = per_label_counts.setdefault(
                            str(label),
                            {"tp": 0, "fp": 0, "fn": 0, "support": 0},
                        )
                        label_tp = int(label_metrics.get("tp", 0))
                        label_fp = int(label_metrics.get("fp", 0))
                        label_fn = int(label_metrics.get("fn", 0))
                        aggregate["tp"] += label_tp
                        aggregate["fp"] += label_fp
                        aggregate["fn"] += label_fn
                        aggregate["support"] += int(
                            label_metrics.get("support", label_tp + label_fn)
                        )
            llm_confidence = result.get("llm_confidence")
            if isinstance(llm_confidence, dict):
                conf = _safe_float(llm_confidence.get("confidence"))
                if conf is not None:
                    confidence_values.append(conf)
            continue
        if status in {"failed", "unavailable"}:
            failed_docs += 1

    processed = completed_docs + failed_docs
    micro = _prf_from_counts(tp, fp, fn) if completed_docs > 0 else _prf_from_counts(0, 0, 0)
    if processed == 0:
        status = "pending"
    elif processed < total_docs:
        status = "running"
    elif failed_docs > 0:
        status = "completed_with_errors"
    else:
        status = "completed"

    per_label_summary: dict[str, dict[str, float | int]] = {}
    for label, counts in sorted(per_label_counts.items()):
        label_prf = _prf_from_counts(counts["tp"], counts["fp"], counts["fn"])
        per_label_summary[label] = {
            **label_prf,
            "support": counts["support"],
        }

    return {
        "id": cell.get("id"),
        "model_id": cell.get("model_id"),
        "model_label": cell.get("model_label"),
        "prompt_id": cell.get("prompt_id"),
        "prompt_label": cell.get("prompt_label"),
        "status": status,
        "total_docs": total_docs,
        "completed_docs": completed_docs,
        "failed_docs": failed_docs,
        "error_count": failed_docs,
        "micro": micro,
        "per_label": per_label_summary,
        "mean_confidence": (
            sum(confidence_values) / len(confidence_values) if confidence_values else None
        ),
    }


def _build_prompt_lab_matrix(run: dict) -> dict:
    total_docs = len(run.get("doc_ids", []))
    prompts = run.get("prompts", [])
    models = run.get("models", [])
    cells_raw = run.get("cells", {})
    cells_dict = cells_raw if isinstance(cells_raw, dict) else {}
    summaries: list[dict] = []
    available_labels: set[str] = set()
    for model in models:
        model_id = str(model.get("id", ""))
        for prompt in prompts:
            prompt_id = str(prompt.get("id", ""))
            cell_id = f"{model_id}__{prompt_id}"
            cell = cells_dict.get(cell_id)
            if not isinstance(cell, dict):
                cell = {
                    "id": cell_id,
                    "model_id": model_id,
                    "model_label": model.get("label", model_id),
                    "prompt_id": prompt_id,
                    "prompt_label": prompt.get("label", prompt_id),
                    "documents": {},
                }
            summary = _build_prompt_lab_cell_summary(cell, total_docs)
            per_label = summary.get("per_label", {})
            if isinstance(per_label, dict):
                available_labels.update(str(label) for label in per_label.keys())
            summaries.append(summary)
    return {
        "models": [{"id": str(item.get("id", "")), "label": str(item.get("label", ""))} for item in models],
        "prompts": [{"id": str(item.get("id", "")), "label": str(item.get("label", ""))} for item in prompts],
        "cells": summaries,
        "available_labels": sorted(available_labels),
    }


def _build_prompt_lab_run_summary(run: dict) -> dict:
    matrix = _build_prompt_lab_matrix(run)
    cells = matrix["cells"]
    completed = 0
    failed = 0
    total = len(run.get("doc_ids", [])) * len(run.get("models", [])) * len(run.get("prompts", []))
    for cell in cells:
        completed += int(cell.get("completed_docs", 0)) + int(cell.get("failed_docs", 0))
        failed += int(cell.get("failed_docs", 0))
    return {
        "id": run.get("id"),
        "name": run.get("name"),
        "status": run.get("status"),
        "created_at": run.get("created_at"),
        "started_at": run.get("started_at"),
        "finished_at": run.get("finished_at"),
        "doc_count": len(run.get("doc_ids", [])),
        "prompt_count": len(run.get("prompts", [])),
        "model_count": len(run.get("models", [])),
        "total_tasks": total,
        "completed_tasks": completed,
        "failed_tasks": failed,
    }


def _build_prompt_lab_run_detail(run: dict) -> dict:
    summary = _build_prompt_lab_run_summary(run)
    matrix = _build_prompt_lab_matrix(run)
    runtime_raw = run.get("runtime", {})
    runtime = runtime_raw if isinstance(runtime_raw, dict) else {}
    return {
        **summary,
        "doc_ids": run.get("doc_ids", []),
        "prompts": run.get("prompts", []),
        "models": run.get("models", []),
        "runtime": {
            "temperature": runtime.get("temperature", 0.0),
            "match_mode": runtime.get("match_mode", "exact"),
            "reference_source": runtime.get("reference_source", "manual"),
            "fallback_reference_source": runtime.get("fallback_reference_source", "pre"),
            "label_profile": runtime.get("label_profile", "simple"),
            "label_projection": runtime.get("label_projection", "native"),
            "api_base": runtime.get("api_base", ""),
            "chunk_mode": runtime.get("chunk_mode", "auto"),
            "chunk_size_chars": runtime.get("chunk_size_chars", DEFAULT_CHUNK_SIZE_CHARS),
        },
        "concurrency": run.get("concurrency", PROMPT_LAB_DEFAULT_CONCURRENCY),
        "warnings": run.get("warnings", []),
        "errors": run.get("errors", []),
        "matrix": matrix,
        "progress": {
            "total_tasks": summary["total_tasks"],
            "completed_tasks": summary["completed_tasks"],
            "failed_tasks": summary["failed_tasks"],
        },
    }


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
    if isinstance(last_run_raw, dict):
        raw_label_profile = str(last_run_raw.get("label_profile", "")).strip().lower()
        if raw_label_profile in {"simple", "advanced"}:
            label_profile = raw_label_profile  # type: ignore[assignment]

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


def _normalize_chunk_mode(raw: object) -> Literal["auto", "off", "force"]:
    value = str(raw or "auto").strip().lower()
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


def _normalize_session_profile(raw: object) -> dict[str, str]:
    if raw is None:
        return dict(DEFAULT_SESSION_PROFILE)
    if not isinstance(raw, dict):
        raise ValueError("Session profile must be an object")

    def _clean(key: str, max_len: int) -> str:
        value = raw.get(key, "")
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ValueError(f"Session profile field '{key}' must be a string")
        normalized = value.strip()
        if len(normalized) > max_len:
            raise ValueError(
                f"Session profile field '{key}' exceeds max length {max_len}"
            )
        return normalized

    return {
        "project_name": _clean("project_name", 120),
        "author": _clean("author", 120),
    }


def _load_session_profile() -> dict[str, str]:
    _ensure_dirs()
    if not PROFILE_PATH.exists():
        return dict(DEFAULT_SESSION_PROFILE)
    try:
        raw = json.loads(PROFILE_PATH.read_text())
        return _normalize_session_profile(raw)
    except Exception:
        # Keep app usable even with a malformed local profile file.
        return dict(DEFAULT_SESSION_PROFILE)


def _save_session_profile(profile: dict[str, str]):
    _ensure_dirs()
    PROFILE_PATH.write_text(json.dumps(profile, indent=2))


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


def _present_env_vars(names: list[str]) -> list[str]:
    return [name for name in names if bool(os.environ.get(name))]


def _resolve_llm_runtime_config(body: "AgentRunBody") -> dict[str, object]:
    cfg = _load_config()
    api_key = (
        body.api_key
        or os.environ.get("LITELLM_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
        or os.environ.get("ANTHROPIC_API_KEY", "")
        or os.environ.get("GEMINI_API_KEY", "")
        or os.environ.get("GOOGLE_API_KEY", "")
    )
    api_base = (
        body.api_base
        or str(cfg.get("api_base", "") or "")
        or os.environ.get("LITELLM_BASE_URL", "")
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
    chunk_mode = _normalize_chunk_mode(body.chunk_mode or "auto")
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


def _build_llm_prompt_snapshot(requested_system_prompt: str) -> dict[str, Any]:
    requested = str(requested_system_prompt or "")
    return {
        "requested_system_prompt": requested,
        "format_guardrail_appended": True,
        "effective_system_prompt": f"{requested}\n\n{FORMAT_GUARDRAIL}",
    }


def _build_method_prompt_snapshot(
    *,
    method_id: str,
    additional_constraints: str,
    method_verify: bool | None,
) -> dict[str, Any]:
    method_definition = METHOD_DEFINITION_BY_ID.get(method_id)
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
                "effective_system_prompt": f"{resolved_prompt}\n\n{FORMAT_GUARDRAIL}",
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
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[CanonicalSpan], list[str], LLMConfidenceMetric]:
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
        )
        spans = _normalize_and_validate_spans(
            normalize_method_spans(llm_result.spans, label_profile=label_profile),
            doc.raw_text,
        )
        if progress_callback is not None:
            progress_callback(1, 1)
        return spans, llm_result.warnings, llm_result.llm_confidence

    chunks = _build_text_chunks(doc, chunk_size_chars)
    warnings: list[str] = []
    all_spans: list[CanonicalSpan] = []
    chunk_metrics: list[LLMConfidenceMetric] = []
    chunk_count = len(chunks)
    chunk_workers = _resolve_chunk_parallel_workers(chunk_count)

    def _run_chunk(idx: int, start: int, end: int) -> tuple[int, list[CanonicalSpan], list[str], LLMConfidenceMetric]:
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
        )
        shifted = _shift_spans(
            normalize_method_spans(llm_result.spans, label_profile=label_profile),
            start,
        )
        chunk_warnings = [f"Chunk {idx + 1}/{chunk_count}: {item}" for item in llm_result.warnings]
        return idx, shifted, chunk_warnings, llm_result.llm_confidence

    chunk_results: dict[int, tuple[list[CanonicalSpan], list[str], LLMConfidenceMetric]] = {}
    if chunk_workers == 1:
        for idx, (start, end) in enumerate(chunks):
            _, shifted, chunk_warnings, chunk_conf = _run_chunk(idx, start, end)
            chunk_results[idx] = (shifted, chunk_warnings, chunk_conf)
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
                    _, shifted, chunk_warnings, chunk_conf = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"Chunk {idx + 1}/{chunk_count} failed during parallel LLM execution: {exc}"
                    ) from exc
                chunk_results[idx] = (shifted, chunk_warnings, chunk_conf)
                completed_count += 1
                if progress_callback is not None:
                    progress_callback(completed_count, chunk_count)

    for idx in range(chunk_count):
        shifted, chunk_warnings, chunk_conf = chunk_results[idx]
        all_spans.extend(shifted)
        warnings.extend(chunk_warnings)
        chunk_metrics.append(chunk_conf)

    normalized = _normalize_and_validate_spans(_dedup_spans(all_spans), doc.raw_text)
    warnings.insert(
        0,
        (
            f"Chunked LLM run used {chunk_count} chunk(s) at ~{chunk_size_chars} chars "
            f"(mode={chunk_mode}, workers={chunk_workers})."
        ),
    )
    return normalized, warnings, _aggregate_llm_confidence(chunk_metrics)


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
) -> tuple[list[CanonicalSpan], list[str]]:
    if not _should_use_chunking(len(doc.raw_text), chunk_mode, chunk_size_chars):
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
        )
        spans = _normalize_and_validate_spans(
            normalize_method_spans(method_result.spans, label_profile=label_profile),
            doc.raw_text,
        )
        if progress_callback is not None:
            progress_callback(1, 1)
        return spans, method_result.warnings

    chunks = _build_text_chunks(doc, chunk_size_chars)
    warnings: list[str] = []
    all_spans: list[CanonicalSpan] = []
    chunk_count = len(chunks)
    chunk_workers = _resolve_chunk_parallel_workers(chunk_count)

    def _run_chunk(idx: int, start: int, end: int) -> tuple[int, list[CanonicalSpan], list[str]]:
        chunk_text = doc.raw_text[start:end]
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
        )
        shifted = _shift_spans(
            normalize_method_spans(method_result.spans, label_profile=label_profile),
            start,
        )
        chunk_warnings = [f"Chunk {idx + 1}/{chunk_count}: {item}" for item in method_result.warnings]
        return idx, shifted, chunk_warnings

    chunk_results: dict[int, tuple[list[CanonicalSpan], list[str]]] = {}
    if chunk_workers == 1:
        for idx, (start, end) in enumerate(chunks):
            _, shifted, chunk_warnings = _run_chunk(idx, start, end)
            chunk_results[idx] = (shifted, chunk_warnings)
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
                    _, shifted, chunk_warnings = future.result()
                except Exception as exc:
                    raise RuntimeError(
                        f"Chunk {idx + 1}/{chunk_count} failed during parallel method execution: {exc}"
                    ) from exc
                chunk_results[idx] = (shifted, chunk_warnings)
                completed_count += 1
                if progress_callback is not None:
                    progress_callback(completed_count, chunk_count)

    for idx in range(chunk_count):
        shifted, chunk_warnings = chunk_results[idx]
        all_spans.extend(shifted)
        warnings.extend(chunk_warnings)

    normalized = _normalize_and_validate_spans(_dedup_spans(all_spans), doc.raw_text)
    warnings.insert(
        0,
        (
            f"Chunked method run used {chunk_count} chunk(s) at ~{chunk_size_chars} chars "
            f"(mode={chunk_mode}, workers={chunk_workers})."
        ),
    )
    return normalized, warnings


# --- Routes ---


@app.get("/api/documents")
async def list_documents():
    session_id = "default"
    ids = _load_session_index(session_id)
    docs = []
    for did in ids:
        doc = _load_doc(did, session_id)
        if doc:
            manual = _load_sidecar(did, "manual", session_id)
            status = "in_progress" if manual else "pending"
            docs.append(
                {
                    "id": doc.id,
                    "filename": doc.filename,
                    "status": status,
                }
            )
    return docs


@app.get("/api/documents/{doc_id}")
async def get_document(doc_id: str):
    session_id = "default"
    doc = _load_doc(doc_id, session_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return _enrich_doc(doc, session_id)


@app.post("/api/documents/upload")
async def upload_file(file: UploadFile = File(...)):
    session_id = "default"
    raw = await file.read()
    filename = file.filename or "unknown"
    doc_id = str(uuid.uuid4())[:8]

    try:
        docs = parse_file(raw, filename, doc_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    ids = _load_session_index(session_id)
    for doc in docs:
        _save_doc(doc, session_id)
        ids.append(doc.id)
    _save_session_index(session_id)

    # Return the first document (for single-file uploads)
    # For JSONL with multiple records, return the first one
    if docs:
        return _enrich_doc(docs[0], session_id)
    raise HTTPException(status_code=400, detail="No documents parsed from file")


@app.delete("/api/documents/{doc_id}")
async def delete_document(doc_id: str):
    session_id = "default"
    ids = _load_session_index(session_id)
    in_index = doc_id in ids
    removed_files = _delete_doc_files(doc_id, session_id)
    if not in_index and not removed_files:
        raise HTTPException(status_code=404, detail="Document not found")

    if in_index:
        _session_docs[session_id] = [did for did in ids if did != doc_id]
        _save_session_index(session_id)

    return {"deleted": True, "doc_id": doc_id}


@app.put("/api/documents/{doc_id}/manual-annotations")
async def save_manual_annotations(doc_id: str, spans: list[CanonicalSpan]):
    session_id = "default"
    doc = _load_doc(doc_id, session_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    normalized = _normalize_and_validate_spans(spans, doc.raw_text)
    _save_sidecar(doc_id, "manual", normalized, session_id)
    _upsert_span_map_entry(
        doc_id,
        MANUAL_RUNS_SIDECAR_KIND,
        "manual",
        normalized,
        session_id,
    )
    _upsert_run_metadata(
        doc_id,
        MANUAL_RUNS_METADATA_SIDECAR_KIND,
        "manual",
        SavedRunMetadata(
            mode="manual",
            updated_at=_now_iso(),
        ),
        session_id,
    )
    return _enrich_doc(doc, session_id)


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


class SessionProfileBody(BaseModel):
    project_name: str | None = None
    author: str | None = None


class PromptLabPromptInput(BaseModel):
    id: str | None = None
    label: str
    system_prompt: str | None = None
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
    match_mode: Literal["exact", "overlap"] = "exact"
    reference_source: Literal["manual", "pre"] = "manual"
    fallback_reference_source: Literal["manual", "pre"] = "pre"
    label_profile: Literal["simple", "advanced"] = "simple"
    label_projection: Literal["native", "coarse_simple"] = "native"
    chunk_mode: Literal["auto", "off", "force"] = "auto"
    chunk_size_chars: int = DEFAULT_CHUNK_SIZE_CHARS


class PromptLabRunCreateBody(BaseModel):
    name: str | None = None
    doc_ids: list[str]
    prompts: list[PromptLabPromptInput]
    models: list[PromptLabModelInput]
    runtime: PromptLabRuntimeInput
    concurrency: int = PROMPT_LAB_DEFAULT_CONCURRENCY


def _resolve_prompt_lab_runtime(runtime: PromptLabRuntimeInput) -> dict[str, object]:
    cfg = _load_config()
    api_key = (
        runtime.api_key
        or os.environ.get("LITELLM_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
        or os.environ.get("ANTHROPIC_API_KEY", "")
        or os.environ.get("GEMINI_API_KEY", "")
        or os.environ.get("GOOGLE_API_KEY", "")
    )
    api_base = (
        runtime.api_base
        or str(cfg.get("api_base", "") or "")
        or os.environ.get("LITELLM_BASE_URL", "")
    )
    chunk_mode = _normalize_chunk_mode(runtime.chunk_mode)
    chunk_size_chars = _normalize_chunk_size(runtime.chunk_size_chars)
    label_profile = _normalize_label_profile(runtime.label_profile)
    label_projection = _normalize_label_projection(runtime.label_projection)
    return {
        "api_key": api_key,
        "api_base": api_base,
        "temperature": runtime.temperature,
        "match_mode": runtime.match_mode,
        "reference_source": runtime.reference_source,
        "fallback_reference_source": runtime.fallback_reference_source,
        "chunk_mode": chunk_mode,
        "chunk_size_chars": chunk_size_chars,
        "label_profile": label_profile,
        "label_projection": label_projection,
    }


def _validate_prompt_lab_request(body: PromptLabRunCreateBody, session_id: str = "default"):
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
    if body.concurrency < 1 or body.concurrency > PROMPT_LAB_MAX_VARIANTS:
        raise HTTPException(status_code=400, detail="concurrency must be between 1 and 6")
    if len(body.prompts) * len(body.models) > PROMPT_LAB_MAX_VARIANTS * PROMPT_LAB_MAX_VARIANTS:
        raise HTTPException(status_code=400, detail="Matrix limit exceeded (max 6x6)")

    known_ids = set(_load_session_index(session_id))
    for doc_id in body.doc_ids:
        if doc_id not in known_ids:
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

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
            if method_id not in PROMPT_LAB_ALLOWED_PRESET_METHODS:
                allowed = ", ".join(sorted(PROMPT_LAB_ALLOWED_PRESET_METHODS))
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"preset_method_id '{method_id}' is not allowed in Prompt Lab. "
                        f"Allowed presets: {allowed}"
                    ),
                )
            if method_id not in METHOD_DEFINITION_BY_ID:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown preset method: {method_id}",
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

                hypothesis_spans, warnings = _run_method_for_document(
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
                )
                llm_confidence: LLMConfidenceMetric | None = None
            else:
                requested_system_prompt = str(prompt.get("system_prompt") or "").strip()
                if not requested_system_prompt:
                    raise ValueError("system_prompt is required for prompt variants.")
                system_prompt = f"{requested_system_prompt}\n\n{FORMAT_GUARDRAIL}"
                hypothesis_spans, warnings, llm_confidence = _run_llm_for_document(
                    doc=enriched,
                    api_key=api_key,
                    api_base=api_base or None,
                    model=str(model["model"]),
                    system_prompt=system_prompt,
                    temperature=temperature,
                    reasoning_effort=str(model["reasoning_effort"]),
                    anthropic_thinking=bool(model["anthropic_thinking"]),
                    anthropic_thinking_budget_tokens=model["anthropic_thinking_budget_tokens"],
                    label_profile=label_profile,  # type: ignore[arg-type]
                    chunk_mode=chunk_mode,
                    chunk_size_chars=chunk_size_chars,
                )

            resolved_reference_source, reference_spans = _resolve_prompt_lab_reference(
                enriched,
                reference_source,  # type: ignore[arg-type]
                fallback_reference_source,  # type: ignore[arg-type]
            )
            projected_reference, projected_hypothesis = _apply_label_projection(
                reference_spans,
                hypothesis_spans,
                label_projection=label_projection,  # type: ignore[arg-type]
            )
            metrics = _serialize_metrics_payload(
                compute_metrics(projected_reference, projected_hypothesis, match_mode)
            )
            return cell_id, doc_id, {
                "status": "completed",
                "reference_source_used": resolved_reference_source,
                "reference_spans": [span.model_dump() for span in reference_spans],
                "hypothesis_spans": [span.model_dump() for span in hypothesis_spans],
                "metrics": metrics,
                "warnings": warnings,
                "llm_confidence": llm_confidence.model_dump() if llm_confidence else None,
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

        system_prompt = f"{requested_system_prompt}\n\n{FORMAT_GUARDRAIL}"
        _start_agent_progress(
            doc_id,
            mode="llm",
            total_chunks=total_chunks,
            session_id=session_id,
        )

        try:
            spans, warnings, llm_confidence = _run_llm_for_document(
                doc=doc,
                api_key=api_key,
                api_base=api_base or None,
                model=model,
                system_prompt=system_prompt,
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
        _save_sidecar(doc_id, "agent.llm", spans, session_id)
        _delete_sidecar(doc_id, "agent.openai", session_id)
        _upsert_span_map_entry(
            doc_id,
            LLM_RUNS_SIDECAR_KIND,
            model,
            spans,
            session_id,
        )
        _upsert_run_metadata(
            doc_id,
            LLM_RUNS_METADATA_SIDECAR_KIND,
            model,
            SavedRunMetadata(
                mode="llm",
                updated_at=_now_iso(),
                model=model,
                label_profile=label_profile,  # type: ignore[arg-type]
                prompt_snapshot=_build_llm_prompt_snapshot(requested_system_prompt),
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
            spans, warnings = _run_method_for_document(
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
                ),
            ),
            session_id,
        )
        _save_json_sidecar(
            doc_id,
            "agent.last_run",
            {
                "mode": "method",
                "method_id": method_id,
                "label_profile": label_profile,
                "updated_at": _now_iso(),
            },
            session_id,
        )
        _finish_agent_progress(doc_id, status="completed", session_id=session_id)
        enriched = _enrich_doc(doc, session_id)
        enriched.agent_run_warnings = warnings
        enriched.agent_run_metrics = AgentRunMetrics(
            llm_confidence=enriched.agent_run_metrics.llm_confidence,
            label_profile=label_profile,  # type: ignore[arg-type]
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
    session_id = "default"
    _validate_prompt_lab_request(body, session_id)
    runtime = _resolve_prompt_lab_runtime(body.runtime)
    api_key = str(runtime["api_key"])
    api_base = str(runtime["api_base"])
    requires_llm = False
    for prompt in body.prompts:
        if prompt.variant_type == "prompt":
            requires_llm = True
            break
        preset_method = METHOD_DEFINITION_BY_ID.get(str(prompt.preset_method_id or ""))
        if preset_method and bool(preset_method.get("uses_llm")):
            requires_llm = True
            break

    if requires_llm:
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail=(
                    "API key required for Prompt Lab runs. Set LITELLM_API_KEY "
                    "or provide runtime.api_key in request."
                ),
            )
        for model in body.models:
            _validate_gateway_model_access(model=model.model, api_base=api_base, api_key=api_key)

    with _prompt_lab_lock:
        run = _initialize_prompt_lab_run(body, session_id, runtime)
        ids = _load_prompt_lab_index(session_id)
        ids.append(str(run["id"]))
        _save_prompt_lab_run(run, session_id)
        _save_prompt_lab_index(session_id)

    worker = threading.Thread(
        target=_run_prompt_lab_job,
        args=(str(run["id"]), session_id, runtime),
        daemon=True,
    )
    worker.start()
    return _build_prompt_lab_run_detail(run)


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
        "warnings": stored.get("warnings", []),
        "reference_source_used": stored.get("reference_source_used"),
        "reference_spans": stored.get("reference_spans", []),
        "hypothesis_spans": stored.get("hypothesis_spans", []),
        "metrics": stored.get("metrics"),
        "llm_confidence": stored.get("llm_confidence"),
        "transcript_text": enriched.raw_text if enriched is not None else None,
        "document": {
            "id": doc_id,
            "filename": enriched.filename if enriched is not None else stored.get("filename"),
        },
        "model": model,
        "prompt": prompt,
    }


@app.get("/api/documents/{doc_id}/metrics")
async def get_metrics(
    doc_id: str,
    reference: str = Query(...),
    hypothesis: str = Query(...),
    match_mode: str = Query("exact"),
    label_projection: str = Query("native"),
):
    session_id = "default"
    doc = _load_doc(doc_id, session_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    enriched = _enrich_doc(doc, session_id)
    ref_spans = _spans_from_source(enriched, reference)
    hyp_spans = _spans_from_source(enriched, hypothesis)
    normalized_projection = _normalize_label_projection(label_projection)
    eval_ref_spans, eval_hyp_spans = _apply_label_projection(
        ref_spans,
        hyp_spans,
        label_projection=normalized_projection,
    )

    result = compute_metrics(eval_ref_spans, eval_hyp_spans, match_mode)
    result["llm_confidence"] = (
        enriched.agent_run_metrics.llm_confidence.model_dump()
        if enriched.agent_run_metrics.llm_confidence is not None
        else None
    )
    result["label_projection"] = normalized_projection
    return result


@app.get("/api/session/profile")
async def get_session_profile():
    return _load_session_profile()


@app.put("/api/session/profile")
async def update_session_profile(body: SessionProfileBody):
    current = _load_session_profile()
    merged = {
        **current,
        **body.model_dump(exclude_none=True),
    }
    try:
        normalized = _normalize_session_profile(merged)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _save_session_profile(normalized)
    return normalized


@app.get("/api/session/export")
async def export_session_bundle():
    session_id = "default"
    ids = _load_session_index(session_id)
    documents: list[dict] = []
    for did in ids:
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
        "project": _load_session_profile(),
        "compatibility": {
            "tool_version": TOOL_VERSION,
            "import_supported_versions": sorted(SUPPORTED_BUNDLE_VERSIONS),
        },
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "documents": documents,
        "prompt_lab_runs": _export_prompt_lab_runs(session_id),
        "config": _load_config(),
    }


@app.get("/api/session/export-ground-truth")
async def export_ground_truth_only(
    source: str = Query("manual"),
):
    session_id = "default"
    ids = _load_session_index(session_id)
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
                "ground_truth_source": source_value,
                "spans": [span.model_dump() for span in spans],
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


@app.post("/api/session/import")
async def import_session_bundle(file: UploadFile = File(...)):
    session_id = "default"
    raw = await file.read()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise HTTPException(
            status_code=400, detail="Import file must be valid UTF-8 JSON"
        ) from exc

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Import payload must be a JSON object")

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
    doc_id_remap: dict[str, str] = {}
    skipped: list[dict[str, str | int]] = []
    warnings: list[str] = []

    incoming_project = payload.get("project")
    if incoming_project is not None:
        try:
            profile = _normalize_session_profile(incoming_project)
            _save_session_profile(profile)
        except ValueError as exc:
            warnings.append(f"Ignored invalid project metadata: {exc}")

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

        new_id = doc.id
        while new_id in existing_ids:
            new_id = str(uuid.uuid4())[:8]
        if new_id != doc.id:
            doc = doc.model_copy(update={"id": new_id})

        _save_doc(doc, session_id)
        if manual_spans:
            _save_sidecar(new_id, "manual", manual_spans, session_id)
        if rule_spans:
            _save_sidecar(new_id, "agent.rule", rule_spans, session_id)
        if llm_spans:
            _save_sidecar(new_id, "agent.llm", llm_spans, session_id)
        for method_id, spans in method_spans.items():
            _save_sidecar(new_id, f"agent.method.{method_id}", spans, session_id)
        if llm_runs:
            _save_span_map_sidecar(new_id, LLM_RUNS_SIDECAR_KIND, llm_runs, session_id)
        if method_runs:
            _save_span_map_sidecar(new_id, METHOD_RUNS_SIDECAR_KIND, method_runs, session_id)
        if llm_run_metadata:
            _save_run_metadata_map_sidecar(
                new_id,
                LLM_RUNS_METADATA_SIDECAR_KIND,
                llm_run_metadata,
                session_id,
            )
        if method_run_metadata:
            _save_run_metadata_map_sidecar(
                new_id,
                METHOD_RUNS_METADATA_SIDECAR_KIND,
                method_run_metadata,
                session_id,
            )
        if llm_confidence is not None:
            _save_json_sidecar(
                new_id,
                "agent.llm.metrics",
                llm_confidence.model_dump(),
                session_id,
            )
        if imported_label_profile is not None:
            _save_json_sidecar(
                new_id,
                "agent.last_run",
                {
                    "mode": "imported",
                    "label_profile": imported_label_profile,
                    "updated_at": _now_iso(),
                },
                session_id,
            )

        ids.append(new_id)
        existing_ids.add(new_id)
        imported_ids.append(new_id)
        doc_id_remap[original_doc_id] = new_id

    _save_session_index(session_id)

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

    return {
        "bundle_version": bundle_version,
        "imported_count": len(imported_ids),
        "imported_ids": imported_ids,
        "skipped_count": len(skipped),
        "skipped": skipped,
        "warnings": warnings,
        "imported_prompt_lab_runs": imported_prompt_lab_runs,
        "total_in_bundle": len(raw_documents),
    }


@app.get("/api/metrics/dashboard")
async def get_dashboard_metrics(
    reference: str = Query(...),
    hypothesis: str = Query(...),
    match_mode: str = Query("exact"),
    label_projection: str = Query("native"),
):
    session_id = "default"
    ids = _load_session_index(session_id)
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
        result = compute_metrics(eval_ref_spans, eval_hyp_spans, match_mode)
        micro = result["micro"]
        macro = result["macro"]

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
        llm_confidence = enriched.agent_run_metrics.llm_confidence
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

    documents.sort(key=lambda item: item["micro"]["f1"])
    return {
        "reference": reference,
        "hypothesis": hypothesis,
        "match_mode": match_mode,
        "label_projection": normalized_projection,
        "total_documents": len(ids),
        "documents_compared": compared,
        "micro": _prf_from_counts(total_tp, total_fp, total_fn),
        "avg_document_micro": avg_doc_micro,
        "avg_document_macro": avg_doc_macro,
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


@app.get("/api/models/presets")
async def list_model_presets():
    return {"presets": MODEL_PRESETS}


@app.get("/api/agent/methods")
async def list_agent_method_catalog():
    return {"methods": list_agent_methods()}


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
