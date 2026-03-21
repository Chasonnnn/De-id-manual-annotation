from __future__ import annotations

import concurrent.futures
import copy
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import HTTPException
from models import CanonicalSpan, LLMConfidenceMetric, ResolutionEvent, SavedRunMetadata


def _server():
    import server

    return server


def _normalize_requested_ids(values: object) -> list[str]:
    if not isinstance(values, list):
        return []
    return [str(item).strip() for item in values if str(item).strip()]


def _resolve_folder_doc_ids(folder_ids: list[str], *, session_id: str) -> list[str]:
    srv = _server()
    resolved: list[str] = []
    for folder_id in folder_ids:
        folder = srv._load_folder(folder_id, session_id)
        if folder is None:
            raise HTTPException(status_code=404, detail=f"Folder not found: {folder_id}")
        resolved.extend(folder.doc_ids)
    return list(dict.fromkeys(resolved))


def _resolve_text_file_path(path_value: str, *, context_dir: Path | None) -> Path:
    raw_path = Path(str(path_value).strip())
    if not raw_path.is_absolute():
        base = context_dir or Path.cwd()
        raw_path = (base / raw_path).resolve()
    return raw_path


def _load_prompt_text_from_file(path: Path) -> str:
    if path.name == "Skills.md":
        raise ValueError(
            "Prompt file 'Skills.md' is not supported. Use the actual skill manifest name 'SKILL.md'."
        )
    if not path.exists():
        raise ValueError(f"Prompt file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Prompt file must be a file: {path}")

    text = path.read_text()
    if path.name == "SKILL.md":
        lines = text.splitlines()
        if lines and lines[0].strip() == "---":
            closing_index = None
            for index in range(1, len(lines)):
                if lines[index].strip() == "---":
                    closing_index = index
                    break
            if closing_index is not None:
                text = "\n".join(lines[closing_index + 1 :])
        text = text.strip()
        if not text:
            raise ValueError(f"Prompt file '{path}' resolved to empty prompt text.")
        return text

    if not text.strip():
        raise ValueError(f"Prompt file '{path}' resolved to empty prompt text.")
    return text


def _resolve_prompt_variant_text(
    *,
    system_prompt: str | None,
    prompt_file: str | None,
    context_dir: Path | None,
) -> str:
    has_system_prompt = bool((system_prompt or "").strip())
    has_prompt_file = bool((prompt_file or "").strip())
    if has_system_prompt == has_prompt_file:
        raise ValueError(
            "Prompt variants require exactly one of system_prompt or prompt_file."
        )
    if has_system_prompt:
        return str(system_prompt or "").strip()
    prompt_path = _resolve_text_file_path(str(prompt_file or ""), context_dir=context_dir)
    return _load_prompt_text_from_file(prompt_path)


def prepare_prompt_lab_body(
    body: Any,
    *,
    session_id: str = "default",
    context_dir: Path | None = None,
):
    srv = _server()

    prompt_items = []
    for prompt in body.prompts:
        variant_type = str(prompt.variant_type or "prompt")
        prompt_file = getattr(prompt, "prompt_file", None)
        if variant_type == "prompt":
            prompt_text = _resolve_prompt_variant_text(
                system_prompt=prompt.system_prompt,
                prompt_file=prompt_file,
                context_dir=context_dir,
            )
            prompt_items.append(
                prompt.model_copy(
                    update={
                        "system_prompt": prompt_text,
                        "prompt_file": None,
                    }
                )
            )
            continue
        if prompt_file:
            raise ValueError(
                "prompt_file is only supported for variant_type='prompt'."
            )
        prompt_items.append(prompt)

    requested_doc_ids = _normalize_requested_ids(body.doc_ids or [])
    requested_folder_ids = _normalize_requested_ids(getattr(body, "folder_ids", []))
    resolved_folder_doc_ids = _resolve_folder_doc_ids(
        requested_folder_ids,
        session_id=session_id,
    )
    if requested_doc_ids or requested_folder_ids:
        doc_ids = list(dict.fromkeys([*requested_doc_ids, *resolved_folder_doc_ids]))
    else:
        doc_ids = list(dict.fromkeys(srv._load_session_index(session_id)))

    if not doc_ids:
        raise HTTPException(
            status_code=400,
            detail=f"No documents found in session '{session_id}'.",
        )

    return body.model_copy(
        update={
            "doc_ids": doc_ids,
            "folder_ids": requested_folder_ids,
            "prompts": prompt_items,
        }
    )


def prepare_methods_lab_body(
    body: Any,
    *,
    session_id: str = "default",
):
    srv = _server()

    requested_doc_ids = _normalize_requested_ids(body.doc_ids or [])
    requested_folder_ids = _normalize_requested_ids(getattr(body, "folder_ids", []))
    resolved_folder_doc_ids = _resolve_folder_doc_ids(
        requested_folder_ids,
        session_id=session_id,
    )
    if requested_doc_ids or requested_folder_ids:
        doc_ids = list(dict.fromkeys([*requested_doc_ids, *resolved_folder_doc_ids]))
    else:
        reference_source = getattr(body.runtime, "reference_source", "manual")
        fallback_reference_source = getattr(body.runtime, "fallback_reference_source", "pre")
        doc_ids: list[str] = []
        for doc_id in srv._load_session_index(session_id):
            doc = srv._load_doc(doc_id, session_id)
            if doc is None:
                continue
            enriched = srv._enrich_doc(doc, session_id)
            _, reference_spans = _resolve_prompt_lab_reference(
                enriched,
                str(reference_source),
                str(fallback_reference_source),
            )
            if reference_spans:
                doc_ids.append(doc_id)

    if not doc_ids:
        raise HTTPException(
            status_code=400,
            detail=f"No reference-annotated documents found in session '{session_id}'.",
        )

    return body.model_copy(update={"doc_ids": doc_ids, "folder_ids": requested_folder_ids})


def resolve_prompt_lab_runtime(runtime: Any) -> dict[str, object]:
    srv = _server()
    cfg = srv._load_config()
    api_key = (
        runtime.api_key
        or srv.os.environ.get("LITELLM_API_KEY", "")
        or srv.os.environ.get("OPENAI_API_KEY", "")
        or srv.os.environ.get("ANTHROPIC_API_KEY", "")
        or srv.os.environ.get("GEMINI_API_KEY", "")
        or srv.os.environ.get("GOOGLE_API_KEY", "")
    )
    api_base = (
        runtime.api_base
        or str(cfg.get("api_base", "") or "")
        or srv.os.environ.get("LITELLM_BASE_URL", "")
    )
    match_mode = srv._normalize_match_mode(runtime.match_mode)
    chunk_mode = srv._normalize_chunk_mode(runtime.chunk_mode)
    chunk_size_chars = srv._normalize_chunk_size(runtime.chunk_size_chars)
    method_bundle = srv._normalize_method_bundle(getattr(runtime, "method_bundle", "audited"))
    return {
        "api_key": api_key,
        "api_base": api_base,
        "temperature": runtime.temperature,
        "match_mode": match_mode,
        "reference_source": runtime.reference_source,
        "fallback_reference_source": runtime.fallback_reference_source,
        "chunk_mode": chunk_mode,
        "chunk_size_chars": chunk_size_chars,
        "method_bundle": method_bundle,
    }


def validate_prompt_lab_request(
    body: Any,
    session_id: str = "default",
    *,
    method_bundle: str = "audited",
):
    srv = _server()
    cfg = srv._load_config()
    prompt_lab_max_concurrency = srv._get_prompt_lab_max_concurrency(cfg)
    if not body.doc_ids:
        raise HTTPException(status_code=400, detail="doc_ids is required")
    if len(body.prompts) < 1 or len(body.prompts) > srv.PROMPT_LAB_MAX_VARIANTS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Prompt variants must be between 1 and {srv.PROMPT_LAB_MAX_VARIANTS}"
            ),
        )
    if len(body.models) < 1 or len(body.models) > srv.PROMPT_LAB_MAX_VARIANTS:
        raise HTTPException(
            status_code=400,
            detail=f"Model variants must be between 1 and {srv.PROMPT_LAB_MAX_VARIANTS}",
        )
    srv._validate_experiment_concurrency(
        body.concurrency,
        max_allowed=prompt_lab_max_concurrency,
    )
    if len(body.prompts) * len(body.models) > srv.PROMPT_LAB_MAX_VARIANTS * srv.PROMPT_LAB_MAX_VARIANTS:
        raise HTTPException(status_code=400, detail="Matrix limit exceeded (max 6x6)")

    for doc_id in body.doc_ids:
        if srv._load_doc(doc_id, session_id) is None:
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
    for folder_id in getattr(body, "folder_ids", []):
        if srv._load_folder(folder_id, session_id) is None:
            raise HTTPException(status_code=404, detail=f"Folder not found: {folder_id}")

    prompt_ids_seen: set[str] = set()
    for index, prompt in enumerate(body.prompts):
        if not prompt.label.strip():
            raise HTTPException(status_code=400, detail=f"Prompt label required at index {index}")
        if prompt.variant_type == "prompt":
            if prompt.method_verify_override is not None:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "method_verify_override is only supported for preset variants "
                        f"(prompt index {index})"
                    ),
                )
            if prompt.preset_method_id:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "preset_method_id is only supported for preset variants "
                        f"(prompt index {index})"
                    ),
                )
            prompt_text = (prompt.system_prompt or "").strip()
            if not prompt_text:
                raise HTTPException(
                    status_code=400,
                    detail=f"system_prompt required at prompt index {index}",
                )
            if "{" in prompt_text or "}" in prompt_text:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Prompt variants must be plain system prompt text "
                        "(templating is not supported)"
                    ),
                )
        elif prompt.variant_type == "preset":
            method_id = (prompt.preset_method_id or "").strip()
            if not method_id:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "preset_method_id required for preset variant "
                        f"at prompt index {index}"
                    ),
                )
            allowed_preset_methods = srv._get_prompt_lab_allowed_preset_methods(
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
            if srv.get_method_definition_by_id(
                method_id,
                method_bundle=method_bundle,  # type: ignore[arg-type]
            ) is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown preset method: {method_id}",
                )
            definition = srv.get_method_definition_by_id(
                method_id,
                method_bundle=method_bundle,  # type: ignore[arg-type]
            )
            if (
                prompt.method_verify_override is not None
                and isinstance(definition, dict)
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
    doc: Any,
    reference_source: str,
    fallback_reference_source: str,
) -> tuple[str, list[Any]]:
    primary = doc.manual_annotations if reference_source == "manual" else doc.pre_annotations
    if primary:
        return reference_source, primary
    fallback = doc.manual_annotations if fallback_reference_source == "manual" else doc.pre_annotations
    return fallback_reference_source, fallback


def _collect_reference_label_set(
    *,
    doc_ids: list[str],
    session_id: str,
    reference_source: str,
    fallback_reference_source: str,
) -> set[str]:
    srv = _server()
    labels: set[str] = set()
    for doc_id in doc_ids:
        doc = srv._load_doc(str(doc_id), session_id)
        if doc is None:
            continue
        enriched = srv._enrich_doc(doc, session_id)
        _resolved_source, reference_spans = _resolve_prompt_lab_reference(
            enriched,
            reference_source,
            fallback_reference_source,
        )
        labels.update(str(span.label) for span in reference_spans)
    return labels


def initialize_prompt_lab_run(
    body: Any,
    session_id: str,
    runtime: dict[str, object],
) -> dict:
    srv = _server()
    run_id = str(srv.uuid.uuid4())[:8]
    now = srv._now_iso()
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

    return {
        "id": run_id,
        "name": (body.name or "").strip() or f"Prompt Lab {now}",
        "status": "queued",
        "created_at": now,
        "started_at": None,
        "finished_at": None,
        "doc_ids": list(dict.fromkeys(body.doc_ids)),
        "folder_ids": list(dict.fromkeys(getattr(body, "folder_ids", []))),
        "prompts": prompts,
        "models": models,
        "runtime": {
            "temperature": runtime["temperature"],
            "match_mode": runtime["match_mode"],
            "reference_source": runtime["reference_source"],
            "fallback_reference_source": runtime["fallback_reference_source"],
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


def _accumulate_metrics(
    metrics: object,
    *,
    counts: dict[str, int],
    per_label_counts: dict[str, dict[str, int]],
    co_primary_counts: dict[str, dict[str, Any]],
) -> None:
    if not isinstance(metrics, dict):
        return
    micro = metrics.get("micro", {})
    if isinstance(micro, dict):
        counts["tp"] += int(micro.get("tp", 0))
        counts["fp"] += int(micro.get("fp", 0))
        counts["fn"] += int(micro.get("fn", 0))
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
            aggregate["support"] += int(label_metrics.get("support", label_tp + label_fn))
    co_primary_metrics = metrics.get("co_primary_metrics", {})
    if isinstance(co_primary_metrics, dict):
        for metric_name, metric_payload in co_primary_metrics.items():
            if not isinstance(metric_payload, dict):
                continue
            aggregate_metric = co_primary_counts.setdefault(
                str(metric_name),
                {
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "per_label": {},
                },
            )
            nested_micro = metric_payload.get("micro", {})
            if isinstance(nested_micro, dict):
                aggregate_metric["tp"] += int(nested_micro.get("tp", 0))
                aggregate_metric["fp"] += int(nested_micro.get("fp", 0))
                aggregate_metric["fn"] += int(nested_micro.get("fn", 0))
            nested_per_label = metric_payload.get("per_label", {})
            if isinstance(nested_per_label, dict):
                for label, label_metrics in nested_per_label.items():
                    if not isinstance(label_metrics, dict):
                        continue
                    aggregate_label = aggregate_metric["per_label"].setdefault(
                        str(label),
                        {"tp": 0, "fp": 0, "fn": 0, "support": 0},
                    )
                    label_tp = int(label_metrics.get("tp", 0))
                    label_fp = int(label_metrics.get("fp", 0))
                    label_fn = int(label_metrics.get("fn", 0))
                    aggregate_label["tp"] += label_tp
                    aggregate_label["fp"] += label_fp
                    aggregate_label["fn"] += label_fn
                    aggregate_label["support"] += int(
                        label_metrics.get("support", label_tp + label_fn)
                    )


def _build_metric_summary(
    *,
    counts: dict[str, int],
    per_label_counts: dict[str, dict[str, int]],
    co_primary_counts: dict[str, dict[str, Any]],
) -> tuple[dict[str, float | int], dict[str, dict[str, float | int]], dict[str, dict[str, Any]]]:
    srv = _server()
    micro = srv._prf_from_counts(counts["tp"], counts["fp"], counts["fn"])
    per_label_summary: dict[str, dict[str, float | int]] = {}
    for label, label_counts in sorted(per_label_counts.items()):
        label_prf = srv._prf_from_counts(
            label_counts["tp"],
            label_counts["fp"],
            label_counts["fn"],
        )
        per_label_summary[label] = {
            **label_prf,
            "support": label_counts["support"],
        }
    co_primary_summary: dict[str, dict[str, Any]] = {}
    for metric_name, metric_counts in sorted(co_primary_counts.items()):
        metric_per_label: dict[str, dict[str, float | int]] = {}
        for label, label_counts in sorted(metric_counts["per_label"].items()):
            label_prf = srv._prf_from_counts(
                label_counts["tp"],
                label_counts["fp"],
                label_counts["fn"],
            )
            metric_per_label[label] = {
                **label_prf,
                "support": label_counts["support"],
            }
        co_primary_summary[metric_name] = {
            "micro": srv._prf_from_counts(
                metric_counts["tp"],
                metric_counts["fp"],
                metric_counts["fn"],
            ),
            "per_label": metric_per_label,
        }
    return micro, per_label_summary, co_primary_summary


def _accumulate_resolution_summary(
    summary: object,
    *,
    totals: dict[str, int],
    by_label_totals: dict[str, dict[str, int]],
) -> None:
    if not isinstance(summary, dict):
        return
    totals["boundary_fix_count"] += int(summary.get("boundary_fix_count", 0))
    totals["augmentation_count"] += int(summary.get("augmentation_count", 0))
    boundary_by_label = summary.get("boundary_fix_count_by_label", {})
    if isinstance(boundary_by_label, dict):
        for label, value in boundary_by_label.items():
            bucket = by_label_totals.setdefault(
                str(label),
                {"boundary_fix_count": 0, "augmentation_count": 0},
            )
            bucket["boundary_fix_count"] += int(value)
    augmentation_by_label = summary.get("augmentation_count_by_label", {})
    if isinstance(augmentation_by_label, dict):
        for label, value in augmentation_by_label.items():
            bucket = by_label_totals.setdefault(
                str(label),
                {"boundary_fix_count": 0, "augmentation_count": 0},
            )
            bucket["augmentation_count"] += int(value)


def _build_prompt_lab_cell_summary(cell: dict, total_docs: int, run_status: str) -> dict:
    srv = _server()
    docs_raw = cell.get("documents", {})
    documents = docs_raw if isinstance(docs_raw, dict) else {}
    completed_docs = 0
    failed_docs = 0
    pending_docs = 0
    metric_counts = {"tp": 0, "fp": 0, "fn": 0}
    raw_metric_counts = {"tp": 0, "fp": 0, "fn": 0}
    confidence_values: list[float] = []
    per_label_counts: dict[str, dict[str, int]] = {}
    raw_per_label_counts: dict[str, dict[str, int]] = {}
    co_primary_counts: dict[str, dict[str, Any]] = {}
    raw_co_primary_counts: dict[str, dict[str, Any]] = {}
    error_families: dict[str, int] = {}
    resolution_totals = {"boundary_fix_count": 0, "augmentation_count": 0}
    resolution_by_label_totals: dict[str, dict[str, int]] = {}

    for result in documents.values():
        if not isinstance(result, dict):
            continue
        status = str(result.get("status", "pending"))
        if status == "completed":
            completed_docs += 1
            _accumulate_metrics(
                result.get("metrics"),
                counts=metric_counts,
                per_label_counts=per_label_counts,
                co_primary_counts=co_primary_counts,
            )
            _accumulate_metrics(
                result.get("raw_metrics"),
                counts=raw_metric_counts,
                per_label_counts=raw_per_label_counts,
                co_primary_counts=raw_co_primary_counts,
            )
            _accumulate_resolution_summary(
                result.get("resolution_summary"),
                totals=resolution_totals,
                by_label_totals=resolution_by_label_totals,
            )
            llm_confidence = result.get("llm_confidence")
            if isinstance(llm_confidence, dict):
                conf = srv._safe_float(llm_confidence.get("confidence"))
                if conf is not None:
                    confidence_values.append(conf)
            continue
        if status in {"failed", "unavailable"}:
            failed_docs += 1
            error_family = result.get("error_family")
            if not isinstance(error_family, str) or not error_family:
                error_family = srv._normalize_error_family(result.get("error"))
            if error_family:
                error_families[error_family] = error_families.get(error_family, 0) + 1
            continue
        if status in {"pending", "running", "queued"}:
            pending_docs += 1

    processed = completed_docs + failed_docs
    if completed_docs > 0:
        micro, per_label_summary, co_primary_summary = _build_metric_summary(
            counts=metric_counts,
            per_label_counts=per_label_counts,
            co_primary_counts=co_primary_counts,
        )
        raw_micro, raw_per_label_summary, raw_co_primary_summary = _build_metric_summary(
            counts=raw_metric_counts,
            per_label_counts=raw_per_label_counts,
            co_primary_counts=raw_co_primary_counts,
        )
    else:
        micro = srv._prf_from_counts(0, 0, 0)
        raw_micro = srv._prf_from_counts(0, 0, 0)
        per_label_summary = {}
        raw_per_label_summary = {}
        co_primary_summary = {}
        raw_co_primary_summary = {}
    if run_status == "cancelled" and processed < total_docs:
        status = "cancelled"
    elif run_status == "cancelling" and processed < total_docs:
        status = "cancelling"
    elif processed == 0:
        status = "pending"
    elif processed < total_docs:
        status = "running"
    elif failed_docs > 0:
        status = "completed_with_errors"
    else:
        status = "completed"
    tolerant_micro = (
        co_primary_summary.get("overlap", {}).get("micro", {})
        if isinstance(co_primary_summary.get("overlap"), dict)
        else {}
    )
    raw_tolerant_micro = (
        raw_co_primary_summary.get("overlap", {}).get("micro", {})
        if isinstance(raw_co_primary_summary.get("overlap"), dict)
        else {}
    )

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
        "pending_docs": pending_docs,
        "error_count": failed_docs,
        "error_families": error_families,
        "micro": micro,
        "raw_micro": raw_micro,
        "co_primary_metrics": co_primary_summary,
        "raw_co_primary_metrics": raw_co_primary_summary,
        "per_label": per_label_summary,
        "raw_per_label": raw_per_label_summary,
        "overlap_gap_f1": float(tolerant_micro.get("f1", 0.0)) - float(micro.get("f1", 0.0)),
        "raw_overlap_gap_f1": float(raw_tolerant_micro.get("f1", 0.0))
        - float(raw_micro.get("f1", 0.0)),
        "resolution_summary": {
            "boundary_fix_count": resolution_totals["boundary_fix_count"],
            "augmentation_count": resolution_totals["augmentation_count"],
            "boundary_fix_count_by_label": {
                label: bucket["boundary_fix_count"]
                for label, bucket in sorted(resolution_by_label_totals.items())
                if bucket["boundary_fix_count"] > 0
            },
            "augmentation_count_by_label": {
                label: bucket["augmentation_count"]
                for label, bucket in sorted(resolution_by_label_totals.items())
                if bucket["augmentation_count"] > 0
            },
        },
        "mean_confidence": (
            sum(confidence_values) / len(confidence_values) if confidence_values else None
        ),
    }


def build_prompt_lab_matrix(run: dict) -> dict:
    total_docs = len(run.get("doc_ids", []))
    run_status = str(run.get("status", ""))
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
            summary = _build_prompt_lab_cell_summary(cell, total_docs, run_status)
            per_label = summary.get("per_label", {})
            if isinstance(per_label, dict):
                available_labels.update(str(label) for label in per_label.keys())
            summaries.append(summary)
    return {
        "models": [
            {"id": str(item.get("id", "")), "label": str(item.get("label", ""))}
            for item in models
        ],
        "prompts": [
            {"id": str(item.get("id", "")), "label": str(item.get("label", ""))}
            for item in prompts
        ],
        "cells": summaries,
        "available_labels": sorted(available_labels),
    }


def _read_method_bundle_for_summary(value: object) -> str:
    raw = str(value or "audited").strip().lower()
    if raw in {
        "legacy",
        "audited",
        "test",
        "v2+post-process",
        "stable",
        "v2",
        "deidentify-v2",
    }:
        return raw
    return "audited"


def build_prompt_lab_run_summary(run: dict) -> dict:
    srv = _server()
    matrix = build_prompt_lab_matrix(run)
    cells = matrix["cells"]
    completed = 0
    failed = 0
    total = len(run.get("doc_ids", [])) * len(run.get("models", [])) * len(run.get("prompts", []))
    run_id = str(run.get("id", ""))
    session_id = str(run.get("session_id", "default") or "default")
    status = str(run.get("status", ""))
    runtime_raw = run.get("runtime", {})
    runtime = runtime_raw if isinstance(runtime_raw, dict) else {}
    method_bundle = _read_method_bundle_for_summary(runtime.get("method_bundle", "audited"))
    for cell in cells:
        completed += int(cell.get("completed_docs", 0)) + int(cell.get("failed_docs", 0))
        failed += int(cell.get("failed_docs", 0))
    return {
        "id": run_id,
        "name": run.get("name"),
        "status": status,
        "created_at": run.get("created_at"),
        "started_at": run.get("started_at"),
        "finished_at": run.get("finished_at"),
        "doc_count": len(run.get("doc_ids", [])),
        "prompt_count": len(run.get("prompts", [])),
        "model_count": len(run.get("models", [])),
        "total_tasks": total,
        "completed_tasks": completed,
        "failed_tasks": failed,
        "method_bundle": method_bundle,
        "cancellable": (
            status in {"queued", "running", "cancelling"}
            and srv._get_prompt_lab_cancel_event(run_id, session_id) is not None
        ),
    }


def build_prompt_lab_run_detail(run: dict) -> dict:
    srv = _server()
    summary = build_prompt_lab_run_summary(run)
    matrix = build_prompt_lab_matrix(run)
    runtime_raw = run.get("runtime", {})
    runtime = runtime_raw if isinstance(runtime_raw, dict) else {}
    diagnostics = srv._build_experiment_run_diagnostics(run, kind="prompt_lab")
    return {
        **summary,
        "doc_ids": run.get("doc_ids", []),
        "folder_ids": run.get("folder_ids", []),
        "prompts": run.get("prompts", []),
        "models": run.get("models", []),
        "runtime": {
            "temperature": runtime.get("temperature", 0.0),
            "match_mode": runtime.get("match_mode", "exact"),
            "reference_source": runtime.get("reference_source", "manual"),
            "fallback_reference_source": runtime.get("fallback_reference_source", "pre"),
            "method_bundle": runtime.get("method_bundle", "audited"),
            "api_base": runtime.get("api_base", ""),
            "chunk_mode": runtime.get("chunk_mode", "off"),
            "chunk_size_chars": runtime.get("chunk_size_chars", srv.DEFAULT_CHUNK_SIZE_CHARS),
        },
        "concurrency": run.get("concurrency", srv.PROMPT_LAB_DEFAULT_CONCURRENCY),
        "warnings": run.get("warnings", []),
        "errors": run.get("errors", []),
        "diagnostics": diagnostics,
        "matrix": matrix,
        "progress": {
            "total_tasks": summary["total_tasks"],
            "completed_tasks": summary["completed_tasks"],
            "failed_tasks": summary["failed_tasks"],
        },
    }


def _mark_pending_documents_cancelled(cells: object, *, updated_at: str):
    if not isinstance(cells, dict):
        return
    for cell in cells.values():
        if not isinstance(cell, dict):
            continue
        documents = cell.get("documents", {})
        if not isinstance(documents, dict):
            continue
        for doc_id, result in list(documents.items()):
            if not isinstance(result, dict):
                documents[doc_id] = {
                    "status": "cancelled",
                    "updated_at": updated_at,
                }
                continue
            if str(result.get("status", "pending")) in {
                "completed",
                "failed",
                "unavailable",
                "cancelled",
            }:
                continue
            result["status"] = "cancelled"
            result["updated_at"] = updated_at


def _mark_prompt_lab_run_cancelled(run_id: str, session_id: str):
    srv = _server()
    with srv._prompt_lab_lock:
        latest = srv._load_prompt_lab_run(run_id, session_id)
        if latest is None:
            return
        if srv._is_terminal_prompt_lab_status(str(latest.get("status", ""))):
            return
        updated_at = srv._now_iso()
        _mark_pending_documents_cancelled(latest.get("cells"), updated_at=updated_at)
        latest["status"] = "cancelled"
        latest["finished_at"] = updated_at
        srv._save_prompt_lab_run(latest, session_id)


def run_prompt_lab_job(
    run_id: str,
    session_id: str,
    runtime: dict[str, object],
    method_bundle: str = "audited",
):
    srv = _server()
    cancel_event = srv._get_prompt_lab_cancel_event(run_id, session_id)
    api_key = str(runtime["api_key"])
    api_base = str(runtime["api_base"])
    temperature = float(runtime["temperature"])
    match_mode = str(runtime["match_mode"])
    reference_source = str(runtime["reference_source"])
    fallback_reference_source = str(runtime["fallback_reference_source"])
    method_bundle = srv._normalize_method_bundle(runtime.get("method_bundle", "audited"))
    chunk_mode = str(runtime["chunk_mode"])
    chunk_size_chars = int(runtime["chunk_size_chars"])
    with srv._prompt_lab_lock:
        run = srv._load_prompt_lab_run(run_id, session_id)
        if run is None:
            srv._clear_prompt_lab_cancel_event(run_id, session_id)
            return
        if cancel_event is not None and cancel_event.is_set():
            srv._save_prompt_lab_run(run, session_id)
            _mark_prompt_lab_run_cancelled(run_id, session_id)
            srv._clear_prompt_lab_cancel_event(run_id, session_id)
            return
        run["status"] = "running"
        run["started_at"] = srv._now_iso()
        srv._save_prompt_lab_run(run, session_id)

    tasks: list[tuple[str, str, str]] = []
    for model in run.get("models", []):
        for prompt in run.get("prompts", []):
            cell_id = f"{model['id']}__{prompt['id']}"
            for doc_id in run.get("doc_ids", []):
                tasks.append((cell_id, str(doc_id), str(model["id"])))

    model_by_id = {str(model["id"]): model for model in run.get("models", [])}
    prompt_by_id = {str(prompt["id"]): prompt for prompt in run.get("prompts", [])}
    reference_label_set = _collect_reference_label_set(
        doc_ids=[str(doc_id) for doc_id in run.get("doc_ids", [])],
        session_id=session_id,
        reference_source=reference_source,
        fallback_reference_source=fallback_reference_source,
    )

    def _execute_task(cell_id: str, doc_id: str, model_id: str) -> tuple[str, str, dict]:
        cell = run.get("cells", {}).get(cell_id, {})
        if not isinstance(cell, dict):
            error = f"Unknown cell '{cell_id}'"
            return cell_id, doc_id, {
                "status": "failed",
                "error": error,
                "error_family": srv._normalize_error_family(error),
            }
        prompt_id = str(cell.get("prompt_id", ""))
        prompt = prompt_by_id.get(prompt_id)
        model = model_by_id.get(model_id)
        if prompt is None or model is None:
            error = "Invalid model/prompt mapping"
            return cell_id, doc_id, {
                "status": "failed",
                "error": error,
                "error_family": srv._normalize_error_family(error),
            }

        doc = srv._load_doc(doc_id, session_id)
        if doc is None:
            return cell_id, doc_id, {"status": "unavailable", "error": "Document no longer exists"}
        enriched = srv._enrich_doc(doc, session_id)
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
                ) = srv._run_method_for_document(
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
                    label_profile="simple",
                    chunk_mode=chunk_mode,
                    chunk_size_chars=chunk_size_chars,
                    method_bundle=method_bundle,  # type: ignore[arg-type]
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
                ) = srv._run_llm_for_document(
                    doc=enriched,
                    api_key=api_key,
                    api_base=api_base or None,
                    model=str(model["model"]),
                    system_prompt=requested_system_prompt,
                    temperature=temperature,
                    reasoning_effort=str(model["reasoning_effort"]),
                    anthropic_thinking=bool(model["anthropic_thinking"]),
                    anthropic_thinking_budget_tokens=model["anthropic_thinking_budget_tokens"],
                    label_profile="simple",
                    chunk_mode=chunk_mode,
                    chunk_size_chars=chunk_size_chars,
                    method_bundle=method_bundle,
                )

            resolved_reference_source, reference_spans = _resolve_prompt_lab_reference(
                enriched,
                reference_source,
                fallback_reference_source,
            )
            projected_reference, projected_hypothesis = srv._prepare_experiment_scoring_spans(
                reference_spans,
                hypothesis_spans,
                method_bundle=method_bundle,
                reference_label_set=reference_label_set,
            )
            projected_reference_raw, projected_hypothesis_raw = srv._prepare_experiment_scoring_spans(
                reference_spans,
                raw_hypothesis_spans,
                method_bundle=method_bundle,
                reference_label_set=reference_label_set,
            )
            scoring_match_mode = srv._normalize_metrics_mode_for_method_bundle(
                match_mode,
                method_bundle=method_bundle,
            )
            metrics = srv._serialize_metrics_payload(
                srv.compute_metrics(
                    projected_reference,
                    projected_hypothesis,
                    scoring_match_mode,
                )
            )
            raw_metrics = srv._serialize_metrics_payload(
                srv.compute_metrics(
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
                "resolution_summary": srv.summarize_resolution_events(resolution_events),
                "updated_at": srv._now_iso(),
                "filename": enriched.filename,
            }
        except Exception as exc:
            message = str(exc).strip() or exc.__class__.__name__
            if len(message) > 800:
                message = f"{message[:800]}..."
            return cell_id, doc_id, {
                "status": "failed",
                "error": message,
                "error_family": srv._normalize_error_family(message),
                "updated_at": srv._now_iso(),
                "filename": enriched.filename,
            }

    max_workers = srv._resolve_experiment_worker_count(
        int(run.get("concurrency", srv.PROMPT_LAB_DEFAULT_CONCURRENCY)),
        total_tasks=len(tasks),
        max_allowed=srv._get_prompt_lab_max_concurrency(),
    )
    executor = srv.ThreadPoolExecutor(max_workers=max_workers)
    cancelled = False
    try:
        future_map = {
            executor.submit(_execute_task, cell_id, doc_id, model_id): (cell_id, doc_id)
            for cell_id, doc_id, model_id in tasks
        }
        while future_map:
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                for future in future_map:
                    future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                _mark_prompt_lab_run_cancelled(run_id, session_id)
                return
            done, _ = concurrent.futures.wait(
                set(future_map.keys()),
                timeout=0.2,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            if not done:
                continue
            for future in done:
                cell_id, doc_id = future_map.pop(future)
                try:
                    _, _, result = future.result()
                except Exception as exc:
                    message = str(exc).strip() or exc.__class__.__name__
                    result = {
                        "status": "failed",
                        "error": message,
                        "error_family": srv._normalize_error_family(message),
                        "updated_at": srv._now_iso(),
                    }
                with srv._prompt_lab_lock:
                    latest = srv._load_prompt_lab_run(run_id, session_id)
                    if latest is None:
                        continue
                    cells = latest.get("cells", {})
                    if isinstance(cells, dict) and isinstance(cells.get(cell_id), dict):
                        cells[cell_id]["documents"][doc_id] = result
                    srv._save_prompt_lab_run(latest, session_id)
    except Exception as exc:
        with srv._prompt_lab_lock:
            latest = srv._load_prompt_lab_run(run_id, session_id)
            if latest is not None:
                latest["status"] = "failed"
                latest["finished_at"] = srv._now_iso()
                errors = latest.get("errors", [])
                if not isinstance(errors, list):
                    errors = []
                errors.append(f"Prompt Lab run failed: {exc}")
                latest["errors"] = errors
                srv._save_prompt_lab_run(latest, session_id)
        return
    finally:
        if not cancelled:
            executor.shutdown(wait=True)
        srv._clear_prompt_lab_cancel_event(run_id, session_id)

    with srv._prompt_lab_lock:
        latest = srv._load_prompt_lab_run(run_id, session_id)
        if latest is None:
            return
        summary = build_prompt_lab_run_summary(latest)
        latest["status"] = (
            "completed_with_errors"
            if int(summary.get("failed_tasks", 0)) > 0
            else "completed"
        )
        latest["finished_at"] = srv._now_iso()
        srv._save_prompt_lab_run(latest, session_id)


def create_prompt_lab_run(
    body: Any,
    *,
    session_id: str = "default",
    context_dir: Path | None = None,
    run_async: bool = False,
) -> dict:
    srv = _server()
    prepared = prepare_prompt_lab_body(body, session_id=session_id, context_dir=context_dir)
    runtime = resolve_prompt_lab_runtime(prepared.runtime)
    method_bundle = str(runtime["method_bundle"])
    validate_prompt_lab_request(prepared, session_id, method_bundle=method_bundle)
    api_key = str(runtime["api_key"])
    api_base = str(runtime["api_base"])
    requires_llm = False
    for prompt in prepared.prompts:
        if prompt.variant_type == "prompt":
            requires_llm = True
            break
        preset_method = srv.get_method_definition_by_id(
            str(prompt.preset_method_id or ""),
            method_bundle=method_bundle,  # type: ignore[arg-type]
        )
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
        for model in prepared.models:
            srv._validate_gateway_model_access(model=model.model, api_base=api_base, api_key=api_key)

    with srv._prompt_lab_lock:
        run = initialize_prompt_lab_run(prepared, session_id, runtime)
        ids = srv._load_prompt_lab_index(session_id)
        ids.append(str(run["id"]))
        srv._save_prompt_lab_run(run, session_id)
        srv._save_prompt_lab_index(session_id)
        srv._register_prompt_lab_cancel_event(str(run["id"]), session_id)

    if run_async:
        worker = threading.Thread(
            target=run_prompt_lab_job,
            args=(str(run["id"]), session_id, runtime, method_bundle),
            daemon=True,
        )
        worker.start()
        return build_prompt_lab_run_detail(run)

    run_prompt_lab_job(str(run["id"]), session_id, runtime, method_bundle)
    latest = srv._load_prompt_lab_run(str(run["id"]), session_id)
    if latest is None:
        raise RuntimeError("Prompt Lab run disappeared before completion.")
    return build_prompt_lab_run_detail(latest)


def resolve_methods_lab_runtime(runtime: Any) -> dict[str, object]:
    srv = _server()
    cfg = srv._load_config()
    api_key = (
        runtime.api_key
        or srv.os.environ.get("LITELLM_API_KEY", "")
        or srv.os.environ.get("OPENAI_API_KEY", "")
        or srv.os.environ.get("ANTHROPIC_API_KEY", "")
        or srv.os.environ.get("GEMINI_API_KEY", "")
        or srv.os.environ.get("GOOGLE_API_KEY", "")
    )
    api_base = (
        runtime.api_base
        or str(cfg.get("api_base", "") or "")
        or srv.os.environ.get("LITELLM_BASE_URL", "")
    )
    match_mode = srv._normalize_match_mode(runtime.match_mode)
    chunk_mode = srv._normalize_chunk_mode(runtime.chunk_mode)
    chunk_size_chars = srv._normalize_chunk_size(runtime.chunk_size_chars)
    method_bundle = srv._normalize_method_bundle(getattr(runtime, "method_bundle", "audited"))
    task_timeout_seconds = srv._safe_float(getattr(runtime, "task_timeout_seconds", None))
    if task_timeout_seconds is not None and task_timeout_seconds <= 0:
        raise HTTPException(status_code=400, detail="task_timeout_seconds must be greater than 0")
    return {
        "api_key": api_key,
        "api_base": api_base,
        "temperature": runtime.temperature,
        "match_mode": match_mode,
        "reference_source": runtime.reference_source,
        "fallback_reference_source": runtime.fallback_reference_source,
        "chunk_mode": chunk_mode,
        "chunk_size_chars": chunk_size_chars,
        "method_bundle": method_bundle,
        "task_timeout_seconds": task_timeout_seconds,
    }


def validate_methods_lab_request(
    body: Any,
    session_id: str = "default",
    *,
    method_bundle: str = "audited",
):
    srv = _server()
    cfg = srv._load_config()
    methods_lab_max_concurrency = srv._get_methods_lab_max_concurrency(cfg)
    if not body.doc_ids:
        raise HTTPException(status_code=400, detail="doc_ids is required")
    if len(body.methods) < 1 or len(body.methods) > srv.METHODS_LAB_MAX_METHOD_VARIANTS:
        raise HTTPException(
            status_code=400,
            detail=(
                "Method variants must be between 1 and "
                f"{srv.METHODS_LAB_MAX_METHOD_VARIANTS}"
            ),
        )
    if len(body.models) < 1 or len(body.models) > srv.PROMPT_LAB_MAX_VARIANTS:
        raise HTTPException(
            status_code=400,
            detail=f"Model variants must be between 1 and {srv.PROMPT_LAB_MAX_VARIANTS}",
        )
    srv._validate_experiment_concurrency(
        body.concurrency,
        max_allowed=methods_lab_max_concurrency,
    )

    for doc_id in body.doc_ids:
        if srv._load_doc(doc_id, session_id) is None:
            raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
    for folder_id in getattr(body, "folder_ids", []):
        if srv._load_folder(folder_id, session_id) is None:
            raise HTTPException(status_code=404, detail=f"Folder not found: {folder_id}")

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
        definition = srv.get_method_definition_by_id(
            method_id,
            method_bundle=method_bundle,  # type: ignore[arg-type]
        )
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


def initialize_methods_lab_run(
    body: Any,
    session_id: str,
    runtime: dict[str, object],
) -> dict:
    srv = _server()
    run_id = str(srv.uuid.uuid4())[:8]
    now = srv._now_iso()
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
        "folder_ids": list(dict.fromkeys(getattr(body, "folder_ids", []))),
        "methods": methods,
        "models": models,
        "runtime": {
            "temperature": runtime["temperature"],
            "match_mode": runtime["match_mode"],
            "reference_source": runtime["reference_source"],
            "fallback_reference_source": runtime["fallback_reference_source"],
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


def _build_methods_lab_cell_summary(cell: dict, total_docs: int, run_status: str) -> dict:
    srv = _server()
    docs_raw = cell.get("documents", {})
    documents = docs_raw if isinstance(docs_raw, dict) else {}
    completed_docs = 0
    failed_docs = 0
    pending_docs = 0
    metric_counts = {"tp": 0, "fp": 0, "fn": 0}
    raw_metric_counts = {"tp": 0, "fp": 0, "fn": 0}
    confidence_values: list[float] = []
    per_label_counts: dict[str, dict[str, int]] = {}
    raw_per_label_counts: dict[str, dict[str, int]] = {}
    co_primary_counts: dict[str, dict[str, Any]] = {}
    raw_co_primary_counts: dict[str, dict[str, Any]] = {}
    error_families: dict[str, int] = {}
    resolution_totals = {"boundary_fix_count": 0, "augmentation_count": 0}
    resolution_by_label_totals: dict[str, dict[str, int]] = {}

    for result in documents.values():
        if not isinstance(result, dict):
            continue
        status = str(result.get("status", "pending"))
        if status == "completed":
            completed_docs += 1
            _accumulate_metrics(
                result.get("metrics"),
                counts=metric_counts,
                per_label_counts=per_label_counts,
                co_primary_counts=co_primary_counts,
            )
            _accumulate_metrics(
                result.get("raw_metrics"),
                counts=raw_metric_counts,
                per_label_counts=raw_per_label_counts,
                co_primary_counts=raw_co_primary_counts,
            )
            _accumulate_resolution_summary(
                result.get("resolution_summary"),
                totals=resolution_totals,
                by_label_totals=resolution_by_label_totals,
            )
            llm_confidence = result.get("llm_confidence")
            if isinstance(llm_confidence, dict):
                conf = srv._safe_float(llm_confidence.get("confidence"))
                if conf is not None:
                    confidence_values.append(conf)
            continue
        if status in {"failed", "unavailable"}:
            failed_docs += 1
            error_family = result.get("error_family")
            if not isinstance(error_family, str) or not error_family:
                error_family = srv._normalize_error_family(result.get("error"))
            if error_family:
                error_families[error_family] = error_families.get(error_family, 0) + 1
            continue
        if status in {"pending", "running", "queued"}:
            pending_docs += 1

    processed = completed_docs + failed_docs
    if completed_docs > 0:
        micro, per_label_summary, co_primary_summary = _build_metric_summary(
            counts=metric_counts,
            per_label_counts=per_label_counts,
            co_primary_counts=co_primary_counts,
        )
        raw_micro, raw_per_label_summary, raw_co_primary_summary = _build_metric_summary(
            counts=raw_metric_counts,
            per_label_counts=raw_per_label_counts,
            co_primary_counts=raw_co_primary_counts,
        )
    else:
        micro = srv._prf_from_counts(0, 0, 0)
        raw_micro = srv._prf_from_counts(0, 0, 0)
        per_label_summary = {}
        raw_per_label_summary = {}
        co_primary_summary = {}
        raw_co_primary_summary = {}
    if run_status == "cancelled" and processed < total_docs:
        status = "cancelled"
    elif run_status == "cancelling" and processed < total_docs:
        status = "cancelling"
    elif processed == 0:
        status = "pending"
    elif processed < total_docs:
        status = "running"
    elif failed_docs > 0:
        status = "completed_with_errors"
    else:
        status = "completed"
    tolerant_micro = (
        co_primary_summary.get("overlap", {}).get("micro", {})
        if isinstance(co_primary_summary.get("overlap"), dict)
        else {}
    )
    raw_tolerant_micro = (
        raw_co_primary_summary.get("overlap", {}).get("micro", {})
        if isinstance(raw_co_primary_summary.get("overlap"), dict)
        else {}
    )

    return {
        "id": cell.get("id"),
        "model_id": cell.get("model_id"),
        "model_label": cell.get("model_label"),
        "method_id": cell.get("method_id"),
        "method_label": cell.get("method_label"),
        "status": status,
        "total_docs": total_docs,
        "completed_docs": completed_docs,
        "failed_docs": failed_docs,
        "pending_docs": pending_docs,
        "error_count": failed_docs,
        "error_families": error_families,
        "micro": micro,
        "raw_micro": raw_micro,
        "co_primary_metrics": co_primary_summary,
        "raw_co_primary_metrics": raw_co_primary_summary,
        "per_label": per_label_summary,
        "raw_per_label": raw_per_label_summary,
        "overlap_gap_f1": float(tolerant_micro.get("f1", 0.0)) - float(micro.get("f1", 0.0)),
        "raw_overlap_gap_f1": float(raw_tolerant_micro.get("f1", 0.0))
        - float(raw_micro.get("f1", 0.0)),
        "resolution_summary": {
            "boundary_fix_count": resolution_totals["boundary_fix_count"],
            "augmentation_count": resolution_totals["augmentation_count"],
            "boundary_fix_count_by_label": {
                label: bucket["boundary_fix_count"]
                for label, bucket in sorted(resolution_by_label_totals.items())
                if bucket["boundary_fix_count"] > 0
            },
            "augmentation_count_by_label": {
                label: bucket["augmentation_count"]
                for label, bucket in sorted(resolution_by_label_totals.items())
                if bucket["augmentation_count"] > 0
            },
        },
        "mean_confidence": (
            sum(confidence_values) / len(confidence_values) if confidence_values else None
        ),
    }


def build_methods_lab_matrix(run: dict) -> dict:
    total_docs = len(run.get("doc_ids", []))
    run_status = str(run.get("status", ""))
    methods = run.get("methods", [])
    models = run.get("models", [])
    cells_raw = run.get("cells", {})
    cells_dict = cells_raw if isinstance(cells_raw, dict) else {}
    summaries: list[dict] = []
    available_labels: set[str] = set()
    for model in models:
        model_id = str(model.get("id", ""))
        for method in methods:
            method_variant_id = str(method.get("id", ""))
            cell_id = f"{model_id}__{method_variant_id}"
            cell = cells_dict.get(cell_id)
            if not isinstance(cell, dict):
                cell = {
                    "id": cell_id,
                    "model_id": model_id,
                    "model_label": model.get("label", model_id),
                    "method_id": method_variant_id,
                    "method_label": method.get("label", method_variant_id),
                    "documents": {},
                }
            summary = _build_methods_lab_cell_summary(cell, total_docs, run_status)
            per_label = summary.get("per_label", {})
            if isinstance(per_label, dict):
                available_labels.update(str(label) for label in per_label.keys())
            summaries.append(summary)
    return {
        "models": [
            {"id": str(item.get("id", "")), "label": str(item.get("label", ""))}
            for item in models
        ],
        "methods": [
            {"id": str(item.get("id", "")), "label": str(item.get("label", ""))}
            for item in methods
        ],
        "cells": summaries,
        "available_labels": sorted(available_labels),
    }


def build_methods_lab_run_summary(run: dict) -> dict:
    srv = _server()
    matrix = build_methods_lab_matrix(run)
    cells = matrix["cells"]
    completed = 0
    failed = 0
    total = len(run.get("doc_ids", [])) * len(run.get("models", [])) * len(run.get("methods", []))
    run_id = str(run.get("id", ""))
    session_id = str(run.get("session_id", "default") or "default")
    status = str(run.get("status", ""))
    runtime_raw = run.get("runtime", {})
    runtime = runtime_raw if isinstance(runtime_raw, dict) else {}
    method_bundle = _read_method_bundle_for_summary(runtime.get("method_bundle", "audited"))
    for cell in cells:
        completed += int(cell.get("completed_docs", 0)) + int(cell.get("failed_docs", 0))
        failed += int(cell.get("failed_docs", 0))
    return {
        "id": run_id,
        "name": run.get("name"),
        "status": status,
        "created_at": run.get("created_at"),
        "started_at": run.get("started_at"),
        "finished_at": run.get("finished_at"),
        "doc_count": len(run.get("doc_ids", [])),
        "method_count": len(run.get("methods", [])),
        "model_count": len(run.get("models", [])),
        "total_tasks": total,
        "completed_tasks": completed,
        "failed_tasks": failed,
        "method_bundle": method_bundle,
        "cancellable": (
            status in {"queued", "running", "cancelling"}
            and srv._get_methods_lab_cancel_event(run_id, session_id) is not None
        ),
    }


def build_methods_lab_run_detail(run: dict) -> dict:
    srv = _server()
    summary = build_methods_lab_run_summary(run)
    matrix = build_methods_lab_matrix(run)
    runtime_raw = run.get("runtime", {})
    runtime = runtime_raw if isinstance(runtime_raw, dict) else {}
    diagnostics = srv._build_experiment_run_diagnostics(run, kind="methods_lab")
    return {
        **summary,
        "doc_ids": run.get("doc_ids", []),
        "folder_ids": run.get("folder_ids", []),
        "methods": run.get("methods", []),
        "models": run.get("models", []),
        "runtime": {
            "temperature": runtime.get("temperature", 0.0),
            "match_mode": runtime.get("match_mode", "exact"),
            "reference_source": runtime.get("reference_source", "manual"),
            "fallback_reference_source": runtime.get("fallback_reference_source", "pre"),
            "method_bundle": runtime.get("method_bundle", "audited"),
            "api_base": runtime.get("api_base", ""),
            "chunk_mode": runtime.get("chunk_mode", "off"),
            "chunk_size_chars": runtime.get("chunk_size_chars", srv.DEFAULT_CHUNK_SIZE_CHARS),
            "task_timeout_seconds": runtime.get("task_timeout_seconds"),
        },
        "concurrency": run.get("concurrency", srv.METHODS_LAB_DEFAULT_CONCURRENCY),
        "warnings": run.get("warnings", []),
        "errors": run.get("errors", []),
        "diagnostics": diagnostics,
        "matrix": matrix,
        "progress": {
            "total_tasks": summary["total_tasks"],
            "completed_tasks": summary["completed_tasks"],
            "failed_tasks": summary["failed_tasks"],
        },
    }


def _mark_methods_lab_run_cancelled(run_id: str, session_id: str):
    srv = _server()
    with srv._methods_lab_lock:
        latest = srv._load_methods_lab_run(run_id, session_id)
        if latest is None:
            return
        if srv._is_terminal_methods_lab_status(str(latest.get("status", ""))):
            return
        updated_at = srv._now_iso()
        _mark_pending_documents_cancelled(latest.get("cells"), updated_at=updated_at)
        latest["status"] = "cancelled"
        latest["finished_at"] = updated_at
        srv._save_methods_lab_run(latest, session_id)


def _persist_methods_lab_document_runtime(
    *,
    run_id: str,
    session_id: str,
    method_variant_id: str,
    doc_id: str,
    target_model_ids: list[str],
    filename: str | None,
    runtime_diagnostics: dict[str, object],
) -> None:
    srv = _server()
    with srv._methods_lab_lock:
        latest = srv._load_methods_lab_run(run_id, session_id)
        if latest is None:
            return
        cells = latest.get("cells", {})
        if not isinstance(cells, dict):
            return
        updated_at = srv._now_iso()
        for target_model_id in target_model_ids:
            cell_id = f"{target_model_id}__{method_variant_id}"
            cell = cells.get(cell_id)
            if not isinstance(cell, dict):
                continue
            documents = cell.get("documents", {})
            if not isinstance(documents, dict):
                continue
            existing = documents.get(doc_id)
            if not isinstance(existing, dict):
                existing = {"status": "pending"}
                documents[doc_id] = existing
            if str(existing.get("status", "")) in {"completed", "failed", "cancelled", "unavailable"}:
                continue
            existing["status"] = "running"
            existing["updated_at"] = updated_at
            if filename:
                existing["filename"] = filename
            existing["runtime_diagnostics"] = copy.deepcopy(runtime_diagnostics)
        srv._save_methods_lab_run(latest, session_id)


def _build_methods_lab_workspace_run_key(
    *,
    method_id: str,
    model: str,
    run_id: str,
) -> str:
    return f"{method_id}::{model}::{run_id}"


def _sync_methods_lab_result_to_workspace(
    *,
    run_id: str,
    session_id: str,
    doc_id: str,
    method_id: str,
    method_verify_override: bool | None,
    target_model_ids: list[str],
    model_by_id: dict[str, dict[str, Any]],
    method_bundle: str,
    result: dict[str, Any],
) -> None:
    srv = _server()
    if str(result.get("status", "")) != "completed":
        return

    hypothesis_spans_raw = result.get("hypothesis_spans", [])
    if not isinstance(hypothesis_spans_raw, list):
        hypothesis_spans_raw = []
    hypothesis_spans = [
        CanonicalSpan.model_validate(item)
        for item in hypothesis_spans_raw
    ]

    raw_hypothesis_spans_raw = result.get("raw_hypothesis_spans", [])
    if not isinstance(raw_hypothesis_spans_raw, list):
        raw_hypothesis_spans_raw = []
    raw_hypothesis_spans = [
        CanonicalSpan.model_validate(item)
        for item in raw_hypothesis_spans_raw
    ]

    resolution_events_raw = result.get("resolution_events", [])
    if not isinstance(resolution_events_raw, list):
        resolution_events_raw = []
    resolution_events = [
        ResolutionEvent.model_validate(item)
        for item in resolution_events_raw
    ]

    llm_confidence = None
    llm_confidence_raw = result.get("llm_confidence")
    if isinstance(llm_confidence_raw, dict):
        llm_confidence = LLMConfidenceMetric.model_validate(llm_confidence_raw)

    updated_at = str(result.get("updated_at") or srv._now_iso())
    for target_model_id in target_model_ids:
        model_item = model_by_id.get(target_model_id, {})
        workspace_model = (
            str(model_item.get("model") or target_model_id or "rule").strip() or "rule"
        )
        run_key = _build_methods_lab_workspace_run_key(
            method_id=method_id,
            model=workspace_model,
            run_id=run_id,
        )
        srv._upsert_span_map_entry(
            doc_id,
            srv.METHOD_RUNS_SIDECAR_KIND,
            run_key,
            hypothesis_spans,
            session_id,
        )
        srv._upsert_run_metadata(
            doc_id,
            srv.METHOD_RUNS_METADATA_SIDECAR_KIND,
            run_key,
            SavedRunMetadata(
                mode="method",
                updated_at=updated_at,
                method_id=method_id,
                model=workspace_model,
                prompt_snapshot=srv._build_method_prompt_snapshot(
                    method_id=method_id,
                    additional_constraints="",
                    method_verify=method_verify_override,
                    method_bundle=method_bundle,
                ),
                llm_confidence=llm_confidence,
                raw_hypothesis_spans=raw_hypothesis_spans,
                resolution_events=resolution_events,
                resolution_policy_version=(
                    str(result.get("resolution_policy_version"))
                    if result.get("resolution_policy_version") is not None
                    else None
                ),
            ),
            session_id,
        )


def run_methods_lab_job(
    run_id: str,
    session_id: str,
    runtime: dict[str, object],
    method_bundle: str = "audited",
):
    srv = _server()
    cancel_event = srv._get_methods_lab_cancel_event(run_id, session_id)
    api_key = str(runtime["api_key"])
    api_base = str(runtime["api_base"])
    temperature = float(runtime["temperature"])
    match_mode = str(runtime["match_mode"])
    reference_source = str(runtime["reference_source"])
    fallback_reference_source = str(runtime["fallback_reference_source"])
    chunk_mode = str(runtime["chunk_mode"])
    chunk_size_chars = int(runtime["chunk_size_chars"])
    task_timeout_seconds = srv._safe_float(runtime.get("task_timeout_seconds"))

    with srv._methods_lab_lock:
        run = srv._load_methods_lab_run(run_id, session_id)
        if run is None:
            srv._clear_methods_lab_cancel_event(run_id, session_id)
            return
        if cancel_event is not None and cancel_event.is_set():
            srv._save_methods_lab_run(run, session_id)
            _mark_methods_lab_run_cancelled(run_id, session_id)
            srv._clear_methods_lab_cancel_event(run_id, session_id)
            return
        run["status"] = "running"
        run["started_at"] = srv._now_iso()
        srv._save_methods_lab_run(run, session_id)

    models = run.get("models", [])
    methods = run.get("methods", [])
    doc_ids = run.get("doc_ids", [])
    model_by_id = {str(model["id"]): model for model in models if isinstance(model, dict)}
    method_by_variant_id = {
        str(method["id"]): method for method in methods if isinstance(method, dict)
    }
    all_model_ids = [str(model.get("id", "")) for model in models if isinstance(model, dict)]
    reference_label_set = _collect_reference_label_set(
        doc_ids=[str(doc_id) for doc_id in doc_ids],
        session_id=session_id,
        reference_source=reference_source,
        fallback_reference_source=fallback_reference_source,
    )

    tasks: list[tuple[str, str, str | None]] = []
    for method in methods:
        if not isinstance(method, dict):
            continue
        method_variant_id = str(method.get("id", ""))
        method_definition = srv.get_method_definition_by_id(
            str(method.get("method_id", "")),
            method_bundle=method_bundle,  # type: ignore[arg-type]
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
                    "error_family": srv._normalize_error_family(
                        f"Unknown method variant '{method_variant_id}'"
                    ),
                    "updated_at": srv._now_iso(),
                },
            )

        definition = srv.get_method_definition_by_id(
            str(method_variant.get("method_id", "")),
            method_bundle=method_bundle,  # type: ignore[arg-type]
        )
        if definition is None:
            return (
                method_variant_id,
                doc_id,
                target_model_ids,
                {
                    "status": "failed",
                    "error": f"Unknown method: {method_variant.get('method_id')}",
                    "error_family": srv._normalize_error_family(
                        f"Unknown method: {method_variant.get('method_id')}"
                    ),
                    "updated_at": srv._now_iso(),
                },
            )

        doc = srv._load_doc(doc_id, session_id)
        if doc is None:
            return (
                method_variant_id,
                doc_id,
                target_model_ids,
                {
                    "status": "unavailable",
                    "error": "Document no longer exists",
                    "updated_at": srv._now_iso(),
                },
            )

        enriched = srv._enrich_doc(doc, session_id)
        resolved_reference_source, reference_spans = _resolve_prompt_lab_reference(
            enriched,
            reference_source,
            fallback_reference_source,
        )
        if not reference_spans and resolved_reference_source != "pre":
            return (
                method_variant_id,
                doc_id,
                target_model_ids,
                {
                    "status": "unavailable",
                    "error": "Methods Lab requires reference annotations for scoring.",
                    "updated_at": srv._now_iso(),
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
        started_at = srv._now_iso()
        runtime_state: dict[str, object] = {
            "started_at": started_at,
            "last_progress_at": started_at,
        }
        runtime_state_lock = threading.Lock()

        def _runtime_snapshot() -> dict[str, object]:
            with runtime_state_lock:
                return copy.deepcopy(runtime_state)

        def _record_runtime_progress(update: dict[str, object]) -> None:
            progress_at = srv._now_iso()
            with runtime_state_lock:
                runtime_state.update(update)
                runtime_state["last_progress_at"] = progress_at
                snapshot = copy.deepcopy(runtime_state)
            _persist_methods_lab_document_runtime(
                run_id=run_id,
                session_id=session_id,
                method_variant_id=method_variant_id,
                doc_id=doc_id,
                target_model_ids=target_model_ids,
                filename=enriched.filename,
                runtime_diagnostics=snapshot,
            )

        _persist_methods_lab_document_runtime(
            run_id=run_id,
            session_id=session_id,
            method_variant_id=method_variant_id,
            doc_id=doc_id,
            target_model_ids=target_model_ids,
            filename=enriched.filename,
            runtime_diagnostics=_runtime_snapshot(),
        )

        try:
            def _run_document():
                return srv._run_method_for_document(
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
                    label_profile="simple",
                    chunk_mode=chunk_mode,
                    chunk_size_chars=chunk_size_chars,
                    runtime_progress_callback=_record_runtime_progress,
                timeout_seconds=task_timeout_seconds,
                method_bundle=method_bundle,
            )

            if task_timeout_seconds is None:
                (
                    hypothesis_spans,
                    warnings,
                    llm_confidence,
                    _chunk_diagnostics,
                    raw_hypothesis_spans,
                    resolution_events,
                    resolution_policy_version,
                    response_debug,
                ) = _run_document()
            else:
                task_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                task_future = task_executor.submit(_run_document)
                timed_out = False
                last_progress_marker = str(_runtime_snapshot().get("last_progress_at") or "")
                last_progress_monotonic = time.monotonic()
                poll_timeout = min(0.2, max(0.01, task_timeout_seconds / 4.0))
                try:
                    while True:
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
                            ) = task_future.result(timeout=poll_timeout)
                            break
                        except concurrent.futures.TimeoutError:
                            snapshot = _runtime_snapshot()
                            current_progress_marker = str(
                                snapshot.get("last_progress_at") or ""
                            )
                            now = time.monotonic()
                            if current_progress_marker != last_progress_marker:
                                last_progress_marker = current_progress_marker
                                last_progress_monotonic = now
                                continue
                            if now - last_progress_monotonic < task_timeout_seconds:
                                continue
                            raise
                except concurrent.futures.TimeoutError as exc:
                    timed_out = True
                    task_future.cancel()
                    message = (
                        "Methods Lab task timed out after "
                        f"{task_timeout_seconds:g} seconds without progress."
                    )
                    return (
                        method_variant_id,
                        doc_id,
                        target_model_ids,
                        {
                            "status": "failed",
                            "error": message,
                            "error_family": "timeout",
                            "runtime_diagnostics": _runtime_snapshot(),
                            "updated_at": srv._now_iso(),
                            "filename": enriched.filename,
                        },
                    )
                finally:
                    task_executor.shutdown(wait=not timed_out, cancel_futures=timed_out)
            projected_reference, projected_hypothesis = srv._prepare_experiment_scoring_spans(
                reference_spans,
                hypothesis_spans,
                method_bundle=method_bundle,
                reference_label_set=reference_label_set,
            )
            projected_reference_raw, projected_hypothesis_raw = srv._prepare_experiment_scoring_spans(
                reference_spans,
                raw_hypothesis_spans,
                method_bundle=method_bundle,
                reference_label_set=reference_label_set,
            )
            scoring_match_mode = srv._normalize_metrics_mode_for_method_bundle(
                match_mode,
                method_bundle=method_bundle,
            )
            metrics = srv._serialize_metrics_payload(
                srv.compute_metrics(
                    projected_reference,
                    projected_hypothesis,
                    scoring_match_mode,
                )
            )
            raw_metrics = srv._serialize_metrics_payload(
                srv.compute_metrics(
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
                    "reference_source_used": resolved_reference_source,
                    "reference_spans": [span.model_dump() for span in reference_spans],
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
                    "resolution_summary": srv.summarize_resolution_events(resolution_events),
                    "runtime_diagnostics": _runtime_snapshot(),
                    "updated_at": srv._now_iso(),
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
                    "error_family": srv._normalize_error_family(message),
                    "runtime_diagnostics": _runtime_snapshot(),
                    "updated_at": srv._now_iso(),
                    "filename": enriched.filename,
                },
            )

    max_workers = srv._resolve_experiment_worker_count(
        int(run.get("concurrency", srv.METHODS_LAB_DEFAULT_CONCURRENCY)),
        total_tasks=len(tasks),
        max_allowed=srv._get_methods_lab_max_concurrency(),
    )
    executor = srv.ThreadPoolExecutor(max_workers=max_workers)
    cancelled = False
    try:
        future_map = {
            executor.submit(_execute_task, method_variant_id, doc_id, model_id): (
                method_variant_id,
                doc_id,
                model_id,
            )
            for method_variant_id, doc_id, model_id in tasks
        }
        while future_map:
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                for future in future_map:
                    future.cancel()
                executor.shutdown(wait=False, cancel_futures=True)
                _mark_methods_lab_run_cancelled(run_id, session_id)
                return
            done, _ = concurrent.futures.wait(
                set(future_map.keys()),
                timeout=0.2,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            if not done:
                continue
            for future in done:
                try:
                    method_variant_id, doc_id, target_model_ids, result = future.result()
                except Exception as exc:
                    message = str(exc).strip() or exc.__class__.__name__
                    method_variant_id, doc_id, model_id = future_map[future]
                    target_model_ids = [model_id] if model_id else all_model_ids
                    result = {
                        "status": "failed",
                        "error": message,
                        "error_family": srv._normalize_error_family(message),
                        "updated_at": srv._now_iso(),
                    }
                future_map.pop(future, None)
                with srv._methods_lab_lock:
                    latest = srv._load_methods_lab_run(run_id, session_id)
                    if latest is None:
                        continue
                    cells = latest.get("cells", {})
                    if not isinstance(cells, dict):
                        continue
                    method_variant = method_by_variant_id.get(method_variant_id)
                    for target_model_id in target_model_ids:
                        cell_id = f"{target_model_id}__{method_variant_id}"
                        if isinstance(cells.get(cell_id), dict):
                            cells[cell_id]["documents"][doc_id] = copy.deepcopy(result)
                    if isinstance(method_variant, dict):
                        _sync_methods_lab_result_to_workspace(
                            run_id=run_id,
                            session_id=session_id,
                            doc_id=doc_id,
                            method_id=str(method_variant.get("method_id", "")),
                            method_verify_override=method_variant.get("method_verify_override"),
                            target_model_ids=target_model_ids,
                            model_by_id=model_by_id,
                            method_bundle=method_bundle,
                            result=result,
                        )
                    srv._save_methods_lab_run(latest, session_id)
    except Exception as exc:
        with srv._methods_lab_lock:
            latest = srv._load_methods_lab_run(run_id, session_id)
            if latest is not None:
                latest["status"] = "failed"
                latest["finished_at"] = srv._now_iso()
                errors = latest.get("errors", [])
                if not isinstance(errors, list):
                    errors = []
                errors.append(f"Methods Lab run failed: {exc}")
                latest["errors"] = errors
                srv._save_methods_lab_run(latest, session_id)
        return
    finally:
        if not cancelled:
            executor.shutdown(wait=True)
        srv._clear_methods_lab_cancel_event(run_id, session_id)

    with srv._methods_lab_lock:
        latest = srv._load_methods_lab_run(run_id, session_id)
        if latest is None:
            return
        summary = build_methods_lab_run_summary(latest)
        latest["status"] = (
            "completed_with_errors"
            if int(summary.get("failed_tasks", 0)) > 0
            else "completed"
        )
        latest["finished_at"] = srv._now_iso()
        srv._save_methods_lab_run(latest, session_id)


def create_methods_lab_run(
    body: Any,
    *,
    session_id: str = "default",
    run_async: bool = False,
    method_bundle: str | None = None,
) -> dict:
    srv = _server()
    prepared = prepare_methods_lab_body(body, session_id=session_id)
    runtime = resolve_methods_lab_runtime(prepared.runtime)
    selected_method_bundle = (
        srv._normalize_method_bundle(method_bundle)
        if method_bundle is not None
        else str(runtime["method_bundle"])
    )
    runtime["method_bundle"] = selected_method_bundle
    validate_methods_lab_request(prepared, session_id, method_bundle=selected_method_bundle)
    api_key = str(runtime["api_key"])
    api_base = str(runtime["api_base"])
    requires_llm = any(
        bool(
            (
                srv.get_method_definition_by_id(
                    str(method.method_id),
                    method_bundle=selected_method_bundle,  # type: ignore[arg-type]
                )
                or {}
            ).get("uses_llm")
        )
        for method in prepared.methods
    )

    if requires_llm:
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail=(
                    "API key required for Methods Lab runs. Set LITELLM_API_KEY "
                    "or provide runtime.api_key in request."
                ),
            )
        for model in prepared.models:
            srv._validate_gateway_model_access(model=model.model, api_base=api_base, api_key=api_key)

    with srv._methods_lab_lock:
        run = initialize_methods_lab_run(prepared, session_id, runtime)
        ids = srv._load_methods_lab_index(session_id)
        ids.append(str(run["id"]))
        srv._save_methods_lab_run(run, session_id)
        srv._save_methods_lab_index(session_id)
        srv._register_methods_lab_cancel_event(str(run["id"]), session_id)

    if run_async:
        worker = threading.Thread(
            target=run_methods_lab_job,
            args=(str(run["id"]), session_id, runtime, selected_method_bundle),
            daemon=True,
        )
        worker.start()
        return build_methods_lab_run_detail(run)

    run_methods_lab_job(str(run["id"]), session_id, runtime, selected_method_bundle)
    latest = srv._load_methods_lab_run(str(run["id"]), session_id)
    if latest is None:
        raise RuntimeError("Methods Lab run disappeared before completion.")
    return build_methods_lab_run_detail(latest)
