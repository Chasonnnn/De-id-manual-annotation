from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import yaml
from fastapi import HTTPException

from experiment_service import create_methods_lab_run, create_prompt_lab_run
from server import (
    MODEL_PRESETS,
    MethodsLabMethodInput,
    MethodsLabRunCreateBody,
    MethodsLabRuntimeInput,
    PromptLabModelInput,
    PromptLabPromptInput,
    PromptLabRunCreateBody,
    PromptLabRuntimeInput,
    _enrich_doc,
    _load_doc,
    _load_methods_lab_run,
    _load_session_index,
    _normalize_error_family,
    list_agent_methods,
)


def _parse_labeled_value(raw: str, *, label_name: str, value_name: str) -> tuple[str, str]:
    label, separator, value = str(raw).partition("=")
    label = label.strip()
    value = value.strip()
    if separator != "=" or not label or not value:
        raise ValueError(
            f"Expected {label_name}={value_name} format, received: {raw!r}"
        )
    return label, value


def _build_prompt_body_from_args(args: argparse.Namespace) -> PromptLabRunCreateBody:
    prompts: list[PromptLabPromptInput] = []
    for value in args.prompt:
        label, prompt_text = _parse_labeled_value(value, label_name="label", value_name="text")
        prompts.append(
            PromptLabPromptInput(
                id=f"prompt_{len(prompts) + 1}",
                label=label,
                system_prompt=prompt_text,
                variant_type="prompt",
            )
        )
    for value in args.prompt_file:
        label, prompt_file = _parse_labeled_value(value, label_name="label", value_name="path")
        prompts.append(
            PromptLabPromptInput(
                id=f"prompt_{len(prompts) + 1}",
                label=label,
                prompt_file=prompt_file,
                variant_type="prompt",
            )
        )
    for value in args.preset:
        label, method_id = _parse_labeled_value(
            value, label_name="label", value_name="preset_method_id"
        )
        prompts.append(
            PromptLabPromptInput(
                id=f"prompt_{len(prompts) + 1}",
                label=label,
                variant_type="preset",
                preset_method_id=method_id,
            )
        )
    if not prompts:
        raise ValueError("At least one prompt variant is required.")

    models = _build_models_from_args(args)
    runtime = PromptLabRuntimeInput(
        api_key=args.api_key,
        api_base=args.api_base,
        temperature=args.temperature,
        match_mode=args.match_mode,
        reference_source=args.reference_source,
        fallback_reference_source=args.fallback_reference_source,
        label_profile=args.label_profile,
        label_projection=args.label_projection,
        chunk_mode=args.chunk_mode,
        chunk_size_chars=args.chunk_size_chars,
    )
    return PromptLabRunCreateBody(
        name=args.name,
        doc_ids=list(dict.fromkeys(args.doc_id)),
        folder_ids=list(dict.fromkeys(args.folder_id)),
        prompts=prompts,
        models=models,
        runtime=runtime,
        concurrency=args.concurrency,
    )


def _build_methods_body_from_args(args: argparse.Namespace) -> MethodsLabRunCreateBody:
    methods: list[MethodsLabMethodInput] = []
    for value in args.method:
        label, method_id = _parse_labeled_value(value, label_name="label", value_name="method_id")
        methods.append(
            MethodsLabMethodInput(
                id=f"method_{len(methods) + 1}",
                label=label,
                method_id=method_id,
            )
        )
    if not methods:
        raise ValueError("At least one method variant is required.")

    models = _build_models_from_args(args)
    runtime = MethodsLabRuntimeInput(
        api_key=args.api_key,
        api_base=args.api_base,
        temperature=args.temperature,
        match_mode=args.match_mode,
        reference_source=args.reference_source,
        fallback_reference_source=args.fallback_reference_source,
        label_profile=args.label_profile,
        label_projection=args.label_projection,
        chunk_mode=args.chunk_mode,
        chunk_size_chars=args.chunk_size_chars,
        task_timeout_seconds=args.task_timeout_seconds,
    )
    return MethodsLabRunCreateBody(
        name=args.name,
        doc_ids=list(dict.fromkeys(args.doc_id)),
        folder_ids=list(dict.fromkeys(args.folder_id)),
        methods=methods,
        models=models,
        runtime=runtime,
        concurrency=args.concurrency,
    )


def _build_models_from_args(args: argparse.Namespace) -> list[PromptLabModelInput]:
    models: list[PromptLabModelInput] = []
    for value in args.model:
        label, model_id = _parse_labeled_value(value, label_name="label", value_name="model")
        models.append(
            PromptLabModelInput(
                id=f"model_{len(models) + 1}",
                label=label,
                model=model_id,
                reasoning_effort=args.reasoning_effort,
                anthropic_thinking=bool(args.anthropic_thinking),
                anthropic_thinking_budget_tokens=args.anthropic_thinking_budget_tokens,
            )
        )
    if not models:
        raise ValueError("At least one model variant is required.")
    return models


def _load_manifest(path_value: str) -> tuple[str, str, Any, Path, str]:
    manifest_path = Path(path_value).resolve()
    if not manifest_path.exists():
        raise ValueError(f"Manifest not found: {manifest_path}")
    payload = manifest_path.read_text()
    if manifest_path.suffix.lower() == ".json":
        data = json.loads(payload)
    else:
        data = yaml.safe_load(payload)
    if not isinstance(data, dict):
        raise ValueError("Manifest must decode to an object.")

    kind = str(data.get("kind") or "").strip()
    session_id = str(data.get("session") or "default").strip() or "default"
    method_bundle = str(data.get("method_bundle") or "audited").strip().lower() or "audited"
    if method_bundle not in {"legacy", "audited", "test", "v2+post-process"}:
        raise ValueError(
            "method_bundle must be one of 'legacy', 'audited', 'test', or 'v2+post-process'."
        )
    body_payload = {key: value for key, value in data.items() if key not in {"kind", "session"}}
    body_payload.pop("method_bundle", None)
    raw_doc_ids = body_payload.get("doc_ids")
    if isinstance(raw_doc_ids, list):
        body_payload["doc_ids"] = [str(value) for value in raw_doc_ids]
    raw_folder_ids = body_payload.get("folder_ids")
    if isinstance(raw_folder_ids, list):
        body_payload["folder_ids"] = [str(value) for value in raw_folder_ids]
    if kind == "prompt_lab":
        body = PromptLabRunCreateBody.model_validate(body_payload)
    elif kind == "methods_lab":
        body = MethodsLabRunCreateBody.model_validate(body_payload)
    else:
        raise ValueError("Manifest kind must be either 'prompt_lab' or 'methods_lab'.")
    return kind, session_id, body, manifest_path.parent, method_bundle


def _write_output_json(path_value: str | None, payload: dict[str, Any]) -> None:
    if not path_value:
        return
    Path(path_value).write_text(json.dumps(payload, indent=2))


def _matrix_rows(kind: str, detail: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in detail.get("matrix", {}).get("cells", []):
        if not isinstance(cell, dict):
            continue
        row = {
            "id": cell.get("id"),
            "status": cell.get("status"),
            "model_id": cell.get("model_id"),
            "model_label": cell.get("model_label"),
            "total_docs": cell.get("total_docs"),
            "completed_docs": cell.get("completed_docs"),
            "failed_docs": cell.get("failed_docs"),
            "pending_docs": cell.get("pending_docs"),
            "precision": cell.get("micro", {}).get("precision"),
            "recall": cell.get("micro", {}).get("recall"),
            "f1": cell.get("micro", {}).get("f1"),
            "raw_precision": cell.get("raw_micro", {}).get("precision"),
            "raw_recall": cell.get("raw_micro", {}).get("recall"),
            "raw_f1": cell.get("raw_micro", {}).get("f1"),
            "overlap_f1": cell.get("co_primary_metrics", {})
            .get("overlap", {})
            .get("micro", {})
            .get("f1"),
            "raw_overlap_f1": cell.get("raw_co_primary_metrics", {})
            .get("overlap", {})
            .get("micro", {})
            .get("f1"),
            "overlap_gap_f1": cell.get("overlap_gap_f1"),
            "raw_overlap_gap_f1": cell.get("raw_overlap_gap_f1"),
            "boundary_fix_count": cell.get("resolution_summary", {}).get("boundary_fix_count"),
            "augmentation_count": cell.get("resolution_summary", {}).get("augmentation_count"),
            "mean_confidence": cell.get("mean_confidence"),
        }
        if kind == "prompt_lab":
            row["prompt_id"] = cell.get("prompt_id")
            row["prompt_label"] = cell.get("prompt_label")
        else:
            row["method_id"] = cell.get("method_id")
            row["method_label"] = cell.get("method_label")
        rows.append(row)
    return rows


def _write_output_csv(path_value: str | None, *, kind: str, detail: dict[str, Any]) -> None:
    if not path_value:
        return
    rows = _matrix_rows(kind, detail)
    if kind == "prompt_lab":
        fieldnames = [
            "id",
            "status",
            "model_id",
            "model_label",
            "prompt_id",
            "prompt_label",
            "total_docs",
            "completed_docs",
            "failed_docs",
            "pending_docs",
            "precision",
            "recall",
            "f1",
            "raw_precision",
            "raw_recall",
            "raw_f1",
            "overlap_f1",
            "raw_overlap_f1",
            "overlap_gap_f1",
            "raw_overlap_gap_f1",
            "boundary_fix_count",
            "augmentation_count",
            "mean_confidence",
        ]
    else:
        fieldnames = [
            "id",
            "status",
            "model_id",
            "model_label",
            "method_id",
            "method_label",
            "total_docs",
            "completed_docs",
            "failed_docs",
            "pending_docs",
            "precision",
            "recall",
            "f1",
            "raw_precision",
            "raw_recall",
            "raw_f1",
            "overlap_f1",
            "raw_overlap_f1",
            "overlap_gap_f1",
            "raw_overlap_gap_f1",
            "boundary_fix_count",
            "augmentation_count",
            "mean_confidence",
        ]
    with Path(path_value).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _prf_from_counts(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _parse_iso_datetime(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None


def _runtime_seconds(document: dict[str, Any]) -> float | None:
    runtime = document.get("runtime_diagnostics")
    started_at = _parse_iso_datetime(runtime.get("started_at")) if isinstance(runtime, dict) else None
    updated_at = _parse_iso_datetime(document.get("updated_at"))
    if started_at is None or updated_at is None:
        return None
    duration = (updated_at - started_at).total_seconds()
    if duration < 0:
        return None
    return duration


def _build_methods_cell_stats(cell: dict[str, Any]) -> dict[str, Any]:
    documents_raw = cell.get("documents", {})
    documents = documents_raw if isinstance(documents_raw, dict) else {}
    tp = 0
    fp = 0
    fn = 0
    completed_count = 0
    failed_count = 0
    cancelled_count = 0
    pending_count = 0
    timeout_count = 0
    empty_content_truncation_count = 0
    runtimes: list[float] = []

    for document in documents.values():
        if not isinstance(document, dict):
            continue
        status = str(document.get("status", "pending"))
        if status == "completed":
            completed_count += 1
            metrics = document.get("metrics")
            micro = metrics.get("micro") if isinstance(metrics, dict) else {}
            if not isinstance(micro, dict):
                micro = {}
            tp += int(micro.get("tp", 0))
            fp += int(micro.get("fp", 0))
            fn += int(micro.get("fn", 0))
            runtime_seconds = _runtime_seconds(document)
            if runtime_seconds is not None:
                runtimes.append(runtime_seconds)
            continue
        if status == "failed":
            failed_count += 1
            family = document.get("error_family")
            if not isinstance(family, str) or not family:
                family = _normalize_error_family(document.get("error"))
            if family == "timeout":
                timeout_count += 1
            if family == "empty_output_finish_reason_length":
                empty_content_truncation_count += 1
            continue
        if status == "cancelled":
            cancelled_count += 1
            continue
        pending_count += 1

    return {
        "micro": _prf_from_counts(tp, fp, fn),
        "completed_count": completed_count,
        "failed_count": failed_count,
        "cancelled_count": cancelled_count,
        "pending_count": pending_count,
        "timeout_count": timeout_count,
        "empty_content_truncation_count": empty_content_truncation_count,
        "mean_runtime_seconds": statistics.mean(runtimes) if runtimes else None,
        "median_runtime_seconds": statistics.median(runtimes) if runtimes else None,
        "documents": documents,
    }


def _build_methods_bundle_stats(run: dict[str, Any]) -> dict[str, Any]:
    cells_raw = run.get("cells", {})
    cells = cells_raw if isinstance(cells_raw, dict) else {}
    total_tp = 0
    total_fp = 0
    total_fn = 0
    completed_count = 0
    failed_count = 0
    cancelled_count = 0
    pending_count = 0
    timeout_count = 0
    empty_content_truncation_count = 0
    runtimes: list[float] = []

    for cell in cells.values():
        if not isinstance(cell, dict):
            continue
        stats = _build_methods_cell_stats(cell)
        micro = stats["micro"]
        total_tp += int(micro.get("tp", 0))
        total_fp += int(micro.get("fp", 0))
        total_fn += int(micro.get("fn", 0))
        completed_count += int(stats["completed_count"])
        failed_count += int(stats["failed_count"])
        cancelled_count += int(stats["cancelled_count"])
        pending_count += int(stats["pending_count"])
        timeout_count += int(stats["timeout_count"])
        empty_content_truncation_count += int(stats["empty_content_truncation_count"])
        cell_runtimes = []
        for document in stats["documents"].values():
            if not isinstance(document, dict):
                continue
            runtime_seconds = _runtime_seconds(document)
            if runtime_seconds is not None:
                cell_runtimes.append(runtime_seconds)
        runtimes.extend(cell_runtimes)

    return {
        "micro": _prf_from_counts(total_tp, total_fp, total_fn),
        "completed_count": completed_count,
        "failed_count": failed_count,
        "cancelled_count": cancelled_count,
        "pending_count": pending_count,
        "timeout_count": timeout_count,
        "empty_content_truncation_count": empty_content_truncation_count,
        "mean_runtime_seconds": statistics.mean(runtimes) if runtimes else None,
        "median_runtime_seconds": statistics.median(runtimes) if runtimes else None,
    }


def _document_status_snapshot(document: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(document, dict):
        return {"status": "missing", "error_family": None, "runtime_seconds": None}
    return {
        "status": str(document.get("status", "pending")),
        "error_family": (
            document.get("error_family")
            if isinstance(document.get("error_family"), str)
            else _normalize_error_family(document.get("error"))
        ),
        "runtime_seconds": _runtime_seconds(document),
    }


def _build_method_bundle_ab_summary(
    *,
    legacy_run: dict[str, Any],
    audited_run: dict[str, Any],
) -> dict[str, Any]:
    legacy_bundle = _build_methods_bundle_stats(legacy_run)
    audited_bundle = _build_methods_bundle_stats(audited_run)

    legacy_cells_raw = legacy_run.get("cells", {})
    audited_cells_raw = audited_run.get("cells", {})
    legacy_cells = legacy_cells_raw if isinstance(legacy_cells_raw, dict) else {}
    audited_cells = audited_cells_raw if isinstance(audited_cells_raw, dict) else {}
    legacy_models = {
        str(item.get("id", "")): item
        for item in legacy_run.get("models", [])
        if isinstance(item, dict)
    }
    audited_models = {
        str(item.get("id", "")): item
        for item in audited_run.get("models", [])
        if isinstance(item, dict)
    }
    legacy_methods = {
        str(item.get("id", "")): item
        for item in legacy_run.get("methods", [])
        if isinstance(item, dict)
    }
    audited_methods = {
        str(item.get("id", "")): item
        for item in audited_run.get("methods", [])
        if isinstance(item, dict)
    }

    pairs: list[dict[str, Any]] = []
    for cell_id in sorted(set(legacy_cells.keys()) | set(audited_cells.keys())):
        legacy_cell = legacy_cells.get(cell_id)
        audited_cell = audited_cells.get(cell_id)
        if not isinstance(legacy_cell, dict) and not isinstance(audited_cell, dict):
            continue
        reference_cell = audited_cell if isinstance(audited_cell, dict) else legacy_cell
        assert isinstance(reference_cell, dict)
        model_variant_id = str(reference_cell.get("model_id", ""))
        method_variant_id = str(reference_cell.get("method_id", ""))
        model = audited_models.get(model_variant_id) or legacy_models.get(model_variant_id) or {}
        method = audited_methods.get(method_variant_id) or legacy_methods.get(method_variant_id) or {}
        legacy_stats = _build_methods_cell_stats(legacy_cell) if isinstance(legacy_cell, dict) else {
            "micro": _prf_from_counts(0, 0, 0),
            "completed_count": 0,
            "failed_count": 0,
            "cancelled_count": 0,
            "pending_count": 0,
            "timeout_count": 0,
            "empty_content_truncation_count": 0,
            "mean_runtime_seconds": None,
            "median_runtime_seconds": None,
            "documents": {},
        }
        audited_stats = _build_methods_cell_stats(audited_cell) if isinstance(audited_cell, dict) else {
            "micro": _prf_from_counts(0, 0, 0),
            "completed_count": 0,
            "failed_count": 0,
            "cancelled_count": 0,
            "pending_count": 0,
            "timeout_count": 0,
            "empty_content_truncation_count": 0,
            "mean_runtime_seconds": None,
            "median_runtime_seconds": None,
            "documents": {},
        }
        documents: dict[str, Any] = {}
        legacy_docs = legacy_stats["documents"]
        audited_docs = audited_stats["documents"]
        for doc_id in sorted(set(legacy_docs.keys()) | set(audited_docs.keys())):
            documents[doc_id] = {
                "legacy": _document_status_snapshot(legacy_docs.get(doc_id)),
                "audited": _document_status_snapshot(audited_docs.get(doc_id)),
            }
        legacy_micro = legacy_stats["micro"]
        audited_micro = audited_stats["micro"]
        pairs.append(
            {
                "cell_id": cell_id,
                "model_variant_id": model_variant_id,
                "model": str(model.get("model", model.get("label", model_variant_id))),
                "model_label": str(model.get("label", model_variant_id)),
                "method_variant_id": method_variant_id,
                "method_id": str(method.get("method_id", method_variant_id)),
                "method_label": str(method.get("label", method_variant_id)),
                "legacy": {
                    key: value
                    for key, value in legacy_stats.items()
                    if key != "documents"
                },
                "audited": {
                    key: value
                    for key, value in audited_stats.items()
                    if key != "documents"
                },
                "delta_vs_legacy": {
                    "micro_precision": float(audited_micro.get("precision", 0.0))
                    - float(legacy_micro.get("precision", 0.0)),
                    "micro_recall": float(audited_micro.get("recall", 0.0))
                    - float(legacy_micro.get("recall", 0.0)),
                    "micro_f1": float(audited_micro.get("f1", 0.0))
                    - float(legacy_micro.get("f1", 0.0)),
                    "failed_count": int(audited_stats["failed_count"])
                    - int(legacy_stats["failed_count"]),
                    "timeout_count": int(audited_stats["timeout_count"])
                    - int(legacy_stats["timeout_count"]),
                    "empty_content_truncation_count": int(
                        audited_stats["empty_content_truncation_count"]
                    )
                    - int(legacy_stats["empty_content_truncation_count"]),
                },
                "documents": documents,
            }
        )

    return {
        "legacy_run_id": legacy_run.get("id"),
        "audited_run_id": audited_run.get("id"),
        "bundles": {
            "legacy": legacy_bundle,
            "audited": audited_bundle,
        },
        "pairs": pairs,
    }


def _print_run_summary(kind: str, detail: dict[str, Any]) -> None:
    print(f"run_id\tstatus\tname")
    print(f"{detail['id']}\t{detail['status']}\t{detail['name']}")
    print()
    if kind == "prompt_lab":
        print("cell_id\tmodel\tprompt\tstatus\tcompleted_docs\tfailed_docs\tf1\tmean_confidence")
        for row in _matrix_rows(kind, detail):
            print(
                "\t".join(
                    [
                        str(row["id"]),
                        str(row["model_label"]),
                        str(row["prompt_label"]),
                        str(row["status"]),
                        str(row["completed_docs"]),
                        str(row["failed_docs"]),
                        str(row["f1"]),
                        str(row["mean_confidence"]),
                    ]
                )
            )
        return

    print("cell_id\tmodel\tmethod\tstatus\tcompleted_docs\tfailed_docs\tf1\tmean_confidence")
    for row in _matrix_rows(kind, detail):
        print(
            "\t".join(
                [
                    str(row["id"]),
                    str(row["model_label"]),
                    str(row["method_label"]),
                    str(row["status"]),
                    str(row["completed_docs"]),
                    str(row["failed_docs"]),
                    str(row["f1"]),
                    str(row["mean_confidence"]),
                ]
            )
        )


def _handle_run(kind: str, detail: dict[str, Any], args: argparse.Namespace) -> int:
    _print_run_summary(kind, detail)
    _write_output_json(args.output_json, detail)
    _write_output_csv(args.output_csv, kind=kind, detail=detail)
    return 0


def _comparison_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pair in summary.get("pairs", []):
        if not isinstance(pair, dict):
            continue
        legacy = pair.get("legacy", {})
        audited = pair.get("audited", {})
        delta = pair.get("delta_vs_legacy", {})
        legacy_micro = legacy.get("micro", {}) if isinstance(legacy, dict) else {}
        audited_micro = audited.get("micro", {}) if isinstance(audited, dict) else {}
        rows.append(
            {
                "model": pair.get("model"),
                "method_id": pair.get("method_id"),
                "method_label": pair.get("method_label"),
                "legacy_f1": legacy_micro.get("f1"),
                "audited_f1": audited_micro.get("f1"),
                "delta_f1": delta.get("micro_f1"),
                "legacy_failed_count": legacy.get("failed_count"),
                "audited_failed_count": audited.get("failed_count"),
                "legacy_timeout_count": legacy.get("timeout_count"),
                "audited_timeout_count": audited.get("timeout_count"),
                "legacy_empty_content_truncation_count": legacy.get(
                    "empty_content_truncation_count"
                ),
                "audited_empty_content_truncation_count": audited.get(
                    "empty_content_truncation_count"
                ),
            }
        )
    return rows


def _write_method_bundle_ab_csv(path_value: str | None, summary: dict[str, Any]) -> None:
    if not path_value:
        return
    rows = _comparison_rows(summary)
    fieldnames = [
        "model",
        "method_id",
        "method_label",
        "legacy_f1",
        "audited_f1",
        "delta_f1",
        "legacy_failed_count",
        "audited_failed_count",
        "legacy_timeout_count",
        "audited_timeout_count",
        "legacy_empty_content_truncation_count",
        "audited_empty_content_truncation_count",
    ]
    with Path(path_value).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_method_bundle_ab_markdown(path_value: str | None, summary: dict[str, Any]) -> None:
    if not path_value:
        return
    rows = _comparison_rows(summary)
    lines = [
        "| Model | Method | Legacy F1 | Audited F1 | Delta F1 | Legacy Failures | Audited Failures |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        legacy_f1 = float(row["legacy_f1"] or 0.0)
        audited_f1 = float(row["audited_f1"] or 0.0)
        delta_f1 = float(row["delta_f1"] or 0.0)
        lines.append(
            "| "
            f"{row['model']} | {row['method_label']} | "
            f"{legacy_f1:.3f} | {audited_f1:.3f} | {delta_f1:.3f} | "
            f"{row['legacy_failed_count']} | {row['audited_failed_count']} |"
        )
    Path(path_value).write_text("\n".join(lines) + "\n")


def _default_artifact_prefix(name: str | None) -> str:
    raw = (name or "methods-bundle-ab").strip().lower()
    slug = "".join(ch if ch.isalnum() else "-" for ch in raw)
    return "-".join(part for part in slug.split("-") if part) or "methods-bundle-ab"


def _print_method_bundle_ab_summary(summary: dict[str, Any]) -> None:
    print("bundle\tcompleted\tfailed\ttimeouts\ttruncations\tf1")
    for bundle_name in ("legacy", "audited"):
        bundle = summary["bundles"][bundle_name]
        micro = bundle.get("micro", {})
        print(
            "\t".join(
                [
                    bundle_name,
                    str(bundle.get("completed_count")),
                    str(bundle.get("failed_count")),
                    str(bundle.get("timeout_count")),
                    str(bundle.get("empty_content_truncation_count")),
                    str(micro.get("f1")),
                ]
            )
        )
    print()
    print("model\tmethod\tlegacy_f1\taudited_f1\tdelta_f1\tlegacy_failed\taudited_failed")
    for row in _comparison_rows(summary):
        print(
            "\t".join(
                [
                    str(row["model"]),
                    str(row["method_label"]),
                    str(row["legacy_f1"]),
                    str(row["audited_f1"]),
                    str(row["delta_f1"]),
                    str(row["legacy_failed_count"]),
                    str(row["audited_failed_count"]),
                ]
            )
        )


def run_methods_bundle_benchmark(args: argparse.Namespace) -> int:
    body = _build_methods_body_from_args(args)
    legacy_detail = create_methods_lab_run(
        body,
        session_id=args.session,
        run_async=False,
        method_bundle="legacy",
    )
    audited_detail = create_methods_lab_run(
        body,
        session_id=args.session,
        run_async=False,
        method_bundle="audited",
    )
    legacy_run = _load_methods_lab_run(str(legacy_detail["id"]), args.session)
    audited_run = _load_methods_lab_run(str(audited_detail["id"]), args.session)
    if legacy_run is None or audited_run is None:
        raise RuntimeError("Methods bundle benchmark runs disappeared before comparison.")
    summary = _build_method_bundle_ab_summary(
        legacy_run=legacy_run,
        audited_run=audited_run,
    )
    _print_method_bundle_ab_summary(summary)
    _write_output_json(args.output_json, summary)
    _write_method_bundle_ab_csv(args.output_csv, summary)
    _write_method_bundle_ab_markdown(args.output_md, summary)
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        prefix = _default_artifact_prefix(args.artifact_prefix or body.name)
        (output_dir / f"{prefix}_legacy.json").write_text(json.dumps(legacy_run, indent=2))
        (output_dir / f"{prefix}_audited.json").write_text(json.dumps(audited_run, indent=2))
        (output_dir / f"{prefix}_comparison.json").write_text(json.dumps(summary, indent=2))
        _write_method_bundle_ab_csv(str(output_dir / f"{prefix}_comparison.csv"), summary)
        _write_method_bundle_ab_markdown(str(output_dir / f"{prefix}_comparison.md"), summary)
    return 0


def run_manifest(args: argparse.Namespace) -> int:
    kind, session_id, body, context_dir, method_bundle = _load_manifest(args.manifest)
    if kind == "prompt_lab":
        detail = create_prompt_lab_run(
            body,
            session_id=session_id,
            context_dir=context_dir,
            run_async=False,
        )
    else:
        detail = create_methods_lab_run(
            body,
            session_id=session_id,
            run_async=False,
            method_bundle=method_bundle,
        )
    return _handle_run(kind, detail, args)


def run_prompt(args: argparse.Namespace) -> int:
    body = _build_prompt_body_from_args(args)
    detail = create_prompt_lab_run(
        body,
        session_id=args.session,
        context_dir=Path.cwd(),
        run_async=False,
    )
    return _handle_run("prompt_lab", detail, args)


def run_methods(args: argparse.Namespace) -> int:
    body = _build_methods_body_from_args(args)
    detail = create_methods_lab_run(
        body,
        session_id=args.session,
        run_async=False,
        method_bundle=args.method_bundle,
    )
    return _handle_run("methods_lab", detail, args)


def list_docs(args: argparse.Namespace) -> int:
    documents: list[dict[str, Any]] = []
    for doc_id in _load_session_index(args.session):
        doc = _load_doc(doc_id, args.session)
        if doc is None:
            continue
        enriched = _enrich_doc(doc, args.session)
        documents.append(
            {
                "id": enriched.id,
                "filename": enriched.filename,
                "manual_annotation_count": len(enriched.manual_annotations),
                "pre_annotation_count": len(enriched.pre_annotations),
                "status": enriched.status,
            }
        )
    print(json.dumps({"session": args.session, "documents": documents}, indent=2))
    return 0


def list_models(_args: argparse.Namespace) -> int:
    print(json.dumps({"models": MODEL_PRESETS}, indent=2))
    return 0


def list_methods(_args: argparse.Namespace) -> int:
    print(json.dumps({"methods": list_agent_methods()}, indent=2))
    return 0


def _add_output_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output-json")
    parser.add_argument("--output-csv")


def _add_runtime_args(parser: argparse.ArgumentParser, *, prompt_mode: bool) -> None:
    parser.add_argument("--session", default="default")
    parser.add_argument("--name")
    parser.add_argument("--doc-id", action="append", default=[])
    parser.add_argument("--folder-id", action="append", default=[])
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--api-key")
    parser.add_argument("--api-base")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--match-mode",
        choices=["exact", "boundary", "overlap"],
        default="exact",
    )
    parser.add_argument(
        "--reference-source",
        choices=["manual", "pre"],
        default="manual",
    )
    parser.add_argument(
        "--fallback-reference-source",
        choices=["manual", "pre"],
        default="pre",
    )
    parser.add_argument(
        "--label-profile",
        choices=["simple", "advanced"],
        default="simple",
    )
    parser.add_argument(
        "--label-projection",
        choices=["native", "coarse_simple"],
        default="native",
    )
    parser.add_argument(
        "--chunk-mode",
        choices=["auto", "off", "force"],
        default="auto",
    )
    parser.add_argument("--chunk-size-chars", type=int, default=10_000)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument(
        "--reasoning-effort",
        choices=["none", "low", "medium", "high", "xhigh"],
        default="none",
    )
    parser.add_argument("--anthropic-thinking", action="store_true")
    parser.add_argument("--anthropic-thinking-budget-tokens", type=int)
    if prompt_mode:
        parser.add_argument("--prompt", action="append", default=[])
        parser.add_argument("--prompt-file", action="append", default=[])
        parser.add_argument("--preset", action="append", default=[])
    else:
        parser.add_argument("--method", action="append", default=[])
        parser.add_argument(
            "--method-bundle",
            choices=["legacy", "audited", "test", "v2+post-process"],
            default="audited",
        )
        parser.add_argument("--task-timeout-seconds", type=float)
    _add_output_args(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Prompt Lab and Methods Lab experiments from the terminal.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run a YAML or JSON experiment manifest.")
    run_parser.add_argument("manifest")
    _add_output_args(run_parser)
    run_parser.set_defaults(func=run_manifest)

    prompt_parser = subparsers.add_parser(
        "prompt", help="Run a prompt x model experiment with convenience flags."
    )
    _add_runtime_args(prompt_parser, prompt_mode=True)
    prompt_parser.set_defaults(func=run_prompt)

    methods_parser = subparsers.add_parser(
        "methods", help="Run a method x model experiment with convenience flags."
    )
    _add_runtime_args(methods_parser, prompt_mode=False)
    methods_parser.set_defaults(func=run_methods)

    benchmark_parser = subparsers.add_parser(
        "benchmark-method-bundles",
        help="Run the same Methods Lab request against legacy and audited method bundles.",
    )
    _add_runtime_args(benchmark_parser, prompt_mode=False)
    benchmark_parser.add_argument("--output-md")
    benchmark_parser.add_argument("--output-dir")
    benchmark_parser.add_argument("--artifact-prefix")
    benchmark_parser.set_defaults(func=run_methods_bundle_benchmark)

    list_docs_parser = subparsers.add_parser("list-docs", help="List documents in a session.")
    list_docs_parser.add_argument("--session", default="default")
    list_docs_parser.set_defaults(func=list_docs)

    list_models_parser = subparsers.add_parser("list-models", help="List model presets.")
    list_models_parser.set_defaults(func=list_models)

    list_methods_parser = subparsers.add_parser("list-methods", help="List method presets.")
    list_methods_parser.set_defaults(func=list_methods)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, str) else json.dumps(exc.detail)
        print(detail, file=sys.stderr)
        return 1
    except Exception as exc:
        print(str(exc) or exc.__class__.__name__, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
