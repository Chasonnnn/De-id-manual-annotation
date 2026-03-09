from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from fastapi import HTTPException

from experiment_service import create_methods_lab_run, create_prompt_lab_run
from server import (
    MethodsLabRunCreateBody,
    PromptLabRunCreateBody,
    _enrich_doc,
    _load_doc,
    _load_session_index,
)

BASELINE_RAW_PROMPT = (
    "You are a PII annotation assistant. Identify all explicit personally identifiable "
    "information (PII) spans in the transcript."
)
DEFAULT_EXPORT_SLUG = "ground-truth-sweep"
DEFAULT_PROMPT_RUNTIME = {
    "temperature": 0.0,
    "match_mode": "exact",
    "reference_source": "manual",
    "fallback_reference_source": "pre",
    "label_profile": "simple",
    "label_projection": "native",
    "chunk_mode": "auto",
    "chunk_size_chars": 10_000,
}
DEFAULT_METHODS_RUNTIME = {
    "temperature": 0.0,
    "match_mode": "exact",
    "label_profile": "simple",
    "label_projection": "native",
    "chunk_mode": "auto",
    "chunk_size_chars": 10_000,
}
DEFAULT_CONCURRENCY = 2
ERROR_FAMILY_COLUMNS = [
    "empty_output_finish_reason_length",
    "connection_error",
    "config_error",
    "timeout",
    "unknown_error",
]


@dataclass(frozen=True)
class ModelVariant:
    id: str
    label: str
    model: str
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh"] = "none"
    anthropic_thinking: bool = False
    anthropic_thinking_budget_tokens: int | None = None

    @property
    def provider(self) -> str:
        if self.model.startswith("anthropic."):
            return "anthropic"
        if self.model.startswith("google."):
            return "gemini"
        return "openai"

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "anthropic_thinking": self.anthropic_thinking,
            "anthropic_thinking_budget_tokens": self.anthropic_thinking_budget_tokens,
        }


@dataclass(frozen=True)
class PromptVariant:
    id: str
    label: str
    variant_type: Literal["prompt", "preset"]
    system_prompt: str | None = None
    preset_method_id: str | None = None
    method_verify_override: bool | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "variant_type": self.variant_type,
            "system_prompt": self.system_prompt if self.variant_type == "prompt" else None,
            "preset_method_id": self.preset_method_id if self.variant_type == "preset" else None,
            "method_verify_override": (
                self.method_verify_override if self.variant_type == "preset" else None
            ),
        }


@dataclass(frozen=True)
class MethodVariant:
    id: str
    label: str
    method_id: str
    method_verify_override: bool | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "method_id": self.method_id,
            "method_verify_override": self.method_verify_override,
        }


@dataclass(frozen=True)
class RunSpec:
    kind: Literal["prompt_lab", "methods_lab"]
    name: str
    variant_batch: str
    model_batch: str
    doc_ids: list[str]
    model_variants: list[ModelVariant]
    prompt_variants: list[PromptVariant]
    method_variants: list[MethodVariant]
    manifest_path: Path
    output_json_path: Path
    output_csv_path: Path


@dataclass(frozen=True)
class GroundTruthSweepPlan:
    session_id: str
    doc_ids: list[str]
    export_root: Path
    runs: list[RunSpec]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _backend_root() -> Path:
    return Path(__file__).resolve().parent


def _default_export_root() -> Path:
    return _backend_root() / ".annotation_tool" / "exports" / "20260308-ground-truth-sweep"


def _load_annotator_agents_prompt() -> str:
    path = _repo_root() / "prompts" / "annotator-agent" / "AGENTS.md"
    return path.read_text().strip()


def _ensure_manual_ground_truth_docs(session_id: str) -> list[str]:
    doc_ids = [str(doc_id) for doc_id in _load_session_index(session_id)]
    if not doc_ids:
        raise ValueError(f"No docs found in session '{session_id}'.")
    missing_manual: list[str] = []
    for doc_id in doc_ids:
        doc = _load_doc(doc_id, session_id)
        if doc is None:
            missing_manual.append(doc_id)
            continue
        enriched = _enrich_doc(doc, session_id)
        if not enriched.manual_annotations:
            missing_manual.append(doc_id)
    if missing_manual:
        joined = ", ".join(missing_manual)
        raise ValueError(f"Ground-truth sweep requires manual annotations for all docs. Missing: {joined}")
    return doc_ids


def _model_batches() -> dict[str, list[ModelVariant]]:
    return {
        "openai": [
            ModelVariant(
                id="codex_xhigh",
                label="codex_xhigh",
                model="openai.gpt-5.3-codex",
                reasoning_effort="xhigh",
            ),
            ModelVariant(
                id="codex_none",
                label="codex_none",
                model="openai.gpt-5.3-codex",
                reasoning_effort="none",
            ),
            ModelVariant(
                id="chat52_xhigh",
                label="chat52_xhigh",
                model="openai.gpt-5.2-chat",
                reasoning_effort="xhigh",
            ),
            ModelVariant(
                id="chat52_none",
                label="chat52_none",
                model="openai.gpt-5.2-chat",
                reasoning_effort="none",
            ),
        ],
        "claude": [
            ModelVariant(
                id="claude_thinking_off",
                label="claude_thinking_off",
                model="anthropic.claude-4.6-opus",
                anthropic_thinking=False,
            ),
            ModelVariant(
                id="claude_thinking_on",
                label="claude_thinking_on",
                model="anthropic.claude-4.6-opus",
                anthropic_thinking=True,
                anthropic_thinking_budget_tokens=2048,
            ),
        ],
        "gemini": [
            ModelVariant(
                id="gemini_pro",
                label="gemini_pro",
                model="google.gemini-3.1-pro-preview",
            ),
            ModelVariant(
                id="gemini_flash_lite",
                label="gemini_flash_lite",
                model="google.gemini-3.1-flash-lite-preview",
            ),
        ],
    }


def _prompt_batches() -> dict[str, list[PromptVariant]]:
    annotator_prompt = _load_annotator_agents_prompt()
    return {
        "prompt_text_core": [
            PromptVariant(
                id="baseline_raw",
                label="baseline_raw",
                variant_type="prompt",
                system_prompt=BASELINE_RAW_PROMPT,
            ),
            PromptVariant(
                id="annotator_agents_raw",
                label="annotator_agents_raw",
                variant_type="prompt",
                system_prompt=annotator_prompt,
            ),
            PromptVariant(
                id="preset_default_verify_off",
                label="preset_default_verify_off",
                variant_type="preset",
                preset_method_id="default",
                method_verify_override=False,
            ),
            PromptVariant(
                id="preset_default_verify_on",
                label="preset_default_verify_on",
                variant_type="preset",
                preset_method_id="default",
                method_verify_override=True,
            ),
            PromptVariant(
                id="preset_extended_verify_off",
                label="preset_extended_verify_off",
                variant_type="preset",
                preset_method_id="extended",
                method_verify_override=False,
            ),
            PromptVariant(
                id="preset_extended_verify_on",
                label="preset_extended_verify_on",
                variant_type="preset",
                preset_method_id="extended",
                method_verify_override=True,
            ),
        ],
        "prompt_preset_llm": [
            PromptVariant(
                id="preset_verified_verify_off",
                label="preset_verified_verify_off",
                variant_type="preset",
                preset_method_id="verified",
                method_verify_override=False,
            ),
            PromptVariant(
                id="preset_verified_verify_on",
                label="preset_verified_verify_on",
                variant_type="preset",
                preset_method_id="verified",
                method_verify_override=True,
            ),
            PromptVariant(
                id="preset_dual_verify_off",
                label="preset_dual_verify_off",
                variant_type="preset",
                preset_method_id="dual",
                method_verify_override=False,
            ),
            PromptVariant(
                id="preset_dual_verify_on",
                label="preset_dual_verify_on",
                variant_type="preset",
                preset_method_id="dual",
                method_verify_override=True,
            ),
            PromptVariant(
                id="preset_dual_split_verify_off",
                label="preset_dual_split_verify_off",
                variant_type="preset",
                preset_method_id="dual-split",
                method_verify_override=False,
            ),
            PromptVariant(
                id="preset_dual_split_verify_on",
                label="preset_dual_split_verify_on",
                variant_type="preset",
                preset_method_id="dual-split",
                method_verify_override=True,
            ),
        ],
        "prompt_preset_presidio": [
            PromptVariant(
                id="preset_presidio",
                label="preset_presidio",
                variant_type="preset",
                preset_method_id="presidio",
                method_verify_override=None,
            ),
            PromptVariant(
                id="preset_presidio_default_verify_off",
                label="preset_presidio_default_verify_off",
                variant_type="preset",
                preset_method_id="presidio+default",
                method_verify_override=False,
            ),
            PromptVariant(
                id="preset_presidio_default_verify_on",
                label="preset_presidio_default_verify_on",
                variant_type="preset",
                preset_method_id="presidio+default",
                method_verify_override=True,
            ),
            PromptVariant(
                id="preset_presidio_llm_split_verify_off",
                label="preset_presidio_llm_split_verify_off",
                variant_type="preset",
                preset_method_id="presidio+llm-split",
                method_verify_override=False,
            ),
            PromptVariant(
                id="preset_presidio_llm_split_verify_on",
                label="preset_presidio_llm_split_verify_on",
                variant_type="preset",
                preset_method_id="presidio+llm-split",
                method_verify_override=True,
            ),
        ],
    }


def _method_batches() -> dict[str, list[MethodVariant]]:
    return {
        "methods_core": [
            MethodVariant("default_verify_off", "default_verify_off", "default", False),
            MethodVariant("default_verify_on", "default_verify_on", "default", True),
            MethodVariant("extended_verify_off", "extended_verify_off", "extended", False),
            MethodVariant("extended_verify_on", "extended_verify_on", "extended", True),
            MethodVariant("verified_verify_off", "verified_verify_off", "verified", False),
            MethodVariant("verified_verify_on", "verified_verify_on", "verified", True),
            MethodVariant("dual_verify_off", "dual_verify_off", "dual", False),
            MethodVariant("dual_verify_on", "dual_verify_on", "dual", True),
            MethodVariant("dual_split_verify_off", "dual_split_verify_off", "dual-split", False),
            MethodVariant("dual_split_verify_on", "dual_split_verify_on", "dual-split", True),
            MethodVariant("presidio", "presidio", "presidio", None),
        ],
        "methods_hybrid": [
            MethodVariant(
                "presidio_default_verify_off",
                "presidio_default_verify_off",
                "presidio+default",
                False,
            ),
            MethodVariant(
                "presidio_default_verify_on",
                "presidio_default_verify_on",
                "presidio+default",
                True,
            ),
            MethodVariant(
                "presidio_llm_split_verify_off",
                "presidio_llm_split_verify_off",
                "presidio+llm-split",
                False,
            ),
            MethodVariant(
                "presidio_llm_split_verify_on",
                "presidio_llm_split_verify_on",
                "presidio+llm-split",
                True,
            ),
        ],
    }


def _public_runtime_payload(kind: Literal["prompt_lab", "methods_lab"], *, api_base: str | None) -> dict[str, Any]:
    payload = dict(DEFAULT_PROMPT_RUNTIME if kind == "prompt_lab" else DEFAULT_METHODS_RUNTIME)
    if api_base:
        payload["api_base"] = api_base
    return payload


def build_ground_truth_sweep_plan(
    *,
    session_id: str = "default",
    doc_ids: list[str] | None = None,
    export_root: Path | None = None,
) -> GroundTruthSweepPlan:
    resolved_doc_ids = list(doc_ids or _ensure_manual_ground_truth_docs(session_id))
    root = (export_root or _default_export_root()).resolve()
    manifests_dir = root / "manifests"
    reports_dir = root / "reports" / "runs"

    runs: list[RunSpec] = []
    for model_batch, models in _model_batches().items():
        for prompt_batch, prompts in _prompt_batches().items():
            stem = f"prompt_lab__{prompt_batch}__{model_batch}"
            runs.append(
                RunSpec(
                    kind="prompt_lab",
                    name=stem,
                    variant_batch=prompt_batch,
                    model_batch=model_batch,
                    doc_ids=resolved_doc_ids,
                    model_variants=models,
                    prompt_variants=prompts,
                    method_variants=[],
                    manifest_path=manifests_dir / f"{stem}.yaml",
                    output_json_path=reports_dir / f"{stem}.json",
                    output_csv_path=reports_dir / f"{stem}.csv",
                )
            )
    for model_batch, models in _model_batches().items():
        for method_batch, methods in _method_batches().items():
            stem = f"methods_lab__{method_batch}__{model_batch}"
            runs.append(
                RunSpec(
                    kind="methods_lab",
                    name=stem,
                    variant_batch=method_batch,
                    model_batch=model_batch,
                    doc_ids=resolved_doc_ids,
                    model_variants=models,
                    prompt_variants=[],
                    method_variants=methods,
                    manifest_path=manifests_dir / f"{stem}.yaml",
                    output_json_path=reports_dir / f"{stem}.json",
                    output_csv_path=reports_dir / f"{stem}.csv",
                )
            )
    return GroundTruthSweepPlan(
        session_id=session_id,
        doc_ids=resolved_doc_ids,
        export_root=root,
        runs=runs,
    )


def _run_manifest_payload(
    spec: RunSpec,
    *,
    session_id: str,
    api_base: str | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "kind": spec.kind,
        "session": session_id,
        "name": spec.name,
        "doc_ids": spec.doc_ids,
        "models": [variant.to_payload() for variant in spec.model_variants],
        "runtime": _public_runtime_payload(spec.kind, api_base=api_base),
        "concurrency": DEFAULT_CONCURRENCY,
    }
    if spec.kind == "prompt_lab":
        payload["prompts"] = [variant.to_payload() for variant in spec.prompt_variants]
    else:
        payload["methods"] = [variant.to_payload() for variant in spec.method_variants]
    return payload


def _execution_payload(
    spec: RunSpec,
    *,
    session_id: str,
    api_key: str | None,
    api_base: str | None,
) -> dict[str, Any]:
    payload = _run_manifest_payload(spec, session_id=session_id, api_base=api_base)
    runtime = dict(payload["runtime"])
    if api_key:
        runtime["api_key"] = api_key
    payload["runtime"] = runtime
    return payload


def _json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _name_tolerant_metrics(cell: dict[str, Any]) -> dict[str, Any]:
    co_primary = cell.get("co_primary_metrics", {})
    if not isinstance(co_primary, dict):
        return {}
    metric = co_primary.get("exact_name_affix_tolerant", {})
    return metric if isinstance(metric, dict) else {}


def _matrix_rows(kind: str, detail: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in detail.get("matrix", {}).get("cells", []):
        if not isinstance(cell, dict):
            continue
        tolerant = _name_tolerant_metrics(cell)
        tolerant_micro = tolerant.get("micro", {}) if isinstance(tolerant, dict) else {}
        raw_tolerant = cell.get("raw_co_primary_metrics", {}).get("exact_name_affix_tolerant", {})
        raw_tolerant_micro = raw_tolerant.get("micro", {}) if isinstance(raw_tolerant, dict) else {}
        error_families = cell.get("error_families", {})
        if not isinstance(error_families, dict):
            error_families = {}
        resolution_summary = cell.get("resolution_summary", {})
        if not isinstance(resolution_summary, dict):
            resolution_summary = {}
        row = {
            "id": cell.get("id"),
            "status": cell.get("status"),
            "model_id": cell.get("model_id"),
            "model_label": cell.get("model_label"),
            "total_docs": cell.get("total_docs"),
            "completed_docs": cell.get("completed_docs"),
            "failed_docs": cell.get("failed_docs"),
            "pending_docs": cell.get("pending_docs", 0),
            "precision": cell.get("micro", {}).get("precision"),
            "recall": cell.get("micro", {}).get("recall"),
            "f1": cell.get("micro", {}).get("f1"),
            "raw_precision": cell.get("raw_micro", {}).get("precision"),
            "raw_recall": cell.get("raw_micro", {}).get("recall"),
            "raw_f1": cell.get("raw_micro", {}).get("f1"),
            "exact_name_affix_tolerant_precision": tolerant_micro.get("precision"),
            "exact_name_affix_tolerant_recall": tolerant_micro.get("recall"),
            "exact_name_affix_tolerant_f1": tolerant_micro.get("f1"),
            "raw_exact_name_affix_tolerant_precision": raw_tolerant_micro.get("precision"),
            "raw_exact_name_affix_tolerant_recall": raw_tolerant_micro.get("recall"),
            "raw_exact_name_affix_tolerant_f1": raw_tolerant_micro.get("f1"),
            "exact_name_affix_gap_f1": cell.get("exact_name_affix_gap_f1"),
            "raw_exact_name_affix_gap_f1": cell.get("raw_exact_name_affix_gap_f1"),
            "boundary_fix_count": resolution_summary.get("boundary_fix_count", 0),
            "augmentation_count": resolution_summary.get("augmentation_count", 0),
            "mean_confidence": cell.get("mean_confidence"),
        }
        for family in ERROR_FAMILY_COLUMNS:
            row[f"error_family_{family}"] = int(error_families.get(family, 0))
        if kind == "prompt_lab":
            row["prompt_id"] = cell.get("prompt_id")
            row["prompt_label"] = cell.get("prompt_label")
        else:
            row["method_id"] = cell.get("method_id")
            row["method_label"] = cell.get("method_label")
        rows.append(row)
    return rows


def _write_run_csv(path: Path, *, kind: str, detail: dict[str, Any]) -> None:
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
            "exact_name_affix_tolerant_precision",
            "exact_name_affix_tolerant_recall",
            "exact_name_affix_tolerant_f1",
            "raw_exact_name_affix_tolerant_precision",
            "raw_exact_name_affix_tolerant_recall",
            "raw_exact_name_affix_tolerant_f1",
            "exact_name_affix_gap_f1",
            "raw_exact_name_affix_gap_f1",
            "boundary_fix_count",
            "augmentation_count",
            "mean_confidence",
            *[f"error_family_{family}" for family in ERROR_FAMILY_COLUMNS],
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
            "exact_name_affix_tolerant_precision",
            "exact_name_affix_tolerant_recall",
            "exact_name_affix_tolerant_f1",
            "raw_exact_name_affix_tolerant_precision",
            "raw_exact_name_affix_tolerant_recall",
            "raw_exact_name_affix_tolerant_f1",
            "exact_name_affix_gap_f1",
            "raw_exact_name_affix_gap_f1",
            "boundary_fix_count",
            "augmentation_count",
            "mean_confidence",
            *[f"error_family_{family}" for family in ERROR_FAMILY_COLUMNS],
        ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _blocked_detail(
    spec: RunSpec,
    *,
    session_id: str,
    api_base: str | None,
    status: str,
    error: str,
) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": None,
        "name": spec.name,
        "kind": spec.kind,
        "manifest_name": spec.manifest_path.name,
        "session_id": session_id,
        "status": status,
        "created_at": None,
        "started_at": None,
        "finished_at": None,
        "doc_ids": spec.doc_ids,
        "runtime": _public_runtime_payload(spec.kind, api_base=api_base),
        "concurrency": DEFAULT_CONCURRENCY,
        "warnings": [],
        "errors": [error],
        "progress": {"total_tasks": 0, "completed_tasks": 0, "failed_tasks": 0},
    }
    if spec.kind == "prompt_lab":
        prompts = [variant.to_payload() for variant in spec.prompt_variants]
        models = [variant.to_payload() for variant in spec.model_variants]
        cells: list[dict[str, Any]] = []
        for model in models:
            for prompt in prompts:
                cells.append(
                    {
                        "id": f"{model['id']}__{prompt['id']}",
                        "model_id": model["id"],
                        "model_label": model["label"],
                        "prompt_id": prompt["id"],
                        "prompt_label": prompt["label"],
                        "status": status,
                        "total_docs": len(spec.doc_ids),
                        "completed_docs": 0,
                        "failed_docs": 0,
                        "pending_docs": len(spec.doc_ids),
                        "error_count": 0,
                        "error_families": {},
                        "micro": {
                            "precision": 0.0,
                            "recall": 0.0,
                            "f1": 0.0,
                            "tp": 0,
                            "fp": 0,
                            "fn": 0,
                        },
                        "co_primary_metrics": {
                            "exact_name_affix_tolerant": {
                                "micro": {
                                    "precision": 0.0,
                                    "recall": 0.0,
                                    "f1": 0.0,
                                    "tp": 0,
                                    "fp": 0,
                                    "fn": 0,
                                },
                                "per_label": {},
                            }
                        },
                        "per_label": {},
                        "mean_confidence": None,
                    }
                )
        base.update(
            {
                "prompt_count": len(prompts),
                "model_count": len(models),
                "total_tasks": len(spec.doc_ids) * len(prompts) * len(models),
                "completed_tasks": 0,
                "failed_tasks": 0,
                "prompts": prompts,
                "models": models,
                "matrix": {
                    "models": [{"id": item["id"], "label": item["label"]} for item in models],
                    "prompts": [{"id": item["id"], "label": item["label"]} for item in prompts],
                    "cells": cells,
                    "available_labels": [],
                },
            }
        )
        return base

    methods = [variant.to_payload() for variant in spec.method_variants]
    models = [variant.to_payload() for variant in spec.model_variants]
    cells = []
    for model in models:
        for method in methods:
            cells.append(
                {
                    "id": f"{model['id']}__{method['id']}",
                    "model_id": model["id"],
                    "model_label": model["label"],
                    "method_id": method["id"],
                    "method_label": method["label"],
                    "status": status,
                    "total_docs": len(spec.doc_ids),
                    "completed_docs": 0,
                    "failed_docs": 0,
                    "pending_docs": len(spec.doc_ids),
                    "error_count": 0,
                    "error_families": {},
                    "micro": {
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1": 0.0,
                        "tp": 0,
                        "fp": 0,
                        "fn": 0,
                    },
                    "co_primary_metrics": {
                        "exact_name_affix_tolerant": {
                            "micro": {
                                "precision": 0.0,
                                "recall": 0.0,
                                "f1": 0.0,
                                "tp": 0,
                                "fp": 0,
                                "fn": 0,
                            },
                            "per_label": {},
                        }
                    },
                    "per_label": {},
                    "mean_confidence": None,
                }
            )
    base.update(
        {
            "method_count": len(methods),
            "model_count": len(models),
            "total_tasks": len(spec.doc_ids) * len(methods) * len(models),
            "completed_tasks": 0,
            "failed_tasks": 0,
            "methods": methods,
            "models": models,
            "matrix": {
                "models": [{"id": item["id"], "label": item["label"]} for item in models],
                "methods": [{"id": item["id"], "label": item["label"]} for item in methods],
                "cells": cells,
                "available_labels": [],
            },
        }
    )
    return base


def _write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False))


def _write_run_json(path: Path, detail: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(detail), indent=2))


def _execute_run(
    spec: RunSpec,
    *,
    session_id: str,
    api_key: str | None,
    api_base: str | None,
) -> dict[str, Any]:
    payload = _execution_payload(spec, session_id=session_id, api_key=api_key, api_base=api_base)
    try:
        if spec.kind == "prompt_lab":
            body = PromptLabRunCreateBody.model_validate(payload)
            detail = create_prompt_lab_run(body, session_id=session_id, run_async=False)
        else:
            body = MethodsLabRunCreateBody.model_validate(payload)
            detail = create_methods_lab_run(body, session_id=session_id, run_async=False)
        detail = {
            **detail,
            "kind": spec.kind,
            "manifest_name": spec.manifest_path.name,
            "variant_batch": spec.variant_batch,
            "model_batch": spec.model_batch,
        }
    except HTTPException as exc:
        error = exc.detail if isinstance(exc.detail, str) else json.dumps(exc.detail)
        detail = _blocked_detail(
            spec,
            session_id=session_id,
            api_base=api_base,
            status="blocked",
            error=error,
        )
    except Exception as exc:  # pragma: no cover - defensive guard
        detail = _blocked_detail(
            spec,
            session_id=session_id,
            api_base=api_base,
            status="failed",
            error=str(exc) or exc.__class__.__name__,
        )

    _write_run_json(spec.output_json_path, detail)
    _write_run_csv(spec.output_csv_path, kind=spec.kind, detail=detail)
    return detail


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _prf_from_counts(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _cell_records(run_details: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompt_records: list[dict[str, Any]] = []
    method_records: list[dict[str, Any]] = []

    for detail in run_details:
        kind = str(detail.get("kind", ""))
        models_by_id = {
            str(model.get("id")): model for model in detail.get("models", []) if isinstance(model, dict)
        }
        for cell in detail.get("matrix", {}).get("cells", []):
            if not isinstance(cell, dict):
                continue
            model_meta = models_by_id.get(str(cell.get("model_id")), {})
            tolerant = _name_tolerant_metrics(cell)
            tolerant_micro = tolerant.get("micro", {}) if isinstance(tolerant, dict) else {}
            raw_tolerant = cell.get("raw_co_primary_metrics", {}).get(
                "exact_name_affix_tolerant", {}
            )
            raw_tolerant_micro = raw_tolerant.get("micro", {}) if isinstance(raw_tolerant, dict) else {}
            error_families = cell.get("error_families", {})
            if not isinstance(error_families, dict):
                error_families = {}
            resolution_summary = cell.get("resolution_summary", {})
            if not isinstance(resolution_summary, dict):
                resolution_summary = {}
            record = {
                "kind": kind,
                "run_name": detail.get("name"),
                "run_id": detail.get("id"),
                "manifest_name": detail.get("manifest_name"),
                "variant_batch": detail.get("variant_batch"),
                "model_batch": detail.get("model_batch"),
                "status": cell.get("status"),
                "model_variant_id": cell.get("model_id"),
                "model_variant_label": cell.get("model_label"),
                "model_name": model_meta.get("model"),
                "provider": _provider_from_model(str(model_meta.get("model", ""))),
                "reasoning_effort": model_meta.get("reasoning_effort"),
                "anthropic_thinking": model_meta.get("anthropic_thinking"),
                "total_docs": int(cell.get("total_docs", 0)),
                "completed_docs": int(cell.get("completed_docs", 0)),
                "failed_docs": int(cell.get("failed_docs", 0)),
                "pending_docs": int(cell.get("pending_docs", 0)),
                "precision": _safe_float(cell.get("micro", {}).get("precision")) or 0.0,
                "recall": _safe_float(cell.get("micro", {}).get("recall")) or 0.0,
                "f1": _safe_float(cell.get("micro", {}).get("f1")) or 0.0,
                "raw_precision": _safe_float(cell.get("raw_micro", {}).get("precision")) or 0.0,
                "raw_recall": _safe_float(cell.get("raw_micro", {}).get("recall")) or 0.0,
                "raw_f1": _safe_float(cell.get("raw_micro", {}).get("f1")) or 0.0,
                "tp": int(cell.get("micro", {}).get("tp", 0)),
                "fp": int(cell.get("micro", {}).get("fp", 0)),
                "fn": int(cell.get("micro", {}).get("fn", 0)),
                "exact_name_affix_tolerant_precision": (
                    _safe_float(tolerant_micro.get("precision")) or 0.0
                ),
                "exact_name_affix_tolerant_recall": (
                    _safe_float(tolerant_micro.get("recall")) or 0.0
                ),
                "exact_name_affix_tolerant_f1": (
                    _safe_float(tolerant_micro.get("f1")) or 0.0
                ),
                "raw_exact_name_affix_tolerant_precision": (
                    _safe_float(raw_tolerant_micro.get("precision")) or 0.0
                ),
                "raw_exact_name_affix_tolerant_recall": (
                    _safe_float(raw_tolerant_micro.get("recall")) or 0.0
                ),
                "raw_exact_name_affix_tolerant_f1": (
                    _safe_float(raw_tolerant_micro.get("f1")) or 0.0
                ),
                "exact_name_affix_tolerant_tp": int(tolerant_micro.get("tp", 0)),
                "exact_name_affix_tolerant_fp": int(tolerant_micro.get("fp", 0)),
                "exact_name_affix_tolerant_fn": int(tolerant_micro.get("fn", 0)),
                "exact_name_affix_gap_f1": _safe_float(cell.get("exact_name_affix_gap_f1"))
                or 0.0,
                "raw_exact_name_affix_gap_f1": _safe_float(
                    cell.get("raw_exact_name_affix_gap_f1")
                )
                or 0.0,
                "boundary_fix_count": int(resolution_summary.get("boundary_fix_count", 0)),
                "augmentation_count": int(resolution_summary.get("augmentation_count", 0)),
                "error_family_counts": {
                    family: int(error_families.get(family, 0))
                    for family in ERROR_FAMILY_COLUMNS
                    if int(error_families.get(family, 0)) > 0
                },
                "mean_confidence": _safe_float(cell.get("mean_confidence")),
                "per_label": cell.get("per_label", {}),
                "raw_per_label": cell.get("raw_per_label", {}),
                "resolution_summary": resolution_summary,
            }
            for family in ERROR_FAMILY_COLUMNS:
                record[f"error_family_{family}"] = int(error_families.get(family, 0))
            if kind == "prompt_lab":
                record["variant_id"] = cell.get("prompt_id")
                record["variant_label"] = cell.get("prompt_label")
                prompt_records.append(record)
            else:
                record["variant_id"] = cell.get("method_id")
                record["variant_label"] = cell.get("method_label")
                method_records.append(record)
    return prompt_records, method_records


def _provider_from_model(model: str) -> str:
    if model.startswith("anthropic."):
        return "Anthropic"
    if model.startswith("google."):
        return "Google Gemini"
    return "OpenAI"


def _primary_metric_value(row: dict[str, Any], field: str) -> float:
    tolerant_field = {
        "precision": "exact_name_affix_tolerant_precision",
        "recall": "exact_name_affix_tolerant_recall",
        "f1": "exact_name_affix_tolerant_f1",
    }.get(field)
    if tolerant_field is not None:
        tolerant_value = _safe_float(row.get(tolerant_field))
        if tolerant_value is not None:
            return tolerant_value
    return _safe_float(row.get(field)) or 0.0


def _aggregate_variant_rankings(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for record in records:
        label = str(record["variant_label"])
        bucket = buckets.setdefault(
            label,
            {
                "variant_label": label,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "exact_name_affix_tolerant_tp": 0,
                "exact_name_affix_tolerant_fp": 0,
                "exact_name_affix_tolerant_fn": 0,
                "completed_cells": 0,
                "failed_cells": 0,
            },
        )
        bucket["tp"] += int(record["tp"])
        bucket["fp"] += int(record["fp"])
        bucket["fn"] += int(record["fn"])
        bucket["exact_name_affix_tolerant_tp"] += int(record["exact_name_affix_tolerant_tp"])
        bucket["exact_name_affix_tolerant_fp"] += int(record["exact_name_affix_tolerant_fp"])
        bucket["exact_name_affix_tolerant_fn"] += int(record["exact_name_affix_tolerant_fn"])
        if str(record["status"]) in {"completed", "completed_with_errors"}:
            bucket["completed_cells"] += 1
        elif str(record["status"]) not in {"pending", "running"}:
            bucket["failed_cells"] += 1
    ranking: list[dict[str, Any]] = []
    for bucket in buckets.values():
        prf = _prf_from_counts(int(bucket["tp"]), int(bucket["fp"]), int(bucket["fn"]))
        tolerant_prf = _prf_from_counts(
            int(bucket["exact_name_affix_tolerant_tp"]),
            int(bucket["exact_name_affix_tolerant_fp"]),
            int(bucket["exact_name_affix_tolerant_fn"]),
        )
        ranking.append(
            {
                **bucket,
                **prf,
                "exact_name_affix_tolerant_precision": tolerant_prf["precision"],
                "exact_name_affix_tolerant_recall": tolerant_prf["recall"],
                "exact_name_affix_tolerant_f1": tolerant_prf["f1"],
            }
        )
    ranking.sort(
        key=lambda item: (-_primary_metric_value(item, "f1"), -float(item["f1"]), item["variant_label"])
    )
    return ranking


def _aggregate_by_group(records: list[dict[str, Any]], *, field: str) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    for record in records:
        key = str(record[field])
        bucket = buckets.setdefault(
            key,
            {
                "label": key,
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "exact_name_affix_tolerant_tp": 0,
                "exact_name_affix_tolerant_fp": 0,
                "exact_name_affix_tolerant_fn": 0,
                "cell_count": 0,
            },
        )
        bucket["tp"] += int(record["tp"])
        bucket["fp"] += int(record["fp"])
        bucket["fn"] += int(record["fn"])
        bucket["exact_name_affix_tolerant_tp"] += int(record["exact_name_affix_tolerant_tp"])
        bucket["exact_name_affix_tolerant_fp"] += int(record["exact_name_affix_tolerant_fp"])
        bucket["exact_name_affix_tolerant_fn"] += int(record["exact_name_affix_tolerant_fn"])
        bucket["cell_count"] += 1
    rows: list[dict[str, Any]] = []
    for bucket in buckets.values():
        prf = _prf_from_counts(bucket["tp"], bucket["fp"], bucket["fn"])
        tolerant_prf = _prf_from_counts(
            bucket["exact_name_affix_tolerant_tp"],
            bucket["exact_name_affix_tolerant_fp"],
            bucket["exact_name_affix_tolerant_fn"],
        )
        rows.append(
            {
                **bucket,
                **prf,
                "exact_name_affix_tolerant_precision": tolerant_prf["precision"],
                "exact_name_affix_tolerant_recall": tolerant_prf["recall"],
                "exact_name_affix_tolerant_f1": tolerant_prf["f1"],
            }
        )
    rows.sort(key=lambda item: (-_primary_metric_value(item, "f1"), -float(item["f1"]), item["label"]))
    return rows


def _aggregate_per_model(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    by_model: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_model.setdefault(str(record["model_variant_label"]), []).append(record)
    for model_label, model_records in sorted(by_model.items()):
        grouped[model_label] = _aggregate_variant_rankings(model_records)
    return grouped


def _verifier_comparisons(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_base: dict[str, dict[str, dict[str, Any]]] = {}
    for row in _aggregate_variant_rankings(records):
        label = str(row["variant_label"])
        state: str | None = None
        base_label = label
        if label.endswith("_verify_off"):
            state = "off"
            base_label = label[: -len("_verify_off")]
        elif label.endswith("_verify_on"):
            state = "on"
            base_label = label[: -len("_verify_on")]
        if state is None:
            continue
        by_base.setdefault(base_label, {})[state] = row
    comparisons: list[dict[str, Any]] = []
    for base_label, states in sorted(by_base.items()):
        off = states.get("off")
        on = states.get("on")
        if off is None or on is None:
            continue
        comparisons.append(
            {
                "base_label": base_label,
                "off_f1": _primary_metric_value(off, "f1"),
                "on_f1": _primary_metric_value(on, "f1"),
                "off_exact_f1": float(off["f1"]),
                "on_exact_f1": float(on["f1"]),
                "delta_f1": _primary_metric_value(on, "f1") - _primary_metric_value(off, "f1"),
            }
        )
    comparisons.sort(key=lambda item: (-abs(item["delta_f1"]), item["base_label"]))
    return comparisons


def _comparison_row(records: list[dict[str, Any]], labels: tuple[str, str]) -> dict[str, Any]:
    rows = _aggregate_variant_rankings([record for record in records if record["variant_label"] in labels])
    by_label = {str(row["variant_label"]): row for row in rows}
    left = by_label.get(labels[0], {"f1": 0.0})
    right = by_label.get(labels[1], {"f1": 0.0})
    return {
        "left_label": labels[0],
        "left_f1": _primary_metric_value(left, "f1"),
        "left_exact_f1": float(left["f1"]),
        "right_label": labels[1],
        "right_f1": _primary_metric_value(right, "f1"),
        "right_exact_f1": float(right["f1"]),
        "delta_f1": _primary_metric_value(left, "f1") - _primary_metric_value(right, "f1"),
    }


def _aggregate_model_family_comparisons(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    families = {
        "codex": ("codex_xhigh", "codex_none"),
        "chat52": ("chat52_xhigh", "chat52_none"),
        "claude": ("claude_thinking_on", "claude_thinking_off"),
    }
    rows: list[dict[str, Any]] = []
    for family, labels in families.items():
        family_records = [record for record in records if record["model_variant_label"] in labels]
        if not family_records:
            continue
        comparison = _comparison_row(family_records, labels)
        rows.append({"family": family, **comparison})
    return rows


def _top_bottom_configurations(
    prompt_records: list[dict[str, Any]], method_records: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = [
        {
            "kind": record["kind"],
            "variant_label": record["variant_label"],
            "model_variant_label": record["model_variant_label"],
            "f1": float(record["f1"]),
            "exact_name_affix_tolerant_f1": float(record["exact_name_affix_tolerant_f1"]),
            "precision": float(record["precision"]),
            "recall": float(record["recall"]),
            "status": record["status"],
        }
        for record in [*prompt_records, *method_records]
        if str(record["status"]) in {"completed", "completed_with_errors"}
    ]
    rows.sort(
        key=lambda item: (
            -_primary_metric_value(item, "f1"),
            -item["f1"],
            item["kind"],
            item["variant_label"],
            item["model_variant_label"],
        )
    )
    if not rows:
        return [], []
    bottom_rows = sorted(
        rows,
        key=lambda item: (
            _primary_metric_value(item, "f1"),
            item["f1"],
            item["kind"],
            item["variant_label"],
            item["model_variant_label"],
        ),
    )[:10]
    return rows[:10], bottom_rows


def _aggregate_label_patterns(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, dict[str, int]] = {}
    for record in records:
        per_label = record.get("per_label", {})
        if not isinstance(per_label, dict):
            continue
        for label, metrics in per_label.items():
            if not isinstance(metrics, dict):
                continue
            bucket = counts.setdefault(str(label), {"tp": 0, "fp": 0, "fn": 0, "support": 0})
            bucket["tp"] += int(metrics.get("tp", 0))
            bucket["fp"] += int(metrics.get("fp", 0))
            bucket["fn"] += int(metrics.get("fn", 0))
            bucket["support"] += int(metrics.get("support", 0))
    rows: list[dict[str, Any]] = []
    for label, bucket in counts.items():
        rows.append({"label": label, **bucket})
    rows.sort(key=lambda item: (-item["fn"], -item["fp"], item["label"]))
    return rows


def _write_aggregate_csv(path: Path, *, kind: str, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "prompt_lab":
        fieldnames = [
            "manifest_name",
            "run_name",
            "run_id",
            "variant_batch",
            "model_batch",
            "provider",
            "model_variant_label",
            "model_name",
            "variant_label",
            "status",
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
            "exact_name_affix_tolerant_precision",
            "exact_name_affix_tolerant_recall",
            "exact_name_affix_tolerant_f1",
            "raw_exact_name_affix_tolerant_precision",
            "raw_exact_name_affix_tolerant_recall",
            "raw_exact_name_affix_tolerant_f1",
            "exact_name_affix_gap_f1",
            "raw_exact_name_affix_gap_f1",
            "tp",
            "fp",
            "fn",
            "boundary_fix_count",
            "augmentation_count",
            "mean_confidence",
            "error_family_empty_output_finish_reason_length",
            "error_family_connection_error",
            "error_family_config_error",
            "error_family_timeout",
            "error_family_unknown_error",
        ]
    else:
        fieldnames = [
            "manifest_name",
            "run_name",
            "run_id",
            "variant_batch",
            "model_batch",
            "provider",
            "model_variant_label",
            "model_name",
            "variant_label",
            "status",
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
            "exact_name_affix_tolerant_precision",
            "exact_name_affix_tolerant_recall",
            "exact_name_affix_tolerant_f1",
            "raw_exact_name_affix_tolerant_precision",
            "raw_exact_name_affix_tolerant_recall",
            "raw_exact_name_affix_tolerant_f1",
            "exact_name_affix_gap_f1",
            "raw_exact_name_affix_gap_f1",
            "tp",
            "fp",
            "fn",
            "boundary_fix_count",
            "augmentation_count",
            "mean_confidence",
            "error_family_empty_output_finish_reason_length",
            "error_family_connection_error",
            "error_family_config_error",
            "error_family_timeout",
            "error_family_unknown_error",
        ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _format_ranking_table(rows: list[dict[str, Any]], *, variant_key: str = "variant_label") -> str:
    if not rows:
        return "_No completed cells._"
    header = (
        "| Variant | NAME-Tolerant F1 | Exact F1 | NAME-Tolerant Precision | NAME-Tolerant Recall | TP | FP | FN |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    )
    body = [
        f"| {row[variant_key]} | {float(row.get('exact_name_affix_tolerant_f1', row['f1'])):.4f} | "
        f"{float(row['f1']):.4f} | "
        f"{float(row.get('exact_name_affix_tolerant_precision', row['precision'])):.4f} | "
        f"{float(row.get('exact_name_affix_tolerant_recall', row['recall'])):.4f} | "
        f"{int(row['tp'])} | {int(row['fp'])} | {int(row['fn'])} |"
        for row in rows
    ]
    return "\n".join([header, *body])


def _format_cell_table(rows: list[dict[str, Any]], *, label_key: str = "variant_label") -> str:
    if not rows:
        return "_No completed cells._"
    header = (
        "| Model | Variant | NAME-Tolerant F1 | Exact F1 | NAME-Tolerant Precision | NAME-Tolerant Recall |\n"
        "| --- | --- | ---: | ---: | ---: | ---: |"
    )
    body = [
        f"| {row['model_variant_label']} | {row[label_key]} "
        f"| {float(row.get('exact_name_affix_tolerant_f1', row['f1'])):.4f} "
        f"| {float(row['f1']):.4f} "
        f"| {float(row.get('exact_name_affix_tolerant_precision', row['precision'])):.4f} "
        f"| {float(row.get('exact_name_affix_tolerant_recall', row['recall'])):.4f} |"
        for row in rows
    ]
    return "\n".join([header, *body])


def _format_run_inventory(runs: list[dict[str, Any]]) -> str:
    header = "| Manifest | Run ID | Status | Failed Tasks | Errors |\n| --- | --- | --- | ---: | --- |"
    body = []
    for run in runs:
        errors = "; ".join(str(item) for item in run.get("errors", [])[:2])
        body.append(
            f"| {run['manifest_name']} | {run.get('id') or '-'} | {run['status']} "
            f"| {int(run.get('failed_tasks', 0))} | {errors or '-'} |"
        )
    return "\n".join([header, *body])


def _format_comparison_rows(rows: list[dict[str, Any]], *, label_field: str) -> str:
    if not rows:
        return "_No comparisons available._"
    header = (
        f"| {label_field} | Off NAME-Tolerant F1 | On NAME-Tolerant F1 | Delta | "
        "Off Exact F1 | On Exact F1 |\n| --- | ---: | ---: | ---: | ---: | ---: |"
    )
    body = [
        f"| {row['base_label']} | {row['off_f1']:.4f} | {row['on_f1']:.4f} | {row['delta_f1']:+.4f} "
        f"| {row['off_exact_f1']:.4f} | {row['on_exact_f1']:.4f} |"
        for row in rows
    ]
    return "\n".join([header, *body])


def _format_family_comparisons(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No family comparisons available._"
    header = (
        "| Family | Left | Left NAME-Tolerant F1 | Right | Right NAME-Tolerant F1 | Delta | "
        "Left Exact F1 | Right Exact F1 |\n| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |"
    )
    body = [
        f"| {row['family']} | {row['left_label']} | {row['left_f1']:.4f} | {row['right_label']} "
        f"| {row['right_f1']:.4f} | {row['delta_f1']:+.4f} "
        f"| {row['left_exact_f1']:.4f} | {row['right_exact_f1']:.4f} |"
        for row in rows
    ]
    return "\n".join([header, *body])


def _format_provider_summary(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No provider summary available._"
    header = (
        "| Provider | NAME-Tolerant F1 | Exact F1 | NAME-Tolerant Precision | NAME-Tolerant Recall | Cells |\n"
        "| --- | ---: | ---: | ---: | ---: | ---: |"
    )
    body = [
        f"| {row['label']} | {float(row.get('exact_name_affix_tolerant_f1', row['f1'])):.4f} | "
        f"{float(row['f1']):.4f} | "
        f"{float(row.get('exact_name_affix_tolerant_precision', row['precision'])):.4f} | "
        f"{float(row.get('exact_name_affix_tolerant_recall', row['recall'])):.4f} | {int(row['cell_count'])} |"
        for row in rows
    ]
    return "\n".join([header, *body])


def _format_label_patterns(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No label patterns available._"
    header = "| Label | FN | FP | TP | Support |\n| --- | ---: | ---: | ---: | ---: |"
    body = [
        f"| {row['label']} | {int(row['fn'])} | {int(row['fp'])} | {int(row['tp'])} | {int(row['support'])} |"
        for row in rows[:10]
    ]
    return "\n".join([header, *body])


def _format_top_bottom(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No completed configurations._"
    header = (
        "| Kind | Variant | Model | NAME-Tolerant F1 | Exact F1 | NAME-Tolerant Precision | NAME-Tolerant Recall |\n"
        "| --- | --- | --- | ---: | ---: | ---: | ---: |"
    )
    body = [
        f"| {row['kind']} | {row['variant_label']} | {row['model_variant_label']} "
        f"| {float(row.get('exact_name_affix_tolerant_f1', row['f1'])):.4f} "
        f"| {row['f1']:.4f} "
        f"| {float(row.get('exact_name_affix_tolerant_precision', row['precision'])):.4f} "
        f"| {float(row.get('exact_name_affix_tolerant_recall', row['recall'])):.4f} |"
        for row in rows
    ]
    return "\n".join([header, *body])


def _format_blocked_runs(runs: list[dict[str, Any]]) -> str:
    blocked = [run for run in runs if run["status"] == "blocked"]
    if not blocked:
        return "_None._"
    header = "| Manifest | Models | Error |\n| --- | --- | --- |"
    body = []
    for run in blocked:
        models = ", ".join(
            str(item.get("label"))
            for item in run.get("models", [])
            if isinstance(item, dict)
        )
        error = "; ".join(str(item) for item in run.get("errors", [])[:1])
        body.append(f"| {run['manifest_name']} | {models} | {error} |")
    return "\n".join([header, *body])


def _generate_report(
    aggregate: dict[str, Any],
    *,
    output_path: Path,
) -> None:
    prompt_per_model = aggregate["prompt_per_model_rankings"]
    method_per_model = aggregate["method_per_model_rankings"]
    raw_prompt_rows = [
        row
        for row in aggregate["prompt_records"]
        if row["variant_label"] in {"baseline_raw", "annotator_agents_raw"}
        and str(row["status"]) in {"completed", "completed_with_errors"}
    ]
    raw_prompt_table = _format_cell_table(
        sorted(
            raw_prompt_rows,
            key=lambda item: (
                item["model_variant_label"],
                item["variant_label"],
            ),
        )
    )

    preset_prompt_sections = []
    for model_label, rows in prompt_per_model.items():
        preset_rows = [
            row for row in rows if row["variant_label"] not in {"baseline_raw", "annotator_agents_raw"}
        ]
        if not preset_rows:
            continue
        preset_prompt_sections.append(f"### {model_label}\n{_format_ranking_table(preset_rows)}")

    method_sections = []
    for model_label, rows in method_per_model.items():
        method_sections.append(f"### {model_label}\n{_format_ranking_table(rows)}")

    lines = [
        "# Ground-Truth Sweep Final Report",
        "",
        "## Overview",
        f"- Export root: `{aggregate['summary']['export_root']}`",
        f"- Runs: {aggregate['summary']['run_count']}",
        f"- Completed runs: {aggregate['summary']['completed_run_count']}",
        f"- Blocked runs: {aggregate['summary']['blocked_run_count']}",
        f"- Failed runs: {aggregate['summary']['failed_run_count']}",
        f"- Documents per run: {aggregate['summary']['doc_count']}",
        "",
        "## Run Inventory",
        _format_run_inventory(aggregate["runs"]),
        "",
        "## Overall Prompt Variant Ranking",
        _format_ranking_table(aggregate["prompt_rankings"]),
        "",
        "## Overall Method Variant Ranking",
        _format_ranking_table(aggregate["method_rankings"]),
        "",
        "## Raw Prompt Comparison By Model",
        raw_prompt_table,
        "",
        "## Preset-Backed Prompt Comparison By Model",
        "\n\n".join(preset_prompt_sections) if preset_prompt_sections else "_No preset-backed prompt results._",
        "",
        "## Method Comparison By Model",
        "\n\n".join(method_sections) if method_sections else "_No method results._",
        "",
        "## Verifier Comparisons",
        "### Prompt Presets",
        _format_comparison_rows(aggregate["prompt_verifier_comparisons"], label_field="Prompt"),
        "",
        "### Methods",
        _format_comparison_rows(aggregate["method_verifier_comparisons"], label_field="Method"),
        "",
        "## Reasoning And Thinking Comparisons",
        _format_family_comparisons(aggregate["model_family_comparisons"]),
        "",
        "## Provider-Level Summary",
        _format_provider_summary(aggregate["provider_summary"]),
        "",
        "## Top 10 Configurations",
        _format_top_bottom(aggregate["top_configurations"]),
        "",
        "## Bottom 10 Configurations",
        _format_top_bottom(aggregate["bottom_configurations"]),
        "",
        "## Notable False-Negative / False-Positive Patterns",
        "### Prompt Lab",
        _format_label_patterns(aggregate["prompt_label_patterns"]),
        "",
        "### Methods Lab",
        _format_label_patterns(aggregate["method_label_patterns"]),
        "",
        "## Blocked Or Unavailable Models",
        _format_blocked_runs(aggregate["runs"]),
    ]
    output_path.write_text("\n".join(lines).strip() + "\n")


def _build_aggregate_summary(
    *,
    plan: GroundTruthSweepPlan,
    run_details: list[dict[str, Any]],
) -> dict[str, Any]:
    prompt_records, method_records = _cell_records(run_details)
    prompt_rankings = _aggregate_variant_rankings(prompt_records)
    method_rankings = _aggregate_variant_rankings(method_records)
    provider_summary = _aggregate_by_group([*prompt_records, *method_records], field="provider")
    top_configurations, bottom_configurations = _top_bottom_configurations(prompt_records, method_records)
    aggregate = {
        "summary": {
            "export_root": str(plan.export_root),
            "session_id": plan.session_id,
            "doc_count": len(plan.doc_ids),
            "run_count": len(run_details),
            "completed_run_count": sum(
                1 for detail in run_details if str(detail.get("status")) == "completed"
            ),
            "blocked_run_count": sum(
                1 for detail in run_details if str(detail.get("status")) == "blocked"
            ),
            "failed_run_count": sum(
                1
                for detail in run_details
                if str(detail.get("status")) not in {"completed", "blocked"}
            ),
        },
        "runs": [
            {
                "name": detail.get("name"),
                "id": detail.get("id"),
                "kind": detail.get("kind"),
                "manifest_name": detail.get("manifest_name"),
                "status": detail.get("status"),
                "failed_tasks": detail.get("failed_tasks", 0),
                "errors": detail.get("errors", []),
                "models": detail.get("models", []),
            }
            for detail in run_details
        ],
        "prompt_rankings": prompt_rankings,
        "method_rankings": method_rankings,
        "prompt_per_model_rankings": _aggregate_per_model(prompt_records),
        "method_per_model_rankings": _aggregate_per_model(method_records),
        "prompt_verifier_comparisons": _verifier_comparisons(prompt_records),
        "method_verifier_comparisons": _verifier_comparisons(method_records),
        "model_family_comparisons": _aggregate_model_family_comparisons(
            [*prompt_records, *method_records]
        ),
        "provider_summary": provider_summary,
        "top_configurations": top_configurations,
        "bottom_configurations": bottom_configurations,
        "prompt_label_patterns": _aggregate_label_patterns(prompt_records),
        "method_label_patterns": _aggregate_label_patterns(method_records),
        "prompt_records": prompt_records,
        "method_records": method_records,
    }
    return aggregate


def run_ground_truth_sweep(
    *,
    session_id: str = "default",
    export_root: Path | None = None,
    api_key: str | None = None,
    api_base: str | None = None,
) -> dict[str, Any]:
    plan = build_ground_truth_sweep_plan(session_id=session_id, export_root=export_root)
    plan.export_root.mkdir(parents=True, exist_ok=True)
    (plan.export_root / "reports" / "runs").mkdir(parents=True, exist_ok=True)
    (plan.export_root / "reports" / "aggregates").mkdir(parents=True, exist_ok=True)

    run_details: list[dict[str, Any]] = []
    for spec in plan.runs:
        manifest_payload = _run_manifest_payload(spec, session_id=session_id, api_base=api_base)
        _write_manifest(spec.manifest_path, manifest_payload)
        run_details.append(
            _execute_run(spec, session_id=session_id, api_key=api_key, api_base=api_base)
        )

    aggregate = _build_aggregate_summary(plan=plan, run_details=run_details)
    aggregates_dir = plan.export_root / "reports" / "aggregates"
    _write_aggregate_csv(
        aggregates_dir / "prompt_lab_all_cells.csv",
        kind="prompt_lab",
        rows=aggregate["prompt_records"],
    )
    _write_aggregate_csv(
        aggregates_dir / "methods_lab_all_cells.csv",
        kind="methods_lab",
        rows=aggregate["method_records"],
    )
    (aggregates_dir / "all_runs_summary.json").write_text(json.dumps(_json_safe(aggregate), indent=2))
    _generate_report(aggregate, output_path=aggregates_dir / "final_report.md")
    return aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the ground-truth model sweep.")
    parser.add_argument("--session", default="default")
    parser.add_argument("--export-root")
    parser.add_argument("--api-key")
    parser.add_argument("--api-base")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    export_root = Path(args.export_root).resolve() if args.export_root else None
    summary = run_ground_truth_sweep(
        session_id=args.session,
        export_root=export_root,
        api_key=args.api_key,
        api_base=args.api_base,
    )
    print(json.dumps(summary["summary"], indent=2))
    print(
        f"Final report: {(Path(summary['summary']['export_root']) / 'reports' / 'aggregates' / 'final_report.md')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
