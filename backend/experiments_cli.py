from __future__ import annotations

import argparse
import csv
import json
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
    _load_session_index,
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


def _load_manifest(path_value: str) -> tuple[str, str, Any, Path]:
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
    body_payload = {key: value for key, value in data.items() if key not in {"kind", "session"}}
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
    return kind, session_id, body, manifest_path.parent


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


def run_manifest(args: argparse.Namespace) -> int:
    kind, session_id, body, context_dir = _load_manifest(args.manifest)
    if kind == "prompt_lab":
        detail = create_prompt_lab_run(
            body,
            session_id=session_id,
            context_dir=context_dir,
            run_async=False,
        )
    else:
        detail = create_methods_lab_run(body, session_id=session_id, run_async=False)
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
    detail = create_methods_lab_run(body, session_id=args.session, run_async=False)
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
