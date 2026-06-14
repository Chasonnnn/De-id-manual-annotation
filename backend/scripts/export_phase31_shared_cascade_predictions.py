from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime
import gc
import importlib
import importlib.util
import json
import math
from pathlib import Path
import re
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a top-level JSON object")
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one candidate export, then one or more Phase31 cascade reviewers."
    )
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args(argv)


def _load_contextshift_cascade_module(repo_root: Path) -> Any:
    script_path = repo_root / "scripts" / "export_phase31_cascade_predictions.py"
    if not script_path.is_file():
        raise FileNotFoundError(f"Missing contextshift cascade script: {script_path}")
    spec = importlib.util.spec_from_file_location(
        "_contextshift_phase31_cascade_predictions",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _safe_file_stem(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return safe or "method"


def _write_progress(
    *,
    progress_path: Path | None,
    stage: str,
    method_id: str | None = None,
    completed_methods: int,
    total_methods: int,
    **extra: Any,
) -> None:
    if progress_path is None:
        return
    payload: dict[str, Any] = {
        "stage": stage,
        "method_id": method_id,
        "completed_methods": completed_methods,
        "total_methods": total_methods,
        "updated_at": datetime.now().astimezone().isoformat(),
    }
    payload.update(extra)
    temp_path = progress_path.with_suffix(progress_path.suffix + ".tmp")
    temp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temp_path.replace(progress_path)


def _candidate_args(config: Mapping[str, Any], module: Any) -> SimpleNamespace:
    return SimpleNamespace(
        workspace_root=Path(str(config["workspace_root"])).expanduser().resolve(),
        action_workspace_root=Path(str(config["action_workspace_root"])).expanduser().resolve(),
        input_dir=Path(str(config["input_dir"])).expanduser().resolve(),
        output_dir=Path(str(config["output_dir"])).expanduser().resolve(),
        subject=str(config.get("subject") or "math"),
        context_profile=str(config.get("context_profile") or "neighbor_v1"),
        id_field=str(config.get("id_field") or "id"),
        transcript_field=str(config.get("transcript_field") or "transcript"),
        freeze_manifest=Path(
            str(config.get("freeze_manifest") or module.DEFAULT_FREEZE_MANIFEST)
        ).expanduser().resolve(),
        uv_bin=str(config["uv_bin"]),
        model_slot=list(config.get("model_slots") or module.DEFAULT_OPERATIONAL_UNION_MODEL_SLOTS),
        span_field="candidate_pii_occurrences",
        device=str(config.get("device") or "auto"),
        batch_size=config.get("batch_size"),
        max_length=config.get("max_length"),
    )


def _review_rows_with_progress(
    *,
    module: Any,
    reviewer_rows: Sequence[Mapping[str, Any]],
    model: Any,
    processor: Any,
    config: Mapping[str, Any],
    backend: str,
    label_format: str,
    direct_address_guard: bool,
    extra_system_rule_text: str,
    model_id: str,
    batch_size: int,
    max_generation_tokens: int,
    checkpoint_path: Path,
    progress_path: Path | None,
    method_id: str,
    completed_methods: int,
    total_methods: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import mlx.core as mx

    reviewer_runtime = importlib.import_module("contextshift_deid.phase31_mlx_reviewer")
    if backend == "mlx-vlm":
        from mlx_vlm.generate import batch_generate as _backend_batch_generate
    elif backend == "mlx-lm":
        from mlx_lm import batch_generate as _backend_batch_generate
    else:
        raise ValueError(f"Unsupported reviewer backend: {backend!r}")

    profile = reviewer_runtime.build_phase31_reviewer_profile(
        model_id=model_id,
        label_format=label_format,
        direct_address_guard=direct_address_guard,
        extra_system_rule_text=extra_system_rule_text,
        requested_max_tokens=max_generation_tokens,
    )
    label_token_ids = None
    if label_format == "letters":
        label_token_ids = reviewer_runtime.phase31_single_token_ids(
            processor,
            [profile.redact_label, profile.keep_label],
        )
    effective_batch_size = 1 if backend == "mlx-vlm" else max(1, int(batch_size))

    runtime_config = dict(config or {})
    predictions: list[dict[str, Any] | None] = [None] * len(reviewer_rows)
    generation_items: list[tuple[int, Mapping[str, Any], Any]] = []
    direct_id_skipped = 0
    for index, row in enumerate(reviewer_rows):
        eval_row = reviewer_runtime.phase31_prompt_version_row(row, prompt_version="surface")
        metadata = dict(row.get("metadata") or {})
        if str(metadata.get("pool_source") or "") == reviewer_runtime.PHASE31_DIRECT_ID_POOL_SOURCE:
            predictions[index] = reviewer_runtime.phase31_prediction_row(
                eval_row,
                raw_output="REDACT",
                predicted_action="REDACT",
                parse_failure=False,
                override_reason="direct_id_rule",
            )
            direct_id_skipped += 1
            continue
        prompt = reviewer_runtime.phase31_format_prompt(
            processor,
            runtime_config,
            eval_row,
            profile,
            prompt_format="plain",
            prompt_version="surface",
        )
        payload: Any
        if backend == "mlx-lm":
            payload = list(processor.encode(prompt, add_special_tokens=True))
        else:
            payload = prompt
        generation_items.append((index, eval_row, payload))

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if direct_id_skipped:
        module.write_jsonl(checkpoint_path, [row for row in predictions if row is not None])

    total_prompt_tokens = 0
    total_generation_tokens = 0
    peak_memory_gb = 0.0
    start_time = perf_counter()
    total_batches = math.ceil(len(generation_items) / effective_batch_size) if generation_items else 0
    completed_generation_rows = 0

    _write_progress(
        progress_path=progress_path,
        stage="review",
        method_id=method_id,
        completed_methods=completed_methods,
        total_methods=total_methods,
        reviewer_rows=len(reviewer_rows),
        reviewer_batch_size=effective_batch_size,
        review_batch_index=0,
        review_batch_count=total_batches,
        completed_reviewer_rows=direct_id_skipped,
        total_reviewer_rows=len(reviewer_rows),
        generated_reviewer_rows=0,
        direct_id_skipped_row_count=direct_id_skipped,
        checkpoint_file=str(checkpoint_path),
    )

    for batch_index, start in enumerate(
        range(0, len(generation_items), effective_batch_size),
        start=1,
    ):
        batch = generation_items[start : start + effective_batch_size]
        logits_processors = (
            [reviewer_runtime.Phase31AllowedLabelTokenProcessor(label_token_ids, keep_penalty=0.0)]
            if label_token_ids
            else None
        )
        if backend == "mlx-vlm":
            response = _backend_batch_generate(
                model,
                processor,
                images=None,
                prompts=[item[2] for item in batch],
                max_tokens=profile.max_tokens,
                verbose=False,
                logits_processors=logits_processors,
            )
        else:
            response = _backend_batch_generate(
                model,
                processor,
                [item[2] for item in batch],
                max_tokens=profile.max_tokens,
                verbose=False,
                logits_processors=logits_processors,
            )
        stats = getattr(response, "stats", None)
        total_prompt_tokens += int(getattr(stats, "prompt_tokens", 0) or 0)
        total_generation_tokens += int(getattr(stats, "generation_tokens", 0) or 0)
        peak_memory_gb = max(peak_memory_gb, float(getattr(stats, "peak_memory", 0.0) or 0.0))
        for (row_index, row, _payload), raw_output in zip(batch, list(response.texts)):
            normalized = reviewer_runtime.normalize_phase31_output_label(
                raw_output,
                output_map=profile.output_map,
            )
            predictions[row_index] = reviewer_runtime.phase31_prediction_row(
                row,
                raw_output=str(raw_output),
                predicted_action=normalized or "REDACT",
                parse_failure=normalized is None,
                override_reason=None,
            )
        completed_generation_rows += len(batch)
        module.write_jsonl(checkpoint_path, [row for row in predictions if row is not None])
        _write_progress(
            progress_path=progress_path,
            stage="review",
            method_id=method_id,
            completed_methods=completed_methods,
            total_methods=total_methods,
            reviewer_rows=len(reviewer_rows),
            reviewer_batch_size=effective_batch_size,
            review_batch_index=batch_index,
            review_batch_count=total_batches,
            completed_reviewer_rows=direct_id_skipped + completed_generation_rows,
            total_reviewer_rows=len(reviewer_rows),
            generated_reviewer_rows=completed_generation_rows,
            direct_id_skipped_row_count=direct_id_skipped,
            prompt_tokens=total_prompt_tokens,
            generation_tokens=total_generation_tokens,
            peak_memory_gb=peak_memory_gb,
            elapsed_seconds=perf_counter() - start_time,
            checkpoint_file=str(checkpoint_path),
        )
        del response
        gc.collect()
        mx.clear_cache()

    final_predictions = reviewer_runtime.phase31_apply_direct_id_override(
        [row for row in predictions if row is not None]
    )
    elapsed_seconds = perf_counter() - start_time
    module.write_jsonl(checkpoint_path, final_predictions)
    return final_predictions, {
        "backend": backend,
        "label_format": label_format,
        "direct_address_guard": bool(direct_address_guard),
        "elapsed_seconds": elapsed_seconds,
        "reviewed_row_count": len(final_predictions),
        "generated_row_count": len(generation_items),
        "direct_id_skipped_row_count": direct_id_skipped,
        "effective_batch_size": effective_batch_size,
        "prompt_tokens": total_prompt_tokens,
        "generation_tokens": total_generation_tokens,
        "peak_memory_gb": peak_memory_gb,
    }


def _run(config: Mapping[str, Any]) -> dict[str, Any]:
    repo_root = Path(str(config["repo_root"])).expanduser().resolve()
    module = _load_contextshift_cascade_module(repo_root)
    input_dir = Path(str(config["input_dir"])).expanduser().resolve()
    output_dir = Path(str(config["output_dir"])).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = (
        Path(str(config["progress_path"])).expanduser().resolve()
        if config.get("progress_path")
        else None
    )
    methods = [dict(method) for method in list(config.get("methods") or [])]
    if not methods:
        raise ValueError("At least one cascade method is required.")

    total_methods = len(methods)
    action_workspace = module.build_workspace_paths(
        Path(str(config["action_workspace_root"])).expanduser().resolve()
    )
    experiment = module.create_experiment_run(
        str(config.get("run_name") or "phase31-shared-cascade-transcript-export"),
        root_dir=action_workspace.experiments_dir,
    )

    _write_progress(
        progress_path=progress_path,
        stage="candidate_export",
        completed_methods=0,
        total_methods=total_methods,
    )
    args = _candidate_args(config, module)
    candidate_payload, candidate_command = module._run_candidate_export(
        args=args,
        candidate_output_dir=experiment.root / "candidate_exports",
    )
    candidate_summary_path = Path(str(candidate_payload["summary_json"])).expanduser().resolve()
    candidate_rows, union_rows, candidate_summary = module._load_candidate_union(
        candidate_summary_path
    )
    predictions_by_id = module.prediction_map(union_rows)
    projections, dataset_summary = module.build_transcript_projections_from_dir(
        input_dir,
        subject=str(config.get("subject") or "math"),
        context_profile=str(config.get("context_profile") or "neighbor_v1"),
        id_field=str(config.get("id_field") or "id"),
        transcript_field=str(config.get("transcript_field") or "transcript"),
    )
    _write_progress(
        progress_path=progress_path,
        stage="candidate_export_completed",
        completed_methods=0,
        total_methods=total_methods,
        candidate_rows=len(candidate_rows),
        union_rows=len(union_rows),
        transcript_count=len(projections),
    )

    method_summaries: list[dict[str, Any]] = []
    for method_index, method in enumerate(methods):
        method_id = str(method["method_id"])
        output_slot = str(method["output_slot"])
        reviewer = dict(method["reviewer"])
        reviewer_model_path = Path(str(reviewer["model_path"])).expanduser().resolve()
        reviewer_adapter_path = Path(str(reviewer["adapter_path"])).expanduser().resolve()
        rule_file = (
            Path(str(reviewer["rule_file"])).expanduser().resolve()
            if reviewer.get("rule_file")
            else None
        )
        method_stem = _safe_file_stem(method_id)

        _write_progress(
            progress_path=progress_path,
            stage="load_reviewer",
            method_id=method_id,
            completed_methods=method_index,
            total_methods=total_methods,
        )
        artifact_files = module.validate_reviewer_artifact(
            model_path=reviewer_model_path,
            adapter_path=reviewer_adapter_path,
        )
        model, processor, runtime_config = module._load_mlx_runtime(
            backend=str(reviewer["backend"]),
            model_path=reviewer_model_path,
            adapter_path=reviewer_adapter_path,
        )
        module.phase31_raise_metal_wired_limit()
        prompt_tokenizer = getattr(processor, "tokenizer", processor)
        reviewer_items = module.build_cascade_reviewer_items(
            candidate_rows=candidate_rows,
            predictions_by_id=predictions_by_id,
            tokenizer=prompt_tokenizer,
            candidate_model_slot=module.DEFAULT_CASCADE_SOURCE_SLOT,
            candidate_model_label=module.DEFAULT_CASCADE_SOURCE_LABEL,
            prompt_variant=str(reviewer["prompt_variant"]),
            max_prompt_tokens=int(
                config.get("max_prompt_tokens") or module.DEFAULT_PHASE31_MAX_PROMPT_TOKENS
            ),
        )
        reviewer_rows = [item.reviewer_row for item in reviewer_items]
        reviewer_rows_path = experiment.root / f"{method_stem}_reviewer_rows.jsonl"
        module.write_jsonl(reviewer_rows_path, reviewer_rows)

        _write_progress(
            progress_path=progress_path,
            stage="review",
            method_id=method_id,
            completed_methods=method_index,
            total_methods=total_methods,
            reviewer_rows=len(reviewer_rows),
            reviewer_batch_size=int(reviewer["batch_size"]),
        )
        reviewer_predictions_path = (
            experiment.predictions_dir / f"{method_stem}_reviewer_predictions.jsonl"
        )
        reviewer_prediction_rows, reviewer_generation = _review_rows_with_progress(
            module=module,
            reviewer_rows=reviewer_rows,
            model=model,
            processor=processor,
            config=runtime_config,
            backend=str(reviewer["backend"]),
            label_format=str(reviewer["label_format"]),
            direct_address_guard=bool(reviewer["direct_address_guard"]),
            extra_system_rule_text=module.phase31_read_extra_system_rule(rule_file),
            model_id=str(reviewer["model_id"]),
            batch_size=int(reviewer["batch_size"]),
            max_generation_tokens=int(
                config.get("max_generation_tokens")
                or module.DEFAULT_PHASE31_MAX_GENERATION_TOKENS
            ),
            checkpoint_path=reviewer_predictions_path,
            progress_path=progress_path,
            method_id=method_id,
            completed_methods=method_index,
            total_methods=total_methods,
        )

        final_output_dir = output_dir / output_slot
        final_output_dir.mkdir(parents=True, exist_ok=True)
        items_by_transcript: dict[str, list[Any]] = defaultdict(list)
        for item in reviewer_items:
            items_by_transcript[str(item.reviewer_row["dialogue_id"])].append(item)
        reviewer_by_id = {str(row["id"]): row for row in reviewer_prediction_rows}

        transcript_summaries: list[dict[str, Any]] = []
        for projection in projections:
            projection_items = items_by_transcript.get(projection.transcript_id, [])
            projection_reviewer_rows = [
                reviewer_by_id[str(item.reviewer_row["id"])] for item in projection_items
            ]
            exported_payload, transcript_summary = module.overlay_reviewer_decisions_onto_payload(
                projection.payload,
                items=projection_items,
                reviewer_rows=projection_reviewer_rows,
                span_field=str(config.get("span_field") or "pii_occurrences"),
                reviewer_artifact_id=str(reviewer["artifact_id"]),
                reviewer_model_path=reviewer_model_path,
                reviewer_adapter_path=reviewer_adapter_path,
                reviewer_prompt_variant=str(reviewer["prompt_variant"]),
                candidate_model_slot=module.DEFAULT_CASCADE_SOURCE_SLOT,
                candidate_model_label=module.DEFAULT_CASCADE_SOURCE_LABEL,
            )
            transcript_summaries.append(
                {
                    "source_file": projection.input_path.name,
                    "transcript_id": projection.transcript_id,
                    **transcript_summary,
                }
            )
            module.write_json(final_output_dir / projection.input_path.name, exported_payload)

        method_summary = {
            "method_id": method_id,
            "output_slot": output_slot,
            "reviewer": {
                "artifact_id": str(reviewer["artifact_id"]),
                "model_id": str(reviewer["model_id"]),
                "model_path": str(reviewer_model_path),
                "adapter_path": str(reviewer_adapter_path),
                "prompt_variant": str(reviewer["prompt_variant"]),
                "backend": str(reviewer["backend"]),
                "label_format": str(reviewer["label_format"]),
                "direct_address_guard": bool(reviewer["direct_address_guard"]),
                "extra_system_rule_file": str(rule_file) if rule_file is not None else None,
                "artifact_files": artifact_files,
                "rows_file": str(reviewer_rows_path),
                "predictions_file": str(reviewer_predictions_path),
                "generation": reviewer_generation,
            },
            "transcripts": transcript_summaries,
            "candidate_span_count": sum(
                int(item["candidate_span_count"]) for item in transcript_summaries
            ),
            "reviewed_span_count": sum(
                int(item["reviewed_span_count"]) for item in transcript_summaries
            ),
            "redacted_span_count": sum(
                int(item["redacted_span_count"]) for item in transcript_summaries
            ),
            "parse_failure_count": sum(
                int(item["parse_failure_count"]) for item in transcript_summaries
            ),
        }
        method_summaries.append(method_summary)
        _write_progress(
            progress_path=progress_path,
            stage="method_completed",
            method_id=method_id,
            completed_methods=method_index + 1,
            total_methods=total_methods,
            effective_batch_size=reviewer_generation.get("effective_batch_size"),
            elapsed_seconds=reviewer_generation.get("elapsed_seconds"),
            peak_memory_gb=reviewer_generation.get("peak_memory_gb"),
        )
        del model, processor, runtime_config
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass

    summary = {
        "status": "completed",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "span_field": str(config.get("span_field") or "pii_occurrences"),
        "dataset": dataset_summary,
        "candidate_export": {
            "summary_json": str(candidate_summary_path),
            "candidate_rows_file": str(candidate_summary["candidate_rows_file"]),
            "command": candidate_command,
        },
        "methods": method_summaries,
    }
    experiment.summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    experiment.report_path.write_text(
        "\n".join(
            [
                "# Phase 31 Shared Cascade Transcript Export",
                "",
                f"- Input dir: `{input_dir}`",
                f"- Export root: `{output_dir}`",
                f"- Methods: `{len(method_summaries)}`",
                f"- Candidate rows: `{len(candidate_rows)}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    module.write_run_metadata(
        experiment.metadata_path,
        {
            "workspace_root": str(action_workspace.root),
            "candidate_workspace_root": str(
                Path(str(config["workspace_root"])).expanduser().resolve()
            ),
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "created_at": datetime.now().astimezone().isoformat(),
        },
    )
    _write_progress(
        progress_path=progress_path,
        stage="completed",
        completed_methods=total_methods,
        total_methods=total_methods,
    )
    return {
        "summary_json": str(experiment.summary_path),
        "report_md": str(experiment.report_path),
        "export_root": str(output_dir),
        "methods": [
            {
                "method_id": str(item["method_id"]),
                "output_slot": str(item["output_slot"]),
            }
            for item in method_summaries
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    config = _load_json(args.config)
    progress_path = (
        Path(str(config["progress_path"])).expanduser().resolve()
        if config.get("progress_path")
        else None
    )
    try:
        payload = _run(config)
    except BaseException as exc:
        _write_progress(
            progress_path=progress_path,
            stage="failed",
            completed_methods=0,
            total_methods=len(list(config.get("methods") or [])),
            error=str(exc),
        )
        raise
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
