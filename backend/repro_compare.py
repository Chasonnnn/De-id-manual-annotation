from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from dataclasses import asdict, is_dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable
from urllib import request

from experiment_service import create_methods_lab_run
from metrics import compute_metrics
from models import CanonicalDocument, CanonicalSpan
from server import (
    MethodsLabMethodInput,
    MethodsLabRunCreateBody,
    MethodsLabRuntimeInput,
    PromptLabModelInput,
    _enrich_doc,
    _load_doc,
    _load_methods_lab_run,
    _load_session_index,
)

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
OUTPUT_ROOT = REPO_ROOT / "output"
DEFAULT_COLLEAGUE_REPO = Path("/Users/chason/deidentify")
DEFAULT_SESSION_ID = "default"
DEFAULT_METHODS_API_URL = "http://localhost:8001/api"
DEFAULT_MODEL = "openai.gpt-5.2-chat"
DEFAULT_MATCH_MODE = "overlap"
DEFAULT_CHUNK_MODE = "off"
DEFAULT_CHUNK_SIZE_CHARS = 10_000
DEFAULT_CURRENT_BUNDLES = ("deidentify-v2", "v2+post-process")
COLLEAGUE_EXPERIMENTS = (
    "dual-v2",
    "regex+dual-v2",
    "presidio-lite+extended-v2",
)
CURRENT_COMPAT_METHODS = COLLEAGUE_EXPERIMENTS
CURRENT_DRIFT_METHOD_PAIRS = (
    ("dual-v2", "dual"),
    ("regex+dual-v2", "presidio+llm-split"),
    ("presidio-lite+extended-v2", "presidio+default"),
)


def _now_slug() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _load_local_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _resolve_gateway_config(
    *,
    api_key: str | None = None,
    api_base: str | None = None,
) -> tuple[str, str | None]:
    _load_local_env(REPO_ROOT / ".env.local")
    resolved_api_key = (
        str(api_key or "").strip()
        or os.getenv("LITELLM_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
        or os.getenv("ANTHROPIC_API_KEY", "").strip()
    )
    resolved_api_base = (
        str(api_base or "").strip()
        or os.getenv("LITELLM_BASE_URL", "").strip()
        or None
    )
    if not resolved_api_key:
        raise SystemExit(
            "Missing API key. Set LITELLM_API_KEY/OPENAI_API_KEY/ANTHROPIC_API_KEY or pass --api-key."
        )
    return resolved_api_key, resolved_api_base


def _resolve_output_dir(path_value: str | None, *, prefix: str) -> Path:
    if path_value:
        output_dir = Path(path_value).expanduser().resolve()
    else:
        output_dir = OUTPUT_ROOT / f"{prefix}-{_now_slug()}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _git_commit(path: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return proc.stdout.strip() or None


def _doc_signature(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_colleague_model_name(model: str) -> str:
    value = str(model).strip()
    if not value or "/" in value:
        return value
    if value.startswith("openai."):
        return f"openai/{value}"
    return value


def load_manual_docs(session_id: str = DEFAULT_SESSION_ID) -> list[CanonicalDocument]:
    docs: list[CanonicalDocument] = []
    for doc_id in _load_session_index(session_id):
        doc = _load_doc(doc_id, session_id)
        if doc is None:
            continue
        enriched = _enrich_doc(doc, session_id)
        if enriched.manual_annotations:
            docs.append(enriched)
    docs.sort(key=lambda item: str(item.id))
    return docs


def _overlapping_utterance_indices(doc: CanonicalDocument, span: CanonicalSpan) -> list[int]:
    return [
        index
        for index, row in enumerate(doc.utterances)
        if max(row.global_start, span.start) < min(row.global_end, span.end)
    ]


def _build_transcript_segments(doc: CanonicalDocument) -> list[tuple[int, int]]:
    utterance_count = len(doc.utterances)
    if utterance_count == 0:
        return []
    safe_split_boundaries = set(range(utterance_count - 1))
    for span in doc.manual_annotations:
        overlapping = _overlapping_utterance_indices(doc, span)
        if not overlapping:
            raise ValueError(
                f"Manual span {span.start}:{span.end} ({span.label}) in doc {doc.id} is outside utterance text."
            )
        for boundary in range(overlapping[0], overlapping[-1]):
            safe_split_boundaries.discard(boundary)

    segments: list[tuple[int, int]] = []
    start_index = 0
    for boundary in range(utterance_count - 1):
        if boundary in safe_split_boundaries:
            segments.append((start_index, boundary))
            start_index = boundary + 1
    segments.append((start_index, utterance_count - 1))
    return segments


def build_colleague_dataset_records(docs: list[CanonicalDocument]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for doc in docs:
        segment_ranges = _build_transcript_segments(doc)
        if not segment_ranges:
            records.append({"session_id": doc.id, "filename": doc.filename, "transcript": []})
            continue

        annotations_by_segment: dict[int, list[dict[str, Any]]] = defaultdict(list)
        segment_bounds: list[tuple[int, int]] = []
        segment_roles: list[str] = []
        segment_content: list[str] = []

        for start_index, end_index in segment_ranges:
            first_row = doc.utterances[start_index]
            last_row = doc.utterances[end_index]
            segment_start = first_row.global_start
            segment_end = last_row.global_end
            segment_bounds.append((segment_start, segment_end))
            segment_roles.append(first_row.speaker)
            segment_content.append(doc.raw_text[segment_start:segment_end])

        for span in doc.manual_annotations:
            for segment_index, (segment_start, segment_end) in enumerate(segment_bounds):
                if segment_start <= span.start and span.end <= segment_end:
                    annotations_by_segment[segment_index].append(
                        {
                            "start": span.start - segment_start,
                            "end": span.end - segment_start,
                            "pii_type": span.label,
                        }
                    )
                    break
            else:
                raise ValueError(
                    f"Manual span {span.start}:{span.end} ({span.label}) in doc {doc.id} does not fit any transcript segment."
                )

        transcript: list[dict[str, Any]] = []
        for index, (segment_start, segment_end) in enumerate(segment_bounds):
            transcript.append(
                {
                    "role": segment_roles[index],
                    "content": segment_content[index],
                    "annotations": annotations_by_segment.get(index, []),
                    "_global_start": segment_start,
                    "_global_end": segment_end,
                }
            )

        records.append(
            {
                "session_id": doc.id,
                "filename": doc.filename,
                "transcript": transcript,
            }
        )
    return records


def write_colleague_dataset(records: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(record) for record in records) + "\n"
    output_path.write_text(payload)


def _transcript_text_from_record(record: dict[str, Any]) -> str:
    transcript = record.get("transcript", [])
    if not isinstance(transcript, list):
        raise ValueError("Dataset record missing transcript list.")
    return "\n".join(str(turn.get("content", "") or "") for turn in transcript if isinstance(turn, dict))


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line:
            continue
        item = json.loads(line)
        if not isinstance(item, dict):
            raise ValueError(f"JSONL record must decode to an object: {line[:80]}")
        records.append(item)
    return records


def select_dataset_records_for_current_docs(
    *,
    current_docs: list[CanonicalDocument],
    dataset_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    records_by_signature: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in dataset_records:
        records_by_signature[_doc_signature(_transcript_text_from_record(record))].append(record)

    selected: list[dict[str, Any]] = []
    for doc in current_docs:
        signature = _doc_signature(doc.raw_text)
        if not records_by_signature.get(signature):
            raise ValueError(
                f"Could not find an exact JSONL transcript match for current doc {doc.id} ({doc.filename})."
            )
        selected.append(records_by_signature[signature].pop(0))
    return selected


def _load_colleague_demo_module(repo_root: Path = DEFAULT_COLLEAGUE_REPO):
    module_path = repo_root / "examples" / "demo" / "app.py"
    src_path = repo_root / "src"
    if not module_path.exists():
        raise FileNotFoundError(f"Colleague demo app not found: {module_path}")
    module_name = f"colleague_demo_app_{hash(str(module_path)) & 0xFFFFFFFF:x}"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    inserted_paths = [str(module_path.parent), str(repo_root)]
    if src_path.exists():
        inserted_paths.insert(0, str(src_path))
    for inserted_path in reversed(inserted_paths):
        sys.path.insert(0, inserted_path)
    try:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        for inserted_path in inserted_paths:
            try:
                sys.path.remove(inserted_path)
            except ValueError:
                pass
    return module


def _to_canonical_span(raw: Any) -> CanonicalSpan:
    if isinstance(raw, CanonicalSpan):
        return raw
    if isinstance(raw, dict):
        if {"start", "end", "label", "text"} <= set(raw.keys()):
            return CanonicalSpan.model_validate(raw)
        if {"start", "end", "entity_type", "text"} <= set(raw.keys()):
            return CanonicalSpan(
                start=int(raw["start"]),
                end=int(raw["end"]),
                label=str(raw["entity_type"]),
                text=str(raw["text"]),
            )
    return CanonicalSpan(
        start=int(getattr(raw, "start")),
        end=int(getattr(raw, "end")),
        label=str(getattr(raw, "label", getattr(raw, "entity_type"))),
        text=str(getattr(raw, "text")),
    )


def _dedup_canonical_spans(spans: Iterable[CanonicalSpan]) -> list[CanonicalSpan]:
    seen: set[tuple[int, int, str, str]] = set()
    deduped: list[CanonicalSpan] = []
    for span in sorted(spans, key=lambda item: (item.start, item.end, item.label, item.text)):
        key = (span.start, span.end, span.label, span.text)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(span)
    return deduped


def project_colleague_predictions_to_current_docs(
    *,
    current_docs: list[CanonicalDocument],
    predicted: dict[int, list[Any]],
    transcript_groups: list[list[int]],
    dataset_records: list[dict[str, Any]] | None = None,
) -> dict[str, list[CanonicalSpan]]:
    projected: dict[str, list[CanonicalSpan]] = {}
    if len(current_docs) != len(transcript_groups):
        raise ValueError(
            f"Current docs ({len(current_docs)}) and transcript groups ({len(transcript_groups)}) must align."
        )
    if dataset_records is not None and len(dataset_records) != len(current_docs):
        raise ValueError(
            f"Dataset records ({len(dataset_records)}) and current docs ({len(current_docs)}) must align."
        )

    for index, (current_doc, group) in enumerate(zip(current_docs, transcript_groups, strict=True)):
        spans: list[CanonicalSpan] = []
        if dataset_records is not None:
            transcript = dataset_records[index].get("transcript", [])
            if not isinstance(transcript, list):
                raise ValueError(f"Dataset record for doc {current_doc.id} is missing transcript list.")
            if len(transcript) != len(group):
                raise ValueError(
                    f"Dataset record for doc {current_doc.id} has {len(transcript)} messages but "
                    f"colleague transcript group has {len(group)} messages."
                )
            offset_entries = _resolve_dataset_record_offsets(
                current_doc=current_doc,
                transcript=transcript,
            )
        else:
            if len(current_doc.utterances) != len(group):
                raise ValueError(
                    f"Current doc {current_doc.id} has {len(current_doc.utterances)} utterances but "
                    f"colleague transcript group has {len(group)} messages."
                )
            offset_entries = [
                (utterance.global_start, utterance.global_end) for utterance in current_doc.utterances
            ]

        for (segment_start, _segment_end), colleague_doc_id in zip(offset_entries, group, strict=True):
            for raw_span in predicted.get(colleague_doc_id, []):
                span = _to_canonical_span(raw_span)
                spans.append(
                    CanonicalSpan(
                        start=segment_start + span.start,
                        end=segment_start + span.end,
                        label=span.label,
                        text=span.text,
                    )
                )
        projected[current_doc.id] = _dedup_canonical_spans(spans)
    return projected


def _resolve_dataset_record_offsets(
    *,
    current_doc: CanonicalDocument,
    transcript: list[Any],
) -> list[tuple[int, int]]:
    if transcript and all(
        isinstance(turn, dict)
        and isinstance(turn.get("_global_start"), int)
        and isinstance(turn.get("_global_end"), int)
        for turn in transcript
    ):
        return [
            (int(turn["_global_start"]), int(turn["_global_end"]))  # type: ignore[index]
            for turn in transcript
            if isinstance(turn, dict)
        ]

    offset_entries: list[tuple[int, int]] = []
    cursor = 0
    for turn in transcript:
        if not isinstance(turn, dict):
            raise ValueError(f"Dataset record for doc {current_doc.id} contains a non-object message.")
        content = str(turn.get("content", "") or "")
        start = cursor
        end = start + len(content)
        if current_doc.raw_text[start:end] != content:
            raise ValueError(
                f"Dataset record for doc {current_doc.id} does not align with current document text."
            )
        offset_entries.append((start, end))
        cursor = end + 1
    return offset_entries


def _aggregate_metrics(per_doc: dict[str, dict[str, Any]]) -> dict[str, Any]:
    tp = sum(int(item["metrics"]["micro"]["tp"]) for item in per_doc.values())
    fp = sum(int(item["metrics"]["micro"]["fp"]) for item in per_doc.values())
    fn = sum(int(item["metrics"]["micro"]["fn"]) for item in per_doc.values())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "micro": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    }


def run_colleague_demo_baseline(
    *,
    current_docs: list[CanonicalDocument],
    records: list[dict[str, Any]] | None = None,
    model: str,
    api_base: str | None,
    repo_root: Path = DEFAULT_COLLEAGUE_REPO,
    experiments: tuple[str, ...] = COLLEAGUE_EXPERIMENTS,
    max_workers: int = 1,
    current_match_mode: str = DEFAULT_MATCH_MODE,
) -> dict[str, Any]:
    module = _load_colleague_demo_module(repo_root)
    selected_records = list(records) if records is not None else build_colleague_dataset_records(current_docs)
    transcripts = [record["transcript"] for record in selected_records]
    gold_docs, transcript_groups = module.transcripts_to_gold_docs(transcripts)
    colleague_model = _normalize_colleague_model_name(model)
    output: dict[str, Any] = {
        "kind": "colleague_demo_v2_baseline",
        "repo_root": str(repo_root),
        "repo_commit": _git_commit(repo_root),
        "model": model,
        "colleague_model": colleague_model,
        "api_base": api_base,
        "concurrency": max_workers,
        "current_match_mode": current_match_mode,
        "doc_ids": [doc.id for doc in current_docs],
        "dataset_records": [
            {
                "session_id": record.get("session_id"),
                "filename": record.get("filename"),
                "message_count": len(record.get("transcript", []))
                if isinstance(record.get("transcript"), list)
                else 0,
            }
            for record in selected_records
        ],
        "experiments": {},
    }

    for experiment in experiments:
        config = module.LLMConfig(model=colleague_model, api_base=api_base)
        pipeline = module.build_pipeline(experiment, config)
        predicted = module.analyze_chunked(
            gold_docs,
            pipeline,
            max_workers=max_workers,
            transcript_groups=transcript_groups,
        )
        all_pred_types = {
            span.entity_type for spans in predicted.values() for span in spans
        }
        all_gold_types = {
            span.entity_type for doc in gold_docs for span in doc.spans
        }
        label_mapping = module.build_label_mapping(all_pred_types, all_gold_types)
        native_eval = module.evaluate_corpus(
            predicted,
            gold_docs,
            label_mapping=label_mapping,
            substring=True,
        )
        transcript_predictions = project_colleague_predictions_to_current_docs(
            current_docs=current_docs,
            predicted=predicted,
            transcript_groups=transcript_groups,
            dataset_records=selected_records,
        )
        per_doc_results: dict[str, dict[str, Any]] = {}
        for doc in current_docs:
            hypothesis_spans = transcript_predictions[doc.id]
            per_doc_results[doc.id] = {
                "filename": doc.filename,
                "hypothesis_spans": [span.model_dump() for span in hypothesis_spans],
                "metrics": compute_metrics(
                    doc.manual_annotations,
                    hypothesis_spans,
                    mode=current_match_mode,
                ),
            }
        output["experiments"][experiment] = {
            "native_eval": _json_safe(native_eval),
            "native_label_mapping": label_mapping,
            "current_repo_metrics": {
                "aggregate": _aggregate_metrics(per_doc_results),
                "documents": per_doc_results,
            },
        }
    return output


def _methods_for_bundle(bundle: str) -> list[tuple[str, str]]:
    if bundle == "deidentify-v2":
        return [
            ("dual_v2", "dual-v2"),
            ("regex_dual_v2", "regex+dual-v2"),
            ("presidio_lite_extended_v2", "presidio-lite+extended-v2"),
        ]
    if bundle == "v2+post-process":
        return [
            ("dual", "dual"),
            ("presidio_llm_split", "presidio+llm-split"),
            ("presidio_default", "presidio+default"),
        ]
    raise ValueError(f"Unsupported bundle for reproduction workflow: {bundle}")


def build_methods_lab_run_body(
    *,
    bundle: str,
    doc_ids: list[str],
    model: str,
    api_key: str,
    api_base: str | None,
    name: str,
    match_mode: str = DEFAULT_MATCH_MODE,
    reference_source: str = "manual",
    fallback_reference_source: str = "pre",
    chunk_mode: str = DEFAULT_CHUNK_MODE,
    chunk_size_chars: int = DEFAULT_CHUNK_SIZE_CHARS,
) -> MethodsLabRunCreateBody:
    methods = [
        MethodsLabMethodInput(id=variant_id, label=method_id, method_id=method_id)
        for variant_id, method_id in _methods_for_bundle(bundle)
    ]
    runtime = MethodsLabRuntimeInput(
        api_key=api_key,
        api_base=api_base,
        temperature=0.0,
        match_mode=match_mode,  # type: ignore[arg-type]
        reference_source=reference_source,  # type: ignore[arg-type]
        fallback_reference_source=fallback_reference_source,  # type: ignore[arg-type]
        method_bundle=bundle,  # type: ignore[arg-type]
        chunk_mode=chunk_mode,  # type: ignore[arg-type]
        chunk_size_chars=chunk_size_chars,
    )
    return MethodsLabRunCreateBody(
        name=name,
        doc_ids=doc_ids,
        methods=methods,
        models=[
            PromptLabModelInput(
                id="model_1",
                label=model,
                model=model,
                reasoning_effort="none",
                anthropic_thinking=False,
                anthropic_thinking_budget_tokens=None,
            )
        ],
        runtime=runtime,
        concurrency=1,
    )


def run_current_methods_lab_sync(
    *,
    session_id: str,
    bundle: str,
    model: str,
    api_key: str,
    api_base: str | None,
    doc_ids: list[str],
    name: str,
) -> dict[str, Any]:
    body = build_methods_lab_run_body(
        bundle=bundle,
        doc_ids=doc_ids,
        model=model,
        api_key=api_key,
        api_base=api_base,
        name=name,
    )
    detail = create_methods_lab_run(
        body,
        session_id=session_id,
        run_async=False,
        method_bundle=bundle,
    )
    run = _load_methods_lab_run(str(detail["id"]), session_id)
    if run is None:
        raise RuntimeError(f"Methods Lab run disappeared after completion: {detail['id']}")
    return run


def launch_current_methods_lab_runs_via_api(
    *,
    api_url: str,
    session_id: str,
    bundles: tuple[str, ...],
    model: str,
    api_key: str,
    api_base: str | None,
    doc_ids: list[str],
) -> dict[str, Any]:
    launched: dict[str, Any] = {"session_id": session_id, "runs": {}}
    for bundle in bundles:
        body = build_methods_lab_run_body(
            bundle=bundle,
            doc_ids=doc_ids,
            model=model,
            api_key=api_key,
            api_base=api_base,
            name=f"Manual27 {bundle}",
        )
        req = request.Request(
            f"{api_url.rstrip('/')}/methods-lab/runs",
            data=json.dumps(body.model_dump()).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(req) as response:
            payload = json.loads(response.read().decode("utf-8"))
        launched["runs"][bundle] = payload
    return launched


def load_methods_lab_run_source(source: str, *, session_id: str = DEFAULT_SESSION_ID) -> dict[str, Any]:
    path = Path(source).expanduser()
    if path.exists():
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise ValueError(f"Run artifact must decode to an object: {path}")
        return payload
    run = _load_methods_lab_run(source, session_id)
    if run is None:
        raise ValueError(f"Methods Lab run not found: {source}")
    return run


def _extract_run_method_variant_id(run: dict[str, Any], method_id: str) -> str:
    for method in run.get("methods", []):
        if str(method.get("method_id")) == method_id:
            return str(method.get("id"))
    raise ValueError(f"Run does not contain method_id '{method_id}'.")


def _extract_run_model_variant_id(run: dict[str, Any], model: str | None) -> str:
    models = run.get("models", [])
    if not isinstance(models, list) or not models:
        raise ValueError("Run does not contain any models.")
    if model is None:
        if len(models) != 1:
            raise ValueError("Run contains multiple models; specify --model.")
        return str(models[0].get("id"))
    for item in models:
        if str(item.get("model")) == model:
            return str(item.get("id"))
    raise ValueError(f"Run does not contain model '{model}'.")


def extract_methods_lab_predictions(
    run: dict[str, Any],
    *,
    method_id: str,
    model: str | None = None,
    span_key: str = "hypothesis_spans",
) -> dict[str, list[CanonicalSpan]]:
    method_variant_id = _extract_run_method_variant_id(run, method_id)
    model_variant_id = _extract_run_model_variant_id(run, model)
    cell_id = f"{model_variant_id}__{method_variant_id}"
    cell = run.get("cells", {}).get(cell_id)
    if not isinstance(cell, dict):
        raise ValueError(f"Run does not contain cell '{cell_id}'.")
    extracted: dict[str, list[CanonicalSpan]] = {}
    for doc_id, doc_result in (cell.get("documents") or {}).items():
        if not isinstance(doc_result, dict):
            continue
        extracted[str(doc_id)] = [
            CanonicalSpan.model_validate(item)
            for item in doc_result.get(span_key, [])
            if isinstance(item, dict)
        ]
    return extracted


def _spans_overlap(a: CanonicalSpan, b: CanonicalSpan) -> bool:
    return max(a.start, b.start) < min(a.end, b.end)


def _span_signature(span: CanonicalSpan) -> tuple[int, int, str, str]:
    return (span.start, span.end, span.label, span.text)


def _compare_span_lists(
    baseline_spans: list[CanonicalSpan],
    candidate_spans: list[CanonicalSpan],
) -> dict[str, Any]:
    baseline_remaining = list(sorted(baseline_spans, key=_span_signature))
    candidate_remaining = list(sorted(candidate_spans, key=_span_signature))
    result = {
        "all_equal": False,
        "exact_match": 0,
        "same_boundary_diff_label": 0,
        "boundary_diff_same_label": 0,
        "overlap_diff_label": 0,
        "only_baseline": 0,
        "only_candidate": 0,
    }
    if [_span_signature(span) for span in baseline_remaining] == [
        _span_signature(span) for span in candidate_remaining
    ]:
        result["all_equal"] = True
        result["exact_match"] = len(baseline_remaining)
        return result

    matched_baseline: set[int] = set()
    matched_candidate: set[int] = set()

    for b_idx, baseline_span in enumerate(baseline_remaining):
        for c_idx, candidate_span in enumerate(candidate_remaining):
            if c_idx in matched_candidate:
                continue
            if _span_signature(baseline_span) == _span_signature(candidate_span):
                matched_baseline.add(b_idx)
                matched_candidate.add(c_idx)
                result["exact_match"] += 1
                break

    for b_idx, baseline_span in enumerate(baseline_remaining):
        if b_idx in matched_baseline:
            continue
        for c_idx, candidate_span in enumerate(candidate_remaining):
            if c_idx in matched_candidate:
                continue
            if baseline_span.start == candidate_span.start and baseline_span.end == candidate_span.end:
                matched_baseline.add(b_idx)
                matched_candidate.add(c_idx)
                result["same_boundary_diff_label"] += 1
                break

    for b_idx, baseline_span in enumerate(baseline_remaining):
        if b_idx in matched_baseline:
            continue
        for c_idx, candidate_span in enumerate(candidate_remaining):
            if c_idx in matched_candidate:
                continue
            if baseline_span.label == candidate_span.label and _spans_overlap(baseline_span, candidate_span):
                matched_baseline.add(b_idx)
                matched_candidate.add(c_idx)
                result["boundary_diff_same_label"] += 1
                break

    for b_idx, baseline_span in enumerate(baseline_remaining):
        if b_idx in matched_baseline:
            continue
        for c_idx, candidate_span in enumerate(candidate_remaining):
            if c_idx in matched_candidate:
                continue
            if _spans_overlap(baseline_span, candidate_span):
                matched_baseline.add(b_idx)
                matched_candidate.add(c_idx)
                result["overlap_diff_label"] += 1
                break

    result["only_baseline"] = sum(1 for index in range(len(baseline_remaining)) if index not in matched_baseline)
    result["only_candidate"] = sum(1 for index in range(len(candidate_remaining)) if index not in matched_candidate)
    return result


def compare_prediction_maps(
    *,
    baseline: dict[str, list[CanonicalSpan]],
    candidate: dict[str, list[CanonicalSpan]],
    baseline_label: str,
    candidate_label: str,
) -> dict[str, Any]:
    documents: dict[str, Any] = {}
    matching_doc_count = 0
    mismatch_doc_count = 0
    aggregate = {
        "exact_match": 0,
        "same_boundary_diff_label": 0,
        "boundary_diff_same_label": 0,
        "overlap_diff_label": 0,
        "only_baseline": 0,
        "only_candidate": 0,
    }
    for doc_id in sorted(set(baseline.keys()) | set(candidate.keys())):
        summary = _compare_span_lists(
            baseline.get(doc_id, []),
            candidate.get(doc_id, []),
        )
        summary[baseline_label] = [span.model_dump() for span in baseline.get(doc_id, [])]
        summary[candidate_label] = [span.model_dump() for span in candidate.get(doc_id, [])]
        documents[doc_id] = summary
        for key in aggregate:
            aggregate[key] += int(summary[key])
        if summary["all_equal"]:
            matching_doc_count += 1
        else:
            mismatch_doc_count += 1
    return {
        "matching_doc_count": matching_doc_count,
        "mismatch_doc_count": mismatch_doc_count,
        "aggregate": aggregate,
        "documents": documents,
    }


def _validate_completed_methods_run(run: dict[str, Any], *, label: str) -> None:
    status = str(run.get("status") or "")
    if status != "completed":
        raise ValueError(f"{label} run must be completed before comparison; got status={status!r}.")


def _compute_current_repo_per_doc_metrics(
    docs_by_id: dict[str, CanonicalDocument],
    predictions: dict[str, list[CanonicalSpan]],
    *,
    match_mode: str,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for doc_id, spans in predictions.items():
        doc = docs_by_id[doc_id]
        results[doc_id] = {
            "filename": doc.filename,
            "hypothesis_spans": [span.model_dump() for span in spans],
            "metrics": compute_metrics(doc.manual_annotations, spans, mode=match_mode),
        }
    return results


def build_repro_comparison_report(
    *,
    colleague_baseline: dict[str, Any],
    compat_run: dict[str, Any],
    current_run: dict[str, Any],
    current_docs: list[CanonicalDocument],
    model: str | None = None,
    match_mode: str = DEFAULT_MATCH_MODE,
) -> dict[str, Any]:
    _validate_completed_methods_run(compat_run, label="compatibility")
    _validate_completed_methods_run(current_run, label="current")
    docs_by_id = {doc.id: doc for doc in current_docs}

    parity_pairs: list[dict[str, Any]] = []
    parity_failed = False
    for method_id in CURRENT_COMPAT_METHODS:
        baseline_docs = {
            doc_id: [
                CanonicalSpan.model_validate(item)
                for item in details["hypothesis_spans"]
            ]
            for doc_id, details in colleague_baseline["experiments"][method_id]["current_repo_metrics"]["documents"].items()
        }
        compat_predictions = extract_methods_lab_predictions(
            compat_run,
            method_id=method_id,
            model=model,
        )
        comparison = compare_prediction_maps(
            baseline=baseline_docs,
            candidate=compat_predictions,
            baseline_label="colleague",
            candidate_label="compatibility",
        )
        compat_metrics = _compute_current_repo_per_doc_metrics(
            docs_by_id,
            compat_predictions,
            match_mode=match_mode,
        )
        pair_summary = {
            "method_id": method_id,
            "colleague_current_repo_metrics": colleague_baseline["experiments"][method_id]["current_repo_metrics"]["aggregate"],
            "compatibility_current_repo_metrics": _aggregate_metrics(compat_metrics),
            "prediction_parity": comparison,
        }
        parity_pairs.append(pair_summary)
        if comparison["mismatch_doc_count"] > 0:
            parity_failed = True

    drift_pairs: list[dict[str, Any]] = []
    if not parity_failed:
        for compat_method_id, current_method_id in CURRENT_DRIFT_METHOD_PAIRS:
            compat_predictions = extract_methods_lab_predictions(
                compat_run,
                method_id=compat_method_id,
                model=model,
            )
            current_predictions = extract_methods_lab_predictions(
                current_run,
                method_id=current_method_id,
                model=model,
            )
            compat_metrics = _compute_current_repo_per_doc_metrics(
                docs_by_id,
                compat_predictions,
                match_mode=match_mode,
            )
            current_metrics = _compute_current_repo_per_doc_metrics(
                docs_by_id,
                current_predictions,
                match_mode=match_mode,
            )
            drift_pairs.append(
                {
                    "compatibility_method_id": compat_method_id,
                    "current_method_id": current_method_id,
                    "compatibility_current_repo_metrics": _aggregate_metrics(compat_metrics),
                    "current_current_repo_metrics": _aggregate_metrics(current_metrics),
                    "prediction_diff": compare_prediction_maps(
                        baseline=compat_predictions,
                        candidate=current_predictions,
                        baseline_label="compatibility",
                        candidate_label="current",
                    ),
                }
            )

    return {
        "status": "parity_failed" if parity_failed else "ok",
        "model": model,
        "parity": {"pairs": parity_pairs},
        "drift": {"pairs": drift_pairs} if drift_pairs else None,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), indent=2))


def _write_markdown_summary(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# Reproducible Baseline + Drift Comparison",
        "",
        f"- Status: {report['status']}",
        f"- Model: {report.get('model')}",
        "",
        "## Prediction Parity",
        "",
        "| Method | Mismatch Docs | Exact Matches | Same Boundary / Diff Label | Boundary Diff / Same Label | Overlap Diff Label |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for pair in report["parity"]["pairs"]:
        parity = pair["prediction_parity"]
        aggregate = parity["aggregate"]
        lines.append(
            "| "
            f"{pair['method_id']} | {parity['mismatch_doc_count']} | {aggregate['exact_match']} | "
            f"{aggregate['same_boundary_diff_label']} | {aggregate['boundary_diff_same_label']} | "
            f"{aggregate['overlap_diff_label']} |"
        )
    if report.get("drift"):
        lines.extend(
            [
                "",
                "## Drift",
                "",
                "| Compatibility | Current | Mismatch Docs | Exact Matches |",
                "| --- | --- | ---: | ---: |",
            ]
        )
        for pair in report["drift"]["pairs"]:
            diff = pair["prediction_diff"]
            lines.append(
                "| "
                f"{pair['compatibility_method_id']} | {pair['current_method_id']} | "
                f"{diff['mismatch_doc_count']} | {diff['aggregate']['exact_match']} |"
            )
    path.write_text("\n".join(lines) + "\n")


def _write_comparison_csv(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "section",
        "baseline_method_id",
        "candidate_method_id",
        "mismatch_doc_count",
        "exact_match",
        "same_boundary_diff_label",
        "boundary_diff_same_label",
        "overlap_diff_label",
        "only_baseline",
        "only_candidate",
        "baseline_f1",
        "candidate_f1",
        "status",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for pair in report["parity"]["pairs"]:
            diff = pair["prediction_parity"]
            aggregate = diff["aggregate"]
            writer.writerow(
                {
                    "section": "parity",
                    "baseline_method_id": pair["method_id"],
                    "candidate_method_id": pair["method_id"],
                    "mismatch_doc_count": diff["mismatch_doc_count"],
                    "exact_match": aggregate["exact_match"],
                    "same_boundary_diff_label": aggregate["same_boundary_diff_label"],
                    "boundary_diff_same_label": aggregate["boundary_diff_same_label"],
                    "overlap_diff_label": aggregate["overlap_diff_label"],
                    "only_baseline": aggregate["only_baseline"],
                    "only_candidate": aggregate["only_candidate"],
                    "baseline_f1": pair["colleague_current_repo_metrics"]["micro"]["f1"],
                    "candidate_f1": pair["compatibility_current_repo_metrics"]["micro"]["f1"],
                    "status": report["status"],
                }
            )
        for pair in (report.get("drift") or {}).get("pairs", []):
            diff = pair["prediction_diff"]
            aggregate = diff["aggregate"]
            writer.writerow(
                {
                    "section": "drift",
                    "baseline_method_id": pair["compatibility_method_id"],
                    "candidate_method_id": pair["current_method_id"],
                    "mismatch_doc_count": diff["mismatch_doc_count"],
                    "exact_match": aggregate["exact_match"],
                    "same_boundary_diff_label": aggregate["same_boundary_diff_label"],
                    "boundary_diff_same_label": aggregate["boundary_diff_same_label"],
                    "overlap_diff_label": aggregate["overlap_diff_label"],
                    "only_baseline": aggregate["only_baseline"],
                    "only_candidate": aggregate["only_candidate"],
                    "baseline_f1": pair["compatibility_current_repo_metrics"]["micro"]["f1"],
                    "candidate_f1": pair["current_current_repo_metrics"]["micro"]["f1"],
                    "status": report["status"],
                }
            )


def _default_output_manifest(output_dir: Path) -> Path:
    return output_dir / "manifest.json"


def _cmd_prepare_dataset(args: argparse.Namespace) -> int:
    docs = load_manual_docs(args.session)
    records = build_colleague_dataset_records(docs)
    output_dir = _resolve_output_dir(args.output_dir, prefix="repro-compare-dataset")
    dataset_path = output_dir / "manual27.colleague.jsonl"
    write_colleague_dataset(records, dataset_path)
    manifest = {
        "kind": "colleague_dataset_export",
        "session_id": args.session,
        "doc_ids": [doc.id for doc in docs],
        "records": [
            {
                "doc_id": doc.id,
                "filename": doc.filename,
                "utterance_count": len(doc.utterances),
                "manual_count": len(doc.manual_annotations),
            }
            for doc in docs
        ],
        "dataset_jsonl": str(dataset_path),
    }
    _write_json(_default_output_manifest(output_dir), manifest)
    print(json.dumps(manifest, indent=2))
    return 0


def _cmd_run_colleague_baseline(args: argparse.Namespace) -> int:
    api_key, api_base = _resolve_gateway_config(api_key=args.api_key, api_base=args.api_base)
    os.environ.setdefault("OPENAI_API_KEY", api_key)
    docs = load_manual_docs(args.session)
    if args.dataset_jsonl:
        dataset_records = load_jsonl_records(Path(args.dataset_jsonl))
        selected_records = select_dataset_records_for_current_docs(
            current_docs=docs,
            dataset_records=dataset_records,
        )
    else:
        selected_records = build_colleague_dataset_records(docs)

    output_dir = _resolve_output_dir(args.output_dir, prefix="repro-compare-baseline")
    dataset_path = output_dir / "selected_dataset.jsonl"
    write_colleague_dataset(selected_records, dataset_path)

    baseline = run_colleague_demo_baseline(
        current_docs=docs,
        records=selected_records,
        model=args.model,
        api_base=api_base,
        repo_root=Path(args.colleague_repo),
        experiments=tuple(args.experiment or COLLEAGUE_EXPERIMENTS),
        max_workers=1,
        current_match_mode=args.match_mode,
    )
    baseline["dataset_jsonl"] = str(dataset_path)
    baseline["source_repo_commit"] = _git_commit(REPO_ROOT)
    _write_json(output_dir / "colleague_demo_v2_baseline.json", baseline)
    print(json.dumps({"output_dir": str(output_dir), "artifact": str(output_dir / "colleague_demo_v2_baseline.json")}, indent=2))
    return 0


def _cmd_run_current_sync(args: argparse.Namespace) -> int:
    api_key, api_base = _resolve_gateway_config(api_key=args.api_key, api_base=args.api_base)
    docs = load_manual_docs(args.session)
    doc_ids = [doc.id for doc in docs]
    bundles = tuple(args.bundle or DEFAULT_CURRENT_BUNDLES)
    output_dir = _resolve_output_dir(args.output_dir, prefix="repro-compare-current")
    manifest: dict[str, Any] = {
        "kind": "current_methods_runs",
        "repo_root": str(REPO_ROOT),
        "repo_commit": _git_commit(REPO_ROOT),
        "session_id": args.session,
        "doc_ids": doc_ids,
        "model": args.model,
        "api_base": api_base,
        "runs": {},
    }
    for bundle in bundles:
        run = run_current_methods_lab_sync(
            session_id=args.session,
            bundle=bundle,
            model=args.model,
            api_key=api_key,
            api_base=api_base,
            doc_ids=doc_ids,
            name=f"Manual27 {bundle}",
        )
        output_path = output_dir / f"{bundle}.methods_lab.json"
        _write_json(output_path, run)
        manifest["runs"][bundle] = {
            "run_id": run["id"],
            "status": run["status"],
            "artifact": str(output_path),
            "method_ids": [method_id for _variant_id, method_id in _methods_for_bundle(bundle)],
        }
    _write_json(_default_output_manifest(output_dir), manifest)
    print(json.dumps(manifest, indent=2))
    return 0


def _cmd_launch_current_api_runs(args: argparse.Namespace) -> int:
    api_key, api_base = _resolve_gateway_config(api_key=args.api_key, api_base=args.api_base)
    docs = load_manual_docs(args.session)
    output_dir = _resolve_output_dir(args.output_dir, prefix="repro-compare-launch")
    launched = launch_current_methods_lab_runs_via_api(
        api_url=args.api_url,
        session_id=args.session,
        bundles=tuple(args.bundle or DEFAULT_CURRENT_BUNDLES),
        model=args.model,
        api_key=api_key,
        api_base=api_base,
        doc_ids=[doc.id for doc in docs],
    )
    _write_json(_default_output_manifest(output_dir), launched)
    print(json.dumps(launched, indent=2))
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    docs = load_manual_docs(args.session)
    colleague_baseline = json.loads(Path(args.colleague_baseline).read_text())
    compat_run = load_methods_lab_run_source(args.compat_run, session_id=args.session)
    current_run = load_methods_lab_run_source(args.current_run, session_id=args.session)
    report = build_repro_comparison_report(
        colleague_baseline=colleague_baseline,
        compat_run=compat_run,
        current_run=current_run,
        current_docs=docs,
        model=args.model,
        match_mode=args.match_mode,
    )
    output_dir = _resolve_output_dir(args.output_dir, prefix="repro-compare-report")
    _write_json(output_dir / "comparison.json", report)
    _write_comparison_csv(output_dir / "comparison.csv", report)
    _write_markdown_summary(output_dir / "comparison.md", report)
    print(json.dumps({"output_dir": str(output_dir), "status": report["status"]}, indent=2))
    return 2 if report["status"] == "parity_failed" else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproducible baseline + drift comparison harness.",
    )
    parser.add_argument("--session", default=DEFAULT_SESSION_ID)
    parser.add_argument("--api-key")
    parser.add_argument("--api-base")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-dataset", help="Export current manual docs to colleague JSONL format.")
    prepare.add_argument("--output-dir")
    prepare.set_defaults(func=_cmd_prepare_dataset)

    baseline = subparsers.add_parser("run-colleague-baseline", help="Run the colleague demo-v2 baseline.")
    baseline.add_argument("--colleague-repo", default=str(DEFAULT_COLLEAGUE_REPO))
    baseline.add_argument("--dataset-jsonl")
    baseline.add_argument("--experiment", action="append", default=[])
    baseline.add_argument("--model", default=DEFAULT_MODEL)
    baseline.add_argument("--match-mode", default=DEFAULT_MATCH_MODE)
    baseline.add_argument("--output-dir")
    baseline.set_defaults(func=_cmd_run_colleague_baseline)

    current = subparsers.add_parser("run-current-sync", help="Run current repo bundles synchronously.")
    current.add_argument("--bundle", action="append", choices=list(DEFAULT_CURRENT_BUNDLES))
    current.add_argument("--model", default=DEFAULT_MODEL)
    current.add_argument("--output-dir")
    current.set_defaults(func=_cmd_run_current_sync)

    launch = subparsers.add_parser(
        "launch-current-api-runs",
        help="Launch current repo Methods Lab runs through the live API for UI observability.",
    )
    launch.add_argument("--bundle", action="append", choices=list(DEFAULT_CURRENT_BUNDLES))
    launch.add_argument("--model", default=DEFAULT_MODEL)
    launch.add_argument("--api-url", default=DEFAULT_METHODS_API_URL)
    launch.add_argument("--output-dir")
    launch.set_defaults(func=_cmd_launch_current_api_runs)

    compare = subparsers.add_parser("compare", help="Compare colleague baseline vs compatibility port vs current bundle.")
    compare.add_argument("--colleague-baseline", required=True)
    compare.add_argument("--compat-run", required=True)
    compare.add_argument("--current-run", required=True)
    compare.add_argument("--model")
    compare.add_argument("--match-mode", default=DEFAULT_MATCH_MODE)
    compare.add_argument("--output-dir")
    compare.set_defaults(func=_cmd_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
