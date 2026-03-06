from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent import SYSTEM_PROMPT  # noqa: E402
from metrics import compute_metrics  # noqa: E402
from server import (  # noqa: E402
    _build_text_chunks,
    _enrich_doc,
    _load_doc,
    _load_session_index,
    _run_llm_for_document,
)
from models import CanonicalSpan  # noqa: E402


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


def _resolve_gateway_config(args: argparse.Namespace) -> tuple[str, str | None]:
    repo_env = ROOT.parent / ".env.local"
    _load_local_env(repo_env)
    api_key = (
        str(args.api_key or "").strip()
        or os.getenv("LITELLM_API_KEY", "").strip()
        or os.getenv("OPENAI_API_KEY", "").strip()
        or os.getenv("ANTHROPIC_API_KEY", "").strip()
    )
    api_base = (
        str(args.api_base or "").strip()
        or os.getenv("LITELLM_BASE_URL", "").strip()
        or None
    )
    if not api_key:
        raise SystemExit(
            "Missing API key. Set LITELLM_API_KEY/OPENAI_API_KEY/ANTHROPIC_API_KEY or pass --api-key."
        )
    return api_key, api_base


def _load_manual_docs(session_id: str) -> list[Any]:
    docs: list[Any] = []
    for doc_id in _load_session_index(session_id):
        doc = _load_doc(doc_id, session_id)
        if doc is None:
            continue
        enriched = _enrich_doc(doc, session_id)
        if enriched.manual_annotations:
            docs.append(enriched)
    return docs


def _run_variant(
    *,
    doc: Any,
    api_key: str,
    api_base: str | None,
    model: str,
    system_prompt: str,
    temperature: float,
    reasoning_effort: str,
    anthropic_thinking: bool,
    anthropic_thinking_budget_tokens: int | None,
    label_profile: str,
    chunk_mode: str,
    chunk_size_chars: int,
    enable_suspicious_empty_retry: bool,
    match_mode: str,
) -> dict[str, Any]:
    spans, warnings, llm_confidence, chunk_diagnostics = _run_llm_for_document(
        doc=doc,
        api_key=api_key,
        api_base=api_base,
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
        reasoning_effort=reasoning_effort,
        anthropic_thinking=anthropic_thinking,
        anthropic_thinking_budget_tokens=anthropic_thinking_budget_tokens,
        label_profile=label_profile,  # type: ignore[arg-type]
        chunk_mode=chunk_mode,
        chunk_size_chars=chunk_size_chars,
        enable_suspicious_empty_retry=enable_suspicious_empty_retry,
    )
    metrics = compute_metrics(doc.manual_annotations, spans, mode=match_mode)
    suspicious_empty_count = sum(1 for item in chunk_diagnostics if item.suspicious_empty)
    return {
        "doc_id": doc.id,
        "filename": doc.filename,
        "metrics": metrics,
        "warnings": warnings,
        "span_count": len(spans),
        "manual_count": len(doc.manual_annotations),
        "suspicious_empty_chunks": suspicious_empty_count,
        "chunk_diagnostics": [item.model_dump() for item in chunk_diagnostics],
        "llm_confidence": llm_confidence.model_dump() if llm_confidence is not None else None,
    }


def _json_safe(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _manual_spans_for_range(doc: Any, start: int, end: int) -> list[CanonicalSpan]:
    result: list[CanonicalSpan] = []
    for span in doc.manual_annotations:
        if span.start < start or span.end > end:
            continue
        result.append(
            CanonicalSpan(
                start=span.start - start,
                end=span.end - start,
                label=span.label,
                text=span.text,
            )
        )
    return result


def _print_compare_table(results: list[dict[str, Any]]) -> None:
    print(
        "model\tchunk_mode\tretry\tdocs\trepeats\tmean_micro_f1\tmin_micro_f1\tmax_micro_f1\tmean_macro_f1\tmean_fn_total\tmean_suspicious_empty_chunks"
    )
    for item in results:
        print(
            "\t".join(
                [
                    str(item["model"]),
                    str(item["chunk_mode"]),
                    str(item["retry_mode"]),
                    str(item["doc_count"]),
                    str(item["repeats"]),
                    f"{item['mean_micro_f1']:.4f}",
                    f"{item['min_micro_f1']:.4f}",
                    f"{item['max_micro_f1']:.4f}",
                    f"{item['mean_macro_f1']:.4f}",
                    f"{item['mean_fn_total']:.1f}",
                    f"{item['mean_suspicious_empty_chunks']:.2f}",
                ]
            )
        )


def run_compare(args: argparse.Namespace) -> int:
    api_key, api_base = _resolve_gateway_config(args)
    docs = _load_manual_docs(args.session)
    if not docs:
        raise SystemExit(f"No manual-annotated docs found in session '{args.session}'.")

    summaries: list[dict[str, Any]] = []
    for model in args.model:
        for chunk_mode in args.chunk_mode:
            for retry_mode in args.retry_mode:
                enable_suspicious_empty_retry = retry_mode == "on"
                run_summaries: list[dict[str, Any]] = []
                for repeat_index in range(args.repeats):
                    per_doc = [
                        _run_variant(
                            doc=doc,
                            api_key=api_key,
                            api_base=api_base,
                            model=model,
                            system_prompt=args.system_prompt,
                            temperature=args.temperature,
                            reasoning_effort=args.reasoning_effort,
                            anthropic_thinking=args.anthropic_thinking,
                            anthropic_thinking_budget_tokens=args.anthropic_thinking_budget_tokens,
                            label_profile=args.label_profile,
                            chunk_mode=chunk_mode,
                            chunk_size_chars=args.chunk_size_chars,
                            enable_suspicious_empty_retry=enable_suspicious_empty_retry,
                            match_mode=args.match_mode,
                        )
                        for doc in docs
                    ]
                    tp_total = sum(int(item["metrics"]["micro"]["tp"]) for item in per_doc)
                    fp_total = sum(int(item["metrics"]["micro"]["fp"]) for item in per_doc)
                    fn_total = sum(int(item["metrics"]["micro"]["fn"]) for item in per_doc)
                    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
                    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
                    micro_f1 = (
                        2 * precision * recall / (precision + recall)
                        if (precision + recall)
                        else 0.0
                    )
                    macro_f1 = statistics.fmean(
                        float(item["metrics"]["micro"]["f1"]) for item in per_doc
                    )
                    run_summaries.append(
                        {
                            "repeat_index": repeat_index + 1,
                            "micro_f1": micro_f1,
                            "macro_f1": macro_f1,
                            "fn_total": fn_total,
                            "suspicious_empty_chunks": sum(
                                int(item["suspicious_empty_chunks"]) for item in per_doc
                            ),
                            "documents": per_doc,
                        }
                    )

                micro_f1_values = [float(item["micro_f1"]) for item in run_summaries]
                macro_f1_values = [float(item["macro_f1"]) for item in run_summaries]
                fn_totals = [int(item["fn_total"]) for item in run_summaries]
                suspicious_empty_totals = [
                    int(item["suspicious_empty_chunks"]) for item in run_summaries
                ]
                summaries.append(
                    {
                        "model": model,
                        "chunk_mode": chunk_mode,
                        "retry_mode": retry_mode,
                        "doc_count": len(docs),
                        "repeats": args.repeats,
                        "mean_micro_f1": statistics.fmean(micro_f1_values),
                        "min_micro_f1": min(micro_f1_values),
                        "max_micro_f1": max(micro_f1_values),
                        "mean_macro_f1": statistics.fmean(macro_f1_values),
                        "mean_fn_total": statistics.fmean(fn_totals),
                        "min_fn_total": min(fn_totals),
                        "max_fn_total": max(fn_totals),
                        "mean_suspicious_empty_chunks": statistics.fmean(
                            suspicious_empty_totals
                        ),
                        "min_suspicious_empty_chunks": min(suspicious_empty_totals),
                        "max_suspicious_empty_chunks": max(suspicious_empty_totals),
                        "runs": run_summaries,
                    }
                )

    _print_compare_table(summaries)
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(_json_safe(summaries), indent=2))
        print(f"\nWrote JSON report to {args.output_json}")
    return 0


def run_probe(args: argparse.Namespace) -> int:
    api_key, api_base = _resolve_gateway_config(args)
    doc = _load_doc(args.doc_id, args.session)
    if doc is None:
        raise SystemExit(f"Unknown doc id: {args.doc_id}")
    enriched = _enrich_doc(doc, args.session)
    chunks = _build_text_chunks(enriched, args.chunk_size_chars)
    if args.chunk_index < 0 or args.chunk_index >= len(chunks):
        raise SystemExit(
            f"chunk_index {args.chunk_index} is out of range for {len(chunks)} chunk(s)."
        )
    start, end = chunks[args.chunk_index]
    chunk_text = enriched.raw_text[start:end]
    manual_chunk_spans = _manual_spans_for_range(enriched, start, end)

    span_counts: list[int] = []
    f1_scores: list[float] = []
    outputs: list[dict[str, Any]] = []
    for run_index in range(args.repeats):
        spans, warnings, llm_confidence, chunk_diagnostics = _run_llm_for_document(
            doc=enriched.model_copy(update={"raw_text": chunk_text}),
            api_key=api_key,
            api_base=api_base,
            model=args.model,
            system_prompt=args.system_prompt,
            temperature=args.temperature,
            reasoning_effort=args.reasoning_effort,
            anthropic_thinking=args.anthropic_thinking,
            anthropic_thinking_budget_tokens=args.anthropic_thinking_budget_tokens,
            label_profile=args.label_profile,  # type: ignore[arg-type]
            chunk_mode="off",
            chunk_size_chars=args.chunk_size_chars,
            enable_suspicious_empty_retry=args.retry_mode == "on",
        )
        span_counts.append(len(spans))
        metrics = compute_metrics(manual_chunk_spans, spans, mode=args.match_mode)
        f1_scores.append(float(metrics["micro"]["f1"]))
        outputs.append(
            {
                "run_index": run_index + 1,
                "span_count": len(spans),
                "metrics": _json_safe(metrics),
                "warnings": warnings,
                "llm_confidence": llm_confidence.model_dump()
                if llm_confidence is not None
                else None,
                "chunk_diagnostics": [item.model_dump() for item in chunk_diagnostics],
            }
        )

    report = {
        "doc_id": enriched.id,
        "chunk_index": args.chunk_index,
        "range": {"start": start, "end": end},
        "repeats": args.repeats,
        "model": args.model,
        "retry_mode": args.retry_mode,
        "counts": span_counts,
        "micro_f1_scores": f1_scores,
        "mean_micro_f1": statistics.fmean(f1_scores) if f1_scores else 0.0,
        "mean": statistics.fmean(span_counts) if span_counts else 0.0,
        "min": min(span_counts) if span_counts else 0,
        "max": max(span_counts) if span_counts else 0,
        "manual_chunk_span_count": len(manual_chunk_spans),
        "runs": outputs,
    }
    print(json.dumps(report, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manual-backed LLM evaluation harness for the annotation tool.",
    )
    parser.add_argument("--session", default="default")
    parser.add_argument("--api-key")
    parser.add_argument("--api-base")
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--reasoning-effort", default="xhigh")
    parser.add_argument("--anthropic-thinking", action="store_true")
    parser.add_argument("--anthropic-thinking-budget-tokens", type=int, default=None)
    parser.add_argument("--label-profile", choices=["simple", "advanced"], default="simple")
    parser.add_argument("--chunk-size-chars", type=int, default=10_000)

    subparsers = parser.add_subparsers(dest="command", required=True)

    compare = subparsers.add_parser("compare", help="Run full-doc manual-backed comparisons.")
    compare.add_argument(
        "--model",
        action="append",
        required=True,
        help="Repeatable. Example: --model openai.gpt-5.2-chat",
    )
    compare.add_argument(
        "--chunk-mode",
        action="append",
        choices=["auto", "off", "force"],
        required=True,
        help="Repeatable. Example: --chunk-mode auto --chunk-mode off",
    )
    compare.add_argument(
        "--retry-mode",
        action="append",
        choices=["on", "off"],
        required=True,
        help="Repeatable. Use 'off' to approximate Plan B without suspicious-empty retry.",
    )
    compare.add_argument("--repeats", type=int, default=1)
    compare.add_argument("--match-mode", choices=["exact", "boundary", "overlap"], default="boundary")
    compare.add_argument("--output-json")
    compare.set_defaults(func=run_compare)

    probe = subparsers.add_parser("probe", help="Run repeated probes on one chunk.")
    probe.add_argument("--model", required=True)
    probe.add_argument("--doc-id", required=True)
    probe.add_argument("--chunk-index", type=int, required=True)
    probe.add_argument("--repeats", type=int, default=3)
    probe.add_argument("--retry-mode", choices=["on", "off"], default="on")
    probe.add_argument("--match-mode", choices=["exact", "boundary", "overlap"], default="boundary")
    probe.set_defaults(func=run_probe)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
