from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping
import zipfile


class Span:
    __slots__ = ("start", "end", "text", "label", "entity_type")

    def __init__(
        self,
        *,
        start: int,
        end: int,
        text: str,
        label: str = "REDACT",
        entity_type: str = "REDACT",
    ) -> None:
        self.start = int(start)
        self.end = int(end)
        self.text = str(text)
        self.label = str(label)
        self.entity_type = str(entity_type)

    def as_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "label": self.label,
            "entity_type": self.entity_type,
        }


def load_jsonl(path_spec: str | Path) -> list[dict[str, Any]]:
    """Load JSONL from a file, or from a zip member using archive.zip::member."""
    spec = str(path_spec)
    lines: list[str]
    if "::" in spec:
        archive_name, member = spec.split("::", 1)
        with zipfile.ZipFile(archive_name) as zf:
            lines = zf.read(member).decode("utf-8").splitlines()
    else:
        lines = Path(spec).read_text(encoding="utf-8").splitlines()

    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"{path_spec}:{line_no}: expected JSON object")
        rows.append(row)
    return rows


def exact_substring_matches(text: str, needle: str) -> list[tuple[int, int]]:
    if not needle:
        return []
    matches: list[tuple[int, int]] = []
    cursor = 0
    while True:
        index = text.find(needle, cursor)
        if index == -1:
            return matches
        matches.append((index, index + len(needle)))
        cursor = index + 1


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(raw)):
        char = raw[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                try:
                    payload = json.loads(raw[start : index + 1])
                except json.JSONDecodeError:
                    return None
                return payload if isinstance(payload, dict) else None
    return None


def _dialogue_id(row: Mapping[str, Any]) -> str:
    raw_id = str(row.get("dialogue_id") or row.get("id") or row.get("document_id") or "")
    if raw_id.endswith("-full-dialogue"):
        raw_id = raw_id[: -len("-full-dialogue")]
    if not raw_id:
        raise ValueError(f"Row missing dialogue_id/id: {row}")
    return raw_id


def _provider(row: Mapping[str, Any]) -> str:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), Mapping) else {}
    return str(row.get("provider") or metadata.get("provider") or "unknown")


def _gold_spans(row: Mapping[str, Any], *, typed_labels: bool) -> list[Span]:
    spans: list[Span] = []
    for item in row.get("gold_spans") or []:
        if not isinstance(item, Mapping):
            continue
        start = int(item["start"])
        end = int(item["end"])
        entity_type = str(item.get("entity_type") or item.get("label") or "REDACT")
        spans.append(
            Span(
                start=start,
                end=end,
                text=str(item.get("text") or ""),
                label=entity_type if typed_labels else "REDACT",
                entity_type=entity_type,
            )
        )
    return spans


def _prediction_payload(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for key in ("redact_spans", "predicted_spans"):
        if isinstance(row.get(key), list):
            return {key: row[key]}

    for key in ("prediction", "predictions", "output", "response"):
        value = row.get(key)
        if isinstance(value, Mapping) and isinstance(value.get("redact_spans"), list):
            return value

    for key in ("completion", "raw_output", "output", "response", "prediction"):
        value = row.get(key)
        if isinstance(value, str):
            payload = _extract_json_object(value)
            if payload is not None and isinstance(payload.get("redact_spans"), list):
                return payload
    return None


def _prediction_items(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    payload = _prediction_payload(row)
    if payload is None:
        return []
    items = payload.get("redact_spans")
    if not isinstance(items, list):
        items = payload.get("predicted_spans")
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, Mapping)]


def _prediction_spans(
    row: Mapping[str, Any],
    *,
    transcript: str,
    typed_labels: bool,
) -> tuple[list[Span], list[dict[str, Any]]]:
    spans: list[Span] = []
    unresolved: list[dict[str, Any]] = []
    dialogue_id = _dialogue_id(row)
    for item in _prediction_items(row):
        entity_type = str(item.get("entity_type") or item.get("label") or "REDACT")
        label = entity_type if typed_labels else "REDACT"
        if "start" in item and "end" in item:
            start = int(item["start"])
            end = int(item["end"])
            if start < 0 or end <= start or end > len(transcript):
                unresolved.append(
                    {
                        "dialogue_id": dialogue_id,
                        "text": str(item.get("text") or ""),
                        "reason": "invalid_offsets",
                        "start": start,
                        "end": end,
                    }
                )
                continue
            spans.append(
                Span(
                    start=start,
                    end=end,
                    text=str(item.get("text") or transcript[start:end]),
                    label=label,
                    entity_type=entity_type,
                )
            )
            continue

        text = str(item.get("text") or "")
        occurrence = int(item.get("occurrence") or 0)
        matches = exact_substring_matches(transcript, text)
        if not text or occurrence < 1 or occurrence > len(matches):
            unresolved.append(
                {
                    "dialogue_id": dialogue_id,
                    "text": text,
                    "occurrence": occurrence,
                    "reason": "occurrence_not_found",
                }
            )
            continue
        start, end = matches[occurrence - 1]
        spans.append(
            Span(
                start=start,
                end=end,
                text=text,
                label=label,
                entity_type=entity_type,
            )
        )
    return spans, unresolved


def _iou(gold: Span, pred: Span) -> float:
    overlap = max(0, min(gold.end, pred.end) - max(gold.start, pred.start))
    if overlap == 0:
        return 0.0
    union = (gold.end - gold.start) + (pred.end - pred.start) - overlap
    return overlap / union if union else 0.0


def _span_overlaps(gold: Span, pred: Span) -> bool:
    return gold.start < pred.end and pred.start < gold.end


def _maximum_cardinality_matches(
    gold_spans: list[Span],
    pred_spans: list[Span],
    *,
    predicate,
) -> list[tuple[int, int]]:
    adj = [
        [pred_index for pred_index, pred in enumerate(pred_spans) if predicate(gold, pred)]
        for gold in gold_spans
    ]
    matched_pred_to_gold = [-1] * len(pred_spans)

    def try_match(gold_index: int, seen: set[int]) -> bool:
        for pred_index in adj[gold_index]:
            if pred_index in seen:
                continue
            seen.add(pred_index)
            if matched_pred_to_gold[pred_index] == -1 or try_match(
                matched_pred_to_gold[pred_index], seen
            ):
                matched_pred_to_gold[pred_index] = gold_index
                return True
        return False

    for gold_index in range(len(gold_spans)):
        try_match(gold_index, set())
    return [
        (gold_index, pred_index)
        for pred_index, gold_index in enumerate(matched_pred_to_gold)
        if gold_index != -1
    ]


def match_spans(
    gold_spans: list[Span],
    pred_spans: list[Span],
    *,
    match_mode: str = "paper-overlap",
    overlap_threshold: float = 0.5,
) -> tuple[list[tuple[Span, Span]], list[Span], list[Span]]:
    if match_mode == "paper-overlap":
        matched_indexes = _maximum_cardinality_matches(
            gold_spans,
            pred_spans,
            predicate=lambda gold, pred: gold.label == pred.label
            and _span_overlaps(gold, pred),
        )
        matched_gold = {gold_index for gold_index, _pred_index in matched_indexes}
        matched_pred = {pred_index for _gold_index, pred_index in matched_indexes}
        matched = [
            (gold_spans[gold_index], pred_spans[pred_index])
            for gold_index, pred_index in matched_indexes
        ]
        false_negatives = [
            span for index, span in enumerate(gold_spans) if index not in matched_gold
        ]
        false_positives = [
            span for index, span in enumerate(pred_spans) if index not in matched_pred
        ]
        return matched, false_negatives, false_positives

    candidates: list[tuple[float, int, int]] = []
    for gold_index, gold in enumerate(gold_spans):
        for pred_index, pred in enumerate(pred_spans):
            if gold.label != pred.label:
                continue
            if match_mode == "exact":
                if gold.start == pred.start and gold.end == pred.end:
                    candidates.append((1.0, gold_index, pred_index))
                continue
            score = _iou(gold, pred)
            if score >= overlap_threshold:
                candidates.append((score, gold_index, pred_index))

    matched_gold: set[int] = set()
    matched_pred: set[int] = set()
    matched: list[tuple[Span, Span]] = []
    for _score, gold_index, pred_index in sorted(candidates, reverse=True):
        if gold_index in matched_gold or pred_index in matched_pred:
            continue
        matched_gold.add(gold_index)
        matched_pred.add(pred_index)
        matched.append((gold_spans[gold_index], pred_spans[pred_index]))

    false_negatives = [
        span for index, span in enumerate(gold_spans) if index not in matched_gold
    ]
    false_positives = [
        span for index, span in enumerate(pred_spans) if index not in matched_pred
    ]
    return matched, false_negatives, false_positives


def _prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _summarize_bucket(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    tp = sum(int(row["tp"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    return {
        "row_count": len(rows),
        "gold_span_count": sum(int(row["gold_span_count"]) for row in rows),
        "predicted_span_count": sum(int(row["predicted_span_count"]) for row in rows),
        "unresolved_prediction_count": sum(
            int(row["unresolved_prediction_count"]) for row in rows
        ),
        "missing_prediction_rows": sum(
            1 for row in rows if bool(row["missing_prediction"])
        ),
        "micro": _prf(tp, fp, fn),
    }


def _prediction_index(prediction_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in prediction_rows:
        dialogue_id = _dialogue_id(row)
        if dialogue_id in index:
            raise ValueError(f"duplicate prediction row for dialogue_id={dialogue_id}")
        index[dialogue_id] = row
    return index


def evaluate_rows(
    gold_rows: list[dict[str, Any]],
    prediction_rows: list[dict[str, Any]] | None = None,
    *,
    match_mode: str = "paper-overlap",
    overlap_threshold: float = 0.5,
    typed_labels: bool = False,
    sample_limit: int = 20,
) -> dict[str, Any]:
    if match_mode not in {"exact", "iou", "paper-overlap"}:
        raise ValueError("match_mode must be exact, iou, or paper-overlap")
    prediction_index = _prediction_index(prediction_rows or gold_rows)
    per_row: list[dict[str, Any]] = []
    false_positive_examples: list[dict[str, Any]] = []
    false_negative_examples: list[dict[str, Any]] = []
    unresolved_examples: list[dict[str, Any]] = []
    entity_type_tp = Counter()
    entity_type_fn = Counter()
    entity_type_support = Counter()

    for gold_row in gold_rows:
        dialogue_id = _dialogue_id(gold_row)
        provider = _provider(gold_row)
        transcript = str(gold_row.get("transcript_text") or gold_row.get("text") or "")
        gold_spans = _gold_spans(gold_row, typed_labels=typed_labels)
        for span in gold_spans:
            entity_type_support[span.entity_type] += 1
        prediction_row = prediction_index.get(dialogue_id)
        missing_prediction = prediction_row is None
        pred_spans: list[Span] = []
        unresolved: list[dict[str, Any]] = []
        if prediction_row is not None:
            pred_spans, unresolved = _prediction_spans(
                prediction_row,
                transcript=transcript,
                typed_labels=typed_labels,
            )

        matched, false_negatives, false_positives = match_spans(
            gold_spans,
            pred_spans,
            match_mode=match_mode,
            overlap_threshold=overlap_threshold,
        )
        for gold_span, _pred_span in matched:
            entity_type_tp[gold_span.entity_type] += 1
        for gold_span in false_negatives:
            entity_type_fn[gold_span.entity_type] += 1

        per_row.append(
            {
                "dialogue_id": dialogue_id,
                "provider": provider,
                "gold_span_count": len(gold_spans),
                "predicted_span_count": len(pred_spans),
                "unresolved_prediction_count": len(unresolved),
                "missing_prediction": missing_prediction,
                "tp": len(matched),
                "fp": len(false_positives),
                "fn": len(false_negatives),
                "micro": _prf(len(matched), len(false_positives), len(false_negatives)),
            }
        )
        for span in false_positives:
            if len(false_positive_examples) >= sample_limit:
                break
            false_positive_examples.append(
                {"dialogue_id": dialogue_id, "provider": provider, "span": span.as_dict()}
            )
        for span in false_negatives:
            if len(false_negative_examples) >= sample_limit:
                break
            false_negative_examples.append(
                {"dialogue_id": dialogue_id, "provider": provider, "span": span.as_dict()}
            )
        for item in unresolved:
            if len(unresolved_examples) >= sample_limit:
                break
            unresolved_examples.append(item)

    by_provider: dict[str, Any] = {}
    rows_by_provider: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_row:
        rows_by_provider[str(row["provider"])].append(row)
    for provider, rows in sorted(rows_by_provider.items()):
        by_provider[provider] = _summarize_bucket(rows)

    by_entity_type: dict[str, Any] = {}
    for entity_type in sorted(entity_type_support):
        tp = entity_type_tp[entity_type]
        fn = entity_type_fn[entity_type]
        by_entity_type[entity_type] = {
            "support": entity_type_support[entity_type],
            "tp": tp,
            "fn": fn,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
        }

    return {
        "match_mode": match_mode,
        "overlap_threshold": overlap_threshold,
        "typed_labels": typed_labels,
        "overall": _summarize_bucket(per_row),
        "by_provider": by_provider,
        "by_entity_type": by_entity_type,
        "rows": per_row,
        "false_positive_examples": false_positive_examples,
        "false_negative_examples": false_negative_examples,
        "unresolved_predictions": unresolved_examples,
    }


def evaluate_files(
    gold_path: str | Path,
    predictions_path: str | Path | None = None,
    *,
    match_mode: str = "paper-overlap",
    overlap_threshold: float = 0.5,
    typed_labels: bool = False,
) -> dict[str, Any]:
    gold_rows = load_jsonl(gold_path)
    prediction_rows = load_jsonl(predictions_path) if predictions_path else None
    report = evaluate_rows(
        gold_rows,
        prediction_rows,
        match_mode=match_mode,
        overlap_threshold=overlap_threshold,
        typed_labels=typed_labels,
    )
    report["gold_path"] = str(gold_path)
    report["predictions_path"] = str(predictions_path) if predictions_path else None
    return report


def _print_summary(report: Mapping[str, Any]) -> None:
    overall = report["overall"]
    micro = overall["micro"]
    print(
        "overall\t"
        f"rows={overall['row_count']}\t"
        f"gold={overall['gold_span_count']}\t"
        f"pred={overall['predicted_span_count']}\t"
        f"tp={micro['tp']}\tfp={micro['fp']}\tfn={micro['fn']}\t"
        f"P={micro['precision']:.4f}\tR={micro['recall']:.4f}\tF1={micro['f1']:.4f}"
    )
    for provider, bucket in report["by_provider"].items():
        micro = bucket["micro"]
        print(
            f"{provider}\t"
            f"rows={bucket['row_count']}\t"
            f"gold={bucket['gold_span_count']}\t"
            f"pred={bucket['predicted_span_count']}\t"
            f"tp={micro['tp']}\tfp={micro['fp']}\tfn={micro['fn']}\t"
            f"P={micro['precision']:.4f}\tR={micro['recall']:.4f}\tF1={micro['f1']:.4f}"
        )
    if overall["unresolved_prediction_count"]:
        print(f"unresolved_predictions={overall['unresolved_prediction_count']}")
    if overall["missing_prediction_rows"]:
        print(f"missing_prediction_rows={overall['missing_prediction_rows']}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate full-dialogue de-identification predictions against exported "
            "gold_spans JSONL. Use archive.zip::member to read JSONL directly from an export ZIP."
        )
    )
    parser.add_argument("--gold", required=True, help="Gold JSONL path, or archive.zip::member.")
    parser.add_argument(
        "--predictions",
        help=(
            "Predictions JSONL path. If omitted, the gold rows' completion field is used "
            "as a perfect-format sanity check."
        ),
    )
    parser.add_argument("--output-json", type=Path, help="Optional JSON report path.")
    parser.add_argument(
        "--match-mode",
        choices=("exact", "iou", "paper-overlap"),
        default="paper-overlap",
        help=(
            "paper-overlap matches the final paper scorer: one-to-one maximum "
            "matching where any overlapping redaction span counts."
        ),
    )
    parser.add_argument("--overlap-threshold", type=float, default=0.5)
    parser.add_argument(
        "--typed-labels",
        action="store_true",
        help="Require predicted entity_type/label to match gold. Default collapses all labels to REDACT.",
    )
    args = parser.parse_args(argv)

    report = evaluate_files(
        args.gold,
        args.predictions,
        match_mode=args.match_mode,
        overlap_threshold=args.overlap_threshold,
        typed_labels=args.typed_labels,
    )
    _print_summary(report)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
