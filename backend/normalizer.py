from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from models import CanonicalDocument, CanonicalSpan, UtteranceRow

PII_TYPE_MAP = {
    "PERSON": "NAME",
}


def _map_label(raw: str) -> str:
    return PII_TYPE_MAP.get(raw.upper(), raw.upper())


def _collect_label_set(*label_lists: list[str]) -> list[str]:
    labels: set[str] = set()
    for values in label_lists:
        for label in values:
            if label:
                labels.add(_map_label(label))
    return sorted(labels)


def _dedup_spans(spans: list[CanonicalSpan]) -> list[CanonicalSpan]:
    seen: set[tuple[int, int, str]] = set()
    result: list[CanonicalSpan] = []
    for s in spans:
        key = (s.start, s.end, s.label)
        if key not in seen:
            seen.add(key)
            result.append(s)
    return result


def _build_utterances(full_text: str) -> list[UtteranceRow]:
    rows: list[UtteranceRow] = []
    offset = 0
    for line in full_text.split("\n"):
        rows.append(
            UtteranceRow(
                speaker="unknown",
                text=line,
                global_start=offset,
                global_end=offset + len(line),
            )
        )
        offset += len(line) + 1  # +1 for newline
    return rows


def _detect_format(data: dict) -> str:
    if "transcript" in data and isinstance(data["transcript"], list):
        first = data["transcript"][0] if data["transcript"] else {}
        if "content" in first and "role" in first:
            return "jsonl"
    if "text" in data and "pii" in data:
        return "hips_v2"
    if "transcript" in data and "pii_occurrences" in data:
        return "hips_v1"
    if "text" in data and "pii_occurrences" in data:
        return "hips_v1"
    raise ValueError(f"Unknown format. Keys: {list(data.keys())}")


def parse_hips_v1(data: dict, filename: str, doc_id: str) -> CanonicalDocument:
    full_text = data.get("transcript") or data.get("text")
    if not isinstance(full_text, str):
        raise ValueError("HIPS v1 payload must include transcript text")

    spans = [
        CanonicalSpan(
            start=p["start"],
            end=p["end"],
            label=_map_label(p["pii_type"]),
            text=p.get("text", full_text[p["start"]:p["end"]]),
        )
        for p in data.get("pii_occurrences", [])
    ]
    deduped = _dedup_spans(spans)
    labels_from_distinct = list(data.get("distinct_pii", {}).keys())
    labels_from_spans = [s.label for s in deduped]
    return CanonicalDocument(
        id=doc_id,
        filename=filename,
        format="hips_v1",
        raw_text=full_text,
        utterances=_build_utterances(full_text),
        pre_annotations=deduped,
        label_set=_collect_label_set(labels_from_distinct, labels_from_spans),
    )


def parse_hips_v2(data: dict, filename: str, doc_id: str) -> CanonicalDocument:
    full_text = data["text"]
    spans = [
        CanonicalSpan(
            start=p["start"],
            end=p["end"],
            label=_map_label(p["type"]),
            text=p.get("pii", full_text[p["start"]:p["end"]]),
        )
        for p in data.get("pii", [])
    ]
    deduped = _dedup_spans(spans)
    labels_from_distinct = list(data.get("distinct_pii", {}).keys())
    labels_from_spans = [s.label for s in deduped]
    return CanonicalDocument(
        id=doc_id,
        filename=filename,
        format="hips_v2",
        raw_text=full_text,
        utterances=_build_utterances(full_text),
        pre_annotations=deduped,
        label_set=_collect_label_set(labels_from_distinct, labels_from_spans),
    )


def parse_jsonl_record(data: dict, filename: str, doc_id: str) -> CanonicalDocument:
    turns = data.get("transcript", [])
    if not isinstance(turns, list):
        raise ValueError("JSONL record must include 'transcript' as a list")

    ordered_turns: list[dict[str, Any]] = []
    for index, turn in enumerate(turns):
        if isinstance(turn, dict):
            ordered_turns.append({"__idx": index, **turn})

    def _turn_sort_key(turn: dict[str, Any]) -> tuple[int, int]:
        sequence = turn.get("sequence_id")
        if isinstance(sequence, int):
            return (0, sequence)
        return (1, int(turn.get("__idx", 0)))

    ordered_turns.sort(key=_turn_sort_key)

    parts: list[str] = []
    utterances: list[UtteranceRow] = []
    spans: list[CanonicalSpan] = []
    offset = 0

    for turn in ordered_turns:
        role = str(turn.get("role", "unknown") or "unknown")
        content = str(turn.get("content", "") or "")
        line_prefix = f"{role}: "
        line_text = f"{line_prefix}{content}"

        if parts:
            offset += 1  # for \n separator
        line_start = offset
        content_start = line_start + len(line_prefix)
        content_end = content_start + len(content)

        parts.append(line_text)
        utterances.append(
            UtteranceRow(
                speaker=role,
                text=content,
                global_start=content_start,
                global_end=content_end,
            )
        )

        for ann in turn.get("annotations", []):
            if not isinstance(ann, dict):
                continue
            start = ann.get("start")
            end = ann.get("end")
            pii_type = ann.get("pii_type")
            if not isinstance(start, int) or not isinstance(end, int) or not isinstance(pii_type, str):
                continue
            abs_start = content_start + start
            abs_end = content_start + end
            if abs_start < content_start or abs_end > content_end or abs_start >= abs_end:
                continue
            label = _map_label(pii_type)
            spans.append(
                CanonicalSpan(
                    start=abs_start,
                    end=abs_end,
                    label=label,
                    text=content[start:end],
                )
            )

        offset = line_start + len(line_text)

    full_text = "\n".join(parts)
    deduped = _dedup_spans(spans)
    labels_from_spans = [s.label for s in deduped]
    return CanonicalDocument(
        id=doc_id,
        filename=filename,
        format="jsonl",
        raw_text=full_text,
        utterances=utterances,
        pre_annotations=deduped,
        label_set=_collect_label_set(labels_from_spans),
    )


def _merge_jsonl_documents(
    record_docs: list[CanonicalDocument], filename: str, doc_id: str
) -> CanonicalDocument:
    parts: list[str] = []
    utterances: list[UtteranceRow] = []
    spans: list[CanonicalSpan] = []
    offset = 0

    for index, record_doc in enumerate(record_docs):
        if index > 0:
            offset += 1  # for \n separator
        parts.append(record_doc.raw_text)

        for row in record_doc.utterances:
            utterances.append(
                UtteranceRow(
                    speaker=row.speaker,
                    text=row.text,
                    global_start=row.global_start + offset,
                    global_end=row.global_end + offset,
                )
            )

        for span in record_doc.pre_annotations:
            spans.append(
                CanonicalSpan(
                    start=span.start + offset,
                    end=span.end + offset,
                    label=span.label,
                    text=span.text,
                )
            )

        offset += len(record_doc.raw_text)

    full_text = "\n".join(parts)
    deduped = _dedup_spans(spans)
    labels_from_docs = [label for doc in record_docs for label in doc.label_set]
    labels_from_spans = [s.label for s in deduped]

    return CanonicalDocument(
        id=doc_id,
        filename=filename,
        format="jsonl",
        raw_text=full_text,
        utterances=utterances,
        pre_annotations=deduped,
        label_set=_collect_label_set(labels_from_docs, labels_from_spans),
    )


def _derive_jsonl_record_display_name(data: dict[str, Any], index: int) -> str:
    transcript = data.get("transcript", [])
    session_ids: set[str] = set()
    if isinstance(data.get("session_id"), (str, int)):
        value = str(data.get("session_id")).strip()
        if value:
            session_ids.add(value)
    if isinstance(transcript, list):
        for turn in transcript:
            if not isinstance(turn, dict):
                continue
            raw_session_id = turn.get("session_id")
            if isinstance(raw_session_id, (str, int)):
                value = str(raw_session_id).strip()
                if value:
                    session_ids.add(value)
    if len(session_ids) == 1:
        return f"Session {next(iter(session_ids))}"
    return f"Record {index + 1}"


def parse_jsonl_file(
    raw_bytes: bytes,
    filename: str,
    doc_id: str,
) -> tuple[CanonicalDocument, list[CanonicalDocument], list[str]]:
    text = raw_bytes.decode("utf-8")
    record_docs: list[CanonicalDocument] = []
    record_display_names: list[str] = []
    stem = Path(filename).stem or "record"
    for i, line in enumerate(text.strip().splitlines()):
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        rid = f"{doc_id}_line{i}"
        record_name = f"{stem}.record-{i + 1:04d}.json"
        record_docs.append(parse_jsonl_record(data, record_name, rid))
        record_display_names.append(_derive_jsonl_record_display_name(data, i))
    if not record_docs:
        raise ValueError("No JSONL records parsed from file")
    merged = _merge_jsonl_documents(record_docs, filename, doc_id)
    return merged, record_docs, record_display_names


def parse_file(raw_bytes: bytes, filename: str, doc_id: str) -> list[CanonicalDocument]:
    docs: list[CanonicalDocument] = []

    if filename.endswith(".jsonl"):
        merged, _, _ = parse_jsonl_file(raw_bytes, filename, doc_id)
        docs.append(merged)
    else:
        text = raw_bytes.decode("utf-8")
        data = json.loads(text)
        fmt = _detect_format(data)
        if fmt == "hips_v1":
            docs.append(parse_hips_v1(data, filename, doc_id))
        elif fmt == "hips_v2":
            docs.append(parse_hips_v2(data, filename, doc_id))
        elif fmt == "jsonl":
            docs.append(parse_jsonl_record(data, filename, doc_id))

    return docs
