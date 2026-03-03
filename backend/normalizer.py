from __future__ import annotations

import json

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
    parts: list[str] = []
    utterances: list[UtteranceRow] = []
    spans: list[CanonicalSpan] = []
    offset = 0

    for turn in turns:
        content = turn.get("content", "")
        if parts:
            offset += 1  # for \n separator
        global_start = offset
        global_end = offset + len(content)

        parts.append(content)
        utterances.append(
            UtteranceRow(
                speaker=turn.get("role", "unknown"),
                text=content,
                global_start=global_start,
                global_end=global_end,
            )
        )

        for ann in turn.get("annotations", []):
            abs_start = global_start + ann["start"]
            abs_end = global_start + ann["end"]
            label = _map_label(ann["pii_type"])
            spans.append(
                CanonicalSpan(
                    start=abs_start,
                    end=abs_end,
                    label=label,
                    text=content[ann["start"]:ann["end"]],
                )
            )

        offset = global_end

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


def parse_file(raw_bytes: bytes, filename: str, doc_id: str) -> list[CanonicalDocument]:
    text = raw_bytes.decode("utf-8")
    docs: list[CanonicalDocument] = []

    if filename.endswith(".jsonl"):
        for i, line in enumerate(text.strip().splitlines()):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            rid = f"{doc_id}_line{i}"
            record_name = f"{filename}#record-{i}"
            docs.append(parse_jsonl_record(data, record_name, rid))
    else:
        data = json.loads(text)
        fmt = _detect_format(data)
        if fmt == "hips_v1":
            docs.append(parse_hips_v1(data, filename, doc_id))
        elif fmt == "hips_v2":
            docs.append(parse_hips_v2(data, filename, doc_id))
        elif fmt == "jsonl":
            docs.append(parse_jsonl_record(data, filename, doc_id))

    return docs
