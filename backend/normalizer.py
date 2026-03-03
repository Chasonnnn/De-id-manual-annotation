from __future__ import annotations

import json
import re
from pathlib import Path

from models import CanonicalDocument, CanonicalSpan, UtteranceRow

PII_TYPE_MAP = {
    "PERSON": "NAME",
}


def _map_label(raw: str) -> str:
    return PII_TYPE_MAP.get(raw.upper(), raw.upper())


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
    full_text = data["transcript"]
    spans = [
        CanonicalSpan(
            start=p["start"],
            end=p["end"],
            label=_map_label(p["pii_type"]),
            text=p.get("text", full_text[p["start"]:p["end"]]),
        )
        for p in data.get("pii_occurrences", [])
    ]
    return CanonicalDocument(
        id=doc_id,
        filename=filename,
        format="hips_v1",
        full_text=full_text,
        utterances=[
            UtteranceRow(speaker="unknown", text=full_text, global_start=0, global_end=len(full_text))
        ],
        gold_spans=spans,
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
    return CanonicalDocument(
        id=doc_id,
        filename=filename,
        format="hips_v2",
        full_text=full_text,
        utterances=[
            UtteranceRow(speaker="unknown", text=full_text, global_start=0, global_end=len(full_text))
        ],
        gold_spans=spans,
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
    return CanonicalDocument(
        id=doc_id,
        filename=filename,
        format="jsonl",
        full_text=full_text,
        utterances=utterances,
        gold_spans=spans,
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
            docs.append(parse_jsonl_record(data, filename, rid))
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
