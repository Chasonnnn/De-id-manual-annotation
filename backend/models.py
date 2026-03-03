from __future__ import annotations

from pydantic import BaseModel


class CanonicalSpan(BaseModel):
    start: int
    end: int
    label: str
    text: str


class UtteranceRow(BaseModel):
    speaker: str
    text: str
    global_start: int
    global_end: int


class CanonicalDocument(BaseModel):
    id: str
    filename: str
    format: str  # "hips_v1" | "hips_v2" | "jsonl"
    full_text: str
    utterances: list[UtteranceRow]
    gold_spans: list[CanonicalSpan]
