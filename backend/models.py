from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


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


class AgentOutputs(BaseModel):
    rule: list[CanonicalSpan] = Field(default_factory=list)
    llm: list[CanonicalSpan] = Field(default_factory=list)


class CanonicalDocument(BaseModel):
    id: str
    filename: str
    format: Literal["hips_v1", "hips_v2", "jsonl"]
    raw_text: str
    utterances: list[UtteranceRow]
    pre_annotations: list[CanonicalSpan]
    label_set: list[str] = Field(default_factory=list)
    manual_annotations: list[CanonicalSpan] = Field(default_factory=list)
    agent_annotations: list[CanonicalSpan] = Field(default_factory=list)
    agent_outputs: AgentOutputs = Field(default_factory=AgentOutputs)
    agent_run_warnings: list[str] = Field(default_factory=list)
    status: Literal["pending", "in_progress", "reviewed"] = "pending"
