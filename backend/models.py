from __future__ import annotations

from typing import Any, Literal

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
    llm_runs: dict[str, list[CanonicalSpan]] = Field(default_factory=dict)
    llm_run_metadata: dict[str, "SavedRunMetadata"] = Field(default_factory=dict)
    methods: dict[str, list[CanonicalSpan]] = Field(default_factory=dict)
    method_run_metadata: dict[str, "SavedRunMetadata"] = Field(default_factory=dict)


class SavedRunMetadata(BaseModel):
    mode: Literal["manual", "rule", "llm", "method"]
    updated_at: str
    model: str | None = None
    method_id: str | None = None
    label_profile: Literal["simple", "advanced"] | None = None
    prompt_snapshot: dict[str, Any] | None = None
    llm_confidence: "LLMConfidenceMetric | None" = None


class LLMConfidenceMetric(BaseModel):
    available: bool
    provider: str
    model: str
    reason: Literal[
        "ok", "unsupported_provider", "missing_logprobs", "empty_completion"
    ]
    token_count: int
    mean_logprob: float | None = None
    confidence: float | None = None
    perplexity: float | None = None
    band: Literal["high", "medium", "low", "na"]
    high_threshold: float = 0.9
    medium_threshold: float = 0.75


class AgentRunMetrics(BaseModel):
    llm_confidence: LLMConfidenceMetric | None = None
    label_profile: Literal["simple", "advanced"] | None = None


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
    agent_run_metrics: AgentRunMetrics = Field(default_factory=AgentRunMetrics)
    status: Literal["pending", "in_progress", "reviewed"] = "pending"
