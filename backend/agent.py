from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Literal

from litellm import completion, get_llm_provider

from models import CanonicalSpan

# Regex patterns for rule-based detection
PATTERNS: dict[str, re.Pattern] = {
    "EMAIL": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "URL": re.compile(r"https?://[^\s,)]+"),
    "PHONE": re.compile(r"\+?\d[\d\- ]{7,}\d"),
    "DATE": re.compile(
        r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
        r"|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)"
        r"\s+\d{1,2},?\s*\d{2,4}\b",
        re.IGNORECASE,
    ),
    "TIME": re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b"),
}


def run_regex(text: str) -> list[CanonicalSpan]:
    spans: list[CanonicalSpan] = []
    for label, pattern in PATTERNS.items():
        for m in pattern.finditer(text):
            spans.append(
                CanonicalSpan(
                    start=m.start(),
                    end=m.end(),
                    label=label,
                    text=m.group(),
                )
            )
    spans.sort(key=lambda s: s.start)
    return spans


SYSTEM_PROMPT = """\
You are a PII annotation assistant. Given a transcript, identify all personally
identifiable information (PII) spans. For each span return a JSON object with:
- "start": character offset (0-based)
- "end": character offset (exclusive)
- "label": PII type (NAME, EMAIL, PHONE, URL, DATE, TIME, ADDRESS, SSN, etc.)
- "text": the exact text of the span

Return ONLY a JSON array of these objects, no other text."""


ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh"]


MODEL_PRESETS: list[dict[str, Any]] = [
    {
        "label": "OpenAI: Codex 5.3 (xhigh)",
        "model": "openai.gpt-5.3-codex",
        "provider": "openai",
        "supports_reasoning_effort": True,
        "supports_anthropic_thinking": False,
        "default_reasoning_effort": "xhigh",
    },
    {
        "label": "OpenAI: ChatGPT 5.2 (xhigh)",
        "model": "openai.gpt-5.2-chat",
        "provider": "openai",
        "supports_reasoning_effort": True,
        "supports_anthropic_thinking": False,
        "default_reasoning_effort": "xhigh",
    },
    {
        "label": "Anthropic: Claude Opus 4.6 (thinking)",
        "model": "anthropic.claude-4.6-opus",
        "provider": "anthropic",
        "supports_reasoning_effort": False,
        "supports_anthropic_thinking": True,
        "default_reasoning_effort": "none",
    },
    {
        "label": "Google: Gemini 3.1 Pro Preview",
        "model": "google.gemini-3.1-pro-preview",
        "provider": "gemini",
        "supports_reasoning_effort": False,
        "supports_anthropic_thinking": False,
        "default_reasoning_effort": "none",
    },
    {
        "label": "Google: Gemini 3 Flash",
        "model": "google.gemini-3-flash-preview",
        "provider": "gemini",
        "supports_reasoning_effort": False,
        "supports_anthropic_thinking": False,
        "default_reasoning_effort": "none",
    },
]


MODEL_PRESET_BY_ID: dict[str, dict[str, Any]] = {
    p["model"]: p for p in MODEL_PRESETS
}


@dataclass
class LLMRunResult:
    spans: list[CanonicalSpan]
    warnings: list[str]


def _strip_code_fences(content: str) -> str:
    """Remove markdown code fences from LLM output."""
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines)
    return content


def _infer_provider(model: str) -> str:
    try:
        _, provider, _, _ = get_llm_provider(model=model)
        return str(provider)
    except Exception:
        if "/" in model:
            return model.split("/", 1)[0]
        return "unknown"


def _supports_reasoning_effort(model: str, provider: str) -> bool:
    preset = MODEL_PRESET_BY_ID.get(model)
    if preset is not None:
        return bool(preset["supports_reasoning_effort"])
    return provider == "openai" and "gpt-5" in model


def _supports_anthropic_thinking(model: str, provider: str) -> bool:
    preset = MODEL_PRESET_BY_ID.get(model)
    if preset is not None:
        return bool(preset["supports_anthropic_thinking"])
    return provider == "anthropic" and (
        "claude-opus-4-6" in model or "claude-4.6-opus" in model
    )


def _supports_custom_temperature(model: str) -> bool:
    # GPT-5 family generally supports default temperature only.
    if model.startswith("openai.gpt-5") or model.startswith("openai/gpt-5"):
        return False
    return True


def _parse_spans_from_response(resp: Any) -> list[CanonicalSpan]:
    content = resp.choices[0].message.content or "[]"
    content = _strip_code_fences(content)
    try:
        raw = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"LLM returned invalid JSON: {exc}. Raw response: {content[:500]}"
        ) from exc
    if not isinstance(raw, list):
        raise ValueError(
            f"Expected JSON array from LLM, got {type(raw).__name__}: {content[:500]}"
        )
    return [
        CanonicalSpan(start=s["start"], end=s["end"], label=s["label"], text=s["text"])
        for s in raw
    ]


def _build_api_base_candidates(api_base: str | None) -> list[str | None]:
    if not api_base:
        return [None]
    normalized = api_base.rstrip("/")
    candidates: list[str | None] = [normalized]
    if not normalized.endswith("/v1"):
        candidates.append(f"{normalized}/v1")
    return candidates


def run_llm_with_metadata(
    text: str,
    api_key: str,
    api_base: str | None = None,
    model: str = "openai/gpt-4o-mini",
    system_prompt: str = SYSTEM_PROMPT,
    temperature: float = 0.0,
    reasoning_effort: ReasoningEffort = "none",
    anthropic_thinking: bool = False,
    anthropic_thinking_budget_tokens: int | None = None,
) -> LLMRunResult:
    """Run LLM-based PII detection with provider-aware advanced parameters."""
    provider = _infer_provider(model)
    supports_reasoning = _supports_reasoning_effort(model, provider)
    supports_thinking = _supports_anthropic_thinking(model, provider)
    use_openai_gateway_format = bool(api_base) and "/" not in model and "." in model

    warnings: list[str] = []
    extra_body: dict[str, Any] = {}
    effective_temperature: float | None = temperature

    if anthropic_thinking and supports_thinking:
        if temperature != 1.0:
            warnings.append(
                f"Model '{model}' requires temperature=1 when thinking is enabled; adjusted automatically."
            )
        effective_temperature = 1.0
    elif not _supports_custom_temperature(model):
        effective_temperature = None

    base_request_kwargs: dict[str, Any] = {
        "model": f"openai/{model}" if use_openai_gateway_format else model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        "api_key": api_key,
    }
    if effective_temperature is not None:
        base_request_kwargs["temperature"] = effective_temperature
    elif temperature != 1.0:
        warnings.append(
            f"Model '{model}' only supports default temperature; custom value ignored."
        )

    if reasoning_effort != "none":
        if supports_reasoning:
            if use_openai_gateway_format:
                extra_body["reasoning_effort"] = reasoning_effort
            else:
                base_request_kwargs["reasoning_effort"] = reasoning_effort
        else:
            warnings.append(
                f"Model '{model}' does not support reasoning_effort; ignored."
            )

    if anthropic_thinking:
        if supports_thinking:
            thinking: dict[str, Any] = {"type": "enabled"}
            if anthropic_thinking_budget_tokens is not None:
                thinking["budget_tokens"] = anthropic_thinking_budget_tokens
                if use_openai_gateway_format:
                    extra_body["max_tokens"] = max(
                        anthropic_thinking_budget_tokens + 256, 4096
                    )
                else:
                    base_request_kwargs["max_tokens"] = max(
                        anthropic_thinking_budget_tokens + 256, 4096
                    )
            if use_openai_gateway_format:
                extra_body["thinking"] = thinking
            else:
                base_request_kwargs["thinking"] = thinking
        else:
            warnings.append(f"Model '{model}' does not support anthropic thinking; ignored.")
    if extra_body:
        base_request_kwargs["extra_body"] = extra_body

    api_base_candidates = _build_api_base_candidates(api_base)
    last_exc: Exception | None = None

    for idx, candidate in enumerate(api_base_candidates):
        request_kwargs = dict(base_request_kwargs)
        if candidate:
            request_kwargs["api_base"] = candidate

        try:
            resp = completion(**request_kwargs)
            spans = _parse_spans_from_response(resp)
            if idx > 0:
                warnings.append(
                    f"Primary api_base failed; succeeded after retrying with '{candidate}'."
                )
            return LLMRunResult(spans=spans, warnings=warnings)
        except Exception as exc:
            last_exc = exc
            if idx < len(api_base_candidates) - 1:
                continue
            raise

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("LLM request failed unexpectedly without exception")


def run_llm(
    text: str,
    api_key: str,
    api_base: str | None = None,
    model: str = "openai/gpt-4o-mini",
    system_prompt: str = SYSTEM_PROMPT,
    temperature: float = 0.0,
) -> list[CanonicalSpan]:
    """Backward-compatible span-only wrapper."""
    result = run_llm_with_metadata(
        text=text,
        api_key=api_key,
        api_base=api_base,
        model=model,
        system_prompt=system_prompt,
        temperature=temperature,
    )
    return result.spans


# Backward compatibility alias
run_openai = run_llm
