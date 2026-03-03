from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
import math
import re
from typing import Any, Literal

from litellm import completion, get_llm_provider

from models import CanonicalSpan, LLMConfidenceMetric

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

FORMAT_GUARDRAIL = """\
STRICT OUTPUT REQUIREMENTS:
- Return valid JSON only (no prose, no markdown, no code fences).
- Top-level must be either:
  1) a JSON array of spans, OR
  2) a JSON object with key "spans" containing that array.
- Each span item MUST include: start (int), end (int), label (str), text (str).
- Do not include confidence fields or extra commentary.
"""

VERIFIER_PROMPT = """\
You are a PII verification analyst. You will receive transcript text and candidate
PII spans. Keep a candidate unless you are highly confident it is not PII.
Remove only obvious false positives such as math expressions, public historical
figures, theorem names, generic context locations, or assignment labels.
Output only JSON with key "decisions", where each item is:
{"index": <int>, "keep": <boolean>}.
"""

METHOD_EXTENDED_PROMPT = """\
You are a PII/PHI analyst reviewing tutoring transcripts.
Detect only explicit PII spans and avoid educational/math context false positives.
Do not label equations, formula symbols, section references, grades, or generic
historical/public figure names as PII.
"""

METHOD_DUAL_NAMES_PROMPT = """\
Detect personal names only. Include private individual names used in dialogue.
Do not include historical/public figures, theorem names, or fictional names.
"""

METHOD_DUAL_IDENTIFIERS_PROMPT = """\
Detect contact and identifier spans: EMAIL, URL, ADDRESS, SSN, PHONE_NUMBER,
FAX_NUMBER, ACCOUNT_NUMBER, DEVICE_IDENTIFIER, IP_ADDRESS, IDENTIFYING_NUMBER.
Ignore math/contextual numbers, scores, and section references.
"""

METHOD_DUAL_NAMES_LOCATION_PROMPT = """\
Detect only NAME and ADDRESS spans. ADDRESS includes specific identifying locations.
Ignore generic country/state/region mentions used as broad context.
"""

METHOD_DUAL_NUMBERS_PROMPT = """\
Detect only numeric/contact identifiers: EMAIL, URL, SSN, PHONE_NUMBER, FAX_NUMBER,
ACCOUNT_NUMBER, DEVICE_IDENTIFIER, IP_ADDRESS, IDENTIFYING_NUMBER.
Ignore educational/math numbers and formula content.
"""

METHOD_SPLIT_LLM_PROMPT = """\
Detect PII spans except common contact identifiers that are handled by Presidio.
Focus on NAME, ADDRESS, DATE, SSN, ACCOUNT_NUMBER, DEVICE_IDENTIFIER,
BIOMETRIC_IDENTIFIER, IMAGE, and IDENTIFYING_NUMBER.
"""


ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh"]

CONFIDENCE_HIGH_THRESHOLD = 0.9
CONFIDENCE_MEDIUM_THRESHOLD = 0.75


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

TOOL_LABEL_MAP: dict[str, str] = {
    "PERSON": "NAME",
    "PER": "NAME",
    "NAME": "NAME",
    "ADDRESS": "LOCATION",
    "LOCATION": "LOCATION",
    "SCHOOL": "SCHOOL",
    "DATE": "DATE",
    "DATE_TIME": "DATE",
    "AGE": "AGE",
    "PHONE": "PHONE",
    "PHONE_NUMBER": "PHONE",
    "FAX_NUMBER": "PHONE",
    "EMAIL": "EMAIL",
    "EMAIL_ADDRESS": "EMAIL",
    "URL": "URL",
    "SSN": "MISC_ID",
    "US_SSN": "MISC_ID",
    "ACCOUNT_NUMBER": "MISC_ID",
    "US_BANK_NUMBER": "MISC_ID",
    "CREDIT_CARD": "MISC_ID",
    "DEVICE_IDENTIFIER": "MISC_ID",
    "IP_ADDRESS": "MISC_ID",
    "BIOMETRIC_IDENTIFIER": "MISC_ID",
    "IMAGE": "MISC_ID",
    "IDENTIFYING_NUMBER": "MISC_ID",
    "MISC_ID": "MISC_ID",
}

PRESIDIO_DEFAULT_MODEL = "en_core_web_lg"
PRESIDIO_SETUP_MESSAGE = (
    "Presidio methods require local setup. Run:\n"
    "1) uv add presidio-analyzer spacy\n"
    "2) uv run python -m spacy download en_core_web_lg"
)


METHOD_DEFINITIONS: list[dict[str, Any]] = [
    {
        "id": "default",
        "label": "Default",
        "description": "Single-pass LLM baseline.",
        "requires_presidio": False,
        "uses_llm": True,
        "supports_verify_override": True,
        "default_verify": False,
        "passes": [
            {"kind": "llm", "prompt": SYSTEM_PROMPT, "entity_types": None},
        ],
    },
    {
        "id": "extended",
        "label": "Extended",
        "description": "Single-pass LLM with stricter false-positive guidance.",
        "requires_presidio": False,
        "uses_llm": True,
        "supports_verify_override": True,
        "default_verify": False,
        "passes": [
            {"kind": "llm", "prompt": METHOD_EXTENDED_PROMPT, "entity_types": None},
        ],
    },
    {
        "id": "verified",
        "label": "Verified",
        "description": "Default LLM pass plus LLM verifier filtering.",
        "requires_presidio": False,
        "uses_llm": True,
        "supports_verify_override": True,
        "default_verify": True,
        "passes": [
            {"kind": "llm", "prompt": SYSTEM_PROMPT, "entity_types": None},
        ],
    },
    {
        "id": "dual",
        "label": "Dual",
        "description": "Two-pass LLM (names + identifiers).",
        "requires_presidio": False,
        "uses_llm": True,
        "supports_verify_override": True,
        "default_verify": False,
        "passes": [
            {
                "kind": "llm",
                "prompt": METHOD_DUAL_NAMES_PROMPT,
                "entity_types": ["NAME"],
            },
            {
                "kind": "llm",
                "prompt": METHOD_DUAL_IDENTIFIERS_PROMPT,
                "entity_types": [
                    "EMAIL",
                    "URL",
                    "ADDRESS",
                    "SSN",
                    "PHONE_NUMBER",
                    "FAX_NUMBER",
                    "ACCOUNT_NUMBER",
                    "DEVICE_IDENTIFIER",
                    "IP_ADDRESS",
                    "IDENTIFYING_NUMBER",
                ],
            },
        ],
    },
    {
        "id": "dual-split",
        "label": "Dual Split",
        "description": "Two-pass LLM split by names/locations and numeric identifiers.",
        "requires_presidio": False,
        "uses_llm": True,
        "supports_verify_override": True,
        "default_verify": False,
        "passes": [
            {
                "kind": "llm",
                "prompt": METHOD_DUAL_NAMES_LOCATION_PROMPT,
                "entity_types": ["NAME", "ADDRESS"],
            },
            {
                "kind": "llm",
                "prompt": METHOD_DUAL_NUMBERS_PROMPT,
                "entity_types": [
                    "EMAIL",
                    "URL",
                    "SSN",
                    "PHONE_NUMBER",
                    "FAX_NUMBER",
                    "ACCOUNT_NUMBER",
                    "DEVICE_IDENTIFIER",
                    "IP_ADDRESS",
                    "IDENTIFYING_NUMBER",
                ],
            },
        ],
    },
    {
        "id": "presidio",
        "label": "Presidio",
        "description": "Presidio-only analyzer.",
        "requires_presidio": True,
        "uses_llm": False,
        "supports_verify_override": False,
        "default_verify": False,
        "passes": [
            {
                "kind": "presidio",
                "entity_types": [
                    "PERSON",
                    "LOCATION",
                    "EMAIL_ADDRESS",
                    "URL",
                    "PHONE_NUMBER",
                    "IP_ADDRESS",
                ],
            },
        ],
    },
    {
        "id": "presidio+default",
        "label": "Presidio + Default",
        "description": "Presidio identifiers combined with default LLM pass.",
        "requires_presidio": True,
        "uses_llm": True,
        "supports_verify_override": True,
        "default_verify": False,
        "passes": [
            {
                "kind": "presidio",
                "entity_types": [
                    "EMAIL_ADDRESS",
                    "URL",
                    "PHONE_NUMBER",
                    "IP_ADDRESS",
                    "US_SSN",
                    "CREDIT_CARD",
                    "US_BANK_NUMBER",
                    "IBAN_CODE",
                ],
            },
            {"kind": "llm", "prompt": SYSTEM_PROMPT, "entity_types": None},
        ],
    },
    {
        "id": "presidio+llm-split",
        "label": "Presidio + LLM Split",
        "description": "Presidio for contact identifiers plus split LLM pass for remaining PII.",
        "requires_presidio": True,
        "uses_llm": True,
        "supports_verify_override": True,
        "default_verify": False,
        "passes": [
            {
                "kind": "presidio",
                "entity_types": [
                    "EMAIL_ADDRESS",
                    "URL",
                    "PHONE_NUMBER",
                    "IP_ADDRESS",
                ],
            },
            {
                "kind": "llm",
                "prompt": METHOD_SPLIT_LLM_PROMPT,
                "entity_types": [
                    "NAME",
                    "ADDRESS",
                    "DATE",
                    "SSN",
                    "ACCOUNT_NUMBER",
                    "DEVICE_IDENTIFIER",
                    "BIOMETRIC_IDENTIFIER",
                    "IMAGE",
                    "IDENTIFYING_NUMBER",
                ],
            },
        ],
    },
]

METHOD_DEFINITION_BY_ID: dict[str, dict[str, Any]] = {
    method["id"]: method for method in METHOD_DEFINITIONS
}


@dataclass
class LLMRunResult:
    spans: list[CanonicalSpan]
    warnings: list[str]
    llm_confidence: LLMConfidenceMetric


@dataclass
class MethodRunResult:
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


def _supports_logprobs(model: str, provider: str) -> bool:
    if not _is_openai_model(provider, model):
        return False
    # GPT-5 family may reject logprobs; avoid noisy retry warnings.
    if model.startswith("openai.gpt-5") or model.startswith("openai/gpt-5"):
        return False
    return True


def _is_openai_model(provider: str, model: str) -> bool:
    if provider == "openai":
        return True
    return model.startswith("openai.") or model.startswith("openai/")


def _build_unavailable_confidence_metric(
    provider: str,
    model: str,
    reason: Literal["unsupported_provider", "missing_logprobs", "empty_completion"],
) -> LLMConfidenceMetric:
    metric_provider = "openai" if _is_openai_model(provider, model) else provider
    return LLMConfidenceMetric(
        available=False,
        provider=metric_provider,
        model=model,
        reason=reason,
        token_count=0,
        mean_logprob=None,
        confidence=None,
        perplexity=None,
        band="na",
        high_threshold=CONFIDENCE_HIGH_THRESHOLD,
        medium_threshold=CONFIDENCE_MEDIUM_THRESHOLD,
    )


def _compute_llm_confidence_metric(
    resp: Any,
    provider: str,
    model: str,
) -> LLMConfidenceMetric:
    if not _is_openai_model(provider, model):
        return _build_unavailable_confidence_metric(provider, model, "unsupported_provider")

    message_content = resp.choices[0].message.content or ""
    normalized_content = message_content.strip()
    logprob_container = getattr(resp.choices[0], "logprobs", None)
    token_logprobs = (
        getattr(logprob_container, "content", None) if logprob_container is not None else None
    )

    if token_logprobs is None:
        missing_reason: Literal["missing_logprobs", "empty_completion"] = (
            "empty_completion" if normalized_content == "" else "missing_logprobs"
        )
        return _build_unavailable_confidence_metric("openai", model, missing_reason)

    numeric_logprobs: list[float] = []
    for token in token_logprobs:
        value = getattr(token, "logprob", None)
        if isinstance(value, (int, float)) and math.isfinite(value):
            numeric_logprobs.append(float(value))

    if not numeric_logprobs:
        missing_reason = "empty_completion" if normalized_content == "" else "missing_logprobs"
        return _build_unavailable_confidence_metric("openai", model, missing_reason)

    mean_logprob = sum(numeric_logprobs) / len(numeric_logprobs)
    confidence = math.exp(mean_logprob)
    perplexity = math.exp(-mean_logprob)
    if confidence >= CONFIDENCE_HIGH_THRESHOLD:
        band: Literal["high", "medium", "low"] = "high"
    elif confidence >= CONFIDENCE_MEDIUM_THRESHOLD:
        band = "medium"
    else:
        band = "low"

    return LLMConfidenceMetric(
        available=True,
        provider="openai",
        model=model,
        reason="ok",
        token_count=len(numeric_logprobs),
        mean_logprob=mean_logprob,
        confidence=confidence,
        perplexity=perplexity,
        band=band,
        high_threshold=CONFIDENCE_HIGH_THRESHOLD,
        medium_threshold=CONFIDENCE_MEDIUM_THRESHOLD,
    )


def _parse_spans_from_response(resp: Any) -> list[CanonicalSpan]:
    content = resp.choices[0].message.content or "[]"
    content = _strip_code_fences(content)

    def _extract_json_candidate(text: str) -> str | None:
        starts = [idx for idx, ch in enumerate(text) if ch in "[{"]
        for start in starts:
            stack: list[str] = []
            for idx in range(start, len(text)):
                ch = text[idx]
                if ch in "[{":
                    stack.append("]" if ch == "[" else "}")
                elif ch in "]}":
                    if not stack or ch != stack[-1]:
                        break
                    stack.pop()
                    if not stack:
                        return text[start : idx + 1]
        return None

    try:
        raw: Any = json.loads(content)
    except json.JSONDecodeError:
        candidate = _extract_json_candidate(content)
        if not candidate:
            raise ValueError(
                "LLM returned non-JSON output. "
                "Ensure prompt requests strict JSON spans with start/end/label/text. "
                f"Raw response: {content[:500]}"
            )
        try:
            raw = json.loads(candidate)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"LLM returned invalid JSON: {exc}. Raw response: {content[:500]}"
            ) from exc

    if isinstance(raw, dict):
        if "spans" not in raw:
            raise ValueError(
                f"Expected top-level array or object with 'spans', got keys: {list(raw.keys())[:10]}"
            )
        raw = raw["spans"]

    if not isinstance(raw, list):
        raise ValueError(
            f"Expected JSON array from LLM, got {type(raw).__name__}: {content[:500]}"
        )
    return [
        CanonicalSpan(start=s["start"], end=s["end"], label=s["label"], text=s["text"])
        for s in raw
    ]


def _is_unsupported_param_error(exc: Exception, param: str) -> bool:
    message = str(exc).lower()
    return (
        "unsupportedparamserror" in message
        and param.lower() in message
    ) or (
        "does not support parameters" in message
        and param.lower() in message
    )


def _build_api_base_candidates(api_base: str | None) -> list[str | None]:
    if not api_base:
        return [None]
    normalized = api_base.rstrip("/")
    candidates: list[str | None] = [normalized]
    if not normalized.endswith("/v1"):
        candidates.append(f"{normalized}/v1")
    return candidates


def normalize_method_label(label: str) -> str:
    normalized = label.strip().upper()
    if not normalized:
        return "MISC_ID"
    return TOOL_LABEL_MAP.get(normalized, "MISC_ID")


def normalize_method_spans(spans: list[CanonicalSpan]) -> list[CanonicalSpan]:
    normalized: list[CanonicalSpan] = []
    for span in spans:
        normalized.append(
            CanonicalSpan(
                start=span.start,
                end=span.end,
                label=normalize_method_label(span.label),
                text=span.text,
            )
        )
    return normalized


def merge_method_spans(spans: list[CanonicalSpan]) -> list[CanonicalSpan]:
    if not spans:
        return []

    indexed = list(enumerate(spans))
    indexed.sort(key=lambda item: (item[1].start, -(item[1].end - item[1].start), item[0]))

    merged: list[CanonicalSpan] = []
    for _, span in indexed:
        if not merged:
            merged.append(span)
            continue

        last = merged[-1]
        if span.start < last.end:
            span_len = span.end - span.start
            last_len = last.end - last.start
            if span_len > last_len:
                merged[-1] = span
            continue
        merged.append(span)

    # Remove exact duplicates, preserving deterministic order.
    seen: set[tuple[int, int, str]] = set()
    deduped: list[CanonicalSpan] = []
    for span in merged:
        key = (span.start, span.end, span.label)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(span)
    return deduped


def _presidio_runtime_error() -> str | None:
    if importlib.util.find_spec("presidio_analyzer") is None:
        return PRESIDIO_SETUP_MESSAGE
    if importlib.util.find_spec("spacy") is None:
        return PRESIDIO_SETUP_MESSAGE

    try:
        import spacy

        spacy.load(PRESIDIO_DEFAULT_MODEL)
    except Exception:
        return (
            "Presidio requires spaCy model 'en_core_web_lg'. Run:\n"
            "uv run python -m spacy download en_core_web_lg"
        )
    return None


def list_agent_methods() -> list[dict[str, Any]]:
    presidio_error = _presidio_runtime_error()
    methods: list[dict[str, Any]] = []
    for method in METHOD_DEFINITIONS:
        unavailable_reason = None
        if method["requires_presidio"] and presidio_error:
            unavailable_reason = presidio_error
        methods.append(
            {
                "id": method["id"],
                "label": method["label"],
                "description": method["description"],
                "requires_presidio": method["requires_presidio"],
                "uses_llm": method["uses_llm"],
                "supports_verify_override": method["supports_verify_override"],
                "available": unavailable_reason is None,
                "unavailable_reason": unavailable_reason,
            }
        )
    return methods


def _parse_verifier_decisions(content: str, total: int) -> set[int]:
    stripped = _strip_code_fences(content)
    payload = json.loads(stripped)
    if not isinstance(payload, dict):
        return set()
    decisions = payload.get("decisions", [])
    if not isinstance(decisions, list):
        return set()
    rejected: set[int] = set()
    for item in decisions:
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        keep = item.get("keep", True)
        if isinstance(index, int) and 0 <= index < total and keep is False:
            rejected.add(index)
    return rejected


def _run_llm_verifier(
    text: str,
    spans: list[CanonicalSpan],
    *,
    api_key: str,
    api_base: str | None,
    model: str,
) -> tuple[list[CanonicalSpan], list[str]]:
    if not spans:
        return [], []

    candidates = "\n".join(
        [
            f'[{i}] {span.label}: "{span.text}" (chars {span.start}-{span.end})'
            for i, span in enumerate(spans)
        ]
    )
    user_message = f"TEXT:\n{text}\n\nCANDIDATES:\n{candidates}"
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "pii_verification",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "decisions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "index": {"type": "integer"},
                                "keep": {"type": "boolean"},
                            },
                            "required": ["index", "keep"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["decisions"],
                "additionalProperties": False,
            },
        },
    }

    use_openai_gateway_format = bool(api_base) and "/" not in model and "." in model
    base_request_kwargs: dict[str, Any] = {
        "model": f"openai/{model}" if use_openai_gateway_format else model,
        "messages": [
            {"role": "system", "content": VERIFIER_PROMPT},
            {"role": "user", "content": user_message},
        ],
        "response_format": response_format,
        "api_key": api_key,
    }

    warnings: list[str] = []
    api_base_candidates = _build_api_base_candidates(api_base)
    last_exc: Exception | None = None
    for idx, candidate in enumerate(api_base_candidates):
        request_kwargs = dict(base_request_kwargs)
        if candidate:
            request_kwargs["api_base"] = candidate
        try:
            resp = completion(**request_kwargs)
            message_content = resp.choices[0].message.content or '{"decisions":[]}'
            rejected = _parse_verifier_decisions(message_content, len(spans))
            filtered = [span for i, span in enumerate(spans) if i not in rejected]
            if idx > 0:
                warnings.append(
                    f"Verifier retried with '{candidate}' after primary api_base failure."
                )
            return filtered, warnings
        except Exception as exc:
            last_exc = exc
            if idx < len(api_base_candidates) - 1:
                continue
            raise

    if last_exc is not None:
        raise last_exc
    return spans, warnings


def _run_presidio_pass(
    text: str,
    entity_types: list[str] | None = None,
) -> list[CanonicalSpan]:
    runtime_error = _presidio_runtime_error()
    if runtime_error is not None:
        raise RuntimeError(runtime_error)

    from presidio_analyzer import AnalyzerEngine
    from presidio_analyzer.nlp_engine import NlpEngineProvider

    config = {
        "nlp_engine_name": "spacy",
        "models": [{"lang_code": "en", "model_name": PRESIDIO_DEFAULT_MODEL}],
    }
    nlp_engine = NlpEngineProvider(nlp_configuration=config).create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
    results = analyzer.analyze(
        text=text,
        language="en",
        entities=entity_types,
        score_threshold=0.0,
    )
    spans: list[CanonicalSpan] = []
    for result in results:
        if result.start >= result.end:
            continue
        spans.append(
            CanonicalSpan(
                start=result.start,
                end=result.end,
                label=str(result.entity_type),
                text=text[result.start : result.end],
            )
        )
    return spans


def _filter_method_pass_spans(
    spans: list[CanonicalSpan],
    entity_types: list[str] | None,
) -> list[CanonicalSpan]:
    if not entity_types:
        return spans
    allowed = {normalize_method_label(entity_type) for entity_type in entity_types}
    return [span for span in spans if normalize_method_label(span.label) in allowed]


def run_method_with_metadata(
    *,
    text: str,
    method_id: str,
    api_key: str | None,
    api_base: str | None,
    model: str,
    system_prompt: str,
    temperature: float,
    reasoning_effort: ReasoningEffort,
    anthropic_thinking: bool,
    anthropic_thinking_budget_tokens: int | None,
    method_verify: bool | None,
) -> MethodRunResult:
    method = METHOD_DEFINITION_BY_ID.get(method_id)
    if method is None:
        raise ValueError(f"Unknown method: {method_id}")

    if method["requires_presidio"]:
        runtime_error = _presidio_runtime_error()
        if runtime_error is not None:
            raise ValueError(runtime_error)

    if method["uses_llm"] and not api_key:
        raise ValueError(
            f"Method '{method_id}' requires an API key. Set LITELLM_API_KEY or provide api_key."
        )

    should_verify = method["default_verify"] if method_verify is None else method_verify
    if should_verify and not api_key:
        raise ValueError(
            f"Method '{method_id}' verification requires an API key. Set LITELLM_API_KEY or provide api_key."
        )

    warnings: list[str] = []
    all_spans: list[CanonicalSpan] = []

    for idx, method_pass in enumerate(method["passes"]):
        kind = method_pass["kind"]
        if kind == "presidio":
            pass_spans = _run_presidio_pass(
                text=text,
                entity_types=method_pass.get("entity_types"),
            )
            all_spans.extend(normalize_method_spans(pass_spans))
            continue

        prompt = str(method_pass.get("prompt") or SYSTEM_PROMPT)
        if system_prompt.strip():
            prompt = f"{prompt}\n\nAdditional constraints:\n{system_prompt.strip()}"
        llm_prompt = f"{prompt}\n\n{FORMAT_GUARDRAIL}"

        llm_result = run_llm_with_metadata(
            text=text,
            api_key=api_key or "",
            api_base=api_base,
            model=model,
            system_prompt=llm_prompt,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            anthropic_thinking=anthropic_thinking,
            anthropic_thinking_budget_tokens=anthropic_thinking_budget_tokens,
        )
        pass_spans = _filter_method_pass_spans(
            normalize_method_spans(llm_result.spans),
            method_pass.get("entity_types"),
        )
        all_spans.extend(pass_spans)
        for warning in llm_result.warnings:
            warnings.append(f"Pass {idx + 1}: {warning}")

    merged = merge_method_spans(all_spans)

    if should_verify:
        try:
            verified_spans, verify_warnings = _run_llm_verifier(
                text=text,
                spans=merged,
                api_key=api_key or "",
                api_base=api_base,
                model=model,
            )
            merged = merge_method_spans(normalize_method_spans(verified_spans))
            warnings.extend(verify_warnings)
        except Exception as exc:
            raise ValueError(f"Method '{method_id}' verifier failed: {exc}") from exc

    return MethodRunResult(spans=merged, warnings=warnings)


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
    is_openai_model = _is_openai_model(provider, model)
    supports_reasoning = _supports_reasoning_effort(model, provider)
    supports_thinking = _supports_anthropic_thinking(model, provider)
    supports_custom_temperature = _supports_custom_temperature(model)
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
    elif not supports_custom_temperature:
        effective_temperature = None

    base_request_kwargs: dict[str, Any] = {
        "model": f"openai/{model}" if use_openai_gateway_format else model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        "api_key": api_key,
    }
    if _supports_logprobs(model, provider):
        base_request_kwargs["logprobs"] = True
    if effective_temperature is not None:
        base_request_kwargs["temperature"] = effective_temperature
    elif not supports_custom_temperature and temperature not in (0.0, 1.0):
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
            llm_confidence = _compute_llm_confidence_metric(resp, provider, model)
            if idx > 0:
                warnings.append(
                    f"Primary api_base failed; succeeded after retrying with '{candidate}'."
                )
            return LLMRunResult(
                spans=spans,
                warnings=warnings,
                llm_confidence=llm_confidence,
            )
        except Exception as exc:
            if request_kwargs.get("logprobs") and _is_unsupported_param_error(exc, "logprobs"):
                fallback_kwargs = dict(request_kwargs)
                fallback_kwargs.pop("logprobs", None)
                try:
                    resp = completion(**fallback_kwargs)
                    spans = _parse_spans_from_response(resp)
                    llm_confidence = _compute_llm_confidence_metric(resp, provider, model)
                    warnings.append(
                        f"Model '{model}' rejected logprobs; retried without logprobs."
                    )
                    if idx > 0:
                        warnings.append(
                            f"Primary api_base failed; succeeded after retrying with '{candidate}'."
                        )
                    return LLMRunResult(
                        spans=spans,
                        warnings=warnings,
                        llm_confidence=llm_confidence,
                    )
                except Exception as retry_exc:
                    last_exc = retry_exc
                    if idx < len(api_base_candidates) - 1:
                        continue
                    raise
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
