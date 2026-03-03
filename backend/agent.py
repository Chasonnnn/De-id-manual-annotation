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

JSON_REPAIR_PROMPT = """\
You repair malformed JSON output for a PII span extractor.
Return valid JSON only.
- Top-level must be either:
  1) {"spans": [...]}
  2) [...]
- Each span must include: start (int), end (int), label (str), text (str).
- Do not add prose, markdown, or extra keys.
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

SIMPLE_LABELS: list[str] = [
    "AGE",
    "DATE",
    "EMAIL",
    "LOCATION",
    "MISC_ID",
    "NAME",
    "PHONE",
    "SCHOOL",
    "URL",
]
SIMPLE_LABEL_SET = set(SIMPLE_LABELS)

ADVANCED_LABELS: list[str] = [
    "AGE",
    "COURSE",
    "DATE",
    "EMAIL_ADDRESS",
    "GRADE_LEVEL",
    "IP_ADDRESS",
    "LOCATION",
    "NRP",
    "PERSON",
    "PHONE_NUMBER",
    "SCHOOL",
    "SOCIAL_HANDLE",
    "URL",
    "US_BANK_NUMBER",
    "US_DRIVER_LICENSE",
    "US_PASSPORT",
    "US_SSN",
]
ADVANCED_LABEL_SET = set(ADVANCED_LABELS)

ADVANCED_TOOL_LABEL_MAP: dict[str, str] = {
    "AGE": "AGE",
    "COURSE": "COURSE",
    "DATE": "DATE",
    "DATE_TIME": "DATE",
    "EMAIL": "EMAIL_ADDRESS",
    "EMAIL_ADDRESS": "EMAIL_ADDRESS",
    "GRADE_LEVEL": "GRADE_LEVEL",
    "IP_ADDRESS": "IP_ADDRESS",
    "LOCATION": "LOCATION",
    "ADDRESS": "LOCATION",
    "GPE": "LOCATION",
    "NRP": "NRP",
    "NATIONALITY": "NRP",
    "NORP": "NRP",
    "RELIGION": "NRP",
    "POLITICAL_GROUP": "NRP",
    "PERSON": "PERSON",
    "PER": "PERSON",
    "NAME": "PERSON",
    "PHONE": "PHONE_NUMBER",
    "PHONE_NUMBER": "PHONE_NUMBER",
    "FAX_NUMBER": "PHONE_NUMBER",
    "SCHOOL": "SCHOOL",
    "SOCIAL_HANDLE": "SOCIAL_HANDLE",
    "URL": "URL",
    "US_BANK_NUMBER": "US_BANK_NUMBER",
    "ACCOUNT_NUMBER": "US_BANK_NUMBER",
    "US_DRIVER_LICENSE": "US_DRIVER_LICENSE",
    "US_PASSPORT": "US_PASSPORT",
    "US_SSN": "US_SSN",
    "SSN": "US_SSN",
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


def _extract_text_from_content_payload(value: Any) -> str:
    """Best-effort extraction of model text payloads across provider shapes."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            extracted = _extract_text_from_content_payload(item)
            if extracted.strip():
                parts.append(extracted)
        return "\n".join(parts)
    if isinstance(value, dict):
        block_type = str(value.get("type", "")).strip().lower()
        if block_type in {"thinking", "reasoning", "redacted_thinking"}:
            return ""

        if "spans" in value:
            try:
                return json.dumps(value)
            except Exception:
                return str(value)

        for key in ("text", "output_text", "content", "value", "arguments", "input"):
            if key not in value:
                continue
            extracted = _extract_text_from_content_payload(value.get(key))
            if extracted.strip():
                return extracted
        return ""

    text_attr = getattr(value, "text", None)
    if isinstance(text_attr, str):
        return text_attr

    content_attr = getattr(value, "content", None)
    if isinstance(content_attr, (str, list, dict)):
        extracted = _extract_text_from_content_payload(content_attr)
        if extracted.strip():
            return extracted

    function_attr = getattr(value, "function", None)
    if function_attr is not None:
        arguments_attr = getattr(function_attr, "arguments", None)
        if isinstance(arguments_attr, (str, list, dict)):
            extracted = _extract_text_from_content_payload(arguments_attr)
            if extracted.strip():
                return extracted

    return ""


def _extract_response_content(resp: Any) -> str:
    """Extract assistant textual output from LiteLLM responses."""
    choices = getattr(resp, "choices", None)
    if choices:
        message = getattr(choices[0], "message", None)
        if message is not None:
            extracted = _extract_text_from_content_payload(getattr(message, "content", None))
            if extracted.strip():
                return extracted
            tool_calls = getattr(message, "tool_calls", None)
            if isinstance(tool_calls, (str, list, dict)):
                extracted = _extract_text_from_content_payload(tool_calls)
                if extracted.strip():
                    return extracted

    output_text = getattr(resp, "output_text", None)
    if isinstance(output_text, (str, list, dict)):
        extracted = _extract_text_from_content_payload(output_text)
        if extracted.strip():
            return extracted

    return ""


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

    message_content = _extract_response_content(resp)
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
    content = _extract_response_content(resp) or "[]"
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


def _build_span_response_format(
    *,
    label_profile: Literal["simple", "advanced"] = "simple",
) -> dict[str, Any]:
    label_schema: dict[str, Any]
    if label_profile == "advanced":
        label_schema = {"type": "string", "enum": ADVANCED_LABELS}
    else:
        label_schema = {"type": "string"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "pii_spans",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "spans": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "start": {"type": "integer"},
                                "end": {"type": "integer"},
                                "label": label_schema,
                                "text": {"type": "string"},
                            },
                            "required": ["start", "end", "label", "text"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["spans"],
                "additionalProperties": False,
            },
        },
    }


def _extract_finish_reason(resp: Any) -> str | None:
    try:
        reason = getattr(resp.choices[0], "finish_reason", None)
    except Exception:
        return None
    if isinstance(reason, str):
        normalized = reason.strip().lower()
        return normalized or None
    return None


_COMMON_NON_NAME_WORDS: set[str] = {
    "a",
    "an",
    "and",
    "are",
    "at",
    "be",
    "because",
    "but",
    "for",
    "from",
    "go",
    "good",
    "got",
    "had",
    "has",
    "have",
    "he",
    "her",
    "here",
    "him",
    "his",
    "i",
    "in",
    "is",
    "it",
    "its",
    "just",
    "look",
    "line",
    "me",
    "minutes",
    "my",
    "no",
    "not",
    "number",
    "of",
    "ok",
    "okay",
    "on",
    "or",
    "our",
    "out",
    "she",
    "so",
    "that",
    "the",
    "their",
    "them",
    "there",
    "they",
    "this",
    "to",
    "up",
    "us",
    "we",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
    "you",
    "your",
}


def _is_plausible_name_text(value: str) -> bool:
    cleaned = value.strip(" \t\r\n.,!?;:\"'()[]{}")
    if not cleaned:
        return False
    if "\n" in cleaned or "\t" in cleaned:
        return False
    if any(ch.isdigit() for ch in cleaned):
        return False
    letters = [ch for ch in cleaned if ch.isalpha()]
    if len(letters) < 2:
        return False

    lowered = cleaned.lower()
    tokens = [token for token in re.split(r"\s+", lowered) if token]
    if len(tokens) > 3:
        return False
    if lowered in _COMMON_NON_NAME_WORDS:
        return False
    if any(token in _COMMON_NON_NAME_WORDS for token in tokens):
        return False
    if any(ch.isupper() for ch in cleaned):
        return True
    return True


def _find_occurrences(text: str, needle: str) -> list[tuple[int, int]]:
    if not needle:
        return []
    out: list[tuple[int, int]] = []
    start = 0
    while True:
        idx = text.find(needle, start)
        if idx < 0:
            break
        out.append((idx, idx + len(needle)))
        start = idx + 1
    return out


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] <= b[0] or b[1] <= a[0])


def _pick_best_occurrence(
    occurrences: list[tuple[int, int]],
    hint_start: int | None,
    taken_ranges: list[tuple[int, int]],
) -> tuple[int, int] | None:
    if not occurrences:
        return None
    available = [
        occ
        for occ in occurrences
        if all(not _overlaps(occ, taken) for taken in taken_ranges)
    ]
    candidates = available or occurrences
    if hint_start is None:
        return candidates[0]
    return min(candidates, key=lambda occ: (abs(occ[0] - hint_start), occ[0], occ[1]))


def _resolve_offsets_from_text(
    raw_text: str,
    span_text: str,
    hint_start: int | None,
    taken_ranges: list[tuple[int, int]],
) -> tuple[int, int] | None:
    variants: list[str] = []
    if span_text:
        variants.append(span_text)
    stripped = span_text.strip()
    if stripped and stripped not in variants:
        variants.append(stripped)
    collapsed_ws = " ".join(stripped.split())
    if collapsed_ws and collapsed_ws not in variants:
        variants.append(collapsed_ws)

    for variant in variants:
        occurrence = _pick_best_occurrence(
            _find_occurrences(raw_text, variant), hint_start, taken_ranges
        )
        if occurrence is not None:
            return occurrence
    return None


def _repair_offset_mismatches(
    raw_text: str,
    spans: list[CanonicalSpan],
    *,
    label_profile: Literal["simple", "advanced"] = "simple",
) -> tuple[list[CanonicalSpan], list[str]]:
    if not spans:
        return spans, []

    total = len(spans)
    n = len(raw_text)
    mismatched = 0
    realigned = 0
    dropped = 0
    repaired: list[CanonicalSpan] = []
    seen: set[tuple[int, int, str]] = set()
    taken_ranges: list[tuple[int, int]] = []

    for span in spans:
        valid_offsets = 0 <= span.start < span.end <= n
        matches_offsets = valid_offsets and raw_text[span.start : span.end] == span.text
        had_mismatch = not matches_offsets
        candidate = span

        if had_mismatch:
            mismatched += 1
            hint_start = span.start if valid_offsets else None
            resolved = _resolve_offsets_from_text(
                raw_text, span.text, hint_start, taken_ranges
            )
            if resolved is not None:
                rs, re_ = resolved
                candidate = CanonicalSpan(
                    start=rs,
                    end=re_,
                    label=span.label,
                    text=raw_text[rs:re_],
                )
                realigned += 1
            elif valid_offsets:
                candidate = CanonicalSpan(
                    start=span.start,
                    end=span.end,
                    label=span.label,
                    text=raw_text[span.start : span.end],
                )
            else:
                dropped += 1
                continue

        if had_mismatch and normalize_method_label(
            candidate.label,
            label_profile=label_profile,
        ) in {"NAME", "PERSON"}:
            if not _is_plausible_name_text(candidate.text):
                dropped += 1
                continue

        key = (candidate.start, candidate.end, candidate.label)
        if key in seen:
            continue
        seen.add(key)
        repaired.append(candidate)
        taken_ranges.append((candidate.start, candidate.end))

    if mismatched == 0:
        return spans, []

    warnings = [
        (
            f"Detected {mismatched}/{total} LLM offset mismatch(es); "
            f"realigned {realigned} span(s) using returned text."
        )
    ]
    if dropped > 0:
        warnings.append(f"Dropped {dropped} low-confidence span(s) after offset repair.")
    return repaired, warnings


def _drop_implausible_name_spans(
    spans: list[CanonicalSpan],
    *,
    label_profile: Literal["simple", "advanced"] = "simple",
) -> tuple[list[CanonicalSpan], int]:
    kept: list[CanonicalSpan] = []
    dropped = 0
    target_label = "NAME" if label_profile == "simple" else "PERSON"
    for span in spans:
        if (
            normalize_method_label(span.label, label_profile=label_profile) == target_label
            and not _is_plausible_name_text(span.text)
        ):
            dropped += 1
            continue
        kept.append(span)
    return kept, dropped


def _is_unsupported_param_error(exc: Exception, param: str) -> bool:
    message = str(exc).lower()
    return (
        "unsupportedparamserror" in message
        and param.lower() in message
    ) or (
        "does not support parameters" in message
        and param.lower() in message
    )


def _is_response_format_unsupported_error(exc: Exception) -> bool:
    message = str(exc).lower()
    if _is_unsupported_param_error(exc, "response_format"):
        return True
    return "json_schema" in message and (
        "unsupported" in message or "not support" in message
    )


def _build_api_base_candidates(api_base: str | None) -> list[str | None]:
    if not api_base:
        return [None]
    normalized = api_base.rstrip("/")
    candidates: list[str | None] = [normalized]
    if not normalized.endswith("/v1"):
        candidates.append(f"{normalized}/v1")
    return candidates


def _complete_with_supported_params(
    request_kwargs: dict[str, Any],
    *,
    model: str,
    warnings: list[str],
) -> tuple[Any, dict[str, Any]]:
    effective_kwargs = dict(request_kwargs)
    while True:
        try:
            return completion(**effective_kwargs), effective_kwargs
        except Exception as exc:
            if "response_format" in effective_kwargs and _is_response_format_unsupported_error(
                exc
            ):
                effective_kwargs.pop("response_format", None)
                warnings.append(
                    f"Model '{model}' rejected response_format/json_schema; retried without schema enforcement."
                )
                continue
            if "logprobs" in effective_kwargs and _is_unsupported_param_error(exc, "logprobs"):
                effective_kwargs.pop("logprobs", None)
                warnings.append(f"Model '{model}' rejected logprobs; retried without logprobs.")
                continue
            raise


def _parse_with_one_repair_retry(
    *,
    resp: Any,
    request_kwargs: dict[str, Any],
    text: str,
    warnings: list[str],
) -> tuple[list[CanonicalSpan], Any]:
    raw_content = _extract_response_content(resp)
    if not raw_content.strip():
        warnings.append("LLM returned empty output content; interpreted as no spans.")

    try:
        return _parse_spans_from_response(resp), resp
    except Exception as parse_exc:
        repair_kwargs = {
            key: value
            for key, value in request_kwargs.items()
            if key not in {"messages", "logprobs"}
        }
        repair_kwargs["messages"] = [
            {"role": "system", "content": JSON_REPAIR_PROMPT},
            {
                "role": "user",
                "content": (
                    "Fix this invalid extractor output into valid JSON.\n\n"
                    f"TRANSCRIPT:\n{text}\n\n"
                    f"INVALID_OUTPUT:\n{raw_content}"
                ),
            },
        ]
        try:
            repair_resp = completion(**repair_kwargs)
            repaired_spans = _parse_spans_from_response(repair_resp)
            warnings.append("Recovered invalid LLM JSON with one repair retry.")
            return repaired_spans, repair_resp
        except Exception as repair_exc:
            raise ValueError(
                f"Failed to parse LLM output and one repair retry failed: {repair_exc}"
            ) from parse_exc


_WEEKDAY_ONLY_RE = re.compile(
    r"^(monday|tuesday|wednesday|thursday|friday|saturday|sunday)$", re.IGNORECASE
)
_MONTH_ONLY_RE = re.compile(
    r"^(january|february|march|april|may|june|july|august|september|october|november|december)$",
    re.IGNORECASE,
)
_YEAR_ONLY_RE = re.compile(r"^\d{4}$")
_SPECIFIC_DATE_PATTERNS = [
    re.compile(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b"),
    re.compile(
        r"\b(?:january|february|march|april|may|june|july|august|september|october|november|december)"
        r"\s+\d{1,2}(?:,\s*\d{2,4})?\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b\d{1,2}\s+"
        r"(?:january|february|march|april|may|june|july|august|september|october|november|december)"
        r"(?:\s+\d{2,4})?\b",
        re.IGNORECASE,
    ),
]
_COURSE_RE = re.compile(r"^[A-Za-z]{2,}(?:\s+[A-Za-z]{1,})*\s+\d{2,}$", re.IGNORECASE)


def _is_specific_date_text(value: str) -> bool:
    cleaned = value.strip(" \t\r\n.,!?;:\"'()[]{}")
    if not cleaned:
        return False
    if _WEEKDAY_ONLY_RE.fullmatch(cleaned):
        return False
    if _MONTH_ONLY_RE.fullmatch(cleaned):
        return False
    if _YEAR_ONLY_RE.fullmatch(cleaned):
        return False
    return any(pattern.search(cleaned) for pattern in _SPECIFIC_DATE_PATTERNS)


def _is_valid_course_text(value: str) -> bool:
    cleaned = " ".join(value.strip().split())
    if not cleaned:
        return False
    return bool(_COURSE_RE.fullmatch(cleaned))


def _passes_label_profile_filters(
    span: CanonicalSpan,
    *,
    label_profile: Literal["simple", "advanced"] = "simple",
) -> bool:
    if label_profile != "advanced":
        return True
    if span.label == "COURSE":
        return _is_valid_course_text(span.text)
    if span.label == "DATE":
        return _is_specific_date_text(span.text)
    return True


def normalize_method_label(
    label: str,
    *,
    label_profile: Literal["simple", "advanced"] = "simple",
) -> str:
    normalized = label.strip().upper()
    if not normalized:
        return "MISC_ID" if label_profile == "simple" else ""
    if label_profile == "advanced":
        mapped = ADVANCED_TOOL_LABEL_MAP.get(normalized, normalized)
        return mapped if mapped in ADVANCED_LABEL_SET else ""
    mapped = TOOL_LABEL_MAP.get(normalized, normalized)
    return mapped if mapped in SIMPLE_LABEL_SET else "MISC_ID"


def normalize_method_spans(
    spans: list[CanonicalSpan],
    *,
    label_profile: Literal["simple", "advanced"] = "simple",
) -> list[CanonicalSpan]:
    normalized: list[CanonicalSpan] = []
    for span in spans:
        mapped_label = normalize_method_label(span.label, label_profile=label_profile)
        if not mapped_label:
            continue
        candidate = CanonicalSpan(
            start=span.start,
            end=span.end,
            label=mapped_label,
            text=span.text,
        )
        if not _passes_label_profile_filters(candidate, label_profile=label_profile):
            continue
        normalized.append(candidate)
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
            message_content = _extract_response_content(resp) or '{"decisions":[]}'
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
    *,
    label_profile: Literal["simple", "advanced"] = "simple",
) -> list[CanonicalSpan]:
    if not entity_types:
        return spans
    allowed = {
        normalize_method_label(entity_type, label_profile=label_profile)
        for entity_type in entity_types
    }
    allowed.discard("")
    return [
        span
        for span in spans
        if normalize_method_label(span.label, label_profile=label_profile) in allowed
    ]


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
    label_profile: Literal["simple", "advanced"] = "simple",
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
            all_spans.extend(normalize_method_spans(pass_spans, label_profile=label_profile))
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
            label_profile=label_profile,
        )
        pass_spans = _filter_method_pass_spans(
            normalize_method_spans(llm_result.spans, label_profile=label_profile),
            method_pass.get("entity_types"),
            label_profile=label_profile,
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
            merged = merge_method_spans(
                normalize_method_spans(verified_spans, label_profile=label_profile)
            )
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
    label_profile: Literal["simple", "advanced"] = "simple",
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
    base_request_kwargs["response_format"] = _build_span_response_format(
        label_profile=label_profile
    )
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
            resp, effective_kwargs = _complete_with_supported_params(
                request_kwargs, model=model, warnings=warnings
            )
            spans, parsed_resp = _parse_with_one_repair_retry(
                resp=resp,
                request_kwargs=effective_kwargs,
                text=text,
                warnings=warnings,
            )
            spans, repair_warnings = _repair_offset_mismatches(
                text,
                spans,
                label_profile=label_profile,
            )
            warnings.extend(repair_warnings)
            spans, dropped_name_spans = _drop_implausible_name_spans(
                spans,
                label_profile=label_profile,
            )
            if dropped_name_spans > 0:
                target_label = "NAME" if label_profile == "simple" else "PERSON"
                warnings.append(
                    f"Dropped {dropped_name_spans} implausible {target_label} span(s) from LLM output."
                )
            spans = normalize_method_spans(spans, label_profile=label_profile)
            finish_reason = _extract_finish_reason(parsed_resp)
            if finish_reason == "length":
                warnings.append(
                    "LLM output may be truncated (finish_reason=length); results can be incomplete."
                )
            llm_confidence = _compute_llm_confidence_metric(parsed_resp, provider, model)
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
