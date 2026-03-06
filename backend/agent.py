from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import json
import math
import re
from typing import Any, Literal

from litellm import completion

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
identifiable information (PII) spans. Extract explicit spans only; do not infer
missing data."""

FORMAT_GUARDRAIL = """\
STRICT OUTPUT REQUIREMENTS:
- Return valid JSON only (no prose, no markdown, no code fences).
- Return exactly one JSON object with key "spans" containing an array of spans.
- If there are no spans, return {"spans": []}.
- Each span item MUST include: start (int), end (int), label (str), text (str).
- Offsets are 0-based and "end" is exclusive.
- Do not include confidence fields or extra commentary.
"""

JSON_REPAIR_PROMPT = """\
You repair malformed JSON output for a PII span extractor.
Return valid JSON only.
- Top-level must be {"spans": [...]}.
- If there are no spans, return {"spans": []}.
- Each span must include: start (int), end (int), label (str), text (str).
- Do not add prose, markdown, or extra keys.
"""

VERIFIER_PROMPT = """\
You are a PII verification analyst. You will receive a text and a list of candidate PII
spans detected by a previous system. All candidates are assumed to be real PII.
For each candidate, decide: keep (it is real PII) or remove (it is NOT PII).
Only remove a candidate if you are very confident it is one of the following:
- A mathematical expression, variable, equation, or formula
- A historical figure, scientist, mathematician, or public figure
- A fictional character, brand name, or product name
- A course name, textbook reference, section number, or assignment label
- A date that is not tied to a specific individual (due dates, semesters, schedules)
- A score, grade, percentage, or number used in an educational or math context
- A country, state, or region mentioned as general context rather than a personal address
- A method or theorem name containing a person's name (e.g., Pythagorean theorem)
When in doubt, keep.
Output data following the provided JSON schema.
"""

METHOD_DEFAULT_PROMPT = """\
You are a PII/PHI analyst. Identify all personally identifiable information (PII) and
protected health information (PHI) in the text. Include all standard HIPAA Safe Harbor
identifiers and anything that could reasonably identify an individual
Extract explicit identifiers only; do not infer missing data. Include partial identifiers,
nicknames, or identifiers in filenames, URLs, or notes.
Categories to detect: names [NAME]; geographic subdivisions smaller than a State [ADDRESS];
dates related to an individual (except year) and ages over 89 [DATE]; phone/fax numbers
[PHONE_NUMBER] [FAX_NUMBER]; email addresses [EMAIL]; Social Security numbers [SSN]; account
numbers [ACCOUNT_NUMBER]; device identifiers [DEVICE_IDENTIFIER]; URLs [URL]; IP addresses
[IP_ADDRESS]; biometric identifiers [BIOMETRIC_IDENTIFIER]; full-face or comparable images
[IMAGE]; and any other unique number, characteristic or code that is capable of identifying
the individual [IDENTIFYING_NUMBER].
Matches should be minimal exact spans of the sensitive value; include anything that could
reasonably be considered PII.
Output data following the provided JSON schema.
"""

METHOD_EXTENDED_PROMPT = """\
You are a PII/PHI analyst reviewing tutoring chat transcripts. Identify personally
identifiable information (PII) per HIPAA Safe Harbor (§164.514(b)(2)).
Extract explicit identifiers only; do not infer missing data.
Categories to detect: names of individuals in the conversation [NAME]; geographic
subdivisions smaller than a State such as street addresses, cities, or school names that
reveal location [ADDRESS]; dates directly related to an individual such as birth dates
(except year) and ages over 89 [DATE]; phone numbers [PHONE_NUMBER]; fax numbers
[FAX_NUMBER]; email addresses [EMAIL]; Social Security numbers [SSN]; account numbers
[ACCOUNT_NUMBER]; device identifiers [DEVICE_IDENTIFIER]; URLs [URL]; IP addresses
[IP_ADDRESS]; biometric identifiers [BIOMETRIC_IDENTIFIER]; full-face or comparable
images [IMAGE]; and any other unique identifying number such as student IDs, medical
record numbers, or license numbers [IDENTIFYING_NUMBER].
DO NOT mark any of the following as PII:
- Historical figures, scientists, mathematicians (e.g., Einstein, Pythagoras, Euler)
- Public figures, celebrities, book authors
- Fictional characters
- Business names, product names, brand names, course or project titles
- Dates that are NOT tied to a specific individual: due dates, assignment dates,
class schedules, publication dates, semesters (e.g., "Fall 2024", "due Monday")
- Mathematical expressions, equations, formulas, or variables
(e.g., f(x)=ax²+bx+c, -b/2a, 6x-4, y=mx+b, 3x+7=22)
- Problem numbers, question numbers, exercise references (e.g., #3, #5, problem 4)
- Calculator model names (e.g., TI-84)
- Course section numbers, textbook references (e.g., section 1.4, chapter 3)
- Scores, grades, percentages, or any number used in an educational/math context
- Countries, states, or regions mentioned as general context rather than a personal address
- Method or theorem names containing a person's name (e.g., "Horner's rule", "L'Hôpital")
Matches should be minimal exact spans of the sensitive value.
Output data following the provided JSON schema.
"""

METHOD_DUAL_NAMES_PROMPT = """\
You are a PII analyst specializing in detecting personal names in tutoring chat transcripts.
Identify all personal names in the text. Mark each as [NAME]. This includes first names,
last names, nicknames, and pseudonyms used to address or refer to specific individuals
in the conversation.
DO NOT mark:
- Historical figures, scientists, or mathematicians (e.g., Einstein, Pythagoras, Euler)
- Book authors, public figures, or celebrities
- Fictional characters
- Method or theorem names that happen to contain a person's name (e.g., "Horner's rule")
- Email addresses, URLs, or usernames
Matches should be minimal exact spans of just the name.
Output data following the provided JSON schema.
"""

METHOD_DUAL_IDENTIFIERS_PROMPT = """\
You are a PII analyst specializing in detecting contact information, numbers, and identifiers
in tutoring chat transcripts. The text contains math problems, equations, and educational
content mixed with personal information.
Identify: email addresses [EMAIL]; URLs [URL]; physical addresses (subdivisions smaller
than a State) [ADDRESS]; Social Security numbers [SSN]; phone numbers [PHONE_NUMBER];
fax numbers [FAX_NUMBER]; financial account numbers such as bank or insurance accounts
[ACCOUNT_NUMBER]; device identifiers [DEVICE_IDENTIFIER]; IP addresses [IP_ADDRESS];
and any other unique identifying number such as student IDs, medical record numbers,
or license numbers [IDENTIFYING_NUMBER].
DO NOT mark any of the following:
- Personal names (these are handled separately)
- Mathematical expressions, equations, formulas, or variables
(e.g., f(x)=ax²+bx+c, -b/2a, 6x-4, y=mx+b)
- Problem numbers, question numbers, or exercise references (e.g., #3, #5, problem 4)
- Calculator model names (e.g., TI-84, TI-89)
- Course section numbers or textbook references (e.g., section 1.4, chapter 3)
- Scores, grades, percentages, or any number used in a math context
- Dates
- City names, regions, or countries mentioned as general locations
Matches should be minimal exact spans.
Output data following the provided JSON schema.
"""

METHOD_DUAL_NAMES_LOCATION_PROMPT = """\
You are a PII analyst specializing in detecting personal names and locations in tutoring
chat transcripts.
Identify all personal names [NAME] and geographic locations smaller than a State [ADDRESS].
Names include first names, last names, nicknames, and pseudonyms. Locations include street
addresses, city names, school names that reveal location, and neighborhood names.
DO NOT mark:
- Historical figures, scientists, or mathematicians (e.g., Einstein, Pythagoras, Euler)
- Book authors, public figures, or celebrities
- Fictional characters
- Method or theorem names (e.g., "Horner's rule")
- Countries, states, or regions used as general context
Matches should be minimal exact spans.
Output data following the provided JSON schema.
"""

METHOD_DUAL_NUMBERS_PROMPT = """\
You are a PII analyst specializing in detecting contact information and identifying numbers
in tutoring chat transcripts. The text contains math problems, equations, and educational
content mixed with personal information.
Identify: email addresses [EMAIL]; URLs [URL]; Social Security numbers [SSN]; phone numbers
[PHONE_NUMBER]; fax numbers [FAX_NUMBER]; account numbers [ACCOUNT_NUMBER];
device identifiers [DEVICE_IDENTIFIER]; IP addresses [IP_ADDRESS]; and unique identifying
numbers such as student IDs, medical record numbers, or license numbers [IDENTIFYING_NUMBER].
DO NOT mark any of the following:
- Personal names or locations (these are handled separately)
- Mathematical expressions, equations, formulas, or variables
(e.g., f(x)=ax²+bx+c, -b/2a, 6x-4, y=mx+b)
- Problem numbers, question numbers, or exercise references (e.g., #3, #5, problem 4)
- Calculator model names (e.g., TI-84, TI-89)
- Course section numbers or textbook references (e.g., section 1.4, chapter 3)
- Scores, grades, percentages, or any number used in a math context
- Dates
Matches should be minimal exact spans.
Output data following the provided JSON schema.
"""

METHOD_SPLIT_LLM_PROMPT = """\
You are a PII/PHI analyst. Identify all personally identifiable information (PII) and
protected health information (PHI) in the text. Include all standard HIPAA Safe Harbor
identifiers and anything that could reasonably identify an individual.
Extract explicit identifiers only; do not infer missing data. Include partial identifiers,
nicknames, or identifiers in filenames or notes.
Categories to detect: names [NAME]; geographic subdivisions smaller than a State [ADDRESS];
dates related to an individual (except year) and ages over 89 [DATE]; Social Security numbers
[SSN]; account numbers [ACCOUNT_NUMBER]; device identifiers [DEVICE_IDENTIFIER];
biometric identifiers [BIOMETRIC_IDENTIFIER]; full-face or comparable images [IMAGE];
and any other unique number, characteristic or code that is capable of identifying the
individual [IDENTIFYING_NUMBER].
Matches should be minimal exact spans of the sensitive value; include anything that could
reasonably be considered PII.
Output data following the provided JSON schema.
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
        "label": "OpenAI: GPT-5.4 (xhigh)",
        "model": "openai.gpt-5.4",
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
        "label": "Google: Gemini 3.1 Flash Lite Preview",
        "model": "google.gemini-3.1-flash-lite-preview",
        "provider": "gemini",
        "supports_reasoning_effort": False,
        "supports_anthropic_thinking": False,
        "default_reasoning_effort": "none",
    },
]


MODEL_PRESET_BY_ID: dict[str, dict[str, Any]] = {
    p["model"]: p for p in MODEL_PRESETS
}

PROVIDER_PREFIX_MAP: dict[str, str] = {
    "openai": "openai",
    "anthropic": "anthropic",
    "google": "gemini",
    "gemini": "gemini",
    "vertex_ai": "gemini",
    "vertex": "gemini",
}

TOOL_LABEL_MAP: dict[str, str] = {
    "PERSON": "NAME",
    "PER": "NAME",
    "NAME": "NAME",
    "PERSON_NAME": "NAME",
    "FIRST_NAME": "NAME",
    "LAST_NAME": "NAME",
    "FULL_NAME": "NAME",
    "GIVEN_NAME": "NAME",
    "SURNAME": "NAME",
    "ADDRESS": "LOCATION",
    "LOCATION": "LOCATION",
    "SCHOOL": "SCHOOL",
    "DATE": "DATE",
    "DATE_TIME": "DATE",
    "TIME": "DATE",
    "TIMESTAMP": "DATE",
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


def _labels_for_profile(
    label_profile: Literal["simple", "advanced"],
) -> list[str]:
    return ADVANCED_LABELS if label_profile == "advanced" else SIMPLE_LABELS


def build_extraction_system_prompt(
    base_prompt: str,
    label_profile: Literal["simple", "advanced"] = "simple",
) -> str:
    prompt = str(base_prompt or "").strip() or SYSTEM_PROMPT.strip()
    allowed_labels = ", ".join(_labels_for_profile(label_profile))
    return (
        f"{prompt}\n\n"
        f"{FORMAT_GUARDRAIL}\n"
        f'- Allowed labels for this run: {allowed_labels}.\n'
        '- The "label" field must be one of the allowed labels above.\n'
        '- The "text" field must exactly match transcript[start:end].\n'
    )

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
            {"kind": "llm", "prompt": METHOD_DEFAULT_PROMPT, "entity_types": None},
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
            {"kind": "llm", "prompt": METHOD_DEFAULT_PROMPT, "entity_types": None},
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
            {"kind": "llm", "prompt": METHOD_DEFAULT_PROMPT, "entity_types": None},
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
    finish_reason: str | None = None


@dataclass
class MethodRunResult:
    spans: list[CanonicalSpan]
    warnings: list[str]
    llm_confidence: LLMConfidenceMetric | None = None


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
    preset = MODEL_PRESET_BY_ID.get(model)
    if preset is not None:
        return str(preset["provider"])

    normalized = model.strip().lower()
    if not normalized:
        return "unknown"

    for separator in ("/", "."):
        if separator not in normalized:
            continue
        prefix = normalized.split(separator, 1)[0]
        if prefix in PROVIDER_PREFIX_MAP:
            return PROVIDER_PREFIX_MAP[prefix]
        if separator == "/":
            return prefix or "unknown"

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


def _supports_logprobs(
    model: str,
    provider: str,
    *,
    reasoning_effort: ReasoningEffort = "none",
) -> bool:
    if not _is_openai_model(provider, model):
        return False
    normalized = model.lower()
    # Empirically incompatible on current OpenAI-compatible gateway route.
    if normalized in {"openai.gpt-5.2-chat", "openai/gpt-5.2-chat"}:
        return False
    # GPT-5 variants can reject logprobs when non-default reasoning is enabled.
    if ("gpt-5" in normalized or "gpt5" in normalized) and reasoning_effort != "none":
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
    *,
    token_count: int = 0,
) -> LLMConfidenceMetric:
    metric_provider = "openai" if _is_openai_model(provider, model) else provider
    return LLMConfidenceMetric(
        available=False,
        provider=metric_provider,
        model=model,
        reason=reason,
        token_count=max(0, int(token_count)),
        mean_logprob=None,
        confidence=None,
        perplexity=None,
        band="na",
        high_threshold=CONFIDENCE_HIGH_THRESHOLD,
        medium_threshold=CONFIDENCE_MEDIUM_THRESHOLD,
    )


def _aggregate_llm_confidence_metrics(
    metrics: list[LLMConfidenceMetric],
) -> LLMConfidenceMetric | None:
    if not metrics:
        return None
    if len(metrics) == 1:
        return metrics[0]

    usable = [
        item
        for item in metrics
        if item.available and item.mean_logprob is not None and item.token_count > 0
    ]
    if not usable:
        return metrics[0]

    total_tokens = sum(item.token_count for item in usable)
    if total_tokens <= 0:
        return metrics[0]

    weighted_mean_logprob = (
        sum(float(item.mean_logprob or 0.0) * item.token_count for item in usable) / total_tokens
    )
    confidence = math.exp(weighted_mean_logprob)
    perplexity = math.exp(-weighted_mean_logprob)
    if confidence >= usable[0].high_threshold:
        band: Literal["high", "medium", "low", "na"] = "high"
    elif confidence >= usable[0].medium_threshold:
        band = "medium"
    else:
        band = "low"

    return LLMConfidenceMetric(
        available=True,
        provider=usable[0].provider,
        model=usable[0].model,
        reason="ok",
        token_count=total_tokens,
        mean_logprob=weighted_mean_logprob,
        confidence=confidence,
        perplexity=perplexity,
        band=band,
        high_threshold=usable[0].high_threshold,
        medium_threshold=usable[0].medium_threshold,
    )


def _compute_llm_confidence_metric(
    resp: Any,
    provider: str,
    model: str,
) -> LLMConfidenceMetric:
    if not _is_openai_model(provider, model):
        return _build_unavailable_confidence_metric(provider, model, "unsupported_provider")

    def _extract_completion_token_count(value: Any) -> int:
        usage = getattr(value, "usage", None)
        if usage is None and isinstance(value, dict):
            usage = value.get("usage")
        if usage is None:
            return 0
        candidates = ("completion_tokens", "output_tokens", "completionTokens", "outputTokens")
        for key in candidates:
            raw: Any = None
            if isinstance(usage, dict):
                raw = usage.get(key)
            else:
                raw = getattr(usage, key, None)
            if isinstance(raw, (int, float)) and raw >= 0:
                return int(raw)
        return 0

    message_content = _extract_response_content(resp)
    normalized_content = message_content.strip()
    completion_token_count = _extract_completion_token_count(resp)
    logprob_container = getattr(resp.choices[0], "logprobs", None)
    token_logprobs = (
        getattr(logprob_container, "content", None) if logprob_container is not None else None
    )

    if token_logprobs is None:
        missing_reason: Literal["missing_logprobs", "empty_completion"] = (
            "empty_completion" if normalized_content == "" else "missing_logprobs"
        )
        return _build_unavailable_confidence_metric(
            "openai",
            model,
            missing_reason,
            token_count=completion_token_count,
        )

    numeric_logprobs: list[float] = []
    for token in token_logprobs:
        value = getattr(token, "logprob", None)
        if isinstance(value, (int, float)) and math.isfinite(value):
            numeric_logprobs.append(float(value))

    if not numeric_logprobs:
        missing_reason = "empty_completion" if normalized_content == "" else "missing_logprobs"
        return _build_unavailable_confidence_metric(
            "openai",
            model,
            missing_reason,
            token_count=completion_token_count,
        )

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
    content = _extract_response_content(resp) or '{"spans":[]}'
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

    if not isinstance(raw, dict) or "spans" not in raw:
        raise ValueError(
            "Expected top-level object with key 'spans', "
            f"got {type(raw).__name__}: {content[:500]}"
        )
    raw = raw["spans"]

    if not isinstance(raw, list):
        raise ValueError(f"Expected 'spans' to be a JSON array, got {type(raw).__name__}.")
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
        label_schema = {"type": "string", "enum": SIMPLE_LABELS}
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


def _has_name_word_boundaries(raw_text: str, start: int, end: int) -> bool:
    if not (0 <= start < end <= len(raw_text)):
        return False
    left_char = raw_text[start - 1] if start > 0 else ""
    right_char = raw_text[end] if end < len(raw_text) else ""
    if left_char and left_char.isalnum():
        return False
    if right_char and right_char.isalnum():
        return False
    return True


def _is_plausible_name_span(raw_text: str, span: CanonicalSpan) -> bool:
    if not _is_plausible_name_text(span.text):
        return False
    return _has_name_word_boundaries(raw_text, span.start, span.end)


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
            if not _is_plausible_name_span(raw_text, candidate):
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
    raw_text: str,
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
            and not _is_plausible_name_span(raw_text, span)
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
        finish_reason = _extract_finish_reason(resp)
        if finish_reason:
            raise ValueError(
                f"LLM returned empty output content (finish_reason={finish_reason})."
            )
        raise ValueError("LLM returned empty output content.")

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
    if mapped in SIMPLE_LABEL_SET:
        return mapped
    # Heuristic fallback for near-miss label names from model outputs.
    if "NAME" in normalized or normalized in {"PERSONAL", "PERSONAL_NAME"}:
        return "NAME"
    if "EMAIL" in normalized:
        return "EMAIL"
    if "PHONE" in normalized or "FAX" in normalized:
        return "PHONE"
    if "URL" in normalized or "URI" in normalized:
        return "URL"
    if "DATE" in normalized or "TIME" in normalized:
        return "DATE"
    if "LOC" in normalized or "ADDR" in normalized or normalized == "GPE":
        return "LOCATION"
    if "SCHOOL" in normalized:
        return "SCHOOL"
    if "AGE" in normalized:
        return "AGE"
    return "MISC_ID"


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
        prompt_templates: list[dict[str, Any]] = []
        for pass_index, method_pass in enumerate(method.get("passes", []), start=1):
            if method_pass.get("kind") != "llm":
                continue
            prompt_templates.append(
                {
                    "pass_index": pass_index,
                    "entity_types": method_pass.get("entity_types"),
                    "system_prompt": str(method_pass.get("prompt") or SYSTEM_PROMPT),
                }
            )
        methods.append(
            {
                "id": method["id"],
                "label": method["label"],
                "description": method["description"],
                "requires_presidio": method["requires_presidio"],
                "uses_llm": method["uses_llm"],
                "supports_verify_override": method["supports_verify_override"],
                "default_verify": method["default_verify"],
                "prompt_templates": prompt_templates,
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
    llm_metrics: list[LLMConfidenceMetric] = []

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

        llm_result = run_llm_with_metadata(
            text=text,
            api_key=api_key or "",
            api_base=api_base,
            model=model,
            system_prompt=prompt,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            anthropic_thinking=anthropic_thinking,
            anthropic_thinking_budget_tokens=anthropic_thinking_budget_tokens,
            label_profile=label_profile,
        )
        llm_metrics.append(llm_result.llm_confidence)
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

    return MethodRunResult(
        spans=merged,
        warnings=warnings,
        llm_confidence=_aggregate_llm_confidence_metrics(llm_metrics),
    )


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
    effective_system_prompt = build_extraction_system_prompt(
        system_prompt,
        label_profile=label_profile,
    )

    if anthropic_thinking and supports_thinking:
        if temperature != 1.0:
            raise ValueError(
                f"Model '{model}' requires temperature=1 when thinking is enabled."
            )
        effective_temperature = 1.0
    elif not supports_custom_temperature:
        effective_temperature = None

    base_request_kwargs: dict[str, Any] = {
        "model": f"openai/{model}" if use_openai_gateway_format else model,
        "messages": [
            {"role": "system", "content": effective_system_prompt},
            {"role": "user", "content": text},
        ],
        "api_key": api_key,
    }
    supports_logprobs = _supports_logprobs(
        model,
        provider,
        reasoning_effort=reasoning_effort,
    )
    if supports_logprobs:
        base_request_kwargs["logprobs"] = True
    elif model.lower() in {"openai.gpt-5.2-chat", "openai/gpt-5.2-chat"}:
        warnings.append(
            "Token logprobs are currently unavailable for model 'openai.gpt-5.2-chat' on this API route."
        )
    elif (
        _is_openai_model(provider, model)
        and ("gpt-5" in model.lower() or "gpt5" in model.lower())
        and reasoning_effort != "none"
    ):
        warnings.append(
            "Logprobs are unavailable for this GPT-5 run when reasoning_effort is not 'none'. "
            "Set reasoning_effort='none' to enable logprob confidence."
        )
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
                text,
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
            if llm_confidence.reason == "missing_logprobs" and _is_openai_model(provider, model):
                warnings.append(
                    "Model response did not include token logprobs. "
                    "This can happen with some gateway/model combinations even when the request succeeds."
                )
            if idx > 0:
                warnings.append(
                    f"Primary api_base failed; succeeded after retrying with '{candidate}'."
                )
            return LLMRunResult(
                spans=spans,
                warnings=warnings,
                llm_confidence=llm_confidence,
                finish_reason=finish_reason,
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
