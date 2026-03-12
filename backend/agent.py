from __future__ import annotations

import copy
from dataclasses import dataclass, field
from functools import lru_cache
import importlib.util
import json
import math
import re
import threading
import time
from typing import Any, Callable, Literal

from litellm import completion

from models import CanonicalSpan, LLMConfidenceMetric, ResolutionEvent
from span_resolution import RESOLUTION_POLICY_VERSION, resolve_spans

# Regex patterns for rule-based detection
DOMAIN_LABEL_PATTERN = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
DOMAIN_SEPARATOR_PATTERN = r"\s*\.\s*"
DOMAIN_PATTERN = (
    rf"(?:{DOMAIN_LABEL_PATTERN}{DOMAIN_SEPARATOR_PATTERN})+[a-z]{{2,63}}"
)

PATTERNS: dict[str, re.Pattern] = {
    "EMAIL": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "URL": re.compile(
        rf"(?<!@)\b(?:https?://)?{DOMAIN_PATTERN}(?:/[^\s,)]+)?",
        re.IGNORECASE,
    ),
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
MethodBundleId = Literal[
    "legacy",
    "audited",
    "test",
    "v2",
    "v2+post-process",
    "deidentify-v2",
]

CONFIDENCE_HIGH_THRESHOLD = 0.9
CONFIDENCE_MEDIUM_THRESHOLD = 0.75
DEFAULT_METHOD_BUNDLE: MethodBundleId = "audited"
DEFAULT_LLM_EXTRACTION_TIMEOUT_SECONDS = 60.0
DEFAULT_LLM_VERIFIER_TIMEOUT_SECONDS = 30.0
DEFAULT_LLM_REPAIR_TIMEOUT_SECONDS = 30.0
DEFAULT_LLM_EXTRACTION_MAX_TOKENS = 4096
DEFAULT_LLM_VERIFIER_MAX_TOKENS = 1024
DEFAULT_LLM_REPAIR_MAX_TOKENS = 1024


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
        "label": "Anthropic: Claude Opus 4.6",
        "model": "anthropic.claude-4.6-opus",
        "provider": "anthropic",
        "supports_reasoning_effort": False,
        "supports_anthropic_thinking": True,
        "default_reasoning_effort": "none",
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
        "label": "Anthropic: Claude Sonnet 4.6",
        "model": "anthropic.claude-4.6-sonnet",
        "provider": "anthropic",
        "supports_reasoning_effort": False,
        "supports_anthropic_thinking": False,
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
    "chatgpt": "openai",
    "anthropic": "anthropic",
    "google": "gemini",
    "gemini": "gemini",
    "vertex_ai": "gemini",
    "vertex": "gemini",
}

OPENAI_MODEL_PREFIXES: tuple[str, ...] = (
    "gpt-",
    "gpt5",
    "o1",
    "o3",
    "o4",
    "codex",
    "chatgpt/",
)

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
    method_bundle: MethodBundleId = DEFAULT_METHOD_BUNDLE,
) -> str:
    prompt = str(base_prompt or "").strip() or SYSTEM_PROMPT.strip()
    allowed_labels = ", ".join(_labels_for_profile(label_profile))
    bundle = _normalize_method_bundle(method_bundle)
    formatted_prompt = (
        f"{prompt}\n\n"
        f"{FORMAT_GUARDRAIL}\n"
        f'- Allowed labels for this run: {allowed_labels}.\n'
        '- The "label" field must be one of the allowed labels above.\n'
        '- The "text" field must exactly match transcript[start:end].\n'
    )
    if bundle == "test":
        formatted_prompt = (
            f"{formatted_prompt}"
            f"{_build_test_bundle_prompt_contract(label_profile)}\n"
        )
    return formatted_prompt

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
            {
                "kind": "llm",
                "prompt": METHOD_DEFAULT_PROMPT,
                "entity_types": None,
                "pass_label": "default:all",
            },
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
            {
                "kind": "llm",
                "prompt": METHOD_EXTENDED_PROMPT,
                "entity_types": None,
                "pass_label": "extended:all",
            },
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
            {
                "kind": "llm",
                "prompt": METHOD_DEFAULT_PROMPT,
                "entity_types": None,
                "pass_label": "verified:all",
            },
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
                "pass_label": "dual:names",
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
                "pass_label": "dual:identifiers",
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
                "pass_label": "dual-split:names_locations",
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
                "pass_label": "dual-split:numeric_identifiers",
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
                "pass_label": "presidio:core",
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
                "pass_label": "presidio+default:presidio",
            },
            {
                "kind": "llm",
                "prompt": METHOD_DEFAULT_PROMPT,
                "entity_types": None,
                "pass_label": "presidio+default:llm",
            },
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
                "pass_label": "presidio+llm-split:presidio",
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
                "pass_label": "presidio+llm-split:llm",
            },
        ],
    },
]

METHOD_DEFINITION_BY_ID: dict[str, dict[str, Any]] = {
    method["id"]: method for method in METHOD_DEFINITIONS
}

LEGACY_METHOD_DEFINITIONS: list[dict[str, Any]] = copy.deepcopy(METHOD_DEFINITIONS)
LEGACY_METHOD_DEFINITION_BY_ID: dict[str, dict[str, Any]] = {
    method["id"]: method for method in LEGACY_METHOD_DEFINITIONS
}

_SIMPLE_PROMPT_LABEL_DESCRIPTIONS: dict[str, str] = {
    "AGE": "ages over 89 or ages that directly identify an individual [AGE]",
    "DATE": "dates directly tied to an individual, excluding standalone years [DATE]",
    "EMAIL": "email addresses [EMAIL]",
    "LOCATION": (
        "geographic locations smaller than a state such as street addresses, cities, and "
        "neighborhood names [LOCATION]"
    ),
    "MISC_ID": (
        "other identifying numbers or codes such as student IDs, account numbers, SSNs, "
        "device identifiers, license numbers, passport numbers, biometric identifiers, or "
        "IP addresses [MISC_ID]"
    ),
    "NAME": "personal names, nicknames, and pseudonyms used for specific individuals [NAME]",
    "PHONE": "phone or fax numbers [PHONE]",
    "SCHOOL": "school names that identify a person or reveal location [SCHOOL]",
    "URL": "URLs or web domains tied to a person or account [URL]",
}

_ADVANCED_PROMPT_LABEL_DESCRIPTIONS: dict[str, str] = {
    "AGE": "ages over 89 or ages that directly identify an individual [AGE]",
    "COURSE": "course names or codes that identify a student's enrollment context [COURSE]",
    "DATE": "dates directly tied to an individual, excluding standalone years [DATE]",
    "EMAIL_ADDRESS": "email addresses [EMAIL_ADDRESS]",
    "GRADE_LEVEL": "grade levels tied to a specific student [GRADE_LEVEL]",
    "IP_ADDRESS": "IP addresses [IP_ADDRESS]",
    "LOCATION": (
        "geographic locations smaller than a state such as street addresses, cities, and "
        "neighborhood names [LOCATION]"
    ),
    "NRP": (
        "nationality, religious, or political-group references when they are stated as a "
        "personal attribute of an individual [NRP]"
    ),
    "PERSON": "personal names, nicknames, and pseudonyms used for specific individuals [PERSON]",
    "PHONE_NUMBER": "phone or fax numbers [PHONE_NUMBER]",
    "SCHOOL": "school names that identify a person or reveal location [SCHOOL]",
    "SOCIAL_HANDLE": "social media handles or usernames [SOCIAL_HANDLE]",
    "URL": "URLs or web domains tied to a person or account [URL]",
    "US_BANK_NUMBER": "bank account or routing numbers [US_BANK_NUMBER]",
    "US_DRIVER_LICENSE": "US driver license numbers [US_DRIVER_LICENSE]",
    "US_PASSPORT": "US passport numbers [US_PASSPORT]",
    "US_SSN": "US Social Security numbers [US_SSN]",
}

_METHOD_FALSE_POSITIVE_GUIDANCE = """\
DO NOT mark any of the following as PII:
- Historical figures, scientists, mathematicians, or other public figures
- Fictional characters, brands, products, course titles, project titles, or textbook references
- Mathematical expressions, equations, formulas, variables, problem numbers, or section references
- Dates not tied to a specific individual, such as deadlines, schedules, semesters, or publication dates
- Scores, grades, percentages, and numbers used only in an educational or math context
- Countries, states, or regions used only as general context rather than a personal location
- Method or theorem names containing a person's name
"""


def _ordered_difference(values: list[str], excluded: set[str]) -> list[str]:
    return [value for value in values if value not in excluded]


def _label_descriptions_for_profile(
    label_profile: Literal["simple", "advanced"],
) -> dict[str, str]:
    return (
        _ADVANCED_PROMPT_LABEL_DESCRIPTIONS
        if label_profile == "advanced"
        else _SIMPLE_PROMPT_LABEL_DESCRIPTIONS
    )


def _render_prompt_category_catalog(
    labels: list[str],
    *,
    label_profile: Literal["simple", "advanced"],
) -> str:
    descriptions = _label_descriptions_for_profile(label_profile)
    parts = [descriptions[label] for label in labels if label in descriptions]
    return "; ".join(parts)


def _build_test_bundle_prompt_contract(
    label_profile: Literal["simple", "advanced"],
) -> str:
    labels = _labels_for_profile(label_profile)
    example_label = "PERSON" if label_profile == "advanced" else "NAME"
    category_catalog = _render_prompt_category_catalog(labels, label_profile=label_profile)
    example_payload = json.dumps(
        {
            "spans": [
                {
                    "start": 12,
                    "end": 20,
                    "label": example_label,
                    "text": "John Doe",
                }
            ]
        },
        separators=(",", ":"),
    )
    empty_payload = json.dumps({"spans": []}, separators=(",", ":"))
    return (
        "TEST BUNDLE EXTRACTION CONTRACT:\n"
        f"- Use only these exact label strings: {', '.join(labels)}.\n"
        f"- Expected categories for this run: {category_catalog}.\n"
        "- If the text uses a legacy taxonomy name, map it to the closest allowed label above.\n"
        "- Do not invent new label names or add extra JSON keys.\n"
        f"- Example valid JSON response: {example_payload}\n"
        f"- Example empty JSON response: {empty_payload}"
    )


def _build_profile_default_prompt(
    label_profile: Literal["simple", "advanced"],
) -> str:
    categories = _render_prompt_category_catalog(
        _labels_for_profile(label_profile),
        label_profile=label_profile,
    )
    return (
        "You are a PII/PHI analyst. Identify all explicit personally identifiable "
        "information in the text. Extract explicit identifiers only; do not infer missing data. "
        f"Categories to detect: {categories}. "
        "Matches should be minimal exact spans of the sensitive value; include partial "
        "identifiers or identifiers embedded in filenames, URLs, or notes when they are "
        "explicit in the text. "
        "Output data following the provided JSON schema."
    )


def _build_profile_extended_prompt(
    label_profile: Literal["simple", "advanced"],
) -> str:
    categories = _render_prompt_category_catalog(
        _labels_for_profile(label_profile),
        label_profile=label_profile,
    )
    return (
        "You are a PII/PHI analyst reviewing tutoring chat transcripts. Identify explicit "
        "personally identifiable information per HIPAA Safe Harbor guidance where applicable. "
        f"Categories to detect: {categories}. "
        f"{_METHOD_FALSE_POSITIVE_GUIDANCE} "
        "Matches should be minimal exact spans of the sensitive value. "
        "Output data following the provided JSON schema."
    )


def _build_profile_names_prompt(
    label_profile: Literal["simple", "advanced"],
) -> str:
    target_label = "PERSON" if label_profile == "advanced" else "NAME"
    return (
        "You are a PII analyst specializing in personal names in tutoring chat transcripts. "
        f"Identify all personal names and label each one [{target_label}]. "
        "This includes first names, last names, nicknames, and pseudonyms used to refer to "
        "specific individuals in the conversation. "
        "DO NOT mark historical figures, scientists, mathematicians, public figures, book "
        "authors, fictional characters, or method/theorem names. "
        "DO NOT mark email addresses, URLs, usernames, or numeric identifiers. "
        "Matches should be minimal exact spans of just the name. "
        "Output data following the provided JSON schema."
    )


def _build_profile_remaining_prompt(
    labels: list[str],
    *,
    label_profile: Literal["simple", "advanced"],
    role: str,
    exclusions: list[str],
) -> str:
    categories = _render_prompt_category_catalog(labels, label_profile=label_profile)
    exclusion_lines = "\n".join(f"- {line}" for line in exclusions)
    return (
        f"You are a PII analyst specializing in {role}. "
        "The text may contain tutoring dialogue, equations, and educational content mixed with "
        "personal information. Identify anything in the following categories that could "
        f"reasonably identify an individual: {categories}. "
        "DO NOT mark any of the following:\n"
        f"{exclusion_lines}\n"
        "Matches should be minimal exact spans. "
        "Output data following the provided JSON schema."
    )


def _copy_profile_mapping(
    values: dict[str, list[str] | None],
) -> dict[str, list[str] | None]:
    return {
        str(profile): (None if items is None else list(items))
        for profile, items in values.items()
    }


def _build_profile_aware_llm_pass(
    *,
    prompt_by_profile: dict[str, str],
    entity_types_by_profile: dict[str, list[str] | None],
    requested_labels_by_profile: dict[str, list[str]],
    pass_label: str,
) -> dict[str, Any]:
    simple_prompt = prompt_by_profile.get("simple") or next(iter(prompt_by_profile.values()))
    simple_entity_types = entity_types_by_profile.get("simple")
    return {
        "kind": "llm",
        "prompt": simple_prompt,
        "prompt_by_profile": dict(prompt_by_profile),
        "entity_types": None if simple_entity_types is None else list(simple_entity_types),
        "entity_types_by_profile": _copy_profile_mapping(entity_types_by_profile),
        "requested_labels_by_profile": {
            str(profile): list(labels) for profile, labels in requested_labels_by_profile.items()
        },
        "pass_label": pass_label,
    }


def _build_profile_aware_presidio_pass(
    *,
    presidio_entities_by_profile: dict[str, list[str]],
    entity_types_by_profile: dict[str, list[str]],
    pass_label: str,
) -> dict[str, Any]:
    simple_entities = presidio_entities_by_profile.get("simple", [])
    simple_entity_types = entity_types_by_profile.get("simple", [])
    return {
        "kind": "presidio",
        "entity_types": list(simple_entity_types),
        "entity_types_by_profile": _copy_profile_mapping(entity_types_by_profile),
        "presidio_entities": list(simple_entities),
        "presidio_entities_by_profile": {
            str(profile): list(values) for profile, values in presidio_entities_by_profile.items()
        },
        "requested_labels_by_profile": {
            str(profile): list(values) for profile, values in entity_types_by_profile.items()
        },
        "pass_label": pass_label,
    }


_AUDITED_DEFAULT_PROMPTS = {
    "simple": _build_profile_default_prompt("simple"),
    "advanced": _build_profile_default_prompt("advanced"),
}
_AUDITED_EXTENDED_PROMPTS = {
    "simple": _build_profile_extended_prompt("simple"),
    "advanced": _build_profile_extended_prompt("advanced"),
}
_AUDITED_NAMES_PROMPTS = {
    "simple": _build_profile_names_prompt("simple"),
    "advanced": _build_profile_names_prompt("advanced"),
}
_DUAL_SIMPLE_REMAINING_LABELS = _ordered_difference(SIMPLE_LABELS, {"NAME"})
_DUAL_ADVANCED_REMAINING_LABELS = _ordered_difference(ADVANCED_LABELS, {"PERSON"})
_DUAL_SPLIT_SIMPLE_PASS_ONE = ["NAME", "LOCATION", "SCHOOL"]
_DUAL_SPLIT_ADVANCED_PASS_ONE = ["PERSON", "LOCATION", "SCHOOL"]
_DUAL_SPLIT_SIMPLE_PASS_TWO = _ordered_difference(
    SIMPLE_LABELS,
    set(_DUAL_SPLIT_SIMPLE_PASS_ONE),
)
_DUAL_SPLIT_ADVANCED_PASS_TWO = _ordered_difference(
    ADVANCED_LABELS,
    set(_DUAL_SPLIT_ADVANCED_PASS_ONE),
)
_PRESIDIO_DEFAULT_SIMPLE_OUTPUT = ["EMAIL", "PHONE", "URL", "MISC_ID"]
_PRESIDIO_DEFAULT_ADVANCED_OUTPUT = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "URL",
    "IP_ADDRESS",
    "US_SSN",
    "US_BANK_NUMBER",
]
_PRESIDIO_DEFAULT_SIMPLE_RESIDUAL = [
    "NAME",
    "LOCATION",
    "SCHOOL",
    "DATE",
    "AGE",
    "MISC_ID",
]
_PRESIDIO_DEFAULT_ADVANCED_RESIDUAL = _ordered_difference(
    ADVANCED_LABELS,
    set(_PRESIDIO_DEFAULT_ADVANCED_OUTPUT),
)
_PRESIDIO_SPLIT_SIMPLE_OUTPUT = ["EMAIL", "PHONE", "URL", "MISC_ID"]
_PRESIDIO_SPLIT_ADVANCED_OUTPUT = [
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "URL",
    "IP_ADDRESS",
]
_PRESIDIO_SPLIT_SIMPLE_RESIDUAL = [
    "NAME",
    "LOCATION",
    "SCHOOL",
    "DATE",
    "AGE",
    "MISC_ID",
]
_PRESIDIO_SPLIT_ADVANCED_RESIDUAL = _ordered_difference(
    ADVANCED_LABELS,
    set(_PRESIDIO_SPLIT_ADVANCED_OUTPUT),
)

_AUDITED_DUAL_REMAINING_PROMPTS = {
    "simple": _build_profile_remaining_prompt(
        _DUAL_SIMPLE_REMAINING_LABELS,
        label_profile="simple",
        role="non-name identifiers and contact details",
        exclusions=[
            "Personal names, nicknames, or pseudonyms",
            "Mathematical expressions, equations, formulas, variables, or problem numbers",
            "Course section references, calculator model names, and textbook references",
        ],
    ),
    "advanced": _build_profile_remaining_prompt(
        _DUAL_ADVANCED_REMAINING_LABELS,
        label_profile="advanced",
        role="non-name identifiers and contact details",
        exclusions=[
            "Personal names, nicknames, or pseudonyms",
            "Mathematical expressions, equations, formulas, variables, or problem numbers",
            "Course section references, calculator model names, and textbook references",
        ],
    ),
}
_AUDITED_DUAL_SPLIT_PASS_ONE_PROMPTS = {
    "simple": _build_profile_remaining_prompt(
        _DUAL_SPLIT_SIMPLE_PASS_ONE,
        label_profile="simple",
        role="personal names and location signals",
        exclusions=[
            "Historical figures, scientists, mathematicians, or public figures",
            "Fictional characters, brands, products, or method/theorem names",
            "Countries, states, or regions used only as general context",
            "Email addresses, phone numbers, URLs, or numeric identifiers",
        ],
    ),
    "advanced": _build_profile_remaining_prompt(
        _DUAL_SPLIT_ADVANCED_PASS_ONE,
        label_profile="advanced",
        role="personal names and location signals",
        exclusions=[
            "Historical figures, scientists, mathematicians, or public figures",
            "Fictional characters, brands, products, or method/theorem names",
            "Countries, states, or regions used only as general context",
            "Email addresses, phone numbers, URLs, or numeric identifiers",
        ],
    ),
}
_AUDITED_DUAL_SPLIT_PASS_TWO_PROMPTS = {
    "simple": _build_profile_remaining_prompt(
        _DUAL_SPLIT_SIMPLE_PASS_TWO,
        label_profile="simple",
        role="dates, contact details, and identifying numbers",
        exclusions=[
            "Personal names, school names, cities, or neighborhood names",
            "Mathematical expressions, equations, formulas, variables, or problem numbers",
            "Course section references, calculator model names, and textbook references",
        ],
    ),
    "advanced": _build_profile_remaining_prompt(
        _DUAL_SPLIT_ADVANCED_PASS_TWO,
        label_profile="advanced",
        role="dates, contact details, and identifying numbers",
        exclusions=[
            "Personal names, school names, cities, or neighborhood names",
            "Mathematical expressions, equations, formulas, variables, or problem numbers",
            "Course section references, calculator model names, and textbook references",
        ],
    ),
}
_AUDITED_PRESIDIO_DEFAULT_PROMPTS = {
    "simple": _build_profile_remaining_prompt(
        _PRESIDIO_DEFAULT_SIMPLE_RESIDUAL,
        label_profile="simple",
        role="the remaining PII not already covered by deterministic identifier detectors",
        exclusions=[
            "Email addresses, phone numbers, URLs, IP addresses, SSNs, or bank numbers already covered elsewhere",
            "Mathematical expressions, equations, formulas, variables, or problem numbers",
            "Course section references or textbook references",
        ],
    ),
    "advanced": _build_profile_remaining_prompt(
        _PRESIDIO_DEFAULT_ADVANCED_RESIDUAL,
        label_profile="advanced",
        role="the remaining PII not already covered by deterministic identifier detectors",
        exclusions=[
            "Email addresses, phone numbers, URLs, IP addresses, US SSNs, or bank numbers already covered elsewhere",
            "Mathematical expressions, equations, formulas, variables, or problem numbers",
            "Course section references or textbook references",
        ],
    ),
}
_AUDITED_SPLIT_LLM_PROMPTS = {
    "simple": _build_profile_remaining_prompt(
        _PRESIDIO_SPLIT_SIMPLE_RESIDUAL,
        label_profile="simple",
        role="remaining PII after contact identifiers are removed",
        exclusions=[
            "Email addresses, phone numbers, URLs, or IP addresses already covered elsewhere",
            "Mathematical expressions, equations, formulas, variables, or problem numbers",
            "Course section references or textbook references",
        ],
    ),
    "advanced": _build_profile_remaining_prompt(
        _PRESIDIO_SPLIT_ADVANCED_RESIDUAL,
        label_profile="advanced",
        role="remaining PII after contact identifiers are removed",
        exclusions=[
            "Email addresses, phone numbers, URLs, or IP addresses already covered elsewhere",
            "Mathematical expressions, equations, formulas, variables, or problem numbers",
            "Course section references or textbook references",
        ],
    ),
}

AUDITED_METHOD_DEFINITIONS: list[dict[str, Any]] = [
    {
        "id": "default",
        "label": "Default",
        "description": "Single-pass LLM baseline.",
        "requires_presidio": False,
        "uses_llm": True,
        "supports_verify_override": True,
        "default_verify": False,
        "supported_label_profiles": ["advanced", "simple"],
        "known_limitations": [],
        "passes": [
            _build_profile_aware_llm_pass(
                prompt_by_profile=_AUDITED_DEFAULT_PROMPTS,
                entity_types_by_profile={
                    "simple": list(SIMPLE_LABELS),
                    "advanced": list(ADVANCED_LABELS),
                },
                requested_labels_by_profile={
                    "simple": list(SIMPLE_LABELS),
                    "advanced": list(ADVANCED_LABELS),
                },
                pass_label="default:all",
            )
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
        "supported_label_profiles": ["advanced", "simple"],
        "known_limitations": [],
        "passes": [
            _build_profile_aware_llm_pass(
                prompt_by_profile=_AUDITED_EXTENDED_PROMPTS,
                entity_types_by_profile={
                    "simple": list(SIMPLE_LABELS),
                    "advanced": list(ADVANCED_LABELS),
                },
                requested_labels_by_profile={
                    "simple": list(SIMPLE_LABELS),
                    "advanced": list(ADVANCED_LABELS),
                },
                pass_label="extended:all",
            )
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
        "supported_label_profiles": ["advanced", "simple"],
        "known_limitations": [],
        "passes": [
            _build_profile_aware_llm_pass(
                prompt_by_profile=_AUDITED_DEFAULT_PROMPTS,
                entity_types_by_profile={
                    "simple": list(SIMPLE_LABELS),
                    "advanced": list(ADVANCED_LABELS),
                },
                requested_labels_by_profile={
                    "simple": list(SIMPLE_LABELS),
                    "advanced": list(ADVANCED_LABELS),
                },
                pass_label="verified:all",
            )
        ],
    },
    {
        "id": "dual",
        "label": "Dual",
        "description": "Two-pass LLM (names + remaining explicit identifiers).",
        "requires_presidio": False,
        "uses_llm": True,
        "supports_verify_override": True,
        "default_verify": False,
        "supported_label_profiles": ["advanced", "simple"],
        "known_limitations": [],
        "passes": [
            _build_profile_aware_llm_pass(
                prompt_by_profile=_AUDITED_NAMES_PROMPTS,
                entity_types_by_profile={"simple": ["NAME"], "advanced": ["PERSON"]},
                requested_labels_by_profile={"simple": ["NAME"], "advanced": ["PERSON"]},
                pass_label="dual:names",
            ),
            _build_profile_aware_llm_pass(
                prompt_by_profile=_AUDITED_DUAL_REMAINING_PROMPTS,
                entity_types_by_profile={
                    "simple": _DUAL_SIMPLE_REMAINING_LABELS,
                    "advanced": _DUAL_ADVANCED_REMAINING_LABELS,
                },
                requested_labels_by_profile={
                    "simple": _DUAL_SIMPLE_REMAINING_LABELS,
                    "advanced": _DUAL_ADVANCED_REMAINING_LABELS,
                },
                pass_label="dual:identifiers",
            ),
        ],
    },
    {
        "id": "dual-split",
        "label": "Dual Split",
        "description": "Two-pass LLM split by names/locations and remaining explicit identifiers.",
        "requires_presidio": False,
        "uses_llm": True,
        "supports_verify_override": True,
        "default_verify": False,
        "supported_label_profiles": ["advanced", "simple"],
        "known_limitations": [],
        "passes": [
            _build_profile_aware_llm_pass(
                prompt_by_profile=_AUDITED_DUAL_SPLIT_PASS_ONE_PROMPTS,
                entity_types_by_profile={
                    "simple": _DUAL_SPLIT_SIMPLE_PASS_ONE,
                    "advanced": _DUAL_SPLIT_ADVANCED_PASS_ONE,
                },
                requested_labels_by_profile={
                    "simple": _DUAL_SPLIT_SIMPLE_PASS_ONE,
                    "advanced": _DUAL_SPLIT_ADVANCED_PASS_ONE,
                },
                pass_label="dual-split:names_locations",
            ),
            _build_profile_aware_llm_pass(
                prompt_by_profile=_AUDITED_DUAL_SPLIT_PASS_TWO_PROMPTS,
                entity_types_by_profile={
                    "simple": _DUAL_SPLIT_SIMPLE_PASS_TWO,
                    "advanced": _DUAL_SPLIT_ADVANCED_PASS_TWO,
                },
                requested_labels_by_profile={
                    "simple": _DUAL_SPLIT_SIMPLE_PASS_TWO,
                    "advanced": _DUAL_SPLIT_ADVANCED_PASS_TWO,
                },
                pass_label="dual-split:numeric_identifiers",
            ),
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
        "supported_label_profiles": ["advanced", "simple"],
        "known_limitations": ["Requires local Presidio and spaCy setup."],
        "passes": [
            _build_profile_aware_presidio_pass(
                presidio_entities_by_profile={
                    "simple": [
                        "PERSON",
                        "LOCATION",
                        "EMAIL_ADDRESS",
                        "URL",
                        "PHONE_NUMBER",
                        "IP_ADDRESS",
                    ],
                    "advanced": [
                        "PERSON",
                        "LOCATION",
                        "EMAIL_ADDRESS",
                        "URL",
                        "PHONE_NUMBER",
                        "IP_ADDRESS",
                    ],
                },
                entity_types_by_profile={
                    "simple": ["NAME", "LOCATION", "EMAIL", "URL", "PHONE", "MISC_ID"],
                    "advanced": [
                        "PERSON",
                        "LOCATION",
                        "EMAIL_ADDRESS",
                        "URL",
                        "PHONE_NUMBER",
                        "IP_ADDRESS",
                    ],
                },
                pass_label="presidio:core",
            )
        ],
    },
    {
        "id": "presidio+default",
        "label": "Presidio + Default",
        "description": "Presidio identifiers combined with a residual LLM pass.",
        "requires_presidio": True,
        "uses_llm": True,
        "supports_verify_override": True,
        "default_verify": False,
        "supported_label_profiles": ["advanced", "simple"],
        "known_limitations": ["Requires local Presidio and spaCy setup."],
        "passes": [
            _build_profile_aware_presidio_pass(
                presidio_entities_by_profile={
                    "simple": [
                        "EMAIL_ADDRESS",
                        "URL",
                        "PHONE_NUMBER",
                        "IP_ADDRESS",
                        "US_SSN",
                        "US_BANK_NUMBER",
                    ],
                    "advanced": [
                        "EMAIL_ADDRESS",
                        "URL",
                        "PHONE_NUMBER",
                        "IP_ADDRESS",
                        "US_SSN",
                        "US_BANK_NUMBER",
                    ],
                },
                entity_types_by_profile={
                    "simple": _PRESIDIO_DEFAULT_SIMPLE_OUTPUT,
                    "advanced": _PRESIDIO_DEFAULT_ADVANCED_OUTPUT,
                },
                pass_label="presidio+default:presidio",
            ),
            _build_profile_aware_llm_pass(
                prompt_by_profile=_AUDITED_PRESIDIO_DEFAULT_PROMPTS,
                entity_types_by_profile={
                    "simple": _PRESIDIO_DEFAULT_SIMPLE_RESIDUAL,
                    "advanced": _PRESIDIO_DEFAULT_ADVANCED_RESIDUAL,
                },
                requested_labels_by_profile={
                    "simple": _PRESIDIO_DEFAULT_SIMPLE_RESIDUAL,
                    "advanced": _PRESIDIO_DEFAULT_ADVANCED_RESIDUAL,
                },
                pass_label="presidio+default:llm",
            ),
        ],
    },
    {
        "id": "presidio+llm-split",
        "label": "Presidio + LLM Split",
        "description": "Presidio for core contact identifiers plus a residual LLM pass.",
        "requires_presidio": True,
        "uses_llm": True,
        "supports_verify_override": True,
        "default_verify": False,
        "supported_label_profiles": ["advanced", "simple"],
        "known_limitations": ["Requires local Presidio and spaCy setup."],
        "passes": [
            _build_profile_aware_presidio_pass(
                presidio_entities_by_profile={
                    "simple": ["EMAIL_ADDRESS", "URL", "PHONE_NUMBER", "IP_ADDRESS"],
                    "advanced": ["EMAIL_ADDRESS", "URL", "PHONE_NUMBER", "IP_ADDRESS"],
                },
                entity_types_by_profile={
                    "simple": _PRESIDIO_SPLIT_SIMPLE_OUTPUT,
                    "advanced": _PRESIDIO_SPLIT_ADVANCED_OUTPUT,
                },
                pass_label="presidio+llm-split:presidio",
            ),
            _build_profile_aware_llm_pass(
                prompt_by_profile=_AUDITED_SPLIT_LLM_PROMPTS,
                entity_types_by_profile={
                    "simple": _PRESIDIO_SPLIT_SIMPLE_RESIDUAL,
                    "advanced": _PRESIDIO_SPLIT_ADVANCED_RESIDUAL,
                },
                requested_labels_by_profile={
                    "simple": _PRESIDIO_SPLIT_SIMPLE_RESIDUAL,
                    "advanced": _PRESIDIO_SPLIT_ADVANCED_RESIDUAL,
                },
                pass_label="presidio+llm-split:llm",
            ),
        ],
    },
]

AUDITED_METHOD_DEFINITION_BY_ID: dict[str, dict[str, Any]] = {
    method["id"]: method for method in AUDITED_METHOD_DEFINITIONS
}
TEST_METHOD_DEFINITIONS: list[dict[str, Any]] = copy.deepcopy(AUDITED_METHOD_DEFINITIONS)
TEST_METHOD_DEFINITION_BY_ID: dict[str, dict[str, Any]] = {
    method["id"]: method for method in TEST_METHOD_DEFINITIONS
}
V2_METHOD_DEFINITIONS: list[dict[str, Any]] = copy.deepcopy(AUDITED_METHOD_DEFINITIONS)
V2_METHOD_DEFINITION_BY_ID: dict[str, dict[str, Any]] = {
    method["id"]: method for method in V2_METHOD_DEFINITIONS
}
V2_POST_PROCESS_METHOD_DEFINITIONS: list[dict[str, Any]] = copy.deepcopy(
    AUDITED_METHOD_DEFINITIONS
)
V2_POST_PROCESS_METHOD_DEFINITION_BY_ID: dict[str, dict[str, Any]] = {
    method["id"]: method for method in V2_POST_PROCESS_METHOD_DEFINITIONS
}

DEIDENTIFY_V2_EXTENDED_PROMPT = """\
You are a PII/PHI analyst reviewing tutoring chat transcripts. Identify personally \
identifiable information (PII) per HIPAA Safe Harbor (§164.514(b)(2)).
Extract explicit identifiers only; do not infer missing data.
Categories to detect:
- First names, last names, nicknames, or usernames of real individuals participating \
in or mentioned in the conversation [NAME]. Exclude historical figures, scientists, \
mathematicians, public figures, fictional characters, and names appearing only as part \
of a method or theorem (e.g., "Horner's method", "L'Hôpital's rule").
- Street addresses, cities, or other geographic subdivisions smaller than a State that \
identify where an individual lives or is located [ADDRESS]. Exclude countries, states, \
and regions mentioned as general context.
- Schools or universities that an individual attends or is associated with \
(e.g., Jackson High, PS 123, University of Phoenix) [SCHOOL]. Exclude institutions \
mentioned only as general references (e.g., "MIT research", "Stanford algorithm").
- Birth dates, admission dates, discharge dates, or dates of death directly tied to a \
specific individual (year may be kept); ages over 89 [DATE]. Exclude all other dates: \
days of the week, times of day, due dates, class schedules, semesters, and dates or \
date ranges appearing in math problems.
- Phone numbers [PHONE_NUMBER]; fax numbers [FAX_NUMBER].
- Email addresses [EMAIL].
- Social Security numbers [SSN].
- Actual account numbers (bank, insurance, etc.) [ACCOUNT_NUMBER]. Exclude general \
mentions of accounts without a number.
- Medical device serial numbers or unique device identifiers [DEVICE_IDENTIFIER].
- URLs that identify or link to an individual [URL]. Exclude generic tool or platform URLs.
- IP addresses [IP_ADDRESS].
- Biometric identifiers such as fingerprints or voiceprints [BIOMETRIC_IDENTIFIER].
- Photographs or images attached or shared that could identify an individual [IMAGE]. \
Exclude descriptions of seeing or sharing images.
- Any other unique identifying number such as student IDs, medical record numbers, \
license numbers, or social media handles [IDENTIFYING_NUMBER]. Exclude problem numbers, \
scores, grades, mathematical expressions, and numbers used in an educational context.
Matches should be minimal exact spans of the sensitive value.
Output data following the provided JSON schema."""

DEIDENTIFY_V2_EXTENDED_TYPES: list[str] = [
    "NAME",
    "ADDRESS",
    "DATE",
    "PHONE_NUMBER",
    "FAX_NUMBER",
    "EMAIL",
    "SSN",
    "ACCOUNT_NUMBER",
    "DEVICE_IDENTIFIER",
    "URL",
    "IP_ADDRESS",
    "BIOMETRIC_IDENTIFIER",
    "IMAGE",
    "IDENTIFYING_NUMBER",
    "SCHOOL",
]

DEIDENTIFY_V2_NAMES_PROMPT = """\
You are a PII analyst reviewing tutoring chat transcripts per HIPAA Safe Harbor.
Identify the names of real individuals participating in or mentioned in the conversation \
[NAME]. This includes first names, last names, nicknames, and pseudonyms.
Exclude historical figures, scientists, mathematicians (e.g., Einstein, Pythagoras), \
public figures, celebrities, fictional characters, and names appearing only as part \
of a method or theorem (e.g., "Horner's method", "L'Hôpital's rule").
Matches should be minimal exact spans of just the name.
Output data following the provided JSON schema."""

DEIDENTIFY_V2_NAMES_TYPES = ["NAME"]

DEIDENTIFY_V2_IDENTIFIERS_PROMPT = """\
You are a PII analyst reviewing tutoring chat transcripts per HIPAA Safe Harbor. \
Personal names are handled separately — do not detect names here.
Categories to detect:
- Street addresses, cities, or other geographic subdivisions smaller than a State that \
identify where an individual lives or is located [ADDRESS]. Exclude countries, states, \
and regions mentioned as general context.
- Schools or universities that an individual attends or is associated with \
(e.g., Jackson High, PS 123, University of Phoenix) [SCHOOL]. Exclude institutions \
mentioned only as general references (e.g., "MIT research", "Stanford algorithm").
- Birth dates, admission dates, discharge dates, or dates of death directly tied to a \
specific individual (year may be kept); ages over 89 [DATE]. Exclude all other dates: \
days of the week, times of day, due dates, class schedules, semesters, and dates or \
date ranges appearing in math problems.
- Email addresses [EMAIL].
- URLs that identify or link to an individual [URL]. Exclude generic tool or platform URLs.
- Phone numbers [PHONE_NUMBER]; fax numbers [FAX_NUMBER].
- Social Security numbers [SSN].
- Actual account numbers (bank, insurance, etc.) [ACCOUNT_NUMBER]. Exclude general \
mentions of accounts without a number.
- Medical device serial numbers or unique device identifiers [DEVICE_IDENTIFIER].
- IP addresses [IP_ADDRESS].
- Biometric identifiers such as fingerprints or voiceprints [BIOMETRIC_IDENTIFIER].
- Photographs or images attached or shared that could identify an individual [IMAGE]. \
Exclude descriptions of seeing or sharing images.
- Any other unique identifying number such as student IDs, medical record numbers, \
license numbers, or social media handles [IDENTIFYING_NUMBER]. Exclude problem numbers, \
scores, grades, mathematical expressions, and numbers used in an educational context.
Matches should be minimal exact spans of the sensitive value.
Output data following the provided JSON schema."""

DEIDENTIFY_V2_IDENTIFIERS_TYPES = [
    "ADDRESS",
    "SCHOOL",
    "DATE",
    "EMAIL",
    "URL",
    "PHONE_NUMBER",
    "FAX_NUMBER",
    "SSN",
    "ACCOUNT_NUMBER",
    "DEVICE_IDENTIFIER",
    "IP_ADDRESS",
    "BIOMETRIC_IDENTIFIER",
    "IMAGE",
    "IDENTIFYING_NUMBER",
]

DEIDENTIFY_V2_PRESIDIO_LITE_TYPES = ["EMAIL_ADDRESS", "PHONE_NUMBER", "IP_ADDRESS"]

DEIDENTIFY_V2_METHOD_DEFINITIONS: list[dict[str, Any]] = [
    {
        "id": "presidio-lite+extended-v2",
        "label": "Regex + LLM Extended v2",
        "description": "Exact compatibility port of the colleague demo v2 hybrid method.",
        "requires_presidio": True,
        "uses_llm": True,
        "supports_verify_override": False,
        "default_verify": False,
        "supported_label_profiles": ["advanced", "simple"],
        "known_limitations": ["Compatibility path preserves legacy demo-v2 labels."],
        "passes": [
            {
                "kind": "presidio",
                "presidio_entities": list(DEIDENTIFY_V2_PRESIDIO_LITE_TYPES),
                "entity_types": ["EMAIL", "PHONE_NUMBER", "IP_ADDRESS"],
                "pass_label": "presidio-lite+extended-v2:presidio",
            },
            {
                "kind": "llm",
                "prompt": DEIDENTIFY_V2_EXTENDED_PROMPT,
                "entity_types": list(DEIDENTIFY_V2_EXTENDED_TYPES),
                "pass_label": "presidio-lite+extended-v2:llm",
            },
        ],
    },
    {
        "id": "dual-v2",
        "label": "Dual LLM v2",
        "description": "Exact compatibility port of the colleague demo dual-v2 method.",
        "requires_presidio": False,
        "uses_llm": True,
        "supports_verify_override": False,
        "default_verify": False,
        "supported_label_profiles": ["advanced", "simple"],
        "known_limitations": ["Compatibility path preserves legacy demo-v2 labels."],
        "passes": [
            {
                "kind": "llm",
                "prompt": DEIDENTIFY_V2_NAMES_PROMPT,
                "entity_types": list(DEIDENTIFY_V2_NAMES_TYPES),
                "pass_label": "dual-v2:names",
            },
            {
                "kind": "llm",
                "prompt": DEIDENTIFY_V2_IDENTIFIERS_PROMPT,
                "entity_types": list(DEIDENTIFY_V2_IDENTIFIERS_TYPES),
                "pass_label": "dual-v2:identifiers",
            },
        ],
    },
    {
        "id": "regex+dual-v2",
        "label": "Regex + Dual LLM v2",
        "description": "Exact compatibility port of the colleague demo regex+dual-v2 method.",
        "requires_presidio": True,
        "uses_llm": True,
        "supports_verify_override": False,
        "default_verify": False,
        "supported_label_profiles": ["advanced", "simple"],
        "known_limitations": ["Compatibility path preserves legacy demo-v2 labels."],
        "passes": [
            {
                "kind": "presidio",
                "presidio_entities": list(DEIDENTIFY_V2_PRESIDIO_LITE_TYPES),
                "entity_types": ["EMAIL", "PHONE_NUMBER", "IP_ADDRESS"],
                "pass_label": "regex+dual-v2:presidio",
            },
            {
                "kind": "llm",
                "prompt": DEIDENTIFY_V2_NAMES_PROMPT,
                "entity_types": list(DEIDENTIFY_V2_NAMES_TYPES),
                "pass_label": "regex+dual-v2:names",
            },
            {
                "kind": "llm",
                "prompt": DEIDENTIFY_V2_IDENTIFIERS_PROMPT,
                "entity_types": list(DEIDENTIFY_V2_IDENTIFIERS_TYPES),
                "pass_label": "regex+dual-v2:identifiers",
            },
        ],
    },
]
DEIDENTIFY_V2_METHOD_DEFINITION_BY_ID: dict[str, dict[str, Any]] = {
    method["id"]: method for method in DEIDENTIFY_V2_METHOD_DEFINITIONS
}
METHOD_DEFINITIONS = AUDITED_METHOD_DEFINITIONS
METHOD_DEFINITION_BY_ID = AUDITED_METHOD_DEFINITION_BY_ID
METHOD_DEFINITIONS_BY_BUNDLE: dict[MethodBundleId, list[dict[str, Any]]] = {
    "legacy": LEGACY_METHOD_DEFINITIONS,
    "audited": METHOD_DEFINITIONS,
    "test": TEST_METHOD_DEFINITIONS,
    "v2": V2_METHOD_DEFINITIONS,
    "v2+post-process": V2_POST_PROCESS_METHOD_DEFINITIONS,
    "deidentify-v2": DEIDENTIFY_V2_METHOD_DEFINITIONS,
}
METHOD_DEFINITION_BY_ID_BY_BUNDLE: dict[MethodBundleId, dict[str, dict[str, Any]]] = {
    "legacy": LEGACY_METHOD_DEFINITION_BY_ID,
    "audited": METHOD_DEFINITION_BY_ID,
    "test": TEST_METHOD_DEFINITION_BY_ID,
    "v2": V2_METHOD_DEFINITION_BY_ID,
    "v2+post-process": V2_POST_PROCESS_METHOD_DEFINITION_BY_ID,
    "deidentify-v2": DEIDENTIFY_V2_METHOD_DEFINITION_BY_ID,
}

_presidio_analyzer_lock = threading.Lock()
_presidio_analyze_lock = threading.Lock()
_presidio_analyzer: Any | None = None


@dataclass
class LLMRunResult:
    spans: list[CanonicalSpan]
    warnings: list[str]
    llm_confidence: LLMConfidenceMetric
    finish_reason: str | None = None
    response_debug: list[str] = field(default_factory=list)
    raw_spans: list[CanonicalSpan] = field(default_factory=list)
    resolution_events: list[ResolutionEvent] = field(default_factory=list)
    resolution_policy_version: str | None = None


@dataclass
class MethodRunResult:
    spans: list[CanonicalSpan]
    warnings: list[str]
    llm_confidence: LLMConfidenceMetric | None = None
    response_debug: list[str] = field(default_factory=list)
    raw_spans: list[CanonicalSpan] = field(default_factory=list)
    resolution_events: list[ResolutionEvent] = field(default_factory=list)
    resolution_policy_version: str | None = None


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


def _truncate_debug_preview(value: str, *, limit: int = 400) -> str:
    compact = value.replace("\n", "\\n").strip()
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit]}..."


def _describe_payload_for_debug(value: Any) -> str:
    if value is None:
        return "None"
    if isinstance(value, str):
        return f"str(len={len(value)}, preview={_truncate_debug_preview(value)!r})"
    if isinstance(value, list):
        return f"list(len={len(value)})"
    if isinstance(value, dict):
        keys = ", ".join(sorted(str(key) for key in value.keys())[:6])
        return f"dict(keys=[{keys}])"
    return type(value).__name__


def _build_response_debug_summary(resp: Any) -> str:
    parts: list[str] = [f"resp_type={type(resp).__name__}"]
    choices = getattr(resp, "choices", None)
    parts.append(f"choices={_describe_payload_for_debug(choices)}")
    finish_reason = _extract_finish_reason(resp)
    if finish_reason:
        parts.append(f"finish_reason={finish_reason}")
    if choices:
        message = getattr(choices[0], "message", None)
        if message is None:
            parts.append("message=None")
        else:
            parts.append(
                f"message.content={_describe_payload_for_debug(getattr(message, 'content', None))}"
            )
            parts.append(
                f"message.tool_calls={_describe_payload_for_debug(getattr(message, 'tool_calls', None))}"
            )
    parts.append(f"output_text={_describe_payload_for_debug(getattr(resp, 'output_text', None))}")

    preview = ""
    for attr_name in ("model_dump", "dict", "json"):
        method = getattr(type(resp), attr_name, None)
        if not callable(method):
            continue
        try:
            payload = getattr(resp, attr_name)()
            if isinstance(payload, str):
                preview = payload
            else:
                preview = json.dumps(payload, default=repr, ensure_ascii=True)
        except Exception:
            continue
        if preview.strip():
            break
    if not preview:
        try:
            preview = repr(resp)
        except Exception:
            preview = ""
    if preview.strip():
        parts.append(f"raw_preview={_truncate_debug_preview(preview)!r}")
    return "; ".join(parts)


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

    if normalized.startswith(OPENAI_MODEL_PREFIXES):
        return "openai"

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
    normalized = model.lower()
    if normalized.startswith(("openai.gpt-5", "openai/gpt-5", "gpt-5", "chatgpt/gpt-5")):
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
    if normalized in {"openai.gpt-5.2-chat", "openai/gpt-5.2-chat", "gpt-5.2-chat"}:
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
    content = _extract_response_content(resp)
    if not content.strip():
        finish_reason = _extract_finish_reason(resp)
        response_debug = _build_response_debug_summary(resp)
        if finish_reason:
            raise ValueError(
                "LLM returned empty output content "
                f"(finish_reason={finish_reason}). response_debug={response_debug}"
            )
        raise ValueError(f"LLM returned empty output content. response_debug={response_debug}")
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


def _recover_partial_spans_from_truncated_output(raw_content: str) -> list[CanonicalSpan]:
    content = _strip_code_fences(raw_content).strip()
    if not content:
        return []

    spans_key_index = content.find('"spans"')
    if spans_key_index < 0:
        return []
    array_start = content.find("[", spans_key_index)
    if array_start < 0:
        return []

    decoder = json.JSONDecoder()
    recovered: list[CanonicalSpan] = []
    idx = array_start + 1
    while idx < len(content):
        while idx < len(content) and content[idx] in " \t\r\n,":
            idx += 1
        if idx >= len(content) or content[idx] == "]":
            break
        if content[idx] != "{":
            break
        try:
            payload, consumed = decoder.raw_decode(content[idx:])
        except json.JSONDecodeError:
            break
        if not isinstance(payload, dict):
            break
        try:
            recovered.append(
                CanonicalSpan(
                    start=int(payload["start"]),
                    end=int(payload["end"]),
                    label=str(payload["label"]),
                    text=str(payload["text"]),
                )
            )
        except Exception:
            break
        idx += consumed
    return recovered


def _compute_repair_max_tokens(
    *,
    raw_content: str,
    requested_max_tokens: object,
    default_repair_max_tokens: int,
) -> int:
    try:
        upper_bound = max(default_repair_max_tokens, int(requested_max_tokens or 0))
    except Exception:
        upper_bound = max(default_repair_max_tokens, DEFAULT_LLM_EXTRACTION_MAX_TOKENS)
    estimated_tokens = math.ceil(len(raw_content) / 4) + 256
    return max(default_repair_max_tokens, min(upper_bound, estimated_tokens))


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
    repair_timeout_seconds: float,
    repair_max_tokens: int,
) -> tuple[list[CanonicalSpan], Any, dict[str, Any] | None, Any | None]:
    raw_content = _extract_response_content(resp)
    if not raw_content.strip():
        finish_reason = _extract_finish_reason(resp)
        response_debug = _build_response_debug_summary(resp)
        if finish_reason:
            raise ValueError(
                "LLM returned empty output content "
                f"(finish_reason={finish_reason}). response_debug={response_debug}"
            )
        raise ValueError(f"LLM returned empty output content. response_debug={response_debug}")

    try:
        return _parse_spans_from_response(resp), resp, None, None
    except Exception as parse_exc:
        finish_reason = _extract_finish_reason(resp)
        partial_spans = (
            _recover_partial_spans_from_truncated_output(raw_content)
            if finish_reason == "length"
            else []
        )
        repair_kwargs = {
            key: value
            for key, value in request_kwargs.items()
            if key not in {"messages", "logprobs", "thinking"}
        }
        repair_kwargs["messages"] = [
            {"role": "system", "content": JSON_REPAIR_PROMPT},
            {
                "role": "user",
                "content": (
                    "Fix this invalid extractor output into valid JSON.\n"
                    "Use only spans recoverable from INVALID_OUTPUT. "
                    "Do not invent new spans or add prose.\n\n"
                    f"INVALID_OUTPUT:\n{raw_content}"
                ),
            },
        ]
        repair_kwargs["timeout"] = repair_timeout_seconds
        repair_kwargs["max_tokens"] = _compute_repair_max_tokens(
            raw_content=raw_content,
            requested_max_tokens=request_kwargs.get("max_tokens"),
            default_repair_max_tokens=repair_max_tokens,
        )
        repair_resp: Any | None = None
        try:
            repair_resp = completion(**repair_kwargs)
            repaired_spans = _parse_spans_from_response(repair_resp)
            warnings.append("Recovered invalid LLM JSON with one repair retry.")
            return repaired_spans, repair_resp, repair_kwargs, repair_resp
        except Exception as repair_exc:
            if partial_spans:
                warnings.append(
                    "Recovered "
                    f"{len(partial_spans)} span(s) from truncated LLM output after repair retry failed."
                )
                return partial_spans, resp, repair_kwargs, repair_resp
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


def _normalize_method_bundle(method_bundle: MethodBundleId | str | None) -> MethodBundleId:
    raw = str(method_bundle or DEFAULT_METHOD_BUNDLE).strip().lower()
    if raw not in {"legacy", "audited", "test", "v2", "v2+post-process", "deidentify-v2"}:
        raise ValueError(
            "method_bundle must be one of: legacy, audited, test, v2, "
            "v2+post-process, deidentify-v2"
        )
    return raw  # type: ignore[return-value]


def get_method_definitions(
    *,
    method_bundle: MethodBundleId = DEFAULT_METHOD_BUNDLE,
) -> list[dict[str, Any]]:
    bundle = _normalize_method_bundle(method_bundle)
    return METHOD_DEFINITIONS_BY_BUNDLE[bundle]


def get_method_definition_by_id(
    method_id: str,
    *,
    method_bundle: MethodBundleId = DEFAULT_METHOD_BUNDLE,
) -> dict[str, Any] | None:
    bundle = _normalize_method_bundle(method_bundle)
    return METHOD_DEFINITION_BY_ID_BY_BUNDLE[bundle].get(method_id)


def _supported_profiles_for_method(method: dict[str, Any]) -> list[Literal["simple", "advanced"]]:
    raw_profiles = method.get("supported_label_profiles")
    if isinstance(raw_profiles, list):
        profiles = [
            str(profile).strip().lower()
            for profile in raw_profiles
            if str(profile).strip().lower() in {"simple", "advanced"}
        ]
        if profiles:
            return sorted(set(profiles))  # type: ignore[return-value]
    return ["advanced", "simple"]


def _resolve_method_pass_prompt(
    method_pass: dict[str, Any],
    *,
    label_profile: Literal["simple", "advanced"],
) -> str:
    prompt_by_profile = method_pass.get("prompt_by_profile")
    if isinstance(prompt_by_profile, dict):
        value = prompt_by_profile.get(label_profile)
        if isinstance(value, str) and value.strip():
            return value
    return str(method_pass.get("prompt") or SYSTEM_PROMPT)


def _resolve_method_pass_entity_types(
    method_pass: dict[str, Any],
    *,
    label_profile: Literal["simple", "advanced"],
) -> list[str] | None:
    entity_types_by_profile = method_pass.get("entity_types_by_profile")
    if isinstance(entity_types_by_profile, dict):
        value = entity_types_by_profile.get(label_profile)
        if value is None:
            return None
        if isinstance(value, list):
            return [str(item) for item in value]
    raw_value = method_pass.get("entity_types")
    if raw_value is None:
        return None
    if isinstance(raw_value, list):
        return [str(item) for item in raw_value]
    return None


def _resolve_method_pass_requested_labels(
    method_pass: dict[str, Any],
    *,
    label_profile: Literal["simple", "advanced"],
) -> list[str]:
    requested_by_profile = method_pass.get("requested_labels_by_profile")
    if isinstance(requested_by_profile, dict):
        value = requested_by_profile.get(label_profile)
        if isinstance(value, list):
            return [str(item) for item in value]
    entity_types = _resolve_method_pass_entity_types(method_pass, label_profile=label_profile)
    return entity_types or []


def _resolve_presidio_pass_entities(
    method_pass: dict[str, Any],
    *,
    label_profile: Literal["simple", "advanced"],
) -> list[str] | None:
    presidio_by_profile = method_pass.get("presidio_entities_by_profile")
    if isinstance(presidio_by_profile, dict):
        value = presidio_by_profile.get(label_profile)
        if isinstance(value, list):
            return [str(item) for item in value]
    raw_value = method_pass.get("presidio_entities")
    if isinstance(raw_value, list):
        return [str(item) for item in raw_value]
    return _resolve_method_pass_entity_types(method_pass, label_profile=label_profile)


def _compute_method_output_labels(
    method: dict[str, Any],
    *,
    label_profile: Literal["simple", "advanced"],
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for method_pass in method.get("passes", []):
        if not isinstance(method_pass, dict):
            continue
        requested = _resolve_method_pass_requested_labels(
            method_pass,
            label_profile=label_profile,
        )
        for label in requested:
            normalized = normalize_method_label(label, label_profile=label_profile)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


def validate_method_contracts(
    *,
    method_bundle: MethodBundleId = DEFAULT_METHOD_BUNDLE,
) -> list[str]:
    if _normalize_method_bundle(method_bundle) == "deidentify-v2":
        return []

    errors: list[str] = []
    for method in get_method_definitions(method_bundle=method_bundle):
        method_id = str(method.get("id", ""))
        for label_profile in _supported_profiles_for_method(method):
            allowed_labels = set(_labels_for_profile(label_profile))
            for index, method_pass in enumerate(method.get("passes", []), start=1):
                if not isinstance(method_pass, dict):
                    continue
                pass_label = str(method_pass.get("pass_label") or f"{method_id}:pass_{index}")
                requested = _resolve_method_pass_requested_labels(
                    method_pass,
                    label_profile=label_profile,
                )
                normalized_requested = {
                    normalize_method_label(label, label_profile=label_profile)
                    for label in requested
                }
                normalized_requested.discard("")
                unsupported_requested = sorted(
                    label for label in normalized_requested if label not in allowed_labels
                )
                if unsupported_requested:
                    errors.append(
                        f"{method_id}/{label_profile}/{pass_label} requests labels outside the "
                        f"profile schema: {', '.join(unsupported_requested)}"
                    )
                pass_filter = _resolve_method_pass_entity_types(
                    method_pass,
                    label_profile=label_profile,
                )
                if pass_filter is not None:
                    normalized_filter = {
                        normalize_method_label(label, label_profile=label_profile)
                        for label in pass_filter
                    }
                    normalized_filter.discard("")
                    missing = sorted(normalized_requested - normalized_filter)
                    if missing:
                        errors.append(
                            f"{method_id}/{label_profile}/{pass_label} drops prompt-requested "
                            f"labels via entity filter: {', '.join(missing)}"
                        )
                if str(method_pass.get("kind")) == "presidio":
                    presidio_entities = _resolve_presidio_pass_entities(
                        method_pass,
                        label_profile=label_profile,
                    ) or []
                    normalized_presidio = {
                        normalize_method_label(label, label_profile=label_profile)
                        for label in presidio_entities
                    }
                    normalized_presidio.discard("")
                    unsupported_presidio = sorted(
                        label for label in normalized_presidio if label not in allowed_labels
                    )
                    if unsupported_presidio:
                        errors.append(
                            f"{method_id}/{label_profile}/{pass_label} includes Presidio "
                            f"entities outside the profile schema: {', '.join(unsupported_presidio)}"
                        )
    return errors


_AUDITED_METHOD_CONTRACT_ERRORS = tuple(validate_method_contracts(method_bundle="audited"))
if _AUDITED_METHOD_CONTRACT_ERRORS:
    joined_errors = "; ".join(_AUDITED_METHOD_CONTRACT_ERRORS)
    raise RuntimeError(f"Audited method contract validation failed: {joined_errors}")

_TEST_METHOD_CONTRACT_ERRORS = tuple(validate_method_contracts(method_bundle="test"))
if _TEST_METHOD_CONTRACT_ERRORS:
    joined_errors = "; ".join(_TEST_METHOD_CONTRACT_ERRORS)
    raise RuntimeError(f"Test method contract validation failed: {joined_errors}")


_V2_METHOD_CONTRACT_ERRORS = tuple(validate_method_contracts(method_bundle="v2"))
if _V2_METHOD_CONTRACT_ERRORS:
    joined_errors = "; ".join(_V2_METHOD_CONTRACT_ERRORS)
    raise RuntimeError(f"V2 method contract validation failed: {joined_errors}")


_V2_POST_PROCESS_METHOD_CONTRACT_ERRORS = tuple(
    validate_method_contracts(method_bundle="v2+post-process")
)
if _V2_POST_PROCESS_METHOD_CONTRACT_ERRORS:
    joined_errors = "; ".join(_V2_POST_PROCESS_METHOD_CONTRACT_ERRORS)
    raise RuntimeError(f"V2+post-process method contract validation failed: {joined_errors}")


_DEIDENTIFY_V2_METHOD_CONTRACT_ERRORS = tuple(
    validate_method_contracts(method_bundle="deidentify-v2")
)
if _DEIDENTIFY_V2_METHOD_CONTRACT_ERRORS:
    joined_errors = "; ".join(_DEIDENTIFY_V2_METHOD_CONTRACT_ERRORS)
    raise RuntimeError(f"deidentify-v2 method contract validation failed: {joined_errors}")


def _bundle_uses_detected_value_post_process(
    method_bundle: MethodBundleId | str | None,
) -> bool:
    return _normalize_method_bundle(method_bundle) == "v2+post-process"


def _bundle_preserves_native_labels(
    method_bundle: MethodBundleId | str | None,
) -> bool:
    return _normalize_method_bundle(method_bundle) == "deidentify-v2"


def _compute_native_method_output_labels(method: dict[str, Any]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for method_pass in method.get("passes", []):
        if not isinstance(method_pass, dict):
            continue
        requested = _resolve_method_pass_requested_labels(
            method_pass,
            label_profile="simple",
        )
        for label in requested:
            normalized = str(label).strip().upper()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


def _is_word_char(value: str) -> bool:
    return value.isalnum() or value == "_"


def _at_word_boundary(text: str, start: int, end: int) -> bool:
    if start > 0 and _is_word_char(text[start - 1]) and _is_word_char(text[start]):
        return False
    if end < len(text) and _is_word_char(text[end - 1]) and _is_word_char(text[end]):
        return False
    return True


def _expand_detected_value_occurrences(
    text: str,
    spans: list[CanonicalSpan],
) -> list[CanonicalSpan]:
    if not spans:
        return []

    expanded: list[CanonicalSpan] = []
    seen: set[tuple[str, str]] = set()
    ordered_spans = sorted(
        (span for span in spans if span.text),
        key=lambda item: len(item.text),
        reverse=True,
    )

    for span in ordered_spans:
        key = (span.label, span.text)
        if key in seen:
            continue

        start_index = 0
        while start_index < len(text):
            found = text.find(span.text, start_index)
            if found == -1:
                break

            new_start = found
            new_end = found + len(span.text)
            if not _at_word_boundary(text, new_start, new_end):
                start_index = found + 1
                continue

            candidate = CanonicalSpan(
                start=new_start,
                end=new_end,
                label=span.label,
                text=text[new_start:new_end],
            )

            overlapping_index: int | None = None
            for index, existing in enumerate(expanded):
                if new_end > existing.start and new_start < existing.end:
                    overlapping_index = index
                    break

            if overlapping_index is None:
                expanded.append(candidate)
            else:
                existing = expanded[overlapping_index]
                if (candidate.end - candidate.start) > (existing.end - existing.start):
                    expanded[overlapping_index] = candidate

            start_index = found + len(span.text)

        seen.add(key)

    return expanded


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


@lru_cache(maxsize=1)
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


def _get_presidio_analyzer():
    global _presidio_analyzer

    analyzer = _presidio_analyzer
    if analyzer is not None:
        return analyzer

    with _presidio_analyzer_lock:
        analyzer = _presidio_analyzer
        if analyzer is not None:
            return analyzer

        from presidio_analyzer import AnalyzerEngine
        from presidio_analyzer.nlp_engine import NlpEngineProvider

        config = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": PRESIDIO_DEFAULT_MODEL}],
        }
        nlp_engine = NlpEngineProvider(nlp_configuration=config).create_engine()
        analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
        _presidio_analyzer = analyzer
        return analyzer


def _reset_presidio_runtime_state():
    global _presidio_analyzer

    _presidio_runtime_error.cache_clear()
    with _presidio_analyzer_lock:
        _presidio_analyzer = None


def list_agent_methods(
    *,
    method_bundle: MethodBundleId = DEFAULT_METHOD_BUNDLE,
) -> list[dict[str, Any]]:
    bundle = _normalize_method_bundle(method_bundle)
    presidio_error = _presidio_runtime_error()
    methods: list[dict[str, Any]] = []
    for method in get_method_definitions(method_bundle=bundle):
        unavailable_reason = None
        if method["requires_presidio"] and presidio_error:
            unavailable_reason = presidio_error
        prompt_templates: list[dict[str, Any]] = []
        for pass_index, method_pass in enumerate(method.get("passes", []), start=1):
            if method_pass.get("kind") != "llm":
                continue
            system_prompt_by_profile = {
                profile: _resolve_method_pass_prompt(
                    method_pass,
                    label_profile=profile,  # type: ignore[arg-type]
                )
                for profile in _supported_profiles_for_method(method)
            }
            entity_types_by_profile = {
                profile: _resolve_method_pass_entity_types(
                    method_pass,
                    label_profile=profile,  # type: ignore[arg-type]
                )
                for profile in _supported_profiles_for_method(method)
            }
            prompt_templates.append(
                {
                    "pass_index": pass_index,
                    "entity_types": entity_types_by_profile.get("simple"),
                    "entity_types_by_profile": entity_types_by_profile,
                    "system_prompt": system_prompt_by_profile.get("simple", SYSTEM_PROMPT),
                    "system_prompt_by_profile": system_prompt_by_profile,
                }
            )
        supported_profiles = _supported_profiles_for_method(method)
        if bundle == "deidentify-v2":
            native_output_labels = _compute_native_method_output_labels(method)
            output_labels_by_profile = {
                profile: list(native_output_labels) for profile in supported_profiles
            }
        else:
            output_labels_by_profile = {
                profile: _compute_method_output_labels(
                    method,
                    label_profile=profile,  # type: ignore[arg-type]
                )
                for profile in supported_profiles
            }
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
                "supported_label_profiles": supported_profiles,
                "output_labels_by_profile": output_labels_by_profile,
                "known_limitations": list(method.get("known_limitations", [])),
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


def _clamp_call_timeout_seconds(
    timeout_seconds: float | None,
    *,
    default_timeout_seconds: float,
) -> float:
    if timeout_seconds is None:
        return default_timeout_seconds
    return max(0.01, min(float(timeout_seconds), default_timeout_seconds))


def _remaining_timeout_seconds(deadline_monotonic: float | None) -> float | None:
    if deadline_monotonic is None:
        return None
    remaining = deadline_monotonic - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("LLM task exhausted its timeout budget before the next provider call.")
    return remaining


def _run_llm_verifier(
    text: str,
    spans: list[CanonicalSpan],
    *,
    api_key: str,
    api_base: str | None,
    model: str,
    timeout_seconds: float | None = None,
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
        "timeout": _clamp_call_timeout_seconds(
            timeout_seconds,
            default_timeout_seconds=DEFAULT_LLM_VERIFIER_TIMEOUT_SECONDS,
        ),
        "max_tokens": DEFAULT_LLM_VERIFIER_MAX_TOKENS,
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
    analyzer = _get_presidio_analyzer()
    with _presidio_analyze_lock:
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


_DEIDENTIFY_V2_PRESIDIO_LABEL_MAP: dict[str, str] = {
    "PERSON": "NAME",
    "LOCATION": "ADDRESS",
    "DATE_TIME": "DATE",
    "EMAIL_ADDRESS": "EMAIL",
    "US_SSN": "SSN",
    "US_BANK_NUMBER": "ACCOUNT_NUMBER",
    "CREDIT_CARD": "ACCOUNT_NUMBER",
}


def _build_deidentify_v2_response_format(entity_types: list[str]) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "pii_matches",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "matches": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "entity_type": {"type": "string", "enum": list(entity_types)},
                                "text": {"type": "string"},
                            },
                            "required": ["entity_type", "text"],
                            "additionalProperties": False,
                        },
                    }
                },
                "required": ["matches"],
                "additionalProperties": False,
            },
        },
    }


def _map_deidentify_v2_presidio_spans(spans: list[CanonicalSpan]) -> list[CanonicalSpan]:
    return [
        CanonicalSpan(
            start=span.start,
            end=span.end,
            label=_DEIDENTIFY_V2_PRESIDIO_LABEL_MAP.get(span.label, span.label),
            text=span.text,
        )
        for span in spans
    ]


def _parse_deidentify_v2_matches(content: str | None) -> list[dict[str, str]]:
    payload = json.loads(content or '{"matches":[]}')
    if not isinstance(payload, dict):
        return []
    matches = payload.get("matches", [])
    if not isinstance(matches, list):
        return []

    parsed: list[dict[str, str]] = []
    for item in matches:
        if not isinstance(item, dict):
            continue
        entity_type = str(item.get("entity_type", "")).strip()
        text = str(item.get("text", ""))
        if not entity_type or not text:
            continue
        parsed.append({"entity_type": entity_type, "text": text})
    return parsed


def _run_deidentify_v2_llm_pass(
    *,
    text: str,
    api_key: str,
    api_base: str | None,
    model: str,
    prompt: str,
    entity_types: list[str],
) -> tuple[list[CanonicalSpan], str]:
    use_openai_gateway_format = bool(api_base) and "/" not in model and "." in model
    request_kwargs: dict[str, Any] = {
        "model": f"openai/{model}" if use_openai_gateway_format else model,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": text},
        ],
        "response_format": _build_deidentify_v2_response_format(entity_types),
        "api_key": api_key,
        "num_retries": 1,
    }
    if api_base:
        request_kwargs["api_base"] = api_base

    response = completion(**request_kwargs)
    matches = _parse_deidentify_v2_matches(_extract_response_content(response))
    seed_spans = [
        CanonicalSpan(start=0, end=0, label=item["entity_type"], text=item["text"])
        for item in matches
    ]
    spans = merge_method_spans(_expand_detected_value_occurrences(text, seed_spans))
    return spans, _build_response_debug_summary(response)


def _run_deidentify_v2_method_with_metadata(
    *,
    text: str,
    method: dict[str, Any],
    method_id: str,
    api_key: str | None,
    api_base: str | None,
    model: str,
    system_prompt: str,
    method_verify: bool | None,
    progress_callback: Callable[[int, str], None] | None = None,
) -> MethodRunResult:
    if method["requires_presidio"]:
        runtime_error = _presidio_runtime_error()
        if runtime_error is not None:
            raise ValueError(runtime_error)

    if method["uses_llm"] and not api_key:
        raise ValueError(
            f"Method '{method_id}' requires an API key. Set LITELLM_API_KEY or provide api_key."
        )

    warnings: list[str] = []
    if system_prompt.strip():
        warnings.append(
            "deidentify-v2 compatibility methods ignore additional system prompt constraints "
            "to preserve the colleague demo prompts exactly."
        )
    if method_verify:
        warnings.append(
            "deidentify-v2 compatibility methods do not attach a verifier; "
            "method_verify was ignored."
        )

    all_raw_spans: list[CanonicalSpan] = []
    response_debug: list[str] = []

    for idx, method_pass in enumerate(method["passes"]):
        pass_label = str(method_pass.get("pass_label") or f"{method_id}:pass_{idx + 1}")
        if progress_callback is not None:
            progress_callback(idx + 1, pass_label)

        if str(method_pass.get("kind")) == "presidio":
            raw_spans = _run_presidio_pass(
                text=text,
                entity_types=[str(item) for item in method_pass.get("presidio_entities", [])],
            )
            all_raw_spans.extend(_map_deidentify_v2_presidio_spans(raw_spans))
            continue

        spans, debug_summary = _run_deidentify_v2_llm_pass(
            text=text,
            api_key=api_key or "",
            api_base=api_base,
            model=model,
            prompt=str(method_pass.get("prompt") or SYSTEM_PROMPT),
            entity_types=[str(item) for item in method_pass.get("entity_types", [])],
        )
        response_debug.append(f"Pass {idx + 1}: {debug_summary}")
        all_raw_spans.extend(spans)

    raw_spans = merge_method_spans(all_raw_spans)
    return MethodRunResult(
        spans=list(raw_spans),
        warnings=warnings,
        llm_confidence=None,
        response_debug=response_debug,
        raw_spans=raw_spans,
        resolution_events=[],
        resolution_policy_version=None,
    )


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
    timeout_seconds: float | None = None,
    progress_callback: Callable[[int, str], None] | None = None,
    method_bundle: MethodBundleId = DEFAULT_METHOD_BUNDLE,
) -> MethodRunResult:
    normalized_method_bundle = _normalize_method_bundle(method_bundle)
    method = get_method_definition_by_id(method_id, method_bundle=normalized_method_bundle)
    if method is None:
        raise ValueError(f"Unknown method: {method_id}")

    if normalized_method_bundle == "deidentify-v2":
        return _run_deidentify_v2_method_with_metadata(
            text=text,
            method=method,
            method_id=method_id,
            api_key=api_key,
            api_base=api_base,
            model=model,
            system_prompt=system_prompt,
            method_verify=method_verify,
            progress_callback=progress_callback,
        )

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
    all_raw_spans: list[CanonicalSpan] = []
    llm_metrics: list[LLMConfidenceMetric] = []
    response_debug: list[str] = []
    deadline_monotonic = (
        time.monotonic() + float(timeout_seconds)
        if timeout_seconds is not None
        else None
    )

    for idx, method_pass in enumerate(method["passes"]):
        kind = method_pass["kind"]
        pass_label = str(method_pass.get("pass_label") or f"{method_id}:pass_{idx + 1}")
        pass_entity_types = _resolve_method_pass_entity_types(
            method_pass,
            label_profile=label_profile,
        )
        if progress_callback is not None:
            progress_callback(idx + 1, pass_label)
        if kind == "presidio":
            presidio_entities = _resolve_presidio_pass_entities(
                method_pass,
                label_profile=label_profile,
            )
            pass_spans = _run_presidio_pass(
                text=text,
                entity_types=presidio_entities,
            )
            all_raw_spans.extend(
                _filter_method_pass_spans(
                    normalize_method_spans(pass_spans, label_profile=label_profile),
                    pass_entity_types,
                    label_profile=label_profile,
                )
            )
            continue

        prompt = _resolve_method_pass_prompt(
            method_pass,
            label_profile=label_profile,
        )
        if system_prompt.strip():
            prompt = f"{prompt}\n\nAdditional constraints:\n{system_prompt.strip()}"

        call_timeout_seconds = _remaining_timeout_seconds(deadline_monotonic)
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
            timeout_seconds=call_timeout_seconds,
            enable_deterministic_augmentation=False,
            method_bundle=normalized_method_bundle,
        )
        llm_metrics.append(llm_result.llm_confidence)
        response_debug.extend(
            [f"Pass {idx + 1}: {item}" for item in getattr(llm_result, "response_debug", [])]
        )
        raw_pass_spans = _filter_method_pass_spans(
            normalize_method_spans(
                getattr(llm_result, "raw_spans", llm_result.spans),
                label_profile=label_profile,
            ),
            pass_entity_types,
            label_profile=label_profile,
        )
        all_raw_spans.extend(raw_pass_spans)
        for warning in llm_result.warnings:
            warnings.append(f"Pass {idx + 1}: {warning}")

    raw_spans = merge_method_spans(all_raw_spans)
    if _bundle_uses_detected_value_post_process(normalized_method_bundle):
        raw_spans = merge_method_spans(_expand_detected_value_occurrences(text, raw_spans))
    pre_verify_resolved, pre_verify_events = resolve_spans(
        text,
        raw_spans,
        label_profile=label_profile,
        enable_augmentation=False,
    )
    merged = merge_method_spans(pre_verify_resolved)

    if should_verify:
        try:
            if progress_callback is not None:
                progress_callback(len(method["passes"]) + 1, f"{method_id}:verifier")
            verifier_timeout_seconds = _remaining_timeout_seconds(deadline_monotonic)
            response_debug.append(
                "Verifier settings: "
                f"timeout={_clamp_call_timeout_seconds(verifier_timeout_seconds, default_timeout_seconds=DEFAULT_LLM_VERIFIER_TIMEOUT_SECONDS):g}; "
                f"max_tokens={DEFAULT_LLM_VERIFIER_MAX_TOKENS}"
            )
            verified_spans, verify_warnings = _run_llm_verifier(
                text=text,
                spans=merged,
                api_key=api_key or "",
                api_base=api_base,
                model=model,
                timeout_seconds=verifier_timeout_seconds,
            )
            merged = merge_method_spans(
                normalize_method_spans(verified_spans, label_profile=label_profile)
            )
            warnings.extend(verify_warnings)
        except Exception as exc:
            raise ValueError(f"Method '{method_id}' verifier failed: {exc}") from exc

    merged, post_verify_events = resolve_spans(
        text,
        merged,
        label_profile=label_profile,
        enable_augmentation=True,
    )
    merged = merge_method_spans(merged)
    merged, dropped_name_spans = _drop_implausible_name_spans(
        text,
        merged,
        label_profile=label_profile,
    )
    if dropped_name_spans > 0:
        target_label = "NAME" if label_profile == "simple" else "PERSON"
        warnings.append(
            f"Dropped {dropped_name_spans} implausible {target_label} span(s) from method output."
        )

    return MethodRunResult(
        spans=merged,
        warnings=warnings,
        llm_confidence=_aggregate_llm_confidence_metrics(llm_metrics),
        response_debug=response_debug,
        raw_spans=raw_spans,
        resolution_events=[*pre_verify_events, *post_verify_events],
        resolution_policy_version=RESOLUTION_POLICY_VERSION,
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
    timeout_seconds: float | None = None,
    enable_deterministic_augmentation: bool = True,
    method_bundle: MethodBundleId = DEFAULT_METHOD_BUNDLE,
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
        method_bundle=method_bundle,
    )
    deadline_monotonic = (
        time.monotonic() + float(timeout_seconds)
        if timeout_seconds is not None
        else None
    )
    effective_timeout_seconds = _clamp_call_timeout_seconds(
        timeout_seconds,
        default_timeout_seconds=DEFAULT_LLM_EXTRACTION_TIMEOUT_SECONDS,
    )
    effective_max_tokens = DEFAULT_LLM_EXTRACTION_MAX_TOKENS

    if anthropic_thinking and supports_thinking:
        if temperature != 1.0:
            warnings.append(
                f"Model '{model}' requires temperature=1 when thinking is enabled; "
                "overriding requested temperature to 1.0."
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
        "timeout": effective_timeout_seconds,
    }
    supports_logprobs = _supports_logprobs(
        model,
        provider,
        reasoning_effort=reasoning_effort,
    )
    if supports_logprobs:
        base_request_kwargs["logprobs"] = True
    elif model.lower() in {"openai.gpt-5.2-chat", "openai/gpt-5.2-chat", "gpt-5.2-chat"}:
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
                effective_max_tokens = max(
                    anthropic_thinking_budget_tokens + 256,
                    DEFAULT_LLM_EXTRACTION_MAX_TOKENS,
                )
            if use_openai_gateway_format:
                extra_body["thinking"] = thinking
            else:
                base_request_kwargs["thinking"] = thinking
        else:
            warnings.append(f"Model '{model}' does not support anthropic thinking; ignored.")
    if use_openai_gateway_format and anthropic_thinking and supports_thinking:
        extra_body["max_tokens"] = effective_max_tokens
    else:
        base_request_kwargs["max_tokens"] = effective_max_tokens
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
            response_debug = [
                "request_settings: "
                f"provider={provider}; timeout={effective_timeout_seconds:g}; "
                f"max_tokens={effective_max_tokens}; label_profile={label_profile}; "
                f"reasoning_effort={reasoning_effort}; anthropic_thinking={anthropic_thinking}",
                _build_response_debug_summary(resp),
            ]
            repair_timeout_seconds = _clamp_call_timeout_seconds(
                _remaining_timeout_seconds(deadline_monotonic),
                default_timeout_seconds=DEFAULT_LLM_REPAIR_TIMEOUT_SECONDS,
            )
            spans, parsed_resp, repair_kwargs, repair_resp = _parse_with_one_repair_retry(
                resp=resp,
                request_kwargs=effective_kwargs,
                text=text,
                warnings=warnings,
                repair_timeout_seconds=repair_timeout_seconds,
                repair_max_tokens=DEFAULT_LLM_REPAIR_MAX_TOKENS,
            )
            if repair_kwargs is not None:
                response_debug.append(
                    "repair_settings: "
                    f"timeout={repair_kwargs['timeout']:g}; max_tokens={repair_kwargs['max_tokens']}"
                )
                if repair_resp is not None:
                    response_debug.append(f"repair: {_build_response_debug_summary(repair_resp)}")
            spans, repair_warnings = _repair_offset_mismatches(
                text,
                spans,
                label_profile=label_profile,
            )
            warnings.extend(repair_warnings)
            raw_spans = normalize_method_spans(spans, label_profile=label_profile)
            spans, resolution_events = resolve_spans(
                text,
                raw_spans,
                label_profile=label_profile,
                enable_augmentation=enable_deterministic_augmentation,
            )
            spans = merge_method_spans(spans)
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
            boundary_fix_count = sum(
                1 for event in resolution_events if event.kind == "boundary_resolution"
            )
            if boundary_fix_count > 0:
                target_label = "NAME" if label_profile == "simple" else "PERSON"
                warnings.append(
                    f"Resolved {boundary_fix_count} supported boundary issue(s) including {target_label} affixes."
                )
            augmentation_count = sum(
                1 for event in resolution_events if event.kind == "augmentation"
            )
            if augmentation_count > 0:
                warnings.append(
                    f"Recovered {augmentation_count} deterministic span(s) from context-aware augmentation."
                )
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
                response_debug=response_debug,
                raw_spans=raw_spans,
                resolution_events=resolution_events,
                resolution_policy_version=RESOLUTION_POLICY_VERSION,
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
