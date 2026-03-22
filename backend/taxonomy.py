from __future__ import annotations

import re

from models import CanonicalSpan

CANONICAL_LABELS: list[str] = [
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
    "AGE",
    "SCHOOL",
    "TUTOR_PROVIDER",
    "CUSTOMIZED_FIELD",
    "OTHER_LOCATIONS_IDENTIFIED",
]
CANONICAL_LABEL_SET = set(CANONICAL_LABELS)

_DROP_LEGACY_LABELS = {"COURSE", "GRADE_LEVEL", "NRP"}

_TUTOR_PROVIDER_TEXTS = {
    "upchieve",
    "saga",
    "saga education",
    "schoolhouse",
    "schoolhouse world",
    "schoolhouse.world",
    "varsity tutors",
    "kumon",
    "mathnasium",
    "sylvan",
    "wyzant",
}

_BROAD_GEO_TEXTS = {
    "alabama",
    "alaska",
    "arizona",
    "arkansas",
    "california",
    "colorado",
    "connecticut",
    "delaware",
    "florida",
    "georgia",
    "hawaii",
    "idaho",
    "illinois",
    "indiana",
    "iowa",
    "kansas",
    "kentucky",
    "louisiana",
    "maine",
    "maryland",
    "massachusetts",
    "michigan",
    "minnesota",
    "mississippi",
    "missouri",
    "montana",
    "nebraska",
    "nevada",
    "new hampshire",
    "new jersey",
    "new mexico",
    "new york",
    "north carolina",
    "north dakota",
    "ohio",
    "oklahoma",
    "oregon",
    "pennsylvania",
    "rhode island",
    "south carolina",
    "south dakota",
    "tennessee",
    "texas",
    "utah",
    "vermont",
    "virginia",
    "washington",
    "west virginia",
    "wisconsin",
    "wyoming",
    "district of columbia",
    "united states",
    "united states of america",
    "usa",
    "us",
    "u s",
    "america",
    "canada",
    "mexico",
    "europe",
    "asia",
    "africa",
    "south america",
    "north america",
    "middle east",
}


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().lower()).strip()


def is_tutor_provider_text(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    if normalized in _TUTOR_PROVIDER_TEXTS:
        return True
    return any(
        normalized == candidate or normalized.startswith(f"{candidate} ")
        for candidate in _TUTOR_PROVIDER_TEXTS
    )


def is_broad_geography_text(text: str) -> bool:
    normalized = _normalize_text(text)
    if not normalized:
        return False
    return normalized in _BROAD_GEO_TEXTS


def canonicalize_label(label: str, *, text: str | None = None) -> str | None:
    normalized = str(label or "").strip().upper()
    if not normalized:
        return None
    if normalized in _DROP_LEGACY_LABELS:
        return None

    if normalized in {
        "NAME",
        "PERSON",
        "PER",
        "PERSON_NAME",
        "FIRST_NAME",
        "LAST_NAME",
        "FULL_NAME",
        "GIVEN_NAME",
        "SURNAME",
        "NAME_STUDENT",
    }:
        return "NAME"

    if normalized in {"ADDRESS", "LOCATION", "GPE"}:
        if text is not None and is_broad_geography_text(text):
            return None
        return "ADDRESS"

    if normalized in {"DATE", "DATE_TIME", "TIME", "TIMESTAMP"}:
        return "DATE"
    if normalized == "AGE":
        return "AGE"
    if normalized in {"PHONE", "PHONE_NUMBER"}:
        return "PHONE_NUMBER"
    if normalized == "FAX_NUMBER":
        return "FAX_NUMBER"
    if normalized in {"EMAIL", "EMAIL_ADDRESS"}:
        return "EMAIL"
    if normalized in {"SSN", "US_SSN"}:
        return "SSN"
    if normalized in {"ACCOUNT_NUMBER", "US_BANK_NUMBER", "CREDIT_CARD"}:
        return "ACCOUNT_NUMBER"
    if normalized == "DEVICE_IDENTIFIER":
        return "DEVICE_IDENTIFIER"
    if normalized in {"URL", "URI"}:
        return "URL"
    if normalized in {"IP", "IP_ADDRESS"}:
        return "IP_ADDRESS"
    if normalized == "BIOMETRIC_IDENTIFIER" or "VOICEPRINT" in normalized or "FINGERPRINT" in normalized:
        return "BIOMETRIC_IDENTIFIER"
    if normalized in {"IMAGE", "PHOTO", "PHOTOGRAPH"}:
        return "IMAGE"
    if normalized == "SCHOOL":
        if text is not None and is_tutor_provider_text(text):
            return "TUTOR_PROVIDER"
        return "SCHOOL"
    if normalized == "TUTOR_PROVIDER":
        return "TUTOR_PROVIDER"
    if normalized in {"IDENTIFYING_NUMBER", "MISC_ID", "SOCIAL_HANDLE", "US_DRIVER_LICENSE", "US_PASSPORT"}:
        return "IDENTIFYING_NUMBER"
    if any(
        token in normalized
        for token in (
            "PASSWORD",
            "USERNAME",
            "USER_NAME",
            "HANDLE",
            "LICENSE",
            "PASSPORT",
            "IDENTIFIER",
            "ACCOUNT_ID",
            "STUDENT_ID",
            "MEDICAL_RECORD",
            "RECORD_NUMBER",
            "AUTH_CODE",
            "ACCESS_CODE",
        )
    ):
        return "IDENTIFYING_NUMBER"
    if normalized in CANONICAL_LABEL_SET:
        return normalized
    return None


def canonicalize_span(span: CanonicalSpan, raw_text: str) -> CanonicalSpan | None:
    if span.start < 0 or span.end > len(raw_text) or span.start >= span.end:
        raise ValueError(
            f"Invalid span offsets [{span.start}:{span.end}] for transcript length {len(raw_text)}."
        )
    exact_text = raw_text[span.start : span.end]
    label = canonicalize_label(span.label, text=exact_text)
    if label is None:
        return None
    return CanonicalSpan(
        start=span.start,
        end=span.end,
        label=label,
        text=exact_text,
    )
