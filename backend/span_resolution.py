from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from models import CanonicalSpan, ResolutionEvent

RESOLUTION_POLICY_VERSION = "2026-03-span-resolution-v2"

_NAME_HONORIFIC_PREFIX_RE = re.compile(r"^(Mr\.?|Mrs\.?|Ms\.?|Miss)\s+", re.IGNORECASE)
_NAME_HONORIFIC_SUFFIX_RE = re.compile(r"(Mr\.?|Mrs\.?|Ms\.?|Miss)\s*$", re.IGNORECASE)
_NAME_TRAILING_POSSESSIVE_SUFFIXES = ("'s", "’s")
_NAME_TRAILING_PUNCTUATION = "."

_DOMAIN_LABEL_PATTERN = r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
_DOMAIN_SEPARATOR_PATTERN = r"\s*\.\s*"
_DOMAIN_PATTERN = (
    rf"(?:{_DOMAIN_LABEL_PATTERN}{_DOMAIN_SEPARATOR_PATTERN})+[a-z]{{2,63}}"
)
_URL_PATTERN = re.compile(
    rf"(?<!@)\b(?:https?://)?{_DOMAIN_PATTERN}(?:/[^\s,)]+)?",
    re.IGNORECASE,
)

_IDENTIFIER_CONTEXT_PATTERNS = [
    re.compile(
        r"\b(?:password|username|login|pin|meeting code|room code|code|id)\b"
        r"(?:\s+(?:is|was|:|-))?\s+([A-Z][A-Z0-9-]{1,11})\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b([A-Z][A-Z0-9-]{1,11})\b"
        r"(?:\s+(?:is|was))?\s+(?:the\s+)?"
        r"(?:password|username|login|pin|meeting code|room code|code|id)\b",
        re.IGNORECASE,
    ),
]

_AGE_CONTEXT_PATTERNS = [
    re.compile(r"\b(?:age|aged)\s+([a-z]+(?:-[a-z]+)?|\d{1,3})\b", re.IGNORECASE),
    re.compile(r"\b([a-z]+(?:-[a-z]+)?|\d{1,3})\s+years?\s+old\b", re.IGNORECASE),
    re.compile(
        r"\b(?:my|our|his|her|their)\s+"
        r"(?:brother|sister|son|daughter|cousin|friend|student|child)\s+is\s+"
        r"([a-z]+(?:-[a-z]+)?|\d{1,3})\b",
        re.IGNORECASE,
    ),
]
_AGE_ONES = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
}
_AGE_TEENS = {
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_AGE_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}


@dataclass
class SpanResolutionResult:
    spans: list[CanonicalSpan]
    events: list[ResolutionEvent]


def canonicalize_name_affix_text(value: str) -> str:
    cleaned = value.strip()
    cleaned = cleaned.strip(" \t\r\n.,!?;:\"'()[]{}")
    cleaned = _NAME_HONORIFIC_PREFIX_RE.sub("", cleaned)
    cleaned = cleaned.strip(" \t\r\n.,!?;:\"'()[]{}")
    for suffix in _NAME_TRAILING_POSSESSIVE_SUFFIXES:
        if cleaned.lower().endswith(suffix.lower()):
            cleaned = cleaned[: -len(suffix)]
            break
    cleaned = cleaned.strip(" \t\r\n.,!?;:\"'()[]{}")
    cleaned = " ".join(cleaned.split())
    return cleaned.casefold()


def _span_key(span: CanonicalSpan) -> tuple[int, int, str]:
    return span.start, span.end, span.label


def _overlaps(a: CanonicalSpan, b: CanonicalSpan) -> bool:
    return not (a.end <= b.start or b.end <= a.start)


def _sorted_unique_spans(spans: list[CanonicalSpan]) -> list[CanonicalSpan]:
    seen: set[tuple[int, int, str]] = set()
    result: list[CanonicalSpan] = []
    for span in sorted(spans, key=lambda item: (item.start, item.end, item.label, item.text)):
        key = _span_key(span)
        if key in seen:
            continue
        seen.add(key)
        result.append(span)
    return result


def _resolve_name_boundary(raw_text: str, span: CanonicalSpan) -> CanonicalSpan:
    start = span.start
    end = span.end
    if not (0 <= start < end <= len(raw_text)):
        return span

    while start < end and raw_text[start].isspace():
        start += 1
    while end > start and raw_text[end - 1].isspace():
        end -= 1

    core = canonicalize_name_affix_text(raw_text[start:end])
    if not core:
        return CanonicalSpan(start=start, end=end, label=span.label, text=raw_text[start:end])

    prefix = raw_text[max(0, start - 16) : start]
    honorific_match = None
    for match in _NAME_HONORIFIC_SUFFIX_RE.finditer(prefix):
        honorific_match = match
    if honorific_match is not None:
        candidate_start = start - (len(prefix) - honorific_match.start())
        candidate_text = raw_text[candidate_start:end]
        if canonicalize_name_affix_text(candidate_text) == core:
            start = candidate_start

    for suffix in _NAME_TRAILING_POSSESSIVE_SUFFIXES:
        if raw_text.startswith(suffix, end):
            candidate_end = end + len(suffix)
            if canonicalize_name_affix_text(raw_text[start:candidate_end]) == core:
                end = candidate_end
                break

    while end < len(raw_text) and raw_text[end] in _NAME_TRAILING_PUNCTUATION:
        candidate_end = end + 1
        if candidate_end < len(raw_text) and raw_text[candidate_end].isalnum():
            break
        if canonicalize_name_affix_text(raw_text[start:candidate_end]) != core:
            break
        end = candidate_end

    return CanonicalSpan(start=start, end=end, label=span.label, text=raw_text[start:end])


def _resolve_url_boundary(raw_text: str, span: CanonicalSpan) -> CanonicalSpan:
    if not (0 <= span.start < span.end <= len(raw_text)):
        return span

    start = span.start
    end = span.end
    while start < end and raw_text[start].isspace():
        start += 1
    while end > start and raw_text[end - 1].isspace():
        end -= 1

    best: tuple[int, int] | None = None
    for match in _URL_PATTERN.finditer(raw_text):
        match_start, match_end = match.span()
        if match_start <= start and match_end >= end:
            candidate = (match_start, match_end)
            if best is None or (candidate[1] - candidate[0]) < (best[1] - best[0]):
                best = candidate
    if best is None:
        return CanonicalSpan(start=start, end=end, label=span.label, text=raw_text[start:end])
    return CanonicalSpan(
        start=best[0],
        end=best[1],
        label=span.label,
        text=raw_text[best[0] : best[1]],
    )


def _make_event(
    *,
    kind: Literal["boundary_resolution", "augmentation"],
    rule: str,
    before: CanonicalSpan | None,
    after: CanonicalSpan,
) -> ResolutionEvent:
    return ResolutionEvent(
        kind=kind,
        label=after.label,
        rule=rule,
        before=before,
        after=after,
    )


def _resolve_boundaries(
    raw_text: str,
    spans: list[CanonicalSpan],
    *,
    label_profile: Literal["simple", "advanced"],
) -> SpanResolutionResult:
    resolved: list[CanonicalSpan] = []
    events: list[ResolutionEvent] = []
    for span in spans:
        updated = span
        rule: str | None = None
        if span.label == "URL":
            updated = _resolve_url_boundary(raw_text, span)
            if _span_key(updated) != _span_key(span) or updated.text != span.text:
                rule = "url_multiline_domain"
        resolved.append(updated)
        if rule is not None:
            events.append(
                _make_event(
                    kind="boundary_resolution",
                    rule=rule,
                    before=span,
                    after=updated,
                )
            )
    return SpanResolutionResult(spans=_sorted_unique_spans(resolved), events=events)


def _candidate_is_identifier_like(value: str) -> bool:
    cleaned = value.strip()
    if not re.fullmatch(r"[A-Z][A-Z0-9-]{1,11}", cleaned):
        return False
    if cleaned.isdigit():
        return False
    return True


def _parse_age_value(value: str) -> int | None:
    cleaned = value.strip().lower()
    if not cleaned:
        return None
    if cleaned.isdigit():
        number = int(cleaned)
        return number if 0 <= number <= 130 else None
    normalized = cleaned.replace("-", " ")
    if normalized in _AGE_ONES:
        return _AGE_ONES[normalized]
    if normalized in _AGE_TEENS:
        return _AGE_TEENS[normalized]
    if normalized in _AGE_TENS:
        return _AGE_TENS[normalized]
    parts = [part for part in normalized.split() if part]
    if len(parts) == 2 and parts[0] in _AGE_TENS and parts[1] in _AGE_ONES:
        return _AGE_TENS[parts[0]] + _AGE_ONES[parts[1]]
    return None


def _augment_misc_id_spans(raw_text: str, existing: list[CanonicalSpan]) -> list[ResolutionEvent]:
    events: list[ResolutionEvent] = []
    occupied = list(existing)
    for pattern in _IDENTIFIER_CONTEXT_PATTERNS:
        for match in pattern.finditer(raw_text):
            start, end = match.span(1)
            candidate_text = raw_text[start:end]
            candidate = CanonicalSpan(
                start=start,
                end=end,
                label="MISC_ID",
                text=candidate_text,
            )
            if not _candidate_is_identifier_like(candidate_text):
                continue
            if any(_overlaps(candidate, span) for span in occupied):
                continue
            occupied.append(candidate)
            events.append(
                _make_event(
                    kind="augmentation",
                    rule="misc_id_context",
                    before=None,
                    after=candidate,
                )
            )
    return events


def _augment_age_spans(raw_text: str, existing: list[CanonicalSpan]) -> list[ResolutionEvent]:
    events: list[ResolutionEvent] = []
    occupied = list(existing)
    for pattern in _AGE_CONTEXT_PATTERNS:
        for match in pattern.finditer(raw_text):
            start, end = match.span(1)
            candidate_text = raw_text[start:end]
            if _parse_age_value(candidate_text) is None:
                continue
            candidate = CanonicalSpan(
                start=start,
                end=end,
                label="AGE",
                text=candidate_text,
            )
            if any(_overlaps(candidate, span) for span in occupied):
                continue
            occupied.append(candidate)
            events.append(
                _make_event(
                    kind="augmentation",
                    rule="age_context",
                    before=None,
                    after=candidate,
                )
            )
    return events


def resolve_spans(
    raw_text: str,
    spans: list[CanonicalSpan],
    *,
    label_profile: Literal["simple", "advanced"] = "simple",
    enable_augmentation: bool = True,
) -> tuple[list[CanonicalSpan], list[ResolutionEvent]]:
    boundary_result = _resolve_boundaries(
        raw_text,
        _sorted_unique_spans(spans),
        label_profile=label_profile,
    )
    resolved = list(boundary_result.spans)
    events = list(boundary_result.events)

    if enable_augmentation:
        augmentation_events = _augment_misc_id_spans(raw_text, resolved)
        for event in augmentation_events:
            resolved.append(event.after)
        age_events = _augment_age_spans(raw_text, resolved)
        for event in age_events:
            resolved.append(event.after)
        events.extend(augmentation_events)
        events.extend(age_events)

    return _sorted_unique_spans(resolved), events


def shift_resolution_events(
    events: list[ResolutionEvent],
    offset: int,
) -> list[ResolutionEvent]:
    shifted: list[ResolutionEvent] = []
    for event in events:
        before = event.before
        shifted_before = (
            CanonicalSpan(
                start=before.start + offset,
                end=before.end + offset,
                label=before.label,
                text=before.text,
            )
            if before is not None
            else None
        )
        after = event.after
        shifted.append(
            ResolutionEvent(
                kind=event.kind,
                label=event.label,
                rule=event.rule,
                before=shifted_before,
                after=CanonicalSpan(
                    start=after.start + offset,
                    end=after.end + offset,
                    label=after.label,
                    text=after.text,
                ),
            )
        )
    return shifted


def summarize_resolution_events(events: list[ResolutionEvent]) -> dict[str, object]:
    boundary_fix_count = 0
    augmentation_count = 0
    boundary_fix_count_by_label: dict[str, int] = {}
    augmentation_count_by_label: dict[str, int] = {}
    for event in events:
        if event.kind == "boundary_resolution":
            boundary_fix_count += 1
            boundary_fix_count_by_label[event.label] = (
                boundary_fix_count_by_label.get(event.label, 0) + 1
            )
        elif event.kind == "augmentation":
            augmentation_count += 1
            augmentation_count_by_label[event.label] = (
                augmentation_count_by_label.get(event.label, 0) + 1
            )
    return {
        "boundary_fix_count": boundary_fix_count,
        "augmentation_count": augmentation_count,
        "boundary_fix_count_by_label": boundary_fix_count_by_label,
        "augmentation_count_by_label": augmentation_count_by_label,
    }
