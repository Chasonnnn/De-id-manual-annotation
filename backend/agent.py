from __future__ import annotations

import json
import re

from openai import OpenAI

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


def run_openai(text: str, api_key: str, model: str = "gpt-4o-mini") -> list[CanonicalSpan]:
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        temperature=0,
    )
    content = resp.choices[0].message.content or "[]"
    # Strip markdown code fences if present
    content = content.strip()
    if content.startswith("```"):
        lines = content.splitlines()
        # Remove first and last lines (fences)
        lines = [l for l in lines if not l.strip().startswith("```")]
        content = "\n".join(lines)
    raw = json.loads(content)
    return [
        CanonicalSpan(start=s["start"], end=s["end"], label=s["label"], text=s["text"])
        for s in raw
    ]
