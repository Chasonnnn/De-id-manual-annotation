from __future__ import annotations

from models import CanonicalSpan
from span_resolution import (
    RESOLUTION_POLICY_VERSION,
    resolve_spans,
    summarize_resolution_events,
)


def test_name_boundary_resolution_expands_supported_affixes():
    text = "Hello Mr. Muhammad, Ana. and Sebastian's notebook."
    spans = [
        CanonicalSpan(start=text.index("Muhammad"), end=text.index("Muhammad") + 8, label="NAME", text="Muhammad"),
        CanonicalSpan(start=text.index("Ana"), end=text.index("Ana") + 3, label="NAME", text="Ana"),
        CanonicalSpan(start=text.index("Sebastian"), end=text.index("Sebastian") + 9, label="NAME", text="Sebastian"),
    ]

    resolved, events = resolve_spans(text, spans, label_profile="simple", enable_augmentation=False)

    assert [(span.start, span.end, span.text) for span in resolved] == [
        (text.index("Muhammad"), text.index("Muhammad") + 8, "Muhammad"),
        (text.index("Ana"), text.index("Ana") + 3, "Ana"),
        (text.index("Sebastian"), text.index("Sebastian") + 9, "Sebastian"),
    ]
    assert events == []


def test_name_boundary_resolution_trims_whitespace_but_does_not_expand_lexical_tokens():
    text = "Javier  met Michael Myers."
    spans = [
        CanonicalSpan(start=0, end=7, label="NAME", text="Javier "),
        CanonicalSpan(start=text.index("Michael"), end=text.index("Michael") + 7, label="NAME", text="Michael"),
    ]

    resolved, events = resolve_spans(text, spans, label_profile="simple", enable_augmentation=False)

    assert [(span.start, span.end, span.text) for span in resolved] == [
        (0, 7, "Javier "),
        (text.index("Michael"), text.index("Michael") + 7, "Michael"),
    ]
    assert events == []


def test_url_boundary_resolution_joins_multiline_domains():
    text = "Use meet.\ngoogle.\ncom and kahoot.\nit today."
    spans = [
        CanonicalSpan(
            start=text.index("google"),
            end=text.index("google") + len("google"),
            label="URL",
            text="google",
        ),
        CanonicalSpan(
            start=text.index("kahoot"),
            end=text.index("kahoot") + len("kahoot"),
            label="URL",
            text="kahoot",
        ),
    ]

    resolved, events = resolve_spans(text, spans, label_profile="simple", enable_augmentation=False)

    assert [(span.start, span.end, span.text) for span in resolved] == [
        (text.index("meet"), text.index("com") + 3, "meet.\ngoogle.\ncom"),
        (text.index("kahoot"), text.index("it") + 2, "kahoot.\nit"),
    ]
    assert len(events) == 2
    assert {event.rule for event in events} == {"url_multiline_domain"}


def test_misc_id_augmentation_requires_identifier_context():
    positive = "Your room code ABC opens the session."
    negative = "The ABC theorem came up in class."

    positive_resolved, positive_events = resolve_spans(
        positive, [], label_profile="simple", enable_augmentation=True
    )
    negative_resolved, negative_events = resolve_spans(
        negative, [], label_profile="simple", enable_augmentation=True
    )

    assert [(span.label, span.text) for span in positive_resolved] == [("MISC_ID", "ABC")]
    assert positive_events[0].kind == "augmentation"
    assert positive_events[0].rule == "misc_id_context"
    assert negative_resolved == []
    assert negative_events == []


def test_age_augmentation_recovers_explicit_age_contexts_and_rejects_counts():
    positive = "My brother is twelve years old and my sister aged 13."
    negative = "The class has twelve students and grade 13 notes."

    positive_resolved, positive_events = resolve_spans(
        positive, [], label_profile="simple", enable_augmentation=True
    )
    negative_resolved, negative_events = resolve_spans(
        negative, [], label_profile="simple", enable_augmentation=True
    )

    assert {(span.label, span.text) for span in positive_resolved} == {
        ("AGE", "twelve"),
        ("AGE", "13"),
    }
    assert all(event.kind == "augmentation" for event in positive_events)
    assert negative_resolved == []
    assert negative_events == []


def test_resolution_event_summary_counts_boundary_and_augmentation_by_label():
    text = "Use meet.\ngoogle.\ncom and password ABC"
    spans = [
        CanonicalSpan(
            start=text.index("google"),
            end=text.index("google") + len("google"),
            label="URL",
            text="google",
        )
    ]

    resolved, events = resolve_spans(text, spans, label_profile="simple", enable_augmentation=True)
    summary = summarize_resolution_events(events)

    assert RESOLUTION_POLICY_VERSION
    assert [span.text for span in resolved] == ["meet.\ngoogle.\ncom", "ABC"]
    assert summary["boundary_fix_count"] == 1
    assert summary["augmentation_count"] == 1
    assert summary["boundary_fix_count_by_label"] == {"URL": 1}
    assert summary["augmentation_count_by_label"] == {"MISC_ID": 1}
