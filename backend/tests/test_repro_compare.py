from __future__ import annotations

from types import SimpleNamespace

import pytest

from models import CanonicalDocument, CanonicalSpan, UtteranceRow
from repro_compare import (
    _load_colleague_demo_module,
    _normalize_colleague_model_name,
    build_colleague_dataset_records,
    build_methods_lab_run_body,
    compare_prediction_maps,
    extract_methods_lab_predictions,
    project_colleague_predictions_to_current_docs,
    run_colleague_demo_baseline,
)


def _make_doc(
    *,
    doc_id: str = "doc-1",
    filename: str = "doc-1.json",
    raw_text: str,
    utterances: list[UtteranceRow],
    manual_annotations: list[CanonicalSpan],
) -> CanonicalDocument:
    return CanonicalDocument(
        id=doc_id,
        filename=filename,
        format="hips_v1",
        raw_text=raw_text,
        utterances=utterances,
        pre_annotations=[],
        manual_annotations=manual_annotations,
        label_set=[],
    )


def test_build_colleague_dataset_records_localizes_manual_spans_to_turns():
    doc = _make_doc(
        raw_text="Hello Ana\nCall 555-1212",
        utterances=[
            UtteranceRow(speaker="student", text="Hello Ana", global_start=0, global_end=9),
            UtteranceRow(speaker="tutor", text="Call 555-1212", global_start=10, global_end=23),
        ],
        manual_annotations=[
            CanonicalSpan(start=6, end=9, label="NAME", text="Ana"),
            CanonicalSpan(start=15, end=23, label="PHONE", text="555-1212"),
        ],
    )

    records = build_colleague_dataset_records([doc])

    assert records == [
        {
            "session_id": "doc-1",
            "filename": "doc-1.json",
            "transcript": [
                {
                    "role": "student",
                    "content": "Hello Ana",
                    "annotations": [
                        {"start": 6, "end": 9, "pii_type": "NAME"},
                    ],
                    "_global_start": 0,
                    "_global_end": 9,
                },
                {
                    "role": "tutor",
                    "content": "Call 555-1212",
                    "annotations": [
                        {"start": 5, "end": 13, "pii_type": "PHONE"},
                    ],
                    "_global_start": 10,
                    "_global_end": 23,
                },
            ],
        }
    ]


def test_build_colleague_dataset_records_merges_turns_for_cross_turn_spans():
    doc = _make_doc(
        raw_text="Hello Ana\nCall 555-1212",
        utterances=[
            UtteranceRow(speaker="student", text="Hello Ana", global_start=0, global_end=9),
            UtteranceRow(speaker="tutor", text="Call 555-1212", global_start=10, global_end=23),
        ],
        manual_annotations=[
            CanonicalSpan(start=8, end=12, label="NAME", text="a\nCa"),
        ],
    )

    records = build_colleague_dataset_records([doc])

    assert records == [
        {
            "session_id": "doc-1",
            "filename": "doc-1.json",
            "transcript": [
                {
                    "role": "student",
                    "content": "Hello Ana\nCall 555-1212",
                    "annotations": [
                        {"start": 8, "end": 12, "pii_type": "NAME"},
                    ],
                    "_global_start": 0,
                    "_global_end": 23,
                }
            ],
        }
    ]


def test_project_colleague_predictions_to_current_docs_shifts_message_offsets():
    doc = _make_doc(
        raw_text="Hello Ana\nCall 555-1212",
        utterances=[
            UtteranceRow(speaker="student", text="Hello Ana", global_start=0, global_end=9),
            UtteranceRow(speaker="tutor", text="Call 555-1212", global_start=10, global_end=23),
        ],
        manual_annotations=[],
    )

    predicted = {
        0: [SimpleNamespace(start=6, end=9, entity_type="NAME", text="Ana")],
        1: [SimpleNamespace(start=5, end=13, entity_type="PHONE", text="555-1212")],
    }

    projected = project_colleague_predictions_to_current_docs(
        current_docs=[doc],
        predicted=predicted,
        transcript_groups=[[0, 1]],
    )

    assert projected == {
        "doc-1": [
            CanonicalSpan(start=6, end=9, label="NAME", text="Ana"),
            CanonicalSpan(start=15, end=23, label="PHONE", text="555-1212"),
        ]
    }


def test_project_colleague_predictions_to_current_docs_uses_dataset_record_bounds():
    doc = _make_doc(
        raw_text="Hello Ana\nCall 555-1212",
        utterances=[
            UtteranceRow(speaker="student", text="Hello Ana", global_start=0, global_end=9),
            UtteranceRow(speaker="tutor", text="Call 555-1212", global_start=10, global_end=23),
        ],
        manual_annotations=[
            CanonicalSpan(start=8, end=12, label="NAME", text="a\nCa"),
        ],
    )
    dataset_records = build_colleague_dataset_records([doc])
    predicted = {
        0: [SimpleNamespace(start=8, end=12, entity_type="NAME", text="a\nCa")],
    }

    projected = project_colleague_predictions_to_current_docs(
        current_docs=[doc],
        predicted=predicted,
        transcript_groups=[[0]],
        dataset_records=dataset_records,
    )

    assert projected == {
        "doc-1": [
            CanonicalSpan(start=8, end=12, label="NAME", text="a\nCa"),
        ]
    }


def test_build_methods_lab_run_body_uses_expected_bundle_methods_and_concurrency():
    compat_body = build_methods_lab_run_body(
        bundle="deidentify-v2",
        doc_ids=["doc-1", "doc-2"],
        model="openai.gpt-5.2-chat",
        api_key="k",
        api_base="https://example.test",
        name="compat",
    )
    current_body = build_methods_lab_run_body(
        bundle="v2+post-process",
        doc_ids=["doc-1", "doc-2"],
        model="openai.gpt-5.2-chat",
        api_key="k",
        api_base="https://example.test",
        name="current",
    )

    assert compat_body.concurrency == 1
    assert compat_body.runtime.method_bundle == "deidentify-v2"
    assert [item.method_id for item in compat_body.methods] == [
        "dual-v2",
        "regex+dual-v2",
        "presidio-lite+extended-v2",
    ]

    assert current_body.concurrency == 1
    assert current_body.runtime.method_bundle == "v2+post-process"
    assert [item.method_id for item in current_body.methods] == [
        "dual",
        "presidio+llm-split",
        "presidio+default",
    ]


def test_extract_methods_lab_predictions_uses_method_id_and_model():
    run = {
        "methods": [
            {"id": "compat_dual", "label": "Dual V2", "method_id": "dual-v2"},
            {"id": "current_dual", "label": "Dual", "method_id": "dual"},
        ],
        "models": [
            {"id": "model_1", "label": "GPT", "model": "openai.gpt-5.2-chat"},
        ],
        "cells": {
            "model_1__compat_dual": {
                "id": "model_1__compat_dual",
                "model_id": "model_1",
                "method_id": "compat_dual",
                "documents": {
                    "doc-1": {
                        "hypothesis_spans": [
                            {"start": 0, "end": 4, "label": "NAME", "text": "Anna"}
                        ]
                    }
                },
            }
        },
    }

    extracted = extract_methods_lab_predictions(
        run,
        method_id="dual-v2",
        model="openai.gpt-5.2-chat",
    )

    assert extracted == {
        "doc-1": [CanonicalSpan(start=0, end=4, label="NAME", text="Anna")]
    }


def test_compare_prediction_maps_reports_doc_level_mismatches():
    baseline = {
        "doc-1": [CanonicalSpan(start=0, end=4, label="NAME", text="Anna")],
        "doc-2": [CanonicalSpan(start=5, end=9, label="NAME", text="John")],
    }
    candidate = {
        "doc-1": [CanonicalSpan(start=0, end=4, label="NAME", text="Anna")],
        "doc-2": [CanonicalSpan(start=5, end=9, label="PERSON", text="John")],
    }

    summary = compare_prediction_maps(
        baseline=baseline,
        candidate=candidate,
        baseline_label="baseline",
        candidate_label="candidate",
    )

    assert summary["matching_doc_count"] == 1
    assert summary["mismatch_doc_count"] == 1
    assert summary["documents"]["doc-2"]["same_boundary_diff_label"] == 1
    assert summary["documents"]["doc-2"]["all_equal"] is False


def test_run_colleague_demo_baseline_uses_supplied_dataset_records(monkeypatch, tmp_path):
    doc = _make_doc(
        raw_text="Hello Ana",
        utterances=[
            UtteranceRow(speaker="student", text="Hello Ana", global_start=0, global_end=9),
        ],
        manual_annotations=[
            CanonicalSpan(start=6, end=9, label="NAME", text="Ana"),
        ],
    )
    supplied_records = [
        {
            "session_id": "doc-1",
            "filename": "doc-1.json",
            "transcript": [
                {
                    "role": "teacher",
                    "content": "Hello Ana",
                    "annotations": [
                        {"start": 6, "end": 9, "pii_type": "NAME"},
                    ],
                }
            ],
        }
    ]
    captured: dict[str, object] = {}

    def _transcripts_to_gold_docs(transcripts):
        captured["transcripts"] = transcripts
        return [SimpleNamespace(id=0, spans=[])], [[0]]

    class _FakeConfig:
        def __init__(self, *, model: str, api_base: str | None):
            self.model = model
            self.api_base = api_base

    fake_module = SimpleNamespace(
        LLMConfig=_FakeConfig,
        transcripts_to_gold_docs=_transcripts_to_gold_docs,
        build_pipeline=lambda experiment, config: {"experiment": experiment, "config": config},
        analyze_chunked=lambda gold_docs, pipeline, max_workers, transcript_groups: {
            0: [SimpleNamespace(start=6, end=9, entity_type="NAME", text="Ana")]
        },
        build_label_mapping=lambda pred_types, gold_types: {"NAME": "NAME"},
        evaluate_corpus=lambda predicted, gold_docs, label_mapping, substring: {
            "micro": {"f1": 1.0}
        },
    )

    monkeypatch.setattr("repro_compare._load_colleague_demo_module", lambda repo_root: fake_module)

    baseline = run_colleague_demo_baseline(
        current_docs=[doc],
        records=supplied_records,
        model="openai.gpt-5.2-chat",
        api_base="https://example.test",
        repo_root=tmp_path,
        experiments=("dual-v2",),
        max_workers=1,
        current_match_mode="exact",
    )

    assert captured["transcripts"] == [supplied_records[0]["transcript"]]
    assert baseline["experiments"]["dual-v2"]["current_repo_metrics"]["aggregate"]["micro"]["f1"] == 1.0


def test_load_colleague_demo_module_supports_src_layout(tmp_path):
    repo_root = tmp_path / "deidentify"
    app_path = repo_root / "examples" / "demo"
    src_path = repo_root / "src" / "deidentify"
    app_path.mkdir(parents=True)
    src_path.mkdir(parents=True)
    (src_path / "__init__.py").write_text("VALUE = 7\n")
    (app_path / "app.py").write_text("from deidentify import VALUE\nDEMO_VALUE = VALUE\n")

    module = _load_colleague_demo_module(repo_root)

    assert module.DEMO_VALUE == 7


def test_normalize_colleague_model_name_adds_openai_provider_prefix():
    assert _normalize_colleague_model_name("openai.gpt-5.2-chat") == "openai/openai.gpt-5.2-chat"
    assert _normalize_colleague_model_name("openai/gpt-5.2-chat") == "openai/gpt-5.2-chat"
