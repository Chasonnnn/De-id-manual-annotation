import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest
from fastapi.testclient import TestClient

from models import CanonicalSpan, LLMConfidenceMetric
from server import (
    _methods_lab_runs,
    _prompt_lab_runs,
    _session_docs,
    app,
)


@pytest.fixture(autouse=True)
def clean_sessions(tmp_path, monkeypatch):
    test_sessions = tmp_path / "sessions"
    test_sessions.mkdir()
    monkeypatch.setattr("server.SESSIONS_DIR", test_sessions)
    monkeypatch.setattr("server.BASE_DIR", tmp_path)
    monkeypatch.setattr("server.CONFIG_PATH", tmp_path / "config.json")
    monkeypatch.setattr("server.PROFILE_PATH", tmp_path / "session_profile.json")
    _session_docs.clear()
    _prompt_lab_runs.clear()
    _methods_lab_runs.clear()
    yield


@pytest.fixture
def client():
    return TestClient(app)


def _make_hips_v1():
    return json.dumps(
        {
            "transcript": "Hello Anna, call Sue please.",
            "pii_occurrences": [
                {"start": 6, "end": 10, "text": "Anna", "pii_type": "NAME"},
            ],
        }
    ).encode()


def _upload(client: TestClient, data: bytes | None = None, filename: str = "test.json") -> str:
    resp = client.post(
        "/api/documents/upload",
        files={"file": (filename, data or _make_hips_v1(), "application/json")},
    )
    assert resp.status_code == 200
    return str(resp.json()["id"])


def _set_manual_annotations(client: TestClient, doc_id: str, spans: list[dict]) -> None:
    resp = client.put(f"/api/documents/{doc_id}/manual-annotations", json=spans)
    assert resp.status_code == 200


def _mock_confidence_metric(
    *,
    available: bool = True,
    provider: str = "openai",
    model: str = "openai.gpt-5.3-codex",
    reason: Literal[
        "ok",
        "unsupported_provider",
        "missing_logprobs",
        "empty_completion",
    ] = "ok",
    token_count: int = 4,
    mean_logprob: float | None = -0.1,
    confidence: float | None = 0.9,
    perplexity: float | None = 1.1,
    band: Literal["high", "medium", "low", "na"] = "high",
) -> LLMConfidenceMetric:
    return LLMConfidenceMetric(
        available=available,
        provider=provider,
        model=model,
        reason=reason,
        token_count=token_count,
        mean_logprob=mean_logprob,
        confidence=confidence,
        perplexity=perplexity,
        band=band,
        high_threshold=0.9,
        medium_threshold=0.75,
    )


def test_build_ground_truth_sweep_plan_respects_variant_limits():
    from ground_truth_sweep import build_ground_truth_sweep_plan

    plan = build_ground_truth_sweep_plan(
        session_id="default",
        doc_ids=["doc_1"],
        export_root=Path("/tmp/ground-truth-sweep"),
    )

    assert len(plan.runs) == 15

    prompt_runs = [run for run in plan.runs if run.kind == "prompt_lab"]
    methods_runs = [run for run in plan.runs if run.kind == "methods_lab"]
    assert len(prompt_runs) == 9
    assert len(methods_runs) == 6

    for run in prompt_runs:
        assert len(run.prompt_variants) <= 6
        assert len(run.model_variants) <= 6
        assert len(run.prompt_variants) * len(run.model_variants) <= 36

    for run in methods_runs:
        assert len(run.method_variants) <= 12
        assert len(run.model_variants) <= 6


def test_run_ground_truth_sweep_writes_manifests_exports_and_final_report(
    client, monkeypatch, tmp_path
):
    from ground_truth_sweep import run_ground_truth_sweep

    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: [
            "openai.gpt-5.3-codex",
            "openai.gpt-5.2-chat",
            "anthropic.claude-4.6-opus",
            "google.gemini-3.1-pro-preview",
            "google.gemini-3.1-flash-lite-preview",
        ],
    )

    def fake_run_llm_with_metadata(**kwargs):
        model = str(kwargs["model"])
        provider = "openai"
        if model.startswith("anthropic."):
            provider = "anthropic"
        elif model.startswith("google."):
            provider = "gemini"
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            warnings=[],
            llm_confidence=_mock_confidence_metric(provider=provider, model=model),
        )

    def fake_run_method_with_metadata(**kwargs):
        model = str(kwargs["model"])
        provider = "openai"
        if model.startswith("anthropic."):
            provider = "anthropic"
        elif model.startswith("google."):
            provider = "gemini"
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            warnings=[],
            llm_confidence=_mock_confidence_metric(provider=provider, model=model),
        )

    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)
    monkeypatch.setattr("server.run_method_with_metadata", fake_run_method_with_metadata)

    doc_id = _upload(client)
    _set_manual_annotations(
        client,
        doc_id,
        [{"start": 6, "end": 10, "label": "NAME", "text": "Anna"}],
    )

    export_root = tmp_path / "20260308-ground-truth-sweep"
    summary = run_ground_truth_sweep(
        session_id="default",
        export_root=export_root,
        api_key="request-key",
        api_base="https://proxy.example.com/v1",
    )

    manifests_dir = export_root / "manifests"
    run_reports_dir = export_root / "reports" / "runs"
    aggregates_dir = export_root / "reports" / "aggregates"

    assert manifests_dir.exists()
    assert len(list(manifests_dir.glob("*.yaml"))) == 15
    assert len(list(run_reports_dir.glob("*.json"))) == 15
    assert len(list(run_reports_dir.glob("*.csv"))) == 15
    assert (aggregates_dir / "prompt_lab_all_cells.csv").exists()
    assert (aggregates_dir / "methods_lab_all_cells.csv").exists()
    assert (aggregates_dir / "all_runs_summary.json").exists()
    assert (aggregates_dir / "final_report.md").exists()

    report_text = (aggregates_dir / "final_report.md").read_text()
    assert "Overall Prompt Variant Ranking" in report_text
    assert "Overall Method Variant Ranking" in report_text
    assert "Verifier Comparisons" in report_text
    assert "Top 10 Configurations" in report_text
    assert "NAME-Tolerant F1" in report_text
    assert "| Variant | NAME-Tolerant F1 | Exact F1 |" in report_text

    aggregate_summary = json.loads((aggregates_dir / "all_runs_summary.json").read_text())
    assert aggregate_summary["summary"]["run_count"] == 15
    assert aggregate_summary["summary"]["completed_run_count"] == 15

    with (aggregates_dir / "prompt_lab_all_cells.csv").open(newline="") as handle:
        prompt_rows = list(csv.DictReader(handle))
    with (aggregates_dir / "methods_lab_all_cells.csv").open(newline="") as handle:
        method_rows = list(csv.DictReader(handle))

    assert prompt_rows
    assert method_rows
    assert all(row["total_docs"] == "1" for row in prompt_rows)
    assert all(row["total_docs"] == "1" for row in method_rows)
    assert "pending_docs" in prompt_rows[0]
    assert "exact_name_affix_tolerant_f1" in prompt_rows[0]
    assert "error_family_empty_output_finish_reason_length" in prompt_rows[0]
    assert "pending_docs" in method_rows[0]
    assert "exact_name_affix_tolerant_f1" in method_rows[0]
    assert "error_family_empty_output_finish_reason_length" in method_rows[0]
    assert "exact_name_affix_tolerant_f1" in aggregate_summary["prompt_records"][0]


def test_aggregate_variant_rankings_sort_by_name_tolerant_f1_first():
    from ground_truth_sweep import _aggregate_variant_rankings

    rows = _aggregate_variant_rankings(
        [
            {
                "variant_label": "baseline_raw",
                "status": "completed",
                "precision": 0.8,
                "recall": 0.8,
                "f1": 0.8,
                "tp": 8,
                "fp": 2,
                "fn": 2,
                "exact_name_affix_tolerant_tp": 8,
                "exact_name_affix_tolerant_fp": 2,
                "exact_name_affix_tolerant_fn": 2,
                "exact_name_affix_tolerant_precision": 0.8,
                "exact_name_affix_tolerant_recall": 0.8,
                "exact_name_affix_tolerant_f1": 0.8,
            },
            {
                "variant_label": "annotator_agents_raw",
                "status": "completed",
                "precision": 0.7,
                "recall": 0.7,
                "f1": 0.7,
                "tp": 7,
                "fp": 3,
                "fn": 3,
                "exact_name_affix_tolerant_tp": 9,
                "exact_name_affix_tolerant_fp": 1,
                "exact_name_affix_tolerant_fn": 1,
                "exact_name_affix_tolerant_precision": 0.95,
                "exact_name_affix_tolerant_recall": 0.85,
                "exact_name_affix_tolerant_f1": 0.9,
            },
        ]
    )

    assert rows[0]["variant_label"] == "annotator_agents_raw"
    assert rows[0]["exact_name_affix_tolerant_f1"] == pytest.approx(0.9)
    assert rows[0]["f1"] == pytest.approx(0.7)


def test_run_ground_truth_sweep_records_blocked_runs(client, monkeypatch, tmp_path):
    from ground_truth_sweep import run_ground_truth_sweep

    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex", "openai.gpt-5.2-chat"],
    )

    def fake_run_llm_with_metadata(**kwargs):
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            warnings=[],
            llm_confidence=_mock_confidence_metric(model=str(kwargs["model"])),
        )

    def fake_run_method_with_metadata(**kwargs):
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            warnings=[],
            llm_confidence=_mock_confidence_metric(model=str(kwargs["model"])),
        )

    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)
    monkeypatch.setattr("server.run_method_with_metadata", fake_run_method_with_metadata)

    doc_id = _upload(client)
    _set_manual_annotations(
        client,
        doc_id,
        [{"start": 6, "end": 10, "label": "NAME", "text": "Anna"}],
    )

    export_root = tmp_path / "20260308-ground-truth-sweep"
    summary = run_ground_truth_sweep(
        session_id="default",
        export_root=export_root,
        api_key="request-key",
        api_base="https://proxy.example.com/v1",
    )

    assert summary["summary"]["blocked_run_count"] > 0
    report_text = (export_root / "reports" / "aggregates" / "final_report.md").read_text()
    assert "Blocked Or Unavailable Models" in report_text
