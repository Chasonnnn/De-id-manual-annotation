import json
import zipfile
from io import BytesIO
from types import SimpleNamespace
from typing import Literal

import pytest
from fastapi.testclient import TestClient

from server import (
    app,
    _enrich_doc,
    _load_doc,
    _methods_lab_runs,
    _prompt_lab_runs,
    _run_llm_for_document,
    _session_dir,
    _session_docs,
)
from models import CanonicalSpan, LLMConfidenceMetric


@pytest.fixture(autouse=True)
def clean_sessions(tmp_path, monkeypatch):
    """Use a temp dir for session storage during tests."""
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


def _make_hips_v1_custom(transcript: str, pii_occurrences: list[dict] | None = None):
    return json.dumps(
        {
            "transcript": transcript,
            "pii_occurrences": pii_occurrences or [],
        }
    ).encode()


def _upload(client, data=None, filename="test.json"):
    if data is None:
        data = _make_hips_v1()
    resp = client.post(
        "/api/documents/upload",
        files={"file": (filename, data, "application/json")},
    )
    return resp


def _wait_for_prompt_lab_terminal(client: TestClient, run_id: str, attempts: int = 30):
    payload = None
    for _ in range(attempts):
        resp = client.get(f"/api/prompt-lab/runs/{run_id}")
        assert resp.status_code == 200
        payload = resp.json()
        if payload["status"] in ("completed", "completed_with_errors", "failed"):
            return payload
    raise AssertionError("Prompt Lab run did not reach a terminal status in time")


def _wait_for_methods_lab_terminal(client: TestClient, run_id: str, attempts: int = 30):
    payload = None
    for _ in range(attempts):
        resp = client.get(f"/api/methods-lab/runs/{run_id}")
        assert resp.status_code == 200
        payload = resp.json()
        if payload["status"] in ("completed", "completed_with_errors", "failed"):
            return payload
    raise AssertionError("Methods Lab run did not reach a terminal status in time")


def _mock_confidence_metric(
    *,
    available: bool = True,
    provider: str = "openai",
    model: str = "openai.gpt-5.2-chat",
    reason: Literal[
        "ok", "unsupported_provider", "missing_logprobs", "empty_completion"
    ] = "ok",
    token_count: int = 4,
    mean_logprob: float | None = -0.1,
    confidence: float | None = 0.9048374180,
    perplexity: float | None = 1.105170185,
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


def test_upload_and_list(client):
    resp = _upload(client)
    assert resp.status_code == 200
    data = resp.json()
    # Upload returns a CanonicalDocument with an id
    assert "id" in data
    doc_id = data["id"]

    resp = client.get("/api/documents")
    assert resp.status_code == 200
    docs = resp.json()
    assert isinstance(docs, list)
    assert len(docs) == 1
    assert docs[0]["id"] == doc_id


def test_get_document(client):
    resp = _upload(client)
    doc_id = resp.json()["id"]

    resp = client.get(f"/api/documents/{doc_id}")
    assert resp.status_code == 200
    doc = resp.json()
    assert doc["raw_text"] == "Hello Anna, call Sue please."
    assert len(doc["pre_annotations"]) == 1
    assert doc["pre_annotations"][0]["label"] == "NAME"
    assert "NAME" in doc["label_set"]
    assert "agent_outputs" in doc
    assert doc["agent_outputs"]["rule"] == []
    assert doc["agent_outputs"]["llm"] == []
    assert "agent_run_metrics" in doc
    assert doc["agent_run_metrics"]["llm_confidence"] is None


def test_get_document_not_found(client):
    resp = client.get("/api/documents/nonexistent")
    assert resp.status_code == 404


def test_delete_document(client):
    resp = _upload(client)
    doc_id = resp.json()["id"]
    client.put(
        f"/api/documents/{doc_id}/manual-annotations",
        json=[{"start": 6, "end": 10, "label": "NAME", "text": "Anna"}],
    )

    resp = client.delete(f"/api/documents/{doc_id}")
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True

    resp = client.get(f"/api/documents/{doc_id}")
    assert resp.status_code == 404
    resp = client.get("/api/documents")
    assert resp.status_code == 200
    assert resp.json() == []


def test_delete_document_not_found(client):
    resp = client.delete("/api/documents/nonexistent")
    assert resp.status_code == 404


def test_session_export_import_roundtrip(client):
    resp = _upload(client)
    doc_id = resp.json()["id"]
    save_resp = client.put(
        f"/api/documents/{doc_id}/manual-annotations",
        json=[{"start": 6, "end": 10, "label": "NAME", "text": "Anna"}],
    )
    assert save_resp.status_code == 200
    rule_resp = client.post(f"/api/documents/{doc_id}/agent", json={"mode": "rule"})
    assert rule_resp.status_code == 200

    export_resp = client.get("/api/session/export")
    assert export_resp.status_code == 200
    bundle = export_resp.json()
    assert bundle["format"] == "annotation_tool_session"
    assert bundle["version"] == 4
    assert "project" in bundle
    assert "compatibility" in bundle
    assert len(bundle["documents"]) == 1
    assert bundle["documents"][0]["source"]["id"] == doc_id
    assert len(bundle["documents"][0]["manual_annotations"]) == 1

    delete_resp = client.delete(f"/api/documents/{doc_id}")
    assert delete_resp.status_code == 200

    import_resp = client.post(
        "/api/session/import",
        files={"file": ("session_bundle.json", json.dumps(bundle).encode(), "application/json")},
    )
    assert import_resp.status_code == 200
    imported = import_resp.json()
    assert imported["bundle_version"] == 4
    assert imported["imported_count"] == 1
    assert imported["skipped_count"] == 0

    imported_id = imported["imported_ids"][0]
    doc_resp = client.get(f"/api/documents/{imported_id}")
    assert doc_resp.status_code == 200
    imported_doc = doc_resp.json()
    assert len(imported_doc["manual_annotations"]) == 1
    assert isinstance(imported_doc["agent_outputs"]["rule"], list)


def test_session_import_deduplicates_existing_ids(client):
    first = _upload(client)
    doc_id = first.json()["id"]

    export_resp = client.get("/api/session/export")
    assert export_resp.status_code == 200
    bundle = export_resp.json()
    assert bundle["documents"][0]["source"]["id"] == doc_id

    import_resp = client.post(
        "/api/session/import",
        files={"file": ("session_bundle.json", json.dumps(bundle).encode(), "application/json")},
    )
    assert import_resp.status_code == 200
    data = import_resp.json()
    assert data["imported_count"] == 1
    assert data["imported_ids"][0] != doc_id

    docs_resp = client.get("/api/documents")
    assert docs_resp.status_code == 200
    assert len(docs_resp.json()) == 2


def test_session_export_import_preserves_agent_llm_metrics(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.2-chat"],
    )

    def fake_run_llm_with_metadata(**kwargs):
        return SimpleNamespace(
            spans=[],
            warnings=[],
            llm_confidence=_mock_confidence_metric(confidence=0.88, band="medium"),
        )

    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)

    upload_resp = _upload(client)
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]
    run_resp = client.post(
        f"/api/documents/{doc_id}/agent",
        json={
            "mode": "llm",
            "model": "openai.gpt-5.2-chat",
            "api_key": "request-key",
            "api_base": "https://proxy.example.com/v1",
        },
    )
    assert run_resp.status_code == 200

    export_resp = client.get("/api/session/export")
    assert export_resp.status_code == 200
    bundle = export_resp.json()
    assert bundle["documents"][0]["agent_metrics"]["llm_confidence"]["confidence"] == pytest.approx(
        0.88
    )

    delete_resp = client.delete(f"/api/documents/{doc_id}")
    assert delete_resp.status_code == 200

    import_resp = client.post(
        "/api/session/import",
        files={
            "file": (
                "session_bundle.json",
                json.dumps(bundle).encode(),
                "application/json",
            )
        },
    )
    assert import_resp.status_code == 200
    imported_id = import_resp.json()["imported_ids"][0]
    doc_resp = client.get(f"/api/documents/{imported_id}")
    assert doc_resp.status_code == 200
    restored = doc_resp.json()
    assert (
        restored["agent_run_metrics"]["llm_confidence"]["confidence"]
        == pytest.approx(0.88)
    )
    assert restored["agent_run_metrics"]["llm_confidence"]["band"] == "medium"


def test_session_import_invalid_payload(client):
    import_resp = client.post(
        "/api/session/import",
        files={"file": ("broken.json", b"{not-json", "application/json")},
    )
    assert import_resp.status_code == 400
    assert "valid UTF-8 JSON" in import_resp.json()["detail"]


def test_session_profile_get_and_update(client):
    get_resp = client.get("/api/session/profile")
    assert get_resp.status_code == 200
    assert get_resp.json()["project_name"] == ""

    put_resp = client.put(
        "/api/session/profile",
        json={
            "project_name": "HIPS QA",
            "author": "Chason",
        },
    )
    assert put_resp.status_code == 200
    profile = put_resp.json()
    assert profile["project_name"] == "HIPS QA"
    assert profile["author"] == "Chason"
    assert "notes" not in profile

    export_resp = client.get("/api/session/export")
    assert export_resp.status_code == 200
    assert export_resp.json()["project"]["project_name"] == "HIPS QA"
    assert "notes" not in export_resp.json()["project"]


def test_session_import_legacy_project_notes_are_ignored(client):
    payload = {
        "format": "annotation_tool_session",
        "version": 3,
        "project": {
            "project_name": "Legacy Bundle",
            "author": "Teammate",
            "notes": "Legacy notes field should be ignored",
        },
        "documents": [],
    }
    import_resp = client.post(
        "/api/session/import",
        files={"file": ("session_bundle.json", json.dumps(payload).encode(), "application/json")},
    )
    assert import_resp.status_code == 200

    profile_resp = client.get("/api/session/profile")
    assert profile_resp.status_code == 200
    profile = profile_resp.json()
    assert profile["project_name"] == "Legacy Bundle"
    assert profile["author"] == "Teammate"
    assert "notes" not in profile


def test_session_import_rejects_unknown_bundle_version(client):
    payload = {
        "format": "annotation_tool_session",
        "version": 999,
        "documents": [],
    }
    import_resp = client.post(
        "/api/session/import",
        files={"file": ("session_bundle.json", json.dumps(payload).encode(), "application/json")},
    )
    assert import_resp.status_code == 400
    assert "Unsupported bundle version" in import_resp.json()["detail"]


def test_session_import_accepts_legacy_v1_format(client):
    upload = _upload(client)
    assert upload.status_code == 200
    doc_id = upload.json()["id"]

    export_resp = client.get("/api/session/export")
    assert export_resp.status_code == 200
    bundle = export_resp.json()
    bundle["format"] = "annotation_tool_session_v1"
    bundle["version"] = 1

    client.delete(f"/api/documents/{doc_id}")

    import_resp = client.post(
        "/api/session/import",
        files={"file": ("legacy_v1.json", json.dumps(bundle).encode(), "application/json")},
    )
    assert import_resp.status_code == 200
    data = import_resp.json()
    assert data["bundle_version"] == 1
    assert data["imported_count"] == 1


def test_manual_annotations(client):
    resp = _upload(client)
    doc_id = resp.json()["id"]

    spans = [{"start": 0, "end": 5, "label": "NAME", "text": "Hello"}]
    resp = client.put(f"/api/documents/{doc_id}/manual-annotations", json=spans)
    assert resp.status_code == 200
    result = resp.json()
    assert len(result["manual_annotations"]) == 1
    assert result["manual_annotations"][0]["text"] == "Hello"

    # Verify persistence via reload
    resp = client.get(f"/api/documents/{doc_id}")
    assert len(resp.json()["manual_annotations"]) == 1


def test_metrics_boundary_mode_ignores_edge_space_and_punctuation(client):
    resp = _upload(
        client,
        data=_make_hips_v1_custom(
            "David met Ana.",
            pii_occurrences=[
                {"start": 0, "end": 5, "text": "David", "pii_type": "NAME"},
                {"start": 10, "end": 13, "text": "Ana", "pii_type": "NAME"},
            ],
        ),
    )
    doc_id = resp.json()["id"]

    manual_spans = [
        {"start": 0, "end": 6, "label": "NAME", "text": "David "},
        {"start": 10, "end": 14, "label": "NAME", "text": "Ana."},
    ]
    save_resp = client.put(f"/api/documents/{doc_id}/manual-annotations", json=manual_spans)
    assert save_resp.status_code == 200

    exact_resp = client.get(
        f"/api/documents/{doc_id}/metrics",
        params={"reference": "pre", "hypothesis": "manual", "match_mode": "exact"},
    )
    assert exact_resp.status_code == 200
    assert exact_resp.json()["micro"]["f1"] == 0.0

    boundary_resp = client.get(
        f"/api/documents/{doc_id}/metrics",
        params={"reference": "pre", "hypothesis": "manual", "match_mode": "boundary"},
    )
    assert boundary_resp.status_code == 200
    assert boundary_resp.json()["micro"]["f1"] == 1.0
    assert boundary_resp.json()["match_mode"] == "boundary"


def test_agent_rule(client):
    # Upload a doc with an email in it
    data = json.dumps(
        {
            "transcript": "Contact me at user@example.com please.",
            "pii_occurrences": [],
        }
    ).encode()
    resp = _upload(client, data)
    doc_id = resp.json()["id"]

    resp = client.post(f"/api/documents/{doc_id}/agent", json={"mode": "rule"})
    assert resp.status_code == 200
    result = resp.json()
    agent_spans = result["agent_annotations"]
    assert any(s["label"] == "EMAIL" for s in agent_spans)
    assert len(result["agent_outputs"]["rule"]) >= 1
    assert result["agent_outputs"]["llm"] == []
    assert result["agent_run_warnings"] == []
    assert result["agent_run_metrics"]["llm_confidence"] is None

    progress_resp = client.get(f"/api/documents/{doc_id}/agent/progress")
    assert progress_resp.status_code == 200
    progress = progress_resp.json()
    assert progress["status"] == "completed"
    assert progress["mode"] == "rule"
    assert progress["completed_chunks"] == 1
    assert progress["total_chunks"] == 1
    assert progress["progress"] == 1.0


def test_agent_combined_excludes_method_outputs(client):
    resp = _upload(client)
    doc_id = resp.json()["id"]

    from server import _save_sidecar

    _save_sidecar(
        doc_id,
        "agent.rule",
        [CanonicalSpan(start=0, end=5, label="NAME", text="Hello")],
    )
    _save_sidecar(
        doc_id,
        "agent.llm",
        [CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
    )
    _save_sidecar(
        doc_id,
        "agent.method.default",
        [CanonicalSpan(start=17, end=20, label="NAME", text="Sue")],
    )

    doc_resp = client.get(f"/api/documents/{doc_id}")
    assert doc_resp.status_code == 200
    payload = doc_resp.json()

    combined = payload["agent_annotations"]
    assert {(s["start"], s["end"]) for s in combined} == {(0, 5), (6, 10)}

    method_spans = payload["agent_outputs"]["methods"]["default"]
    assert {(s["start"], s["end"]) for s in method_spans} == {(17, 20)}


def test_agent_llm_response_includes_llm_confidence_metrics(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.2-chat"],
    )

    def fake_run_llm_with_metadata(**kwargs):
        return SimpleNamespace(
            spans=[],
            warnings=[],
            llm_confidence=_mock_confidence_metric(),
        )

    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)

    resp = _upload(client)
    doc_id = resp.json()["id"]
    run_resp = client.post(
        f"/api/documents/{doc_id}/agent",
        json={
            "mode": "llm",
            "model": "openai.gpt-5.2-chat",
            "api_key": "request-key",
            "api_base": "https://proxy.example.com/v1",
        },
    )
    assert run_resp.status_code == 200
    payload = run_resp.json()
    assert payload["agent_run_metrics"]["llm_confidence"]["available"] is True
    assert payload["agent_run_metrics"]["llm_confidence"]["reason"] == "ok"


def test_agent_llm_persists_label_profile(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.2-chat"],
    )

    def fake_run_llm_with_metadata(**kwargs):
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="PERSON", text="Anna")],
            warnings=[],
            llm_confidence=_mock_confidence_metric(),
        )

    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)

    resp = _upload(client)
    doc_id = resp.json()["id"]
    run_resp = client.post(
        f"/api/documents/{doc_id}/agent",
        json={
            "mode": "llm",
            "model": "openai.gpt-5.2-chat",
            "api_key": "request-key",
            "api_base": "https://proxy.example.com/v1",
            "label_profile": "advanced",
        },
    )
    assert run_resp.status_code == 200
    payload = run_resp.json()
    assert payload["agent_outputs"]["llm"][0]["label"] == "PERSON"
    assert payload["agent_run_metrics"]["label_profile"] == "advanced"

    doc_resp = client.get(f"/api/documents/{doc_id}")
    assert doc_resp.status_code == 200
    reloaded = doc_resp.json()
    assert reloaded["agent_run_metrics"]["label_profile"] == "advanced"


def test_agent_llm_normalizes_person_label_to_name(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.2-chat"],
    )

    def fake_run_llm_with_metadata(**kwargs):
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="PERSON", text="Anna")],
            warnings=[],
            llm_confidence=_mock_confidence_metric(),
        )

    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)

    resp = _upload(client)
    doc_id = resp.json()["id"]
    run_resp = client.post(
        f"/api/documents/{doc_id}/agent",
        json={
            "mode": "llm",
            "model": "openai.gpt-5.2-chat",
            "api_key": "request-key",
            "api_base": "https://proxy.example.com/v1",
        },
    )
    assert run_resp.status_code == 200
    payload = run_resp.json()
    llm_spans = payload["agent_outputs"]["llm"]
    assert len(llm_spans) == 1
    assert llm_spans[0]["label"] == "NAME"


def test_agent_llm_prefers_canonical_sidecar_over_legacy_openai(client):
    upload_resp = _upload(client)
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    client.put(
        f"/api/documents/{doc_id}/manual-annotations",
        json=[],
    )

    from server import _save_sidecar

    _save_sidecar(
        doc_id,
        "agent.llm",
        [CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
    )
    _save_sidecar(
        doc_id,
        "agent.openai",
        [CanonicalSpan(start=0, end=5, label="NAME", text="Hello")],
    )

    doc_resp = client.get(f"/api/documents/{doc_id}")
    assert doc_resp.status_code == 200
    payload = doc_resp.json()
    llm_spans = payload["agent_outputs"]["llm"]
    assert len(llm_spans) == 1
    assert llm_spans[0]["start"] == 6
    assert llm_spans[0]["text"] == "Anna"


def test_agent_llm_run_deletes_legacy_openai_sidecar(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.2-chat"],
    )

    def fake_run_llm_with_metadata(**kwargs):
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            warnings=[],
            llm_confidence=_mock_confidence_metric(),
        )

    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)

    upload_resp = _upload(client)
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    from server import _save_sidecar, _session_dir

    _save_sidecar(
        doc_id,
        "agent.openai",
        [CanonicalSpan(start=0, end=5, label="NAME", text="Hello")],
    )

    run_resp = client.post(
        f"/api/documents/{doc_id}/agent",
        json={
            "mode": "llm",
            "model": "openai.gpt-5.2-chat",
            "api_key": "request-key",
            "api_base": "https://proxy.example.com/v1",
        },
    )
    assert run_resp.status_code == 200
    payload = run_resp.json()
    llm_spans = payload["agent_outputs"]["llm"]
    assert len(llm_spans) == 1
    assert llm_spans[0]["start"] == 6
    assert llm_spans[0]["text"] == "Anna"
    assert not (_session_dir() / f"{doc_id}.agent.openai.json").exists()


def test_metrics_include_latest_llm_confidence(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.2-chat"],
    )

    def fake_run_llm_with_metadata(**kwargs):
        return SimpleNamespace(
            spans=[],
            warnings=[],
            llm_confidence=_mock_confidence_metric(confidence=0.91, band="high"),
        )

    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)

    resp = _upload(client)
    doc_id = resp.json()["id"]
    run_resp = client.post(
        f"/api/documents/{doc_id}/agent",
        json={
            "mode": "llm",
            "model": "openai.gpt-5.2-chat",
            "api_key": "request-key",
            "api_base": "https://proxy.example.com/v1",
        },
    )
    assert run_resp.status_code == 200

    metrics_resp = client.get(
        f"/api/documents/{doc_id}/metrics",
        params={"reference": "pre", "hypothesis": "agent.llm", "match_mode": "exact"},
    )
    assert metrics_resp.status_code == 200
    metrics = metrics_resp.json()
    assert metrics["llm_confidence"]["available"] is True
    assert metrics["llm_confidence"]["band"] == "high"


def test_metrics_use_selected_method_run_llm_confidence(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: [
            "google.gemini-3.1-flash-lite-preview",
            "openai.gpt-5.2-chat",
        ],
    )

    def fake_run_llm_with_metadata(**kwargs):
        model = str(kwargs.get("model"))
        if model == "google.gemini-3.1-flash-lite-preview":
            return SimpleNamespace(
                spans=[],
                warnings=[],
                llm_confidence=_mock_confidence_metric(
                    available=False,
                    provider="unknown",
                    model=model,
                    reason="unsupported_provider",
                    token_count=0,
                    mean_logprob=None,
                    confidence=None,
                    perplexity=None,
                    band="na",
                ),
            )
        return SimpleNamespace(
            spans=[],
            warnings=[],
            llm_confidence=_mock_confidence_metric(
                model=model,
                confidence=0.91,
                band="high",
            ),
        )

    def fake_run_method_with_metadata(**kwargs):
        model = str(kwargs.get("model"))
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            warnings=[],
            llm_confidence=_mock_confidence_metric(
                model=model,
                token_count=6,
                mean_logprob=-0.08,
                confidence=0.9231163464,
                perplexity=1.0832870677,
                band="high",
            ),
        )

    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)
    monkeypatch.setattr("server.run_method_with_metadata", fake_run_method_with_metadata)

    upload_resp = _upload(client)
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    gemini_run = client.post(
        f"/api/documents/{doc_id}/agent",
        json={
            "mode": "llm",
            "model": "google.gemini-3.1-flash-lite-preview",
            "api_key": "request-key",
            "api_base": "https://proxy.example.com/v1",
        },
    )
    assert gemini_run.status_code == 200
    assert gemini_run.json()["agent_run_metrics"]["llm_confidence"]["reason"] == "unsupported_provider"

    method_run = client.post(
        f"/api/documents/{doc_id}/agent",
        json={
            "mode": "method",
            "method_id": "default",
            "model": "openai.gpt-5.2-chat",
            "api_key": "request-key",
            "api_base": "https://proxy.example.com/v1",
        },
    )
    assert method_run.status_code == 200
    assert method_run.json()["agent_run_metrics"]["llm_confidence"]["reason"] == "ok"

    metrics_resp = client.get(
        f"/api/documents/{doc_id}/metrics",
        params={
            "reference": "pre",
            "hypothesis": "agent.method.default::openai.gpt-5.2-chat",
            "match_mode": "exact",
        },
    )
    assert metrics_resp.status_code == 200
    metrics = metrics_resp.json()
    assert metrics["llm_confidence"]["reason"] == "ok"
    assert metrics["llm_confidence"]["model"] == "openai.gpt-5.2-chat"
    assert metrics["llm_confidence"]["token_count"] == 6

    dashboard_resp = client.get(
        "/api/metrics/dashboard",
        params={
            "reference": "pre",
            "hypothesis": "agent.method.default::openai.gpt-5.2-chat",
            "match_mode": "exact",
        },
    )
    assert dashboard_resp.status_code == 200
    first_doc = dashboard_resp.json()["documents"][0]
    assert first_doc["llm_confidence"]["reason"] == "ok"
    assert first_doc["llm_confidence"]["model"] == "openai.gpt-5.2-chat"


def test_non_openai_llm_confidence_returns_na_without_failing_run(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["anthropic.claude-4.6-opus"],
    )

    def fake_run_llm_with_metadata(**kwargs):
        return SimpleNamespace(
            spans=[],
            warnings=[],
            llm_confidence=_mock_confidence_metric(
                available=False,
                provider="anthropic",
                model="anthropic.claude-4.6-opus",
                reason="unsupported_provider",
                token_count=0,
                mean_logprob=None,
                confidence=None,
                perplexity=None,
                band="na",
            ),
        )

    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)

    resp = _upload(client)
    doc_id = resp.json()["id"]
    run_resp = client.post(
        f"/api/documents/{doc_id}/agent",
        json={
            "mode": "llm",
            "model": "anthropic.claude-4.6-opus",
            "api_key": "request-key",
            "api_base": "https://proxy.example.com/v1",
        },
    )
    assert run_resp.status_code == 200
    payload = run_resp.json()
    assert payload["agent_run_metrics"]["llm_confidence"]["available"] is False
    assert payload["agent_run_metrics"]["llm_confidence"]["reason"] == "unsupported_provider"
    assert payload["agent_run_metrics"]["llm_confidence"]["band"] == "na"


def test_agent_unknown_mode(client):
    resp = _upload(client)
    doc_id = resp.json()["id"]

    resp = client.post(f"/api/documents/{doc_id}/agent", json={"mode": "unknown"})
    assert resp.status_code == 422


def test_agent_llm_no_key(client, monkeypatch):
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    resp = _upload(client)
    doc_id = resp.json()["id"]

    resp = client.post(f"/api/documents/{doc_id}/agent", json={"mode": "llm"})
    assert resp.status_code == 400
    assert "API key" in resp.json()["detail"]


def test_agent_llm_uses_litellm_env_key_and_base(client, monkeypatch):
    captured: dict[str, object] = {}

    def fake_run_llm_with_metadata(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            spans=[],
            warnings=[],
            llm_confidence=_mock_confidence_metric(),
        )

    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.2-chat"],
    )
    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)
    monkeypatch.setenv("LITELLM_API_KEY", "litellm-key")
    monkeypatch.setenv("LITELLM_BASE_URL", "https://proxy.example.com/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    resp = _upload(client)
    doc_id = resp.json()["id"]

    resp = client.post(
        f"/api/documents/{doc_id}/agent",
        json={"mode": "llm", "model": "openai.gpt-5.2-chat"},
    )
    assert resp.status_code == 200
    assert captured["api_key"] == "litellm-key"
    assert captured["api_base"] == "https://proxy.example.com/v1"


def test_agent_llm_request_overrides_env_key_and_base(client, monkeypatch):
    captured: dict[str, object] = {}

    def fake_run_llm_with_metadata(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            spans=[],
            warnings=[],
            llm_confidence=_mock_confidence_metric(),
        )

    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.2-chat"],
    )
    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)
    monkeypatch.setenv("LITELLM_API_KEY", "litellm-env-key")
    monkeypatch.setenv("LITELLM_BASE_URL", "https://env.example.com/v1")

    resp = _upload(client)
    doc_id = resp.json()["id"]

    resp = client.post(
        f"/api/documents/{doc_id}/agent",
        json={
            "mode": "llm",
            "model": "openai.gpt-5.2-chat",
            "api_key": "request-key",
            "api_base": "https://request.example.com/v1",
        },
    )
    assert resp.status_code == 200
    assert captured["api_key"] == "request-key"
    assert captured["api_base"] == "https://request.example.com/v1"


def test_agent_llm_chunk_mode_auto_chunks_long_text(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.2-chat"],
    )
    call_count = {"count": 0}

    def fake_run_llm_with_metadata(**kwargs):
        call_count["count"] += 1
        return SimpleNamespace(
            spans=[],
            warnings=[],
            llm_confidence=_mock_confidence_metric(confidence=0.8, band="medium"),
        )

    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)
    long_text = "Hello Anna.\n" * 1500
    resp = _upload(client, _make_hips_v1_custom(long_text))
    doc_id = resp.json()["id"]

    run_resp = client.post(
        f"/api/documents/{doc_id}/agent",
        json={
            "mode": "llm",
            "model": "openai.gpt-5.2-chat",
            "api_key": "request-key",
            "api_base": "https://proxy.example.com/v1",
            "chunk_mode": "auto",
            "chunk_size_chars": 2000,
        },
    )
    assert run_resp.status_code == 200
    payload = run_resp.json()
    assert call_count["count"] > 1
    assert any("Chunked LLM run used" in warning for warning in payload["agent_run_warnings"])


def test_agent_llm_chunk_mode_off_stays_single_pass(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.2-chat"],
    )
    call_count = {"count": 0}

    def fake_run_llm_with_metadata(**kwargs):
        call_count["count"] += 1
        return SimpleNamespace(
            spans=[],
            warnings=[],
            llm_confidence=_mock_confidence_metric(confidence=0.8, band="medium"),
        )

    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)
    long_text = "Hello Anna.\n" * 1500
    resp = _upload(client, _make_hips_v1_custom(long_text))
    doc_id = resp.json()["id"]

    run_resp = client.post(
        f"/api/documents/{doc_id}/agent",
        json={
            "mode": "llm",
            "model": "openai.gpt-5.2-chat",
            "api_key": "request-key",
            "api_base": "https://proxy.example.com/v1",
            "chunk_mode": "off",
            "chunk_size_chars": 2000,
        },
    )
    assert run_resp.status_code == 200
    payload = run_resp.json()
    assert call_count["count"] == 1
    assert all(
        "Chunked LLM run used" not in warning for warning in payload["agent_run_warnings"]
    )


def test_agent_llm_chunk_mode_force_chunks_short_text(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.2-chat"],
    )
    call_count = {"count": 0}

    def fake_run_llm_with_metadata(**kwargs):
        call_count["count"] += 1
        return SimpleNamespace(
            spans=[],
            warnings=[],
            llm_confidence=_mock_confidence_metric(confidence=0.8, band="medium"),
        )

    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)
    resp = _upload(client)
    doc_id = resp.json()["id"]

    run_resp = client.post(
        f"/api/documents/{doc_id}/agent",
        json={
            "mode": "llm",
            "model": "openai.gpt-5.2-chat",
            "api_key": "request-key",
            "api_base": "https://proxy.example.com/v1",
            "chunk_mode": "force",
            "chunk_size_chars": 2000,
        },
    )
    assert run_resp.status_code == 200
    payload = run_resp.json()
    assert call_count["count"] == 1
    assert any("mode=force" in warning for warning in payload["agent_run_warnings"])


def test_agent_llm_returns_actionable_error_details(client, monkeypatch):
    def fake_run_llm_with_metadata(**kwargs):
        raise Exception("Upstream 404 from proxy")

    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.2-chat"],
    )
    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)

    resp = _upload(client)
    doc_id = resp.json()["id"]

    resp = client.post(
        f"/api/documents/{doc_id}/agent",
        json={
            "mode": "llm",
            "model": "openai.gpt-5.2-chat",
            "api_key": "request-key",
            "api_base": "https://proxy.example.com",
        },
    )
    assert resp.status_code == 502
    detail = resp.json()["detail"]
    assert "LLM request failed" in detail
    assert "Upstream 404 from proxy" in detail
    assert "/v1" in detail


def test_agent_llm_rejects_model_not_in_gateway_catalog(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["google.gemini-3.1-pro-preview"],
    )

    resp = _upload(client)
    doc_id = resp.json()["id"]

    resp = client.post(
        f"/api/documents/{doc_id}/agent",
        json={
            "mode": "llm",
            "model": "openai.gpt-5.3-codex",
            "api_key": "request-key",
            "api_base": "https://proxy.example.com",
        },
    )
    assert resp.status_code == 400
    assert "not available for this API key" in resp.json()["detail"]


def test_agent_llm_gateway_model_requires_api_base(client):
    resp = _upload(client)
    doc_id = resp.json()["id"]

    resp = client.post(
        f"/api/documents/{doc_id}/agent",
        json={
            "mode": "llm",
            "model": "google.gemini-3.1-pro-preview",
            "api_key": "request-key",
        },
    )
    assert resp.status_code == 400
    assert "gateway model ID" in resp.json()["detail"]


def test_metrics(client):
    resp = _upload(client)
    doc_id = resp.json()["id"]

    # Run rule agent first
    client.post(f"/api/documents/{doc_id}/agent", json={"mode": "rule"})

    resp = client.get(
        f"/api/documents/{doc_id}/metrics",
        params={"reference": "pre", "hypothesis": "agent", "match_mode": "exact"},
    )
    assert resp.status_code == 200
    result = resp.json()
    assert "micro" in result
    assert "per_label" in result
    assert "cohens_kappa" in result
    assert "confusion_matrix" in result
    assert "false_positives" in result
    assert "false_negatives" in result
    assert "support" in next(iter(result["per_label"].values()))
    assert "llm_confidence" in result
    assert result["llm_confidence"] is None


def test_metrics_supports_coarse_label_projection(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.2-chat"],
    )

    def fake_run_llm_with_metadata(**kwargs):
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="PERSON", text="Anna")],
            warnings=[],
            llm_confidence=_mock_confidence_metric(),
        )

    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)

    resp = _upload(client)
    doc_id = resp.json()["id"]
    run_resp = client.post(
        f"/api/documents/{doc_id}/agent",
        json={
            "mode": "llm",
            "model": "openai.gpt-5.2-chat",
            "api_key": "request-key",
            "api_base": "https://proxy.example.com/v1",
            "label_profile": "advanced",
        },
    )
    assert run_resp.status_code == 200

    native_resp = client.get(
        f"/api/documents/{doc_id}/metrics",
        params={
            "reference": "pre",
            "hypothesis": "agent.llm",
            "match_mode": "exact",
            "label_projection": "native",
        },
    )
    assert native_resp.status_code == 200
    native = native_resp.json()
    assert native["micro"]["f1"] == pytest.approx(0.0)

    coarse_resp = client.get(
        f"/api/documents/{doc_id}/metrics",
        params={
            "reference": "pre",
            "hypothesis": "agent.llm",
            "match_mode": "exact",
            "label_projection": "coarse_simple",
        },
    )
    assert coarse_resp.status_code == 200
    coarse = coarse_resp.json()
    assert coarse["label_projection"] == "coarse_simple"
    assert coarse["micro"]["f1"] == pytest.approx(1.0)
    assert coarse["per_label"]["NAME"]["tp"] == 1


def test_metrics_unknown_source(client):
    resp = _upload(client)
    doc_id = resp.json()["id"]

    resp = client.get(
        f"/api/documents/{doc_id}/metrics",
        params={"reference": "pre", "hypothesis": "bogus"},
    )
    assert resp.status_code == 400
    assert "Unknown source" in resp.json()["detail"]


def test_metrics_dashboard(client):
    resp = _upload(client)
    doc_id = resp.json()["id"]

    # Add a second document to exercise aggregation.
    second = _upload(
        client,
        data=json.dumps(
            {
                "transcript": "Email joe@example.com then call 555-123-4567.",
                "pii_occurrences": [
                    {
                        "start": 6,
                        "end": 21,
                        "text": "joe@example.com",
                        "pii_type": "EMAIL",
                    }
                ],
            }
        ).encode(),
    )
    assert second.status_code == 200

    # Seed one hypothesis source.
    run_resp = client.post(f"/api/documents/{doc_id}/agent", json={"mode": "rule"})
    assert run_resp.status_code == 200

    resp = client.get(
        "/api/metrics/dashboard",
        params={"reference": "pre", "hypothesis": "agent.rule", "match_mode": "exact"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["reference"] == "pre"
    assert data["hypothesis"] == "agent.rule"
    assert data["match_mode"] == "exact"
    assert data["total_documents"] == 2
    assert data["documents_compared"] == 2
    assert "micro" in data and "f1" in data["micro"]
    assert "avg_document_micro" in data
    assert "avg_document_macro" in data
    assert "llm_confidence_summary" in data
    assert "band_counts" in data["llm_confidence_summary"]
    assert len(data["documents"]) == 2
    first_doc = data["documents"][0]
    assert "id" in first_doc and "filename" in first_doc
    assert "micro" in first_doc and "f1" in first_doc["micro"]
    assert "macro" in first_doc and "precision" in first_doc["macro"]
    assert "cohens_kappa" in first_doc
    assert "mean_iou" in first_doc
    assert "llm_confidence" in first_doc


def test_metrics_dashboard_aggregates_llm_confidence(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.2-chat"],
    )
    scores = iter([0.92, 0.8])

    def fake_run_llm_with_metadata(**kwargs):
        confidence = next(scores)
        band = "high" if confidence >= 0.9 else "medium"
        return SimpleNamespace(
            spans=[],
            warnings=[],
            llm_confidence=_mock_confidence_metric(
                confidence=confidence,
                band=band,
            ),
        )

    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)

    first = _upload(client)
    second = _upload(
        client,
        data=json.dumps(
            {
                "transcript": "Email joe@example.com then call 555-123-4567.",
                "pii_occurrences": [],
            }
        ).encode(),
    )
    assert first.status_code == 200
    assert second.status_code == 200
    for doc_id in (first.json()["id"], second.json()["id"]):
        run_resp = client.post(
            f"/api/documents/{doc_id}/agent",
            json={
                "mode": "llm",
                "model": "openai.gpt-5.2-chat",
                "api_key": "request-key",
                "api_base": "https://proxy.example.com/v1",
            },
        )
        assert run_resp.status_code == 200

    dashboard_resp = client.get(
        "/api/metrics/dashboard",
        params={"reference": "pre", "hypothesis": "agent.llm", "match_mode": "exact"},
    )
    assert dashboard_resp.status_code == 200
    data = dashboard_resp.json()
    summary = data["llm_confidence_summary"]
    assert summary["documents_with_confidence"] == 2
    assert summary["mean_confidence"] == pytest.approx((0.92 + 0.8) / 2)
    assert summary["band_counts"]["high"] == 1
    assert summary["band_counts"]["medium"] == 1
    assert summary["band_counts"]["na"] == 0


def test_metrics_dashboard_unknown_source(client):
    _upload(client)
    resp = client.get(
        "/api/metrics/dashboard",
        params={"reference": "pre", "hypothesis": "bogus"},
    )
    assert resp.status_code == 400
    assert "Unknown source" in resp.json()["detail"]


def test_config(client):
    resp = client.get("/api/config")
    assert resp.status_code == 200

    resp = client.put("/api/config", json={"openai_model": "gpt-4o"})
    assert resp.status_code == 200

    resp = client.get("/api/config")
    assert resp.json()["openai_model"] == "gpt-4o"


def test_upload_invalid_file(client):
    resp = client.post(
        "/api/documents/upload",
        files={"file": ("bad.json", b"not valid json", "application/json")},
    )
    assert resp.status_code == 400


def test_model_presets_endpoint(client):
    resp = client.get("/api/models/presets")
    assert resp.status_code == 200
    data = resp.json()
    assert "presets" in data
    model_ids = {p["model"] for p in data["presets"]}
    assert "openai.gpt-5.3-codex" in model_ids
    assert "gpt-5.4" in model_ids
    assert "openai.gpt-5.2-chat" in model_ids
    assert "google.gemini-3.1-flash-lite-preview" in model_ids


def test_agent_credentials_status(client, monkeypatch):
    monkeypatch.setenv("LITELLM_API_KEY", "litellm-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("LITELLM_BASE_URL", "https://proxy.example.com")

    resp = client.get("/api/agent/credentials/status")
    assert resp.status_code == 200
    data = resp.json()
    assert data["has_api_key"] is True
    assert "LITELLM_API_KEY" in data["api_key_sources"]
    assert "OPENAI_API_KEY" in data["api_key_sources"]
    assert data["has_api_base"] is True
    assert "LITELLM_BASE_URL" in data["api_base_sources"]


def test_prompt_lab_run_completes_and_falls_back_to_pre(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
    )

    def fake_run_llm_with_metadata(**kwargs):
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            warnings=[],
            llm_confidence=_mock_confidence_metric(confidence=0.91, band="high"),
        )

    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)

    upload_resp = _upload(client)
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    create_resp = client.post(
        "/api/prompt-lab/runs",
        json={
            "name": "fallback-check",
            "doc_ids": [doc_id],
            "prompts": [
                {
                    "id": "p1",
                    "label": "Baseline",
                    "system_prompt": "Detect pii spans as strict JSON",
                }
            ],
            "models": [
                {
                    "id": "m1",
                    "label": "Codex",
                    "model": "openai.gpt-5.3-codex",
                    "reasoning_effort": "xhigh",
                    "anthropic_thinking": False,
                    "anthropic_thinking_budget_tokens": None,
                }
            ],
            "runtime": {
                "api_key": "request-key",
                "api_base": "https://proxy.example.com/v1",
                "temperature": 0.0,
                "match_mode": "exact",
                "reference_source": "manual",
                "fallback_reference_source": "pre",
            },
            "concurrency": 4,
        },
    )
    assert create_resp.status_code == 200
    run_id = create_resp.json()["id"]

    final = _wait_for_prompt_lab_terminal(client, run_id)
    assert final["status"] == "completed"
    assert final["matrix"]["cells"][0]["completed_docs"] == 1
    assert final["matrix"]["cells"][0]["failed_docs"] == 0
    assert final["matrix"]["cells"][0]["micro"]["f1"] == pytest.approx(1.0)

    cell_id = final["matrix"]["cells"][0]["id"]
    detail_resp = client.get(
        f"/api/prompt-lab/runs/{run_id}/cells/{cell_id}/documents/{doc_id}"
    )
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["reference_source_used"] == "pre"
    assert len(detail["hypothesis_spans"]) == 1
    assert not (_session_dir() / f"{doc_id}.agent.llm.json").exists()


def test_prompt_lab_run_handles_mismatch_metrics_without_crashing(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
    )

    def fake_run_llm_with_metadata(**kwargs):
        return SimpleNamespace(
            spans=[],
            warnings=[],
            llm_confidence=_mock_confidence_metric(confidence=0.62, band="low"),
        )

    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)

    upload_resp = _upload(client)
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    create_resp = client.post(
        "/api/prompt-lab/runs",
        json={
            "name": "mismatch-check",
            "doc_ids": [doc_id],
            "prompts": [
                {
                    "id": "p1",
                    "label": "Baseline",
                    "system_prompt": "Detect pii spans as strict JSON",
                }
            ],
            "models": [
                {
                    "id": "m1",
                    "label": "Codex",
                    "model": "openai.gpt-5.3-codex",
                    "reasoning_effort": "xhigh",
                    "anthropic_thinking": False,
                    "anthropic_thinking_budget_tokens": None,
                }
            ],
            "runtime": {
                "api_key": "request-key",
                "api_base": "https://proxy.example.com/v1",
                "temperature": 0.0,
                "match_mode": "exact",
                "reference_source": "manual",
                "fallback_reference_source": "pre",
            },
            "concurrency": 1,
        },
    )
    assert create_resp.status_code == 200
    run_id = create_resp.json()["id"]

    final = _wait_for_prompt_lab_terminal(client, run_id)
    assert final["status"] == "completed"
    cell = final["matrix"]["cells"][0]
    assert cell["completed_docs"] == 1
    assert cell["micro"]["f1"] == pytest.approx(0.0)


def test_prompt_lab_runtime_supports_label_profile_and_coarse_projection(
    client,
    monkeypatch,
):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
    )

    def fake_run_llm_with_metadata(**kwargs):
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="PERSON", text="Anna")],
            warnings=[],
            llm_confidence=_mock_confidence_metric(confidence=0.91, band="high"),
        )

    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)

    upload_resp = _upload(client)
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    create_resp = client.post(
        "/api/prompt-lab/runs",
        json={
            "name": "advanced-profile-run",
            "doc_ids": [doc_id],
            "prompts": [
                {
                    "id": "p1",
                    "label": "Baseline",
                    "system_prompt": "Detect pii spans as strict JSON",
                }
            ],
            "models": [
                {
                    "id": "m1",
                    "label": "Codex",
                    "model": "openai.gpt-5.3-codex",
                    "reasoning_effort": "xhigh",
                    "anthropic_thinking": False,
                    "anthropic_thinking_budget_tokens": None,
                }
            ],
            "runtime": {
                "api_key": "request-key",
                "api_base": "https://proxy.example.com/v1",
                "temperature": 0.0,
                "match_mode": "exact",
                "reference_source": "manual",
                "fallback_reference_source": "pre",
                "label_profile": "advanced",
                "label_projection": "coarse_simple",
            },
            "concurrency": 1,
        },
    )
    assert create_resp.status_code == 200
    run_id = create_resp.json()["id"]

    final = _wait_for_prompt_lab_terminal(client, run_id)
    assert final["status"] == "completed"
    assert final["runtime"]["label_profile"] == "advanced"
    assert final["runtime"]["label_projection"] == "coarse_simple"
    assert final["matrix"]["cells"][0]["micro"]["f1"] == pytest.approx(1.0)


def test_prompt_lab_preset_variant_executes_method_pipeline(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
    )
    calls = {"method": 0, "llm": 0}

    def fake_run_method_with_metadata(**kwargs):
        calls["method"] += 1
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            warnings=["method-path"],
        )

    def fake_run_llm_with_metadata(**kwargs):
        calls["llm"] += 1
        return SimpleNamespace(
            spans=[],
            warnings=[],
            llm_confidence=_mock_confidence_metric(),
        )

    monkeypatch.setattr("server.run_method_with_metadata", fake_run_method_with_metadata)
    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)

    upload_resp = _upload(client)
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    create_resp = client.post(
        "/api/prompt-lab/runs",
        json={
            "name": "preset-method-run",
            "doc_ids": [doc_id],
            "prompts": [
                {
                    "id": "preset1",
                    "label": "Verified preset",
                    "variant_type": "preset",
                    "preset_method_id": "verified",
                    "method_verify_override": True,
                }
            ],
            "models": [
                {
                    "id": "m1",
                    "label": "Codex",
                    "model": "openai.gpt-5.3-codex",
                    "reasoning_effort": "xhigh",
                    "anthropic_thinking": False,
                    "anthropic_thinking_budget_tokens": None,
                }
            ],
            "runtime": {
                "api_key": "request-key",
                "api_base": "https://proxy.example.com/v1",
                "temperature": 0.0,
                "match_mode": "exact",
                "reference_source": "manual",
                "fallback_reference_source": "pre",
            },
            "concurrency": 1,
        },
    )
    assert create_resp.status_code == 200
    run_id = create_resp.json()["id"]

    final = _wait_for_prompt_lab_terminal(client, run_id)
    assert final["status"] == "completed"
    assert calls["method"] >= 1
    assert calls["llm"] == 0
    assert final["prompts"][0]["variant_type"] == "preset"
    assert final["prompts"][0]["preset_method_id"] == "verified"


def test_prompt_lab_matrix_includes_per_label_and_available_labels(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
    )

    def fake_run_llm_with_metadata(**kwargs):
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            warnings=[],
            llm_confidence=_mock_confidence_metric(confidence=0.91, band="high"),
        )

    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)

    upload_resp = _upload(client)
    doc_id = upload_resp.json()["id"]
    create_resp = client.post(
        "/api/prompt-lab/runs",
        json={
            "doc_ids": [doc_id],
            "prompts": [
                {"id": "p1", "label": "Baseline", "system_prompt": "normal run"}
            ],
            "models": [
                {
                    "id": "m1",
                    "label": "Codex",
                    "model": "openai.gpt-5.3-codex",
                    "reasoning_effort": "xhigh",
                    "anthropic_thinking": False,
                    "anthropic_thinking_budget_tokens": None,
                }
            ],
            "runtime": {
                "api_key": "request-key",
                "api_base": "https://proxy.example.com/v1",
                "temperature": 0.0,
                "match_mode": "exact",
                "reference_source": "manual",
                "fallback_reference_source": "pre",
            },
            "concurrency": 1,
        },
    )
    assert create_resp.status_code == 200
    run_id = create_resp.json()["id"]

    final = _wait_for_prompt_lab_terminal(client, run_id)
    cell = final["matrix"]["cells"][0]
    assert "per_label" in cell
    assert "NAME" in cell["per_label"]
    assert cell["per_label"]["NAME"]["tp"] == 1
    assert cell["per_label"]["NAME"]["support"] == 1
    assert "available_labels" in final["matrix"]
    assert "NAME" in final["matrix"]["available_labels"]


def test_prompt_lab_presidio_preset_fails_per_task_without_aborting_run(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
    )

    def fake_run_method_with_metadata(**kwargs):
        raise ValueError("Presidio methods require local setup.")

    monkeypatch.setattr("server.run_method_with_metadata", fake_run_method_with_metadata)

    upload_resp = _upload(client)
    doc_id = upload_resp.json()["id"]
    create_resp = client.post(
        "/api/prompt-lab/runs",
        json={
            "doc_ids": [doc_id],
            "prompts": [
                {
                    "id": "p1",
                    "label": "Presidio split",
                    "variant_type": "preset",
                    "preset_method_id": "presidio+llm-split",
                }
            ],
            "models": [
                {
                    "id": "m1",
                    "label": "Codex",
                    "model": "openai.gpt-5.3-codex",
                    "reasoning_effort": "xhigh",
                    "anthropic_thinking": False,
                    "anthropic_thinking_budget_tokens": None,
                }
            ],
            "runtime": {
                "api_key": "request-key",
                "api_base": "https://proxy.example.com/v1",
                "temperature": 0.0,
                "match_mode": "exact",
                "reference_source": "manual",
                "fallback_reference_source": "pre",
            },
            "concurrency": 1,
        },
    )
    assert create_resp.status_code == 200
    run_id = create_resp.json()["id"]
    final = _wait_for_prompt_lab_terminal(client, run_id)
    assert final["status"] == "completed_with_errors"
    cell = final["matrix"]["cells"][0]
    assert cell["failed_docs"] == 1


def test_prompt_lab_preset_variant_requires_preset_method_id(client):
    upload_resp = _upload(client)
    doc_id = upload_resp.json()["id"]
    resp = client.post(
        "/api/prompt-lab/runs",
        json={
            "doc_ids": [doc_id],
            "prompts": [
                {
                    "id": "preset1",
                    "label": "Preset missing id",
                    "variant_type": "preset",
                }
            ],
            "models": [
                {
                    "id": "m1",
                    "label": "Codex",
                    "model": "openai.gpt-5.3-codex",
                    "reasoning_effort": "xhigh",
                    "anthropic_thinking": False,
                    "anthropic_thinking_budget_tokens": None,
                }
            ],
            "runtime": {
                "api_key": "request-key",
                "api_base": "https://proxy.example.com/v1",
                "temperature": 0.0,
                "match_mode": "exact",
                "reference_source": "manual",
                "fallback_reference_source": "pre",
            },
            "concurrency": 1,
        },
    )
    assert resp.status_code == 400
    assert "preset_method_id required" in resp.json()["detail"]


def test_prompt_lab_preset_variant_allows_presidio_plus_default(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
    )

    calls = {"method": 0}

    def fake_run_method_with_metadata(**kwargs):
        calls["method"] += 1
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            warnings=[],
        )

    monkeypatch.setattr("server.run_method_with_metadata", fake_run_method_with_metadata)

    upload_resp = _upload(client)
    doc_id = upload_resp.json()["id"]
    create_resp = client.post(
        "/api/prompt-lab/runs",
        json={
            "doc_ids": [doc_id],
            "prompts": [
                {
                    "id": "p1",
                    "label": "Presidio + Default",
                    "variant_type": "preset",
                    "preset_method_id": "presidio+default",
                }
            ],
            "models": [
                {
                    "id": "m1",
                    "label": "Codex",
                    "model": "openai.gpt-5.3-codex",
                    "reasoning_effort": "xhigh",
                    "anthropic_thinking": False,
                    "anthropic_thinking_budget_tokens": None,
                }
            ],
            "runtime": {
                "api_key": "request-key",
                "api_base": "https://proxy.example.com/v1",
                "temperature": 0.0,
                "match_mode": "exact",
                "reference_source": "manual",
                "fallback_reference_source": "pre",
            },
            "concurrency": 1,
        },
    )
    assert create_resp.status_code == 200
    run_id = create_resp.json()["id"]
    final = _wait_for_prompt_lab_terminal(client, run_id)
    assert final["status"] == "completed"
    assert calls["method"] >= 1
    assert final["prompts"][0]["preset_method_id"] == "presidio+default"


def test_prompt_lab_enforces_variant_limits(client):
    upload_resp = _upload(client)
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    prompts = [
        {
            "id": f"p{idx}",
            "label": f"Prompt {idx}",
            "system_prompt": "Detect pii spans as strict JSON",
        }
        for idx in range(1, 8)
    ]

    resp = client.post(
        "/api/prompt-lab/runs",
        json={
            "doc_ids": [doc_id],
            "prompts": prompts,
            "models": [
                {
                    "id": "m1",
                    "label": "Codex",
                    "model": "openai.gpt-5.3-codex",
                    "reasoning_effort": "xhigh",
                    "anthropic_thinking": False,
                    "anthropic_thinking_budget_tokens": None,
                }
            ],
            "runtime": {
                "api_key": "request-key",
                "api_base": "https://proxy.example.com/v1",
                "temperature": 0.0,
                "match_mode": "exact",
                "reference_source": "manual",
                "fallback_reference_source": "pre",
            },
            "concurrency": 4,
        },
    )
    assert resp.status_code == 400
    assert "between 1 and 6" in resp.json()["detail"]


def test_prompt_lab_completed_with_errors_when_some_cells_fail(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
    )

    def fake_run_llm_with_metadata(**kwargs):
        prompt = kwargs.get("system_prompt", "")
        if isinstance(prompt, str) and "force-fail" in prompt:
            raise RuntimeError("Simulated run failure")
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            warnings=[],
            llm_confidence=_mock_confidence_metric(confidence=0.85, band="medium"),
        )

    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)

    upload_resp = _upload(client)
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    create_resp = client.post(
        "/api/prompt-lab/runs",
        json={
            "doc_ids": [doc_id],
            "prompts": [
                {
                    "id": "ok",
                    "label": "ok",
                    "system_prompt": "normal run",
                },
                {
                    "id": "bad",
                    "label": "bad",
                    "system_prompt": "force-fail",
                },
            ],
            "models": [
                {
                    "id": "m1",
                    "label": "Codex",
                    "model": "openai.gpt-5.3-codex",
                    "reasoning_effort": "xhigh",
                    "anthropic_thinking": False,
                    "anthropic_thinking_budget_tokens": None,
                }
            ],
            "runtime": {
                "api_key": "request-key",
                "api_base": "https://proxy.example.com/v1",
                "temperature": 0.0,
                "match_mode": "exact",
                "reference_source": "manual",
                "fallback_reference_source": "pre",
            },
            "concurrency": 2,
        },
    )
    assert create_resp.status_code == 200
    run_id = create_resp.json()["id"]

    final = _wait_for_prompt_lab_terminal(client, run_id)
    assert final["status"] == "completed_with_errors"
    error_cells = [cell for cell in final["matrix"]["cells"] if cell["failed_docs"] > 0]
    assert len(error_cells) == 1


def test_prompt_lab_export_import_remaps_doc_ids(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
    )

    def fake_run_llm_with_metadata(**kwargs):
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            warnings=[],
            llm_confidence=_mock_confidence_metric(confidence=0.93, band="high"),
        )

    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)

    upload_resp = _upload(client)
    assert upload_resp.status_code == 200
    original_doc_id = upload_resp.json()["id"]

    create_resp = client.post(
        "/api/prompt-lab/runs",
        json={
            "name": "portable-run",
            "doc_ids": [original_doc_id],
            "prompts": [{"id": "p1", "label": "Baseline", "system_prompt": "normal run"}],
            "models": [
                {
                    "id": "m1",
                    "label": "Codex",
                    "model": "openai.gpt-5.3-codex",
                    "reasoning_effort": "xhigh",
                    "anthropic_thinking": False,
                    "anthropic_thinking_budget_tokens": None,
                }
            ],
            "runtime": {
                "api_key": "request-key",
                "api_base": "https://proxy.example.com/v1",
                "temperature": 0.0,
                "match_mode": "exact",
                "reference_source": "manual",
                "fallback_reference_source": "pre",
            },
            "concurrency": 1,
        },
    )
    assert create_resp.status_code == 200
    original_run_id = create_resp.json()["id"]
    _wait_for_prompt_lab_terminal(client, original_run_id)

    export_resp = client.get("/api/session/export")
    assert export_resp.status_code == 200
    bundle = export_resp.json()
    assert len(bundle["prompt_lab_runs"]) >= 1

    import_resp = client.post(
        "/api/session/import",
        files={
            "file": (
                "session_bundle.json",
                json.dumps(bundle).encode(),
                "application/json",
            )
        },
    )
    assert import_resp.status_code == 200
    imported = import_resp.json()
    assert imported["imported_prompt_lab_runs"] >= 1
    new_doc_id = imported["imported_ids"][0]
    assert new_doc_id != original_doc_id

    runs_resp = client.get("/api/prompt-lab/runs")
    assert runs_resp.status_code == 200
    runs = runs_resp.json()["runs"]
    imported_run = next((row for row in runs if row["id"] != original_run_id), None)
    assert imported_run is not None

    detail_resp = client.get(f"/api/prompt-lab/runs/{imported_run['id']}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert new_doc_id in detail["doc_ids"]


def test_methods_lab_accepts_eight_method_variants_and_repeated_method_ids(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
    )

    def fake_run_method_with_metadata(**kwargs):
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            warnings=[],
            llm_confidence=_mock_confidence_metric(),
        )

    monkeypatch.setattr("server.run_method_with_metadata", fake_run_method_with_metadata)

    upload_resp = _upload(client)
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]
    manual_resp = client.put(
        f"/api/documents/{doc_id}/manual-annotations",
        json=[{"start": 6, "end": 10, "label": "NAME", "text": "Anna"}],
    )
    assert manual_resp.status_code == 200

    methods = [
        {"id": "m_default", "label": "Default", "method_id": "default"},
        {"id": "m_extended", "label": "Extended", "method_id": "extended"},
        {"id": "m_verified_on", "label": "Verified On", "method_id": "verified"},
        {"id": "m_verified_off", "label": "Verified Off", "method_id": "verified", "method_verify_override": False},
        {"id": "m_dual", "label": "Dual", "method_id": "dual"},
        {"id": "m_dual_split", "label": "Dual Split", "method_id": "dual-split"},
        {"id": "m_pd", "label": "Presidio Default", "method_id": "presidio+default"},
        {"id": "m_pls", "label": "Presidio Split", "method_id": "presidio+llm-split"},
    ]

    create_resp = client.post(
        "/api/methods-lab/runs",
        json={
            "name": "eight-methods",
            "doc_ids": [doc_id],
            "methods": methods,
            "models": [
                {
                    "id": "model_1",
                    "label": "Codex",
                    "model": "openai.gpt-5.3-codex",
                    "reasoning_effort": "xhigh",
                    "anthropic_thinking": False,
                    "anthropic_thinking_budget_tokens": None,
                }
            ],
            "runtime": {
                "api_key": "request-key",
                "api_base": "https://proxy.example.com/v1",
                "temperature": 0.0,
                "match_mode": "exact",
                "label_profile": "simple",
                "label_projection": "native",
                "chunk_mode": "auto",
                "chunk_size_chars": 10000,
            },
            "concurrency": 1,
        },
    )
    assert create_resp.status_code == 200
    run_id = create_resp.json()["id"]

    final = _wait_for_methods_lab_terminal(client, run_id)
    assert final["status"] == "completed"
    assert len(final["methods"]) == 8
    assert final["methods"][2]["method_id"] == "verified"
    assert final["methods"][3]["method_id"] == "verified"
    assert final["methods"][3]["method_verify_override"] is False


def test_methods_lab_rejects_duplicate_method_variant_ids_and_unknown_method(client):
    upload_resp = _upload(client)
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    duplicate_resp = client.post(
        "/api/methods-lab/runs",
        json={
            "doc_ids": [doc_id],
            "methods": [
                {"id": "dup", "label": "First", "method_id": "default"},
                {"id": "dup", "label": "Second", "method_id": "extended"},
            ],
            "models": [
                {
                    "id": "model_1",
                    "label": "Codex",
                    "model": "openai.gpt-5.3-codex",
                    "reasoning_effort": "xhigh",
                    "anthropic_thinking": False,
                    "anthropic_thinking_budget_tokens": None,
                }
            ],
            "runtime": {
                "api_key": "request-key",
                "api_base": "https://proxy.example.com/v1",
                "temperature": 0.0,
                "match_mode": "exact",
            },
            "concurrency": 1,
        },
    )
    assert duplicate_resp.status_code == 400
    assert "Duplicate method id" in duplicate_resp.json()["detail"]

    unknown_resp = client.post(
        "/api/methods-lab/runs",
        json={
            "doc_ids": [doc_id],
            "methods": [{"id": "m1", "label": "Unknown", "method_id": "does-not-exist"}],
            "models": [
                {
                    "id": "model_1",
                    "label": "Codex",
                    "model": "openai.gpt-5.3-codex",
                    "reasoning_effort": "xhigh",
                    "anthropic_thinking": False,
                    "anthropic_thinking_budget_tokens": None,
                }
            ],
            "runtime": {
                "api_key": "request-key",
                "api_base": "https://proxy.example.com/v1",
                "temperature": 0.0,
                "match_mode": "exact",
            },
            "concurrency": 1,
        },
    )
    assert unknown_resp.status_code == 400
    assert "Unknown method" in unknown_resp.json()["detail"]


def test_methods_lab_requires_api_key_only_for_llm_methods(client, monkeypatch):
    monkeypatch.setattr("server.run_method_with_metadata", lambda **kwargs: SimpleNamespace(spans=[], warnings=[]))

    upload_resp = _upload(client)
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    presidio_only_resp = client.post(
        "/api/methods-lab/runs",
        json={
            "doc_ids": [doc_id],
            "methods": [{"id": "m1", "label": "Presidio", "method_id": "presidio"}],
            "models": [
                {
                    "id": "model_1",
                    "label": "Codex",
                    "model": "openai.gpt-5.3-codex",
                    "reasoning_effort": "xhigh",
                    "anthropic_thinking": False,
                    "anthropic_thinking_budget_tokens": None,
                }
            ],
            "runtime": {"temperature": 0.0, "match_mode": "exact"},
            "concurrency": 1,
        },
    )
    assert presidio_only_resp.status_code == 200

    llm_resp = client.post(
        "/api/methods-lab/runs",
        json={
            "doc_ids": [doc_id],
            "methods": [{"id": "m1", "label": "Default", "method_id": "default"}],
            "models": [
                {
                    "id": "model_1",
                    "label": "Codex",
                    "model": "openai.gpt-5.3-codex",
                    "reasoning_effort": "xhigh",
                    "anthropic_thinking": False,
                    "anthropic_thinking_budget_tokens": None,
                }
            ],
            "runtime": {"temperature": 0.0, "match_mode": "exact"},
            "concurrency": 1,
        },
    )
    assert llm_resp.status_code == 400
    assert "API key required for Methods Lab runs" in llm_resp.json()["detail"]


def test_methods_lab_marks_docs_without_manual_annotations_unavailable(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
    )

    def fake_run_method_with_metadata(**kwargs):
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            warnings=[],
            llm_confidence=_mock_confidence_metric(),
        )

    monkeypatch.setattr("server.run_method_with_metadata", fake_run_method_with_metadata)

    with_manual = _upload(client)
    without_manual = _upload(client, filename="test-2.json")
    assert with_manual.status_code == 200
    assert without_manual.status_code == 200
    manual_doc_id = with_manual.json()["id"]
    missing_doc_id = without_manual.json()["id"]

    manual_resp = client.put(
        f"/api/documents/{manual_doc_id}/manual-annotations",
        json=[{"start": 6, "end": 10, "label": "NAME", "text": "Anna"}],
    )
    assert manual_resp.status_code == 200

    create_resp = client.post(
        "/api/methods-lab/runs",
        json={
            "doc_ids": [manual_doc_id, missing_doc_id],
            "methods": [{"id": "m1", "label": "Default", "method_id": "default"}],
            "models": [
                {
                    "id": "model_1",
                    "label": "Codex",
                    "model": "openai.gpt-5.3-codex",
                    "reasoning_effort": "xhigh",
                    "anthropic_thinking": False,
                    "anthropic_thinking_budget_tokens": None,
                }
            ],
            "runtime": {
                "api_key": "request-key",
                "api_base": "https://proxy.example.com/v1",
                "temperature": 0.0,
                "match_mode": "exact",
            },
            "concurrency": 1,
        },
    )
    assert create_resp.status_code == 200
    run_id = create_resp.json()["id"]

    final = _wait_for_methods_lab_terminal(client, run_id)
    assert final["status"] == "completed_with_errors"
    cell = final["matrix"]["cells"][0]
    assert cell["completed_docs"] == 1
    assert cell["failed_docs"] == 1

    missing_detail_resp = client.get(
        f"/api/methods-lab/runs/{run_id}/cells/model_1__m1/documents/{missing_doc_id}"
    )
    assert missing_detail_resp.status_code == 200
    missing_detail = missing_detail_resp.json()
    assert missing_detail["status"] == "unavailable"
    assert "manual annotations" in missing_detail["error"].lower()


def test_methods_lab_presidio_style_methods_execute_once_per_doc_and_reuse_across_models(
    client, monkeypatch
):
    calls = {"count": 0}

    def fake_run_method_with_metadata(**kwargs):
        calls["count"] += 1
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            warnings=[],
            llm_confidence=None,
        )

    monkeypatch.setattr("server.run_method_with_metadata", fake_run_method_with_metadata)

    upload_resp = _upload(client)
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]
    manual_resp = client.put(
        f"/api/documents/{doc_id}/manual-annotations",
        json=[{"start": 6, "end": 10, "label": "NAME", "text": "Anna"}],
    )
    assert manual_resp.status_code == 200

    create_resp = client.post(
        "/api/methods-lab/runs",
        json={
            "doc_ids": [doc_id],
            "methods": [{"id": "m1", "label": "Presidio", "method_id": "presidio"}],
            "models": [
                {
                    "id": "model_1",
                    "label": "Codex",
                    "model": "openai.gpt-5.3-codex",
                    "reasoning_effort": "xhigh",
                    "anthropic_thinking": False,
                    "anthropic_thinking_budget_tokens": None,
                },
                {
                    "id": "model_2",
                    "label": "Chat",
                    "model": "openai.gpt-5.2-chat",
                    "reasoning_effort": "xhigh",
                    "anthropic_thinking": False,
                    "anthropic_thinking_budget_tokens": None,
                },
            ],
            "runtime": {"temperature": 0.0, "match_mode": "exact"},
            "concurrency": 2,
        },
    )
    assert create_resp.status_code == 200
    run_id = create_resp.json()["id"]

    final = _wait_for_methods_lab_terminal(client, run_id)
    assert final["status"] == "completed"
    assert calls["count"] == 1

    detail_one = client.get(f"/api/methods-lab/runs/{run_id}/cells/model_1__m1/documents/{doc_id}")
    detail_two = client.get(f"/api/methods-lab/runs/{run_id}/cells/model_2__m1/documents/{doc_id}")
    assert detail_one.status_code == 200
    assert detail_two.status_code == 200
    assert detail_one.json()["hypothesis_spans"] == detail_two.json()["hypothesis_spans"]


def test_methods_lab_export_import_remaps_doc_ids(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
    )

    def fake_run_method_with_metadata(**kwargs):
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            warnings=[],
            llm_confidence=_mock_confidence_metric(confidence=0.93, band="high"),
        )

    monkeypatch.setattr("server.run_method_with_metadata", fake_run_method_with_metadata)

    upload_resp = _upload(client)
    assert upload_resp.status_code == 200
    original_doc_id = upload_resp.json()["id"]
    manual_resp = client.put(
        f"/api/documents/{original_doc_id}/manual-annotations",
        json=[{"start": 6, "end": 10, "label": "NAME", "text": "Anna"}],
    )
    assert manual_resp.status_code == 200

    create_resp = client.post(
        "/api/methods-lab/runs",
        json={
            "name": "portable-methods-run",
            "doc_ids": [original_doc_id],
            "methods": [{"id": "m1", "label": "Default", "method_id": "default"}],
            "models": [
                {
                    "id": "model_1",
                    "label": "Codex",
                    "model": "openai.gpt-5.3-codex",
                    "reasoning_effort": "xhigh",
                    "anthropic_thinking": False,
                    "anthropic_thinking_budget_tokens": None,
                }
            ],
            "runtime": {
                "api_key": "request-key",
                "api_base": "https://proxy.example.com/v1",
                "temperature": 0.0,
                "match_mode": "exact",
            },
            "concurrency": 1,
        },
    )
    assert create_resp.status_code == 200
    original_run_id = create_resp.json()["id"]
    _wait_for_methods_lab_terminal(client, original_run_id)

    export_resp = client.get("/api/session/export")
    assert export_resp.status_code == 200
    bundle = export_resp.json()
    assert len(bundle["methods_lab_runs"]) >= 1

    import_resp = client.post(
        "/api/session/import",
        files={
            "file": (
                "session_bundle.json",
                json.dumps(bundle).encode(),
                "application/json",
            )
        },
    )
    assert import_resp.status_code == 200
    imported = import_resp.json()
    assert imported["imported_methods_lab_runs"] >= 1
    new_doc_id = imported["imported_ids"][0]
    assert new_doc_id != original_doc_id

    runs_resp = client.get("/api/methods-lab/runs")
    assert runs_resp.status_code == 200
    runs = runs_resp.json()["runs"]
    imported_run = next((row for row in runs if row["id"] != original_run_id), None)
    assert imported_run is not None

    detail_resp = client.get(f"/api/methods-lab/runs/{imported_run['id']}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert new_doc_id in detail["doc_ids"]


def test_agent_llm_persists_outputs_per_model(client, monkeypatch):
    supported_models = {"openai.gpt-5.2-chat", "openai.gpt-5.3-codex"}
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: sorted(supported_models),
    )

    def fake_run_llm_with_metadata(**kwargs):
        model = str(kwargs.get("model"))
        if model == "openai.gpt-5.2-chat":
            spans = [CanonicalSpan(start=6, end=10, label="NAME", text="Anna")]
        else:
            spans = [CanonicalSpan(start=0, end=5, label="NAME", text="Hello")]
        return SimpleNamespace(
            spans=spans,
            warnings=[],
            llm_confidence=_mock_confidence_metric(model=model),
        )

    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)

    upload_resp = _upload(client)
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    run_a = client.post(
        f"/api/documents/{doc_id}/agent",
        json={
            "mode": "llm",
            "model": "openai.gpt-5.2-chat",
            "api_key": "request-key",
            "api_base": "https://proxy.example.com/v1",
        },
    )
    assert run_a.status_code == 200
    run_b = client.post(
        f"/api/documents/{doc_id}/agent",
        json={
            "mode": "llm",
            "model": "openai.gpt-5.3-codex",
            "api_key": "request-key",
            "api_base": "https://proxy.example.com/v1",
        },
    )
    assert run_b.status_code == 200

    doc_resp = client.get(f"/api/documents/{doc_id}")
    assert doc_resp.status_code == 200
    payload = doc_resp.json()
    assert payload["agent_outputs"]["llm"][0]["text"] == "Hello"
    assert set(payload["agent_outputs"]["llm_runs"].keys()) == supported_models
    assert payload["agent_outputs"]["llm_runs"]["openai.gpt-5.2-chat"][0]["text"] == "Anna"
    assert payload["agent_outputs"]["llm_runs"]["openai.gpt-5.3-codex"][0]["text"] == "Hello"
    assert payload["agent_outputs"]["llm_run_metadata"]["openai.gpt-5.2-chat"]["mode"] == "llm"
    assert (
        payload["agent_outputs"]["llm_run_metadata"]["openai.gpt-5.2-chat"]["model"]
        == "openai.gpt-5.2-chat"
    )
    llm_prompt_snapshot = payload["agent_outputs"]["llm_run_metadata"]["openai.gpt-5.2-chat"][
        "prompt_snapshot"
    ]
    assert llm_prompt_snapshot["format_guardrail_appended"] is True
    assert "requested_system_prompt" in llm_prompt_snapshot
    assert "effective_system_prompt" in llm_prompt_snapshot
    assert "updated_at" in payload["agent_outputs"]["llm_run_metadata"]["openai.gpt-5.2-chat"]

    metrics_resp = client.get(
        f"/api/documents/{doc_id}/metrics",
        params={
            "reference": "pre",
            "hypothesis": "agent.llm_run.openai.gpt-5.2-chat",
            "match_mode": "exact",
        },
    )
    assert metrics_resp.status_code == 200


def test_agent_llm_chunk_retry_persists_diagnostics(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.2-chat"],
    )

    call_counts: dict[str, int] = {}

    def fake_run_llm_with_metadata(**kwargs):
        chunk_text = str(kwargs.get("text", ""))
        count = call_counts.get(chunk_text, 0) + 1
        call_counts[chunk_text] = count
        if len(call_counts) == 1:
            spans = [CanonicalSpan(start=0, end=3, label="NAME", text=chunk_text[:3])]
        elif count == 1:
            spans = []
        else:
            spans = [CanonicalSpan(start=0, end=3, label="NAME", text=chunk_text[:3])]
        return SimpleNamespace(
            spans=spans,
            warnings=[],
            llm_confidence=_mock_confidence_metric(model="openai.gpt-5.2-chat"),
            finish_reason=None,
        )

    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)

    transcript = (("A" * 2400) + "\n") + (("B" * 2400) + "\n") + (("C" * 2400) + "\n")
    upload_resp = _upload(client, data=_make_hips_v1_custom(transcript))
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    run_resp = client.post(
        f"/api/documents/{doc_id}/agent",
        json={
            "mode": "llm",
            "model": "openai.gpt-5.2-chat",
            "api_key": "request-key",
            "api_base": "https://proxy.example.com/v1",
            "chunk_mode": "force",
            "chunk_size_chars": 2000,
        },
    )
    assert run_resp.status_code == 200
    payload = run_resp.json()
    diagnostics = payload["agent_run_metrics"]["chunk_diagnostics"]
    assert len(diagnostics) >= 2
    assert any(item["suspicious_empty"] is True for item in diagnostics)
    recovered = next(item for item in diagnostics if item["suspicious_empty"] is True)
    assert recovered["attempt_count"] == 2
    assert recovered["retry_used"] is True
    assert recovered["span_count"] > 0
    assert any("retry recovered" in warning for warning in payload["agent_run_warnings"])

    doc_resp = client.get(f"/api/documents/{doc_id}")
    assert doc_resp.status_code == 200
    reloaded = doc_resp.json()
    reloaded_diagnostics = reloaded["agent_run_metrics"]["chunk_diagnostics"]
    assert reloaded_diagnostics == diagnostics
    saved_meta = reloaded["agent_outputs"]["llm_run_metadata"]["openai.gpt-5.2-chat"]
    assert saved_meta["chunk_diagnostics"] == diagnostics


def test_run_llm_for_document_can_disable_suspicious_empty_retry(client, monkeypatch):
    call_counts: dict[str, int] = {}

    def fake_run_llm_with_metadata(**kwargs):
        chunk_text = str(kwargs.get("text", ""))
        call_counts[chunk_text] = call_counts.get(chunk_text, 0) + 1
        if chunk_text.startswith("A"):
            spans = [CanonicalSpan(start=0, end=3, label="NAME", text=chunk_text[:3])]
        else:
            spans = []
        return SimpleNamespace(
            spans=spans,
            warnings=[],
            llm_confidence=_mock_confidence_metric(model="openai.gpt-5.2-chat"),
            finish_reason=None,
        )

    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)

    transcript = (("A" * 2400) + "\n") + (("B" * 2400) + "\n") + (("C" * 2400) + "\n")
    upload_resp = _upload(client, data=_make_hips_v1_custom(transcript))
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]
    doc = _load_doc(doc_id, "default")
    assert doc is not None
    enriched = _enrich_doc(doc, "default")

    spans, warnings, _llm_confidence, diagnostics = _run_llm_for_document(
        doc=enriched,
        api_key="request-key",
        api_base="https://proxy.example.com/v1",
        model="openai.gpt-5.2-chat",
        system_prompt="Detect PII spans.",
        temperature=0.0,
        reasoning_effort="xhigh",
        anthropic_thinking=False,
        anthropic_thinking_budget_tokens=None,
        label_profile="simple",
        chunk_mode="force",
        chunk_size_chars=2000,
        enable_suspicious_empty_retry=False,
    )

    assert len(spans) > 0
    suspicious = [item for item in diagnostics if item.suspicious_empty]
    assert len(suspicious) >= 1
    assert all(item.attempt_count == 1 for item in suspicious)
    assert all(item.retry_used is False for item in suspicious)
    assert any("retry disabled" in warning for warning in warnings)
    assert max(call_counts.values()) == 1


def test_method_persists_outputs_per_method_model(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.2-chat", "openai.gpt-5.3-codex"],
    )

    def fake_run_method_with_metadata(**kwargs):
        model = str(kwargs.get("model"))
        if model == "openai.gpt-5.2-chat":
            spans = [CanonicalSpan(start=6, end=10, label="NAME", text="Anna")]
        else:
            spans = [CanonicalSpan(start=17, end=20, label="NAME", text="Sue")]
        return SimpleNamespace(spans=spans, warnings=[])

    monkeypatch.setattr("server.run_method_with_metadata", fake_run_method_with_metadata)

    upload_resp = _upload(client)
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    run_a = client.post(
        f"/api/documents/{doc_id}/agent",
        json={
            "mode": "method",
            "method_id": "default",
            "model": "openai.gpt-5.2-chat",
            "api_key": "request-key",
            "api_base": "https://proxy.example.com/v1",
        },
    )
    assert run_a.status_code == 200
    run_b = client.post(
        f"/api/documents/{doc_id}/agent",
        json={
            "mode": "method",
            "method_id": "default",
            "model": "openai.gpt-5.3-codex",
            "api_key": "request-key",
            "api_base": "https://proxy.example.com/v1",
        },
    )
    assert run_b.status_code == 200

    doc_resp = client.get(f"/api/documents/{doc_id}")
    assert doc_resp.status_code == 200
    payload = doc_resp.json()
    method_outputs = payload["agent_outputs"]["methods"]
    assert method_outputs["default"][0]["text"] == "Sue"
    assert method_outputs["default::openai.gpt-5.2-chat"][0]["text"] == "Anna"
    assert method_outputs["default::openai.gpt-5.3-codex"][0]["text"] == "Sue"
    assert payload["agent_outputs"]["method_run_metadata"]["default::openai.gpt-5.2-chat"][
        "mode"
    ] == "method"
    assert payload["agent_outputs"]["method_run_metadata"]["default::openai.gpt-5.2-chat"][
        "method_id"
    ] == "default"
    assert payload["agent_outputs"]["method_run_metadata"]["default::openai.gpt-5.2-chat"][
        "model"
    ] == "openai.gpt-5.2-chat"
    method_prompt_snapshot = payload["agent_outputs"]["method_run_metadata"][
        "default::openai.gpt-5.2-chat"
    ]["prompt_snapshot"]
    assert method_prompt_snapshot["method_id"] == "default"
    assert "passes" in method_prompt_snapshot
    assert isinstance(method_prompt_snapshot["passes"], list)
    assert len(method_prompt_snapshot["passes"]) >= 1


def test_export_ground_truth_zip_for_selected_source(client):
    upload_resp = _upload(client)
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    manual_resp = client.put(
        f"/api/documents/{doc_id}/manual-annotations",
        json=[{"start": 6, "end": 10, "label": "NAME", "text": "Anna"}],
    )
    assert manual_resp.status_code == 200

    export_resp = client.get(
        "/api/session/export-ground-truth",
        params={"source": "manual"},
    )
    assert export_resp.status_code == 200
    assert "application/zip" in export_resp.headers["content-type"]

    with zipfile.ZipFile(BytesIO(export_resp.content), mode="r") as archive:
        names = archive.namelist()
        assert len(names) == 1
        payload = json.loads(archive.read(names[0]).decode("utf-8"))
        assert payload["id"] == doc_id
        assert payload["transcript"] == "Hello Anna, call Sue please."
        assert payload["ground_truth_source"] == "manual"
        assert payload["spans"][0]["text"] == "Anna"
        assert payload["pii_occurrences"][0]["pii_type"] == "NAME"


def test_session_import_accepts_ground_truth_zip_without_existing_source(client):
    transcript = "Hello Anna, call Sue please."
    upload_resp = _upload(client, _make_hips_v1_custom(transcript))
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    manual_resp = client.put(
        f"/api/documents/{doc_id}/manual-annotations",
        json=[{"start": 6, "end": 10, "label": "NAME", "text": "Anna"}],
    )
    assert manual_resp.status_code == 200

    export_resp = client.get(
        "/api/session/export-ground-truth",
        params={"source": "manual"},
    )
    assert export_resp.status_code == 200

    delete_resp = client.delete(f"/api/documents/{doc_id}")
    assert delete_resp.status_code == 200

    import_resp = client.post(
        "/api/session/import",
        files={"file": ("ground-truth-manual.zip", export_resp.content, "application/zip")},
    )
    assert import_resp.status_code == 200
    imported = import_resp.json()
    assert imported["imported_count"] == 1
    assert imported["skipped_count"] == 0

    imported_id = imported["imported_ids"][0]
    doc_resp = client.get(f"/api/documents/{imported_id}")
    assert doc_resp.status_code == 200
    imported_doc = doc_resp.json()
    assert imported_doc["raw_text"] == transcript
    assert imported_doc["manual_annotations"] == [
        {"start": 6, "end": 10, "label": "NAME", "text": "Anna"}
    ]


def test_session_export_includes_saved_run_metadata(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.2-chat"],
    )

    def fake_run_llm_with_metadata(**kwargs):
        model = str(kwargs.get("model"))
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            warnings=[],
            llm_confidence=_mock_confidence_metric(model=model),
        )

    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)

    upload_resp = _upload(client)
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    run_resp = client.post(
        f"/api/documents/{doc_id}/agent",
        json={
            "mode": "llm",
            "model": "openai.gpt-5.2-chat",
            "api_key": "request-key",
            "api_base": "https://proxy.example.com/v1",
        },
    )
    assert run_resp.status_code == 200

    export_resp = client.get("/api/session/export")
    assert export_resp.status_code == 200
    bundle = export_resp.json()
    assert len(bundle["documents"]) == 1
    saved = bundle["documents"][0]["agent_saved_outputs"]
    assert "llm_runs" in saved
    assert "llm_run_metadata" in saved
    assert saved["llm_run_metadata"]["openai.gpt-5.2-chat"]["mode"] == "llm"
    assert "prompt_snapshot" in saved["llm_run_metadata"]["openai.gpt-5.2-chat"]
