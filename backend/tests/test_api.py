import json
from types import SimpleNamespace
from typing import Literal

import pytest
from fastapi.testclient import TestClient

from server import app, _session_docs
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


def _upload(client, data=None, filename="test.json"):
    if data is None:
        data = _make_hips_v1()
    resp = client.post(
        "/api/documents/upload",
        files={"file": (filename, data, "application/json")},
    )
    return resp


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
    assert bundle["version"] == 2
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
    assert imported["bundle_version"] == 2
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
            "notes": "Compare pre-annotations vs manual and llm outputs",
        },
    )
    assert put_resp.status_code == 200
    profile = put_resp.json()
    assert profile["project_name"] == "HIPS QA"
    assert profile["author"] == "Chason"

    export_resp = client.get("/api/session/export")
    assert export_resp.status_code == 200
    assert export_resp.json()["project"]["project_name"] == "HIPS QA"


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
