import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import app, SESSIONS_DIR, _session_docs


@pytest.fixture(autouse=True)
def clean_sessions(tmp_path, monkeypatch):
    """Use a temp dir for session storage during tests."""
    test_sessions = tmp_path / "sessions"
    test_sessions.mkdir()
    monkeypatch.setattr("server.SESSIONS_DIR", test_sessions)
    monkeypatch.setattr("server.BASE_DIR", tmp_path)
    monkeypatch.setattr("server.CONFIG_PATH", tmp_path / "config.json")
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


def test_get_document_not_found(client):
    resp = client.get("/api/documents/nonexistent")
    assert resp.status_code == 404


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


def test_agent_unknown_mode(client):
    resp = _upload(client)
    doc_id = resp.json()["id"]

    resp = client.post(f"/api/documents/{doc_id}/agent", json={"mode": "unknown"})
    assert resp.status_code == 422


def test_agent_llm_no_key(client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    resp = _upload(client)
    doc_id = resp.json()["id"]

    resp = client.post(f"/api/documents/{doc_id}/agent", json={"mode": "llm"})
    assert resp.status_code == 400
    assert "API key" in resp.json()["detail"]


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


def test_metrics_unknown_source(client):
    resp = _upload(client)
    doc_id = resp.json()["id"]

    resp = client.get(
        f"/api/documents/{doc_id}/metrics",
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
    assert "openai/gpt-5.2-codex" in model_ids
