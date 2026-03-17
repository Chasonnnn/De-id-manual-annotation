import json
import os
import threading
import time
import zipfile
from io import BytesIO
from types import SimpleNamespace
from typing import Literal

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from server import (
    app,
    _enrich_doc,
    _load_repo_env_file,
    _resolve_env_api_base,
    _resolve_env_api_key,
    _load_doc,
    _run_method_for_document,
    _methods_lab_cancel_events,
    _methods_lab_runs,
    _prompt_lab_runs,
    _prompt_lab_cancel_events,
    _prompt_lab_run_path,
    _prepare_experiment_scoring_spans,
    _run_llm_for_document,
    _session_dir,
    _session_docs,
    _methods_lab_run_path,
    ROOT_ENV_PATH,
)
from models import (
    CanonicalDocument,
    CanonicalSpan,
    FolderRecord,
    LLMConfidenceMetric,
    ResolutionEvent,
    UtteranceRow,
)


@pytest.fixture(autouse=True)
def clean_sessions(tmp_path, monkeypatch):
    """Use a temp dir for session storage during tests."""
    test_sessions = tmp_path / "sessions"
    test_sessions.mkdir()
    monkeypatch.setattr("server.SESSIONS_DIR", test_sessions)
    monkeypatch.setattr("server.BASE_DIR", tmp_path)
    monkeypatch.setattr("server.CONFIG_PATH", tmp_path / "config.json")
    _session_docs.clear()
    _prompt_lab_runs.clear()
    _prompt_lab_cancel_events.clear()
    _methods_lab_runs.clear()
    _methods_lab_cancel_events.clear()
    yield


@pytest.fixture
def client():
    return TestClient(app)


def test_list_prompt_lab_runs_reloads_index_from_disk_when_cache_is_stale(client):
    run_id = "prompt-cli-run"
    _prompt_lab_runs["default"] = ["stale-prompt-run"]

    run = {
        "id": run_id,
        "name": "prompt cli run",
        "status": "running",
        "created_at": "2026-03-10T00:00:00+00:00",
        "started_at": "2026-03-10T00:00:01+00:00",
        "finished_at": None,
        "doc_ids": ["doc-1"],
        "folder_ids": [],
        "prompts": [{"id": "prompt_1", "label": "Baseline"}],
        "models": [
            {
                "id": "model_1",
                "label": "Codex",
                "model": "openai.gpt-5.3-codex",
            }
        ],
        "runtime": {},
        "concurrency": 1,
        "cells": {
            "model_1__prompt_1": {
                "id": "model_1__prompt_1",
                "model_id": "model_1",
                "model_label": "Codex",
                "prompt_id": "prompt_1",
                "prompt_label": "Baseline",
                "documents": {},
            }
        },
    }
    path = _prompt_lab_run_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run))
    (path.parent / "_index.json").write_text(json.dumps([run_id]))

    resp = client.get("/api/prompt-lab/runs")

    assert resp.status_code == 200
    assert [item["id"] for item in resp.json()["runs"]] == [run_id]


def test_list_methods_lab_runs_reloads_index_from_disk_when_cache_is_stale(client):
    run_id = "methods-cli-run"
    _methods_lab_runs["default"] = ["stale-methods-run"]

    run = {
        "id": run_id,
        "name": "methods cli run",
        "status": "running",
        "created_at": "2026-03-10T00:00:00+00:00",
        "started_at": "2026-03-10T00:00:01+00:00",
        "finished_at": None,
        "doc_ids": ["doc-1"],
        "folder_ids": [],
        "methods": [{"id": "method_1", "label": "Default", "method_id": "default"}],
        "models": [
            {
                "id": "model_1",
                "label": "Codex",
                "model": "openai.gpt-5.3-codex",
            }
        ],
        "runtime": {},
        "concurrency": 1,
        "cells": {
            "model_1__method_1": {
                "id": "model_1__method_1",
                "model_id": "model_1",
                "model_label": "Codex",
                "method_id": "method_1",
                "method_label": "Default",
                "documents": {},
            }
        },
    }
    path = _methods_lab_run_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run))
    (path.parent / "_index.json").write_text(json.dumps([run_id]))

    resp = client.get("/api/methods-lab/runs")

    assert resp.status_code == 200
    assert [item["id"] for item in resp.json()["runs"]] == [run_id]


@pytest.mark.parametrize(
    ("bundle_id", "expected_bundle"),
    [
        ("v2", "v2"),
        ("stable", "stable"),
    ],
)
def test_list_methods_lab_runs_accepts_historical_method_bundle_ids(
    client, bundle_id, expected_bundle
):
    run_id = f"methods-{bundle_id}-run"
    run = {
        "id": run_id,
        "name": f"methods {bundle_id} run",
        "status": "completed",
        "created_at": "2026-03-10T00:00:00+00:00",
        "started_at": "2026-03-10T00:00:01+00:00",
        "finished_at": "2026-03-10T00:00:02+00:00",
        "doc_ids": ["doc-1"],
        "folder_ids": [],
        "methods": [{"id": "method_1", "label": "Default", "method_id": "default"}],
        "models": [
            {
                "id": "model_1",
                "label": "Codex",
                "model": "openai.gpt-5.3-codex",
            }
        ],
        "runtime": {"method_bundle": bundle_id},
        "concurrency": 1,
        "cells": {
            "model_1__method_1": {
                "id": "model_1__method_1",
                "model_id": "model_1",
                "model_label": "Codex",
                "method_id": "method_1",
                "method_label": "Default",
                "documents": {},
            }
        },
    }
    path = _methods_lab_run_path(run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(run))
    (path.parent / "_index.json").write_text(json.dumps([run_id]))

    resp = client.get("/api/methods-lab/runs")

    assert resp.status_code == 200
    listed = resp.json()["runs"][0]
    assert listed["id"] == run_id
    assert listed["method_bundle"] == expected_bundle


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


def _make_multi_record_jsonl() -> bytes:
    records = [
        {
            "transcript": [
                {
                    "role": "volunteer",
                    "content": "Hello Liam",
                    "sequence_id": 0,
                    "session_id": "session-001",
                    "annotations": [
                        {"start": 6, "end": 10, "pii_type": "NAME", "text": "Liam"}
                    ],
                }
            ]
        },
        {
            "transcript": [
                {
                    "role": "student",
                    "content": "Meet Ava",
                    "sequence_id": 0,
                    "session_id": "session-002",
                    "annotations": [
                        {"start": 5, "end": 8, "pii_type": "NAME", "text": "Ava"}
                    ],
                }
            ]
        },
    ]
    return "\n".join(json.dumps(record) for record in records).encode()


def _make_mixed_pre_multi_record_jsonl() -> bytes:
    records = [
        {
            "transcript": [
                {
                    "role": "volunteer",
                    "content": "Hello Liam",
                    "sequence_id": 0,
                    "session_id": "session-001",
                    "annotations": [
                        {"start": 6, "end": 10, "pii_type": "NAME", "text": "Liam"}
                    ],
                }
            ]
        },
        {
            "transcript": [
                {
                    "role": "student",
                    "content": "Hello Nora",
                    "sequence_id": 0,
                    "session_id": "session-002",
                    "annotations": [],
                }
            ]
        },
    ]
    return "\n".join(json.dumps(record) for record in records).encode()


def _make_prunable_multi_record_jsonl() -> bytes:
    records = [
        {
            "transcript": [
                {
                    "role": "volunteer",
                    "content": "Hello Liam",
                    "sequence_id": 0,
                    "session_id": "session-001",
                    "annotations": [
                        {"start": 6, "end": 10, "pii_type": "NAME", "text": "Liam"}
                    ],
                }
            ]
        },
        {
            "transcript": [
                {
                    "role": "student",
                    "content": "Hello Nora",
                    "sequence_id": 0,
                    "session_id": "session-002",
                    "annotations": [],
                }
            ]
        },
        {
            "transcript": [
                {
                    "role": "student",
                    "content": "Hello Omar",
                    "sequence_id": 0,
                    "session_id": "session-003",
                    "annotations": [],
                }
            ]
        },
    ]
    return "\n".join(json.dumps(record) for record in records).encode()


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
    for attempt in range(attempts):
        resp = client.get(f"/api/prompt-lab/runs/{run_id}")
        assert resp.status_code == 200
        payload = resp.json()
        if payload["status"] in ("completed", "completed_with_errors", "failed", "cancelled"):
            return payload
        if attempt < attempts - 1:
            time.sleep(0.01)
    raise AssertionError("Prompt Lab run did not reach a terminal status in time")


def _wait_for_methods_lab_terminal(client: TestClient, run_id: str, attempts: int = 30):
    payload = None
    for attempt in range(attempts):
        resp = client.get(f"/api/methods-lab/runs/{run_id}")
        assert resp.status_code == 200
        payload = resp.json()
        if payload["status"] in ("completed", "completed_with_errors", "failed", "cancelled"):
            return payload
        if attempt < attempts - 1:
            time.sleep(0.01)
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


def _write_test_config(payload: dict) -> None:
    import server

    server.CONFIG_PATH.write_text(json.dumps(payload))


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


def test_upload_multirecord_jsonl_creates_import_folder_and_hidden_child_docs(client):
    resp = _upload(
        client,
        data=_make_multi_record_jsonl(),
        filename="sessions.jsonl",
    )
    assert resp.status_code == 200
    merged = resp.json()

    docs_resp = client.get("/api/documents")
    assert docs_resp.status_code == 200
    docs = docs_resp.json()
    assert [item["id"] for item in docs] == [merged["id"]]

    folders_resp = client.get("/api/folders")
    assert folders_resp.status_code == 200
    folders = folders_resp.json()
    assert len(folders) == 1
    folder = folders[0]
    assert folder["name"] == "sessions"
    assert folder["kind"] == "import"
    assert folder["merged_doc_id"] == merged["id"]
    assert folder["doc_count"] == 2
    assert folder["child_folder_count"] == 0

    detail_resp = client.get(f"/api/folders/{folder['id']}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert [item["display_name"] for item in detail["documents"]] == [
        "Session session-001",
        "Session session-002",
    ]
    assert [item["filename"] for item in detail["documents"]] == [
        "sessions.record-0001.json",
        "sessions.record-0002.json",
    ]
    assert detail["child_folder_ids"] == []
    assert len(detail["doc_ids"]) == 2

    hidden_doc_id = detail["documents"][0]["id"]
    hidden_doc_resp = client.get(f"/api/documents/{hidden_doc_id}")
    assert hidden_doc_resp.status_code == 200
    assert hidden_doc_resp.json()["raw_text"] == "volunteer: Hello Liam"


def test_sample_folder_creation_and_import_folder_deletion(client):
    upload_resp = _upload(
        client,
        data=_make_multi_record_jsonl(),
        filename="sessions.jsonl",
    )
    assert upload_resp.status_code == 200
    merged_doc_id = upload_resp.json()["id"]

    folder_id = client.get("/api/folders").json()[0]["id"]
    import_detail = client.get(f"/api/folders/{folder_id}").json()
    hidden_doc_id = import_detail["documents"][0]["id"]

    sample_resp = client.post(
        f"/api/folders/{folder_id}/samples",
        json={"count": 1},
    )
    assert sample_resp.status_code == 200
    sample = sample_resp.json()
    assert sample["kind"] == "sample"
    assert sample["parent_folder_id"] is None
    assert sample["source_folder_id"] == folder_id
    assert sample["sample_size"] == 1
    assert sample["sample_seed"] is not None
    assert len(sample["doc_ids"]) == 1

    folders_after_sample = client.get("/api/folders").json()
    assert len(folders_after_sample) == 2
    assert {item["id"] for item in folders_after_sample} == {folder_id, sample["id"]}

    import_detail_after_sample = client.get(f"/api/folders/{folder_id}").json()
    assert [item["id"] for item in import_detail_after_sample["documents"]] == [
        item["id"] for item in import_detail["documents"]
    ]
    assert import_detail_after_sample["doc_ids"] == import_detail["doc_ids"]
    assert len(import_detail_after_sample["doc_ids"]) == 2

    too_large_resp = client.post(
        f"/api/folders/{folder_id}/samples",
        json={"count": 3},
    )
    assert too_large_resp.status_code == 400

    hidden_delete_resp = client.delete(f"/api/documents/{hidden_doc_id}")
    assert hidden_delete_resp.status_code == 409

    delete_sample_resp = client.delete(f"/api/folders/{sample['id']}")
    assert delete_sample_resp.status_code == 200
    assert client.get(f"/api/documents/{hidden_doc_id}").status_code == 200

    sample_resp = client.post(
        f"/api/folders/{folder_id}/samples",
        json={"count": 1},
    )
    assert sample_resp.status_code == 200
    sample = sample_resp.json()

    delete_import_resp = client.delete(f"/api/folders/{folder_id}")
    assert delete_import_resp.status_code == 200
    assert client.get(f"/api/documents/{hidden_doc_id}").status_code == 404
    assert client.get(f"/api/folders/{sample['id']}").status_code == 404
    assert client.get(f"/api/documents/{merged_doc_id}").status_code == 200
    assert client.get("/api/folders").json() == []
    assert [item["id"] for item in client.get("/api/documents").json()] == [merged_doc_id]


def test_prune_folder_removes_docs_without_pre_or_manual_annotations_and_updates_samples(client):
    upload_resp = _upload(
        client,
        data=_make_prunable_multi_record_jsonl(),
        filename="sessions.jsonl",
    )
    assert upload_resp.status_code == 200

    folder_id = client.get("/api/folders").json()[0]["id"]
    import_detail = client.get(f"/api/folders/{folder_id}").json()
    kept_pre_doc_id, kept_manual_doc_id, removed_doc_id = [
        item["id"] for item in import_detail["documents"]
    ]

    save_manual_resp = client.put(
        f"/api/documents/{kept_manual_doc_id}/manual-annotations",
        json=[{"start": 6, "end": 10, "label": "NAME", "text": "Nora"}],
    )
    assert save_manual_resp.status_code == 200

    sample_resp = client.post(
        f"/api/folders/{folder_id}/samples",
        json={"count": 3},
    )
    assert sample_resp.status_code == 200
    sample_id = sample_resp.json()["id"]

    prune_resp = client.post(f"/api/folders/{folder_id}/prune-empty-docs")
    assert prune_resp.status_code == 200
    assert prune_resp.json() == {
        "folder_id": folder_id,
        "removed_count": 1,
        "removed_doc_ids": [removed_doc_id],
        "updated_folder_ids": [folder_id, sample_id],
    }

    folders = {item["id"]: item for item in client.get("/api/folders").json()}
    assert folders[folder_id]["doc_count"] == 2
    assert folders[sample_id]["doc_count"] == 2

    refreshed_import = client.get(f"/api/folders/{folder_id}").json()
    assert refreshed_import["doc_ids"] == [kept_pre_doc_id, kept_manual_doc_id]

    refreshed_sample = client.get(f"/api/folders/{sample_id}").json()
    assert set(refreshed_sample["doc_ids"]) == {kept_pre_doc_id, kept_manual_doc_id}

    assert client.get(f"/api/documents/{removed_doc_id}").status_code == 404
    assert client.get(f"/api/documents/{kept_pre_doc_id}").status_code == 200
    assert client.get(f"/api/documents/{kept_manual_doc_id}").status_code == 200


def test_create_manual_folder_and_nested_subfolder(client):
    create_root_resp = client.post(
        "/api/folders",
        json={"name": "Round 1"},
    )
    assert create_root_resp.status_code == 200
    root = create_root_resp.json()
    assert root["name"] == "Round 1"
    assert root["kind"] == "manual"
    assert root["parent_folder_id"] is None
    assert root["doc_ids"] == []
    assert root["child_folder_ids"] == []

    create_child_resp = client.post(
        "/api/folders",
        json={"name": "Adjudication", "parent_folder_id": root["id"]},
    )
    assert create_child_resp.status_code == 200
    child = create_child_resp.json()
    assert child["name"] == "Adjudication"
    assert child["kind"] == "manual"
    assert child["parent_folder_id"] == root["id"]

    folders_resp = client.get("/api/folders")
    assert folders_resp.status_code == 200
    folders = folders_resp.json()
    assert [item["name"] for item in folders] == ["Round 1", "Adjudication"]

    root_detail_resp = client.get(f"/api/folders/{root['id']}")
    assert root_detail_resp.status_code == 200
    root_detail = root_detail_resp.json()
    assert root_detail["child_folder_ids"] == [child["id"]]
    assert [item["id"] for item in root_detail["child_folders"]] == [child["id"]]

    delete_root_resp = client.delete(f"/api/folders/{root['id']}")
    assert delete_root_resp.status_code == 200
    assert client.get(f"/api/folders/{root['id']}").status_code == 404
    assert client.get(f"/api/folders/{child['id']}").status_code == 404
    assert client.get("/api/folders").json() == []


def test_prompt_lab_run_accepts_folder_ids(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
    )

    def fake_run_llm_with_metadata(**kwargs):
        text = str(kwargs["text"])
        target = "Liam" if "Liam" in text else "Ava"
        start = text.index(target)
        return SimpleNamespace(
            spans=[
                CanonicalSpan(
                    start=start,
                    end=start + len(target),
                    label="NAME",
                    text=target,
                )
            ],
            warnings=[],
            llm_confidence=_mock_confidence_metric(),
        )

    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)

    upload_resp = _upload(
        client,
        data=_make_multi_record_jsonl(),
        filename="sessions.jsonl",
    )
    assert upload_resp.status_code == 200
    folder_id = client.get("/api/folders").json()[0]["id"]
    folder_detail = client.get(f"/api/folders/{folder_id}").json()
    child_doc_ids = [item["id"] for item in folder_detail["documents"]]

    create_resp = client.post(
        "/api/prompt-lab/runs",
        json={
            "name": "folder-prompt-run",
            "folder_ids": [folder_id],
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
                "reference_source": "pre",
                "fallback_reference_source": "pre",
            },
            "concurrency": 2,
        },
    )
    assert create_resp.status_code == 200
    run_id = create_resp.json()["id"]

    final = _wait_for_prompt_lab_terminal(client, run_id)
    assert final["status"] == "completed"
    assert final["folder_ids"] == [folder_id]
    assert final["doc_ids"] == child_doc_ids
    assert final["matrix"]["cells"][0]["completed_docs"] == 2


def test_methods_lab_run_accepts_folder_ids(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
    )

    def fake_run_method_with_metadata(**kwargs):
        text = str(kwargs["text"])
        target = "Liam" if "Liam" in text else "Ava"
        start = text.index(target)
        return SimpleNamespace(
            spans=[
                CanonicalSpan(
                    start=start,
                    end=start + len(target),
                    label="NAME",
                    text=target,
                )
            ],
            warnings=[],
            llm_confidence=_mock_confidence_metric(),
        )

    monkeypatch.setattr("server.run_method_with_metadata", fake_run_method_with_metadata)

    upload_resp = _upload(
        client,
        data=_make_multi_record_jsonl(),
        filename="sessions.jsonl",
    )
    assert upload_resp.status_code == 200
    folder_id = client.get("/api/folders").json()[0]["id"]
    folder_detail = client.get(f"/api/folders/{folder_id}").json()
    child_doc_ids = [item["id"] for item in folder_detail["documents"]]

    for doc_id in child_doc_ids:
        document = client.get(f"/api/documents/{doc_id}").json()
        save_resp = client.put(
            f"/api/documents/{doc_id}/manual-annotations",
            json=document["pre_annotations"],
        )
        assert save_resp.status_code == 200

    create_resp = client.post(
        "/api/methods-lab/runs",
        json={
            "name": "folder-methods-run",
            "folder_ids": [folder_id],
            "methods": [
                {
                    "id": "method_1",
                    "label": "Default",
                    "method_id": "default",
                }
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
                "label_profile": "simple",
                "label_projection": "native",
                "chunk_mode": "off",
                "chunk_size_chars": 10000,
            },
            "concurrency": 1,
        },
    )
    assert create_resp.status_code == 200
    run_id = create_resp.json()["id"]

    final = _wait_for_methods_lab_terminal(client, run_id)
    assert final["status"] == "completed"
    assert final["folder_ids"] == [folder_id]
    assert final["doc_ids"] == child_doc_ids
    assert final["matrix"]["cells"][0]["completed_docs"] == 2


def test_methods_lab_can_score_against_pre_annotations(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
    )

    def fake_run_method_with_metadata(**kwargs):
        text = str(kwargs["text"])
        start = text.index("Anna")
        return SimpleNamespace(
            spans=[
                CanonicalSpan(
                    start=start,
                    end=start + 4,
                    label="NAME",
                    text="Anna",
                )
            ],
            warnings=[],
            llm_confidence=_mock_confidence_metric(),
        )

    monkeypatch.setattr("server.run_method_with_metadata", fake_run_method_with_metadata)

    upload_resp = _upload(client)
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    create_resp = client.post(
        "/api/methods-lab/runs",
        json={
            "name": "methods-vs-pre",
            "doc_ids": [doc_id],
            "methods": [
                {
                    "id": "method_1",
                    "label": "Default",
                    "method_id": "default",
                }
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
                "reference_source": "pre",
                "fallback_reference_source": "pre",
                "label_profile": "simple",
                "label_projection": "native",
                "chunk_mode": "off",
                "chunk_size_chars": 10000,
            },
            "concurrency": 1,
        },
    )
    assert create_resp.status_code == 200
    run_id = create_resp.json()["id"]

    final = _wait_for_methods_lab_terminal(client, run_id)
    assert final["status"] == "completed"
    assert final["runtime"]["reference_source"] == "pre"
    assert final["runtime"]["fallback_reference_source"] == "pre"
    assert final["matrix"]["cells"][0]["completed_docs"] == 1

    detail_resp = client.get(
        f"/api/methods-lab/runs/{run_id}/cells/{final['matrix']['cells'][0]['id']}/documents/{doc_id}"
    )
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["status"] == "completed"
    assert detail["reference_source_used"] == "pre"
    assert detail["metrics"]["micro"]["f1"] == pytest.approx(1.0)


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
    assert bundle["version"] == 5
    assert "project" not in bundle
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
    assert imported["bundle_version"] == 5
    assert imported["imported_count"] == 1
    assert imported["skipped_count"] == 0

    imported_id = imported["imported_ids"][0]
    doc_resp = client.get(f"/api/documents/{imported_id}")
    assert doc_resp.status_code == 200
    imported_doc = doc_resp.json()
    assert len(imported_doc["manual_annotations"]) == 1
    assert isinstance(imported_doc["agent_outputs"]["rule"], list)


def test_session_export_import_roundtrip_preserves_folders_and_folder_run_selections(
    client,
    monkeypatch,
):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
    )

    def fake_run_llm_with_metadata(**kwargs):
        text = str(kwargs["text"])
        target = "Liam" if "Liam" in text else "Ava"
        start = text.index(target)
        return SimpleNamespace(
            spans=[
                CanonicalSpan(
                    start=start,
                    end=start + len(target),
                    label="NAME",
                    text=target,
                )
            ],
            warnings=[],
            llm_confidence=_mock_confidence_metric(),
        )

    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)

    upload_resp = _upload(
        client,
        data=_make_multi_record_jsonl(),
        filename="sessions.jsonl",
    )
    assert upload_resp.status_code == 200
    merged_doc_id = upload_resp.json()["id"]
    folder_id = client.get("/api/folders").json()[0]["id"]

    run_resp = client.post(
        "/api/prompt-lab/runs",
        json={
            "name": "folder-prompt-run",
            "folder_ids": [folder_id],
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
                "reference_source": "pre",
                "fallback_reference_source": "pre",
            },
            "concurrency": 2,
        },
    )
    assert run_resp.status_code == 200
    original_run_id = run_resp.json()["id"]
    _wait_for_prompt_lab_terminal(client, original_run_id)

    export_resp = client.get("/api/session/export")
    assert export_resp.status_code == 200
    bundle = export_resp.json()
    assert bundle["version"] == 5
    assert len(bundle["documents"]) == 3
    assert len(bundle["folders"]) == 1
    assert bundle["prompt_lab_runs"][0]["folder_ids"] == [folder_id]

    delete_folder_resp = client.delete(f"/api/folders/{folder_id}")
    assert delete_folder_resp.status_code == 200
    delete_doc_resp = client.delete(f"/api/documents/{merged_doc_id}")
    assert delete_doc_resp.status_code == 200
    assert client.get("/api/documents").json() == []

    import_resp = client.post(
        "/api/session/import",
        files={"file": ("session_bundle.json", json.dumps(bundle).encode(), "application/json")},
    )
    assert import_resp.status_code == 200
    imported = import_resp.json()
    assert imported["bundle_version"] == 5
    assert imported["imported_count"] == 3
    assert imported["imported_prompt_lab_runs"] >= 1

    docs_resp = client.get("/api/documents")
    assert docs_resp.status_code == 200
    docs = docs_resp.json()
    assert len(docs) == 1

    folders_resp = client.get("/api/folders")
    assert folders_resp.status_code == 200
    folders = folders_resp.json()
    assert len(folders) == 1
    imported_folder_id = folders[0]["id"]

    folder_detail_resp = client.get(f"/api/folders/{imported_folder_id}")
    assert folder_detail_resp.status_code == 200
    assert len(folder_detail_resp.json()["documents"]) == 2

    runs_resp = client.get("/api/prompt-lab/runs")
    assert runs_resp.status_code == 200
    runs = runs_resp.json()["runs"]
    imported_run = next((item for item in runs if item["id"] != original_run_id), None)
    assert imported_run is not None

    imported_detail_resp = client.get(f"/api/prompt-lab/runs/{imported_run['id']}")
    assert imported_detail_resp.status_code == 200
    imported_detail = imported_detail_resp.json()
    assert imported_detail["folder_ids"] == [imported_folder_id]
    assert len(imported_detail["doc_ids"]) == 2


def test_session_import_reuses_existing_matching_document(client):
    first = _upload(client)
    doc_id = first.json()["id"]
    save_resp = client.put(
        f"/api/documents/{doc_id}/manual-annotations",
        json=[{"start": 6, "end": 10, "label": "NAME", "text": "Anna"}],
    )
    assert save_resp.status_code == 200

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
    assert data["imported_ids"] == [doc_id]

    docs_resp = client.get("/api/documents")
    assert docs_resp.status_code == 200
    assert len(docs_resp.json()) == 1


def test_session_import_matches_existing_doc_by_filename_and_transcript_and_prefers_richer_manual_annotations(
    client,
):
    upload_resp = _upload(
        client,
        data=_make_hips_v1_custom(
            "Hello Anna, call Sue please.",
            pii_occurrences=[{"start": 6, "end": 10, "text": "Anna", "pii_type": "NAME"}],
        ),
        filename="shared.json",
    )
    assert upload_resp.status_code == 200
    existing_doc_id = upload_resp.json()["id"]

    existing_save = client.put(
        f"/api/documents/{existing_doc_id}/manual-annotations",
        json=[{"start": 6, "end": 10, "label": "NAME", "text": "Anna"}],
    )
    assert existing_save.status_code == 200

    existing_doc = client.get(f"/api/documents/{existing_doc_id}").json()
    imported_source = {
        **existing_doc,
        "id": "imported-doc-1",
        "manual_annotations": [],
        "agent_annotations": [],
        "agent_outputs": {
            "rule": [],
            "llm": [],
            "llm_runs": {},
            "llm_run_metadata": {},
            "methods": {},
            "method_run_metadata": {},
        },
        "agent_run_warnings": [],
        "agent_run_metrics": {"llm_confidence": None, "label_profile": None, "chunk_diagnostics": []},
    }
    import_payload = {
        "format": "annotation_tool_session",
        "version": 4,
        "compatibility": {"tool_version": "2026.03.03", "import_supported_versions": [1, 2, 3, 4]},
        "documents": [
            {
                "source": imported_source,
                "manual_annotations": [
                    {"start": 0, "end": 5, "label": "NAME", "text": "Hello"},
                    {"start": 6, "end": 10, "label": "NAME", "text": "Anna"},
                ],
            }
        ],
        "prompt_lab_runs": [],
        "methods_lab_runs": [],
    }

    import_resp = client.post(
        "/api/session/import",
        files={
            "file": (
                "session_bundle.json",
                json.dumps(import_payload).encode(),
                "application/json",
            )
        },
    )
    assert import_resp.status_code == 200
    data = import_resp.json()
    assert data["imported_ids"] == [existing_doc_id]

    docs_resp = client.get("/api/documents")
    assert docs_resp.status_code == 200
    assert len(docs_resp.json()) == 1

    refreshed = client.get(f"/api/documents/{existing_doc_id}")
    assert refreshed.status_code == 200
    refreshed_doc = refreshed.json()
    assert len(refreshed_doc["manual_annotations"]) == 2


def test_session_import_keep_current_conflict_preserves_existing_document(client):
    upload_resp = _upload(
        client,
        data=_make_hips_v1_custom(
            "Hello Anna, call Sue please.",
            pii_occurrences=[{"start": 6, "end": 10, "text": "Anna", "pii_type": "NAME"}],
        ),
        filename="shared.json",
    )
    assert upload_resp.status_code == 200
    existing_doc_id = upload_resp.json()["id"]

    existing_save = client.put(
        f"/api/documents/{existing_doc_id}/manual-annotations",
        json=[
            {"start": 0, "end": 5, "label": "NAME", "text": "Hello"},
            {"start": 6, "end": 10, "label": "NAME", "text": "Anna"},
        ],
    )
    assert existing_save.status_code == 200

    existing_doc = client.get(f"/api/documents/{existing_doc_id}").json()
    imported_source = {
        **existing_doc,
        "id": "imported-doc-keep",
        "manual_annotations": [],
        "agent_annotations": [],
        "agent_outputs": {
            "rule": [],
            "llm": [],
            "llm_runs": {},
            "llm_run_metadata": {},
            "methods": {},
            "method_run_metadata": {},
        },
        "agent_run_warnings": [],
        "agent_run_metrics": {"llm_confidence": None, "label_profile": None, "chunk_diagnostics": []},
    }
    import_payload = {
        "format": "annotation_tool_session",
        "version": 4,
        "compatibility": {"tool_version": "2026.03.03", "import_supported_versions": [1, 2, 3, 4]},
        "documents": [
            {
                "source": imported_source,
                "manual_annotations": [
                    {"start": 6, "end": 10, "label": "NAME", "text": "Anna"},
                ],
            }
        ],
        "prompt_lab_runs": [],
        "methods_lab_runs": [],
    }

    import_resp = client.post(
        "/api/session/import",
        data={"conflict_policy": "keep_current"},
        files={
            "file": (
                "session_bundle.json",
                json.dumps(import_payload).encode(),
                "application/json",
            )
        },
    )
    assert import_resp.status_code == 200
    data = import_resp.json()
    assert data["imported_ids"] == [existing_doc_id]
    assert data["created_count"] == 0
    assert data["kept_current_count"] == 1

    docs_resp = client.get("/api/documents")
    assert docs_resp.status_code == 200
    assert len(docs_resp.json()) == 1

    refreshed = client.get(f"/api/documents/{existing_doc_id}")
    assert refreshed.status_code == 200
    refreshed_doc = refreshed.json()
    assert refreshed_doc["manual_annotations"] == [
        {"start": 0, "end": 5, "label": "NAME", "text": "Hello"},
        {"start": 6, "end": 10, "label": "NAME", "text": "Anna"},
    ]


def test_session_import_add_new_conflict_creates_duplicate_document(client):
    upload_resp = _upload(
        client,
        data=_make_hips_v1_custom(
            "Hello Anna, call Sue please.",
            pii_occurrences=[{"start": 6, "end": 10, "text": "Anna", "pii_type": "NAME"}],
        ),
        filename="shared.json",
    )
    assert upload_resp.status_code == 200
    existing_doc_id = upload_resp.json()["id"]

    existing_save = client.put(
        f"/api/documents/{existing_doc_id}/manual-annotations",
        json=[
            {"start": 0, "end": 5, "label": "NAME", "text": "Hello"},
            {"start": 6, "end": 10, "label": "NAME", "text": "Anna"},
        ],
    )
    assert existing_save.status_code == 200

    existing_doc = client.get(f"/api/documents/{existing_doc_id}").json()
    imported_source = {
        **existing_doc,
        "id": existing_doc_id,
        "manual_annotations": [],
        "agent_annotations": [],
        "agent_outputs": {
            "rule": [],
            "llm": [],
            "llm_runs": {},
            "llm_run_metadata": {},
            "methods": {},
            "method_run_metadata": {},
        },
        "agent_run_warnings": [],
        "agent_run_metrics": {"llm_confidence": None, "label_profile": None, "chunk_diagnostics": []},
    }
    import_payload = {
        "format": "annotation_tool_session",
        "version": 4,
        "compatibility": {"tool_version": "2026.03.03", "import_supported_versions": [1, 2, 3, 4]},
        "documents": [
            {
                "source": imported_source,
                "manual_annotations": [
                    {"start": 6, "end": 10, "label": "NAME", "text": "Anna"},
                ],
            }
        ],
        "prompt_lab_runs": [],
        "methods_lab_runs": [],
    }

    import_resp = client.post(
        "/api/session/import",
        data={"conflict_policy": "add_new"},
        files={
            "file": (
                "session_bundle.json",
                json.dumps(import_payload).encode(),
                "application/json",
            )
        },
    )
    assert import_resp.status_code == 200
    data = import_resp.json()
    assert data["created_count"] == 1
    assert data["added_as_new_count"] == 1

    imported_id = data["imported_ids"][0]
    assert imported_id != existing_doc_id

    docs_resp = client.get("/api/documents")
    assert docs_resp.status_code == 200
    assert len(docs_resp.json()) == 2

    existing_doc = client.get(f"/api/documents/{existing_doc_id}").json()
    imported_doc = client.get(f"/api/documents/{imported_id}").json()
    assert existing_doc["manual_annotations"] == [
        {"start": 0, "end": 5, "label": "NAME", "text": "Hello"},
        {"start": 6, "end": 10, "label": "NAME", "text": "Anna"},
    ]
    assert imported_doc["manual_annotations"] == [
        {"start": 6, "end": 10, "label": "NAME", "text": "Anna"},
    ]


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


def test_session_profile_routes_are_removed(client):
    get_resp = client.get("/api/session/profile")
    assert get_resp.status_code == 404

    put_resp = client.put(
        "/api/session/profile",
        json={
            "project_name": "HIPS QA",
            "author": "Chason",
        },
    )
    assert put_resp.status_code == 404


def test_session_import_ignores_legacy_project_metadata(client):
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
    data = import_resp.json()
    assert data["imported_count"] == 0
    assert data["skipped_count"] == 0


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


def test_metrics_substring_mode_accepts_nested_same_label(client):
    resp = _upload(
        client,
        data=_make_hips_v1_custom(
            "Mr. Evans joined.",
            pii_occurrences=[
                {"start": 0, "end": 9, "text": "Mr. Evans", "pii_type": "NAME"},
            ],
        ),
    )
    doc_id = resp.json()["id"]

    manual_spans = [
        {"start": 4, "end": 9, "label": "NAME", "text": "Evans"},
    ]
    save_resp = client.put(f"/api/documents/{doc_id}/manual-annotations", json=manual_spans)
    assert save_resp.status_code == 200

    substring_resp = client.get(
        f"/api/documents/{doc_id}/metrics",
        params={"reference": "pre", "hypothesis": "manual", "match_mode": "substring"},
    )
    assert substring_resp.status_code == 200
    assert substring_resp.json()["micro"]["f1"] == 1.0
    assert substring_resp.json()["match_mode"] == "substring"


def test_metrics_endpoint_defaults_to_overlap_matching(client):
    resp = _upload(
        client,
        data=_make_hips_v1_custom(
            "Hello Anna, call Sue please.",
            pii_occurrences=[
                {"start": 6, "end": 10, "text": "Anna", "pii_type": "NAME"},
            ],
        ),
    )
    doc_id = resp.json()["id"]

    manual_resp = client.put(
        f"/api/documents/{doc_id}/manual-annotations",
        json=[{"start": 6, "end": 11, "label": "NAME", "text": "Anna,"}],
    )
    assert manual_resp.status_code == 200

    metrics_resp = client.get(
        f"/api/documents/{doc_id}/metrics",
        params={"reference": "pre", "hypothesis": "manual"},
    )
    assert metrics_resp.status_code == 200
    assert metrics_resp.json()["match_mode"] == "overlap"
    assert metrics_resp.json()["micro"]["f1"] == 1.0


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


def test_agent_llm_default_chunk_mode_stays_single_pass_for_long_text(client, monkeypatch):
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
        },
    )
    assert run_resp.status_code == 200
    payload = run_resp.json()
    assert call_count["count"] == 1
    assert all(
        "Chunked LLM run used" not in warning for warning in payload["agent_run_warnings"]
    )


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


def test_metrics_exact_includes_name_tolerant_co_primary_metric(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
    )

    text = "Hello Mr. Muhammad"
    muhammad_start = text.index("Muhammad")

    def fake_run_llm_with_metadata(**kwargs):
        return SimpleNamespace(
            spans=[
                CanonicalSpan(
                    start=muhammad_start,
                    end=muhammad_start + len("Muhammad"),
                    label="NAME",
                    text="Muhammad",
                )
            ],
            warnings=[],
            llm_confidence=_mock_confidence_metric(),
        )

    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)

    upload_resp = _upload(client, _make_hips_v1_custom(text))
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    manual_resp = client.put(
        f"/api/documents/{doc_id}/manual-annotations",
        json=[
            {
                "start": text.index("Mr."),
                "end": len(text),
                "label": "NAME",
                "text": "Mr. Muhammad",
            }
        ],
    )
    assert manual_resp.status_code == 200

    run_resp = client.post(
        f"/api/documents/{doc_id}/agent",
        json={
            "mode": "llm",
            "model": "openai.gpt-5.3-codex",
            "api_key": "request-key",
            "api_base": "https://proxy.example.com/v1",
        },
    )
    assert run_resp.status_code == 200

    metrics_resp = client.get(
        f"/api/documents/{doc_id}/metrics",
        params={"reference": "manual", "hypothesis": "agent.llm", "match_mode": "exact"},
    )
    assert metrics_resp.status_code == 200
    payload = metrics_resp.json()
    assert payload["micro"]["f1"] == pytest.approx(0.0)
    overlap = payload["co_primary_metrics"]["overlap"]
    assert overlap["micro"]["f1"] == pytest.approx(1.0)
    assert overlap["per_label"]["NAME"]["tp"] == 1


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
    assert "co_primary_metrics" in data
    assert "overlap" in data["co_primary_metrics"]
    assert "avg_document_micro" in data
    assert "avg_document_macro" in data
    assert "llm_confidence_summary" in data
    assert "band_counts" in data["llm_confidence_summary"]
    assert len(data["documents"]) == 2
    first_doc = data["documents"][0]
    assert "id" in first_doc and "filename" in first_doc
    assert "micro" in first_doc and "f1" in first_doc["micro"]
    assert "macro" in first_doc and "precision" in first_doc["macro"]
    assert "co_primary_metrics" in first_doc
    assert "overlap" in first_doc["co_primary_metrics"]
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


def test_experiment_limits_endpoint_uses_configured_caps(client):
    _write_test_config(
        {
            "prompt_lab_max_concurrency": 17,
            "methods_lab_max_concurrency": 99,
        }
    )

    resp = client.get("/api/experiments/limits")
    assert resp.status_code == 200
    assert resp.json() == {
        "prompt_lab_default_concurrency": 10,
        "prompt_lab_max_concurrency": 17,
        "methods_lab_default_concurrency": 10,
        "methods_lab_max_concurrency": 32,
    }


def test_experiment_diagnostics_endpoint_reports_gateway_catalog(client, monkeypatch):
    _write_test_config({"api_base": "https://proxy.example.com"})
    monkeypatch.setenv("LITELLM_API_KEY", "litellm-key")
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: [
            "openai.gpt-4.1-nano",
            "openai.gpt-5.3-codex",
        ],
    )

    resp = client.get("/api/experiments/diagnostics")

    assert resp.status_code == 200
    assert resp.json() == {
        "resolved_api_base": "https://proxy.example.com",
        "api_base_host": "proxy.example.com",
        "prompt_lab_max_concurrency": 16,
        "methods_lab_max_concurrency": 16,
        "gateway_catalog": {
            "reachable": True,
            "model_count": 2,
            "error": None,
            "checked_at": resp.json()["gateway_catalog"]["checked_at"],
        },
    }
    assert resp.json()["gateway_catalog"]["checked_at"]


def test_experiment_diagnostics_endpoint_reports_gateway_catalog_error(client, monkeypatch):
    _write_test_config({"api_base": "https://proxy.example.com"})
    monkeypatch.setenv("LITELLM_API_KEY", "litellm-key")

    def fail_fetch(_api_base: str, _api_key: str) -> list[str]:
        raise HTTPException(status_code=502, detail="Gateway lookup failed for diagnostics")

    monkeypatch.setattr("server._fetch_gateway_model_ids", fail_fetch)

    resp = client.get("/api/experiments/diagnostics")

    assert resp.status_code == 200
    assert resp.json() == {
        "resolved_api_base": "https://proxy.example.com",
        "api_base_host": "proxy.example.com",
        "prompt_lab_max_concurrency": 16,
        "methods_lab_max_concurrency": 16,
        "gateway_catalog": {
            "reachable": False,
            "model_count": None,
            "error": "Gateway lookup failed for diagnostics",
            "checked_at": resp.json()["gateway_catalog"]["checked_at"],
        },
    }
    assert resp.json()["gateway_catalog"]["checked_at"]


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
    assert "anthropic.claude-4.6-opus" in model_ids
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


def test_load_repo_env_file_reads_root_env_without_overwriting_process_env(
    monkeypatch, tmp_path
):
    env_path = tmp_path / ".env.local"
    env_path.write_text(
        "\n".join(
            [
                "LITELLM_API_KEY=file-key",
                "LITELLM_BASE_URL=https://file.example.com/v1",
                "OPENAI_API_KEY=file-openai-key",
            ]
        )
    )
    monkeypatch.setenv("OPENAI_API_KEY", "process-openai-key")
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)

    loaded = _load_repo_env_file(env_path)

    assert loaded == {
        "LITELLM_API_KEY": "file-key",
        "LITELLM_BASE_URL": "https://file.example.com/v1",
    }
    assert ROOT_ENV_PATH != env_path


def test_resolve_env_runtime_credentials_ignore_root_env_file(monkeypatch, tmp_path):
    env_path = tmp_path / ".env.local"
    env_path.write_text(
        "\n".join(
            [
                "LITELLM_API_KEY=file-key",
                "LITELLM_BASE_URL=https://file.example.com/v1",
            ]
        )
    )
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.setattr("server.ROOT_ENV_PATH", env_path)

    assert _resolve_env_api_key() == ""
    assert _resolve_env_api_base() == ""
    assert "LITELLM_API_KEY" not in os.environ
    assert "LITELLM_BASE_URL" not in os.environ


def test_agent_credentials_status_loads_root_env_file(client, monkeypatch, tmp_path):
    env_path = tmp_path / ".env.local"
    env_path.write_text(
        "\n".join(
            [
                "LITELLM_API_KEY=file-key",
                "LITELLM_BASE_URL=https://file.example.com/v1",
            ]
        )
    )
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("LITELLM_BASE_URL", raising=False)
    monkeypatch.setattr("server.ROOT_ENV_PATH", env_path)

    resp = client.get("/api/agent/credentials/status")

    assert resp.status_code == 200
    data = resp.json()
    assert data["has_api_key"] is True
    assert data["api_key_sources"] == ["LITELLM_API_KEY"]
    assert data["has_api_base"] is True
    assert data["api_base_sources"] == ["LITELLM_BASE_URL"]


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


def test_prompt_lab_detail_persists_response_debug(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
    )

    def fake_run_llm_with_metadata(**kwargs):
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            warnings=[],
            llm_confidence=_mock_confidence_metric(confidence=0.91, band="high"),
            response_debug=[
                "resp_type=ModelResponse; finish_reason=stop; message.content=str(len=42)"
            ],
        )

    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)

    upload_resp = _upload(client)
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]
    manual_resp = client.put(
        f"/api/documents/{doc_id}/manual-annotations",
        json=[{"start": 6, "end": 10, "label": "NAME", "text": "Anna"}],
    )
    assert manual_resp.status_code == 200

    create_resp = client.post(
        "/api/prompt-lab/runs",
        json={
            "name": "response-debug-check",
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
                    "reasoning_effort": "none",
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

    final = _wait_for_prompt_lab_terminal(client, run_id)
    assert final["status"] == "completed"
    cell_id = final["matrix"]["cells"][0]["id"]

    stored = json.loads(_prompt_lab_run_path(run_id).read_text())
    assert (
        stored["cells"][cell_id]["documents"][doc_id]["response_debug"]
        == ["resp_type=ModelResponse; finish_reason=stop; message.content=str(len=42)"]
    )

    detail_resp = client.get(
        f"/api/prompt-lab/runs/{run_id}/cells/{cell_id}/documents/{doc_id}"
    )
    assert detail_resp.status_code == 200
    assert detail_resp.json()["response_debug"] == [
        "resp_type=ModelResponse; finish_reason=stop; message.content=str(len=42)"
    ]


def test_prompt_lab_cell_summary_includes_overlap_metrics_and_error_families(
    client, monkeypatch
):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
    )

    first_text = "Hello Mr. Muhammad"
    second_text = "Hello Zara"
    muhammad_start = first_text.index("Muhammad")

    def fake_run_llm_with_metadata(**kwargs):
        text = str(kwargs["text"])
        if "Zara" in text:
            raise ValueError("LLM returned empty output content (finish_reason=length).")
        return SimpleNamespace(
            spans=[
                CanonicalSpan(
                    start=muhammad_start,
                    end=muhammad_start + len("Muhammad"),
                    label="NAME",
                    text="Muhammad",
                )
            ],
            warnings=[],
            llm_confidence=_mock_confidence_metric(confidence=0.91, band="high"),
        )

    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)

    first_upload = _upload(client, _make_hips_v1_custom(first_text), filename="prompt-a.json")
    second_upload = _upload(client, _make_hips_v1_custom(second_text), filename="prompt-b.json")
    assert first_upload.status_code == 200
    assert second_upload.status_code == 200
    first_doc_id = first_upload.json()["id"]
    second_doc_id = second_upload.json()["id"]

    first_manual = client.put(
        f"/api/documents/{first_doc_id}/manual-annotations",
        json=[
            {
                "start": first_text.index("Mr."),
                "end": len(first_text),
                "label": "NAME",
                "text": "Mr. Muhammad",
            }
        ],
    )
    second_manual = client.put(
        f"/api/documents/{second_doc_id}/manual-annotations",
        json=[
            {
                "start": second_text.index("Zara"),
                "end": len(second_text),
                "label": "NAME",
                "text": "Zara",
            }
        ],
    )
    assert first_manual.status_code == 200
    assert second_manual.status_code == 200

    create_resp = client.post(
        "/api/prompt-lab/runs",
        json={
            "name": "overlap-prompt-run",
            "doc_ids": [first_doc_id, second_doc_id],
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
                    "reasoning_effort": "none",
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
    assert cell["pending_docs"] == 0
    assert cell["micro"]["f1"] == pytest.approx(0.0)
    overlap = cell["co_primary_metrics"]["overlap"]
    assert overlap["micro"]["f1"] == pytest.approx(1.0)
    assert cell["error_families"]["empty_output_finish_reason_length"] == 1

    detail_resp = client.get(
        f"/api/prompt-lab/runs/{run_id}/cells/{cell['id']}/documents/{second_doc_id}"
    )
    assert detail_resp.status_code == 200
    assert detail_resp.json()["error_family"] == "empty_output_finish_reason_length"


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


def test_prompt_lab_preset_variant_accepts_legacy_method_bundle(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
    )
    seen_method_bundles: list[str] = []

    def fake_run_method_with_metadata(**kwargs):
        seen_method_bundles.append(str(kwargs.get("method_bundle")))
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            raw_spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            warnings=[],
            llm_confidence=None,
            response_debug=[],
            resolution_events=[],
            resolution_policy_version=None,
        )

    monkeypatch.setattr("server.run_method_with_metadata", fake_run_method_with_metadata)

    upload_resp = _upload(client)
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    create_resp = client.post(
        "/api/prompt-lab/runs",
        json={
            "name": "preset-method-legacy-run",
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
                    "reasoning_effort": "none",
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
                "method_bundle": "legacy",
            },
            "concurrency": 1,
        },
    )
    assert create_resp.status_code == 200
    run_id = create_resp.json()["id"]
    assert create_resp.json()["method_bundle"] == "legacy"

    final = _wait_for_prompt_lab_terminal(client, run_id)
    assert final["status"] == "completed"
    assert final["method_bundle"] == "legacy"
    assert final["runtime"]["method_bundle"] == "legacy"
    runs_resp = client.get("/api/prompt-lab/runs")
    assert runs_resp.status_code == 200
    runs = runs_resp.json()["runs"]
    listed = next((row for row in runs if row["id"] == run_id), None)
    assert listed is not None
    assert listed["method_bundle"] == "legacy"
    assert seen_method_bundles == ["legacy"]


def test_prompt_lab_preset_variant_accepts_v2_post_process_method_bundle(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
    )
    seen_method_bundles: list[str] = []

    def fake_run_method_with_metadata(**kwargs):
        seen_method_bundles.append(str(kwargs.get("method_bundle")))
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            raw_spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            warnings=[],
            llm_confidence=None,
            response_debug=[],
            resolution_events=[],
            resolution_policy_version=None,
        )

    monkeypatch.setattr("server.run_method_with_metadata", fake_run_method_with_metadata)

    upload_resp = _upload(client)
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    create_resp = client.post(
        "/api/prompt-lab/runs",
        json={
            "name": "preset-method-v2-post-process-run",
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
                    "reasoning_effort": "none",
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
                "method_bundle": "v2+post-process",
            },
            "concurrency": 1,
        },
    )
    assert create_resp.status_code == 200
    run_id = create_resp.json()["id"]
    assert create_resp.json()["method_bundle"] == "v2+post-process"

    final = _wait_for_prompt_lab_terminal(client, run_id)
    assert final["status"] == "completed"
    assert final["method_bundle"] == "v2+post-process"
    assert final["runtime"]["method_bundle"] == "v2+post-process"
    runs_resp = client.get("/api/prompt-lab/runs")
    assert runs_resp.status_code == 200
    runs = runs_resp.json()["runs"]
    listed = next((row for row in runs if row["id"] == run_id), None)
    assert listed is not None
    assert listed["method_bundle"] == "v2+post-process"
    assert seen_method_bundles == ["v2+post-process"]


def test_prompt_lab_preset_variant_accepts_deidentify_v2_method_bundle(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
    )
    seen_method_bundles: list[str] = []

    def fake_run_method_with_metadata(**kwargs):
        seen_method_bundles.append(str(kwargs.get("method_bundle")))
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            raw_spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            warnings=[],
            llm_confidence=None,
            response_debug=[],
            resolution_events=[],
            resolution_policy_version=None,
        )

    monkeypatch.setattr("server.run_method_with_metadata", fake_run_method_with_metadata)

    upload_resp = _upload(client)
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    create_resp = client.post(
        "/api/prompt-lab/runs",
        json={
            "name": "preset-method-deidentify-v2-run",
            "doc_ids": [doc_id],
            "prompts": [
                {
                    "id": "preset1",
                    "label": "Dual V2 preset",
                    "variant_type": "preset",
                    "preset_method_id": "dual-v2",
                }
            ],
            "models": [
                {
                    "id": "m1",
                    "label": "Codex",
                    "model": "openai.gpt-5.3-codex",
                    "reasoning_effort": "none",
                    "anthropic_thinking": False,
                    "anthropic_thinking_budget_tokens": None,
                }
            ],
            "runtime": {
                "api_key": "request-key",
                "api_base": "https://proxy.example.com/v1",
                "temperature": 0.0,
                "match_mode": "substring",
                "reference_source": "pre",
                "fallback_reference_source": "pre",
                "method_bundle": "deidentify-v2",
            },
            "concurrency": 1,
        },
    )
    assert create_resp.status_code == 200
    run_id = create_resp.json()["id"]
    assert create_resp.json()["method_bundle"] == "deidentify-v2"

    final = _wait_for_prompt_lab_terminal(client, run_id)
    assert final["status"] == "completed"
    assert final["method_bundle"] == "deidentify-v2"
    assert final["runtime"]["method_bundle"] == "deidentify-v2"
    runs_resp = client.get("/api/prompt-lab/runs")
    assert runs_resp.status_code == 200
    runs = runs_resp.json()["runs"]
    listed = next((row for row in runs if row["id"] == run_id), None)
    assert listed is not None
    assert listed["method_bundle"] == "deidentify-v2"
    assert seen_method_bundles == ["deidentify-v2"]


def test_prompt_lab_preset_variant_rejects_unsupported_verify_override_for_deidentify_v2(
    client, monkeypatch
):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
    )

    upload_resp = _upload(client)
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    create_resp = client.post(
        "/api/prompt-lab/runs",
        json={
            "name": "preset-method-deidentify-v2-verify-override",
            "doc_ids": [doc_id],
            "prompts": [
                {
                    "id": "preset1",
                    "label": "Dual V2 preset",
                    "variant_type": "preset",
                    "preset_method_id": "dual-v2",
                    "method_verify_override": True,
                }
            ],
            "models": [
                {
                    "id": "m1",
                    "label": "Codex",
                    "model": "openai.gpt-5.3-codex",
                    "reasoning_effort": "none",
                    "anthropic_thinking": False,
                    "anthropic_thinking_budget_tokens": None,
                }
            ],
            "runtime": {
                "api_key": "request-key",
                "api_base": "https://proxy.example.com/v1",
                "temperature": 0.0,
                "match_mode": "substring",
                "reference_source": "pre",
                "fallback_reference_source": "pre",
                "method_bundle": "deidentify-v2",
            },
            "concurrency": 1,
        },
    )
    assert create_resp.status_code == 400
    assert "method_verify_override is not supported" in create_resp.json()["detail"]


def test_prompt_lab_preset_variant_accepts_v2_method_bundle(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
    )
    seen_method_bundles: list[str] = []

    def fake_run_method_with_metadata(**kwargs):
        seen_method_bundles.append(str(kwargs.get("method_bundle")))
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            raw_spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            warnings=[],
            llm_confidence=None,
            response_debug=[],
            resolution_events=[],
            resolution_policy_version=None,
        )

    monkeypatch.setattr("server.run_method_with_metadata", fake_run_method_with_metadata)

    upload_resp = _upload(client)
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    create_resp = client.post(
        "/api/prompt-lab/runs",
        json={
            "name": "preset-method-v2-run",
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
                    "reasoning_effort": "none",
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
                "method_bundle": "v2",
            },
            "concurrency": 1,
        },
    )
    assert create_resp.status_code == 200
    run_id = create_resp.json()["id"]
    assert create_resp.json()["method_bundle"] == "v2"

    final = _wait_for_prompt_lab_terminal(client, run_id)
    assert final["status"] == "completed"
    assert final["method_bundle"] == "v2"
    assert final["runtime"]["method_bundle"] == "v2"
    runs_resp = client.get("/api/prompt-lab/runs")
    assert runs_resp.status_code == 200
    runs = runs_resp.json()["runs"]
    listed = next((row for row in runs if row["id"] == run_id), None)
    assert listed is not None
    assert listed["method_bundle"] == "v2"
    assert seen_method_bundles == ["v2"]


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


def test_prompt_lab_accepts_concurrency_above_legacy_limit(client, monkeypatch):
    _write_test_config({"prompt_lab_max_concurrency": 16})
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
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

    create_resp = client.post(
        "/api/prompt-lab/runs",
        json={
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
            "concurrency": 12,
        },
    )
    assert create_resp.status_code == 200
    assert create_resp.json()["concurrency"] == 12

    final = _wait_for_prompt_lab_terminal(client, create_resp.json()["id"])
    assert final["status"] == "completed"
    assert final["concurrency"] == 12


def test_prompt_lab_rejects_concurrency_above_configured_cap(client):
    _write_test_config({"prompt_lab_max_concurrency": 16})

    upload_resp = _upload(client)
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    resp = client.post(
        "/api/prompt-lab/runs",
        json={
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
                "temperature": 0.0,
                "match_mode": "exact",
                "reference_source": "manual",
                "fallback_reference_source": "pre",
            },
            "concurrency": 17,
        },
    )
    assert resp.status_code == 400
    assert "concurrency must be between 1 and 16" in resp.json()["detail"]


def test_prompt_lab_executor_workers_clamp_to_total_tasks(client, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor as RealThreadPoolExecutor

    _write_test_config({"prompt_lab_max_concurrency": 16})
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
    )

    def fake_run_llm_with_metadata(**kwargs):
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            warnings=[],
            llm_confidence=_mock_confidence_metric(),
        )

    captured_workers: list[int] = []

    class RecordingExecutor:
        def __init__(self, max_workers, *args, **kwargs):
            captured_workers.append(int(max_workers))
            self._inner = RealThreadPoolExecutor(max_workers=max_workers, *args, **kwargs)

        def submit(self, *args, **kwargs):
            return self._inner.submit(*args, **kwargs)

        def shutdown(self, *args, **kwargs):
            return self._inner.shutdown(*args, **kwargs)

    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)
    monkeypatch.setattr("server.ThreadPoolExecutor", RecordingExecutor)

    upload_resp = _upload(client)
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    create_resp = client.post(
        "/api/prompt-lab/runs",
        json={
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
            "concurrency": 12,
        },
    )
    assert create_resp.status_code == 200

    final = _wait_for_prompt_lab_terminal(client, create_resp.json()["id"])
    assert final["status"] == "completed"
    assert captured_workers == [1]


def test_prompt_lab_run_detail_reports_clamped_worker_diagnostics(client, monkeypatch):
    _write_test_config({"prompt_lab_max_concurrency": 16})
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
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

    create_resp = client.post(
        "/api/prompt-lab/runs",
        json={
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
            "concurrency": 16,
        },
    )
    assert create_resp.status_code == 200

    final = _wait_for_prompt_lab_terminal(client, create_resp.json()["id"])
    assert final["status"] == "completed"
    assert final["diagnostics"] == {
        "requested_concurrency": 16,
        "effective_worker_count": 1,
        "max_allowed_concurrency": 16,
        "total_tasks": 1,
        "clamped_by_task_count": True,
        "clamped_by_server_cap": False,
        "api_base_host": "proxy.example.com",
    }


def test_prompt_lab_run_detail_reports_effective_worker_diagnostics_for_sixteen_tasks(
    client, monkeypatch
):
    _write_test_config({"prompt_lab_max_concurrency": 16})
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
    )

    def fake_run_llm_with_metadata(**kwargs):
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            warnings=[],
            llm_confidence=_mock_confidence_metric(),
        )

    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)

    doc_ids: list[str] = []
    for index in range(4):
        upload_resp = _upload(
            client,
            data=_make_hips_v1_custom(f"Hello Anna {index}.", [{"start": 6, "end": 10, "text": "Anna", "pii_type": "NAME"}]),
            filename=f"prompt-{index}.json",
        )
        assert upload_resp.status_code == 200
        doc_ids.append(upload_resp.json()["id"])

    create_resp = client.post(
        "/api/prompt-lab/runs",
        json={
            "doc_ids": doc_ids,
            "prompts": [
                {
                    "id": "p1",
                    "label": "Baseline",
                    "system_prompt": "Detect pii spans as strict JSON",
                }
            ],
            "models": [
                {
                    "id": f"m{index + 1}",
                    "label": f"Codex {index + 1}",
                    "model": "openai.gpt-5.3-codex",
                    "reasoning_effort": "xhigh",
                    "anthropic_thinking": False,
                    "anthropic_thinking_budget_tokens": None,
                }
                for index in range(4)
            ],
            "runtime": {
                "api_key": "request-key",
                "api_base": "https://proxy.example.com/v1",
                "temperature": 0.0,
                "match_mode": "exact",
                "reference_source": "manual",
                "fallback_reference_source": "pre",
            },
            "concurrency": 16,
        },
    )
    assert create_resp.status_code == 200

    final = _wait_for_prompt_lab_terminal(client, create_resp.json()["id"])
    assert final["status"] == "completed"
    assert final["diagnostics"] == {
        "requested_concurrency": 16,
        "effective_worker_count": 16,
        "max_allowed_concurrency": 16,
        "total_tasks": 16,
        "clamped_by_task_count": False,
        "clamped_by_server_cap": False,
        "api_base_host": "proxy.example.com",
    }


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
    assert new_doc_id == original_doc_id

    runs_resp = client.get("/api/prompt-lab/runs")
    assert runs_resp.status_code == 200
    runs = runs_resp.json()["runs"]
    imported_run = next((row for row in runs if row["id"] != original_run_id), None)
    assert imported_run is not None

    detail_resp = client.get(f"/api/prompt-lab/runs/{imported_run['id']}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert new_doc_id in detail["doc_ids"]


def test_prompt_lab_run_can_be_deleted(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
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

    create_resp = client.post(
        "/api/prompt-lab/runs",
        json={
            "name": "delete-me",
            "doc_ids": [doc_id],
            "prompts": [{"id": "p1", "label": "Baseline", "system_prompt": "normal run"}],
            "models": [
                {
                    "id": "m1",
                    "label": "Codex",
                    "model": "openai.gpt-5.3-codex",
                    "reasoning_effort": "none",
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
    _wait_for_prompt_lab_terminal(client, run_id)
    assert _prompt_lab_run_path(run_id).exists()

    delete_resp = client.delete(f"/api/prompt-lab/runs/{run_id}")
    assert delete_resp.status_code == 200
    assert delete_resp.json() == {"ok": True, "id": run_id}
    assert not _prompt_lab_run_path(run_id).exists()

    detail_resp = client.get(f"/api/prompt-lab/runs/{run_id}")
    assert detail_resp.status_code == 404

    runs_resp = client.get("/api/prompt-lab/runs")
    assert runs_resp.status_code == 200
    assert all(run["id"] != run_id for run in runs_resp.json()["runs"])


def test_prompt_lab_run_can_be_cancelled(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
    )

    started = threading.Event()
    release = threading.Event()

    def fake_run_llm_with_metadata(**kwargs):
        started.set()
        assert release.wait(timeout=5), "Prompt Lab worker did not release in time"
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            warnings=[],
            llm_confidence=_mock_confidence_metric(),
        )

    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)

    first_upload = _upload(client)
    assert first_upload.status_code == 200
    first_doc_id = first_upload.json()["id"]
    second_upload = _upload(client, filename="second.json")
    assert second_upload.status_code == 200
    second_doc_id = second_upload.json()["id"]
    for doc_id in [first_doc_id, second_doc_id]:
        manual_resp = client.put(
            f"/api/documents/{doc_id}/manual-annotations",
            json=[{"start": 6, "end": 10, "label": "NAME", "text": "Anna"}],
        )
        assert manual_resp.status_code == 200

    create_resp = client.post(
        "/api/prompt-lab/runs",
        json={
            "name": "cancel-prompt-run",
            "doc_ids": [first_doc_id, second_doc_id],
            "prompts": [{"id": "p1", "label": "Baseline", "system_prompt": "normal run"}],
            "models": [
                {
                    "id": "model_1",
                    "label": "Codex",
                    "model": "openai.gpt-5.3-codex",
                    "reasoning_effort": "none",
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
    assert started.wait(timeout=5)

    cancel_resp = client.post(f"/api/prompt-lab/runs/{run_id}/cancel")
    assert cancel_resp.status_code == 200
    assert cancel_resp.json() == {"ok": True, "id": run_id, "status": "cancelling"}

    release.set()
    final = _wait_for_prompt_lab_terminal(client, run_id)
    assert final["status"] == "cancelled"
    assert final["finished_at"] is not None
    assert final["completed_tasks"] < final["total_tasks"]
    assert any(cell["status"] == "cancelled" for cell in final["matrix"]["cells"])

    first_detail = client.get(
        f"/api/prompt-lab/runs/{run_id}/cells/model_1__p1/documents/{first_doc_id}"
    )
    second_detail = client.get(
        f"/api/prompt-lab/runs/{run_id}/cells/model_1__p1/documents/{second_doc_id}"
    )
    assert first_detail.status_code == 200
    assert second_detail.status_code == 200
    assert first_detail.json()["status"] in {"completed", "cancelled"}
    assert second_detail.json()["status"] == "cancelled"


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

    final = _wait_for_methods_lab_terminal(client, run_id, attempts=200)
    assert final["status"] == "completed"
    assert len(final["methods"]) == 8
    assert final["methods"][2]["method_id"] == "verified"
    assert final["methods"][3]["method_id"] == "verified"
    assert final["methods"][3]["method_verify_override"] is False


def test_methods_lab_cell_summary_includes_overlap_metrics_and_error_families(
    client, monkeypatch
):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
    )

    first_text = "Hello Ms. Uddin"
    second_text = "Hello Zara"
    uddin_start = first_text.index("Uddin")

    def fake_run_method_with_metadata(**kwargs):
        text = str(kwargs["text"])
        if "Zara" in text:
            raise ValueError("LLM returned empty output content (finish_reason=length).")
        return SimpleNamespace(
            spans=[
                CanonicalSpan(
                    start=uddin_start,
                    end=uddin_start + len("Uddin"),
                    label="NAME",
                    text="Uddin",
                )
            ],
            warnings=[],
            llm_confidence=_mock_confidence_metric(confidence=0.88, band="medium"),
        )

    monkeypatch.setattr("server.run_method_with_metadata", fake_run_method_with_metadata)

    first_upload = _upload(client, _make_hips_v1_custom(first_text), filename="method-a.json")
    second_upload = _upload(client, _make_hips_v1_custom(second_text), filename="method-b.json")
    assert first_upload.status_code == 200
    assert second_upload.status_code == 200
    first_doc_id = first_upload.json()["id"]
    second_doc_id = second_upload.json()["id"]

    first_manual = client.put(
        f"/api/documents/{first_doc_id}/manual-annotations",
        json=[
            {
                "start": first_text.index("Ms."),
                "end": len(first_text),
                "label": "NAME",
                "text": "Ms. Uddin",
            }
        ],
    )
    second_manual = client.put(
        f"/api/documents/{second_doc_id}/manual-annotations",
        json=[
            {
                "start": second_text.index("Zara"),
                "end": len(second_text),
                "label": "NAME",
                "text": "Zara",
            }
        ],
    )
    assert first_manual.status_code == 200
    assert second_manual.status_code == 200

    create_resp = client.post(
        "/api/methods-lab/runs",
        json={
            "name": "overlap-method-run",
            "doc_ids": [first_doc_id, second_doc_id],
            "methods": [{"id": "m1", "label": "Default", "method_id": "default"}],
            "models": [
                {
                    "id": "model_1",
                    "label": "Codex",
                    "model": "openai.gpt-5.3-codex",
                    "reasoning_effort": "none",
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

    final = _wait_for_methods_lab_terminal(client, run_id, attempts=200)
    assert final["status"] == "completed_with_errors"
    cell = final["matrix"]["cells"][0]
    assert cell["failed_docs"] == 1
    assert cell["pending_docs"] == 0
    assert cell["micro"]["f1"] == pytest.approx(0.0)
    overlap = cell["co_primary_metrics"]["overlap"]
    assert overlap["micro"]["f1"] == pytest.approx(1.0)
    assert cell["error_families"]["empty_output_finish_reason_length"] == 1

    detail_resp = client.get(
        f"/api/methods-lab/runs/{run_id}/cells/{cell['id']}/documents/{second_doc_id}"
    )
    assert detail_resp.status_code == 200
    assert detail_resp.json()["error_family"] == "empty_output_finish_reason_length"


def test_methods_lab_accepts_concurrency_above_legacy_limit(client, monkeypatch):
    _write_test_config({"methods_lab_max_concurrency": 16})
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

    create_resp = client.post(
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
            "runtime": {
                "api_key": "request-key",
                "api_base": "https://proxy.example.com/v1",
                "temperature": 0.0,
                "match_mode": "exact",
            },
            "concurrency": 12,
        },
    )
    assert create_resp.status_code == 200
    assert create_resp.json()["concurrency"] == 12

    final = _wait_for_methods_lab_terminal(client, create_resp.json()["id"])
    assert final["status"] == "completed"
    assert final["concurrency"] == 12


def test_methods_lab_run_detail_reports_clamped_worker_diagnostics(client, monkeypatch):
    _write_test_config({"methods_lab_max_concurrency": 16})
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

    create_resp = client.post(
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
            "runtime": {
                "api_key": "request-key",
                "api_base": "https://proxy.example.com/v1",
                "temperature": 0.0,
                "match_mode": "exact",
            },
            "concurrency": 16,
        },
    )
    assert create_resp.status_code == 200

    final = _wait_for_methods_lab_terminal(client, create_resp.json()["id"])
    assert final["status"] == "completed"
    assert final["diagnostics"] == {
        "requested_concurrency": 16,
        "effective_worker_count": 1,
        "max_allowed_concurrency": 16,
        "total_tasks": 1,
        "clamped_by_task_count": True,
        "clamped_by_server_cap": False,
        "api_base_host": "proxy.example.com",
    }


def test_methods_lab_run_detail_reports_effective_worker_diagnostics_for_sixteen_tasks(
    client, monkeypatch
):
    _write_test_config({"methods_lab_max_concurrency": 16})
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

    doc_ids: list[str] = []
    for index in range(4):
        upload_resp = _upload(
            client,
            data=_make_hips_v1_custom(f"Hello Anna {index}.", [{"start": 6, "end": 10, "text": "Anna", "pii_type": "NAME"}]),
            filename=f"methods-{index}.json",
        )
        assert upload_resp.status_code == 200
        doc_id = upload_resp.json()["id"]
        manual_resp = client.put(
            f"/api/documents/{doc_id}/manual-annotations",
            json=[{"start": 6, "end": 10, "label": "NAME", "text": "Anna"}],
        )
        assert manual_resp.status_code == 200
        doc_ids.append(doc_id)

    create_resp = client.post(
        "/api/methods-lab/runs",
        json={
            "doc_ids": doc_ids,
            "methods": [{"id": "m1", "label": "Default", "method_id": "default"}],
            "models": [
                {
                    "id": f"model_{index + 1}",
                    "label": f"Codex {index + 1}",
                    "model": "openai.gpt-5.3-codex",
                    "reasoning_effort": "xhigh",
                    "anthropic_thinking": False,
                    "anthropic_thinking_budget_tokens": None,
                }
                for index in range(4)
            ],
            "runtime": {
                "api_key": "request-key",
                "api_base": "https://proxy.example.com/v1",
                "temperature": 0.0,
                "match_mode": "exact",
            },
            "concurrency": 16,
        },
    )
    assert create_resp.status_code == 200

    final = _wait_for_methods_lab_terminal(client, create_resp.json()["id"])
    assert final["status"] == "completed"
    assert final["diagnostics"] == {
        "requested_concurrency": 16,
        "effective_worker_count": 16,
        "max_allowed_concurrency": 16,
        "total_tasks": 16,
        "clamped_by_task_count": False,
        "clamped_by_server_cap": False,
        "api_base_host": "proxy.example.com",
    }


def test_methods_lab_rejects_concurrency_above_configured_cap(client):
    _write_test_config({"methods_lab_max_concurrency": 16})

    upload_resp = _upload(client)
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    resp = client.post(
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
            "runtime": {
                "api_key": "request-key",
                "temperature": 0.0,
                "match_mode": "exact",
            },
            "concurrency": 17,
        },
    )
    assert resp.status_code == 400
    assert "concurrency must be between 1 and 16" in resp.json()["detail"]


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


def test_methods_lab_marks_docs_without_selected_reference_annotations_unavailable(
    client, monkeypatch
):
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
                "reference_source": "manual",
                "fallback_reference_source": "manual",
            },
            "concurrency": 1,
        },
    )
    assert create_resp.status_code == 200
    run_id = create_resp.json()["id"]

    final = _wait_for_methods_lab_terminal(client, run_id, attempts=200)
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
    assert "reference annotations" in missing_detail["error"].lower()


def test_methods_lab_allows_empty_pre_reference_docs_and_syncs_workspace_runs(
    client, monkeypatch
):
    model_id = "anthropic.claude-4.6-sonnet"
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: [model_id],
    )

    def fake_run_method_with_metadata(**kwargs):
        text = str(kwargs["text"])
        target = "Liam" if "Liam" in text else "Nora"
        start = text.index(target)
        return SimpleNamespace(
            spans=[
                CanonicalSpan(
                    start=start,
                    end=start + len(target),
                    label="NAME",
                    text=target,
                )
            ],
            warnings=[],
            llm_confidence=_mock_confidence_metric(provider="anthropic", model=model_id),
        )

    monkeypatch.setattr("server.run_method_with_metadata", fake_run_method_with_metadata)

    upload_resp = _upload(
        client,
        data=_make_mixed_pre_multi_record_jsonl(),
        filename="mixed-pre.jsonl",
    )
    assert upload_resp.status_code == 200
    folder_id = client.get("/api/folders").json()[0]["id"]
    folder_detail = client.get(f"/api/folders/{folder_id}").json()
    child_doc_ids = [item["id"] for item in folder_detail["documents"]]

    docs_by_pre_count: dict[int, list[str]] = {0: [], 1: []}
    for doc_id in child_doc_ids:
        doc_resp = client.get(f"/api/documents/{doc_id}")
        assert doc_resp.status_code == 200
        docs_by_pre_count[len(doc_resp.json()["pre_annotations"])].append(doc_id)

    assert len(docs_by_pre_count[1]) == 1
    assert len(docs_by_pre_count[0]) == 1
    annotated_doc_id = docs_by_pre_count[1][0]
    empty_pre_doc_id = docs_by_pre_count[0][0]

    create_resp = client.post(
        "/api/methods-lab/runs",
        json={
            "name": "mixed-pre-folder-run",
            "folder_ids": [folder_id],
            "methods": [
                {
                    "id": "method_1",
                    "label": "Regex + LLM Extended v2",
                    "method_id": "presidio-lite+extended-v2",
                }
            ],
            "models": [
                {
                    "id": "model_1",
                    "label": "Claude Sonnet 4.6",
                    "model": model_id,
                    "reasoning_effort": "none",
                    "anthropic_thinking": False,
                    "anthropic_thinking_budget_tokens": None,
                }
            ],
            "runtime": {
                "api_key": "request-key",
                "api_base": "https://proxy.example.com/v1",
                "temperature": 0.0,
                "match_mode": "overlap",
                "reference_source": "pre",
                "fallback_reference_source": "pre",
                "label_profile": "advanced",
                "label_projection": "native",
                "chunk_mode": "off",
                "chunk_size_chars": 10000,
                "method_bundle": "deidentify-v2",
            },
            "concurrency": 1,
        },
    )
    assert create_resp.status_code == 200
    run_id = create_resp.json()["id"]

    final = _wait_for_methods_lab_terminal(client, run_id, attempts=200)
    assert final["status"] == "completed"
    assert final["doc_ids"] == child_doc_ids
    cell = final["matrix"]["cells"][0]
    assert cell["completed_docs"] == 2
    assert cell["failed_docs"] == 0

    empty_detail_resp = client.get(
        f"/api/methods-lab/runs/{run_id}/cells/{cell['id']}/documents/{empty_pre_doc_id}"
    )
    assert empty_detail_resp.status_code == 200
    empty_detail = empty_detail_resp.json()
    assert empty_detail["status"] == "completed"
    assert empty_detail["reference_source_used"] == "pre"
    assert empty_detail["reference_spans"] == []
    assert empty_detail["metrics"]["micro"]["tp"] == 0
    assert empty_detail["metrics"]["micro"]["fp"] == 1
    assert empty_detail["metrics"]["micro"]["fn"] == 0

    annotated_detail_resp = client.get(
        f"/api/methods-lab/runs/{run_id}/cells/{cell['id']}/documents/{annotated_doc_id}"
    )
    assert annotated_detail_resp.status_code == 200
    annotated_detail = annotated_detail_resp.json()
    assert annotated_detail["status"] == "completed"
    assert annotated_detail["metrics"]["micro"]["f1"] == pytest.approx(1.0)

    run_key = f"presidio-lite+extended-v2::{model_id}::{run_id}"
    empty_doc_resp = client.get(f"/api/documents/{empty_pre_doc_id}")
    assert empty_doc_resp.status_code == 200
    empty_doc_payload = empty_doc_resp.json()
    assert "presidio-lite+extended-v2" not in empty_doc_payload["agent_outputs"]["methods"]
    assert empty_doc_payload["agent_outputs"]["methods"][run_key][0]["text"] == "Nora"
    run_meta = empty_doc_payload["agent_outputs"]["method_run_metadata"][run_key]
    assert run_meta["mode"] == "method"
    assert run_meta["method_id"] == "presidio-lite+extended-v2"
    assert run_meta["model"] == model_id
    assert run_meta["label_profile"] == "advanced"

    runs_sidecar = (
        _session_dir("default") / f"{empty_pre_doc_id}.agent.method.runs.json"
    )
    runs_meta_sidecar = (
        _session_dir("default") / f"{empty_pre_doc_id}.agent.method.runs.meta.json"
    )
    assert run_key in json.loads(runs_sidecar.read_text())
    assert run_key in json.loads(runs_meta_sidecar.read_text())


def test_methods_lab_workspace_sync_uses_distinct_run_keys_for_reruns(client, monkeypatch):
    model_id = "google.gemini-3.1-pro-preview"
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: [model_id],
    )

    def fake_run_method_with_metadata(**kwargs):
        text = str(kwargs["text"])
        start = text.index("Anna")
        return SimpleNamespace(
            spans=[CanonicalSpan(start=start, end=start + 4, label="NAME", text="Anna")],
            warnings=[],
            llm_confidence=_mock_confidence_metric(provider="gemini", model=model_id),
        )

    monkeypatch.setattr("server.run_method_with_metadata", fake_run_method_with_metadata)

    upload_resp = _upload(client)
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    run_ids: list[str] = []
    for attempt in range(2):
        create_resp = client.post(
            "/api/methods-lab/runs",
            json={
                "name": f"rerun-{attempt}",
                "doc_ids": [doc_id],
                "methods": [
                    {
                        "id": "method_1",
                        "label": "Regex + LLM Extended v2",
                        "method_id": "presidio-lite+extended-v2",
                    }
                ],
                "models": [
                    {
                        "id": "model_1",
                        "label": "Gemini 3.1 Pro Preview",
                        "model": model_id,
                        "reasoning_effort": "none",
                        "anthropic_thinking": False,
                        "anthropic_thinking_budget_tokens": None,
                    }
                ],
                "runtime": {
                    "api_key": "request-key",
                    "api_base": "https://proxy.example.com/v1",
                    "temperature": 0.0,
                    "match_mode": "overlap",
                    "reference_source": "pre",
                    "fallback_reference_source": "pre",
                    "label_profile": "advanced",
                    "label_projection": "native",
                    "chunk_mode": "off",
                    "chunk_size_chars": 10000,
                    "method_bundle": "deidentify-v2",
                },
                "concurrency": 1,
            },
        )
        assert create_resp.status_code == 200
        run_id = create_resp.json()["id"]
        run_ids.append(run_id)
        final = _wait_for_methods_lab_terminal(client, run_id, attempts=200)
        assert final["status"] == "completed"

    doc_resp = client.get(f"/api/documents/{doc_id}")
    assert doc_resp.status_code == 200
    payload = doc_resp.json()
    run_keys = sorted(
        key
        for key in payload["agent_outputs"]["methods"]
        if key.startswith(f"presidio-lite+extended-v2::{model_id}::")
    )
    assert run_keys == sorted(
        [
            f"presidio-lite+extended-v2::{model_id}::{run_ids[0]}",
            f"presidio-lite+extended-v2::{model_id}::{run_ids[1]}",
        ]
    )

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
    assert new_doc_id == original_doc_id

    runs_resp = client.get("/api/methods-lab/runs")
    assert runs_resp.status_code == 200
    runs = runs_resp.json()["runs"]
    imported_run = next((row for row in runs if row["id"] != original_run_id), None)
    assert imported_run is not None

    detail_resp = client.get(f"/api/methods-lab/runs/{imported_run['id']}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert new_doc_id in detail["doc_ids"]


def test_methods_lab_run_can_be_deleted(client, monkeypatch):
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

    create_resp = client.post(
        "/api/methods-lab/runs",
        json={
            "name": "delete-method-run",
            "doc_ids": [doc_id],
            "methods": [{"id": "m1", "label": "Default", "method_id": "default"}],
            "models": [
                {
                    "id": "model_1",
                    "label": "Codex",
                    "model": "openai.gpt-5.3-codex",
                    "reasoning_effort": "none",
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
    _wait_for_methods_lab_terminal(client, run_id)
    assert _methods_lab_run_path(run_id).exists()

    delete_resp = client.delete(f"/api/methods-lab/runs/{run_id}")
    assert delete_resp.status_code == 200
    assert delete_resp.json() == {"ok": True, "id": run_id}
    assert not _methods_lab_run_path(run_id).exists()

    detail_resp = client.get(f"/api/methods-lab/runs/{run_id}")
    assert detail_resp.status_code == 404

    runs_resp = client.get("/api/methods-lab/runs")
    assert runs_resp.status_code == 200
    assert all(run["id"] != run_id for run in runs_resp.json()["runs"])


def test_methods_lab_run_can_be_cancelled(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
    )

    started = threading.Event()
    release = threading.Event()

    def fake_run_method_with_metadata(**kwargs):
        started.set()
        assert release.wait(timeout=5), "Methods Lab worker did not release in time"
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            warnings=[],
            llm_confidence=_mock_confidence_metric(),
        )

    monkeypatch.setattr("server.run_method_with_metadata", fake_run_method_with_metadata)

    first_upload = _upload(client)
    assert first_upload.status_code == 200
    first_doc_id = first_upload.json()["id"]
    second_upload = _upload(client, filename="second-methods.json")
    assert second_upload.status_code == 200
    second_doc_id = second_upload.json()["id"]
    for doc_id in [first_doc_id, second_doc_id]:
        manual_resp = client.put(
            f"/api/documents/{doc_id}/manual-annotations",
            json=[{"start": 6, "end": 10, "label": "NAME", "text": "Anna"}],
        )
        assert manual_resp.status_code == 200

    create_resp = client.post(
        "/api/methods-lab/runs",
        json={
            "name": "cancel-methods-run",
            "doc_ids": [first_doc_id, second_doc_id],
            "methods": [{"id": "m1", "label": "Default", "method_id": "default"}],
            "models": [
                {
                    "id": "model_1",
                    "label": "Codex",
                    "model": "openai.gpt-5.3-codex",
                    "reasoning_effort": "none",
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
    assert started.wait(timeout=5)

    cancel_resp = client.post(f"/api/methods-lab/runs/{run_id}/cancel")
    assert cancel_resp.status_code == 200
    assert cancel_resp.json() == {"ok": True, "id": run_id, "status": "cancelling"}

    release.set()
    final = _wait_for_methods_lab_terminal(client, run_id)
    assert final["status"] == "cancelled"
    assert final["finished_at"] is not None
    assert final["completed_tasks"] < final["total_tasks"]
    assert any(cell["status"] == "cancelled" for cell in final["matrix"]["cells"])

    first_detail = client.get(
        f"/api/methods-lab/runs/{run_id}/cells/model_1__m1/documents/{first_doc_id}"
    )
    second_detail = client.get(
        f"/api/methods-lab/runs/{run_id}/cells/model_1__m1/documents/{second_doc_id}"
    )
    assert first_detail.status_code == 200
    assert second_detail.status_code == 200
    assert first_detail.json()["status"] in {"completed", "cancelled"}
    assert second_detail.json()["status"] == "cancelled"


def test_methods_lab_timeout_persists_runtime_diagnostics(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["google.gemini-3.1-pro-preview"],
    )

    release = threading.Event()

    def fake_run_method_with_metadata(**kwargs):
        progress_callback = kwargs.get("progress_callback")
        if callable(progress_callback):
            progress_callback(1, "dual:names")
        assert release.wait(timeout=5), "Timed-out worker did not release in time"
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            warnings=[],
            llm_confidence=_mock_confidence_metric(
                provider="gemini",
                model="google.gemini-3.1-pro-preview",
            ),
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
            "name": "timeout-runtime-diagnostics",
            "doc_ids": [doc_id],
            "methods": [{"id": "dual_method", "label": "Dual", "method_id": "dual"}],
            "models": [
                {
                    "id": "gemini_pro",
                    "label": "Gemini Pro",
                    "model": "google.gemini-3.1-pro-preview",
                    "reasoning_effort": "none",
                    "anthropic_thinking": False,
                    "anthropic_thinking_budget_tokens": None,
                }
            ],
            "runtime": {
                "api_key": "request-key",
                "api_base": "https://proxy.example.com/v1",
                "temperature": 0.0,
                "match_mode": "exact",
                "task_timeout_seconds": 0.05,
            },
            "concurrency": 1,
        },
    )
    assert create_resp.status_code == 200
    run_id = create_resp.json()["id"]

    final = _wait_for_methods_lab_terminal(client, run_id, attempts=200)
    release.set()

    assert final["status"] == "completed_with_errors"
    assert final["runtime"]["task_timeout_seconds"] == pytest.approx(0.05)
    cell = final["matrix"]["cells"][0]
    assert cell["failed_docs"] == 1
    assert cell["error_families"]["timeout"] == 1

    detail_resp = client.get(
        f"/api/methods-lab/runs/{run_id}/cells/{cell['id']}/documents/{doc_id}"
    )
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["status"] == "failed"
    assert detail["error_family"] == "timeout"
    assert "timed out" in detail["error"].lower()
    diagnostics = detail["runtime_diagnostics"]
    assert diagnostics["current_chunk_index"] == 1
    assert diagnostics["total_chunks"] == 1
    assert diagnostics["current_pass_index"] == 1
    assert diagnostics["current_pass_label"] == "dual:names"
    assert diagnostics["last_progress_at"] is not None


def test_methods_lab_progress_resets_timeout_window(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["google.gemini-3.1-pro-preview"],
    )

    def fake_run_method_with_metadata(**kwargs):
        progress_callback = kwargs.get("progress_callback")
        if callable(progress_callback):
            progress_callback(1, "dual:names")
        time.sleep(0.1)
        if callable(progress_callback):
            progress_callback(2, "dual:identifiers")
        time.sleep(0.1)
        if callable(progress_callback):
            progress_callback(2, "dual:identifiers")
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            warnings=[],
            llm_confidence=_mock_confidence_metric(
                provider="gemini",
                model="google.gemini-3.1-pro-preview",
            ),
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
            "name": "timeout-progress-window",
            "doc_ids": [doc_id],
            "methods": [{"id": "dual_method", "label": "Dual", "method_id": "dual"}],
            "models": [
                {
                    "id": "gemini_pro",
                    "label": "Gemini Pro",
                    "model": "google.gemini-3.1-pro-preview",
                    "reasoning_effort": "none",
                    "anthropic_thinking": False,
                    "anthropic_thinking_budget_tokens": None,
                }
            ],
                "runtime": {
                    "api_key": "request-key",
                    "api_base": "https://proxy.example.com/v1",
                    "temperature": 0.0,
                    "match_mode": "exact",
                    "task_timeout_seconds": 0.18,
                },
                "concurrency": 1,
            },
    )
    assert create_resp.status_code == 200
    run_id = create_resp.json()["id"]

    final = _wait_for_methods_lab_terminal(client, run_id, attempts=200)

    assert final["status"] == "completed"
    cell = final["matrix"]["cells"][0]
    assert cell["failed_docs"] == 0
    assert cell["completed_docs"] == 1

    detail_resp = client.get(
        f"/api/methods-lab/runs/{run_id}/cells/{cell['id']}/documents/{doc_id}"
    )
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["status"] == "completed"
    diagnostics = detail["runtime_diagnostics"]
    assert diagnostics["current_pass_label"] == "dual:identifiers"
    assert diagnostics["last_progress_at"] is not None


def test_methods_lab_detail_persists_response_debug(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
    )

    def fake_run_method_with_metadata(**kwargs):
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            warnings=[],
            llm_confidence=_mock_confidence_metric(),
            response_debug=[
                "Pass 1: resp_type=ModelResponse; finish_reason=stop",
                "Pass 2: resp_type=ModelResponse; finish_reason=stop",
            ],
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
            "name": "methods-response-debug-check",
            "doc_ids": [doc_id],
            "methods": [{"id": "m1", "label": "Default", "method_id": "default"}],
            "models": [
                {
                    "id": "model_1",
                    "label": "Codex",
                    "model": "openai.gpt-5.3-codex",
                    "reasoning_effort": "none",
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
    assert final["status"] == "completed"
    cell_id = final["matrix"]["cells"][0]["id"]

    stored = json.loads(_methods_lab_run_path(run_id).read_text())
    assert stored["cells"][cell_id]["documents"][doc_id]["response_debug"] == [
        "Pass 1: resp_type=ModelResponse; finish_reason=stop",
        "Pass 2: resp_type=ModelResponse; finish_reason=stop",
    ]

    detail_resp = client.get(
        f"/api/methods-lab/runs/{run_id}/cells/{cell_id}/documents/{doc_id}"
    )
    assert detail_resp.status_code == 200
    assert detail_resp.json()["response_debug"] == [
        "Pass 1: resp_type=ModelResponse; finish_reason=stop",
        "Pass 2: resp_type=ModelResponse; finish_reason=stop",
    ]


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
    llm_runs = payload["agent_outputs"]["llm_runs"]
    llm_run_metadata = payload["agent_outputs"]["llm_run_metadata"]
    assert len(llm_runs) == 2
    assert set(llm_run_metadata.keys()) == set(llm_runs.keys())
    run_keys_by_model = {
        meta["model"]: run_key for run_key, meta in llm_run_metadata.items()
    }
    assert set(run_keys_by_model.keys()) == supported_models
    run_key_a = run_keys_by_model["openai.gpt-5.2-chat"]
    run_key_b = run_keys_by_model["openai.gpt-5.3-codex"]
    assert llm_runs[run_key_a][0]["text"] == "Anna"
    assert llm_runs[run_key_b][0]["text"] == "Hello"
    assert llm_run_metadata[run_key_a]["mode"] == "llm"
    assert llm_run_metadata[run_key_a]["model"] == "openai.gpt-5.2-chat"
    llm_prompt_snapshot = llm_run_metadata[run_key_a]["prompt_snapshot"]
    assert llm_prompt_snapshot["format_guardrail_appended"] is True
    assert "requested_system_prompt" in llm_prompt_snapshot
    assert "effective_system_prompt" in llm_prompt_snapshot
    assert "updated_at" in llm_run_metadata[run_key_a]

    metrics_resp = client.get(
        f"/api/documents/{doc_id}/metrics",
        params={
            "reference": "pre",
            "hypothesis": f"agent.llm_run.{run_key_a}",
            "match_mode": "exact",
        },
    )
    assert metrics_resp.status_code == 200


def test_agent_llm_persists_multiple_runs_for_same_model(client, monkeypatch):
    model = "openai.gpt-5.3-codex"
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: [model],
    )

    def fake_run_llm_with_metadata(**kwargs):
        prompt = str(kwargs.get("system_prompt"))
        spans = (
            [CanonicalSpan(start=6, end=10, label="NAME", text="Anna")]
            if "baseline" in prompt.lower()
            else [CanonicalSpan(start=17, end=20, label="NAME", text="Sue")]
        )
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
            "model": model,
            "system_prompt": "Baseline prompt",
            "api_key": "request-key",
            "api_base": "https://proxy.example.com/v1",
        },
    )
    assert run_a.status_code == 200
    run_b = client.post(
        f"/api/documents/{doc_id}/agent",
        json={
            "mode": "llm",
            "model": model,
            "system_prompt": "Annotator prompt",
            "api_key": "request-key",
            "api_base": "https://proxy.example.com/v1",
        },
    )
    assert run_b.status_code == 200

    doc_resp = client.get(f"/api/documents/{doc_id}")
    assert doc_resp.status_code == 200
    payload = doc_resp.json()
    assert payload["agent_outputs"]["llm"][0]["text"] == "Sue"
    llm_runs = payload["agent_outputs"]["llm_runs"]
    llm_run_metadata = payload["agent_outputs"]["llm_run_metadata"]
    assert len(llm_runs) == 2
    assert set(llm_runs.keys()) == set(llm_run_metadata.keys())
    assert all(meta["model"] == model for meta in llm_run_metadata.values())
    requested_prompts = {
        meta["prompt_snapshot"]["requested_system_prompt"]
        for meta in llm_run_metadata.values()
    }
    assert requested_prompts == {"Baseline prompt", "Annotator prompt"}

    for run_key in llm_runs:
        metrics_resp = client.get(
            f"/api/documents/{doc_id}/metrics",
            params={
                "reference": "pre",
                "hypothesis": f"agent.llm_run.{run_key}",
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
    saved_meta = next(iter(reloaded["agent_outputs"]["llm_run_metadata"].values()))
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

    (
        spans,
        warnings,
        _llm_confidence,
        diagnostics,
        _raw_hypothesis_spans,
        _resolution_events,
        _resolution_policy_version,
        _response_debug,
    ) = _run_llm_for_document(
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


def test_agent_method_default_chunk_mode_stays_single_pass_for_long_text(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.2-chat"],
    )
    call_count = {"count": 0}

    def fake_run_method_with_metadata(**kwargs):
        call_count["count"] += 1
        return SimpleNamespace(
            spans=[],
            warnings=[],
            llm_confidence=_mock_confidence_metric(confidence=0.8, band="medium"),
        )

    monkeypatch.setattr("server.run_method_with_metadata", fake_run_method_with_metadata)
    long_text = "Hello Anna.\n" * 1500
    resp = _upload(client, _make_hips_v1_custom(long_text))
    doc_id = resp.json()["id"]

    run_resp = client.post(
        f"/api/documents/{doc_id}/agent",
        json={
            "mode": "method",
            "method_id": "dual",
            "model": "openai.gpt-5.2-chat",
            "api_key": "request-key",
            "api_base": "https://proxy.example.com/v1",
        },
    )
    assert run_resp.status_code == 200
    payload = run_resp.json()
    assert call_count["count"] == 1
    assert all(
        "Chunked method run used" not in warning for warning in payload["agent_run_warnings"]
    )


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


def test_export_ground_truth_zip_can_scope_to_recursive_folder_docs(client):
    top_level_resp = _upload(client, _make_hips_v1_custom("Top level Anna"), filename="top.json")
    assert top_level_resp.status_code == 200
    top_level_id = top_level_resp.json()["id"]

    parent_resp = _upload(client, _make_hips_v1_custom("Parent Liam"), filename="parent.json")
    assert parent_resp.status_code == 200
    parent_doc_id = parent_resp.json()["id"]

    child_resp = _upload(client, _make_hips_v1_custom("Child Ava"), filename="child.json")
    assert child_resp.status_code == 200
    child_doc_id = child_resp.json()["id"]

    for doc_id, text in (
        (top_level_id, "Anna"),
        (parent_doc_id, "Liam"),
        (child_doc_id, "Ava"),
    ):
        manual_resp = client.put(
            f"/api/documents/{doc_id}/manual-annotations",
            json=[{"start": 0, "end": len(text), "label": "NAME", "text": text}],
        )
        assert manual_resp.status_code == 200

    folder_dir = _session_dir() / "folders"
    folder_dir.mkdir(parents=True, exist_ok=True)
    parent_folder = FolderRecord(
        id="folder-parent",
        name="TIMSS",
        kind="manual",
        parent_folder_id=None,
        doc_ids=[parent_doc_id],
        child_folder_ids=["folder-child"],
        created_at="2026-03-17T00:00:00Z",
        doc_display_names={parent_doc_id: "Parent transcript"},
    )
    child_folder = FolderRecord(
        id="folder-child",
        name="AU",
        kind="manual",
        parent_folder_id="folder-parent",
        doc_ids=[child_doc_id],
        child_folder_ids=[],
        created_at="2026-03-17T00:00:01Z",
        doc_display_names={child_doc_id: "Child transcript"},
    )
    (folder_dir / "_index.json").write_text(json.dumps([parent_folder.id, child_folder.id]))
    (folder_dir / f"{parent_folder.id}.json").write_text(parent_folder.model_dump_json(indent=2))
    (folder_dir / f"{child_folder.id}.json").write_text(child_folder.model_dump_json(indent=2))

    export_resp = client.get(
        "/api/session/export-ground-truth",
        params={"source": "manual", "scope": "folder", "folder_id": parent_folder.id},
    )
    assert export_resp.status_code == 200

    with zipfile.ZipFile(BytesIO(export_resp.content), mode="r") as archive:
        names = sorted(archive.namelist())
        assert len(names) == 2
        payloads = [
            json.loads(archive.read(name).decode("utf-8"))
            for name in names
        ]
        assert {payload["id"] for payload in payloads} == {parent_doc_id, child_doc_id}
        assert {payload["transcript"] for payload in payloads} == {"Parent Liam", "Child Ava"}
        assert all(payload["ground_truth_source"] == "manual" for payload in payloads)
        assert top_level_id not in {payload["id"] for payload in payloads}


def test_export_ground_truth_zip_requires_folder_id_for_folder_scope(client):
    export_resp = client.get(
        "/api/session/export-ground-truth",
        params={"source": "manual", "scope": "folder"},
    )

    assert export_resp.status_code == 400
    assert export_resp.json() == {"detail": "folder_id is required when scope=folder"}


def test_export_ground_truth_zip_returns_404_for_unknown_folder_scope(client):
    export_resp = client.get(
        "/api/session/export-ground-truth",
        params={"source": "manual", "scope": "folder", "folder_id": "missing-folder"},
    )

    assert export_resp.status_code == 404
    assert export_resp.json() == {"detail": "Folder not found"}


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


def test_session_import_accepts_direct_ground_truth_json_payload(client):
    transcript = "Hello Anna, call Sue please."
    ground_truth_payload = {
        "id": "gt-doc-1",
        "filename": "shared.ground_truth.json",
        "transcript": transcript,
        "ground_truth_source": "manual",
        "spans": [{"start": 6, "end": 10, "label": "NAME", "text": "Anna"}],
        "pii_occurrences": [{"start": 6, "end": 10, "pii_type": "NAME", "text": "Anna"}],
    }

    import_resp = client.post(
        "/api/session/import",
        files={
            "file": (
                "shared.ground_truth.json",
                json.dumps(ground_truth_payload).encode(),
                "application/json",
            )
        },
    )
    assert import_resp.status_code == 200
    imported = import_resp.json()
    assert imported["bundle_version"] is None
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


def test_ground_truth_import_replace_conflict_updates_manual_and_preserves_pre_annotations(client):
    transcript = "Hello Anna, call Sue please."
    upload_resp = _upload(
        client,
        _make_hips_v1_custom(
            transcript,
            pii_occurrences=[{"start": 6, "end": 10, "text": "Anna", "pii_type": "NAME"}],
        ),
        filename="shared.json",
    )
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    manual_resp = client.put(
        f"/api/documents/{doc_id}/manual-annotations",
        json=[{"start": 0, "end": 5, "label": "NAME", "text": "Hello"}],
    )
    assert manual_resp.status_code == 200

    ground_truth_payload = {
        "id": "gt-doc-1",
        "filename": "shared.json",
        "transcript": transcript,
        "ground_truth_source": "manual",
        "spans": [{"start": 6, "end": 10, "label": "NAME", "text": "Anna"}],
        "pii_occurrences": [{"start": 6, "end": 10, "pii_type": "NAME", "text": "Anna"}],
    }

    import_resp = client.post(
        "/api/session/import",
        data={"conflict_policy": "replace"},
        files={
            "file": (
                "shared.ground_truth.json",
                json.dumps(ground_truth_payload).encode(),
                "application/json",
            )
        },
    )
    assert import_resp.status_code == 200
    imported = import_resp.json()
    assert imported["imported_ids"] == [doc_id]
    assert imported["created_count"] == 0
    assert imported["replaced_count"] == 1

    doc_resp = client.get(f"/api/documents/{doc_id}")
    assert doc_resp.status_code == 200
    imported_doc = doc_resp.json()
    assert imported_doc["pre_annotations"] == [
        {"start": 6, "end": 10, "label": "NAME", "text": "Anna"}
    ]
    assert imported_doc["manual_annotations"] == [
        {"start": 6, "end": 10, "label": "NAME", "text": "Anna"}
    ]


def test_ground_truth_import_keep_current_conflict_preserves_existing_manual_annotations(client):
    transcript = "Hello Anna, call Sue please."
    upload_resp = _upload(client, _make_hips_v1_custom(transcript), filename="shared.json")
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    manual_resp = client.put(
        f"/api/documents/{doc_id}/manual-annotations",
        json=[{"start": 0, "end": 5, "label": "NAME", "text": "Hello"}],
    )
    assert manual_resp.status_code == 200

    ground_truth_payload = {
        "id": "gt-doc-1",
        "filename": "shared.json",
        "transcript": transcript,
        "ground_truth_source": "manual",
        "spans": [{"start": 6, "end": 10, "label": "NAME", "text": "Anna"}],
        "pii_occurrences": [{"start": 6, "end": 10, "pii_type": "NAME", "text": "Anna"}],
    }

    import_resp = client.post(
        "/api/session/import",
        data={"conflict_policy": "keep_current"},
        files={
            "file": (
                "shared.ground_truth.json",
                json.dumps(ground_truth_payload).encode(),
                "application/json",
            )
        },
    )
    assert import_resp.status_code == 200
    imported = import_resp.json()
    assert imported["imported_ids"] == [doc_id]
    assert imported["created_count"] == 0
    assert imported["kept_current_count"] == 1

    doc_resp = client.get(f"/api/documents/{doc_id}")
    assert doc_resp.status_code == 200
    imported_doc = doc_resp.json()
    assert imported_doc["manual_annotations"] == [
        {"start": 0, "end": 5, "label": "NAME", "text": "Hello"}
    ]


def test_ground_truth_import_add_new_conflict_creates_duplicate_document(client):
    transcript = "Hello Anna, call Sue please."
    upload_resp = _upload(client, _make_hips_v1_custom(transcript), filename="shared.json")
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    manual_resp = client.put(
        f"/api/documents/{doc_id}/manual-annotations",
        json=[{"start": 0, "end": 5, "label": "NAME", "text": "Hello"}],
    )
    assert manual_resp.status_code == 200

    ground_truth_payload = {
        "id": doc_id,
        "filename": "shared.json",
        "transcript": transcript,
        "ground_truth_source": "manual",
        "spans": [{"start": 6, "end": 10, "label": "NAME", "text": "Anna"}],
        "pii_occurrences": [{"start": 6, "end": 10, "pii_type": "NAME", "text": "Anna"}],
    }

    import_resp = client.post(
        "/api/session/import",
        data={"conflict_policy": "add_new"},
        files={
            "file": (
                "shared.ground_truth.json",
                json.dumps(ground_truth_payload).encode(),
                "application/json",
            )
        },
    )
    assert import_resp.status_code == 200
    imported = import_resp.json()
    assert imported["created_count"] == 1
    assert imported["added_as_new_count"] == 1
    imported_id = imported["imported_ids"][0]
    assert imported_id != doc_id

    docs_resp = client.get("/api/documents")
    assert docs_resp.status_code == 200
    assert len(docs_resp.json()) == 2

    original_doc = client.get(f"/api/documents/{doc_id}").json()
    imported_doc = client.get(f"/api/documents/{imported_id}").json()
    assert original_doc["manual_annotations"] == [
        {"start": 0, "end": 5, "label": "NAME", "text": "Hello"}
    ]
    assert imported_doc["manual_annotations"] == [
        {"start": 6, "end": 10, "label": "NAME", "text": "Anna"}
    ]


def test_session_import_accepts_exported_ground_truth_json_member(client):
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

    with zipfile.ZipFile(BytesIO(export_resp.content), mode="r") as archive:
        names = archive.namelist()
        assert len(names) == 1
        member_name = names[0]
        member_raw = archive.read(member_name)

    delete_resp = client.delete(f"/api/documents/{doc_id}")
    assert delete_resp.status_code == 200

    import_resp = client.post(
        "/api/session/import",
        files={"file": (member_name, member_raw, "application/json")},
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


def test_session_ingest_routes_raw_upload_payloads_to_document_upload(client):
    ingest_resp = client.post(
        "/api/session/ingest",
        files={"file": ("transcript.json", _make_hips_v1(), "application/json")},
    )
    assert ingest_resp.status_code == 200
    payload = ingest_resp.json()
    assert payload["mode"] == "upload"
    assert payload["uploaded_count"] == 1
    assert payload["imported_count"] == 0
    assert payload["created_count"] == 1
    assert len(payload["created_ids"]) == 1
    created_id = payload["created_ids"][0]

    doc_resp = client.get(f"/api/documents/{created_id}")
    assert doc_resp.status_code == 200
    assert doc_resp.json()["raw_text"] == "Hello Anna, call Sue please."


def test_session_ingest_routes_timss_txt_payloads_to_document_upload(client):
    timss_raw = (
        "Top of Form\r"
        "00:00:04\tSN\tPlease teach us well.\r"
        "00:00:06\tT\tOkay.\r"
        "00:00:08\tSN\tJos\xe9 is ready.\r"
        "\xca\r"
        "Bottom of Form\r"
    ).encode("latin-1")

    ingest_resp = client.post(
        "/api/session/ingest",
        files={"file": ("Science JP1 transcript.txt", timss_raw, "text/plain")},
    )

    assert ingest_resp.status_code == 200
    payload = ingest_resp.json()
    assert payload["mode"] == "upload"
    assert payload["uploaded_count"] == 1
    assert payload["imported_count"] == 0
    assert payload["created_count"] == 1
    assert len(payload["created_ids"]) == 1

    created_id = payload["created_ids"][0]
    doc_resp = client.get(f"/api/documents/{created_id}")
    assert doc_resp.status_code == 200
    doc = doc_resp.json()
    assert doc["filename"] == "Science JP1 transcript.txt"
    assert doc["raw_text"] == "SN: Please teach us well.\nT: Okay.\nSN: José is ready."
    assert doc["pre_annotations"] == []


def test_session_ingest_routes_ground_truth_zip_to_import_processing(client):
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

    delete_resp = client.delete(f"/api/documents/{doc_id}")
    assert delete_resp.status_code == 200

    ingest_resp = client.post(
        "/api/session/ingest",
        files={"file": ("ground-truth-manual.zip", export_resp.content, "application/zip")},
    )
    assert ingest_resp.status_code == 200
    payload = ingest_resp.json()
    assert payload["mode"] == "import"
    assert payload["uploaded_count"] == 0
    assert payload["imported_count"] == 1


def test_session_ingest_forwards_import_conflict_policy(client):
    transcript = "Hello Anna, call Sue please."
    upload_resp = _upload(client, _make_hips_v1_custom(transcript), filename="shared.json")
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    ingest_resp = client.post(
        "/api/session/ingest",
        data={"conflict_policy": "add_new"},
        files={
            "file": (
                "shared.ground_truth.json",
                json.dumps(
                    {
                        "id": doc_id,
                        "filename": "shared.json",
                        "transcript": transcript,
                        "ground_truth_source": "manual",
                        "spans": [
                            {"start": 6, "end": 10, "label": "NAME", "text": "Anna"},
                        ],
                        "pii_occurrences": [
                            {"start": 6, "end": 10, "pii_type": "NAME", "text": "Anna"},
                        ],
                    }
                ).encode(),
                "application/json",
            )
        },
    )
    assert ingest_resp.status_code == 200
    payload = ingest_resp.json()
    assert payload["mode"] == "import"
    assert payload["created_count"] == 1
    assert payload["added_as_new_count"] == 1
    assert payload["imported_count"] == 1

    docs_resp = client.get("/api/documents")
    assert docs_resp.status_code == 200
    assert len(docs_resp.json()) == 2
    assert payload["created_count"] == 1
    assert len(payload["created_ids"]) == 1


def test_session_ingest_routes_exported_bundle_to_import_processing(client):
    upload_resp = _upload(client)
    assert upload_resp.status_code == 200
    original_doc_id = upload_resp.json()["id"]

    export_resp = client.get("/api/session/export")
    assert export_resp.status_code == 200

    delete_resp = client.delete(f"/api/documents/{original_doc_id}")
    assert delete_resp.status_code == 200

    ingest_resp = client.post(
        "/api/session/ingest",
        files={
            "file": (
                "session-bundle.json",
                json.dumps(export_resp.json()).encode(),
                "application/json",
            )
        },
    )
    assert ingest_resp.status_code == 200
    payload = ingest_resp.json()
    assert payload["mode"] == "import"
    assert payload["uploaded_count"] == 0
    assert payload["imported_count"] == 1
    assert payload["created_count"] == 1
    assert len(payload["created_ids"]) == 1


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
    assert len(saved["llm_run_metadata"]) == 1
    saved_meta = next(iter(saved["llm_run_metadata"].values()))
    assert saved_meta["mode"] == "llm"
    assert "prompt_snapshot" in saved_meta


def test_agent_run_persists_raw_and_resolved_spans_with_resolution_metadata(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.2-chat"],
    )

    text = "Hello Mr. Muhammad"

    def fake_run_llm_with_metadata(**kwargs):
        return SimpleNamespace(
            spans=[
                CanonicalSpan(
                    start=text.index("Mr."),
                    end=len(text),
                    label="NAME",
                    text="Mr. Muhammad",
                )
            ],
            raw_spans=[
                CanonicalSpan(
                    start=text.index("Muhammad"),
                    end=len(text),
                    label="NAME",
                    text="Muhammad",
                )
            ],
            warnings=[],
            llm_confidence=_mock_confidence_metric(model=str(kwargs["model"])),
            resolution_events=[
                ResolutionEvent(
                    kind="boundary_resolution",
                    label="NAME",
                    rule="name_honorific_prefix",
                    before=CanonicalSpan(
                        start=text.index("Muhammad"),
                        end=len(text),
                        label="NAME",
                        text="Muhammad",
                    ),
                    after=CanonicalSpan(
                        start=text.index("Mr."),
                        end=len(text),
                        label="NAME",
                        text="Mr. Muhammad",
                    ),
                )
            ],
            resolution_policy_version="2026-03-span-resolution-v2",
        )

    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)

    upload_resp = _upload(client, _make_hips_v1_custom(text))
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
    payload = run_resp.json()
    assert payload["agent_run_metrics"]["resolution_policy_version"] == "2026-03-span-resolution-v2"
    assert payload["agent_run_metrics"]["raw_hypothesis_spans"][0]["text"] == "Muhammad"
    assert payload["agent_run_metrics"]["resolution_events"][0]["rule"] == "name_honorific_prefix"

    export_resp = client.get("/api/session/export")
    assert export_resp.status_code == 200
    saved_meta = next(
        iter(export_resp.json()["documents"][0]["agent_saved_outputs"]["llm_run_metadata"].values())
    )
    assert saved_meta["raw_hypothesis_spans"][0]["text"] == "Muhammad"
    assert saved_meta["resolution_events"][0]["rule"] == "name_honorific_prefix"
    assert saved_meta["resolution_policy_version"] == "2026-03-span-resolution-v2"


def test_prompt_lab_detail_includes_raw_metrics_and_resolution_metadata(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
    )

    text = "Hello Mr. Muhammad"

    def fake_run_llm_with_metadata(**kwargs):
        return SimpleNamespace(
            spans=[
                CanonicalSpan(
                    start=text.index("Mr."),
                    end=len(text),
                    label="NAME",
                    text="Mr. Muhammad",
                )
            ],
            raw_spans=[
                CanonicalSpan(
                    start=text.index("Muhammad"),
                    end=len(text),
                    label="NAME",
                    text="Muhammad",
                )
            ],
            warnings=[],
            llm_confidence=_mock_confidence_metric(model=str(kwargs["model"])),
            resolution_events=[
                ResolutionEvent(
                    kind="boundary_resolution",
                    label="NAME",
                    rule="name_honorific_prefix",
                    before=CanonicalSpan(
                        start=text.index("Muhammad"),
                        end=len(text),
                        label="NAME",
                        text="Muhammad",
                    ),
                    after=CanonicalSpan(
                        start=text.index("Mr."),
                        end=len(text),
                        label="NAME",
                        text="Mr. Muhammad",
                    ),
                )
            ],
            resolution_policy_version="2026-03-span-resolution-v2",
        )

    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)

    upload_resp = _upload(client, _make_hips_v1_custom(text))
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]
    manual_resp = client.put(
        f"/api/documents/{doc_id}/manual-annotations",
        json=[
            {
                "start": text.index("Mr."),
                "end": len(text),
                "label": "NAME",
                "text": "Mr. Muhammad",
            }
        ],
    )
    assert manual_resp.status_code == 200

    create_resp = client.post(
        "/api/prompt-lab/runs",
        json={
            "name": "resolution-audit",
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
                    "reasoning_effort": "none",
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
    detail_resp = client.get(
        f"/api/prompt-lab/runs/{run_id}/cells/{final['matrix']['cells'][0]['id']}/documents/{doc_id}"
    )
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["raw_hypothesis_spans"][0]["text"] == "Muhammad"
    assert detail["resolution_events"][0]["rule"] == "name_honorific_prefix"
    assert detail["resolution_policy_version"] == "2026-03-span-resolution-v2"
    assert detail["raw_metrics"]["micro"]["f1"] == pytest.approx(0.0)
    assert detail["metrics"]["micro"]["f1"] == pytest.approx(1.0)


def test_methods_lab_accepts_legacy_method_bundle(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
    )
    seen_method_bundles: list[str] = []

    def fake_run_method_with_metadata(**kwargs):
        seen_method_bundles.append(str(kwargs.get("method_bundle")))
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            raw_spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            warnings=[],
            llm_confidence=None,
            response_debug=[],
            resolution_events=[],
            resolution_policy_version=None,
        )

    monkeypatch.setattr("server.run_method_with_metadata", fake_run_method_with_metadata)

    upload_resp = _upload(client)
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    create_resp = client.post(
        "/api/methods-lab/runs",
        json={
            "name": "methods-legacy-run",
            "doc_ids": [doc_id],
            "methods": [{"id": "method_1", "label": "Default", "method_id": "default"}],
            "models": [
                {
                    "id": "m1",
                    "label": "Codex",
                    "model": "openai.gpt-5.3-codex",
                    "reasoning_effort": "none",
                    "anthropic_thinking": False,
                    "anthropic_thinking_budget_tokens": None,
                }
            ],
            "runtime": {
                "api_key": "request-key",
                "api_base": "https://proxy.example.com/v1",
                "temperature": 0.0,
                "match_mode": "exact",
                "reference_source": "pre",
                "fallback_reference_source": "pre",
                "method_bundle": "legacy",
            },
            "concurrency": 1,
        },
    )
    assert create_resp.status_code == 200
    run_id = create_resp.json()["id"]
    assert create_resp.json()["method_bundle"] == "legacy"

    final = _wait_for_methods_lab_terminal(client, run_id)
    assert final["status"] == "completed"
    assert final["method_bundle"] == "legacy"
    assert final["runtime"]["method_bundle"] == "legacy"
    runs_resp = client.get("/api/methods-lab/runs")
    assert runs_resp.status_code == 200
    runs = runs_resp.json()["runs"]
    listed = next((row for row in runs if row["id"] == run_id), None)
    assert listed is not None
    assert listed["method_bundle"] == "legacy"
    assert seen_method_bundles == ["legacy"]


def test_methods_lab_accepts_test_method_bundle(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["google.gemini-3.1-pro-preview"],
    )
    seen_method_bundles: list[str] = []

    def fake_run_method_with_metadata(**kwargs):
        seen_method_bundles.append(str(kwargs.get("method_bundle")))
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            raw_spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            warnings=[],
            llm_confidence=None,
            response_debug=[],
            resolution_events=[],
            resolution_policy_version=None,
        )

    monkeypatch.setattr("server.run_method_with_metadata", fake_run_method_with_metadata)

    upload_resp = _upload(client)
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    create_resp = client.post(
        "/api/methods-lab/runs",
        json={
            "name": "methods-test-run",
            "doc_ids": [doc_id],
            "methods": [{"id": "method_1", "label": "Default", "method_id": "default"}],
            "models": [
                {
                    "id": "m1",
                    "label": "Gemini Pro",
                    "model": "google.gemini-3.1-pro-preview",
                    "reasoning_effort": "none",
                    "anthropic_thinking": False,
                    "anthropic_thinking_budget_tokens": None,
                }
            ],
            "runtime": {
                "api_key": "request-key",
                "api_base": "https://proxy.example.com/v1",
                "temperature": 0.0,
                "match_mode": "exact",
                "reference_source": "pre",
                "fallback_reference_source": "pre",
                "method_bundle": "test",
            },
            "concurrency": 1,
        },
    )
    assert create_resp.status_code == 200
    run_id = create_resp.json()["id"]
    assert create_resp.json()["method_bundle"] == "test"

    final = _wait_for_methods_lab_terminal(client, run_id)
    assert final["status"] == "completed"
    assert final["method_bundle"] == "test"
    assert final["runtime"]["method_bundle"] == "test"
    runs_resp = client.get("/api/methods-lab/runs")
    assert runs_resp.status_code == 200
    runs = runs_resp.json()["runs"]
    listed = next((row for row in runs if row["id"] == run_id), None)
    assert listed is not None
    assert listed["method_bundle"] == "test"
    assert seen_method_bundles == ["test"]


def test_methods_lab_accepts_v2_post_process_method_bundle(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["google.gemini-3.1-pro-preview"],
    )
    seen_method_bundles: list[str] = []

    def fake_run_method_with_metadata(**kwargs):
        seen_method_bundles.append(str(kwargs.get("method_bundle")))
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            raw_spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            warnings=[],
            llm_confidence=None,
            response_debug=[],
            resolution_events=[],
            resolution_policy_version=None,
        )

    monkeypatch.setattr("server.run_method_with_metadata", fake_run_method_with_metadata)

    upload_resp = _upload(client)
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    create_resp = client.post(
        "/api/methods-lab/runs",
        json={
            "name": "methods-v2-post-process-run",
            "doc_ids": [doc_id],
            "methods": [{"id": "method_1", "label": "Default", "method_id": "default"}],
            "models": [
                {
                    "id": "m1",
                    "label": "Gemini Pro",
                    "model": "google.gemini-3.1-pro-preview",
                    "reasoning_effort": "none",
                    "anthropic_thinking": False,
                    "anthropic_thinking_budget_tokens": None,
                }
            ],
            "runtime": {
                "api_key": "request-key",
                "api_base": "https://proxy.example.com/v1",
                "temperature": 0.0,
                "match_mode": "exact",
                "reference_source": "pre",
                "fallback_reference_source": "pre",
                "method_bundle": "v2+post-process",
            },
            "concurrency": 1,
        },
    )
    assert create_resp.status_code == 200
    run_id = create_resp.json()["id"]
    assert create_resp.json()["method_bundle"] == "v2+post-process"

    final = _wait_for_methods_lab_terminal(client, run_id)
    assert final["status"] == "completed"
    assert final["method_bundle"] == "v2+post-process"
    assert final["runtime"]["method_bundle"] == "v2+post-process"
    runs_resp = client.get("/api/methods-lab/runs")
    assert runs_resp.status_code == 200
    runs = runs_resp.json()["runs"]
    listed = next((row for row in runs if row["id"] == run_id), None)
    assert listed is not None
    assert listed["method_bundle"] == "v2+post-process"
    assert seen_method_bundles == ["v2+post-process"]


def test_methods_lab_accepts_v2_method_bundle(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["google.gemini-3.1-pro-preview"],
    )
    seen_method_bundles: list[str] = []

    def fake_run_method_with_metadata(**kwargs):
        seen_method_bundles.append(str(kwargs.get("method_bundle")))
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            raw_spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            warnings=[],
            llm_confidence=None,
            response_debug=[],
            resolution_events=[],
            resolution_policy_version=None,
        )

    monkeypatch.setattr("server.run_method_with_metadata", fake_run_method_with_metadata)

    upload_resp = _upload(client)
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    create_resp = client.post(
        "/api/methods-lab/runs",
        json={
            "name": "methods-v2-run",
            "doc_ids": [doc_id],
            "methods": [{"id": "method_1", "label": "Default", "method_id": "default"}],
            "models": [
                {
                    "id": "m1",
                    "label": "Gemini Pro",
                    "model": "google.gemini-3.1-pro-preview",
                    "reasoning_effort": "none",
                    "anthropic_thinking": False,
                    "anthropic_thinking_budget_tokens": None,
                }
            ],
            "runtime": {
                "api_key": "request-key",
                "api_base": "https://proxy.example.com/v1",
                "temperature": 0.0,
                "match_mode": "exact",
                "reference_source": "pre",
                "fallback_reference_source": "pre",
                "method_bundle": "v2",
            },
            "concurrency": 1,
        },
    )
    assert create_resp.status_code == 200
    run_id = create_resp.json()["id"]
    assert create_resp.json()["method_bundle"] == "v2"

    final = _wait_for_methods_lab_terminal(client, run_id)
    assert final["status"] == "completed"
    assert final["method_bundle"] == "v2"
    assert final["runtime"]["method_bundle"] == "v2"
    runs_resp = client.get("/api/methods-lab/runs")
    assert runs_resp.status_code == 200
    runs = runs_resp.json()["runs"]
    listed = next((row for row in runs if row["id"] == run_id), None)
    assert listed is not None
    assert listed["method_bundle"] == "v2"
    assert seen_method_bundles == ["v2"]


def test_methods_lab_accepts_deidentify_v2_method_bundle(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["google.gemini-3.1-pro-preview"],
    )
    seen_method_bundles: list[str] = []

    def fake_run_method_with_metadata(**kwargs):
        seen_method_bundles.append(str(kwargs.get("method_bundle")))
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            raw_spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            warnings=[],
            llm_confidence=None,
            response_debug=[],
            resolution_events=[],
            resolution_policy_version=None,
        )

    monkeypatch.setattr("server.run_method_with_metadata", fake_run_method_with_metadata)

    upload_resp = _upload(client)
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    create_resp = client.post(
        "/api/methods-lab/runs",
        json={
            "name": "methods-deidentify-v2-run",
            "doc_ids": [doc_id],
            "methods": [{"id": "method_1", "label": "Dual V2", "method_id": "dual-v2"}],
            "models": [
                {
                    "id": "m1",
                    "label": "Gemini Pro",
                    "model": "google.gemini-3.1-pro-preview",
                    "reasoning_effort": "none",
                    "anthropic_thinking": False,
                    "anthropic_thinking_budget_tokens": None,
                }
            ],
            "runtime": {
                "api_key": "request-key",
                "api_base": "https://proxy.example.com/v1",
                "temperature": 0.0,
                "match_mode": "substring",
                "reference_source": "pre",
                "fallback_reference_source": "pre",
                "method_bundle": "deidentify-v2",
            },
            "concurrency": 1,
        },
    )
    assert create_resp.status_code == 200
    run_id = create_resp.json()["id"]
    assert create_resp.json()["method_bundle"] == "deidentify-v2"

    final = _wait_for_methods_lab_terminal(client, run_id)
    assert final["status"] == "completed"
    assert final["method_bundle"] == "deidentify-v2"
    assert final["runtime"]["method_bundle"] == "deidentify-v2"
    runs_resp = client.get("/api/methods-lab/runs")
    assert runs_resp.status_code == 200
    runs = runs_resp.json()["runs"]
    listed = next((row for row in runs if row["id"] == run_id), None)
    assert listed is not None
    assert listed["method_bundle"] == "deidentify-v2"
    assert seen_method_bundles == ["deidentify-v2"]


def test_methods_lab_deidentify_v2_scores_with_legacy_label_aliases(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["google.gemini-3.1-pro-preview"],
    )

    def fake_run_method_for_document(**kwargs):
        return (
            [CanonicalSpan(start=0, end=12, label="ADDRESS", text="123 Main St.")],
            [],
            None,
            [],
            [CanonicalSpan(start=0, end=12, label="ADDRESS", text="123 Main St.")],
            [],
            None,
            [],
        )

    monkeypatch.setattr("server._run_method_for_document", fake_run_method_for_document)

    upload_resp = _upload(
        client,
        data=_make_hips_v1_custom(
            "123 Main St.",
            pii_occurrences=[
                {"start": 0, "end": 12, "text": "123 Main St.", "pii_type": "LOCATION"},
            ],
        ),
    )
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]

    create_resp = client.post(
        "/api/methods-lab/runs",
        json={
            "name": "methods-deidentify-v2-alias-run",
            "doc_ids": [doc_id],
            "methods": [{"id": "method_1", "label": "Dual V2", "method_id": "dual-v2"}],
            "models": [
                {
                    "id": "m1",
                    "label": "Gemini Pro",
                    "model": "google.gemini-3.1-pro-preview",
                    "reasoning_effort": "none",
                    "anthropic_thinking": False,
                    "anthropic_thinking_budget_tokens": None,
                }
            ],
            "runtime": {
                "api_key": "request-key",
                "api_base": "https://proxy.example.com/v1",
                "temperature": 0.0,
                "match_mode": "exact",
                "reference_source": "pre",
                "fallback_reference_source": "pre",
                "method_bundle": "deidentify-v2",
            },
            "concurrency": 1,
        },
    )
    assert create_resp.status_code == 200
    run_id = create_resp.json()["id"]

    final = _wait_for_methods_lab_terminal(client, run_id)
    assert final["status"] == "completed"
    detail_resp = client.get(
        f"/api/methods-lab/runs/{run_id}/cells/{final['matrix']['cells'][0]['id']}/documents/{doc_id}"
    )
    assert detail_resp.status_code == 200
    result = detail_resp.json()
    assert result["status"] == "completed"
    assert result["metrics"]["micro"]["f1"] == 1.0
    assert result["raw_metrics"]["micro"]["f1"] == 1.0


def test_prepare_experiment_scoring_spans_deidentify_v2_uses_run_level_gold_labels():
    reference_spans = [
        CanonicalSpan(start=0, end=12, label="LOCATION", text="123 Main St."),
    ]
    hypothesis_spans = [
        CanonicalSpan(start=0, end=12, label="ADDRESS", text="123 Main St."),
    ]

    _projected_reference, projected_hypothesis = _prepare_experiment_scoring_spans(
        reference_spans,
        hypothesis_spans,
        label_projection="native",
        method_bundle="deidentify-v2",
        reference_label_set={"ADDRESS"},
    )

    assert projected_hypothesis[0].label == "ADDRESS"


def test_run_method_for_document_v2_post_process_expands_repeated_occurrences(
    client, monkeypatch
):
    doc_id = str(_upload(client).json()["id"])
    doc = _load_doc(doc_id)
    assert doc is not None
    doc.raw_text = "Anna met Anna."

    def fake_run_method_with_metadata(**kwargs):
        return SimpleNamespace(
            spans=[CanonicalSpan(start=0, end=4, label="NAME", text="Anna")],
            raw_spans=[CanonicalSpan(start=0, end=4, label="NAME", text="Anna")],
            warnings=[],
            llm_confidence=None,
            response_debug=[],
            resolution_events=[],
            resolution_policy_version=None,
        )

    monkeypatch.setattr("server.run_method_with_metadata", fake_run_method_with_metadata)

    spans, warnings, _confidence, _diag, raw_spans, _events, _policy, _debug = _run_method_for_document(
        doc=doc,
        method_id="default",
        api_key="request-key",
        api_base="https://proxy.example.com/v1",
        model="google.gemini-3.1-pro-preview",
        system_prompt="",
        temperature=0.0,
        reasoning_effort="none",
        anthropic_thinking=False,
        anthropic_thinking_budget_tokens=None,
        method_verify=False,
        label_profile="simple",
        chunk_mode="off",
        chunk_size_chars=10000,
        method_bundle="v2+post-process",
    )

    assert warnings == []
    assert [(span.start, span.end, span.text) for span in raw_spans] == [
        (0, 4, "Anna"),
        (9, 13, "Anna"),
    ]
    assert [(span.start, span.end, span.text) for span in spans] == [
        (0, 4, "Anna"),
        (9, 13, "Anna"),
    ]


def test_run_method_for_document_deidentify_v2_uses_legacy_chunking_and_propagation(monkeypatch):
    first_text = "Anna " + ("x" * 30010)
    second_text = "Anna " + ("y" * 30010)
    first_prefix = "student: "
    second_prefix = "volunteer: "
    raw_text = f"{first_prefix}{first_text}\n{second_prefix}{second_text}"
    doc = CanonicalDocument(
        id="doc-legacy-v2",
        filename="session.jsonl.record-0001.json",
        format="jsonl",
        raw_text=raw_text,
        utterances=[
            UtteranceRow(
                speaker="student",
                text=first_text,
                global_start=len(first_prefix),
                global_end=len(first_prefix) + len(first_text),
            ),
            UtteranceRow(
                speaker="volunteer",
                text=second_text,
                global_start=len(first_prefix) + len(first_text) + 1 + len(second_prefix),
                global_end=len(first_prefix)
                + len(first_text)
                + 1
                + len(second_prefix)
                + len(second_text),
            ),
        ],
        pre_annotations=[],
        label_set=["NAME"],
    )

    seen_texts: list[str] = []

    def fake_run_method_with_metadata(**kwargs):
        chunk_text = str(kwargs["text"])
        seen_texts.append(chunk_text)
        if "[MSG-1:student]" in chunk_text:
            start = chunk_text.index("Anna")
            return SimpleNamespace(
                spans=[CanonicalSpan(start=start, end=start + 4, label="NAME", text="Anna")],
                raw_spans=[CanonicalSpan(start=start, end=start + 4, label="NAME", text="Anna")],
                warnings=[],
                llm_confidence=None,
                response_debug=[],
                resolution_events=[],
                resolution_policy_version=None,
            )
        return SimpleNamespace(
            spans=[],
            raw_spans=[],
            warnings=[],
            llm_confidence=None,
            response_debug=[],
            resolution_events=[],
            resolution_policy_version=None,
        )

    monkeypatch.setattr("server.run_method_with_metadata", fake_run_method_with_metadata)

    spans, warnings, _confidence, diagnostics, raw_spans, events, policy, _debug = _run_method_for_document(
        doc=doc,
        method_id="dual-v2",
        api_key="request-key",
        api_base="https://proxy.example.com/v1",
        model="google.gemini-3.1-pro-preview",
        system_prompt="ignored",
        temperature=0.0,
        reasoning_effort="none",
        anthropic_thinking=False,
        anthropic_thinking_budget_tokens=None,
        method_verify=False,
        label_profile="simple",
        chunk_mode="off",
        chunk_size_chars=10000,
        method_bundle="deidentify-v2",
    )

    assert len(seen_texts) == 2
    assert seen_texts[0].startswith("[MSG-1:student] ")
    assert seen_texts[1].startswith("[MSG-2:volunteer] ")
    assert warnings == []
    assert diagnostics
    assert events == []
    assert policy is None
    assert [(span.start, span.end, span.label, span.text) for span in raw_spans] == [
        (len(first_prefix), len(first_prefix) + 4, "NAME", "Anna"),
        (
            len(first_prefix) + len(first_text) + 1 + len(second_prefix),
            len(first_prefix) + len(first_text) + 1 + len(second_prefix) + 4,
            "NAME",
            "Anna",
        ),
    ]
    assert [(span.start, span.end, span.label, span.text) for span in spans] == [
        (len(first_prefix), len(first_prefix) + 4, "NAME", "Anna"),
        (
            len(first_prefix) + len(first_text) + 1 + len(second_prefix),
            len(first_prefix) + len(first_text) + 1 + len(second_prefix) + 4,
            "NAME",
            "Anna",
        ),
    ]


def test_methods_lab_detail_includes_raw_metrics_and_resolution_metadata(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
    )

    text = "Hello Mr. Muhammad"

    def fake_run_method_with_metadata(**kwargs):
        return SimpleNamespace(
            spans=[
                CanonicalSpan(
                    start=text.index("Mr."),
                    end=len(text),
                    label="NAME",
                    text="Mr. Muhammad",
                )
            ],
            raw_spans=[
                CanonicalSpan(
                    start=text.index("Muhammad"),
                    end=len(text),
                    label="NAME",
                    text="Muhammad",
                )
            ],
            warnings=[],
            llm_confidence=_mock_confidence_metric(model=str(kwargs["model"])),
            resolution_events=[
                ResolutionEvent(
                    kind="boundary_resolution",
                    label="NAME",
                    rule="name_honorific_prefix",
                    before=CanonicalSpan(
                        start=text.index("Muhammad"),
                        end=len(text),
                        label="NAME",
                        text="Muhammad",
                    ),
                    after=CanonicalSpan(
                        start=text.index("Mr."),
                        end=len(text),
                        label="NAME",
                        text="Mr. Muhammad",
                    ),
                )
            ],
            resolution_policy_version="2026-03-span-resolution-v2",
        )

    monkeypatch.setattr("server.run_method_with_metadata", fake_run_method_with_metadata)

    upload_resp = _upload(client, _make_hips_v1_custom(text))
    assert upload_resp.status_code == 200
    doc_id = upload_resp.json()["id"]
    manual_resp = client.put(
        f"/api/documents/{doc_id}/manual-annotations",
        json=[
            {
                "start": text.index("Mr."),
                "end": len(text),
                "label": "NAME",
                "text": "Mr. Muhammad",
            }
        ],
    )
    assert manual_resp.status_code == 200

    create_resp = client.post(
        "/api/methods-lab/runs",
        json={
            "name": "method-resolution-audit",
            "doc_ids": [doc_id],
            "methods": [{"id": "method_1", "label": "Default", "method_id": "default"}],
            "models": [
                {
                    "id": "m1",
                    "label": "Codex",
                    "model": "openai.gpt-5.3-codex",
                    "reasoning_effort": "none",
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
    assert final["status"] == "completed"
    detail_resp = client.get(
        f"/api/methods-lab/runs/{run_id}/cells/{final['matrix']['cells'][0]['id']}/documents/{doc_id}"
    )
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["raw_hypothesis_spans"][0]["text"] == "Muhammad"
    assert detail["resolution_events"][0]["rule"] == "name_honorific_prefix"
    assert detail["resolution_policy_version"] == "2026-03-span-resolution-v2"
    assert detail["raw_metrics"]["micro"]["f1"] == pytest.approx(0.0)
    assert detail["metrics"]["micro"]["f1"] == pytest.approx(1.0)
