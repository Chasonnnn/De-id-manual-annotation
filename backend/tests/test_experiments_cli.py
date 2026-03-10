import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest
from fastapi.testclient import TestClient

from experiments_cli import main
from models import CanonicalSpan, LLMConfidenceMetric
from server import (
    _load_methods_lab_index,
    _load_methods_lab_run,
    _load_prompt_lab_index,
    _load_prompt_lab_run,
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
    _session_docs.clear()
    _prompt_lab_runs.clear()
    _methods_lab_runs.clear()
    yield


@pytest.fixture
def client():
    return TestClient(app)


def _make_hips_v1(
    transcript: str = "Hello Anna, call Sue please.",
    pii_occurrences: list[dict] | None = None,
) -> bytes:
    return json.dumps(
        {
            "transcript": transcript,
            "pii_occurrences": pii_occurrences
            or [{"start": 6, "end": 10, "text": "Anna", "pii_type": "NAME"}],
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


def test_manifest_prompt_run_persists_artifacts_and_writes_reports(
    client, monkeypatch, tmp_path, capsys
):
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

    doc_id = _upload(client)
    manifest_path = tmp_path / "prompt_manifest.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "kind: prompt_lab",
                "name: manifest-prompt",
                "session: default",
                "doc_ids:",
                f"  - {doc_id}",
                "prompts:",
                "  - id: p1",
                "    label: Baseline",
                "    system_prompt: Detect pii spans as strict JSON",
                "models:",
                "  - id: m1",
                "    label: Codex",
                "    model: openai.gpt-5.3-codex",
                "    reasoning_effort: xhigh",
                "runtime:",
                "  api_key: request-key",
                "  api_base: https://proxy.example.com/v1",
                "  temperature: 0.0",
                "  match_mode: exact",
                "  reference_source: manual",
                "  fallback_reference_source: pre",
                "concurrency: 1",
            ]
        )
    )
    output_json = tmp_path / "prompt_run.json"
    output_csv = tmp_path / "prompt_run.csv"

    exit_code = main(
        [
            "run",
            str(manifest_path),
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ]
    )

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "completed" in stdout

    run_ids = _load_prompt_lab_index()
    assert len(run_ids) == 1
    saved = _load_prompt_lab_run(run_ids[0])
    assert saved is not None
    assert saved["status"] == "completed"
    assert saved["doc_ids"] == [doc_id]

    report = json.loads(output_json.read_text())
    assert report["id"] == run_ids[0]
    assert report["status"] == "completed"

    with output_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["status"] == "completed"
    assert rows[0]["prompt_id"] == "p1"


def test_manifest_methods_run_persists_artifacts(client, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
    )

    def fake_run_method_with_metadata(**kwargs):
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            warnings=[],
            llm_confidence=_mock_confidence_metric(model="openai.gpt-5.3-codex"),
        )

    monkeypatch.setattr("server.run_method_with_metadata", fake_run_method_with_metadata)

    doc_id = _upload(client)
    _set_manual_annotations(
        client,
        doc_id,
        [{"start": 6, "end": 10, "label": "NAME", "text": "Anna"}],
    )
    manifest_path = tmp_path / "methods_manifest.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "kind: methods_lab",
                "name: manifest-methods",
                "session: default",
                "doc_ids:",
                f"  - {doc_id}",
                "methods:",
                "  - id: method_1",
                "    label: Default",
                "    method_id: default",
                "models:",
                "  - id: model_1",
                "    label: Codex",
                "    model: openai.gpt-5.3-codex",
                "    reasoning_effort: xhigh",
                "runtime:",
                "  api_key: request-key",
                "  api_base: https://proxy.example.com/v1",
                "  temperature: 0.0",
                "  match_mode: exact",
                "concurrency: 1",
            ]
        )
    )

    exit_code = main(["run", str(manifest_path)])

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert "completed" in stdout

    run_ids = _load_methods_lab_index()
    assert len(run_ids) == 1
    saved = _load_methods_lab_run(run_ids[0])
    assert saved is not None
    assert saved["status"] == "completed"
    assert saved["doc_ids"] == [doc_id]


def test_methods_cli_run_accepts_pre_reference_source(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
    )

    def fake_run_method_with_metadata(**kwargs):
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            warnings=[],
            llm_confidence=_mock_confidence_metric(model="openai.gpt-5.3-codex"),
        )

    monkeypatch.setattr("server.run_method_with_metadata", fake_run_method_with_metadata)

    doc_id = _upload(client)

    exit_code = main(
        [
            "methods",
            "--doc-id",
            doc_id,
            "--method",
            "Default=default",
            "--model",
            "Codex=openai.gpt-5.3-codex",
            "--api-key",
            "request-key",
            "--api-base",
            "https://proxy.example.com/v1",
            "--reference-source",
            "pre",
            "--fallback-reference-source",
            "pre",
        ]
    )

    assert exit_code == 0
    run_ids = _load_methods_lab_index()
    assert len(run_ids) == 1
    saved = _load_methods_lab_run(run_ids[0])
    assert saved is not None
    assert saved["status"] == "completed"
    assert saved["runtime"]["reference_source"] == "pre"
    assert saved["runtime"]["fallback_reference_source"] == "pre"


def test_methods_cli_run_accepts_task_timeout_seconds(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
    )

    def fake_run_method_with_metadata(**kwargs):
        return SimpleNamespace(
            spans=[CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
            warnings=[],
            llm_confidence=_mock_confidence_metric(model="openai.gpt-5.3-codex"),
        )

    monkeypatch.setattr("server.run_method_with_metadata", fake_run_method_with_metadata)

    doc_id = _upload(client)
    _set_manual_annotations(
        client,
        doc_id,
        [{"start": 6, "end": 10, "label": "NAME", "text": "Anna"}],
    )

    exit_code = main(
        [
            "methods",
            "--doc-id",
            doc_id,
            "--method",
            "Default=default",
            "--model",
            "Codex=openai.gpt-5.3-codex",
            "--api-key",
            "request-key",
            "--api-base",
            "https://proxy.example.com/v1",
            "--task-timeout-seconds",
            "12.5",
        ]
    )

    assert exit_code == 0
    run_ids = _load_methods_lab_index()
    assert len(run_ids) == 1
    saved = _load_methods_lab_run(run_ids[0])
    assert saved is not None
    assert saved["status"] == "completed"
    assert saved["runtime"]["task_timeout_seconds"] == pytest.approx(12.5)


def test_manifest_prompt_run_accepts_folder_ids(client, monkeypatch, tmp_path):
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

    _upload(client, data=_make_multi_record_jsonl(), filename="sessions.jsonl")
    folder_id = client.get("/api/folders").json()[0]["id"]
    folder_detail = client.get(f"/api/folders/{folder_id}").json()
    child_doc_ids = [item["id"] for item in folder_detail["documents"]]

    manifest_path = tmp_path / "prompt_folder_manifest.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "kind: prompt_lab",
                "name: manifest-prompt-folder",
                "session: default",
                "folder_ids:",
                f"  - {folder_id}",
                "prompts:",
                "  - id: p1",
                "    label: Baseline",
                "    system_prompt: Detect pii spans as strict JSON",
                "models:",
                "  - id: m1",
                "    label: Codex",
                "    model: openai.gpt-5.3-codex",
                "    reasoning_effort: xhigh",
                "runtime:",
                "  api_key: request-key",
                "  api_base: https://proxy.example.com/v1",
                "  temperature: 0.0",
                "  match_mode: exact",
                "  reference_source: pre",
                "  fallback_reference_source: pre",
                "concurrency: 2",
            ]
        )
    )

    exit_code = main(["run", str(manifest_path)])

    assert exit_code == 0
    run_ids = _load_prompt_lab_index()
    saved = _load_prompt_lab_run(run_ids[0])
    assert saved is not None
    assert saved["folder_ids"] == [folder_id]
    assert saved["doc_ids"] == child_doc_ids


def test_flag_methods_run_accepts_folder_id(client, monkeypatch):
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
            llm_confidence=_mock_confidence_metric(model="openai.gpt-5.3-codex"),
        )

    monkeypatch.setattr("server.run_method_with_metadata", fake_run_method_with_metadata)

    _upload(client, data=_make_multi_record_jsonl(), filename="sessions.jsonl")
    folder_id = client.get("/api/folders").json()[0]["id"]
    folder_detail = client.get(f"/api/folders/{folder_id}").json()
    child_doc_ids = [item["id"] for item in folder_detail["documents"]]
    for doc_id in child_doc_ids:
        document = client.get(f"/api/documents/{doc_id}").json()
        _set_manual_annotations(client, doc_id, document["pre_annotations"])

    exit_code = main(
        [
            "methods",
            "--session",
            "default",
            "--folder-id",
            folder_id,
            "--method",
            "Default=default",
            "--model",
            "Codex=openai.gpt-5.3-codex",
            "--api-key",
            "request-key",
            "--api-base",
            "https://proxy.example.com/v1",
        ]
    )

    assert exit_code == 0
    run_ids = _load_methods_lab_index()
    saved = _load_methods_lab_run(run_ids[0])
    assert saved is not None
    assert saved["folder_ids"] == [folder_id]
    assert saved["doc_ids"] == child_doc_ids


def test_manifest_prompt_run_accepts_concurrency_above_legacy_limit(
    client, monkeypatch, tmp_path
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

    doc_id = _upload(client)
    manifest_path = tmp_path / "prompt_manifest_high_concurrency.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "kind: prompt_lab",
                "name: manifest-prompt-high-concurrency",
                "session: default",
                "doc_ids:",
                f"  - {doc_id}",
                "prompts:",
                "  - id: p1",
                "    label: Baseline",
                "    system_prompt: Detect pii spans as strict JSON",
                "models:",
                "  - id: m1",
                "    label: Codex",
                "    model: openai.gpt-5.3-codex",
                "    reasoning_effort: xhigh",
                "runtime:",
                "  api_key: request-key",
                "  api_base: https://proxy.example.com/v1",
                "  temperature: 0.0",
                "  match_mode: exact",
                "  reference_source: manual",
                "  fallback_reference_source: pre",
                "concurrency: 12",
            ]
        )
    )

    exit_code = main(["run", str(manifest_path)])

    assert exit_code == 0
    run_ids = _load_prompt_lab_index()
    saved = _load_prompt_lab_run(run_ids[0])
    assert saved is not None
    assert saved["concurrency"] == 12


def test_manifest_methods_run_accepts_concurrency_above_legacy_limit(
    client, monkeypatch, tmp_path
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
            llm_confidence=_mock_confidence_metric(model="openai.gpt-5.3-codex"),
        )

    monkeypatch.setattr("server.run_method_with_metadata", fake_run_method_with_metadata)

    doc_id = _upload(client)
    _set_manual_annotations(
        client,
        doc_id,
        [{"start": 6, "end": 10, "label": "NAME", "text": "Anna"}],
    )
    manifest_path = tmp_path / "methods_manifest_high_concurrency.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "kind: methods_lab",
                "name: manifest-methods-high-concurrency",
                "session: default",
                "doc_ids:",
                f"  - {doc_id}",
                "methods:",
                "  - id: method_1",
                "    label: Default",
                "    method_id: default",
                "models:",
                "  - id: model_1",
                "    label: Codex",
                "    model: openai.gpt-5.3-codex",
                "    reasoning_effort: xhigh",
                "runtime:",
                "  api_key: request-key",
                "  api_base: https://proxy.example.com/v1",
                "  temperature: 0.0",
                "  match_mode: exact",
                "concurrency: 12",
            ]
        )
    )

    exit_code = main(["run", str(manifest_path)])

    assert exit_code == 0
    run_ids = _load_methods_lab_index()
    saved = _load_methods_lab_run(run_ids[0])
    assert saved is not None
    assert saved["concurrency"] == 12


def test_manifest_prompt_run_rejects_concurrency_above_configured_cap(tmp_path, capsys):
    _write_test_config({"prompt_lab_max_concurrency": 16})
    manifest_path = tmp_path / "invalid_prompt_manifest.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "kind: prompt_lab",
                "name: invalid-prompt-concurrency",
                "session: default",
                "doc_ids:",
                "  - missing-doc",
                "prompts:",
                "  - id: p1",
                "    label: Baseline",
                "    system_prompt: Detect pii spans as strict JSON",
                "models:",
                "  - id: m1",
                "    label: Codex",
                "    model: openai.gpt-5.3-codex",
                "runtime:",
                "  api_key: request-key",
                "concurrency: 17",
            ]
        )
    )

    exit_code = main(["run", str(manifest_path)])

    assert exit_code == 1
    assert "concurrency must be between 1 and 16" in capsys.readouterr().err


def test_flag_prompt_run_compiles_and_succeeds(client, monkeypatch, tmp_path):
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

    doc_id = _upload(client)
    output_json = tmp_path / "flag_prompt.json"
    exit_code = main(
        [
            "prompt",
            "--session",
            "default",
            "--doc-id",
            doc_id,
            "--prompt",
            "Baseline=Detect pii spans as strict JSON",
            "--model",
            "Codex=openai.gpt-5.3-codex",
            "--api-key",
            "request-key",
            "--api-base",
            "https://proxy.example.com/v1",
            "--reasoning-effort",
            "xhigh",
            "--output-json",
            str(output_json),
        ]
    )

    assert exit_code == 0
    run_ids = _load_prompt_lab_index()
    saved = _load_prompt_lab_run(run_ids[0])
    assert saved is not None
    assert saved["prompts"][0]["label"] == "Baseline"
    assert saved["prompts"][0]["system_prompt"] == "Detect pii spans as strict JSON"
    assert saved["models"][0]["model"] == "openai.gpt-5.3-codex"
    assert json.loads(output_json.read_text())["status"] == "completed"


def test_flag_methods_run_compiles_and_succeeds(client, monkeypatch):
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

    doc_id = _upload(client)
    _set_manual_annotations(
        client,
        doc_id,
        [{"start": 6, "end": 10, "label": "NAME", "text": "Anna"}],
    )
    exit_code = main(
        [
            "methods",
            "--session",
            "default",
            "--doc-id",
            doc_id,
            "--method",
            "Default=default",
            "--model",
            "Codex=openai.gpt-5.3-codex",
            "--api-key",
            "request-key",
            "--api-base",
            "https://proxy.example.com/v1",
            "--reasoning-effort",
            "xhigh",
        ]
    )

    assert exit_code == 0
    run_ids = _load_methods_lab_index()
    saved = _load_methods_lab_run(run_ids[0])
    assert saved is not None
    assert saved["methods"][0]["label"] == "Default"
    assert saved["methods"][0]["method_id"] == "default"
    assert saved["models"][0]["model"] == "openai.gpt-5.3-codex"


def test_agents_prompt_file_is_loaded_verbatim(client, monkeypatch, tmp_path):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
    )
    captured: dict[str, object] = {}

    def fake_run_llm_with_metadata(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            spans=[],
            warnings=[],
            llm_confidence=_mock_confidence_metric(),
        )

    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)

    doc_id = _upload(client)
    agents_path = tmp_path / "AGENTS.md"
    agents_text = "Keep this prompt exactly as written.\nDo not rewrite it."
    agents_path.write_text(agents_text)
    manifest_path = tmp_path / "agents_manifest.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "kind: prompt_lab",
                "session: default",
                "doc_ids:",
                f"  - {doc_id}",
                "prompts:",
                "  - id: p1",
                "    label: Agents",
                "    prompt_file: AGENTS.md",
                "models:",
                "  - id: m1",
                "    label: Codex",
                "    model: openai.gpt-5.3-codex",
                "runtime:",
                "  api_key: request-key",
                "  api_base: https://proxy.example.com/v1",
            ]
        )
    )

    exit_code = main(["run", str(manifest_path)])

    assert exit_code == 0
    assert captured["system_prompt"] == agents_text


def test_skill_prompt_file_strips_frontmatter(client, monkeypatch, tmp_path):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
    )
    captured: dict[str, object] = {}

    def fake_run_llm_with_metadata(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            spans=[],
            warnings=[],
            llm_confidence=_mock_confidence_metric(),
        )

    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)

    doc_id = _upload(client)
    skill_path = tmp_path / "SKILL.md"
    skill_path.write_text(
        "\n".join(
            [
                "---",
                "name: example-skill",
                "description: Example skill",
                "---",
                "",
                "Follow the skill instructions.",
                "Return strict JSON.",
            ]
        )
    )
    manifest_path = tmp_path / "skill_manifest.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "kind: prompt_lab",
                "session: default",
                "doc_ids:",
                f"  - {doc_id}",
                "prompts:",
                "  - id: p1",
                "    label: Skill",
                "    prompt_file: SKILL.md",
                "models:",
                "  - id: m1",
                "    label: Codex",
                "    model: openai.gpt-5.3-codex",
                "runtime:",
                "  api_key: request-key",
                "  api_base: https://proxy.example.com/v1",
            ]
        )
    )

    exit_code = main(["run", str(manifest_path)])

    assert exit_code == 0
    assert captured["system_prompt"] == "Follow the skill instructions.\nReturn strict JSON."


@pytest.mark.parametrize(
    ("manifest_lines", "error_substring"),
    [
        (
            [
                "kind: prompt_lab",
                "session: default",
                "prompts:",
                "  - id: p1",
                "    label: Missing File",
                "    prompt_file: missing.md",
                "models:",
                "  - id: m1",
                "    label: Codex",
                "    model: openai.gpt-5.3-codex",
                "runtime:",
                "  api_key: request-key",
                "  api_base: https://proxy.example.com/v1",
            ],
            "Prompt file not found",
        ),
        (
            [
                "kind: prompt_lab",
                "session: default",
                "prompts:",
                "  - id: p1",
                "    label: Conflicting",
                "    system_prompt: Inline prompt",
                "    prompt_file: AGENTS.md",
                "models:",
                "  - id: m1",
                "    label: Codex",
                "    model: openai.gpt-5.3-codex",
                "runtime:",
                "  api_key: request-key",
                "  api_base: https://proxy.example.com/v1",
            ],
            "exactly one of system_prompt or prompt_file",
        ),
        (
            [
                "kind: prompt_lab",
                "session: default",
                "prompts:",
                "  - id: p1",
                "    label: Empty Skill",
                "    prompt_file: SKILL.md",
                "models:",
                "  - id: m1",
                "    label: Codex",
                "    model: openai.gpt-5.3-codex",
                "runtime:",
                "  api_key: request-key",
                "  api_base: https://proxy.example.com/v1",
            ],
            "resolved to empty prompt text",
        ),
    ],
)
def test_invalid_prompt_file_inputs_fail_fast(
    client, tmp_path, capsys, manifest_lines, error_substring
):
    _upload(client)
    (tmp_path / "AGENTS.md").write_text("Inline instructions")
    (tmp_path / "SKILL.md").write_text("---\nname: empty\ndescription: empty\n---\n")
    manifest_path = tmp_path / "invalid_manifest.yaml"
    manifest_path.write_text("\n".join(manifest_lines))

    exit_code = main(["run", str(manifest_path)])

    assert exit_code == 1
    assert error_substring in capsys.readouterr().err
    assert _load_prompt_lab_index() == []


def test_default_doc_selection_matches_run_kind(client, monkeypatch):
    monkeypatch.setattr(
        "server._fetch_gateway_model_ids",
        lambda api_base, api_key: ["openai.gpt-5.3-codex"],
    )

    def fake_run_llm_with_metadata(**kwargs):
        return SimpleNamespace(
            spans=[],
            warnings=[],
            llm_confidence=_mock_confidence_metric(),
        )

    def fake_run_method_with_metadata(**kwargs):
        return SimpleNamespace(
            spans=[],
            warnings=[],
            llm_confidence=_mock_confidence_metric(),
        )

    monkeypatch.setattr("server.run_llm_with_metadata", fake_run_llm_with_metadata)
    monkeypatch.setattr("server.run_method_with_metadata", fake_run_method_with_metadata)

    manual_doc_id = _upload(client)
    no_manual_doc_id = _upload(client, filename="second.json")
    _set_manual_annotations(
        client,
        manual_doc_id,
        [{"start": 6, "end": 10, "label": "NAME", "text": "Anna"}],
    )

    prompt_exit = main(
        [
            "prompt",
            "--session",
            "default",
            "--prompt",
            "Baseline=Detect pii spans as strict JSON",
            "--model",
            "Codex=openai.gpt-5.3-codex",
            "--api-key",
            "request-key",
            "--api-base",
            "https://proxy.example.com/v1",
        ]
    )
    assert prompt_exit == 0
    prompt_run = _load_prompt_lab_run(_load_prompt_lab_index()[0])
    assert prompt_run is not None
    assert set(prompt_run["doc_ids"]) == {manual_doc_id, no_manual_doc_id}

    methods_exit = main(
        [
            "methods",
            "--session",
            "default",
            "--method",
            "Default=default",
            "--model",
            "Codex=openai.gpt-5.3-codex",
            "--api-key",
            "request-key",
            "--api-base",
            "https://proxy.example.com/v1",
        ]
    )
    assert methods_exit == 0
    methods_run = _load_methods_lab_run(_load_methods_lab_index()[0])
    assert methods_run is not None
    assert set(methods_run["doc_ids"]) == {manual_doc_id, no_manual_doc_id}


def test_manifest_doc_ids_are_normalized_to_strings(tmp_path):
    manifest_path = tmp_path / "numeric_doc_ids.yaml"
    manifest_path.write_text(
        "\n".join(
            [
                "kind: methods_lab",
                "session: default",
                "doc_ids:",
                "  - 26028249",
                "methods:",
                "  - id: m1",
                "    label: Presidio",
                "    method_id: presidio",
                "models:",
                "  - id: model_1",
                "    label: Codex",
                "    model: openai.gpt-5.3-codex",
                "runtime:",
                "  temperature: 0.0",
                "  match_mode: exact",
            ]
        )
    )

    from experiments_cli import _load_manifest

    kind, session_id, body, _context_dir = _load_manifest(str(manifest_path))

    assert kind == "methods_lab"
    assert session_id == "default"
    assert body.doc_ids == ["26028249"]


def test_list_commands_emit_machine_readable_json(client, capsys):
    doc_id = _upload(client)

    docs_exit = main(["list-docs", "--session", "default"])
    assert docs_exit == 0
    docs_payload = json.loads(capsys.readouterr().out)
    assert docs_payload["session"] == "default"
    assert docs_payload["documents"][0]["id"] == doc_id

    models_exit = main(["list-models"])
    assert models_exit == 0
    models_payload = json.loads(capsys.readouterr().out)
    assert any(item["model"] == "openai.gpt-5.3-codex" for item in models_payload["models"])

    methods_exit = main(["list-methods"])
    assert methods_exit == 0
    methods_payload = json.loads(capsys.readouterr().out)
    assert any(item["id"] == "default" for item in methods_payload["methods"])
