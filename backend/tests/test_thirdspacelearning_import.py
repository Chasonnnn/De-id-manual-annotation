import csv
import importlib.util
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import (
    app,
    _methods_lab_cancel_events,
    _methods_lab_runs,
    _prompt_lab_cancel_events,
    _prompt_lab_runs,
    _session_docs,
)


def _load_thirdspace_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "thirdspacelearning_import.py"
    spec = importlib.util.spec_from_file_location("thirdspacelearning_import", module_path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def clean_sessions(tmp_path, monkeypatch):
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


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def test_build_thirdspace_records_groups_sorts_and_maps_labels(tmp_path):
    module = _load_thirdspace_module()
    transcript_path = tmp_path / "transcripts.csv"
    index_path = tmp_path / "indices.csv"

    _write_csv(
        transcript_path,
        ["session_id", "sequence_id", "start_time", "end_time", "speaker", "content"],
        [
            {
                "session_id": "session-a",
                "sequence_id": "2",
                "start_time": "00:00:02,000",
                "end_time": "00:00:03,000",
                "speaker": "",
                "content": "Meet Liam at Oakhaven.",
            },
            {
                "session_id": "session-a",
                "sequence_id": "1",
                "start_time": "00:00:00,000",
                "end_time": "00:00:01,000",
                "speaker": "Tutor",
                "content": "Hello there.",
            },
            {
                "session_id": "session-b",
                "sequence_id": "1",
                "start_time": "00:00:05,000",
                "end_time": "00:00:06,000",
                "speaker": "Background",
                "content": "Welcome to third space.",
            },
        ],
    )
    _write_csv(
        index_path,
        [
            "session_id",
            "sequence_id",
            "pii_id",
            "pii_type",
            "surrogate",
            "starting index",
            "ending index",
            "content",
        ],
        [
            {
                "session_id": "session-a",
                "sequence_id": "2",
                "pii_id": "0",
                "pii_type": "NAME",
                "surrogate": "Liam",
                "starting index": "5",
                "ending index": "9",
                "content": "Meet Liam at Oakhaven.",
            },
            {
                "session_id": "session-a",
                "sequence_id": "2",
                "pii_id": "1",
                "pii_type": "OTHER_LOCATION",
                "surrogate": "Oakhaven",
                "starting index": "13",
                "ending index": "21",
                "content": "Meet Liam at Oakhaven.",
            },
            {
                "session_id": "session-b",
                "sequence_id": "1",
                "pii_id": "0",
                "pii_type": "TUTORING_PROVIDER",
                "surrogate": "",
                "starting index": "11",
                "ending index": "22",
                "content": "Welcome to third space.",
            },
        ],
    )

    records = module.build_thirdspace_records(transcript_path, index_path)

    assert len(records) == 2
    by_session = {record["session_id"]: record for record in records}

    session_a = by_session["session-a"]
    assert [turn["sequence_id"] for turn in session_a["transcript"]] == [1, 2]
    assert session_a["transcript"][1]["role"] == "unknown"
    assert session_a["transcript"][1]["start_time"] == "00:00:02,000"
    assert session_a["transcript"][1]["end_time"] == "00:00:03,000"
    assert session_a["transcript"][1]["annotations"] == [
        {"start": 5, "end": 9, "text": "Liam", "pii_type": "NAME"},
        {
            "start": 13,
            "end": 21,
            "text": "Oakhaven",
            "pii_type": "OTHER_LOCATIONS_IDENTIFIED",
        },
    ]

    session_b = by_session["session-b"]
    assert session_b["transcript"][0]["role"] == "Background"
    assert session_b["transcript"][0]["annotations"] == [
        {
            "start": 11,
            "end": 22,
            "text": "third space",
            "pii_type": "TUTOR_PROVIDER",
        }
    ]


def test_generated_thirdspace_jsonl_upload_creates_expected_import_folder(client, tmp_path):
    module = _load_thirdspace_module()
    transcript_path = tmp_path / "transcripts.csv"
    index_path = tmp_path / "indices.csv"
    output_path = tmp_path / "thirdspacelearning.jsonl"

    _write_csv(
        transcript_path,
        ["session_id", "sequence_id", "start_time", "end_time", "speaker", "content"],
        [
            {
                "session_id": "session-001",
                "sequence_id": "2",
                "start_time": "00:00:02,000",
                "end_time": "00:00:03,000",
                "speaker": "Tutor",
                "content": "Meet Liam.",
            },
            {
                "session_id": "session-001",
                "sequence_id": "1",
                "start_time": "00:00:00,000",
                "end_time": "00:00:01,000",
                "speaker": "Student",
                "content": "Hello.",
            },
            {
                "session_id": "session-002",
                "sequence_id": "1",
                "start_time": "00:00:04,000",
                "end_time": "00:00:05,000",
                "speaker": "",
                "content": "Welcome to third space.",
            },
        ],
    )
    _write_csv(
        index_path,
        [
            "session_id",
            "sequence_id",
            "pii_id",
            "pii_type",
            "surrogate",
            "starting index",
            "ending index",
            "content",
        ],
        [
            {
                "session_id": "session-001",
                "sequence_id": "2",
                "pii_id": "0",
                "pii_type": "NAME",
                "surrogate": "Liam",
                "starting index": "5",
                "ending index": "9",
                "content": "Meet Liam.",
            },
            {
                "session_id": "session-002",
                "sequence_id": "1",
                "pii_id": "0",
                "pii_type": "TUTORING_PROVIDER",
                "surrogate": "",
                "starting index": "11",
                "ending index": "22",
                "content": "Welcome to third space.",
            },
        ],
    )

    records = module.build_thirdspace_records(transcript_path, index_path)
    module.write_thirdspace_jsonl(records, output_path)

    upload_resp = client.post(
        "/api/documents/upload",
        files={"file": ("thirdspacelearning.jsonl", output_path.read_bytes(), "application/json")},
    )

    assert upload_resp.status_code == 200
    folders_resp = client.get("/api/folders")
    assert folders_resp.status_code == 200
    folders = folders_resp.json()
    assert len(folders) == 1
    assert folders[0]["name"] == "thirdspacelearning"
    assert folders[0]["doc_count"] == 2

    detail_resp = client.get(f"/api/folders/{folders[0]['id']}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert [item["display_name"] for item in detail["documents"]] == [
        "Session session-001",
        "Session session-002",
    ]

    first_doc = client.get(f"/api/documents/{detail['documents'][0]['id']}").json()
    second_doc = client.get(f"/api/documents/{detail['documents'][1]['id']}").json()

    assert first_doc["raw_text"] == "Student: Hello.\nTutor: Meet Liam."
    assert [span["label"] for span in first_doc["pre_annotations"]] == ["NAME"]
    assert second_doc["raw_text"] == "unknown: Welcome to third space."
    assert [span["label"] for span in second_doc["pre_annotations"]] == ["TUTOR_PROVIDER"]


def test_import_thirdspace_jsonl_fails_when_folder_already_exists(tmp_path):
    module = _load_thirdspace_module()
    output_path = tmp_path / "thirdspacelearning.jsonl"
    output_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "session_id": "session-001",
                        "transcript": [
                            {
                                "session_id": "session-001",
                                "sequence_id": 1,
                                "role": "Tutor",
                                "start_time": "00:00:00,000",
                                "end_time": "00:00:01,000",
                                "content": "Hello Liam",
                                "annotations": [
                                    {
                                        "start": 6,
                                        "end": 10,
                                        "text": "Liam",
                                        "pii_type": "NAME",
                                    }
                                ],
                            }
                        ],
                        "annotations": [],
                    }
                ),
                json.dumps(
                    {
                        "session_id": "session-002",
                        "transcript": [
                            {
                                "session_id": "session-002",
                                "sequence_id": 1,
                                "role": "Student",
                                "start_time": "00:00:00,000",
                                "end_time": "00:00:01,000",
                                "content": "Welcome to third space.",
                                "annotations": [
                                    {
                                        "start": 11,
                                        "end": 22,
                                        "text": "third space",
                                        "pii_type": "TUTOR_PROVIDER",
                                    }
                                ],
                            }
                        ],
                        "annotations": [],
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )

    imported = module.import_thirdspace_jsonl(output_path)
    assert imported.filename == "thirdspacelearning.jsonl"

    with pytest.raises(ValueError, match="thirdspacelearning"):
        module.import_thirdspace_jsonl(output_path)
