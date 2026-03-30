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


def _load_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "name_location_csv_import.py"
    spec = importlib.util.spec_from_file_location("name_location_csv_import", module_path)
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


def _write_chat_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["timestamp_seconds", "user_role", "message_type", "message"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_build_records_parses_csv_rows_and_placeholder_spans(tmp_path):
    module = _load_module()
    input_dir = tmp_path / "dataset"
    input_dir.mkdir()

    _write_chat_csv(
        input_dir / "10.csv",
        [
            {
                "timestamp_seconds": "1.5",
                "user_role": "volunteer",
                "message_type": "text",
                "message": "hello <PERSON> from <LOCATION> and <PERSON>",
            },
            {
                "timestamp_seconds": "2.5",
                "user_role": "student",
                "message_type": "voice",
                "message": "ok",
            },
        ],
    )
    _write_chat_csv(
        input_dir / "2.csv",
        [
            {
                "timestamp_seconds": "0.5",
                "user_role": "student",
                "message_type": "text",
                "message": "<LOCATION> first",
            }
        ],
    )

    records = module.build_name_location_records(input_dir)

    assert [record["session_id"] for record in records] == ["2", "10"]

    second = records[1]
    assert [turn["sequence_id"] for turn in second["transcript"]] == [1, 2]
    assert second["transcript"][0]["role"] == "volunteer"
    assert second["transcript"][0]["timestamp_seconds"] == "1.5"
    assert second["transcript"][0]["message_type"] == "text"
    assert second["transcript"][0]["annotations"] == [
        {"start": 6, "end": 14, "text": "<PERSON>", "pii_type": "NAME"},
        {"start": 20, "end": 30, "text": "<LOCATION>", "pii_type": "ADDRESS"},
        {"start": 35, "end": 43, "text": "<PERSON>", "pii_type": "NAME"},
    ]
    assert second["transcript"][1]["annotations"] == []


def test_generated_jsonl_upload_creates_expected_folder(client, tmp_path):
    module = _load_module()
    input_dir = tmp_path / "dataset"
    input_dir.mkdir()
    output_path = tmp_path / "01_topic_1_prePII_noMoreThan_NAME_LOCATION.jsonl"

    _write_chat_csv(
        input_dir / "7.csv",
        [
            {
                "timestamp_seconds": "9.1",
                "user_role": "student",
                "message_type": "text",
                "message": "hi <PERSON>",
            }
        ],
    )
    _write_chat_csv(
        input_dir / "9.csv",
        [
            {
                "timestamp_seconds": "10.2",
                "user_role": "volunteer",
                "message_type": "voice",
                "message": "from <LOCATION>",
            }
        ],
    )

    records = module.build_name_location_records(input_dir)
    module.write_name_location_jsonl(records, output_path)

    upload_resp = client.post(
        "/api/documents/upload",
        files={
            "file": (
                "01_topic_1_prePII_noMoreThan_NAME_LOCATION.jsonl",
                output_path.read_bytes(),
                "application/json",
            )
        },
    )

    assert upload_resp.status_code == 200
    folders_resp = client.get("/api/folders")
    assert folders_resp.status_code == 200
    folders = folders_resp.json()
    assert len(folders) == 1
    assert folders[0]["name"] == "01_topic_1_prePII_noMoreThan_NAME_LOCATION"
    assert folders[0]["doc_count"] == 2

    detail_resp = client.get(f"/api/folders/{folders[0]['id']}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert [item["display_name"] for item in detail["documents"]] == [
        "Session 7",
        "Session 9",
    ]

    first_doc = client.get(f"/api/documents/{detail['documents'][0]['id']}").json()
    second_doc = client.get(f"/api/documents/{detail['documents'][1]['id']}").json()

    assert first_doc["raw_text"] == "student: hi <PERSON>"
    assert [(span["label"], span["text"]) for span in first_doc["pre_annotations"]] == [
        ("NAME", "<PERSON>")
    ]
    assert first_doc["manual_annotations"] == []

    assert second_doc["raw_text"] == "volunteer: from <LOCATION>"
    assert [(span["label"], span["text"]) for span in second_doc["pre_annotations"]] == [
        ("ADDRESS", "<LOCATION>")
    ]
    assert second_doc["manual_annotations"] == []


def test_build_records_preserves_message_text_and_allows_blank_messages(tmp_path):
    module = _load_module()
    input_dir = tmp_path / "dataset"
    input_dir.mkdir()

    _write_chat_csv(
        input_dir / "8491.csv",
        [
            {
                "timestamp_seconds": "1.0",
                "user_role": "student",
                "message_type": "text",
                "message": "  <PERSON>  ",
            },
            {
                "timestamp_seconds": "2.0",
                "user_role": "student",
                "message_type": "text",
                "message": "\n",
            },
        ],
    )

    records = module.build_name_location_records(input_dir)

    assert records[0]["transcript"][0]["content"] == "  <PERSON>  "
    assert records[0]["transcript"][0]["annotations"] == [
        {"start": 2, "end": 10, "text": "<PERSON>", "pii_type": "NAME"}
    ]
    assert records[0]["transcript"][1]["content"] == ""
    assert records[0]["transcript"][1]["annotations"] == []


def test_import_jsonl_fails_when_folder_already_exists(tmp_path):
    module = _load_module()
    output_path = tmp_path / "01_topic_1_prePII_noMoreThan_NAME_LOCATION.jsonl"
    output_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "session_id": "7",
                        "transcript": [
                            {
                                "session_id": "7",
                                "sequence_id": 1,
                                "role": "student",
                                "timestamp_seconds": "9.1",
                                "message_type": "text",
                                "content": "hi <PERSON>",
                                "annotations": [
                                    {
                                        "start": 3,
                                        "end": 11,
                                        "text": "<PERSON>",
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
                        "session_id": "9",
                        "transcript": [
                            {
                                "session_id": "9",
                                "sequence_id": 1,
                                "role": "volunteer",
                                "timestamp_seconds": "10.2",
                                "message_type": "text",
                                "content": "from <LOCATION>",
                                "annotations": [
                                    {
                                        "start": 5,
                                        "end": 15,
                                        "text": "<LOCATION>",
                                        "pii_type": "ADDRESS",
                                    }
                                ],
                            }
                        ],
                        "annotations": [],
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    imported = module.import_name_location_jsonl(output_path)
    assert imported.filename == "01_topic_1_prePII_noMoreThan_NAME_LOCATION.jsonl"

    with pytest.raises(ValueError, match="01_topic_1_prePII_noMoreThan_NAME_LOCATION"):
        module.import_name_location_jsonl(output_path)
