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
    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "upchieve_surrogate_v4_import.py"
    )
    spec = importlib.util.spec_from_file_location("upchieve_surrogate_v4_import", module_path)
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


def _write_source_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def test_build_records_repairs_offsets_and_surrogate_token_substitutions(tmp_path):
    module = _load_module()
    source_path = tmp_path / "sample.jsonl"

    _write_source_jsonl(
        source_path,
        [
            {
                "dialogue_id": "dialogue-1",
                "transcript_turns": [
                    {
                        "user_role": "student",
                        "message_type": "text",
                        "timestamp_seconds": "1.0",
                        "message": "Hi Clearview",
                        "message_original": "Hi <LOCATION>",
                        "surrogate_spans": [
                            {
                                "span_text": "Clearview",
                                "entity_type": "ADDRESS",
                                "original_entity_type": "LOCATION",
                                "tag_start": 3,
                                "tag_end": 12,
                            }
                        ],
                    },
                    {
                        "user_role": "volunteer",
                        "message_type": "text",
                        "timestamp_seconds": "2.0",
                        "message": "Treaty writing workshop and Oakwood",
                        "message_original": "Treaty <COURSE> and <LOCATION>",
                        "surrogate_spans": [
                            {
                                "span_text": "writing workshop",
                                "entity_type": "COURSE",
                                "original_entity_type": "COURSE",
                                "tag_start": 7,
                                "tag_end": 23,
                            },
                            {
                                "span_text": "Oakwood",
                                "entity_type": "ADDRESS",
                                "original_entity_type": "LOCATION",
                                "tag_start": 29,
                                "tag_end": 36,
                            },
                        ],
                    },
                    {
                        "user_role": "student",
                        "message_type": "text",
                        "timestamp_seconds": "3.0",
                        "message": "[REDACTED-ID-0001] is here",
                        "message_original": "<SOCIAL_HANDLE> is here",
                        "surrogate_spans": [
                            {
                                "span_text": "[REDACTED-ID-0001]",
                                "entity_type": "IDENTIFYING_NUMBER",
                                "original_entity_type": "SOCIAL_HANDLE",
                                "tag_start": 0,
                                "tag_end": 18,
                            }
                        ],
                    },
                ],
                "action_v4_spans": [
                    {
                        "span_text": "Clearview",
                        "entity_type": "ADDRESS",
                        "metadata": {"turn_index": 0, "tag_start": 3, "tag_end": 12},
                    },
                    {
                        "span_text": "Oakwood",
                        "entity_type": "ADDRESS",
                        "metadata": {
                            "turn_index": 1,
                            "tag_start": 20,
                            "tag_end": 27,
                            "turn_text": "Treaty Psychology 101 and Oakwood",
                        },
                    },
                    {
                        "span_text": "@user_00001",
                        "entity_type": "IDENTIFYING_NUMBER",
                        "metadata": {
                            "turn_index": 2,
                            "tag_start": 0,
                            "tag_end": 11,
                            "turn_text": "@user_00001 is here",
                            "turn_text_original": "<SOCIAL_HANDLE> is here",
                            "original_entity_type": "SOCIAL_HANDLE",
                        },
                    },
                ],
            }
        ],
    )

    records = module.build_upchieve_surrogate_records(source_path)

    assert [record["session_id"] for record in records] == ["dialogue-1"]
    transcript = records[0]["transcript"]
    assert [turn["sequence_id"] for turn in transcript] == [1, 2, 3]
    assert transcript[0]["annotations"] == [
        {"start": 3, "end": 12, "text": "Clearview", "pii_type": "ADDRESS"}
    ]
    assert transcript[1]["annotations"] == [
        {"start": 28, "end": 35, "text": "Oakwood", "pii_type": "ADDRESS"}
    ]
    assert transcript[2]["annotations"] == [
        {
            "start": 0,
            "end": 18,
            "text": "[REDACTED-ID-0001]",
            "pii_type": "IDENTIFYING_NUMBER",
        }
    ]


def test_generated_jsonl_upload_creates_expected_folder(client, tmp_path):
    module = _load_module()
    source_path = tmp_path / "upchieve_social_studies_only_surrogate_replaced_with_v4_spans.jsonl"
    output_dir = tmp_path / "out"

    _write_source_jsonl(
        source_path,
        [
            {
                "dialogue_id": "22",
                "transcript_turns": [
                    {
                        "user_role": "student",
                        "message_type": "text",
                        "timestamp_seconds": "1.0",
                        "message": "Hi Clearview",
                        "message_original": "Hi <LOCATION>",
                        "surrogate_spans": [],
                    }
                ],
                "action_v4_spans": [
                    {
                        "span_text": "Clearview",
                        "entity_type": "ADDRESS",
                        "metadata": {"turn_index": 0, "tag_start": 3, "tag_end": 12},
                    }
                ],
            },
            {
                "dialogue_id": "23",
                "transcript_turns": [
                    {
                        "user_role": "volunteer",
                        "message_type": "text",
                        "timestamp_seconds": "2.0",
                        "message": "Hello Ryan",
                        "message_original": "Hello <PERSON>",
                        "surrogate_spans": [],
                    }
                ],
                "action_v4_spans": [
                    {
                        "span_text": "Ryan",
                        "entity_type": "NAME",
                        "metadata": {"turn_index": 0, "tag_start": 6, "tag_end": 10},
                    }
                ],
            },
        ],
    )

    result = module.convert_and_import_upchieve_surrogate_export(
        source_path=source_path,
        output_dir=output_dir,
    )

    assert result["record_count"] == 2
    assert result["output_path"] == str(
        output_dir / "upchieve_social_studies_only_surrogate_replaced_with_v4_spans.jsonl"
    )
    assert result["label_counts"] == {"ADDRESS": 1, "NAME": 1}

    folders_resp = client.get("/api/folders")
    assert folders_resp.status_code == 200
    folders = folders_resp.json()
    assert len(folders) == 1
    assert folders[0]["name"] == "upchieve_social_studies_only_surrogate_replaced_with_v4_spans"
    assert folders[0]["doc_count"] == 2

    detail_resp = client.get(f"/api/folders/{folders[0]['id']}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert [item["display_name"] for item in detail["documents"]] == [
        "Session 22",
        "Session 23",
    ]

    first_doc = client.get(f"/api/documents/{detail['documents'][0]['id']}").json()
    second_doc = client.get(f"/api/documents/{detail['documents'][1]['id']}").json()

    assert first_doc["raw_text"] == "student: Hi Clearview"
    assert [(span["label"], span["text"]) for span in first_doc["pre_annotations"]] == [
        ("ADDRESS", "Clearview")
    ]
    assert first_doc["manual_annotations"] == []

    assert second_doc["raw_text"] == "volunteer: Hello Ryan"
    assert [(span["label"], span["text"]) for span in second_doc["pre_annotations"]] == [
        ("NAME", "Ryan")
    ]


def test_import_jsonl_fails_when_folder_already_exists(tmp_path):
    module = _load_module()
    output_path = tmp_path / "upchieve_english_only_surrogate_replaced_with_v4_spans.jsonl"
    output_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "session_id": "23",
                        "transcript": [
                            {
                                "session_id": "23",
                                "sequence_id": 1,
                                "role": "student",
                                "content": "Hi Ryan",
                                "annotations": [
                                    {
                                        "start": 3,
                                        "end": 7,
                                        "text": "Ryan",
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
                        "session_id": "24",
                        "transcript": [
                            {
                                "session_id": "24",
                                "sequence_id": 1,
                                "role": "student",
                                "content": "Hi Sam",
                                "annotations": [
                                    {
                                        "start": 3,
                                        "end": 6,
                                        "text": "Sam",
                                        "pii_type": "NAME",
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

    first = module.import_upchieve_surrogate_jsonl(output_path)
    assert first.filename == "upchieve_english_only_surrogate_replaced_with_v4_spans.jsonl"

    with pytest.raises(ValueError, match="already exists"):
        module.import_upchieve_surrogate_jsonl(output_path)


def test_build_records_errors_when_span_cannot_be_resolved(tmp_path):
    module = _load_module()
    source_path = tmp_path / "bad.jsonl"

    _write_source_jsonl(
        source_path,
        [
            {
                "dialogue_id": "dialogue-bad",
                "transcript_turns": [
                    {
                        "user_role": "student",
                        "message_type": "text",
                        "timestamp_seconds": "1.0",
                        "message": "hello there",
                        "message_original": "hello there",
                        "surrogate_spans": [],
                    }
                ],
                "action_v4_spans": [
                    {
                        "span_text": "missing",
                        "entity_type": "NAME",
                        "metadata": {"turn_index": 0, "tag_start": 0, "tag_end": 7},
                    }
                ],
            }
        ],
    )

    with pytest.raises(ValueError, match="Unable to resolve action_v4 span"):
        module.build_upchieve_surrogate_records(source_path)
