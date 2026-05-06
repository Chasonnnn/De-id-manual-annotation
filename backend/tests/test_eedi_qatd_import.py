import csv
import importlib.util
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
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "eedi_qatd_import.py"
    spec = importlib.util.spec_from_file_location("eedi_qatd_import", module_path)
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


def test_build_eedi_records_groups_dialogues_and_attaches_metadata(tmp_path):
    module = _load_module()
    source_dir = tmp_path / "eedi"
    anchored_dir = source_dir / "anchored-dialogues"
    anchored_dir.mkdir(parents=True)

    _write_csv(
        anchored_dir / "train.csv",
        [
            "InterventionId",
            "TutorId",
            "QuestionId_DQ",
            "MessageSequence",
            "IsTutor",
            "MessageString",
            "TalkMovePrediction",
        ],
        [
            {
                "InterventionId": "10",
                "TutorId": "749",
                "QuestionId_DQ": "104614",
                "MessageSequence": "2",
                "IsTutor": "1",
                "MessageString": "Let's look at the first decimal place.",
                "TalkMovePrediction": "<Press for Accuracy>",
            },
            {
                "InterventionId": "10",
                "TutorId": "-1",
                "QuestionId_DQ": "104614",
                "MessageSequence": "1",
                "IsTutor": "0",
                "MessageString": "I think both are wrong.",
                "TalkMovePrediction": "",
            },
            {
                "InterventionId": "11",
                "TutorId": "801",
                "QuestionId_DQ": "200001",
                "MessageSequence": "1",
                "IsTutor": "1",
                "MessageString": "Hello there.",
                "TalkMovePrediction": "<None>",
            },
        ],
    )
    _write_csv(
        anchored_dir / "test.csv",
        [
            "InterventionId",
            "TutorId",
            "QuestionId_DQ",
            "MessageSequence",
            "IsTutor",
            "MessageString",
            "TalkMovePrediction",
        ],
        [],
    )
    _write_csv(
        source_dir / "dq-question-metadata.csv",
        [
            "QuestionId_DQ",
            "InterventionId",
            "MetaDataId",
            "Text",
            "Sequence",
            "MetaDataTagId",
            "Label",
        ],
        [
            {
                "QuestionId_DQ": "104614",
                "InterventionId": "10",
                "MetaDataId": "1",
                "Text": "Round 5.4598 to 1 dp.",
                "Sequence": "1",
                "MetaDataTagId": "10",
                "Label": "Question Text",
            },
            {
                "QuestionId_DQ": "104614",
                "InterventionId": "10",
                "MetaDataId": "2",
                "Text": "5.5",
                "Sequence": "2",
                "MetaDataTagId": "20",
                "Label": "Answer A Text",
            },
        ],
    )
    _write_csv(
        source_dir / "dialogue-subjects.csv",
        [
            "InterventionId",
            "SubjectId",
            "ParentSubjectId",
            "SubjectName",
            "SubjectLevel",
            "SubjectType",
        ],
        [
            {
                "InterventionId": "10",
                "SubjectId": "32",
                "ParentSubjectId": "3",
                "SubjectName": "Number",
                "SubjectLevel": "1",
                "SubjectType": "Subject",
            },
            {
                "InterventionId": "10",
                "SubjectId": "141",
                "ParentSubjectId": "32",
                "SubjectName": "Rounding and Estimating",
                "SubjectLevel": "2",
                "SubjectType": "Topic",
            },
        ],
    )

    records = module.build_eedi_qatd_records(source_dir)

    assert [record["session_id"] for record in records] == ["10", "11"]
    first = records[0]
    assert first["question_id_dq"] == "104614"
    assert [turn["sequence_id"] for turn in first["transcript"]] == [1, 2]
    assert first["transcript"][0]["role"] == "Student"
    assert first["transcript"][1]["role"] == "Tutor"
    assert first["transcript"][1]["talk_move_prediction"] == "<Press for Accuracy>"
    assert first["transcript"][1]["annotations"] == []
    assert [item["Label"] for item in first["dq_question_metadata"]] == [
        "Question Text",
        "Answer A Text",
    ]
    assert [item["SubjectType"] for item in first["dialogue_subjects"]] == [
        "Subject",
        "Topic",
    ]


def test_generated_eedi_jsonl_upload_creates_expected_import_folder(client, tmp_path):
    module = _load_module()
    source_dir = tmp_path / "eedi"
    anchored_dir = source_dir / "anchored-dialogues"
    anchored_dir.mkdir(parents=True)
    output_path = tmp_path / "eedi_qatd_2k.jsonl"

    _write_csv(
        anchored_dir / "train.csv",
        [
            "InterventionId",
            "TutorId",
            "QuestionId_DQ",
            "MessageSequence",
            "IsTutor",
            "MessageString",
            "TalkMovePrediction",
        ],
        [
            {
                "InterventionId": "10",
                "TutorId": "749",
                "QuestionId_DQ": "104614",
                "MessageSequence": "1",
                "IsTutor": "0",
                "MessageString": "I need help",
                "TalkMovePrediction": "",
            },
            {
                "InterventionId": "10",
                "TutorId": "749",
                "QuestionId_DQ": "104614",
                "MessageSequence": "2",
                "IsTutor": "1",
                "MessageString": "Sure, let's solve it.",
                "TalkMovePrediction": "<Keep Together>",
            },
            {
                "InterventionId": "11",
                "TutorId": "801",
                "QuestionId_DQ": "200001",
                "MessageSequence": "1",
                "IsTutor": "1",
                "MessageString": "Welcome back.",
                "TalkMovePrediction": "<None>",
            },
        ],
    )
    _write_csv(
        anchored_dir / "test.csv",
        [
            "InterventionId",
            "TutorId",
            "QuestionId_DQ",
            "MessageSequence",
            "IsTutor",
            "MessageString",
            "TalkMovePrediction",
        ],
        [],
    )
    _write_csv(
        source_dir / "dq-question-metadata.csv",
        [
            "QuestionId_DQ",
            "InterventionId",
            "MetaDataId",
            "Text",
            "Sequence",
            "MetaDataTagId",
            "Label",
        ],
        [],
    )
    _write_csv(
        source_dir / "dialogue-subjects.csv",
        [
            "InterventionId",
            "SubjectId",
            "ParentSubjectId",
            "SubjectName",
            "SubjectLevel",
            "SubjectType",
        ],
        [],
    )

    records = module.build_eedi_qatd_records(source_dir)
    module.write_eedi_qatd_jsonl(records, output_path)

    response = client.post(
        "/api/documents/upload",
        files={"file": (output_path.name, output_path.read_bytes(), "application/json")},
    )
    assert response.status_code == 200, response.text

    merged_doc = response.json()
    assert merged_doc["filename"] == "eedi_qatd_2k.jsonl"
    assert "Student: I need help\nTutor: Sure, let's solve it." in merged_doc["raw_text"]
    assert "Tutor: Welcome back." in merged_doc["raw_text"]

    folder = client.get("/api/folders").json()[0]
    assert folder["name"] == "eedi_qatd_2k"
    folder_detail = client.get(f"/api/folders/{folder['id']}").json()
    assert len(folder_detail["doc_ids"]) == 2
    child_doc_id = folder_detail["doc_ids"][0]
    imported_doc = client.get(f"/api/documents/{child_doc_id}").json()
    assert imported_doc["filename"] == "eedi_qatd_2k.record-0001.json"
    assert imported_doc["raw_text"] == "Student: I need help\nTutor: Sure, let's solve it."
    assert imported_doc["pre_annotations"] == []
