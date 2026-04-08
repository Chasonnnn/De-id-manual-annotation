import importlib.util
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from server import (
    _methods_lab_cancel_events,
    _methods_lab_runs,
    _prompt_lab_cancel_events,
    _prompt_lab_runs,
    _session_docs,
    app,
)


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "word_problem_heavy_surrogate_import.py"
    )
    spec = importlib.util.spec_from_file_location(
        "word_problem_heavy_surrogate_import", module_path
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(autouse=True)
def clean_sessions(tmp_path, monkeypatch):
    test_sessions = tmp_path / "sessions"
    (test_sessions / "default" / "folders").mkdir(parents=True)
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


def _write_source_folder(sessions_dir: Path, folder_id: str, doc_ids: list[str]) -> None:
    folders_dir = sessions_dir / "default" / "folders"
    folders_dir.mkdir(parents=True, exist_ok=True)
    folder_record = {
        "id": folder_id,
        "name": "Word Problem Heavy 100",
        "kind": "manual",
        "doc_ids": doc_ids,
        "child_folder_ids": [],
        "created_at": "2026-04-07T00:00:00Z",
        "doc_display_names": {doc_id: f"Session {doc_id}" for doc_id in doc_ids},
    }
    (folders_dir / f"{folder_id}.json").write_text(json.dumps(folder_record))
    index_path = folders_dir / "_index.json"
    existing = []
    if index_path.exists():
        existing = json.loads(index_path.read_text())
    if folder_id not in existing:
        existing.append(folder_id)
    index_path.write_text(json.dumps(existing))


def _write_source_doc(
    sessions_dir: Path,
    doc_id: str,
    utterances: list[tuple[str, str]],
) -> None:
    """Write a minimal <doc_id>.source.json file."""
    parts: list[str] = []
    utt_records: list[dict] = []
    offset = 0
    for index, (speaker, text) in enumerate(utterances):
        line_prefix = f"{speaker}: "
        line_text = f"{line_prefix}{text}"
        if parts:
            offset += 1  # for \n
        line_start = offset
        content_start = line_start + len(line_prefix)
        content_end = content_start + len(text)
        parts.append(line_text)
        utt_records.append(
            {
                "speaker": speaker,
                "text": text,
                "global_start": content_start,
                "global_end": content_end,
            }
        )
        offset = line_start + len(line_text)
    raw_text = "\n".join(parts)
    doc = {
        "id": doc_id,
        "filename": f"{doc_id}.json",
        "format": "jsonl",
        "raw_text": raw_text,
        "utterances": utt_records,
        "pre_annotations": [],
        "label_set": [],
        "manual_annotations": [],
        "agent_annotations": [],
        "status": "pending",
    }
    (sessions_dir / "default" / f"{doc_id}.source.json").write_text(json.dumps(doc))


def test_apply_surrogates_replaces_all_occurrences(tmp_path):
    module = _load_module()
    entries = [
        {"original": "Parthipan", "label": "NAME", "surrogate": "Aarav"},
        {"original": "Nila", "label": "NAME", "surrogate": "Maya"},
    ]
    text = "Hi Parthipan, this is Nila and Parthipan again"
    new_text, spans = module.apply_surrogates_to_utterance(text, entries)
    assert new_text == "Hi Aarav, this is Maya and Aarav again"
    assert spans == [
        {"start": 3, "end": 8, "text": "Aarav", "pii_type": "NAME"},
        {"start": 18, "end": 22, "text": "Maya", "pii_type": "NAME"},
        {"start": 27, "end": 32, "text": "Aarav", "pii_type": "NAME"},
    ]


def test_apply_surrogates_surrogates_placeholder_tag(tmp_path):
    module = _load_module()
    entries = [
        {"original": "<LOCATION>", "label": "ADDRESS", "surrogate": "Clearview"},
    ]
    text = "I live near <LOCATION> downtown"
    new_text, spans = module.apply_surrogates_to_utterance(text, entries)
    assert new_text == "I live near Clearview downtown"
    assert spans == [
        {"start": 12, "end": 21, "text": "Clearview", "pii_type": "ADDRESS"},
    ]


def test_apply_surrogates_raises_on_overlap(tmp_path):
    module = _load_module()
    entries = [
        {"original": "Alex Smith", "label": "NAME", "surrogate": "Pat Jones"},
        {"original": "Smith", "label": "NAME", "surrogate": "Doe"},
    ]
    text = "Meet Alex Smith today"
    with pytest.raises(ValueError, match="[Oo]verlap"):
        module.apply_surrogates_to_utterance(text, entries)


def test_apply_surrogates_pii_leak_guard(tmp_path):
    module = _load_module()
    entries = [
        {"original": "Alex", "label": "NAME", "surrogate": "Alexandria"},
    ]
    text = "Hi Alex how are you"
    with pytest.raises(ValueError, match="leak"):
        module.apply_surrogates_to_utterance(text, entries)


def test_build_surrogate_record_from_source_doc(tmp_path):
    module = _load_module()
    source_doc = {
        "id": "abc_line5",
        "raw_text": "volunteer: Hi Parthipan\nstudent: My name is Nila",
        "utterances": [
            {"speaker": "volunteer", "text": "Hi Parthipan", "global_start": 11, "global_end": 23},
            {"speaker": "student", "text": "My name is Nila", "global_start": 33, "global_end": 48},
        ],
    }
    entries = [
        {"original": "Parthipan", "label": "NAME", "surrogate": "Aarav"},
        {"original": "Nila", "label": "NAME", "surrogate": "Maya"},
    ]
    record = module.build_surrogate_record_for_doc(source_doc, entries)
    assert record["session_id"] == "abc_line5"
    assert record["transcript"] == [
        {
            "session_id": "abc_line5",
            "sequence_id": 1,
            "role": "volunteer",
            "content": "Hi Aarav",
            "annotations": [
                {"start": 3, "end": 8, "text": "Aarav", "pii_type": "NAME"},
            ],
        },
        {
            "session_id": "abc_line5",
            "sequence_id": 2,
            "role": "student",
            "content": "My name is Maya",
            "annotations": [
                {"start": 11, "end": 15, "text": "Maya", "pii_type": "NAME"},
            ],
        },
    ]
    assert record["annotations"] == []


def test_user_label_to_canonical_mapping(tmp_path):
    module = _load_module()
    assert module._canonical_label("NAME") == "NAME"
    assert module._canonical_label("TUTORING_PROVIDER") == "TUTOR_PROVIDER"
    assert module._canonical_label("OTHER_LOCATION") == "OTHER_LOCATIONS_IDENTIFIED"
    assert module._canonical_label("OTHER_IDENTIFYING_NUMBER") == "IDENTIFYING_NUMBER"
    with pytest.raises(ValueError, match="Unknown PII label"):
        module._canonical_label("BOGUS")


def test_end_to_end_creates_new_folder(client, tmp_path):
    module = _load_module()
    sessions_dir = tmp_path / "sessions"
    _write_source_folder(sessions_dir, "fb8af36d", ["abc_line0", "abc_line1"])
    _write_source_doc(
        sessions_dir,
        "abc_line0",
        [
            ("volunteer", "Hi Parthipan"),
            ("student", "I am Nila"),
        ],
    )
    _write_source_doc(
        sessions_dir,
        "abc_line1",
        [
            ("volunteer", "Hello Karan"),
            ("student", "Hi from <LOCATION>"),
        ],
    )

    surrogate_map = {
        "docs": {
            "abc_line0": {
                "entries": [
                    {"original": "Parthipan", "label": "NAME", "surrogate": "Aarav"},
                    {"original": "Nila", "label": "NAME", "surrogate": "Maya"},
                ]
            },
            "abc_line1": {
                "entries": [
                    {"original": "Karan", "label": "NAME", "surrogate": "Devan"},
                    {"original": "<LOCATION>", "label": "ADDRESS", "surrogate": "Clearview"},
                ]
            },
        }
    }
    map_path = tmp_path / "map.json"
    map_path.write_text(json.dumps(surrogate_map))
    output_dir = tmp_path / "out"

    result = module.convert_and_import(
        source_folder_id="fb8af36d",
        surrogate_map_path=map_path,
        target_name="word_problem_heavy_100_surrogated",
        session_id="default",
        output_dir=output_dir,
    )

    assert result["record_count"] == 2
    assert result["folder_name"] == "word_problem_heavy_100_surrogated"
    assert result["label_counts"] == {"NAME": 3, "ADDRESS": 1}

    folders_resp = client.get("/api/folders")
    assert folders_resp.status_code == 200
    folders = folders_resp.json()
    names = {folder["name"]: folder for folder in folders}
    assert "word_problem_heavy_100_surrogated" in names
    new_folder = names["word_problem_heavy_100_surrogated"]
    assert new_folder["doc_count"] == 2

    detail = client.get(f"/api/folders/{new_folder['id']}").json()
    doc_ids = [item["id"] for item in detail["documents"]]
    assert len(doc_ids) == 2

    first_doc = client.get(f"/api/documents/{doc_ids[0]}").json()
    assert "Parthipan" not in first_doc["raw_text"]
    assert "Nila" not in first_doc["raw_text"]
    assert "Aarav" in first_doc["raw_text"]
    assert "Maya" in first_doc["raw_text"]
    assert sorted((s["label"], s["text"]) for s in first_doc["pre_annotations"]) == [
        ("NAME", "Aarav"),
        ("NAME", "Maya"),
    ]

    second_doc = client.get(f"/api/documents/{doc_ids[1]}").json()
    assert "<LOCATION>" not in second_doc["raw_text"]
    assert "Clearview" in second_doc["raw_text"]
    assert "Devan" in second_doc["raw_text"]
    labels = sorted((s["label"], s["text"]) for s in second_doc["pre_annotations"])
    assert labels == [("ADDRESS", "Clearview"), ("NAME", "Devan")]


def test_missing_doc_in_map_raises(tmp_path):
    module = _load_module()
    sessions_dir = tmp_path / "sessions"
    _write_source_folder(sessions_dir, "fb8af36d", ["abc_line0", "abc_line1"])
    _write_source_doc(sessions_dir, "abc_line0", [("volunteer", "Hi there")])
    _write_source_doc(sessions_dir, "abc_line1", [("student", "Hello")])

    # Map only covers one doc
    map_path = tmp_path / "map.json"
    map_path.write_text(json.dumps({"docs": {"abc_line0": {"entries": []}}}))
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="abc_line1"):
        module.convert_and_import(
            source_folder_id="fb8af36d",
            surrogate_map_path=map_path,
            target_name="word_problem_heavy_100_surrogated",
            session_id="default",
            output_dir=output_dir,
        )


def test_folder_name_collision_raises(tmp_path):
    module = _load_module()
    sessions_dir = tmp_path / "sessions"
    _write_source_folder(sessions_dir, "fb8af36d", ["abc_line0"])
    _write_source_doc(sessions_dir, "abc_line0", [("volunteer", "Hi there")])
    # Seed an existing folder with the target name
    _write_source_folder(sessions_dir, "deadbeef", [])
    # Rewrite its name to the target
    existing_path = sessions_dir / "default" / "folders" / "deadbeef.json"
    data = json.loads(existing_path.read_text())
    data["name"] = "word_problem_heavy_100_surrogated"
    existing_path.write_text(json.dumps(data))

    map_path = tmp_path / "map.json"
    map_path.write_text(json.dumps({"docs": {"abc_line0": {"entries": []}}}))
    output_dir = tmp_path / "out"

    with pytest.raises(ValueError, match="already exists"):
        module.convert_and_import(
            source_folder_id="fb8af36d",
            surrogate_map_path=map_path,
            target_name="word_problem_heavy_100_surrogated",
            session_id="default",
            output_dir=output_dir,
        )
