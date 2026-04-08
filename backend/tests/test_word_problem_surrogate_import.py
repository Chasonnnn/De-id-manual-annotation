import csv
import importlib.util
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from models import CanonicalDocument, CanonicalSpan, FolderRecord, UtteranceRow
from server import (
    app,
    _load_doc,
    _load_folder,
    _methods_lab_cancel_events,
    _methods_lab_runs,
    _prompt_lab_cancel_events,
    _prompt_lab_runs,
    _save_doc,
    _save_folder,
    _save_folder_index,
    _save_session_index,
    _save_sidecar,
    _session_docs,
)


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "word_problem_surrogate_import.py"
    )
    spec = importlib.util.spec_from_file_location("word_problem_surrogate_import", module_path)
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


def _build_doc(doc_id: str, turns: list[tuple[str, str]]) -> CanonicalDocument:
    raw_lines: list[str] = []
    utterances: list[UtteranceRow] = []
    offset = 0
    for index, (speaker, content) in enumerate(turns):
        line = f"{speaker}: {content}"
        if index > 0:
            offset += 1
        content_start = offset + len(f"{speaker}: ")
        content_end = content_start + len(content)
        raw_lines.append(line)
        utterances.append(
            UtteranceRow(
                speaker=speaker,
                text=content,
                global_start=content_start,
                global_end=content_end,
            )
        )
        offset += len(line)
    return CanonicalDocument(
        id=doc_id,
        filename=f"{doc_id}.json",
        format="jsonl",
        raw_text="\n".join(raw_lines),
        utterances=utterances,
        pre_annotations=[],
        label_set=[],
    )


def _make_source_folder(doc_ids: list[str]) -> FolderRecord:
    return FolderRecord(
        id="sourcefld",
        name="01_topic_1_prePII_noMoreThan_NAME_LOCATION",
        kind="import",
        doc_ids=doc_ids,
        created_at="2026-04-07T00:00:00Z",
        source_filename="01_topic_1_prePII_noMoreThan_NAME_LOCATION.jsonl",
        doc_display_names={doc_id: f"Session {index + 1}" for index, doc_id in enumerate(doc_ids)},
    )


def _write_ranking_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "doc_id",
                "csv_filename",
                "score",
                "explicit_word_problem_hits",
                "question_stem_hits",
                "total_left_hits",
                "rate_ratio_hits",
                "equation_prompt_hits",
                "object_noun_hits",
                "quoted_problem_hits",
                "number_hits",
                "question_hits",
                "preview",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _read_mapping_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_build_dataset_selects_first_eligible_docs_with_backfill_and_conservative_union_filter(tmp_path):
    module = _load_module()

    docs = {
        "doc-a": _build_doc("doc-a", [("volunteer", "hi <PERSON>")]),
        "doc-b": _build_doc("doc-b", [("student", "hi amina")]),
        "doc-c": _build_doc("doc-c", [("student", "i need a job now")]),
        "doc-d": _build_doc("doc-d", [("student", "from <LOCATION>")]),
        "doc-e": _build_doc("doc-e", [("student", "plain algebra chat")]),
    }
    for doc in docs.values():
        _save_doc(doc)

    _save_sidecar(
        "doc-b",
        "agent.method.deid_pipeline_union",
        [CanonicalSpan(start=12, end=17, label="NAME", text="amina")],
    )
    _save_sidecar(
        "doc-c",
        "agent.method.deid_pipeline_union",
        [CanonicalSpan(start=17, end=20, label="ADDRESS", text="job")],
    )

    source_folder = _make_source_folder(list(docs))
    _save_folder(source_folder)
    _save_folder_index([source_folder.id])
    _save_session_index()

    ranking_path = tmp_path / "ranking.csv"
    _write_ranking_csv(
        ranking_path,
        [
            {"doc_id": "doc-a", "csv_filename": "1.csv", "score": "9", "explicit_word_problem_hits": "0", "question_stem_hits": "0", "total_left_hits": "0", "rate_ratio_hits": "0", "equation_prompt_hits": "0", "object_noun_hits": "0", "quoted_problem_hits": "0", "number_hits": "0", "question_hits": "0", "preview": "a"},
            {"doc_id": "doc-b", "csv_filename": "2.csv", "score": "8", "explicit_word_problem_hits": "0", "question_stem_hits": "0", "total_left_hits": "0", "rate_ratio_hits": "0", "equation_prompt_hits": "0", "object_noun_hits": "0", "quoted_problem_hits": "0", "number_hits": "0", "question_hits": "0", "preview": "b"},
            {"doc_id": "doc-c", "csv_filename": "3.csv", "score": "7", "explicit_word_problem_hits": "0", "question_stem_hits": "0", "total_left_hits": "0", "rate_ratio_hits": "0", "equation_prompt_hits": "0", "object_noun_hits": "0", "quoted_problem_hits": "0", "number_hits": "0", "question_hits": "0", "preview": "c"},
            {"doc_id": "doc-d", "csv_filename": "4.csv", "score": "6", "explicit_word_problem_hits": "0", "question_stem_hits": "0", "total_left_hits": "0", "rate_ratio_hits": "0", "equation_prompt_hits": "0", "object_noun_hits": "0", "quoted_problem_hits": "0", "number_hits": "0", "question_hits": "0", "preview": "d"},
            {"doc_id": "doc-e", "csv_filename": "5.csv", "score": "5", "explicit_word_problem_hits": "0", "question_stem_hits": "0", "total_left_hits": "0", "rate_ratio_hits": "0", "equation_prompt_hits": "0", "object_noun_hits": "0", "quoted_problem_hits": "0", "number_hits": "0", "question_hits": "0", "preview": "e"},
        ],
    )

    dataset = module.build_word_problem_surrogate_dataset(
        ranking_path=ranking_path,
        target_count=3,
    )

    assert [item["original_doc_id"] for item in dataset["selected_docs"]] == [
        "doc-a",
        "doc-b",
        "doc-d",
    ]
    assert dataset["summary"]["skipped_only_false_positive_method"] == 1
    assert dataset["summary"]["skipped_no_signal"] == 1


def test_build_dataset_replaces_placeholders_and_keeps_repeated_entity_surrogates_stable(tmp_path):
    module = _load_module()

    doc = _build_doc(
        "doc-sur",
        [
            ("volunteer", "hi <PERSON>"),
            ("volunteer", "can you try that <PERSON> in <LOCATION>?"),
            ("student", "my friend Amina said hi"),
        ],
    )
    _save_doc(doc)
    amina_start = doc.raw_text.index("Amina")
    _save_sidecar(
        "doc-sur",
        "agent.method.deid_pipeline_union",
        [CanonicalSpan(start=amina_start, end=amina_start + 5, label="NAME", text="Amina")],
    )

    source_folder = _make_source_folder(["doc-sur"])
    _save_folder(source_folder)
    _save_folder_index([source_folder.id])
    _save_session_index()

    ranking_path = tmp_path / "ranking.csv"
    _write_ranking_csv(
        ranking_path,
        [
            {"doc_id": "doc-sur", "csv_filename": "10.csv", "score": "10", "explicit_word_problem_hits": "0", "question_stem_hits": "0", "total_left_hits": "0", "rate_ratio_hits": "0", "equation_prompt_hits": "0", "object_noun_hits": "0", "quoted_problem_hits": "0", "number_hits": "0", "question_hits": "0", "preview": "sur"},
        ],
    )

    dataset = module.build_word_problem_surrogate_dataset(
        ranking_path=ranking_path,
        target_count=1,
    )

    record = dataset["records"][0]
    transcript = record["transcript"]
    assert "<PERSON>" not in transcript[0]["content"]
    assert "<PERSON>" not in transcript[1]["content"]
    assert "<LOCATION>" not in transcript[1]["content"]
    first_name = transcript[0]["annotations"][0]["text"]
    second_name = transcript[1]["annotations"][0]["text"]
    assert first_name == second_name
    assert transcript[2]["annotations"][0]["pii_type"] == "NAME"
    assert transcript[2]["annotations"][0]["text"] != "Amina"


def test_build_dataset_varies_same_placeholder_pattern_across_documents(tmp_path):
    module = _load_module()

    doc_a = _build_doc("doc-a", [("volunteer", "hi <PERSON>")])
    doc_b = _build_doc("doc-b", [("volunteer", "hi <PERSON>")])
    _save_doc(doc_a)
    _save_doc(doc_b)

    source_folder = _make_source_folder(["doc-a", "doc-b"])
    _save_folder(source_folder)
    _save_folder_index([source_folder.id])
    _save_session_index()

    ranking_path = tmp_path / "ranking.csv"
    _write_ranking_csv(
        ranking_path,
        [
            {"doc_id": "doc-a", "csv_filename": "201.csv", "score": "10", "explicit_word_problem_hits": "0", "question_stem_hits": "0", "total_left_hits": "0", "rate_ratio_hits": "0", "equation_prompt_hits": "0", "object_noun_hits": "0", "quoted_problem_hits": "0", "number_hits": "0", "question_hits": "0", "preview": "a"},
            {"doc_id": "doc-b", "csv_filename": "202.csv", "score": "9", "explicit_word_problem_hits": "0", "question_stem_hits": "0", "total_left_hits": "0", "rate_ratio_hits": "0", "equation_prompt_hits": "0", "object_noun_hits": "0", "quoted_problem_hits": "0", "number_hits": "0", "question_hits": "0", "preview": "b"},
        ],
    )

    dataset = module.build_word_problem_surrogate_dataset(
        ranking_path=ranking_path,
        target_count=2,
    )

    first_name = dataset["records"][0]["transcript"][0]["annotations"][0]["text"]
    second_name = dataset["records"][1]["transcript"][0]["annotations"][0]["text"]
    assert first_name != second_name


def test_convert_and_import_creates_folder_mapping_and_empty_manual_annotations(client, tmp_path):
    module = _load_module()

    doc_a = _build_doc("doc-a", [("volunteer", "hi <PERSON>")])
    doc_b = _build_doc("doc-b", [("student", "hello lance")])
    _save_doc(doc_a)
    _save_doc(doc_b)
    _save_sidecar(
        "doc-b",
        "agent.method.deid_pipeline_union",
        [CanonicalSpan(start=15, end=20, label="NAME", text="lance")],
    )

    source_folder = _make_source_folder(["doc-a", "doc-b"])
    _save_folder(source_folder)
    _save_folder_index([source_folder.id])
    _save_session_index()

    ranking_path = tmp_path / "ranking.csv"
    output_path = tmp_path / "Word Problem Heavy 100 Surrogate.jsonl"
    mapping_path = tmp_path / "Word Problem Heavy 100 Surrogate_doc_mapping.csv"
    _write_ranking_csv(
        ranking_path,
        [
            {"doc_id": "doc-a", "csv_filename": "101.csv", "score": "10", "explicit_word_problem_hits": "0", "question_stem_hits": "0", "total_left_hits": "0", "rate_ratio_hits": "0", "equation_prompt_hits": "0", "object_noun_hits": "0", "quoted_problem_hits": "0", "number_hits": "0", "question_hits": "0", "preview": "a"},
            {"doc_id": "doc-b", "csv_filename": "102.csv", "score": "9", "explicit_word_problem_hits": "0", "question_stem_hits": "0", "total_left_hits": "0", "rate_ratio_hits": "0", "equation_prompt_hits": "0", "object_noun_hits": "0", "quoted_problem_hits": "0", "number_hits": "0", "question_hits": "0", "preview": "b"},
        ],
    )

    result = module.build_and_import_word_problem_surrogates(
        ranking_path=ranking_path,
        output_path=output_path,
        mapping_path=mapping_path,
        target_count=2,
    )

    assert result["selected_count"] == 2
    assert result["imported_folder_name"] == "Word Problem Heavy 100 Surrogate"
    assert mapping_path.exists()

    folders_resp = client.get("/api/folders")
    assert folders_resp.status_code == 200
    folders = folders_resp.json()
    assert len(folders) == 2

    imported_folder = next(
        folder for folder in folders if folder["name"] == "Word Problem Heavy 100 Surrogate"
    )
    detail_resp = client.get(f"/api/folders/{imported_folder['id']}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert len(detail["documents"]) == 2

    for item in detail["documents"]:
        doc = client.get(f"/api/documents/{item['id']}").json()
        assert "<PERSON>" not in doc["raw_text"]
        assert doc["manual_annotations"] == []
        assert doc["pre_annotations"]

    mapping_rows = _read_mapping_csv(mapping_path)
    assert len(mapping_rows) == 2
    assert {row["original_doc_id"] for row in mapping_rows} == {"doc-a", "doc-b"}
    assert {row["original_csv_filename"] for row in mapping_rows} == {"101.csv", "102.csv"}

    original_doc = _load_doc("doc-a")
    assert original_doc is not None
    assert "<PERSON>" in original_doc.raw_text

    folder_record = _load_folder(imported_folder["id"])
    assert folder_record is not None
    assert folder_record.source_folder_id == source_folder.id
