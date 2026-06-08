import importlib.util
import json
from pathlib import Path
import zipfile

import pytest


def _load_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "evaluate_full_dialogue_redactions.py"
    )
    spec = importlib.util.spec_from_file_location(
        "evaluate_full_dialogue_redactions", module_path
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row))
            handle.write("\n")


def test_scores_completion_predictions_by_occurrence(tmp_path):
    module = _load_module()
    transcript = "Tutor: Hi Jamie.\nStudent: Jamie is here."
    gold_path = tmp_path / "gold.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"

    _write_jsonl(
        gold_path,
        [
            {
                "dialogue_id": "d1",
                "provider": "upchieve_math",
                "transcript_text": transcript,
                "gold_spans": [
                    {
                        "start": transcript.index("Jamie"),
                        "end": transcript.index("Jamie") + len("Jamie"),
                        "text": "Jamie",
                        "entity_type": "NAME",
                    }
                ],
            }
        ],
    )
    _write_jsonl(
        predictions_path,
        [
            {
                "dialogue_id": "d1",
                "completion": json.dumps(
                    {"redact_spans": [{"text": "Jamie", "occurrence": 1}]}
                ),
            }
        ],
    )

    report = module.evaluate_files(gold_path, predictions_path, match_mode="exact")

    assert report["overall"]["row_count"] == 1
    assert report["overall"]["micro"] == {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "tp": 1,
        "fp": 0,
        "fn": 0,
    }
    assert report["by_provider"]["upchieve_math"]["micro"]["f1"] == 1.0
    assert report["by_entity_type"]["NAME"]["recall"] == 1.0


def test_paper_overlap_mode_counts_partial_boundary_match(tmp_path):
    module = _load_module()
    transcript = "Tutor: Hello Amelia."
    gold_path = tmp_path / "gold.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    start = transcript.index("Amelia")

    _write_jsonl(
        gold_path,
        [
            {
                "dialogue_id": "d1",
                "provider": "thirdspace",
                "transcript_text": transcript,
                "gold_spans": [
                    {
                        "start": start,
                        "end": start + len("Amelia"),
                        "text": "Amelia",
                        "entity_type": "NAME",
                    }
                ],
            }
        ],
    )
    _write_jsonl(
        predictions_path,
        [
            {
                "dialogue_id": "d1",
                "redact_spans": [
                    {
                        "start": start,
                        "end": start + len("Ameli"),
                        "text": "Ameli",
                    }
                ],
            }
        ],
    )

    exact = module.evaluate_files(gold_path, predictions_path, match_mode="exact")
    paper_overlap = module.evaluate_files(
        gold_path,
        predictions_path,
        match_mode="paper-overlap",
    )

    assert exact["overall"]["micro"]["f1"] == 0.0
    assert paper_overlap["overall"]["micro"]["f1"] == 1.0


def test_paper_overlap_matches_final_scorer_any_overlap_semantics(tmp_path):
    module = _load_module()
    transcript = "Tutor: Hello Amelia."
    gold_path = tmp_path / "gold.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"
    start = transcript.index("Amelia")

    _write_jsonl(
        gold_path,
        [
            {
                "dialogue_id": "d1",
                "provider": "thirdspace",
                "transcript_text": transcript,
                "gold_spans": [
                    {
                        "start": start,
                        "end": start + len("Amelia"),
                        "text": "Amelia",
                        "entity_type": "NAME",
                    }
                ],
            }
        ],
    )
    _write_jsonl(
        predictions_path,
        [
            {
                "dialogue_id": "d1",
                "redact_spans": [
                    {
                        "start": start,
                        "end": start + 1,
                        "text": "A",
                    }
                ],
            }
        ],
    )

    paper_overlap = module.evaluate_files(gold_path, predictions_path)
    iou = module.evaluate_files(
        gold_path,
        predictions_path,
        match_mode="iou",
        overlap_threshold=0.5,
    )

    assert paper_overlap["match_mode"] == "paper-overlap"
    assert paper_overlap["overall"]["micro"]["f1"] == 1.0
    assert iou["overall"]["micro"]["f1"] == 0.0


def test_reports_unresolved_text_occurrence_predictions(tmp_path):
    module = _load_module()
    gold_path = tmp_path / "gold.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"

    _write_jsonl(
        gold_path,
        [
            {
                "dialogue_id": "d1",
                "provider": "upchieve_math",
                "transcript_text": "Tutor: Hello.",
                "gold_spans": [],
            }
        ],
    )
    _write_jsonl(
        predictions_path,
        [{"dialogue_id": "d1", "redact_spans": [{"text": "Missing", "occurrence": 1}]}],
    )

    report = module.evaluate_files(gold_path, predictions_path)

    assert report["overall"]["unresolved_prediction_count"] == 1
    assert report["unresolved_predictions"][0]["reason"] == "occurrence_not_found"


def test_loads_gold_from_zip_member(tmp_path):
    module = _load_module()
    archive_path = tmp_path / "export.zip"
    member = "latest_refreshed_testing_sets/combined_test_200.jsonl"
    row = {
        "dialogue_id": "d1",
        "provider": "upchieve_math",
        "transcript_text": "Tutor: Hello.",
        "gold_spans": [],
    }
    with zipfile.ZipFile(archive_path, "w") as zf:
        zf.writestr(member, json.dumps(row) + "\n")

    rows = module.load_jsonl(f"{archive_path}::{member}")

    assert rows == [row]


def test_rejects_duplicate_prediction_dialogue_ids(tmp_path):
    module = _load_module()
    gold_path = tmp_path / "gold.jsonl"
    predictions_path = tmp_path / "predictions.jsonl"

    _write_jsonl(
        gold_path,
        [
            {
                "dialogue_id": "d1",
                "provider": "upchieve_math",
                "transcript_text": "Tutor: Hello.",
                "gold_spans": [],
            }
        ],
    )
    _write_jsonl(
        predictions_path,
        [
            {"dialogue_id": "d1", "redact_spans": []},
            {"dialogue_id": "d1", "redact_spans": []},
        ],
    )

    with pytest.raises(ValueError, match="duplicate prediction row"):
        module.evaluate_files(gold_path, predictions_path)
