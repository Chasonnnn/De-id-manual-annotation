from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server import _load_all_folders, _upload_document_payload  # noqa: E402


DEFAULT_OUTPUT_DIR = ROOT / "output"
DEFAULT_OUTPUT_BASENAME = "eedi_question_anchored_tutoring_dialogues_2k.jsonl"
DEFAULT_SESSION_ID = "default"


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise ValueError(f"Missing required CSV file: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sort_int_text(value: str | None) -> tuple[int, str]:
    text = str(value or "").strip()
    if text.lstrip("-").isdigit():
        return (0, f"{int(text):012d}")
    return (1, text)


def _normalize_role(is_tutor: str | None) -> str:
    text = str(is_tutor or "").strip()
    if text == "1":
        return "Tutor"
    if text == "0":
        return "Student"
    return "unknown"


def build_eedi_qatd_records(source_dir: Path | str) -> list[dict[str, Any]]:
    root = Path(source_dir)
    anchored_train = _read_csv_rows(root / "anchored-dialogues" / "train.csv")
    anchored_test = _read_csv_rows(root / "anchored-dialogues" / "test.csv")
    question_rows = _read_csv_rows(root / "dq-question-metadata.csv")
    subject_rows = _read_csv_rows(root / "dialogue-subjects.csv")

    messages_by_intervention: dict[str, list[dict[str, str]]] = {}
    for row in [*anchored_train, *anchored_test]:
        intervention_id = str(row.get("InterventionId") or "").strip()
        if not intervention_id:
            raise ValueError("anchored-dialogues rows must include InterventionId")
        messages_by_intervention.setdefault(intervention_id, []).append(dict(row))

    question_by_intervention: dict[str, list[dict[str, str]]] = {}
    for row in question_rows:
        intervention_id = str(row.get("InterventionId") or "").strip()
        if not intervention_id:
            continue
        question_by_intervention.setdefault(intervention_id, []).append(dict(row))

    subjects_by_intervention: dict[str, list[dict[str, str]]] = {}
    for row in subject_rows:
        intervention_id = str(row.get("InterventionId") or "").strip()
        if not intervention_id:
            continue
        subjects_by_intervention.setdefault(intervention_id, []).append(dict(row))

    records: list[dict[str, Any]] = []
    for intervention_id in sorted(messages_by_intervention, key=lambda item: _sort_int_text(item)):
        rows = messages_by_intervention[intervention_id]
        ordered_rows = sorted(
            rows,
            key=lambda row: (
                _sort_int_text(row.get("MessageSequence")),
                _sort_int_text(row.get("TutorId")),
                str(row.get("MessageString") or ""),
            ),
        )
        question_id_dq = str(ordered_rows[0].get("QuestionId_DQ") or "").strip()
        transcript: list[dict[str, Any]] = []
        for row in ordered_rows:
            sequence_text = str(row.get("MessageSequence") or "").strip()
            sequence_id = int(sequence_text) if sequence_text.lstrip("-").isdigit() else None
            talk_move = str(row.get("TalkMovePrediction") or "").strip()
            transcript.append(
                {
                    "session_id": intervention_id,
                    "sequence_id": sequence_id,
                    "role": _normalize_role(row.get("IsTutor")),
                    "content": str(row.get("MessageString") or ""),
                    "is_tutor": str(row.get("IsTutor") or "").strip(),
                    "tutor_id": str(row.get("TutorId") or "").strip(),
                    "question_id_dq": question_id_dq,
                    "talk_move_prediction": talk_move or None,
                    "annotations": [],
                }
            )

        question_metadata = sorted(
            question_by_intervention.get(intervention_id, []),
            key=lambda row: (
                _sort_int_text(row.get("Sequence")),
                _sort_int_text(row.get("MetaDataId")),
            ),
        )
        subject_metadata = sorted(
            subjects_by_intervention.get(intervention_id, []),
            key=lambda row: (
                _sort_int_text(row.get("SubjectLevel")),
                _sort_int_text(row.get("SubjectId")),
            ),
        )
        records.append(
            {
                "session_id": intervention_id,
                "question_id_dq": question_id_dq,
                "transcript": transcript,
                "dq_question_metadata": question_metadata,
                "dialogue_subjects": subject_metadata,
                "annotations": [],
            }
        )

    if not records:
        raise ValueError(f"No Eedi dialogue records found in {root}")
    return records


def write_eedi_qatd_jsonl(records: list[dict[str, Any]], output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
    return path


def import_eedi_qatd_jsonl(output_path: Path | str, *, session_id: str = DEFAULT_SESSION_ID):
    path = Path(output_path)
    if path.suffix.casefold() != ".jsonl":
        raise ValueError(f"Expected a .jsonl file, got {path.name!r}.")

    folder_name = path.stem
    existing_names = {
        str(folder.name or "").strip().casefold() for folder in _load_all_folders(session_id)
    }
    if folder_name.casefold() in existing_names:
        raise ValueError(f"Folder '{folder_name}' already exists in session '{session_id}'.")

    return _upload_document_payload(path.read_bytes(), path.name, session_id)


def convert_and_import_eedi_qatd(
    *,
    source_dir: Path | str,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    output_name: str = DEFAULT_OUTPUT_BASENAME,
    session_id: str = DEFAULT_SESSION_ID,
) -> dict[str, Any]:
    source = Path(source_dir)
    records = build_eedi_qatd_records(source)
    output_path = Path(output_dir) / output_name
    output = write_eedi_qatd_jsonl(records, output_path)
    imported_doc = import_eedi_qatd_jsonl(output, session_id=session_id)
    return {
        "source_dir": str(source),
        "record_count": len(records),
        "output_path": str(output),
        "folder_name": output.stem,
        "imported_doc_id": imported_doc.id,
        "imported_filename": imported_doc.filename,
        "session_id": session_id,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert the Eedi Question-Anchored Tutoring Dialogues CSV bundle into "
            "dialogue-level JSONL records compatible with the annotation tool and import them."
        )
    )
    parser.add_argument("source_dir", help="Directory containing the Eedi CSV bundle.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_BASENAME)
    parser.add_argument("--session", default=DEFAULT_SESSION_ID)
    args = parser.parse_args(argv)

    result = convert_and_import_eedi_qatd(
        source_dir=Path(args.source_dir),
        output_dir=Path(args.output_dir),
        output_name=str(args.output_name or DEFAULT_OUTPUT_BASENAME),
        session_id=str(args.session or DEFAULT_SESSION_ID),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
