from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server import _load_all_folders, _upload_document_payload  # noqa: E402


IMPORT_FOLDER_NAME = "thirdspacelearning"
DEFAULT_TRANSCRIPTS_CSV = (
    ROOT.parent / "Thirdspace" / "DeID_GT_ThirdSpaceLearning_math_481_transcripts.csv"
)
DEFAULT_INDICES_CSV = (
    ROOT.parent / "Thirdspace" / "DeID_GT_ThirdSpaceLearning_math_481_transcripts_indices.csv"
)
DEFAULT_OUTPUT_PATH = ROOT.parent / "Thirdspace" / f"{IMPORT_FOLDER_NAME}.jsonl"
DEFAULT_SESSION_ID = "default"

THIRDSPACE_LABEL_MAP: dict[str, str] = {
    "NAME": "NAME",
    "TUTORING_PROVIDER": "TUTOR_PROVIDER",
    "ADDRESS": "ADDRESS",
    "OTHER_LOCATION": "OTHER_LOCATIONS_IDENTIFIED",
    "AGE": "AGE",
    "DATE": "DATE",
}


def _require_int(value: str | int | None, *, field_name: str) -> int:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError(f"Missing required integer field '{field_name}'.")
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid integer for '{field_name}': {raw!r}") from exc


def _require_text(value: str | None, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Missing required field '{field_name}'.")
    return text


def _normalize_role(speaker: str | None) -> str:
    role = str(speaker or "").strip()
    return role or "unknown"


def _map_label(raw_label: str | None) -> str:
    normalized = str(raw_label or "").strip().upper()
    mapped = THIRDSPACE_LABEL_MAP.get(normalized)
    if mapped is None:
        raise ValueError(f"Unsupported ThirdSpace pii_type: {raw_label!r}")
    return mapped


def build_thirdspace_records(
    transcript_csv_path: Path | str,
    index_csv_path: Path | str,
) -> list[dict[str, Any]]:
    transcript_path = Path(transcript_csv_path)
    indices_path = Path(index_csv_path)

    sessions: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    turns_by_key: dict[tuple[str, int], dict[str, Any]] = {}

    with transcript_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            session_id = _require_text(row.get("session_id"), field_name="session_id")
            sequence_id = _require_int(row.get("sequence_id"), field_name="sequence_id")
            key = (session_id, sequence_id)
            if key in turns_by_key:
                raise ValueError(
                    f"Duplicate transcript row for session_id={session_id!r} sequence_id={sequence_id}."
                )
            turn = {
                "session_id": session_id,
                "sequence_id": sequence_id,
                "role": _normalize_role(row.get("speaker")),
                "start_time": str(row.get("start_time") or ""),
                "end_time": str(row.get("end_time") or ""),
                "content": str(row.get("content") or ""),
                "annotations": [],
            }
            sessions.setdefault(session_id, []).append(turn)
            turns_by_key[key] = turn

    with indices_path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            session_id = _require_text(row.get("session_id"), field_name="session_id")
            sequence_id = _require_int(row.get("sequence_id"), field_name="sequence_id")
            key = (session_id, sequence_id)
            turn = turns_by_key.get(key)
            if turn is None:
                raise ValueError(
                    f"Missing transcript row for annotation session_id={session_id!r} sequence_id={sequence_id}."
                )

            transcript_content = str(turn["content"])
            annotation_content = str(row.get("content") or "")
            if transcript_content != annotation_content:
                raise ValueError(
                    "Annotation content does not match transcript content for "
                    f"session_id={session_id!r} sequence_id={sequence_id}."
                )

            start = _require_int(row.get("starting index"), field_name="starting index")
            end = _require_int(row.get("ending index"), field_name="ending index")
            if start < 0 or end > len(transcript_content) or start >= end:
                raise ValueError(
                    "Invalid annotation bounds for "
                    f"session_id={session_id!r} sequence_id={sequence_id}: [{start}, {end})."
                )

            turn["annotations"].append(
                {
                    "start": start,
                    "end": end,
                    "text": transcript_content[start:end],
                    "pii_type": _map_label(row.get("pii_type")),
                }
            )

    records: list[dict[str, Any]] = []
    for session_id, turns in sessions.items():
        ordered_turns = sorted(turns, key=lambda turn: int(turn["sequence_id"]))
        records.append(
            {
                "session_id": session_id,
                "transcript": ordered_turns,
                "annotations": [],
            }
        )
    return records


def write_thirdspace_jsonl(records: list[dict[str, Any]], output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
    return path


def import_thirdspace_jsonl(
    output_path: Path | str,
    *,
    session_id: str = DEFAULT_SESSION_ID,
):
    path = Path(output_path)
    filename = path.name
    if filename != f"{IMPORT_FOLDER_NAME}.jsonl":
        raise ValueError(
            f"Expected output filename '{IMPORT_FOLDER_NAME}.jsonl', got {filename!r}."
        )

    existing_names = {str(folder.name or "").strip().casefold() for folder in _load_all_folders(session_id)}
    if IMPORT_FOLDER_NAME.casefold() in existing_names:
        raise ValueError(
            f"Folder '{IMPORT_FOLDER_NAME}' already exists in session '{session_id}'."
        )

    return _upload_document_payload(path.read_bytes(), filename, session_id)


def convert_and_import_thirdspace(
    *,
    transcript_csv_path: Path | str = DEFAULT_TRANSCRIPTS_CSV,
    index_csv_path: Path | str = DEFAULT_INDICES_CSV,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    session_id: str = DEFAULT_SESSION_ID,
) -> dict[str, Any]:
    records = build_thirdspace_records(transcript_csv_path, index_csv_path)
    output = write_thirdspace_jsonl(records, output_path)
    imported_doc = import_thirdspace_jsonl(output, session_id=session_id)
    return {
        "session_count": len(records),
        "output_path": str(output),
        "imported_doc_id": imported_doc.id,
        "imported_filename": imported_doc.filename,
        "session_id": session_id,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert ThirdSpace CSV transcripts into app JSONL and import them."
    )
    parser.add_argument("--transcripts", default=str(DEFAULT_TRANSCRIPTS_CSV))
    parser.add_argument("--indices", default=str(DEFAULT_INDICES_CSV))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--session", default=DEFAULT_SESSION_ID)
    args = parser.parse_args(argv)

    result = convert_and_import_thirdspace(
        transcript_csv_path=Path(args.transcripts),
        index_csv_path=Path(args.indices),
        output_path=Path(args.output),
        session_id=str(args.session or DEFAULT_SESSION_ID),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
