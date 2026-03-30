from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from collections import OrderedDict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from normalizer import parse_jsonl_record  # noqa: E402
from server import (  # noqa: E402
    _load_all_folders,
    _load_doc,
    _now_iso,
    _persist_manual_annotations,
    _save_doc,
    _upload_document_payload,
)


IMPORT_FOLDER_NAME = "thirdspacelearning"
DEFAULT_TRANSCRIPTS_CSV = (
    ROOT.parent / "Thirdspace" / "DeID_GT_ThirdSpaceLearning_math_481_transcripts.csv"
)
DEFAULT_INDICES_CSV = (
    ROOT.parent / "Thirdspace" / "DeID_GT_ThirdSpaceLearning_math_481_transcripts_indices.csv"
)
DEFAULT_OUTPUT_PATH = ROOT.parent / "Thirdspace" / f"{IMPORT_FOLDER_NAME}.jsonl"
DEFAULT_UPDATED_TRANSCRIPTS_CSV = Path(
    "/Users/chason/Downloads/drive-download-20260324T200607Z-1-001/"
    "DeID_GT_ThirdSpaceLearning_math_481_transcripts.csv"
)
DEFAULT_UPDATED_INDICES_CSV = Path(
    "/Users/chason/Downloads/drive-download-20260324T200607Z-1-001/"
    "DeID_GT_ThirdSpaceLearning_math_481_transcripts_indices_updated.csv"
)
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


def _records_by_session(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_session: dict[str, dict[str, Any]] = {}
    for record in records:
        session_id = _require_text(record.get("session_id"), field_name="session_id")
        if session_id in by_session:
            raise ValueError(f"Duplicate ThirdSpace record for session_id={session_id!r}.")
        by_session[session_id] = record
    return by_session


def _load_existing_import_folder(
    *,
    session_id: str = DEFAULT_SESSION_ID,
):
    matching = [
        folder
        for folder in _load_all_folders(session_id)
        if str(folder.name or "").strip().casefold() == IMPORT_FOLDER_NAME.casefold()
    ]
    if not matching:
        raise ValueError(
            f"Folder '{IMPORT_FOLDER_NAME}' was not found in session '{session_id}'."
        )
    if len(matching) > 1:
        raise ValueError(
            f"Multiple folders named '{IMPORT_FOLDER_NAME}' exist in session '{session_id}'."
        )
    return matching[0]


def _session_id_from_display_name(display_name: str) -> str:
    text = str(display_name or "").strip()
    prefix = "Session "
    if not text.startswith(prefix):
        raise ValueError(
            f"Expected ThirdSpace display name to start with {prefix!r}, got {display_name!r}."
        )
    session_id = text[len(prefix) :].strip()
    if not session_id:
        raise ValueError(f"Missing session id in display name {display_name!r}.")
    return session_id


def sync_existing_thirdspace_folder(
    *,
    transcript_csv_path: Path | str = DEFAULT_UPDATED_TRANSCRIPTS_CSV,
    index_csv_path: Path | str = DEFAULT_UPDATED_INDICES_CSV,
    session_id: str = DEFAULT_SESSION_ID,
) -> dict[str, Any]:
    records = build_thirdspace_records(transcript_csv_path, index_csv_path)
    records_by_session = _records_by_session(records)
    folder = _load_existing_import_folder(session_id=session_id)

    folder_doc_ids = list(folder.doc_ids)
    if len(folder_doc_ids) != len(records_by_session):
        raise ValueError(
            "Existing ThirdSpace folder doc count does not match rebuilt session count: "
            f"{len(folder_doc_ids)} != {len(records_by_session)}."
        )

    updated_doc_ids: list[str] = []
    updated_at = _now_iso()
    label_counts: Counter[str] = Counter()

    for doc_id in folder_doc_ids:
        display_name = folder.doc_display_names.get(doc_id)
        if display_name is None:
            raise ValueError(f"Missing display name for ThirdSpace doc {doc_id!r}.")
        session_key = _session_id_from_display_name(display_name)
        record = records_by_session.pop(session_key, None)
        if record is None:
            raise ValueError(
                f"No updated ThirdSpace record found for session_id={session_key!r}."
            )

        existing_doc = _load_doc(doc_id, session_id)
        if existing_doc is None:
            raise ValueError(f"Missing existing ThirdSpace document {doc_id!r}.")

        rebuilt_doc = parse_jsonl_record(record, existing_doc.filename, existing_doc.id)
        if rebuilt_doc.raw_text != existing_doc.raw_text:
            raise ValueError(
                f"Transcript mismatch for existing ThirdSpace document {doc_id!r}."
            )

        synced_spans = list(rebuilt_doc.pre_annotations)
        for span in synced_spans:
            label_counts[span.label] += 1

        _save_doc(
            existing_doc.model_copy(
                update={
                    "pre_annotations": synced_spans,
                    "manual_annotations": synced_spans,
                    "label_set": list(rebuilt_doc.label_set),
                }
            ),
            session_id,
        )
        _persist_manual_annotations(
            doc_id,
            synced_spans,
            session_id,
            updated_at=updated_at,
        )
        updated_doc_ids.append(doc_id)

    if records_by_session:
        leftover = sorted(records_by_session)[:5]
        raise ValueError(
            "Updated ThirdSpace CSV contains sessions that are not present in the existing "
            f"folder, for example: {leftover}"
        )

    return {
        "folder_id": folder.id,
        "folder_name": folder.name,
        "processed_count": len(updated_doc_ids),
        "updated_doc_ids": updated_doc_ids,
        "label_counts": dict(sorted(label_counts.items())),
        "session_id": session_id,
    }


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
        description="Convert or sync ThirdSpace CSV transcripts in the annotation tool."
    )
    parser.add_argument("--sync-existing", action="store_true")
    parser.add_argument("--transcripts", default=str(DEFAULT_TRANSCRIPTS_CSV))
    parser.add_argument("--indices", default=str(DEFAULT_INDICES_CSV))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument(
        "--updated-transcripts",
        default=str(DEFAULT_UPDATED_TRANSCRIPTS_CSV),
    )
    parser.add_argument(
        "--updated-indices",
        default=str(DEFAULT_UPDATED_INDICES_CSV),
    )
    parser.add_argument("--session", default=DEFAULT_SESSION_ID)
    args = parser.parse_args(argv)

    if args.sync_existing:
        result = sync_existing_thirdspace_folder(
            transcript_csv_path=Path(args.updated_transcripts),
            index_csv_path=Path(args.updated_indices),
            session_id=str(args.session or DEFAULT_SESSION_ID),
        )
    else:
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
