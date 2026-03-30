from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server import _load_all_folders, _upload_document_payload  # noqa: E402


IMPORT_FOLDER_NAME = "01_topic_1_prePII_noMoreThan_NAME_LOCATION"
DEFAULT_INPUT_DIR = Path("/Users/chason/Downloads") / IMPORT_FOLDER_NAME
DEFAULT_OUTPUT_PATH = ROOT / "output" / f"{IMPORT_FOLDER_NAME}.jsonl"
DEFAULT_SESSION_ID = "default"

CSV_FIELDS = ("timestamp_seconds", "user_role", "message_type", "message")
PLACEHOLDER_PATTERN = re.compile(r"<(PERSON|LOCATION)>")
PLACEHOLDER_LABEL_MAP: dict[str, str] = {
    "PERSON": "NAME",
    "LOCATION": "ADDRESS",
}


def _require_text(value: str | None, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Missing required field '{field_name}'.")
    return text


def _coerce_message(value: str | None) -> str:
    if value is None:
        raise ValueError("Missing required field 'message'.")
    raw = str(value)
    if raw.strip() == "":
        return ""
    return raw


def _sorted_csv_paths(input_dir: Path) -> list[Path]:
    def sort_key(path: Path) -> tuple[int, Any]:
        stem = path.stem
        if stem.isdigit():
            return (0, int(stem))
        return (1, stem)

    return sorted(input_dir.glob("*.csv"), key=sort_key)


def _build_annotations(message: str) -> list[dict[str, Any]]:
    annotations: list[dict[str, Any]] = []
    for match in PLACEHOLDER_PATTERN.finditer(message):
        raw_label = match.group(1)
        mapped_label = PLACEHOLDER_LABEL_MAP.get(raw_label)
        if mapped_label is None:
            raise ValueError(f"Unsupported placeholder label: {raw_label!r}")
        annotations.append(
            {
                "start": match.start(),
                "end": match.end(),
                "text": match.group(0),
                "pii_type": mapped_label,
            }
        )
    return annotations


def build_name_location_records(input_dir: Path | str) -> list[dict[str, Any]]:
    directory = Path(input_dir)
    if not directory.exists():
        raise ValueError(f"Input directory does not exist: {directory}")
    if not directory.is_dir():
        raise ValueError(f"Input path is not a directory: {directory}")

    csv_paths = _sorted_csv_paths(directory)
    if not csv_paths:
        raise ValueError(f"No CSV files found in {directory}")

    records: list[dict[str, Any]] = []
    for path in csv_paths:
        transcript: list[dict[str, Any]] = []
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = tuple(reader.fieldnames or ())
            if fieldnames != CSV_FIELDS:
                raise ValueError(
                    f"Unexpected CSV headers in {path.name}: {fieldnames!r}. Expected {CSV_FIELDS!r}."
                )
            for index, row in enumerate(reader, start=1):
                timestamp_seconds = _require_text(
                    row.get("timestamp_seconds"), field_name="timestamp_seconds"
                )
                user_role = _require_text(row.get("user_role"), field_name="user_role")
                message_type = _require_text(row.get("message_type"), field_name="message_type")
                message = _coerce_message(row.get("message"))
                transcript.append(
                    {
                        "session_id": path.stem,
                        "sequence_id": index,
                        "role": user_role,
                        "timestamp_seconds": timestamp_seconds,
                        "message_type": message_type,
                        "content": message,
                        "annotations": _build_annotations(message),
                    }
                )

        records.append(
            {
                "session_id": path.stem,
                "transcript": transcript,
                "annotations": [],
            }
        )
    return records


def write_name_location_jsonl(records: list[dict[str, Any]], output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
    return path


def import_name_location_jsonl(
    output_path: Path | str,
    *,
    session_id: str = DEFAULT_SESSION_ID,
):
    path = Path(output_path)
    filename = path.name
    expected_filename = f"{IMPORT_FOLDER_NAME}.jsonl"
    if filename != expected_filename:
        raise ValueError(f"Expected output filename '{expected_filename}', got {filename!r}.")

    existing_names = {
        str(folder.name or "").strip().casefold() for folder in _load_all_folders(session_id)
    }
    if IMPORT_FOLDER_NAME.casefold() in existing_names:
        raise ValueError(
            f"Folder '{IMPORT_FOLDER_NAME}' already exists in session '{session_id}'."
        )

    return _upload_document_payload(path.read_bytes(), filename, session_id)


def convert_and_import_name_location(
    *,
    input_dir: Path | str = DEFAULT_INPUT_DIR,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    session_id: str = DEFAULT_SESSION_ID,
) -> dict[str, Any]:
    records = build_name_location_records(input_dir)
    output = write_name_location_jsonl(records, output_path)
    imported_doc = import_name_location_jsonl(output, session_id=session_id)

    label_counts: Counter[str] = Counter()
    for record in records:
        for turn in record["transcript"]:
            for annotation in turn["annotations"]:
                label_counts[str(annotation["pii_type"])] += 1

    return {
        "record_count": len(records),
        "output_path": str(output),
        "imported_doc_id": imported_doc.id,
        "imported_filename": imported_doc.filename,
        "label_counts": dict(sorted(label_counts.items())),
        "session_id": session_id,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Convert a directory of CSV transcripts with inline placeholders into app JSONL and import it."
    )
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--session", default=DEFAULT_SESSION_ID)
    args = parser.parse_args(argv)

    result = convert_and_import_name_location(
        input_dir=Path(args.input_dir),
        output_path=Path(args.output),
        session_id=str(args.session or DEFAULT_SESSION_ID),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
