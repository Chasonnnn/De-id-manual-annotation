from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from server import _load_all_folders, _upload_document_payload  # noqa: E402


DEFAULT_SOCIAL_STUDIES_PATH = Path(
    "/Users/chason/contextshift-deid/workspaces/candidate/artifacts/experiments/"
    "20260402_231451_upchieve-english-social-raw-v4-export/"
    "upchieve_social_studies_only_surrogate_replaced_with_v4_spans.jsonl"
)
DEFAULT_ENGLISH_PATH = Path(
    "/Users/chason/contextshift-deid/workspaces/candidate/artifacts/experiments/"
    "20260402_231451_upchieve-english-social-raw-v4-export/"
    "upchieve_english_only_surrogate_replaced_with_v4_spans.jsonl"
)
DEFAULT_SOURCE_PATHS = [DEFAULT_SOCIAL_STUDIES_PATH, DEFAULT_ENGLISH_PATH]
DEFAULT_OUTPUT_DIR = ROOT / "output"
DEFAULT_SESSION_ID = "default"


def _require_text(value: str | None, *, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"Missing required field '{field_name}'.")
    return text


def _normalize_role(value: str | None) -> str:
    role = str(value or "").strip()
    return role or "unknown"


def _find_unique_occurrence(text: str, needle: str) -> tuple[int, int] | None:
    if not needle:
        return None
    first = text.find(needle)
    if first < 0:
        return None
    second = text.find(needle, first + 1)
    if second >= 0:
        return None
    return (first, first + len(needle))


def _resolve_via_surrogate_spans(
    turn: dict[str, Any],
    *,
    entity_type: str,
    tag_start: int | None,
    original_entity_type: str | None,
) -> tuple[int, int, str] | None:
    candidates: list[tuple[int, int, str]] = []
    for raw_span in turn.get("surrogate_spans") or []:
        if not isinstance(raw_span, dict):
            continue
        current_text = str(raw_span.get("span_text") or "")
        current_entity_type = str(raw_span.get("entity_type") or "").strip()
        current_original_entity_type = str(raw_span.get("original_entity_type") or "").strip()
        start = raw_span.get("tag_start")
        end = raw_span.get("tag_end")
        if (
            not current_text
            or not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end <= start
        ):
            continue
        if current_entity_type != entity_type:
            continue
        if tag_start is not None and start != tag_start:
            continue
        if original_entity_type and current_original_entity_type not in {"", original_entity_type}:
            continue
        candidates.append((start, end, current_text))

    if not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError(
            "Unable to resolve action_v4 span uniquely from surrogate_spans: "
            f"entity_type={entity_type!r}, tag_start={tag_start!r}, "
            f"original_entity_type={original_entity_type!r}."
        )
    return candidates[0]


def _resolve_action_annotation(turn: dict[str, Any], action_span: dict[str, Any]) -> dict[str, Any]:
    turn_text = str(turn.get("message") or "")
    span_text = _require_text(action_span.get("span_text"), field_name="action_v4_spans[].span_text")
    entity_type = _require_text(
        action_span.get("entity_type"),
        field_name="action_v4_spans[].entity_type",
    )
    metadata = action_span.get("metadata") or {}
    tag_start = metadata.get("tag_start")
    tag_end = metadata.get("tag_end")
    original_entity_type = str(metadata.get("original_entity_type") or "").strip() or None

    if (
        isinstance(tag_start, int)
        and isinstance(tag_end, int)
        and 0 <= tag_start <= tag_end <= len(turn_text)
        and tag_start < tag_end
        and turn_text[tag_start:tag_end] == span_text
    ):
        return {
            "start": tag_start,
            "end": tag_end,
            "text": span_text,
            "pii_type": entity_type,
        }

    surrogate_match = _resolve_via_surrogate_spans(
        turn,
        entity_type=entity_type,
        tag_start=tag_start if isinstance(tag_start, int) else None,
        original_entity_type=original_entity_type,
    )
    if surrogate_match is not None:
        start, end, current_text = surrogate_match
        return {
            "start": start,
            "end": end,
            "text": current_text,
            "pii_type": entity_type,
        }

    unique_match = _find_unique_occurrence(turn_text, span_text)
    if unique_match is not None:
        start, end = unique_match
        return {
            "start": start,
            "end": end,
            "text": span_text,
            "pii_type": entity_type,
        }

    raise ValueError(
        "Unable to resolve action_v4 span against transcript turn: "
        f"entity_type={entity_type!r}, span_text={span_text!r}, "
        f"tag_start={tag_start!r}, tag_end={tag_end!r}, "
        f"turn_text={turn_text!r}."
    )


def _dedup_annotations(annotations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[int, int, str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for annotation in sorted(
        annotations,
        key=lambda item: (
            int(item["start"]),
            int(item["end"]),
            str(item["pii_type"]),
            str(item["text"]),
        ),
    ):
        key = (
            int(annotation["start"]),
            int(annotation["end"]),
            str(annotation["pii_type"]),
            str(annotation["text"]),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(annotation)
    return deduped


def build_upchieve_surrogate_records(source_path: Path | str) -> list[dict[str, Any]]:
    path = Path(source_path)
    if not path.exists():
        raise ValueError(f"Source file does not exist: {path}")

    records: list[dict[str, Any]] = []
    seen_session_ids: set[str] = set()

    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw:
                continue
            data = json.loads(raw)
            session_id = _require_text(data.get("dialogue_id"), field_name="dialogue_id")
            if session_id in seen_session_ids:
                raise ValueError(f"Duplicate dialogue_id {session_id!r} in {path.name}.")
            seen_session_ids.add(session_id)

            transcript_turns = data.get("transcript_turns")
            if not isinstance(transcript_turns, list):
                raise ValueError(
                    f"Record {line_number} in {path.name} must include 'transcript_turns' as a list."
                )

            annotations_by_turn: dict[int, list[dict[str, Any]]] = {
                idx: [] for idx in range(len(transcript_turns))
            }
            for action_span in data.get("action_v4_spans") or []:
                if not isinstance(action_span, dict):
                    continue
                metadata = action_span.get("metadata") or {}
                turn_index = metadata.get("turn_index")
                if (
                    not isinstance(turn_index, int)
                    or turn_index < 0
                    or turn_index >= len(transcript_turns)
                ):
                    raise ValueError(
                        "Invalid action_v4 span turn_index for "
                        f"dialogue_id={session_id!r}: {turn_index!r}."
                    )
                try:
                    annotation = _resolve_action_annotation(
                        transcript_turns[turn_index],
                        action_span,
                    )
                except ValueError as exc:
                    raise ValueError(
                        "Unable to resolve action_v4 span for "
                        f"dialogue_id={session_id!r}, turn_index={turn_index}: {exc}"
                    ) from exc
                annotations_by_turn[turn_index].append(annotation)

            transcript: list[dict[str, Any]] = []
            for index, turn in enumerate(transcript_turns, start=1):
                if not isinstance(turn, dict):
                    raise ValueError(
                        f"Turn {index} for dialogue_id={session_id!r} must be an object."
                    )
                transcript.append(
                    {
                        "session_id": session_id,
                        "sequence_id": index,
                        "role": _normalize_role(turn.get("user_role")),
                        "timestamp_seconds": str(turn.get("timestamp_seconds") or ""),
                        "message_type": str(turn.get("message_type") or ""),
                        "message_original": str(turn.get("message_original") or ""),
                        "content": str(turn.get("message") or ""),
                        "annotations": _dedup_annotations(annotations_by_turn[index - 1]),
                    }
                )

            records.append(
                {
                    "session_id": session_id,
                    "transcript": transcript,
                    "annotations": [],
                }
            )

    if not records:
        raise ValueError(f"No JSONL records found in {path}")
    return records


def write_upchieve_surrogate_jsonl(records: list[dict[str, Any]], output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
    return path


def import_upchieve_surrogate_jsonl(
    output_path: Path | str,
    *,
    session_id: str = DEFAULT_SESSION_ID,
):
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


def convert_and_import_upchieve_surrogate_export(
    *,
    source_path: Path | str,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    session_id: str = DEFAULT_SESSION_ID,
) -> dict[str, Any]:
    source = Path(source_path)
    records = build_upchieve_surrogate_records(source)
    output_path = Path(output_dir) / source.name
    output = write_upchieve_surrogate_jsonl(records, output_path)
    imported_doc = import_upchieve_surrogate_jsonl(output, session_id=session_id)

    label_counts: Counter[str] = Counter()
    for record in records:
        for turn in record["transcript"]:
            for annotation in turn["annotations"]:
                label_counts[str(annotation["pii_type"])] += 1

    return {
        "source_path": str(source),
        "record_count": len(records),
        "output_path": str(output),
        "folder_name": output.stem,
        "imported_doc_id": imported_doc.id,
        "imported_filename": imported_doc.filename,
        "label_counts": dict(sorted(label_counts.items())),
        "session_id": session_id,
    }


def convert_and_import_many(
    *,
    source_paths: list[Path | str] | None = None,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    session_id: str = DEFAULT_SESSION_ID,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for source_path in source_paths or DEFAULT_SOURCE_PATHS:
        results.append(
            convert_and_import_upchieve_surrogate_export(
                source_path=source_path,
                output_dir=output_dir,
                session_id=session_id,
            )
        )
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Convert surrogate-replaced Upchieve v4 exports into app JSONL, "
            "repairing action span offsets against transcript_turns, and import them."
        )
    )
    parser.add_argument("sources", nargs="*", default=[str(path) for path in DEFAULT_SOURCE_PATHS])
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--session", default=DEFAULT_SESSION_ID)
    args = parser.parse_args(argv)

    results = convert_and_import_many(
        source_paths=[Path(item) for item in args.sources],
        output_dir=Path(args.output_dir),
        session_id=str(args.session or DEFAULT_SESSION_ID),
    )
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
