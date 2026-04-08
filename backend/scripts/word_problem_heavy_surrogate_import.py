"""Import a surrogate-replaced copy of an existing session folder.

Reads a human-curated surrogate map (produced by Claude reading each source
document in the target folder) and produces a new folder whose documents have
every PII span replaced by a natural-reading surrogate, recorded as
``pre_annotations``.

Designed for the "Word Problem Heavy 100" folder (id ``fb8af36d``) but is
generic: any session folder + matching surrogate map will work.
"""

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

from server import (  # noqa: E402
    _load_all_folders,
    _load_folder,
    _session_dir,
    _upload_document_payload,
)

DEFAULT_SESSION_ID = "default"
DEFAULT_OUTPUT_DIR = ROOT / "output"

# Maps the user's 16-type PII taxonomy to the backend canonical labels that
# ``backend/taxonomy.py::canonicalize_label`` accepts.
USER_LABEL_TO_CANONICAL: dict[str, str] = {
    "NAME": "NAME",
    "AGE": "AGE",
    "ADDRESS": "ADDRESS",
    "OTHER_LOCATION": "OTHER_LOCATIONS_IDENTIFIED",
    "DATE": "DATE",
    "SCHOOL": "SCHOOL",
    "PHONE_NUMBER": "PHONE_NUMBER",
    "FAX_NUMBER": "FAX_NUMBER",
    "EMAIL": "EMAIL",
    "SSN": "SSN",
    "ACCOUNT_NUMBER": "ACCOUNT_NUMBER",
    "DEVICE_IDENTIFIER": "DEVICE_IDENTIFIER",
    "URL": "URL",
    "IP_ADDRESS": "IP_ADDRESS",
    "OTHER_IDENTIFYING_NUMBER": "IDENTIFYING_NUMBER",
    "TUTORING_PROVIDER": "TUTOR_PROVIDER",
}


def _canonical_label(label: str) -> str:
    normalized = str(label or "").strip().upper()
    if normalized not in USER_LABEL_TO_CANONICAL:
        raise ValueError(
            f"Unknown PII label {label!r}. Expected one of "
            f"{sorted(USER_LABEL_TO_CANONICAL)}."
        )
    return USER_LABEL_TO_CANONICAL[normalized]


def _validate_entry(entry: dict[str, Any]) -> dict[str, str]:
    original = str(entry.get("original") or "")
    surrogate = str(entry.get("surrogate") or "")
    label = str(entry.get("label") or "")
    if not original:
        raise ValueError(f"Surrogate entry missing 'original': {entry!r}")
    if not surrogate:
        raise ValueError(f"Surrogate entry missing 'surrogate': {entry!r}")
    if not label:
        raise ValueError(f"Surrogate entry missing 'label': {entry!r}")
    canonical = _canonical_label(label)
    if original in surrogate:
        raise ValueError(
            f"PII leak: surrogate {surrogate!r} contains original {original!r}."
        )
    return {"original": original, "surrogate": surrogate, "label": canonical}


def apply_surrogates_to_utterance(
    text: str,
    entries: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    """Apply every surrogate entry to ``text`` and return the rewritten text
    alongside annotation spans whose offsets point at the surrogate substrings
    in the new text.

    Raises on overlapping spans and on residual PII leaks.
    """
    normalized_entries = [_validate_entry(entry) for entry in entries]

    # Collect every occurrence of every original in the source text.
    hits: list[tuple[int, int, str, str, str]] = []
    for entry in normalized_entries:
        original = entry["original"]
        pos = 0
        while True:
            idx = text.find(original, pos)
            if idx < 0:
                break
            hits.append(
                (idx, idx + len(original), original, entry["surrogate"], entry["label"])
            )
            pos = idx + len(original)

    hits.sort(key=lambda item: (item[0], item[1]))
    for left, right in zip(hits, hits[1:]):
        if left[1] > right[0]:
            raise ValueError(
                "Overlapping surrogate spans detected: "
                f"{left[2]!r} at [{left[0]}:{left[1]}] vs "
                f"{right[2]!r} at [{right[0]}:{right[1]}]."
            )

    new_parts: list[str] = []
    out_spans: list[dict[str, Any]] = []
    cursor = 0
    new_offset = 0
    for start, end, _original, surrogate, canonical_label in hits:
        prefix = text[cursor:start]
        new_parts.append(prefix)
        new_offset += len(prefix)
        new_parts.append(surrogate)
        out_spans.append(
            {
                "start": new_offset,
                "end": new_offset + len(surrogate),
                "text": surrogate,
                "pii_type": canonical_label,
            }
        )
        new_offset += len(surrogate)
        cursor = end
    new_parts.append(text[cursor:])
    new_text = "".join(new_parts)

    # PII-leak guard: no original substring may survive in the rewritten text.
    for entry in normalized_entries:
        if entry["original"] in new_text:
            raise ValueError(
                f"PII leak after replacement: {entry['original']!r} still "
                f"present in rewritten text."
            )

    return new_text, out_spans


def build_surrogate_record_for_doc(
    source_doc: dict[str, Any],
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a JSONL record (transcript-style) for one source document."""
    doc_id = str(source_doc.get("id") or "")
    if not doc_id:
        raise ValueError("Source doc is missing 'id'.")
    utterances = source_doc.get("utterances") or []
    if not isinstance(utterances, list) or not utterances:
        raise ValueError(f"Source doc {doc_id!r} has no utterances.")

    transcript: list[dict[str, Any]] = []
    for index, utt in enumerate(utterances, start=1):
        speaker = str(utt.get("speaker") or "unknown")
        text = str(utt.get("text") or "")
        new_text, spans = apply_surrogates_to_utterance(text, entries)
        transcript.append(
            {
                "session_id": doc_id,
                "sequence_id": index,
                "role": speaker,
                "content": new_text,
                "annotations": spans,
            }
        )

    return {
        "session_id": doc_id,
        "transcript": transcript,
        "annotations": [],
    }


def _load_surrogate_map(path: Path) -> dict[str, list[dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    docs = data.get("docs")
    if not isinstance(docs, dict):
        raise ValueError(
            f"Surrogate map {path} must have a top-level 'docs' object."
        )
    out: dict[str, list[dict[str, Any]]] = {}
    for doc_id, entry_block in docs.items():
        if not isinstance(entry_block, dict):
            raise ValueError(
                f"Surrogate map entry for {doc_id!r} must be an object."
            )
        entries = entry_block.get("entries")
        if not isinstance(entries, list):
            raise ValueError(
                f"Surrogate map entry for {doc_id!r} missing 'entries' list."
            )
        out[str(doc_id)] = entries
    return out


def _load_source_doc(session_id: str, doc_id: str) -> dict[str, Any]:
    path = _session_dir(session_id) / f"{doc_id}.source.json"
    if not path.exists():
        raise ValueError(
            f"Source document not found: {path} (doc_id={doc_id!r}, "
            f"session={session_id!r})."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def build_jsonl_records(
    source_folder_id: str,
    surrogate_map_path: Path,
    session_id: str,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    folder = _load_folder(source_folder_id, session_id)
    if folder is None:
        raise ValueError(
            f"Source folder {source_folder_id!r} not found in session "
            f"{session_id!r}."
        )
    doc_ids = list(folder.doc_ids)
    if limit is not None:
        doc_ids = doc_ids[:limit]

    surrogate_map = _load_surrogate_map(Path(surrogate_map_path))

    # Fail explicitly on any missing doc.
    missing = [did for did in doc_ids if did not in surrogate_map]
    if missing:
        raise ValueError(
            f"Surrogate map is missing entries for {len(missing)} doc(s): "
            f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
        )

    records: list[dict[str, Any]] = []
    for doc_id in doc_ids:
        source_doc = _load_source_doc(session_id, doc_id)
        entries = surrogate_map[doc_id]
        records.append(build_surrogate_record_for_doc(source_doc, entries))
    return records


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
    return output_path


def _assert_target_name_available(target_name: str, session_id: str) -> None:
    existing = {
        str(folder.name or "").strip().casefold()
        for folder in _load_all_folders(session_id)
    }
    if target_name.casefold() in existing:
        raise ValueError(
            f"Folder {target_name!r} already exists in session {session_id!r}."
        )


def convert_and_import(
    *,
    source_folder_id: str,
    surrogate_map_path: Path | str,
    target_name: str,
    session_id: str = DEFAULT_SESSION_ID,
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    _assert_target_name_available(target_name, session_id)

    records = build_jsonl_records(
        source_folder_id=source_folder_id,
        surrogate_map_path=Path(surrogate_map_path),
        session_id=session_id,
        limit=limit,
    )

    output_path = Path(output_dir) / f"{target_name}.jsonl"
    write_jsonl(records, output_path)

    label_counts: Counter[str] = Counter()
    for record in records:
        for turn in record["transcript"]:
            for ann in turn["annotations"]:
                label_counts[str(ann["pii_type"])] += 1

    summary: dict[str, Any] = {
        "source_folder_id": source_folder_id,
        "session_id": session_id,
        "record_count": len(records),
        "folder_name": target_name,
        "output_path": str(output_path),
        "label_counts": dict(sorted(label_counts.items())),
        "dry_run": dry_run,
    }

    if dry_run:
        return summary

    imported = _upload_document_payload(
        output_path.read_bytes(),
        output_path.name,
        session_id,
    )
    summary["imported_merged_doc_id"] = imported.id
    summary["imported_filename"] = imported.filename
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Produce a surrogate-replaced copy of a session folder using a "
            "human-curated surrogate map. The new folder is imported into "
            "the same session with pre_annotations for each surrogate span."
        )
    )
    parser.add_argument("--source-folder-id", required=True)
    parser.add_argument("--surrogate-map", required=True, type=Path)
    parser.add_argument(
        "--target-name",
        default="word_problem_heavy_100_surrogated",
    )
    parser.add_argument("--session", default=DEFAULT_SESSION_ID)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    summary = convert_and_import(
        source_folder_id=str(args.source_folder_id),
        surrogate_map_path=Path(args.surrogate_map),
        target_name=str(args.target_name),
        session_id=str(args.session),
        output_dir=Path(args.output_dir),
        limit=args.limit,
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
