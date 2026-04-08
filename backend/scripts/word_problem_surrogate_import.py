from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models import CanonicalDocument, CanonicalSpan, FolderRecord
from server import (  # noqa: E402
    _load_all_folders,
    _load_doc,
    _load_folder,
    _load_sidecar,
    _save_folder,
    _upload_document_payload,
)
from taxonomy import is_broad_geography_text  # noqa: E402


IMPORT_FOLDER_NAME = "Word Problem Heavy 100 Surrogate"
SOURCE_FOLDER_NAME = "01_topic_1_prePII_noMoreThan_NAME_LOCATION"
UNION_METHOD_ID = "deid_pipeline_union"
DEFAULT_RANKING_PATH = (
    ROOT / "output" / "01_topic_1_prePII_noMoreThan_NAME_LOCATION_top_200_word_problem_heavy.csv"
)
DEFAULT_OUTPUT_PATH = ROOT / "output" / f"{IMPORT_FOLDER_NAME}.jsonl"
DEFAULT_MAPPING_PATH = ROOT / "output" / f"{IMPORT_FOLDER_NAME}_doc_mapping.csv"
DEFAULT_SESSION_ID = "default"
DEFAULT_TARGET_COUNT = 100

PLACEHOLDER_PATTERN = re.compile(r"<(PERSON|LOCATION)>")
SELF_INTRO_PERSON_PATTERN = re.compile(
    r"\b(my name is|i am|i'm|im|this is)\s+<PERSON>\b",
    re.IGNORECASE,
)
DIRECT_ADDRESS_PERSON_PATTERN = re.compile(
    r"(^|[\s,;:!?.-])(hi|hello|hey|thanks|thank you|bye|goodbye|dear)\s+<PERSON>\b",
    re.IGNORECASE,
)
PLACEHOLDER_LABEL_MAP: dict[str, str] = {
    "PERSON": "NAME",
    "LOCATION": "ADDRESS",
}
NAME_TEXT_RE = re.compile(r"^[A-Za-z][A-Za-z' -]{1,40}$")
ADDRESS_TEXT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9' .,#-]{2,60}$")
NAME_CONTEXT_RE = re.compile(
    r"\b(hi|hello|hey|thanks|thank you|bye|goodbye|dear|name is|i am|i'm|im|this is)\b",
    re.IGNORECASE,
)
ADDRESS_CONTEXT_RE = re.compile(
    r"\b(live in|lives in|from|at|near|around|moved to|located in)\b",
    re.IGNORECASE,
)

NAME_FALSE_POSITIVE_TEXTS = {
    "maam",
    "sir",
    "miss",
    "teacher",
    "student",
    "tutor",
    "buddy",
    "friend",
    "bro",
    "sis",
    "mom",
    "dad",
    "kinda",
    "alr",
    "pleas",
    "station",
    "crackers",
    "red",
    "job",
    "i",
    "h",
}
ADDRESS_FALSE_POSITIVE_TEXTS = {
    "job",
    "home",
    "class",
    "school",
    "problem",
    "question",
    "board",
}

NAME_SURROGATES = [
    "Marisol",
    "Elliot",
    "Jalen",
    "Nadia",
    "Tariq",
    "Mina",
    "Rowan",
    "Celeste",
    "Darius",
    "Imani",
    "Keira",
    "Noel",
    "Sarai",
    "Daphne",
    "Zane",
    "Priya",
    "Lena",
    "Malik",
    "Tessa",
    "Jonah",
    "Arielle",
    "Kellen",
    "Leona",
    "Marcus",
    "Sonia",
    "Nico",
    "Camila",
    "Brennan",
    "Selene",
    "Rafael",
    "Talia",
    "Emerson",
    "Karina",
    "Micah",
    "Jocelyn",
    "Theo",
    "Aisha",
    "Reina",
    "Matteo",
    "Liora",
    "Devin",
    "Paloma",
    "Yasmin",
    "Corbin",
    "Alina",
    "Kian",
    "Maren",
    "Soren",
    "Vivian",
    "Luca",
    "Adele",
    "Nolan",
    "Farah",
    "Gideon",
    "Kiara",
    "Rosalie",
    "Desmond",
    "Anika",
    "Callum",
    "Maeve",
]

ADDRESS_SURROGATES = [
    "Riverton",
    "Maple Glen",
    "Cedar Falls",
    "Oakridge",
    "Lakeside",
    "Pine Hollow",
    "Brookfield",
    "Fairhaven",
    "Stonebridge",
    "Meadow Park",
    "Summit Ridge",
    "Willow Creek",
    "Clearwater",
    "Harbor Point",
    "Redwood Terrace",
    "Silver Lake",
    "Autumn Grove",
    "Northgate",
    "Briarwood",
    "Westfield",
    "Fox Hollow",
    "Glenhaven",
    "Riverbend",
    "Sunnybank",
    "Evergreen Point",
    "Kingsley Park",
    "Heather Hill",
    "Millbrook",
    "Crescent Bay",
    "Parker Heights",
]

LABEL_SURROGATE_POOLS: dict[str, list[str]] = {
    "NAME": NAME_SURROGATES,
    "ADDRESS": ADDRESS_SURROGATES,
}


def _normalize_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").strip().lower()).strip()


def _is_too_similar(original_text: str, candidate: str) -> bool:
    original = _normalize_text(original_text)
    current = _normalize_text(candidate)
    if not original or original in {"person", "location"}:
        return False
    if original == current:
        return True
    if len(original) >= 3 and original in current:
        return True
    if len(current) >= 3 and current in original:
        return True
    if original[:2] and current.startswith(original[:2]):
        return True
    return False


def _find_source_folder(
    *,
    source_folder_name: str = SOURCE_FOLDER_NAME,
    session_id: str = DEFAULT_SESSION_ID,
) -> FolderRecord:
    for folder in _load_all_folders(session_id):
        if str(folder.name or "").strip().casefold() == source_folder_name.casefold():
            return folder
    raise ValueError(f"Folder '{source_folder_name}' not found in session '{session_id}'.")


def _read_ranking_rows(path: Path | str) -> list[dict[str, str]]:
    ranking_path = Path(path)
    if not ranking_path.exists():
        raise ValueError(f"Ranking CSV does not exist: {ranking_path}")
    with ranking_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No ranking rows found in {ranking_path}")
    return rows


def _map_global_span_to_turn(
    doc: CanonicalDocument,
    span: CanonicalSpan,
) -> tuple[int, int, int] | None:
    for turn_index, row in enumerate(doc.utterances):
        if span.start >= row.global_start and span.end <= row.global_end:
            return (
                turn_index,
                span.start - row.global_start,
                span.end - row.global_start,
            )
    return None


def _looks_like_obvious_name(span: CanonicalSpan, doc: CanonicalDocument) -> bool:
    normalized = _normalize_text(span.text)
    if not normalized or normalized in NAME_FALSE_POSITIVE_TEXTS:
        return False
    if not NAME_TEXT_RE.fullmatch(span.text):
        return False
    if span.text.strip().isdigit():
        return False

    context_start = max(0, span.start - 24)
    context_end = min(len(doc.raw_text), span.end + 24)
    context = doc.raw_text[context_start:context_end]
    if NAME_CONTEXT_RE.search(context):
        return True
    if any(char.isupper() for char in span.text):
        return True
    return span.text.isalpha() and len(span.text) >= 4


def _looks_like_obvious_address(span: CanonicalSpan, doc: CanonicalDocument) -> bool:
    normalized = _normalize_text(span.text)
    if not normalized or normalized in ADDRESS_FALSE_POSITIVE_TEXTS:
        return False
    if is_broad_geography_text(span.text):
        return False
    if not ADDRESS_TEXT_RE.fullmatch(span.text):
        return False

    context_start = max(0, span.start - 30)
    context_end = min(len(doc.raw_text), span.end + 30)
    context = doc.raw_text[context_start:context_end]
    if ADDRESS_CONTEXT_RE.search(context):
        return True
    if any(char.isdigit() for char in span.text):
        return True
    return bool(re.search(r"[A-Z]", span.text)) and len(span.text.split()) >= 2


def _filter_obvious_union_spans(doc: CanonicalDocument, spans: list[CanonicalSpan]) -> list[CanonicalSpan]:
    accepted: list[CanonicalSpan] = []
    seen: set[tuple[int, int, str, str]] = set()
    for span in spans:
        if span.start < 0 or span.end > len(doc.raw_text) or span.start >= span.end:
            continue
        if doc.raw_text[span.start:span.end] != span.text:
            continue
        if span.label == "NAME":
            keep = _looks_like_obvious_name(span, doc)
        elif span.label == "ADDRESS":
            keep = _looks_like_obvious_address(span, doc)
        else:
            keep = False
        if not keep:
            continue
        key = (span.start, span.end, span.label, span.text)
        if key in seen:
            continue
        seen.add(key)
        accepted.append(span)
    return sorted(accepted, key=lambda item: (item.start, item.end, item.label, item.text))


def _placeholder_entity_key(
    speaker: str,
    content: str,
    *,
    start: int,
    label: str,
    label_count_in_turn: int,
) -> str:
    if label != "NAME":
        left = _normalize_text(content[max(0, start - 16) : start])
        right = _normalize_text(content[start : min(len(content), start + 24)])
        return f"{label}:placeholder:{left}|{right}"

    if SELF_INTRO_PERSON_PATTERN.search(content):
        return f"NAME:speaker_identity:{speaker}"
    if DIRECT_ADDRESS_PERSON_PATTERN.search(content):
        return f"NAME:counterparty_of:{speaker}"
    if label_count_in_turn == 1:
        return f"NAME:counterparty_of:{speaker}"
    left = _normalize_text(content[max(0, start - 16) : start])
    right = _normalize_text(content[start : min(len(content), start + 24)])
    return f"NAME:placeholder:{speaker}:{left}|{right}"


def _choose_surrogate(
    label: str,
    *,
    entity_key: str,
    original_text: str,
    used_surrogates: set[str],
) -> str:
    pool = LABEL_SURROGATE_POOLS.get(label)
    if not pool:
        raise ValueError(f"No surrogate pool configured for label {label!r}.")
    seed = hashlib.sha1(entity_key.encode("utf-8")).hexdigest()
    start_index = int(seed[:8], 16) % len(pool)
    for offset in range(len(pool)):
        candidate = pool[(start_index + offset) % len(pool)]
        if candidate in used_surrogates:
            continue
        if _is_too_similar(original_text, candidate):
            continue
        used_surrogates.add(candidate)
        return candidate
    raise ValueError(f"Unable to choose a unique surrogate for label {label!r}.")


def _build_turn_occurrences(
    doc: CanonicalDocument,
    union_spans: list[CanonicalSpan],
) -> list[list[dict[str, Any]]]:
    occurrences: list[list[dict[str, Any]]] = [[] for _ in doc.utterances]

    for turn_index, row in enumerate(doc.utterances):
        raw_placeholder_counts = Counter(PLACEHOLDER_PATTERN.findall(row.text))
        for match in PLACEHOLDER_PATTERN.finditer(row.text):
            raw_label = match.group(1)
            label = PLACEHOLDER_LABEL_MAP[raw_label]
            occurrences[turn_index].append(
                {
                    "turn_index": turn_index,
                    "start": match.start(),
                    "end": match.end(),
                    "label": label,
                    "original_text": match.group(0),
                    "source": "placeholder",
                    "entity_key": _placeholder_entity_key(
                        row.speaker,
                        row.text,
                        start=match.start(),
                        label=label,
                        label_count_in_turn=int(raw_placeholder_counts.get(raw_label, 0)),
                    ),
                }
            )

    for span in union_spans:
        mapped = _map_global_span_to_turn(doc, span)
        if mapped is None:
            continue
        turn_index, local_start, local_end = mapped
        occurrences[turn_index].append(
            {
                "turn_index": turn_index,
                "start": local_start,
                "end": local_end,
                "label": span.label,
                "original_text": span.text,
                "source": "union",
                "entity_key": f"{span.label}:union:{_normalize_text(span.text)}",
            }
        )

    for turn_occurrences in occurrences:
        turn_occurrences.sort(
            key=lambda item: (int(item["start"]), int(item["end"]), str(item["label"]), str(item["source"]))
        )

    return occurrences


def _apply_surrogates_to_doc(
    doc: CanonicalDocument,
    *,
    union_spans: list[CanonicalSpan],
    original_doc_id: str,
    csv_filename: str,
    rank: int,
) -> tuple[dict[str, Any], dict[str, Counter[str]]]:
    turn_occurrences = _build_turn_occurrences(doc, union_spans)
    assignments: dict[str, str] = {}
    used_surrogates: set[str] = set()
    placeholder_counts: Counter[str] = Counter()
    method_counts: Counter[str] = Counter()
    transcript: list[dict[str, Any]] = []

    for turn_index, row in enumerate(doc.utterances):
        content = row.text
        annotations: list[dict[str, Any]] = []
        replacements = []
        for occurrence in turn_occurrences[turn_index]:
            entity_key = f"{original_doc_id}:{occurrence['entity_key']}"
            if entity_key not in assignments:
                assignments[entity_key] = _choose_surrogate(
                    str(occurrence["label"]),
                    entity_key=entity_key,
                    original_text=str(occurrence["original_text"]),
                    used_surrogates=used_surrogates,
                )
            replacement_text = assignments[entity_key]
            replacements.append({**occurrence, "replacement_text": replacement_text})
            if occurrence["source"] == "placeholder":
                placeholder_counts[str(occurrence["label"])] += 1
            else:
                method_counts[str(occurrence["label"])] += 1

        replacements.sort(key=lambda item: int(item["start"]), reverse=True)
        updated = content
        for item in replacements:
            start = int(item["start"])
            end = int(item["end"])
            replacement_text = str(item["replacement_text"])
            updated = updated[:start] + replacement_text + updated[end:]

        rebuilt_annotations: list[dict[str, Any]] = []
        cursor = 0
        for item in sorted(replacements, key=lambda entry: int(entry["start"])):
            original_start = int(item["start"])
            original_end = int(item["end"])
            replacement_text = str(item["replacement_text"])
            new_start = cursor + original_start
            new_end = new_start + len(replacement_text)
            cursor += len(replacement_text) - (original_end - original_start)
            rebuilt_annotations.append(
                {
                    "start": new_start,
                    "end": new_end,
                    "text": replacement_text,
                    "pii_type": str(item["label"]),
                }
            )

        transcript.append(
            {
                "session_id": original_doc_id,
                "sequence_id": turn_index + 1,
                "role": row.speaker,
                "content": updated,
                "annotations": rebuilt_annotations,
                "source_doc_id": original_doc_id,
                "source_csv_filename": csv_filename,
                "source_rank": rank,
            }
        )

    record = {
        "session_id": original_doc_id,
        "source_doc_id": original_doc_id,
        "source_csv_filename": csv_filename,
        "source_rank": rank,
        "transcript": transcript,
        "annotations": [],
    }
    return record, {
        "placeholder_counts": placeholder_counts,
        "method_counts": method_counts,
    }


def build_word_problem_surrogate_dataset(
    *,
    ranking_path: Path | str = DEFAULT_RANKING_PATH,
    source_folder_name: str = SOURCE_FOLDER_NAME,
    session_id: str = DEFAULT_SESSION_ID,
    target_count: int = DEFAULT_TARGET_COUNT,
) -> dict[str, Any]:
    if target_count <= 0:
        raise ValueError("target_count must be positive.")

    source_folder = _find_source_folder(source_folder_name=source_folder_name, session_id=session_id)
    source_doc_ids = set(source_folder.doc_ids)
    ranking_rows = _read_ranking_rows(ranking_path)

    selected_docs: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    placeholder_totals: Counter[str] = Counter()
    method_totals: Counter[str] = Counter()
    skipped_no_signal = 0
    skipped_only_false_positive_method = 0
    skipped_missing_doc = 0
    skipped_outside_source_folder = 0

    for rank, row in enumerate(ranking_rows, start=1):
        original_doc_id = str(row.get("doc_id") or "").strip()
        csv_filename = str(row.get("csv_filename") or "").strip()
        if not original_doc_id:
            continue
        if original_doc_id not in source_doc_ids:
            skipped_outside_source_folder += 1
            continue

        doc = _load_doc(original_doc_id, session_id)
        if doc is None:
            skipped_missing_doc += 1
            continue

        placeholder_hits = sum(len(PLACEHOLDER_PATTERN.findall(row.text)) for row in doc.utterances)
        union_sidecar = _load_sidecar(
            original_doc_id,
            f"agent.method.{UNION_METHOD_ID}",
            session_id,
        ) or []
        obvious_union = _filter_obvious_union_spans(doc, union_sidecar)

        if placeholder_hits == 0 and not union_sidecar:
            skipped_no_signal += 1
            continue
        if placeholder_hits == 0 and not obvious_union:
            skipped_only_false_positive_method += 1
            continue

        if len(selected_docs) >= target_count:
            continue

        record, counters = _apply_surrogates_to_doc(
            doc,
            union_spans=obvious_union,
            original_doc_id=original_doc_id,
            csv_filename=csv_filename,
            rank=rank,
        )
        records.append(record)
        selected_docs.append(
            {
                "original_doc_id": original_doc_id,
                "original_csv_filename": csv_filename,
                "rank": rank,
                "score": str(row.get("score") or ""),
                "placeholder_count": placeholder_hits,
                "accepted_union_count": len(obvious_union),
            }
        )
        placeholder_totals.update(counters["placeholder_counts"])
        method_totals.update(counters["method_counts"])

    if len(selected_docs) != target_count:
        raise ValueError(
            f"Only selected {len(selected_docs)} eligible transcripts from ranking pool; expected {target_count}."
        )

    return {
        "records": records,
        "selected_docs": selected_docs,
        "summary": {
            "selected_count": len(selected_docs),
            "skipped_no_signal": skipped_no_signal,
            "skipped_only_false_positive_method": skipped_only_false_positive_method,
            "skipped_missing_doc": skipped_missing_doc,
            "skipped_outside_source_folder": skipped_outside_source_folder,
            "placeholder_replacements_by_label": dict(sorted(placeholder_totals.items())),
            "method_additions_by_label": dict(sorted(method_totals.items())),
        },
        "source_folder_id": source_folder.id,
        "source_folder_name": source_folder.name,
    }


def write_word_problem_surrogate_jsonl(
    records: list[dict[str, Any]],
    output_path: Path | str,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")
    return path


def _ensure_import_folder_available(
    folder_name: str,
    *,
    session_id: str = DEFAULT_SESSION_ID,
) -> None:
    existing_names = {
        str(folder.name or "").strip().casefold() for folder in _load_all_folders(session_id)
    }
    if folder_name.casefold() in existing_names:
        raise ValueError(f"Folder '{folder_name}' already exists in session '{session_id}'.")


def _find_imported_folder(
    folder_name: str,
    *,
    session_id: str = DEFAULT_SESSION_ID,
) -> FolderRecord:
    for folder in _load_all_folders(session_id):
        if str(folder.name or "").strip() == folder_name:
            return folder
    raise ValueError(f"Imported folder '{folder_name}' was not found in session '{session_id}'.")


def write_word_problem_surrogate_mapping_csv(
    selected_docs: list[dict[str, Any]],
    imported_folder: FolderRecord,
    mapping_path: Path | str,
) -> Path:
    if len(imported_folder.doc_ids) != len(selected_docs):
        raise ValueError(
            "Imported folder doc count does not match selected doc count: "
            f"{len(imported_folder.doc_ids)} != {len(selected_docs)}"
        )

    path = Path(mapping_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "derived_doc_id",
                "derived_display_name",
                "original_doc_id",
                "original_csv_filename",
                "source_rank",
                "source_score",
            ],
        )
        writer.writeheader()
        for derived_doc_id, selected in zip(imported_folder.doc_ids, selected_docs, strict=True):
            writer.writerow(
                {
                    "derived_doc_id": derived_doc_id,
                    "derived_display_name": imported_folder.doc_display_names.get(derived_doc_id, ""),
                    "original_doc_id": selected["original_doc_id"],
                    "original_csv_filename": selected["original_csv_filename"],
                    "source_rank": selected["rank"],
                    "source_score": selected["score"],
                }
            )
    return path


def build_and_import_word_problem_surrogates(
    *,
    ranking_path: Path | str = DEFAULT_RANKING_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    mapping_path: Path | str = DEFAULT_MAPPING_PATH,
    source_folder_name: str = SOURCE_FOLDER_NAME,
    session_id: str = DEFAULT_SESSION_ID,
    target_count: int = DEFAULT_TARGET_COUNT,
) -> dict[str, Any]:
    dataset = build_word_problem_surrogate_dataset(
        ranking_path=ranking_path,
        source_folder_name=source_folder_name,
        session_id=session_id,
        target_count=target_count,
    )
    output = write_word_problem_surrogate_jsonl(dataset["records"], output_path)
    folder_name = output.stem
    _ensure_import_folder_available(folder_name, session_id=session_id)
    imported_doc = _upload_document_payload(output.read_bytes(), output.name, session_id)
    imported_folder = _find_imported_folder(folder_name, session_id=session_id)
    _save_folder(
        imported_folder.model_copy(update={"source_folder_id": dataset["source_folder_id"]}),
        session_id,
    )
    imported_folder = _load_folder(imported_folder.id, session_id)
    if imported_folder is None:
        raise ValueError(f"Unable to reload imported folder '{folder_name}'.")
    mapping = write_word_problem_surrogate_mapping_csv(
        dataset["selected_docs"],
        imported_folder,
        mapping_path,
    )

    return {
        "selected_count": len(dataset["selected_docs"]),
        "output_path": str(output),
        "mapping_path": str(mapping),
        "imported_doc_id": imported_doc.id,
        "imported_filename": imported_doc.filename,
        "imported_folder_id": imported_folder.id,
        "imported_folder_name": imported_folder.name,
        "source_folder_id": dataset["source_folder_id"],
        **dataset["summary"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build and import a surrogate-backed word-problem-heavy folder."
    )
    parser.add_argument("--ranking", default=str(DEFAULT_RANKING_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument("--mapping", default=str(DEFAULT_MAPPING_PATH))
    parser.add_argument("--source-folder", default=SOURCE_FOLDER_NAME)
    parser.add_argument("--session", default=DEFAULT_SESSION_ID)
    parser.add_argument("--target-count", type=int, default=DEFAULT_TARGET_COUNT)
    args = parser.parse_args(argv)

    result = build_and_import_word_problem_surrogates(
        ranking_path=Path(args.ranking),
        output_path=Path(args.output),
        mapping_path=Path(args.mapping),
        source_folder_name=str(args.source_folder or SOURCE_FOLDER_NAME),
        session_id=str(args.session or DEFAULT_SESSION_ID),
        target_count=int(args.target_count or DEFAULT_TARGET_COUNT),
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
