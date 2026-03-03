import json
import pytest
from normalizer import parse_file


def test_parse_hips_v1():
    data = {
        "transcript": "Hello Anna, call Sue please.",
        "pii_occurrences": [
            {"start": 6, "end": 10, "text": "Anna", "pii_type": "NAME"},
            {"start": 17, "end": 20, "text": "Sue", "pii_type": "NAME"},
        ],
    }
    raw = json.dumps(data).encode()
    docs = parse_file(raw, "test_v1.json", "doc1")
    assert len(docs) == 1
    doc = docs[0]
    assert doc.format == "hips_v1"
    assert doc.raw_text == "Hello Anna, call Sue please."
    assert len(doc.pre_annotations) == 2
    assert doc.pre_annotations[0].label == "NAME"
    assert doc.pre_annotations[0].text == "Anna"
    assert "NAME" in doc.label_set


def test_parse_hips_v2():
    data = {
        "text": "Hi Michael, meet Raymond.",
        "pii": [
            {"pii": "Michael", "type": "NAME", "start": 3, "end": 10},
            {"pii": "Raymond", "type": "NAME", "start": 17, "end": 24},
        ],
    }
    raw = json.dumps(data).encode()
    docs = parse_file(raw, "test_2909.json", "doc2")
    assert len(docs) == 1
    doc = docs[0]
    assert doc.format == "hips_v2"
    assert len(doc.pre_annotations) == 2
    assert doc.pre_annotations[1].text == "Raymond"
    assert "NAME" in doc.label_set


def test_parse_jsonl():
    record = {
        "transcript": [
            {
                "_id": "0",
                "role": "volunteer",
                "content": "Hi Chloe!",
                "session_id": 1,
                "sequence_id": 0,
                "annotations": [
                    {"pii_type": "PERSON", "surrogate": "Chloe", "start": 3, "end": 8}
                ],
            },
            {
                "_id": "1",
                "role": "student",
                "content": "Hello there",
                "session_id": 1,
                "sequence_id": 1,
                "annotations": [],
            },
        ],
        "annotations": [],
    }
    raw = json.dumps(record).encode()
    # Single JSON object that looks like JSONL record
    docs = parse_file(raw, "test.json", "doc3")
    assert len(docs) == 1
    doc = docs[0]
    assert doc.format == "jsonl"
    assert doc.raw_text == "Hi Chloe!\nHello there"
    assert len(doc.pre_annotations) == 1
    span = doc.pre_annotations[0]
    assert span.label == "NAME"  # PERSON -> NAME
    assert span.text == "Chloe"
    assert span.start == 3
    assert span.end == 8
    assert "NAME" in doc.label_set


def test_parse_jsonl_file():
    records = [
        {
            "transcript": [
                {
                    "_id": "0",
                    "role": "a",
                    "content": "Hi Bob",
                    "session_id": 1,
                    "sequence_id": 0,
                    "annotations": [
                        {"pii_type": "PERSON", "surrogate": "Bob", "start": 3, "end": 6}
                    ],
                },
            ],
            "annotations": [],
        },
        {
            "transcript": [
                {
                    "_id": "0",
                    "role": "b",
                    "content": "Hello",
                    "session_id": 2,
                    "sequence_id": 0,
                    "annotations": [],
                },
            ],
            "annotations": [],
        },
    ]
    raw = "\n".join(json.dumps(r) for r in records).encode()
    docs = parse_file(raw, "test.jsonl", "doc4")
    assert len(docs) == 2
    assert docs[0].pre_annotations[0].text == "Bob"


def test_jsonl_offset_remapping():
    """Verify that annotations in later turns get correct global offsets."""
    record = {
        "transcript": [
            {
                "_id": "0",
                "role": "a",
                "content": "Hello world",
                "session_id": 1,
                "sequence_id": 0,
                "annotations": [],
            },
            {
                "_id": "1",
                "role": "b",
                "content": "Hi Alice!",
                "session_id": 1,
                "sequence_id": 1,
                "annotations": [
                    {"pii_type": "PERSON", "surrogate": "Alice", "start": 3, "end": 8}
                ],
            },
        ],
        "annotations": [],
    }
    raw = json.dumps(record).encode()
    docs = parse_file(raw, "test.json", "doc5")
    doc = docs[0]
    # "Hello world\nHi Alice!"
    #  01234567890 1 234567890
    #  offset 0    12
    assert doc.raw_text == "Hello world\nHi Alice!"
    span = doc.pre_annotations[0]
    assert span.start == 15  # 12 (len("Hello world") + 1 for \n) + 3
    assert span.end == 20
    assert doc.raw_text[span.start : span.end] == "Alice"


def test_hips_v2_full():
    """HIPS v2 format: 'text' + 'pii' keys with {start, end, pii, type}."""
    data = {
        "text": "Call Dr. Smith at 123 Main St.",
        "pii": [
            {"pii": "Dr. Smith", "type": "NAME", "start": 5, "end": 14},
            {"pii": "123 Main St.", "type": "LOCATION", "start": 18, "end": 30},
        ],
    }
    raw = json.dumps(data).encode()
    docs = parse_file(raw, "sample_2909.json", "hv2")
    assert len(docs) == 1
    doc = docs[0]
    assert doc.format == "hips_v2"
    assert doc.raw_text == "Call Dr. Smith at 123 Main St."
    assert len(doc.pre_annotations) == 2
    assert doc.pre_annotations[0].label == "NAME"
    assert doc.pre_annotations[0].text == "Dr. Smith"
    assert doc.pre_annotations[0].start == 5
    assert doc.pre_annotations[0].end == 14
    assert doc.pre_annotations[1].label == "LOCATION"
    assert doc.pre_annotations[1].text == "123 Main St."
    # Verify utterance building
    assert len(doc.utterances) == 1
    assert doc.utterances[0].global_start == 0
    assert doc.utterances[0].global_end == 30


def test_hips_v2_label_mapping():
    """HIPS v2 with PERSON label should map to NAME."""
    data = {
        "text": "Ask Alice.",
        "pii": [
            {"pii": "Alice", "type": "PERSON", "start": 4, "end": 9},
        ],
    }
    raw = json.dumps(data).encode()
    docs = parse_file(raw, "test_v2_mapping.json", "hv2m")
    assert docs[0].pre_annotations[0].label == "NAME"


def test_span_dedup():
    """Duplicate spans should be deduplicated."""
    data = {
        "transcript": "Hello Anna.",
        "pii_occurrences": [
            {"start": 6, "end": 10, "text": "Anna", "pii_type": "NAME"},
            {"start": 6, "end": 10, "text": "Anna", "pii_type": "NAME"},
        ],
    }
    raw = json.dumps(data).encode()
    docs = parse_file(raw, "dedup.json", "dup1")
    assert len(docs[0].pre_annotations) == 1


def test_hips_v2_empty_pii():
    """HIPS v2 with no PII should produce empty annotations."""
    data = {
        "text": "No PII here.",
        "pii": [],
    }
    raw = json.dumps(data).encode()
    docs = parse_file(raw, "empty_v2.json", "ev2")
    assert docs[0].pre_annotations == []
    assert docs[0].raw_text == "No PII here."


def test_hips_v1_with_text_key_supported():
    data = {
        "text": "Hi Anna.",
        "pii_occurrences": [
            {"start": 3, "end": 7, "text": "Anna", "pii_type": "NAME"},
        ],
    }
    raw = json.dumps(data).encode()
    docs = parse_file(raw, "text_key_v1.json", "tv1")
    assert len(docs) == 1
    assert docs[0].raw_text == "Hi Anna."
    assert docs[0].pre_annotations[0].text == "Anna"
