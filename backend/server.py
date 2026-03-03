from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models import CanonicalDocument, CanonicalSpan, AgentOutputs
from normalizer import parse_file
from agent import MODEL_PRESETS, run_regex, run_llm_with_metadata
from metrics import compute_metrics

app = FastAPI(title="Annotation Tool")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(".annotation_tool")
SESSIONS_DIR = BASE_DIR / "sessions"
CONFIG_PATH = BASE_DIR / "config.json"

DEFAULT_CONFIG = {
    "system_prompt": (
        "You are a PII de-identification expert. Analyze the transcript and identify ALL PII instances.\n"
        "Return ONLY a JSON array where each element has:\n"
        '- "text": exact text as it appears\n'
        '- "label": one of NAME, LOCATION, SCHOOL, DATE, AGE, PHONE, EMAIL, URL, MISC_ID\n'
        '- "confidence": 0.0-1.0\n\n'
        "PII types: NAME (personal names), LOCATION (addresses, cities), SCHOOL (institutions),\n"
        "DATE (specific dates), AGE (ages/birth years), PHONE, EMAIL, URL, MISC_ID (IDs, case numbers)"
    ),
    "model": "openai/gpt-5.2-codex",
    "temperature": 0.0,
    "reasoning_effort": "xhigh",
    "anthropic_thinking": False,
    "anthropic_thinking_budget_tokens": None,
}


def _ensure_dirs():
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def _session_dir(session_id: str = "default") -> Path:
    return SESSIONS_DIR / session_id


def _save_doc(doc: CanonicalDocument, session_id: str = "default"):
    d = _session_dir(session_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{doc.id}.source.json").write_text(doc.model_dump_json(indent=2))


def _load_doc(doc_id: str, session_id: str = "default") -> CanonicalDocument | None:
    p = _session_dir(session_id) / f"{doc_id}.source.json"
    if not p.exists():
        return None
    return CanonicalDocument.model_validate_json(p.read_text())


def _save_sidecar(
    doc_id: str, kind: str, spans: list[CanonicalSpan], session_id: str = "default"
):
    d = _session_dir(session_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{doc_id}.{kind}.json").write_text(
        json.dumps([s.model_dump() for s in spans], indent=2)
    )


def _load_sidecar(
    doc_id: str, kind: str, session_id: str = "default"
) -> list[CanonicalSpan] | None:
    p = _session_dir(session_id) / f"{doc_id}.{kind}.json"
    if not p.exists():
        return None
    raw = json.loads(p.read_text())
    return [CanonicalSpan(**s) for s in raw]


# Session index management
_session_docs: dict[str, list[str]] = {}


def _load_session_index(session_id: str = "default") -> list[str]:
    if session_id in _session_docs:
        return _session_docs[session_id]
    idx_path = _session_dir(session_id) / "_index.json"
    if idx_path.exists():
        ids = json.loads(idx_path.read_text())
        _session_docs[session_id] = ids
        return ids
    _session_docs[session_id] = []
    return _session_docs[session_id]


def _save_session_index(session_id: str = "default"):
    d = _session_dir(session_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / "_index.json").write_text(json.dumps(_session_docs.get(session_id, [])))


def _enrich_doc(
    doc: CanonicalDocument, session_id: str = "default"
) -> CanonicalDocument:
    """Load sidecars and merge into document."""
    manual = _load_sidecar(doc.id, "manual", session_id)
    agent_rule = _load_sidecar(doc.id, "agent.rule", session_id)
    agent_llm = _load_sidecar(doc.id, "agent.llm", session_id)
    # Backward compat: also check old sidecar name
    agent_openai = _load_sidecar(doc.id, "agent.openai", session_id)

    llm_spans = (agent_llm or []) + (agent_openai or [])

    doc.manual_annotations = manual or []
    doc.agent_outputs = AgentOutputs(rule=agent_rule or [], llm=llm_spans)
    doc.agent_annotations = doc.agent_outputs.rule + doc.agent_outputs.llm
    doc.agent_run_warnings = []

    if manual:
        doc.status = "in_progress"

    return doc


def _dedup_spans(spans: list[CanonicalSpan]) -> list[CanonicalSpan]:
    seen: set[tuple[int, int, str]] = set()
    result: list[CanonicalSpan] = []
    for span in sorted(spans, key=lambda s: (s.start, s.end, s.label)):
        key = (span.start, span.end, span.label)
        if key in seen:
            continue
        seen.add(key)
        result.append(span)
    return result


def _normalize_and_validate_spans(
    spans: list[CanonicalSpan], raw_text: str
) -> list[CanonicalSpan]:
    normalized: list[CanonicalSpan] = []
    n = len(raw_text)
    for span in spans:
        if span.start < 0 or span.end > n or span.start >= span.end:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Invalid span offsets [{span.start}:{span.end}] for transcript length {n}."
                ),
            )
        expected = raw_text[span.start : span.end]
        normalized.append(
            CanonicalSpan(
                start=span.start,
                end=span.end,
                label=span.label.upper(),
                text=expected,
            )
        )
    return _dedup_spans(normalized)


# --- Config ---


def _load_config() -> dict:
    _ensure_dirs()
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return dict(DEFAULT_CONFIG)


def _save_config(cfg: dict):
    _ensure_dirs()
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))


# --- Routes ---


@app.get("/api/documents")
async def list_documents():
    session_id = "default"
    ids = _load_session_index(session_id)
    docs = []
    for did in ids:
        doc = _load_doc(did, session_id)
        if doc:
            manual = _load_sidecar(did, "manual", session_id)
            status = "in_progress" if manual else "pending"
            docs.append(
                {
                    "id": doc.id,
                    "filename": doc.filename,
                    "status": status,
                }
            )
    return docs


@app.get("/api/documents/{doc_id}")
async def get_document(doc_id: str):
    session_id = "default"
    doc = _load_doc(doc_id, session_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return _enrich_doc(doc, session_id)


@app.post("/api/documents/upload")
async def upload_file(file: UploadFile = File(...)):
    session_id = "default"
    raw = await file.read()
    filename = file.filename or "unknown"
    doc_id = str(uuid.uuid4())[:8]

    try:
        docs = parse_file(raw, filename, doc_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    ids = _load_session_index(session_id)
    for doc in docs:
        _save_doc(doc, session_id)
        ids.append(doc.id)
    _save_session_index(session_id)

    # Return the first document (for single-file uploads)
    # For JSONL with multiple records, return the first one
    if docs:
        return _enrich_doc(docs[0], session_id)
    raise HTTPException(status_code=400, detail="No documents parsed from file")


@app.put("/api/documents/{doc_id}/manual-annotations")
async def save_manual_annotations(doc_id: str, spans: list[CanonicalSpan]):
    session_id = "default"
    doc = _load_doc(doc_id, session_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    normalized = _normalize_and_validate_spans(spans, doc.raw_text)
    _save_sidecar(doc_id, "manual", normalized, session_id)
    return _enrich_doc(doc, session_id)


class AgentRunBody(BaseModel):
    mode: Literal["rule", "llm", "openai"] = "rule"
    system_prompt: str | None = None
    model: str | None = None
    temperature: float | None = None
    api_key: str | None = None
    reasoning_effort: Literal["none", "low", "medium", "high", "xhigh"] | None = None
    anthropic_thinking: bool | None = None
    anthropic_thinking_budget_tokens: int | None = None


@app.post("/api/documents/{doc_id}/agent")
async def run_agent(doc_id: str, body: AgentRunBody):
    session_id = "default"
    doc = _load_doc(doc_id, session_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    if body.mode == "rule":
        spans = _normalize_and_validate_spans(run_regex(doc.raw_text), doc.raw_text)
        _save_sidecar(doc_id, "agent.rule", spans, session_id)
        enriched = _enrich_doc(doc, session_id)
        enriched.agent_run_warnings = []
        return enriched

    if body.mode in ("llm", "openai"):
        cfg = _load_config()
        api_key = (
            body.api_key
            or os.environ.get("OPENAI_API_KEY", "")
            or os.environ.get("ANTHROPIC_API_KEY", "")
        )
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail="API key required. Set an env var (OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.) or provide in request.",
            )
        model = body.model or cfg.get("model", DEFAULT_CONFIG["model"])
        system_prompt = body.system_prompt or cfg.get(
            "system_prompt", DEFAULT_CONFIG["system_prompt"]
        )
        temperature = (
            body.temperature
            if body.temperature is not None
            else float(cfg.get("temperature", 0.0))
        )
        reasoning_effort = body.reasoning_effort or cfg.get(
            "reasoning_effort", "none"
        )
        anthropic_thinking = (
            body.anthropic_thinking
            if body.anthropic_thinking is not None
            else bool(cfg.get("anthropic_thinking", False))
        )
        anthropic_thinking_budget_tokens = (
            body.anthropic_thinking_budget_tokens
            if body.anthropic_thinking_budget_tokens is not None
            else cfg.get("anthropic_thinking_budget_tokens")
        )

        llm_result = run_llm_with_metadata(
            text=doc.raw_text,
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            anthropic_thinking=anthropic_thinking,
            anthropic_thinking_budget_tokens=anthropic_thinking_budget_tokens,
        )
        spans = _normalize_and_validate_spans(llm_result.spans, doc.raw_text)
        _save_sidecar(doc_id, "agent.llm", spans, session_id)
        enriched = _enrich_doc(doc, session_id)
        enriched.agent_run_warnings = llm_result.warnings
        return enriched

    raise HTTPException(status_code=400, detail=f"Unknown agent mode: {body.mode}")


@app.get("/api/documents/{doc_id}/metrics")
async def get_metrics(
    doc_id: str,
    reference: str = Query(...),
    hypothesis: str = Query(...),
    match_mode: str = Query("exact"),
):
    session_id = "default"
    doc = _load_doc(doc_id, session_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    enriched = _enrich_doc(doc, session_id)

    def _get_spans(source: str) -> list[CanonicalSpan]:
        if source == "pre":
            return enriched.pre_annotations
        if source == "manual":
            return enriched.manual_annotations
        if source == "agent":
            return enriched.agent_annotations
        if source == "agent.rule":
            return enriched.agent_outputs.rule
        if source == "agent.llm":
            return enriched.agent_outputs.llm
        raise HTTPException(status_code=400, detail=f"Unknown source: {source}")

    ref_spans = _get_spans(reference)
    hyp_spans = _get_spans(hypothesis)

    result = compute_metrics(ref_spans, hyp_spans, match_mode)
    return result


@app.get("/api/config")
async def get_config():
    return _load_config()


@app.put("/api/config")
async def update_config(body: dict):
    cfg = _load_config()
    cfg.update(body)
    _save_config(cfg)
    return cfg


@app.get("/api/models/presets")
async def list_model_presets():
    return {"presets": MODEL_PRESETS}
