# De-id-manual-annotation

## Start frontend + backend together

From repo root:

```bash
./run.sh
```

First-time setup (installs dependencies, then starts both):

```bash
./run.sh --install
```

You can override ports:

```bash
BACKEND_PORT=8001 FRONTEND_PORT=5174 ./run.sh
```

If a root `.env.local` file exists, `run.sh` auto-loads it.

## LiteLLM gateway/proxy config

You can set key and base URL in two ways:

1. In UI (Agent panel):
- `API Key`
- `LiteLLM Base URL`

2. Environment fallback:
- `LITELLM_API_KEY`
- `LITELLM_BASE_URL`
- provider keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `GOOGLE_API_KEY`)

## Where results are saved (DB or files?)

Current implementation is **file-based**, not a database.

Storage root:

```text
backend/.annotation_tool/
```

Important files:
- `backend/.annotation_tool/config.json` (saved config defaults)
- `backend/.annotation_tool/sessions/default/_index.json` (document index)
- `backend/.annotation_tool/sessions/default/<doc_id>.source.json` (parsed source doc)
- `backend/.annotation_tool/sessions/default/<doc_id>.manual.json` (manual annotations)
- `backend/.annotation_tool/sessions/default/<doc_id>.agent.rule.json` (rule-agent output)
- `backend/.annotation_tool/sessions/default/<doc_id>.agent.llm.json` (LLM-agent output)

So right now persistence is local JSON sidecars per document/session.
