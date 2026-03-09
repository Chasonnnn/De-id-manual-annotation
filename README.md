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

Prompt Lab and Methods Lab runs are also session-backed sidecars:
- `backend/.annotation_tool/sessions/default/prompt_lab/_index.json`
- `backend/.annotation_tool/sessions/default/prompt_lab/<run_id>.json`
- `backend/.annotation_tool/sessions/default/methods_lab/_index.json`
- `backend/.annotation_tool/sessions/default/methods_lab/<run_id>.json`

## Experiment CLI

Run the backend CLI from `backend/`:

```bash
cd backend
uv sync
uv run annotation-experiments list-docs
```

Available commands:

```bash
uv run annotation-experiments run manifest.yaml
uv run annotation-experiments prompt ...
uv run annotation-experiments methods ...
uv run annotation-experiments list-docs --session default
uv run annotation-experiments list-models
uv run annotation-experiments list-methods
```

Behavior:
- runs read documents from `.annotation_tool/sessions/<session>/`
- prompt runs default to all docs in the session when `doc_ids` are omitted
- methods runs default to docs with manual annotations when `doc_ids` are omitted
- run artifacts are persisted to the same Prompt Lab / Methods Lab JSON files used by the UI
- `AGENTS.md` and `SKILL.md` are supported only as prompt text sources for LiteLLM runs
- `SKILL.md` strips optional YAML frontmatter before use
- `Skills.md` is not supported; use the actual file name `SKILL.md`

Optional outputs:

```bash
uv run annotation-experiments run manifest.yaml \
  --output-json /tmp/run.json \
  --output-csv /tmp/run.csv
```

### Prompt Flags

Inline prompt variants:

```bash
uv run annotation-experiments prompt \
  --session default \
  --prompt 'Baseline=Extract PII spans as strict JSON.' \
  --model 'Codex=openai.gpt-5.3-codex' \
  --api-key "$LITELLM_API_KEY" \
  --api-base "$LITELLM_BASE_URL"
```

Prompt files:

```bash
uv run annotation-experiments prompt \
  --session default \
  --prompt-file 'Agents=../prompts/AGENTS.md' \
  --model 'Codex=openai.gpt-5.3-codex' \
  --api-key "$LITELLM_API_KEY" \
  --api-base "$LITELLM_BASE_URL"
```

Preset prompt variants backed by an existing method definition:

```bash
uv run annotation-experiments prompt \
  --session default \
  --preset 'Default Prompt=default' \
  --model 'Codex=openai.gpt-5.3-codex' \
  --api-key "$LITELLM_API_KEY" \
  --api-base "$LITELLM_BASE_URL"
```

### Methods Flags

```bash
uv run annotation-experiments methods \
  --session default \
  --method 'Default=default' \
  --method 'Reasoning=reasoning' \
  --model 'Codex=openai.gpt-5.3-codex' \
  --api-key "$LITELLM_API_KEY" \
  --api-base "$LITELLM_BASE_URL"
```

### Manifest Schema

Prompt Lab manifests use `kind: prompt_lab` and include `prompts` plus `models`:

```yaml
kind: prompt_lab
session: default
name: inline-prompt-sweep
doc_ids:
  - doc_1
prompts:
  - id: baseline
    label: Baseline
    system_prompt: Extract PII spans as strict JSON.
models:
  - id: codex
    label: Codex
    model: openai.gpt-5.3-codex
    reasoning_effort: xhigh
runtime:
  api_key: ${LITELLM_API_KEY}
  api_base: ${LITELLM_BASE_URL}
  temperature: 0.0
  match_mode: exact
  reference_source: manual
  fallback_reference_source: pre
concurrency: 12
```

`AGENTS.md` prompt file example:

```yaml
kind: prompt_lab
session: default
name: agents-file-prompt
prompts:
  - id: agents
    label: Agents Instructions
    prompt_file: ../prompts/AGENTS.md
models:
  - id: codex
    label: Codex
    model: openai.gpt-5.3-codex
runtime:
  api_key: ${LITELLM_API_KEY}
  api_base: ${LITELLM_BASE_URL}
```

`SKILL.md` prompt file example:

```yaml
kind: prompt_lab
session: default
name: skill-file-prompt
prompts:
  - id: skill
    label: Skill Instructions
    prompt_file: ../skills/example/SKILL.md
models:
  - id: codex
    label: Codex
    model: openai.gpt-5.3-codex
runtime:
  api_key: ${LITELLM_API_KEY}
  api_base: ${LITELLM_BASE_URL}
```

Methods Lab manifests use `kind: methods_lab` and include `methods` plus `models`:

```yaml
kind: methods_lab
session: default
name: method-sweep
methods:
  - id: default_method
    label: Default
    method_id: default
  - id: reasoning_method
    label: Reasoning
    method_id: reasoning
models:
  - id: codex
    label: Codex
    model: openai.gpt-5.3-codex
    reasoning_effort: xhigh
runtime:
  api_key: ${LITELLM_API_KEY}
  api_base: ${LITELLM_BASE_URL}
  temperature: 0.0
  match_mode: exact
concurrency: 8
```

Concurrency notes:
- Prompt Lab and Methods Lab default to `4` workers.
- The server-configurable max defaults to `16` and is hard-capped at `32`.
- Higher concurrency mainly helps LLM-backed sweeps. Local Presidio-heavy runs benefit less.

## Qwen3.5 9B local runner

If you want to run the local `Qwen3.5-9B` MLX-VLM checkpoint from this repo,
use:

```bash
python3 backend/scripts/qwen35_9b_generate.py \
  --prompt 'Return exactly this JSON and nothing else: {"status":"ok"}'
```

Notes:
- The script defaults to `mlx-community/Qwen3.5-9B-MLX-4bit`.
- Thinking mode is disabled by default. Add `--enable-thinking` to turn it on.
- If you hit memory pressure on the M2 Max, retry with `--prefill-step-size 512`.
- Required dependency:

```bash
cd backend
uv add "mlx-vlm[torch]"
```
