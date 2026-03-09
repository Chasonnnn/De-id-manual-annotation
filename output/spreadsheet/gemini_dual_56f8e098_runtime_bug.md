# Runtime Bug Ticket: `gemini_pro + Dual + 56f8e098` Hangs

## Summary

`google.gemini-3.1-pro-preview` hangs on doc `56f8e098` when run through the `Dual` method, while nearby control cases complete successfully.

This issue reproduced twice:

- inside the hard-doc regression Methods Lab run `1d1bfc14`
- inside the one-doc rerun `d2b59511`

In both cases, the run stayed `running` on disk with the document still `pending`, and no completed or failed result was ever written for that cell.

## Severity

`P2`

The issue is narrow, but it blocks full completion of targeted regression runs and makes Gemini `Dual` unreliable on at least one known hard doc.

## Isolation Result

This is **not** a generic document failure, **not** a generic Gemini failure, and **not** a generic `Dual` failure.

The evidence points to a **combination-specific runtime issue**:

- provider/model side: `gemini_pro`
- method side: `dual`
- document side: `56f8e098`

Most likely scope:

- `gemini_pro` on one of the `dual`-specific prompts for this transcript under chunked execution

## Implemented Debug Result

After adding persisted runtime breadcrumbs and a doc timeout, the rerun no longer stayed silently pending.

Debug rerun:

- run id: `6d5d5cf6`
- status: `completed_with_errors`
- doc status: `failed`
- error family: `timeout`
- timeout: `45s`

Last persisted runtime breadcrumb before timeout:

- `current_chunk_index = 2`
- `total_chunks = 4`
- `chunk_start = 9987`
- `chunk_end = 19927`
- `current_pass_index = 2`
- `current_pass_label = dual:identifiers`

This narrows the live stall to:

- doc `56f8e098`
- chunk `2/4`
- `Dual` pass `2`
- prompt family `dual:identifiers`

Control rerun on the isolated chunk:

- `gemini_pro + dual:identifiers + chunk 2` completed in about `11.9s`
- `gemini_pro + dual-split:numeric_identifiers + chunk 2` completed in about `2.6s`

So the strongest current conclusion is:

- the chunk text itself is not enough to reproduce the hang in isolation
- the stall emerges inside the full chunked `Dual` execution path
- the most likely trigger is **parallel multi-chunk Gemini fan-out**, with the stuck call occurring on `chunk 2 / dual:identifiers`

## Reproduction Matrix

Doc facts:

- doc id: `56f8e098`
- size: `31,568` chars
- manual spans: `10`
- chunking in successful runs: `4` chunks at about `10,000` chars

Observed outcomes:

| Case | Result | Notes |
| --- | --- | --- |
| Prompt Lab, `gemini_pro + annotator_agents_raw` | completed | same doc succeeds with Gemini in prompt flow |
| Methods Lab, `gemini_pro + Default` | completed | same doc succeeds with Gemini in methods flow |
| Methods Lab, `gemini_pro + Dual Split` | completed | same doc succeeds with Gemini in another two-pass method |
| Methods Lab, `codex_none + Dual` | completed | same method succeeds on same doc with OpenAI |
| Methods Lab, `claude_thinking_off + Dual` | completed | same method succeeds on same doc with Claude |
| Methods Lab, `gemini_pro + Dual` in run `1d1bfc14` | hung | remained `pending` |
| Methods Lab, `gemini_pro + Dual` in rerun `d2b59511` | hung again | remained `pending` |

## Relevant Method Definitions

`Dual` in [agent.py](/Users/chason/De-id-manual-annotation/backend/agent.py#L495) is a two-pass LLM method:

1. names pass
2. identifiers pass

`Dual Split` in [agent.py](/Users/chason/De-id-manual-annotation/backend/agent.py#L527) also uses two LLM passes, but with a different prompt split:

1. names + locations
2. numeric/contact identifiers

Because `Dual Split` completes on the same doc with `gemini_pro`, the likely failure is **not** “Gemini cannot handle two-pass chunked methods” in general. It is more likely tied to one of the **`Dual`-specific prompts** on this transcript.

## Evidence From Successful Controls

Successful control warnings for `56f8e098`:

- `codex_none + Dual`:
  - completed
  - `4` chunks
  - `21` warnings
- `codex_none + Dual Split`:
  - completed
  - `4` chunks
  - `20` warnings
- `gemini_pro + Default`:
  - completed
  - `4` chunks
  - `5` warnings
- `claude_thinking_off + Dual`:
  - completed
  - `4` chunks
  - `4` warnings
- `gemini_pro + annotator_agents_raw`:
  - completed
  - `4` chunks
  - `4` warnings

These controls show:

- the document can complete under chunking
- Gemini can complete the document on other prompt/method paths
- `Dual` can complete the document on other models

## Most Likely Root Cause

Primary hypothesis:

- `gemini_pro` becomes unstable under the full parallel chunked `Dual` fan-out on this doc, and the stuck in-flight request is `chunk 2 / dual:identifiers`

Secondary hypotheses:

- the Cornell gateway or LiteLLM route is mishandling one of the concurrent Gemini requests rather than the prompt text itself
- the issue may require the combination of:
  - `4` chunk workers
  - `2` sequential LLM passes per chunk
  - the specific `dual:identifiers` request shape for chunk `2`

What we can say with confidence:

- this is not a scoring issue
- this is not a manual-annotation issue
- this is not caused by the new NAME-tolerant metric
- this is no longer an “unknown hang location”; we know the last chunk/pass reached before timeout

## Why The Current Platform Makes This Hard To Debug

Current limitations:

- no per-task timeout
- no persisted `in_progress` state at chunk/pass level
- no persisted “entered pass 1/pass 2” markers
- no persisted per-chunk start/end timestamps
- no automatic orphaned-run reconciliation for these stale `running` docs

Because of that, the system only shows `pending`, even after the worker is clearly stuck.

## Recommended Next Fixes

### 1. Instrumentation

Add persisted runtime breadcrumbs for methods execution:

- `current_chunk_index`
- `total_chunks`
- `current_pass_index`
- `current_pass_kind`
- `current_pass_prompt_family`
- `started_at` / `updated_at` for the active doc task

### 2. Timeout

Add a per-doc or per-pass timeout for experiment jobs.

Minimum acceptable behavior:

- if a task exceeds the timeout, mark the doc `failed`
- attach `error_family=timeout`
- do not leave the doc `pending`

### 3. Provider-Specific Logging

Persist request/response metadata for failures and timeouts:

- model
- provider
- method id
- doc id
- chunk index
- pass index
- finish_reason if present
- request elapsed time

### 4. Recovery

On backend startup, reconcile stale `running` runs whose worker no longer exists:

- mark them `interrupted`
- mark still-pending docs `cancelled` or `interrupted`

## Acceptance Criteria For A Fix

The issue is considered fixed when:

1. `gemini_pro + Dual + 56f8e098` reaches a terminal doc state instead of hanging.
2. If it times out, the doc is explicitly marked failed with `error_family=timeout`.
3. The run no longer stays indefinitely `running` because of one stuck Gemini request.
4. The persisted doc result clearly identifies which chunk/pass caused the problem.
