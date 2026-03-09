# Hard-Doc Resolution V2 Replay

- Policy version: `2026-03-span-resolution-v2`
- This replay uses the saved `raw_hypothesis_spans` from the live hard-doc reruns and reapplies the current resolver locally.
- No new model calls were made; this isolates the resolver change itself.

## Against Raw Exact
- Positive: `1`
- Negative: `12`
- Flat: `70`

## Against Prior V1 Resolver
- Improved: `55`
- Worse: `0`
- Unchanged: `28`

## Biggest Improvements Over V1
- `prompt / codex_none / annotator_agents_raw / 56f8e098`: v1 `0.3750` -> v2 `0.7500` (`+0.3750`)
- `methods / codex_none / Default / 56f8e098`: v1 `0.4706` -> v2 `0.8235` (`+0.3529`)
- `methods / codex_none / Dual Split / 56f8e098`: v1 `0.2000` -> v2 `0.5000` (`+0.3000`)
- `methods / claude_thinking_off / Dual / 56f8e098`: v1 `0.3750` -> v2 `0.6250` (`+0.2500`)
- `methods / claude_thinking_off / Dual / d4df4c18`: v1 `0.4364` -> v2 `0.6727` (`+0.2364`)
- `prompt / codex_none / annotator_agents_raw / d4df4c18`: v1 `0.4000` -> v2 `0.6286` (`+0.2286`)
- `methods / codex_none / Dual / d4df4c18`: v1 `0.4000` -> v2 `0.6286` (`+0.2286`)
- `methods / claude_thinking_off / Default / 56f8e098`: v1 `0.1111` -> v2 `0.3333` (`+0.2222`)

## Remaining Differences From Raw
- `methods / claude_thinking_off / Dual / f5d5b0e9`: raw `0.8727` -> v2 `0.8649` (`-0.0079`), augmentations `2`
- `prompt / claude_thinking_off / annotator_agents_raw / f5d5b0e9`: raw `0.7980` -> v2 `0.7902` (`-0.0078`), augmentations `2`
- `methods / codex_none / Dual / f5d5b0e9`: raw `0.7980` -> v2 `0.7902` (`-0.0078`), augmentations `2`
- `methods / claude_thinking_off / Dual Split / f5d5b0e9`: raw `0.8584` -> v2 `0.8507` (`-0.0078`), augmentations `2`
- `methods / claude_thinking_off / Default / f5d5b0e9`: raw `0.8479` -> v2 `0.8402` (`-0.0077`), augmentations `2`
- `methods / gemini_pro / Dual Split / f5d5b0e9`: raw `0.8821` -> v2 `0.8745` (`-0.0076`), augmentations `2`
- `methods / gemini_pro / Dual / f5d5b0e9`: raw `0.8722` -> v2 `0.8646` (`-0.0076`), augmentations `2`
- `methods / codex_none / Dual Split / f5d5b0e9`: raw `0.7685` -> v2 `0.7610` (`-0.0075`), augmentations `2`

## Artifact
- `/Users/chason/De-id-manual-annotation/output/spreadsheet/hard_doc_resolution_v2_replay/doc_replay.csv`
