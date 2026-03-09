# Hard-Doc Resolution V1 Regression

## Result
- Backend tests passed: `238/238`.
- The live hard-doc reruns completed for Prompt Lab (`a51ceb42`) and Methods Lab (`f73189f2`).
- The current resolution pipeline is **not ready to keep enabled for official exact scoring**.
- On this slice, the only clearly helpful rule was `url_multiline_domain`; the current NAME rules regressed exact scoring.

## Aggregate Signal
- Cell-level resolved-vs-raw exact: `0` positive, `12` negative, `0` flat.
- Doc-level resolved-vs-raw exact: `1` positive, `55` negative, `27` flat completed docs.
- The remaining failed case was the known `gemini_pro + Dual + 56f8e098`, now surfaced correctly as `timeout` instead of hanging forever.

## What Helped
- `methods_lab / gemini_pro / Default / fe8b0fe8`: raw `0.0000` -> resolved `0.5000` (`+0.5000`)

## What Hurt
- `prompt_lab / codex_none / annotator_agents_raw / 56f8e098`: raw `0.7500` -> resolved `0.3750` (`-0.3750`)
- `methods_lab / codex_none / Default / 56f8e098`: raw `0.8235` -> resolved `0.4706` (`-0.3529`)
- `methods_lab / codex_none / Dual Split / 56f8e098`: raw `0.5000` -> resolved `0.2000` (`-0.3000`)
- `methods_lab / claude_thinking_off / Dual / 56f8e098`: raw `0.6250` -> resolved `0.3750` (`-0.2500`)
- `methods_lab / claude_thinking_off / Dual / d4df4c18`: raw `0.6727` -> resolved `0.4364` (`-0.2364`)
- `prompt_lab / codex_none / annotator_agents_raw / d4df4c18`: raw `0.6286` -> resolved `0.4000` (`-0.2286`)
- `methods_lab / codex_none / Dual / d4df4c18`: raw `0.6286` -> resolved `0.4000` (`-0.2286`)
- `methods_lab / claude_thinking_off / Default / 56f8e098`: raw `0.3333` -> resolved `0.1111` (`-0.2222`)
- `methods_lab / gemini_pro / Default / d4df4c18`: raw `0.7969` -> resolved `0.5781` (`-0.2187`)
- `methods_lab / gemini_pro / Dual / d4df4c18`: raw `0.6909` -> resolved `0.4727` (`-0.2182`)

## Root Cause From Live Examples
- `name_terminal_punctuation` is overshooting. Example from `codex_none / annotator_agents_raw / 56f8e098`: raw correct spans `Michael` and `Raymond` were changed into `Michael.` and `Raymond.`, which created exact false positives and false negatives.
- The same rule also expands into ellipses, e.g. `Michael...`, `Tony.`, `Okafor...`, which is not how the manual ground truth is annotated.
- `name_trailing_possessive` is also overshooting in the current benchmark. Examples like `Opeyemi` -> `Opeyemi's` became exact mismatches instead of improvements.
- `url_multiline_domain` did help exactly as intended. Example from `gemini_pro / Default / fe8b0fe8`: raw `https://rodriguez.com/since
` was corrected to `https://rodriguez.com/since`, improving exact F1 from `0.0` to `0.5` on that doc.

## Recommendation
- Keep the new telemetry: `raw_hypothesis_spans`, `raw_metrics`, `resolution_events`, `resolution_policy_version`, timeout diagnostics.
- Disable or sharply narrow these v1 resolution rules before broader rollout:
  - `name_terminal_punctuation`
  - `name_trailing_possessive`
  - likely `name_honorific_prefix` and `misc_id_context` too, pending a cleaner positive-control suite
- Keep `exact_name_affix_tolerant` as the diagnostic companion metric; it is still useful for analysis without corrupting official exact spans.
- Keep `url_multiline_domain`; it produced the one clear exact-score improvement in the live rerun.

## Artifacts
- `/Users/chason/De-id-manual-annotation/output/spreadsheet/hard_doc_resolution_v1/summary.md`
- `/Users/chason/De-id-manual-annotation/output/spreadsheet/hard_doc_resolution_v1/cell_delta.csv`
- `/Users/chason/De-id-manual-annotation/output/spreadsheet/hard_doc_resolution_v1/doc_delta.csv`
