# Leaderboard Report

Source files:
- `output/spreadsheet/prompt_lab_15fad00a_all_results_wide.csv`
- `output/spreadsheet/methods_lab_5bcf6ed4_all_results_wide.csv`

Generated: 2026-03-08

## Scope

- Prompt export combines:
  - `15fad00a` OpenAI prompt run
  - `8fa104db` Gemini prompt run
- Methods export combines:
  - `5bcf6ed4` Codex methods run
  - `9d9278e6` Gemini methods run
- Important caveat:
  - Prompt comparisons are partially biased by the incomplete `chat52_none` run.
  - For fair prompt-variant ranking, use only fully completed model-prompt cells.

## Executive Summary

- Best overall model in the current exports: `gemini_pro`
- Best production candidate right now: `gemini_pro + Dual Split`
- Highest raw-scoring method cell: `gemini_pro + Dual` at `86.1%` micro F1, but it had `1/16` failed docs
- Best stable method cell: `gemini_pro + Dual Split` at `86.0%` micro F1 with `0/16` failed docs
- Best prompt-only cell: `gemini_pro + preset_extended_verify_off` at `84.5%` micro F1 with `0/16` failed docs
- Best cross-model prompt template: `annotator_agents_raw`
- Best OpenAI prompt baseline: `codex_none + annotator_agents_raw` at `77.5%`
- Best OpenAI method baseline: `Codex + Dual` at `75.5%`

## Model Leaderboard

### Prompt Lab

| Model | Completed | Failed | Pending | Failure Rate | Micro F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|---:|
| `gemini_pro` | 96 | 0 | 0 | 0.0% | 82.6% | 84.1% | 81.0% |
| `gemini_flash_lite` | 85 | 11 | 0 | 11.5% | 80.8% | 85.4% | 76.7% |
| `codex_xhigh` | 96 | 0 | 0 | 0.0% | 73.5% | 84.1% | 65.2% |
| `codex_none` | 96 | 0 | 0 | 0.0% | 73.4% | 84.0% | 65.2% |
| `chat52_xhigh` | 96 | 0 | 0 | 0.0% | 59.9% | 88.3% | 45.4% |
| `chat52_none` | 15 | 0 | 81 | n/a | 52.0% | 82.0% | 38.1% |

### Methods Lab

| Model | Completed | Failed | Failure Rate | Micro F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| `gemini_pro` | 127 | 1 | 0.8% | 81.4% | 82.1% | 80.7% |
| `gemini_flash_lite` | 114 | 14 | 10.9% | 77.2% | 79.7% | 74.8% |
| `Codex` | 128 | 0 | 0.0% | 70.7% | 77.6% | 64.9% |

## Prompt Leaderboard

### Best prompt cells

| Rank | Model | Prompt | Completed | Failed | Failure Rate | Micro F1 |
|---|---|---|---:|---:|---:|---:|
| 1 | `gemini_pro` | `preset_extended_verify_off` | 16 | 0 | 0.0% | 84.5% |
| 2 | `gemini_pro` | `preset_default_verify_off` | 16 | 0 | 0.0% | 83.6% |
| 3 | `gemini_flash_lite` | `annotator_agents_raw` | 13 | 3 | 18.8% | 83.2% |
| 4 | `gemini_pro` | `annotator_agents_raw` | 16 | 0 | 0.0% | 82.6% |
| 5 | `gemini_pro` | `preset_default_verify_on` | 16 | 0 | 0.0% | 81.9% |

### Fair prompt-variant ranking across fully completed model cells

| Prompt Variant | Mean F1 | Mean Failure Rate | Notes |
|---|---:|---:|---|
| `annotator_agents_raw` | 78.6% | 3.8% | Best cross-model prompt family |
| `preset_default_verify_on` | 74.6% | 3.8% | Strong but less stable than extended-off |
| `preset_extended_verify_off` | 73.2% | 0.0% | Best stable preset-backed prompt |
| `preset_default_verify_off` | 73.0% | 3.8% | Slightly below verify-on |
| `baseline_raw` | 72.8% | 2.5% | Useful control, not best |
| `preset_extended_verify_on` | 71.7% | 0.0% | Verifier hurt this preset overall |

## Method Leaderboard

### Best method cells

| Rank | Model | Method | Completed | Failed | Failure Rate | Micro F1 |
|---|---|---|---:|---:|---:|---:|
| 1 | `gemini_pro` | `Dual` | 15 | 1 | 6.3% | 86.1% |
| 2 | `gemini_pro` | `Dual Split` | 16 | 0 | 0.0% | 86.0% |
| 3 | `gemini_flash_lite` | `Dual Split` | 14 | 2 | 12.5% | 84.5% |
| 4 | `gemini_pro` | `Extended` | 16 | 0 | 0.0% | 84.5% |
| 5 | `gemini_pro` | `Presidio + LLM Split` | 16 | 0 | 0.0% | 84.4% |

### Overall method ranking

| Method | Completed | Failed | Failure Rate | Micro F1 | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| `Dual Split` | 46 | 2 | 4.2% | 81.2% | 85.9% | 77.1% |
| `Dual` | 45 | 3 | 6.3% | 81.1% | 87.1% | 75.8% |
| `Extended` | 48 | 0 | 0.0% | 79.3% | 85.9% | 73.5% |
| `Verified` | 45 | 3 | 6.3% | 78.2% | 85.4% | 72.2% |
| `Presidio + Default` | 45 | 3 | 6.3% | 77.8% | 79.1% | 76.5% |
| `Presidio + LLM Split` | 47 | 1 | 2.1% | 77.5% | 81.0% | 74.2% |
| `Default` | 45 | 3 | 6.3% | 77.2% | 79.4% | 75.2% |
| `Presidio` | 48 | 0 | 0.0% | 62.5% | 61.2% | 63.8% |

## Label-Level Findings

- `NAME` is the main differentiator.
  - Prompt Lab `NAME` F1:
    - `gemini_pro`: `86.6%`
    - `gemini_flash_lite`: `85.0%`
    - `codex_none`: `76.9%`
    - `codex_xhigh`: `76.6%`
- `LOCATION` remains mediocre for all model families.
- `DATE`, `AGE`, and `MISC_ID` are weak across the board.
- `SCHOOL` appears misaligned with the current evaluation set.
  - Support is effectively zero in the exports, but models still predict it, so this becomes pure false-positive pressure.

## Reliability Findings

- `gemini_pro` is both the strongest and the most stable model in the current exports.
- `gemini_flash_lite` is accuracy-competitive, but repeated failures make it a weaker production choice.
- Prompt-side `gemini_flash_lite` failures cluster around a small set of repeated docs:
  - `c9c3cf4c`
  - `d4df4c18`
  - `35abe043`
  - `f5d5b0e9`
- Methods-side `gemini_flash_lite` failures show the same pattern.
- `claude_thinking_on` should be treated as non-viable for this workflow based on runtime behavior, not just score.

## Recommended Production Choice

### Primary

- `gemini_pro + Dual Split`

Why:
- highest stable method result
- `86.0%` micro F1
- `0/16` failed docs in the best cell
- strongest overall model-family quality and stability mix

### Secondary

- `gemini_pro + Dual`

Why:
- slightly higher peak score: `86.1%`
- but less stable than `Dual Split`

### Best prompt-only baseline

- `gemini_pro + preset_extended_verify_off`

Why:
- best prompt-only score: `84.5%`
- stable
- useful if you want a simpler prompt-lab-style configuration without moving to full method orchestration

### Best OpenAI fallback

- `codex_none + annotator_agents_raw`

Why:
- strongest OpenAI prompt result: `77.5%`
- `xhigh` offered no meaningful net advantage over `none`

## Recommended Next Experiments

1. Run a focused confirmation sweep on only the finalists:
   - `gemini_pro + Dual Split`
   - `gemini_pro + Dual`
   - `gemini_pro + Extended`
   - `gemini_pro + preset_extended_verify_off`
   - `codex_none + annotator_agents_raw`
   - `codex_none + Dual`

2. Run a hard-doc failure study on the repeated failure documents:
   - `c9c3cf4c`
   - `d4df4c18`
   - `35abe043`
   - `f5d5b0e9`
   - `56f8e098`

3. Do a label-taxonomy cleanup experiment:
   - tighten `SCHOOL`
   - add stronger positive examples for `MISC_ID`
   - improve handling rules for `DATE` and `AGE`

4. Run a prompt ablation on `annotator_agents_raw`:
   - keep the participant-memory guidance
   - test removing low-value verbosity
   - test adding explicit `LOCATION` and `MISC_ID` counterexamples

5. Improve runtime reliability before the next large sweep:
   - add per-task timeout
   - persist `in_progress`
   - reconcile orphaned `running` runs after backend restart
   - keep provider-separated run batches

## Bottom Line

- Gemini beat the current OpenAI baselines clearly.
- Methods beat prompt-only configurations for Gemini.
- The best balance of quality and stability is `gemini_pro + Dual Split`.

## Boundary-Tolerant Diagnostic

Hard-doc regression artifacts:
- `output/spreadsheet/hard_doc_regression/summary.md`
- `output/spreadsheet/hard_doc_regression/cell_comparison.csv`
- `output/spreadsheet/hard_doc_regression/doc_comparison.csv`

- Scope: `7` hard docs x `3` stable models x `1` prompt + `3` methods = `84` doc-runs.
- Completed evidence: `83/84` doc-runs produced usable comparisons. The remaining case, `gemini_pro + Dual + 56f8e098`, hung twice, including a one-doc rerun.
- Model-level average gain from `exact` to `exact_name_affix_tolerant` on completed docs:
- `claude_thinking_off`: exact `0.7369` -> tolerant `0.8843` (`+0.1474`) across `28` completed docs
- `codex_none`: exact `0.6611` -> tolerant `0.7894` (`+0.1284`) across `28` completed docs
- `gemini_pro`: exact `0.7638` -> tolerant `0.9190` (`+0.1552`) across `27` completed docs
- Strongest improving docs were boundary-loss cases:
- `9bb358ed`: avg exact `0.7049` -> tolerant `0.9319` (`+0.2269`), improved `12/12` completed evaluations
- `26028249`: avg exact `0.7505` -> tolerant `0.9442` (`+0.1937`), improved `12/12` completed evaluations
- `d4df4c18`: avg exact `0.6710` -> tolerant `0.8336` (`+0.1626`), improved `12/12` completed evaluations
- `f5d5b0e9`: avg exact `0.7553` -> tolerant `0.8872` (`+0.1318`), improved `12/12` completed evaluations
- Control docs stayed flat:
- `56f8e098`: avg exact `0.5795` -> tolerant `0.5795` (`+0.0000`), improved `0/11` completed evaluations
- `ad20d7f7`: avg exact `0.7208` -> tolerant `0.7208` (`+0.0000`), improved `0/12` completed evaluations
- `fe8b0fe8`: avg exact `0.4833` -> tolerant `0.4833` (`+0.0000`), improved `0/12` completed evaluations
- Conclusion: `exact_name_affix_tolerant` is behaving as intended. It recovers NAME boundary-affix misses without inflating unrelated error classes.
