# Hard-Doc Regression Summary

- Prompt run: `e521988e` (`completed`)
- Methods run: `1d1bfc14` (`running` on disk, but analysis fixed at `62/63` completed after the worker was stopped)
- Supplemental rerun: `d2b59511` for `gemini_pro + Dual + 56f8e098` also hung without producing a completed doc result

## Model-Level Delta
- `claude_thinking_off`: exact `0.7369` -> tolerant `0.8843` (`+0.1474`) across `28` completed docs
- `codex_none`: exact `0.6611` -> tolerant `0.7894` (`+0.1284`) across `28` completed docs
- `gemini_pro`: exact `0.7638` -> tolerant `0.9190` (`+0.1552`) across `27` completed docs

## Boundary-Sensitive Docs That Improved
- `9bb358ed`: avg exact `0.7049` -> tolerant `0.9319` (`+0.2269`), improved `12/12` completed evaluations
- `26028249`: avg exact `0.7505` -> tolerant `0.9442` (`+0.1937`), improved `12/12` completed evaluations
- `d4df4c18`: avg exact `0.6710` -> tolerant `0.8336` (`+0.1626`), improved `12/12` completed evaluations
- `f5d5b0e9`: avg exact `0.7553` -> tolerant `0.8872` (`+0.1318`), improved `12/12` completed evaluations

## Control Docs With No Change
- `56f8e098`: avg exact `0.5795` -> tolerant `0.5795` (`+0.0000`), improved `0/11` completed evaluations
- `ad20d7f7`: avg exact `0.7208` -> tolerant `0.7208` (`+0.0000`), improved `0/12` completed evaluations
- `fe8b0fe8`: avg exact `0.4833` -> tolerant `0.4833` (`+0.0000`), improved `0/12` completed evaluations

## Interpretation
- The tolerant metric consistently improves the known NAME-boundary docs and does not move the non-boundary controls.
- `9bb358ed`, `26028249`, `d4df4c18`, and `f5d5b0e9` are the core validation set for boundary-loss recovery.
- `56f8e098`, `ad20d7f7`, and `fe8b0fe8` staying flat is the evidence that the metric is not inflating unrelated errors.
