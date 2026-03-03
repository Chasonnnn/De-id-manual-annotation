# Project Rules

## 0) Package Manager Standard
- Use `uv` for all Python dependency and environment management.
- Do not use `pip install ...` workflows in project docs or CI unless required to bootstrap `uv`.
- Preferred commands: `uv sync`, `uv add`, `uv remove`, `uv run ...`.

## 1) Latest Dependencies Only
- Always use current stable package versions when adding or upgrading dependencies.
- Do not intentionally pin to older package versions for compatibility.
- When touching dependency manifests, upgrade related packages rather than preserving legacy ranges.

## 2) No Backward Compatibility Requirement (Pre-Release)
- This project is under active development and does not require backward compatibility.
- Breaking API/schema changes are allowed when they simplify architecture or speed delivery.
- Prefer removing transitional compatibility layers instead of maintaining legacy behavior.

## 7) No Fallback Routes
- No fallback routes. Expose issues so we can fix them.
- Do not hide integration or runtime errors behind silent/default route behavior.
- Return explicit failure responses with actionable error details.

## 8) Official Documentation First (Planning)
- When making plans, implementation decisions, or technical recommendations, check official documentation first.
- Prefer primary sources (official docs/specs/repos) over memory or third-party summaries.
- If documentation is missing or ambiguous, call it out explicitly and list assumptions.
