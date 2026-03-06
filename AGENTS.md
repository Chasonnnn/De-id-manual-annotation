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
- Hard rule: do not add compatibility shims, dual paths, legacy-preserving branches, or transitional behavior unless the user explicitly asks for them.

## 3) TDD Is Required
- Follow test-driven development for backend work:
  1. Write or adjust failing tests first.
  2. Implement the minimal code to pass tests.
  3. Refactor while keeping tests green.
- No feature is complete without automated tests that cover expected behavior and key failure modes.

## 4) Reproduce Admin-Reported Bugs First
- When admin reports a bug, first reproduce it with code before changing implementation.
- After reproduction, identify the root cause, then implement and verify the fix.

## 7) No Fallback Routes
- No fallback routes. Expose issues so they can be fixed.
- Do not hide integration or runtime errors behind silent/default route behavior.
- Hard rule: do not silently downgrade behavior, switch models or modes, retry with a different strategy, or substitute an alternative implementation.
- Return explicit failure responses with actionable error details.

## 8) Official Documentation First (Planning)
- When making plans, implementation decisions, or technical recommendations, check official documentation first.
- Prefer primary sources such as official docs, specs, and repositories over memory or third-party summaries.
- If documentation is missing or ambiguous, call it out explicitly and list assumptions.
