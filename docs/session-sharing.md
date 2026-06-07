# Session Sharing

Use exported session bundles as the GitHub sharing format. Do not share or sync
`backend/.annotation_tool/` as the canonical workflow.

## Export a Bundle

1. Start the app with `./run.sh`.
2. In the sidebar, set `Export Type` to `Full Session Bundle`.
3. Click `Export Full Session`.
4. Save the downloaded `annotation-session-<timestamp>.json` in the private
   GitHub repo location your team uses for intentional shared sessions.

Full session bundles include documents, manual annotations, folders, Prompt Lab
runs, Methods Lab runs, and an export summary with bundle version and counts.

## Commit Only Intentional Files

Commit exported bundle files that colleagues should import. Do not commit:

- `backend/.annotation_tool/`
- `.env.local`
- API keys, LiteLLM keys, or provider keys
- ad hoc browser downloads that are not meant to be imported

Keep bundle filenames descriptive enough for review, for example:

```text
sessions/2026-06-07-upchieve-methods-reviewed.annotation-session.json
```

## Import a Colleague Bundle

1. Pull the latest private repo changes.
2. Start the app with `./run.sh`.
3. Choose the sidebar `Import Conflicts` policy.
4. Drop the exported `.json` session bundle into the sidebar import area.

Conflict policies:

- `Replace Current`: update matching imported document IDs and keep imported
  annotations/runs as the source of truth.
- `Add as New`: keep both copies by creating new IDs for conflicting imported
  documents.
- `Keep Current`: skip incoming documents that conflict with local document IDs.

Use `Replace Current` when reviewing the same shared session together. Use
`Add as New` when comparing two versions side by side. Use `Keep Current` when
you only want non-conflicting new material from a bundle.

## Local Runtime Data

The app persists live working data under:

```text
backend/.annotation_tool/
```

That directory is intentionally ignored by Git. Treat it as local runtime state,
not the team sharing format. Export a full session bundle whenever a colleague
should receive or review your current state.
