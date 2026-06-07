import { afterEach, describe, expect, it, vi } from "vitest";

import {
  exportGroundTruth,
  getHealth,
  getWorkspace,
  listMethodsLabRuns,
  listPromptLabRuns,
  mirrorMethodToManual,
  mirrorPreToManual,
} from "./client";

describe("run summary normalization", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("preserves the deidentify-v2 method bundle for Methods Lab run summaries", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response(
          JSON.stringify({
            runs: [
              {
                id: "run-1",
                name: "Colleague demo V2",
                status: "running",
                method_bundle: "deidentify-v2",
                cancellable: true,
                created_at: "2026-03-12T00:00:00Z",
                started_at: "2026-03-12T00:00:01Z",
                finished_at: null,
                doc_count: 27,
                method_count: 3,
                model_count: 1,
                total_tasks: 81,
                completed_tasks: 24,
                failed_tasks: 0,
              },
            ],
          }),
          {
            status: 200,
            headers: { "Content-Type": "application/json" },
          },
        ),
      ),
    );

    const runs = await listMethodsLabRuns();

    expect(runs[0]?.method_bundle).toBe("deidentify-v2");
  });

  it("preserves the deidentify-v2 method bundle for Prompt Lab run summaries", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response(
          JSON.stringify({
            runs: [
              {
                id: "run-1",
                name: "Colleague demo V2",
                status: "running",
                method_bundle: "deidentify-v2",
                cancellable: true,
                created_at: "2026-03-12T00:00:00Z",
                started_at: "2026-03-12T00:00:01Z",
                finished_at: null,
                doc_count: 27,
                prompt_count: 3,
                model_count: 1,
                total_tasks: 81,
                completed_tasks: 24,
                failed_tasks: 0,
              },
            ],
          }),
          {
            status: 200,
            headers: { "Content-Type": "application/json" },
          },
        ),
      ),
    );

    const runs = await listPromptLabRuns();

    expect(runs[0]?.method_bundle).toBe("deidentify-v2");
  });
});

describe("readiness and workspace", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  const healthPayload = {
    status: "warning",
    tool_version: "2026.03.03",
    storage: {
      root: "/tmp/backend/.annotation_tool",
      session_dir: "/tmp/backend/.annotation_tool/sessions/default",
      exists: true,
    },
    counts: {
      documents: 1,
      folders: 1,
      prompt_lab_runs: 2,
      methods_lab_runs: 3,
    },
    credentials: {
      has_api_key: true,
      api_key_sources: ["LITELLM_API_KEY"],
      has_api_base: true,
      api_base_sources: ["LITELLM_BASE_URL"],
    },
    method_availability_warnings: [
      {
        id: "pipeline",
        label: "Local Pipeline",
        reason: "DEID_PIPELINE_REPO_ROOT is not configured",
      },
    ],
    config_warnings: ["Config warning"],
    dependency_warnings: ["Dependency warning"],
  };

  it("normalizes the readiness health endpoint", async () => {
    const fetchMock = vi.fn(async () =>
      new Response(JSON.stringify(healthPayload), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    const health = await getHealth();

    expect(fetchMock).toHaveBeenCalledWith("/api/health", undefined);
    expect(health.status).toBe("warning");
    expect(health.counts.methods_lab_runs).toBe(3);
    expect(health.credentials.api_key_sources).toEqual(["LITELLM_API_KEY"]);
    expect(health.method_availability_warnings[0]?.reason).toContain("DEID_PIPELINE");
  });

  it("normalizes the workspace endpoint without requiring folder details", async () => {
    const fetchMock = vi.fn(async () =>
      new Response(
        JSON.stringify({
          documents: [
            {
              id: "doc-1",
              filename: "session.json",
              display_name: "Session 1",
              status: "pending",
            },
          ],
          folders: [
            {
              id: "folder-1",
              name: "Review Queue",
              kind: "manual",
              parent_folder_id: null,
              merged_doc_id: null,
              doc_count: 10,
              child_folder_count: 0,
              source_filename: null,
              source_folder_id: null,
              sample_size: null,
              sample_seed: null,
              created_at: "2026-03-17T00:00:00Z",
            },
          ],
          folder_details: {},
          health: healthPayload,
        }),
        {
          status: 200,
          headers: { "Content-Type": "application/json" },
        },
      ),
    );
    vi.stubGlobal("fetch", fetchMock);

    const workspace = await getWorkspace();

    expect(fetchMock).toHaveBeenCalledWith("/api/workspace", undefined);
    expect(workspace.documents[0]?.display_name).toBe("Session 1");
    expect(workspace.folders[0]?.doc_count).toBe(10);
    expect(workspace.folder_details).toEqual({});
    expect(workspace.health.counts.prompt_lab_runs).toBe(2);
  });
});

describe("ground-truth export", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("serializes top-level export scope explicitly", async () => {
    const fetchMock = vi.fn(async () => new Response(new Blob(["zip"]), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    await exportGroundTruth("manual", { kind: "top_level" });

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/session/export-ground-truth?source=manual&scope=top_level",
    );
  });

  it("serializes folder export scope with folder_id", async () => {
    const fetchMock = vi.fn(async () => new Response(new Blob(["zip"]), { status: 200 }));
    vi.stubGlobal("fetch", fetchMock);

    await exportGroundTruth("manual", { kind: "folder", folderId: "folder-import" });

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/session/export-ground-truth?source=manual&scope=folder&folder_id=folder-import",
    );
  });
});

describe("mirror pre to manual", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("serializes top-level mirror scope explicitly", async () => {
    const fetchMock = vi.fn(async () =>
      new Response(JSON.stringify({ processed_count: 3 }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await mirrorPreToManual({ kind: "top_level" });

    expect(fetchMock).toHaveBeenCalledWith("/api/session/mirror-pre-to-manual?scope=top_level", {
      method: "POST",
    });
  });

  it("serializes folder mirror scope with folder_id", async () => {
    const fetchMock = vi.fn(async () =>
      new Response(JSON.stringify({ processed_count: 2 }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await mirrorPreToManual({ kind: "folder", folderId: "folder-import" });

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/session/mirror-pre-to-manual?scope=folder&folder_id=folder-import",
      { method: "POST" },
    );
  });
});

describe("mirror method to manual", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("serializes folder mirror scope with method run and cell ids", async () => {
    const fetchMock = vi.fn(async () =>
      new Response(JSON.stringify({ processed_count: 2 }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await mirrorMethodToManual(
      { kind: "folder", folderId: "folder-import" },
      { runId: "run-1", cellId: "cell-a" },
    );

    expect(fetchMock).toHaveBeenCalledWith(
      "/api/session/mirror-method-to-manual?scope=folder&folder_id=folder-import&run_id=run-1&cell_id=cell-a",
      { method: "POST" },
    );
  });
});
