import { afterEach, describe, expect, it, vi } from "vitest";

import {
  exportGroundTruth,
  listMethodsLabRuns,
  listPromptLabRuns,
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
