import { afterEach, describe, expect, it, vi } from "vitest";

import { listMethodsLabRuns, listPromptLabRuns } from "./client";

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
