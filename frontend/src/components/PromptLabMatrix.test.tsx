import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import PromptLabMatrix from "./PromptLabMatrix";
import type { PromptLabRunDetail } from "../types";

describe("PromptLabMatrix", () => {
  it("defaults to overlap metrics when available and keeps exact as a diagnostic", () => {
    const run: PromptLabRunDetail = {
      id: "run_1",
      name: "prompt_run",
      status: "completed",
      cancellable: false,
      created_at: "2026-03-09T00:00:00Z",
      started_at: "2026-03-09T00:00:00Z",
      finished_at: "2026-03-09T00:01:00Z",
      doc_count: 1,
      prompt_count: 1,
      model_count: 1,
      total_tasks: 1,
      completed_tasks: 1,
      failed_tasks: 0,
      doc_ids: ["doc-1"],
      prompts: [{ id: "baseline_raw", label: "baseline_raw", variant_type: "prompt", system_prompt: "test" }],
      models: [
        {
          id: "gemini_pro",
          label: "gemini_pro",
          model: "google.gemini-3.1-pro-preview",
          reasoning_effort: "none",
          anthropic_thinking: false,
          anthropic_thinking_budget_tokens: null,
        },
      ],
      runtime: {
        temperature: 0,
        match_mode: "exact",
        reference_source: "manual",
        fallback_reference_source: "pre",
        label_profile: "simple",
        label_projection: "native",
        chunk_mode: "auto",
        chunk_size_chars: 10000,
      },
      concurrency: 1,
      warnings: [],
      errors: [],
      matrix: {
        models: [{ id: "gemini_pro", label: "gemini_pro" }],
        prompts: [{ id: "baseline_raw", label: "baseline_raw" }],
        available_labels: [],
        cells: [
          {
            id: "gemini_pro__baseline_raw",
            model_id: "gemini_pro",
            model_label: "gemini_pro",
            prompt_id: "baseline_raw",
            prompt_label: "baseline_raw",
            status: "completed",
            total_docs: 1,
            completed_docs: 1,
            failed_docs: 0,
            error_count: 0,
            micro: { precision: 0.5, recall: 0.3, f1: 0.4, tp: 2, fp: 2, fn: 4 },
            per_label: {},
            co_primary_metrics: {
              overlap: {
                micro: { precision: 1, recall: 0.8, f1: 0.9, tp: 6, fp: 0, fn: 1 },
                macro: { precision: 1, recall: 0.8, f1: 0.9 },
                per_label: {},
              },
            },
            mean_confidence: null,
          },
        ],
      },
      progress: {
        total_tasks: 1,
        completed_tasks: 1,
        failed_tasks: 0,
      },
    };

    render(
      <PromptLabMatrix run={run} selectedCellId={null} onSelectCell={vi.fn()} />,
    );

    expect(screen.getByText("Overlap Overall F1")).not.toBeNull();
    expect(screen.getByText("90.0%")).not.toBeNull();
    expect(screen.getByText("Exact F1 40.0% · Exact R 30.0%")).not.toBeNull();
  });
});
