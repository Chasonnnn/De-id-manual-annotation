import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import PromptLabMatrix from "./PromptLabMatrix";
import type { PromptLabRunDetail } from "../types";

describe("PromptLabMatrix", () => {
  afterEach(() => {
    cleanup();
  });

  it("defaults to overlap metrics when available and keeps exact as a diagnostic", () => {
    const run = {
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
      folder_ids: [],
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
      diagnostics: {
        requested_concurrency: 16,
        effective_worker_count: 1,
        max_allowed_concurrency: 16,
        total_tasks: 1,
        clamped_by_task_count: true,
        clamped_by_server_cap: false,
        api_base_host: "api.ai.it.cornell.edu",
      },
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
    } as PromptLabRunDetail;

    render(
      <PromptLabMatrix run={run} selectedCellId={null} onSelectCell={vi.fn()} />,
    );

    expect(screen.getByText("Overlap Overall F1")).not.toBeNull();
    expect(screen.getByText("90.0%")).not.toBeNull();
    expect(screen.getByText("Exact F1 40.0% · Exact R 30.0%")).not.toBeNull();
  });

  it("renders the diagnostics strip and clamp explanation for a clamped run", () => {
    const run = {
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
      folder_ids: [],
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
      diagnostics: {
        requested_concurrency: 16,
        effective_worker_count: 1,
        max_allowed_concurrency: 16,
        total_tasks: 1,
        clamped_by_task_count: true,
        clamped_by_server_cap: false,
        api_base_host: "api.ai.it.cornell.edu",
      },
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
    } as PromptLabRunDetail;

    const view = render(
      <PromptLabMatrix
        run={run}
        experimentDiagnostics={{
          resolved_api_base: "https://api.ai.it.cornell.edu",
          api_base_host: "api.ai.it.cornell.edu",
          prompt_lab_max_concurrency: 16,
          methods_lab_max_concurrency: 16,
          gateway_catalog: {
            reachable: true,
            model_count: 189,
            error: null,
            checked_at: "2026-03-10T00:00:00Z",
          },
        }}
        selectedCellId={null}
        onSelectCell={vi.fn()}
      />,
    );

    const matrix = within(view.container);
    expect(matrix.getByText("Requested 16")).not.toBeNull();
    expect(matrix.getByText("Effective 1")).not.toBeNull();
    expect(matrix.getByText("Tasks 1")).not.toBeNull();
    expect(matrix.getByText("Cap 16")).not.toBeNull();
    expect(matrix.getByText("Gateway api.ai.it.cornell.edu")).not.toBeNull();
    expect(matrix.getByText("Catalog status reachable · 189 models")).not.toBeNull();
    expect(
      matrix.getByText("Only 1 task exists for this run, so the backend started 1 worker."),
    ).not.toBeNull();
  });
});
