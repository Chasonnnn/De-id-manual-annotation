import { cleanup, fireEvent, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import MetricsCompareDashboard from "./MetricsCompareDashboard";
import type { MetricsCandidate, MetricsCompareResult } from "../types";

const candidates: MetricsCandidate[] = [
  {
    id: "manual",
    source: "manual",
    kind: "manual",
    label: "Manual annotations",
    document_count: 2,
    method_bundle: "audited",
  },
  {
    id: "agent.method.method-a::model-a",
    source: "agent.method.method-a::model-a",
    kind: "method_run",
    label: "Method A / model-a",
    document_count: 2,
    method_bundle: "deidentify-v2",
  },
  {
    id: "methods_lab.run-1.cell-1",
    source: "methods_lab.run-1.cell-1",
    kind: "methods_lab_cell",
    label: "Lab run / cell 1",
    document_count: 1,
    method_bundle: "deidentify-v2",
    run_id: "run-1",
    cell_id: "cell-1",
  },
];

const result: MetricsCompareResult = {
  reference: "manual",
  match_mode: "overlap",
  primary_metric: "recall",
  total_documents: 2,
  hypotheses: [
    {
      ...candidates[1]!,
      source: "agent.method.method-a::model-a",
      match_mode: "overlap",
      micro: { precision: 0.9, recall: 0.5, f1: 0.643, tp: 5, fp: 1, fn: 5 },
      avg_document_micro: { precision: 0.9, recall: 0.5, f1: 0.643 },
      avg_document_macro: { precision: 0.9, recall: 0.5, f1: 0.643 },
      per_label: {},
      missed_label_counts: { NAME: 5 },
      exact_micro: { precision: 0.8, recall: 0.4, f1: 0.533, tp: 4, fp: 1, fn: 6 },
      overlap_micro: { precision: 0.9, recall: 0.5, f1: 0.643, tp: 5, fp: 1, fn: 5 },
      exact_overlap_gap_f1: 0.11,
      coverage: {
        total_documents: 2,
        compared_documents: 2,
        skipped_documents: 0,
        skipped: [],
      },
      llm_confidence_summary: {
        documents_with_confidence: 1,
        mean_confidence: 0.8,
        band_counts: { high: 1, medium: 0, low: 0, na: 0 },
      },
      documents: [
        {
          id: "doc-a",
          filename: "doc-a.json",
          reference_count: 10,
          hypothesis_count: 6,
          micro: { precision: 0.9, recall: 0.5, f1: 0.643, tp: 5, fp: 1, fn: 5 },
          macro: { precision: 0.9, recall: 0.5, f1: 0.643 },
          exact_micro: { precision: 0.8, recall: 0.4, f1: 0.533, tp: 4, fp: 1, fn: 6 },
          overlap_micro: { precision: 0.9, recall: 0.5, f1: 0.643, tp: 5, fp: 1, fn: 5 },
          cohens_kappa: 0.7,
          matched_span_mean_iou: 0.8,
        },
      ],
    },
    {
      ...candidates[2]!,
      source: "methods_lab.run-1.cell-1",
      match_mode: "overlap",
      micro: { precision: 0.7, recall: 0.9, f1: 0.788, tp: 9, fp: 4, fn: 1 },
      avg_document_micro: { precision: 0.7, recall: 0.9, f1: 0.788 },
      avg_document_macro: { precision: 0.7, recall: 0.9, f1: 0.788 },
      per_label: {},
      missed_label_counts: { NAME: 1 },
      exact_micro: { precision: 0.5, recall: 0.6, f1: 0.545, tp: 6, fp: 6, fn: 4 },
      overlap_micro: { precision: 0.7, recall: 0.9, f1: 0.788, tp: 9, fp: 4, fn: 1 },
      exact_overlap_gap_f1: 0.243,
      coverage: {
        total_documents: 2,
        compared_documents: 1,
        skipped_documents: 1,
        skipped: [{ doc_id: "doc-b", reason: "candidate_unavailable" }],
      },
      llm_confidence_summary: {
        documents_with_confidence: 1,
        mean_confidence: 0.93,
        band_counts: { high: 1, medium: 0, low: 0, na: 0 },
      },
      documents: [
        {
          id: "doc-b",
          filename: "doc-b.json",
          reference_count: 10,
          hypothesis_count: 13,
          micro: { precision: 0.7, recall: 0.9, f1: 0.788, tp: 9, fp: 4, fn: 1 },
          macro: { precision: 0.7, recall: 0.9, f1: 0.788 },
          exact_micro: { precision: 0.5, recall: 0.6, f1: 0.545, tp: 6, fp: 6, fn: 4 },
          overlap_micro: { precision: 0.7, recall: 0.9, f1: 0.788, tp: 9, fp: 4, fn: 1 },
          cohens_kappa: 0.6,
          matched_span_mean_iou: 0.65,
        },
      ],
    },
  ],
};

describe("MetricsCompareDashboard", () => {
  afterEach(() => {
    cleanup();
  });

  it("ranks candidates by recall and exposes exact/overlap diagnostics plus drilldown", () => {
    const onOpenDocument = vi.fn();

    render(
      <MetricsCompareDashboard
        candidates={candidates}
        reference="manual"
        selectedHypotheses={["agent.method.method-a::model-a", "methods_lab.run-1.cell-1"]}
        matchMode="overlap"
        loading={false}
        result={result}
        onReferenceChange={vi.fn()}
        onHypothesesChange={vi.fn()}
        onMatchModeChange={vi.fn()}
        onRefresh={vi.fn()}
        onExportCsv={vi.fn()}
        onOpenDocument={onOpenDocument}
      />,
    );

    const cards = screen.getAllByRole("button", { name: /TP/ });
    expect(cards[0]?.textContent).toContain("Lab run / cell 1");
    expect(cards[0]?.textContent).toContain("90.0%");
    expect(cards[0]?.textContent).toContain("1 skipped");
    expect(screen.getByText("Exact F1 54.5%")).toBeTruthy();
    expect(screen.getByText("Overlap F1 78.8%")).toBeTruthy();
    expect(screen.getByText("Missed labels NAME:1")).toBeTruthy();

    const docRow = screen.getByRole("row", { name: /doc-b.json/ });
    fireEvent.click(within(docRow).getByRole("button", { name: "Open" }));
    expect(onOpenDocument).toHaveBeenCalledWith("methods_lab.run-1.cell-1", "doc-b");
  });

  it("updates selected hypotheses and delegates CSV export", () => {
    const onHypothesesChange = vi.fn();
    const onExportCsv = vi.fn();

    render(
      <MetricsCompareDashboard
        candidates={candidates}
        reference="manual"
        selectedHypotheses={["agent.method.method-a::model-a"]}
        matchMode="exact"
        loading={false}
        result={result}
        onReferenceChange={vi.fn()}
        onHypothesesChange={onHypothesesChange}
        onMatchModeChange={vi.fn()}
        onRefresh={vi.fn()}
        onExportCsv={onExportCsv}
        onOpenDocument={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByLabelText(/Method A \/ model-a/));
    expect(onHypothesesChange).toHaveBeenCalledWith([]);

    fireEvent.click(screen.getByRole("button", { name: "Export CSV" }));
    expect(onExportCsv).toHaveBeenCalledTimes(1);
  });
});
