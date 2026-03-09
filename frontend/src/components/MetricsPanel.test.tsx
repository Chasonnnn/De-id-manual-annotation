import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import MetricsPanel from "./MetricsPanel";
import type { MetricsResult } from "../types";

describe("MetricsPanel", () => {
  it("uses the overlap exact companion metrics as the primary display when exact mode is selected", () => {
    const metrics: MetricsResult = {
      micro: { precision: 0.5, recall: 0.3, f1: 0.4 },
      macro: { precision: 0.45, recall: 0.35, f1: 0.39 },
      per_label: {},
      co_primary_metrics: {
        overlap: {
          micro: { precision: 1, recall: 0.8, f1: 0.9 },
          macro: { precision: 1, recall: 0.8, f1: 0.9 },
          per_label: {},
        },
      },
      false_positives: [],
      false_negatives: [],
    };

    render(
      <MetricsPanel
        reference="manual"
        hypothesis="agent.llm"
        matchMode="exact"
        sourceOptions={[
          { value: "manual", label: "Manual" },
          { value: "agent.llm", label: "LLM" },
        ]}
        metrics={metrics}
        loading={false}
        onRefresh={vi.fn()}
        onReferenceChange={vi.fn()}
        onHypothesisChange={vi.fn()}
        onMatchModeChange={vi.fn()}
      />,
    );

    expect(screen.getByText("Overlap Micro F1")).not.toBeNull();
    expect(screen.getAllByText("90.0%").length).toBeGreaterThan(0);
    expect(screen.getByText("Exact diagnostic: P 50.0% · R 30.0% · F1 40.0%")).not.toBeNull();
  });
});
