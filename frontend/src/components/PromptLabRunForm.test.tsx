import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import PromptLabRunForm from "./PromptLabRunForm";
import type { DocumentSummary } from "../types";

describe("PromptLabRunForm", () => {
  const documents: DocumentSummary[] = [
    { id: "doc-1", filename: "doc-1.txt", status: "reviewed" },
  ];

  it("uses the provided concurrency max for the input and validation", async () => {
    const onRun = vi.fn().mockResolvedValue(undefined);

    render(
      <PromptLabRunForm
        documents={documents}
        selectedDocumentId="doc-1"
        methods={[]}
        onRun={onRun}
        running={false}
        concurrencyMax={12}
      />,
    );

    await waitFor(() => {
      expect((screen.getByLabelText("doc-1.txt") as HTMLInputElement).checked).toBe(true);
    });

    const concurrencyInput = screen.getByLabelText("Concurrency") as HTMLInputElement;
    expect(concurrencyInput.max).toBe("12");
    expect(concurrencyInput.value).toBe("10");

    fireEvent.change(concurrencyInput, { target: { value: "13" } });
    fireEvent.click(screen.getByRole("button", { name: "Run Prompt Lab" }));

    expect(await screen.findByText("Concurrency must be 1 to 12")).toBeTruthy();
    expect(onRun).not.toHaveBeenCalled();
  });
});
