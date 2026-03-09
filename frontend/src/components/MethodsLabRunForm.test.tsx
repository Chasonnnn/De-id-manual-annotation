import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import MethodsLabRunForm from "./MethodsLabRunForm";
import type { DocumentSummary } from "../types";

describe("MethodsLabRunForm", () => {
  const documents: DocumentSummary[] = [
    { id: "doc-1", filename: "doc-1.txt", status: "reviewed" },
  ];

  it("uses the provided concurrency max for the input and validation", async () => {
    const onRun = vi.fn().mockResolvedValue(undefined);

    render(
      <MethodsLabRunForm
        documents={documents}
        selectedDocumentId="doc-1"
        methods={[]}
        onRun={onRun}
        running={false}
        concurrencyMax={12}
      />,
    );

    await waitFor(() => {
      expect((screen.getAllByLabelText("doc-1.txt")[0] as HTMLInputElement).checked).toBe(true);
    });

    const concurrencyInput = screen.getByLabelText("Concurrency") as HTMLInputElement;
    expect(concurrencyInput.max).toBe("12");
    expect(concurrencyInput.value).toBe("10");

    fireEvent.change(concurrencyInput, { target: { value: "13" } });
    fireEvent.click(screen.getByRole("button", { name: "Run Methods Lab" }));

    expect(await screen.findByText("Concurrency must be 1 to 12")).toBeTruthy();
    expect(onRun).not.toHaveBeenCalled();
  });

  it("defaults methods lab chunk mode to off", async () => {
    render(
      <MethodsLabRunForm
        documents={documents}
        selectedDocumentId="doc-1"
        methods={[]}
        onRun={vi.fn()}
        running={false}
        concurrencyMax={12}
      />,
    );

    await waitFor(() => {
      expect((screen.getAllByLabelText("doc-1.txt")[0] as HTMLInputElement).checked).toBe(true);
    });

    expect((screen.getByLabelText("Chunk Mode") as HTMLSelectElement).value).toBe("off");
  });
});
