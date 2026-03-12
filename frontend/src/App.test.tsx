import { cleanup, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";

vi.mock("./api/client", () => ({
  createFolder: vi.fn(),
  createFolderSample: vi.fn(),
  deleteFolder: vi.fn(),
  deleteDocument: vi.fn(),
  exportGroundTruth: vi.fn(),
  exportSession: vi.fn(),
  getFolder: vi.fn(),
  getAgentProgress: vi.fn(),
  getAgentMethods: vi.fn().mockResolvedValue([]),
  getDocument: vi.fn(),
  getMetricsDashboard: vi.fn(),
  getMetrics: vi.fn(),
  ingestSessionFile: vi.fn(),
  listFolders: vi.fn().mockResolvedValue([]),
  listDocuments: vi.fn().mockResolvedValue([]),
  pruneEmptyFolderDocs: vi.fn(),
  runAgent: vi.fn(),
  updateManualAnnotations: vi.fn(),
}));

describe("App", () => {
  const consoleError = vi.spyOn(console, "error").mockImplementation(() => {});

  beforeEach(() => {
    consoleError.mockClear();
  });

  afterEach(() => {
    cleanup();
  });

  it("renders the workspace without hitting the error boundary", async () => {
    render(<App />);

    await waitFor(() => {
      expect(screen.getByText("Workspace")).toBeTruthy();
    });

    expect(screen.queryByText("Something went wrong")).toBeNull();
    expect(screen.getByText("Select a document or upload a file to begin")).toBeTruthy();
  });
});
