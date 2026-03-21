import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

import AgentPane from "./AgentPane";
import { getAgentCredentialStatus } from "../api/client";

vi.mock("../api/client", () => ({
  getAgentCredentialStatus: vi.fn(),
}));

describe("AgentPane", () => {
  beforeEach(() => {
    sessionStorage.clear();
    vi.mocked(getAgentCredentialStatus).mockResolvedValue({
      has_api_key: false,
      api_key_sources: [],
      has_api_base: false,
      api_base_sources: [],
    });
  });

  afterEach(() => {
    cleanup();
  });

  it("uses saved credential overrides without auto-expanding their editors", async () => {
    sessionStorage.setItem("agent_api_key", "saved-key");
    sessionStorage.setItem("agent_api_base", "https://proxy.example.com/v1");
    const onRunAgent = vi.fn().mockResolvedValue(undefined);

    render(
      <AgentPane
        text="Example transcript"
        spans={[]}
        activeOutput="combined"
        onActiveOutputChange={vi.fn()}
        llmRunOptions={[]}
        activeLlmRunKey="__latest__"
        onActiveLlmRunKeyChange={vi.fn()}
        onRunAgent={onRunAgent}
        running={false}
        onScroll={vi.fn()}
      />,
    );

    await waitFor(() => {
      expect(getAgentCredentialStatus).toHaveBeenCalled();
    });

    fireEvent.click(screen.getByRole("button", { name: "Show Config" }));
    fireEvent.change(screen.getByLabelText("Mode"), { target: { value: "llm" } });

    expect(screen.getAllByText("Local override saved in this browser session.")).toHaveLength(2);
    expect(screen.queryByPlaceholderText("LiteLLM gateway key or provider key")).toBeNull();
    expect(screen.queryByPlaceholderText("https://your-litellm-gateway/v1")).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Run Agent" }));

    await waitFor(() => {
      expect(onRunAgent).toHaveBeenCalledWith(
        expect.objectContaining({
          api_key: "saved-key",
          api_base: "https://proxy.example.com/v1",
        }),
      );
    });

    fireEvent.click(screen.getByRole("button", { name: "Edit base URL" }));
    expect(screen.getByPlaceholderText("https://your-litellm-gateway/v1")).toBeTruthy();
  });

  it("defaults llm chunk mode to off", async () => {
    render(
      <AgentPane
        text="Example transcript"
        spans={[]}
        activeOutput="combined"
        onActiveOutputChange={vi.fn()}
        llmRunOptions={[]}
        activeLlmRunKey="__latest__"
        onActiveLlmRunKeyChange={vi.fn()}
        onRunAgent={vi.fn()}
        running={false}
        onScroll={vi.fn()}
      />,
    );

    await waitFor(() => {
      expect(getAgentCredentialStatus).toHaveBeenCalled();
    });

    fireEvent.click(screen.getByRole("button", { name: "Show Config" }));
    fireEvent.change(screen.getByLabelText("Mode"), { target: { value: "llm" } });

    expect((screen.getByLabelText("Chunk Mode") as HTMLSelectElement).value).toBe("off");
    expect(screen.queryByLabelText("Label Profile")).toBeNull();
  });
});
