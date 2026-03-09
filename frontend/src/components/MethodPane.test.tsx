import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

import MethodPane from "./MethodPane";
import { getAgentCredentialStatus } from "../api/client";
import type { AgentMethodOption } from "../types";

vi.mock("../api/client", () => ({
  getAgentCredentialStatus: vi.fn(),
}));

const methodOptions: AgentMethodOption[] = [
  {
    id: "dual-split",
    label: "Dual Split",
    description: "Two-pass method",
    requires_presidio: false,
    uses_llm: true,
    supports_verify_override: true,
    default_verify: false,
    prompt_templates: [],
    available: true,
    unavailable_reason: null,
  },
];

describe("MethodPane", () => {
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

  it("keeps saved credential overrides collapsed until explicitly edited", async () => {
    sessionStorage.setItem("agent_api_key", "saved-key");
    sessionStorage.setItem("agent_api_base", "https://proxy.example.com/v1");
    const onRunMethod = vi.fn().mockResolvedValue(undefined);

    render(
      <MethodPane
        text="Example transcript"
        spans={[]}
        methods={methodOptions}
        activeMethod="dual-split"
        onActiveMethodChange={vi.fn()}
        onRunMethod={onRunMethod}
        running={false}
        onScroll={vi.fn()}
      />,
    );

    await waitFor(() => {
      expect(getAgentCredentialStatus).toHaveBeenCalled();
    });

    fireEvent.click(screen.getByRole("button", { name: "Show Config" }));

    expect(screen.getAllByText("Local override saved in this browser session.")).toHaveLength(2);
    expect(screen.queryByPlaceholderText("LiteLLM gateway key or provider key")).toBeNull();
    expect(screen.queryByPlaceholderText("https://your-litellm-gateway/v1")).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Run Method" }));

    await waitFor(() => {
      expect(onRunMethod).toHaveBeenCalledWith(
        expect.objectContaining({
          api_key: "saved-key",
          api_base: "https://proxy.example.com/v1",
        }),
      );
    });

    fireEvent.click(screen.getByRole("button", { name: "Edit key" }));
    expect(screen.getByPlaceholderText("LiteLLM gateway key or provider key")).toBeTruthy();
  });

  it("defaults method chunk mode to off", async () => {
    render(
      <MethodPane
        text="Example transcript"
        spans={[]}
        methods={methodOptions}
        activeMethod="dual-split"
        onActiveMethodChange={vi.fn()}
        onRunMethod={vi.fn()}
        running={false}
        onScroll={vi.fn()}
      />,
    );

    await waitFor(() => {
      expect(getAgentCredentialStatus).toHaveBeenCalled();
    });

    fireEvent.click(screen.getByRole("button", { name: "Show Config" }));

    expect((screen.getByLabelText("Chunk Mode") as HTMLSelectElement).value).toBe("off");
  });
});
