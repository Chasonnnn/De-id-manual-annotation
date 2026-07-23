import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";

import MethodPane from "./MethodPane";
import { getAgentCredentialStatus } from "../api/client";
import type { AgentMethodOption } from "../types";

vi.mock("../api/client", () => ({
  getAgentCredentialStatus: vi.fn(),
}));

const methodOptions: AgentMethodOption[] = [
  {
    id: "dual",
    label: "Dual",
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

const localMethodOptions: AgentMethodOption[] = [
  {
    id: "deid_pipeline_cascade_gemma31b",
    label: "Operational union + Gemma 31B reviewer",
    description: "Local cascade method",
    requires_presidio: false,
    uses_llm: false,
    supports_verify_override: false,
    default_verify: false,
    prompt_templates: [],
    available: true,
    unavailable_reason: null,
  },
];
const storedRetiredMethod: AgentMethodOption = {
  id: "deid_pipeline_cascade_gemma12b",
  label: "Stored Gemma 12B",
  description: "Saved method run output",
  requires_presidio: false,
  uses_llm: false,
  supports_verify_override: false,
  default_verify: false,
  prompt_templates: [],
  available: true,
  unavailable_reason: null,
};

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
        offeredMethods={methodOptions}
        activeMethod="dual"
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
        offeredMethods={methodOptions}
        activeMethod="dual"
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
    expect(screen.queryByLabelText("Label Profile")).toBeNull();
  });

  it("explains when a method runs locally without LLM setup", async () => {
    render(
      <MethodPane
        text="Example transcript"
        spans={[]}
        methods={localMethodOptions}
        offeredMethods={localMethodOptions}
        activeMethod="deid_pipeline_cascade_gemma31b"
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

    expect(
      screen.getByText(
        /This method runs locally without an LLM\. Model, API key, base URL, reasoning, temperature, and chunk settings are not used\./i,
      ),
    ).toBeTruthy();
    expect(screen.queryByLabelText("Model")).toBeNull();
    expect(screen.queryByLabelText("API Key")).toBeNull();
  });

  it("keeps stored retired methods viewable but excludes them from new runs", async () => {
    const onRunMethod = vi.fn().mockResolvedValue(undefined);
    render(
      <MethodPane
        text="Example transcript"
        spans={[]}
        methods={[...methodOptions, storedRetiredMethod]}
        offeredMethods={methodOptions}
        activeMethod="deid_pipeline_cascade_gemma12b"
        onActiveMethodChange={vi.fn()}
        onRunMethod={onRunMethod}
        running={false}
        onScroll={vi.fn()}
      />,
    );

    const viewSelect = screen.getByLabelText("View:") as HTMLSelectElement;
    expect(within(viewSelect).getByRole("option", { name: "Stored Gemma 12B" })).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: "Show Config" }));
    const runSelect = screen.getByLabelText("Method") as HTMLSelectElement;
    expect(within(runSelect).getAllByRole("option")).toHaveLength(1);
    expect(within(runSelect).getByRole("option", { name: "Dual" })).toBeTruthy();
    expect(within(runSelect).queryByRole("option", { name: "Stored Gemma 12B" })).toBeNull();

    fireEvent.click(screen.getByRole("button", { name: "Run Method" }));
    await waitFor(() => {
      expect(onRunMethod).toHaveBeenCalledWith(
        expect.objectContaining({
          method_id: "dual",
        }),
      );
    });
  });
});
