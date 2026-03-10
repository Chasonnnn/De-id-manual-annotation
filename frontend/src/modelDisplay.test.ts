import { describe, expect, it } from "vitest";

import { formatExperimentModelLabel, getExperimentModelLabelById } from "./modelDisplay";

describe("modelDisplay", () => {
  it("prefers the model id and appends reasoning effort when present", () => {
    expect(
      formatExperimentModelLabel({
        label: "Model 2",
        model: "openai.gpt-5.2-chat",
        reasoning_effort: "high",
        anthropic_thinking: false,
      }),
    ).toBe("openai.gpt-5.2-chat (high)");
  });

  it("appends anthropic thinking when enabled", () => {
    expect(
      formatExperimentModelLabel({
        label: "Claude",
        model: "anthropic.claude-4.6-sonnet",
        reasoning_effort: "none",
        anthropic_thinking: true,
      }),
    ).toBe("anthropic.claude-4.6-sonnet (thinking)");
  });

  it("falls back to a resolved model by id", () => {
    expect(
      getExperimentModelLabelById(
        [
          {
            id: "m_high",
            label: "Model 4",
            model: "openai.gpt-5.2-chat",
            reasoning_effort: "xhigh",
          },
        ],
        "m_high",
        "Model 4",
      ),
    ).toBe("openai.gpt-5.2-chat (xhigh)");
  });
});
