import { describe, expect, it } from "vitest";

import {
  AGENT_PROMPT_PRESETS,
  AGENT_VIEW_DESCRIPTIONS,
  detectAgentPromptPresetId,
  getAgentPromptPreset,
  getPromptPresetLabelFromSnapshot,
} from "./agentPromptPresets";

describe("agentPromptPresets", () => {
  it("detects built-in prompt presets by prompt text", () => {
    const baseline = getAgentPromptPreset("baseline");
    const annotator = getAgentPromptPreset("annotator_agents");

    expect(detectAgentPromptPresetId(baseline.systemPrompt)).toBe("baseline");
    expect(detectAgentPromptPresetId(annotator.systemPrompt)).toBe("annotator_agents");
  });

  it("falls back to custom for non-preset prompt text", () => {
    expect(detectAgentPromptPresetId("custom prompt")).toBe("custom");
  });

  it("extracts preset labels from saved prompt snapshots", () => {
    const annotator = getAgentPromptPreset("annotator_agents");

    expect(
      getPromptPresetLabelFromSnapshot({
        requested_system_prompt: annotator.systemPrompt,
      }),
    ).toBe(annotator.label);
    expect(getPromptPresetLabelFromSnapshot({ requested_system_prompt: "custom prompt" })).toBe(
      null,
    );
  });

  it("defines all agent view descriptions", () => {
    expect(AGENT_PROMPT_PRESETS.map((item) => item.id)).toEqual([
      "baseline",
      "annotator_agents",
    ]);
    expect(AGENT_VIEW_DESCRIPTIONS.combined).toContain("rule");
    expect(AGENT_VIEW_DESCRIPTIONS.rule).toContain("rule");
    expect(AGENT_VIEW_DESCRIPTIONS.llm).toContain("LLM");
  });
});
