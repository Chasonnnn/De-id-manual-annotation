import { describe, expect, it } from "vitest";

import { MODEL_PRESETS } from "./modelPresets";

describe("MODEL_PRESETS", () => {
  it("includes Claude Opus 4.6 instead of GPT-5.4", () => {
    const ids = MODEL_PRESETS.map((preset) => preset.id);
    expect(ids).toContain("anthropic.claude-4.6-opus");
    expect(ids).not.toContain("gpt-5.4");
  });
});
