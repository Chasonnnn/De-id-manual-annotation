import { describe, expect, it } from "vitest";

import { formatMethodBundleLabel } from "./experimentDisplay";

describe("formatMethodBundleLabel", () => {
  it("renders the v2+post-process bundle label", () => {
    expect(formatMethodBundleLabel("v2+post-process")).toBe("V2 + post-process methods");
  });

  it("renders historical bundle labels truthfully", () => {
    expect(formatMethodBundleLabel("v2")).toBe("V2 methods");
    expect(formatMethodBundleLabel("stable")).toBe("Stable methods");
    expect(formatMethodBundleLabel("deidentify-v2")).toBe("Colleague demo V2 methods");
  });
});
