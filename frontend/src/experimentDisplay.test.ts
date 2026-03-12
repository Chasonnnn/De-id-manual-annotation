import { describe, expect, it } from "vitest";

import { formatMethodBundleLabel } from "./experimentDisplay";

describe("formatMethodBundleLabel", () => {
  it("renders the v2+post-process bundle label", () => {
    expect(formatMethodBundleLabel("v2+post-process")).toBe("V2 + post-process methods");
  });
});
