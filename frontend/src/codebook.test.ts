import { describe, expect, it } from "vitest";
import { CODEBOOK } from "./codebook";
import { ENTITY_TYPES } from "./hosted/types";

describe("annotation codebook", () => {
  it("covers every canonical entity type once with two examples", () => {
    expect(CODEBOOK.map((entry) => entry.label)).toEqual([...ENTITY_TYPES]);
    for (const entry of CODEBOOK) {
      expect(entry.definition.trim()).not.toBe("");
      expect(entry.examples).toHaveLength(2);
      expect(entry.examples.every((example) => example.trim() !== "")).toBe(true);
    }
  });
});
