import { describe, expect, it } from "vitest";

import {
  buildCodePointOffsetTable,
  codeUnitOffsetToCodePointOffset,
  getCodePointLength,
  sliceByCodePointOffsets,
} from "./textOffsets";

describe("textOffsets", () => {
  it("slices by backend code-point offsets even after emoji", () => {
    const text = "prefix 😂\nstudent: Greenville\nvolunteer: hi";
    const start = Array.from("prefix 😂\nstudent: ").length;
    const end = start + Array.from("Greenville").length;

    expect(text.slice(start, end)).not.toBe("Greenville");
    expect(sliceByCodePointOffsets(text, start, end)).toBe("Greenville");
  });

  it("converts DOM UTF-16 offsets back to code-point offsets", () => {
    const text = "A😂BC";
    expect(codeUnitOffsetToCodePointOffset(text, 0)).toBe(0);
    expect(codeUnitOffsetToCodePointOffset(text, 1)).toBe(1);
    expect(codeUnitOffsetToCodePointOffset(text, 3)).toBe(2);
    expect(codeUnitOffsetToCodePointOffset(text, 4)).toBe(3);
  });

  it("builds a reusable code-point to code-unit lookup", () => {
    const text = "A😂BC";
    const table = buildCodePointOffsetTable(text, [0, 1, 2, 4]);

    expect(getCodePointLength(text)).toBe(4);
    expect(table.totalCodePoints).toBe(4);
    expect(table.toCodeUnit(0)).toBe(0);
    expect(table.toCodeUnit(1)).toBe(1);
    expect(table.toCodeUnit(2)).toBe(3);
    expect(table.toCodeUnit(4)).toBe(5);
  });
});
