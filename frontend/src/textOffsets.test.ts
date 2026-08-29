import { describe, expect, it } from "vitest";

import {
  codeUnitOffsetToCodePointOffset,
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
});
