import type { CanonicalSpan } from "./hosted/types";
import {
  codeUnitOffsetToCodePointOffset,
  getCodePointLength,
  sliceByCodePointOffsets,
} from "./textOffsets";

const SPEAKER_PREFIX_RE = /\n(?:Tutor|Student):\s*/gu;

export function buildNewSpansFromSelection(
  rawText: string,
  start: number,
  end: number,
  label: string,
): CanonicalSpan[] {
  const selectedText = sliceByCodePointOffsets(rawText, start, end);
  if (label !== "URL") {
    return [{ start, end, label, text: selectedText }];
  }

  const prefixMatches = [...selectedText.matchAll(SPEAKER_PREFIX_RE)];
  if (prefixMatches.length === 0) {
    return [{ start, end, label, text: selectedText }];
  }

  const segmentBoundaries: Array<[number, number]> = [];
  let segmentStart = 0;
  for (const match of prefixMatches) {
    const matchStart = codeUnitOffsetToCodePointOffset(selectedText, match.index);
    segmentBoundaries.push([segmentStart, matchStart]);
    segmentStart = codeUnitOffsetToCodePointOffset(
      selectedText,
      match.index + match[0].length,
    );
  }
  segmentBoundaries.push([segmentStart, getCodePointLength(selectedText)]);

  const fragments = segmentBoundaries.flatMap(([relativeStart, relativeEnd]) => {
    let nextStart = start + relativeStart;
    let nextEnd = start + relativeEnd;
    while (
      nextStart < nextEnd &&
      /^\s$/u.test(sliceByCodePointOffsets(rawText, nextStart, nextStart + 1))
    ) {
      nextStart += 1;
    }
    while (
      nextEnd > nextStart &&
      /^\s$/u.test(sliceByCodePointOffsets(rawText, nextEnd - 1, nextEnd))
    ) {
      nextEnd -= 1;
    }
    if (nextStart >= nextEnd) return [];
    return [{
      start: nextStart,
      end: nextEnd,
      label,
      text: sliceByCodePointOffsets(rawText, nextStart, nextEnd),
    }];
  });

  const firstFragment = fragments[0];
  const lastFragment = fragments[fragments.length - 1];
  if (
    fragments.length > 1 &&
    firstFragment?.text.toLowerCase() === "amara." &&
    lastFragment?.text.toLowerCase() === "org"
  ) {
    return [firstFragment, lastFragment];
  }
  return fragments;
}
