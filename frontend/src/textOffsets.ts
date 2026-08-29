export function getCodePointLength(text: string): number {
  return Array.from(text).length;
}

export function codeUnitOffsetToCodePointOffset(
  text: string,
  codeUnitOffset: number,
): number {
  const clampedOffset = Math.max(0, Math.min(codeUnitOffset, text.length));
  return Array.from(text.slice(0, clampedOffset)).length;
}

export function sliceByCodePointOffsets(
  text: string,
  start: number,
  end: number,
): string {
  if (start >= end) return "";
  return Array.from(text).slice(Math.max(0, start), Math.max(0, end)).join("");
}
