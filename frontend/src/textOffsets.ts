export interface CodePointOffsetTable {
  totalCodePoints: number;
  toCodeUnit(offset: number): number;
}

export function getCodePointLength(text: string): number {
  let count = 0;
  for (const _char of text) {
    count += 1;
  }
  return count;
}

export function codeUnitOffsetToCodePointOffset(
  text: string,
  codeUnitOffset: number,
): number {
  const clampedOffset = Math.max(0, Math.min(codeUnitOffset, text.length));
  let codePointOffset = 0;
  let cursor = 0;

  while (cursor < clampedOffset) {
    const codePoint = text.codePointAt(cursor);
    if (codePoint == null) break;
    cursor += codePoint > 0xffff ? 2 : 1;
    codePointOffset += 1;
  }

  return codePointOffset;
}

export function buildCodePointOffsetTable(
  text: string,
  requestedOffsets: number[],
): CodePointOffsetTable {
  const normalizedOffsets = Array.from(
    new Set(requestedOffsets.map((offset) => Math.max(0, offset))),
  ).sort((a, b) => a - b);
  const codeUnitByCodePoint = new Map<number, number>();

  let requestedIndex = 0;
  let codePointOffset = 0;
  let codeUnitOffset = 0;

  while (
    requestedIndex < normalizedOffsets.length &&
    normalizedOffsets[requestedIndex] === 0
  ) {
    codeUnitByCodePoint.set(0, 0);
    requestedIndex += 1;
  }

  while (codeUnitOffset < text.length) {
    while (
      requestedIndex < normalizedOffsets.length &&
      normalizedOffsets[requestedIndex] === codePointOffset
    ) {
      codeUnitByCodePoint.set(codePointOffset, codeUnitOffset);
      requestedIndex += 1;
    }

    const codePoint = text.codePointAt(codeUnitOffset);
    if (codePoint == null) break;
    codeUnitOffset += codePoint > 0xffff ? 2 : 1;
    codePointOffset += 1;
  }

  while (
    requestedIndex < normalizedOffsets.length &&
    (normalizedOffsets[requestedIndex] ?? Number.POSITIVE_INFINITY) <= codePointOffset
  ) {
    const requestedOffset = normalizedOffsets[requestedIndex];
    if (requestedOffset == null) {
      break;
    }
    codeUnitByCodePoint.set(requestedOffset, codeUnitOffset);
    requestedIndex += 1;
  }

  const totalCodePoints = codePointOffset;
  const totalCodeUnits = codeUnitOffset;

  return {
    totalCodePoints,
    toCodeUnit(offset: number) {
      const clampedOffset = Math.max(0, Math.min(offset, totalCodePoints));
      const cached = codeUnitByCodePoint.get(clampedOffset);
      if (cached != null) {
        return cached;
      }

      let cursorCodePoint = 0;
      let cursorCodeUnit = 0;
      while (cursorCodeUnit < text.length && cursorCodePoint < clampedOffset) {
        const codePoint = text.codePointAt(cursorCodeUnit);
        if (codePoint == null) break;
        cursorCodeUnit += codePoint > 0xffff ? 2 : 1;
        cursorCodePoint += 1;
      }
      return cursorCodePoint >= clampedOffset ? cursorCodeUnit : totalCodeUnits;
    },
  };
}

export function sliceByCodePointOffsets(
  text: string,
  start: number,
  end: number,
  table?: CodePointOffsetTable,
): string {
  if (start >= end) return "";
  const offsetTable = table ?? buildCodePointOffsetTable(text, [start, end]);
  return text.slice(
    offsetTable.toCodeUnit(start),
    offsetTable.toCodeUnit(end),
  );
}
