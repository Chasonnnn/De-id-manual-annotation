import type { CSSProperties } from "react";
import type { CanonicalSpan } from "../types";
import { getLabelColor } from "../types";
import {
  buildCodePointOffsetTable,
  type CodePointOffsetTable,
  sliceByCodePointOffsets,
} from "../textOffsets";

export interface DiffSpanInfo {
  start: number;
  end: number;
  type: "added" | "removed";
}

interface Props {
  text: string;
  spans: CanonicalSpan[];
  clickable?: boolean;
  onSpanClick?: (index: number, e: React.MouseEvent) => void;
  diffSpans?: DiffSpanInfo[];
}

interface RenderSegment {
  start: number;
  end: number;
  activeSpanIndices: number[];
  diffClass: string;
}

export default function AnnotatedText({
  text,
  spans,
  clickable = false,
  onSpanClick,
  diffSpans = [],
}: Props) {
  const sorted = [...spans].sort((a, b) => a.start - b.start);
  const offsetTable = buildCodePointOffsetTable(text, [
    0,
    ...sorted.flatMap((span) => [span.start, span.end]),
    ...diffSpans.flatMap((span) => [span.start, span.end]),
  ]);
  const segments = buildRenderSegments(
    offsetTable.totalCodePoints,
    sorted,
    diffSpans,
  );

  return (
    <>
      {segments.map((segment) =>
        renderSegment(
          text,
          segment,
          sorted,
          clickable,
          onSpanClick,
          offsetTable,
        ),
      )}
    </>
  );
}

function buildRenderSegments(
  totalCodePoints: number,
  spans: CanonicalSpan[],
  diffSpans: DiffSpanInfo[],
): RenderSegment[] {
  const boundaries = new Set<number>([0, totalCodePoints]);
  for (const span of spans) {
    boundaries.add(Math.max(0, Math.min(span.start, totalCodePoints)));
    boundaries.add(Math.max(0, Math.min(span.end, totalCodePoints)));
  }
  for (const diffSpan of diffSpans) {
    boundaries.add(Math.max(0, Math.min(diffSpan.start, totalCodePoints)));
    boundaries.add(Math.max(0, Math.min(diffSpan.end, totalCodePoints)));
  }

  const sortedBoundaries = Array.from(boundaries).sort((a, b) => a - b);
  const segments: RenderSegment[] = [];

  for (let i = 0; i < sortedBoundaries.length - 1; i += 1) {
    const start = sortedBoundaries[i]!;
    const end = sortedBoundaries[i + 1]!;
    if (start >= end) continue;

    const activeSpanIndices = spans.flatMap((span, index) =>
      span.start < end && span.end > start ? [index] : [],
    );
    segments.push({
      start,
      end,
      activeSpanIndices,
      diffClass: getDiffClass(start, end, diffSpans),
    });
  }

  return segments;
}

function renderSegment(
  text: string,
  segment: RenderSegment,
  sortedSpans: CanonicalSpan[],
  clickable: boolean,
  onSpanClick: Props["onSpanClick"],
  offsetTable: CodePointOffsetTable,
): React.ReactNode {
  const { start, end, activeSpanIndices, diffClass } = segment;
  const segmentText = sliceByCodePointOffsets(text, start, end, offsetTable);

  if (activeSpanIndices.length === 0) {
    return (
      <span
        key={`segment-${start}-${end}`}
        className={diffClass || undefined}
        data-offset={start}
        data-offset-end={end}
      >
        {segmentText}
      </span>
    );
  }

  const labels = Array.from(
    new Set(activeSpanIndices.map((index) => sortedSpans[index]!.label)),
  );
  const clickableIndex = pickClickableSpanIndex(activeSpanIndices, sortedSpans);
  const color = getLabelColor(labels[0] ?? "MISC_ID");

  return (
    <span
      key={`segment-${start}-${end}`}
      className={`ann-span ${clickable && clickableIndex !== null ? "clickable" : ""} ${diffClass}`}
      style={{ "--ann-color": color } as CSSProperties}
      data-offset={start}
      data-offset-end={end}
      onClick={
        clickable && onSpanClick && clickableIndex !== null
          ? (e) => onSpanClick(clickableIndex, e)
          : undefined
      }
    >
      {segmentText}
      <span
        className="ann-span-label"
        data-annotation-label="true"
        aria-hidden="true"
      >
        {labels.join(" · ")}
      </span>
    </span>
  );
}

function pickClickableSpanIndex(
  indices: number[],
  spans: CanonicalSpan[],
): number | null {
  if (indices.length === 0) return null;
  return indices.reduce((best, current) => {
    const bestSpan = spans[best]!;
    const currentSpan = spans[current]!;
    const bestLength = bestSpan.end - bestSpan.start;
    const currentLength = currentSpan.end - currentSpan.start;
    if (currentLength < bestLength) return current;
    if (currentLength > bestLength) return best;
    if (currentSpan.start > bestSpan.start) return current;
    return best;
  });
}

function getDiffClass(
  start: number,
  end: number,
  diffSpans: DiffSpanInfo[],
): string {
  for (const d of diffSpans) {
    if (start < d.end && end > d.start) {
      return d.type === "added" ? "diff-added" : "diff-removed";
    }
  }
  return "";
}
