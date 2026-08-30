import type { CSSProperties } from "react";
import type { CanonicalSpan } from "../hosted/types";
import { getLabelColor } from "../hosted/types";

interface Props {
  text: string;
  spans: CanonicalSpan[];
  clickable?: boolean;
  onSpanClick?: (index: number, e: React.MouseEvent | React.KeyboardEvent) => void;
  comparisonMode?: boolean;
  comparisonSpans?: CanonicalSpan[];
  startOffset?: number;
  endOffset?: number;
}

interface RenderSegment {
  start: number;
  end: number;
  activeSpanIndices: number[];
  activeComparisonIndices: number[];
}

export default function AnnotatedText({
  text,
  spans,
  clickable = false,
  onSpanClick,
  comparisonMode = false,
  comparisonSpans = [],
  startOffset = 0,
  endOffset,
}: Props) {
  const sorted = spans.toSorted((a, b) => a.start - b.start);
  const sortedComparison = comparisonSpans.toSorted((a, b) => a.start - b.start);
  const codePoints = Array.from(text);
  const rangeEnd = Math.min(endOffset ?? codePoints.length, codePoints.length);
  const segments = buildRenderSegments(
    codePoints.length,
    Math.max(0, startOffset),
    rangeEnd,
    sorted,
    comparisonMode ? sortedComparison : [],
  );

  return (
    <>
      {segments.map((segment) =>
        renderSegment(
          codePoints,
          segment,
          sorted,
          sortedComparison,
          clickable,
          onSpanClick,
          comparisonMode,
        ),
      )}
    </>
  );
}

function buildRenderSegments(
  totalCodePoints: number,
  startOffset: number,
  endOffset: number,
  spans: CanonicalSpan[],
  comparisonSpans: CanonicalSpan[],
): RenderSegment[] {
  const rangeStart = Math.min(startOffset, totalCodePoints);
  const rangeEnd = Math.max(rangeStart, Math.min(endOffset, totalCodePoints));
  const boundaries = new Set<number>([rangeStart, rangeEnd]);
  for (const span of [...spans, ...comparisonSpans]) {
    if (span.end <= rangeStart || span.start >= rangeEnd) continue;
    boundaries.add(Math.max(rangeStart, Math.min(span.start, rangeEnd)));
    boundaries.add(Math.max(rangeStart, Math.min(span.end, rangeEnd)));
  }
  const sortedBoundaries = Array.from(boundaries).sort((a, b) => a - b);
  const segments: RenderSegment[] = [];

  for (let index = 0; index < sortedBoundaries.length - 1; index += 1) {
    const start = sortedBoundaries[index]!;
    const end = sortedBoundaries[index + 1]!;
    if (start >= end) continue;
    segments.push({
      start,
      end,
      activeSpanIndices: spans.flatMap((span, spanIndex) =>
        span.start < end && span.end > start ? [spanIndex] : [],
      ),
      activeComparisonIndices: comparisonSpans.flatMap((span, spanIndex) =>
        span.start < end && span.end > start ? [spanIndex] : [],
      ),
    });
  }

  return segments;
}

function spansMatch(left: CanonicalSpan, right: CanonicalSpan): boolean {
  return left.start === right.start && left.end === right.end && left.label === right.label;
}

function comparisonClass(
  activeSpans: CanonicalSpan[],
  activeComparisonSpans: CanonicalSpan[],
  allSpans: CanonicalSpan[],
  allComparisonSpans: CanonicalSpan[],
): "comparison-match" | "comparison-difference" | null {
  if (activeSpans.length === 0 && activeComparisonSpans.length === 0) return null;
  const ownMatched = activeSpans.every((span) =>
    allComparisonSpans.some((candidate) => spansMatch(span, candidate)),
  );
  const comparisonMatched = activeComparisonSpans.every((span) =>
    allSpans.some((candidate) => spansMatch(span, candidate)),
  );
  return ownMatched && comparisonMatched ? "comparison-match" : "comparison-difference";
}

function renderSegment(
  codePoints: string[],
  segment: RenderSegment,
  sortedSpans: CanonicalSpan[],
  sortedComparison: CanonicalSpan[],
  clickable: boolean,
  onSpanClick: Props["onSpanClick"],
  comparisonMode: boolean,
): React.ReactNode {
  const { start, end, activeSpanIndices, activeComparisonIndices } = segment;
  const segmentText = codePoints.slice(start, end).join("");
  const activeSpans = activeSpanIndices.map((index) => sortedSpans[index]!);
  const activeComparisonSpans = activeComparisonIndices.map(
    (index) => sortedComparison[index]!,
  );
  const comparisonState = comparisonMode
    ? comparisonClass(activeSpans, activeComparisonSpans, sortedSpans, sortedComparison)
    : null;

  if (activeSpanIndices.length === 0 && comparisonState === null) {
    return (
      <span key={`segment-${start}-${end}`} data-offset={start} data-offset-end={end}>
        {segmentText}
      </span>
    );
  }

  const labels = Array.from(new Set(activeSpans.map((span) => span.label)));
  const clickableIndex = pickClickableSpanIndex(activeSpanIndices, sortedSpans);
  const color = getLabelColor(labels[0] ?? "IDENTIFYING_NUMBER");
  const isClickable = clickable && onSpanClick && clickableIndex !== null;
  const className = [
    "ann-span",
    isClickable ? "clickable" : "",
    comparisonState ?? "",
  ].filter(Boolean).join(" ");
  const handleActivate = (event: React.MouseEvent | React.KeyboardEvent) => {
    if (clickableIndex === null || !onSpanClick) return;
    onSpanClick(clickableIndex, event);
  };

  if (isClickable) {
    return (
      <button
        type="button"
        key={`segment-${start}-${end}`}
        className={className}
        style={{ "--ann-color": color } as CSSProperties}
        data-offset={start}
        data-offset-end={end}
        aria-label={`${labels.join(" and ")}: ${segmentText}`}
        onClick={handleActivate}
      >
        {segmentText}
      </button>
    );
  }

  return (
    <span
      key={`segment-${start}-${end}`}
      className={className}
      style={{ "--ann-color": color } as CSSProperties}
      data-offset={start}
      data-offset-end={end}
    >
      {segmentText}
    </span>
  );
}

function pickClickableSpanIndex(indices: number[], spans: CanonicalSpan[]): number | null {
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
