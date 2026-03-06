import { forwardRef, useCallback, useEffect, useRef, useState } from "react";
import type { CanonicalSpan } from "../types";
import AnnotatedText from "./AnnotatedText";
import AnnotationPopup from "./AnnotationPopup";

interface Props {
  text: string;
  spans: CanonicalSpan[];
  labels: string[];
  diffSpans?: { start: number; end: number; type: "added" | "removed" }[];
  onSpansChange: (spans: CanonicalSpan[]) => void;
  onScroll: (scrollTop: number) => void;
}

interface PopupState {
  x: number;
  y: number;
  selStart: number;
  selEnd: number;
  selText: string;
  editIndex: number | null;
}

const BOUNDARY_IGNORABLE_RE = /[\p{P}\s]/u;

/**
 * Walk up from a DOM node to find the nearest ancestor (or self) that has
 * a `data-offset` attribute. Returns the numeric offset value, or null.
 */
function findDataOffset(node: Node): number | null {
  let current: Node | null = node;
  while (current) {
    if (current instanceof HTMLElement && current.dataset.offset != null) {
      return parseInt(current.dataset.offset, 10);
    }
    current = current.parentNode;
  }
  return null;
}

/**
 * Calculate the precise character offset within raw_text for a given
 * DOM position (node + offset). Uses data-offset attributes set by
 * AnnotatedText on every text-containing element.
 */
function resolveCharOffset(node: Node, domOffset: number): number | null {
  // If the node is a text node, find the parent element with data-offset
  // and add the DOM offset within that text node.
  if (node.nodeType === Node.TEXT_NODE) {
    const parent = node.parentElement;
    if (!parent) return null;

    // Find the element with data-offset (might be the parent itself or an ancestor)
    const baseOffset = findDataOffset(parent);
    if (baseOffset === null) return null;

    // If the text node is the only child of its parent, DOM offset is directly
    // the position within the element.
    // If there are multiple children, we need to count text content before this node.
    let charsBefore = 0;
    for (const child of parent.childNodes) {
      if (child === node) break;
      // Skip annotation label overlays -- they don't correspond to raw text
      if (
        child instanceof HTMLElement &&
        (child.tagName === "SUP" ||
          child.dataset.annotationLabel === "true" ||
          child.classList.contains("ann-span-label"))
      ) {
        continue;
      }
      charsBefore += child.textContent?.length ?? 0;
    }

    return baseOffset + charsBefore + domOffset;
  }

  // If the node is an element node, the offset is the child index
  if (node.nodeType === Node.ELEMENT_NODE) {
    const el = node as HTMLElement;
    const baseOffset = findDataOffset(el);
    if (baseOffset !== null) return baseOffset;
    // If the element itself doesn't have data-offset, try the child at domOffset
    const child = el.childNodes[domOffset];
    if (child) {
      const childOffset = findDataOffset(child);
      if (childOffset !== null) return childOffset;
    }
  }

  return null;
}

function trimBoundarySelection(
  rawText: string,
  start: number,
  end: number,
): { start: number; end: number; text: string } {
  let nextStart = start;
  let nextEnd = end;
  while (nextStart < nextEnd && BOUNDARY_IGNORABLE_RE.test(rawText[nextStart] ?? "")) {
    nextStart += 1;
  }
  while (nextEnd > nextStart && BOUNDARY_IGNORABLE_RE.test(rawText[nextEnd - 1] ?? "")) {
    nextEnd -= 1;
  }
  if (nextStart >= nextEnd) {
    return { start, end, text: rawText.slice(start, end) };
  }
  return {
    start: nextStart,
    end: nextEnd,
    text: rawText.slice(nextStart, nextEnd),
  };
}

const ManualAnnotationPane = forwardRef<HTMLDivElement, Props>(
  ({ text, spans, labels, diffSpans = [], onSpansChange, onScroll }, ref) => {
    const [popup, setPopup] = useState<PopupState | null>(null);
    const [trimBoundaries, setTrimBoundaries] = useState(() => {
      try {
        const saved = sessionStorage.getItem("manual_trim_boundaries");
        return saved == null ? true : saved === "true";
      } catch {
        return true;
      }
    });
    // Internal ref for DOM access -- merged with forwarded ref
    const localRef = useRef<HTMLDivElement>(null);

    useEffect(() => {
      try {
        sessionStorage.setItem("manual_trim_boundaries", String(trimBoundaries));
      } catch {
        // Best-effort persistence only.
      }
    }, [trimBoundaries]);

    // Merge callback: sets both our local ref and the forwarded ref
    const mergedRefCallback = useCallback(
      (el: HTMLDivElement | null) => {
        localRef.current = el;
        if (typeof ref === "function") {
          ref(el);
        } else if (ref) {
          (ref as React.MutableRefObject<HTMLDivElement | null>).current = el;
        }
      },
      [ref],
    );

    const handleMouseUp = useCallback(() => {
      const sel = window.getSelection();
      if (!sel || sel.isCollapsed || !sel.rangeCount) return;

      const range = sel.getRangeAt(0);
      const container = localRef.current;
      if (!container || !container.contains(range.commonAncestorContainer))
        return;

      const selectedText = sel.toString();
      if (!selectedText.trim()) return;

      // Calculate precise character offsets using data-offset attributes
      const start = resolveCharOffset(
        range.startContainer,
        range.startOffset,
      );
      const end = resolveCharOffset(range.endContainer, range.endOffset);

      if (start === null || end === null || start >= end) return;

      // Verify the offset is within bounds and the text matches
      if (start < 0 || end > text.length) return;
      const verifiedText = text.slice(start, end);
      // Allow minor whitespace differences from selection vs raw text
      if (
        verifiedText.replace(/\s+/g, " ").trim() !==
        selectedText.replace(/\s+/g, " ").trim()
      ) {
        return;
      }

      const normalizedSelection = trimBoundaries
        ? trimBoundarySelection(text, start, end)
        : { start, end, text: verifiedText };
      if (!normalizedSelection.text.trim()) return;

      const rect = range.getBoundingClientRect();
      setPopup({
        x: rect.left,
        y: rect.bottom + 4,
        selStart: normalizedSelection.start,
        selEnd: normalizedSelection.end,
        selText: normalizedSelection.text,
        editIndex: null,
      });
      sel.removeAllRanges();
    }, [text, trimBoundaries]);

    const handleSpanClick = useCallback(
      (index: number, e: React.MouseEvent) => {
        const sorted = [...spans].sort((a, b) => a.start - b.start);
        const span = sorted[index];
        if (!span) return;
        const originalIndex = spans.indexOf(span);
        setPopup({
          x: e.clientX,
          y: e.clientY + 4,
          selStart: span.start,
          selEnd: span.end,
          selText: span.text,
          editIndex: originalIndex,
        });
      },
      [spans],
    );

    const handleLabelSelect = useCallback(
      (label: string) => {
        if (!popup) return;
        if (popup.editIndex !== null) {
          const updated = spans.map((s, i) =>
            i === popup.editIndex ? { ...s, label } : s,
          );
          onSpansChange(updated);
        } else {
          const newSpan: CanonicalSpan = {
            start: popup.selStart,
            end: popup.selEnd,
            label,
            text: popup.selText,
          };
          onSpansChange([...spans, newSpan]);
        }
        setPopup(null);
      },
      [popup, spans, onSpansChange],
    );

    const handleDelete = useCallback(() => {
      if (!popup || popup.editIndex === null) return;
      onSpansChange(spans.filter((_, i) => i !== popup.editIndex));
      setPopup(null);
    }, [popup, spans, onSpansChange]);

    return (
      <div className="pane">
        <div className="pane-header pane-header-manual">
          <span>Manual Annotations</span>
          <label className="pane-header-toggle" title="Trim leading/trailing spaces and punctuation from new selections">
            <input
              type="checkbox"
              checked={trimBoundaries}
              onChange={(e) => setTrimBoundaries(e.target.checked)}
            />
            Trim Space/Punct
          </label>
        </div>
        <div
          className="pane-body"
          ref={mergedRefCallback}
          onScroll={(e) => onScroll((e.target as HTMLDivElement).scrollTop)}
          onMouseUp={handleMouseUp}
        >
          <AnnotatedText
            text={text}
            spans={[...spans].sort((a, b) => a.start - b.start)}
            clickable
            onSpanClick={handleSpanClick}
            diffSpans={diffSpans}
          />
        </div>
        {popup && (
          <AnnotationPopup
            x={popup.x}
            y={popup.y}
            labels={labels}
            onSelect={handleLabelSelect}
            onDelete={popup.editIndex !== null ? handleDelete : undefined}
            onClose={() => setPopup(null)}
          />
        )}
      </div>
    );
  },
);

ManualAnnotationPane.displayName = "ManualAnnotationPane";
export default ManualAnnotationPane;
