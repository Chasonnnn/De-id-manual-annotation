import { forwardRef } from "react";
import type { CanonicalSpan } from "../types";
import AnnotatedText from "./AnnotatedText";

interface Props {
  text: string;
  spans: CanonicalSpan[];
  diffSpans?: { start: number; end: number; type: "added" | "removed" }[];
  onScroll: (scrollTop: number) => void;
}

const PreAnnotationPane = forwardRef<HTMLDivElement, Props>(
  ({ text, spans, diffSpans = [], onScroll }, ref) => {
    return (
      <div className="pane">
        <div className="pane-header">Pre-annotations</div>
        <div
          className="pane-body"
          ref={ref}
          onScroll={(e) => onScroll((e.target as HTMLDivElement).scrollTop)}
        >
          <AnnotatedText text={text} spans={spans} diffSpans={diffSpans} />
        </div>
      </div>
    );
  },
);

PreAnnotationPane.displayName = "PreAnnotationPane";
export default PreAnnotationPane;
