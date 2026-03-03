import { forwardRef } from "react";
import AnnotatedText from "./AnnotatedText";

interface Props {
  text: string;
  diffSpans?: { start: number; end: number; type: "added" | "removed" }[];
  onScroll: (scrollTop: number) => void;
}

const RawPane = forwardRef<HTMLDivElement, Props>(
  ({ text, diffSpans = [], onScroll }, ref) => {
  return (
    <div className="pane">
      <div className="pane-header">Raw Transcript</div>
      <div
        className="pane-body"
        ref={ref}
        onScroll={(e) => onScroll((e.target as HTMLDivElement).scrollTop)}
      >
        <AnnotatedText text={text} spans={[]} diffSpans={diffSpans} />
      </div>
    </div>
  );
  },
);

RawPane.displayName = "RawPane";
export default RawPane;
