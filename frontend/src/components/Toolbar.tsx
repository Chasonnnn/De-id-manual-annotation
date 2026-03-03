import type {
  AnnotationSource,
  LabelProjection,
  MatchMode,
  PaneType,
} from "../types";

type SaveStatus = "idle" | "saving" | "saved";

interface Props {
  visiblePanes: PaneType[];
  onTogglePane: (pane: PaneType) => void;
  diffMode: boolean;
  onToggleDiff: () => void;
  reference: AnnotationSource;
  onReferenceChange: (ref: AnnotationSource) => void;
  hypothesis: AnnotationSource;
  onHypothesisChange: (hyp: AnnotationSource) => void;
  sourceOptions: Array<{ value: AnnotationSource; label: string }>;
  matchMode: MatchMode;
  onMatchModeChange: (mode: MatchMode) => void;
  labelProjection: LabelProjection;
  onLabelProjectionChange: (projection: LabelProjection) => void;
  saveStatus?: SaveStatus;
}

export default function Toolbar({
  visiblePanes,
  onTogglePane,
  diffMode,
  onToggleDiff,
  reference,
  onReferenceChange,
  hypothesis,
  onHypothesisChange,
  sourceOptions,
  matchMode,
  onMatchModeChange,
  labelProjection,
  onLabelProjectionChange,
  saveStatus = "idle",
}: Props) {
  const paneButtons: { type: PaneType; label: string }[] = [
    { type: "raw", label: "+Raw" },
    { type: "pre", label: "+Pre-annotations" },
    { type: "manual", label: "+Manual" },
    { type: "agent", label: "+Agent" },
    { type: "methods", label: "+Methods" },
  ];

  return (
    <div className="toolbar">
      <div className="toolbar-group">
        <span className="toolbar-label">Panes:</span>
        {paneButtons.map((p) => (
          <button
            key={p.type}
            className={visiblePanes.includes(p.type) ? "active" : ""}
            onClick={() => onTogglePane(p.type)}
          >
            {p.label}
          </button>
        ))}
      </div>
      <div className="toolbar-group">
        <button
          className={diffMode ? "active" : ""}
          onClick={onToggleDiff}
        >
          Diff Mode
        </button>
      </div>
      <div className="toolbar-group">
        <label htmlFor="toolbar-reference-select">Reference:</label>
        <select
          id="toolbar-reference-select"
          name="reference"
          value={reference}
          onChange={(e) =>
            onReferenceChange(e.target.value as AnnotationSource)
          }
        >
          {sourceOptions.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>
      <div className="toolbar-group">
        <label htmlFor="toolbar-hypothesis-select">Hypothesis:</label>
        <select
          id="toolbar-hypothesis-select"
          name="hypothesis"
          value={hypothesis}
          onChange={(e) =>
            onHypothesisChange(e.target.value as AnnotationSource)
          }
        >
          {sourceOptions.map((option) => (
            <option key={option.value} value={option.value}>
              {option.label}
            </option>
          ))}
        </select>
      </div>
      <div className="toolbar-group">
        <label htmlFor="toolbar-match-select">Match:</label>
        <select
          id="toolbar-match-select"
          name="match_mode"
          value={matchMode}
          onChange={(e) => onMatchModeChange(e.target.value as MatchMode)}
        >
          <option value="exact">Exact</option>
          <option value="overlap">Overlap</option>
        </select>
      </div>
      <div className="toolbar-group">
        <label htmlFor="toolbar-label-projection-select">Labels:</label>
        <select
          id="toolbar-label-projection-select"
          name="label_projection"
          value={labelProjection}
          onChange={(e) =>
            onLabelProjectionChange(e.target.value as LabelProjection)
          }
        >
          <option value="native">Native</option>
          <option value="coarse_simple">Coarse (advanced→simple)</option>
        </select>
      </div>
      {saveStatus !== "idle" && (
        <div className="toolbar-group">
          <span className={`save-indicator ${saveStatus}`}>
            {saveStatus === "saving" ? "Saving..." : "Saved"}
          </span>
        </div>
      )}
    </div>
  );
}
