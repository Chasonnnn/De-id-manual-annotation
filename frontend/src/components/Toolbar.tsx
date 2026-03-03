import type { AnnotationSource, MatchMode, PaneType } from "../types";

type SaveStatus = "idle" | "saving" | "saved";

interface Props {
  visiblePanes: PaneType[];
  onTogglePane: (pane: PaneType) => void;
  diffMode: boolean;
  onToggleDiff: () => void;
  reference: AnnotationSource;
  onReferenceChange: (ref: AnnotationSource) => void;
  matchMode: MatchMode;
  onMatchModeChange: (mode: MatchMode) => void;
  saveStatus?: SaveStatus;
}

export default function Toolbar({
  visiblePanes,
  onTogglePane,
  diffMode,
  onToggleDiff,
  reference,
  onReferenceChange,
  matchMode,
  onMatchModeChange,
  saveStatus = "idle",
}: Props) {
  const paneButtons: { type: PaneType; label: string }[] = [
    { type: "pre", label: "+Pre-annotations" },
    { type: "manual", label: "+Manual" },
    { type: "agent", label: "+Agent" },
  ];

  return (
    <div className="toolbar">
      <div className="toolbar-group">
        <label>Panes:</label>
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
        <label>Reference:</label>
        <select
          value={reference}
          onChange={(e) =>
            onReferenceChange(e.target.value as AnnotationSource)
          }
        >
          <option value="pre">Pre-annotations</option>
          <option value="manual">Manual</option>
          <option value="agent">Agent (Combined)</option>
          <option value="agent.rule">Agent (Rule)</option>
          <option value="agent.llm">Agent (LLM)</option>
        </select>
      </div>
      <div className="toolbar-group">
        <label>Match:</label>
        <select
          value={matchMode}
          onChange={(e) => onMatchModeChange(e.target.value as MatchMode)}
        >
          <option value="exact">Exact</option>
          <option value="overlap">Overlap</option>
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
