import { Component, useCallback, useEffect, useRef, useState } from "react";
import { PII_LABELS } from "./types";
import type {
  AnnotationSource,
  AgentConfig,
  AgentView,
  CanonicalDocument,
  CanonicalSpan,
  DocumentSummary,
  MatchMode,
  MetricsResult,
  PaneType,
} from "./types";
import {
  getDocument,
  getMetrics,
  listDocuments,
  runAgent,
  updateManualAnnotations,
  uploadDocument,
} from "./api/client";
import Sidebar from "./components/Sidebar";
import Toolbar from "./components/Toolbar";
import PaneContainer, { useSyncScroll } from "./components/PaneContainer";
import RawPane from "./components/RawPane";
import PreAnnotationPane from "./components/PreAnnotationPane";
import ManualAnnotationPane from "./components/ManualAnnotationPane";
import AgentPane from "./components/AgentPane";
import MetricsPanel from "./components/MetricsPanel";
import { computeDiff } from "./components/DiffOverlay";

// ---------------------------------------------------------------------------
// 4.3: Error boundary to prevent white-screen crashes
// ---------------------------------------------------------------------------
interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

class ErrorBoundary extends Component<
  { children: React.ReactNode },
  ErrorBoundaryState
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  render() {
    if (this.state.hasError) {
      return (
        <div
          style={{
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            height: "100vh",
            gap: 16,
            color: "#555",
            fontFamily: "sans-serif",
          }}
        >
          <h2>Something went wrong</h2>
          <p style={{ color: "#888", maxWidth: 500, textAlign: "center" }}>
            {this.state.error?.message ?? "An unexpected error occurred."}
          </p>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            style={{
              padding: "8px 20px",
              background: "#4a6cf7",
              color: "#fff",
              border: "none",
              borderRadius: 6,
              cursor: "pointer",
              fontSize: 14,
            }}
          >
            Try Again
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

// ---------------------------------------------------------------------------
// Save status type
// ---------------------------------------------------------------------------
type SaveStatus = "idle" | "saving" | "saved";

// ---------------------------------------------------------------------------
// Main App
// ---------------------------------------------------------------------------
function AppContent() {
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [doc, setDoc] = useState<CanonicalDocument | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [warning, setWarning] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false); // 4.2: upload loading

  const [visiblePanes, setVisiblePanes] = useState<PaneType[]>(["pre"]);
  const [diffMode, setDiffMode] = useState(false);
  const [reference, setReference] = useState<AnnotationSource>("pre");
  const [matchMode, setMatchMode] = useState<MatchMode>("exact");
  const [agentView, setAgentView] = useState<AgentView>("combined");

  const [agentRunning, setAgentRunning] = useState(false);
  const [metrics, setMetrics] = useState<MetricsResult | null>(null);
  const [metricsLoading, setMetricsLoading] = useState(false);

  const [saveStatus, setSaveStatus] = useState<SaveStatus>("idle"); // 4.1
  const saveTimer = useRef<ReturnType<typeof setTimeout>>(null);
  const savedTimer = useRef<ReturnType<typeof setTimeout>>(null);

  const { registerPane, handleScroll } = useSyncScroll();

  // Load document list
  useEffect(() => {
    listDocuments()
      .then(setDocuments)
      .catch((e: unknown) => setError(String(e)));
  }, []);

  // Load selected document
  useEffect(() => {
    if (!selectedId) {
      setDoc(null);
      return;
    }
    setLoading(true);
    getDocument(selectedId)
      .then((d) => {
        setDoc(d);
        setMetrics(null);
        setSaveStatus("idle");
        setWarning(null);
      })
      .catch((e: unknown) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [selectedId]);

  // 4.2: Upload with loading indicator
  const handleUpload = useCallback(
    async (file: File) => {
      setUploading(true);
      try {
        const newDoc = await uploadDocument(file);
        const refreshed = await listDocuments();
        setDocuments(refreshed);
        setSelectedId(newDoc.id);
      } catch (e: unknown) {
        setError(String(e));
      } finally {
        setUploading(false);
      }
    },
    [],
  );

  const handleTogglePane = useCallback((pane: PaneType) => {
    setVisiblePanes((prev) =>
      prev.includes(pane) ? prev.filter((p) => p !== pane) : [...prev, pane],
    );
  }, []);

  // 4.1: Save status in debounced auto-save
  const handleManualChange = useCallback(
    (spans: CanonicalSpan[]) => {
      if (!doc) return;
      setDoc({ ...doc, manual_annotations: spans });

      // Debounced auto-save
      if (saveTimer.current) clearTimeout(saveTimer.current);
      if (savedTimer.current) clearTimeout(savedTimer.current);
      setSaveStatus("saving");
      saveTimer.current = setTimeout(() => {
        updateManualAnnotations(doc.id, spans)
          .then(() => {
            setSaveStatus("saved");
            savedTimer.current = setTimeout(() => setSaveStatus("idle"), 2000);
          })
          .catch((e: unknown) => {
            setSaveStatus("idle");
            setError(String(e));
          });
      }, 1000);
    },
    [doc],
  );

  const handleRunAgent = useCallback(
    async (config: AgentConfig) => {
      if (!doc) return;
      setAgentRunning(true);
      try {
        const updated = await runAgent(doc.id, config);
        setDoc(updated);
        if ((updated.agent_run_warnings ?? []).length > 0) {
          setWarning(updated.agent_run_warnings.join(" "));
        } else {
          setWarning(null);
        }
      } catch (e: unknown) {
        setError(String(e));
      } finally {
        setAgentRunning(false);
      }
    },
    [doc],
  );

  const handleMetricsRefresh = useCallback(async () => {
    if (!doc) return;
    const hypothesisOptions = getSelectableSources(visiblePanes).filter(
      (source) => source !== reference,
    );
    const hypothesis = hypothesisOptions[0];
    if (!hypothesis) {
      setError("Need at least two annotation panes for metrics comparison");
      return;
    }
    setMetricsLoading(true);
    try {
      const result = await getMetrics(doc.id, reference, hypothesis, matchMode);
      setMetrics(result);
    } catch (e: unknown) {
      setError(String(e));
    } finally {
      setMetricsLoading(false);
    }
  }, [doc, visiblePanes, reference, matchMode]);

  // Compute diffs if diff mode is on
  const getSpansForSource = (source: AnnotationSource): CanonicalSpan[] => {
    if (!doc) return [];
    switch (source) {
      case "pre":
        return doc.pre_annotations;
      case "manual":
        return doc.manual_annotations;
      case "agent":
        return doc.agent_annotations;
      case "agent.rule":
        return doc.agent_outputs?.rule ?? [];
      case "agent.llm":
        return doc.agent_outputs?.llm ?? [];
      default:
        return [];
    }
  };

  const getAgentSpans = (): CanonicalSpan[] => {
    if (!doc) return [];
    if (agentView === "rule") return doc.agent_outputs?.rule ?? [];
    if (agentView === "llm") return doc.agent_outputs?.llm ?? [];
    return doc.agent_annotations;
  };

  const getPaneSource = (paneType: PaneType): AnnotationSource | null => {
    if (paneType === "pre") return "pre";
    if (paneType === "manual") return "manual";
    if (paneType === "agent") {
      if (agentView === "rule") return "agent.rule";
      if (agentView === "llm") return "agent.llm";
      return "agent";
    }
    return null;
  };

  const dedupeDiffs = (
    diffs: { start: number; end: number; type: "added" | "removed" }[],
  ) => {
    const seen = new Set<string>();
    const out: { start: number; end: number; type: "added" | "removed" }[] = [];
    for (const diff of diffs) {
      const key = `${diff.start}:${diff.end}:${diff.type}`;
      if (!seen.has(key)) {
        seen.add(key);
        out.push(diff);
      }
    }
    return out;
  };

  const getGlobalDiffForSource = (source: AnnotationSource) => {
    if (!doc || !diffMode) return [];
    const visibleSources = getDisplayedSources(visiblePanes, agentView).filter(
      (item) => item !== source,
    );
    if (visibleSources.length === 0) return [];

    const target = getSpansForSource(source);
    const allDiffs = visibleSources.flatMap((other) =>
      computeDiff(getSpansForSource(other), target, matchMode),
    );
    return dedupeDiffs(allDiffs);
  };

  const getDiffSpans = (paneType: PaneType) => {
    if (!doc || !diffMode) return [];
    if (paneType === "raw") {
      const visibleSources = getDisplayedSources(visiblePanes, agentView);
      return dedupeDiffs(
        visibleSources.flatMap((source) => getGlobalDiffForSource(source)),
      );
    }
    const source = getPaneSource(paneType);
    if (!source) return [];
    return getGlobalDiffForSource(source);
  };

  // Build ordered list of panes to render
  const allPanes: PaneType[] = ["raw", ...visiblePanes];
  let paneIndex = 0;

  return (
    <div className="app-layout">
      <Sidebar
        documents={documents}
        selectedId={selectedId}
        onSelect={setSelectedId}
        onUpload={handleUpload}
        uploading={uploading}
      />
      <div className="main-area">
        {!doc && !loading && (
          <div className="empty-state">
            Select a document or upload a file to begin
          </div>
        )}
        {loading && <div className="loading">Loading document...</div>}
        {doc && (
          <>
            <Toolbar
              visiblePanes={visiblePanes}
              onTogglePane={handleTogglePane}
              diffMode={diffMode}
              onToggleDiff={() => setDiffMode(!diffMode)}
              reference={reference}
              onReferenceChange={setReference}
              matchMode={matchMode}
              onMatchModeChange={setMatchMode}
              saveStatus={saveStatus}
            />
            <PaneContainer>
              {allPanes.map((paneType) => {
                const idx = paneIndex++;
                switch (paneType) {
                  case "raw":
                    return (
                      <RawPane
                        key="raw"
                        ref={registerPane(idx)}
                        text={doc.raw_text}
                        diffSpans={getDiffSpans("raw")}
                        onScroll={handleScroll(idx)}
                      />
                    );
                  case "pre":
                    return (
                      <PreAnnotationPane
                        key="pre"
                        ref={registerPane(idx)}
                        text={doc.raw_text}
                        spans={doc.pre_annotations}
                        diffSpans={getDiffSpans("pre")}
                        onScroll={handleScroll(idx)}
                      />
                    );
                  case "manual":
                    return (
                      <ManualAnnotationPane
                        key="manual"
                        ref={registerPane(idx)}
                        text={doc.raw_text}
                        labels={doc.label_set?.length ? doc.label_set : PII_LABELS}
                        spans={doc.manual_annotations}
                        diffSpans={getDiffSpans("manual")}
                        onSpansChange={handleManualChange}
                        onScroll={handleScroll(idx)}
                      />
                    );
                  case "agent":
                    return (
                      <AgentPane
                        key="agent"
                        ref={registerPane(idx)}
                        text={doc.raw_text}
                        spans={getAgentSpans()}
                        activeOutput={agentView}
                        onActiveOutputChange={setAgentView}
                        diffSpans={getDiffSpans("agent")}
                        onRunAgent={handleRunAgent}
                        running={agentRunning}
                        onScroll={handleScroll(idx)}
                      />
                    );
                }
              })}
            </PaneContainer>
            <MetricsPanel
              metrics={metrics}
              loading={metricsLoading}
              onRefresh={handleMetricsRefresh}
            />
          </>
        )}
      </div>
      {error && (
        <div className="error-toast" onClick={() => setError(null)}>
          {error}
        </div>
      )}
      {warning && (
        <div className="warning-toast" onClick={() => setWarning(null)}>
          {warning}
        </div>
      )}
    </div>
  );
}

function getSelectableSources(visiblePanes: PaneType[]): AnnotationSource[] {
  const sources: AnnotationSource[] = [];
  if (visiblePanes.includes("pre")) sources.push("pre");
  if (visiblePanes.includes("manual")) sources.push("manual");
  if (visiblePanes.includes("agent")) {
    sources.push("agent", "agent.rule", "agent.llm");
  }
  return sources;
}

function getDisplayedSources(
  visiblePanes: PaneType[],
  agentView: AgentView,
): AnnotationSource[] {
  const sources: AnnotationSource[] = [];
  if (visiblePanes.includes("pre")) sources.push("pre");
  if (visiblePanes.includes("manual")) sources.push("manual");
  if (visiblePanes.includes("agent")) {
    if (agentView === "rule") sources.push("agent.rule");
    else if (agentView === "llm") sources.push("agent.llm");
    else sources.push("agent");
  }
  return sources;
}

export default function App() {
  return (
    <ErrorBoundary>
      <AppContent />
    </ErrorBoundary>
  );
}
