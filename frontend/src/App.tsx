import { Component, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { PII_LABELS } from "./types";
import type {
  AnnotationSource,
  AgentConfig,
  AgentMethodOption,
  AgentView,
  CanonicalDocument,
  CanonicalSpan,
  DashboardMetricsResult,
  DocumentSummary,
  MatchMode,
  MethodView,
  MetricsResult,
  PaneType,
  SessionProfile,
} from "./types";
import {
  deleteDocument,
  exportSession,
  getAgentMethods,
  getDocument,
  getMetricsDashboard,
  getMetrics,
  getSessionProfile,
  importSession,
  listDocuments,
  runAgent,
  updateSessionProfile,
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
import MethodPane from "./components/MethodPane";
import MetricsPanel from "./components/MetricsPanel";
import DashboardPanel from "./components/DashboardPanel";
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

const CANONICAL_PANE_ORDER: PaneType[] = ["raw", "pre", "manual", "agent", "methods"];

function normalizePaneOrder(panes: PaneType[]): PaneType[] {
  const visible = new Set(panes);
  return CANONICAL_PANE_ORDER.filter((pane) => visible.has(pane));
}

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
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);
  const [importing, setImporting] = useState(false);
  const [savingProfile, setSavingProfile] = useState(false);
  const [sessionProfile, setSessionProfile] = useState<SessionProfile>({
    project_name: "",
    author: "",
    notes: "",
  });

  const [visiblePanes, setVisiblePanes] = useState<PaneType[]>(() =>
    normalizePaneOrder(["raw", "pre"]),
  );
  const [diffMode, setDiffMode] = useState(false);
  const [reference, setReference] = useState<AnnotationSource>("pre");
  const [hypothesis, setHypothesis] = useState<AnnotationSource>("manual");
  const [matchMode, setMatchMode] = useState<MatchMode>("exact");
  const [agentView, setAgentView] = useState<AgentView>("combined");
  const [agentMethods, setAgentMethods] = useState<AgentMethodOption[]>([]);
  const [methodView, setMethodView] = useState<MethodView>("default");

  const [agentRunning, setAgentRunning] = useState(false);
  const [methodRunning, setMethodRunning] = useState(false);
  const [metrics, setMetrics] = useState<MetricsResult | null>(null);
  const [metricsLoading, setMetricsLoading] = useState(false);
  const [dashboard, setDashboard] = useState<DashboardMetricsResult | null>(null);
  const [dashboardLoading, setDashboardLoading] = useState(false);

  const [saveStatus, setSaveStatus] = useState<SaveStatus>("idle"); // 4.1
  const saveTimer = useRef<ReturnType<typeof setTimeout>>(null);
  const savedTimer = useRef<ReturnType<typeof setTimeout>>(null);

  const { registerPane, handleScroll } = useSyncScroll();

  const refreshDocuments = useCallback(async () => {
    const docs = await listDocuments();
    setDocuments(docs);
    return docs;
  }, []);

  // Load document list
  useEffect(() => {
    refreshDocuments().catch((e: unknown) => setError(String(e)));
  }, [refreshDocuments]);

  useEffect(() => {
    getSessionProfile()
      .then(setSessionProfile)
      .catch((e: unknown) => setError(String(e)));
  }, []);

  useEffect(() => {
    getAgentMethods()
      .then((methods) => setAgentMethods(methods))
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
        await refreshDocuments();
        setSelectedId(newDoc.id);
      } catch (e: unknown) {
        setError(String(e));
      } finally {
        setUploading(false);
      }
    },
    [refreshDocuments],
  );

  const handleDeleteDocument = useCallback(
    async (docId: string) => {
      if (!window.confirm("Delete this document and its annotations?")) return;
      setDeletingId(docId);
      try {
        await deleteDocument(docId);
        const refreshed = await refreshDocuments();
        if (selectedId === docId) {
          const nextId = refreshed[0]?.id ?? null;
          setSelectedId(nextId);
          if (!nextId) {
            setDoc(null);
            setMetrics(null);
          }
        }
      } catch (e: unknown) {
        setError(String(e));
      } finally {
        setDeletingId(null);
      }
    },
    [refreshDocuments, selectedId],
  );

  const handleExportSession = useCallback(async () => {
    setExporting(true);
    try {
      await updateSessionProfile(sessionProfile);
      const bundle = await exportSession();
      const stamp = new Date().toISOString().replace(/[:.]/g, "-");
      const blob = new Blob([JSON.stringify(bundle, null, 2)], {
        type: "application/json",
      });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `annotation-session-${stamp}.json`;
      document.body.appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
    } catch (e: unknown) {
      setError(String(e));
    } finally {
      setExporting(false);
    }
  }, [sessionProfile]);

  const handleImportSession = useCallback(
    async (file: File) => {
      setImporting(true);
      try {
        const result = await importSession(file);
        const refreshed = await refreshDocuments();
        if (result.imported_ids.length > 0) {
          const firstImportedId = result.imported_ids[0] ?? null;
          const selectedStillExists =
            selectedId !== null && refreshed.some((docItem) => docItem.id === selectedId);
          if (!selectedStillExists && firstImportedId) {
            setSelectedId(firstImportedId);
          }
        }
        const warnings: string[] = [];
        if (result.skipped_count > 0) {
          warnings.push(
            `Imported ${result.imported_count} document(s), skipped ${result.skipped_count}.`,
          );
        } else {
          warnings.push(`Imported ${result.imported_count} document(s).`);
        }
        if ((result.warnings ?? []).length > 0) {
          warnings.push(`Import warnings: ${result.warnings?.join(" | ")}`);
        }
        setWarning(warnings.join(" "));
        const refreshedProfile = await getSessionProfile();
        setSessionProfile(refreshedProfile);
      } catch (e: unknown) {
        setError(String(e));
      } finally {
        setImporting(false);
      }
    },
    [refreshDocuments, selectedId],
  );

  const handleSaveSessionProfile = useCallback(async () => {
    setSavingProfile(true);
    try {
      const saved = await updateSessionProfile(sessionProfile);
      setSessionProfile(saved);
      setWarning("Bundle info saved.");
    } catch (e: unknown) {
      setError(String(e));
    } finally {
      setSavingProfile(false);
    }
  }, [sessionProfile]);

  const handleTogglePane = useCallback((pane: PaneType) => {
    setVisiblePanes((prev) => {
      const next = prev.includes(pane)
        ? prev.filter((p) => p !== pane)
        : [...prev, pane];
      return normalizePaneOrder(next);
    });
  }, []);

  const orderedVisiblePanes = useMemo(
    () => normalizePaneOrder(visiblePanes),
    [visiblePanes],
  );

  const methodOutputIds = useMemo(
    () => Object.keys(doc?.agent_outputs?.methods ?? {}),
    [doc],
  );

  const methodCatalog = useMemo<AgentMethodOption[]>(() => {
    const merged = [...agentMethods];
    const knownIds = new Set(agentMethods.map((method) => method.id));
    for (const methodId of methodOutputIds) {
      if (knownIds.has(methodId)) continue;
      merged.push({
        id: methodId,
        label: methodId,
        description: "Imported method output",
        requires_presidio: false,
        uses_llm: true,
        supports_verify_override: false,
        available: true,
        unavailable_reason: null,
      });
      knownIds.add(methodId);
    }
    return merged;
  }, [agentMethods, methodOutputIds]);

  useEffect(() => {
    if (methodCatalog.length === 0) return;
    const existing = methodCatalog.some((method) => method.id === methodView);
    if (existing) return;
    const withOutput = methodCatalog.find(
      (method) => (doc?.agent_outputs?.methods?.[method.id]?.length ?? 0) > 0,
    );
    const fallbackMethodId = withOutput?.id ?? methodCatalog[0]?.id ?? null;
    if (fallbackMethodId) {
      setMethodView(fallbackMethodId);
    }
  }, [doc, methodCatalog, methodView]);

  const sourceOptions = useMemo<Array<{ value: AnnotationSource; label: string }>>(() => {
    const options: Array<{ value: AnnotationSource; label: string }> = [
      { value: "pre", label: "Pre-annotations" },
      { value: "manual", label: "Manual annotations" },
      { value: "agent", label: "Agent (combined)" },
      { value: "agent.rule", label: "Agent (rule)" },
      { value: "agent.llm", label: "Agent (llm)" },
    ];
    for (const method of methodCatalog) {
      options.push({
        value: `agent.method.${method.id}` as AnnotationSource,
        label: `Method: ${method.label}`,
      });
    }
    return options;
  }, [methodCatalog]);

  useEffect(() => {
    if (sourceOptions.length === 0) return;
    const values = sourceOptions.map((option) => option.value);
    const primary = values[0];
    if (!primary) return;
    if (!values.includes(reference)) {
      setReference(primary);
    }
    if (!values.includes(hypothesis)) {
      const fallback = values.find((value) => value !== reference) ?? primary;
      setHypothesis(fallback);
      return;
    }
    if (reference === hypothesis && values.length > 1) {
      const fallback = values.find((value) => value !== reference) ?? primary;
      setHypothesis(fallback);
    }
  }, [hypothesis, reference, sourceOptions]);

  const handleDashboardRefresh = useCallback(async () => {
    if (documents.length === 0) {
      setDashboard(null);
      return;
    }
    if (reference === hypothesis) {
      setError("Reference and hypothesis must be different for metrics");
      return;
    }
    setDashboardLoading(true);
    try {
      const result = await getMetricsDashboard(reference, hypothesis, matchMode);
      setDashboard(result);
    } catch (e: unknown) {
      setError(String(e));
    } finally {
      setDashboardLoading(false);
    }
  }, [documents.length, reference, hypothesis, matchMode]);

  useEffect(() => {
    void handleDashboardRefresh();
  }, [handleDashboardRefresh]);

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
            void handleDashboardRefresh();
          })
          .catch((e: unknown) => {
            setSaveStatus("idle");
            setError(String(e));
          });
      }, 1000);
    },
    [doc, handleDashboardRefresh],
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
        void handleDashboardRefresh();
      } catch (e: unknown) {
        setError(String(e));
      } finally {
        setAgentRunning(false);
      }
    },
    [doc, handleDashboardRefresh],
  );

  const handleRunMethod = useCallback(
    async (config: AgentConfig) => {
      if (!doc) return;
      setMethodRunning(true);
      try {
        const updated = await runAgent(doc.id, config);
        setDoc(updated);
        if ((updated.agent_run_warnings ?? []).length > 0) {
          setWarning(updated.agent_run_warnings.join(" "));
        } else {
          setWarning(null);
        }
        void handleDashboardRefresh();
      } catch (e: unknown) {
        setError(String(e));
      } finally {
        setMethodRunning(false);
      }
    },
    [doc, handleDashboardRefresh],
  );

  const handleMetricsRefresh = useCallback(async () => {
    if (!doc) return;
    if (reference === hypothesis) {
      setError("Reference and hypothesis must be different for metrics");
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
  }, [doc, reference, hypothesis, matchMode]);

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
        if (source.startsWith("agent.method.")) {
          const methodId = source.slice("agent.method.".length);
          return doc.agent_outputs?.methods?.[methodId] ?? [];
        }
        return [];
    }
  };

  const getAgentSpans = (): CanonicalSpan[] => {
    if (!doc) return [];
    if (agentView === "rule") return doc.agent_outputs?.rule ?? [];
    if (agentView === "llm") return doc.agent_outputs?.llm ?? [];
    return doc.agent_annotations;
  };

  const getMethodSpans = (): CanonicalSpan[] => {
    if (!doc || !methodView) return [];
    return doc.agent_outputs?.methods?.[methodView] ?? [];
  };

  const getPaneSource = (paneType: PaneType): AnnotationSource | null => {
    if (paneType === "pre") return "pre";
    if (paneType === "manual") return "manual";
    if (paneType === "agent") {
      if (agentView === "rule") return "agent.rule";
      if (agentView === "llm") return "agent.llm";
      return "agent";
    }
    if (paneType === "methods") {
      return methodView ? (`agent.method.${methodView}` as AnnotationSource) : null;
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
    const methodSource = methodView
      ? (`agent.method.${methodView}` as AnnotationSource)
      : null;
    const visibleSources = getDisplayedSources(
      orderedVisiblePanes,
      agentView,
      methodSource,
    ).filter((item) => item !== source);
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
      const methodSource = methodView
        ? (`agent.method.${methodView}` as AnnotationSource)
        : null;
      const visibleSources = getDisplayedSources(
        orderedVisiblePanes,
        agentView,
        methodSource,
      );
      return dedupeDiffs(
        visibleSources.flatMap((source) => getGlobalDiffForSource(source)),
      );
    }
    const source = getPaneSource(paneType);
    if (!source) return [];
    return getGlobalDiffForSource(source);
  };

  // Build ordered list of panes to render
  const allPanes: PaneType[] = orderedVisiblePanes;
  let paneIndex = 0;

  return (
    <div className="app-layout">
      <Sidebar
        documents={documents}
        selectedId={selectedId}
        onSelect={setSelectedId}
        onUpload={handleUpload}
        onDelete={handleDeleteDocument}
        onExportSession={handleExportSession}
        onImportSession={handleImportSession}
        sessionProfile={sessionProfile}
        onSessionProfileChange={setSessionProfile}
        onSaveSessionProfile={handleSaveSessionProfile}
        uploading={uploading}
        deletingId={deletingId}
        exporting={exporting}
        importing={importing}
        savingProfile={savingProfile}
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
              visiblePanes={orderedVisiblePanes}
              onTogglePane={handleTogglePane}
              diffMode={diffMode}
              onToggleDiff={() => setDiffMode(!diffMode)}
              reference={reference}
              onReferenceChange={setReference}
              hypothesis={hypothesis}
              onHypothesisChange={setHypothesis}
              sourceOptions={sourceOptions}
              matchMode={matchMode}
              onMatchModeChange={setMatchMode}
              saveStatus={saveStatus}
            />
            <DashboardPanel
              dashboard={dashboard}
              loading={dashboardLoading}
              onRefresh={handleDashboardRefresh}
              selectedId={selectedId}
              onSelectDocument={setSelectedId}
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
                  case "methods":
                    return (
                      <MethodPane
                        key="methods"
                        ref={registerPane(idx)}
                        text={doc.raw_text}
                        spans={getMethodSpans()}
                        methods={methodCatalog}
                        activeMethod={methodView}
                        onActiveMethodChange={setMethodView}
                        diffSpans={getDiffSpans("methods")}
                        onRunMethod={handleRunMethod}
                        running={methodRunning}
                        onScroll={handleScroll(idx)}
                      />
                    );
                }
              })}
            </PaneContainer>
            <MetricsPanel
              reference={reference}
              hypothesis={hypothesis}
              matchMode={matchMode}
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

function getDisplayedSources(
  visiblePanes: PaneType[],
  agentView: AgentView,
  methodSource: AnnotationSource | null,
): AnnotationSource[] {
  const orderedPanes = normalizePaneOrder(visiblePanes);
  const sources: AnnotationSource[] = [];
  if (orderedPanes.includes("pre")) sources.push("pre");
  if (orderedPanes.includes("manual")) sources.push("manual");
  if (orderedPanes.includes("agent")) {
    if (agentView === "rule") sources.push("agent.rule");
    else if (agentView === "llm") sources.push("agent.llm");
    else sources.push("agent");
  }
  if (orderedPanes.includes("methods") && methodSource) {
    sources.push(methodSource);
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
