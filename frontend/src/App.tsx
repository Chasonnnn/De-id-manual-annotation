import { Component, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { PII_LABELS } from "./types";
import type {
  AnnotationSource,
  AgentConfig,
  AgentChunkDiagnostic,
  AgentMethodOption,
  AgentRunProgress,
  AgentView,
  CanonicalDocument,
  CanonicalSpan,
  DashboardMetricsResult,
  DocumentSummary,
  LabelProjection,
  MatchMode,
  MethodView,
  MetricsResult,
  PaneType,
  SessionProfile,
} from "./types";
import {
  deleteDocument,
  exportGroundTruth,
  exportSession,
  getAgentProgress,
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
import PromptLabTab from "./components/PromptLabTab";
import MethodsLabTab from "./components/MethodsLabTab";
import { importSessionFiles } from "./importSessionFiles";

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
type RunToastKind = "success" | "error" | "info";

interface RunToast {
  id: string;
  kind: RunToastKind;
  message: string;
}

const CANONICAL_PANE_ORDER: PaneType[] = ["raw", "pre", "manual", "agent", "methods"];

function normalizePaneOrder(panes: PaneType[]): PaneType[] {
  const visible = new Set(panes);
  return CANONICAL_PANE_ORDER.filter((pane) => visible.has(pane));
}

function isChunkWarning(message: string): boolean {
  const trimmed = message.trim();
  return trimmed.startsWith("Chunked ") || /^Chunk \d+\/\d+:/.test(trimmed);
}

function hasChunkWarnings(messages: string[]): boolean {
  return messages.some(isChunkWarning);
}

function formatRunTimestamp(value: string | null | undefined): string {
  if (!value) return "";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString([], {
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function nonChunkWarningMessage(messages: string[]): string | null {
  const filtered = messages.filter((message) => !isChunkWarning(message));
  return filtered.length > 0 ? filtered.join(" ") : null;
}

function getChunkDiagnosticsCount(
  diagnostics: AgentChunkDiagnostic[] | null | undefined,
): number {
  return diagnostics?.length ?? 0;
}

function resolveMethodChunkDiagnostics(
  doc: CanonicalDocument | null,
  methodId: string,
): AgentChunkDiagnostic[] {
  if (!doc || !methodId) return [];
  const runMeta = doc.agent_outputs?.method_run_metadata ?? {};
  const exact = runMeta[methodId]?.chunk_diagnostics;
  if (exact && exact.length > 0) return exact;
  if (methodId.includes("::")) return exact ?? [];

  let latestDiagnostics: AgentChunkDiagnostic[] = [];
  let latestTimestamp = Number.NEGATIVE_INFINITY;
  for (const metadata of Object.values(runMeta)) {
    if (metadata.method_id !== methodId) continue;
    const timestamp = metadata.updated_at ? new Date(metadata.updated_at).getTime() : 0;
    if (!Number.isFinite(timestamp) || timestamp < latestTimestamp) continue;
    latestTimestamp = timestamp;
    latestDiagnostics = metadata.chunk_diagnostics ?? [];
  }
  return latestDiagnostics;
}

function toPromptTemplatesFromSnapshot(
  snapshot: unknown,
): AgentMethodOption["prompt_templates"] {
  if (!snapshot || typeof snapshot !== "object") {
    return undefined;
  }
  const snapshotObj = snapshot as Record<string, unknown>;
  const passes = Array.isArray(snapshotObj.passes) ? snapshotObj.passes : [];
  const templates: NonNullable<AgentMethodOption["prompt_templates"]> = [];

  for (let i = 0; i < passes.length; i += 1) {
    const rawPass = passes[i];
    if (!rawPass || typeof rawPass !== "object") continue;
    const pass = rawPass as Record<string, unknown>;

    const promptCandidate =
      typeof pass.resolved_system_prompt === "string"
        ? pass.resolved_system_prompt
        : typeof pass.base_system_prompt === "string"
          ? pass.base_system_prompt
          : typeof pass.effective_system_prompt === "string"
            ? pass.effective_system_prompt
            : "";
    if (!promptCandidate) continue;

    const entityTypes = Array.isArray(pass.entity_types)
      ? pass.entity_types.filter((item): item is string => typeof item === "string")
      : null;

    templates.push({
      pass_index:
        typeof pass.pass_index === "number" && Number.isFinite(pass.pass_index)
          ? pass.pass_index
          : i + 1,
      entity_types: entityTypes,
      system_prompt: promptCandidate,
      source: "saved",
    });
  }

  return templates.length > 0 ? templates : undefined;
}

// ---------------------------------------------------------------------------
// Main App
// ---------------------------------------------------------------------------
function AppContent() {
  const [mainTab, setMainTab] = useState<
    "workspace" | "prompt_lab" | "methods_lab" | "dashboard"
  >(
    "workspace",
  );
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [doc, setDoc] = useState<CanonicalDocument | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [warning, setWarning] = useState<string | null>(null);
  const [runToasts, setRunToasts] = useState<RunToast[]>([]);
  const [agentRunProgress, setAgentRunProgress] = useState<AgentRunProgress | null>(null);
  const [uploading, setUploading] = useState(false); // 4.2: upload loading
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);
  const [importing, setImporting] = useState(false);
  const [savingProfile, setSavingProfile] = useState(false);
  const [sessionProfile, setSessionProfile] = useState<SessionProfile>({
    project_name: "",
    author: "",
  });

  const [visiblePanes, setVisiblePanes] = useState<PaneType[]>(() =>
    normalizePaneOrder(["raw", "pre"]),
  );
  const [diffMode, setDiffMode] = useState(false);
  const [reference, setReference] = useState<AnnotationSource>("pre");
  const [hypothesis, setHypothesis] = useState<AnnotationSource>("manual");
  const [matchMode, setMatchMode] = useState<MatchMode>("exact");
  const [labelProjection, setLabelProjection] = useState<LabelProjection>("native");
  const [agentView, setAgentView] = useState<AgentView>("combined");
  const [agentLlmRun, setAgentLlmRun] = useState<string>("__latest__");
  const [agentMethods, setAgentMethods] = useState<AgentMethodOption[]>([]);
  const [methodView, setMethodView] = useState<MethodView>("default");

  const [agentRunning, setAgentRunning] = useState(false);
  const [methodRunning, setMethodRunning] = useState(false);
  const [agentChunked, setAgentChunked] = useState(false);
  const [methodChunked, setMethodChunked] = useState(false);
  const [metrics, setMetrics] = useState<MetricsResult | null>(null);
  const [metricsLoading, setMetricsLoading] = useState(false);
  const [dashboard, setDashboard] = useState<DashboardMetricsResult | null>(null);
  const [dashboardLoading, setDashboardLoading] = useState(false);

  const [saveStatus, setSaveStatus] = useState<SaveStatus>("idle"); // 4.1
  const saveTimer = useRef<ReturnType<typeof setTimeout>>(null);
  const savedTimer = useRef<ReturnType<typeof setTimeout>>(null);
  const dashboardRefreshTimer = useRef<ReturnType<typeof setTimeout>>(null);

  const { registerPane, handleScroll } = useSyncScroll();

  const pushRunToast = useCallback((kind: RunToastKind, message: string) => {
    const id = `${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    setRunToasts((prev) => [...prev, { id, kind, message }]);
    window.setTimeout(() => {
      setRunToasts((prev) => prev.filter((item) => item.id !== id));
    }, 4500);
  }, []);

  const dismissRunToast = useCallback((id: string) => {
    setRunToasts((prev) => prev.filter((item) => item.id !== id));
  }, []);

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
      setAgentChunked(false);
      setMethodChunked(false);
      setAgentRunProgress(null);
      return;
    }
    setLoading(true);
    getDocument(selectedId)
      .then((d) => {
        setDoc(d);
        setMetrics(null);
        setSaveStatus("idle");
        setWarning(null);
        setAgentChunked(false);
        setMethodChunked(false);
      })
      .catch((e: unknown) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [selectedId]);

  useEffect(() => {
    if (!doc || (!agentRunning && !methodRunning)) return;
    let cancelled = false;
    let pollTimer: ReturnType<typeof setTimeout> | null = null;
    const poll = async () => {
      try {
        const progress = await getAgentProgress(doc.id);
        if (cancelled) return;
        setAgentRunProgress(progress);
        if (progress.status === "running") {
          pollTimer = window.setTimeout(() => {
            void poll();
          }, 1000);
        }
      } catch {
        if (cancelled) return;
        // Best-effort polling; retry on transient failures with a slower cadence.
        pollTimer = window.setTimeout(() => {
          void poll();
        }, 1500);
      }
    };
    void poll();
    return () => {
      cancelled = true;
      if (pollTimer) {
        window.clearTimeout(pollTimer);
      }
    };
  }, [doc, agentRunning, methodRunning]);

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

  const handleExportSession = useCallback(async (
    mode: "full" | "ground_truth",
    source: AnnotationSource,
  ) => {
    setExporting(true);
    try {
      await updateSessionProfile(sessionProfile);
      const stamp = new Date().toISOString().replace(/[:.]/g, "-");
      const fullBundle = mode === "full" ? await exportSession() : null;
      const blob =
        mode === "full"
          ? new Blob([JSON.stringify(fullBundle, null, 2)], {
              type: "application/json",
            })
          : await exportGroundTruth(source);
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download =
        mode === "full"
          ? `annotation-session-${stamp}.json`
          : `ground-truth-${stamp}.zip`;
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
    async (files: File[]) => {
      if (files.length === 0) return;
      setImporting(true);
      setError(null);
      setWarning(null);
      try {
        const result = await importSessionFiles(files, importSession);
        if (result.imported_ids.length > 0) {
          const refreshed = await refreshDocuments();
          const firstImportedId = result.imported_ids[0] ?? null;
          const selectedStillExists =
            selectedId !== null && refreshed.some((docItem) => docItem.id === selectedId);
          if (!selectedStillExists && firstImportedId) {
            setSelectedId(firstImportedId);
          }
        }
        if (result.imported_count > 0) {
          const refreshedProfile = await getSessionProfile();
          setSessionProfile(refreshedProfile);
        }

        const warnings: string[] = [];
        if (result.imported_count > 0) {
          warnings.push(
            `Imported ${result.imported_count} document(s) from ${result.succeeded_file_count} file(s).`,
          );
        }
        if ((result.imported_prompt_lab_runs ?? 0) > 0) {
          warnings.push(
            `Imported ${result.imported_prompt_lab_runs} Prompt Lab run(s).`,
          );
        }
        if ((result.imported_methods_lab_runs ?? 0) > 0) {
          warnings.push(
            `Imported ${result.imported_methods_lab_runs} Methods Lab run(s).`,
          );
        }
        if (result.skipped_count > 0) {
          warnings.push(`Skipped ${result.skipped_count} item(s).`);
        }
        if ((result.warnings ?? []).length > 0) {
          warnings.push(`Import warnings: ${result.warnings?.join(" | ")}`);
        }
        if (warnings.length > 0) {
          setWarning(warnings.join(" "));
        }
        if (result.failed_file_count > 0) {
          const failureSummary = result.failed_files
            .map((item) => `${item.file_name}: ${item.message}`)
            .join(" | ");
          setError(
            `Failed to import ${result.failed_file_count} file(s). ${failureSummary}`,
          );
        }
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
    const runMeta = doc?.agent_outputs?.method_run_metadata ?? {};
    for (const methodId of methodOutputIds) {
      if (knownIds.has(methodId)) continue;
      const runParts = methodId.split("::");
      const meta = runMeta[methodId];
      const runTime = formatRunTimestamp(meta?.updated_at);
      const methodName = meta?.method_id ?? runParts[0] ?? "method";
      const modelName = meta?.model ?? runParts[1] ?? "run";
      const inferredLabel =
        runParts.length === 2 || meta
          ? `${methodName} @ ${modelName}${runTime ? ` • ${runTime}` : ""}`
          : methodId;
      merged.push({
        id: methodId,
        label: inferredLabel,
        description: runParts.length === 2 ? "Saved method run output" : "Imported method output",
        requires_presidio: false,
        uses_llm: true,
        supports_verify_override: false,
        prompt_templates: toPromptTemplatesFromSnapshot(meta?.prompt_snapshot),
        available: true,
        unavailable_reason: null,
      });
      knownIds.add(methodId);
    }
    return merged;
  }, [agentMethods, doc, methodOutputIds]);

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
    const llmRunMeta = doc?.agent_outputs?.llm_run_metadata ?? {};
    for (const modelKey of Object.keys(doc?.agent_outputs?.llm_runs ?? {})) {
      const meta = llmRunMeta[modelKey];
      const modelName = meta?.model ?? modelKey;
      const runTime = formatRunTimestamp(meta?.updated_at);
      options.push({
        value: `agent.llm_run.${modelKey}` as AnnotationSource,
        label: `Agent LLM: ${modelName}${runTime ? ` • ${runTime}` : ""}`,
      });
    }
    const methodOptionsSeed =
      methodCatalog.length > 0
        ? methodCatalog
        : ([
            "default",
            "extended",
            "verified",
            "dual",
            "dual-split",
            "presidio",
            "presidio+default",
            "presidio+llm-split",
          ].map((id) => ({
            id,
            label: id,
          })) as Array<{ id: string; label: string }>);
    for (const method of methodOptionsSeed) {
      options.push({
        value: `agent.method.${method.id}` as AnnotationSource,
        label: `Method: ${method.label}`,
      });
    }
    return options;
  }, [doc, methodCatalog]);

  const llmRunOptions = useMemo(() => {
    const runs = doc?.agent_outputs?.llm_runs ?? {};
    const meta = doc?.agent_outputs?.llm_run_metadata ?? {};
    return Object.keys(runs).map((key) => {
      const runMeta = meta[key];
      const modelName = runMeta?.model ?? key;
      const runTime = formatRunTimestamp(runMeta?.updated_at);
      return {
        key,
        label: modelName,
        subtitle: runTime,
      };
    });
  }, [doc]);

  const agentChunkDiagnostics = useMemo<AgentChunkDiagnostic[]>(() => {
    if (!doc) return [];
    if (agentView === "rule") return [];
    if (agentView === "llm" && agentLlmRun !== "__latest__") {
      return doc.agent_outputs?.llm_run_metadata?.[agentLlmRun]?.chunk_diagnostics ?? [];
    }
    return doc.agent_run_metrics?.chunk_diagnostics ?? [];
  }, [agentLlmRun, agentView, doc]);

  const methodChunkDiagnostics = useMemo<AgentChunkDiagnostic[]>(
    () => resolveMethodChunkDiagnostics(doc, methodView),
    [doc, methodView],
  );

  useEffect(() => {
    if (agentLlmRun === "__latest__") return;
    const exists = llmRunOptions.some((item) => item.key === agentLlmRun);
    if (!exists) {
      setAgentLlmRun("__latest__");
    }
  }, [agentLlmRun, llmRunOptions]);

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
  }, [hypothesis, reference, sourceOptions]);

  const handleDashboardRefresh = useCallback(async () => {
    if (documents.length === 0) {
      setDashboard(null);
      return;
    }
    setDashboardLoading(true);
    try {
      const result = await getMetricsDashboard(
        reference,
        hypothesis,
        matchMode,
        labelProjection,
      );
      setDashboard(result);
    } catch (e: unknown) {
      setError(String(e));
    } finally {
      setDashboardLoading(false);
    }
  }, [documents.length, reference, hypothesis, matchMode, labelProjection]);

  useEffect(() => {
    void handleDashboardRefresh();
  }, [handleDashboardRefresh]);

  const scheduleDashboardRefresh = useCallback(
    (delayMs = 1500) => {
      if (dashboardRefreshTimer.current) {
        clearTimeout(dashboardRefreshTimer.current);
      }
      dashboardRefreshTimer.current = setTimeout(() => {
        dashboardRefreshTimer.current = null;
        void handleDashboardRefresh();
      }, delayMs);
    },
    [handleDashboardRefresh],
  );

  useEffect(() => () => {
    if (saveTimer.current) clearTimeout(saveTimer.current);
    if (savedTimer.current) clearTimeout(savedTimer.current);
    if (dashboardRefreshTimer.current) clearTimeout(dashboardRefreshTimer.current);
  }, []);

  // 4.1: Save status in debounced auto-save
  const handleManualChange = useCallback(
    (spans: CanonicalSpan[]) => {
      if (!doc) return;
      setDoc({ ...doc, manual_annotations: spans });
      setDocuments((prev) =>
        prev.map((item) =>
          item.id === doc.id
            ? {
                ...item,
                status: spans.length > 0 ? "in_progress" : "pending",
              }
            : item,
        ),
      );

      // Debounced auto-save
      if (saveTimer.current) clearTimeout(saveTimer.current);
      if (savedTimer.current) clearTimeout(savedTimer.current);
      if (dashboardRefreshTimer.current) clearTimeout(dashboardRefreshTimer.current);
      setSaveStatus("saving");
      const docId = doc.id;
      saveTimer.current = setTimeout(() => {
        updateManualAnnotations(docId, spans)
          .then((savedDoc) => {
            setDoc((current) => (current?.id === savedDoc.id ? savedDoc : current));
            setSaveStatus("saved");
            savedTimer.current = setTimeout(() => setSaveStatus("idle"), 2000);
            scheduleDashboardRefresh();
          })
          .catch((e: unknown) => {
            setSaveStatus("idle");
            setError(String(e));
          });
      }, 1000);
    },
    [doc, scheduleDashboardRefresh],
  );

  const handleRunAgent = useCallback(
    async (config: AgentConfig) => {
      if (!doc) return;
      const fileLabel = doc.filename || doc.id;
      setAgentRunning(true);
      try {
        const updated = await runAgent(doc.id, config);
        setDoc(updated);
        const warnings = updated.agent_run_warnings ?? [];
        setAgentChunked(hasChunkWarnings(warnings));
        const nonChunkWarning = nonChunkWarningMessage(warnings);
        setWarning(nonChunkWarning);
        const spanCount =
          config.mode === "rule"
            ? updated.agent_outputs?.rule?.length ?? 0
            : updated.agent_outputs?.llm?.length ?? 0;
        const modelLabel = config.model?.trim();
        if (spanCount === 0) {
          pushRunToast(
            "info",
            modelLabel
              ? `Agent run completed for ${fileLabel} (${modelLabel}) with 0 spans.${nonChunkWarning ? ` ${nonChunkWarning}` : ""}`
              : `Agent run completed for ${fileLabel} with 0 spans.${nonChunkWarning ? ` ${nonChunkWarning}` : ""}`,
          );
        } else {
          pushRunToast(
            "success",
            modelLabel
              ? `Agent run completed for ${fileLabel} (${modelLabel}) with ${spanCount} span(s).`
              : `Agent run completed for ${fileLabel} with ${spanCount} span(s).`,
          );
        }
        void handleDashboardRefresh();
      } catch (e: unknown) {
        setError(String(e));
        pushRunToast("error", `Agent run failed for ${fileLabel}.`);
      } finally {
        setAgentRunning(false);
      }
    },
    [doc, handleDashboardRefresh, pushRunToast],
  );

  const handleRunMethod = useCallback(
    async (config: AgentConfig) => {
      if (!doc) return;
      const fileLabel = doc.filename || doc.id;
      setMethodRunning(true);
      try {
        const updated = await runAgent(doc.id, config);
        setDoc(updated);
        const warnings = updated.agent_run_warnings ?? [];
        setMethodChunked(hasChunkWarnings(warnings));
        const nonChunkWarning = nonChunkWarningMessage(warnings);
        setWarning(nonChunkWarning);
        const methodKey = config.method_id?.trim() ?? "";
        const spanCount = methodKey
          ? updated.agent_outputs?.methods?.[methodKey]?.length ?? 0
          : 0;
        const methodLabel = config.method_id?.trim();
        const modelLabel = config.model?.trim();
        const scope = [methodLabel, modelLabel].filter(Boolean).join(" @ ");
        if (spanCount === 0) {
          pushRunToast(
            "info",
            scope
              ? `Method run completed for ${fileLabel} (${scope}) with 0 spans.${nonChunkWarning ? ` ${nonChunkWarning}` : ""}`
              : `Method run completed for ${fileLabel} with 0 spans.${nonChunkWarning ? ` ${nonChunkWarning}` : ""}`,
          );
        } else {
          pushRunToast(
            "success",
            scope
              ? `Method run completed for ${fileLabel} (${scope}) with ${spanCount} span(s).`
              : `Method run completed for ${fileLabel} with ${spanCount} span(s).`,
          );
        }
        void handleDashboardRefresh();
      } catch (e: unknown) {
        setError(String(e));
        pushRunToast("error", `Method run failed for ${fileLabel}.`);
      } finally {
        setMethodRunning(false);
      }
    },
    [doc, handleDashboardRefresh, pushRunToast],
  );

  const handleMetricsRefresh = useCallback(async () => {
    if (!doc) return;
    setMetricsLoading(true);
    try {
      const result = await getMetrics(
        doc.id,
        reference,
        hypothesis,
        matchMode,
        labelProjection,
      );
      setMetrics(result);
    } catch (e: unknown) {
      setError(String(e));
    } finally {
      setMetricsLoading(false);
    }
  }, [doc, reference, hypothesis, matchMode, labelProjection]);

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
        if (source.startsWith("agent.llm_run.")) {
          const runKey = source.slice("agent.llm_run.".length);
          return doc.agent_outputs?.llm_runs?.[runKey] ?? [];
        }
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
    if (agentView === "llm") {
      if (agentLlmRun !== "__latest__") {
        return doc.agent_outputs?.llm_runs?.[agentLlmRun] ?? [];
      }
      return doc.agent_outputs?.llm ?? [];
    }
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
      if (agentView === "llm") {
        return agentLlmRun === "__latest__"
          ? "agent.llm"
          : (`agent.llm_run.${agentLlmRun}` as AnnotationSource);
      }
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
        exportSourceOptions={sourceOptions}
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
        <div className="main-tabbar">
          <button
            type="button"
            className={mainTab === "workspace" ? "active" : ""}
            onClick={() => setMainTab("workspace")}
          >
            Workspace
          </button>
          <button
            type="button"
            className={mainTab === "prompt_lab" ? "active" : ""}
            onClick={() => setMainTab("prompt_lab")}
          >
            Prompt Lab
          </button>
          <button
            type="button"
            className={mainTab === "methods_lab" ? "active" : ""}
            onClick={() => setMainTab("methods_lab")}
          >
            Methods Lab
          </button>
          <button
            type="button"
            className={mainTab === "dashboard" ? "active" : ""}
            onClick={() => setMainTab("dashboard")}
          >
            Dashboard
          </button>
        </div>
        {mainTab === "prompt_lab" ? (
          <PromptLabTab
            documents={documents}
            selectedDocumentId={selectedId}
            onSelectDocument={setSelectedId}
          />
        ) : mainTab === "methods_lab" ? (
          <MethodsLabTab
            documents={documents}
            selectedDocumentId={selectedId}
            onSelectDocument={setSelectedId}
          />
        ) : mainTab === "dashboard" ? (
          <section className="dashboard-tab">
            <div className="dashboard-tab-toolbar">
              <div className="dashboard-tab-controls">
                <label>
                  Reference
                  <select value={reference} onChange={(e) => setReference(e.target.value as AnnotationSource)}>
                    {sourceOptions.map((option) => (
                      <option key={`dash-ref-${option.value}`} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Hypothesis
                  <select
                    value={hypothesis}
                    onChange={(e) => setHypothesis(e.target.value as AnnotationSource)}
                  >
                    {sourceOptions.map((option) => (
                      <option key={`dash-hyp-${option.value}`} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  Match
                  <select
                    value={matchMode}
                    onChange={(e) => setMatchMode(e.target.value as MatchMode)}
                  >
                    <option value="exact">Exact</option>
                    <option value="boundary">Trim Space/Punct</option>
                    <option value="overlap">Overlap</option>
                  </select>
                </label>
                <label>
                  Label Compare
                  <select
                    value={labelProjection}
                    onChange={(e) => setLabelProjection(e.target.value as LabelProjection)}
                  >
                    <option value="native">Native</option>
                    <option value="coarse_simple">Coarse (advanced→simple)</option>
                  </select>
                </label>
              </div>
              <button
                type="button"
                className="dashboard-tab-refresh"
                onClick={() => void handleDashboardRefresh()}
              >
                Refresh Dashboard
              </button>
            </div>
            <DashboardPanel
              dashboard={dashboard}
              loading={dashboardLoading}
              onRefresh={handleDashboardRefresh}
              selectedId={selectedId}
              onSelectDocument={(docId) => {
                setSelectedId(docId);
                setMainTab("workspace");
              }}
            />
            {!dashboard && !dashboardLoading && (
              <div className="dashboard-tab-empty">
                No dashboard metrics yet. Click <strong>Refresh Dashboard</strong>.
              </div>
            )}
          </section>
        ) : (
          <>
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
                  labelProjection={labelProjection}
                  onLabelProjectionChange={setLabelProjection}
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
                            labels={PII_LABELS}
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
                            runProgress={agentRunning ? agentRunProgress : null}
                            processedWithChunking={
                              agentChunked || getChunkDiagnosticsCount(agentChunkDiagnostics) > 0
                            }
                            chunkDiagnostics={agentChunkDiagnostics}
                            activeOutput={agentView}
                            onActiveOutputChange={setAgentView}
                            llmRunOptions={llmRunOptions}
                            activeLlmRunKey={agentLlmRun}
                            onActiveLlmRunKeyChange={setAgentLlmRun}
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
                            runProgress={methodRunning ? agentRunProgress : null}
                            methods={methodCatalog}
                            processedWithChunking={
                              methodChunked || getChunkDiagnosticsCount(methodChunkDiagnostics) > 0
                            }
                            chunkDiagnostics={methodChunkDiagnostics}
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
                  sourceOptions={sourceOptions}
                  metrics={metrics}
                  loading={metricsLoading}
                  onRefresh={handleMetricsRefresh}
                  onReferenceChange={setReference}
                  onHypothesisChange={setHypothesis}
                  onMatchModeChange={setMatchMode}
                />
              </>
            )}
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
      {runToasts.length > 0 && (
        <div className="run-toast-stack" aria-live="polite" aria-atomic="false">
          {runToasts.map((toast) => (
            <div
              key={toast.id}
              className={`run-toast ${toast.kind}`}
              onClick={() => dismissRunToast(toast.id)}
              role="status"
            >
              {toast.message}
            </div>
          ))}
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
