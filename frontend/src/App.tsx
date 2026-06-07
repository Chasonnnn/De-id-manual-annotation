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
  DocumentSummary,
  FolderDetail,
  FolderSummary,
  GroundTruthExportScope,
  ImportConflictPolicy,
  MatchMode,
  MetricsCandidate,
  MetricsCandidateSource,
  MetricsCompareResult,
  MethodView,
  MetricsResult,
  PaneInstance,
  PaneType,
  ReadinessHealth,
  WorkspaceState,
} from "./types";
import {
  createFolder,
  createFolderSample,
  deleteFolder,
  deleteFolderDocument,
  deleteDocument,
  exportGroundTruth,
  exportSession,
  getFolder,
  getAgentProgress,
  getAgentMethods,
  getDocument,
  getMethodsLabDocResult,
  getMetricsCandidates,
  getMetrics,
  getWorkspace,
  compareMetrics,
  ingestSessionFile,
  mirrorPreToManual,
  pruneEmptyFolderDocs,
  runAgent,
  updateManualAnnotations,
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
import MetricsCompareDashboard from "./components/MetricsCompareDashboard";
import { computeDiff } from "./components/DiffOverlay";
import PromptLabTab from "./components/PromptLabTab";
import MethodsLabTab from "./components/MethodsLabTab";
import ConfirmDialog from "./components/ConfirmDialog";
import { ingestFiles } from "./ingestFiles";
import { getPromptPresetLabelFromSnapshot } from "./agentPromptPresets";

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

interface WorkspaceRefreshResult {
  documents: DocumentSummary[];
  allDocIds: Set<string>;
  firstAvailableId: string | null;
}

interface PendingConfirm {
  title: string;
  message: string;
  confirmLabel?: string;
  destructive?: boolean;
  onConfirm: () => void;
}

const CANONICAL_PANE_ORDER: PaneType[] = ["raw", "pre", "manual", "agent", "methods"];

function makePaneInstance(type: PaneType, sourceRef?: MetricsCandidateSource): PaneInstance {
  const singletonId = type === "methods" ? `methods-${Date.now()}-${Math.random().toString(36).slice(2, 8)}` : type;
  return {
    id: singletonId,
    type,
    title:
      type === "raw"
        ? "Raw"
        : type === "pre"
          ? "Pre-annotations"
          : type === "manual"
            ? "Manual"
            : type === "agent"
              ? "Agent"
              : "Methods",
    source_ref: sourceRef,
  };
}

function normalizePaneInstances(panes: PaneInstance[]): PaneInstance[] {
  const seenSingletons = new Set<PaneType>();
  return [...panes]
    .filter((pane) => {
      if (pane.type === "methods") return true;
      if (seenSingletons.has(pane.type)) return false;
      seenSingletons.add(pane.type);
      return true;
    })
    .sort((a, b) => {
      const order = CANONICAL_PANE_ORDER.indexOf(a.type) - CANONICAL_PANE_ORDER.indexOf(b.type);
      return order === 0 ? a.id.localeCompare(b.id) : order;
    });
}

function loadInitialPaneInstances(): PaneInstance[] {
  try {
    const raw = window.localStorage.getItem("annotation_tool_pane_instances_v1");
    if (raw) {
      const parsed = JSON.parse(raw) as PaneInstance[];
      if (Array.isArray(parsed)) {
        const normalized = normalizePaneInstances(
          parsed.filter((item) => item && typeof item.id === "string" && typeof item.type === "string"),
        );
        if (normalized.length > 0) return normalized;
      }
    }
  } catch {
    // localStorage unavailable or stale payload
  }
  return [makePaneInstance("raw"), makePaneInstance("pre")];
}

function sourceToMethodView(source: MetricsCandidateSource | undefined): MethodView | null {
  if (!source || !String(source).startsWith("agent.method.")) return null;
  return String(source).slice("agent.method.".length);
}

function parseMethodsLabSource(
  source: MetricsCandidateSource | undefined,
): { runId: string; cellId: string } | null {
  if (!source || !String(source).startsWith("methods_lab.")) return null;
  const rest = String(source).slice("methods_lab.".length);
  const [runId, ...cellParts] = rest.split(".");
  const cellId = cellParts.join(".");
  return runId && cellId ? { runId, cellId } : null;
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

function getSortableTimestamp(value: string | null | undefined): number {
  if (!value) return Number.NEGATIVE_INFINITY;
  const date = new Date(value);
  const timestamp = date.getTime();
  return Number.isFinite(timestamp) ? timestamp : Number.NEGATIVE_INFINITY;
}

function getChunkDiagnosticsCount(
  diagnostics: AgentChunkDiagnostic[] | null | undefined,
): number {
  return diagnostics?.length ?? 0;
}

function buildWorkspaceRefreshResult(
  documents: DocumentSummary[],
  folderDetailsById: Record<string, FolderDetail>,
): WorkspaceRefreshResult {
  const folderDocIds = Object.values(folderDetailsById).flatMap((detail) =>
    detail.documents.map((item) => item.id),
  );
  return {
    documents,
    allDocIds: new Set([
      ...documents.map((item) => item.id),
      ...folderDocIds,
    ]),
    firstAvailableId: documents[0]?.id ?? folderDocIds[0] ?? null,
  };
}

function getReadinessWarnings(health: ReadinessHealth | null): string[] {
  if (!health) return [];
  return [
    ...health.config_warnings,
    ...health.dependency_warnings,
    ...health.method_availability_warnings.map(
      (item) => `${item.label}: ${item.reason}`,
    ),
  ];
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

function getLatestReadyMethodView(doc: CanonicalDocument | null): MethodView | null {
  if (!doc) return null;
  const outputs = doc.agent_outputs?.methods ?? {};
  const metadata = doc.agent_outputs?.method_run_metadata ?? {};
  const outputIds = Object.keys(outputs);
  if (outputIds.length === 0) return null;

  let latestId: string | null = null;
  let latestTimestamp = Number.NEGATIVE_INFINITY;
  for (const outputId of outputIds) {
    const timestamp = getSortableTimestamp(metadata[outputId]?.updated_at);
    if (latestId === null || timestamp >= latestTimestamp) {
      latestId = outputId;
      latestTimestamp = timestamp;
    }
  }
  return latestId;
}

function getLatestReadyAgentSelection(
  doc: CanonicalDocument | null,
): { view: AgentView; llmRunKey: string } | null {
  if (!doc) return null;
  const llmRuns = doc.agent_outputs?.llm_runs ?? {};
  const llmMetadata = doc.agent_outputs?.llm_run_metadata ?? {};
  const llmRunKeys = Object.keys(llmRuns);
  if (llmRunKeys.length > 0) {
    let latestKey: string | null = null;
    let latestTimestamp = Number.NEGATIVE_INFINITY;
    for (const runKey of llmRunKeys) {
      const timestamp = getSortableTimestamp(llmMetadata[runKey]?.updated_at);
      if (latestKey === null || timestamp >= latestTimestamp) {
        latestKey = runKey;
        latestTimestamp = timestamp;
      }
    }
    if (latestKey) {
      return { view: "llm", llmRunKey: latestKey };
    }
  }
  if ((doc.agent_outputs?.llm?.length ?? 0) > 0) {
    return { view: "llm", llmRunKey: "__latest__" };
  }
  if ((doc.agent_outputs?.rule?.length ?? 0) > 0) {
    return { view: "rule", llmRunKey: "__latest__" };
  }
  if ((doc.agent_annotations?.length ?? 0) > 0) {
    return { view: "combined", llmRunKey: "__latest__" };
  }
  return null;
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
function useAppContentController() {
  const [mainTab, setMainTab] = useState<
    "workspace" | "prompt_lab" | "methods_lab" | "dashboard"
  >(
    "workspace",
  );
  const [documents, setDocuments] = useState<DocumentSummary[]>([]);
  const [folders, setFolders] = useState<FolderSummary[]>([]);
  const [folderDetailsById, setFolderDetailsById] = useState<Record<string, FolderDetail>>({});
  const [folderDetailLoadingById, setFolderDetailLoadingById] = useState<Record<string, boolean>>({});
  const [health, setHealth] = useState<ReadinessHealth | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [doc, setDoc] = useState<CanonicalDocument | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [warning, setWarning] = useState<string | null>(null);
  const [runToasts, setRunToasts] = useState<RunToast[]>([]);
  const [pendingConfirm, setPendingConfirm] = useState<PendingConfirm | null>(null);
  const [agentRunProgress, setAgentRunProgress] = useState<AgentRunProgress | null>(null);
  const [ingesting, setIngesting] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);
  const [folderBusyId, setFolderBusyId] = useState<string | null>(null);
  const [exporting, setExporting] = useState(false);
  const [mirroringPreToManual, setMirroringPreToManual] = useState(false);

  const [paneInstances, setPaneInstances] = useState<PaneInstance[]>(loadInitialPaneInstances);
  const [diffMode, setDiffMode] = useState(false);
  const [reference, setReference] = useState<AnnotationSource>("pre");
  const [hypothesis, setHypothesis] = useState<AnnotationSource>("manual");
  const [matchMode, setMatchMode] = useState<MatchMode>("overlap");
  const [agentView, setAgentView] = useState<AgentView>("combined");
  const [agentLlmRun, setAgentLlmRun] = useState<string>("__latest__");
  const [agentMethods, setAgentMethods] = useState<AgentMethodOption[]>([]);
  const [methodView, setMethodView] = useState<MethodView>("default");
  const [methodsLabPaneSpans, setMethodsLabPaneSpans] = useState<Record<string, CanonicalSpan[]>>({});
  const [methodsLabPaneLoading, setMethodsLabPaneLoading] = useState<Record<string, boolean>>({});

  const [agentRunning, setAgentRunning] = useState(false);
  const [methodRunning, setMethodRunning] = useState(false);
  const [agentChunked, setAgentChunked] = useState(false);
  const [methodChunked, setMethodChunked] = useState(false);
  const [metrics, setMetrics] = useState<MetricsResult | null>(null);
  const [metricsLoading, setMetricsLoading] = useState(false);
  const [metricsCandidates, setMetricsCandidates] = useState<MetricsCandidate[]>([]);
  const [compareReference, setCompareReference] = useState<MetricsCandidateSource>("manual");
  const [compareHypotheses, setCompareHypotheses] = useState<MetricsCandidateSource[]>([]);
  const [compareResult, setCompareResult] = useState<MetricsCompareResult | null>(null);
  const [compareLoading, setCompareLoading] = useState(false);

  const [saveStatus, setSaveStatus] = useState<SaveStatus>("idle"); // 4.1
  const saveTimer = useRef<ReturnType<typeof setTimeout>>(null);
  const savedTimer = useRef<ReturnType<typeof setTimeout>>(null);
  const compareRefreshTimer = useRef<ReturnType<typeof setTimeout>>(null);
  const agentViewRef = useRef<AgentView>("combined");
  const agentLlmRunRef = useRef<string>("__latest__");
  const methodViewRef = useRef<MethodView>("default");
  const folderDetailsRef = useRef<Record<string, FolderDetail>>({});
  const folderDetailLoadingRef = useRef<Set<string>>(new Set());

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

  const applyWorkspaceState = useCallback(
    (
      workspace: WorkspaceState,
      options: { preserveFolderDetails?: boolean } = {},
    ): WorkspaceRefreshResult => {
      const preserveFolderDetails = options.preserveFolderDetails ?? true;
      const activeFolderIds = new Set(workspace.folders.map((folder) => folder.id));
      const nextDetails: Record<string, FolderDetail> = {};

      if (preserveFolderDetails) {
        for (const [folderId, detail] of Object.entries(folderDetailsRef.current)) {
          if (activeFolderIds.has(folderId)) {
            nextDetails[folderId] = detail;
          }
        }
      }
      for (const [folderId, detail] of Object.entries(workspace.folder_details)) {
        if (activeFolderIds.has(folderId)) {
          nextDetails[folderId] = detail;
        }
      }

      folderDetailsRef.current = nextDetails;
      setDocuments(workspace.documents);
      setFolders(workspace.folders);
      setFolderDetailsById(nextDetails);
      setHealth(workspace.health);
      return buildWorkspaceRefreshResult(workspace.documents, nextDetails);
    },
    [],
  );

  const refreshDocuments = useCallback(
    async (
      options: { preserveFolderDetails?: boolean } = {},
    ): Promise<WorkspaceRefreshResult> => {
      const workspace = await getWorkspace();
      return applyWorkspaceState(workspace, options);
    },
    [applyWorkspaceState],
  );

  const ensureFolderDetail = useCallback(
    async (folderId: string) => {
      if (folderDetailsRef.current[folderId] || folderDetailLoadingRef.current.has(folderId)) {
        return;
      }
      folderDetailLoadingRef.current.add(folderId);
      setFolderDetailLoadingById((prev) => ({ ...prev, [folderId]: true }));
      try {
        const detail = await getFolder(folderId);
        const nextDetails = {
          ...folderDetailsRef.current,
          [detail.id]: detail,
        };
        folderDetailsRef.current = nextDetails;
        setFolderDetailsById(nextDetails);
      } catch (e: unknown) {
        setError(String(e));
      } finally {
        folderDetailLoadingRef.current.delete(folderId);
        setFolderDetailLoadingById((prev) => {
          if (!prev[folderId]) return prev;
          const next = { ...prev };
          delete next[folderId];
          return next;
        });
      }
    },
    [],
  );

  // Load document list
  useEffect(() => {
    refreshDocuments().catch((e: unknown) => setError(String(e)));
  }, [refreshDocuments]);

  useEffect(() => {
    getAgentMethods()
      .then((methods) => setAgentMethods(methods))
      .catch((e: unknown) => setError(String(e)));
  }, []);

  const clearSelectedDocumentState = useCallback(() => {
    setDoc(null);
    setAgentChunked(false);
    setMethodChunked(false);
    setAgentRunProgress(null);
  }, []);

  const applyLoadedDocumentState = useCallback((nextDoc: CanonicalDocument) => {
    setDoc(nextDoc);
    setMetrics(null);
    setSaveStatus("idle");
    setWarning(null);
    setAgentChunked(false);
    setMethodChunked(false);
  }, []);

  const loadSelectedDocument = useCallback(async (docId: string | null) => {
    if (!docId) {
      clearSelectedDocumentState();
      return;
    }
    setLoading(true);
    try {
      const nextDoc = await getDocument(docId);
      applyLoadedDocumentState(nextDoc);
    } catch (e: unknown) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }, [applyLoadedDocumentState, clearSelectedDocumentState]);

  // Load selected document
  useEffect(() => {
    void loadSelectedDocument(selectedId);
  }, [loadSelectedDocument, selectedId]);

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

  const handleIngestFiles = useCallback(
    async (files: File[], conflictPolicy: ImportConflictPolicy) => {
      if (files.length === 0) return;
      setIngesting(true);
      setError(null);
      setWarning(null);
      try {
        const result = await ingestFiles(files, (file) => ingestSessionFile(file, conflictPolicy));
        if (result.created_ids.length > 0 || result.imported_file_count > 0) {
          const refreshed = await refreshDocuments({ preserveFolderDetails: false });
          const firstCreatedId = result.created_ids[0] ?? null;
          const selectedStillExists =
            selectedId !== null && refreshed.allDocIds.has(selectedId);
          if (!selectedStillExists) {
            setSelectedId(firstCreatedId ?? refreshed.documents[0]?.id ?? null);
          }
        }

        const warnings: string[] = [];
        if (result.uploaded_count > 0) {
          warnings.push(
            `Added ${result.uploaded_count} raw document(s) from ${result.uploaded_file_count} file(s).`,
          );
        }
        if (result.imported_count > 0) {
          warnings.push(
            `Imported ${result.imported_count} document(s) from ${result.imported_file_count} file(s).`,
          );
        }
        if ((result.replaced_count ?? 0) > 0) {
          warnings.push(`Replaced ${result.replaced_count} conflicting document(s).`);
        }
        if ((result.kept_current_count ?? 0) > 0) {
          warnings.push(`Kept ${result.kept_current_count} current document(s) on conflict.`);
        }
        if ((result.added_as_new_count ?? 0) > 0) {
          warnings.push(`Added ${result.added_as_new_count} conflicting document(s) as new.`);
        }
        if ((result.imported_prompt_lab_runs ?? 0) > 0) {
          warnings.push(`Imported ${result.imported_prompt_lab_runs} Prompt Lab run(s).`);
        }
        if ((result.imported_methods_lab_runs ?? 0) > 0) {
          warnings.push(`Imported ${result.imported_methods_lab_runs} Methods Lab run(s).`);
        }
        if (result.skipped_count > 0) {
          warnings.push(`Skipped ${result.skipped_count} item(s).`);
        }
        const ingestWarnings = result.warnings ?? [];
        if (ingestWarnings.length > 0) {
          warnings.push(`Ingest warnings: ${ingestWarnings.join(" | ")}`);
        }
        if (warnings.length > 0) {
          setWarning(warnings.join(" "));
        }
        if (result.failed_file_count > 0) {
          const failureSummary = result.failed_files
            .map((item) => `${item.file_name}: ${item.message}`)
            .join(" | ");
          setError(`Failed to process ${result.failed_file_count} file(s). ${failureSummary}`);
        }
      } catch (e: unknown) {
        setError(String(e));
      } finally {
        setIngesting(false);
      }
    },
    [refreshDocuments, selectedId],
  );

  const doDeleteDocument = useCallback(
    async (docId: string) => {
      setDeletingId(docId);
      try {
        await deleteDocument(docId);
        const refreshed = await refreshDocuments({ preserveFolderDetails: false });
        if (selectedId === docId || !refreshed.allDocIds.has(selectedId ?? "")) {
          const nextId = refreshed.firstAvailableId;
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

  const doDeleteFolderDocument = useCallback(
    async (folderId: string, docId: string) => {
      setDeletingId(docId);
      try {
        await deleteFolderDocument(folderId, docId);
        const refreshed = await refreshDocuments({ preserveFolderDetails: false });
        if (selectedId === docId || !refreshed.allDocIds.has(selectedId ?? "")) {
          const nextId = refreshed.firstAvailableId;
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

  const handleDeleteDocument = useCallback(
    (docId: string) => {
      setPendingConfirm({
        title: "Delete Document",
        message: "Delete this document and its annotations? This cannot be undone.",
        confirmLabel: "Delete",
        destructive: true,
        onConfirm: () => {
          setPendingConfirm(null);
          void doDeleteDocument(docId);
        },
      });
    },
    [doDeleteDocument],
  );

  const handleDeleteFolderDocument = useCallback(
    (folderId: string, docId: string) => {
      setPendingConfirm({
        title: "Delete Transcript",
        message:
          "Delete this transcript and its annotations? This removes it from any folders that include it. This cannot be undone.",
        confirmLabel: "Delete Transcript",
        destructive: true,
        onConfirm: () => {
          setPendingConfirm(null);
          void doDeleteFolderDocument(folderId, docId);
        },
      });
    },
    [doDeleteFolderDocument],
  );

  const handleCreateFolderSample = useCallback(
    async (folderId: string, count: number) => {
      setFolderBusyId(folderId);
      try {
        await createFolderSample(folderId, count);
        await refreshDocuments({ preserveFolderDetails: false });
      } catch (e: unknown) {
        setError(String(e));
      } finally {
        setFolderBusyId(null);
      }
    },
    [refreshDocuments],
  );

  const handleCreateFolder = useCallback(
    async (name: string, parentFolderId: string | null) => {
      setFolderBusyId(parentFolderId ?? "__new__");
      try {
        await createFolder(name, parentFolderId);
        await refreshDocuments({ preserveFolderDetails: false });
      } catch (e: unknown) {
        setError(String(e));
      } finally {
        setFolderBusyId(null);
      }
    },
    [refreshDocuments],
  );

  const doDeleteFolder = useCallback(
    async (folderId: string) => {
      setFolderBusyId(folderId);
      try {
        await deleteFolder(folderId);
        const refreshed = await refreshDocuments({ preserveFolderDetails: false });
        if (selectedId && !refreshed.allDocIds.has(selectedId)) {
          const nextId = refreshed.firstAvailableId;
          setSelectedId(nextId);
          if (!nextId) {
            setDoc(null);
            setMetrics(null);
          }
        }
      } catch (e: unknown) {
        setError(String(e));
      } finally {
        setFolderBusyId(null);
      }
    },
    [refreshDocuments, selectedId],
  );

  const handleDeleteFolder = useCallback(
    (folderId: string) => {
      setPendingConfirm({
        title: "Delete Folder",
        message: "Delete this folder and its managed transcript set? This cannot be undone.",
        confirmLabel: "Delete Folder",
        destructive: true,
        onConfirm: () => {
          setPendingConfirm(null);
          void doDeleteFolder(folderId);
        },
      });
    },
    [doDeleteFolder],
  );

  const doPruneFolder = useCallback(
    async (folderId: string) => {
      setFolderBusyId(folderId);
      try {
        const result = await pruneEmptyFolderDocs(folderId);
        const refreshed = await refreshDocuments({ preserveFolderDetails: false });
        if (selectedId && !refreshed.allDocIds.has(selectedId)) {
          const nextId = refreshed.firstAvailableId;
          setSelectedId(nextId);
          if (!nextId) {
            setDoc(null);
            setMetrics(null);
          }
        }
        setWarning(
          result.removed_count > 0
            ? `Removed ${result.removed_count} unannotated file(s) from the folder.`
            : "No unannotated files were found in the folder.",
        );
      } catch (e: unknown) {
        setError(String(e));
      } finally {
        setFolderBusyId(null);
      }
    },
    [refreshDocuments, selectedId],
  );

  const handlePruneFolder = useCallback(
    (folderId: string) => {
      setPendingConfirm({
        title: "Prune Empty Documents",
        message:
          "Remove direct files in this folder that have neither pre-annotations nor manual annotations?",
        confirmLabel: "Prune",
        destructive: true,
        onConfirm: () => {
          setPendingConfirm(null);
          void doPruneFolder(folderId);
        },
      });
    },
    [doPruneFolder],
  );

  const doMirrorPreToManual = useCallback(
    async (scope: GroundTruthExportScope) => {
      setMirroringPreToManual(true);
      try {
        const result = await mirrorPreToManual(scope);
        const refreshed = await refreshDocuments({ preserveFolderDetails: false });
        if (selectedId && refreshed.allDocIds.has(selectedId)) {
          await loadSelectedDocument(selectedId);
        }
        setWarning(
          `Mirrored pre-annotations into manual for ${result.processed_count} document(s): ${result.copied_count} copied, ${result.cleared_count} empty.`,
        );
      } catch (e: unknown) {
        setError(String(e));
      } finally {
        setMirroringPreToManual(false);
      }
    },
    [loadSelectedDocument, refreshDocuments, selectedId],
  );

  const handleWorkspaceChanged = useCallback(async () => {
    const refreshed = await refreshDocuments({ preserveFolderDetails: false });
    if (selectedId && refreshed.allDocIds.has(selectedId)) {
      await loadSelectedDocument(selectedId);
    }
  }, [loadSelectedDocument, refreshDocuments, selectedId]);

  const handleMirrorPreToManual = useCallback(
    (scope: GroundTruthExportScope) => {
      const scopeLabel =
        scope.kind === "folder"
          ? `the folder "${folders.find((folder) => folder.id === scope.folderId)?.name ?? scope.folderId}"`
          : "all documents in the current session";
      setPendingConfirm({
        title: "Mirror Pre to Manual",
        message:
          `Copy pre-annotations into manual annotations for ${scopeLabel}? ` +
          "Existing manual annotations in that scope will be replaced.",
        confirmLabel: "Mirror",
        destructive: false,
        onConfirm: () => {
          setPendingConfirm(null);
          void doMirrorPreToManual(scope);
        },
      });
    },
    [doMirrorPreToManual, folders],
  );

  const handleExportSession = useCallback(async (
    mode: "full" | "ground_truth",
    source: AnnotationSource,
    exportScope: GroundTruthExportScope,
  ) => {
    setExporting(true);
    try {
      const stamp = new Date().toISOString().replace(/[:.]/g, "-");
      const fullBundle = mode === "full" ? await exportSession() : null;
      const blob =
        mode === "full"
          ? new Blob([JSON.stringify(fullBundle, null, 2)], {
              type: "application/json",
            })
          : await exportGroundTruth(source, exportScope);
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
  }, []);

  useEffect(() => {
    try {
      window.localStorage.setItem(
        "annotation_tool_pane_instances_v1",
        JSON.stringify(paneInstances),
      );
    } catch {
      // localStorage unavailable
    }
  }, [paneInstances]);

  const handleTogglePane = useCallback((pane: PaneType) => {
    setPaneInstances((prev) => {
      if (pane === "methods") {
        return normalizePaneInstances([...prev, makePaneInstance("methods")]);
      }
      const exists = prev.some((item) => item.type === pane);
      if (exists) {
        return normalizePaneInstances(prev.filter((item) => item.type !== pane));
      }
      return normalizePaneInstances([...prev, makePaneInstance(pane)]);
    });
  }, []);

  const orderedPaneInstances = useMemo(
    () => normalizePaneInstances(paneInstances),
    [paneInstances],
  );
  const orderedVisiblePanes = useMemo(
    () => orderedPaneInstances.map((pane) => pane.type),
    [orderedPaneInstances],
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
      const presetLabel = getPromptPresetLabelFromSnapshot(meta?.prompt_snapshot ?? null);
      const detail = [presetLabel, runTime].filter(Boolean).join(" • ");
      options.push({
        value: `agent.llm_run.${modelKey}` as AnnotationSource,
        label: `Agent LLM: ${modelName}${detail ? ` • ${detail}` : ""}`,
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
      const presetLabel = getPromptPresetLabelFromSnapshot(runMeta?.prompt_snapshot ?? null);
      return {
        key,
        label: presetLabel ? `${modelName} • ${presetLabel}` : modelName,
        subtitle: runTime,
      };
    });
  }, [doc]);

  const agentOutputSignature = useMemo(() => {
    if (!doc) return "";
    const llmRunMeta = doc.agent_outputs?.llm_run_metadata ?? {};
    const llmRunKeys = Object.keys(doc.agent_outputs?.llm_runs ?? {}).sort();
    return JSON.stringify({
      docId: doc.id,
      llmRunKeys,
      llmRunTimestamps: llmRunKeys.map((key) => llmRunMeta[key]?.updated_at ?? null),
      llmCount: doc.agent_outputs?.llm?.length ?? 0,
      ruleCount: doc.agent_outputs?.rule?.length ?? 0,
      combinedCount: doc.agent_annotations?.length ?? 0,
    });
  }, [doc]);

  const methodOutputSignature = useMemo(() => {
    if (!doc) return "";
    const methodMeta = doc.agent_outputs?.method_run_metadata ?? {};
    const methodIds = Object.keys(doc.agent_outputs?.methods ?? {}).sort();
    return JSON.stringify({
      docId: doc.id,
      methodIds,
      methodTimestamps: methodIds.map((id) => methodMeta[id]?.updated_at ?? null),
    });
  }, [doc]);

  const readinessWarnings = useMemo(() => getReadinessWarnings(health), [health]);

  useEffect(() => {
    agentViewRef.current = agentView;
  }, [agentView]);

  useEffect(() => {
    agentLlmRunRef.current = agentLlmRun;
  }, [agentLlmRun]);

  useEffect(() => {
    methodViewRef.current = methodView;
  }, [methodView]);

  const agentChunkDiagnostics = useMemo<AgentChunkDiagnostic[]>(() => {
    if (!doc) return [];
    if (agentView === "rule") return [];
    if (agentView === "llm" && agentLlmRun !== "__latest__") {
      return doc.agent_outputs?.llm_run_metadata?.[agentLlmRun]?.chunk_diagnostics ?? [];
    }
    return doc.agent_run_metrics?.chunk_diagnostics ?? [];
  }, [agentLlmRun, agentView, doc]);

  useEffect(() => {
    const activeDocId = selectedId;
    if (!activeDocId) {
      setMethodsLabPaneSpans({});
      setMethodsLabPaneLoading({});
      return;
    }
    for (const pane of orderedPaneInstances) {
      if (pane.type !== "methods") continue;
      const parsed = parseMethodsLabSource(pane.source_ref);
      if (!parsed) continue;
      const cacheKey = `${pane.id}:${activeDocId}:${pane.source_ref}`;
      setMethodsLabPaneLoading((prev) => ({ ...prev, [pane.id]: true }));
      getMethodsLabDocResult(parsed.runId, parsed.cellId, activeDocId)
        .then((detail) => {
          setMethodsLabPaneSpans((prev) => ({
            ...prev,
            [cacheKey]: detail.hypothesis_spans,
          }));
        })
        .catch((err: unknown) => {
          setError(String(err));
          setMethodsLabPaneSpans((prev) => ({ ...prev, [cacheKey]: [] }));
        })
        .finally(() => {
          setMethodsLabPaneLoading((prev) => {
            const next = { ...prev };
            delete next[pane.id];
            return next;
          });
        });
    }
  }, [orderedPaneInstances, selectedId]);

  useEffect(() => {
    if (agentLlmRun === "__latest__") return;
    const exists = llmRunOptions.some((item) => item.key === agentLlmRun);
    if (!exists) {
      setAgentLlmRun("__latest__");
    }
  }, [agentLlmRun, llmRunOptions]);

  useEffect(() => {
    const nextSelection = getLatestReadyAgentSelection(doc);
    if (!nextSelection) return;
    if (nextSelection.view !== agentViewRef.current) {
      setAgentView(nextSelection.view);
    }
    if (nextSelection.llmRunKey !== agentLlmRunRef.current) {
      setAgentLlmRun(nextSelection.llmRunKey);
    }
  }, [agentOutputSignature, doc]);

  useEffect(() => {
    const latestMethodView = getLatestReadyMethodView(doc);
    if (!latestMethodView || latestMethodView === methodViewRef.current) return;
    setMethodView(latestMethodView);
  }, [doc, methodOutputSignature]);

  const refreshMetricCandidates = useCallback(async () => {
    const candidates = await getMetricsCandidates();
    setMetricsCandidates(candidates);
    const nextReference = candidates.some((candidate) => candidate.source === compareReference)
      ? compareReference
      : candidates.find((candidate) => candidate.source === "manual")?.source ??
        candidates[0]?.source ??
        "manual";
    if (nextReference !== compareReference) {
      setCompareReference(nextReference);
    }
    setCompareHypotheses((prev) => {
      const available = new Set(candidates.map((candidate) => candidate.source));
      const retained = prev.filter((source) => available.has(source) && source !== nextReference);
      if (retained.length > 0) return retained;
      return candidates
        .filter(
          (candidate) =>
            candidate.source !== nextReference &&
            (candidate.kind === "method_run" || candidate.kind === "methods_lab_cell"),
        )
        .map((candidate) => candidate.source);
    });
    return candidates;
  }, [compareReference]);

  useEffect(() => {
    if (mainTab !== "dashboard") return;
    refreshMetricCandidates().catch((err: unknown) => setError(String(err)));
  }, [mainTab, refreshMetricCandidates]);

  const handleCompareRefresh = useCallback(async () => {
    setCompareLoading(true);
    try {
      const candidates = metricsCandidates.length > 0 ? metricsCandidates : await refreshMetricCandidates();
      const available = new Set(candidates.map((candidate) => candidate.source));
      const hypotheses = compareHypotheses.filter(
        (source) => available.has(source) && source !== compareReference,
      );
      if (hypotheses.length === 0) {
        setCompareResult(null);
        setWarning("Select at least one saved output to compare.");
        return;
      }
      const result = await compareMetrics(compareReference, hypotheses, matchMode, "recall");
      setCompareResult(result);
    } catch (err: unknown) {
      setError(String(err));
    } finally {
      setCompareLoading(false);
    }
  }, [compareHypotheses, compareReference, matchMode, metricsCandidates, refreshMetricCandidates]);

  const addMethodPaneForSource = useCallback((source: MetricsCandidateSource) => {
    setPaneInstances((prev) =>
      normalizePaneInstances([...prev, makePaneInstance("methods", source)]),
    );
    const methodId = sourceToMethodView(source);
    if (methodId) {
      setMethodView(methodId);
    }
  }, []);

  const handleOpenCompareDocument = useCallback(
    (source: MetricsCandidateSource, docId: string) => {
      setSelectedId(docId);
      if (String(source).startsWith("agent.method.") || String(source).startsWith("methods_lab.")) {
        addMethodPaneForSource(source);
      }
      setMainTab("workspace");
    },
    [addMethodPaneForSource],
  );

  const handleCompareExportCsv = useCallback(() => {
    if (!compareResult) return;
    const rows = [
      [
        "candidate",
        "document",
        "recall",
        "f1",
        "precision",
        "tp",
        "fp",
        "fn",
        "exact_f1",
        "overlap_f1",
        "coverage_compared",
        "coverage_total",
      ],
    ];
    for (const hypothesisItem of compareResult.hypotheses) {
      for (const row of hypothesisItem.documents) {
        rows.push([
          hypothesisItem.label,
          row.filename,
          String(row.micro.recall),
          String(row.micro.f1),
          String(row.micro.precision),
          String(row.micro.tp),
          String(row.micro.fp),
          String(row.micro.fn),
          String(row.exact_micro.f1),
          String(row.overlap_micro.f1),
          String(hypothesisItem.coverage.compared_documents),
          String(hypothesisItem.coverage.total_documents),
        ]);
      }
    }
    const csv = rows
      .map((row) => row.map((cell) => `"${cell.replace(/"/g, '""')}"`).join(","))
      .join("\n");
    const blob = new Blob([csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `metrics-comparison-${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }, [compareResult]);

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

  const scheduleCompareRefresh = useCallback(
    (delayMs = 1500) => {
      if (!compareResult) return;
      if (compareRefreshTimer.current) {
        clearTimeout(compareRefreshTimer.current);
      }
      compareRefreshTimer.current = setTimeout(() => {
        compareRefreshTimer.current = null;
        void handleCompareRefresh();
      }, delayMs);
    },
    [compareResult, handleCompareRefresh],
  );

  useEffect(() => () => {
    if (saveTimer.current) clearTimeout(saveTimer.current);
    if (savedTimer.current) clearTimeout(savedTimer.current);
    if (compareRefreshTimer.current) clearTimeout(compareRefreshTimer.current);
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
      if (compareRefreshTimer.current) clearTimeout(compareRefreshTimer.current);
      setSaveStatus("saving");
      const docId = doc.id;
      saveTimer.current = setTimeout(() => {
        updateManualAnnotations(docId, spans)
          .then((savedDoc) => {
            setDoc((current) => (current?.id === savedDoc.id ? savedDoc : current));
            setSaveStatus("saved");
            savedTimer.current = setTimeout(() => setSaveStatus("idle"), 2000);
            scheduleCompareRefresh();
          })
          .catch((e: unknown) => {
            setSaveStatus("idle");
            setError(String(e));
          });
      }, 1000);
    },
    [doc, scheduleCompareRefresh],
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
        void handleCompareRefresh();
      } catch (e: unknown) {
        setError(String(e));
        pushRunToast("error", `Agent run failed for ${fileLabel}.`);
      } finally {
        setAgentRunning(false);
      }
    },
    [doc, handleCompareRefresh, pushRunToast],
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
        const methodRunKeys = Object.keys(updated.agent_outputs?.methods ?? {}).filter(
          (key) => methodKey && key.startsWith(`${methodKey}::`),
        );
        const latestMethodKey =
          methodRunKeys.length > 0 ? methodRunKeys[methodRunKeys.length - 1] : methodKey;
        const spanCount = latestMethodKey
          ? updated.agent_outputs?.methods?.[latestMethodKey]?.length ?? 0
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
        void handleCompareRefresh();
      } catch (e: unknown) {
        setError(String(e));
        pushRunToast("error", `Method run failed for ${fileLabel}.`);
      } finally {
        setMethodRunning(false);
      }
    },
    [doc, handleCompareRefresh, pushRunToast],
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
      );
      setMetrics(result);
    } catch (e: unknown) {
      setError(String(e));
    } finally {
      setMetricsLoading(false);
    }
  }, [doc, reference, hypothesis, matchMode]);

  // Compute diffs if diff mode is on
  const getSpansForSource = (source: MetricsCandidateSource): CanonicalSpan[] => {
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

  const getMethodSpans = (sourceRef?: MetricsCandidateSource, paneId?: string): CanonicalSpan[] => {
    if (!doc) return [];
    const methodsLabSource = parseMethodsLabSource(sourceRef);
    if (methodsLabSource && paneId && selectedId) {
      return methodsLabPaneSpans[`${paneId}:${selectedId}:${sourceRef}`] ?? [];
    }
    const methodId = sourceToMethodView(sourceRef) ?? methodView;
    if (!methodId) return [];
    return doc.agent_outputs?.methods?.[methodId] ?? [];
  };

  const getPaneSource = (pane: PaneInstance): MetricsCandidateSource | null => {
    if (pane.type === "pre") return "pre";
    if (pane.type === "manual") return "manual";
    if (pane.type === "agent") {
      if (agentView === "rule") return "agent.rule";
      if (agentView === "llm") {
        return agentLlmRun === "__latest__"
          ? "agent.llm"
          : (`agent.llm_run.${agentLlmRun}` as AnnotationSource);
      }
      return "agent";
    }
    if (pane.type === "methods") {
      if (pane.source_ref) return pane.source_ref;
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

  const getGlobalDiffForSource = (source: MetricsCandidateSource) => {
    if (!doc || !diffMode) return [];
    if (!String(source).startsWith("agent.") && source !== "pre" && source !== "manual") return [];
    const visibleSources = getDisplayedSources(
      orderedPaneInstances,
      agentView,
    ).filter((item) => item !== source);
    if (visibleSources.length === 0) return [];

    const target = getSpansForSource(source);
    const allDiffs = visibleSources.flatMap((other) =>
      computeDiff(getSpansForSource(other), target, matchMode),
    );
    return dedupeDiffs(allDiffs);
  };

  const getDiffSpans = (pane: PaneInstance) => {
    if (!doc || !diffMode) return [];
    if (pane.type === "raw") {
      const visibleSources = getDisplayedSources(
        orderedPaneInstances,
        agentView,
      );
      return dedupeDiffs(
        visibleSources.flatMap((source) => getGlobalDiffForSource(source)),
      );
    }
    const source = getPaneSource(pane);
    if (!source || !String(source).startsWith("agent.")) return [];
    return getGlobalDiffForSource(source);
  };

  return {
    mainTab,
    setMainTab,
    documents,
    folders,
    folderDetailsById,
    folderDetailLoadingById,
    health,
    readinessWarnings,
    selectedId,
    setSelectedId,
    doc,
    loading,
    error,
    setError,
    warning,
    setWarning,
    runToasts,
    dismissRunToast,
    pendingConfirm,
    setPendingConfirm,
    agentRunProgress,
    ingesting,
    deletingId,
    folderBusyId,
    exporting,
    mirroringPreToManual,
    orderedPaneInstances,
    setPaneInstances,
    orderedVisiblePanes,
    diffMode,
    setDiffMode,
    reference,
    setReference,
    hypothesis,
    setHypothesis,
    matchMode,
    setMatchMode,
    agentView,
    setAgentView,
    agentLlmRun,
    setAgentLlmRun,
    methodCatalog,
    methodView,
    setMethodView,
    agentRunning,
    methodRunning,
    agentChunked,
    methodChunked,
    metrics,
    metricsLoading,
    metricsCandidates,
    compareReference,
    setCompareReference,
    compareHypotheses,
    setCompareHypotheses,
    compareResult,
    compareLoading,
    saveStatus,
    registerPane,
    handleScroll,
    sourceOptions,
    llmRunOptions,
    agentChunkDiagnostics,
    methodsLabPaneLoading,
    handleIngestFiles,
    handleDeleteDocument,
    handleDeleteFolderDocument,
    handleCreateFolder,
    handleCreateFolderSample,
    handleDeleteFolder,
    handlePruneFolder,
    handleMirrorPreToManual,
    handleExportSession,
    handleCompareRefresh,
    handleCompareExportCsv,
    handleOpenCompareDocument,
    handleTogglePane,
    handleManualChange,
    handleRunAgent,
    handleRunMethod,
    handleMetricsRefresh,
    getDiffSpans,
    getAgentSpans,
    getMethodSpans,
    ensureFolderDetail,
    handleWorkspaceChanged,
  };
}

type AppContentController = ReturnType<typeof useAppContentController>;

function renderAppContent(controller: AppContentController) {
  const {
    mainTab,
    setMainTab,
    documents,
    folders,
    folderDetailsById,
    folderDetailLoadingById,
    health,
    readinessWarnings,
    selectedId,
    setSelectedId,
    doc,
    loading,
    error,
    setError,
    warning,
    setWarning,
    runToasts,
    dismissRunToast,
    pendingConfirm,
    setPendingConfirm,
    agentRunProgress,
    ingesting,
    deletingId,
    folderBusyId,
    exporting,
    mirroringPreToManual,
    orderedPaneInstances,
    setPaneInstances,
    orderedVisiblePanes,
    diffMode,
    setDiffMode,
    reference,
    setReference,
    hypothesis,
    setHypothesis,
    matchMode,
    setMatchMode,
    agentView,
    setAgentView,
    agentLlmRun,
    setAgentLlmRun,
    methodCatalog,
    methodView,
    setMethodView,
    agentRunning,
    methodRunning,
    agentChunked,
    methodChunked,
    metrics,
    metricsLoading,
    metricsCandidates,
    compareReference,
    setCompareReference,
    compareHypotheses,
    setCompareHypotheses,
    compareResult,
    compareLoading,
    saveStatus,
    registerPane,
    handleScroll,
    sourceOptions,
    llmRunOptions,
    agentChunkDiagnostics,
    methodsLabPaneLoading,
    handleIngestFiles,
    handleDeleteDocument,
    handleDeleteFolderDocument,
    handleCreateFolder,
    handleCreateFolderSample,
    handleDeleteFolder,
    handlePruneFolder,
    handleMirrorPreToManual,
    handleExportSession,
    handleCompareRefresh,
    handleCompareExportCsv,
    handleOpenCompareDocument,
    handleTogglePane,
    handleManualChange,
    handleRunAgent,
    handleRunMethod,
    handleMetricsRefresh,
    getDiffSpans,
    getAgentSpans,
    getMethodSpans,
    ensureFolderDetail,
    handleWorkspaceChanged,
  } = controller;

  const allPanes: PaneInstance[] = orderedPaneInstances;
  let paneIndex = 0;

  return (
    <div className="app-layout">
      <Sidebar
        documents={documents}
        folders={folders}
        folderDetailsById={folderDetailsById}
        selectedId={selectedId}
        onSelect={setSelectedId}
        onIngestFiles={handleIngestFiles}
        onDelete={handleDeleteDocument}
        onDeleteFolderDocument={handleDeleteFolderDocument}
        onCreateFolder={handleCreateFolder}
        onCreateFolderSample={handleCreateFolderSample}
        onDeleteFolder={handleDeleteFolder}
        onPruneFolder={handlePruneFolder}
        onMirrorPreToManual={handleMirrorPreToManual}
        onEnsureFolderDetail={ensureFolderDetail}
        onExportSession={handleExportSession}
        exportSourceOptions={sourceOptions}
        ingesting={ingesting}
        deletingId={deletingId}
        folderBusyId={folderBusyId}
        folderDetailLoadingById={folderDetailLoadingById}
        exporting={exporting}
        mirroringPreToManual={mirroringPreToManual}
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
        {readinessWarnings.length > 0 && (
          <div className="readiness-banner" role="status">
            <div>
              <strong>Readiness warnings</strong>
              <span>
                {health?.counts.documents ?? 0} docs, {health?.counts.folders ?? 0} folders,
                storage {health?.storage.exists ? "ready" : "missing"}
              </span>
            </div>
            <ul>
              {readinessWarnings.slice(0, 4).map((message) => (
                <li key={message}>{message}</li>
              ))}
            </ul>
          </div>
        )}
        {mainTab === "prompt_lab" ? (
          <PromptLabTab
            documents={documents}
            folders={folders}
            selectedDocumentId={selectedId}
            onSelectDocument={setSelectedId}
          />
        ) : mainTab === "methods_lab" ? (
          <MethodsLabTab
            documents={documents}
            folders={folders}
            selectedDocumentId={selectedId}
            onSelectDocument={setSelectedId}
            onWorkspaceChanged={handleWorkspaceChanged}
            onOpenCellInWorkspace={handleOpenCompareDocument}
            onCompareCells={(sources) => {
              setCompareReference("manual");
              setCompareHypotheses(sources);
              setMainTab("dashboard");
            }}
          />
        ) : mainTab === "dashboard" ? (
          <section className="dashboard-tab">
            <MetricsCompareDashboard
              candidates={metricsCandidates}
              reference={compareReference}
              selectedHypotheses={compareHypotheses}
              matchMode={matchMode}
              loading={compareLoading}
              result={compareResult}
              onReferenceChange={setCompareReference}
              onHypothesesChange={setCompareHypotheses}
              onMatchModeChange={setMatchMode}
              onRefresh={() => void handleCompareRefresh()}
              onExportCsv={handleCompareExportCsv}
              onOpenDocument={handleOpenCompareDocument}
            />
          </section>
        ) : (
          <>
            {!doc && !loading && (
              <div className="empty-state">
                <h2>Start with a transcript or session bundle</h2>
                <p>
                  Drop a JSON, JSONL, TXT, ZIP, or exported full-session bundle in the
                  sidebar. Imported folders load their transcript lists only when expanded.
                </p>
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
                <PaneContainer>
                  {allPanes.map((pane) => {
                    const idx = paneIndex++;
                    switch (pane.type) {
                      case "raw":
                        return (
                          <RawPane
                            key={pane.id}
                            ref={registerPane(idx)}
                            text={doc.raw_text}
                            diffSpans={getDiffSpans(pane)}
                            onScroll={handleScroll(idx)}
                          />
                        );
                      case "pre":
                        return (
                          <PreAnnotationPane
                            key={pane.id}
                            ref={registerPane(idx)}
                            text={doc.raw_text}
                            spans={doc.pre_annotations}
                            diffSpans={getDiffSpans(pane)}
                            onScroll={handleScroll(idx)}
                          />
                        );
                      case "manual":
                        return (
                          <ManualAnnotationPane
                            key={pane.id}
                            ref={registerPane(idx)}
                            text={doc.raw_text}
                            labels={PII_LABELS}
                            spans={doc.manual_annotations}
                            diffSpans={getDiffSpans(pane)}
                            onSpansChange={handleManualChange}
                            onScroll={handleScroll(idx)}
                          />
                        );
                      case "agent":
                        return (
                          <AgentPane
                            key={pane.id}
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
                            diffSpans={getDiffSpans(pane)}
                            onRunAgent={handleRunAgent}
                            running={agentRunning}
                            onScroll={handleScroll(idx)}
                          />
                        );
                      case "methods":
                        {
                          const activeMethod = sourceToMethodView(pane.source_ref) ?? methodView;
                          const methodSource =
                            pane.source_ref ??
                            (activeMethod
                              ? (`agent.method.${activeMethod}` as MetricsCandidateSource)
                              : undefined);
                          const isMethodsLabPane = Boolean(parseMethodsLabSource(methodSource));
                          return (
                            <MethodPane
                              key={pane.id}
                              ref={registerPane(idx)}
                              text={doc.raw_text}
                              spans={getMethodSpans(methodSource, pane.id)}
                              runProgress={methodRunning && !isMethodsLabPane ? agentRunProgress : null}
                              methods={methodCatalog}
                              processedWithChunking={
                                !isMethodsLabPane &&
                                (methodChunked || getChunkDiagnosticsCount(resolveMethodChunkDiagnostics(doc, activeMethod)) > 0)
                              }
                              chunkDiagnostics={isMethodsLabPane ? [] : resolveMethodChunkDiagnostics(doc, activeMethod)}
                              activeMethod={activeMethod}
                              onActiveMethodChange={(nextMethod) => {
                                setPaneInstances((prev) =>
                                  normalizePaneInstances(
                                    prev.map((item) =>
                                      item.id === pane.id
                                        ? {
                                            ...item,
                                            source_ref: `agent.method.${nextMethod}` as MetricsCandidateSource,
                                          }
                                        : item,
                                    ),
                                  ),
                                );
                                setMethodView(nextMethod);
                              }}
                              diffSpans={getDiffSpans(pane)}
                              onRunMethod={handleRunMethod}
                              running={methodRunning || Boolean(methodsLabPaneLoading[pane.id])}
                              onScroll={handleScroll(idx)}
                            />
                          );
                        }
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
        <button type="button" className="error-toast" onClick={() => setError(null)}>
          {error}
        </button>
      )}
      {warning && (
        <button type="button" className="warning-toast" onClick={() => setWarning(null)}>
          {warning}
        </button>
      )}
      {runToasts.length > 0 && (
        <div className="run-toast-stack" aria-live="polite" aria-atomic="false">
          {runToasts.map((toast) => (
            <button
              key={toast.id}
              type="button"
              className={`run-toast ${toast.kind}`}
              onClick={() => dismissRunToast(toast.id)}
            >
              {toast.message}
            </button>
          ))}
        </div>
      )}
      {pendingConfirm && (
        <ConfirmDialog
          title={pendingConfirm.title}
          message={pendingConfirm.message}
          confirmLabel={pendingConfirm.confirmLabel}
          destructive={pendingConfirm.destructive}
          onConfirm={pendingConfirm.onConfirm}
          onCancel={() => setPendingConfirm(null)}
        />
      )}
    </div>
  );
}

function AppContent() {
  return renderAppContent(useAppContentController());
}

function getDisplayedSources(
  panes: PaneInstance[],
  agentView: AgentView,
): MetricsCandidateSource[] {
  const sources: MetricsCandidateSource[] = [];
  if (panes.some((pane) => pane.type === "pre")) sources.push("pre");
  if (panes.some((pane) => pane.type === "manual")) sources.push("manual");
  if (panes.some((pane) => pane.type === "agent")) {
    if (agentView === "rule") sources.push("agent.rule");
    else if (agentView === "llm") sources.push("agent.llm");
    else sources.push("agent");
  }
  for (const pane of panes) {
    if (pane.type !== "methods") continue;
    if (pane.source_ref) {
      sources.push(pane.source_ref);
    }
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
