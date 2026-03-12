import { useEffect, useMemo, useState } from "react";
import { MODEL_PRESETS, getModelPreset } from "../modelPresets";
import type {
  AgentMethodOption,
  DocumentSummary,
  FolderSummary,
  MatchMode,
  MethodBundle,
  MethodsLabMethodInput,
  MethodsLabRunCreateRequest,
  PromptLabModelInput,
} from "../types";

interface Props {
  documents: DocumentSummary[];
  folders: FolderSummary[];
  selectedDocumentId: string | null;
  methods: AgentMethodOption[];
  concurrencyMax: number;
  onRun: (payload: MethodsLabRunCreateRequest) => Promise<void>;
  running: boolean;
  collapsed?: boolean;
  onToggleCollapsed?: () => void;
}

type MethodsLabFormState = {
  name: string;
  selectedDocIds: string[];
  selectedFolderIds: string[];
  methodVariants: MethodsLabMethodInput[];
  models: PromptLabModelInput[];
};

type MethodsLabRuntimeState = {
  temperature: number;
  matchMode: MatchMode;
  referenceSource: "manual" | "pre";
  apiKey: string;
  apiBase: string;
  concurrency: number;
  labelProfile: "simple" | "advanced";
  labelProjection: "native" | "coarse_simple";
  methodBundle: MethodBundle;
  chunkMode: "auto" | "off" | "force";
  chunkSizeChars: number;
};

type MethodsLabUiState = {
  submitting: boolean;
  error: string | null;
  localCollapsed: boolean;
};

type RuntimePatch = Partial<MethodsLabRuntimeState>;

const REASONING_EFFORT_OPTIONS = ["none", "low", "medium", "high", "xhigh"] as const;
const FALLBACK_METHOD_IDS = [
  "default",
  "extended",
  "verified",
  "dual",
  "dual-split",
  "presidio",
  "presidio+default",
  "presidio+llm-split",
] as const;
const MAX_METHOD_VARIANTS = 12;

function readSessionValue(key: string): string {
  try {
    return sessionStorage.getItem(key) ?? "";
  } catch {
    return "";
  }
}

function makeMethod(index: number, methodOptions: AgentMethodOption[]): MethodsLabMethodInput {
  const selected = methodOptions[0];
  return {
    id: `method_${index + 1}`,
    label: selected?.label ?? `Method ${index + 1}`,
    method_id: selected?.id ?? "default",
    method_verify_override:
      selected?.supports_verify_override && typeof selected.default_verify === "boolean"
        ? selected.default_verify
        : null,
  };
}

function makeModel(index: number): PromptLabModelInput {
  const preset = MODEL_PRESETS[0];
  return {
    id: `model_${index + 1}`,
    label: index === 0 ? "Codex 5.3" : `Model ${index + 1}`,
    model: preset?.id ?? "openai.gpt-5.3-codex",
    reasoning_effort: preset?.defaultReasoningEffort ?? "none",
    anthropic_thinking: false,
    anthropic_thinking_budget_tokens: null,
  };
}

function flattenFolders(
  folders: FolderSummary[],
): Array<{ folder: FolderSummary; depth: number }> {
  const byParent = new Map<string | null, FolderSummary[]>();
  for (const folder of folders) {
    const bucket = byParent.get(folder.parent_folder_id) ?? [];
    bucket.push(folder);
    byParent.set(folder.parent_folder_id, bucket);
  }
  const ordered: Array<{ folder: FolderSummary; depth: number }> = [];
  const visit = (parentId: string | null, depth: number) => {
    const children = byParent.get(parentId) ?? [];
    for (const folder of children) {
      ordered.push({ folder, depth });
      visit(folder.id, depth + 1);
    }
  };
  visit(null, 0);
  return ordered;
}

function MethodsLabRunHeader({
  isCollapsed,
  requestEstimate,
  docCount,
  methodCount,
  modelCount,
  onToggleCollapsed,
}: {
  isCollapsed: boolean;
  requestEstimate: number;
  docCount: number;
  methodCount: number;
  modelCount: number;
  onToggleCollapsed: () => void;
}) {
  return (
    <div className="prompt-lab-run-header">
      <h3>Methods Lab Setup</h3>
      <div className="prompt-lab-estimate">
        Requests: <strong>{requestEstimate}</strong> ({docCount} docs × {methodCount} methods ×{" "}
        {modelCount} models)
      </div>
      <button type="button" className="prompt-lab-toggle-btn" onClick={onToggleCollapsed}>
        {isCollapsed ? "Show Setup" : "Hide Setup"}
      </button>
    </div>
  );
}

function MethodsLabConfigGrid({
  name,
  runtime,
  concurrencyMax,
  onNameChange,
  onRuntimeChange,
}: {
  name: string;
  runtime: MethodsLabRuntimeState;
  concurrencyMax: number;
  onNameChange: (value: string) => void;
  onRuntimeChange: (patch: RuntimePatch) => void;
}) {
  return (
    <div className="prompt-lab-form-grid">
      <div className="prompt-lab-field">
        <label htmlFor="methods-lab-name">Run Name (optional)</label>
        <input
          id="methods-lab-name"
          type="text"
          value={name}
          onChange={(e) => onNameChange(e.target.value)}
          placeholder="Methods Lab experiment"
        />
      </div>

      <div className="prompt-lab-field">
        <label htmlFor="methods-lab-temp">Temperature ({runtime.temperature.toFixed(1)})</label>
        <input
          id="methods-lab-temp"
          type="range"
          min="0"
          max="2"
          step="0.1"
          value={runtime.temperature}
          onChange={(e) => onRuntimeChange({ temperature: Number.parseFloat(e.target.value) })}
        />
      </div>

      <div className="prompt-lab-field prompt-lab-inline">
        <label htmlFor="methods-lab-match">Match</label>
        <select
          id="methods-lab-match"
          value={runtime.matchMode}
          onChange={(e) => onRuntimeChange({ matchMode: e.target.value as MatchMode })}
        >
          <option value="exact">Exact</option>
          <option value="boundary">Trim Space/Punct</option>
          <option value="overlap">Overlap</option>
        </select>
      </div>

      <div className="prompt-lab-field prompt-lab-inline">
        <label htmlFor="methods-lab-reference">Reference</label>
        <select
          id="methods-lab-reference"
          value={runtime.referenceSource}
          onChange={(e) =>
            onRuntimeChange({ referenceSource: e.target.value as "manual" | "pre" })
          }
        >
          <option value="manual">Manual annotations</option>
          <option value="pre">Pre-annotations</option>
        </select>
      </div>

      <div className="prompt-lab-field prompt-lab-inline">
        <label htmlFor="methods-lab-concurrency">Concurrency</label>
        <input
          id="methods-lab-concurrency"
          type="number"
          min={1}
          max={concurrencyMax}
          value={runtime.concurrency}
          onChange={(e) =>
            onRuntimeChange({ concurrency: Number.parseInt(e.target.value, 10) || 1 })
          }
        />
        <div className="prompt-lab-config-note">
          Max {concurrencyMax}. Higher values mostly help LLM-backed sweeps.
        </div>
      </div>

      <div className="prompt-lab-field prompt-lab-inline">
        <label htmlFor="methods-lab-label-profile">Label Profile</label>
        <select
          id="methods-lab-label-profile"
          value={runtime.labelProfile}
          onChange={(e) =>
            onRuntimeChange({ labelProfile: e.target.value as "simple" | "advanced" })
          }
        >
          <option value="simple">Simple</option>
          <option value="advanced">Advanced (UPchieve)</option>
        </select>
      </div>

      <div className="prompt-lab-field prompt-lab-inline">
        <label htmlFor="methods-lab-label-projection">Label Compare</label>
        <select
          id="methods-lab-label-projection"
          value={runtime.labelProjection}
          onChange={(e) =>
            onRuntimeChange({
              labelProjection: e.target.value as "native" | "coarse_simple",
            })
          }
        >
          <option value="native">Native</option>
          <option value="coarse_simple">Coarse (advanced→simple)</option>
        </select>
      </div>

      <div className="prompt-lab-field prompt-lab-inline">
        <label htmlFor="methods-lab-method-bundle">Method Bundle</label>
        <select
          id="methods-lab-method-bundle"
          value={runtime.methodBundle}
          onChange={(e) =>
            onRuntimeChange({ methodBundle: e.target.value as MethodBundle })
          }
        >
          <option value="audited">Audited</option>
          <option value="test">Test</option>
          <option value="v2">V2</option>
          <option value="v2+post-process">V2 + post-process</option>
          <option value="legacy">Legacy</option>
        </select>
      </div>

      <div className="prompt-lab-field prompt-lab-inline">
        <label htmlFor="methods-lab-chunk-mode">Chunk Mode</label>
        <select
          id="methods-lab-chunk-mode"
          value={runtime.chunkMode}
          onChange={(e) =>
            onRuntimeChange({ chunkMode: e.target.value as "auto" | "off" | "force" })
          }
        >
          <option value="auto">Auto</option>
          <option value="off">Off</option>
          <option value="force">Force</option>
        </select>
      </div>

      {runtime.chunkMode !== "off" && (
        <div className="prompt-lab-field prompt-lab-inline">
          <label htmlFor="methods-lab-chunk-size">Chunk Size (chars)</label>
          <input
            id="methods-lab-chunk-size"
            type="number"
            min={2000}
            max={30000}
            step={500}
            value={runtime.chunkSizeChars}
            onChange={(e) =>
              onRuntimeChange({
                chunkSizeChars: Number.parseInt(e.target.value, 10) || 10000,
              })
            }
          />
        </div>
      )}

      <div className="prompt-lab-field">
        <label htmlFor="methods-lab-api-key">API Key (optional override)</label>
        <input
          id="methods-lab-api-key"
          type="password"
          value={runtime.apiKey}
          onChange={(e) => onRuntimeChange({ apiKey: e.target.value })}
          placeholder="Uses env key when not set"
        />
      </div>

      <div className="prompt-lab-field">
        <label htmlFor="methods-lab-api-base">LiteLLM Base URL (optional override)</label>
        <input
          id="methods-lab-api-base"
          type="text"
          value={runtime.apiBase}
          onChange={(e) => onRuntimeChange({ apiBase: e.target.value })}
          placeholder="https://your-gateway/v1"
        />
      </div>
    </div>
  );
}

function MethodsLabDocumentsSection({
  documents,
  folders,
  selectedDocIds,
  selectedFolderIds,
  onToggleDoc,
  onToggleFolder,
}: {
  documents: DocumentSummary[];
  folders: FolderSummary[];
  selectedDocIds: string[];
  selectedFolderIds: string[];
  onToggleDoc: (docId: string) => void;
  onToggleFolder: (folderId: string) => void;
}) {
  const flattenedFolders = useMemo(() => flattenFolders(folders), [folders]);

  return (
    <div className="prompt-lab-section">
      <div className="prompt-lab-section-header">
        <span>Documents and Folders</span>
        <small>{selectedDocIds.length} docs • {selectedFolderIds.length} folders</small>
      </div>
      <div className="prompt-lab-config-note">
        Methods Lab scores against the selected reference source. Documents without a matching
        reference are marked unavailable.
      </div>
      <div className="prompt-lab-doc-grid">
        {documents.map((doc) => (
          <label key={doc.id} className="prompt-lab-doc-option">
            <input
              type="checkbox"
              checked={selectedDocIds.includes(doc.id)}
              onChange={() => onToggleDoc(doc.id)}
            />
            <span title={doc.display_name}>{doc.display_name}</span>
          </label>
        ))}
        {flattenedFolders.map(({ folder, depth }) => (
          <label key={folder.id} className="prompt-lab-doc-option">
            <input
              type="checkbox"
              checked={selectedFolderIds.includes(folder.id)}
              onChange={() => onToggleFolder(folder.id)}
            />
            <span title={folder.name}>
              {`${"  ".repeat(depth)}${folder.name} (${folder.doc_count} docs)`}
            </span>
          </label>
        ))}
      </div>
    </div>
  );
}

function MethodVariantsSection({
  methodVariants,
  methodOptions,
  onAddAllAvailable,
  onAddMethod,
  onRemoveMethod,
  onUpdateMethod,
  onMethodPresetChange,
}: {
  methodVariants: MethodsLabMethodInput[];
  methodOptions: AgentMethodOption[];
  onAddAllAvailable: () => void;
  onAddMethod: () => void;
  onRemoveMethod: (index: number) => void;
  onUpdateMethod: (index: number, patch: Partial<MethodsLabMethodInput>) => void;
  onMethodPresetChange: (index: number, methodId: string) => void;
}) {
  return (
    <div className="prompt-lab-section">
      <div className="prompt-lab-section-header">
        <span>Method Variants</span>
        <div>
          <button
            type="button"
            onClick={onAddAllAvailable}
            disabled={methodOptions.filter((item) => item.available).length === 0}
          >
            Add All Available
          </button>
          <button
            type="button"
            onClick={onAddMethod}
            disabled={methodVariants.length >= MAX_METHOD_VARIANTS}
          >
            + Add Method
          </button>
        </div>
      </div>
      {methodVariants.map((method, index) => {
        const selectedMethod = methodOptions.find((item) => item.id === method.method_id);
        return (
          <div key={method.id ?? index} className="prompt-lab-card">
            <div className="prompt-lab-card-header">
              <input
                type="text"
                value={method.label}
                onChange={(e) => onUpdateMethod(index, { label: e.target.value })}
                placeholder="Method label"
              />
              <button
                type="button"
                onClick={() => onRemoveMethod(index)}
                disabled={methodVariants.length <= 1}
              >
                Remove
              </button>
            </div>

            <div className="prompt-lab-model-row">
              <label>
                Built-in Method
                <select
                  value={method.method_id}
                  onChange={(e) => onMethodPresetChange(index, e.target.value)}
                >
                  {methodOptions.map((option) => (
                    <option key={option.id} value={option.id} disabled={!option.available}>
                      {option.label}
                      {!option.available ? " (setup required)" : ""}
                    </option>
                  ))}
                </select>
              </label>
            </div>

            {selectedMethod && (
              <div className="prompt-lab-config-note">
                {selectedMethod.description}
                {selectedMethod.unavailable_reason ? ` ${selectedMethod.unavailable_reason}` : ""}
              </div>
            )}

            {selectedMethod?.supports_verify_override && (
              <label className="inline-checkbox-label">
                <input
                  type="checkbox"
                  checked={Boolean(method.method_verify_override)}
                  onChange={(e) =>
                    onUpdateMethod(index, { method_verify_override: e.target.checked })
                  }
                />
                Verifier Override
              </label>
            )}
          </div>
        );
      })}
    </div>
  );
}

function ModelVariantsSection({
  models,
  onAddModel,
  onRemoveModel,
  onUpdateModel,
  onModelPresetChange,
}: {
  models: PromptLabModelInput[];
  onAddModel: () => void;
  onRemoveModel: (index: number) => void;
  onUpdateModel: (index: number, patch: Partial<PromptLabModelInput>) => void;
  onModelPresetChange: (index: number, presetId: string) => void;
}) {
  return (
    <div className="prompt-lab-section">
      <div className="prompt-lab-section-header">
        <span>Model Variants</span>
        <button type="button" onClick={onAddModel} disabled={models.length >= 6}>
          + Add Model
        </button>
      </div>
      {models.map((model, index) => {
        const preset = getModelPreset(model.model);
        const presetValue = preset ? model.model : "__custom__";
        const supportsReasoning = preset?.supportsReasoningEffort ?? false;
        const supportsThinking = preset?.supportsAnthropicThinking ?? false;
        return (
          <div key={model.id ?? index} className="prompt-lab-card">
            <div className="prompt-lab-card-header">
              <input
                type="text"
                value={model.label}
                onChange={(e) => onUpdateModel(index, { label: e.target.value })}
                placeholder="Model label"
              />
              <button type="button" onClick={() => onRemoveModel(index)} disabled={models.length <= 1}>
                Remove
              </button>
            </div>

            <div className="prompt-lab-model-row">
              <select value={presetValue} onChange={(e) => onModelPresetChange(index, e.target.value)}>
                {MODEL_PRESETS.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.label}
                  </option>
                ))}
                <option value="__custom__">Custom...</option>
              </select>
              {presetValue === "__custom__" && (
                <input
                  type="text"
                  value={model.model}
                  onChange={(e) => onUpdateModel(index, { model: e.target.value })}
                  placeholder="provider.model-name"
                />
              )}
            </div>

            <div className="prompt-lab-model-row">
              <label>
                Reasoning
                <select
                  value={model.reasoning_effort ?? "none"}
                  disabled={!supportsReasoning}
                  onChange={(e) =>
                    onUpdateModel(index, {
                      reasoning_effort: e.target.value as
                        | "none"
                        | "low"
                        | "medium"
                        | "high"
                        | "xhigh",
                    })
                  }
                >
                  {REASONING_EFFORT_OPTIONS.map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={Boolean(model.anthropic_thinking)}
                  disabled={!supportsThinking}
                  onChange={(e) =>
                    onUpdateModel(index, { anthropic_thinking: e.target.checked })
                  }
                />
                Anthropic thinking
              </label>
              <input
                type="number"
                min={256}
                step={256}
                disabled={!supportsThinking || !model.anthropic_thinking}
                value={model.anthropic_thinking_budget_tokens ?? 2048}
                onChange={(e) =>
                  onUpdateModel(index, {
                    anthropic_thinking_budget_tokens: Number.parseInt(e.target.value, 10) || 2048,
                  })
                }
                placeholder="Thinking budget"
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function useMethodsLabRunFormController({
  folders,
  selectedDocumentId,
  methods,
  concurrencyMax,
  onRun,
  collapsed,
  onToggleCollapsed,
}: Omit<Props, "documents" | "running">) {
  const folderDocCountById = useMemo(
    () =>
      Object.fromEntries(folders.map((folder) => [folder.id, folder.doc_count])) as Record<
        string,
        number
      >,
    [folders],
  );
  const methodOptions = useMemo(
    () =>
      methods.length > 0
        ? methods
        : FALLBACK_METHOD_IDS.map((id) => ({
            id,
            label: id,
            description: "",
            requires_presidio: id.includes("presidio"),
            uses_llm: id !== "presidio",
            supports_verify_override: id !== "presidio",
            default_verify: id === "verified",
            available: true,
            unavailable_reason: null,
          })),
    [methods],
  );

  const [formState, setFormState] = useState<MethodsLabFormState>({
    name: "",
    selectedDocIds: [],
    selectedFolderIds: [],
    methodVariants: [makeMethod(0, methodOptions)],
    models: [makeModel(0)],
  });
  const [runtimeState, setRuntimeState] = useState<MethodsLabRuntimeState>({
    temperature: 0,
    matchMode: "overlap",
    referenceSource: "manual",
    apiKey: readSessionValue("methods_lab_api_key"),
    apiBase: readSessionValue("methods_lab_api_base"),
    concurrency: 10,
    labelProfile: "simple",
    labelProjection: "native",
    methodBundle: "audited",
    chunkMode: "off",
    chunkSizeChars: 10000,
  });
  const [uiState, setUiState] = useState<MethodsLabUiState>({
    submitting: false,
    error: null,
    localCollapsed: false,
  });

  useEffect(() => {
    if (!selectedDocumentId) return;
    setFormState((prev) =>
      prev.selectedDocIds.length > 0 ? prev : { ...prev, selectedDocIds: [selectedDocumentId] },
    );
  }, [selectedDocumentId]);

  useEffect(() => {
    setRuntimeState((prev) => ({
      ...prev,
      concurrency: Math.min(Math.max(prev.concurrency, 1), concurrencyMax),
    }));
  }, [concurrencyMax]);

  useEffect(() => {
    try {
      if (runtimeState.apiKey) {
        sessionStorage.setItem("methods_lab_api_key", runtimeState.apiKey);
      } else {
        sessionStorage.removeItem("methods_lab_api_key");
      }
    } catch {
      // sessionStorage unavailable
    }
  }, [runtimeState.apiKey]);

  useEffect(() => {
    try {
      if (runtimeState.apiBase) {
        sessionStorage.setItem("methods_lab_api_base", runtimeState.apiBase);
      } else {
        sessionStorage.removeItem("methods_lab_api_base");
      }
    } catch {
      // sessionStorage unavailable
    }
  }, [runtimeState.apiBase]);

  useEffect(() => {
    setFormState((prev) =>
      prev.methodVariants.length > 0
        ? prev
        : { ...prev, methodVariants: [makeMethod(0, methodOptions)] },
    );
  }, [methodOptions]);

  const selectedFolderDocCount = formState.selectedFolderIds.reduce(
    (sum, folderId) => sum + (folderDocCountById[folderId] ?? 0),
    0,
  );
  const requestEstimate =
    (formState.selectedDocIds.length + selectedFolderDocCount) *
    formState.methodVariants.length *
    formState.models.length;
  const selectedTotalDocs = formState.selectedDocIds.length + selectedFolderDocCount;
  const isCollapsed = collapsed ?? uiState.localCollapsed;

  const updateRuntime = (patch: RuntimePatch) => {
    setRuntimeState((prev) => ({ ...prev, ...patch }));
  };
  const setUiError = (error: string | null) => {
    setUiState((prev) => ({ ...prev, error }));
  };
  const setSubmitting = (submitting: boolean) => {
    setUiState((prev) => ({ ...prev, submitting }));
  };
  const toggleCollapsed = () => {
    if (onToggleCollapsed) {
      onToggleCollapsed();
      return;
    }
    setUiState((prev) => ({ ...prev, localCollapsed: !prev.localCollapsed }));
  };

  const handleNameChange = (name: string) => {
    setFormState((prev) => ({ ...prev, name }));
  };

  const toggleDoc = (docId: string) => {
    setFormState((prev) => ({
      ...prev,
      selectedDocIds: prev.selectedDocIds.includes(docId)
        ? prev.selectedDocIds.filter((id) => id !== docId)
        : [...prev.selectedDocIds, docId],
    }));
  };

  const toggleFolder = (folderId: string) => {
    setFormState((prev) => ({
      ...prev,
      selectedFolderIds: prev.selectedFolderIds.includes(folderId)
        ? prev.selectedFolderIds.filter((id) => id !== folderId)
        : [...prev.selectedFolderIds, folderId],
    }));
  };

  const updateMethodVariant = (index: number, patch: Partial<MethodsLabMethodInput>) => {
    setFormState((prev) => ({
      ...prev,
      methodVariants: prev.methodVariants.map((item, i) =>
        i === index ? { ...item, ...patch } : item,
      ),
    }));
  };

  const handleMethodPresetChange = (index: number, methodId: string) => {
    const selected = methodOptions.find((item) => item.id === methodId);
    updateMethodVariant(index, {
      method_id: methodId,
      label: selected?.label ?? methodId,
      method_verify_override:
        selected?.supports_verify_override && typeof selected.default_verify === "boolean"
          ? selected.default_verify
          : null,
    });
  };

  const addAllAvailableMethods = () => {
    const available = methodOptions.filter((item) => item.available).slice(0, MAX_METHOD_VARIANTS);
    if (available.length === 0) return;
    setFormState((prev) => ({
      ...prev,
      methodVariants: available.map((method, index) => ({
        id: `method_${index + 1}`,
        label: method.label,
        method_id: method.id,
        method_verify_override:
          method.supports_verify_override && typeof method.default_verify === "boolean"
            ? method.default_verify
            : null,
      })),
    }));
  };

  const handleAddMethod = () => {
    setFormState((prev) => ({
      ...prev,
      methodVariants: [...prev.methodVariants, makeMethod(prev.methodVariants.length, methodOptions)],
    }));
  };

  const handleRemoveMethod = (index: number) => {
    setFormState((prev) => ({
      ...prev,
      methodVariants: prev.methodVariants.filter((_, i) => i !== index),
    }));
  };

  const updateModel = (index: number, patch: Partial<PromptLabModelInput>) => {
    setFormState((prev) => ({
      ...prev,
      models: prev.models.map((item, i) => (i === index ? { ...item, ...patch } : item)),
    }));
  };

  const handleModelPresetChange = (index: number, presetId: string) => {
    if (presetId === "__custom__") {
      updateModel(index, { model: "", reasoning_effort: "none" });
      return;
    }
    const preset = getModelPreset(presetId);
    updateModel(index, {
      model: presetId,
      reasoning_effort: preset?.defaultReasoningEffort ?? "none",
      anthropic_thinking: false,
      anthropic_thinking_budget_tokens: null,
    });
  };

  const handleAddModel = () => {
    setFormState((prev) => ({
      ...prev,
      models: [...prev.models, makeModel(prev.models.length)],
    }));
  };

  const handleRemoveModel = (index: number) => {
    setFormState((prev) => ({
      ...prev,
      models: prev.models.filter((_, i) => i !== index),
    }));
  };

  const validate = (): string | null => {
    if (formState.selectedDocIds.length === 0 && formState.selectedFolderIds.length === 0) {
      return "Select at least one document or folder";
    }
    if (
      formState.methodVariants.length === 0 ||
      formState.methodVariants.length > MAX_METHOD_VARIANTS
    ) {
      return `Method variants must be 1 to ${MAX_METHOD_VARIANTS}`;
    }
    if (formState.models.length === 0 || formState.models.length > 6) {
      return "Model variants must be 1 to 6";
    }
    if (runtimeState.concurrency < 1 || runtimeState.concurrency > concurrencyMax) {
      return `Concurrency must be 1 to ${concurrencyMax}`;
    }
    if (runtimeState.chunkSizeChars < 2000 || runtimeState.chunkSizeChars > 30000) {
      return "Chunk size must be between 2000 and 30000";
    }
    const seenIds = new Set<string>();
    for (const method of formState.methodVariants) {
      const id = method.id?.trim() ?? "";
      if (!id || seenIds.has(id)) {
        return "Every method variant needs a unique id";
      }
      seenIds.add(id);
      if (!method.label.trim()) {
        return "Every method variant needs a label";
      }
      if (!(method.method_id ?? "").trim()) {
        return "Every method variant needs a built-in method";
      }
    }
    for (const model of formState.models) {
      if (!model.label.trim() || !model.model?.trim()) {
        return "Every model needs a label and model id";
      }
    }
    return null;
  };

  const handleSubmit = async () => {
    const validationError = validate();
    setUiError(validationError);
    if (validationError) return;

    const payload: MethodsLabRunCreateRequest = {
      name: formState.name.trim() || undefined,
      doc_ids: formState.selectedDocIds,
      folder_ids: formState.selectedFolderIds,
      methods: formState.methodVariants.map((item, index) => ({
        id: item.id?.trim() || `method_${index + 1}`,
        label: item.label.trim(),
        method_id: item.method_id.trim(),
        method_verify_override: item.method_verify_override ?? null,
      })),
      models: formState.models.map((item, index) => ({
        id: item.id?.trim() || `model_${index + 1}`,
        label: item.label.trim(),
        model: item.model.trim(),
        reasoning_effort: item.reasoning_effort ?? "none",
        anthropic_thinking: Boolean(item.anthropic_thinking),
        anthropic_thinking_budget_tokens: item.anthropic_thinking
          ? item.anthropic_thinking_budget_tokens ?? 2048
          : null,
      })),
      runtime: {
        api_key: runtimeState.apiKey || undefined,
        api_base: runtimeState.apiBase || undefined,
        temperature: runtimeState.temperature,
        match_mode: runtimeState.matchMode,
        reference_source: runtimeState.referenceSource,
        fallback_reference_source: runtimeState.referenceSource,
        label_profile: runtimeState.labelProfile,
        label_projection: runtimeState.labelProjection,
        method_bundle: runtimeState.methodBundle,
        chunk_mode: runtimeState.chunkMode,
        chunk_size_chars: runtimeState.chunkSizeChars,
      },
      concurrency: runtimeState.concurrency,
    };

    setSubmitting(true);
    try {
      await onRun(payload);
      setUiError(null);
      if (!onToggleCollapsed && collapsed === undefined) {
        setUiState((prev) => ({ ...prev, localCollapsed: true }));
      }
    } catch (e: unknown) {
      setUiError(String(e));
    } finally {
      setSubmitting(false);
    }
  };

  return {
    formState,
    runtimeState,
    uiState,
    methodOptions,
    requestEstimate,
    selectedTotalDocs,
    isCollapsed,
    updateRuntime,
    toggleCollapsed,
    toggleDoc,
    toggleFolder,
    handleNameChange,
    updateMethodVariant,
    handleMethodPresetChange,
    addAllAvailableMethods,
    handleAddMethod,
    handleRemoveMethod,
    updateModel,
    handleModelPresetChange,
    handleAddModel,
    handleRemoveModel,
    handleSubmit,
  };
}

export default function MethodsLabRunForm(props: Props) {
  const { documents, folders, concurrencyMax, running } = props;
  const {
    formState,
    runtimeState,
    uiState,
    methodOptions,
    requestEstimate,
    selectedTotalDocs,
    isCollapsed,
    updateRuntime,
    toggleCollapsed,
    toggleDoc,
    toggleFolder,
    handleNameChange,
    updateMethodVariant,
    handleMethodPresetChange,
    addAllAvailableMethods,
    handleAddMethod,
    handleRemoveMethod,
    updateModel,
    handleModelPresetChange,
    handleAddModel,
    handleRemoveModel,
    handleSubmit,
  } = useMethodsLabRunFormController(props);

  return (
    <section className={`prompt-lab-run-form ${isCollapsed ? "collapsed" : ""}`}>
      <MethodsLabRunHeader
        isCollapsed={isCollapsed}
        requestEstimate={requestEstimate}
        docCount={selectedTotalDocs}
        methodCount={formState.methodVariants.length}
        modelCount={formState.models.length}
        onToggleCollapsed={toggleCollapsed}
      />

      {!isCollapsed && (
        <>
          <MethodsLabConfigGrid
            name={formState.name}
            runtime={runtimeState}
            concurrencyMax={concurrencyMax}
            onNameChange={handleNameChange}
            onRuntimeChange={updateRuntime}
          />
          <MethodsLabDocumentsSection
            documents={documents}
            folders={folders}
            selectedDocIds={formState.selectedDocIds}
            selectedFolderIds={formState.selectedFolderIds}
            onToggleDoc={toggleDoc}
            onToggleFolder={toggleFolder}
          />
          <MethodVariantsSection
            methodVariants={formState.methodVariants}
            methodOptions={methodOptions}
            onAddAllAvailable={addAllAvailableMethods}
            onAddMethod={handleAddMethod}
            onRemoveMethod={handleRemoveMethod}
            onUpdateMethod={updateMethodVariant}
            onMethodPresetChange={handleMethodPresetChange}
          />
          <ModelVariantsSection
            models={formState.models}
            onAddModel={handleAddModel}
            onRemoveModel={handleRemoveModel}
            onUpdateModel={updateModel}
            onModelPresetChange={handleModelPresetChange}
          />

          {uiState.error && <div className="prompt-lab-error">{uiState.error}</div>}

          <div className="prompt-lab-actions">
            <button type="button" onClick={handleSubmit} disabled={running || uiState.submitting}>
              {running || uiState.submitting ? "Starting..." : "Run Methods Lab"}
            </button>
          </div>
        </>
      )}
    </section>
  );
}
