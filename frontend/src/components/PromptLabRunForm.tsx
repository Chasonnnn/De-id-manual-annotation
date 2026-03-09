import { useEffect, useMemo, useState } from "react";
import { MODEL_PRESETS, getModelPreset } from "../modelPresets";
import type {
  AgentMethodOption,
  DocumentSummary,
  FolderSummary,
  MatchMode,
  PromptLabModelInput,
  PromptLabPromptInput,
  PromptLabRunCreateRequest,
} from "../types";

interface Props {
  documents: DocumentSummary[];
  folders: FolderSummary[];
  selectedDocumentId: string | null;
  methods: AgentMethodOption[];
  concurrencyMax: number;
  onRun: (payload: PromptLabRunCreateRequest) => Promise<void>;
  running: boolean;
  collapsed?: boolean;
  onToggleCollapsed?: () => void;
}

type PromptLabFormState = {
  name: string;
  selectedDocIds: string[];
  selectedFolderIds: string[];
  prompts: PromptLabPromptInput[];
  models: PromptLabModelInput[];
};

type PromptLabRuntimeState = {
  temperature: number;
  matchMode: MatchMode;
  referenceSource: "manual" | "pre";
  fallbackSource: "manual" | "pre";
  apiKey: string;
  apiBase: string;
  concurrency: number;
  labelProfile: "simple" | "advanced";
  labelProjection: "native" | "coarse_simple";
  chunkMode: "auto" | "off" | "force";
  chunkSizeChars: number;
};

type PromptLabUiState = {
  submitting: boolean;
  error: string | null;
  localCollapsed: boolean;
};

type RuntimePatch = Partial<PromptLabRuntimeState>;

const REASONING_EFFORT_OPTIONS = ["none", "low", "medium", "high", "xhigh"] as const;
const FALLBACK_PRESET_METHOD_IDS = [
  "default",
  "extended",
  "verified",
  "dual",
  "dual-split",
  "presidio",
  "presidio+default",
  "presidio+llm-split",
] as const;

const DEFAULT_PROMPT =
  'You are a PII annotation assistant. Return ONLY a JSON array of objects with start (0-based), end (exclusive), label, and text for each PII span.';

function readSessionValue(key: string): string {
  try {
    return sessionStorage.getItem(key) ?? "";
  } catch {
    return "";
  }
}

function makePrompt(index: number): PromptLabPromptInput {
  return {
    id: `prompt_${index + 1}`,
    label: index === 0 ? "Baseline" : `Prompt ${index + 1}`,
    variant_type: "prompt",
    system_prompt: DEFAULT_PROMPT,
    preset_method_id: null,
    method_verify_override: null,
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

function PromptLabRunHeader({
  isCollapsed,
  requestEstimate,
  docCount,
  promptCount,
  modelCount,
  onToggleCollapsed,
}: {
  isCollapsed: boolean;
  requestEstimate: number;
  docCount: number;
  promptCount: number;
  modelCount: number;
  onToggleCollapsed: () => void;
}) {
  return (
    <div className="prompt-lab-run-header">
      <h3>Prompt Lab Setup</h3>
      <div className="prompt-lab-estimate">
        Requests: <strong>{requestEstimate}</strong> ({docCount} docs × {promptCount} prompts ×{" "}
        {modelCount} models)
      </div>
      <button type="button" className="prompt-lab-toggle-btn" onClick={onToggleCollapsed}>
        {isCollapsed ? "Show Setup" : "Hide Setup"}
      </button>
    </div>
  );
}

function PromptLabConfigGrid({
  name,
  runtime,
  concurrencyMax,
  onNameChange,
  onRuntimeChange,
}: {
  name: string;
  runtime: PromptLabRuntimeState;
  concurrencyMax: number;
  onNameChange: (value: string) => void;
  onRuntimeChange: (patch: RuntimePatch) => void;
}) {
  return (
    <div className="prompt-lab-form-grid">
      <div className="prompt-lab-field">
        <label htmlFor="prompt-lab-name">Run Name (optional)</label>
        <input
          id="prompt-lab-name"
          type="text"
          value={name}
          onChange={(e) => onNameChange(e.target.value)}
          placeholder="Prompt Lab experiment"
        />
      </div>

      <div className="prompt-lab-field">
        <label htmlFor="prompt-lab-temp">Temperature ({runtime.temperature.toFixed(1)})</label>
        <input
          id="prompt-lab-temp"
          type="range"
          min="0"
          max="2"
          step="0.1"
          value={runtime.temperature}
          onChange={(e) => onRuntimeChange({ temperature: Number.parseFloat(e.target.value) })}
        />
      </div>

      <div className="prompt-lab-field prompt-lab-inline">
        <label htmlFor="prompt-lab-match">Match</label>
        <select
          id="prompt-lab-match"
          value={runtime.matchMode}
          onChange={(e) => onRuntimeChange({ matchMode: e.target.value as MatchMode })}
        >
          <option value="exact">Exact</option>
          <option value="boundary">Trim Space/Punct</option>
          <option value="overlap">Overlap</option>
        </select>
      </div>

      <div className="prompt-lab-field prompt-lab-inline">
        <label htmlFor="prompt-lab-reference">Reference</label>
        <select
          id="prompt-lab-reference"
          value={runtime.referenceSource}
          onChange={(e) =>
            onRuntimeChange({ referenceSource: e.target.value as "manual" | "pre" })
          }
        >
          <option value="manual">Manual</option>
          <option value="pre">Pre-annotations</option>
        </select>
      </div>

      <div className="prompt-lab-field prompt-lab-inline">
        <label htmlFor="prompt-lab-fallback">Fallback</label>
        <select
          id="prompt-lab-fallback"
          value={runtime.fallbackSource}
          onChange={(e) =>
            onRuntimeChange({ fallbackSource: e.target.value as "manual" | "pre" })
          }
        >
          <option value="pre">Pre-annotations</option>
          <option value="manual">Manual</option>
        </select>
      </div>

      <div className="prompt-lab-field prompt-lab-inline">
        <label htmlFor="prompt-lab-concurrency">Concurrency</label>
        <input
          id="prompt-lab-concurrency"
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
        <label htmlFor="prompt-lab-label-profile">Label Profile</label>
        <select
          id="prompt-lab-label-profile"
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
        <label htmlFor="prompt-lab-label-projection">Label Compare</label>
        <select
          id="prompt-lab-label-projection"
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
        <label htmlFor="prompt-lab-chunk-mode">Chunk Mode</label>
        <select
          id="prompt-lab-chunk-mode"
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
          <label htmlFor="prompt-lab-chunk-size">Chunk Size (chars)</label>
          <input
            id="prompt-lab-chunk-size"
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
        <label htmlFor="prompt-lab-api-key">API Key (optional override)</label>
        <input
          id="prompt-lab-api-key"
          type="password"
          value={runtime.apiKey}
          onChange={(e) => onRuntimeChange({ apiKey: e.target.value })}
          placeholder="Uses env key when not set"
        />
      </div>

      <div className="prompt-lab-field">
        <label htmlFor="prompt-lab-api-base">LiteLLM Base URL (optional override)</label>
        <input
          id="prompt-lab-api-base"
          type="text"
          value={runtime.apiBase}
          onChange={(e) => onRuntimeChange({ apiBase: e.target.value })}
          placeholder="https://your-gateway/v1"
        />
      </div>
    </div>
  );
}

function PromptLabDocumentsSection({
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

function PromptVariantsSection({
  prompts,
  presetMethodOptions,
  onAddPrompt,
  onRemovePrompt,
  onUpdatePrompt,
  onPromptVariantChange,
}: {
  prompts: PromptLabPromptInput[];
  presetMethodOptions: AgentMethodOption[];
  onAddPrompt: () => void;
  onRemovePrompt: (index: number) => void;
  onUpdatePrompt: (index: number, patch: Partial<PromptLabPromptInput>) => void;
  onPromptVariantChange: (index: number, variantType: "prompt" | "preset") => void;
}) {
  return (
    <div className="prompt-lab-section">
      <div className="prompt-lab-section-header">
        <span>Prompt Variants</span>
        <button type="button" onClick={onAddPrompt} disabled={prompts.length >= 6}>
          + Add Prompt
        </button>
      </div>
      {prompts.map((prompt, index) => {
        const variantType = prompt.variant_type ?? "prompt";
        const selectedPreset = presetMethodOptions.find(
          (method) => method.id === prompt.preset_method_id,
        );
        return (
          <div key={prompt.id ?? index} className="prompt-lab-card">
            <div className="prompt-lab-card-header">
              <input
                type="text"
                value={prompt.label}
                onChange={(e) => onUpdatePrompt(index, { label: e.target.value })}
                placeholder="Prompt label"
              />
              <button
                type="button"
                onClick={() => onRemovePrompt(index)}
                disabled={prompts.length <= 1}
              >
                Remove
              </button>
            </div>

            <div className="prompt-lab-model-row">
              <label>
                Variant
                <select
                  value={variantType}
                  onChange={(e) =>
                    onPromptVariantChange(index, e.target.value as "prompt" | "preset")
                  }
                >
                  <option value="prompt">System Prompt</option>
                  <option value="preset">Experiment Preset</option>
                </select>
              </label>
            </div>

            {variantType === "prompt" ? (
              <textarea
                value={prompt.system_prompt ?? ""}
                onChange={(e) => onUpdatePrompt(index, { system_prompt: e.target.value })}
                rows={4}
              />
            ) : (
              <>
                <div className="prompt-lab-model-row">
                  <label>
                    Preset
                    <select
                      value={prompt.preset_method_id ?? ""}
                      onChange={(e) =>
                        onUpdatePrompt(index, { preset_method_id: e.target.value || null })
                      }
                    >
                      {presetMethodOptions.map((method) => (
                        <option key={method.id} value={method.id} disabled={!method.available}>
                          {method.label}
                          {!method.available ? " (setup required)" : ""}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
                {selectedPreset && (
                  <div className="prompt-lab-config-note">
                    {selectedPreset.description}
                    {selectedPreset.unavailable_reason
                      ? ` ${selectedPreset.unavailable_reason}`
                      : ""}
                  </div>
                )}
                {selectedPreset?.supports_verify_override && (
                  <label className="inline-checkbox-label">
                    <input
                      type="checkbox"
                      checked={Boolean(prompt.method_verify_override)}
                      onChange={(e) =>
                        onUpdatePrompt(index, { method_verify_override: e.target.checked })
                      }
                    />
                    Verifier Override
                  </label>
                )}
              </>
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

function usePromptLabRunFormController({
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
  const [formState, setFormState] = useState<PromptLabFormState>({
    name: "",
    selectedDocIds: [],
    selectedFolderIds: [],
    prompts: [makePrompt(0)],
    models: [makeModel(0)],
  });
  const [runtimeState, setRuntimeState] = useState<PromptLabRuntimeState>({
    temperature: 0,
    matchMode: "exact",
    referenceSource: "manual",
    fallbackSource: "pre",
    apiKey: readSessionValue("prompt_lab_api_key"),
    apiBase: readSessionValue("prompt_lab_api_base"),
    concurrency: 10,
    labelProfile: "simple",
    labelProjection: "native",
    chunkMode: "off",
    chunkSizeChars: 10000,
  });
  const [uiState, setUiState] = useState<PromptLabUiState>({
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
      if (runtimeState.apiKey) sessionStorage.setItem("prompt_lab_api_key", runtimeState.apiKey);
      else sessionStorage.removeItem("prompt_lab_api_key");
    } catch {
      // sessionStorage unavailable
    }
  }, [runtimeState.apiKey]);

  useEffect(() => {
    try {
      if (runtimeState.apiBase) sessionStorage.setItem("prompt_lab_api_base", runtimeState.apiBase);
      else sessionStorage.removeItem("prompt_lab_api_base");
    } catch {
      // sessionStorage unavailable
    }
  }, [runtimeState.apiBase]);

  const presetMethodOptions = useMemo(
    () =>
      methods.length > 0
        ? methods
        : FALLBACK_PRESET_METHOD_IDS.map((id) => ({
            id,
            label: id,
            description: "",
            requires_presidio: id.includes("presidio"),
            uses_llm: id !== "presidio",
            supports_verify_override: id !== "presidio",
            available: true,
            unavailable_reason: null,
          })),
    [methods],
  );

  const selectedFolderDocCount = formState.selectedFolderIds.reduce(
    (sum, folderId) => sum + (folderDocCountById[folderId] ?? 0),
    0,
  );
  const requestEstimate =
    (formState.selectedDocIds.length + selectedFolderDocCount) *
    formState.prompts.length *
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

  const handleNameChange = (name: string) => {
    setFormState((prev) => ({ ...prev, name }));
  };

  const updatePrompt = (index: number, patch: Partial<PromptLabPromptInput>) => {
    setFormState((prev) => ({
      ...prev,
      prompts: prev.prompts.map((item, i) => (i === index ? { ...item, ...patch } : item)),
    }));
  };

  const handlePromptVariantChange = (index: number, variantType: "prompt" | "preset") => {
    setFormState((prev) => ({
      ...prev,
      prompts: prev.prompts.map((item, i) => {
        if (i !== index) return item;
        if (variantType === "prompt") {
          return {
            ...item,
            variant_type: "prompt",
            preset_method_id: null,
            method_verify_override: null,
            system_prompt: item.system_prompt || DEFAULT_PROMPT,
          };
        }
        return {
          ...item,
          variant_type: "preset",
          system_prompt: undefined,
          preset_method_id: item.preset_method_id || presetMethodOptions[0]?.id || null,
          method_verify_override:
            typeof item.method_verify_override === "boolean"
              ? item.method_verify_override
              : null,
        };
      }),
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

  const handleAddPrompt = () => {
    setFormState((prev) => ({
      ...prev,
      prompts: [...prev.prompts, makePrompt(prev.prompts.length)],
    }));
  };

  const handleRemovePrompt = (index: number) => {
    setFormState((prev) => ({
      ...prev,
      prompts: prev.prompts.filter((_, i) => i !== index),
    }));
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
    if (formState.prompts.length === 0 || formState.prompts.length > 6) {
      return "Prompt variants must be 1 to 6";
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
    for (const prompt of formState.prompts) {
      if (!prompt.label.trim()) {
        return "Every prompt needs a label";
      }
      if ((prompt.variant_type ?? "prompt") === "prompt") {
        const promptText = prompt.system_prompt?.trim() ?? "";
        if (!promptText) {
          return "Every System Prompt variant needs system prompt text";
        }
        if (promptText.includes("{") || promptText.includes("}")) {
          return "Prompt text must be plain system prompt text (no templating)";
        }
      } else if (!(prompt.preset_method_id ?? "").trim()) {
        return "Every Experiment Preset variant needs a preset method";
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

    const payload: PromptLabRunCreateRequest = {
      name: formState.name.trim() || undefined,
      doc_ids: formState.selectedDocIds,
      folder_ids: formState.selectedFolderIds,
      prompts: formState.prompts.map((item, index) => ({
        id: item.id?.trim() || `prompt_${index + 1}`,
        label: item.label.trim(),
        variant_type: item.variant_type ?? "prompt",
        system_prompt:
          (item.variant_type ?? "prompt") === "prompt" ? item.system_prompt ?? "" : undefined,
        preset_method_id:
          (item.variant_type ?? "prompt") === "preset" ? item.preset_method_id ?? null : null,
        method_verify_override:
          (item.variant_type ?? "prompt") === "preset"
            ? item.method_verify_override ?? null
            : null,
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
        fallback_reference_source: runtimeState.fallbackSource,
        label_profile: runtimeState.labelProfile,
        label_projection: runtimeState.labelProjection,
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
    presetMethodOptions,
    requestEstimate,
    selectedTotalDocs,
    isCollapsed,
    updateRuntime,
    toggleCollapsed,
    toggleDoc,
    toggleFolder,
    handleNameChange,
    updatePrompt,
    handlePromptVariantChange,
    handleAddPrompt,
    handleRemovePrompt,
    updateModel,
    handleModelPresetChange,
    handleAddModel,
    handleRemoveModel,
    handleSubmit,
  };
}

export default function PromptLabRunForm(props: Props) {
  const { documents, folders, concurrencyMax, running } = props;
  const {
    formState,
    runtimeState,
    uiState,
    presetMethodOptions,
    requestEstimate,
    selectedTotalDocs,
    isCollapsed,
    updateRuntime,
    toggleCollapsed,
    toggleDoc,
    toggleFolder,
    handleNameChange,
    updatePrompt,
    handlePromptVariantChange,
    handleAddPrompt,
    handleRemovePrompt,
    updateModel,
    handleModelPresetChange,
    handleAddModel,
    handleRemoveModel,
    handleSubmit,
  } = usePromptLabRunFormController(props);

  return (
    <section className={`prompt-lab-run-form ${isCollapsed ? "collapsed" : ""}`}>
      <PromptLabRunHeader
        isCollapsed={isCollapsed}
        requestEstimate={requestEstimate}
        docCount={selectedTotalDocs}
        promptCount={formState.prompts.length}
        modelCount={formState.models.length}
        onToggleCollapsed={toggleCollapsed}
      />

      {!isCollapsed && (
        <>
          <PromptLabConfigGrid
            name={formState.name}
            runtime={runtimeState}
            concurrencyMax={concurrencyMax}
            onNameChange={handleNameChange}
            onRuntimeChange={updateRuntime}
          />
          <PromptLabDocumentsSection
            documents={documents}
            folders={folders}
            selectedDocIds={formState.selectedDocIds}
            selectedFolderIds={formState.selectedFolderIds}
            onToggleDoc={toggleDoc}
            onToggleFolder={toggleFolder}
          />
          <PromptVariantsSection
            prompts={formState.prompts}
            presetMethodOptions={presetMethodOptions}
            onAddPrompt={handleAddPrompt}
            onRemovePrompt={handleRemovePrompt}
            onUpdatePrompt={updatePrompt}
            onPromptVariantChange={handlePromptVariantChange}
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
              {running || uiState.submitting ? "Starting..." : "Run Prompt Lab"}
            </button>
          </div>
        </>
      )}
    </section>
  );
}
