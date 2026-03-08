import { useEffect, useMemo, useState } from "react";
import { MODEL_PRESETS, getModelPreset } from "../modelPresets";
import type {
  AgentMethodOption,
  DocumentSummary,
  MatchMode,
  MethodsLabMethodInput,
  MethodsLabRunCreateRequest,
  PromptLabModelInput,
} from "../types";

interface Props {
  documents: DocumentSummary[];
  selectedDocumentId: string | null;
  methods: AgentMethodOption[];
  onRun: (payload: MethodsLabRunCreateRequest) => Promise<void>;
  running: boolean;
  forceCollapsed?: boolean;
}

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

function makeMethod(
  index: number,
  methodOptions: AgentMethodOption[],
): MethodsLabMethodInput {
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

export default function MethodsLabRunForm({
  documents,
  selectedDocumentId,
  methods,
  onRun,
  running,
  forceCollapsed = false,
}: Props) {
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
  const [name, setName] = useState("");
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);
  const [methodVariants, setMethodVariants] = useState<MethodsLabMethodInput[]>([
    makeMethod(0, methodOptions),
  ]);
  const [models, setModels] = useState<PromptLabModelInput[]>([makeModel(0)]);
  const [temperature, setTemperature] = useState(0);
  const [matchMode, setMatchMode] = useState<MatchMode>("exact");
  const [apiKey, setApiKey] = useState(() => {
    try {
      return sessionStorage.getItem("methods_lab_api_key") ?? "";
    } catch {
      return "";
    }
  });
  const [apiBase, setApiBase] = useState(() => {
    try {
      return sessionStorage.getItem("methods_lab_api_base") ?? "";
    } catch {
      return "";
    }
  });
  const [concurrency, setConcurrency] = useState(4);
  const [labelProfile, setLabelProfile] = useState<"simple" | "advanced">("simple");
  const [labelProjection, setLabelProjection] = useState<"native" | "coarse_simple">(
    "native",
  );
  const [chunkMode, setChunkMode] = useState<"auto" | "off" | "force">("auto");
  const [chunkSizeChars, setChunkSizeChars] = useState(10000);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [collapsed, setCollapsed] = useState(false);

  useEffect(() => {
    if (forceCollapsed) {
      setCollapsed(true);
    }
  }, [forceCollapsed]);

  useEffect(() => {
    if (!selectedDocumentId) return;
    setSelectedDocIds((prev) => (prev.length > 0 ? prev : [selectedDocumentId]));
  }, [selectedDocumentId]);

  useEffect(() => {
    try {
      if (apiKey) sessionStorage.setItem("methods_lab_api_key", apiKey);
      else sessionStorage.removeItem("methods_lab_api_key");
    } catch {
      // sessionStorage unavailable
    }
  }, [apiKey]);

  useEffect(() => {
    try {
      if (apiBase) sessionStorage.setItem("methods_lab_api_base", apiBase);
      else sessionStorage.removeItem("methods_lab_api_base");
    } catch {
      // sessionStorage unavailable
    }
  }, [apiBase]);

  useEffect(() => {
    setMethodVariants((prev) => {
      if (prev.length > 0) return prev;
      return [makeMethod(0, methodOptions)];
    });
  }, [methodOptions]);

  const requestEstimate = useMemo(
    () => selectedDocIds.length * methodVariants.length * models.length,
    [selectedDocIds.length, methodVariants.length, models.length],
  );

  const toggleDoc = (docId: string) => {
    setSelectedDocIds((prev) =>
      prev.includes(docId) ? prev.filter((id) => id !== docId) : [...prev, docId],
    );
  };

  const updateMethodVariant = (index: number, patch: Partial<MethodsLabMethodInput>) => {
    setMethodVariants((prev) =>
      prev.map((item, i) => (i === index ? { ...item, ...patch } : item)),
    );
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
    setMethodVariants(
      available.map((method, index) => ({
        id: `method_${index + 1}`,
        label: method.label,
        method_id: method.id,
        method_verify_override:
          method.supports_verify_override && typeof method.default_verify === "boolean"
            ? method.default_verify
            : null,
      })),
    );
  };

  const updateModel = (index: number, patch: Partial<PromptLabModelInput>) => {
    setModels((prev) => prev.map((item, i) => (i === index ? { ...item, ...patch } : item)));
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

  const validate = (): string | null => {
    if (selectedDocIds.length === 0) return "Select at least one document";
    if (methodVariants.length === 0 || methodVariants.length > MAX_METHOD_VARIANTS) {
      return `Method variants must be 1 to ${MAX_METHOD_VARIANTS}`;
    }
    if (models.length === 0 || models.length > 6) return "Model variants must be 1 to 6";
    if (concurrency < 1 || concurrency > 6) return "Concurrency must be 1 to 6";
    if (chunkSizeChars < 2000 || chunkSizeChars > 30000) {
      return "Chunk size must be between 2000 and 30000";
    }
    const seenIds = new Set<string>();
    for (const method of methodVariants) {
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
    for (const model of models) {
      if (!model.label.trim() || !model.model?.trim()) {
        return "Every model needs a label and model id";
      }
    }
    return null;
  };

  const handleSubmit = async () => {
    const validationError = validate();
    setError(validationError);
    if (validationError) return;
    const payload: MethodsLabRunCreateRequest = {
      name: name.trim() || undefined,
      doc_ids: selectedDocIds,
      methods: methodVariants.map((item, index) => ({
        id: item.id?.trim() || `method_${index + 1}`,
        label: item.label.trim(),
        method_id: item.method_id.trim(),
        method_verify_override: item.method_verify_override ?? null,
      })),
      models: models.map((item, index) => ({
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
        api_key: apiKey || undefined,
        api_base: apiBase || undefined,
        temperature,
        match_mode: matchMode,
        label_profile: labelProfile,
        label_projection: labelProjection,
        chunk_mode: chunkMode,
        chunk_size_chars: chunkSizeChars,
      },
      concurrency,
    };

    setSubmitting(true);
    try {
      await onRun(payload);
      setError(null);
      setCollapsed(true);
    } catch (e: unknown) {
      setError(String(e));
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <section className={`prompt-lab-run-form ${collapsed ? "collapsed" : ""}`}>
      <div className="prompt-lab-run-header">
        <h3>Methods Lab Setup</h3>
        <div className="prompt-lab-estimate">
          Requests: <strong>{requestEstimate}</strong> ({selectedDocIds.length} docs × {methodVariants.length} methods × {models.length} models)
        </div>
        <button
          type="button"
          className="prompt-lab-toggle-btn"
          onClick={() => setCollapsed((prev) => !prev)}
        >
          {collapsed ? "Show Setup" : "Hide Setup"}
        </button>
      </div>

      {!collapsed && (
        <>
          <div className="prompt-lab-form-grid">
            <div className="prompt-lab-field">
              <label htmlFor="methods-lab-name">Run Name (optional)</label>
              <input
                id="methods-lab-name"
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Methods Lab experiment"
              />
            </div>

            <div className="prompt-lab-field">
              <label htmlFor="methods-lab-temp">Temperature ({temperature.toFixed(1)})</label>
              <input
                id="methods-lab-temp"
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(Number.parseFloat(e.target.value))}
              />
            </div>

            <div className="prompt-lab-field prompt-lab-inline">
              <label htmlFor="methods-lab-match">Match</label>
              <select
                id="methods-lab-match"
                value={matchMode}
                onChange={(e) => setMatchMode(e.target.value as MatchMode)}
              >
                <option value="exact">Exact</option>
                <option value="boundary">Trim Space/Punct</option>
                <option value="overlap">Overlap</option>
              </select>
            </div>

            <div className="prompt-lab-field prompt-lab-inline">
              <label htmlFor="methods-lab-concurrency">Concurrency</label>
              <input
                id="methods-lab-concurrency"
                type="number"
                min={1}
                max={6}
                value={concurrency}
                onChange={(e) => setConcurrency(Number.parseInt(e.target.value, 10) || 1)}
              />
            </div>

            <div className="prompt-lab-field prompt-lab-inline">
              <label htmlFor="methods-lab-label-profile">Label Profile</label>
              <select
                id="methods-lab-label-profile"
                value={labelProfile}
                onChange={(e) => setLabelProfile(e.target.value as "simple" | "advanced")}
              >
                <option value="simple">Simple</option>
                <option value="advanced">Advanced (UPchieve)</option>
              </select>
            </div>

            <div className="prompt-lab-field prompt-lab-inline">
              <label htmlFor="methods-lab-label-projection">Label Compare</label>
              <select
                id="methods-lab-label-projection"
                value={labelProjection}
                onChange={(e) =>
                  setLabelProjection(e.target.value as "native" | "coarse_simple")
                }
              >
                <option value="native">Native</option>
                <option value="coarse_simple">Coarse (advanced→simple)</option>
              </select>
            </div>

            <div className="prompt-lab-field prompt-lab-inline">
              <label htmlFor="methods-lab-chunk-mode">Chunk Mode</label>
              <select
                id="methods-lab-chunk-mode"
                value={chunkMode}
                onChange={(e) => setChunkMode(e.target.value as "auto" | "off" | "force")}
              >
                <option value="auto">Auto</option>
                <option value="off">Off</option>
                <option value="force">Force</option>
              </select>
            </div>

            {chunkMode !== "off" && (
              <div className="prompt-lab-field prompt-lab-inline">
                <label htmlFor="methods-lab-chunk-size">Chunk Size (chars)</label>
                <input
                  id="methods-lab-chunk-size"
                  type="number"
                  min={2000}
                  max={30000}
                  step={500}
                  value={chunkSizeChars}
                  onChange={(e) =>
                    setChunkSizeChars(Number.parseInt(e.target.value, 10) || 10000)
                  }
                />
              </div>
            )}

            <div className="prompt-lab-field">
              <label htmlFor="methods-lab-api-key">API Key (optional override)</label>
              <input
                id="methods-lab-api-key"
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="Uses env key when not set"
              />
            </div>

            <div className="prompt-lab-field">
              <label htmlFor="methods-lab-api-base">LiteLLM Base URL (optional override)</label>
              <input
                id="methods-lab-api-base"
                type="text"
                value={apiBase}
                onChange={(e) => setApiBase(e.target.value)}
                placeholder="https://your-gateway/v1"
              />
            </div>
          </div>

          <div className="prompt-lab-section">
            <div className="prompt-lab-section-header">
              <span>Documents</span>
              <small>{selectedDocIds.length} selected</small>
            </div>
            <div className="prompt-lab-config-note">
              Methods Lab scores against manual annotations only. Documents without manual annotations are marked unavailable.
            </div>
            <div className="prompt-lab-doc-grid">
              {documents.map((doc) => (
                <label key={doc.id} className="prompt-lab-doc-option">
                  <input
                    type="checkbox"
                    checked={selectedDocIds.includes(doc.id)}
                    onChange={() => toggleDoc(doc.id)}
                  />
                  <span title={doc.filename}>{doc.filename}</span>
                </label>
              ))}
            </div>
          </div>

          <div className="prompt-lab-section">
            <div className="prompt-lab-section-header">
              <span>Method Variants</span>
              <div>
                <button
                  type="button"
                  onClick={addAllAvailableMethods}
                  disabled={methodOptions.filter((item) => item.available).length === 0}
                >
                  Add All Available
                </button>
                <button
                  type="button"
                  onClick={() =>
                    setMethodVariants((prev) => [...prev, makeMethod(prev.length, methodOptions)])
                  }
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
                      onChange={(e) => updateMethodVariant(index, { label: e.target.value })}
                      placeholder="Method label"
                    />
                    <button
                      type="button"
                      onClick={() =>
                        setMethodVariants((prev) => prev.filter((_, i) => i !== index))
                      }
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
                        onChange={(e) => handleMethodPresetChange(index, e.target.value)}
                      >
                        {methodOptions.map((option) => (
                          <option
                            key={option.id}
                            value={option.id}
                            disabled={!option.available}
                          >
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
                      {selectedMethod.unavailable_reason
                        ? ` ${selectedMethod.unavailable_reason}`
                        : ""}
                    </div>
                  )}

                  {selectedMethod?.supports_verify_override && (
                    <label className="inline-checkbox-label">
                      <input
                        type="checkbox"
                        checked={Boolean(method.method_verify_override)}
                        onChange={(e) =>
                          updateMethodVariant(index, {
                            method_verify_override: e.target.checked,
                          })
                        }
                      />
                      Verifier Override
                    </label>
                  )}
                </div>
              );
            })}
          </div>

          <div className="prompt-lab-section">
            <div className="prompt-lab-section-header">
              <span>Model Variants</span>
              <button
                type="button"
                onClick={() => setModels((prev) => [...prev, makeModel(prev.length)])}
                disabled={models.length >= 6}
              >
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
                      onChange={(e) => updateModel(index, { label: e.target.value })}
                      placeholder="Model label"
                    />
                    <button
                      type="button"
                      onClick={() => setModels((prev) => prev.filter((_, i) => i !== index))}
                      disabled={models.length <= 1}
                    >
                      Remove
                    </button>
                  </div>

                  <div className="prompt-lab-model-row">
                    <select
                      value={presetValue}
                      onChange={(e) => handleModelPresetChange(index, e.target.value)}
                    >
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
                        onChange={(e) => updateModel(index, { model: e.target.value })}
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
                          updateModel(index, {
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
                          updateModel(index, { anthropic_thinking: e.target.checked })
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
                        updateModel(index, {
                          anthropic_thinking_budget_tokens:
                            Number.parseInt(e.target.value, 10) || 2048,
                        })
                      }
                      placeholder="Thinking budget"
                    />
                  </div>
                </div>
              );
            })}
          </div>

          {error && <div className="prompt-lab-error">{error}</div>}

          <div className="prompt-lab-actions">
            <button type="button" onClick={handleSubmit} disabled={running || submitting}>
              {running || submitting ? "Starting..." : "Run Methods Lab"}
            </button>
          </div>
        </>
      )}
    </section>
  );
}
