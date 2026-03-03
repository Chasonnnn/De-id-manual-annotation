import { useEffect, useMemo, useState } from "react";
import { MODEL_PRESETS, getModelPreset } from "../modelPresets";
import type {
  DocumentSummary,
  PromptLabModelInput,
  PromptLabPromptInput,
  PromptLabRunCreateRequest,
} from "../types";

interface Props {
  documents: DocumentSummary[];
  selectedDocumentId: string | null;
  onRun: (payload: PromptLabRunCreateRequest) => Promise<void>;
  running: boolean;
}

const REASONING_EFFORT_OPTIONS = ["none", "low", "medium", "high", "xhigh"] as const;

const DEFAULT_PROMPT =
  'You are a PII annotation assistant. Return ONLY a JSON array of objects with start (0-based), end (exclusive), label, and text for each PII span.';

function makePrompt(index: number): PromptLabPromptInput {
  return {
    id: `prompt_${index + 1}`,
    label: index === 0 ? "Baseline" : `Prompt ${index + 1}`,
    system_prompt: DEFAULT_PROMPT,
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

export default function PromptLabRunForm({
  documents,
  selectedDocumentId,
  onRun,
  running,
}: Props) {
  const [name, setName] = useState("");
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);
  const [prompts, setPrompts] = useState<PromptLabPromptInput[]>([makePrompt(0)]);
  const [models, setModels] = useState<PromptLabModelInput[]>([makeModel(0)]);
  const [temperature, setTemperature] = useState(0);
  const [matchMode, setMatchMode] = useState<"exact" | "overlap">("exact");
  const [referenceSource, setReferenceSource] = useState<"manual" | "pre">("manual");
  const [fallbackSource, setFallbackSource] = useState<"manual" | "pre">("pre");
  const [apiKey, setApiKey] = useState(() => {
    try {
      return sessionStorage.getItem("prompt_lab_api_key") ?? "";
    } catch {
      return "";
    }
  });
  const [apiBase, setApiBase] = useState(() => {
    try {
      return sessionStorage.getItem("prompt_lab_api_base") ?? "";
    } catch {
      return "";
    }
  });
  const [concurrency, setConcurrency] = useState(4);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [collapsed, setCollapsed] = useState(false);

  useEffect(() => {
    if (!selectedDocumentId) return;
    setSelectedDocIds((prev) => (prev.length > 0 ? prev : [selectedDocumentId]));
  }, [selectedDocumentId]);

  useEffect(() => {
    try {
      if (apiKey) sessionStorage.setItem("prompt_lab_api_key", apiKey);
      else sessionStorage.removeItem("prompt_lab_api_key");
    } catch {
      // sessionStorage unavailable
    }
  }, [apiKey]);

  useEffect(() => {
    try {
      if (apiBase) sessionStorage.setItem("prompt_lab_api_base", apiBase);
      else sessionStorage.removeItem("prompt_lab_api_base");
    } catch {
      // sessionStorage unavailable
    }
  }, [apiBase]);

  const requestEstimate = useMemo(
    () => selectedDocIds.length * prompts.length * models.length,
    [selectedDocIds.length, prompts.length, models.length],
  );

  const toggleDoc = (docId: string) => {
    setSelectedDocIds((prev) =>
      prev.includes(docId) ? prev.filter((id) => id !== docId) : [...prev, docId],
    );
  };

  const updatePrompt = (index: number, patch: Partial<PromptLabPromptInput>) => {
    setPrompts((prev) => prev.map((item, i) => (i === index ? { ...item, ...patch } : item)));
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
    if (prompts.length === 0 || prompts.length > 6) return "Prompt variants must be 1 to 6";
    if (models.length === 0 || models.length > 6) return "Model variants must be 1 to 6";
    if (concurrency < 1 || concurrency > 6) return "Concurrency must be 1 to 6";
    for (const prompt of prompts) {
      if (!prompt.label.trim() || !prompt.system_prompt.trim()) {
        return "Every prompt needs a label and system prompt";
      }
      if (prompt.system_prompt.includes("{") || prompt.system_prompt.includes("}")) {
        return "Prompt text must be plain system prompt text (no templating)";
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
    const payload: PromptLabRunCreateRequest = {
      name: name.trim() || undefined,
      doc_ids: selectedDocIds,
      prompts: prompts.map((item, index) => ({
        id: item.id?.trim() || `prompt_${index + 1}`,
        label: item.label.trim(),
        system_prompt: item.system_prompt,
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
        reference_source: referenceSource,
        fallback_reference_source: fallbackSource,
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
        <h3>Prompt Lab Setup</h3>
        <div className="prompt-lab-estimate">
          Requests: <strong>{requestEstimate}</strong> ({selectedDocIds.length} docs × {prompts.length} prompts × {models.length} models)
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
              <label htmlFor="prompt-lab-name">Run Name (optional)</label>
              <input
                id="prompt-lab-name"
                type="text"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="Prompt Lab experiment"
              />
            </div>

            <div className="prompt-lab-field">
              <label htmlFor="prompt-lab-temp">Temperature ({temperature.toFixed(1)})</label>
              <input
                id="prompt-lab-temp"
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={temperature}
                onChange={(e) => setTemperature(Number.parseFloat(e.target.value))}
              />
            </div>

            <div className="prompt-lab-field prompt-lab-inline">
              <label htmlFor="prompt-lab-match">Match</label>
              <select
                id="prompt-lab-match"
                value={matchMode}
                onChange={(e) => setMatchMode(e.target.value as "exact" | "overlap")}
              >
                <option value="exact">Exact</option>
                <option value="overlap">Overlap</option>
              </select>
            </div>

            <div className="prompt-lab-field prompt-lab-inline">
              <label htmlFor="prompt-lab-reference">Reference</label>
              <select
                id="prompt-lab-reference"
                value={referenceSource}
                onChange={(e) => setReferenceSource(e.target.value as "manual" | "pre")}
              >
                <option value="manual">Manual</option>
                <option value="pre">Pre-annotations</option>
              </select>
            </div>

            <div className="prompt-lab-field prompt-lab-inline">
              <label htmlFor="prompt-lab-fallback">Fallback</label>
              <select
                id="prompt-lab-fallback"
                value={fallbackSource}
                onChange={(e) => setFallbackSource(e.target.value as "manual" | "pre")}
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
                max={6}
                value={concurrency}
                onChange={(e) => setConcurrency(Number.parseInt(e.target.value, 10) || 1)}
              />
            </div>

            <div className="prompt-lab-field">
              <label htmlFor="prompt-lab-api-key">API Key (optional override)</label>
              <input
                id="prompt-lab-api-key"
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="Uses env key when not set"
              />
            </div>

            <div className="prompt-lab-field">
              <label htmlFor="prompt-lab-api-base">LiteLLM Base URL (optional override)</label>
              <input
                id="prompt-lab-api-base"
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
              <span>Prompt Variants</span>
              <button
                type="button"
                onClick={() => setPrompts((prev) => [...prev, makePrompt(prev.length)])}
                disabled={prompts.length >= 6}
              >
                + Add Prompt
              </button>
            </div>
            {prompts.map((prompt, index) => (
              <div key={prompt.id ?? index} className="prompt-lab-card">
                <div className="prompt-lab-card-header">
                  <input
                    type="text"
                    value={prompt.label}
                    onChange={(e) => updatePrompt(index, { label: e.target.value })}
                    placeholder="Prompt label"
                  />
                  <button
                    type="button"
                    onClick={() => setPrompts((prev) => prev.filter((_, i) => i !== index))}
                    disabled={prompts.length <= 1}
                  >
                    Remove
                  </button>
                </div>
                <textarea
                  value={prompt.system_prompt}
                  onChange={(e) => updatePrompt(index, { system_prompt: e.target.value })}
                  rows={4}
                />
              </div>
            ))}
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
                        onChange={(e) => updateModel(index, { anthropic_thinking: e.target.checked })}
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
              {running || submitting ? "Starting..." : "Run Prompt Lab"}
            </button>
          </div>
        </>
      )}
    </section>
  );
}
