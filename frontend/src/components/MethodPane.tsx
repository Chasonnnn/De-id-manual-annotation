import { forwardRef, useEffect, useState } from "react";
import { getAgentCredentialStatus } from "../api/client";
import { MODEL_PRESETS, getModelPreset } from "../modelPresets";
import type {
  AgentConfig,
  AgentCredentialStatus,
  AgentMethodOption,
  CanonicalSpan,
  LabelProfile,
  MethodView,
} from "../types";
import AnnotatedText from "./AnnotatedText";

interface Props {
  text: string;
  spans: CanonicalSpan[];
  methods: AgentMethodOption[];
  processedWithChunking?: boolean;
  activeMethod: MethodView;
  onActiveMethodChange: (methodId: MethodView) => void;
  diffSpans?: { start: number; end: number; type: "added" | "removed" }[];
  onRunMethod: (config: AgentConfig) => Promise<void>;
  running: boolean;
  onScroll: (scrollTop: number) => void;
}

const MODEL_GROUPS = ["OpenAI", "Anthropic", "Google Gemini"] as const;
const REASONING_EFFORT_OPTIONS = ["none", "low", "medium", "high", "xhigh"] as const;

const MethodPane = forwardRef<HTMLDivElement, Props>(
  (
    {
      text,
      spans,
      methods,
      processedWithChunking = false,
      activeMethod,
      onActiveMethodChange,
      diffSpans = [],
      onRunMethod,
      running,
      onScroll,
    },
    ref,
  ) => {
    const [configOpen, setConfigOpen] = useState(false);
    const [systemPrompt, setSystemPrompt] = useState(
      "Identify all PII in the transcript. Label each span with: NAME, LOCATION, SCHOOL, DATE, AGE, PHONE, EMAIL, URL, or MISC_ID.",
    );
    const [model, setModel] = useState("openai.gpt-5.3-codex");
    const [customModel, setCustomModel] = useState("");
    const [temperature, setTemperature] = useState(0);
    const [labelProfile, setLabelProfile] = useState<LabelProfile>("simple");
    const [chunkMode, setChunkMode] = useState<"auto" | "off" | "force">("auto");
    const [chunkSizeChars, setChunkSizeChars] = useState(10000);
    const [reasoningEffort, setReasoningEffort] = useState<
      "none" | "low" | "medium" | "high" | "xhigh"
    >("xhigh");
    const [anthropicThinking, setAnthropicThinking] = useState(false);
    const [anthropicThinkingBudget, setAnthropicThinkingBudget] = useState(2048);
    const [methodVerify, setMethodVerify] = useState(false);
    const [apiKey, setApiKey] = useState(() => {
      try {
        return sessionStorage.getItem("agent_api_key") ?? "";
      } catch {
        return "";
      }
    });
    const [apiBase, setApiBase] = useState(() => {
      try {
        return sessionStorage.getItem("agent_api_base") ?? "";
      } catch {
        return "";
      }
    });
    const [credentialStatus, setCredentialStatus] = useState<AgentCredentialStatus | null>(
      null,
    );
    const [showApiKeyInput, setShowApiKeyInput] = useState(() => Boolean(apiKey));
    const [showApiBaseInput, setShowApiBaseInput] = useState(() => Boolean(apiBase));

    useEffect(() => {
      getAgentCredentialStatus()
        .then((status) => {
          setCredentialStatus(status);
          if (status.has_api_key && !apiKey) {
            setShowApiKeyInput(false);
          }
          if (status.has_api_base && !apiBase) {
            setShowApiBaseInput(false);
          }
        })
        .catch(() => {
          setCredentialStatus(null);
        });
      // Load once; local overrides are tracked separately in component state.
      // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    useEffect(() => {
      const selected = methods.find((method) => method.id === activeMethod);
      if (selected?.supports_verify_override) {
        setMethodVerify(Boolean(selected.id === "verified"));
      } else {
        setMethodVerify(false);
      }
    }, [activeMethod, methods]);

    const selectedMethod = methods.find((method) => method.id === activeMethod) ?? null;
    const methodAvailable = selectedMethod?.available ?? false;
    const methodUsesLlm = selectedMethod?.uses_llm ?? true;
    const supportsVerifyOverride = selectedMethod?.supports_verify_override ?? false;

    const effectiveModel = model === "__custom__" ? customModel.trim() : model;
    const selectedPreset =
      model === "__custom__" ? undefined : getModelPreset(effectiveModel);
    const supportsReasoningEffort = selectedPreset?.supportsReasoningEffort ?? false;
    const supportsAnthropicThinking = selectedPreset?.supportsAnthropicThinking ?? false;

    const handleApiKeyChange = (value: string) => {
      setApiKey(value);
      try {
        if (value) {
          sessionStorage.setItem("agent_api_key", value);
        } else {
          sessionStorage.removeItem("agent_api_key");
        }
      } catch {
        // sessionStorage unavailable
      }
    };

    const handleApiBaseChange = (value: string) => {
      setApiBase(value);
      try {
        if (value) {
          sessionStorage.setItem("agent_api_base", value);
        } else {
          sessionStorage.removeItem("agent_api_base");
        }
      } catch {
        // sessionStorage unavailable
      }
    };

    const handleRun = () => {
      if (!selectedMethod) return;
      const payload: AgentConfig = {
        mode: "method",
        method_id: selectedMethod.id,
        label_profile: labelProfile,
      };
      if (supportsVerifyOverride) {
        payload.method_verify = methodVerify;
      }
      if (methodUsesLlm) {
        payload.system_prompt = systemPrompt;
        payload.model = effectiveModel;
        payload.temperature = temperature;
        payload.api_key = apiKey || undefined;
        payload.api_base = apiBase || undefined;
        payload.reasoning_effort = reasoningEffort;
        payload.anthropic_thinking = anthropicThinking;
        payload.anthropic_thinking_budget_tokens = anthropicThinking
          ? anthropicThinkingBudget
          : undefined;
        payload.chunk_mode = chunkMode;
        payload.chunk_size_chars = chunkSizeChars;
      }
      void onRunMethod(payload);
    };

    const runDisabled =
      running || !selectedMethod || !methodAvailable || (methodUsesLlm && !effectiveModel);

    return (
      <div className="pane">
        <div className="pane-header pane-header-agent">
          <div className="pane-header-agent-left">
            <span>Method Annotations</span>
            {processedWithChunking && (
              <span className="chunk-badge" title="Backend auto-chunked this run for reliability.">
                Processed with chunking
              </span>
            )}
            <span className="agent-view-control">
              <label htmlFor="method-view-select">View:</label>
              <select
                id="method-view-select"
                name="method_view"
                value={activeMethod}
                onChange={(e) => onActiveMethodChange(e.target.value)}
              >
                {methods.map((method) => (
                  <option key={method.id} value={method.id}>
                    {method.label}
                  </option>
                ))}
              </select>
            </span>
          </div>
          <button className="config-toggle" onClick={() => setConfigOpen(!configOpen)}>
            {configOpen ? "Hide Config" : "Show Config"}
          </button>
        </div>
        <div className={`agent-config ${configOpen ? "" : "collapsed"}`}>
          <div className="field">
            <label htmlFor="method-select">Method</label>
            <select
              id="method-select"
              name="method_select"
              value={activeMethod}
              onChange={(e) => onActiveMethodChange(e.target.value)}
            >
              {methods.map((method) => (
                <option key={method.id} value={method.id} disabled={!method.available}>
                  {method.label}
                  {!method.available ? " (setup required)" : ""}
                </option>
              ))}
            </select>
            {selectedMethod && (
              <span className="config-note">
                {selectedMethod.description}
              </span>
            )}
            {selectedMethod?.unavailable_reason && (
              <span className="config-warning">
                {selectedMethod.unavailable_reason}
              </span>
            )}
          </div>
          {supportsVerifyOverride && (
            <div className="field">
              <label className="inline-checkbox-label">
                <input
                  type="checkbox"
                  checked={methodVerify}
                  onChange={(e) => setMethodVerify(e.target.checked)}
                />
                Method Verifier
              </label>
            </div>
          )}
          {methodUsesLlm && (
            <>
              <div className="field">
                <label htmlFor="method-system-prompt">System Prompt</label>
                <textarea
                  id="method-system-prompt"
                  name="method_system_prompt"
                  value={systemPrompt}
                  onChange={(e) => setSystemPrompt(e.target.value)}
                />
              </div>
              <div className="field">
                <label htmlFor="method-model">Model</label>
                <select
                  id="method-model"
                  name="method_model"
                  value={model}
                  onChange={(e) => {
                    const value = e.target.value;
                    setModel(value);
                    const preset = value === "__custom__" ? undefined : getModelPreset(value);
                    setReasoningEffort(preset?.defaultReasoningEffort ?? "none");
                  }}
                >
                  {MODEL_GROUPS.map((group) => (
                    <optgroup key={group} label={group}>
                      {MODEL_PRESETS.filter((preset) => preset.group === group).map((opt) => (
                        <option key={opt.id} value={opt.id}>
                          {opt.label}
                        </option>
                      ))}
                    </optgroup>
                  ))}
                  <option value="__custom__">Custom model...</option>
                </select>
                {model === "__custom__" && (
                  <input
                    id="method-custom-model"
                    name="method_custom_model"
                    type="text"
                    value={customModel}
                    onChange={(e) => setCustomModel(e.target.value)}
                    placeholder="provider.model-name (e.g., openai.gpt-5.3-codex)"
                    style={{ marginTop: 4 }}
                  />
                )}
              </div>
              <div className={`field ${supportsReasoningEffort ? "" : "field-disabled"}`}>
                <label htmlFor="method-reasoning-effort">
                  Reasoning Effort
                  {!supportsReasoningEffort && " (not supported for this model)"}
                </label>
                <select
                  id="method-reasoning-effort"
                  name="method_reasoning_effort"
                  value={reasoningEffort}
                  disabled={!supportsReasoningEffort}
                  onChange={(e) =>
                    setReasoningEffort(
                      e.target.value as "none" | "low" | "medium" | "high" | "xhigh",
                    )
                  }
                >
                  {REASONING_EFFORT_OPTIONS.map((value) => (
                    <option key={value} value={value}>
                      {value}
                    </option>
                  ))}
                </select>
              </div>
              <div className={`field ${supportsAnthropicThinking ? "" : "field-disabled"}`}>
                <label htmlFor="method-anthropic-thinking">
                  Anthropic Thinking
                  {!supportsAnthropicThinking && " (not supported for this model)"}
                </label>
                <div className="inline-control-row">
                  <input
                    id="method-anthropic-thinking"
                    name="method_anthropic_thinking"
                    type="checkbox"
                    checked={anthropicThinking}
                    disabled={!supportsAnthropicThinking}
                    onChange={(e) => setAnthropicThinking(e.target.checked)}
                  />
                  <input
                    id="method-anthropic-thinking-budget"
                    name="method_anthropic_thinking_budget_tokens"
                    type="number"
                    min={256}
                    step={256}
                    disabled={!supportsAnthropicThinking || !anthropicThinking}
                    value={anthropicThinkingBudget}
                    onChange={(e) =>
                      setAnthropicThinkingBudget(Number.parseInt(e.target.value, 10) || 2048)
                    }
                    placeholder="Budget tokens"
                  />
                </div>
              </div>
              <div className="field">
                <label htmlFor="method-temperature">Temperature: {temperature}</label>
                <input
                  id="method-temperature"
                  name="method_temperature"
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                />
              </div>
              <div className="field">
                <label htmlFor="method-label-profile">Label Profile</label>
                <select
                  id="method-label-profile"
                  name="method_label_profile"
                  value={labelProfile}
                  onChange={(e) =>
                    setLabelProfile(e.target.value as LabelProfile)
                  }
                >
                  <option value="simple">Simple</option>
                  <option value="advanced">Advanced (UPchieve)</option>
                </select>
              </div>
              <div className="field">
                <label htmlFor="method-chunk-mode">Chunk Mode</label>
                <select
                  id="method-chunk-mode"
                  name="method_chunk_mode"
                  value={chunkMode}
                  onChange={(e) =>
                    setChunkMode(e.target.value as "auto" | "off" | "force")
                  }
                >
                  <option value="auto">Auto</option>
                  <option value="off">Off</option>
                  <option value="force">Force</option>
                </select>
              </div>
              {chunkMode !== "off" && (
                <div className="field">
                  <label htmlFor="method-chunk-size">Chunk Size (chars)</label>
                  <input
                    id="method-chunk-size"
                    name="method_chunk_size_chars"
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
              <div className="field">
                {credentialStatus?.has_api_key && !showApiKeyInput && !apiKey ? (
                  <>
                    <div className="field-label">API Key</div>
                    <div style={{ fontSize: 12, color: "#666", lineHeight: 1.4 }}>
                      Server env key detected ({credentialStatus.api_key_sources.join(", ")}).
                      <button
                        type="button"
                        onClick={() => setShowApiKeyInput(true)}
                        style={{
                          marginLeft: 8,
                          border: "none",
                          background: "none",
                          color: "#4a6cf7",
                          cursor: "pointer",
                          padding: 0,
                        }}
                      >
                        Override key
                      </button>
                    </div>
                  </>
                ) : (
                  <>
                    <label htmlFor="method-api-key">API Key</label>
                    <input
                      id="method-api-key"
                      name="method_api_key"
                      type="password"
                      value={apiKey}
                      onChange={(e) => handleApiKeyChange(e.target.value)}
                      placeholder="LiteLLM gateway key or provider key"
                    />
                    <span style={{ fontSize: 10, color: "#888" }}>
                      Uses LITELLM_API_KEY first, then provider env vars as fallback
                    </span>
                  </>
                )}
              </div>
              <div className="field">
                {credentialStatus?.has_api_base && !showApiBaseInput && !apiBase ? (
                  <>
                    <div className="field-label">LiteLLM Base URL (optional)</div>
                    <div style={{ fontSize: 12, color: "#666", lineHeight: 1.4 }}>
                      Server env base URL detected ({credentialStatus.api_base_sources.join(", ")}).
                      <button
                        type="button"
                        onClick={() => setShowApiBaseInput(true)}
                        style={{
                          marginLeft: 8,
                          border: "none",
                          background: "none",
                          color: "#4a6cf7",
                          cursor: "pointer",
                          padding: 0,
                        }}
                      >
                        Override base URL
                      </button>
                    </div>
                  </>
                ) : (
                  <>
                    <label htmlFor="method-api-base">LiteLLM Base URL (optional)</label>
                    <input
                      id="method-api-base"
                      name="method_api_base"
                      type="text"
                      value={apiBase}
                      onChange={(e) => handleApiBaseChange(e.target.value)}
                      placeholder="https://your-litellm-gateway/v1"
                    />
                    <span style={{ fontSize: 10, color: "#888" }}>
                      Uses LITELLM_BASE_URL env var when not set here
                    </span>
                  </>
                )}
              </div>
            </>
          )}
          <button className="run-btn" onClick={handleRun} disabled={runDisabled}>
            {running ? "Running..." : "Run Method"}
          </button>
        </div>
        <div
          className="pane-body"
          ref={ref}
          onScroll={(e) => onScroll((e.target as HTMLDivElement).scrollTop)}
        >
          {spans.length === 0 && !running ? (
            <span style={{ color: "#888" }}>
              No method annotations yet for the selected method.
            </span>
          ) : (
            <AnnotatedText text={text} spans={spans} diffSpans={diffSpans} />
          )}
        </div>
      </div>
    );
  },
);

MethodPane.displayName = "MethodPane";
export default MethodPane;
