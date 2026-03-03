import { forwardRef, useEffect, useState } from "react";
import { getAgentCredentialStatus } from "../api/client";
import { MODEL_PRESETS, getModelPreset } from "../modelPresets";
import type {
  AgentConfig,
  AgentCredentialStatus,
  AgentView,
  CanonicalSpan,
  LabelProfile,
} from "../types";
import AnnotatedText from "./AnnotatedText";

interface Props {
  text: string;
  spans: CanonicalSpan[];
  processedWithChunking?: boolean;
  activeOutput: AgentView;
  onActiveOutputChange: (view: AgentView) => void;
  diffSpans?: { start: number; end: number; type: "added" | "removed" }[];
  onRunAgent: (config: AgentConfig) => Promise<void>;
  running: boolean;
  onScroll: (scrollTop: number) => void;
}

const MODEL_GROUPS = ["OpenAI", "Anthropic", "Google Gemini"] as const;
const REASONING_EFFORT_OPTIONS = ["none", "low", "medium", "high", "xhigh"] as const;

const AgentPane = forwardRef<HTMLDivElement, Props>(
  (
    {
      text,
      spans,
      processedWithChunking = false,
      activeOutput,
      onActiveOutputChange,
      diffSpans = [],
      onRunAgent,
      running,
      onScroll,
    },
    ref,
  ) => {
    const [configOpen, setConfigOpen] = useState(false);
    const [mode, setMode] = useState<"rule" | "llm">("rule");
    const [systemPrompt, setSystemPrompt] = useState(
      "You are a PII annotation assistant. Return ONLY a JSON array of objects with start (0-based), end (exclusive), label, and text for each PII span.",
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
    }, []);

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

    const effectiveModel = model === "__custom__" ? customModel : model;
    const selectedPreset =
      model === "__custom__" ? undefined : getModelPreset(effectiveModel);
    const supportsReasoningEffort = selectedPreset?.supportsReasoningEffort ?? false;
    const supportsAnthropicThinking =
      selectedPreset?.supportsAnthropicThinking ?? false;

    const handleRun = () => {
      onRunAgent({
        mode,
        system_prompt: systemPrompt,
        model: effectiveModel,
        temperature,
        api_key: apiKey || undefined,
        api_base: apiBase || undefined,
        reasoning_effort: reasoningEffort,
        anthropic_thinking: anthropicThinking,
        anthropic_thinking_budget_tokens: anthropicThinking
          ? anthropicThinkingBudget
          : undefined,
        label_profile: labelProfile,
        chunk_mode: chunkMode,
        chunk_size_chars: chunkSizeChars,
      });
    };

    return (
      <div className="pane">
        <div className="pane-header pane-header-agent">
          <div className="pane-header-agent-left">
            <span>Agent Annotations</span>
            {processedWithChunking && (
              <span className="chunk-badge" title="Backend auto-chunked this run for reliability.">
                Processed with chunking
              </span>
            )}
            <span className="agent-view-control">
              <label htmlFor="agent-view-select">View:</label>
              <select
                id="agent-view-select"
                name="agent_view"
                value={activeOutput}
                onChange={(e) =>
                  onActiveOutputChange(e.target.value as "combined" | "rule" | "llm")
                }
              >
                <option value="combined">Combined</option>
                <option value="rule">Rule Only</option>
                <option value="llm">LLM Only</option>
              </select>
            </span>
          </div>
          <button className="config-toggle" onClick={() => setConfigOpen(!configOpen)}>
            {configOpen ? "Hide Config" : "Show Config"}
          </button>
        </div>
        <div className={`agent-config ${configOpen ? "" : "collapsed"}`}>
          <div className="field">
            <label htmlFor="agent-mode">Mode</label>
            <select
              id="agent-mode"
              name="agent_mode"
              value={mode}
              onChange={(e) => setMode(e.target.value as "rule" | "llm")}
            >
              <option value="rule">Rule-based</option>
              <option value="llm">LLM (via LiteLLM)</option>
            </select>
          </div>
          {mode === "llm" && (
            <>
              <div className="field">
                <label htmlFor="agent-system-prompt">System Prompt</label>
                <textarea
                  id="agent-system-prompt"
                  name="agent_system_prompt"
                  value={systemPrompt}
                  onChange={(e) => setSystemPrompt(e.target.value)}
                />
              </div>
              <div className="field">
                <label htmlFor="agent-model">Model</label>
                <select
                  id="agent-model"
                  name="agent_model"
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
                    id="agent-custom-model"
                    name="agent_custom_model"
                    type="text"
                    value={customModel}
                    onChange={(e) => setCustomModel(e.target.value)}
                    placeholder="provider.model-name (e.g., openai.gpt-5.3-codex)"
                    style={{ marginTop: 4 }}
                  />
                )}
              </div>
              <div className={`field ${supportsReasoningEffort ? "" : "field-disabled"}`}>
                <label htmlFor="agent-reasoning-effort">
                  Reasoning Effort
                  {!supportsReasoningEffort && " (not supported for this model)"}
                </label>
                <select
                  id="agent-reasoning-effort"
                  name="agent_reasoning_effort"
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
                <label htmlFor="agent-anthropic-thinking">
                  Anthropic Thinking
                  {!supportsAnthropicThinking && " (not supported for this model)"}
                </label>
                <div className="inline-control-row">
                  <input
                    id="agent-anthropic-thinking"
                    name="agent_anthropic_thinking"
                    type="checkbox"
                    checked={anthropicThinking}
                    disabled={!supportsAnthropicThinking}
                    onChange={(e) => setAnthropicThinking(e.target.checked)}
                  />
                  <input
                    id="agent-anthropic-thinking-budget"
                    name="agent_anthropic_thinking_budget_tokens"
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
                <label htmlFor="agent-temperature">Temperature: {temperature}</label>
                <input
                  id="agent-temperature"
                  name="agent_temperature"
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                />
              </div>
              <div className="field">
                <label htmlFor="agent-label-profile">Label Profile</label>
                <select
                  id="agent-label-profile"
                  name="agent_label_profile"
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
                <label htmlFor="agent-chunk-mode">Chunk Mode</label>
                <select
                  id="agent-chunk-mode"
                  name="agent_chunk_mode"
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
                  <label htmlFor="agent-chunk-size">Chunk Size (chars)</label>
                  <input
                    id="agent-chunk-size"
                    name="agent_chunk_size_chars"
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
                    <label htmlFor="agent-api-key">API Key</label>
                    <input
                      id="agent-api-key"
                      name="agent_api_key"
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
                    <label htmlFor="agent-api-base">LiteLLM Base URL (optional)</label>
                    <input
                      id="agent-api-base"
                      name="agent_api_base"
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
          <button className="run-btn" onClick={handleRun} disabled={running}>
            {running ? "Running..." : "Run Agent"}
          </button>
        </div>
        <div
          className="pane-body"
          ref={ref}
          onScroll={(e) => onScroll((e.target as HTMLDivElement).scrollTop)}
        >
          {spans.length === 0 && !running ? (
            <span style={{ color: "#888" }}>
              No agent annotations yet. Configure and run the agent above.
            </span>
          ) : (
            <AnnotatedText text={text} spans={spans} diffSpans={diffSpans} />
          )}
        </div>
      </div>
    );
  },
);

AgentPane.displayName = "AgentPane";
export default AgentPane;
