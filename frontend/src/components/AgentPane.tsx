import { forwardRef, useState } from "react";
import { MODEL_PRESETS, getModelPreset } from "../modelPresets";
import type { AgentConfig, AgentView, CanonicalSpan } from "../types";
import AnnotatedText from "./AnnotatedText";

interface Props {
  text: string;
  spans: CanonicalSpan[];
  activeOutput: AgentView;
  onActiveOutputChange: (view: AgentView) => void;
  diffSpans?: { start: number; end: number; type: "added" | "removed" }[];
  onRunAgent: (config: AgentConfig) => Promise<void>;
  running: boolean;
  onScroll: (scrollTop: number) => void;
}

const MODEL_GROUPS = ["OpenAI", "Anthropic", "Google Gemini", "Ollama"] as const;
const REASONING_EFFORT_OPTIONS = ["none", "low", "medium", "high", "xhigh"] as const;

const AgentPane = forwardRef<HTMLDivElement, Props>(
  (
    {
      text,
      spans,
      activeOutput,
      onActiveOutputChange,
      diffSpans = [],
      onRunAgent,
      running,
      onScroll,
    },
    ref,
  ) => {
    const [configOpen, setConfigOpen] = useState(true);
    const [mode, setMode] = useState<"rule" | "llm">("rule");
    const [systemPrompt, setSystemPrompt] = useState(
      "Identify all PII in the transcript. Label each span with: NAME, LOCATION, SCHOOL, DATE, AGE, PHONE, EMAIL, URL, or MISC_ID.",
    );
    const [model, setModel] = useState("openai/gpt-5.2-codex");
    const [customModel, setCustomModel] = useState("");
    const [temperature, setTemperature] = useState(0);
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
        reasoning_effort: reasoningEffort,
        anthropic_thinking: anthropicThinking,
        anthropic_thinking_budget_tokens: anthropicThinking
          ? anthropicThinkingBudget
          : undefined,
      });
    };

    return (
      <div className="pane">
        <div className="pane-header">
          Agent Annotations
          <span style={{ marginLeft: 8, fontSize: 11, color: "#888" }}>
            View:
            <select
              style={{ marginLeft: 6 }}
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
          <button
            className="config-toggle"
            onClick={() => setConfigOpen(!configOpen)}
            style={{ float: "right" }}
          >
            {configOpen ? "Hide Config" : "Show Config"}
          </button>
        </div>
        <div className={`agent-config ${configOpen ? "" : "collapsed"}`}>
          <div className="field">
            <label>Mode</label>
            <select
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
                <label>System Prompt</label>
                <textarea
                  value={systemPrompt}
                  onChange={(e) => setSystemPrompt(e.target.value)}
                />
              </div>
              <div className="field">
                <label>Model</label>
                <select
                  value={model}
                  onChange={(e) => {
                    const value = e.target.value;
                    setModel(value);
                    const preset = value === "__custom__" ? undefined : getModelPreset(value);
                    if (preset?.defaultReasoningEffort) {
                      setReasoningEffort(preset.defaultReasoningEffort);
                    }
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
                    type="text"
                    value={customModel}
                    onChange={(e) => setCustomModel(e.target.value)}
                    placeholder="provider/model-name (e.g., ollama/llama3)"
                    style={{ marginTop: 4 }}
                  />
                )}
              </div>
              <div className={`field ${supportsReasoningEffort ? "" : "field-disabled"}`}>
                <label>
                  Reasoning Effort
                  {!supportsReasoningEffort && " (not supported for this model)"}
                </label>
                <select
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
                <label>
                  Anthropic Thinking
                  {!supportsAnthropicThinking && " (not supported for this model)"}
                </label>
                <div
                  style={{
                    display: "flex",
                    gap: 8,
                    alignItems: "center",
                  }}
                >
                  <input
                    type="checkbox"
                    checked={anthropicThinking}
                    disabled={!supportsAnthropicThinking}
                    onChange={(e) => setAnthropicThinking(e.target.checked)}
                    style={{ width: 16, height: 16 }}
                  />
                  <input
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
                <label>Temperature: {temperature}</label>
                <input
                  type="range"
                  min="0"
                  max="2"
                  step="0.1"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                />
              </div>
              <div className="field">
                <label>API Key</label>
                <input
                  type="password"
                  value={apiKey}
                  onChange={(e) => handleApiKeyChange(e.target.value)}
                  placeholder="Provider API key (or set env var)"
                />
                <span style={{ fontSize: 10, color: "#888" }}>
                  Uses OPENAI_API_KEY, ANTHROPIC_API_KEY, etc. env vars as fallback
                </span>
              </div>
            </>
          )}
          <button
            className="run-btn"
            onClick={handleRun}
            disabled={running}
          >
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
