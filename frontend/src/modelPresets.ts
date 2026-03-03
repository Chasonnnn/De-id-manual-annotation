export interface ModelPreset {
  id: string;
  label: string;
  group: "OpenAI" | "Anthropic" | "Google Gemini" | "Ollama";
  supportsReasoningEffort: boolean;
  supportsAnthropicThinking: boolean;
  defaultReasoningEffort?: "none" | "low" | "medium" | "high" | "xhigh";
}

export const MODEL_PRESETS: ModelPreset[] = [
  {
    id: "openai/gpt-5.2-codex",
    label: "GPT-5.2 Codex (xhigh)",
    group: "OpenAI",
    supportsReasoningEffort: true,
    supportsAnthropicThinking: false,
    defaultReasoningEffort: "xhigh",
  },
  {
    id: "openai/gpt-5.2-chat-latest",
    label: "GPT-5.2 Chat Latest (xhigh)",
    group: "OpenAI",
    supportsReasoningEffort: true,
    supportsAnthropicThinking: false,
    defaultReasoningEffort: "xhigh",
  },
  {
    id: "openai/gpt-4o-mini",
    label: "GPT-4o Mini",
    group: "OpenAI",
    supportsReasoningEffort: false,
    supportsAnthropicThinking: false,
  },
  {
    id: "anthropic/claude-opus-4-6",
    label: "Claude Opus 4.6 (thinking)",
    group: "Anthropic",
    supportsReasoningEffort: false,
    supportsAnthropicThinking: true,
  },
  {
    id: "anthropic/claude-opus-4-6-20260210",
    label: "Claude Opus 4.6 Snapshot 20260210",
    group: "Anthropic",
    supportsReasoningEffort: false,
    supportsAnthropicThinking: true,
  },
  {
    id: "gemini/gemini-3-pro-preview",
    label: "Gemini 3 Pro Preview",
    group: "Google Gemini",
    supportsReasoningEffort: false,
    supportsAnthropicThinking: false,
  },
  {
    id: "ollama/llama3",
    label: "Llama 3 (Ollama)",
    group: "Ollama",
    supportsReasoningEffort: false,
    supportsAnthropicThinking: false,
  },
];

const byId = new Map(MODEL_PRESETS.map((preset) => [preset.id, preset]));

export function getModelPreset(modelId: string): ModelPreset | undefined {
  return byId.get(modelId);
}

