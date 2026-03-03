"""Tests for agent.py: regex-based and LLM-based PII detection via LiteLLM."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent import (
    MODEL_PRESETS,
    SYSTEM_PROMPT,
    _strip_code_fences,
    run_llm,
    run_llm_with_metadata,
    run_regex,
)
from models import CanonicalSpan


# ---------------------------------------------------------------------------
# Regex agent tests
# ---------------------------------------------------------------------------


class TestRunRegex:
    def test_detects_email(self):
        spans = run_regex("Contact user@example.com for info.")
        emails = [s for s in spans if s.label == "EMAIL"]
        assert len(emails) == 1
        assert emails[0].text == "user@example.com"
        assert emails[0].start == 8
        assert emails[0].end == 24

    def test_detects_url(self):
        spans = run_regex("Visit https://example.com/page today.")
        urls = [s for s in spans if s.label == "URL"]
        assert len(urls) == 1
        assert urls[0].text == "https://example.com/page"

    def test_detects_phone(self):
        spans = run_regex("Call +1-555-867-5309 now.")
        phones = [s for s in spans if s.label == "PHONE"]
        assert len(phones) >= 1
        assert any("+1-555-867-5309" in s.text for s in phones)

    def test_detects_date_numeric(self):
        spans = run_regex("Born on 12/25/1990 at noon.")
        dates = [s for s in spans if s.label == "DATE"]
        assert len(dates) >= 1
        assert any("12/25/1990" in s.text for s in dates)

    def test_detects_date_named_month(self):
        spans = run_regex("Meeting on January 15, 2024 at 3pm.")
        dates = [s for s in spans if s.label == "DATE"]
        assert len(dates) >= 1
        assert any("January 15, 2024" in s.text for s in dates)

    def test_detects_time(self):
        spans = run_regex("The session starts at 2:30 PM.")
        times = [s for s in spans if s.label == "TIME"]
        assert len(times) >= 1
        assert any("2:30 PM" in s.text for s in times)

    def test_no_false_positive_on_clean_text(self):
        spans = run_regex("Hello, how are you doing today?")
        assert len(spans) == 0

    def test_multiple_types(self):
        text = "Email me at a@b.com or visit https://x.com"
        spans = run_regex(text)
        labels = {s.label for s in spans}
        assert "EMAIL" in labels
        assert "URL" in labels

    def test_spans_sorted_by_start(self):
        text = "Visit https://b.com then email a@b.com"
        spans = run_regex(text)
        starts = [s.start for s in spans]
        assert starts == sorted(starts)

    def test_offset_consistency(self):
        """Every span's text must match the slice of the input at its offsets."""
        text = "Send to user@test.org on 01/15/2025 at 10:30 AM via https://foo.bar"
        spans = run_regex(text)
        assert len(spans) > 0
        for s in spans:
            assert text[s.start : s.end] == s.text, f"Offset mismatch for {s}"


# ---------------------------------------------------------------------------
# Code fence stripping
# ---------------------------------------------------------------------------


class TestStripCodeFences:
    def test_plain_json(self):
        assert _strip_code_fences('[{"a": 1}]') == '[{"a": 1}]'

    def test_json_code_fence(self):
        content = '```json\n[{"a": 1}]\n```'
        assert _strip_code_fences(content) == '[{"a": 1}]'

    def test_bare_code_fence(self):
        content = '```\n[{"a": 1}]\n```'
        assert _strip_code_fences(content) == '[{"a": 1}]'

    def test_whitespace_preserved(self):
        content = '  ```json\n  [{"a": 1}]\n  ```  '
        result = _strip_code_fences(content)
        assert '"a"' in result


# ---------------------------------------------------------------------------
# LLM agent tests (mocked via litellm.completion)
# ---------------------------------------------------------------------------


def _mock_completion_response(content: str):
    """Build a mock LiteLLM completion response (OpenAI format)."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    resp = MagicMock()
    resp.choices = [choice]
    return resp


class TestRunLLM:
    @patch("agent.completion")
    def test_parses_valid_json(self, mock_completion):
        payload = json.dumps(
            [
                {"start": 0, "end": 4, "label": "NAME", "text": "John"},
            ]
        )
        mock_completion.return_value = _mock_completion_response(payload)

        spans = run_llm("Hi John!", api_key="test-key")
        assert len(spans) == 1
        assert spans[0].label == "NAME"
        assert spans[0].text == "John"

    @patch("agent.completion")
    def test_strips_markdown_fences(self, mock_completion):
        payload = (
            '```json\n[{"start": 0, "end": 4, "label": "NAME", "text": "John"}]\n```'
        )
        mock_completion.return_value = _mock_completion_response(payload)

        spans = run_llm("Hi John!", api_key="test-key")
        assert len(spans) == 1

    @patch("agent.completion")
    def test_raises_on_invalid_json(self, mock_completion):
        mock_completion.return_value = _mock_completion_response("not json at all")

        with pytest.raises(ValueError, match="invalid JSON"):
            run_llm("text", api_key="test-key")

    @patch("agent.completion")
    def test_raises_on_non_array_json(self, mock_completion):
        mock_completion.return_value = _mock_completion_response('{"not": "an array"}')

        with pytest.raises(ValueError, match="Expected JSON array"):
            run_llm("text", api_key="test-key")

    @patch("agent.completion")
    def test_empty_response(self, mock_completion):
        mock_completion.return_value = _mock_completion_response("[]")

        spans = run_llm("clean text", api_key="test-key")
        assert spans == []

    @patch("agent.completion")
    def test_none_content_returns_empty(self, mock_completion):
        message = MagicMock()
        message.content = None
        choice = MagicMock()
        choice.message = message
        resp = MagicMock()
        resp.choices = [choice]
        mock_completion.return_value = resp

        spans = run_llm("text", api_key="test-key")
        assert spans == []

    @patch("agent.completion")
    def test_custom_system_prompt_and_temperature(self, mock_completion):
        mock_completion.return_value = _mock_completion_response("[]")

        custom_prompt = "You are a custom PII detector."
        run_llm(
            "text",
            api_key="k",
            model="openai/gpt-4o",
            system_prompt=custom_prompt,
            temperature=0.5,
        )

        call_kwargs = mock_completion.call_args
        assert call_kwargs.kwargs["model"] == "openai/gpt-4o"
        assert call_kwargs.kwargs["temperature"] == 0.5
        messages = call_kwargs.kwargs["messages"]
        assert messages[0]["content"] == custom_prompt

    @patch("agent.completion")
    def test_default_system_prompt(self, mock_completion):
        mock_completion.return_value = _mock_completion_response("[]")

        run_llm("text", api_key="k")

        call_kwargs = mock_completion.call_args
        messages = call_kwargs.kwargs["messages"]
        assert messages[0]["content"] == SYSTEM_PROMPT

    @patch("agent.completion")
    def test_api_key_passed_to_litellm(self, mock_completion):
        """Verify api_key is forwarded to litellm.completion."""
        mock_completion.return_value = _mock_completion_response("[]")

        run_llm("text", api_key="sk-test-123", model="openai/gpt-4o")

        call_kwargs = mock_completion.call_args
        assert call_kwargs.kwargs["api_key"] == "sk-test-123"

    @patch("agent.completion")
    def test_api_base_passed_to_litellm(self, mock_completion):
        """Verify api_base is forwarded to litellm.completion when set."""
        mock_completion.return_value = _mock_completion_response("[]")

        run_llm(
            "text",
            api_key="sk-test-123",
            api_base="https://litellm.local/v1",
            model="openai/gpt-4o",
        )

        call_kwargs = mock_completion.call_args
        assert call_kwargs.kwargs["api_base"] == "https://litellm.local/v1"

    @patch("agent.completion")
    def test_default_model_uses_litellm_prefix(self, mock_completion):
        """Default model should use the provider/model format."""
        mock_completion.return_value = _mock_completion_response("[]")

        run_llm("text", api_key="k")

        call_kwargs = mock_completion.call_args
        assert call_kwargs.kwargs["model"] == "openai/gpt-4o-mini"

    @patch("agent.completion")
    def test_anthropic_model(self, mock_completion):
        """LiteLLM should support non-OpenAI models like Anthropic."""
        mock_completion.return_value = _mock_completion_response("[]")

        run_llm("text", api_key="k", model="anthropic/claude-sonnet-4-20250514")

        call_kwargs = mock_completion.call_args
        assert call_kwargs.kwargs["model"] == "anthropic/claude-sonnet-4-20250514"

    @patch("agent.completion")
    def test_openai_reasoning_effort_is_passed(self, mock_completion):
        mock_completion.return_value = _mock_completion_response("[]")
        run_llm_with_metadata(
            text="text",
            api_key="k",
            model="openai/gpt-5.2-codex",
            reasoning_effort="xhigh",
        )
        call_kwargs = mock_completion.call_args
        assert call_kwargs.kwargs["reasoning_effort"] == "xhigh"

    @patch("agent.completion")
    def test_anthropic_thinking_is_passed(self, mock_completion):
        mock_completion.return_value = _mock_completion_response("[]")
        run_llm_with_metadata(
            text="text",
            api_key="k",
            model="anthropic/claude-opus-4-6",
            anthropic_thinking=True,
            anthropic_thinking_budget_tokens=2048,
        )
        call_kwargs = mock_completion.call_args
        assert call_kwargs.kwargs["thinking"]["type"] == "enabled"
        assert call_kwargs.kwargs["thinking"]["budget_tokens"] == 2048

    @patch("agent.completion")
    def test_fallback_retries_without_advanced_params(self, mock_completion):
        mock_completion.side_effect = [
            Exception("Unsupported parameter: reasoning_effort"),
            _mock_completion_response("[]"),
        ]
        result = run_llm_with_metadata(
            text="text",
            api_key="k",
            model="openai/gpt-5.2-chat-latest",
            reasoning_effort="xhigh",
        )
        assert result.spans == []
        assert len(result.warnings) == 1
        assert "retried without them" in result.warnings[0]

        first_call = mock_completion.call_args_list[0].kwargs
        second_call = mock_completion.call_args_list[1].kwargs
        assert "reasoning_effort" in first_call
        assert "reasoning_effort" not in second_call


def test_model_presets_include_requested_options():
    model_ids = {preset["model"] for preset in MODEL_PRESETS}
    assert "openai/gpt-5.2-codex" in model_ids
    assert "openai/gpt-5.2-chat-latest" in model_ids
    assert "anthropic/claude-opus-4-6" in model_ids
    assert "anthropic/claude-opus-4-6-20260210" in model_ids
    assert "gemini/gemini-3-pro-preview" in model_ids
    assert "ollama/llama3" in model_ids
