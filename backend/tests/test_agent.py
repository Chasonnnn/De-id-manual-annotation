"""Tests for agent.py: regex-based and LLM-based PII detection via LiteLLM."""

from __future__ import annotations

import json
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

from agent import (
    METHOD_DEFINITION_BY_ID,
    MODEL_PRESETS,
    SYSTEM_PROMPT,
    _expand_detected_value_occurrences,
    _infer_provider,
    _normalize_method_bundle,
    _presidio_runtime_error,
    _reset_presidio_runtime_state,
    _run_presidio_pass,
    build_extraction_system_prompt,
    _strip_code_fences,
    run_llm,
    run_llm_with_metadata,
    run_method_with_metadata,
    run_regex,
)
import agent
from models import CanonicalSpan, LLMConfidenceMetric


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

    def test_detects_bare_domain_url(self):
        spans = run_regex("Visit upchieve.org for more info.")
        urls = [s for s in spans if s.label == "URL"]
        assert len(urls) == 1
        assert urls[0].text == "upchieve.org"

    def test_detects_bare_domain_split_after_dot(self):
        text = "Visit upchieve.\norg for more info."
        spans = run_regex(text)
        urls = [s for s in spans if s.label == "URL"]
        assert len(urls) == 1
        assert urls[0].text == "upchieve.\norg"

    def test_detects_bare_domain_split_before_dot(self):
        text = "Visit upchieve\n.org for more info."
        spans = run_regex(text)
        urls = [s for s in spans if s.label == "URL"]
        assert len(urls) == 1
        assert urls[0].text == "upchieve\n.org"

    def test_detects_multi_line_chained_domain(self):
        text = "Can you go to meet.\ngoogle.\ncom and type this in?"
        spans = run_regex(text)
        urls = [s for s in spans if s.label == "URL"]
        assert len(urls) == 1
        assert urls[0].text == "meet.\ngoogle.\ncom"

    def test_detects_multi_line_chained_domain_with_split_before_dot(self):
        text = "Open docs\n.google.\ncom for the instructions."
        spans = run_regex(text)
        urls = [s for s in spans if s.label == "URL"]
        assert len(urls) == 1
        assert urls[0].text == "docs\n.google.\ncom"

    def test_detects_split_country_tld_domain(self):
        text = "So everybody go to kahoot.\nit"
        spans = run_regex(text)
        urls = [s for s in spans if s.label == "URL"]
        assert len(urls) == 1
        assert urls[0].text == "kahoot.\nit"

    def test_does_not_double_count_email_domain_as_url(self):
        spans = run_regex("Contact user@example.com for info.")
        urls = [s for s in spans if s.label == "URL"]
        emails = [s for s in spans if s.label == "EMAIL"]
        assert len(emails) == 1
        assert urls == []

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


class TestInferProvider:
    def test_recognizes_common_dot_and_slash_model_prefixes(self):
        assert _infer_provider("openai.gpt-5.2-chat") == "openai"
        assert _infer_provider("openai/gpt-4o") == "openai"
        assert _infer_provider("chatgpt/gpt-5.4") == "openai"
        assert _infer_provider("anthropic.claude-4.6-opus") == "anthropic"
        assert _infer_provider("anthropic/claude-sonnet-4-20250514") == "anthropic"
        assert _infer_provider("google.gemini-3.1-pro-preview") == "gemini"
        assert _infer_provider("google/gemini-2.5-pro") == "gemini"

    def test_falls_back_without_litellm_provider_lookup(self):
        assert _infer_provider("gpt-5.4") == "openai"
        assert _infer_provider("vertex_ai/gemini-2.5-pro") == "gemini"
        assert _infer_provider("custom-provider/model-name") == "custom-provider"
        assert _infer_provider("opaque-model-id") == "unknown"


class TestConfigureLiteLLMRuntime:
    def test_suppresses_litellm_debug_noise(self):
        fake_litellm = types.SimpleNamespace(
            set_verbose=True,
            suppress_debug_info=False,
            log_level="DEBUG",
        )

        agent._configure_litellm_runtime(fake_litellm)

        assert fake_litellm.set_verbose is False
        assert fake_litellm.suppress_debug_info is True
        assert fake_litellm.log_level == "ERROR"


# ---------------------------------------------------------------------------
# LLM agent tests (mocked via litellm.completion)
# ---------------------------------------------------------------------------


def _mock_completion_response(
    content: object,
    token_logprobs: list[float] | None = None,
    finish_reason: str | None = None,
    completion_tokens: int | None = None,
):
    """Build a mock LiteLLM completion response (OpenAI format)."""
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason
    if token_logprobs is not None:
        tokens = []
        for value in token_logprobs:
            token = MagicMock()
            token.logprob = value
            tokens.append(token)
        choice.logprobs = MagicMock()
        choice.logprobs.content = tokens
    else:
        choice.logprobs = None
    resp = MagicMock()
    resp.choices = [choice]
    if completion_tokens is not None:
        usage = MagicMock()
        usage.completion_tokens = completion_tokens
        resp.usage = usage
    else:
        resp.usage = None
    return resp


def _spans_payload(items: list[dict[str, object]]) -> str:
    return json.dumps({"spans": items})


class TestRunLLM:
    @patch("agent.completion")
    def test_parses_valid_json(self, mock_completion):
        payload = _spans_payload(
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
    def test_repairs_offset_mismatches_using_span_text(self, mock_completion):
        payload = _spans_payload(
            [
                {"start": 0, "end": 1, "label": "NAME", "text": "Anna"},
                {"start": 2, "end": 4, "label": "NAME", "text": "Sue"},
            ]
        )
        mock_completion.return_value = _mock_completion_response(payload)

        result = run_llm_with_metadata(
            text="Hello Anna and Sue.",
            api_key="test-key",
            model="openai/gpt-4o",
        )

        assert [(span.start, span.end, span.text) for span in result.spans] == [
            (6, 10, "Anna"),
            (15, 18, "Sue"),
        ]
        assert any("offset mismatch" in warning for warning in result.warnings)

    @patch("agent.completion")
    def test_drops_low_confidence_name_after_failed_realign(self, mock_completion):
        payload = _spans_payload(
            [
                {"start": 0, "end": 3, "label": "NAME", "text": "Anna"},
            ]
        )
        mock_completion.return_value = _mock_completion_response(payload)

        result = run_llm_with_metadata(
            text="and value",
            api_key="test-key",
            model="openai/gpt-4o",
        )

        assert result.spans == []
        assert any("Dropped 1" in warning for warning in result.warnings)

    @patch("agent.completion")
    def test_matching_offsets_skip_repair_warning(self, mock_completion):
        payload = _spans_payload(
            [
                {"start": 6, "end": 10, "label": "NAME", "text": "Anna"},
            ]
        )
        mock_completion.return_value = _mock_completion_response(payload)

        result = run_llm_with_metadata(
            text="Hello Anna.",
            api_key="test-key",
            model="openai/gpt-4o",
        )

        assert len(result.spans) == 1
        assert result.spans[0].text == "Anna"
        assert all("offset mismatch" not in warning for warning in result.warnings)

    @patch("agent.completion")
    def test_preserves_name_span_in_official_output_when_honorific_is_missing(self, mock_completion):
        text = "Hello Mr. Muhammad"
        muhammad_start = text.index("Muhammad")
        payload = _spans_payload(
            [
                {
                    "start": muhammad_start,
                    "end": muhammad_start + len("Muhammad"),
                    "label": "NAME",
                    "text": "Muhammad",
                }
            ]
        )
        mock_completion.return_value = _mock_completion_response(payload)

        result = run_llm_with_metadata(
            text=text,
            api_key="test-key",
            model="openai/gpt-4o",
        )

        assert [(span.start, span.end, span.text) for span in result.spans] == [
            (muhammad_start, muhammad_start + len("Muhammad"), "Muhammad")
        ]

    @patch("agent.completion")
    def test_preserves_name_span_in_official_output_when_possessive_is_missing(self, mock_completion):
        text = "Sebastian's notebook"
        sebastian_start = text.index("Sebastian")
        payload = _spans_payload(
            [
                {
                    "start": sebastian_start,
                    "end": sebastian_start + len("Sebastian"),
                    "label": "NAME",
                    "text": "Sebastian",
                }
            ]
        )
        mock_completion.return_value = _mock_completion_response(payload)

        result = run_llm_with_metadata(
            text=text,
            api_key="test-key",
            model="openai/gpt-4o",
        )

        assert [(span.start, span.end, span.text) for span in result.spans] == [
            (sebastian_start, sebastian_start + len("Sebastian"), "Sebastian")
        ]

    @patch("agent.completion")
    def test_warns_when_finish_reason_is_length(self, mock_completion):
        payload = _spans_payload(
            [
                {"start": 6, "end": 10, "label": "NAME", "text": "Anna"},
            ]
        )
        mock_completion.return_value = _mock_completion_response(
            payload, finish_reason="length"
        )

        result = run_llm_with_metadata(
            text="Hello Anna.",
            api_key="test-key",
            model="openai/gpt-4o",
        )

        assert len(result.spans) == 1
        assert any("finish_reason=length" in warning for warning in result.warnings)

    @patch("agent.completion")
    def test_parses_object_with_spans_key(self, mock_completion):
        payload = _spans_payload(
            [
                {"start": 0, "end": 4, "label": "NAME", "text": "John"},
            ]
        )
        mock_completion.return_value = _mock_completion_response(payload)
        spans = run_llm("Hi John!", api_key="test-key")
        assert len(spans) == 1
        assert spans[0].text == "John"

    @patch("agent.completion")
    def test_parses_json_embedded_in_text(self, mock_completion):
        payload = (
            "I found entities.\n"
            '{"spans":[{"start": 0, "end": 4, "label": "NAME", "text": "John"}]}'
        )
        mock_completion.return_value = _mock_completion_response(payload)
        spans = run_llm("Hi John!", api_key="test-key")
        assert len(spans) == 1

    @patch("agent.completion")
    def test_parses_anthropic_content_blocks(self, mock_completion):
        payload = [
            {"type": "thinking", "text": "internal reasoning"},
            {
                "type": "text",
                "text": '{"spans":[{"start": 3, "end": 7, "label": "NAME", "text": "John"}]}',
            },
        ]
        mock_completion.return_value = _mock_completion_response(payload)
        spans = run_llm(
            "Hi John!",
            api_key="test-key",
            model="anthropic.claude-4.6-opus",
        )
        assert len(spans) == 1
        assert spans[0].start == 3
        assert spans[0].end == 7
        assert spans[0].text == "John"

    @patch("agent.completion")
    def test_strips_markdown_fences(self, mock_completion):
        payload = (
            '```json\n{"spans":[{"start": 0, "end": 4, "label": "NAME", "text": "John"}]}\n```'
        )
        mock_completion.return_value = _mock_completion_response(payload)

        spans = run_llm("Hi John!", api_key="test-key")
        assert len(spans) == 1

    @patch("agent.completion")
    def test_raises_on_invalid_json(self, mock_completion):
        mock_completion.return_value = _mock_completion_response("not json at all")

        with pytest.raises(ValueError, match="non-JSON output"):
            run_llm("text", api_key="test-key")

    @patch("agent.completion")
    def test_raises_on_non_array_json(self, mock_completion):
        mock_completion.return_value = _mock_completion_response('{"not": "an array"}')

        with pytest.raises(ValueError, match="top-level object with key 'spans'"):
            run_llm("text", api_key="test-key")

    @patch("agent.completion")
    def test_empty_response(self, mock_completion):
        mock_completion.return_value = _mock_completion_response('{"spans":[]}')

        spans = run_llm("clean text", api_key="test-key")
        assert spans == []

    @patch("agent.completion")
    def test_none_content_raises_error(self, mock_completion):
        message = MagicMock()
        message.content = None
        choice = MagicMock()
        choice.message = message
        resp = MagicMock()
        resp.choices = [choice]
        mock_completion.return_value = resp

        with pytest.raises(ValueError, match="empty output content"):
            run_llm("text", api_key="test-key")

    @patch("agent.completion")
    def test_none_content_raises_error_with_finish_reason(self, mock_completion):
        message = MagicMock()
        message.content = None
        choice = MagicMock()
        choice.message = message
        choice.finish_reason = "length"
        resp = MagicMock()
        resp.choices = [choice]
        mock_completion.return_value = resp

        with pytest.raises(ValueError, match="empty output content \\(finish_reason=length\\)"):
            run_llm_with_metadata(
                text="text",
                api_key="test-key",
                model="openai/gpt-4o",
            )

    @patch("agent.completion")
    def test_empty_content_error_includes_response_debug_summary(self, mock_completion):
        message = types.SimpleNamespace(content=None, tool_calls=None)
        choice = types.SimpleNamespace(message=message, finish_reason="length")
        resp = types.SimpleNamespace(choices=[choice], output_text=None)
        mock_completion.return_value = resp

        with pytest.raises(ValueError) as exc_info:
            run_llm_with_metadata(
                text="text",
                api_key="test-key",
                model="openai/gpt-4o",
            )

        message_text = str(exc_info.value)
        assert "empty output content (finish_reason=length)" in message_text
        assert "response_debug=" in message_text
        assert "message.content=None" in message_text
        assert "message.tool_calls=None" in message_text
        assert "output_text=None" in message_text
        assert "raw_preview=" in message_text

    @patch("agent.completion")
    def test_custom_system_prompt_and_temperature(self, mock_completion):
        mock_completion.return_value = _mock_completion_response('{"spans":[]}')

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
        assert messages[0]["content"] == build_extraction_system_prompt(custom_prompt)

    @patch("agent.completion")
    def test_default_system_prompt(self, mock_completion):
        mock_completion.return_value = _mock_completion_response('{"spans":[]}')

        run_llm("text", api_key="k")

        call_kwargs = mock_completion.call_args
        messages = call_kwargs.kwargs["messages"]
        assert messages[0]["content"] == build_extraction_system_prompt(SYSTEM_PROMPT)

    @patch("agent.completion")
    def test_api_key_passed_to_litellm(self, mock_completion):
        """Verify api_key is forwarded to litellm.completion."""
        mock_completion.return_value = _mock_completion_response('{"spans":[]}')

        run_llm("text", api_key="sk-test-123", model="openai/gpt-4o")

        call_kwargs = mock_completion.call_args
        assert call_kwargs.kwargs["api_key"] == "sk-test-123"

    @patch("agent.completion")
    def test_api_base_passed_to_litellm(self, mock_completion):
        """Verify api_base is forwarded to litellm.completion when set."""
        mock_completion.return_value = _mock_completion_response('{"spans":[]}')

        run_llm(
            "text",
            api_key="sk-test-123",
            api_base="https://litellm.local/v1",
            model="openai/gpt-4o",
        )

        call_kwargs = mock_completion.call_args
        assert call_kwargs.kwargs["api_base"] == "https://litellm.local/v1"

    @patch("agent.completion")
    def test_timeout_is_passed_to_litellm(self, mock_completion):
        mock_completion.return_value = _mock_completion_response('{"spans":[]}')

        run_llm_with_metadata(
            text="text",
            api_key="k",
            model="openai/gpt-4o",
            timeout_seconds=12.5,
        )

        call_kwargs = mock_completion.call_args
        assert call_kwargs.kwargs["timeout"] == pytest.approx(12.5)

    @patch("agent.completion")
    def test_extraction_defaults_timeout_and_max_tokens(self, mock_completion):
        mock_completion.return_value = _mock_completion_response('{"spans":[]}')

        run_llm_with_metadata(
            text="text",
            api_key="k",
            model="openai/gpt-4o",
        )

        call_kwargs = mock_completion.call_args
        assert call_kwargs.kwargs["timeout"] == pytest.approx(60.0)
        assert call_kwargs.kwargs["max_tokens"] == 4096

    @patch("agent.completion")
    def test_extraction_clamps_timeout_to_explicit_budget(self, mock_completion):
        mock_completion.return_value = _mock_completion_response('{"spans":[]}')

        run_llm_with_metadata(
            text="text",
            api_key="k",
            model="openai/gpt-4o",
            timeout_seconds=15.0,
        )

        call_kwargs = mock_completion.call_args
        assert call_kwargs.kwargs["timeout"] == pytest.approx(15.0)
        assert call_kwargs.kwargs["max_tokens"] == 4096

    @patch("agent.completion")
    def test_default_model_uses_litellm_prefix(self, mock_completion):
        """Default model should use the provider/model format."""
        mock_completion.return_value = _mock_completion_response('{"spans":[]}')

        run_llm("text", api_key="k")

        call_kwargs = mock_completion.call_args
        assert call_kwargs.kwargs["model"] == "openai/gpt-4o-mini"

    @patch("agent.completion")
    def test_anthropic_model(self, mock_completion):
        """LiteLLM should support non-OpenAI models like Anthropic."""
        mock_completion.return_value = _mock_completion_response('{"spans":[]}')

        run_llm("text", api_key="k", model="anthropic/claude-sonnet-4-20250514")

        call_kwargs = mock_completion.call_args
        assert call_kwargs.kwargs["model"] == "anthropic/claude-sonnet-4-20250514"

    @patch("agent.completion")
    def test_openai_reasoning_effort_is_passed(self, mock_completion):
        mock_completion.return_value = _mock_completion_response('{"spans":[]}')
        run_llm_with_metadata(
            text="text",
            api_key="k",
            model="openai.gpt-5.3-codex",
            reasoning_effort="xhigh",
        )
        call_kwargs = mock_completion.call_args
        assert call_kwargs.kwargs["reasoning_effort"] == "xhigh"

    @patch("agent.completion")
    def test_drops_implausible_name_spans_even_with_valid_offsets(self, mock_completion):
        payload = _spans_payload(
            [
                {"start": 0, "end": 4, "label": "NAME", "text": "Good"},
                {"start": 6, "end": 10, "label": "NAME", "text": "Anna"},
            ]
        )
        mock_completion.return_value = _mock_completion_response(payload)
        result = run_llm_with_metadata(
            text="Good. Anna",
            api_key="k",
            model="openai/gpt-4o",
        )
        assert [(span.start, span.end, span.text) for span in result.spans] == [
            (6, 10, "Anna")
        ]
        assert any("implausible NAME span" in warning for warning in result.warnings)

    @patch("agent.completion")
    def test_drops_partial_word_name_fragments_with_valid_offsets(self, mock_completion):
        text = (
            "So how did you come up late?\n"
            "Three over three and they both look like X's which one is which?\n"
            "Wait, they both look like X's which one is which?\n"
            "Good job, Anna."
        )
        three_idx = text.index("Three")
        wait_idx = text.index("Wait")
        and_they_idx = text.index("and they")
        anna_idx = text.index("Anna")
        payload = _spans_payload(
            [
                {"start": three_idx + 2, "end": three_idx + 5, "label": "NAME", "text": "ree"},
                {"start": wait_idx + 1, "end": wait_idx + 5, "label": "NAME", "text": "ait,"},
                {"start": and_they_idx + 2, "end": and_they_idx + 6, "label": "NAME", "text": "d th"},
                {"start": anna_idx, "end": anna_idx + 4, "label": "NAME", "text": "Anna"},
            ]
        )
        mock_completion.return_value = _mock_completion_response(payload)

        result = run_llm_with_metadata(
            text=text,
            api_key="k",
            model="openai/gpt-4o",
        )
        assert [(span.start, span.end, span.text) for span in result.spans] == [
            (anna_idx, anna_idx + 4, "Anna")
        ]
        assert any("implausible NAME span" in warning for warning in result.warnings)

    @patch("agent.completion")
    def test_anthropic_thinking_is_passed(self, mock_completion):
        mock_completion.return_value = _mock_completion_response('{"spans":[]}')
        run_llm_with_metadata(
            text="text",
            api_key="k",
            model="anthropic.claude-4.6-opus",
            anthropic_thinking=True,
            anthropic_thinking_budget_tokens=2048,
            temperature=1.0,
        )
        call_kwargs = mock_completion.call_args
        assert call_kwargs.kwargs["thinking"]["type"] == "enabled"
        assert call_kwargs.kwargs["thinking"]["budget_tokens"] == 2048
        assert call_kwargs.kwargs["max_tokens"] > call_kwargs.kwargs["thinking"]["budget_tokens"]

    @patch("agent.completion")
    def test_unsupported_advanced_params_raise_error(self, mock_completion):
        mock_completion.side_effect = [
            Exception("Unsupported parameter: reasoning_effort"),
        ]
        with pytest.raises(Exception, match="Unsupported parameter"):
            run_llm_with_metadata(
                text="text",
                api_key="k",
                model="openai.gpt-5.2-chat",
                reasoning_effort="xhigh",
            )

    @patch("agent.completion")
    def test_api_base_retries_with_v1_suffix(self, mock_completion):
        mock_completion.side_effect = [
            Exception("404 Not Found"),
            _mock_completion_response('{"spans":[]}'),
        ]
        result = run_llm_with_metadata(
            text="text",
            api_key="k",
            api_base="https://api.example.com",
            model="openai.gpt-5.2-chat",
        )
        assert result.spans == []
        first_call = mock_completion.call_args_list[0].kwargs
        second_call = mock_completion.call_args_list[1].kwargs
        assert first_call["api_base"] == "https://api.example.com"
        assert second_call["api_base"] == "https://api.example.com/v1"
        assert any("succeeded after retrying" in warning for warning in result.warnings)

    @patch("agent.completion")
    def test_chat_model_omits_custom_temperature(self, mock_completion):
        mock_completion.return_value = _mock_completion_response('{"spans":[]}')
        result = run_llm_with_metadata(
            text="text",
            api_key="k",
            model="openai.gpt-5.2-chat",
            temperature=0.5,
        )
        assert result.spans == []
        call_kwargs = mock_completion.call_args
        assert "temperature" not in call_kwargs.kwargs
        assert any("default temperature" in warning for warning in result.warnings)

    @patch("agent.completion")
    def test_bare_gpt5_model_omits_custom_temperature(self, mock_completion):
        mock_completion.return_value = _mock_completion_response('{"spans":[]}')
        result = run_llm_with_metadata(
            text="text",
            api_key="k",
            model="gpt-5.4",
            temperature=0.5,
        )
        assert result.spans == []
        call_kwargs = mock_completion.call_args
        assert "temperature" not in call_kwargs.kwargs
        assert any("default temperature" in warning for warning in result.warnings)

    @patch("agent.completion")
    def test_gateway_dot_model_uses_openai_provider_format(self, mock_completion):
        mock_completion.return_value = _mock_completion_response('{"spans":[]}')
        run_llm_with_metadata(
            text="text",
            api_key="k",
            api_base="https://proxy.example.com",
            model="google.gemini-3.1-pro-preview",
        )
        call_kwargs = mock_completion.call_args
        assert call_kwargs.kwargs["model"] == "openai/google.gemini-3.1-pro-preview"

    @patch("agent.completion")
    def test_gateway_reasoning_passed_via_extra_body(self, mock_completion):
        mock_completion.return_value = _mock_completion_response('{"spans":[]}')
        run_llm_with_metadata(
            text="text",
            api_key="k",
            api_base="https://proxy.example.com",
            model="openai.gpt-5.3-codex",
            reasoning_effort="xhigh",
        )
        call_kwargs = mock_completion.call_args
        assert "reasoning_effort" not in call_kwargs.kwargs
        assert call_kwargs.kwargs["extra_body"]["reasoning_effort"] == "xhigh"

    @patch("agent.completion")
    def test_gateway_thinking_passed_via_extra_body(self, mock_completion):
        mock_completion.return_value = _mock_completion_response('{"spans":[]}')
        result = run_llm_with_metadata(
            text="text",
            api_key="k",
            api_base="https://proxy.example.com",
            model="anthropic.claude-4.6-opus",
            anthropic_thinking=True,
            anthropic_thinking_budget_tokens=2048,
            temperature=1.0,
        )
        call_kwargs = mock_completion.call_args
        assert "thinking" not in call_kwargs.kwargs
        assert call_kwargs.kwargs["temperature"] == 1.0
        assert call_kwargs.kwargs["extra_body"]["thinking"]["type"] == "enabled"
        assert (
            call_kwargs.kwargs["extra_body"]["max_tokens"]
            > call_kwargs.kwargs["extra_body"]["thinking"]["budget_tokens"]
        )
        assert result.spans == []

    @patch("agent.completion")
    def test_gateway_thinking_forces_temperature_one(self, mock_completion):
        mock_completion.return_value = _mock_completion_response('{"spans":[]}')
        result = run_llm_with_metadata(
            text="text",
            api_key="k",
            api_base="https://proxy.example.com",
            model="anthropic.claude-4.6-opus",
            anthropic_thinking=True,
            anthropic_thinking_budget_tokens=2048,
            temperature=0.0,
        )

        call_kwargs = mock_completion.call_args
        assert call_kwargs.kwargs["temperature"] == 1.0
        assert any(
            "overriding requested temperature to 1.0" in warning
            for warning in result.warnings
        )

    @patch("agent.completion")
    def test_anthropic_empty_thinking_output_raises_error(self, mock_completion):
        mock_completion.return_value = _mock_completion_response(None, finish_reason="length")

        with pytest.raises(ValueError, match="empty output content \\(finish_reason=length\\)"):
            run_llm_with_metadata(
                text="Hello Anna.",
                api_key="k",
                model="anthropic.claude-4.6-opus",
                anthropic_thinking=True,
                anthropic_thinking_budget_tokens=2048,
                temperature=1.0,
            )

        call_kwargs = mock_completion.call_args
        assert call_kwargs.kwargs["thinking"]["type"] == "enabled"

    @patch("agent.completion")
    def test_openai_model_enables_logprobs_and_computes_confidence(self, mock_completion):
        mock_completion.return_value = _mock_completion_response(
            '{"spans":[]}', token_logprobs=[-0.1, -0.2, -0.3]
        )
        result = run_llm_with_metadata(
            text="text",
            api_key="k",
            model="openai/gpt-4o",
        )
        call_kwargs = mock_completion.call_args
        assert call_kwargs.kwargs["logprobs"] is True
        assert result.llm_confidence.available is True
        assert result.llm_confidence.reason == "ok"
        assert result.llm_confidence.token_count == 3
        assert result.llm_confidence.confidence is not None
        assert result.llm_confidence.perplexity is not None
        assert result.llm_confidence.band in {"high", "medium", "low"}

    @patch("agent.completion")
    def test_non_openai_model_omits_logprobs(self, mock_completion):
        mock_completion.return_value = _mock_completion_response('{"spans":[]}')
        result = run_llm_with_metadata(
            text="text",
            api_key="k",
            model="anthropic.claude-4.6-opus",
        )
        call_kwargs = mock_completion.call_args
        assert "logprobs" not in call_kwargs.kwargs
        assert result.llm_confidence.available is False
        assert result.llm_confidence.reason == "unsupported_provider"
        assert result.llm_confidence.band == "na"

    @patch("agent.completion")
    def test_openai_missing_logprobs_sets_unavailable_metric(self, mock_completion):
        mock_completion.return_value = _mock_completion_response('{"spans":[]}')
        result = run_llm_with_metadata(
            text="text",
            api_key="k",
            model="openai/gpt-4o",
        )
        assert result.llm_confidence.available is False
        assert result.llm_confidence.reason == "missing_logprobs"
        assert result.llm_confidence.band == "na"
        assert any("did not include token logprobs" in warning for warning in result.warnings)

    @patch("agent.completion")
    def test_openai_missing_logprobs_uses_usage_completion_tokens(self, mock_completion):
        mock_completion.return_value = _mock_completion_response(
            '{"spans":[]}', completion_tokens=42
        )
        result = run_llm_with_metadata(
            text="text",
            api_key="k",
            model="openai/gpt-4o",
        )
        assert result.llm_confidence.available is False
        assert result.llm_confidence.reason == "missing_logprobs"
        assert result.llm_confidence.token_count == 42

    @patch("agent.completion")
    def test_openai_logprobs_fallback_retries_without_param(self, mock_completion):
        mock_completion.side_effect = [
            Exception("UnsupportedParamsError: openai does not support parameters: ['logprobs']"),
            _mock_completion_response('{"spans":[]}'),
        ]
        result = run_llm_with_metadata(
            text="text",
            api_key="k",
            model="openai/gpt-4o",
        )
        assert result.spans == []
        first_call = mock_completion.call_args_list[0].kwargs
        second_call = mock_completion.call_args_list[1].kwargs
        assert first_call["logprobs"] is True
        assert "logprobs" not in second_call
        assert any("rejected logprobs" in warning for warning in result.warnings)

    @patch("agent.completion")
    def test_openai_gpt5_family_attempts_logprobs(self, mock_completion):
        mock_completion.return_value = _mock_completion_response(
            '{"spans":[]}', token_logprobs=[-0.2, -0.1]
        )
        result = run_llm_with_metadata(
            text="text",
            api_key="k",
            model="openai.gpt-5.3-codex",
        )
        call_kwargs = mock_completion.call_args
        assert call_kwargs.kwargs["logprobs"] is True
        assert result.llm_confidence.available is True
        assert result.llm_confidence.reason == "ok"

    @patch("agent.completion")
    def test_openai_gpt5_chat_skips_logprobs_probe(self, mock_completion):
        mock_completion.return_value = _mock_completion_response('{"spans":[]}')
        result = run_llm_with_metadata(
            text="text",
            api_key="k",
            model="openai.gpt-5.2-chat",
            reasoning_effort="xhigh",
        )
        call_kwargs = mock_completion.call_args
        assert "logprobs" not in call_kwargs.kwargs
        assert any("currently unavailable for model 'openai.gpt-5.2-chat'" in warning for warning in result.warnings)

    @patch("agent.completion")
    def test_openai_gpt5_chat_with_none_still_skips_logprobs_probe(self, mock_completion):
        mock_completion.return_value = _mock_completion_response('{"spans":[]}')
        result = run_llm_with_metadata(
            text="text",
            api_key="k",
            model="openai.gpt-5.2-chat",
            reasoning_effort="none",
        )
        call_kwargs = mock_completion.call_args
        assert "logprobs" not in call_kwargs.kwargs
        assert result.llm_confidence.available is False
        assert result.llm_confidence.reason == "missing_logprobs"
        assert any("currently unavailable for model 'openai.gpt-5.2-chat'" in warning for warning in result.warnings)

    @patch("agent.completion")
    def test_response_format_fallback_retries_without_schema(self, mock_completion):
        calls: list[dict] = []

        def _side_effect(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise Exception(
                    "UnsupportedParamsError: openai does not support parameters: ['response_format']"
                )
            return _mock_completion_response('{"spans":[]}')

        mock_completion.side_effect = _side_effect
        result = run_llm_with_metadata(
            text="text",
            api_key="k",
            model="openai/gpt-4o",
        )
        assert result.spans == []
        assert "response_format" in calls[0]
        assert "response_format" not in calls[1]
        assert any("rejected response_format" in warning for warning in result.warnings)

    @patch("agent.completion")
    def test_advanced_profile_response_schema_enforces_label_enum(self, mock_completion):
        mock_completion.return_value = _mock_completion_response('{"spans":[]}')
        run_llm_with_metadata(
            text="text",
            api_key="k",
            model="openai/gpt-4o",
            label_profile="advanced",
        )
        call_kwargs = mock_completion.call_args
        enum_values = (
            call_kwargs.kwargs["response_format"]["json_schema"]["schema"]["properties"]["spans"][
                "items"
            ]["properties"]["label"]["enum"]
        )
        assert "PERSON" in enum_values
        assert "COURSE" in enum_values
        assert "US_SSN" in enum_values

    @patch("agent.completion")
    def test_simple_profile_response_schema_enforces_label_enum(self, mock_completion):
        mock_completion.return_value = _mock_completion_response('{"spans":[]}')
        run_llm_with_metadata(
            text="text",
            api_key="k",
            model="openai/gpt-4o",
            label_profile="simple",
        )
        call_kwargs = mock_completion.call_args
        enum_values = (
            call_kwargs.kwargs["response_format"]["json_schema"]["schema"]["properties"]["spans"][
                "items"
            ]["properties"]["label"]["enum"]
        )
        assert "NAME" in enum_values
        assert "EMAIL" in enum_values
        assert "MISC_ID" in enum_values

    def test_build_extraction_system_prompt_is_label_profile_specific(self):
        simple_prompt = build_extraction_system_prompt(SYSTEM_PROMPT, label_profile="simple")
        advanced_prompt = build_extraction_system_prompt(
            SYSTEM_PROMPT,
            label_profile="advanced",
        )

        assert '{"spans": []}' in simple_prompt
        assert "Allowed labels for this run: AGE, DATE, EMAIL, LOCATION, MISC_ID, NAME, PHONE, SCHOOL, URL." in simple_prompt
        assert "TIME" not in simple_prompt
        assert "PERSON" in advanced_prompt
        assert "US_SSN" in advanced_prompt

    @patch("agent.completion")
    def test_simple_profile_maps_person_name_and_time_labels(self, mock_completion):
        payload = _spans_payload(
            [
                {"start": 0, "end": 4, "label": "PERSON_NAME", "text": "John"},
                {"start": 13, "end": 29, "label": "EMAIL", "text": "john@example.com"},
                {"start": 33, "end": 40, "label": "TIME", "text": "3:00 PM"},
            ]
        )
        mock_completion.return_value = _mock_completion_response(payload)
        result = run_llm_with_metadata(
            text="John emailed john@example.com at 3:00 PM",
            api_key="k",
            model="openai/gpt-4o",
            label_profile="simple",
        )
        assert [(span.label, span.text) for span in result.spans] == [
            ("NAME", "John"),
            ("EMAIL", "john@example.com"),
            ("DATE", "3:00 PM"),
        ]

    @patch("agent.completion")
    def test_parse_failure_uses_one_repair_retry(self, mock_completion):
        mock_completion.side_effect = [
            _mock_completion_response("not valid json"),
            _mock_completion_response('{"spans":[{"start":3,"end":7,"label":"NAME","text":"John"}]}'),
        ]
        result = run_llm_with_metadata(
            text="Hi John!",
            api_key="k",
            model="openai/gpt-4o",
        )
        assert len(result.spans) == 1
        assert result.spans[0].text == "John"
        assert mock_completion.call_count == 2
        assert any("repair retry" in warning for warning in result.warnings)

    @patch("agent.completion")
    def test_parse_failure_repair_retry_uses_repair_limits(self, mock_completion):
        calls: list[dict[str, object]] = []

        def side_effect(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return _mock_completion_response("not valid json")
            return _mock_completion_response('{"spans":[]}')

        mock_completion.side_effect = side_effect

        run_llm_with_metadata(
            text="Hi John!",
            api_key="k",
            model="openai/gpt-4o",
        )

        assert calls[0]["timeout"] == pytest.approx(60.0)
        assert calls[0]["max_tokens"] == 4096
        assert calls[1]["timeout"] == pytest.approx(30.0)
        assert calls[1]["max_tokens"] == 1024

    @patch("agent.completion")
    def test_parse_failure_repair_retry_scales_budget_for_large_truncated_output(
        self, mock_completion
    ):
        calls: list[dict[str, object]] = []
        items = [
            {
                "start": idx * 10,
                "end": idx * 10 + 4,
                "label": "NAME",
                "text": f"N{idx:02d}",
            }
            for idx in range(96)
        ]
        truncated_payload = _spans_payload(items)[:-24]

        def side_effect(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                return _mock_completion_response(truncated_payload, finish_reason="length")
            return _mock_completion_response('{"spans":[]}')

        mock_completion.side_effect = side_effect

        run_llm_with_metadata(
            text=" ".join(f"name{idx}" for idx in range(200)),
            api_key="k",
            model="openai/gpt-4o",
        )

        assert calls[0]["max_tokens"] == 4096
        assert calls[1]["timeout"] == pytest.approx(30.0)
        assert calls[1]["max_tokens"] > 1024
        repair_messages = calls[1]["messages"]
        assert "INVALID_OUTPUT:" in repair_messages[1]["content"]
        assert "TRANSCRIPT:" not in repair_messages[1]["content"]

    @patch("agent.completion")
    def test_truncated_output_repair_failure_recovers_partial_spans(self, mock_completion):
        text = "John met Mary yesterday."
        truncated_payload = (
            '{"spans":['
            '{"start":0,"end":4,"label":"NAME","text":"John"},'
            '{"start":9,"end":13,"label":"NAME","text":"Mary"},'
            '{"start":14'
        )
        mock_completion.side_effect = [
            _mock_completion_response(truncated_payload, finish_reason="length"),
            _mock_completion_response(None, finish_reason="length"),
        ]

        result = run_llm_with_metadata(
            text=text,
            api_key="k",
            model="openai/gpt-4o",
        )

        assert [(span.label, span.text) for span in result.spans] == [
            ("NAME", "John"),
            ("NAME", "Mary"),
        ]
        assert any(
            "Recovered 2 span(s) from truncated LLM output after repair retry failed."
            in warning
            for warning in result.warnings
        )

    @patch("agent.completion")
    def test_parse_failure_after_repair_raises_clear_error(self, mock_completion):
        mock_completion.side_effect = [
            _mock_completion_response("not valid json"),
            _mock_completion_response("still not json"),
        ]
        with pytest.raises(ValueError, match="one repair retry failed"):
            run_llm_with_metadata(
                text="Hi John!",
                api_key="k",
                model="openai/gpt-4o",
            )

    @patch("agent.completion")
    def test_advanced_profile_filters_course_and_non_specific_date(self, mock_completion):
        text = "Monday algebra Algebra 300 January 5, 2026"
        monday_start = text.index("Monday")
        monday_end = monday_start + len("Monday")
        invalid_course_start = text.index("algebra")
        invalid_course_end = invalid_course_start + len("algebra")
        valid_course_start = text.index("Algebra 300")
        valid_course_end = valid_course_start + len("Algebra 300")
        valid_date_start = text.index("January 5, 2026")
        valid_date_end = valid_date_start + len("January 5, 2026")

        payload = _spans_payload(
            [
                {
                    "start": monday_start,
                    "end": monday_end,
                    "label": "DATE",
                    "text": "Monday",
                },
                {
                    "start": invalid_course_start,
                    "end": invalid_course_end,
                    "label": "COURSE",
                    "text": "algebra",
                },
                {
                    "start": valid_course_start,
                    "end": valid_course_end,
                    "label": "COURSE",
                    "text": "Algebra 300",
                },
                {
                    "start": valid_date_start,
                    "end": valid_date_end,
                    "label": "DATE",
                    "text": "January 5, 2026",
                },
            ]
        )
        mock_completion.return_value = _mock_completion_response(payload)

        result = run_llm_with_metadata(
            text=text,
            api_key="k",
            model="openai/gpt-4o",
            label_profile="advanced",
        )
        assert [(span.label, span.text) for span in result.spans] == [
            ("COURSE", "Algebra 300"),
            ("DATE", "January 5, 2026"),
        ]

def test_run_method_with_metadata_emits_dual_pass_progress(monkeypatch):
    progress_events: list[tuple[int, str]] = []

    def fake_run_llm_with_metadata(**kwargs):
        return types.SimpleNamespace(
            spans=[CanonicalSpan(start=0, end=4, label="NAME", text="Anna")],
            warnings=[],
            llm_confidence=LLMConfidenceMetric(
                available=False,
                provider="gemini",
                model="google.gemini-3.1-pro-preview",
                reason="unsupported_provider",
                token_count=0,
                band="na",
            ),
        )

    monkeypatch.setattr("agent.run_llm_with_metadata", fake_run_llm_with_metadata)

    result = run_method_with_metadata(
        text="Anna emailed anna@example.com",
        method_id="dual",
        api_key="k",
        api_base="https://proxy.example.com/v1",
        model="google.gemini-3.1-pro-preview",
        system_prompt="",
        temperature=0.0,
        reasoning_effort="none",
        anthropic_thinking=False,
        anthropic_thinking_budget_tokens=None,
        method_verify=False,
        progress_callback=lambda pass_index, pass_label: progress_events.append(
            (pass_index, pass_label)
        ),
    )

    assert result.spans
    assert progress_events == [(1, "dual:names"), (2, "dual:identifiers")]


@patch("agent.completion")
def test_run_llm_verifier_defaults_timeout_and_max_tokens(mock_completion):
    mock_completion.return_value = _mock_completion_response('{"decisions":[]}')

    spans, warnings = agent._run_llm_verifier(
        "Hello Anna",
        [CanonicalSpan(start=6, end=10, label="NAME", text="Anna")],
        api_key="k",
        api_base=None,
        model="openai/gpt-4o",
    )

    assert spans == [CanonicalSpan(start=6, end=10, label="NAME", text="Anna")]
    assert warnings == []
    call_kwargs = mock_completion.call_args
    assert call_kwargs.kwargs["timeout"] == pytest.approx(30.0)
    assert call_kwargs.kwargs["max_tokens"] == 1024


def test_model_presets_include_requested_options():
    model_ids = {preset["model"] for preset in MODEL_PRESETS}
    assert "openai.gpt-5.3-codex" in model_ids
    assert "openai.gpt-5.2-chat" in model_ids
    assert "anthropic.claude-4.6-opus" in model_ids
    assert "anthropic.claude-4.6-sonnet" in model_ids
    assert "google.gemini-3.1-pro-preview" in model_ids
    assert "google.gemini-3.1-flash-lite-preview" in model_ids


def test_method_prompts_are_migrated_from_experiment_presets():
    extended_prompt = METHOD_DEFINITION_BY_ID["extended"]["passes"][0]["prompt"]
    assert "HIPAA Safe Harbor" in extended_prompt
    assert "Method or theorem names containing a person's name" in extended_prompt

    dual_numbers_prompt = METHOD_DEFINITION_BY_ID["dual-split"]["passes"][1]["prompt"]
    assert "Mathematical expressions" in dual_numbers_prompt
    assert "Output data following the provided JSON schema." in dual_numbers_prompt

    split_prompt = METHOD_DEFINITION_BY_ID["presidio+llm-split"]["passes"][1]["prompt"]
    assert "reasonably identify an individual" in split_prompt


def test_audited_method_contracts_validate_cleanly():
    assert agent.validate_method_contracts(method_bundle="audited") == []


def test_test_method_contracts_validate_cleanly():
    assert agent.validate_method_contracts(method_bundle="test") == []


def test_v2_post_process_method_contracts_validate_cleanly():
    assert agent.validate_method_contracts(method_bundle="v2+post-process") == []


def test_v2_method_contracts_validate_cleanly():
    assert agent.validate_method_contracts(method_bundle="v2") == []


def test_deidentify_v2_method_contracts_validate_cleanly():
    assert agent.validate_method_contracts(method_bundle="deidentify-v2") == []


def test_test_bundle_prompt_wrapper_includes_examples_and_label_catalog():
    prompt = build_extraction_system_prompt(
        "Base prompt",
        label_profile="simple",
        method_bundle="test",
    )

    assert "Use only these exact label strings" in prompt
    assert '{"spans":[]}' in prompt
    assert '"label":"NAME"' in prompt
    assert "ages over 89" in prompt
    assert "geographic locations smaller than a state" in prompt


def test_normalize_method_bundle_accepts_v2_post_process():
    assert _normalize_method_bundle("v2+post-process") == "v2+post-process"


def test_normalize_method_bundle_accepts_v2():
    assert _normalize_method_bundle("v2") == "v2"


def test_normalize_method_bundle_accepts_deidentify_v2():
    assert _normalize_method_bundle("deidentify-v2") == "deidentify-v2"


def test_expand_detected_value_occurrences_repeats_same_name():
    text = "Anna met Anna at school."
    spans = [CanonicalSpan(start=0, end=4, label="NAME", text="Anna")]

    expanded = _expand_detected_value_occurrences(text, spans)

    assert [(span.start, span.end, span.label, span.text) for span in expanded] == [
        (0, 4, "NAME", "Anna"),
        (9, 13, "NAME", "Anna"),
    ]


def test_expand_detected_value_occurrences_prefers_longer_overlap():
    text = "Call 555-123-4567 now"
    spans = [
        CanonicalSpan(start=5, end=17, label="PHONE", text="555-123-4567"),
        CanonicalSpan(start=9, end=17, label="MISC_ID", text="123-4567"),
    ]

    expanded = _expand_detected_value_occurrences(text, spans)

    assert [(span.label, span.text) for span in expanded] == [("PHONE", "555-123-4567")]


def test_list_agent_methods_reports_profile_metadata_for_audited_bundle():
    methods = agent.list_agent_methods(method_bundle="audited")
    dual_split = next(item for item in methods if item["id"] == "dual-split")

    assert dual_split["supported_label_profiles"] == ["advanced", "simple"]
    assert "SCHOOL" in dual_split["output_labels_by_profile"]["simple"]
    assert "SCHOOL" in dual_split["output_labels_by_profile"]["advanced"]
    assert isinstance(dual_split["known_limitations"], list)


def test_list_agent_methods_reports_deidentify_v2_catalog_entries():
    methods = agent.list_agent_methods(method_bundle="deidentify-v2")

    method_ids = {item["id"] for item in methods}

    assert "dual-v2" in method_ids
    assert "regex+dual-v2" in method_ids
    assert "presidio-lite+extended-v2" in method_ids


def test_deidentify_v2_dual_method_uses_exact_legacy_prompt_text():
    definition = agent.get_method_definition_by_id("dual-v2", method_bundle="deidentify-v2")
    assert definition is not None

    names_prompt = definition["passes"][0]["prompt"]
    identifiers_prompt = definition["passes"][1]["prompt"]

    assert "reviewing tutoring chat transcripts per HIPAA Safe Harbor" in names_prompt
    assert "Personal names are handled separately" in identifiers_prompt
    assert "[SCHOOL]" in identifiers_prompt


def test_audited_presidio_default_uses_residual_llm_scope():
    definition = agent.get_method_definition_by_id(
        "presidio+default",
        method_bundle="audited",
    )
    assert definition is not None

    llm_pass = definition["passes"][1]
    assert llm_pass["entity_types_by_profile"]["simple"] == [
        "NAME",
        "LOCATION",
        "SCHOOL",
        "DATE",
        "AGE",
        "MISC_ID",
    ]
    assert set(llm_pass["entity_types_by_profile"]["advanced"]) == {
        "AGE",
        "COURSE",
        "DATE",
        "GRADE_LEVEL",
        "LOCATION",
        "NRP",
        "PERSON",
        "SCHOOL",
        "SOCIAL_HANDLE",
        "US_DRIVER_LICENSE",
        "US_PASSPORT",
    }


def test_audited_presidio_llm_split_advanced_labels_stay_within_schema():
    definition = agent.get_method_definition_by_id(
        "presidio+llm-split",
        method_bundle="audited",
    )
    assert definition is not None

    llm_pass = definition["passes"][1]
    advanced_labels = set(llm_pass["entity_types_by_profile"]["advanced"])
    assert "DEVICE_IDENTIFIER" not in advanced_labels
    assert "BIOMETRIC_IDENTIFIER" not in advanced_labels
    assert "IMAGE" not in advanced_labels
    assert "IDENTIFYING_NUMBER" not in advanced_labels
    assert advanced_labels <= set(agent.ADVANCED_LABELS)


def test_run_method_with_metadata_audited_dual_split_keeps_school_output(monkeypatch):
    llm_calls: list[dict[str, object]] = []

    def fake_run_llm_with_metadata(**kwargs):
        llm_calls.append(kwargs)
        if len(llm_calls) == 1:
            spans = [CanonicalSpan(start=0, end=4, label="SCHOOL", text="Yale")]
        else:
            spans = []
        return types.SimpleNamespace(
            spans=spans,
            raw_spans=spans,
            warnings=[],
            llm_confidence=LLMConfidenceMetric(
                available=False,
                provider="gemini",
                model="google.gemini-3.1-pro-preview",
                reason="unsupported_provider",
                token_count=0,
                band="na",
            ),
            response_debug=[],
            resolution_events=[],
            resolution_policy_version=None,
        )

    monkeypatch.setattr("agent.run_llm_with_metadata", fake_run_llm_with_metadata)

    result = run_method_with_metadata(
        text="Yale",
        method_id="dual-split",
        api_key="k",
        api_base="https://proxy.example.com/v1",
        model="google.gemini-3.1-pro-preview",
        system_prompt="",
        temperature=0.0,
        reasoning_effort="none",
        anthropic_thinking=False,
        anthropic_thinking_budget_tokens=None,
        method_verify=False,
        label_profile="simple",
        method_bundle="audited",
    )

    assert [(span.label, span.text) for span in result.spans] == [("SCHOOL", "Yale")]


def test_run_method_with_metadata_v2_post_process_expands_repeated_occurrences(monkeypatch):
    llm_call_count = 0

    def fake_run_llm_with_metadata(**kwargs):
        nonlocal llm_call_count
        llm_call_count += 1
        spans = [CanonicalSpan(start=0, end=4, label="NAME", text="Anna")]
        return types.SimpleNamespace(
            spans=spans,
            raw_spans=spans,
            warnings=[],
            llm_confidence=LLMConfidenceMetric(
                available=False,
                provider="gemini",
                model="google.gemini-3.1-pro-preview",
                reason="unsupported_provider",
                token_count=0,
                band="na",
            ),
            response_debug=[],
            resolution_events=[],
            resolution_policy_version=None,
        )

    monkeypatch.setattr("agent.run_llm_with_metadata", fake_run_llm_with_metadata)

    result = run_method_with_metadata(
        text="Anna met Anna.",
        method_id="default",
        api_key="k",
        api_base="https://proxy.example.com/v1",
        model="google.gemini-3.1-pro-preview",
        system_prompt="",
        temperature=0.0,
        reasoning_effort="none",
        anthropic_thinking=False,
        anthropic_thinking_budget_tokens=None,
        method_verify=False,
        label_profile="simple",
        method_bundle="v2+post-process",
    )

    assert llm_call_count == 1
    assert [(span.start, span.end, span.text) for span in result.raw_spans] == [
        (0, 4, "Anna"),
        (9, 13, "Anna"),
    ]
    assert [(span.start, span.end, span.text) for span in result.spans] == [
        (0, 4, "Anna"),
        (9, 13, "Anna"),
    ]


def test_run_method_with_metadata_deidentify_v2_expands_text_matches(monkeypatch):
    completion_calls: list[dict[str, object]] = []

    def fake_completion(**kwargs):
        completion_calls.append(kwargs)
        if len(completion_calls) == 1:
            content = json.dumps({"matches": [{"entity_type": "NAME", "text": "Anna"}]})
        else:
            content = json.dumps({"matches": []})
        return _mock_completion_response(content)

    monkeypatch.setattr("agent.completion", fake_completion)

    result = run_method_with_metadata(
        text="Anna met Anna.",
        method_id="dual-v2",
        api_key="k",
        api_base="https://proxy.example.com/v1",
        model="google.gemini-3.1-pro-preview",
        system_prompt="",
        temperature=0.0,
        reasoning_effort="none",
        anthropic_thinking=False,
        anthropic_thinking_budget_tokens=None,
        method_verify=False,
        label_profile="simple",
        method_bundle="deidentify-v2",
    )

    assert len(completion_calls) == 2
    assert completion_calls[0]["response_format"]["json_schema"]["name"] == "pii_matches"
    assert [(span.start, span.end, span.label, span.text) for span in result.raw_spans] == [
        (0, 4, "NAME", "Anna"),
        (9, 13, "NAME", "Anna"),
    ]
    assert [(span.start, span.end, span.label, span.text) for span in result.spans] == [
        (0, 4, "NAME", "Anna"),
        (9, 13, "NAME", "Anna"),
    ]
    assert result.resolution_events == []


def test_run_method_with_metadata_deidentify_v2_respects_word_boundaries(monkeypatch):
    completion_calls: list[dict[str, object]] = []

    def fake_completion(**kwargs):
        completion_calls.append(kwargs)
        if len(completion_calls) == 1:
            content = json.dumps({"matches": [{"entity_type": "ADDRESS", "text": "Chicago"}]})
        else:
            content = json.dumps({"matches": []})
        return _mock_completion_response(content)

    monkeypatch.setattr("agent.completion", fake_completion)

    result = run_method_with_metadata(
        text="Chicagoan Chicago",
        method_id="dual-v2",
        api_key="k",
        api_base="https://proxy.example.com/v1",
        model="google.gemini-3.1-pro-preview",
        system_prompt="",
        temperature=0.0,
        reasoning_effort="none",
        anthropic_thinking=False,
        anthropic_thinking_budget_tokens=None,
        method_verify=False,
        label_profile="simple",
        method_bundle="deidentify-v2",
    )

    assert [(span.start, span.end, span.label, span.text) for span in result.raw_spans] == [
        (10, 17, "ADDRESS", "Chicago"),
    ]


def test_run_method_with_metadata_deidentify_v2_extended_uses_original_entity_order(
    monkeypatch,
):
    completion_calls: list[dict[str, object]] = []

    def fake_completion(**kwargs):
        completion_calls.append(kwargs)
        return _mock_completion_response(json.dumps({"matches": []}))

    monkeypatch.setattr("agent.completion", fake_completion)

    run_method_with_metadata(
        text="Nothing to find here.",
        method_id="presidio-lite+extended-v2",
        api_key="k",
        api_base="https://proxy.example.com/v1",
        model="google.gemini-3.1-pro-preview",
        system_prompt="",
        temperature=0.0,
        reasoning_effort="none",
        anthropic_thinking=False,
        anthropic_thinking_budget_tokens=None,
        method_verify=False,
        label_profile="simple",
        method_bundle="deidentify-v2",
    )

    llm_call = completion_calls[0]
    enum_values = llm_call["response_format"]["json_schema"]["schema"]["properties"]["matches"][
        "items"
    ]["properties"]["entity_type"]["enum"]
    assert enum_values == [
        "NAME",
        "ADDRESS",
        "DATE",
        "PHONE_NUMBER",
        "FAX_NUMBER",
        "EMAIL",
        "SSN",
        "ACCOUNT_NUMBER",
        "DEVICE_IDENTIFIER",
        "URL",
        "IP_ADDRESS",
        "BIOMETRIC_IDENTIFIER",
        "IMAGE",
        "IDENTIFYING_NUMBER",
        "SCHOOL",
    ]


def test_run_method_with_metadata_audited_does_not_expand_repeated_occurrences(monkeypatch):
    def fake_run_llm_with_metadata(**kwargs):
        spans = [CanonicalSpan(start=0, end=4, label="NAME", text="Anna")]
        return types.SimpleNamespace(
            spans=spans,
            raw_spans=spans,
            warnings=[],
            llm_confidence=LLMConfidenceMetric(
                available=False,
                provider="gemini",
                model="google.gemini-3.1-pro-preview",
                reason="unsupported_provider",
                token_count=0,
                band="na",
            ),
            response_debug=[],
            resolution_events=[],
            resolution_policy_version=None,
        )

    monkeypatch.setattr("agent.run_llm_with_metadata", fake_run_llm_with_metadata)

    result = run_method_with_metadata(
        text="Anna met Anna.",
        method_id="default",
        api_key="k",
        api_base="https://proxy.example.com/v1",
        model="google.gemini-3.1-pro-preview",
        system_prompt="",
        temperature=0.0,
        reasoning_effort="none",
        anthropic_thinking=False,
        anthropic_thinking_budget_tokens=None,
        method_verify=False,
        label_profile="simple",
        method_bundle="audited",
    )

    assert [(span.start, span.end, span.text) for span in result.raw_spans] == [
        (0, 4, "Anna"),
    ]
    assert [(span.start, span.end, span.text) for span in result.spans] == [
        (0, 4, "Anna"),
    ]


def test_presidio_runtime_error_caches_success(monkeypatch):
    _reset_presidio_runtime_state()

    fake_spacy = types.SimpleNamespace(load=MagicMock())
    monkeypatch.setattr(
        "agent.importlib.util.find_spec",
        lambda name: object() if name in {"presidio_analyzer", "spacy"} else None,
    )
    monkeypatch.setitem(sys.modules, "spacy", fake_spacy)

    assert _presidio_runtime_error() is None
    assert _presidio_runtime_error() is None

    assert fake_spacy.load.call_count == 1


def test_run_presidio_pass_reuses_analyzer_across_threads(monkeypatch):
    _reset_presidio_runtime_state()

    analyzer_init_count = 0
    provider_init_count = 0

    class FakeResult:
        def __init__(self, start: int, end: int, entity_type: str):
            self.start = start
            self.end = end
            self.entity_type = entity_type

    class FakeAnalyzerEngine:
        def __init__(self, *, nlp_engine):
            nonlocal analyzer_init_count
            analyzer_init_count += 1
            self._nlp_engine = nlp_engine

        def analyze(self, *, text, language, entities, score_threshold):
            assert language == "en"
            assert score_threshold == 0.0
            return [FakeResult(0, min(len(text), 4), entities[0] if entities else "PERSON")]

    class FakeNlpEngineProvider:
        def __init__(self, *, nlp_configuration):
            nonlocal provider_init_count
            provider_init_count += 1
            self._nlp_configuration = nlp_configuration

        def create_engine(self):
            return {"config": self._nlp_configuration}

    fake_presidio_module = types.ModuleType("presidio_analyzer")
    fake_presidio_module.AnalyzerEngine = FakeAnalyzerEngine
    fake_nlp_module = types.ModuleType("presidio_analyzer.nlp_engine")
    fake_nlp_module.NlpEngineProvider = FakeNlpEngineProvider

    monkeypatch.setattr("agent._presidio_runtime_error", lambda: None)
    monkeypatch.setitem(sys.modules, "presidio_analyzer", fake_presidio_module)
    monkeypatch.setitem(sys.modules, "presidio_analyzer.nlp_engine", fake_nlp_module)

    with ThreadPoolExecutor(max_workers=3) as pool:
        texts = ["Alice Example", "Bob Example", "Carol Example"]
        results = list(pool.map(lambda text: _run_presidio_pass(text, ["PERSON"]), texts))

    assert [[span.text for span in spans] for spans in results] == [["Alic"], ["Bob "], ["Caro"]]
    assert provider_init_count == 1
    assert analyzer_init_count == 1


def test_run_presidio_pass_reuses_analyzer_within_thread(monkeypatch):
    _reset_presidio_runtime_state()

    analyzer_init_count = 0
    provider_init_count = 0

    class FakeResult:
        def __init__(self, start: int, end: int, entity_type: str):
            self.start = start
            self.end = end
            self.entity_type = entity_type

    class FakeAnalyzerEngine:
        def __init__(self, *, nlp_engine):
            nonlocal analyzer_init_count
            analyzer_init_count += 1
            self._nlp_engine = nlp_engine

        def analyze(self, *, text, language, entities, score_threshold):
            assert language == "en"
            assert score_threshold == 0.0
            return [FakeResult(0, min(len(text), 4), entities[0] if entities else "PERSON")]

    class FakeNlpEngineProvider:
        def __init__(self, *, nlp_configuration):
            nonlocal provider_init_count
            provider_init_count += 1
            self._nlp_configuration = nlp_configuration

        def create_engine(self):
            return {"config": self._nlp_configuration}

    fake_presidio_module = types.ModuleType("presidio_analyzer")
    fake_presidio_module.AnalyzerEngine = FakeAnalyzerEngine
    fake_nlp_module = types.ModuleType("presidio_analyzer.nlp_engine")
    fake_nlp_module.NlpEngineProvider = FakeNlpEngineProvider

    monkeypatch.setattr("agent._presidio_runtime_error", lambda: None)
    monkeypatch.setitem(sys.modules, "presidio_analyzer", fake_presidio_module)
    monkeypatch.setitem(sys.modules, "presidio_analyzer.nlp_engine", fake_nlp_module)

    spans_first = _run_presidio_pass("Alice Example", ["PERSON"])
    spans_second = _run_presidio_pass("Bob Example", ["PERSON"])

    assert [span.text for span in spans_first] == ["Alic"]
    assert [span.text for span in spans_second] == ["Bob "]
    assert provider_init_count == 1
    assert analyzer_init_count == 1


def test_agent_does_not_keep_stale_name_affix_helpers():
    assert not hasattr(agent, "_snap_name_boundary_affixes")
    assert not hasattr(agent, "_canonicalize_name_boundary_text")
