"""Tests for agent.py: regex-based and LLM-based PII detection via LiteLLM."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from agent import (
    METHOD_DEFINITION_BY_ID,
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


def _mock_completion_response(
    content: object,
    token_logprobs: list[float] | None = None,
    finish_reason: str | None = None,
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
    def test_repairs_offset_mismatches_using_span_text(self, mock_completion):
        payload = json.dumps(
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
        payload = json.dumps(
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
        payload = json.dumps(
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
    def test_warns_when_finish_reason_is_length(self, mock_completion):
        payload = json.dumps(
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
        payload = json.dumps(
            {
                "spans": [
                    {"start": 0, "end": 4, "label": "NAME", "text": "John"},
                ]
            }
        )
        mock_completion.return_value = _mock_completion_response(payload)
        spans = run_llm("Hi John!", api_key="test-key")
        assert len(spans) == 1
        assert spans[0].text == "John"

    @patch("agent.completion")
    def test_parses_json_embedded_in_text(self, mock_completion):
        payload = (
            "I found entities.\n"
            '[{"start": 0, "end": 4, "label": "NAME", "text": "John"}]'
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
                "text": '[{"start": 3, "end": 7, "label": "NAME", "text": "John"}]',
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
            '```json\n[{"start": 0, "end": 4, "label": "NAME", "text": "John"}]\n```'
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

        with pytest.raises(ValueError, match="top-level array or object with 'spans'"):
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
    def test_none_content_adds_empty_output_warning(self, mock_completion):
        message = MagicMock()
        message.content = None
        choice = MagicMock()
        choice.message = message
        resp = MagicMock()
        resp.choices = [choice]
        mock_completion.return_value = resp

        result = run_llm_with_metadata(
            text="text",
            api_key="test-key",
            model="openai/gpt-4o",
        )
        assert result.spans == []
        assert any("empty output content" in warning for warning in result.warnings)

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
            model="openai.gpt-5.3-codex",
            reasoning_effort="xhigh",
        )
        call_kwargs = mock_completion.call_args
        assert call_kwargs.kwargs["reasoning_effort"] == "xhigh"

    @patch("agent.completion")
    def test_drops_implausible_name_spans_even_with_valid_offsets(self, mock_completion):
        payload = json.dumps(
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
    def test_anthropic_thinking_is_passed(self, mock_completion):
        mock_completion.return_value = _mock_completion_response("[]")
        run_llm_with_metadata(
            text="text",
            api_key="k",
            model="anthropic.claude-4.6-opus",
            anthropic_thinking=True,
            anthropic_thinking_budget_tokens=2048,
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
            _mock_completion_response("[]"),
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
        mock_completion.return_value = _mock_completion_response("[]")
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
    def test_gateway_dot_model_uses_openai_provider_format(self, mock_completion):
        mock_completion.return_value = _mock_completion_response("[]")
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
        mock_completion.return_value = _mock_completion_response("[]")
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
        mock_completion.return_value = _mock_completion_response("[]")
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
        assert "thinking" not in call_kwargs.kwargs
        assert call_kwargs.kwargs["temperature"] == 1.0
        assert call_kwargs.kwargs["extra_body"]["thinking"]["type"] == "enabled"
        assert (
            call_kwargs.kwargs["extra_body"]["max_tokens"]
            > call_kwargs.kwargs["extra_body"]["thinking"]["budget_tokens"]
        )
        assert any("requires temperature=1" in warning for warning in result.warnings)

    @patch("agent.completion")
    def test_openai_model_enables_logprobs_and_computes_confidence(self, mock_completion):
        mock_completion.return_value = _mock_completion_response(
            "[]", token_logprobs=[-0.1, -0.2, -0.3]
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
        mock_completion.return_value = _mock_completion_response("[]")
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
        mock_completion.return_value = _mock_completion_response("[]")
        result = run_llm_with_metadata(
            text="text",
            api_key="k",
            model="openai/gpt-4o",
        )
        assert result.llm_confidence.available is False
        assert result.llm_confidence.reason == "missing_logprobs"
        assert result.llm_confidence.band == "na"

    @patch("agent.completion")
    def test_openai_logprobs_fallback_retries_without_param(self, mock_completion):
        mock_completion.side_effect = [
            Exception("UnsupportedParamsError: openai does not support parameters: ['logprobs']"),
            _mock_completion_response("[]"),
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
        mock_completion.return_value = _mock_completion_response("[]", token_logprobs=[-0.2, -0.1])
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
    def test_response_format_fallback_retries_without_schema(self, mock_completion):
        calls: list[dict] = []

        def _side_effect(**kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise Exception(
                    "UnsupportedParamsError: openai does not support parameters: ['response_format']"
                )
            return _mock_completion_response("[]")

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
        mock_completion.return_value = _mock_completion_response("[]")
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
    def test_parse_failure_uses_one_repair_retry(self, mock_completion):
        mock_completion.side_effect = [
            _mock_completion_response("not valid json"),
            _mock_completion_response(
                '{"spans":[{"start":3,"end":7,"label":"NAME","text":"John"}]}'
            ),
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

        payload = json.dumps(
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


def test_model_presets_include_requested_options():
    model_ids = {preset["model"] for preset in MODEL_PRESETS}
    assert "openai.gpt-5.3-codex" in model_ids
    assert "openai.gpt-5.2-chat" in model_ids
    assert "anthropic.claude-4.6-opus" in model_ids
    assert "google.gemini-3.1-pro-preview" in model_ids
    assert "google.gemini-3-flash-preview" in model_ids


def test_method_prompts_are_migrated_from_experiment_presets():
    extended_prompt = METHOD_DEFINITION_BY_ID["extended"]["passes"][0]["prompt"]
    assert "HIPAA Safe Harbor" in extended_prompt
    assert "Method or theorem names containing a person's name" in extended_prompt

    dual_numbers_prompt = METHOD_DEFINITION_BY_ID["dual-split"]["passes"][1]["prompt"]
    assert "Mathematical expressions" in dual_numbers_prompt
    assert "Output data following the provided JSON schema." in dual_numbers_prompt

    split_prompt = METHOD_DEFINITION_BY_ID["presidio+llm-split"]["passes"][1]["prompt"]
    assert "reasonably identify an individual" in split_prompt
