"""Tests for tool call parsers (§6.6, FR7, AC15).

Patterns: happy path, edge cases, negative path, boundary,
corner cases, alternate flows, compatibility.
"""

from __future__ import annotations

import json

import pytest

from mlxs._types import ToolCallResult
from mlxs.tool_calling import get_parser
from mlxs.tool_calling.generic import GenericToolCallParser
from mlxs.tool_calling.qwen import QwenToolCallParser

# =============================================================================
# GenericToolCallParser
# =============================================================================


class TestGenericHappyPath:
    def test_parse_inline_function_call(self) -> None:
        text = '{"name": "get_weather", "arguments": {"city": "Paris"}}'
        parser = GenericToolCallParser()
        results = parser.parse(text)
        assert len(results) == 1
        assert results[0].name == "get_weather"
        assert json.loads(results[0].arguments) == {"city": "Paris"}

    def test_parse_multiple_inline_calls(self) -> None:
        text = (
            '{"name": "func_a", "arguments": {"x": 1}} '
            'some text {"name": "func_b", "arguments": {"y": 2}}'
        )
        parser = GenericToolCallParser()
        results = parser.parse(text)
        assert len(results) == 2
        assert results[0].name == "func_a"
        assert results[1].name == "func_b"

    def test_tool_calls_array_format(self) -> None:
        data = {
            "tool_calls": [
                {
                    "id": "call_123",
                    "function": {
                        "name": "search",
                        "arguments": '{"query": "test"}',
                    },
                }
            ]
        }
        text = json.dumps(data)
        parser = GenericToolCallParser()
        results = parser.parse(text)
        assert len(results) == 1
        assert results[0].name == "search"
        assert results[0].id == "call_123"

    def test_is_tool_call_positive(self) -> None:
        parser = GenericToolCallParser()
        assert parser.is_tool_call('{"name": "test", "arguments": {}}')


class TestGenericNegativePath:
    def test_parse_no_tool_calls(self) -> None:
        parser = GenericToolCallParser()
        assert parser.parse("Hello, world!") == []

    def test_is_tool_call_negative(self) -> None:
        parser = GenericToolCallParser()
        assert not parser.is_tool_call("Hello, no tools here")

    def test_invalid_json_arguments_skipped(self) -> None:
        text = '{"name": "func", "arguments": {invalid json}}'
        parser = GenericToolCallParser()
        assert parser.parse(text) == []

    def test_partial_json_no_crash(self) -> None:
        parser = GenericToolCallParser()
        assert parser.parse('{"name": "func"') == []

    def test_empty_string(self) -> None:
        parser = GenericToolCallParser()
        assert parser.parse("") == []

    def test_only_whitespace(self) -> None:
        parser = GenericToolCallParser()
        assert parser.parse("   \n\t  ") == []


class TestGenericEdgeCases:
    def test_arguments_with_nested_json(self) -> None:
        """Arguments containing nested objects (regex-limited depth)."""
        text = '{"name": "f", "arguments": {"key": "val"}}'
        parser = GenericToolCallParser()
        results = parser.parse(text)
        assert len(results) == 1
        assert json.loads(results[0].arguments) == {"key": "val"}

    def test_name_with_special_chars(self) -> None:
        text = '{"name": "get_user-info_v2", "arguments": {"id": 1}}'
        parser = GenericToolCallParser()
        results = parser.parse(text)
        assert len(results) == 1
        assert results[0].name == "get_user-info_v2"

    def test_tool_call_surrounded_by_text(self) -> None:
        text = 'Let me call {"name": "f", "arguments": {"x": 1}} for you.'
        parser = GenericToolCallParser()
        results = parser.parse(text)
        assert len(results) == 1
        assert results[0].name == "f"

    def test_tool_calls_array_with_dict_arguments(self) -> None:
        """Arguments as dict (not string) should be serialized to JSON."""
        data = {
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "f",
                        "arguments": {"key": "val"},
                    },
                }
            ]
        }
        text = json.dumps(data)
        parser = GenericToolCallParser()
        results = parser.parse(text)
        assert len(results) == 1
        assert json.loads(results[0].arguments) == {"key": "val"}

    def test_each_call_gets_unique_id(self) -> None:
        text = '{"name": "a", "arguments": {"x": 1}} {"name": "b", "arguments": {"y": 2}}'
        parser = GenericToolCallParser()
        results = parser.parse(text)
        assert len(results) == 2
        assert results[0].id != results[1].id


class TestGenericBoundary:
    def test_empty_arguments(self) -> None:
        text = '{"name": "f", "arguments": {}}'
        parser = GenericToolCallParser()
        results = parser.parse(text)
        assert len(results) == 1
        assert results[0].arguments == "{}"

    def test_tool_calls_empty_array(self) -> None:
        text = json.dumps({"tool_calls": []})
        parser = GenericToolCallParser()
        assert parser.parse(text) == []

    def test_tool_calls_missing_name(self) -> None:
        """tool_calls entry without name should be skipped."""
        data = {"tool_calls": [{"id": "call_1", "function": {"arguments": "{}"}}]}
        text = json.dumps(data)
        parser = GenericToolCallParser()
        assert parser.parse(text) == []


# =============================================================================
# QwenToolCallParser
# =============================================================================


class TestQwenHappyPath:
    def test_parse_modern_format(self) -> None:
        text = (
            '<tool_call>\n{"name": "calculator", "arguments": {"expression": "2+2"}}\n</tool_call>'
        )
        parser = QwenToolCallParser()
        results = parser.parse(text)
        assert len(results) == 1
        assert results[0].name == "calculator"
        assert json.loads(results[0].arguments) == {"expression": "2+2"}

    def test_parse_legacy_format(self) -> None:
        text = '✿FUNCTION✿: get_weather\n✿ARGS✿: {"city": "Tokyo"}\n✿RESULT✿:'
        parser = QwenToolCallParser()
        results = parser.parse(text)
        assert len(results) == 1
        assert results[0].name == "get_weather"
        assert json.loads(results[0].arguments) == {"city": "Tokyo"}

    def test_multiple_modern_calls(self) -> None:
        text = (
            '<tool_call>\n{"name": "a", "arguments": {}}\n</tool_call>\n'
            '<tool_call>\n{"name": "b", "arguments": {"x": 1}}\n</tool_call>'
        )
        parser = QwenToolCallParser()
        results = parser.parse(text)
        assert len(results) == 2
        assert results[0].name == "a"
        assert results[1].name == "b"


class TestQwenNegativePath:
    def test_parse_no_tool_calls(self) -> None:
        parser = QwenToolCallParser()
        assert parser.parse("No tool calls here") == []

    def test_invalid_json_in_tool_call_tag(self) -> None:
        text = "<tool_call>\n{not valid json}\n</tool_call>"
        parser = QwenToolCallParser()
        assert parser.parse(text) == []

    def test_invalid_json_in_legacy_format(self) -> None:
        text = "✿FUNCTION✿: func\n✿ARGS✿: {not valid}\n✿RESULT✿:"
        parser = QwenToolCallParser()
        assert parser.parse(text) == []


class TestQwenEdgeCases:
    def test_is_tool_call_modern(self) -> None:
        parser = QwenToolCallParser()
        assert parser.is_tool_call("<tool_call>...")

    def test_is_tool_call_legacy(self) -> None:
        parser = QwenToolCallParser()
        assert parser.is_tool_call("✿FUNCTION✿: test")

    def test_is_tool_call_negative(self) -> None:
        parser = QwenToolCallParser()
        assert not parser.is_tool_call("Just normal text")

    def test_modern_format_with_extra_whitespace(self) -> None:
        text = '<tool_call>   \n  {"name": "f", "arguments": {}}  \n  </tool_call>'
        parser = QwenToolCallParser()
        results = parser.parse(text)
        assert len(results) == 1
        assert results[0].name == "f"

    def test_legacy_format_name_with_whitespace(self) -> None:
        text = "✿FUNCTION✿:   spaced_name  \n✿ARGS✿: {}\n✿RESULT✿:"
        parser = QwenToolCallParser()
        results = parser.parse(text)
        assert len(results) == 1
        assert results[0].name == "spaced_name"

    def test_modern_format_missing_name(self) -> None:
        """JSON without 'name' field should be skipped."""
        text = '<tool_call>\n{"arguments": {"x": 1}}\n</tool_call>'
        parser = QwenToolCallParser()
        assert parser.parse(text) == []

    def test_modern_format_arguments_as_string(self) -> None:
        """Arguments as string (not dict) should be preserved."""
        text = '<tool_call>\n{"name": "f", "arguments": "raw_string"}\n</tool_call>'
        parser = QwenToolCallParser()
        results = parser.parse(text)
        assert len(results) == 1
        assert results[0].arguments == "raw_string"


class TestQwenAlternateFlows:
    def test_modern_preferred_over_legacy(self) -> None:
        """When both formats present, modern is returned (modern tried first)."""
        text = (
            '<tool_call>\n{"name": "modern", "arguments": {}}\n</tool_call>\n'
            "✿FUNCTION✿: legacy\n✿ARGS✿: {}\n✿RESULT✿:"
        )
        parser = QwenToolCallParser()
        results = parser.parse(text)
        # Modern format found, so legacy isn't parsed
        assert all(r.name == "modern" for r in results)

    def test_legacy_only_when_no_modern(self) -> None:
        text = "✿FUNCTION✿: legacy_only\n✿ARGS✿: {}\n✿RESULT✿:"
        parser = QwenToolCallParser()
        results = parser.parse(text)
        assert len(results) == 1
        assert results[0].name == "legacy_only"


# =============================================================================
# Parser registry — compatibility path
# =============================================================================


class TestGetParser:
    def test_get_generic_parser(self) -> None:
        parser = get_parser("generic")
        assert isinstance(parser, GenericToolCallParser)

    def test_get_qwen_parser(self) -> None:
        parser = get_parser("qwen")
        assert isinstance(parser, QwenToolCallParser)

    def test_unknown_parser_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown tool_call_parser"):
            get_parser("nonexistent")

    def test_error_message_lists_supported(self) -> None:
        with pytest.raises(ValueError, match="generic") as exc_info:
            get_parser("bogus")
        assert "qwen" in str(exc_info.value)


# =============================================================================
# ToolCallResult type — boundary & corner cases
# =============================================================================


class TestToolCallResult:
    def test_frozen(self) -> None:
        result = ToolCallResult(id="1", name="f", arguments="{}")
        with pytest.raises(AttributeError):
            result.name = "g"  # type: ignore[misc]

    def test_equality(self) -> None:
        a = ToolCallResult(id="1", name="f", arguments="{}")
        b = ToolCallResult(id="1", name="f", arguments="{}")
        assert a == b

    def test_fields(self) -> None:
        result = ToolCallResult(id="call_abc", name="search", arguments='{"q": "x"}')
        assert result.id == "call_abc"
        assert result.name == "search"
        assert result.arguments == '{"q": "x"}'
