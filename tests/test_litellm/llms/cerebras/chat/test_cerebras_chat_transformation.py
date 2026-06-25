"""
Unit tests for CerebrasConfig.map_openai_params.

Cerebras does not support tools and response_format in the same request.
When both are supplied, response_format must be translated into an extra
tool call (the same strategy Fireworks AI uses).
"""

import pytest

from litellm.llms.cerebras.chat import CerebrasConfig


MODEL = "cerebras/llama-4-scout-17b-16e-instruct"

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Return the weather for a location.",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        },
    }
]

_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
    },
}


def _map(non_default_params: dict) -> dict:
    return CerebrasConfig().map_openai_params(
        non_default_params=non_default_params,
        optional_params={},
        model=MODEL,
        drop_params=False,
    )


def test_response_format_alone_passes_through():
    """response_format with no tools must reach the API unchanged."""
    result = _map({"response_format": _RESPONSE_FORMAT})
    assert result.get("response_format") == _RESPONSE_FORMAT
    assert "tools" not in result


def test_tools_alone_pass_through():
    """tools with no response_format must reach the API unchanged."""
    result = _map({"tools": _TOOLS})
    assert result.get("tools") == _TOOLS
    assert "response_format" not in result


def test_tools_and_response_format_converts_format_to_tool():
    """When tools and response_format are both set, response_format must be
    removed and its json_schema appended as an extra tool call so Cerebras
    does not reject the request."""
    result = _map({"tools": _TOOLS, "response_format": _RESPONSE_FORMAT})

    # response_format must not appear in the outgoing request
    assert "response_format" not in result

    # original user tool is preserved
    tools = result.get("tools", [])
    assert any(
        t.get("function", {}).get("name") == "get_weather" for t in tools
    ), "original tool must be preserved"

    # the response_format schema must be appended as an extra tool
    assert any(
        t.get("function", {}).get("parameters")
        == _RESPONSE_FORMAT["json_schema"]["schema"]
        for t in tools
    ), "response_format schema must become a tool parameter"


def test_tools_and_json_object_response_format():
    """json_object response_format (no schema) alongside tools should not
    crash and should not emit a response_format key."""
    result = _map({"tools": _TOOLS, "response_format": {"type": "json_object"}})
    assert "response_format" not in result
    # tools must still be present
    assert result.get("tools") is not None
