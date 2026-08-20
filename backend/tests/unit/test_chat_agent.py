"""How the agent is assembled: which tools exist, and under what conditions."""

import pytest
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.toolsets import CombinedToolset
from pydantic_ai.usage import RunUsage

from app.core.config import LLMSettings, TavilySettings
from app.features.chat.agent import build_agent

LLM = LLMSettings(base_url="https://example.test/v1", model="vllm/some-model")
TAVILY = TavilySettings(api_key="tvly-test")

VAULT_TOOLS = {
    "read_card_section",
    "grep_card",
    "list_models",
    "read_model",
    "read_benchmark",
}


async def _tool_names(agent) -> dict:
    """The tool definitions the model would be offered.

    `Agent` has no public `get_toolset()`/`get_tools(None)` in installed
    pydantic-ai 2.32.0 -- that pair from the brief does not exist on this
    version. The public surface is `Agent.toolsets` (a `Sequence[AbstractToolset]`)
    combined with the public `CombinedToolset`, whose `get_tools` takes a real
    `RunContext` rather than `None`. `deps=None` is safe here: resolving tool
    *definitions* (schemas) never touches `ctx.deps` -- only calling a tool
    body would.
    """
    ctx = RunContext(deps=None, model=TestModel(), usage=RunUsage())
    toolset = CombinedToolset(agent.toolsets)
    return await toolset.get_tools(ctx)


@pytest.mark.anyio
async def test_the_vault_tools_are_registered():
    agent = build_agent(LLM, None)

    assert VAULT_TOOLS <= (await _tool_names(agent)).keys()


@pytest.mark.anyio
async def test_tavily_is_absent_when_no_key_is_configured():
    """An unavailable tool the model can see is one it will try and apologise for."""
    names = (await _tool_names(build_agent(LLM, None))).keys()

    assert not any("tavily" in name or "search" in name for name in names)


@pytest.mark.anyio
async def test_tavily_is_present_when_a_key_is_configured():
    names = (await _tool_names(build_agent(LLM, TAVILY))).keys()

    assert any("tavily" in name or "search" in name for name in names)


@pytest.mark.anyio
async def test_no_tool_declares_a_parameter_called_name():
    """research.md 6e - this endpoint fills such a parameter with the tool's own name."""
    tools = await _tool_names(build_agent(LLM, TAVILY))

    for name, tool in tools.items():
        properties = (tool.tool_def.parameters_json_schema or {}).get("properties", {})
        assert "name" not in properties, f"{name} declares a `name` parameter"
