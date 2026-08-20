"""The chat agent, assembled per request.

Per request rather than at import because the model settings come from a
dependency and the tool list varies with whether Tavily is configured.

Tools are registered as thin wrappers over `tools.py`, whose functions take
`ChatDeps` directly. The docstring on each wrapper is what the model reads, so
it is written for that reader rather than for us.
"""

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from app.core.config import LLMSettings, TavilySettings
from app.features.chat import context, prompts, tools
from app.features.chat.deps import ChatDeps


def build_agent(llm: LLMSettings, tavily: TavilySettings | None) -> Agent[ChatDeps, str]:
    """One agent for one request.

    `tool_choice` is never forced: this endpoint returns a malformed call when
    a specific function is pinned (research.md 6e). pydantic-ai's default is
    "auto", which is what we want.
    """
    model = OpenAIChatModel(
        llm.model,
        provider=OpenAIProvider(base_url=llm.base_url, api_key=llm.api_key),
    )

    extra = []
    if tavily is not None:
        # Only when configured: an unavailable tool the model can see is one it
        # will call, retry, and apologise for.
        from pydantic_ai.common_tools.tavily import tavily_search_tool

        extra.append(tavily_search_tool(tavily.api_key))

    agent = Agent(
        model,
        deps_type=ChatDeps,
        instructions=prompts.SYSTEM,
        tools=extra,
    )

    @agent.instructions
    def seed(ctx: RunContext[ChatDeps]) -> str:
        """R9.3 / R9.4 - rebuilt every turn from the store."""
        return context.seed(ctx.deps)

    @agent.tool
    def read_card_section(ctx: RunContext[ChatDeps], section: str) -> str:
        """Read one section of this model's card, verbatim.

        Args:
            section: The exact heading, as listed in card_sections.
        """
        return tools.read_card_section(ctx.deps, section)

    @agent.tool
    def grep_card(ctx: RunContext[ChatDeps], query: str) -> str:
        """Find every line of this model's card containing a string.

        Args:
            query: Text to look for. Case-insensitive.
        """
        return tools.grep_card(ctx.deps, query)

    @agent.tool
    def list_models(ctx: RunContext[ChatDeps]) -> str:
        """List every model in the vault with its headline numbers."""
        return tools.list_models(ctx.deps)

    @agent.tool
    def read_model(ctx: RunContext[ChatDeps], model_id: str) -> str:
        """Read another model's whole reviewed document.

        Args:
            model_id: A Hugging Face ID, as shown by list_models.
        """
        return tools.read_model(ctx.deps, model_id)

    @agent.tool
    def read_benchmark(ctx: RunContext[ChatDeps], slug: str) -> str:
        """Read what a benchmark measures and how to read its scores.

        Args:
            slug: The benchmark's slug, e.g. swe-bench-verified.
        """
        return tools.read_benchmark(ctx.deps, slug)

    return agent
