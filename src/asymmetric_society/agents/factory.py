"""Agent factory: build the right backend from an :class:`AgentConfig`."""

from __future__ import annotations

from asymmetric_society.agents.anthropic_agent import AnthropicAgent
from asymmetric_society.agents.base import BaseAgent
from asymmetric_society.agents.gemini_agent import GeminiAgent
from asymmetric_society.agents.ollama_agent import OllamaAgent
from asymmetric_society.agents.openai_agent import OpenAIAgent
from asymmetric_society.config import AgentConfig
from asymmetric_society.runner.budget import BudgetTracker

_REGISTRY: dict[str, type[BaseAgent]] = {
    "anthropic": AnthropicAgent,
    "openai": OpenAIAgent,
    "gemini": GeminiAgent,
    "ollama": OllamaAgent,
}


def create_agent(
    config: AgentConfig,
    *,
    budget: BudgetTracker,
    db_path: str,
) -> BaseAgent:
    """Instantiate the backend named by ``config.backend``.

    Args:
        config: Validated per-agent configuration.
        budget: Shared budget tracker passed to the agent.
        db_path: SQLite database for call logging.

    Returns:
        A ready-to-use :class:`BaseAgent` subclass instance.

    Raises:
        ValueError: If ``config.backend`` is not a known backend.
    """
    try:
        cls = _REGISTRY[config.backend]
    except KeyError as exc:
        raise ValueError(f"Unknown backend: {config.backend!r}") from exc
    return cls(
        config.agent_id,
        config.model_name,
        config.capability_tier,
        budget=budget,
        db_path=db_path,
        temperature=config.temperature,
    )
