"""Agent abstraction layer: a unified interface over four LLM backends."""

from asymmetric_society.agents.base import (
    Action,
    ActionResult,
    AgentAPIError,
    BaseAgent,
    CompletionResult,
)
from asymmetric_society.agents.factory import create_agent

__all__ = [
    "Action",
    "ActionResult",
    "AgentAPIError",
    "BaseAgent",
    "CompletionResult",
    "create_agent",
]
