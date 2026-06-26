"""Game abstraction layer (:class:`BaseGame`).

A game owns the rules: it renders the per-agent prompt, supplies the validator
and fallback action that the agent's retry loop needs, and advances state given
the agents' actions. The agent layer stays game-agnostic. Concrete games (Public
Goods, Trust) arrive in later tickets - this module defines the contract only.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

# A game-defined action is a small JSON-compatible dict, e.g. {"contribution": 5}.
Action = dict[str, Any]
# A validator turns raw parsed JSON into a clean Action or raises ValueError.
Validator = Callable[[dict[str, Any]], Action]


class BaseGame(ABC):
    """Abstract base class for all economic games.

    Subclasses implement the rules; the runner drives ``reset`` then repeated
    ``render_prompt`` -> agent.act -> ``step`` cycles until ``done``.
    """

    @abstractmethod
    def reset(self) -> dict[str, Any]:
        """Initialize game state for a fresh run.

        Returns:
            The initial public state dict.
        """

    @abstractmethod
    def render_prompt(
        self, agent_id: str, state: dict[str, Any], history: list[dict[str, Any]]
    ) -> str:
        """Build the prompt shown to one agent this round.

        Args:
            agent_id: The agent the prompt is for.
            state: Current public game state.
            history: All past rounds (anonymized).

        Returns:
            The fully rendered prompt text.
        """

    @abstractmethod
    def validator(self, agent_id: str, state: dict[str, Any]) -> Validator:
        """Return a validator for one agent's action this round.

        The returned callable takes the JSON the agent produced and returns a
        clean :data:`Action`, or raises ``ValueError`` if it is out of range or
        malformed. The agent's retry loop uses this to decide parse success.
        """

    @abstractmethod
    def fallback_action(self, agent_id: str, state: dict[str, Any]) -> Action:
        """Return the safe default action when an agent never parses.

        Per project convention these are the maximally cautious choices
        (e.g. contribute 0, send 0, return 0).
        """

    @abstractmethod
    def step(
        self, actions: dict[str, Action]
    ) -> tuple[dict[str, Any], dict[str, float], bool]:
        """Advance the game by one round.

        Args:
            actions: Mapping of ``agent_id`` to that agent's action.

        Returns:
            A tuple ``(new_state, payoffs, done)`` where ``payoffs`` maps
            ``agent_id`` to the payoff earned this round and ``done`` signals
            the game has ended.
        """
