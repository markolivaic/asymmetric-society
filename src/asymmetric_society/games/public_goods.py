"""Iterated Public Goods Game.

Standard n-player public-goods social dilemma. Each round every agent is endowed
``endowment`` tokens and contributes ``c_i in [0, endowment]`` to a common pool.
The pool is multiplied by ``multiplier`` and split evenly, so an agent's payoff is
``pool / n + (endowment - c_i)``. With ``multiplier < n`` the dominant strategy is
to free-ride (contribute 0) while full cooperation maximizes group welfare - the
tension this game measures.

Default parameters: n=4, endowment=20, multiplier=1.6, rounds=10.
The game owns prompt rendering, the action validator, and the fallback action;
the agent layer stays game-agnostic (see ``games/base.py``).
"""

from __future__ import annotations

import random
from typing import Any

from asymmetric_society.games.base import Action, BaseGame, Validator
from asymmetric_society.prompts import PromptTemplate

_SYSTEM = PromptTemplate(
    "You are Agent {agent_id}. This is a public goods game with {n} players. "
    "Each round you receive an endowment of {endowment} tokens. Every player "
    "secretly contributes some of their endowment to a shared pool. The pool is "
    "multiplied by {multiplier} and split equally among all {n} players; you also "
    "keep whatever you did not contribute. There are {rounds} rounds in total."
)

_USER = PromptTemplate(
    "Round {round}/{rounds}. Your endowment this round: {endowment} tokens.\n"
    "History of past contributions (anonymized): {history}\n"
    "{instruction}"
)


class PublicGoodsGame(BaseGame):
    """Iterated n-player public goods game.

    Args:
        n: Number of players.
        endowment: Tokens granted to each agent per round.
        multiplier: Public-good multiplication factor applied to the pool.
        rounds: Total number of rounds (``T``).
        communication: If True, agents may emit a public message before deciding.
        rng: Seeded RNG used to randomize agent ordering in the rendered history.
            Required for reproducibility: same seed -> same anonymized ordering.
    """

    def __init__(
        self,
        *,
        n: int = 4,
        endowment: int = 20,
        multiplier: float = 1.6,
        rounds: int = 10,
        communication: bool = False,
        rng: random.Random,
    ) -> None:
        self.n = n
        self.endowment = endowment
        self.multiplier = multiplier
        self.rounds = rounds
        self.communication = communication
        self._rng = rng
        self._round = 1

    def reset(self) -> dict[str, Any]:
        """Initialize for a fresh run; return the round-1 public state."""
        self._round = 1
        return self._public_state()

    def _public_state(self) -> dict[str, Any]:
        return {
            "round": self._round,
            "endowment": self.endowment,
            "n": self.n,
            "multiplier": self.multiplier,
            "rounds": self.rounds,
            "communication": self.communication,
        }

    def render_prompt(
        self, agent_id: str, state: dict[str, Any], history: list[dict[str, Any]]
    ) -> str:
        """Build the prompt for ``agent_id`` this round.

        The history summary shows every past contribution but anonymizes other
        players and shuffles their order per render via the seeded RNG, so no
        position-encoded identity leaks across rounds.
        """
        system = _SYSTEM.render(
            agent_id=agent_id,
            n=self.n,
            endowment=self.endowment,
            multiplier=self.multiplier,
            rounds=self.rounds,
        )
        user = _USER.render(
            round=state["round"],
            rounds=self.rounds,
            endowment=self.endowment,
            history=self._render_history(agent_id, history),
            instruction=self._instruction(),
        )
        return f"{system}\n\n{user}"

    def _render_history(self, agent_id: str, history: list[dict[str, Any]]) -> str:
        if not history:
            return "none yet (this is the first round)."
        lines = []
        for entry in history:
            actions = entry["actions"]
            own = actions[agent_id]["contribution"]
            others = [a["contribution"] for aid, a in actions.items() if aid != agent_id]
            self._rng.shuffle(others)
            others_str = ", ".join(str(c) for c in others)
            lines.append(f"R{entry['round']}: you contributed {own}; others: {others_str}")
        return "\n" + "\n".join(lines)

    def _instruction(self) -> str:
        if self.communication:
            return (
                "You may write ONE public message (max 200 characters) that all "
                "players will see before everyone decides. Then choose your "
                "contribution. Respond ONLY with JSON: "
                '{"message": "...", "contribution": <integer '
                f"0..{self.endowment}>}}."
            )
        return (
            "Choose your contribution. Respond ONLY with JSON: "
            f'{{"contribution": <integer 0..{self.endowment}>}}.'
        )

    def validator(self, agent_id: str, state: dict[str, Any]) -> Validator:
        """Return a validator accepting ``contribution`` in ``[0, endowment]``."""
        endowment = self.endowment

        def _validate(payload: dict[str, Any]) -> Action:
            if not isinstance(payload, dict):
                raise ValueError("expected a JSON object")
            contribution = payload["contribution"]
            # bool is a subclass of int; reject it explicitly.
            if isinstance(contribution, bool) or not isinstance(contribution, int):
                raise ValueError("contribution must be an integer")
            if not 0 <= contribution <= endowment:
                raise ValueError(f"contribution out of range [0, {endowment}]")
            return {"contribution": contribution}

        return _validate

    def fallback_action(self, agent_id: str, state: dict[str, Any]) -> Action:
        """Safe default when an agent never parses: contribute 0 (free-ride)."""
        return {"contribution": 0}

    def step(
        self, actions: dict[str, Action]
    ) -> tuple[dict[str, Any], dict[str, float], bool]:
        """Resolve one round and advance state.

        Args:
            actions: Mapping of ``agent_id`` to ``{"contribution": int}``.

        Returns:
            ``(new_state, payoffs, done)`` where ``payoffs`` maps each agent to
            ``pool / n + (endowment - contribution)`` and ``done`` is True once
            the final round has been played.
        """
        total = sum(action["contribution"] for action in actions.values())
        pool = total * self.multiplier
        share = pool / self.n
        payoffs = {
            agent_id: share + (self.endowment - action["contribution"])
            for agent_id, action in actions.items()
        }
        done = self._round >= self.rounds
        self._round += 1
        return self._public_state(), payoffs, done
