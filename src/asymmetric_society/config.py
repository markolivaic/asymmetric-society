"""Experiment configuration schema (pydantic).

Every experiment is fully specified by a JSON config plus an integer seed, so
that ``same config + seed`` reproduces the same game setup, agent ordering, and
randomization. Note: this does NOT guarantee bit-identical LLM responses - see
``docs/architecture.md`` on reproducibility.

Pattern borrowed from FAIRGAME (arXiv:2504.14325): config-driven experiments
via JSON.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Backend = Literal["anthropic", "openai", "gemini", "ollama"]
CapabilityTier = Literal["strong", "weak"]


def new_agent_id() -> str:
    """Return a fresh anonymous 8-character agent ID.

    We never use names like ``Agent 1`` because position-encoded identities can
    leak ordering information into prompts and bias behavior.

    Returns:
        An 8-character hex string.
    """
    return uuid.uuid4().hex[:8]


class AgentConfig(BaseModel):
    """Configuration for a single agent in an experiment."""

    backend: Backend
    model_name: str
    capability_tier: CapabilityTier
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    agent_id: str = Field(default_factory=new_agent_id)

    @field_validator("agent_id")
    @classmethod
    def _non_empty_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("agent_id must be non-empty")
        return value


class GameConfig(BaseModel):
    """Configuration for the game being played.

    ``params`` holds game-specific knobs (endowment, rounds, multiplier, ...)
    validated by the concrete game in a later ticket.
    """

    name: str
    params: dict[str, object] = Field(default_factory=dict)


class ExperimentConfig(BaseModel):
    """Top-level, fully-reproducible specification of one experiment run."""

    name: str
    seed: int
    game: GameConfig
    agents: list[AgentConfig]
    rounds: int = Field(gt=0)
    db_path: str = "results/experiments.db"

    @field_validator("agents")
    @classmethod
    def _unique_agent_ids(cls, agents: list[AgentConfig]) -> list[AgentConfig]:
        ids = [a.agent_id for a in agents]
        if len(ids) != len(set(ids)):
            raise ValueError("agent_id values must be unique within an experiment")
        if not agents:
            raise ValueError("an experiment needs at least one agent")
        return agents

    @model_validator(mode="after")
    def _rounds_consistent(self) -> ExperimentConfig:
        """Reject a config where ``rounds`` disagrees with ``game.params``.

        The runner drives the loop from the game's own round count, so a silent
        mismatch (e.g. top-level ``rounds=10`` but ``game.params['rounds']=5``)
        would let the config lie about how long the run was. If both are given
        they must be equal.
        """
        game_rounds = self.game.params.get("rounds")
        if game_rounds is not None and game_rounds != self.rounds:
            raise ValueError(
                f"rounds mismatch: ExperimentConfig.rounds={self.rounds} but "
                f"game.params['rounds']={game_rounds}; they must be equal"
            )
        return self


def load_experiment(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment config from a JSON file.

    Args:
        path: Path to the JSON config file.

    Returns:
        The parsed and validated :class:`ExperimentConfig`.

    Raises:
        pydantic.ValidationError: If the file does not match the schema.
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ExperimentConfig.model_validate(data)
