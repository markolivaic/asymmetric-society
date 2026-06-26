"""Smoke test: package imports resolve and core scaffolding works."""

from __future__ import annotations

import pytest

from asymmetric_society.config import AgentConfig, new_agent_id
from asymmetric_society.prompts import PromptTemplate


def test_prompt_template_renders() -> None:
    template = PromptTemplate("Round {round} of {total}.")
    assert template.render(round=1, total=10) == "Round 1 of 10."


def test_prompt_template_rejects_missing_value() -> None:
    template = PromptTemplate("Hello {name}")
    with pytest.raises(KeyError):
        template.render()


def test_agent_id_is_anonymous_8_chars() -> None:
    aid = new_agent_id()
    assert len(aid) == 8 and aid.isalnum()


def test_agent_config_validates_tier() -> None:
    cfg = AgentConfig(backend="ollama", model_name="llama3.2:3b", capability_tier="weak")
    assert cfg.capability_tier == "weak"
    with pytest.raises(ValueError):
        AgentConfig(backend="ollama", model_name="x", capability_tier="superhuman")
