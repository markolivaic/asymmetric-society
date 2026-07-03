"""Shared scoring primitive (metrics/_llm_scoring) + classifier back-compat.

Verifies the seam the message classifier and the deception metric both ride: a scripted backend goes through
the real ``act`` pipeline, calls log out of the parse gate, and the classifier's
old names still resolve to the moved primitive.
"""

from __future__ import annotations

import json
import sqlite3

from asymmetric_society.agents.base import BaseAgent, CompletionResult
from asymmetric_society.metrics import classifier
from asymmetric_society.metrics._llm_scoring import (
    SCORING_MODEL,
    build_scoring_agent,
    llm_score,
)
from asymmetric_society.runner.budget import BudgetTracker


class ScoringMockAgent(BaseAgent):
    """Backend that replays one scripted raw response."""

    def __init__(self, raw: str, **kwargs: object) -> None:
        super().__init__("scorer-mock", "mock", "classifier", **kwargs)  # type: ignore[arg-type]
        self._raw = raw
        self.calls = 0

    def _complete(self, prompt: str, *, json_mode: bool) -> CompletionResult:
        self.calls += 1
        return CompletionResult(raw_text=self._raw, input_tokens=5, output_tokens=2, cost_eur=0.0002)


def _query(db_path: str, sql: str) -> list:
    con = sqlite3.connect(db_path)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def test_llm_score_validates_and_logs_out_of_gate(db_path: str, budget: BudgetTracker) -> None:
    agent = ScoringMockAgent(json.dumps({"x": 1}), budget=budget, db_path=db_path)

    def validator(payload: dict) -> dict:
        if "x" not in payload:
            raise ValueError("no x")
        return {"x": payload["x"]}

    result = llm_score(agent, "prompt", validator=validator, fallback={"x": 0})
    assert result.action == {"x": 1}
    # Logged with NULL experiment_id and the scorer tier -> out of weak/strong gate.
    assert _query(db_path, "SELECT experiment_id, capability_tier FROM llm_calls") == [
        (None, "classifier")
    ]


def test_build_scoring_agent_fields(db_path: str, budget: BudgetTracker) -> None:
    agent = build_scoring_agent(db_path, budget=budget, agent_id="foo", tier="bar", max_tokens=32)
    assert agent.agent_id == "foo"
    assert agent.capability_tier == "bar"
    assert agent.model_name == SCORING_MODEL
    assert agent.max_tokens == 32


def test_classifier_back_compat_aliases(db_path: str, budget: BudgetTracker) -> None:
    # The classifier's old name now points at the moved primitive.
    assert classifier.llm_classify is llm_score
    agent = classifier.build_classifier_agent(db_path, budget=budget)
    assert agent.agent_id == "classifier"
    assert agent.capability_tier == "classifier"
    assert agent.model_name == SCORING_MODEL
