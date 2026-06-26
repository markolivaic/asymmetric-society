"""Message classifier (ZAV-25) tested with a scripted backend - no API calls.

The classifier drives the real :meth:`BaseAgent.act` pipeline (prefill/parse/
retry/budget/log); only ``_complete`` is mocked, so the validator, fallback,
SQLite logging, idempotent upsert, and gate-isolation are all exercised offline.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from asymmetric_society.agents.base import BaseAgent, CompletionResult
from asymmetric_society.metrics.classifier import (
    LABELS,
    agreement,
    batch_classify,
    classify_message,
    label_validator,
)
from asymmetric_society.runner import db
from asymmetric_society.runner.budget import BudgetTracker
from asymmetric_society.runner.results import parse_failure_stats

EXP = "exp-classify-01"


class ClassifierMockAgent(BaseAgent):
    """Backend that returns scripted classification JSON instead of calling Haiku."""

    def __init__(
        self,
        *,
        label: str | None = None,
        responses: list[str] | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__("classifier", "mock-haiku", "classifier", **kwargs)  # type: ignore[arg-type]
        self._label = label
        self._responses = list(responses) if responses is not None else None
        self.calls = 0

    def _complete(self, prompt: str, *, json_mode: bool) -> CompletionResult:
        self.calls += 1
        if self._responses is not None:
            raw = self._responses.pop(0)
        else:
            raw = json.dumps({"label": self._label})
        return CompletionResult(raw_text=raw, input_tokens=10, output_tokens=3, cost_eur=0.0001)


def _query(db_path: str, sql: str) -> list:
    con = sqlite3.connect(db_path)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# label_validator (pure)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("label", list(LABELS))
def test_label_validator_accepts_each_class(label: str) -> None:
    assert label_validator({"label": label}) == {"label": label}


def test_label_validator_strips_whitespace() -> None:
    assert label_validator({"label": "  threat "}) == {"label": "threat"}


@pytest.mark.parametrize(
    "bad",
    [
        {"label": "cooperative"},  # not an exact class
        {"label": ""},
        {"label": 3},
        {"note": "no label key"},
        {},
    ],
)
def test_label_validator_rejects_bad(bad: dict) -> None:
    with pytest.raises(ValueError):
        label_validator(bad)


def test_label_validator_rejects_non_dict() -> None:
    with pytest.raises(ValueError):
        label_validator("threat")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# classify_message
# --------------------------------------------------------------------------- #


def test_classify_message_returns_scripted_label(db_path: str, budget: BudgetTracker) -> None:
    agent = ClassifierMockAgent(label="threat", budget=budget, db_path=db_path)
    result = classify_message("Cross me and you get nothing.", "Trust Game", agent=agent)
    assert result.label == "threat"
    assert not result.fallback_used
    assert result.cost_eur > 0
    # Exactly one llm_calls row, logged out-of-experiment (NULL) and own tier.
    rows = _query(db_path, "SELECT experiment_id, capability_tier FROM llm_calls")
    assert rows == [(None, "classifier")]


def test_classify_message_falls_back_to_neutral_on_garbage(
    db_path: str, budget: BudgetTracker
) -> None:
    agent = ClassifierMockAgent(
        responses=["not json", "still not", "nope", "bad"], budget=budget, db_path=db_path
    )
    result = classify_message("???", "Public Goods Game", agent=agent)
    assert result.label == "neutral"
    assert result.fallback_used
    assert agent.calls == 4  # max_parse_retries (3) + 1


# --------------------------------------------------------------------------- #
# batch_classify
# --------------------------------------------------------------------------- #


def _seed_actions(db_path: str) -> None:
    rows = [
        (1, "aaaa0001", "I promise to contribute fully.", False),
        (1, "bbbb0002", None, False),  # no message -> skipped
        (2, "aaaa0001", "   ", False),  # whitespace-only -> skipped
        (2, "bbbb0002", "If you defect I punish you.", False),
        (3, "aaaa0001", "The multiplier makes cooperation optimal.", False),
    ]
    for round_num, agent_id, message, fb in rows:
        db.insert_action(
            db_path,
            db.ActionRecord(
                experiment_id=EXP,
                round_num=round_num,
                agent_id=agent_id,
                action_json={"contribution": 10},
                message=message,
                retries=0,
                cost_eur=0.0,
                fallback_used=fb,
            ),
        )


def test_batch_classify_populates_table_and_skips_empty(
    db_path: str, budget: BudgetTracker
) -> None:
    _seed_actions(db_path)
    agent = ClassifierMockAgent(label="informational", budget=budget, db_path=db_path)
    n = batch_classify(db_path, [EXP], agent=agent, game_context="Public Goods Game")
    assert n == 3  # 5 rows, 2 without usable messages
    labels = _query(db_path, "SELECT label FROM message_classifications")
    assert [r[0] for r in labels] == ["informational"] * 3


def test_batch_classify_is_idempotent(db_path: str, budget: BudgetTracker) -> None:
    _seed_actions(db_path)
    agent = ClassifierMockAgent(label="neutral", budget=budget, db_path=db_path)
    batch_classify(db_path, [EXP], agent=agent, game_context="g")
    batch_classify(db_path, [EXP], agent=agent, game_context="g")  # re-run
    count = _query(db_path, "SELECT COUNT(*) FROM message_classifications")[0][0]
    assert count == 3  # upsert on (experiment_id, round_num, agent_id), no duplicates


def test_classifier_calls_excluded_from_parse_gate(
    db_path: str, budget: BudgetTracker
) -> None:
    _seed_actions(db_path)
    agent = ClassifierMockAgent(label="neutral", budget=budget, db_path=db_path)
    batch_classify(db_path, [EXP], agent=agent, game_context="g")
    # The classifier logged calls with NULL experiment_id, so a gate query scoped
    # to the experiment sees none of them (no weak/strong contamination).
    assert parse_failure_stats(db_path, [EXP]) == []
    tiers = _query(db_path, "SELECT DISTINCT capability_tier FROM llm_calls")
    assert tiers == [("classifier",)]


# --------------------------------------------------------------------------- #
# agreement (pure)
# --------------------------------------------------------------------------- #


def test_agreement_perfect() -> None:
    gold = ["threat", "neutral", "informational"]
    rep = agreement(gold, list(gold))
    assert rep["accuracy"] == pytest.approx(1.0)
    assert rep["correct"] == 3


def test_agreement_counts_confusion_and_per_label() -> None:
    gold = ["threat", "threat", "neutral", "neutral"]
    pred = ["threat", "neutral", "neutral", "informational"]
    rep = agreement(gold, pred)
    assert rep["accuracy"] == pytest.approx(0.5)  # 2 of 4 correct
    assert rep["confusion"][("threat", "neutral")] == 1
    assert rep["per_label"]["threat"]["tp"] == 1
    assert rep["per_label"]["threat"]["fn"] == 1
    assert rep["per_label"]["neutral"]["fp"] == 1  # threat->neutral mislabel
    assert rep["per_label"]["informational"]["fp"] == 1


def test_agreement_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        agreement(["threat"], ["threat", "neutral"])
