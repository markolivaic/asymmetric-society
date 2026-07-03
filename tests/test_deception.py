"""KL deception metric: validator, pure math, and end-to-end scoring.

All offline. A scripted backend returns a distribution JSON so the LLM seam is
exercised without network; the KL/surprisal math is tested directly on synthetic
distributions, including the smoothing guard that keeps KL finite.
"""

from __future__ import annotations

import json
import math
import sqlite3

import pytest

from asymmetric_society.agents.base import BaseAgent, CompletionResult
from asymmetric_society.metrics.deception import (
    bin_to_support,
    deception_scores,
    distribution_validator,
    empirical_distribution,
    extract_stated,
    kl_divergence,
    score_game,
    smooth,
    uniform_fallback,
)
from asymmetric_society.runner import db
from asymmetric_society.runner.budget import BudgetTracker

SUPPORT = (0, 5, 10, 15, 20)
EXP = "exp-deception-01"


class DistMockAgent(BaseAgent):
    """Backend that returns a fixed stated distribution over the support."""

    def __init__(self, dist: dict[int, float], **kwargs: object) -> None:
        super().__init__("scorer", "mock", "classifier", **kwargs)  # type: ignore[arg-type]
        self._dist = dist
        self.calls = 0

    def _complete(self, prompt: str, *, json_mode: bool) -> CompletionResult:
        self.calls += 1
        raw = json.dumps({"distribution": {str(k): v for k, v in self._dist.items()}})
        return CompletionResult(raw_text=raw, input_tokens=8, output_tokens=4, cost_eur=0.0003)


def _query(db_path: str, sql: str) -> list:
    con = sqlite3.connect(db_path)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


# --------------------------------------------------------------------------- #
# distribution_validator
# --------------------------------------------------------------------------- #


def test_distribution_validator_accepts_and_normalizes() -> None:
    v = distribution_validator(SUPPORT)
    out = v({"distribution": {"0": 0.21, "5": 0.21, "10": 0.21, "15": 0.21, "20": 0.21}})
    dist = out["distribution"]
    assert set(dist) == set(SUPPORT)
    assert sum(dist.values()) == pytest.approx(1.0)
    assert dist[0] == pytest.approx(0.2)  # 0.21 / 1.05


@pytest.mark.parametrize(
    "bad",
    [
        {"distribution": {"0": 0.5, "5": 0.5, "10": 0.5, "15": 0.5, "20": 0.5}},  # sum 2.5
        {"distribution": {"0": -0.1, "5": 0.4, "10": 0.3, "15": 0.2, "20": 0.2}},  # negative
        {"distribution": {"0": 0.5, "5": 0.5}},  # missing keys
        {"distribution": "nope"},  # not an object
        {"foo": 1},  # no 'distribution'
    ],
)
def test_distribution_validator_rejects(bad: dict) -> None:
    v = distribution_validator(SUPPORT)
    with pytest.raises(ValueError):
        v(bad)


def test_uniform_fallback() -> None:
    fb = uniform_fallback(SUPPORT)["distribution"]
    assert fb == {v: pytest.approx(0.2) for v in SUPPORT}


# --------------------------------------------------------------------------- #
# pure math
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("value,expected", [(13, 15), (2, 0), (7, 5), (8, 10), (20, 20)])
def test_bin_to_support(value: int, expected: int) -> None:
    assert bin_to_support(value, SUPPORT) == expected


def test_empirical_distribution_counts_and_normalizes() -> None:
    dist = empirical_distribution([0, 0, 10], SUPPORT)
    assert dist.tolist() == pytest.approx([2 / 3, 0.0, 1 / 3, 0.0, 0.0])


def test_empirical_distribution_empty_is_uniform() -> None:
    assert empirical_distribution([], SUPPORT).tolist() == pytest.approx([0.2] * 5)


def test_kl_zero_for_identical() -> None:
    p = [0.2, 0.2, 0.2, 0.2, 0.2]
    assert kl_divergence(p, p) == pytest.approx(0.0)


def test_kl_positive_for_divergent() -> None:
    p = [1.0, 0.0, 0.0, 0.0, 0.0]
    q = [0.2, 0.2, 0.2, 0.2, 0.2]
    assert kl_divergence(p, q) > 0.0


def test_smoothing_prevents_inf() -> None:
    p = [0.5, 0.5, 0.0, 0.0, 0.0]
    q = [0.0, 0.0, 1.0, 0.0, 0.0]  # zero mass where p has mass
    assert math.isinf(kl_divergence(p, q))  # unsmoothed -> infinite by design
    finite = kl_divergence(p, smooth(q))
    assert math.isfinite(finite) and finite > 0.0


# --------------------------------------------------------------------------- #
# deception_scores
# --------------------------------------------------------------------------- #


def test_deception_scores_high_when_words_contradict_acts() -> None:
    revealed = [0, 0, 0, 0, 0]  # always free-rides
    stated = [{0: 0.0, 5: 0.0, 10: 0.0, 15: 0.0, 20: 1.0}] * 5  # "I'll contribute 20"
    rep = deception_scores(revealed, stated, SUPPORT)
    assert rep["n_rounds"] == 5
    assert rep["kl_headline"] > 5.0
    assert rep["surprisal"] > 5.0


def test_deception_scores_low_when_honest() -> None:
    revealed = [10, 10, 10, 10, 10]
    stated = [{0: 0.0, 5: 0.0, 10: 1.0, 15: 0.0, 20: 0.0}] * 5
    rep = deception_scores(revealed, stated, SUPPORT)
    assert rep["kl_headline"] < 0.1
    assert rep["surprisal"] < 0.1


def test_deception_scores_skips_rounds_without_message() -> None:
    revealed = [0, 5, 10]
    stated = [{0: 1.0, 5: 0.0, 10: 0.0, 15: 0.0, 20: 0.0}, None, None]
    rep = deception_scores(revealed, stated, SUPPORT)
    assert rep["n_rounds"] == 1  # only the round with a stated distribution


# --------------------------------------------------------------------------- #
# extract_stated + score_game (through the mock LLM)
# --------------------------------------------------------------------------- #


def test_extract_stated_returns_distribution(db_path: str, budget: BudgetTracker) -> None:
    agent = DistMockAgent({0: 0.0, 5: 0.1, 10: 0.6, 15: 0.2, 20: 0.1}, budget=budget, db_path=db_path)
    out = extract_stated("I'll contribute around 10.", "PGG", SUPPORT, agent=agent)
    assert out.distribution[10] == pytest.approx(0.6)
    assert sum(out.distribution.values()) == pytest.approx(1.0)


def _seed_pgg_actions(db_path: str) -> None:
    rows = [
        ("aaaa0001", 1, 0, "I'll contribute everything!"),
        ("aaaa0001", 2, 0, "Cooperating fully this round."),
        ("bbbb0002", 1, 10, "Going to put in a fair share."),
        ("bbbb0002", 2, 10, None),  # no message -> skipped for stated, kept for revealed
    ]
    for agent_id, round_num, contribution, message in rows:
        db.insert_action(
            db_path,
            db.ActionRecord(
                experiment_id=EXP,
                round_num=round_num,
                agent_id=agent_id,
                action_json={"contribution": contribution},
                message=message,
                retries=0,
                cost_eur=0.0,
                fallback_used=False,
            ),
        )


def test_score_game_persists_idempotently(db_path: str, budget: BudgetTracker) -> None:
    _seed_pgg_actions(db_path)
    agent = DistMockAgent({0: 0.0, 5: 0.0, 10: 0.0, 15: 0.0, 20: 1.0}, budget=budget, db_path=db_path)
    results = score_game(db_path, EXP, game="public_goods", agent=agent, game_context="PGG")
    assert {r.agent_id for r in results} == {"aaaa0001", "bbbb0002"}
    rows = _query(db_path, "SELECT COUNT(*) FROM deception_scores")
    assert rows[0][0] == 2
    # Re-run: upsert on (experiment_id, agent_id, direction), still 2 rows.
    score_game(db_path, EXP, game="public_goods", agent=agent, game_context="PGG")
    assert _query(db_path, "SELECT COUNT(*) FROM deception_scores")[0][0] == 2
    # Agent that always free-rides while "promising 20" should score high KL.
    free = next(r for r in results if r.agent_id == "aaaa0001")
    assert free.kl_headline > 5.0
