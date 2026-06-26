"""C3 validation harness: per-round export decomposes the headline; correlate math.

No real LLM: a counter-based mock scoring agent makes both ``score_game`` and
``export_round_records`` deterministic, so the per-round KL exported for C3 must
average back to the agent/game ``kl_headline`` (the decomposition guarantee).
"""

from __future__ import annotations

import math

import pytest

from asymmetric_society.agents.base import ActionResult
from asymmetric_society.metrics import deception
from asymmetric_society.metrics.deception_validation import correlate, export_round_records
from asymmetric_society.runner import db

_SUPPORT = (0, 5, 10, 15, 20)


class _MockScorer:
    """Returns a rotating concentrated distribution; deterministic across runs."""

    def __init__(self) -> None:
        self.i = 0

    def act(self, prompt, validator, fallback, *, json_mode=True, experiment_id=None, round_no=None):
        peak = _SUPPORT[self.i % len(_SUPPORT)]
        self.i += 1
        dist = {v: (0.6 if v == peak else 0.4 / (len(_SUPPORT) - 1)) for v in _SUPPORT}
        return ActionResult(action={"distribution": dist}, cost_eur=0.0, fallback_used=False)


def _seed_pgg_game(db_path: str, exp: str, contribs_and_msgs: list[tuple[int, str | None]]) -> None:
    db.insert_experiment(
        db_path, db.ExperimentRecord(id=exp, config_json="{}", seed=0, start_time="t")
    )
    db.insert_agent(
        db_path,
        db.AgentRecord(experiment_id=exp, agent_id="a1", model_name="m", capability_tier="weak"),
    )
    for round_num, (contrib, msg) in enumerate(contribs_and_msgs, start=1):
        db.insert_action(
            db_path,
            db.ActionRecord(
                experiment_id=exp,
                round_num=round_num,
                agent_id="a1",
                action_json={"contribution": contrib},
                message=msg,
                retries=0,
                cost_eur=0.0,
                fallback_used=False,
            ),
        )


def test_export_decomposes_headline(db_path: str) -> None:
    _seed_pgg_game(db_path, "exp1", [(0, "msg a"), (10, "msg b"), (20, "msg c"), (5, "msg d")])

    records = export_round_records(db_path, "exp1", game="public_goods", agent=_MockScorer())
    results = deception.score_game(
        db_path, "exp1", game="public_goods", agent=_MockScorer(), persist=False
    )
    kl_headline = results[0].kl_headline
    surprisal = results[0].surprisal

    assert len(records) == 4
    assert sum(r.kl_round for r in records) / len(records) == pytest.approx(kl_headline)
    assert sum(r.surprisal_round for r in records) / len(records) == pytest.approx(surprisal)


def test_export_skips_rounds_without_message(db_path: str) -> None:
    _seed_pgg_game(db_path, "exp2", [(0, "hi"), (10, None), (20, "  "), (5, "bye")])
    records = export_round_records(db_path, "exp2", game="public_goods", agent=_MockScorer())
    # only the two non-empty messages are scored
    assert len(records) == 2
    assert [r.round_num for r in records] == [1, 4]
    # history captured is the prior actions (round 4 has seen 0, 10, 20)
    assert records[1].history == [0.0, 10.0, 20.0]
    assert records[1].action == 5.0


def test_correlate_perfect_monotonic() -> None:
    res = correlate([0, 1, 2, 3], [0.1, 0.5, 1.0, 2.0], n_perm=500, n_boot=500)
    assert res["n"] == 4
    assert res["spearman_rho"] == pytest.approx(1.0)
    assert res["auc"] == pytest.approx(1.0)  # threshold 2 -> labels [0,0,1,1]
    assert res["n_deceptive"] == 2


def test_correlate_anticorrelated() -> None:
    res = correlate([0, 1, 2, 3], [2.0, 1.0, 0.5, 0.1], n_perm=500, n_boot=500)
    assert res["spearman_rho"] == pytest.approx(-1.0)
    assert res["auc"] == pytest.approx(0.0)


def test_correlate_auc_na_when_one_class() -> None:
    res = correlate([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4], deceptive_threshold=2)
    assert math.isnan(res["auc"])
    assert res["auc_na"] is not None
    assert res["n_deceptive"] == 0
