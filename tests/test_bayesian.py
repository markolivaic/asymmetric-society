"""Bayesian capability inference: feature math, extraction, and the model.

All offline. The synthetic generator drives the model tests with known-separable
(and known-overlapping) classes, so the honest gate (balanced accuracy / AUC) is
checked both ways - it must pass on separable data and FAIL on overlap.
"""

from __future__ import annotations

import numpy as np
import pytest

from asymmetric_society.metrics.bayesian import (
    CONSISTENCY_FEATURE,
    FEATURE_NAMES,
    extract_features,
    f_endgame_drop,
    f_mean,
    f_reciprocity,
    f_std,
    f_trend_slope,
    fit_infer,
    make_synthetic,
)
from asymmetric_society.runner import db


# --------------------------------------------------------------------------- #
# Pure feature helpers
# --------------------------------------------------------------------------- #


def test_f_mean_and_std() -> None:
    assert f_mean([10, 20, 30]) == pytest.approx(20.0)
    assert f_std([5, 5, 5]) == pytest.approx(0.0)
    assert f_mean([]) == 0.0


def test_f_trend_slope() -> None:
    assert f_trend_slope([1, 2, 3, 4], [10, 8, 6, 4]) == pytest.approx(-2.0)
    assert f_trend_slope([1], [10]) == 0.0  # too short


def test_f_endgame_drop() -> None:
    assert f_endgame_drop([10, 10, 10, 0]) == pytest.approx(10.0)  # defects on last round
    assert f_endgame_drop([5, 5]) == pytest.approx(0.0)
    assert f_endgame_drop([7]) == 0.0


def test_f_reciprocity() -> None:
    assert f_reciprocity([2, 4, 6], [1, 2, 3]) == pytest.approx(1.0)  # perfectly tracks
    assert f_reciprocity([5, 5, 5], [1, 2, 3]) == 0.0  # no own variance
    assert f_reciprocity([5], [1]) == 0.0  # too few pairs


# --------------------------------------------------------------------------- #
# Feature extraction from a seeded DB
# --------------------------------------------------------------------------- #


def _seed_two_agents(db_path: str) -> None:
    exp = "exp-bayes-01"
    db.insert_agent(db_path, db.AgentRecord(exp, "strong01", "claude", "strong"))
    db.insert_agent(db_path, db.AgentRecord(exp, "weak01", "llama", "weak"))
    # strong: high & stable; weak: low & noisy.
    plays = {
        "strong01": [20, 20, 20, 20],
        "weak01": [0, 5, 0, 5],
    }
    for agent_id, seq in plays.items():
        for rnd, contribution in enumerate(seq, start=1):
            db.insert_action(
                db_path,
                db.ActionRecord(
                    experiment_id=exp,
                    round_num=rnd,
                    agent_id=agent_id,
                    action_json={"contribution": contribution},
                    message=None,
                    retries=0,
                    cost_eur=0.0,
                    fallback_used=False,
                ),
            )


def test_extract_features_shapes_and_labels(db_path: str) -> None:
    _seed_two_agents(db_path)
    X, y, ids, names = extract_features(db_path)
    assert X.shape == (2, len(FEATURE_NAMES))
    assert names == list(FEATURE_NAMES)
    assert sorted(y.tolist()) == [0, 1]  # one strong, one weak
    # The strong agent's mean_contribution (col 0) is far higher than the weak one.
    label = {aid: y[i] for i, (_exp, aid) in enumerate(ids)}
    means = {aid: X[i, 0] for i, (_exp, aid) in enumerate(ids)}
    assert label["strong01"] == 1 and means["strong01"] == pytest.approx(20.0)
    assert means["strong01"] > means["weak01"]


def test_extract_features_with_consistency(db_path: str) -> None:
    _seed_two_agents(db_path)
    # strong = consistent (low surprisal); weak = inconsistent (high surprisal).
    db.insert_deception_score(db_path, db.DeceptionRecord(
        experiment_id="exp-bayes-01", agent_id="strong01", kl_headline=0.1,
        surprisal=0.2, direction="revealed_given_stated", n_rounds=4))
    db.insert_deception_score(db_path, db.DeceptionRecord(
        experiment_id="exp-bayes-01", agent_id="weak01", kl_headline=2.0,
        surprisal=1.8, direction="revealed_given_stated", n_rounds=4))

    X, y, ids, names = extract_features(db_path, with_consistency=True)
    assert X.shape == (2, len(FEATURE_NAMES) + 1)
    assert names[-1] == CONSISTENCY_FEATURE
    # consistency = -surprisal, so the consistent strong agent scores higher.
    cons = {aid: X[i, -1] for i, (_exp, aid) in enumerate(ids)}
    assert cons["strong01"] > cons["weak01"]

    # Regression: default extraction is unchanged (5 features, original names).
    X5, _y5, _ids5, names5 = extract_features(db_path)
    assert X5.shape[1] == len(FEATURE_NAMES)
    assert names5 == list(FEATURE_NAMES)


# --------------------------------------------------------------------------- #
# Model: separable passes, overlap fails (honest gate)
# --------------------------------------------------------------------------- #


def test_separable_classes_beat_the_bar() -> None:
    X, y = make_synthetic(n_per_class=60, separation=3.0, seed=0)
    rep = fit_infer(X, y, seed=0)
    assert rep["balanced_accuracy"] >= 0.8  # well past the 0.65 target
    assert rep["roc_auc"] >= 0.9
    assert rep["majority_baseline"] == pytest.approx(0.5)  # balanced synthetic


def test_overlapping_classes_fail_the_bar() -> None:
    X, y = make_synthetic(n_per_class=60, separation=0.0, seed=0)
    rep = fit_infer(X, y, seed=0)
    assert rep["balanced_accuracy"] < 0.65  # honest: chance-level, not a rubber stamp


def test_posteriors_track_class() -> None:
    X, y = make_synthetic(n_per_class=60, separation=3.0, seed=1)
    rep = fit_infer(X, y, seed=1)
    post = np.asarray(rep["posteriors"])
    assert post.min() >= 0.0 and post.max() <= 1.0
    y = np.asarray(y)
    assert post[y == 1].mean() > post[y == 0].mean()  # strong class gets higher P(strong)


def test_majority_baseline_reports_imbalance() -> None:
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (40, 5))
    y = [1] * 10 + [0] * 30  # 1:3 like the real pilot
    rep = fit_infer(X, y, seed=0)
    assert rep["majority_baseline"] == pytest.approx(0.75)
    assert rep["n_strong"] == 10 and rep["n_weak"] == 30


def test_determinism_same_seed() -> None:
    X, y = make_synthetic(n_per_class=40, separation=1.5, seed=7)
    a = fit_infer(X, y, seed=3)
    b = fit_infer(X, y, seed=3)
    assert a["balanced_accuracy"] == b["balanced_accuracy"]
    assert a["roc_auc"] == b["roc_auc"]
