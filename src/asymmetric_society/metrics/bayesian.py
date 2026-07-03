"""Bayesian capability inference: infer tier from behaviour.

Trains a logistic regression on per-agent behavioural features and outputs a
posterior ``P(strong | behavior)``. ``predict_proba`` is the posterior under the
logistic likelihood; the intercept encodes the empirical prior (class base rate),
optionally overridable.

Features are PGG, communication-OFF, and **action-based only** (not payoff - we
test whether *behaviour* is tier-distinguishable, not whether the *outcome* leaks
the tier, which would be circular):

* ``mean_contribution`` / ``std_contribution`` - level and volatility of play.
* ``trend_slope`` - linear slope of contribution over rounds (end-game decline).
* ``endgame_drop`` - last-round contribution vs the earlier mean (defection spike).
* ``reciprocity`` - corr(own action_t, others' mean action_{t-1}) (conditional
  cooperation).

A ``message_action_consistency`` feature (comm-ON) is a deliberate follow-on for
the human-rating pilot once real messages exist; it is not part of the active
feature set here.

**Honest evaluation (see Decision Log).** On the labelled pilot the classes are
imbalanced 1:3, so raw accuracy ≥ 0.75 is the trivial majority baseline. The
meaningful "tier-distinguishable" signal is **balanced accuracy / ROC-AUC vs
0.50**. And the pilot has only 10 strong agents of a *single* model (Haiku) vs a
single weak model (Llama), so any pilot number is a smoke test of the pipeline,
not a finding - proper evaluation needs Week-3 scale and the full strong tier.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from asymmetric_society.runner import db

#: Active behavioural features (PGG, comm-OFF), in fixed column order.
FEATURE_NAMES: tuple[str, ...] = (
    "mean_contribution",
    "std_contribution",
    "trend_slope",
    "endgame_drop",
    "reciprocity",
)

#: Optional 6th feature (comm-ON only): message-action consistency, derived from
#: the deception metric's per-agent surprisal (consistency = -surprisal, so higher
#: = the action is LESS surprising given the stated message). NOT independent of
#: the KL deception metric (same stated-vs-revealed signal) -- it is a tier feature,
#: never an independent confirmation of the deception result.
CONSISTENCY_FEATURE = "message_action_consistency"

#: Which parsed-action field holds the scalar action for each game family.
_ACTION_KEY: dict[str, str] = {"public_goods": "contribution"}


def _read_surprisal(db_path: str, experiment_ids: Sequence[str]) -> dict[tuple[str, str], float]:
    """Per-agent surprisal from ``deception_scores`` (for the consistency feature).

    Only comm-ON games have rows (that is where deception was scored); a (exp, agent)
    pair absent here has no message signal.
    """
    unique = sorted(set(experiment_ids))
    if not unique:
        return {}
    placeholders = ",".join("?" * len(unique))
    con = db.connect(db_path)
    try:
        rows = con.execute(
            f"SELECT experiment_id, agent_id, surprisal FROM deception_scores "
            f"WHERE experiment_id IN ({placeholders})",
            tuple(unique),
        ).fetchall()
    finally:
        con.close()
    return {(exp, aid): float(s) for exp, aid, s in rows if s is not None}


# --------------------------------------------------------------------------- #
# Pure feature helpers
# --------------------------------------------------------------------------- #


def f_mean(seq: Sequence[float]) -> float:
    """Mean action over rounds (0.0 for an empty sequence)."""
    return float(np.mean(seq)) if len(seq) else 0.0


def f_std(seq: Sequence[float]) -> float:
    """Population standard deviation of the action over rounds."""
    return float(np.std(seq)) if len(seq) else 0.0


def f_trend_slope(rounds: Sequence[int], seq: Sequence[float]) -> float:
    """Linear slope of action vs round index (0.0 if fewer than 2 points)."""
    if len(seq) < 2:
        return 0.0
    return float(np.polyfit(np.asarray(rounds, dtype=float), np.asarray(seq, dtype=float), 1)[0])


def f_endgame_drop(seq: Sequence[float]) -> float:
    """Earlier-rounds mean minus the final action (positive = end-game defection)."""
    if len(seq) < 2:
        return 0.0
    return float(np.mean(seq[:-1]) - seq[-1])


def f_reciprocity(own_t: Sequence[float], others_prev: Sequence[float]) -> float:
    """Correlation between own action_t and others' mean action_{t-1}.

    Returns 0.0 when there are fewer than two aligned pairs or either side has no
    variance (correlation undefined).
    """
    if len(own_t) < 2 or len(others_prev) < 2:
        return 0.0
    a = np.asarray(own_t, dtype=float)
    b = np.asarray(others_prev, dtype=float)
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    corr = np.corrcoef(a, b)[0, 1]
    return float(corr) if np.isfinite(corr) else 0.0


def _agent_features(rounds: list[int], own: dict[int, float], others_mean: dict[int, float]) -> list[float]:
    """Assemble the feature vector for one agent from its per-round data."""
    own_rounds = [r for r in rounds if r in own]
    own_seq = [own[r] for r in own_rounds]
    own_t: list[float] = []
    others_prev: list[float] = []
    for r in own_rounds:
        if (r - 1) in others_mean:
            own_t.append(own[r])
            others_prev.append(others_mean[r - 1])
    return [
        f_mean(own_seq),
        f_std(own_seq),
        f_trend_slope(own_rounds, own_seq),
        f_endgame_drop(own_seq),
        f_reciprocity(own_t, others_prev),
    ]


# --------------------------------------------------------------------------- #
# Feature extraction from the results DB
# --------------------------------------------------------------------------- #


def extract_features(
    db_path: str,
    experiment_ids: Sequence[str] | None = None,
    *,
    game: str = "public_goods",
    with_consistency: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, str]], list[str]]:
    """Build the per-agent feature matrix and tier labels from a results DB.

    Args:
        db_path: SQLite results database.
        experiment_ids: Restrict to these experiments (all experiments if None).
        game: Game family selecting the scalar action field.

    Returns:
        ``(X, y, agent_ids, feature_names)`` where ``X`` is ``(n_agents,
        n_features)``, ``y`` is 1 for strong / 0 for weak, and ``agent_ids`` are
        ``(experiment_id, agent_id)`` pairs aligned with the rows.
    """
    action_key = _ACTION_KEY[game]
    con = db.connect(db_path)
    try:
        if experiment_ids:
            ph = ",".join("?" * len(experiment_ids))
            agent_rows = con.execute(
                f"SELECT experiment_id, agent_id, capability_tier FROM agents "
                f"WHERE experiment_id IN ({ph})",
                tuple(experiment_ids),
            ).fetchall()
            act_rows = con.execute(
                f"SELECT experiment_id, round_num, agent_id, action_json FROM actions "
                f"WHERE experiment_id IN ({ph})",
                tuple(experiment_ids),
            ).fetchall()
        else:
            agent_rows = con.execute(
                "SELECT experiment_id, agent_id, capability_tier FROM agents"
            ).fetchall()
            act_rows = con.execute(
                "SELECT experiment_id, round_num, agent_id, action_json FROM actions"
            ).fetchall()
    finally:
        con.close()

    # contributions[experiment][round][agent] = action value
    contributions: dict[str, dict[int, dict[str, float]]] = {}
    for exp, rnd, aid, action_json in act_rows:
        value = json.loads(action_json).get(action_key)
        if value is None:
            continue
        contributions.setdefault(exp, {}).setdefault(int(rnd), {})[aid] = float(value)

    rows_x: list[list[float]] = []
    rows_y: list[int] = []
    ids: list[tuple[str, str]] = []
    for exp, aid, tier in sorted(agent_rows):
        per_round = contributions.get(exp, {})
        rounds = sorted(per_round.keys())
        own = {r: per_round[r][aid] for r in rounds if aid in per_round[r]}
        if not own:
            continue
        others_mean: dict[int, float] = {}
        for r in rounds:
            others = [v for a, v in per_round[r].items() if a != aid]
            if others:
                others_mean[r] = float(np.mean(others))
        rows_x.append(_agent_features(rounds, own, others_mean))
        rows_y.append(1 if tier == "strong" else 0)
        ids.append((exp, aid))

    feature_names = list(FEATURE_NAMES)
    if with_consistency:
        surprisal = _read_surprisal(db_path, [exp for exp, _ in ids])
        values = np.array([surprisal.get(pair, np.nan) for pair in ids], dtype=float)
        if np.isnan(values).any():
            fill = float(np.nanmean(values)) if not np.isnan(values).all() else 0.0
            values = np.where(np.isnan(values), fill, values)
        for row, surp in zip(rows_x, values):
            row.append(-float(surp))  # consistency = -surprisal (higher = more consistent)
        feature_names.append(CONSISTENCY_FEATURE)

    return (
        np.asarray(rows_x, dtype=float),
        np.asarray(rows_y, dtype=int),
        ids,
        feature_names,
    )


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def _pipeline() -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000)),
        ]
    )


def fit_infer(
    X: np.ndarray,
    y: Sequence[int],
    *,
    n_splits: int = 5,
    seed: int = 0,
    prior: float | None = None,
) -> dict[str, Any]:
    """Fit the standardized logistic regression and report honest CV metrics.

    Args:
        X: Feature matrix ``(n_agents, n_features)``.
        y: Tier labels (1 strong / 0 weak).
        n_splits: Desired CV folds (capped by the smaller class count).
        seed: RNG seed for the stratified split (reproducible).
        prior: Optional P(strong) prior; shifts the intercept via a log-odds
            offset. Default uses the empirical base rate.

    Returns:
        ``accuracy``, ``balanced_accuracy``, ``roc_auc`` (cross-validated; ``nan``
        when a class has < 2 members), ``cv_folds``, ``majority_baseline``, ``n``,
        ``n_strong``, ``n_weak``, ``posteriors`` (P(strong) per agent), and
        ``coefficients`` (standardized logistic weights per feature).
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(list(y), dtype=int)
    n = len(y)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    base = n_pos / n if n else float("nan")
    majority = max(n_pos, n_neg) / n if n else float("nan")

    pipe = _pipeline()
    acc = bal = auc = float("nan")
    folds = 0
    if n_pos >= 2 and n_neg >= 2:
        folds = min(n_splits, n_pos, n_neg)
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        acc = float(cross_val_score(pipe, X, y, cv=skf, scoring="accuracy").mean())
        bal = float(cross_val_score(pipe, X, y, cv=skf, scoring="balanced_accuracy").mean())
        try:
            auc = float(cross_val_score(pipe, X, y, cv=skf, scoring="roc_auc").mean())
        except ValueError:
            auc = float("nan")

    pipe.fit(X, y)
    lr: LogisticRegression = pipe.named_steps["lr"]
    strong_col = list(lr.classes_).index(1)
    posteriors = pipe.predict_proba(X)[:, strong_col]
    if prior is not None and 0.0 < prior < 1.0 and 0.0 < base < 1.0:
        p = np.clip(posteriors, 1e-9, 1 - 1e-9)
        shift = math.log(prior / (1 - prior)) - math.log(base / (1 - base))
        logits = np.log(p / (1 - p)) + shift
        posteriors = 1.0 / (1.0 + np.exp(-logits))

    names = list(FEATURE_NAMES) if X.shape[1] == len(FEATURE_NAMES) else [
        f"f{i}" for i in range(X.shape[1])
    ]
    coefficients = dict(zip(names, lr.coef_[0].tolist()))

    return {
        "accuracy": acc,
        "balanced_accuracy": bal,
        "roc_auc": auc,
        "cv_folds": folds,
        "majority_baseline": float(majority),
        "n": int(n),
        "n_strong": n_pos,
        "n_weak": n_neg,
        "posteriors": posteriors.tolist(),
        "coefficients": coefficients,
    }


def make_synthetic(
    n_per_class: int,
    separation: float,
    *,
    n_features: int = 5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Two balanced Gaussian clusters ``separation`` apart, for unit tests.

    Class 0 ~ N(0, 1); class 1 ~ N(separation, 1). Large ``separation`` is easily
    separable; ``separation = 0`` is pure overlap (chance-level).
    """
    rng = np.random.default_rng(seed)
    x0 = rng.normal(0.0, 1.0, (n_per_class, n_features))
    x1 = rng.normal(separation, 1.0, (n_per_class, n_features))
    X = np.vstack([x0, x1])
    y = np.array([0] * n_per_class + [1] * n_per_class, dtype=int)
    return X, y


def infer_from_db(
    db_path: str,
    experiment_ids: Sequence[str] | None = None,
    *,
    game: str = "public_goods",
    n_splits: int = 5,
    seed: int = 0,
    with_consistency: bool = False,
) -> dict[str, Any]:
    """Convenience: extract features from a DB and fit/evaluate the classifier."""
    X, y, agent_ids, feature_names = extract_features(
        db_path, experiment_ids, game=game, with_consistency=with_consistency
    )
    result = fit_infer(X, y, n_splits=n_splits, seed=seed)
    result["feature_names"] = feature_names
    result["agent_ids"] = agent_ids
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: infer tier from behaviour on a results DB and print an honest report.

    Reads an existing database only - no API spend.
    """
    parser = argparse.ArgumentParser(description="Bayesian capability inference.")
    parser.add_argument("--db", required=True, help="Results database (e.g. results/pilot.db).")
    parser.add_argument("--game", default="public_goods", help="Game family.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    result = infer_from_db(args.db, game=args.game, seed=args.seed)
    print(f"Bayesian capability inference on {args.db} ({args.game})\n")
    print(f"  agents: {result['n']}  (strong={result['n_strong']}, weak={result['n_weak']})")
    print(f"  majority baseline (raw): {result['majority_baseline']:.3f}")
    print(f"  CV folds: {result['cv_folds']}")
    print(f"  raw accuracy:      {result['accuracy']:.3f}")
    print(f"  balanced accuracy: {result['balanced_accuracy']:.3f}   <- honest tier-distinguishability")
    print(f"  ROC-AUC:           {result['roc_auc']:.3f}   <- (chance = 0.5)")
    print("\n  feature coefficients (standardized; + => predicts strong):")
    for name, coef in sorted(
        result["coefficients"].items(), key=lambda kv: abs(kv[1]), reverse=True
    ):
        print(f"    {name:20} {coef:+.3f}")
    print("\n  NOTE: pilot is imbalanced 1:3, N=10 strong of a single model (Haiku) vs")
    print("  a single weak model (Llama). This is a SMOKE TEST of the pipeline, not a")
    print("  finding; real evaluation needs Week-3 scale and the full strong tier.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
