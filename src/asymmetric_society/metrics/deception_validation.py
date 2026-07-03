"""C3 validation of the KL deception metric against blind human ratings.

The KL metric (``metrics/deception.py``) is built and synthetically checked, but
its *validity* - does a high KL actually correspond to deception a human sees? -
can only be tested on real messages with human scoring (the KL deception metric's "C3" caveat).
This module provides the two pieces that test needs:

  * :func:`export_round_records` - re-runs the per-message Haiku extraction over a
    game and emits one record per scored round with the *per-round* KL and
    surprisal (the components that average to the agent/game ``kl_headline`` and
    ``surprisal``), plus the message, the action taken THAT round, and the agent's
    prior-action history. This is the per-message granularity the validation
    correlates against, and it decomposes the headline exactly (see the test).
  * :func:`correlate` - given aligned human ordinal scores (0-3) and a metric
    column, computes Spearman rho with a permutation p-value and bootstrap CI, and
    a binary-deception ROC-AUC (human >= threshold). Run AFTER blind rating.

Blindness (the methodological core, see ``docs/zav29_c3_rubric.md``): the rating
sheet a human fills carries only message/action/history - never KL, surprisal,
tier, or ids. KL meets the human scores only here, after the ratings are in.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from asymmetric_society.agents.anthropic_agent import AnthropicAgent
from asymmetric_society.metrics.deception import (
    ACTION_SUPPORT,
    _ACTION_KEY,
    deception_scores,
    extract_stated,
)
from asymmetric_society.runner import db


@dataclass
class RoundRecord:
    """One scored (message-bearing) round, the unit of C3 validation."""

    experiment_id: str
    agent_id: str
    round_num: int
    game: str
    message: str
    action: float
    history: list[float]
    kl_round: float
    surprisal_round: float


def _read_rounds_with_num(
    db_path: str, experiment_id: str, action_key: str
) -> dict[str, list[tuple[int, float | None, str | None]]]:
    """Per agent, ordered ``(round_num, action_value, message)`` across rounds.

    Like ``deception._read_agent_rounds`` but keeps ``round_num`` so an exported
    record can be joined back to a hand label.
    """
    con = db.connect(db_path)
    try:
        rows = con.execute(
            "SELECT agent_id, round_num, action_json, message FROM actions "
            "WHERE experiment_id = ? ORDER BY agent_id, round_num",
            (experiment_id,),
        ).fetchall()
    finally:
        con.close()
    out: dict[str, list[tuple[int, float | None, str | None]]] = {}
    for agent_id, round_num, action_json, message in rows:
        action = json.loads(action_json)
        out.setdefault(agent_id, []).append((int(round_num), action.get(action_key), message))
    return out


def export_round_records(
    db_path: str,
    experiment_id: str,
    *,
    game: str,
    agent: AnthropicAgent,
    game_context: str = "",
    epsilon: float = 1e-3,
) -> list[RoundRecord]:
    """Emit one :class:`RoundRecord` per message-bearing scored round of a game.

    Mirrors ``deception.score_game``'s extraction loop but, instead of averaging,
    keeps each round's KL/surprisal (reusing the canonical per-round values that
    ``deception_scores`` now returns, so the export decomposes the headline
    exactly). The Haiku extraction is repeated here (one call per message).

    Args:
        db_path: Results database to read actions/messages from.
        experiment_id: The game to export.
        game: Deception game family (``"public_goods"`` or ``"trust_send"``).
        agent: The Haiku scoring agent (inject a mock in tests).
        game_context: Context string passed to every extraction.
        epsilon: Laplace smoothing (must match the scoring run).

    Returns:
        Per-agent, per-scored-round records in round order.
    """
    support = ACTION_SUPPORT[game]
    action_key = _ACTION_KEY[game]
    per_agent = _read_rounds_with_num(db_path, experiment_id, action_key)

    records: list[RoundRecord] = []
    for agent_id, rounds in per_agent.items():
        revealed_actions: list[float] = []
        stated_per_round: list[dict[int, float] | None] = []
        scored_meta: list[tuple[int, str, float, list[float]]] = []
        history: list[float] = []
        for round_num, action_value, message in rounds:
            if action_value is None:
                continue  # no-op / off-role turn carries no action
            action = float(action_value)
            revealed_actions.append(action)
            if message and message.strip():
                stated = extract_stated(message, game_context, support, agent=agent)
                stated_per_round.append(stated.distribution)
                scored_meta.append((round_num, message, action, list(history)))
            else:
                stated_per_round.append(None)
            history.append(action)

        scores = deception_scores(revealed_actions, stated_per_round, support, epsilon=epsilon)
        for (round_num, message, action, hist), per in zip(scored_meta, scores["per_round"]):
            records.append(
                RoundRecord(
                    experiment_id=experiment_id,
                    agent_id=agent_id,
                    round_num=round_num,
                    game=game,
                    message=message,
                    action=action,
                    history=hist,
                    kl_round=float(per["kl"]),
                    surprisal_round=float(per["surprisal"]),
                )
            )
    return records


# --------------------------------------------------------------------------- #
# Correlation / AUC (run AFTER blind human rating)
# --------------------------------------------------------------------------- #


def correlate(
    human: Sequence[float],
    metric: Sequence[float],
    *,
    deceptive_threshold: int = 2,
    n_perm: int = 10000,
    n_boot: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    """Validate a metric column against blind human ordinal deception scores.

    Args:
        human: Ordinal human scores (0-3), one per rated message.
        metric: The metric value (per-round KL or surprisal) for the same messages.
        deceptive_threshold: Human score at/above which a message is "deceptive"
            for the binary ROC-AUC (default 2).
        n_perm: Label permutations for the Spearman null p-value.
        n_boot: Bootstrap resamples for the rho CI.
        seed: RNG seed for reproducible permutation/bootstrap.

    Returns:
        ``{"n", "spearman_rho", "perm_p", "rho_ci", "auc", "n_deceptive", "auc_na"}``.
        ``auc`` is ``nan`` (``auc_na`` reason set) if one binary class is empty.
    """
    h = np.asarray(human, dtype=float)
    m = np.asarray(metric, dtype=float)
    mask = ~(np.isnan(h) | np.isnan(m))
    h, m = h[mask], m[mask]
    n = int(h.size)
    rng = np.random.default_rng(seed)

    rho = float(spearmanr(h, m).statistic) if n >= 2 else float("nan")

    # Permutation null: how often a shuffled labelling matches/beats |rho|.
    perm_p = float("nan")
    if n >= 3 and not math.isnan(rho):
        count = 0
        for _ in range(n_perm):
            if abs(spearmanr(rng.permutation(h), m).statistic) >= abs(rho):
                count += 1
        perm_p = (count + 1) / (n_perm + 1)

    # Bootstrap CI for rho.
    rho_ci: tuple[float, float] = (float("nan"), float("nan"))
    if n >= 3 and not math.isnan(rho):
        boots = []
        idx = np.arange(n)
        for _ in range(n_boot):
            s = rng.choice(idx, size=n, replace=True)
            if np.unique(h[s]).size < 2 or np.unique(m[s]).size < 2:
                continue
            boots.append(spearmanr(h[s], m[s]).statistic)
        if boots:
            rho_ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))

    # Binary ROC-AUC: human >= threshold is the positive (deceptive) class.
    labels = (h >= deceptive_threshold).astype(int)
    n_deceptive = int(labels.sum())
    auc = float("nan")
    auc_na = None
    if 0 < n_deceptive < n:
        auc = float(roc_auc_score(labels, m))
    else:
        auc_na = f"only one class present (deceptive={n_deceptive}/{n})"

    return {
        "n": n,
        "spearman_rho": rho,
        "perm_p": perm_p,
        "rho_ci": rho_ci,
        "auc": auc,
        "n_deceptive": n_deceptive,
        "auc_na": auc_na,
    }
