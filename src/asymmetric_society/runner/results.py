"""Read-side queries over the results database (stdlib ``sqlite3``).

The write helpers live in :mod:`asymmetric_society.runner.db`; this module is the
read counterpart used by the pilot runner's report and later analysis. It computes
the parse-failure gate, pulls a full game transcript for eyeball verification that
the model actually plays (not just emits valid JSON), surfaces raw failure samples
for diagnosis, and reads final payoffs for the inequality metrics.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from asymmetric_society.runner.db import connect


def _placeholders(n: int) -> str:
    return ",".join("?" * n)


def _query(db_path: str | Path, sql: str, params: tuple[Any, ...]) -> list[dict[str, Any]]:
    con = connect(db_path)
    try:
        con.row_factory = sqlite3.Row
        rows = con.execute(sql, params).fetchall()
    finally:
        con.close()
    return [dict(row) for row in rows]


def parse_failure_stats(
    db_path: str | Path, experiment_ids: list[str]
) -> list[dict[str, Any]]:
    """Parse-failure rates per capability tier across the given experiments.

    Returns one dict per tier with three rates (all reported; the project gate is
    ``attempt_fail_rate`` < 0.15):

    * ``fallback_rate`` - fraction of calls that gave up entirely (fell back).
    * ``any_parse_fail_rate`` - fraction of calls with >= 1 bad attempt.
    * ``attempt_fail_rate`` - bad attempts / total attempts (the gate metric).

    No-op turns are excluded. A sequential game (e.g. the Trust Game) prompts the
    off-role agent for a placeholder ``{"noop": true}`` turn that carries no
    decision; counting these would pad the gate with trivially-parsed calls. We
    filter on ``parsed_action`` (the validated action this very table stores), so
    no join is needed and no real call can be dropped - a games without no-ops
    (Public Goods) is unaffected and the gate numbers are identical.
    """
    if not experiment_ids:
        return []
    sql = f"""
        SELECT capability_tier,
               COUNT(*) AS calls,
               AVG(CAST(fallback_used AS REAL))                          AS fallback_rate,
               AVG(CASE WHEN parse_fail_count > 0 THEN 1.0 ELSE 0.0 END) AS any_parse_fail_rate,
               CAST(SUM(parse_fail_count) AS REAL)
                 / NULLIF(SUM(parse_fail_count + parse_ok), 0)           AS attempt_fail_rate
        FROM llm_calls
        WHERE experiment_id IN ({_placeholders(len(experiment_ids))})
          AND (parsed_action IS NULL OR parsed_action NOT LIKE '%"noop"%')
        GROUP BY capability_tier
        ORDER BY capability_tier
    """
    return _query(db_path, sql, tuple(experiment_ids))


def parse_failure_examples(
    db_path: str | Path, experiment_ids: list[str], *, limit: int = 10
) -> list[dict[str, Any]]:
    """Raw responses that failed to parse or fell back, for borderline diagnosis.

    A borderline gate (e.g. 12-18%) is often a cheap prompt fix (model wraps JSON
    in ```json fences or prepends prose), not a model-capability problem - these
    samples are what a human inspects to decide.
    """
    if not experiment_ids:
        return []
    sql = f"""
        SELECT experiment_id, round_no, agent_id, capability_tier,
               parse_fail_count, fallback_used, raw_response
        FROM llm_calls
        WHERE experiment_id IN ({_placeholders(len(experiment_ids))})
          AND (parse_fail_count > 0 OR fallback_used = 1)
        ORDER BY experiment_id, round_no, agent_id
        LIMIT ?
    """
    return _query(db_path, sql, (*experiment_ids, limit))


def final_payoffs(
    db_path: str | Path, experiment_ids: list[str]
) -> list[dict[str, Any]]:
    """Final (last-round) cumulative payoff per agent/tier for each experiment."""
    if not experiment_ids:
        return []
    sql = f"""
        SELECT p.experiment_id, p.agent_id, a.capability_tier, p.cumulative_payoff
        FROM payoffs p
        JOIN agents a
          ON a.experiment_id = p.experiment_id AND a.agent_id = p.agent_id
        WHERE p.experiment_id IN ({_placeholders(len(experiment_ids))})
          AND p.round_num = (
              SELECT MAX(p2.round_num) FROM payoffs p2
              WHERE p2.experiment_id = p.experiment_id
          )
        ORDER BY p.experiment_id, a.capability_tier, p.agent_id
    """
    return _query(db_path, sql, tuple(experiment_ids))


def total_cost(db_path: str | Path, experiment_ids: list[str]) -> float:
    """Sum ``total_cost_eur`` across the given experiments (€0 for all-local runs)."""
    if not experiment_ids:
        return 0.0
    rows = _query(
        db_path,
        f"SELECT COALESCE(SUM(total_cost_eur), 0.0) AS total FROM experiments "
        f"WHERE id IN ({_placeholders(len(experiment_ids))})",
        tuple(experiment_ids),
    )
    return float(rows[0]["total"])


def game_transcript(db_path: str | Path, experiment_id: str) -> list[dict[str, Any]]:
    """Full per-round, per-agent transcript of one game for eyeball checks.

    Each row: round_num, agent_id, capability_tier, message (if any), round_payoff,
    and the per-game action fields parsed out of the action JSON - ``contribution``
    (Public Goods) and ``send``/``return`` (Trust Game); absent fields are ``None``.
    Lets a human confirm the model actually varies its moves and responds to history
    rather than emitting a constant.
    """
    sql = """
        SELECT ac.round_num, ac.agent_id, ag.capability_tier,
               ac.action_json, ac.message, ac.fallback_used, p.round_payoff
        FROM actions ac
        JOIN agents ag
          ON ag.experiment_id = ac.experiment_id AND ag.agent_id = ac.agent_id
        LEFT JOIN payoffs p
          ON p.experiment_id = ac.experiment_id
         AND p.round_num = ac.round_num
         AND p.agent_id = ac.agent_id
        WHERE ac.experiment_id = ?
        ORDER BY ac.round_num, ac.agent_id
    """
    rows = _query(db_path, sql, (experiment_id,))
    for row in rows:
        action = json.loads(row.pop("action_json"))
        row["contribution"] = action.get("contribution")
        row["send"] = action.get("send")
        row["return"] = action.get("return")
    return rows
