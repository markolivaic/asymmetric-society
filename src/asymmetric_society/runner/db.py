"""SQLite logging of every LLM call (stdlib ``sqlite3``, not SQLAlchemy).

Every agent call is logged with its full prompt, raw response, parsed action,
token counts, cost, and the two failure counters. This table is the source of
truth for the KL-deception metric, the ZAV-20 parse-failure gate, budget
reconciliation, and debugging.

Failure accounting is split deliberately:
  * ``api_error_count`` - transient transport failures (rate limit, timeout,
    5xx) that were retried with backoff. NOT model mistakes.
  * ``parse_fail_count`` - the model returned text but the JSON was invalid or
    failed validation. These are the real model errors the ZAV-20 gate measures.

Later tickets (ZAV-19) add experiments/agents/rounds/actions/payoffs tables
alongside ``llm_calls``; ``init_db`` is the single place to extend.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_calls (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ts               TEXT    NOT NULL,
    experiment_id    TEXT,
    round_no         INTEGER,
    agent_id         TEXT    NOT NULL,
    model_name       TEXT    NOT NULL,
    capability_tier  TEXT    NOT NULL,
    prompt           TEXT    NOT NULL,
    raw_response     TEXT,
    parsed_action    TEXT,
    api_error_count  INTEGER NOT NULL DEFAULT 0,
    parse_fail_count INTEGER NOT NULL DEFAULT 0,
    input_tokens     INTEGER NOT NULL DEFAULT 0,
    output_tokens    INTEGER NOT NULL DEFAULT 0,
    cost_eur         REAL    NOT NULL DEFAULT 0.0,
    parse_ok         INTEGER NOT NULL DEFAULT 0,
    fallback_used    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS experiments (
    id              TEXT    PRIMARY KEY,
    config_json     TEXT    NOT NULL,
    seed            INTEGER NOT NULL,
    start_time      TEXT    NOT NULL,
    end_time        TEXT,
    total_cost_eur  REAL    NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS agents (
    experiment_id    TEXT NOT NULL,
    agent_id         TEXT NOT NULL,
    model_name       TEXT NOT NULL,
    capability_tier  TEXT NOT NULL,
    PRIMARY KEY (experiment_id, agent_id)
);

CREATE TABLE IF NOT EXISTS rounds (
    experiment_id  TEXT    NOT NULL,
    round_num      INTEGER NOT NULL,
    PRIMARY KEY (experiment_id, round_num)
);

-- One row per agent per round. The (experiment_id, round_num, agent_id) key is
-- unique and matches the same triple on llm_calls 1:1 (act() logs exactly one
-- llm_calls row per call), so full prompt/response are reached by joining to
-- llm_calls rather than duplicated here.
CREATE TABLE IF NOT EXISTS actions (
    experiment_id  TEXT    NOT NULL,
    round_num      INTEGER NOT NULL,
    agent_id       TEXT    NOT NULL,
    action_json    TEXT    NOT NULL,
    message        TEXT,
    retries        INTEGER NOT NULL DEFAULT 0,
    cost_eur       REAL    NOT NULL DEFAULT 0.0,
    fallback_used  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (experiment_id, round_num, agent_id)
);

CREATE TABLE IF NOT EXISTS payoffs (
    experiment_id      TEXT    NOT NULL,
    round_num          INTEGER NOT NULL,
    agent_id           TEXT    NOT NULL,
    round_payoff       REAL    NOT NULL,
    cumulative_payoff  REAL    NOT NULL,
    PRIMARY KEY (experiment_id, round_num, agent_id)
);

-- ZAV-25 message classification. One row per classified agent message; the
-- (experiment_id, round_num, agent_id) triple matches the source row in
-- ``actions`` 1:1, so re-classifying a database is idempotent (INSERT OR
-- REPLACE on the UNIQUE key). The raw classifier LLM call is logged separately
-- in ``llm_calls`` (with capability_tier='classifier' and a NULL experiment_id
-- so it never enters the weak/strong parse-failure gate).
CREATE TABLE IF NOT EXISTS message_classifications (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             TEXT    NOT NULL,
    experiment_id  TEXT,
    round_num      INTEGER,
    agent_id       TEXT    NOT NULL,
    message        TEXT    NOT NULL,
    label          TEXT    NOT NULL,
    cost_eur       REAL    NOT NULL DEFAULT 0.0,
    fallback_used  INTEGER NOT NULL DEFAULT 0,
    UNIQUE (experiment_id, round_num, agent_id)
);

-- ZAV-22 KL deception scores. One row per agent per game per KL direction; the
-- UNIQUE key makes re-scoring a database idempotent (INSERT OR REPLACE).
CREATE TABLE IF NOT EXISTS deception_scores (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    ts             TEXT    NOT NULL,
    experiment_id  TEXT    NOT NULL,
    agent_id       TEXT    NOT NULL,
    kl_headline    REAL,
    surprisal      REAL,
    direction      TEXT    NOT NULL,
    n_rounds       INTEGER NOT NULL DEFAULT 0,
    cost_eur       REAL    NOT NULL DEFAULT 0.0,
    UNIQUE (experiment_id, agent_id, direction)
);
"""


@dataclass
class CallRecord:
    """One row destined for ``llm_calls``.

    Mirrors the table columns; booleans are stored as 0/1 and
    ``parsed_action`` is JSON-encoded on write.
    """

    agent_id: str
    model_name: str
    capability_tier: str
    prompt: str
    raw_response: str | None
    parsed_action: dict[str, Any] | None
    api_error_count: int
    parse_fail_count: int
    input_tokens: int
    output_tokens: int
    cost_eur: float
    parse_ok: bool
    fallback_used: bool
    experiment_id: str | None = None
    round_no: int | None = None


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite connection, creating the parent directory if needed.

    Args:
        db_path: Path to the database file.

    Returns:
        An open connection (caller is responsible for closing).
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(str(path))


def init_db(db_path: str | Path) -> None:
    """Create the ``llm_calls`` table if it does not yet exist.

    Args:
        db_path: Path to the database file.
    """
    con = connect(db_path)
    try:
        con.executescript(_SCHEMA)
        con.commit()
    finally:
        con.close()


def log_call(db_path: str | Path, record: CallRecord) -> int:
    """Insert one call record and return its row id.

    Args:
        db_path: Path to the database file.
        record: The call to persist.

    Returns:
        The autoincrement ``id`` of the inserted row.
    """
    con = connect(db_path)
    try:
        cur = con.execute(
            """
            INSERT INTO llm_calls (
                ts, experiment_id, round_no, agent_id, model_name, capability_tier,
                prompt, raw_response, parsed_action, api_error_count, parse_fail_count,
                input_tokens, output_tokens, cost_eur, parse_ok, fallback_used
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now(timezone.utc).isoformat(),
                record.experiment_id,
                record.round_no,
                record.agent_id,
                record.model_name,
                record.capability_tier,
                record.prompt,
                record.raw_response,
                json.dumps(record.parsed_action) if record.parsed_action is not None else None,
                record.api_error_count,
                record.parse_fail_count,
                record.input_tokens,
                record.output_tokens,
                record.cost_eur,
                int(record.parse_ok),
                int(record.fallback_used),
            ),
        )
        con.commit()
        return int(cur.lastrowid)
    finally:
        con.close()


# ---------------------------------------------------------------------------
# ZAV-19 experiment tables (experiments / agents / rounds / actions / payoffs).
# These record the game-level run alongside the per-call ``llm_calls`` log.
# ---------------------------------------------------------------------------


@dataclass
class ExperimentRecord:
    """One row for the ``experiments`` table (``end_time``/cost set at finalize)."""

    id: str
    config_json: str
    seed: int
    start_time: str


@dataclass
class AgentRecord:
    """One row for the ``agents`` table."""

    experiment_id: str
    agent_id: str
    model_name: str
    capability_tier: str


@dataclass
class ActionRecord:
    """One row for the ``actions`` table.

    ``action_json`` is the validated game action (JSON-encoded on write); full
    prompt/response live in ``llm_calls`` and are reached by joining on
    ``(experiment_id, round_num, agent_id)``.
    """

    experiment_id: str
    round_num: int
    agent_id: str
    action_json: dict[str, Any]
    message: str | None
    retries: int
    cost_eur: float
    fallback_used: bool


@dataclass
class PayoffRecord:
    """One row for the ``payoffs`` table."""

    experiment_id: str
    round_num: int
    agent_id: str
    round_payoff: float
    cumulative_payoff: float


def insert_experiment(db_path: str | Path, record: ExperimentRecord) -> None:
    """Insert the opening row for an experiment (before any rounds run)."""
    con = connect(db_path)
    try:
        con.execute(
            "INSERT INTO experiments (id, config_json, seed, start_time) "
            "VALUES (?, ?, ?, ?)",
            (record.id, record.config_json, record.seed, record.start_time),
        )
        con.commit()
    finally:
        con.close()


def finalize_experiment(
    db_path: str | Path, experiment_id: str, *, end_time: str, total_cost_eur: float
) -> None:
    """Write ``end_time`` and ``total_cost_eur`` once a run has completed."""
    con = connect(db_path)
    try:
        con.execute(
            "UPDATE experiments SET end_time = ?, total_cost_eur = ? WHERE id = ?",
            (end_time, total_cost_eur, experiment_id),
        )
        con.commit()
    finally:
        con.close()


def insert_agent(db_path: str | Path, record: AgentRecord) -> None:
    """Insert one participating agent for an experiment."""
    con = connect(db_path)
    try:
        con.execute(
            "INSERT INTO agents (experiment_id, agent_id, model_name, capability_tier) "
            "VALUES (?, ?, ?, ?)",
            (
                record.experiment_id,
                record.agent_id,
                record.model_name,
                record.capability_tier,
            ),
        )
        con.commit()
    finally:
        con.close()


def insert_round(db_path: str | Path, experiment_id: str, round_num: int) -> None:
    """Record that a round started."""
    con = connect(db_path)
    try:
        con.execute(
            "INSERT INTO rounds (experiment_id, round_num) VALUES (?, ?)",
            (experiment_id, round_num),
        )
        con.commit()
    finally:
        con.close()


def insert_action(db_path: str | Path, record: ActionRecord) -> None:
    """Insert one agent's action for one round."""
    con = connect(db_path)
    try:
        con.execute(
            "INSERT INTO actions ("
            "experiment_id, round_num, agent_id, action_json, message, "
            "retries, cost_eur, fallback_used"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                record.experiment_id,
                record.round_num,
                record.agent_id,
                json.dumps(record.action_json),
                record.message,
                record.retries,
                record.cost_eur,
                int(record.fallback_used),
            ),
        )
        con.commit()
    finally:
        con.close()


def insert_payoff(db_path: str | Path, record: PayoffRecord) -> None:
    """Insert one agent's round and cumulative payoff for one round."""
    con = connect(db_path)
    try:
        con.execute(
            "INSERT INTO payoffs ("
            "experiment_id, round_num, agent_id, round_payoff, cumulative_payoff"
            ") VALUES (?, ?, ?, ?, ?)",
            (
                record.experiment_id,
                record.round_num,
                record.agent_id,
                record.round_payoff,
                record.cumulative_payoff,
            ),
        )
        con.commit()
    finally:
        con.close()


# ---------------------------------------------------------------------------
# ZAV-25 message classification (one row per classified agent message).
# ---------------------------------------------------------------------------


@dataclass
class ClassificationRecord:
    """One row for the ``message_classifications`` table.

    ``experiment_id``/``round_num`` may be ``None`` for ad-hoc classification
    (e.g. the standalone accuracy eval) that is not tied to a stored game round.
    """

    agent_id: str
    message: str
    label: str
    cost_eur: float = 0.0
    fallback_used: bool = False
    experiment_id: str | None = None
    round_num: int | None = None


def insert_classification(db_path: str | Path, record: ClassificationRecord) -> None:
    """Upsert one message classification.

    Uses ``INSERT OR REPLACE`` on the ``(experiment_id, round_num, agent_id)``
    unique key so re-classifying a results database is idempotent rather than
    raising or duplicating rows.
    """
    con = connect(db_path)
    try:
        con.execute(
            "INSERT OR REPLACE INTO message_classifications ("
            "ts, experiment_id, round_num, agent_id, message, label, "
            "cost_eur, fallback_used"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                record.experiment_id,
                record.round_num,
                record.agent_id,
                record.message,
                record.label,
                record.cost_eur,
                int(record.fallback_used),
            ),
        )
        con.commit()
    finally:
        con.close()


# ---------------------------------------------------------------------------
# ZAV-22 KL deception scores (one row per agent per game per KL direction).
# ---------------------------------------------------------------------------


@dataclass
class DeceptionRecord:
    """One row for the ``deception_scores`` table.

    ``kl_headline``/``surprisal`` may be ``None`` (e.g. an agent with no scored
    rounds) and are stored as SQL NULL.
    """

    experiment_id: str
    agent_id: str
    kl_headline: float | None
    surprisal: float | None
    direction: str
    n_rounds: int
    cost_eur: float = 0.0


def insert_deception_score(db_path: str | Path, record: DeceptionRecord) -> None:
    """Upsert one agent's deception scores for a game.

    Uses ``INSERT OR REPLACE`` on ``(experiment_id, agent_id, direction)`` so
    re-scoring a database is idempotent. A NaN score is stored as NULL.
    """

    def _clean(value: float | None) -> float | None:
        if value is None or (isinstance(value, float) and value != value):
            return None
        return float(value)

    con = connect(db_path)
    try:
        con.execute(
            "INSERT OR REPLACE INTO deception_scores ("
            "ts, experiment_id, agent_id, kl_headline, surprisal, "
            "direction, n_rounds, cost_eur"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(),
                record.experiment_id,
                record.agent_id,
                _clean(record.kl_headline),
                _clean(record.surprisal),
                record.direction,
                record.n_rounds,
                record.cost_eur,
            ),
        )
        con.commit()
    finally:
        con.close()
