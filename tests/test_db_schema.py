"""Experiment tables: existence and insert/round-trip (pure SQLite)."""

from __future__ import annotations

import json
import sqlite3

from asymmetric_society.runner import db

NEW_TABLES = {"experiments", "agents", "rounds", "actions", "payoffs"}


def _tables(db_path: str) -> set[str]:
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    finally:
        con.close()
    return {r[0] for r in rows}


def test_init_db_creates_experiment_tables(db_path: str) -> None:
    # llm_calls plus the five experiment tables all live in one init_db.
    assert NEW_TABLES.issubset(_tables(db_path))
    assert "llm_calls" in _tables(db_path)


def test_experiment_round_trip(db_path: str) -> None:
    db.insert_experiment(
        db_path,
        db.ExperimentRecord(id="exp1", config_json='{"name": "x"}', seed=7, start_time="t0"),
    )
    db.finalize_experiment(db_path, "exp1", end_time="t1", total_cost_eur=1.23)

    con = sqlite3.connect(db_path)
    try:
        row = con.execute(
            "SELECT id, seed, start_time, end_time, total_cost_eur FROM experiments"
        ).fetchone()
    finally:
        con.close()
    assert row == ("exp1", 7, "t0", "t1", 1.23)


def test_agent_round_round_trip(db_path: str) -> None:
    db.insert_agent(db_path, db.AgentRecord("exp1", "ag1", "llama3.2:3b", "weak"))
    db.insert_round(db_path, "exp1", 1)

    con = sqlite3.connect(db_path)
    try:
        assert con.execute("SELECT model_name FROM agents").fetchone()[0] == "llama3.2:3b"
        assert con.execute("SELECT round_num FROM rounds").fetchone()[0] == 1
    finally:
        con.close()


def test_action_and_payoff_round_trip(db_path: str) -> None:
    db.insert_action(
        db_path,
        db.ActionRecord(
            experiment_id="exp1",
            round_num=1,
            agent_id="ag1",
            action_json={"contribution": 5},
            message="cooperate",
            retries=2,
            cost_eur=0.0,
            fallback_used=False,
        ),
    )
    db.insert_payoff(
        db_path,
        db.PayoffRecord(
            experiment_id="exp1",
            round_num=1,
            agent_id="ag1",
            round_payoff=12.0,
            cumulative_payoff=12.0,
        ),
    )

    con = sqlite3.connect(db_path)
    try:
        action_json, message, retries, fallback = con.execute(
            "SELECT action_json, message, retries, fallback_used FROM actions"
        ).fetchone()
        payoff = con.execute("SELECT round_payoff FROM payoffs").fetchone()[0]
    finally:
        con.close()
    assert json.loads(action_json) == {"contribution": 5}
    assert message == "cooperate"
    assert retries == 2
    assert fallback == 0
    assert payoff == 12.0
