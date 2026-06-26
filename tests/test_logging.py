"""SQLite call logging: rows persist with both failure counters intact."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from asymmetric_society.runner.db import CallRecord, init_db, log_call


def test_log_call_round_trips(tmp_path: Path) -> None:
    path = tmp_path / "log.db"
    init_db(path)
    row_id = log_call(
        path,
        CallRecord(
            agent_id="abc12345",
            model_name="claude-haiku-4-5-20251001",
            capability_tier="strong",
            prompt="the prompt",
            raw_response='{"contribution": 4}',
            parsed_action={"contribution": 4},
            api_error_count=1,
            parse_fail_count=2,
            input_tokens=120,
            output_tokens=30,
            cost_eur=0.0042,
            parse_ok=True,
            fallback_used=False,
            experiment_id="exp1",
            round_no=3,
        ),
    )
    assert row_id == 1

    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    row = con.execute("SELECT * FROM llm_calls WHERE id = ?", (row_id,)).fetchone()
    con.close()

    assert row["agent_id"] == "abc12345"
    assert row["api_error_count"] == 1
    assert row["parse_fail_count"] == 2
    assert row["parse_ok"] == 1
    assert row["fallback_used"] == 0
    assert row["round_no"] == 3
    assert json.loads(row["parsed_action"]) == {"contribution": 4}


def test_init_db_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "idem.db"
    init_db(path)
    init_db(path)  # second call must not error or wipe data
    log_call(
        path,
        CallRecord(
            agent_id="a",
            model_name="m",
            capability_tier="weak",
            prompt="p",
            raw_response=None,
            parsed_action=None,
            api_error_count=0,
            parse_fail_count=0,
            input_tokens=0,
            output_tokens=0,
            cost_eur=0.0,
            parse_ok=False,
            fallback_used=True,
        ),
    )
    con = sqlite3.connect(str(path))
    count = con.execute("SELECT COUNT(*) FROM llm_calls").fetchone()[0]
    con.close()
    assert count == 1
