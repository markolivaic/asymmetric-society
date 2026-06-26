"""Results-reader SQL on hand-inserted rows (pure SQLite, no Ollama)."""

from __future__ import annotations

import pytest

from asymmetric_society.runner import db, results


def _call(db_path: str, agent_id: str, *, parse_fail: int, parse_ok: bool,
          fallback: bool, raw: str, exp: str = "e1", rnd: int = 1) -> None:
    db.log_call(
        db_path,
        db.CallRecord(
            agent_id=agent_id, model_name="llama3.2:3b", capability_tier="weak",
            prompt="p", raw_response=raw, parsed_action={"contribution": 5} if parse_ok else None,
            api_error_count=0, parse_fail_count=parse_fail,
            input_tokens=0, output_tokens=0, cost_eur=0.0,
            parse_ok=parse_ok, fallback_used=fallback, experiment_id=exp, round_no=rnd,
        ),
    )


def test_parse_failure_stats_rates(db_path: str) -> None:
    # w1: 1 bad attempt then ok (attempts 2, fails 1). w2: 3 bad, gave up (attempts 3, fails 3).
    _call(db_path, "w1", parse_fail=1, parse_ok=True, fallback=False, raw='{"contribution":5}')
    _call(db_path, "w2", parse_fail=3, parse_ok=False, fallback=True, raw="garbage")

    (row,) = results.parse_failure_stats(db_path, ["e1"])
    assert row["capability_tier"] == "weak"
    assert row["calls"] == 2
    assert row["fallback_rate"] == pytest.approx(0.5)  # 1 of 2 fell back
    assert row["any_parse_fail_rate"] == pytest.approx(1.0)  # both had >=1 bad attempt
    # attempt_fail_rate = (1+3) / ((1+1) + (3+0)) = 4/5
    assert row["attempt_fail_rate"] == pytest.approx(0.8)


def test_parse_failure_examples(db_path: str) -> None:
    _call(db_path, "w1", parse_fail=0, parse_ok=True, fallback=False, raw='{"contribution":5}')
    _call(db_path, "w2", parse_fail=2, parse_ok=True, fallback=False, raw="```json {}```")
    examples = results.parse_failure_examples(db_path, ["e1"])
    assert len(examples) == 1  # only the one with parse_fail_count > 0
    assert examples[0]["agent_id"] == "w2"
    assert "json" in examples[0]["raw_response"]


def test_empty_inputs_return_empty(db_path: str) -> None:
    assert results.parse_failure_stats(db_path, []) == []
    assert results.final_payoffs(db_path, []) == []
    assert results.parse_failure_examples(db_path, []) == []


def test_final_payoffs_uses_last_round(db_path: str) -> None:
    db.insert_agent(db_path, db.AgentRecord("e1", "w1", "llama3.2:3b", "weak"))
    for rnd, cum in [(1, 10.0), (2, 25.0)]:
        db.insert_payoff(db_path, db.PayoffRecord("e1", rnd, "w1", 5.0, cum))

    (row,) = results.final_payoffs(db_path, ["e1"])
    assert row["agent_id"] == "w1"
    assert row["capability_tier"] == "weak"
    assert row["cumulative_payoff"] == pytest.approx(25.0)  # last round only


def test_game_transcript_parses_contribution(db_path: str) -> None:
    db.insert_agent(db_path, db.AgentRecord("e1", "w1", "llama3.2:3b", "weak"))
    db.insert_action(
        db_path,
        db.ActionRecord("e1", 1, "w1", {"contribution": 7}, "hi", 0, 0.0, False),
    )
    db.insert_payoff(db_path, db.PayoffRecord("e1", 1, "w1", 12.0, 12.0))

    (row,) = results.game_transcript(db_path, "e1")
    assert row["contribution"] == 7
    assert row["message"] == "hi"
    assert row["round_payoff"] == pytest.approx(12.0)
    assert "action_json" not in row  # popped after parsing
