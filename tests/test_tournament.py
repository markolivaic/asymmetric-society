"""Runner end-to-end with mock agents: no API calls, no real cost.

``create_agent`` is monkeypatched to return ``FixedAgent`` (a ``BaseAgent`` whose
``_complete`` returns scripted JSON), so the whole orchestration loop and SQLite
logging are exercised against a real ``PublicGoodsGame`` without any network.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from asymmetric_society.agents.base import BaseAgent, CompletionResult
from asymmetric_society.config import AgentConfig, ExperimentConfig, GameConfig
from asymmetric_society.runner.budget import BudgetExceededError, BudgetTracker
from asymmetric_society.runner.tournament import run_experiment

CONTRIB = {"aa000001": 0, "aa000002": 5, "aa000003": 10, "aa000004": 15}


class FixedAgent(BaseAgent):
    """Backend that always replies with a fixed contribution (and maybe a message)."""

    def __init__(
        self,
        contribution: int,
        *,
        message: str | None = None,
        raw_override: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._contribution = contribution
        self._message = message
        self._raw_override = raw_override
        self.calls = 0

    def _complete(self, prompt: str, *, json_mode: bool) -> CompletionResult:
        self.calls += 1
        if self._raw_override is not None:
            raw = self._raw_override
        else:
            payload: dict = {"contribution": self._contribution}
            if self._message is not None:
                payload["message"] = self._message
            raw = json.dumps(payload)
        return CompletionResult(raw_text=raw, input_tokens=10, output_tokens=5, cost_eur=0.0)


def _config(db_path: str, *, rounds: int = 3, communication: bool = False) -> ExperimentConfig:
    return ExperimentConfig(
        name="test-pgg",
        seed=12345,
        rounds=rounds,
        db_path=db_path,
        game=GameConfig(
            name="public_goods",
            params={"n": 4, "endowment": 20, "multiplier": 1.6,
                    "rounds": rounds, "communication": communication},
        ),
        agents=[
            AgentConfig(backend="ollama", model_name="llama3.2:3b",
                        capability_tier="weak", agent_id=aid)
            for aid in CONTRIB
        ],
    )


def _patch_agents(monkeypatch: pytest.MonkeyPatch, *, raw_override: str | None = None,
                  message: str | None = None) -> None:
    def _factory(config: AgentConfig, *, budget: BudgetTracker, db_path: str) -> FixedAgent:
        return FixedAgent(
            CONTRIB[config.agent_id],
            message=message,
            raw_override=raw_override,
            agent_id=config.agent_id,
            model_name=config.model_name,
            capability_tier=config.capability_tier,
            budget=budget,
            db_path=db_path,
            temperature=config.temperature,
        )

    monkeypatch.setattr("asymmetric_society.runner.tournament.create_agent", _factory)


def _query(db_path: str, sql: str) -> list:
    con = sqlite3.connect(db_path)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


async def test_run_populates_all_tables(monkeypatch: pytest.MonkeyPatch, db_path: str) -> None:
    _patch_agents(monkeypatch)
    exp_id = await run_experiment(_config(db_path, rounds=3))

    (exp_row,) = _query(
        db_path, "SELECT id, seed, end_time, total_cost_eur FROM experiments"
    )
    assert exp_row[0] == exp_id
    assert exp_row[1] == 12345
    assert exp_row[2] is not None  # finalized
    assert exp_row[3] == 0.0  # mock agents cost nothing

    assert _query(db_path, "SELECT COUNT(*) FROM agents")[0][0] == 4
    assert _query(db_path, "SELECT COUNT(*) FROM rounds")[0][0] == 3
    assert _query(db_path, "SELECT COUNT(*) FROM actions")[0][0] == 12  # 4 agents * 3 rounds
    assert _query(db_path, "SELECT COUNT(*) FROM payoffs")[0][0] == 12
    # The agent layer logged exactly one llm_calls row per act() call -> 1:1 join.
    assert _query(db_path, "SELECT COUNT(*) FROM llm_calls")[0][0] == 12


async def test_payoffs_match_pgg_math(monkeypatch: pytest.MonkeyPatch, db_path: str) -> None:
    _patch_agents(monkeypatch)
    exp_id = await run_experiment(_config(db_path, rounds=3))

    # contributions 0/5/10/15 -> total 30, pool 48, share 12.
    # per round: a=12+20=32, b=12+15=27, c=12+10=22, d=12+5=17. Over 3 rounds x3.
    expected_final = {"aa000001": 96.0, "aa000002": 81.0,
                      "aa000003": 66.0, "aa000004": 51.0}
    rows = _query(
        db_path,
        f"SELECT agent_id, cumulative_payoff FROM payoffs "
        f"WHERE experiment_id='{exp_id}' AND round_num=3",
    )
    result = dict(rows)
    for agent_id, expected in expected_final.items():
        assert result[agent_id] == pytest.approx(expected)


async def test_actions_record_action_and_no_fallback(
    monkeypatch: pytest.MonkeyPatch, db_path: str
) -> None:
    _patch_agents(monkeypatch)
    await run_experiment(_config(db_path, rounds=2))
    rows = _query(db_path, "SELECT action_json, retries, fallback_used FROM actions")
    for action_json, retries, fallback in rows:
        assert "contribution" in json.loads(action_json)
        assert retries == 0
        assert fallback == 0


async def test_invalid_responses_fall_back_to_zero(
    monkeypatch: pytest.MonkeyPatch, db_path: str
) -> None:
    _patch_agents(monkeypatch, raw_override="not valid json")
    await run_experiment(_config(db_path, rounds=1))
    rows = _query(db_path, "SELECT action_json, retries, fallback_used FROM actions")
    assert len(rows) == 4
    for action_json, retries, fallback in rows:
        assert json.loads(action_json) == {"contribution": 0}
        assert fallback == 1
        assert retries == 4  # max_parse_retries (3) + 1 initial attempt


async def test_determinism_same_seed_identical_prompts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_agents(monkeypatch)
    db_a, db_b = str(tmp_path / "a.db"), str(tmp_path / "b.db")
    await run_experiment(_config(db_a, rounds=3))
    await run_experiment(_config(db_b, rounds=3))

    # Order by (round_no, agent_id), not insertion id: concurrent act() threads
    # finish in a nondeterministic order, but the prompt *content* per agent per
    # round is deterministic given the seed.
    order = "ORDER BY round_no, agent_id"
    prompts_a = _query(db_a, f"SELECT round_no, agent_id, prompt FROM llm_calls {order}")
    prompts_b = _query(db_b, f"SELECT round_no, agent_id, prompt FROM llm_calls {order}")
    # Same seed -> identical anonymized history ordering -> identical prompts.
    assert prompts_a == prompts_b


async def test_communication_message_logged(
    monkeypatch: pytest.MonkeyPatch, db_path: str
) -> None:
    _patch_agents(monkeypatch, message="let's cooperate")
    await run_experiment(_config(db_path, rounds=1, communication=True))
    messages = _query(db_path, "SELECT message FROM actions")
    assert all(m == "let's cooperate" for (m,) in messages)


async def test_budget_limit_tripwire_halts_run(
    monkeypatch: pytest.MonkeyPatch, db_path: str
) -> None:
    # A zero cap must abort before the first call (spent 0.0 >= limit 0.0).
    _patch_agents(monkeypatch)
    with pytest.raises(BudgetExceededError):
        await run_experiment(_config(db_path, rounds=1), budget_limit_eur=0.0)
