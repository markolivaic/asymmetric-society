"""TrustGame through the real runner with a scripted mock backend (no network).

A two-phase sequential game must still drive the simultaneous runner correctly and
log clean rows. The mock reads the rendered prompt to decide what to emit (send /
return / no-op), exactly as a real model would, so the whole orchestration loop and
SQLite logging are exercised without any API calls. The key new invariant: the
off-role no-op turns must NOT count toward the parse-failure gate.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from asymmetric_society.agents.base import BaseAgent, CompletionResult
from asymmetric_society.config import AgentConfig, ExperimentConfig, GameConfig
from asymmetric_society.runner.budget import BudgetTracker
from asymmetric_society.runner.results import parse_failure_stats
from asymmetric_society.runner.tournament import run_experiment

INVESTOR = "10000001"  # first in config order -> investor under fixed roles
TRUSTEE = "20000002"


class TrustMockAgent(BaseAgent):
    """Backend that answers by reading the prompt: send / return / no-op."""

    def __init__(self, *, send: int, ret: int, **kwargs: object) -> None:
        super().__init__(**kwargs)  # type: ignore[arg-type]
        self._send = send
        self._ret = ret
        self.calls = 0

    def _complete(self, prompt: str, *, json_mode: bool) -> CompletionResult:
        self.calls += 1
        if '"send"' in prompt:
            raw = json.dumps({"send": self._send})
        elif '"return"' in prompt:
            raw = json.dumps({"return": self._ret})
        else:
            raw = json.dumps({"noop": True})
        return CompletionResult(raw_text=raw, input_tokens=10, output_tokens=5, cost_eur=0.0)


def _config(db_path: str, *, rounds: int = 2, roles: str = "fixed",
            communication: bool = False) -> ExperimentConfig:
    return ExperimentConfig(
        name="test-trust",
        seed=12345,
        rounds=rounds,
        db_path=db_path,
        game=GameConfig(
            name="trust_game",
            params={"endowment": 10, "multiplier": 3, "rounds": rounds,
                    "roles": roles, "communication": communication},
        ),
        agents=[
            AgentConfig(backend="ollama", model_name="llama3.2:3b",
                        capability_tier="weak", agent_id=INVESTOR),
            AgentConfig(backend="ollama", model_name="llama3.2:3b",
                        capability_tier="weak", agent_id=TRUSTEE),
        ],
    )


def _patch_agents(monkeypatch: pytest.MonkeyPatch, *, send: int, ret: int) -> None:
    def _factory(config: AgentConfig, *, budget: BudgetTracker, db_path: str) -> TrustMockAgent:
        return TrustMockAgent(
            send=send, ret=ret,
            agent_id=config.agent_id, model_name=config.model_name,
            capability_tier=config.capability_tier, budget=budget,
            db_path=db_path, temperature=config.temperature,
        )

    monkeypatch.setattr("asymmetric_society.runner.tournament.create_agent", _factory)


def _query(db_path: str, sql: str) -> list:
    con = sqlite3.connect(db_path)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


async def test_two_phase_run_populates_all_tables(
    monkeypatch: pytest.MonkeyPatch, db_path: str
) -> None:
    _patch_agents(monkeypatch, send=5, ret=3)
    await run_experiment(_config(db_path, rounds=2))
    # 2 Berg rounds -> 4 runner rounds; 2 agents each -> 8 action/payoff/call rows.
    assert _query(db_path, "SELECT COUNT(*) FROM rounds")[0][0] == 4
    assert _query(db_path, "SELECT COUNT(*) FROM actions")[0][0] == 8
    assert _query(db_path, "SELECT COUNT(*) FROM payoffs")[0][0] == 8
    assert _query(db_path, "SELECT COUNT(*) FROM llm_calls")[0][0] == 8


async def test_fixed_role_cumulative_payoffs(
    monkeypatch: pytest.MonkeyPatch, db_path: str
) -> None:
    _patch_agents(monkeypatch, send=5, ret=3)
    exp_id = await run_experiment(_config(db_path, rounds=2))
    # Per Berg round: investor 10-5+3=8, trustee 15-3=12. Over 2 rounds: 16 / 24.
    rows = _query(
        db_path,
        f"SELECT agent_id, cumulative_payoff FROM payoffs "
        f"WHERE experiment_id='{exp_id}' AND round_num=4",
    )
    final = dict(rows)
    assert final[INVESTOR] == pytest.approx(16.0)
    assert final[TRUSTEE] == pytest.approx(24.0)


async def test_noop_turns_excluded_from_parse_gate(
    monkeypatch: pytest.MonkeyPatch, db_path: str
) -> None:
    _patch_agents(monkeypatch, send=5, ret=3)
    exp_id = await run_experiment(_config(db_path, rounds=2))
    # 8 total calls, but 4 are off-role no-ops. The gate must see only the 4
    # real decisions, otherwise the no-op turns pad the parse-success count.
    stats = parse_failure_stats(db_path, [exp_id])
    assert len(stats) == 1  # single "weak" tier
    assert stats[0]["calls"] == 4
    assert stats[0]["attempt_fail_rate"] == pytest.approx(0.0)


async def test_actions_log_decisions_and_noops(
    monkeypatch: pytest.MonkeyPatch, db_path: str
) -> None:
    _patch_agents(monkeypatch, send=5, ret=3)
    exp_id = await run_experiment(_config(db_path, rounds=2))
    actions = [
        json.loads(a)
        for (a,) in _query(
            db_path, f"SELECT action_json FROM actions WHERE experiment_id='{exp_id}'"
        )
    ]
    assert sum("send" in a for a in actions) == 2  # one invest decision per Berg round
    assert sum("return" in a for a in actions) == 2  # one return decision per Berg round
    assert sum("noop" in a for a in actions) == 4  # one idle turn per phase
