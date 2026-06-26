"""Batch seed derivation (mock-only: run_experiment is stubbed, no Ollama)."""

from __future__ import annotations

import pytest

from asymmetric_society.config import AgentConfig, ExperimentConfig, GameConfig
from asymmetric_society.runner import batch


def _config() -> ExperimentConfig:
    return ExperimentConfig(
        name="b",
        seed=100,
        rounds=2,
        game=GameConfig(
            name="public_goods",
            params={"n": 4, "endowment": 20, "multiplier": 1.6, "rounds": 2,
                    "communication": False},
        ),
        agents=[
            AgentConfig(backend="ollama", model_name="llama3.2:3b",
                        capability_tier="weak", agent_id=f"a000000{i}")
            for i in range(4)
        ],
    )


async def test_run_batch_derives_sequential_seeds(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[int, str]] = []

    async def fake_run(cfg: ExperimentConfig, **_kw: object) -> str:
        seen.append((cfg.seed, cfg.name))
        return f"exp{cfg.seed}"

    monkeypatch.setattr("asymmetric_society.runner.batch.run_experiment", fake_run)
    ids = await batch.run_batch(_config(), 3)

    assert [seed for seed, _ in seen] == [100, 101, 102]
    assert len({name for _, name in seen}) == 3  # unique per-game names
    assert ids == ["exp100", "exp101", "exp102"]


async def test_run_batch_base_seed_override(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[int] = []

    async def fake_run(cfg: ExperimentConfig, **_kw: object) -> str:
        seen.append(cfg.seed)
        return "x"

    monkeypatch.setattr("asymmetric_society.runner.batch.run_experiment", fake_run)
    await batch.run_batch(_config(), 2, base_seed=500)
    assert seen == [500, 501]


async def test_run_batch_progress_callback(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_run(cfg: ExperimentConfig, **_kw: object) -> str:
        return f"exp{cfg.seed}"

    monkeypatch.setattr("asymmetric_society.runner.batch.run_experiment", fake_run)
    calls: list[tuple[int, str]] = []
    await batch.run_batch(_config(), 2, on_game_complete=lambda i, eid, el: calls.append((i, eid)))
    assert calls == [(0, "exp100"), (1, "exp101")]


async def test_run_batch_rejects_zero_games() -> None:
    with pytest.raises(ValueError):
        await batch.run_batch(_config(), 0)


async def test_run_batch_continues_past_api_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    from asymmetric_society.agents.base import AgentAPIError

    attempted: list[int] = []

    async def fake_run(cfg: ExperimentConfig, **_kw: object) -> str:
        attempted.append(cfg.seed)
        if cfg.seed == 101:  # a transient blip on the second game
            raise AgentAPIError("getaddrinfo failed")
        return f"exp{cfg.seed}"

    monkeypatch.setattr("asymmetric_society.runner.batch.run_experiment", fake_run)
    failed: list[int] = []
    ids = await batch.run_batch(_config(), 3, on_game_failed=lambda i, e: failed.append(i))

    assert attempted == [100, 101, 102]  # all three attempted
    assert ids == ["exp100", "exp102"]  # the failed game is skipped, not fatal
    assert failed == [1]


async def test_run_batch_propagates_budget_error(monkeypatch: pytest.MonkeyPatch) -> None:
    # Budget exhaustion must stop the whole batch, not be swallowed like a blip.
    from asymmetric_society.runner.budget import BudgetExceededError

    async def fake_run(cfg: ExperimentConfig, **_kw: object) -> str:
        raise BudgetExceededError("cap reached")

    monkeypatch.setattr("asymmetric_society.runner.batch.run_experiment", fake_run)
    with pytest.raises(BudgetExceededError):
        await batch.run_batch(_config(), 3)
