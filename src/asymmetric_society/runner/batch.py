"""Run a config for N games sequentially (ZAV-20 pilot driver).

Ollama serves one model instance and serializes concurrent requests, so there is
nothing to gain from running games in parallel - they run one after another. Each
game gets a distinct derived seed so the N games differ from one another while the
whole batch stays reproducible (same base seed -> same N seeds -> same game setup,
agent ordering, and randomization; LLM outputs are not bit-reproducible).
"""

from __future__ import annotations

import time
from typing import Callable

from asymmetric_society.agents.base import AgentAPIError
from asymmetric_society.config import ExperimentConfig
from asymmetric_society.runner.tournament import run_experiment

#: Called after each game with (game_index, experiment_id, elapsed_seconds).
ProgressCallback = Callable[[int, str, float], None]
#: Called when a game aborts on a transport failure, with (game_index, error).
FailureCallback = Callable[[int, str], None]


async def run_batch(
    config: ExperimentConfig,
    n_games: int,
    *,
    base_seed: int | None = None,
    budget_limit_eur: float | None = None,
    on_game_complete: ProgressCallback | None = None,
    on_game_failed: FailureCallback | None = None,
) -> list[str]:
    """Run ``n_games`` independent games and return the successful experiment ids.

    A single game that aborts on a transport failure (:class:`AgentAPIError` -
    e.g. a momentary network/DNS outage during an overnight run) is logged and
    skipped so the rest of the batch still completes; its partial rows stay in
    the DB but are excluded from analysis because only successful ids are
    returned. :class:`BudgetExceededError` is deliberately NOT caught - once the
    spend cap is hit the whole batch must stop rather than keep spending.

    Args:
        config: Base experiment configuration; reused for every game with only
            the seed and name changed.
        n_games: Number of games to play (>= 1).
        base_seed: Seed for the first game; game ``i`` uses ``base_seed + i``.
            Defaults to ``config.seed``.
        budget_limit_eur: Optional spend cap applied to every game (a tripwire on
            paid runs). The cap is cumulative across games via the shared DB.
        on_game_complete: Optional callback invoked after each successful game
            with the game index, its experiment id, and wall-clock seconds.
        on_game_failed: Optional callback invoked when a game aborts on a
            transport failure, with the game index and the error message.

    Returns:
        The successful experiment ids, in play order.

    Raises:
        ValueError: If ``n_games < 1``.
        BudgetExceededError: If the spend cap is reached (propagates to stop the
            whole batch).
    """
    if n_games < 1:
        raise ValueError("n_games must be >= 1")

    base = config.seed if base_seed is None else base_seed
    experiment_ids: list[str] = []
    for i in range(n_games):
        cfg_i = config.model_copy(update={"seed": base + i, "name": f"{config.name}-g{i:02d}"})
        start = time.perf_counter()
        try:
            experiment_id = await run_experiment(cfg_i, budget_limit_eur=budget_limit_eur)
        except AgentAPIError as exc:
            if on_game_failed is not None:
                on_game_failed(i, str(exc))
            continue
        elapsed = time.perf_counter() - start
        experiment_ids.append(experiment_id)
        if on_game_complete is not None:
            on_game_complete(i, experiment_id, elapsed)
    return experiment_ids
