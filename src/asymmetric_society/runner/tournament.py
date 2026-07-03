"""Experiment orchestration: run one game end-to-end and log it to SQLite.

``run_experiment`` wires a :class:`~asymmetric_society.config.ExperimentConfig`
into a game and its agents, then drives the
``render_prompt -> agent.act -> step`` loop until the game reports ``done``,
recording the run across the ``experiments / agents / rounds / actions / payoffs``
tables (the per-call prompt/response log lives separately in ``llm_calls``).

Concurrency: ``act()`` is synchronous (see the Week-1 decision log). Each round
fans out the agents' calls with ``asyncio.to_thread(agent.act)`` + ``gather`` so
the paid API calls overlap with the serial Ollama queue. ``gather`` preserves
input order and ``step`` aggregates by ``agent_id``, so concurrency never changes
the outcome.

Reproducibility: same config + seed reproduces the same game setup, agent
ordering, and randomization (not bit-identical LLM text). The loop is driven by
the game's own round count; ``ExperimentConfig`` validates that its ``rounds``
agrees with ``game.params``.
"""

from __future__ import annotations

import asyncio
import random
import uuid
from datetime import datetime, timezone

from asymmetric_society.agents.factory import create_agent
from asymmetric_society.config import ExperimentConfig
from asymmetric_society.games.factory import create_game
from asymmetric_society.runner import db
from asymmetric_society.runner.budget import BudgetTracker


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def run_experiment(config: ExperimentConfig, *, budget_limit_eur: float | None = None) -> str:
    """Run one game end-to-end and persist every round to SQLite.

    Args:
        config: Fully validated experiment specification.
        budget_limit_eur: Optional spend cap for this run, overriding the default
            €100 hard cap. Use a low value (e.g. 5.0) as a tripwire on paid runs.

    Returns:
        The generated ``experiment_id`` (hex), the primary key tying every
        logged row of this run together.

    Raises:
        BudgetExceededError: If the cap is reached mid-run (propagates from
            the agent layer's budget check).
    """
    db.init_db(config.db_path)
    if budget_limit_eur is None:
        budget = BudgetTracker(config.db_path)
    else:
        budget = BudgetTracker(config.db_path, limit_eur=budget_limit_eur)
    rng = random.Random(config.seed)
    game = create_game(config.game, rng=rng)
    agents = {
        agent_cfg.agent_id: create_agent(agent_cfg, budget=budget, db_path=config.db_path)
        for agent_cfg in config.agents
    }

    experiment_id = uuid.uuid4().hex
    db.insert_experiment(
        config.db_path,
        db.ExperimentRecord(
            id=experiment_id,
            config_json=config.model_dump_json(),
            seed=config.seed,
            start_time=_now(),
        ),
    )
    for agent_cfg in config.agents:
        db.insert_agent(
            config.db_path,
            db.AgentRecord(
                experiment_id=experiment_id,
                agent_id=agent_cfg.agent_id,
                model_name=agent_cfg.model_name,
                capability_tier=agent_cfg.capability_tier,
            ),
        )

    agent_ids = list(agents)
    history: list[dict] = []
    cumulative: dict[str, float] = {agent_id: 0.0 for agent_id in agent_ids}
    total_cost = 0.0

    state = game.reset()
    done = False
    while not done:
        round_num = state["round"]
        db.insert_round(config.db_path, experiment_id, round_num)

        prompts = {aid: game.render_prompt(aid, state, history) for aid in agent_ids}
        validators = {aid: game.validator(aid, state) for aid in agent_ids}
        fallbacks = {aid: game.fallback_action(aid, state) for aid in agent_ids}

        results = await asyncio.gather(
            *(
                asyncio.to_thread(
                    agents[aid].act,
                    prompts[aid],
                    validators[aid],
                    fallbacks[aid],
                    experiment_id=experiment_id,
                    round_no=round_num,
                )
                for aid in agent_ids
            )
        )
        result_by_id = dict(zip(agent_ids, results))
        actions = {aid: result.action for aid, result in result_by_id.items()}

        for aid in agent_ids:
            result = result_by_id[aid]
            total_cost += result.cost_eur
            db.insert_action(
                config.db_path,
                db.ActionRecord(
                    experiment_id=experiment_id,
                    round_num=round_num,
                    agent_id=aid,
                    action_json=result.action,
                    message=result.message,
                    retries=result.parse_fail_count + result.api_error_count,
                    cost_eur=result.cost_eur,
                    fallback_used=result.fallback_used,
                ),
            )

        history.append({"round": round_num, "actions": actions})

        state, payoffs, done = game.step(actions)

        for aid in agent_ids:
            cumulative[aid] += payoffs[aid]
            db.insert_payoff(
                config.db_path,
                db.PayoffRecord(
                    experiment_id=experiment_id,
                    round_num=round_num,
                    agent_id=aid,
                    round_payoff=payoffs[aid],
                    cumulative_payoff=cumulative[aid],
                ),
            )

    db.finalize_experiment(
        config.db_path,
        experiment_id,
        end_time=_now(),
        total_cost_eur=total_cost,
    )
    return experiment_id
