"""Experiment orchestration: budget enforcement and SQLite call logging."""

from asymmetric_society.runner.budget import (
    BUDGET_LIMIT_EUR,
    USD_TO_EUR,
    BudgetExceededError,
    BudgetTracker,
)
from asymmetric_society.runner.db import (
    ActionRecord,
    AgentRecord,
    CallRecord,
    ExperimentRecord,
    PayoffRecord,
    finalize_experiment,
    init_db,
    insert_action,
    insert_agent,
    insert_experiment,
    insert_payoff,
    insert_round,
    log_call,
)

# NOTE: ``tournament`` is intentionally NOT imported here. It pulls in the agent
# factory (and thus ``agents.base``), which imports back from this package; an
# eager import would create a partially-initialized circular import. Import it
# directly: ``from asymmetric_society.runner.tournament import run_experiment``.

__all__ = [
    "BUDGET_LIMIT_EUR",
    "USD_TO_EUR",
    "ActionRecord",
    "AgentRecord",
    "BudgetExceededError",
    "BudgetTracker",
    "CallRecord",
    "ExperimentRecord",
    "PayoffRecord",
    "finalize_experiment",
    "init_db",
    "insert_action",
    "insert_agent",
    "insert_experiment",
    "insert_payoff",
    "insert_round",
    "log_call",
]
