"""Hard budget cap enforcement and restart-safe spend reloading."""

from __future__ import annotations

from pathlib import Path

import pytest

from asymmetric_society.runner.budget import BudgetExceededError, BudgetTracker
from asymmetric_society.runner.db import CallRecord, init_db, log_call


def _record(cost_eur: float) -> CallRecord:
    return CallRecord(
        agent_id="a1",
        model_name="m",
        capability_tier="strong",
        prompt="p",
        raw_response="r",
        parsed_action={"x": 1},
        api_error_count=0,
        parse_fail_count=0,
        input_tokens=1,
        output_tokens=1,
        cost_eur=cost_eur,
        parse_ok=True,
        fallback_used=False,
    )


def test_check_passes_under_cap(db_path: str) -> None:
    tracker = BudgetTracker(db_path, limit_eur=10.0)
    tracker.record(5.0)
    tracker.check()  # should not raise
    assert tracker.remaining_eur == pytest.approx(5.0)


def test_check_raises_at_cap(db_path: str) -> None:
    tracker = BudgetTracker(db_path, limit_eur=10.0)
    tracker.record(10.0)
    with pytest.raises(BudgetExceededError):
        tracker.check()


def test_reloads_cumulative_spend_from_db(tmp_path: Path) -> None:
    path = tmp_path / "spend.db"
    init_db(path)
    log_call(path, _record(30.0))
    log_call(path, _record(20.0))
    tracker = BudgetTracker(path)
    assert tracker.spent_eur == pytest.approx(50.0)


def test_reload_then_record_crosses_cap(tmp_path: Path) -> None:
    path = tmp_path / "spend.db"
    init_db(path)
    log_call(path, _record(99.5))
    tracker = BudgetTracker(path)  # default €100 cap
    tracker.check()  # still under
    tracker.record(1.0)
    with pytest.raises(BudgetExceededError):
        tracker.check()
