"""Shared pytest fixtures: an isolated SQLite db and a budget tracker."""

from __future__ import annotations

from pathlib import Path

import pytest

from asymmetric_society.runner.budget import BudgetTracker
from asymmetric_society.runner.db import init_db


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    """Path to a fresh, initialized SQLite database for one test."""
    path = tmp_path / "test.db"
    init_db(path)
    return str(path)


@pytest.fixture
def budget(db_path: str) -> BudgetTracker:
    """A budget tracker over the test database with the default €100 cap."""
    return BudgetTracker(db_path)
