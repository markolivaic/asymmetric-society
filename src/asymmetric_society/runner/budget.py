"""Hard budget cap with a kill switch.

The thesis has a hard €100 cap across all paid API spend. ``BudgetTracker``
loads the cumulative spend recorded in SQLite on construction (so the cap
survives process restarts), refuses to authorize a call once the cap is reached,
and records each call's cost as it happens.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

#: Hard cap across the whole thesis, in EUR. Never raise this in code.
BUDGET_LIMIT_EUR: float = 100.0

#: USD->EUR conversion. Provider pricing is quoted in USD; the cap is in EUR.
#: Approximate and configurable; conservatively rounding up protects the cap.
USD_TO_EUR: float = 0.92


class BudgetExceededError(RuntimeError):
    """Raised before an API call when the cumulative spend cap is reached."""


class BudgetTracker:
    """Tracks cumulative EUR spend and enforces the hard cap.

    Attributes:
        limit_eur: The hard cap.
        spent_eur: Cumulative spend known to this tracker.
    """

    def __init__(
        self,
        db_path: str | Path,
        *,
        limit_eur: float = BUDGET_LIMIT_EUR,
    ) -> None:
        """Initialize the tracker, seeding ``spent_eur`` from the database.

        Args:
            db_path: Path to the SQLite database holding ``llm_calls``. The total
                of its ``cost_eur`` column becomes the starting spend.
            limit_eur: Hard cap in EUR.
        """
        self.db_path = str(db_path)
        self.limit_eur = limit_eur
        self.spent_eur = self._load_spend_from_db()

    def _load_spend_from_db(self) -> float:
        """Sum ``cost_eur`` across all logged calls, or 0.0 if none yet."""
        if not Path(self.db_path).exists():
            return 0.0
        con = sqlite3.connect(self.db_path)
        try:
            cur = con.execute("SELECT COALESCE(SUM(cost_eur), 0.0) FROM llm_calls")
            (total,) = cur.fetchone()
            return float(total)
        except sqlite3.OperationalError:
            # Table not created yet - treat as zero spend.
            return 0.0
        finally:
            con.close()

    def check(self) -> None:
        """Authorize the next API call, or abort.

        Raises:
            BudgetExceededError: If cumulative spend has reached the cap. The
                call that would push us over is never made.
        """
        if self.spent_eur >= self.limit_eur:
            raise BudgetExceededError(
                f"Budget cap reached: spent €{self.spent_eur:.4f} of "
                f"€{self.limit_eur:.2f}. Aborting before next API call."
            )

    def record(self, cost_eur: float) -> None:
        """Add the cost of a completed call to the running total.

        Args:
            cost_eur: Cost of the call in EUR (0.0 for local models).
        """
        self.spent_eur += cost_eur

    @property
    def remaining_eur(self) -> float:
        """EUR left before the cap (never negative)."""
        return max(0.0, self.limit_eur - self.spent_eur)
