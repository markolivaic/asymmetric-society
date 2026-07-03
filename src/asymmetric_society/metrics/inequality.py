"""Welfare-inequality metrics: Gini, Atkinson, capability premium.

Inequality of realized payoffs is the thesis headline metric (Decision Log,
2026-05-15): almost no LLM-agent paper foregrounds it. These functions operate on
plain payoff vectors; mapping the SQLite results into vectors lives in the caller
(see ``runner.results.final_payoffs``), keeping the metrics dependency-free and
trivially testable on synthetic data.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import bootstrap


def gini(payoffs: list[float] | np.ndarray) -> float:
    """Gini coefficient of a payoff vector.

    0 = perfect equality; approaches 1 as one agent takes everything (exactly
    ``(n-1)/n`` when a single agent holds the whole pie).

    Args:
        payoffs: Non-negative payoff values (at least one).

    Returns:
        The Gini coefficient in ``[0, 1]``.

    Raises:
        ValueError: If empty or any value is negative.
    """
    x = np.asarray(payoffs, dtype=float)
    if x.size == 0:
        raise ValueError("payoffs must be non-empty")
    if np.any(x < 0):
        raise ValueError("Gini requires non-negative values")
    mean = x.mean()
    if mean == 0:
        return 0.0
    abs_diff_sum = np.abs(x[:, None] - x[None, :]).sum()
    return float(abs_diff_sum / (2.0 * x.size**2 * mean))


def atkinson(payoffs: list[float] | np.ndarray, epsilon: float = 1.0) -> float:
    """Atkinson inequality index at inequality-aversion ``epsilon``.

    0 = perfect equality; 1 = maximal inequality. Run at ``epsilon`` in
    ``{0.5, 1.0, 2.0}`` per the thesis spec. With ``epsilon >= 1`` any zero
    payoff drives the index to 1 (the equally-distributed-equivalent income is 0).

    Args:
        payoffs: Non-negative payoff values (at least one).
        epsilon: Inequality-aversion parameter (>= 0).

    Returns:
        The Atkinson index in ``[0, 1]``.

    Raises:
        ValueError: If empty or any value is negative.
    """
    x = np.asarray(payoffs, dtype=float)
    if x.size == 0:
        raise ValueError("payoffs must be non-empty")
    if np.any(x < 0):
        raise ValueError("Atkinson requires non-negative values")
    mean = x.mean()
    if mean == 0:
        return 0.0
    if epsilon == 1.0:
        if np.any(x == 0):
            return 1.0
        ede = np.exp(np.mean(np.log(x)))
    elif epsilon > 1.0:
        if np.any(x == 0):
            return 1.0
        ede = np.mean(x ** (1.0 - epsilon)) ** (1.0 / (1.0 - epsilon))
    else:  # 0 <= epsilon < 1
        ede = np.mean(x ** (1.0 - epsilon)) ** (1.0 / (1.0 - epsilon))
    return float(1.0 - ede / mean)


def capability_premium(
    strong_payoffs: list[float] | np.ndarray,
    weak_payoffs: list[float] | np.ndarray,
    *,
    max_payoff: float | None = None,
) -> float:
    """Mean strong payoff minus mean weak payoff, optionally normalized.

    Args:
        strong_payoffs: Payoffs of strong-tier agents (non-empty).
        weak_payoffs: Payoffs of weak-tier agents (non-empty).
        max_payoff: If given, divide the difference by this (e.g. the maximum
            attainable payoff) to return a normalized premium; otherwise return
            the raw payoff-unit difference.

    Returns:
        The (optionally normalized) capability premium.

    Raises:
        ValueError: If either group is empty or ``max_payoff <= 0``.
    """
    strong = np.asarray(strong_payoffs, dtype=float)
    weak = np.asarray(weak_payoffs, dtype=float)
    if strong.size == 0 or weak.size == 0:
        raise ValueError("both strong and weak payoff groups must be non-empty")
    diff = float(strong.mean() - weak.mean())
    if max_payoff is None:
        return diff
    if max_payoff <= 0:
        raise ValueError("max_payoff must be positive")
    return diff / max_payoff


def gini_bootstrap(
    payoffs: list[float] | np.ndarray,
    *,
    n_resamples: int = 10000,
    confidence_level: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Gini point estimate with a bootstrap confidence interval.

    Args:
        payoffs: Payoff vector.
        n_resamples: Bootstrap resamples (default 10,000 per spec).
        confidence_level: CI level (default 0.95).
        seed: Seed for reproducible resampling.

    Returns:
        ``(point_estimate, ci_lower, ci_upper)``. With fewer than 2 values the CI
        collapses to the point estimate.
    """
    x = np.asarray(payoffs, dtype=float)
    point = gini(x)
    if x.size < 2:
        return (point, point, point)
    rng = np.random.default_rng(seed)
    result = bootstrap(
        (x,),
        gini,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        vectorized=False,
        method="percentile",
        random_state=rng,
    )
    ci = result.confidence_interval
    return (point, float(ci.low), float(ci.high))
