"""Inequality metrics validated on synthetic data (equality->0, inequality->1)."""

from __future__ import annotations

import pytest

from asymmetric_society.metrics import inequality


# --- Gini ------------------------------------------------------------------

def test_gini_perfect_equality_is_zero() -> None:
    assert inequality.gini([10, 10, 10, 10]) == pytest.approx(0.0)


def test_gini_perfect_inequality_approaches_one() -> None:
    # One agent holds everything; Gini = (n-1)/n -> ~1 for large n.
    payoffs = [0.0] * 99 + [100.0]
    assert inequality.gini(payoffs) == pytest.approx(0.99, abs=1e-6)


def test_gini_two_person_one_holder_is_half() -> None:
    assert inequality.gini([0, 1]) == pytest.approx(0.5)


def test_gini_is_scale_invariant() -> None:
    assert inequality.gini([1, 2, 3, 4]) == pytest.approx(inequality.gini([10, 20, 30, 40]))


@pytest.mark.parametrize("bad", [[], [-1.0, 2.0]])
def test_gini_rejects_bad_input(bad: list) -> None:
    with pytest.raises(ValueError):
        inequality.gini(bad)


# --- Atkinson --------------------------------------------------------------

@pytest.mark.parametrize("eps", [0.5, 1.0, 2.0])
def test_atkinson_equality_is_zero(eps: float) -> None:
    assert inequality.atkinson([7, 7, 7, 7], epsilon=eps) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("eps", [0.5, 1.0, 2.0])
def test_atkinson_inequality_is_positive(eps: float) -> None:
    assert inequality.atkinson([1, 5, 20, 100], epsilon=eps) > 0.0


@pytest.mark.parametrize("eps", [1.0, 2.0])
def test_atkinson_zero_payoff_drives_to_one(eps: float) -> None:
    # With eps >= 1 a single zero makes the equally-distributed equivalent 0.
    assert inequality.atkinson([0, 50, 50, 50], epsilon=eps) == pytest.approx(1.0)


def test_atkinson_rejects_negative() -> None:
    with pytest.raises(ValueError):
        inequality.atkinson([-1, 2, 3])


# --- Capability premium ----------------------------------------------------

def test_capability_premium_raw_difference() -> None:
    assert inequality.capability_premium([30, 40], [10, 20]) == pytest.approx(20.0)


def test_capability_premium_normalized() -> None:
    assert inequality.capability_premium([30, 40], [10, 20], max_payoff=40) == pytest.approx(0.5)


def test_capability_premium_can_be_negative() -> None:
    # Weak out-earning strong yields a negative premium (exploitation signal).
    assert inequality.capability_premium([10], [25]) == pytest.approx(-15.0)


@pytest.mark.parametrize("kwargs", [{"strong_payoffs": [], "weak_payoffs": [1]}])
def test_capability_premium_rejects_empty(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        inequality.capability_premium(**kwargs)


# --- Gini bootstrap --------------------------------------------------------

def test_gini_bootstrap_point_matches_gini_and_brackets() -> None:
    payoffs = [5, 8, 12, 20, 3, 9, 15, 7]
    point, lo, hi = inequality.gini_bootstrap(payoffs, n_resamples=500, seed=42)
    assert point == pytest.approx(inequality.gini(payoffs))
    assert lo <= point <= hi


def test_gini_bootstrap_is_deterministic_with_seed() -> None:
    payoffs = [5, 8, 12, 20, 3, 9, 15, 7]
    a = inequality.gini_bootstrap(payoffs, n_resamples=500, seed=7)
    b = inequality.gini_bootstrap(payoffs, n_resamples=500, seed=7)
    assert a == b


def test_gini_bootstrap_single_value_collapses() -> None:
    assert inequality.gini_bootstrap([10.0]) == (0.0, 0.0, 0.0)
