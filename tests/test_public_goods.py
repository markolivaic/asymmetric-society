"""PublicGoodsGame rules: validator, fallback, payoff math, determinism (no LLM)."""

from __future__ import annotations

import random

import pytest

from asymmetric_society.games.public_goods import PublicGoodsGame

AGENTS = ["a", "b", "c", "d"]


def _game(**kw: object) -> PublicGoodsGame:
    kw.setdefault("rng", random.Random(0))
    return PublicGoodsGame(**kw)  # type: ignore[arg-type]


def test_reset_initial_state() -> None:
    state = _game().reset()
    assert state["round"] == 1
    assert state["endowment"] == 20
    assert state["rounds"] == 10
    assert state["n"] == 4


def test_validator_accepts_in_range() -> None:
    g = _game()
    v = g.validator("a", g.reset())
    assert v({"contribution": 0}) == {"contribution": 0}
    assert v({"contribution": 20}) == {"contribution": 20}
    # extra keys (like a message) are ignored by the validator
    assert v({"contribution": 7, "message": "hi"}) == {"contribution": 7}


@pytest.mark.parametrize(
    "bad",
    [
        {"contribution": -1},
        {"contribution": 21},
        {"contribution": 1.5},
        {"contribution": True},  # bool is a subclass of int - must be rejected
        {"contribution": "5"},
        {},  # missing key
    ],
)
def test_validator_rejects_bad(bad: dict) -> None:
    g = _game()
    v = g.validator("a", g.reset())
    with pytest.raises((ValueError, KeyError)):
        v(bad)


def test_validator_rejects_non_object() -> None:
    g = _game()
    v = g.validator("a", g.reset())
    with pytest.raises(ValueError):
        v([1, 2, 3])  # type: ignore[arg-type]


def test_fallback_is_zero() -> None:
    g = _game()
    assert g.fallback_action("a", g.reset()) == {"contribution": 0}


def test_step_symmetric_payoffs() -> None:
    g = _game()
    g.reset()
    actions = {aid: {"contribution": 10} for aid in AGENTS}
    state, payoffs, done = g.step(actions)
    # pool = 40 * 1.6 = 64; share = 16; payoff = 16 + (20 - 10) = 26
    assert all(payoffs[aid] == pytest.approx(26.0) for aid in AGENTS)
    assert not done
    assert state["round"] == 2


def test_step_free_rider_beats_cooperator() -> None:
    g = _game()
    g.reset()
    actions = {"a": {"contribution": 20}, "b": {"contribution": 0},
               "c": {"contribution": 0}, "d": {"contribution": 0}}
    _, payoffs, _ = g.step(actions)
    # pool = 20 * 1.6 = 32; share = 8; cooperator: 8 + 0 = 8; free-rider: 8 + 20 = 28
    assert payoffs["a"] == pytest.approx(8.0)
    assert payoffs["b"] == payoffs["c"] == payoffs["d"] == pytest.approx(28.0)


def test_done_after_T_rounds() -> None:
    g = _game(rounds=3)
    g.reset()
    actions = {aid: {"contribution": 0} for aid in AGENTS}
    dones = [g.step(actions)[2] for _ in range(3)]
    assert dones == [False, False, True]


def test_render_history_deterministic_same_seed() -> None:
    g1 = _game(rng=random.Random(42))
    g2 = _game(rng=random.Random(42))
    s1, s2 = g1.reset(), g2.reset()
    history = [
        {"round": 1, "actions": {"a": {"contribution": 5}, "b": {"contribution": 12},
                                 "c": {"contribution": 0}, "d": {"contribution": 20}}},
    ]
    for aid in AGENTS:  # identical call order -> identical seeded shuffles
        assert g1.render_prompt(aid, s1, history) == g2.render_prompt(aid, s2, history)


def test_prompt_omits_message_field_without_communication() -> None:
    g = _game(communication=False)
    p = g.render_prompt("a", g.reset(), [])
    assert '"contribution"' in p
    assert "message" not in p


def test_prompt_includes_message_field_with_communication() -> None:
    g = _game(communication=True)
    p = g.render_prompt("a", g.reset(), [])
    assert "message" in p
    assert "200 characters" in p
