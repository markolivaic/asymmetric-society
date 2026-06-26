"""TrustGame rules: phases, role assignment, validators, payoff math (no LLM).

Pure game-logic tests mirroring ``test_public_goods.py``: no agents, no network.
The two-phase structure (invest -> return per Berg round) and the role bookkeeping
(learned in config order; swapped each round under alternating play) are the parts
unique to this game, so they get the most coverage.
"""

from __future__ import annotations

import random

import pytest

from asymmetric_society.games.trust_game import TrustGame

INV = "aaaa0001"  # first in config order -> initial investor
TRU = "bbbb0002"


def _game(**kw: object) -> TrustGame:
    kw.setdefault("rng", random.Random(0))
    return TrustGame(**kw)  # type: ignore[arg-type]


def _assign_roles(g: TrustGame, state: dict, inv: str = INV, tru: str = TRU) -> None:
    """Mimic the runner: render both agents in config order so roles are learned."""
    g.render_prompt(inv, state, [])
    g.render_prompt(tru, state, [])


def _play(g: TrustGame, send: int, ret: int) -> tuple[list[bool], dict[str, float]]:
    """Drive a full game with constant send/return; return (dones, cumulative)."""
    state = g.reset()
    _assign_roles(g, state)
    cumulative = {INV: 0.0, TRU: 0.0}
    dones: list[bool] = []
    while True:
        berg, phase = state["berg_round"], state["phase"]
        investor, trustee = g._ids_for_round(berg)  # white-box: who acts this phase
        if phase == "invest":
            actions = {investor: {"send": send}, trustee: {"noop": True}}
        else:
            actions = {trustee: {"return": ret}, investor: {"noop": True}}
        state, payoffs, done = g.step(actions)
        for aid in cumulative:
            cumulative[aid] += payoffs[aid]
        dones.append(done)
        if done:
            break
    return dones, cumulative


def test_reset_initial_state() -> None:
    state = _game().reset()
    assert state["round"] == 1
    assert state["berg_round"] == 1
    assert state["phase"] == "invest"
    assert state["endowment"] == 10
    assert state["multiplier"] == 3
    assert state["rounds"] == 10
    assert "received" not in state


def test_invalid_roles_rejected() -> None:
    with pytest.raises(ValueError):
        _game(roles="sideways")


def test_invest_phase_prompts_split_by_role() -> None:
    g = _game()
    s = g.reset()
    p_inv = g.render_prompt(INV, s, [])
    p_tru = g.render_prompt(TRU, s, [])
    assert "INVESTOR" in p_inv and '"send"' in p_inv
    assert '"noop"' in p_tru and '"send"' not in p_tru  # trustee idle in invest phase


def test_more_than_two_players_rejected() -> None:
    g = _game()
    s = g.reset()
    g.render_prompt(INV, s, [])
    g.render_prompt(TRU, s, [])
    with pytest.raises(ValueError):
        g.render_prompt("cccc0003", s, [])


def test_send_validator_range() -> None:
    g = _game()
    s = g.reset()
    _assign_roles(g, s)
    v = g.validator(INV, s)
    assert v({"send": 0}) == {"send": 0}
    assert v({"send": 10}) == {"send": 10}
    assert v({"send": 7, "message": "hi"}) == {"send": 7}  # message ignored by validator


@pytest.mark.parametrize(
    "bad",
    [
        {"send": -1},
        {"send": 11},
        {"send": 1.5},
        {"send": True},  # bool is a subclass of int -> rejected
        {"send": "5"},
        {},
    ],
)
def test_send_validator_rejects_bad(bad: dict) -> None:
    g = _game()
    s = g.reset()
    _assign_roles(g, s)
    v = g.validator(INV, s)
    with pytest.raises((ValueError, KeyError)):
        v(bad)


def test_invest_step_sets_received_and_zero_payoffs() -> None:
    g = _game()
    s = g.reset()
    _assign_roles(g, s)
    s2, payoffs, done = g.step({INV: {"send": 5}, TRU: {"noop": True}})
    assert s2["phase"] == "return"
    assert s2["received"] == 15  # 3 * 5
    assert payoffs == {INV: 0.0, TRU: 0.0}
    assert not done


def test_return_validator_bounded_by_received() -> None:
    g = _game()
    s = g.reset()
    _assign_roles(g, s)
    s2, _, _ = g.step({INV: {"send": 5}, TRU: {"noop": True}})
    v = g.validator(TRU, s2)
    assert v({"return": 0}) == {"return": 0}
    assert v({"return": 15}) == {"return": 15}
    with pytest.raises(ValueError):
        v({"return": 16})


def test_payoff_math_berg_round() -> None:
    g = _game()
    s = g.reset()
    _assign_roles(g, s)
    g.step({INV: {"send": 5}, TRU: {"noop": True}})  # invest
    _, payoffs, _ = g.step({TRU: {"return": 9}, INV: {"noop": True}})  # return
    # e=10, x=5, 3x=15, y=9 -> investor 10-5+9=14, trustee 15-9=6.
    assert payoffs[INV] == pytest.approx(14.0)
    assert payoffs[TRU] == pytest.approx(6.0)


def test_done_after_2T_runner_rounds() -> None:
    dones, _ = _play(_game(rounds=2), send=5, ret=3)
    # 2 Berg rounds -> 4 runner rounds; done only on the final return phase.
    assert dones == [False, False, False, True]


def test_fixed_roles_cumulative_payoffs() -> None:
    _, cumulative = _play(_game(rounds=2), send=5, ret=3)
    # Each Berg round: investor 10-5+3=8, trustee 15-3=12. Over 2 rounds.
    assert cumulative[INV] == pytest.approx(16.0)
    assert cumulative[TRU] == pytest.approx(24.0)


def test_fallback_actions_by_role_and_phase() -> None:
    g = _game()
    s = g.reset()
    _assign_roles(g, s)
    assert g.fallback_action(INV, s) == {"send": 0}  # investor, invest phase
    assert g.fallback_action(TRU, s) == {"noop": True}  # trustee idle in invest
    s2, _, _ = g.step({INV: {"send": 4}, TRU: {"noop": True}})
    assert g.fallback_action(TRU, s2) == {"return": 0}  # trustee, return phase
    assert g.fallback_action(INV, s2) == {"noop": True}  # investor idle in return


def test_noop_validator_is_permissive() -> None:
    g = _game()
    s = g.reset()
    _assign_roles(g, s)
    v = g.validator(TRU, s)  # trustee is idle in the invest phase
    assert v({"noop": True}) == {"noop": True}
    assert v({"message": "I'll be fair", "noop": True}) == {"noop": True}
    with pytest.raises(ValueError):
        v("not a dict")  # type: ignore[arg-type]


def test_alternating_swaps_active_role() -> None:
    g = _game(roles="alternating", rounds=2)
    s = g.reset()
    _assign_roles(g, s)
    s, _, _ = g.step({INV: {"send": 5}, TRU: {"noop": True}})  # berg1 invest
    s, _, _ = g.step({TRU: {"return": 3}, INV: {"noop": True}})  # berg1 return
    assert s["berg_round"] == 2 and s["phase"] == "invest"
    # In berg 2 the original trustee is now the investor.
    assert '"send"' in g.render_prompt(TRU, s, [])
    assert '"noop"' in g.render_prompt(INV, s, [])


def test_alternating_history_attributes_roles_per_round() -> None:
    """Note #1: each past round must read with the role the agent held *then*."""
    g = _game(roles="alternating", rounds=3)
    s = g.reset()
    _assign_roles(g, s)
    g.step({INV: {"send": 5}, TRU: {"noop": True}})  # berg1 invest (INV invests)
    s, _, _ = g.step({TRU: {"return": 3}, INV: {"noop": True}})  # berg1 return
    history = [
        {"round": 1, "actions": {INV: {"send": 5}, TRU: {"noop": True}}},
        {"round": 2, "actions": {TRU: {"return": 3}, INV: {"noop": True}}},
    ]
    inv_view = g.render_prompt(INV, s, history)
    tru_view = g.render_prompt(TRU, s, history)
    assert "you were INVESTOR - you sent 5" in inv_view
    assert "the trustee returned 3" in inv_view
    assert "you were TRUSTEE" in tru_view
    assert "the investor sent 5 (you received 15); you returned 3" in tru_view


def test_communication_prompts_offer_message() -> None:
    g = _game(communication=True)
    s = g.reset()
    p_inv = g.render_prompt(INV, s, [])
    p_tru = g.render_prompt(TRU, s, [])
    assert '"message"' in p_inv and '"send"' in p_inv
    assert '"message"' in p_tru and '"noop"' in p_tru  # trustee cheap-talk turn
    assert "promise" in p_tru


def test_no_communication_omits_message() -> None:
    g = _game(communication=False)
    s = g.reset()
    assert "message" not in g.render_prompt(INV, s, [])
    assert "message" not in g.render_prompt(TRU, s, [])


def test_determinism_same_seed_identical_prompts() -> None:
    g1, g2 = _game(rng=random.Random(42)), _game(rng=random.Random(42))
    history = [
        {"round": 1, "actions": {INV: {"send": 6}, TRU: {"noop": True}}},
        {"round": 2, "actions": {TRU: {"return": 4}, INV: {"noop": True}}},
    ]
    # advance both to berg 2 so history renders
    for g in (g1, g2):
        _assign_roles(g, g.reset())
        g.step({INV: {"send": 6}, TRU: {"noop": True}})
        g.step({TRU: {"return": 4}, INV: {"noop": True}})
    s1, s2 = g1._public_state(), g2._public_state()
    assert g1.render_prompt(INV, s1, history) == g2.render_prompt(INV, s2, history)
