"""Parse-retry and API-error-retry logic, with a mock backend (no real calls)."""

from __future__ import annotations

import pytest

from asymmetric_society.agents.base import (
    AgentAPIError,
    BaseAgent,
    CompletionResult,
)
from asymmetric_society.runner.budget import BudgetTracker

FALLBACK = {"contribution": 0}


def validator(payload: dict) -> dict:
    """Accept only {"contribution": int in [0, 20]}."""
    c = payload["contribution"]
    if not isinstance(c, int) or isinstance(c, bool) or not 0 <= c <= 20:
        raise ValueError("contribution out of range")
    return {"contribution": c}


class FakeAgent(BaseAgent):
    """Backend whose ``_complete`` replays a scripted list of outcomes.

    Each scripted item is either a string (returned as raw text) or an
    ``Exception`` instance (raised). This lets us drive every branch of the
    retry loop deterministically without touching a network.
    """

    def __init__(self, responses: list, **kwargs: object) -> None:
        super().__init__("agent01", "fake-model", "weak", **kwargs)  # type: ignore[arg-type]
        self._responses = list(responses)
        self.calls = 0

    def _complete(self, prompt: str, *, json_mode: bool) -> CompletionResult:
        self.calls += 1
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return CompletionResult(raw_text=item, input_tokens=10, output_tokens=5, cost_eur=0.001)


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make exponential backoff instant during tests."""
    monkeypatch.setattr("asymmetric_society.agents.base.time.sleep", lambda _s: None)


def _agent(responses: list, budget: BudgetTracker, db_path: str, **kw: object) -> FakeAgent:
    return FakeAgent(responses, budget=budget, db_path=db_path, **kw)


def test_valid_first_try(budget: BudgetTracker, db_path: str) -> None:
    agent = _agent(['{"contribution": 5}'], budget, db_path)
    result = agent.act("p", validator, FALLBACK)
    assert result.action == {"contribution": 5}
    assert result.parse_ok and not result.fallback_used
    assert result.parse_fail_count == 0
    assert agent.calls == 1


def test_retries_then_succeeds(budget: BudgetTracker, db_path: str) -> None:
    agent = _agent(["not json", '{"contribution": 99}', '{"contribution": 7}'], budget, db_path)
    result = agent.act("p", validator, FALLBACK)
    assert result.action == {"contribution": 7}
    assert result.parse_ok
    assert result.parse_fail_count == 2  # bad JSON + out-of-range
    assert result.api_error_count == 0


def test_exhausts_parse_retries_then_fallback(budget: BudgetTracker, db_path: str) -> None:
    agent = _agent(["x", "y", "z", "w"], budget, db_path, max_parse_retries=3)
    result = agent.act("p", validator, FALLBACK)
    assert result.fallback_used
    assert result.action == FALLBACK
    assert result.parse_fail_count == 4
    assert agent.calls == 4


def test_api_error_uses_backoff_not_parse_failure(
    budget: BudgetTracker, db_path: str
) -> None:
    agent = _agent(
        [AgentAPIError("429"), AgentAPIError("503"), '{"contribution": 3}'],
        budget,
        db_path,
    )
    result = agent.act("p", validator, FALLBACK)
    assert result.action == {"contribution": 3}
    assert result.api_error_count == 2
    assert result.parse_fail_count == 0  # network failures are NOT parse failures


def test_persistent_api_error_raises_no_silent_fallback(
    budget: BudgetTracker, db_path: str
) -> None:
    agent = _agent([AgentAPIError("down")] * 4, budget, db_path, max_api_retries=3)
    with pytest.raises(AgentAPIError):
        agent.act("p", validator, FALLBACK)
    assert agent.calls == 4  # initial + 3 backoff retries


def test_parses_json_wrapped_in_markdown_fences(budget: BudgetTracker, db_path: str) -> None:
    # The Claude failure mode from the first paid pilot: valid JSON in a code fence.
    agent = _agent(['```json\n{"contribution": 8}\n```'], budget, db_path)
    result = agent.act("p", validator, FALLBACK)
    assert result.action == {"contribution": 8}
    assert result.parse_ok and not result.fallback_used
    assert agent.calls == 1


def test_parses_json_with_surrounding_prose(budget: BudgetTracker, db_path: str) -> None:
    # Reasoning before and commentary after the object must not break parsing.
    agent = _agent(['Let me think. {"contribution": 6} because cooperation pays.'], budget, db_path)
    result = agent.act("p", validator, FALLBACK)
    assert result.action == {"contribution": 6}
    assert result.parse_ok


def test_plain_prose_without_object_still_fails(budget: BudgetTracker, db_path: str) -> None:
    agent = _agent(["I will analyze the game first."], budget, db_path, max_parse_retries=0)
    result = agent.act("p", validator, FALLBACK)
    assert result.fallback_used
    assert result.parse_fail_count == 1
