"""Agent abstraction layer (:class:`BaseAgent`).

All four backends share one interface and one retry/log/budget pipeline, which
lives here in the base class. Subclasses implement only ``_complete`` (the raw
model call plus cost accounting). This guarantees the retry semantics are tested
once, not four times.

The retry loop distinguishes two failure classes (see ``docs/architecture.md``):

  * :class:`AgentAPIError` - transport failure (rate limit, timeout, 5xx). Retry
    with exponential backoff; counted in ``api_error_count``; never a parse
    failure. If retries are exhausted we re-raise: a dead API is not a model
    giving a bad answer, and silently falling back would corrupt the data.
  * parse / validation failure - the model replied but the JSON was invalid or
    the game's validator rejected it. Retry up to ``max_parse_retries``; counted
    in ``parse_fail_count``; on exhaustion we use the game's fallback action.
"""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from asymmetric_society.runner import db
from asymmetric_society.runner.budget import BudgetTracker

Action = dict[str, Any]
Validator = Callable[[dict[str, Any]], Action]


def _parse_json_object(text: str) -> Any:
    """Parse the first JSON object from model output, tolerant of wrappers.

    Models that follow instructions emit bare JSON, but some (e.g. Claude) wrap
    it in ```json fences or surround it with prose reasoning. This strips a
    leading/trailing markdown code fence and then uses ``raw_decode`` from the
    first ``{`` so any text before or after the object is ignored. Genuinely
    non-JSON output (no object) still raises and counts as a parse failure.
    """
    s = text.strip()
    if s.startswith("```"):
        s = s[3:]
        if s[:4].lower() == "json":
            s = s[4:]
        fence = s.rfind("```")
        if fence != -1:
            s = s[:fence]
        s = s.strip()
    start = s.find("{")
    if start == -1:
        raise ValueError("no JSON object found in response")
    obj, _ = json.JSONDecoder().raw_decode(s[start:])
    return obj


class AgentAPIError(RuntimeError):
    """Transport-level failure from a backend (rate limit, timeout, 5xx).

    Backends catch their SDK's transient exceptions and re-raise as this shared
    type so the base retry loop can treat them uniformly and, crucially, keep
    them out of the parse-failure statistics.
    """


@dataclass
class CompletionResult:
    """Raw output of a single model call, returned by ``_complete``."""

    raw_text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_eur: float = 0.0
    #: Anthropic cache accounting (zero for other backends); kept for auditing.
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0


@dataclass
class ActionResult:
    """Outcome of :meth:`BaseAgent.act`."""

    action: Action
    message: str | None = None
    raw_response: str | None = None
    parsed_action: dict[str, Any] | None = None
    api_error_count: int = 0
    parse_fail_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_eur: float = 0.0
    parse_ok: bool = False
    fallback_used: bool = False
    _attempts: int = field(default=0, repr=False)


class BaseAgent(ABC):
    """Unified interface over a single LLM backend.

    Args:
        agent_id: Anonymous 8-char identity.
        model_name: Exact dated model string.
        capability_tier: ``"strong"`` or ``"weak"``.
        budget: Shared budget tracker; checked before every call.
        db_path: SQLite database for call logging.
        temperature: Sampling temperature.
        max_parse_retries: Max attempts on invalid JSON before fallback.
        max_api_retries: Max backoff retries on transport failure.
        backoff_base: Base seconds for exponential backoff (1, 2, 4, ...).
    """

    def __init__(
        self,
        agent_id: str,
        model_name: str,
        capability_tier: str,
        *,
        budget: BudgetTracker,
        db_path: str,
        temperature: float = 1.0,
        max_parse_retries: int = 3,
        max_api_retries: int = 3,
        backoff_base: float = 1.0,
    ) -> None:
        self.agent_id = agent_id
        self.model_name = model_name
        self.capability_tier = capability_tier
        self.budget = budget
        self.db_path = db_path
        self.temperature = temperature
        self.max_parse_retries = max_parse_retries
        self.max_api_retries = max_api_retries
        self.backoff_base = backoff_base

    @abstractmethod
    def _complete(self, prompt: str, *, json_mode: bool) -> CompletionResult:
        """Make one raw model call and return text plus cost accounting.

        Implementations must translate their SDK's transient errors into
        :class:`AgentAPIError`. They should not retry - the base class owns that.

        Args:
            prompt: The fully rendered prompt.
            json_mode: Whether to request JSON-formatted output.

        Returns:
            The raw completion and its token/cost accounting.

        Raises:
            AgentAPIError: On any transient transport failure.
        """

    def _call_with_backoff(
        self, prompt: str, *, json_mode: bool, result: ActionResult
    ) -> CompletionResult:
        """Call ``_complete`` with budget check and exponential backoff.

        Transport failures are retried with backoff and tallied in
        ``result.api_error_count``. Budget is checked before every attempt.

        Raises:
            AgentAPIError: If all transport retries are exhausted.
            BudgetExceededError: If the cap is reached (propagates from budget).
        """
        attempt = 0
        while True:
            self.budget.check()
            try:
                completion = self._complete(prompt, json_mode=json_mode)
            except AgentAPIError:
                result.api_error_count += 1
                if attempt >= self.max_api_retries:
                    raise
                time.sleep(self.backoff_base * (2**attempt))
                attempt += 1
                continue
            self.budget.record(completion.cost_eur)
            return completion

    def act(
        self,
        prompt: str,
        validator: Validator,
        fallback: Action,
        *,
        json_mode: bool = True,
        experiment_id: str | None = None,
        round_no: int | None = None,
    ) -> ActionResult:
        """Produce a validated action for ``prompt``, retrying as needed.

        Args:
            prompt: The fully rendered prompt (the game owns rendering).
            validator: Turns raw parsed JSON into a clean action or raises
                ``ValueError``/``KeyError`` on a bad/out-of-range response.
            fallback: Safe default action used if every parse attempt fails.
            json_mode: Request JSON output from the backend.
            experiment_id: Optional experiment id for the log row.
            round_no: Optional round number for the log row.

        Returns:
            An :class:`ActionResult`; ``fallback_used`` is True iff the model
            never produced a valid action.

        Raises:
            AgentAPIError: If the backend keeps failing at the transport level.
            BudgetExceededError: If the budget cap is reached.
        """
        result = ActionResult(action=fallback)
        last_raw: str | None = None

        for attempt in range(self.max_parse_retries + 1):
            result._attempts = attempt + 1
            completion = self._call_with_backoff(prompt, json_mode=json_mode, result=result)
            last_raw = completion.raw_text
            result.input_tokens += completion.input_tokens
            result.output_tokens += completion.output_tokens
            result.cost_eur += completion.cost_eur

            try:
                payload = _parse_json_object(completion.raw_text)
                action = validator(payload)
            except (json.JSONDecodeError, ValueError, KeyError, TypeError):
                result.parse_fail_count += 1
                continue

            result.action = action
            result.message = payload.get("message") if isinstance(payload, dict) else None
            result.parsed_action = action
            result.parse_ok = True
            break
        else:
            # Loop exhausted without break: every attempt failed to parse.
            result.fallback_used = True
            result.parsed_action = fallback

        result.raw_response = last_raw
        self._log(result, prompt, experiment_id, round_no)
        return result

    def _log(
        self,
        result: ActionResult,
        prompt: str,
        experiment_id: str | None,
        round_no: int | None,
    ) -> None:
        """Persist the call to SQLite."""
        db.log_call(
            self.db_path,
            db.CallRecord(
                agent_id=self.agent_id,
                model_name=self.model_name,
                capability_tier=self.capability_tier,
                prompt=prompt,
                raw_response=result.raw_response,
                parsed_action=result.parsed_action,
                api_error_count=result.api_error_count,
                parse_fail_count=result.parse_fail_count,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cost_eur=result.cost_eur,
                parse_ok=result.parse_ok,
                fallback_used=result.fallback_used,
                experiment_id=experiment_id,
                round_no=round_no,
            ),
        )
