"""Shared Haiku message-scoring seam (the message classifier + the deception metric).

Both message metrics ask Haiku to read a message and emit structured JSON: the
classifier a category label, the deception metric a probability distribution over
an action support. They
share one tested call path - :func:`llm_score` over :meth:`BaseAgent.act` - so the
prefill-``{``, tolerant parser, parse-retry, €100 budget check, and ``llm_calls``
logging are written and tested once. Only the prompt, validator, and fallback
differ per task.

Extracted from ``classifier.py`` once the deception metric became the second consumer (the
"refactor when a real second user exists" rule); ``classifier.py`` re-exports the
names for back-compatibility.

Scoring calls log with ``capability_tier="classifier"`` and ``experiment_id=None``
so they form their own ``llm_calls`` group and never contaminate the weak/strong
parse-failure gate, while still being counted against the budget.
"""

from __future__ import annotations

from typing import Any

from asymmetric_society.agents.anthropic_agent import AnthropicAgent
from asymmetric_society.agents.base import ActionResult, Validator
from asymmetric_society.runner.budget import BudgetTracker

#: Exact dated scorer model string (Claude Haiku 4.5).
SCORING_MODEL = "claude-haiku-4-5-20251001"


def build_scoring_agent(
    db_path: str,
    *,
    budget: BudgetTracker,
    agent_id: str = "scorer",
    tier: str = "classifier",
    max_tokens: int = 64,
) -> AnthropicAgent:
    """Construct a Haiku agent for a message-scoring task.

    ``tier`` is a free-form capability-tier string (BaseAgent stores it verbatim);
    keeping it off ``"strong"``/``"weak"`` is what excludes these calls from the
    parse-failure gate. Temperature is 0 for stable, repeatable scoring.

    Args:
        db_path: SQLite database for call logging.
        budget: Shared budget tracker enforcing the €100 cap.
        agent_id: Logical id recorded on each call (e.g. ``"classifier"``).
        tier: Capability-tier label for the log rows (kept out of the gate).
        max_tokens: Output cap (labels/short distributions are tiny).

    Returns:
        A ready-to-use :class:`AnthropicAgent`.
    """
    return AnthropicAgent(
        agent_id,
        SCORING_MODEL,
        tier,
        budget=budget,
        db_path=db_path,
        temperature=0.0,
        max_tokens=max_tokens,
    )


def llm_score(
    agent: AnthropicAgent,
    prompt: str,
    *,
    validator: Validator,
    fallback: dict[str, Any],
    experiment_id: str | None = None,
    round_no: int | None = None,
) -> ActionResult:
    """Run one scoring LLM call through the shared ``act`` pipeline.

    The single seam every message-scoring task reuses: supply a task-specific
    ``validator`` and ``fallback`` and inherit the prefill/parse/retry/budget/log
    machinery.

    Args:
        agent: The scoring agent.
        prompt: The fully rendered prompt.
        validator: Turns parsed JSON into a clean action or raises.
        fallback: Safe default used if every parse attempt fails.
        experiment_id: Optional experiment id for the log row (kept ``None`` for
            scoring calls so they stay out of the gate).
        round_no: Optional round number for the log row.

    Returns:
        The :class:`ActionResult` from :meth:`BaseAgent.act`.
    """
    return agent.act(
        prompt,
        validator,
        fallback,
        json_mode=True,
        experiment_id=experiment_id,
        round_no=round_no,
    )
