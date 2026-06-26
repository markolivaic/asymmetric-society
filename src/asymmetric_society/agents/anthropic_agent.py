"""Claude Haiku 4.5 backend (Anthropic SDK), with prompt caching.

Cost accounting is cache-aware: Anthropic bills cache *writes* at 1.25x and
cache *reads* at 0.10x the base input rate, reported as separate fields on the
``usage`` object. Multiplying a flat ``input_tokens`` by the base rate would
both over-charge the budget tracker and hide the savings that keep us under the
€100 cap, so we read every cache field explicitly.
"""

from __future__ import annotations

import anthropic

from asymmetric_society.agents.base import AgentAPIError, BaseAgent, CompletionResult
from asymmetric_society.runner.budget import USD_TO_EUR

# USD per million tokens for Claude Haiku 4.5.
_USD_IN_PER_M = 1.0
_USD_OUT_PER_M = 5.0
_CACHE_WRITE_MULT = 1.25
_CACHE_READ_MULT = 0.10

_TRANSIENT = (
    anthropic.RateLimitError,
    anthropic.APITimeoutError,
    anthropic.APIConnectionError,
    anthropic.InternalServerError,
)


class AnthropicAgent(BaseAgent):
    """Backend for ``claude-haiku-4-5-20251001`` and other Anthropic models."""

    def __init__(self, *args: object, max_tokens: int = 512, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.max_tokens = max_tokens
        self._client = anthropic.Anthropic()

    def _complete(self, prompt: str, *, json_mode: bool) -> CompletionResult:
        messages: list[dict[str, object]] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            }
        ]
        # Prefill the assistant turn with "{" so the model must continue a JSON
        # object instead of prefacing it with prose or wrapping it in ```json
        # fences (the failure mode observed in the first paid pilot). We prepend
        # the "{" back below to reconstruct the full object.
        if json_mode:
            messages.append({"role": "assistant", "content": "{"})

        try:
            resp = self._client.messages.create(
                model=self.model_name,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=messages,
            )
        except _TRANSIENT as exc:
            raise AgentAPIError(str(exc)) from exc
        except anthropic.APIStatusError as exc:
            if exc.status_code in (429,) or exc.status_code >= 500:
                raise AgentAPIError(str(exc)) from exc
            raise

        text = "".join(block.text for block in resp.content if block.type == "text")
        if json_mode:
            text = "{" + text
        usage = resp.usage
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cost_usd = (
            usage.input_tokens / 1e6 * _USD_IN_PER_M
            + cache_write / 1e6 * _USD_IN_PER_M * _CACHE_WRITE_MULT
            + cache_read / 1e6 * _USD_IN_PER_M * _CACHE_READ_MULT
            + usage.output_tokens / 1e6 * _USD_OUT_PER_M
        )
        return CompletionResult(
            raw_text=text,
            input_tokens=usage.input_tokens + cache_write + cache_read,
            output_tokens=usage.output_tokens,
            cost_eur=cost_usd * USD_TO_EUR,
            cache_creation_tokens=cache_write,
            cache_read_tokens=cache_read,
        )
