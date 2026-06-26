"""GPT-5-mini backend (OpenAI SDK).

Note: the exact ``gpt-5-mini`` model string and its parameter constraints should
be verified against the live OpenAI docs before any real spend. The model string
is config-overridable.
"""

from __future__ import annotations

import openai

from asymmetric_society.agents.base import AgentAPIError, BaseAgent, CompletionResult
from asymmetric_society.runner.budget import USD_TO_EUR

# USD per million tokens for GPT-5-mini.
_USD_IN_PER_M = 0.25
_USD_OUT_PER_M = 2.0

_TRANSIENT = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


class OpenAIAgent(BaseAgent):
    """Backend for ``gpt-5-mini`` and other OpenAI chat models."""

    def __init__(self, *args: object, max_tokens: int = 512, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.max_tokens = max_tokens
        self._client = openai.OpenAI()

    def _complete(self, prompt: str, *, json_mode: bool) -> CompletionResult:
        kwargs: dict[str, object] = {
            "model": self.model_name,
            "max_completion_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        # GPT-5 reasoning models accept ONLY the default temperature (1); any
        # other value is a 400 BadRequest. Other OpenAI models honor temperature.
        # Consequence: gpt-5-mini effectively runs at temperature 1 regardless of
        # config (documented heterogeneity vs the other backends).
        if not self.model_name.startswith("gpt-5"):
            kwargs["temperature"] = self.temperature
        else:
            # Cap reasoning to "minimal" (no chain-of-thought): the default effort
            # consumed ~900 reasoning tokens on a game prompt, overrunning
            # max_completion_tokens (-> empty output -> parse fail) AND billing
            # those tokens at the output rate. "minimal" gives 0 reasoning tokens,
            # a clean JSON answer, and play comparable to the non-reasoning
            # backends (a documented design choice).
            kwargs["reasoning_effort"] = "minimal"
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except _TRANSIENT as exc:
            raise AgentAPIError(str(exc)) from exc
        except openai.APIStatusError as exc:
            if exc.status_code in (429,) or exc.status_code >= 500:
                raise AgentAPIError(str(exc)) from exc
            raise

        text = resp.choices[0].message.content or ""
        usage = resp.usage
        in_tok = usage.prompt_tokens if usage else 0
        out_tok = usage.completion_tokens if usage else 0
        cost_usd = in_tok / 1e6 * _USD_IN_PER_M + out_tok / 1e6 * _USD_OUT_PER_M
        return CompletionResult(
            raw_text=text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_eur=cost_usd * USD_TO_EUR,
        )
