"""Gemini Flash backend (google-genai SDK migration).

Migrated off the deprecated ``google-generativeai`` package to ``google-genai``
(the ``google.genai`` client). Two things matter for this project:

* **Thinking is disabled** (``thinking_budget=0``). Gemini 2.5+ Flash bills
  thinking tokens at the full *output* rate, which would silently inflate cost
  well beyond the headline price; a single game action needs no chain-of-thought.
* **JSON mode** uses ``response_mime_type="application/json"`` (the SDK's native
  constrained-JSON), parsed by the shared tolerant parser in the base class.
"""

from __future__ import annotations

import os

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from asymmetric_society.agents.base import AgentAPIError, BaseAgent, CompletionResult
from asymmetric_society.runner.budget import USD_TO_EUR

# USD per million tokens for Gemini Flash (verify per model string before launch).
_USD_IN_PER_M = 0.10
_USD_OUT_PER_M = 0.40

_TRANSIENT_HTTPX = (
    httpx.TimeoutException,
    httpx.ConnectError,
)


class GeminiAgent(BaseAgent):
    """Backend for ``gemini-2.5-flash`` and other Gemini models (thinking off)."""

    def __init__(self, *args: object, max_tokens: int = 512, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self.max_tokens = max_tokens
        self._client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    def _complete(self, prompt: str, *, json_mode: bool) -> CompletionResult:
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
            response_mime_type="application/json" if json_mode else "text/plain",
        )
        try:
            resp = self._client.models.generate_content(
                model=self.model_name, contents=prompt, config=config
            )
        except _TRANSIENT_HTTPX as exc:
            raise AgentAPIError(str(exc)) from exc
        except genai_errors.APIError as exc:
            # Retry only on rate limit (429) and server errors (5xx); a 4xx like a
            # bad request is a real error that must surface, not be retried away.
            code = getattr(exc, "code", None)
            if code == 429 or (isinstance(code, int) and code >= 500):
                raise AgentAPIError(str(exc)) from exc
            raise

        try:
            text = resp.text or ""
        except (ValueError, AttributeError):
            # .text raises if the candidate carried no text part (e.g. blocked or
            # empty); treat as an empty completion so the parse-retry loop handles it.
            text = ""

        usage = resp.usage_metadata
        in_tok = getattr(usage, "prompt_token_count", 0) or 0
        out_tok = getattr(usage, "candidates_token_count", 0) or 0
        think_tok = getattr(usage, "thoughts_token_count", 0) or 0
        # thoughts should be 0 with thinking disabled; bill them at the output rate
        # if any slip through, so the budget tracker is never under-counted.
        cost_usd = in_tok / 1e6 * _USD_IN_PER_M + (out_tok + think_tok) / 1e6 * _USD_OUT_PER_M
        return CompletionResult(
            raw_text=text,
            input_tokens=in_tok,
            output_tokens=out_tok + think_tok,
            cost_eur=cost_usd * USD_TO_EUR,
        )
