"""Llama 3.2 3B backend (Ollama, local).

JSON is requested via ``format="json"`` - prompt-level coercion, NOT true
constrained decoding (per project constraints). The base class's validate-and-
retry loop handles the malformed output a 3B model occasionally produces. Local
inference is free, so ``cost_eur`` is always 0.0; Ollama serializes concurrent
requests, which the runner accounts for separately.
"""

from __future__ import annotations

import httpx
import ollama

from asymmetric_society.agents.base import AgentAPIError, BaseAgent, CompletionResult

_TRANSIENT = (
    ollama.ResponseError,
    httpx.TimeoutException,
    httpx.ConnectError,
    ConnectionError,
)


class OllamaAgent(BaseAgent):
    """Backend for a locally served ``llama3.2:3b`` (or similar) model."""

    def __init__(
        self,
        *args: object,
        host: str = "http://localhost:11434",
        **kwargs: object,
    ) -> None:
        super().__init__(*args, **kwargs)  # type: ignore[arg-type]
        self._client = ollama.Client(host=host)

    def _complete(self, prompt: str, *, json_mode: bool) -> CompletionResult:
        try:
            resp = self._client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                format="json" if json_mode else "",
                options={"temperature": self.temperature},
            )
        except _TRANSIENT as exc:
            raise AgentAPIError(str(exc)) from exc

        text = resp["message"]["content"]
        return CompletionResult(
            raw_text=text,
            input_tokens=resp.get("prompt_eval_count", 0) or 0,
            output_tokens=resp.get("eval_count", 0) or 0,
            cost_eur=0.0,
        )
