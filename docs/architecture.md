# Architecture

Design notes for the Asymmetric Society simulation framework. This documents the
ZAV-16 scaffold and the ZAV-17 agent abstraction layer.

## Why build from scratch (not fork FAIRGAME)

FAIRGAME (arXiv:2504.14325) is cited as related work but not forked: its
architecture (2-player, single-model, matrix-only games) doesn't support our
core need (n-player, mixed-model, diverse game types). We borrow three patterns
and cite them:

1. **Config-driven experiments via JSON** - `config.py` (pydantic schema).
2. **Text-based prompt templates with placeholders** - `prompts.py`.
3. **Retry-on-parse-failure for LLM responses** - `agents/base.py`.

## Layers

```
config.py     ExperimentConfig / AgentConfig / GameConfig (pydantic, JSON-loaded)
prompts.py    PromptTemplate - {placeholder} fill with strict validation
games/base.py BaseGame ABC - owns rules, prompt rendering, validator, fallback
agents/       BaseAgent ABC + 4 backends + factory - owns the LLM call
runner/       budget.py (hard cap) + db.py (SQLite call log)
```

### Agent layer: one pipeline, four backends

The retry / logging / budget pipeline lives once in `BaseAgent.act`. Each backend
implements only `_complete(prompt, *, json_mode) -> CompletionResult` (the raw
model call plus cost accounting). This guarantees retry semantics are tested
once, not four times.

`act` deliberately refines the ticket's literal `act(state, history)` signature
to `act(prompt, validator, fallback, ...)`. Rationale: the **game** owns prompt
rendering and the action schema; the **agent** stays game-agnostic and owns the
LLM call, retry, and logging. Clean separation of concerns.

### Two failure classes (critical for the ZAV-20 gate)

The retry loop separates two kinds of failure, because the ZAV-20 gate
(parse-failure rate < 15%) must measure *model* mistakes, not network weather:

| Failure | Trigger | Handling | Counter |
|---|---|---|---|
| **API / transport** | rate limit (429), timeout, 5xx → `AgentAPIError` | exponential backoff, then re-raise | `api_error_count` |
| **Parse / validation** | model replied but JSON invalid or validator rejects | retry up to N, then fallback action | `parse_fail_count` |

A persistent transport failure **re-raises** rather than silently falling back -
a dead API is not a model giving a bad answer, and a silent fallback would
corrupt the behavioral data. Both counters are logged on every `llm_calls` row;
ZAV-20 derives its rate from `parse_fail_count` only.

### Budget cap

`runner/budget.py` enforces a hard €100 cap. `BudgetTracker` seeds its running
total from the sum of `cost_eur` in SQLite on construction (so the cap survives
restarts), `check()` raises `BudgetExceededError` *before* a call once the cap is
reached, and `record()` updates the total after each call. Provider pricing is in
USD; `USD_TO_EUR` converts to the EUR cap.

Anthropic cost is **cache-aware**: cache writes bill at 1.25× and cache reads at
0.10× the base input rate, reported as separate `usage` fields. We read them
explicitly instead of multiplying a flat `input_tokens`, so the tracker neither
over-charges nor hides the caching savings we rely on to stay under budget.

### Logging

`runner/db.py` uses stdlib `sqlite3` (not SQLAlchemy - per project constraints).
Every call logs prompt, raw response, parsed action, token counts, cost, the two
failure counters, and `parse_ok` / `fallback_used`. This table feeds the
KL-deception metric, the ZAV-20 gate, budget reconciliation, and debugging.
ZAV-19 adds experiments/agents/rounds/actions/payoffs tables alongside it.

## Reproducibility - read before writing the thesis

`same config + seed` reproduces the **game setup, agent ordering, and
randomization** deterministically. It does **not** guarantee bit-identical LLM
outputs: even at `temperature=0`, hosted models vary across runs (batching,
hardware, silent model updates). Describe the guarantee as *pipeline
reproducibility* / *deterministic given fixed LLM outputs* - never claim
bit-identical model text, or a defense examiner can rightly challenge it.

## Concurrency

`act()` is synchronous (matches the ticket; simplest to test). Ollama serves one
model instance and serializes concurrent requests, so the runner (ZAV-19) will
overlap paid API calls with the serial local calls via
`asyncio.to_thread(agent.act, ...)` + `asyncio.gather`, rather than making the
backends natively async.
