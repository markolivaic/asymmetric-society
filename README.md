# Asymmetric Society

**Capability heterogeneity, strategic exploitation, and welfare inequality in LLM agent populations**

> Bachelor's thesis - Francesco Marko Livaić - Tehničko veleučilište u Zagrebu (TVZ) - 2026

## What this is

When language models of very different capability share a task, does the stronger one come
out ahead? This project puts strong frontier models (Claude Haiku 4.5, GPT-5-mini, Gemini 2.5
Flash) and a small local model (Llama 3.2 3B, served through Ollama) into the same repeated
economic games and measures who exploits whom, how inequality emerges, and whether the agents
deceive each other.

The headline result reverses the intuition. With no communication the strong models earn a
capability premium; add a condition in which each agent states a public pledge before it acts,
and the premium inverts: the strong models earn less. The effect belongs to the group game and
fades in the two-player exchange. A second contribution is methodological. A contemporaneous
measure of deception separates the capability tiers where the obvious global divergence fails,
because that divergence rewards behavioral consistency rather than dishonesty.

The study is pre-registered ([osf.io/rnqgs](https://osf.io/rnqgs/)) and the full write-up is the
LaTeX thesis under `thesis/`.

## The games

- **Iterated Public Goods Game** (4 players): contribute to a shared pool or free-ride on the others.
- **Trust Game** (Berg, Dickhaut & McCabe 1995, 2 players): send an investment that is tripled, then decide how much to return.

Each game runs with and without the communication condition, across a sixteen-cell matrix of
population compositions (from a lone strong agent among the weak to the reverse), replicated over
twenty seeds: 320 games in total.

## What it measures

- **Welfare inequality**: Gini coefficient and Atkinson index, with bootstrap confidence intervals.
- **Capability premium**: the strong tier's payoff minus the weak tier's.
- **Deception**: a contemporaneous, per-round surprisal of an agent's action under its own stated policy (the primary, exploratory signal), set against a global policy-KL divergence kept as a negative control, plus a blind message classifier.
- **Behavioral tier inference**: a logistic regression that recovers an agent's capability tier from its behavior alone.
- **Interaction networks**: the Trust token-transfer graph and a behavioral-similarity graph.

## Requirements

- Python 3.11 and [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) with `llama3.2:3b` pulled, for the weak tier
- API keys for Anthropic, OpenAI, and Google (only needed to run new experiments, not to reproduce the analysis)

## Setup

```bash
git clone https://github.com/markolivaic/asymmetric-society.git
cd asymmetric-society

uv venv --python 3.11
uv sync

cp .env.example .env   # then fill in your API keys
```

## Reproduce the analysis

The scored production dataset is committed (`results/production.db`, 320 games), so the results
and figures reproduce without any API calls:

```bash
# Per-hypothesis results (premiums, deception, tier inference, networks). Read-only, no cost.
uv run python scripts/analyze_production.py

# Regenerate every figure into figures/. Read-only, no cost.
uv run python scripts/make_figures.py

# Test suite (173 tests, no API calls).
uv run pytest
```

The thesis (`thesis/main.tex`) builds with [Tectonic](https://tectonic-typesetting.github.io/):
`tectonic thesis/main.tex`.

## Run a new experiment (spends API budget)

Running fresh games costs real money. A hard cap stops the process before it can exceed the limit
(a 100 EUR ceiling, with an 85 EUR tripwire in the production script; the actual study spent under
6 EUR). Ollama must be running for the weak tier.

```bash
uv run python scripts/run_production.py --print-plan        # show the matrix, no cost
uv run python scripts/run_production.py --seeds 1 --dry-run  # one cheap smoke run
uv run python scripts/run_production.py                      # the full matrix
uv run python scripts/score_production.py                    # score deception + classify messages
```

Every experiment is fully specified by a JSON config and an integer seed, and every API call is
logged to SQLite with its prompt, response, cost, and retries.

## Project structure

```
src/asymmetric_society/
  agents/     BaseAgent + Anthropic / OpenAI / Gemini / Ollama backends
  games/      Public Goods Game, Trust Game
  metrics/    inequality, deception, tier inference, networks, message classifier
  runner/     tournament orchestrator, SQLite logging, budget cap, config schema
scripts/      run the study, score it, analyze it, make figures
experiments/  experiment configs (JSON)
tests/        pytest suite
figures/      the thesis figures (F1-F7)
thesis/       LaTeX source for the thesis
frontend/     a small visualization app for the results
data/         the human deception-validation rating sheet and key
docs/         design notes
results/      output databases and logs (production.db is committed; the rest is gitignored)
```

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT, see [LICENSE](LICENSE).
