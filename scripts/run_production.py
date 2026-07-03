"""Step 3: production experiment harness (the ~320-game matrix).

Builds the full experimental matrix in code and runs it into ONE
``results/production.db`` so the EUR 100 cap is global:

  * PGG (uniform n=4): 5 compositions -- all_strong (4S), all_weak (4W),
    mixed (2S+2W), lone_elite (1S+3W), lone_vulnerable (3S+1W).
  * Trust (dyadic n=2): 3 compositions -- ss, ww, sw (alternating roles).
  * x communication ON/OFF x 20 seeds  ->  5*2*20 + 3*2*20 = 320 games.

The strong tier is ROTATED across {Haiku, GPT-5-mini, Gemini} per slot via
``STRONG[(seed + slot) % 3]`` -- a pure, reproducible, balanced assignment so
"strong" is genuinely diverse (this is what breaks the pilot's Haiku-only
pseudoreplication). Agent ids are anonymous 8-hex (a seeded RNG, so reproducible)
and the agent list is shuffled, so neither id nor position leaks the tier into the
prompt. Communication ON/OFF share the same seed AND the same agent setup, so the
only difference within a seed-pair is the communication flag.

Modes:
  --print-plan          print the matrix + per-model rotation balance; EUR 0, no runs.
  --all-weak            swap every strong slot to Llama (EUR 0 local) -- the
                        STRUCTURAL smoke proving every cell runs and every config
                        validates without spending money (Step-3 gate).
  --dry-run             run ONLY the most expensive cells (all_strong/ss comm-ON)
                        at --seeds, to measure EUR/game before the full run (Step 4).
  --seeds N             seed count (default 20; use 1-2 for smoke/dry-run).
  --gemini-model STR    Gemini model string (default gemini-2.5-flash; override per
                        run, recorded in the run name).
  --budget FLOAT        cumulative EUR cap on production.db (default 85 = the stop tripwire).

Step-3 gate (EUR 0):  uv run python scripts/run_production.py --print-plan
                      uv run python scripts/run_production.py --all-weak --seeds 1
"""

from __future__ import annotations

import argparse
import asyncio
import random
from collections import Counter

from asymmetric_society.agents.base import AgentAPIError
from asymmetric_society.config import AgentConfig, ExperimentConfig, GameConfig
from asymmetric_society.runner import db
from asymmetric_society.runner.budget import BudgetExceededError
from asymmetric_society.runner.tournament import run_experiment

BASE_SEED = 2000
WEAK = ("ollama", "llama3.2:3b")

#: Strong tier, rotated per slot. Gemini string is overridable at run time.
def _strong_pool(gemini_model: str) -> list[tuple[str, str]]:
    return [
        ("anthropic", "claude-haiku-4-5-20251001"),
        ("openai", "gpt-5-mini"),
        ("gemini", gemini_model),
    ]


#: (n_strong, n_weak) per composition.
PGG_COMPS: dict[str, tuple[int, int]] = {
    "all_strong": (4, 0),
    "all_weak": (0, 4),
    "mixed": (2, 2),
    "lone_elite": (1, 3),
    "lone_vulnerable": (3, 1),
}
TRUST_COMPS: dict[str, tuple[int, int]] = {"ss": (2, 0), "ww": (0, 2), "sw": (1, 1)}


def cells() -> list[tuple[str, str, int, int, bool]]:
    """All (game, composition, n_strong, n_weak, communication) cells."""
    out: list[tuple[str, str, int, int, bool]] = []
    for comp, (ns, nw) in PGG_COMPS.items():
        for comm in (False, True):
            out.append(("public_goods", comp, ns, nw, comm))
    for comp, (ns, nw) in TRUST_COMPS.items():
        for comm in (False, True):
            out.append(("trust_game", comp, ns, nw, comm))
    return out


def build_config(
    game: str, comp: str, n_strong: int, n_weak: int, comm: bool, seed: int,
    *, db_path: str, gemini_model: str, all_weak: bool,
) -> ExperimentConfig:
    """Construct one fully-specified, reproducible experiment config for a cell+seed."""
    # RNG keyed on (game, comp, seed) ONLY (not comm) so a seed's comm-ON and
    # comm-OFF games get identical agents -- a clean paired comparison.
    rng = random.Random(f"{game}|{comp}|{seed}")
    pool = _strong_pool(gemini_model)
    used: set[str] = set()

    def hexid() -> str:
        while True:
            x = "".join(rng.choice("0123456789abcdef") for _ in range(8))
            if x not in used:
                used.add(x)
                return x

    agents: list[AgentConfig] = []
    for slot in range(n_strong):
        if all_weak:
            backend, model, tier = (*WEAK, "weak")
        else:
            backend, model = pool[(seed + slot) % len(pool)]
            tier = "strong"
        agents.append(AgentConfig(backend=backend, model_name=model, capability_tier=tier,
                                  temperature=1.0, agent_id=hexid()))
    for _ in range(n_weak):
        agents.append(AgentConfig(backend=WEAK[0], model_name=WEAK[1], capability_tier="weak",
                                  temperature=1.0, agent_id=hexid()))
    rng.shuffle(agents)  # list position must not correlate with tier

    if game == "public_goods":
        params: dict[str, object] = {
            "n": n_strong + n_weak, "endowment": 20, "multiplier": 1.6,
            "rounds": 10, "communication": comm,
        }
    else:
        params = {"endowment": 10, "multiplier": 3, "rounds": 10,
                  "roles": "alternating", "communication": comm}

    tag = "aw" if all_weak else "prod"
    name = f"{tag}-{game[:4]}-{comp}-{'on' if comm else 'off'}-s{seed}"
    return ExperimentConfig(name=name, seed=seed, rounds=10, db_path=db_path,
                            game=GameConfig(name=game, params=params), agents=agents)


def plan(seeds: list[int], gemini_model: str, all_weak: bool, db_path: str) -> list[ExperimentConfig]:
    """Build every config for the chosen seeds (full matrix)."""
    configs: list[ExperimentConfig] = []
    for game, comp, ns, nw, comm in cells():
        for seed in seeds:
            configs.append(build_config(game, comp, ns, nw, comm, seed,
                                        db_path=db_path, gemini_model=gemini_model, all_weak=all_weak))
    return configs


def print_plan(seeds: list[int], gemini_model: str) -> None:
    """Print the matrix, game count, and per-strong-model rotation balance (EUR 0)."""
    configs = plan(seeds, gemini_model, all_weak=False, db_path="(none)")
    print(f"Matrix: {len(cells())} cells x {len(seeds)} seeds = {len(configs)} games")
    print(f"  PGG compositions:   {list(PGG_COMPS)}")
    print(f"  Trust compositions: {list(TRUST_COMPS)}")
    model_games: Counter[str] = Counter()
    model_slots: Counter[str] = Counter()
    for cfg in configs:
        seen = set()
        for a in cfg.agents:
            if a.capability_tier == "strong":
                model_slots[a.model_name] += 1
                seen.add(a.model_name)
        for m in seen:
            model_games[m] += 1
    print("  Strong-model rotation balance (slots filled across the run):")
    for m, n in model_slots.most_common():
        print(f"    {m:<30} {n:>4} slots   ({model_games[m]} games)")


async def run_all(configs: list[ExperimentConfig], budget_cap: float) -> tuple[int, int]:
    """Run every config into the shared DB; resilient per game, hard-stop on budget."""
    db.init_db(configs[0].db_path)
    done = failed = 0
    for i, cfg in enumerate(configs, 1):
        try:
            await run_experiment(cfg, budget_limit_eur=budget_cap)
            done += 1
            print(f"  [{i}/{len(configs)}] ok   {cfg.name}")
        except BudgetExceededError as exc:
            print(f"  BUDGET CAP HIT -> stopping: {exc}")
            break
        except (AgentAPIError, Exception) as exc:  # noqa: BLE001 - isolate one game
            failed += 1
            print(f"  [{i}/{len(configs)}] SKIP {cfg.name}: {type(exc).__name__}: {str(exc)[:80]}")
    return done, failed


def _resume_filter(configs: list[ExperimentConfig], db_path: str) -> list[ExperimentConfig]:
    """For --resume: purge partial (unfinished) experiments and drop already-finished cells.

    A finished experiment has ``end_time`` set; its config ``name`` (cell+seed) is the
    idempotency key. Partial games (process died mid-run) are deleted across all tables
    so they re-run cleanly with no duplicate rows.
    """
    import json

    con = db.connect(db_path)
    try:
        partial = [r[0] for r in con.execute(
            "SELECT id FROM experiments WHERE end_time IS NULL").fetchall()]
        for eid in partial:
            for table in ("agents", "rounds", "actions", "payoffs", "llm_calls"):
                con.execute(f"DELETE FROM {table} WHERE experiment_id = ?", (eid,))
            con.execute("DELETE FROM experiments WHERE id = ?", (eid,))
        con.commit()
        finished = {
            json.loads(r[0])["name"]
            for r in con.execute(
                "SELECT config_json FROM experiments WHERE end_time IS NOT NULL").fetchall()
        }
    finally:
        con.close()
    remaining = [c for c in configs if c.name not in finished]
    print(f"RESUME: {len(finished)} finished kept, {len(partial)} partial purged, "
          f"{len(remaining)} games remaining")
    return remaining


def main() -> int:
    """Build the matrix and run the chosen mode."""
    parser = argparse.ArgumentParser(description="production experiment harness")
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--all-weak", action="store_true", help="swap strong->Llama (EUR 0 smoke)")
    parser.add_argument("--dry-run", action="store_true", help="run only the expensive cells")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--gemini-model", default="gemini-2.5-flash")
    parser.add_argument("--budget", type=float, default=85.0)
    parser.add_argument("--db", default="results/production.db")
    parser.add_argument("--resume", action="store_true",
                        help="skip cells already finished in --db; purge any partial (unfinished) games")
    args = parser.parse_args()

    seeds = [BASE_SEED + i for i in range(args.seeds)]

    if args.print_plan:
        print_plan(seeds, args.gemini_model)
        return 0

    from dotenv import load_dotenv

    load_dotenv(override=True)

    if args.dry_run:
        # The two most expensive cells (all-strong, comm-ON) for a EUR/game estimate.
        configs = []
        for game, comp, ns, nw, comm in cells():
            if comm and comp in ("all_strong", "ss"):
                for seed in seeds:
                    configs.append(build_config(game, comp, ns, nw, comm, seed,
                                                db_path=args.db, gemini_model=args.gemini_model, all_weak=False))
        label = "DRY-RUN (expensive cells)"
    else:
        configs = plan(seeds, args.gemini_model, all_weak=args.all_weak, db_path=args.db)
        if args.resume:
            configs = _resume_filter(configs, args.db)
        label = (
            "ALL-WEAK structural smoke (EUR 0)" if args.all_weak
            else "RESUME production" if args.resume
            else "PRODUCTION RUN"
        )
        if not configs:
            print("Nothing to run: every cell already finished.")
            return 0

    print("=" * 64)
    print(f"{label}: {len(configs)} games -> {args.db}  (budget cap EUR {args.budget})")
    print(f"Gemini string: {args.gemini_model}")
    print("=" * 64)
    done, failed = asyncio.run(run_all(configs, args.budget))

    con = db.connect(args.db)
    try:
        total = con.execute("SELECT COALESCE(SUM(cost_eur), 0.0) FROM llm_calls").fetchone()[0]
    finally:
        con.close()
    print("=" * 64)
    print(f"Completed {done}/{len(configs)} games ({failed} failed/skipped). Spend so far: EUR {total:.4f}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
