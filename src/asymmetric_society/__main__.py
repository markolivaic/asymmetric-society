"""Command-line entry point: run an experiment config for N games and report.

Usage:
    python -m asymmetric_society run <config.json> --games N [--seed S] [--db PATH]

Runs the config ``--games`` times (sequentially), then prints the per-tier
parse-failure table with the gate verdict (attempt_fail_rate < 0.15), one full
game transcript, and any raw parse-failure samples. Intended for the ZAV-20
pilot; the all_weak config runs entirely on local Ollama at zero cost.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import socket
import sys
import time

from dotenv import load_dotenv

from asymmetric_society.config import ExperimentConfig, load_experiment
from asymmetric_society.runner import results
from asymmetric_society.runner.batch import run_batch

GATE = 0.15


def _probe_ollama(host: str = "127.0.0.1", port: int = 11434, timeout: float = 2.0) -> bool:
    """Return True if something is listening on the Ollama port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _print_parse_table(stats: list[dict]) -> dict | None:
    """Print the per-tier parse-failure table; return the weak-tier row if any."""
    print("\n=== Parse-failure rates (per tier) ===")
    print(f"{'tier':<8}{'calls':>7}{'attempt_fail':>14}{'fallback':>11}{'any_fail':>10}")
    weak = None
    for row in stats:
        print(
            f"{row['capability_tier']:<8}{row['calls']:>7}"
            f"{row['attempt_fail_rate'] or 0.0:>14.3f}"
            f"{row['fallback_rate'] or 0.0:>11.3f}"
            f"{row['any_parse_fail_rate'] or 0.0:>10.3f}"
        )
        if row["capability_tier"] == "weak":
            weak = row
    return weak


def _print_gate(weak: dict | None) -> None:
    print("\n=== GATE (weak-tier attempt_fail_rate < 0.15) ===")
    if weak is None:
        print("  no weak-tier calls found - cannot evaluate gate")
        return
    rate = weak["attempt_fail_rate"] or 0.0
    verdict = "PASS" if rate < GATE else "FAIL"
    print(f"  attempt_fail_rate = {rate:.3f}  ->  {verdict}")
    if rate >= GATE:
        print("  >= 0.15: STOP, no paid spend. Inspect failure samples below before any pivot.")
    elif rate >= 0.12:
        print("  borderline (>=0.12): inspect failure samples; likely a cheap prompt fix.")


def _print_transcript(db_path: str, experiment_id: str) -> None:
    rows = results.game_transcript(db_path, experiment_id)
    print(f"\n=== Sample transcript (experiment {experiment_id}) ===")
    current = None
    for row in rows:
        if row["round_num"] != current:
            current = row["round_num"]
            print(f"-- round {current} --")
        msg = f'  msg="{row["message"]}"' if row["message"] else ""
        flag = "  [FALLBACK]" if row["fallback_used"] else ""
        payoff = row["round_payoff"]
        payoff_str = f"{payoff:.1f}" if payoff is not None else "?"
        fields = [
            f"{name}={row[name]}"
            for name in ("contribution", "send", "return")
            if row.get(name) is not None
        ]
        action_str = " ".join(fields) if fields else "noop"
        print(
            f"  {row['agent_id']} ({row['capability_tier']}): "
            f"{action_str:<16} payoff={payoff_str}{msg}{flag}"
        )


def _print_failures(db_path: str, experiment_ids: list[str]) -> None:
    examples = results.parse_failure_examples(db_path, experiment_ids, limit=10)
    if not examples:
        print("\n=== Parse-failure samples: none ===")
        return
    print(f"\n=== Parse-failure samples ({len(examples)} shown) ===")
    for ex in examples:
        raw = (ex["raw_response"] or "")[:300]
        print(
            f"  [{ex['capability_tier']} {ex['agent_id']} r{ex['round_no']} "
            f"fails={ex['parse_fail_count']} fallback={ex['fallback_used']}]"
        )
        print(f"    raw: {raw!r}")


def _cmd_run(args: argparse.Namespace) -> int:
    # .env is the source of truth for this research tool; override empty/stale
    # shell env vars (e.g. an empty ANTHROPIC_API_KEY exported by the OS would
    # otherwise shadow the real key in .env). Conscious decision, see Decision Log.
    load_dotenv(override=True)
    config: ExperimentConfig = load_experiment(args.config)
    updates: dict = {}
    if args.db:
        updates["db_path"] = args.db
    if args.seed is not None:
        updates["seed"] = args.seed
    if updates:
        config = config.model_copy(update=updates)

    if any(a.backend == "ollama" for a in config.agents) and not _probe_ollama():
        print("ERROR: Ollama server not reachable at 127.0.0.1:11434. "
              "Start it (`ollama serve`) and ensure the model is pulled.", file=sys.stderr)
        return 1
    if any(a.backend == "anthropic" for a in config.agents) and not os.getenv("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY not set (checked env and .env). "
              "Cannot run paid Anthropic agents.", file=sys.stderr)
        return 1

    budget_note = f", budget cap=€{args.budget:.2f}" if args.budget is not None else ""
    print(f"Running '{config.name}' x{args.games} game(s), seed base={config.seed}, "
          f"db={config.db_path}{budget_note}")

    def _on_done(i: int, experiment_id: str, elapsed: float) -> None:
        print(f"  game {i + 1}/{args.games} done in {elapsed:.1f}s  ({experiment_id})")

    failures: list[int] = []

    def _on_failed(i: int, error: str) -> None:
        failures.append(i)
        print(f"  game {i + 1}/{args.games} FAILED (API error): {error[:120]} - continuing",
              file=sys.stderr)

    start = time.perf_counter()
    experiment_ids = asyncio.run(
        run_batch(
            config, args.games, base_seed=config.seed,
            budget_limit_eur=args.budget,
            on_game_complete=_on_done, on_game_failed=_on_failed,
        )
    )
    total = time.perf_counter() - start

    failed_note = f" ({len(failures)} failed and skipped)" if failures else ""
    print(f"\nCompleted {len(experiment_ids)}/{args.games} games in {total:.1f}s "
          f"({total / max(len(experiment_ids), 1):.1f}s/game){failed_note}.")
    print(f"Total spend: €{results.total_cost(config.db_path, experiment_ids):.4f}")

    stats = results.parse_failure_stats(config.db_path, experiment_ids)
    weak = _print_parse_table(stats)
    _print_gate(weak)
    if experiment_ids:
        _print_transcript(config.db_path, experiment_ids[0])
    _print_failures(config.db_path, experiment_ids)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="asymmetric_society")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="Run an experiment config for N games.")
    run.add_argument("config", help="Path to the experiment JSON config.")
    run.add_argument("--games", type=int, default=10, help="Number of games (default 10).")
    run.add_argument("--seed", type=int, default=None, help="Override base seed.")
    run.add_argument("--db", default=None, help="Override config db_path.")
    run.add_argument("--budget", type=float, default=None,
                     help="Spend cap in EUR for this run (tripwire on paid backends).")
    run.set_defaults(func=_cmd_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
