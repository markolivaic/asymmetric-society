"""C3 human-rating pilot sweep: PGG + Trust x communication ON/OFF into one shared DB.

``run_batch`` runs a single config N times; it does NOT sweep conditions. This
wrapper invokes the CLI once per condition (4 configs) with a SHARED results DB
and a SHARED budget tripwire, so the per-DB ``BudgetTracker`` enforces one
cumulative cap across all four conditions (and the later metric passes that log
to the same DB). The same base seed is used for every condition, so condition i
runs games with seeds ``base..base+games-1`` -- paired comm ON/OFF on identical
scaffolding.

This is the PAID run (Haiku in the mixed compositions). Expected ~EUR 0.3-0.6 for
28 games. STOP for review before launching (checkpoint B).

Run:  uv run python scripts/run_zav29.py                       # 7 games/cond = 28
      uv run python scripts/run_zav29.py --games 7 --budget 5.0
"""

from __future__ import annotations

import argparse
import subprocess
import sys

#: (label, config path) per condition, in run order.
CONDITIONS: list[tuple[str, str]] = [
    ("PGG   comm OFF", "experiments/zav29_pgg_comm_off.json"),
    ("PGG   comm ON ", "experiments/zav29_pgg_comm_on.json"),
    ("Trust comm OFF", "experiments/zav29_trust_comm_off.json"),
    ("Trust comm ON ", "experiments/zav29_trust_comm_on.json"),
]


def main() -> int:
    """Run each condition through the CLI with a shared DB and budget tripwire."""
    parser = argparse.ArgumentParser(description="C3 human-rating multi-condition pilot sweep")
    parser.add_argument("--games", type=int, default=7,
                        help="games per condition (default 7 -> 28 total)")
    parser.add_argument("--seed", type=int, default=1000,
                        help="shared base seed (paired across conditions)")
    parser.add_argument("--db", default="results/zav29.db", help="shared results DB")
    parser.add_argument("--budget", type=float, default=5.0,
                        help="shared cumulative EUR tripwire (per-DB)")
    args = parser.parse_args()

    print("=" * 64)
    print(f"C3 HUMAN-RATING PILOT SWEEP  games/cond={args.games}  seed={args.seed}")
    print(f"db={args.db}  budget=EUR {args.budget:.2f}  total games={len(CONDITIONS) * args.games}")
    print("=" * 64)

    for label, cfg in CONDITIONS:
        print(f"\n--- {label}  ({cfg}) ---")
        cmd = [
            sys.executable, "-m", "asymmetric_society", "run", cfg,
            "--games", str(args.games), "--seed", str(args.seed),
            "--db", args.db, "--budget", str(args.budget),
        ]
        returncode = subprocess.run(cmd).returncode
        if returncode != 0:
            print(f"[!] Condition '{label.strip()}' exited {returncode}. Stopping sweep "
                  "(budget tripwire or setup error).", file=sys.stderr)
            return returncode

    print("\n" + "=" * 64)
    print(f"Sweep done. All {len(CONDITIONS) * args.games} games in {args.db}.")
    print("Next (checkpoint C): KL scoring, classifier, tier-inference, premium.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
