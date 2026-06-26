"""Parallel + resumable production scoring for H4 (KL/surprisal + classification).

The serial scorer (zav29_metrics) would take ~3-4h on 160 comm-ON games and a
sleep would lose all of it. This driver:

  * RESUMES - skips experiments already in ``deception_scores`` (KL) /
    ``message_classifications`` (classify), so a sleep/kill loses only in-flight
    games; re-running finishes the rest with no re-pay for completed ones.
  * PARALLELIZES the Haiku I/O with a thread pool (each thread its own scoring
    agents; the budget object is shared so the EUR 85 cap still applies). SQLite
    is put in WAL mode so concurrent small writes (per-call llm_calls log,
    per-agent deception_scores) don't block each other.

Surprisal is the C3-validated deception signal; global ``kl_headline`` is the
negative control. EUR-tracked; expected ~EUR 2-3, ~30-45 min.

Run:  uv run python scripts/score_production.py [--db results/production.db] [--workers 6]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sqlite3
import threading

from asymmetric_society.metrics import classifier, deception
from asymmetric_society.metrics._llm_scoring import build_scoring_agent
from asymmetric_society.runner import db
from asymmetric_society.runner.budget import BudgetExceededError, BudgetTracker

_FAM = {"public_goods": "public_goods", "trust_game": "trust_send"}


def _comm_on(db_path: str) -> list[tuple[str, str]]:
    con = db.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            "SELECT id, config_json FROM experiments WHERE end_time IS NOT NULL"
        ).fetchall()
    finally:
        con.close()
    out = []
    for r in rows:
        cfg = json.loads(r["config_json"])
        if cfg["game"]["params"].get("communication"):
            out.append((r["id"], cfg["game"]["name"]))
    return out


def _distinct(db_path: str, table: str) -> set[str]:
    con = db.connect(db_path)
    try:
        return {
            r[0] for r in con.execute(
                f"SELECT DISTINCT experiment_id FROM {table} WHERE experiment_id IS NOT NULL"
            ).fetchall()
        }
    finally:
        con.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Parallel resumable production scoring (H4)")
    parser.add_argument("--db", default="results/production.db")
    parser.add_argument("--budget", type=float, default=85.0)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(override=True)
    db.init_db(args.db)

    # WAL: concurrent readers + queued small writers without "database is locked".
    con = sqlite3.connect(args.db)
    con.execute("PRAGMA journal_mode=WAL")
    con.close()

    budget = BudgetTracker(args.db, limit_eur=args.budget)
    tls = threading.local()

    def agents():
        if not hasattr(tls, "scorer"):
            tls.scorer = build_scoring_agent(
                args.db, budget=budget, agent_id="scorer", tier="classifier", max_tokens=256)
            tls.clf = classifier.build_classifier_agent(args.db, budget=budget)
        return tls.scorer, tls.clf

    comm_on = _comm_on(args.db)
    kl_todo = [(e, g) for (e, g) in comm_on if e not in _distinct(args.db, "deception_scores")]
    cl_todo = [(e, g) for (e, g) in comm_on if e not in _distinct(args.db, "message_classifications")]
    print(f"comm-ON games={len(comm_on)}  KL todo={len(kl_todo)} (skip {len(comm_on)-len(kl_todo)})  "
          f"classify todo={len(cl_todo)} (skip {len(comm_on)-len(cl_todo)})")

    def do_kl(item):
        e, g = item
        scorer, _ = agents()
        deception.score_game(args.db, e, game=_FAM[g], agent=scorer, game_context=f"{g} game")

    def do_cl(item):
        e, g = item
        _, clf = agents()
        classifier.batch_classify(args.db, [e], agent=clf, game_context=f"{g} game")

    def run_phase(name, fn, items):
        done = fail = 0
        with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(fn, it): it for it in items}
            for fut in cf.as_completed(futs):
                try:
                    fut.result()
                    done += 1
                except BudgetExceededError:
                    print(f"  [{name}] BUDGET CAP -> stopping")
                    break
                except Exception as exc:  # noqa: BLE001 - isolate one game; resume retries it
                    fail += 1
                    print(f"  [{name}] SKIP {futs[fut][0][:8]}: {type(exc).__name__}: {str(exc)[:60]}")
                if (done + fail) % 20 == 0:
                    print(f"  [{name}] {done+fail}/{len(items)}  (EUR {budget.spent_eur:.3f})")
        print(f"  [{name}] done={done} failed={fail}")

    print("=== KL/surprisal scoring ===")
    run_phase("KL", do_kl, kl_todo)
    print("=== message classification ===")
    run_phase("classify", do_cl, cl_todo)
    print(f"TOTAL spend now: EUR {budget.spent_eur:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
