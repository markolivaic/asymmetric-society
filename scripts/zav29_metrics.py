"""ZAV-29 checkpoint-C metrics driver: KL + classifier + tier-inference + premium.

Runs the EXISTING metric passes over ``results/zav29.db`` (the pilot produced by
``scripts/run_zav29.py``) and prints the checkpoint-C report:

  * KL / surprisal distribution on real comm-ON messages, with a degeneracy flag
    (the cheap sanity before investing in C3 hand-rating: are KL values spread,
    not all ~0 and not all saturated ~6.9?);
  * message-class label distribution (is it sensible, not all one label?);
  * Bayesian tier-inference balanced-accuracy / ROC-AUC (honest gate vs 0.5);
  * capability premium per condition (mean strong payoff - mean weak payoff);
  * a compact comm-ON transcript per game for eyeballing; and the Haiku spend.

Paid: KL extraction + classification call Haiku (~EUR 0.3-0.5). All calls log to
the same DB, so the shared budget tripwire applies. Sequential, so allow ~15-20 min.

Run:  uv run python scripts/zav29_metrics.py [--db results/zav29.db] [--budget 5.0]
"""

from __future__ import annotations

import argparse
import json
import math

from asymmetric_society.metrics import bayesian, classifier, deception, inequality
from asymmetric_society.metrics._llm_scoring import build_scoring_agent
from asymmetric_society.runner import db, results
from asymmetric_society.runner.budget import BudgetTracker

#: game.name -> deception score family (selects action support + action field).
_DECEPTION_GAME = {"public_goods": "public_goods", "trust_game": "trust_send"}


def _experiments_by_condition(db_path: str) -> dict[tuple[str, bool], list[str]]:
    """Group experiment ids by ``(game_name, communication)`` from config_json."""
    con = db.connect(db_path)
    try:
        rows = con.execute(
            "SELECT id, config_json FROM experiments ORDER BY start_time"
        ).fetchall()
    finally:
        con.close()
    groups: dict[tuple[str, bool], list[str]] = {}
    for exp_id, config_json in rows:
        cfg = json.loads(config_json)
        key = (cfg["game"]["name"], bool(cfg["game"]["params"].get("communication", False)))
        groups.setdefault(key, []).append(exp_id)
    return groups


def _dist_summary(values: list[float]) -> tuple[str, list[float]]:
    """Return a human summary string and the cleaned (non-NaN) sorted values."""
    vals = sorted(v for v in values if not math.isnan(v))
    if not vals:
        return "no (non-NaN) values", []
    n = len(vals)

    def q(p: float) -> float:
        return vals[min(n - 1, int(p * n))]

    summary = (
        f"n={n}  min={vals[0]:.3f}  q25={q(0.25):.3f}  median={q(0.5):.3f}  "
        f"q75={q(0.75):.3f}  max={vals[-1]:.3f}  mean={sum(vals) / n:.3f}"
    )
    return summary, vals


def _degeneracy_verdict(vals: list[float], *, saturate: float = 6.5) -> str:
    """Flag whether a KL/surprisal distribution carries usable spread for C3."""
    if len(vals) < 2:
        return "WARNING: too few values to judge"
    spread = vals[-1] - vals[0]
    near_zero = sum(1 for v in vals if v < 0.05) / len(vals)
    saturated = sum(1 for v in vals if v > saturate) / len(vals)
    if spread < 0.1:
        return f"DEGENERATE: spread={spread:.3f} (all clustered) -> C3 rating not worthwhile"
    if near_zero > 0.95:
        return f"DEGENERATE: {near_zero:.0%} near 0 -> metric separates nothing"
    if saturated > 0.95:
        return f"DEGENERATE: {saturated:.0%} saturated (>{saturate}) -> all maxed out"
    return (f"NON-DEGENERATE: spread={spread:.3f}, {near_zero:.0%} near-0, "
            f"{saturated:.0%} saturated -> usable range for C3")


def _print_transcript(db_path: str, experiment_id: str, label: str, max_rounds: int = 6) -> None:
    """Compact message+action transcript for one game (eyeball signal)."""
    rows = results.game_transcript(db_path, experiment_id)
    print(f"\n  [{label}] experiment {experiment_id[:8]}")
    shown = 0
    current = None
    for row in rows:
        if row["round_num"] != current:
            current = row["round_num"]
            shown += 1
            if shown > max_rounds:
                break
        action = next(
            (f"{k}={row[k]}" for k in ("contribution", "send", "return") if row.get(k) is not None),
            "noop",
        )
        msg = (row["message"] or "")[:90]
        print(f"    r{row['round_num']:<2} {row['agent_id']}({row['capability_tier'][:4]}) "
              f"{action:<14} msg={msg!r}")


def main() -> int:
    """Run the checkpoint-C metric passes and print the report."""
    parser = argparse.ArgumentParser(description="ZAV-29 checkpoint-C metrics")
    parser.add_argument("--db", default="results/zav29.db")
    parser.add_argument("--budget", type=float, default=5.0, help="EUR tripwire (shared per-DB)")
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(override=True)

    db.init_db(args.db)
    budget = BudgetTracker(args.db, limit_eur=args.budget)
    scorer = build_scoring_agent(
        args.db, budget=budget, agent_id="scorer", tier="classifier", max_tokens=256
    )
    clf = classifier.build_classifier_agent(args.db, budget=budget)

    groups = _experiments_by_condition(args.db)
    all_ids = [e for ids in groups.values() for e in ids]
    pgg_ids = [e for (g, _c), ids in groups.items() if g == "public_goods" for e in ids]
    comm_on = {g: ids for (g, c), ids in groups.items() if c}

    print("=" * 70)
    print("ZAV-29 CHECKPOINT C -- metrics on real comm-ON data")
    print("=" * 70)
    for (g, c), ids in sorted(groups.items()):
        print(f"  {g:<13} comm={'ON ' if c else 'OFF'}  games={len(ids)}")
    print(f"  TOTAL games={len(all_ids)}")

    # --- 1. KL deception (comm-ON games) ---
    print("\n--- 1. KL deception (comm-ON games; Haiku extraction) ---")
    kls: list[float] = []
    surps: list[float] = []
    for game_name, ids in comm_on.items():
        fam = _DECEPTION_GAME[game_name]
        ctx = f"{game_name} game"
        for exp in ids:
            for r in deception.score_game(args.db, exp, game=fam, agent=scorer, game_context=ctx):
                kls.append(r.kl_headline)
                surps.append(r.surprisal)
    kl_summary, kl_vals = _dist_summary(kls)
    sp_summary, sp_vals = _dist_summary(surps)
    print(f"  kl_headline : {kl_summary}")
    print(f"               {_degeneracy_verdict(kl_vals)}")
    print(f"  surprisal   : {sp_summary}")
    print(f"               {_degeneracy_verdict(sp_vals)}")

    # --- 2. Message classification (comm-ON) ---
    print("\n--- 2. Message classification (comm-ON; Haiku) ---")
    n_classified = 0
    for game_name, ids in comm_on.items():
        n_classified += classifier.batch_classify(
            args.db, ids, agent=clf, game_context=f"{game_name} game"
        )
    con = db.connect(args.db)
    try:
        label_rows = con.execute(
            "SELECT label, COUNT(*) FROM message_classifications GROUP BY label ORDER BY 2 DESC"
        ).fetchall()
    finally:
        con.close()
    print(f"  classified {n_classified} messages; label distribution:")
    for label, count in label_rows:
        print(f"    {label:<20} {count}")

    # --- 3. Tier-inference (PGG, all comm conditions; action-only features) ---
    print("\n--- 3. Bayesian tier-inference (PGG, 5 action features) ---")
    try:
        res = bayesian.infer_from_db(args.db, pgg_ids, game="public_goods")
        print(f"  n={res['n']}  strong={res['n_strong']}  weak={res['n_weak']}  "
              f"majority_baseline={res['majority_baseline']:.3f}")
        print(f"  accuracy={res['accuracy']:.3f}  balanced_accuracy={res['balanced_accuracy']:.3f}  "
              f"roc_auc={res['roc_auc']:.3f}")
        print("  gate (honest): balanced_accuracy & ROC-AUC vs chance 0.50")
        coeffs = res["coefficients"]
        if not isinstance(coeffs, dict):
            coeffs = dict(zip(res["feature_names"], coeffs))
        print("  coefficients: " + "  ".join(f"{k}={float(v):+.2f}" for k, v in coeffs.items()))
    except Exception as exc:  # noqa: BLE001 - report, don't crash the whole report
        print(f"  [!] tier-inference failed: {exc}")

    # --- 4. Capability premium (per condition) ---
    print("\n--- 4. Capability premium (mean strong - mean weak final payoff) ---")
    for (g, c), ids in sorted(groups.items()):
        rows = results.final_payoffs(args.db, ids)
        strong = [r["cumulative_payoff"] for r in rows if r["capability_tier"] == "strong"]
        weak = [r["cumulative_payoff"] for r in rows if r["capability_tier"] == "weak"]
        if strong and weak:
            prem = inequality.capability_premium(strong, weak)
            print(f"  {g:<13} comm={'ON ' if c else 'OFF'}  "
                  f"strong_mean={sum(strong) / len(strong):.1f}  "
                  f"weak_mean={sum(weak) / len(weak):.1f}  premium={prem:+.1f}")
        else:
            print(f"  {g:<13} comm={'ON ' if c else 'OFF'}  (missing a tier)")

    # --- 5. Sample transcripts (one comm-ON game per family) ---
    print("\n--- 5. Sample comm-ON transcripts (first 6 rounds) ---")
    for game_name, ids in comm_on.items():
        if ids:
            _print_transcript(args.db, ids[0], f"{game_name} comm-ON")

    # --- 6. Spend ---
    con = db.connect(args.db)
    try:
        total = con.execute("SELECT COALESCE(SUM(cost_eur), 0.0) FROM llm_calls").fetchone()[0]
        n_msgs = con.execute(
            "SELECT COUNT(*) FROM actions WHERE message IS NOT NULL AND TRIM(message) <> ''"
        ).fetchone()[0]
    finally:
        con.close()
    print("\n" + "=" * 70)
    print(f"Total messages persisted: {n_msgs}")
    print(f"Total spend in {args.db}: EUR {total:.4f}  (budget tripwire EUR {args.budget})")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
