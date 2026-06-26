"""ZAV-29 / C3 analysis: correlate blind human ratings with KL and surprisal.

Run AFTER the human has filled ``human_score`` (0-3) in the rating sheet. Joins
the sheet to the KL key by ``rating_id`` and reports, for BOTH the per-round KL
and the contemporaneous surprisal: Spearman rho (permutation p, bootstrap CI) and
the binary-deception ROC-AUC (human >= 2), against the pre-registered PASS
threshold in ``docs/zav29_c3_rubric.md`` (rho >= 0.40 & p < 0.05; HOLD THE LINE).

Run:  uv run python scripts/zav29_c3_analyze.py [--sheet ...] [--key ...]
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter

from asymmetric_society.metrics.deception_validation import correlate


def _load(sheet_path: str, key_path: str) -> tuple[list[float], list[float], list[float], int]:
    """Join rated rows to the KL key by rating_id; return human/kl/surprisal + n_unrated."""
    with open(key_path, encoding="utf-8") as kf:
        keys = {row["rating_id"]: row for row in csv.DictReader(kf)}
    human: list[float] = []
    kl: list[float] = []
    surp: list[float] = []
    n_unrated = 0
    with open(sheet_path, encoding="utf-8") as sf:
        for row in csv.DictReader(sf):
            score = (row.get("human_score") or "").strip()
            if score == "":
                n_unrated += 1
                continue
            k = keys.get(row["rating_id"])
            if k is None:
                continue
            human.append(float(score))
            kl.append(float(k["kl_round"]))
            surp.append(float(k["surprisal_round"]))
    return human, kl, surp, n_unrated


def _report(name: str, res: dict) -> None:
    rho = res["spearman_rho"]
    lo, hi = res["rho_ci"]
    print(f"\n  {name}:")
    print(f"    Spearman rho = {rho:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  perm p = {res['perm_p']:.4f}")
    if res["auc_na"]:
        print(f"    ROC-AUC = n/a ({res['auc_na']})")
    else:
        print(f"    ROC-AUC (human>=2) = {res['auc']:.3f}  (deceptive n={res['n_deceptive']}/{res['n']})")
    is_pass = (not math.isnan(rho)) and rho >= 0.40 and res["perm_p"] < 0.05
    if is_pass:
        verdict = "PASS (rho>=0.40 & p<0.05)"
    elif not math.isnan(rho) and rho >= 0.20:
        verdict = "below PASS -> exploratory (rho 0.20-0.40; do NOT move the threshold)"
    else:
        verdict = "not validated at pilot scale (rho<0.20 or n.s.)"
    print(f"    -> {verdict}")


def main() -> int:
    """Load ratings, join to the KL key, and report the C3 validation."""
    parser = argparse.ArgumentParser(description="ZAV-29 C3 correlation analysis")
    parser.add_argument("--sheet", default="results/zav29_c3_rating_sheet.csv")
    parser.add_argument("--key", default="results/zav29_c3_key.csv")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    human, kl, surp, n_unrated = _load(args.sheet, args.key)

    print("=" * 64)
    print("ZAV-29 C3 -- KL deception validation vs BLIND human ratings")
    print("=" * 64)
    print(f"rated messages: {len(human)}  (unrated/skipped: {n_unrated})")
    if not human:
        print("[!] no rated rows found -- fill human_score (0-3) in the sheet first.")
        return 1
    if len(human) < 40:
        print("[!] fewer than ~40 rated messages -- Spearman underpowered (rubric floor).")
    print(f"human score distribution: {dict(sorted(Counter(human).items()))}")

    _report("per-round KL (global-flavoured)", correlate(human, kl, seed=args.seed))
    _report("surprisal (contemporaneous)", correlate(human, surp, seed=args.seed))

    print("\nNote: the rubric construct is contemporaneous (message vs THIS round's")
    print("action), which matches surprisal. If surprisal out-correlates the global")
    print("per-round KL, that gap is itself the headline methodological finding.")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
