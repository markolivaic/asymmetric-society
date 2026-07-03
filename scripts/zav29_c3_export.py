"""C3 export: build the BLIND rating sheet + the (separate) KL key.

Re-runs the per-message Haiku extraction over the comm-ON games to get each
message's per-round KL and surprisal (``deception_validation.export_round_records``),
dumps the full records, then draws a KL-STRATIFIED sample (so the rare high-KL
tail is represented, which a purely random sample would starve) and writes:

  * ``results/zav29_c3_rating_sheet.csv`` -- BLIND. Columns: rating_id, game,
    message, action_taken, history, human_score(empty). No KL/surprisal/tier/
    agent/experiment columns. This is what the human fills in, one pass.
  * ``results/zav29_c3_key.csv`` -- rating_id -> experiment_id, agent_id,
    round_num, kl_round, surprisal_round. NOT opened during rating; joined back
    by scripts/zav29_c3_analyze.py only AFTER the ratings are in.

Paid: ~EUR 0.3 (one Haiku extraction per comm-ON message). Run:
  uv run python scripts/zav29_c3_export.py [--db results/zav29.db] [--n 100] [--seed 0]
"""

from __future__ import annotations

import argparse
import csv
import json
import random

from asymmetric_society.metrics import deception_validation as dv
from asymmetric_society.metrics._llm_scoring import build_scoring_agent
from asymmetric_society.runner import db
from asymmetric_society.runner.budget import BudgetTracker

#: game.name -> (deception family, endowment, action verb) for display.
_GAME_INFO = {
    "public_goods": ("public_goods", 20, "contributed"),
    "trust_game": ("trust_send", 10, "sent"),
}


def _comm_on_experiments(db_path: str) -> list[tuple[str, str]]:
    """Return ``(experiment_id, game_name)`` for communication=ON experiments."""
    con = db.connect(db_path)
    try:
        rows = con.execute("SELECT id, config_json FROM experiments").fetchall()
    finally:
        con.close()
    out: list[tuple[str, str]] = []
    for exp_id, config_json in rows:
        cfg = json.loads(config_json)
        if cfg["game"]["params"].get("communication"):
            out.append((exp_id, cfg["game"]["name"]))
    return out


def _stratified_sample(records: list[dv.RoundRecord], target: int, rng: random.Random):
    """KL-stratified sample of ~``target`` across 5 KL-quantile bins."""
    if len(records) <= target:
        return list(records), [len(records)]
    ordered = sorted(records, key=lambda r: r.kl_round)
    n, bins = len(ordered), 5
    per_bin = target // bins
    sample: list[dv.RoundRecord] = []
    counts: list[int] = []
    for b in range(bins):
        chunk = ordered[b * n // bins:(b + 1) * n // bins]
        k = min(per_bin, len(chunk))
        picked = rng.sample(chunk, k)
        sample.extend(picked)
        counts.append(k)
    return sample, counts


def main() -> int:
    """Export the blind rating sheet and the separate KL key."""
    parser = argparse.ArgumentParser(description="C3 blind rating sheet + key export")
    parser.add_argument("--db", default="results/zav29.db")
    parser.add_argument("--n", type=int, default=100, help="target rated sample size")
    parser.add_argument("--seed", type=int, default=0, help="sampling/shuffle seed")
    parser.add_argument("--budget", type=float, default=5.0)
    parser.add_argument("--sheet", default="results/zav29_c3_rating_sheet.csv")
    parser.add_argument("--key", default="results/zav29_c3_key.csv")
    parser.add_argument("--records", default="results/zav29_round_records.json")
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(override=True)

    db.init_db(args.db)
    budget = BudgetTracker(args.db, limit_eur=args.budget)
    scorer = build_scoring_agent(args.db, budget=budget, agent_id="scorer", tier="classifier", max_tokens=256)

    # 1. Extract per-round KL records over all comm-ON games.
    all_records: list[dv.RoundRecord] = []
    for exp_id, game_name in _comm_on_experiments(args.db):
        family = _GAME_INFO[game_name][0]
        all_records.extend(
            dv.export_round_records(
                args.db, exp_id, game=family, agent=scorer, game_context=f"{game_name} game"
            )
        )
    print(f"Extracted {len(all_records)} KL-scored comm-ON messages.")

    # 2. Dump full records (source of truth, includes KL).
    with open(args.records, "w", encoding="utf-8") as fp:
        json.dump([vars(r) for r in all_records], fp, indent=2)

    # 3. KL-stratified sample + shuffle, then assign opaque rating ids.
    rng = random.Random(args.seed)
    sample, bin_counts = _stratified_sample(all_records, args.n, rng)
    rng.shuffle(sample)

    family_to_game = {"public_goods": "public_goods", "trust_send": "trust_game"}

    # 4. Write the BLIND rating sheet and the separate key.
    with open(args.sheet, "w", encoding="utf-8", newline="") as sf, \
         open(args.key, "w", encoding="utf-8", newline="") as kf:
        sheet = csv.writer(sf)
        key = csv.writer(kf)
        sheet.writerow(["rating_id", "game", "message", "action_taken", "history", "human_score"])
        key.writerow(["rating_id", "experiment_id", "agent_id", "round_num", "kl_round", "surprisal_round"])
        for i, r in enumerate(sample, start=1):
            rating_id = f"m{i:04d}"
            game_name = family_to_game[r.game]
            _family, endowment, verb = _GAME_INFO[game_name]
            action_taken = f"{verb} {int(r.action)} of {endowment}"
            hist = ", ".join(str(int(h)) for h in r.history) or "(first round)"
            sheet.writerow([rating_id, game_name, r.message, action_taken, hist, ""])
            key.writerow([rating_id, r.experiment_id, r.agent_id, r.round_num,
                          f"{r.kl_round:.6f}", f"{r.surprisal_round:.6f}"])

    print(f"Corpus={len(all_records)}  sampled={len(sample)}  bin_counts(low->high KL)={bin_counts}")
    print(f"Blind rating sheet: {args.sheet}")
    print(f"  columns: rating_id, game, message, action_taken, history, human_score  (NO KL/tier/id)")
    print(f"Key (do NOT open while rating): {args.key}")
    print(f"Full records (with KL): {args.records}")
    print(f"Spend in {args.db}: budget-tracked, cap EUR {args.budget}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
