# C3 deception-validation dataset (ZAV-29)

The single most important validation artifact in the thesis: the **blind human
deception ratings** that the KL deception metric (ZAV-22) was tested against, plus
the metric key. Tracked here (not in gitignored `results/`) because it is thesis
appendix material and the evidence that C3 was actually carried out.

## Files

- **`zav29_c3_rating_sheet.csv`** - 100 messages from the comm-ON pilot games,
  rated 0–3 for deception by the student (single human rater), BLIND: the sheet
  carries only `rating_id, game, message, action_taken, history, human_score` - no
  KL/surprisal/tier/agent/experiment columns. KL-stratified sample of the 350
  comm-ON messages (20 per KL-quintile) so the rare high-KL tail is represented.
- **`zav29_c3_key.csv`** - `rating_id -> experiment_id, agent_id, round_num,
  kl_round, surprisal_round`. Held separate from the sheet during rating (the
  blindness guarantee); joined back only by `scripts/zav29_c3_analyze.py` AFTER the
  ratings were in.

## Protocol & result (see `docs/zav29_c3_rubric.md`, committed before any rating)

- Rubric FROZEN and committed before rating; threshold **pre-registered at Spearman
  ρ ≥ 0.40 & p < 0.05** and **NOT moved** despite the outcome.
- Human distribution: 92×0, 6×1, 2×2, 0×3 (real deception is rare).
- Result: global `kl_headline` NOT validated (ρ = −0.01, p = 0.90; misses both clear
  deceptions). Contemporaneous **surprisal** correlates weakly but significantly
  (ρ = +0.22, p = 0.028) - below the PASS threshold, so **exploratory**. The headline
  is the gap: contemporaneous surprisal >> global policy-KL for tracking human-
  perceived deception.
- Caveats: underpowered (2/100 clearly deceptive), single rater = the metric's
  author (mitigated: blind, frozen rubric, one pass), pilot scale, pseudoreplication.

Reproduce: `uv run python scripts/zav29_c3_analyze.py --sheet data/zav29_c3_rating_sheet.csv --key data/zav29_c3_key.csv`
