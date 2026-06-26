"""ZAV-5 / Phase 4: production analysis -> H1-H5 results on results/production.db.

Reads the 320-game production DB and produces the per-hypothesis report. The paid
scoring (KL/surprisal in ``deception_scores``, labels in ``message_classifications``)
is written by ``scripts/zav29_metrics.py``; the deception/classification sections
here read those tables and degrade gracefully if scoring hasn't finished yet.

Hypotheses (operationalized from RQ1-3 + the C3 pivot; cross-check vs osf.io/rnqgs):
  H1  capability asymmetry -> welfare inequality (mixed/asym Gini > homogeneous;
      capability premium != 0 in mixed-tier games)
  H2  communication modulates the exploitation effect (premium/Gini ON vs OFF)
  H3  the pattern generalizes across games (PGG vs Trust)
  H4  deception: contemporaneous surprisal tracks deception (validated, C3);
      global policy-KL is the negative control (NOT validated)
  H5  capability tier is inferable from behaviour (full strong tier, not pilot)

EUR 0 (read-only). Run:  uv run python scripts/analyze_production.py
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import warnings

import numpy as np
import pandas as pd

from asymmetric_society.metrics import bayesian, inequality, network
from asymmetric_society.runner import db

warnings.filterwarnings("ignore")

_MODEL_SHORT = {
    "claude-haiku-4-5-20251001": "haiku",
    "gpt-5-mini": "gpt5mini",
    "gemini-2.5-flash": "gemini",
    "llama3.2:3b": "llama",
}
# Compositions where both tiers are present -> the capability-premium test set.
_MIXED_COMPS = {"mixed", "lone_elite", "lone_vulnerable", "sw"}


def _bootstrap_mean(values, n: int = 10000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of ``values`` with a percentile bootstrap 95% CI."""
    v = np.asarray([x for x in values if x == x], dtype=float)
    if v.size == 0:
        return (float("nan"),) * 3
    if v.size == 1:
        return (float(v[0]),) * 3
    rng = np.random.default_rng(seed)
    boots = rng.choice(v, size=(n, v.size), replace=True).mean(axis=1)
    return float(v.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def load_frame(db_path: str) -> pd.DataFrame:
    """Agent-level rows: experiment, game, composition, comm, tier, model, payoff."""
    con = db.connect(db_path)
    con.row_factory = sqlite3.Row
    try:
        cfg = {
            r["id"]: json.loads(r["config_json"])
            for r in con.execute(
                "SELECT id, config_json FROM experiments WHERE end_time IS NOT NULL"
            ).fetchall()
        }
        rows = con.execute(
            """SELECT p.experiment_id e, p.agent_id a, ag.capability_tier tier,
                      ag.model_name model, p.cumulative_payoff pay
               FROM payoffs p JOIN agents ag
                 ON ag.experiment_id=p.experiment_id AND ag.agent_id=p.agent_id
               WHERE p.round_num=(SELECT MAX(round_num) FROM payoffs p2
                                  WHERE p2.experiment_id=p.experiment_id)"""
        ).fetchall()
    finally:
        con.close()
    recs = []
    for r in rows:
        c = cfg.get(r["e"])
        if not c:
            continue
        recs.append(dict(
            exp=r["e"], game=c["game"]["name"], comp=c["name"].split("-")[2],
            comm=bool(c["game"]["params"].get("communication")),
            tier=r["tier"], model=_MODEL_SHORT.get(r["model"], r["model"]), pay=float(r["pay"]),
        ))
    return pd.DataFrame(recs)


def h1_inequality(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70 + "\nH1 - capability asymmetry -> welfare inequality\n" + "=" * 70)
    # per-game Gini, aggregated per condition (mean + bootstrap CI across 20 seeds)
    gini_by_game = df.groupby("exp").pay.apply(lambda s: inequality.gini(s.values))
    meta = df.groupby("exp").agg(game=("game", "first"), comp=("comp", "first"), comm=("comm", "first"))
    meta["gini"] = gini_by_game
    print("  Gini per condition (mean [95% CI] across 20 games):")
    for (game, comp, comm), grp in meta.groupby(["game", "comp", "comm"]):
        m, lo, hi = _bootstrap_mean(grp.gini.values)
        tag = "homog" if comp in ("all_strong", "all_weak", "ss", "ww") else "MIXED"
        print(f"    {game[:5]:<5} {comp:<16} {'ON ' if comm else 'OFF'} [{tag}]  Gini={m:.3f} [{lo:.3f},{hi:.3f}]")
    # Atkinson index per game at eps in {0.5,1,2}, pre-registered alongside Gini as a
    # PRIMARY inequality outcome (OSF sec.4, osf.io/rnqgs). Same per-game vector, same
    # per-cell bootstrap CI. Reported as the pre-registered robustness check on the Gini.
    eps_list = (0.5, 1.0, 2.0)
    for e in eps_list:
        meta[f"atk{e}"] = df.groupby("exp").pay.apply(lambda s: inequality.atkinson(s.values, epsilon=e))
    print("\n  Atkinson index per condition (mean [95% CI] across 20 games):")
    for (game, comp, comm), grp in meta.groupby(["game", "comp", "comm"]):
        tag = "homog" if comp in ("all_strong", "all_weak", "ss", "ww") else "MIXED"
        cells = []
        for e in eps_list:
            am, alo, ahi = _bootstrap_mean(grp[f"atk{e}"].values)
            cells.append(f"A{e}={am:.3f}[{alo:.3f},{ahi:.3f}]")
        print(f"    {game[:5]:<5} {comp:<16} {'ON ' if comm else 'OFF'} [{tag}]  " + "  ".join(cells))
    # Robustness verdict: does the Gini inequality ordering survive under Atkinson?
    print("\n  Pre-registered robustness: mixed-vs-homogeneous inequality, PGG, Gini vs Atkinson:")
    pgg = meta[meta.game == "public_goods"].copy()
    pgg["is_mixed"] = ~pgg.comp.isin(("all_strong", "all_weak"))
    for col, name in (("gini", "Gini"), ("atk0.5", "Atk.5"), ("atk1.0", "Atk1"), ("atk2.0", "Atk2")):
        hi_m = pgg[pgg.is_mixed][col].mean()
        lo_m = pgg[~pgg.is_mixed][col].mean()
        verdict = "mixed>homog" if hi_m > lo_m else "homog>=mixed"
        print(f"    {name:<6} mixed={hi_m:.3f}  homog={lo_m:.3f}  diff={hi_m - lo_m:+.3f}  -> {verdict}")
    print("  => Atkinson agrees with Gini iff every row reads mixed>homog (H1 robust to the index).")
    # capability premium in mixed-tier games (per-game strong-mean - weak-mean)
    print("\n  Capability premium (strong - weak payoff) in MIXED-tier games:")
    mix = df[df.comp.isin(_MIXED_COMPS)]
    for (game, comp, comm), grp in mix.groupby(["game", "comp", "comm"]):
        prem = grp.groupby("exp").apply(
            lambda g: g[g.tier == "strong"].pay.mean() - g[g.tier == "weak"].pay.mean())
        m, lo, hi = _bootstrap_mean(prem.values)
        sign = "STRONG>weak" if m > 0 else "weak>STRONG (INVERTED)"
        print(f"    {game[:5]:<5} {comp:<16} {'ON ' if comm else 'OFF'}  premium={m:+.1f} [{lo:+.1f},{hi:+.1f}]  {sign}")


def inverted_by_model(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70 + "\nINVERTED PREMIUM: per-model (Haiku idiosyncrasy or general?)\n" + "=" * 70)
    mix = df[df.comp.isin(_MIXED_COMPS)]
    # each strong model's payoff vs the weak (llama) mean IN THE SAME mixed games,
    # SPLIT by communication (pooling hides the comm flip).
    for comm_val, label in ((False, "comm-OFF"), (True, "comm-ON ")):
        sub_c = mix[mix.comm == comm_val]
        print(f"  {label}: each strong model's payoff vs Llama (same games):")
        for model in ("haiku", "gpt5mini", "gemini"):
            exps = sub_c[sub_c.model == model].exp.unique()
            sub = sub_c[sub_c.exp.isin(exps)]
            per_game = sub.groupby("exp").apply(
                lambda g: g[g.model == model].pay.mean() - g[g.tier == "weak"].pay.mean())
            m, lo, hi = _bootstrap_mean(per_game.values)
            verdict = "MORE" if m > 0 else "LESS (exploited)"
            print(f"    {model:<9} vs Llama: {m:+.1f} [{lo:+.1f},{hi:+.1f}]  earns {verdict}  (n={len(per_game)})")
    print("  => the comm-ON rows answer it: inversion is GENERAL iff all three are negative there.")


def h2_h3_regression(df: pd.DataFrame) -> None:
    print("\n" + "=" * 70 + "\nH2/H3 - mixed-effects regression (payoff ~ tier*comm*game)\n" + "=" * 70)
    import statsmodels.formula.api as smf
    mix = df[df.comp.isin(_MIXED_COMPS)].copy()
    mix["strong"] = (mix.tier == "strong").astype(int)
    mix["commb"] = mix.comm.astype(int)
    mix["pgg"] = (mix.game == "public_goods").astype(int)
    # comm and game are constant WITHIN a game (group), so their main effects are
    # absorbed by the random intercept; include only the within-game strong terms.
    terms = ("strong", "strong:commb", "strong:pgg", "strong:commb:pgg")

    def _report(res, kind: str) -> None:
        print(f"  [{kind}] n={int(res.nobs)} agent-rows, {mix.exp.nunique()} games")
        for term in terms:
            if term in res.params.index:
                b, p = res.params[term], res.pvalues[term]
                star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
                print(f"    {term:<20} beta={b:+.2f}  p={p:.4f} {star}")
        print("  reading: strong = premium in ref cell (comm-OFF, Trust); strong:commb = comm")
        print("           flips/modulates premium (H2); strong:pgg = PGG-vs-Trust (H3).")

    try:
        md = smf.mixedlm("pay ~ strong + strong:commb + strong:pgg + strong:commb:pgg",
                         data=mix, groups=mix["exp"])
        _report(md.fit(method="lbfgs", maxiter=300), "mixed-effects, random intercept/game")
    except Exception as exc:  # noqa: BLE001 - fall back to OLS + cluster-robust SE
        print(f"  mixed-effects singular ({str(exc)[:40]}); OLS + cluster-robust SE by game:")
        res = smf.ols("pay ~ strong * commb * pgg", data=mix).fit(
            cov_type="cluster", cov_kwds={"groups": mix["exp"]})
        _report(res, "OLS, cluster-robust by game")


def h5_tier_inference(db_path: str) -> None:
    print("\n" + "=" * 70 + "\nH5 - tier inference from behaviour (FULL strong tier)\n" + "=" * 70)
    con = db.connect(db_path); con.row_factory = sqlite3.Row
    try:
        pgg_ids = [r["id"] for r in con.execute(
            "SELECT id, config_json FROM experiments WHERE end_time IS NOT NULL").fetchall()
            if json.loads(r["config_json"])["game"]["name"] == "public_goods"]
    finally:
        con.close()
    try:
        res = bayesian.infer_from_db(db_path, pgg_ids, game="public_goods")
        print(f"  PGG agents n={res['n']} (strong={res['n_strong']} weak={res['n_weak']})  "
              f"majority_baseline={res['majority_baseline']:.3f}")
        print(f"  accuracy={res['accuracy']:.3f}  balanced_accuracy={res['balanced_accuracy']:.3f}  "
              f"roc_auc={res['roc_auc']:.3f}")
        co = res["coefficients"]
        co = co if isinstance(co, dict) else dict(zip(res["feature_names"], co))
        print("  coefficients: " + "  ".join(f"{k}={float(v):+.2f}" for k, v in co.items()))
        print("  NOTE: with the full strong tier this is a REAL cross-tier eval (not pilot")
        print("        pseudoreplication). If ~1.0 it means the 3 strong models are still")
        print("        jointly separable from Llama; coefficients show which features carry it.")
    except Exception as exc:  # noqa: BLE001
        print(f"  [!] tier-inference failed: {exc}")


def h4_deception(db_path: str) -> None:
    print("\n" + "=" * 70 + "\nH4 - deception: surprisal (validated) vs global KL (neg. control)\n" + "=" * 70)
    con = db.connect(db_path); con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """SELECT d.experiment_id e, d.kl_headline kl, d.surprisal sp,
                      ag.capability_tier tier, ag.model_name model
               FROM deception_scores d JOIN agents ag
                 ON ag.experiment_id=d.experiment_id AND ag.agent_id=d.agent_id"""
        ).fetchall()
    finally:
        con.close()
    if not rows:
        print("  (no deception_scores yet -- paid scoring still running; re-run after it finishes)")
        return
    con = db.connect(db_path); con.row_factory = sqlite3.Row
    try:
        cfg = {r["id"]: json.loads(r["config_json"]) for r in con.execute(
            "SELECT id, config_json FROM experiments WHERE end_time IS NOT NULL").fetchall()}
    finally:
        con.close()
    recs = []
    for r in rows:
        c = cfg.get(r["e"]) if "e" in r.keys() else None
        recs.append(dict(kl=r["kl"], sp=r["sp"], tier=r["tier"],
                         model=_MODEL_SHORT.get(r["model"], r["model"]),
                         comp=c["name"].split("-")[2] if c else "?"))
    d = pd.DataFrame(recs).dropna(subset=["sp"])

    def _diff_ci(a, b, n=10000, seed=0):
        a = np.asarray([x for x in a if x == x]); b = np.asarray([x for x in b if x == x])
        rng = np.random.default_rng(seed)
        boots = rng.choice(a, (n, a.size), True).mean(1) - rng.choice(b, (n, b.size), True).mean(1)
        return float(a.mean() - b.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))

    print(f"  scored agent-games: {len(d)} (strong={int((d.tier=='strong').sum())}, weak={int((d.tier=='weak').sum())})")
    print("  SURPRISAL (C3-validated; lower = actions match words = less deceptive):")
    for t in ("strong", "weak"):
        m, lo, hi = _bootstrap_mean(d[d.tier == t].sp.values)
        print(f"    {t:<7} {m:.3f} [{lo:.3f},{hi:.3f}]")
    dm, dlo, dhi = _diff_ci(d[d.tier == "strong"].sp.values, d[d.tier == "weak"].sp.values)
    sep = "SEPARATES" if (dlo > 0) == (dhi > 0) else "overlaps 0 (no separation)"
    print(f"    strong-weak diff = {dm:+.3f} [{dlo:+.3f},{dhi:+.3f}]  -> {sep}")
    print("  surprisal by strong model:")
    for mdl, g in d[d.tier == "strong"].groupby("model"):
        m, lo, hi = _bootstrap_mean(g.sp.values)
        print(f"    {mdl:<9} {m:.3f} [{lo:.3f},{hi:.3f}]")
    print("  global kl_headline (NEGATIVE CONTROL -- expected NOT to track deception):")
    for t in ("strong", "weak"):
        m, lo, hi = _bootstrap_mean(d[d.tier == t].kl.values)
        print(f"    {t:<7} {m:.3f} [{lo:.3f},{hi:.3f}]")
    km, klo, khi = _diff_ci(d[d.tier == "strong"].kl.values, d[d.tier == "weak"].kl.values)
    ksep = "separates" if (klo > 0) == (khi > 0) else "overlaps 0 (no separation -- good, it's the control)"
    print(f"    strong-weak diff = {km:+.3f} [{klo:+.3f},{khi:+.3f}]  -> {ksep}")
    print("  surprisal by composition (comm-ON)  [strong | weak]:")
    for comp, g in d.groupby("comp"):
        s = _bootstrap_mean(g[g.tier == "strong"].sp.values)[0]
        w = _bootstrap_mean(g[g.tier == "weak"].sp.values)[0]
        print(f"    {comp:<16} strong={s:.3f}  weak={w:.3f}")


def classification(db_path: str) -> None:
    print("\n" + "=" * 70 + "\nMessage classification (5 classes, comm-ON)\n" + "=" * 70)
    con = db.connect(db_path); con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """SELECT mc.label, ag.capability_tier tier
               FROM message_classifications mc JOIN agents ag
                 ON ag.experiment_id=mc.experiment_id AND ag.agent_id=mc.agent_id"""
        ).fetchall()
    finally:
        con.close()
    if not rows:
        print("  (no classifications yet -- paid scoring still running)")
        return
    d = pd.DataFrame([dict(label=r["label"], tier=r["tier"]) for r in rows])
    print(f"  classified {len(d)} messages. distribution by tier:")
    print(d.groupby(["tier", "label"]).size().unstack(fill_value=0).to_string().replace("\n", "\n    "))


def network_summary(db_path: str) -> None:
    print("\n" + "=" * 70 + "\nNetwork: Trust transfer asymmetry + PGG centrality by tier\n" + "=" * 70)
    con = db.connect(db_path); con.row_factory = sqlite3.Row
    try:
        sw = [r["id"] for r in con.execute(
            "SELECT id, config_json FROM experiments WHERE end_time IS NOT NULL").fetchall()
            if json.loads(r["config_json"])["name"].split("-")[2] == "sw"]
    finally:
        con.close()
    s2w = w2s = 0.0
    for exp in sw:
        g = network.trust_transfer_graph(db_path, exp)
        for a, b, data in g.edges(data=True):
            if g.nodes[a]["tier"] == "strong" and g.nodes[b]["tier"] == "weak":
                s2w += data["weight"]
            elif g.nodes[a]["tier"] == "weak" and g.nodes[b]["tier"] == "strong":
                w2s += data["weight"]
    print(f"  Trust SW transfers ({len(sw)} games): strong->weak={s2w:.0f}  weak->strong={w2s:.0f}  "
          f"net to {'weak' if s2w > w2s else 'strong'}={abs(s2w-w2s):.0f}")
    cent = network.centrality(network.pgg_cocontribution_graph(db_path, k=8))
    bt = network.centrality_by_tier(network.pgg_cocontribution_graph(db_path, k=8), cent)
    print("  PGG similarity-graph eigenvector centrality by tier (caveat: NOT independent of H5):")
    for t, m in sorted(bt.items()):
        print(f"    {t:<7} eigenvector={m['eigenvector']:.3f}")


def main() -> int:
    p = argparse.ArgumentParser(description="Phase-4 production analysis (H1-H5)")
    p.add_argument("--db", default="results/production.db")
    args = p.parse_args()
    df = load_frame(args.db)
    print("=" * 70)
    print(f"PRODUCTION ANALYSIS -- {df.exp.nunique()} games, {len(df)} agent-rows")
    print("=" * 70)
    h1_inequality(df)
    inverted_by_model(df)
    h2_h3_regression(df)
    h5_tier_inference(args.db)
    network_summary(args.db)
    h4_deception(args.db)
    classification(args.db)
    print("\n" + "=" * 70 + "\nDONE. (H4/classification require the paid scoring pass to be complete.)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
