"""ZAV-35 Phase 2: export the scored production DB to static JSON for the frontend.

EUR 0, read-only. Writes two files the React site imports directly:
  frontend/src/data/aggregates.json  - every chart's numbers (premium, Gini,
      deception, message classes, Trust flow, tier-inference), reusing the EXACT
      analyze_production / make_figures computations so the site matches the
      figures and the Decision Log to the decimal.
  frontend/src/data/games.json       - 2-3 full game replays for the dark Theater
      (round-by-round messages, contributions, payoffs), including a matched
      mixed-PGG comm ON/OFF pair (same seed) so the ON/OFF toggle is like-for-like.

Run:  uv run python scripts/export_web_data.py   (or .venv/Scripts/python.exe ...)
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import analyze_production as ap
import make_figures as mf
from asymmetric_society.metrics import bayesian, inequality, network
from asymmetric_society.runner import db

DB = "results/production.db"
OUT = Path("frontend/src/data")
SEED = mf.SEED
MODELS = ("haiku", "gpt5mini", "gemini")


def _ci(values) -> dict:
    m, lo, hi = ap._bootstrap_mean(np.asarray(values, dtype=float), seed=SEED)
    return {"mean": round(m, 3), "lo": round(lo, 3), "hi": round(hi, 3)}


def _experiments_full(db_path: str) -> list[dict]:
    """id, game, comp, comm, seed for every finished experiment."""
    con = db.connect(db_path); con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            "SELECT id, config_json, seed FROM experiments WHERE end_time IS NOT NULL"
        ).fetchall()
    finally:
        con.close()
    out = []
    for r in rows:
        c = json.loads(r["config_json"])
        out.append(dict(id=r["id"], game=c["game"]["name"], comp=c["name"].split("-")[2],
                        comm=bool(c["game"]["params"].get("communication")), seed=r["seed"]))
    return out


def build_aggregates(df, exps) -> dict:
    con = db.connect(DB); con.row_factory = sqlite3.Row
    try:
        spend = con.execute("SELECT COALESCE(SUM(cost_eur),0) s FROM llm_calls").fetchone()["s"]
        mc_rows = con.execute(
            """SELECT ag.capability_tier tier, mc.label, COUNT(*) c
               FROM message_classifications mc JOIN agents ag
                 ON ag.experiment_id=mc.experiment_id AND ag.agent_id=mc.agent_id
               GROUP BY tier, label"""
        ).fetchall()
    finally:
        con.close()

    # F1 - capability premium by model x communication
    prem = mf._premium_by_model_comm(df)
    premium = {}
    for m in MODELS:
        off_m, off_lo, off_hi = prem[(False, m)]
        on_m, on_lo, on_hi = prem[(True, m)]
        premium[m] = {
            "off": {"mean": round(off_m, 3), "lo": round(off_lo, 3), "hi": round(off_hi, 3)},
            "on": {"mean": round(on_m, 3), "lo": round(on_lo, 3), "hi": round(on_hi, 3)},
        }

    # Dissociation - strong-vs-weak premium by GAME (PGG vs Trust), comm OFF/ON.
    # The headline flip is a Public-Goods phenomenon; in dyadic Trust the premium is ~0.
    PGG_COMPS = {"mixed", "lone_elite", "lone_vulnerable"}

    def _game_premium(game, comm_val, comps):
        sub = df[(df.game == game) & (df.comp.isin(comps)) & (df.comm == comm_val)]
        per = sub.groupby("exp").apply(lambda g: g[g.tier == "strong"].pay.mean() - g[g.tier == "weak"].pay.mean())
        return _ci(per.dropna().values)

    premium_by_game = {
        "public_goods": {"off": _game_premium("public_goods", False, PGG_COMPS), "on": _game_premium("public_goods", True, PGG_COMPS)},
        "trust_game": {"off": _game_premium("trust_game", False, {"sw"}), "on": _game_premium("trust_game", True, {"sw"})},
    }

    # F2 - Gini by composition x game x comm
    gini_by_game = df.groupby("exp").pay.apply(lambda s: inequality.gini(s.values))
    gmeta = df.groupby("exp").agg(game=("game", "first"), comp=("comp", "first"), comm=("comm", "first"))
    gmeta["gini"] = gini_by_game
    gini = []
    for (game, comp, comm), grp in gmeta.groupby(["game", "comp", "comm"]):
        gini.append({"game": game, "comp": comp, "comm": bool(comm), **_ci(grp.gini.values)})

    # F3 - deception: surprisal vs global KL by tier (+ per strong model)
    d = mf._deception_frame(DB)
    deception = {}
    for col in ("sp", "kl"):
        deception["surprisal" if col == "sp" else "kl"] = {
            "strong": _ci(d[d.tier == "strong"][col].dropna().values),
            "weak": _ci(d[d.tier == "weak"][col].dropna().values),
        }
    deception["surprisal"]["by_model"] = {
        m: round(ap._bootstrap_mean(d[(d.tier == "strong") & (d.model == m)].sp.values, seed=SEED)[0], 3)
        for m in MODELS
    }

    # F4 - message-class counts by tier
    msg = {"strong": {}, "weak": {}}
    for r in mc_rows:
        msg.setdefault(r["tier"], {})[r["label"]] = r["c"]

    # F5 - Trust strong-weak token flow (+ caveated PGG-similarity assortativity)
    sw = [e["id"] for e in exps if e["game"] == "trust_game" and e["comp"] == "sw"]
    s2w = w2s = 0.0
    per_game_net = []
    for exp in sw:
        g = network.trust_transfer_graph(DB, exp)
        a = b = 0.0
        for u, v, data in g.edges(data=True):
            if g.nodes[u]["tier"] == "strong" and g.nodes[v]["tier"] == "weak":
                a += data["weight"]
            elif g.nodes[u]["tier"] == "weak" and g.nodes[v]["tier"] == "strong":
                b += data["weight"]
        s2w += a; w2s += b
        per_game_net.append(round(a - b, 1))
    assort = network.tier_assortativity(network.pgg_cocontribution_graph(DB, k=8))
    trust_flow = {
        "s2w": round(s2w), "w2s": round(w2s), "net": round(s2w - w2s),
        "net_to": "weak" if s2w > w2s else "strong", "n_games": len(sw),
        "per_game_net": per_game_net, "assortativity": round(float(assort), 3),
    }

    # F6 - tier inference: honest cross-validated ROC + confusion
    pgg = [e["id"] for e in exps if e["game"] == "public_goods"]
    X, y, _ids, _names = bayesian.extract_features(DB, pgg, game="public_goods")
    summary = bayesian.fit_infer(X, y, seed=SEED)
    pipe = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    proba = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba")[:, 1]
    fpr, tpr, _ = roc_curve(y, proba)
    grid = np.linspace(0, 1, 50)
    roc = [{"fpr": round(float(f), 3), "tpr": round(float(np.interp(f, fpr, tpr)), 3)} for f in grid]
    cm = confusion_matrix(y, (proba >= 0.5).astype(int))  # rows: true weak(0)/strong(1)
    tier_inference = {
        "auc": round(float(auc(fpr, tpr)), 3),
        "balanced_accuracy": round(float(summary["balanced_accuracy"]), 3),
        "n": int(summary["n"]), "n_strong": int(summary["n_strong"]), "n_weak": int(summary["n_weak"]),
        "roc": roc,
        "confusion": {"true_weak_pred_weak": int(cm[0, 0]), "true_weak_pred_strong": int(cm[0, 1]),
                      "true_strong_pred_weak": int(cm[1, 0]), "true_strong_pred_strong": int(cm[1, 1])},
    }

    return {
        "meta": {"n_games": int(df.exp.nunique()), "spend_eur": round(float(spend), 2), "cells": 16},
        "premium_by_model_comm": premium,
        "premium_by_game": premium_by_game,
        "gini": gini,
        "deception": deception,
        "message_classes": msg,
        "trust_flow": trust_flow,
        "tier_inference": tier_inference,
    }


def _replay(exp_id: str, label: str) -> dict:
    con = db.connect(DB); con.row_factory = sqlite3.Row
    try:
        cfg = json.loads(con.execute("SELECT config_json FROM experiments WHERE id=?", (exp_id,)).fetchone()["config_json"])
        agents = con.execute("SELECT agent_id, capability_tier, model_name FROM agents WHERE experiment_id=?", (exp_id,)).fetchall()
        acts = con.execute("SELECT round_num, agent_id, action_json, message FROM actions WHERE experiment_id=? ORDER BY round_num, agent_id", (exp_id,)).fetchall()
        pays = con.execute("SELECT round_num, agent_id, round_payoff, cumulative_payoff FROM payoffs WHERE experiment_id=?", (exp_id,)).fetchall()
    finally:
        con.close()
    pay_idx = {(p["round_num"], p["agent_id"]): p for p in pays}
    rounds: dict[int, dict] = {}
    for a in acts:
        action = json.loads(a["action_json"])
        rn = a["round_num"]
        p = pay_idx.get((rn, a["agent_id"]))
        rounds.setdefault(rn, {"round": rn, "actions": []})["actions"].append({
            "agent": a["agent_id"],
            "contribution": action.get("contribution"),
            "message": a["message"],
            "round_payoff": round(float(p["round_payoff"]), 2) if p else None,
            "cumulative": round(float(p["cumulative_payoff"]), 2) if p else None,
        })
    return {
        "id": exp_id, "label": label,
        "game": cfg["game"]["name"], "comp": cfg["name"].split("-")[2],
        "comm": bool(cfg["game"]["params"].get("communication")),
        "agents": [{"id": a["agent_id"], "tier": a["capability_tier"],
                    "model": ap._MODEL_SHORT.get(a["model_name"], a["model_name"])} for a in agents],
        "rounds": [rounds[k] for k in sorted(rounds)],
    }


def select_theater_games(df, exps) -> list[dict]:
    """Matched mixed-PGG comm ON/OFF pair (same seed, clearest exploitation) + one all-strong comm-ON."""
    by_id = {e["id"]: e for e in exps}
    # per-game strong-minus-weak premium for mixed PGG comm-ON games
    def premium_of(exp_id):
        g = df[df.exp == exp_id]
        s, w = g[g.tier == "strong"].pay, g[g.tier == "weak"].pay
        return (s.mean() - w.mean()) if len(s) and len(w) else np.nan

    mixed_on = [e for e in exps if e["game"] == "public_goods" and e["comp"] == "mixed" and e["comm"]]
    mixed_on.sort(key=lambda e: premium_of(e["id"]))  # most-negative (most exploited) first
    pick_on = mixed_on[0]
    # its comm-OFF twin by seed
    twin = next((e for e in exps if e["game"] == "public_goods" and e["comp"] == "mixed"
                 and not e["comm"] and e["seed"] == pick_on["seed"]), None)
    games = [_replay(pick_on["id"], "mixed-PGG · communication ON"),
             _replay(twin["id"], "mixed-PGG · communication OFF") if twin else None]
    # bonus: an all-strong comm-ON game
    allstrong = next((e for e in exps if e["game"] == "public_goods" and e["comp"] == "all_strong" and e["comm"]), None)
    if allstrong:
        games.append(_replay(allstrong["id"], "all-strong · communication ON"))
    return [g for g in games if g]


def main() -> int:
    df = ap.load_frame(DB)
    exps = _experiments_full(DB)
    OUT.mkdir(parents=True, exist_ok=True)

    agg = build_aggregates(df, exps)
    (OUT / "aggregates.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")

    games = select_theater_games(df, exps)
    (OUT / "games.json").write_text(json.dumps(games, indent=2), encoding="utf-8")

    print(f"wrote {OUT}/aggregates.json + games.json")
    print(f"  meta: {agg['meta']}")
    p = agg["premium_by_model_comm"]
    print(f"  F1 premium OFF/ON: haiku {p['haiku']['off']['mean']:+}/{p['haiku']['on']['mean']:+} "
          f"gpt5mini {p['gpt5mini']['off']['mean']:+}/{p['gpt5mini']['on']['mean']:+} "
          f"gemini {p['gemini']['off']['mean']:+}/{p['gemini']['on']['mean']:+}")
    print(f"  F3 surprisal: strong {agg['deception']['surprisal']['strong']['mean']} < weak {agg['deception']['surprisal']['weak']['mean']}"
          f" | KL strong {agg['deception']['kl']['strong']['mean']} > weak {agg['deception']['kl']['weak']['mean']}")
    print(f"  F5 trust: s2w {agg['trust_flow']['s2w']} / w2s {agg['trust_flow']['w2s']} net {agg['trust_flow']['net']} to {agg['trust_flow']['net_to']} (assort {agg['trust_flow']['assortativity']})")
    print(f"  F6 inference: AUC {agg['tier_inference']['auc']} balacc {agg['tier_inference']['balanced_accuracy']}")
    print(f"  theater games: {[(g['label'], len(g['rounds'])) for g in games]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
