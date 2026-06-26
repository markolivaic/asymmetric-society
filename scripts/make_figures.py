"""ZAV-34: final result figures from the scored production DB (results/production.db).

Turns the Phase-4 analysis numbers (already in the DB) into publication-quality
figures for the thesis Results chapter and the React frontend. EUR 0 (read-only):
reads ``payoffs`` / ``deception_scores`` / ``message_classifications`` / ``actions``
and reuses the metric modules, never calls an LLM.

Design rules (per ZAV-34 review):
  * NO descriptive/marketing titles baked into the figures. Final titles live on
    the Confluence page "Figure Titles (thesis + frontend)" and are added at
    write-up / frontend time. Here we ship only neutral technical axis labels,
    units, legends, and CIs.
  * Colours are FIXED across every figure (tier + per-model + message-class) so
    the thesis and the frontend bind to one palette.
  * F1 (headline) draws an emphasised zero line: the whole story is the premium
    crossing from positive (comm-OFF) to negative (comm-ON).

Each figure is saved as BOTH a raster PNG (frontend/preview) and a vector PDF
(LaTeX thesis) under the tracked ``figures/`` directory.

Run:  uv run python scripts/make_figures.py            # all seven
      uv run python scripts/make_figures.py --only f1  # one figure
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write files, never open a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, Patch

# Reuse the EXACT analysis pipeline (same data frame + bootstrap) so figure
# numbers reproduce analyze_production.py / the Decision Log rather than drifting.
import analyze_production as ap  # scripts/ is on sys.path[0] when run from repo root
from asymmetric_society.metrics import bayesian, inequality, network
from asymmetric_society.runner import db

# --------------------------------------------------------------------------- #
# Shared style - one palette, bound by name, reused by the frontend.
# Okabe-Ito colourblind-safe colours. Strong tier/models in cool hues, weak in warm.
# --------------------------------------------------------------------------- #
FIG_DIR = Path("figures")
SEED = 0  # deterministic bootstrap everywhere
# Frontend page base: the SVG export blends into the site; PNG/PDF stay white for print.
SITE_BASE = "#f4f2ee"

TIER_COLORS = {"strong": "#0072B2", "weak": "#D55E00"}
MODEL_COLORS = {
    "haiku": "#0072B2",     # blue
    "gpt5mini": "#009E73",  # bluish green
    "gemini": "#56B4E9",    # sky blue
    "llama": "#D55E00",     # vermillion (weak)
}
MODEL_LABEL = {"haiku": "Claude Haiku 4.5", "gpt5mini": "GPT-5-mini", "gemini": "Gemini 2.5 Flash"}
# 5 message classes - green (good) → grey (neutral) → warm (threat/deception).
CLASS_ORDER = ["cooperative-promise", "informational", "neutral", "threat", "deceptive-claim"]
CLASS_COLORS = {
    "cooperative-promise": "#009E73",
    "informational": "#56B4E9",
    "neutral": "#BBBBBB",
    "threat": "#E69F00",
    "deceptive-claim": "#D55E00",
}
ZERO_LINE = dict(color="#222222", lw=2.0, ls="--", zorder=1)  # emphasised baseline

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "font.size": 11,
    "axes.titlesize": 11,
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "axes.axisbelow": True,
    "axes.facecolor": "none",  # transparent plot area inherits the per-format savefig facecolor
})


def _register_fonts() -> None:
    """Render figures in the vendored JetBrains Mono so they match the site's
    data charts (instead of matplotlib's default DejaVu Sans - the visual seam)."""
    import matplotlib.font_manager as fm

    ttf = Path("scripts/_fonts/JetBrainsMono.ttf")
    if not ttf.exists():
        return
    fm.fontManager.addfont(str(ttf))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(ttf)).get_name()


def _save(fig, name: str) -> None:
    """Write a figure as both PNG (frontend) and PDF (thesis), tightly cropped."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext, facecolor in (("png", "white"), ("pdf", "white"), ("svg", SITE_BASE)):
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight", facecolor=facecolor)
    plt.close(fig)
    print(f"  wrote figures/{name}.png + .pdf (white) + .svg ({SITE_BASE})")


def _experiments(db_path: str) -> list[dict]:
    """(id, game, comp, comm) for every finished experiment."""
    con = db.connect(db_path); con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            "SELECT id, config_json FROM experiments WHERE end_time IS NOT NULL"
        ).fetchall()
    finally:
        con.close()
    out = []
    for r in rows:
        c = json.loads(r["config_json"])
        out.append(dict(
            id=r["id"], game=c["game"]["name"], comp=c["name"].split("-")[2],
            comm=bool(c["game"]["params"].get("communication")),
        ))
    return out


# --------------------------------------------------------------------------- #
# F1 - headline: capability premium by communication, per strong model.
# Story: communication inverts the premium (positive -> negative) for all three.
# --------------------------------------------------------------------------- #
def _premium_by_model_comm(df: pd.DataFrame) -> dict[tuple[bool, str], tuple[float, float, float]]:
    """Replicates analyze_production.inverted_by_model: each strong model's payoff
    minus the weak (Llama) mean in the SAME mixed games, split by communication."""
    mix = df[df.comp.isin(ap._MIXED_COMPS)]
    out: dict[tuple[bool, str], tuple[float, float, float]] = {}
    for comm_val in (False, True):
        sub_c = mix[mix.comm == comm_val]
        for model in ("haiku", "gpt5mini", "gemini"):
            exps = sub_c[sub_c.model == model].exp.unique()
            sub = sub_c[sub_c.exp.isin(exps)]
            per_game = sub.groupby("exp").apply(
                lambda g: g[g.model == model].pay.mean() - g[g.tier == "weak"].pay.mean())
            out[(comm_val, model)] = ap._bootstrap_mean(per_game.values, seed=SEED)
    return out


def fig1_premium(df: pd.DataFrame) -> None:
    prem = _premium_by_model_comm(df)
    models = ("haiku", "gpt5mini", "gemini")
    x = np.arange(len(models))
    w = 0.38

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for k, comm_val in enumerate((False, True)):
        offs = (k - 0.5) * w
        for i, m in enumerate(models):
            mean, lo, hi = prem[(comm_val, m)]
            ax.bar(x[i] + offs, mean, w, color=MODEL_COLORS[m],
                   hatch="////" if comm_val else None, edgecolor="white", linewidth=0.6,
                   alpha=0.95 if not comm_val else 0.80, zorder=2)
            ax.errorbar(x[i] + offs, mean, yerr=[[mean - lo], [hi - mean]],
                        fmt="none", ecolor="#222222", elinewidth=1.3, capsize=3, zorder=3)
    ax.axhline(0, **ZERO_LINE)  # emphasised: positive (OFF) vs negative (ON)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL[m] for m in models])
    ax.set_ylabel("Capability premium\n(strong − weak cumulative payoff)")
    # legend: OFF vs ON encoded by hatch (colour carries model identity)
    handles = [Patch(facecolor="#888888", edgecolor="white", label="Communication OFF"),
               Patch(facecolor="#888888", edgecolor="white", hatch="////", label="Communication ON")]
    ax.legend(handles=handles, loc="upper right")
    ax.margins(y=0.18)
    _save(fig, "f1_premium_by_communication")


# --------------------------------------------------------------------------- #
# F2 - welfare inequality (Gini) by composition and game, split by communication.
# Story: inequality is modestly higher in mixed/asymmetric populations.
# --------------------------------------------------------------------------- #
_COMP_ORDER = {
    "public_goods": ["all_strong", "all_weak", "mixed", "lone_elite", "lone_vulnerable"],
    "trust_game": ["ss", "ww", "sw"],
}
_HOMOG = {"all_strong", "all_weak", "ss", "ww"}
_COMP_LABEL = {
    "all_strong": "all-strong", "all_weak": "all-weak", "mixed": "mixed-equal",
    "lone_elite": "lone-elite", "lone_vulnerable": "lone-vuln.",
    "ss": "strong-strong", "ww": "weak-weak", "sw": "strong-weak",
}


def fig2_inequality(df: pd.DataFrame) -> None:
    gini_by_game = df.groupby("exp").pay.apply(lambda s: inequality.gini(s.values))
    meta = df.groupby("exp").agg(game=("game", "first"), comp=("comp", "first"),
                                 comm=("comm", "first"))
    meta["gini"] = gini_by_game

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6),
                             gridspec_kw={"width_ratios": [5, 3]})
    for ax, game in zip(axes, ("public_goods", "trust_game")):
        comps = _COMP_ORDER[game]
        x = np.arange(len(comps))
        w = 0.38
        for k, comm_val in enumerate((False, True)):
            means, los, his = [], [], []
            for comp in comps:
                grp = meta[(meta.game == game) & (meta.comp == comp) & (meta.comm == comm_val)]
                m, lo, hi = ap._bootstrap_mean(grp.gini.values, seed=SEED)
                means.append(m); los.append(m - lo); his.append(hi - m)
            ax.bar(x + (k - 0.5) * w, means, w,
                   color="#4C72B0" if comm_val == False else "#4C72B0",
                   hatch="////" if comm_val else None, edgecolor="white", linewidth=0.6,
                   alpha=0.85 if not comm_val else 0.7,
                   yerr=[los, his], error_kw=dict(ecolor="#222", elinewidth=1.1, capsize=2.5))
        ax.set_xticks(x)
        ax.set_xticklabels([_COMP_LABEL[c] for c in comps], rotation=25, ha="right")
        # neutral facet tag (technical, not a story title)
        ax.set_title("Public Goods" if game == "public_goods" else "Trust", fontsize=10, color="#555")
        # mark homogeneous vs mixed groups
        for i, comp in enumerate(comps):
            if comp not in _HOMOG:
                ax.axvspan(i - 0.5, i + 0.5, color="#000000", alpha=0.04, zorder=0)
    axes[0].set_ylabel("Gini coefficient of payoffs\n(mean of 20 games, 95% CI)")
    handles = [Patch(facecolor="#4C72B0", alpha=0.85, label="Communication OFF"),
               Patch(facecolor="#4C72B0", alpha=0.7, hatch="////", label="Communication ON"),
               Patch(facecolor="#000000", alpha=0.06, label="mixed-tier composition")]
    axes[1].legend(handles=handles, loc="upper left", fontsize=9)
    _save(fig, "f2_inequality_by_composition")


# --------------------------------------------------------------------------- #
# F3 - deception: contemporaneous surprisal vs global policy-KL, by tier.
# Story: surprisal separates the tiers (strong < weak); global KL gives the
# opposite (artefactual) sign -> the registered negative control.
# --------------------------------------------------------------------------- #
def _deception_frame(db_path: str) -> pd.DataFrame:
    con = db.connect(db_path); con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """SELECT d.surprisal sp, d.kl_headline kl, ag.capability_tier tier,
                      ag.model_name model
               FROM deception_scores d JOIN agents ag
                 ON ag.experiment_id=d.experiment_id AND ag.agent_id=d.agent_id"""
        ).fetchall()
    finally:
        con.close()
    return pd.DataFrame([dict(sp=r["sp"], kl=r["kl"], tier=r["tier"],
                              model=ap._MODEL_SHORT.get(r["model"], r["model"])) for r in rows])


def fig3_deception(db_path: str) -> None:
    d = _deception_frame(db_path)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.6), sharey=False)
    panels = [("sp", "Contemporaneous surprisal\n(lower = actions match words)", axes[0]),
              ("kl", "Global policy-KL\n(negative control)", axes[1])]
    for col, ylabel, ax in panels:
        for i, tier in enumerate(("strong", "weak")):
            vals = d[d.tier == tier][col].dropna().values
            m, lo, hi = ap._bootstrap_mean(vals, seed=SEED)
            ax.bar(i, m, 0.6, color=TIER_COLORS[tier], alpha=0.9,
                   yerr=[[m - lo], [hi - m]], error_kw=dict(ecolor="#222", elinewidth=1.3, capsize=4))
        # per strong-model surprisal dots (shows it holds across all three)
        if col == "sp":
            # jitter the three model dots + dark edge so Haiku (same blue as the
            # strong bar) stays visible instead of vanishing into the bar.
            xoff = {"haiku": -0.14, "gemini": 0.0, "gpt5mini": 0.14}
            for m in ("haiku", "gpt5mini", "gemini"):
                mm = ap._bootstrap_mean(d[(d.tier == "strong") & (d.model == m)].sp.values, seed=SEED)[0]
                ax.scatter(xoff[m], mm, s=60, color=MODEL_COLORS[m], edgecolor="#222222",
                           linewidth=1.0, zorder=4, label=MODEL_LABEL[m])
            ax.legend(loc="upper left", fontsize=8, title="strong models")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["strong", "weak"])
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.15)  # headroom so the note clears the bars/legend
        # annotate the strong−weak direction (top-right; legend sits top-left)
        sm = ap._bootstrap_mean(d[d.tier == "strong"][col].dropna().values, seed=SEED)[0]
        wm = ap._bootstrap_mean(d[d.tier == "weak"][col].dropna().values, seed=SEED)[0]
        arrow = "strong < weak" if sm < wm else "strong > weak"
        ax.text(0.97, 0.97, arrow, transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color="#555")
    _save(fig, "f3_deception_surprisal_vs_kl")


# --------------------------------------------------------------------------- #
# F4 - message-class distribution by tier (comm-ON).
# Story: strong agents talk in cooperative promises; weak agents hold the
# deceptive-claim and threat mass.
# --------------------------------------------------------------------------- #
def fig4_message_classes(db_path: str) -> None:
    con = db.connect(db_path); con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """SELECT ag.capability_tier tier, mc.label
               FROM message_classifications mc JOIN agents ag
                 ON ag.experiment_id=mc.experiment_id AND ag.agent_id=mc.agent_id"""
        ).fetchall()
    finally:
        con.close()
    d = pd.DataFrame([dict(tier=r["tier"], label=r["label"]) for r in rows])
    counts = d.groupby(["tier", "label"]).size().unstack(fill_value=0).reindex(columns=CLASS_ORDER, fill_value=0)
    prop = counts.div(counts.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(9.2, 3.4))
    tiers = ["strong", "weak"]
    left = np.zeros(len(tiers))
    y = np.arange(len(tiers))
    for cls in CLASS_ORDER:
        vals = prop.loc[tiers, cls].values
        ax.barh(y, vals, left=left, color=CLASS_COLORS[cls], edgecolor="white", label=cls)
        for i, tier in enumerate(tiers):
            n = int(counts.loc[tier, cls])
            if vals[i] > 0.04:  # label only segments wide enough to read
                ax.text(left[i] + vals[i] / 2, y[i], f"{n}", ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")
        left += vals
    ax.set_yticks(y); ax.set_yticklabels([f"{t}\n(n={int(counts.loc[t].sum())} msgs)" for t in tiers])
    ax.set_xlabel("Share of messages within tier")
    ax.set_xlim(0, 1)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.22), fontsize=8.5)
    # call out the asymmetry that carries the story
    ax.text(1.005, 1, f"deceptive-claim:\nstrong {int(counts.loc['strong','deceptive-claim'])}, "
            f"weak {int(counts.loc['weak','deceptive-claim'])}", transform=ax.get_yaxis_transform(),
            fontsize=8, color="#555", va="top")
    _save(fig, "f4_message_classes_by_tier")


# --------------------------------------------------------------------------- #
# F5 - Trust strong-weak token-transfer flow (the genuine interaction graph).
# Story: in strong-weak Trust games net token flow runs slightly toward the weak.
# Tier-assortativity (0.847) is a SEPARATE PGG-similarity result, shown caveated.
# --------------------------------------------------------------------------- #
def fig5_trust_transfer(db_path: str) -> None:
    sw = [e["id"] for e in _experiments(db_path) if e["game"] == "trust_game" and e["comp"] == "sw"]
    s2w = w2s = 0.0
    per_game_net = []
    for exp in sw:
        g = network.trust_transfer_graph(db_path, exp)
        gs2w = gw2s = 0.0
        for a, b, data in g.edges(data=True):
            if g.nodes[a]["tier"] == "strong" and g.nodes[b]["tier"] == "weak":
                gs2w += data["weight"]
            elif g.nodes[a]["tier"] == "weak" and g.nodes[b]["tier"] == "strong":
                gw2s += data["weight"]
        s2w += gs2w; w2s += gw2s
        per_game_net.append(gs2w - gw2s)  # >0 = strong gives more to weak
    per_game_net = np.asarray(per_game_net, dtype=float)

    # assortativity is on the PGG behavioural-similarity graph (NOT this graph)
    assort = network.tier_assortativity(network.pgg_cocontribution_graph(db_path, k=8))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(10.5, 4.4),
                                   gridspec_kw={"width_ratios": [3, 3]})

    # --- left: aggregate 2-node flow diagram ---
    axL.set_xlim(0, 1); axL.set_ylim(0, 1); axL.axis("off")
    axL.set_aspect("equal")  # keep the tier nodes circular, not squashed ellipses
    ps, pw = (0.20, 0.5), (0.80, 0.5)
    for (px, py), tier in ((ps, "strong"), (pw, "weak")):
        axL.add_patch(plt.Circle((px, py), 0.11, color=TIER_COLORS[tier], zorder=3))
        axL.text(px, py, tier, ha="center", va="center", color="white", fontweight="bold", zorder=4)
    wmax = max(s2w, w2s)
    # strong -> weak (upper arc), weak -> strong (lower arc); width ∝ total tokens
    axL.add_patch(FancyArrowPatch(ps, pw, connectionstyle="arc3,rad=0.28",
                  arrowstyle="-|>", mutation_scale=22, lw=1 + 6 * s2w / wmax,
                  color=TIER_COLORS["strong"], alpha=0.85, zorder=2))
    axL.add_patch(FancyArrowPatch(pw, ps, connectionstyle="arc3,rad=0.28",
                  arrowstyle="-|>", mutation_scale=22, lw=1 + 6 * w2s / wmax,
                  color=TIER_COLORS["weak"], alpha=0.85, zorder=2))
    axL.text(0.5, 0.80, f"strong → weak: {s2w:,.0f} tokens", ha="center", fontsize=9,
             color=TIER_COLORS["strong"])
    axL.text(0.5, 0.20, f"weak → strong: {w2s:,.0f} tokens", ha="center", fontsize=9,
             color=TIER_COLORS["weak"])
    net_to = "weak" if s2w > w2s else "strong"
    axL.text(0.5, 0.02, f"net {abs(s2w - w2s):,.0f} tokens toward {net_to}  "
             f"({len(sw)} strong–weak Trust games)", ha="center", fontsize=9, color="#333")

    # --- right: per-game net distribution (rigor behind the aggregate) ---
    axR.hist(per_game_net, bins=12, color="#888888", edgecolor="white", zorder=2)
    axR.axvline(0, **ZERO_LINE)
    axR.axvline(per_game_net.mean(), color="#0072B2", lw=2,
                label=f"mean net = {per_game_net.mean():+.0f}")
    axR.set_xlabel("Per-game net flow, tokens\n(strong→weak − weak→strong)")
    axR.set_ylabel("Trust games")
    axR.set_ylim(0, axR.get_ylim()[1] * 1.12)  # headroom: don't clip the tallest bar
    axR.legend(loc="upper right", fontsize=9)
    # caveated secondary result: assortativity belongs to a DIFFERENT graph,
    # centred under the whole figure so it reads as an aside, not a histogram label.
    fig.subplots_adjust(bottom=0.24)
    fig.text(0.5, 0.05, f"PGG behavioural-similarity graph: tier-assortativity = {assort:.3f} "
             f"(homophily; not independent of tier inference)", ha="center", fontsize=8, color="#777")
    _save(fig, "f5_trust_token_transfer")


# --------------------------------------------------------------------------- #
# F6 - tier inference from behaviour: honest cross-validated ROC + confusion.
# Story: behaviour alone separates strong from weak agents (AUC ~0.97).
# --------------------------------------------------------------------------- #
def fig6_tier_inference(db_path: str) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import auc, confusion_matrix, roc_curve
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pgg = [e["id"] for e in _experiments(db_path) if e["game"] == "public_goods"]
    X, y, _ids, _names = bayesian.extract_features(db_path, pgg, game="public_goods")
    summary = bayesian.fit_infer(X, y, seed=SEED)  # canonical CV metrics (0.97 / 0.92)

    pipe = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(max_iter=1000))])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    proba = cross_val_predict(pipe, X, y, cv=skf, method="predict_proba")[:, 1]
    fpr, tpr, _ = roc_curve(y, proba)
    cv_auc = auc(fpr, tpr)
    cm = confusion_matrix(y, (proba >= 0.5).astype(int))  # rows: true weak(0)/strong(1)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.6, 4.6),
                                   gridspec_kw={"width_ratios": [3, 2.4]})
    # ROC (honest pooled cross-validation)
    axL.plot(fpr, tpr, color="#0072B2", lw=2.2, label=f"5-fold CV  AUC = {cv_auc:.3f}")
    axL.plot([0, 1], [0, 1], ls="--", color="#999", lw=1.2, label="chance (AUC = 0.5)")
    axL.set_xlabel("False positive rate"); axL.set_ylabel("True positive rate")
    axL.set_xlim(0, 1); axL.set_ylim(0, 1.02)
    axL.legend(loc="lower right", fontsize=9)
    axL.text(0.04, 0.30, f"n = {summary['n']} PGG agents\n"
             f"(strong {summary['n_strong']} / weak {summary['n_weak']})\n"
             f"balanced accuracy = {summary['balanced_accuracy']:.2f}",
             transform=axL.transAxes, fontsize=8.5, color="#444", va="bottom", ha="left")
    # confusion matrix (CV predictions, 0.5 threshold)
    im = axR.imshow(cm, cmap="Blues")
    axR.set_xticks([0, 1]); axR.set_xticklabels(["pred. weak", "pred. strong"])
    axR.set_yticks([0, 1]); axR.set_yticklabels(["true weak", "true strong"])
    thresh = cm.max() / 2
    for i in range(2):
        for j in range(2):
            axR.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "#222", fontsize=13, fontweight="bold")
    fig.colorbar(im, ax=axR, fraction=0.046, pad=0.04, label="agents")
    _save(fig, "f6_tier_inference_roc")
    print(f"    [F6] canonical fit_infer AUC={summary['roc_auc']:.3f}  "
          f"balanced_acc={summary['balanced_accuracy']:.3f}  (pooled-CV curve AUC={cv_auc:.3f})")


# --------------------------------------------------------------------------- #
# F7 - dissociation: capability premium by GAME (PGG vs Trust), comm OFF/ON.
# Story: the communication-driven flip is a Public-Goods phenomenon; in dyadic
# Trust the premium is near zero and the data is uncertain. A forest plot (point
# estimate + 95% bootstrap CI) shows the contrast in both magnitude and
# certainty. Premium is recomputed from the frame (same path as F1), never
# hardcoded from the frontend.
# --------------------------------------------------------------------------- #
_PGG_COMPS = {"mixed", "lone_elite", "lone_vulnerable"}


def _game_premium(df: pd.DataFrame, game: str, comm_val: bool, comps: set) -> tuple[float, float, float]:
    sub = df[(df.game == game) & (df.comp.isin(comps)) & (df.comm == comm_val)]
    per = sub.groupby("exp").apply(
        lambda g: g[g.tier == "strong"].pay.mean() - g[g.tier == "weak"].pay.mean())
    return ap._bootstrap_mean(per.dropna().values, seed=SEED)


def fig7_dissociation(df: pd.DataFrame) -> None:
    rows = [
        ("Public Goods, comm OFF", _game_premium(df, "public_goods", False, _PGG_COMPS)),
        ("Public Goods, comm ON", _game_premium(df, "public_goods", True, _PGG_COMPS)),
        ("Trust, comm OFF", _game_premium(df, "trust_game", False, {"sw"})),
        ("Trust, comm ON", _game_premium(df, "trust_game", True, {"sw"})),
    ]
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ys = list(range(len(rows)))[::-1]
    for y, (label, (m, lo, hi)) in zip(ys, rows):
        color = TIER_COLORS["strong"] if m >= 0 else TIER_COLORS["weak"]
        ax.plot([lo, hi], [y, y], color="#555555", lw=1.5, zorder=2)
        for xx in (lo, hi):
            ax.plot([xx, xx], [y - 0.13, y + 0.13], color="#555555", lw=1.5, zorder=2)
        ax.scatter([m], [y], s=90, color=color, edgecolor="white", linewidth=1.4, zorder=3)
        ax.annotate(f"{m:+.0f}", (m, y), textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=10, color="#1a1a18")
    ax.axvline(0, **ZERO_LINE)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlim(-46, 34)
    ax.set_xlabel("Capability premium  (strong − weak cumulative payoff, 95% CI)")
    _save(fig, "f7_dissociation_by_game")


def main() -> int:
    p = argparse.ArgumentParser(description="ZAV-34 final result figures (EUR 0, read-only)")
    p.add_argument("--db", default="results/production.db")
    p.add_argument("--only", default=None, help="f1..f7 to render a single figure")
    args = p.parse_args()
    _register_fonts()

    df = ap.load_frame(args.db)
    print(f"Loaded {df.exp.nunique()} games, {len(df)} agent-rows -> figures/")

    figs = {
        "f1": lambda: fig1_premium(df),
        "f2": lambda: fig2_inequality(df),
        "f3": lambda: fig3_deception(args.db),
        "f4": lambda: fig4_message_classes(args.db),
        "f5": lambda: fig5_trust_transfer(args.db),
        "f6": lambda: fig6_tier_inference(args.db),
        "f7": lambda: fig7_dissociation(df),
    }
    todo = [args.only] if args.only else list(figs)
    for key in todo:
        print(f"[{key}]")
        figs[key]()
    print("DONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
