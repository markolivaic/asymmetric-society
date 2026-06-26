"""Network metrics (ZAV-27): interaction/behaviour graphs + centrality + Louvain.

Two deliberately distinct graph mappings (a game maps to a graph two ways, and
conflating them would be a thesis error):

* :func:`trust_transfer_graph` - a **real directed token-transfer graph**. Trust
  gives literal A→B transfers (investor sends, trustee returns), so edges and
  weights are genuine interaction. *But* our Trust games are n=2 pairs, so a
  single game is a 2-node graph: centrality is degenerate and Louvain meaningless
  until ZAV-29 scale. The honest signal now is transfer asymmetry, not centrality.
* :func:`pgg_cocontribution_graph` - a **behavioural-similarity graph, NOT an
  interaction graph**. PGG has no A→B transfers (shared pool), so per the ticket's
  "co-contribution patterns" we connect agents by similarity of contribution
  behaviour (the ZAV-26 features), sparsified to a kNN graph. "Central" =
  behaviourally typical; "community" = behavioural cluster.

Caveats the thesis must carry (see Decision Log):

* The PGG similarity graph is built *from the ZAV-26 features*, so a high
  Louvain-vs-tier ARI is the **same information as ZAV-26 balanced accuracy with a
  different tool - NOT independent confirmation**. Only the Trust transfer graph is
  independent interaction structure.
* The kNN ``k`` (and RBF gamma) are a hidden threshold-confound: dense enough
  graphs always yield *some* communities. Sweep ``k`` and check ARI stability.
* Pseudoreplication: tier-aligned clusters on the pilot mean "Haiku-vs-Llama",
  not "strong-vs-weak in general".
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Sequence

import community as community_louvain
import networkx as nx
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.pairwise import rbf_kernel

from asymmetric_society.metrics.bayesian import extract_features
from asymmetric_society.runner import db

#: kNN neighbourhood sizes swept in the CLI to expose threshold sensitivity.
K_SWEEP: tuple[int, ...] = (4, 8, 15)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def trust_transfer_graph(db_path: str, experiment_id: str) -> nx.DiGraph:
    """Directed weighted token-transfer graph for one Trust game (n=2).

    An edge ``A -> B`` carries the total tokens A transferred to B across the game
    (investor sends, trustee returns). Nodes carry a ``tier`` attribute.

    Raises:
        ValueError: If the game has more than two agents (Trust is 2-player; the
            counterparty of a transfer is otherwise ambiguous).
    """
    con = db.connect(db_path)
    try:
        agent_rows = con.execute(
            "SELECT agent_id, capability_tier FROM agents WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchall()
        act_rows = con.execute(
            "SELECT agent_id, action_json FROM actions WHERE experiment_id = ? ORDER BY round_num",
            (experiment_id,),
        ).fetchall()
    finally:
        con.close()

    tiers = {aid: tier for aid, tier in agent_rows}
    agent_ids = list(tiers)
    if len(agent_ids) > 2:
        raise ValueError("trust_transfer_graph supports 2-player Trust games only")

    g = nx.DiGraph()
    for aid, tier in tiers.items():
        g.add_node(aid, tier=tier)

    for agent_id, action_json in act_rows:
        action = json.loads(action_json)
        amount = action.get("send", action.get("return"))
        if amount is None:
            continue  # no-op / idle turn carries no transfer
        others = [a for a in agent_ids if a != agent_id]
        if not others:
            continue
        other = others[0]
        weight = float(amount)
        if g.has_edge(agent_id, other):
            g[agent_id][other]["weight"] += weight
        else:
            g.add_edge(agent_id, other, weight=weight)
    return g


def pgg_cocontribution_graph(
    db_path: str,
    experiment_ids: Sequence[str] | None = None,
    *,
    k: int = 8,
    gamma: float | None = None,
) -> nx.Graph:
    """Behavioural-similarity graph over PGG agents (NOT an interaction graph).

    Nodes are ``(experiment_id, agent_id)`` agent-instances pooled across games;
    edge weight is the RBF similarity of their standardized ZAV-26 behaviour
    vectors, sparsified to each node's ``k`` nearest neighbours. Nodes carry a
    ``tier`` attribute.

    Args:
        db_path: Results database.
        experiment_ids: Restrict to these experiments (all if None).
        k: Neighbours kept per node (capped at n-1).
        gamma: RBF kernel width (sklearn default ``1/n_features`` if None).

    Returns:
        An undirected weighted similarity graph.
    """
    X, y, ids, _names = extract_features(db_path, experiment_ids, game="public_goods")
    g = nx.Graph()
    for (exp, aid), label in zip(ids, y):
        g.add_node((exp, aid), tier="strong" if label == 1 else "weak")
    n = len(ids)
    if n < 2:
        return g

    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    xz = (X - mu) / sd
    sim = rbf_kernel(xz, gamma=gamma)
    np.fill_diagonal(sim, 0.0)

    kk = min(k, n - 1)
    for i in range(n):
        neighbours = np.argsort(sim[i])[::-1][:kk]
        for j in neighbours:
            w = float(sim[i, j])
            if w <= 0:
                continue
            a, b = ids[i], ids[j]
            if g.has_edge(a, b):
                g[a][b]["weight"] = max(g[a][b]["weight"], w)
            else:
                g.add_edge(a, b, weight=w)
    return g


# --------------------------------------------------------------------------- #
# Graph metrics (game-agnostic)
# --------------------------------------------------------------------------- #


def _undirected(g: nx.Graph) -> nx.Graph:
    """Undirected weighted view (directed edge weights summed per pair)."""
    if not g.is_directed():
        return g
    u = nx.Graph()
    u.add_nodes_from(g.nodes(data=True))
    for a, b, data in g.edges(data=True):
        w = float(data.get("weight", 1.0))
        if u.has_edge(a, b):
            u[a][b]["weight"] += w
        else:
            u.add_edge(a, b, weight=w)
    return u


def centrality(g: nx.Graph) -> dict[Any, dict[str, float]]:
    """Per-node degree, betweenness, and eigenvector centrality.

    Computed on the undirected weighted view. Degree and eigenvector use edge
    weights (tie strength); betweenness is topological (unweighted) to avoid
    misreading similarity weights as path distances. Each measure degrades to 0.0
    on graphs too small/degenerate for it.
    """
    u = _undirected(g)
    nodes = list(u.nodes())
    degree = nx.degree_centrality(u) if len(nodes) > 1 else {n: 0.0 for n in nodes}
    betweenness = nx.betweenness_centrality(u) if len(nodes) > 2 else {n: 0.0 for n in nodes}
    # eigenvector_centrality_numpy raises AmbiguousSolution on a DISCONNECTED graph
    # (common at scale), which previously zeroed every node. Compute it per connected
    # component so centrality stays meaningful within each component.
    eigenvector: dict[Any, float] = {}
    for comp in nx.connected_components(u):
        sub = u.subgraph(comp)
        try:
            eigenvector.update(nx.eigenvector_centrality_numpy(sub, weight="weight"))
        except (nx.NetworkXException, ValueError, TypeError):
            eigenvector.update({n: 0.0 for n in sub.nodes()})
    return {
        n: {
            "degree": float(degree.get(n, 0.0)),
            "betweenness": float(betweenness.get(n, 0.0)),
            "eigenvector": float(eigenvector.get(n, 0.0)),
        }
        for n in nodes
    }


def louvain(g: nx.Graph, *, seed: int = 0) -> dict[Any, int]:
    """Louvain community assignment per node (seeded for reproducibility)."""
    u = _undirected(g)
    if u.number_of_edges() == 0:
        return {n: i for i, n in enumerate(u.nodes())}
    return community_louvain.best_partition(u, weight="weight", random_state=seed)


def modularity(g: nx.Graph, partition: dict[Any, int]) -> float:
    """Modularity of ``partition`` on the (undirected) graph."""
    u = _undirected(g)
    if u.number_of_edges() == 0:
        return 0.0
    return float(community_louvain.modularity(partition, u, weight="weight"))


def tier_assortativity(g: nx.Graph) -> float:
    """Assortativity by the ``tier`` node attribute (do same-tier agents connect?).

    Returns ``nan`` when undefined (e.g. a single tier or too few edges).
    """
    u = _undirected(g)
    if u.number_of_edges() == 0:
        return float("nan")
    try:
        return float(nx.attribute_assortativity_coefficient(u, "tier"))
    except (nx.NetworkXError, ValueError, KeyError):
        return float("nan")


def community_tier_alignment(
    partition: dict[Any, int], tiers: dict[Any, str]
) -> dict[str, Any]:
    """Adjusted Rand index between a Louvain partition and tier labels."""
    nodes = [n for n in partition if n in tiers]
    if not nodes:
        return {"ari": float("nan"), "n_communities": 0}
    pred = [partition[n] for n in nodes]
    true = [tiers[n] for n in nodes]
    return {
        "ari": float(adjusted_rand_score(true, pred)),
        "n_communities": len(set(pred)),
    }


def centrality_by_tier(
    g: nx.Graph, cent: dict[Any, dict[str, float]]
) -> dict[str, dict[str, float]]:
    """Mean of each centrality measure grouped by node tier."""
    groups: dict[str, list[dict[str, float]]] = {}
    for node, measures in cent.items():
        tier = g.nodes[node].get("tier", "unknown")
        groups.setdefault(tier, []).append(measures)
    out: dict[str, dict[str, float]] = {}
    for tier, rows in groups.items():
        out[tier] = {
            key: float(np.mean([r[key] for r in rows]))
            for key in ("degree", "betweenness", "eigenvector")
        }
    return out


def graph_summary(g: nx.Graph, *, seed: int = 0) -> dict[str, Any]:
    """Bundle the structural metrics and tier reads for one graph."""
    cent = centrality(g)
    partition = louvain(g, seed=seed)
    tiers = {n: g.nodes[n].get("tier", "unknown") for n in g.nodes()}
    alignment = community_tier_alignment(partition, tiers)
    return {
        "n_nodes": g.number_of_nodes(),
        "n_edges": g.number_of_edges(),
        "density": float(nx.density(g)) if g.number_of_nodes() > 1 else 0.0,
        "modularity": modularity(g, partition),
        "n_communities": alignment["n_communities"],
        "ari_vs_tier": alignment["ari"],
        "tier_assortativity": tier_assortativity(g),
        "centrality": cent,
        "centrality_by_tier": centrality_by_tier(g, cent),
        "partition": partition,
    }


# --------------------------------------------------------------------------- #
# CLI (read-only; €0)
# --------------------------------------------------------------------------- #


def _experiment_ids(db_path: str) -> list[str]:
    con = db.connect(db_path)
    try:
        return [row[0] for row in con.execute("SELECT id FROM experiments").fetchall()]
    finally:
        con.close()


def _run_pgg(db_path: str, seed: int) -> None:
    print("PGG co-contribution graph (BEHAVIOURAL SIMILARITY, not interaction)\n")
    base = pgg_cocontribution_graph(db_path, k=8)
    tiers = {n: base.nodes[n]["tier"] for n in base.nodes()}
    n_strong = sum(t == "strong" for t in tiers.values())
    print(f"  agents (nodes): {base.number_of_nodes()}  (strong={n_strong}, "
          f"weak={base.number_of_nodes() - n_strong})")
    print(f"  tier assortativity (k=8): {tier_assortativity(base):+.3f}")
    cent = centrality(base)
    by_tier = centrality_by_tier(base, cent)
    print("  mean eigenvector centrality by tier (k=8):")
    for tier, measures in sorted(by_tier.items()):
        print(f"    {tier:7} eigenvector={measures['eigenvector']:.3f}  degree={measures['degree']:.3f}")
    print("\n  Louvain-vs-tier ARI sensitivity over k (is the cluster real or a threshold artifact?):")
    for k in K_SWEEP:
        g = pgg_cocontribution_graph(db_path, k=k)
        part = louvain(g, seed=seed)
        align = community_tier_alignment(part, {n: g.nodes[n]["tier"] for n in g.nodes()})
        print(f"    k={k:<2}  communities={align['n_communities']}  "
              f"modularity={modularity(g, part):.3f}  ARI_vs_tier={align['ari']:+.3f}")
    print("\n  NOTE: this graph is derived from the ZAV-26 features, so ARI-vs-tier is the SAME")
    print("  information as ZAV-26 balanced accuracy, not independent confirmation. Only the Trust")
    print("  transfer graph is independent interaction structure. Pilot = Haiku-vs-Llama, not general.")


def _run_trust(db_path: str, seed: int) -> None:
    ids = _experiment_ids(db_path)
    if not ids:
        print("No experiments in DB.")
        return
    exp = ids[0]
    g = trust_transfer_graph(db_path, exp)
    print(f"Trust token-transfer graph (REAL directed interaction), experiment {exp}\n")
    print(f"  nodes: {g.number_of_nodes()}  edges: {g.number_of_edges()}")
    for a, b, data in g.edges(data=True):
        ta, tb = g.nodes[a]["tier"], g.nodes[b]["tier"]
        print(f"    {a} ({ta}) -> {b} ({tb})  transferred {data['weight']:.0f}")
    print("\n  NOTE: n=2 pair -> centrality/Louvain are degenerate here. This is an EXTRACTION")
    print("  PROOF only, not a result; a real transfer network needs ZAV-29 scale.")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI: build a graph from a results DB and print structural metrics (€0)."""
    parser = argparse.ArgumentParser(description="Network metrics (ZAV-27).")
    parser.add_argument("--db", required=True, help="Results database.")
    parser.add_argument("--game", choices=("trust", "pgg"), default="pgg")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.game == "pgg":
        _run_pgg(args.db, args.seed)
    else:
        _run_trust(args.db, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
