"""Network metrics (ZAV-27): synthetic structure recovery + DB builders.

Offline. Synthetic graphs with a planted structure verify that Louvain recovers
known communities and centrality flags the obvious hub; builder tests confirm the
Trust transfer edges and the PGG similarity clustering on seeded databases.
"""

from __future__ import annotations

import networkx as nx
import pytest

from asymmetric_society.metrics.network import (
    centrality,
    community_tier_alignment,
    louvain,
    modularity,
    pgg_cocontribution_graph,
    tier_assortativity,
    trust_transfer_graph,
)
from asymmetric_society.runner import db


def _two_cliques() -> nx.Graph:
    g = nx.Graph()
    for a in range(4):
        for b in range(a + 1, 4):
            g.add_edge(a, b, weight=1.0)
    for a in range(4, 8):
        for b in range(a + 1, 8):
            g.add_edge(a, b, weight=1.0)
    g.add_edge(3, 4, weight=0.1)  # weak bridge
    return g


# --------------------------------------------------------------------------- #
# Synthetic structure
# --------------------------------------------------------------------------- #


def test_louvain_recovers_two_communities() -> None:
    g = _two_cliques()
    part = louvain(g, seed=0)
    assert len(set(part.values())) == 2
    # The two cliques land in different communities.
    assert part[0] == part[1] == part[2] == part[3]
    assert part[4] == part[5] == part[6] == part[7]
    assert part[0] != part[4]
    assert modularity(g, part) > 0.0


def test_centrality_flags_star_hub() -> None:
    g = nx.star_graph(5)  # node 0 is the centre
    cent = centrality(g)
    assert max(cent, key=lambda n: cent[n]["degree"]) == 0
    assert max(cent, key=lambda n: cent[n]["betweenness"]) == 0


def test_tier_assortativity_positive_when_same_tier_connected() -> None:
    g = _two_cliques()
    for n in range(4):
        g.nodes[n]["tier"] = "strong"
    for n in range(4, 8):
        g.nodes[n]["tier"] = "weak"
    assert tier_assortativity(g) > 0.0  # like wires to like


def test_community_tier_alignment_ari() -> None:
    partition = {0: 0, 1: 0, 2: 1, 3: 1}
    tiers = {0: "strong", 1: "strong", 2: "weak", 3: "weak"}
    assert community_tier_alignment(partition, tiers)["ari"] == pytest.approx(1.0)
    shuffled = {0: 0, 1: 1, 2: 0, 3: 1}
    assert community_tier_alignment(shuffled, tiers)["ari"] < 1.0


def test_louvain_determinism() -> None:
    g = _two_cliques()
    assert louvain(g, seed=3) == louvain(g, seed=3)


# --------------------------------------------------------------------------- #
# Trust transfer graph
# --------------------------------------------------------------------------- #


def test_trust_transfer_graph_directed_weights(db_path: str) -> None:
    exp = "exp-trust-net"
    db.insert_agent(db_path, db.AgentRecord(exp, "inv", "claude", "strong"))
    db.insert_agent(db_path, db.AgentRecord(exp, "tru", "llama", "weak"))
    db.insert_action(db_path, db.ActionRecord(exp, 1, "inv", {"send": 5}, None, 0, 0.0, False))
    db.insert_action(db_path, db.ActionRecord(exp, 2, "tru", {"return": 3}, None, 0, 0.0, False))
    g = trust_transfer_graph(db_path, exp)
    assert g.is_directed()
    assert g["inv"]["tru"]["weight"] == pytest.approx(5.0)
    assert g["tru"]["inv"]["weight"] == pytest.approx(3.0)
    assert g.nodes["inv"]["tier"] == "strong"


# --------------------------------------------------------------------------- #
# PGG co-contribution similarity graph
# --------------------------------------------------------------------------- #


def _seed_two_behaviour_clusters(db_path: str) -> None:
    exp = "exp-pgg-net"
    high = ["h1", "h2", "h3"]  # high stable contributors -> labelled strong
    low = ["l1", "l2", "l3"]  # low contributors -> labelled weak
    for aid in high:
        db.insert_agent(db_path, db.AgentRecord(exp, aid, "m", "strong"))
    for aid in low:
        db.insert_agent(db_path, db.AgentRecord(exp, aid, "m", "weak"))
    plays = {**{a: [20, 20, 20, 20] for a in high}, **{a: [0, 1, 0, 1] for a in low}}
    for aid, seq in plays.items():
        for rnd, contribution in enumerate(seq, start=1):
            db.insert_action(
                db_path,
                db.ActionRecord(exp, rnd, aid, {"contribution": contribution}, None, 0, 0.0, False),
            )


def test_pgg_cocontribution_graph_clusters_by_behaviour(db_path: str) -> None:
    _seed_two_behaviour_clusters(db_path)
    g = pgg_cocontribution_graph(db_path, k=2)
    assert g.number_of_nodes() == 6
    part = louvain(g, seed=0)
    tiers = {n: g.nodes[n]["tier"] for n in g.nodes()}
    # The planted high/low behavioural split aligns with the tier labels.
    assert community_tier_alignment(part, tiers)["ari"] == pytest.approx(1.0)
