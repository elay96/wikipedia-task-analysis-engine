"""M83 pure-logic helpers: slug<->title, network metrics, forward flow,
BH score, FDR-BH. Imported by m83a, m83b, and m83 main scripts. Fully
unit-tested in scripts/cleaning/test_m83_utils.py.
"""
from __future__ import annotations

from typing import Sequence
import networkx as nx
import numpy as np


def title_to_slug(title: str) -> str:
    """Wikipedia title (with spaces, possibly anchor) -> URL slug."""
    if not title:
        return ""
    head = title.split("#", 1)[0]
    return head.strip().replace(" ", "_")


def slug_to_title(slug: str) -> str:
    """URL slug -> human title (underscores -> spaces)."""
    return slug.replace("_", " ")


def network_metrics(G: nx.Graph) -> dict:
    """Compute Zhou's 4 BH-score metrics + sanity stats on an undirected graph.

    Returns dict with: n_unique, n_edges, clustering, char_path_length,
    global_efficiency, lcc_fraction. char_path_length is computed on the
    largest connected component (paper's formula assumes connectivity).
    """
    n = G.number_of_nodes()
    if n == 0:
        return {"n_unique": 0, "n_edges": 0, "clustering": 0.0,
                "char_path_length": 0.0, "global_efficiency": 0.0,
                "lcc_fraction": 0.0}

    if n == 1:
        return {"n_unique": 1, "n_edges": 0, "clustering": 0.0,
                "char_path_length": 0.0, "global_efficiency": 0.0,
                "lcc_fraction": 1.0}

    n_edges = G.number_of_edges()
    clustering = nx.average_clustering(G)
    global_eff = nx.global_efficiency(G)

    components = sorted(nx.connected_components(G), key=len, reverse=True)
    lcc = G.subgraph(components[0]).copy()
    lcc_fraction = len(lcc) / n
    if len(lcc) >= 2:
        char_path = nx.average_shortest_path_length(lcc)
    else:
        char_path = 0.0

    return {
        "n_unique": n,
        "n_edges": n_edges,
        "clustering": float(clustering),
        "char_path_length": float(char_path),
        "global_efficiency": float(global_eff),
        "lcc_fraction": float(lcc_fraction),
    }
