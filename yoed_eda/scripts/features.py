"""Per-participant semantic + structural browsing features."""
from __future__ import annotations

import numpy as np
import networkx as nx

import _bootstrap  # noqa: F401
from m83_utils import forward_flow  # from repo scripts/


def step_distances(vectors):
    """Mean and variance of cosine distance between consecutive vectors."""
    if len(vectors) < 2:
        return float("nan"), float("nan")
    V = np.stack([np.asarray(v, dtype=float) for v in vectors])
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    U = V / norms
    dists = [1.0 - float(U[i] @ U[i + 1]) for i in range(len(U) - 1)]
    return float(np.mean(dists)), float(np.var(dists))


def structural_features(visits: list, n_searches: int) -> dict:
    articles = [v["article"] for v in visits]
    dwells = [v["dwell_ms"] for v in visits if v.get("dwell_ms") is not None and not _isnan(v["dwell_ms"])]
    n_pages = len(articles)
    n_unique = len(set(articles))
    revisit_rate = (n_pages - n_unique) / n_pages if n_pages else float("nan")
    return {
        "n_pages": n_pages,
        "n_unique_pages": n_unique,
        "n_searches": n_searches,
        "revisit_rate": revisit_rate,
        "search_vs_link_ratio": (n_searches / n_pages) if n_pages else float("nan"),
        "mean_dwell": float(np.mean(dwells)) if dwells else float("nan"),
        "var_dwell": float(np.var(dwells)) if dwells else float("nan"),
    }


def visited_subgraph(visited_slugs, outlinks_by_slug) -> nx.Graph:
    """Undirected subgraph: nodes = unique visited slugs, edges where an
    outlink points to another visited slug."""
    nodes = list(dict.fromkeys(visited_slugs))
    node_set = set(nodes)
    G = nx.Graph()
    G.add_nodes_from(nodes)
    for s in nodes:
        for tgt in outlinks_by_slug.get(s, []):
            if tgt in node_set and tgt != s:
                G.add_edge(s, tgt)
    return G


def semantic_features(vectors) -> dict:
    mean_d, var_d = step_distances(vectors)
    return {
        "mean_step_distance": mean_d,
        "var_step_distance": var_d,
        "forward_flow": forward_flow(vectors) if len(vectors) >= 2 else float("nan"),
    }


def _isnan(x) -> bool:
    try:
        return np.isnan(x)
    except (TypeError, ValueError):
        return False
