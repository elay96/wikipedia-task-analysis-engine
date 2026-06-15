"""Per-participant semantic + structural browsing features."""
from __future__ import annotations

import numpy as np
import networkx as nx

import _bootstrap  # noqa: F401
from m83_utils import forward_flow  # from repo scripts/


def step_distance_series(vectors):
    """Ordered list of cosine distances between consecutive vectors."""
    if len(vectors) < 2:
        return []
    V = np.stack([np.asarray(v, dtype=float) for v in vectors])
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    U = V / norms
    return [1.0 - float(U[i] @ U[i + 1]) for i in range(len(U) - 1)]


def step_distances(vectors):
    """Mean and variance of cosine distance between consecutive vectors."""
    dists = step_distance_series(vectors)
    if not dists:
        return float("nan"), float("nan")
    return float(np.mean(dists)), float(np.var(dists))


MIN_STEPS_FOR_DYNAMICS = 3


def dynamics_features(vectors) -> dict:
    """Within-session explore->exploit dynamics of semantic step distances.

    dyn_slope: OLS slope of step distance vs normalized step index (0..1).
        Negative = distances shrink over the session (converging / exploit);
        positive = distances grow (diverging / explore).
    dyn_early_late_delta: mean(second half of steps) - mean(first half).
    dyn_n_steps: number of step distances the participant contributed.
    Requires >= MIN_STEPS_FOR_DYNAMICS steps; otherwise slope/delta are nan."""
    dists = step_distance_series(vectors)
    n = len(dists)
    if n < MIN_STEPS_FOR_DYNAMICS:
        return {"dyn_slope": float("nan"), "dyn_early_late_delta": float("nan"),
                "dyn_n_steps": n}
    d = np.asarray(dists, dtype=float)
    x = np.linspace(0.0, 1.0, n)
    slope = float(np.polyfit(x, d, 1)[0])
    half = n // 2
    early_late = float(np.mean(d[half:]) - np.mean(d[:half]))
    return {"dyn_slope": slope, "dyn_early_late_delta": early_late, "dyn_n_steps": n}


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
