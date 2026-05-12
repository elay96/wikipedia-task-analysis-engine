"""M83 pure-logic helpers: slug<->title, network metrics, forward flow,
BH score, FDR-BH. Imported by m83a, m83b, and m83 main scripts. Fully
unit-tested in scripts/cleaning/test_m83_utils.py.
"""
from __future__ import annotations

from typing import Sequence
import networkx as nx
import numpy as np
import pandas as pd


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


def forward_flow(vectors: Sequence[np.ndarray]) -> float:
    """Average cosine distance from each vector to all earlier vectors,
    averaged across positions i >= 2. Returns NaN if sequence has fewer
    than 2 vectors. Vectors do not need to be pre-normalised; the function
    normalises on the fly.

    Matches Gray (2019) definition, used by Zhou (2024) for Dancer style.
    """
    n = len(vectors)
    if n < 2:
        return float("nan")

    V = np.stack([np.asarray(v, dtype=float) for v in vectors])
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    V_unit = V / norms

    per_position = []
    for i in range(1, n):
        sims = V_unit[:i] @ V_unit[i]
        dists = 1.0 - sims
        per_position.append(float(np.mean(dists)))
    return float(np.mean(per_position))


def bh_score(df: pd.DataFrame) -> list:
    """Aggregate Zhou's 4 metrics into a single busybody-hunter score per row.

    Higher = more hunter-like (tight, connected). Lower = more busybody-like.
    Requires columns: n_edges, clustering, global_efficiency, char_path_length.
    Z-scoring is within the input DataFrame (i.e., within-cohort).
    Constant columns contribute 0 (no division-by-zero).
    """
    cols = ["n_edges", "clustering", "global_efficiency", "char_path_length"]
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"bh_score: missing column {c!r}")

    def z(s: pd.Series) -> np.ndarray:
        a = s.to_numpy(dtype=float)
        sd = float(np.std(a, ddof=0))
        if sd == 0:
            return np.zeros_like(a)
        return (a - float(np.mean(a))) / sd

    z_edges = z(df["n_edges"])
    z_clust = z(df["clustering"])
    z_eff = z(df["global_efficiency"])
    z_path = z(df["char_path_length"])
    return list(z_edges + z_clust + z_eff - z_path)


def fdr_bh(pvals: Sequence[float]) -> list:
    """Benjamini-Hochberg adjusted p-values, clipped to [0, 1].

    NaN inputs are preserved as NaN; m equals the count of finite p-values.
    The returned list has the same length and order as the input.
    """
    p = np.asarray(pvals, dtype=float)
    finite_mask = np.isfinite(p)
    out = np.full(p.shape, np.nan)

    m = int(finite_mask.sum())
    if m == 0:
        return out.tolist()

    finite_idx = np.where(finite_mask)[0]
    p_finite = p[finite_idx]
    order = np.argsort(p_finite)
    ranked = p_finite[order]
    adj = ranked * m / (np.arange(m) + 1.0)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)

    out_finite = np.empty(m)
    out_finite[order] = adj
    out[finite_idx] = out_finite
    return out.tolist()


def collect_diffuse_slugs(game_df: pd.DataFrame) -> set:
    """Unique ArticleSlug from article_open rows in Condition=='diffuse'."""
    cond = game_df["Condition"].astype(str).str.lower() == "diffuse"
    opens = game_df[cond & (game_df["Action"] == "article_open")]
    slugs = opens["ArticleSlug"].dropna().astype(str).unique()
    return set(slugs)
