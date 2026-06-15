"""Canonical correlation between two variable blocks, with a permutation test.

CCA overfits when the number of variables is large relative to N (here ~28
variables vs ~107 participants), so the in-sample canonical correlation is
optimistic. The permutation test (shuffle the rows of one block, recompute)
gives an honest significance estimate that absorbs that optimism. PLSCanonical
is reported alongside as a more shrinkage-robust cross-check."""
from __future__ import annotations

import numpy as np
from sklearn.cross_decomposition import CCA, PLSCanonical
from sklearn.preprocessing import StandardScaler


def _first_canonical_corr(model, Xs, Ys) -> float:
    xc, yc = model.transform(Xs, Ys)
    r = np.corrcoef(xc[:, 0], yc[:, 0])[0, 1]
    return abs(float(r))


def _prep(X, Y):
    """Median-impute then z-score each block; return scaled arrays."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    for A in (X, Y):
        for j in range(A.shape[1]):
            col = A[:, j]
            med = np.nanmedian(col)
            col[np.isnan(col)] = med if not np.isnan(med) else 0.0
    Xs = StandardScaler().fit_transform(X)
    Ys = StandardScaler().fit_transform(Y)
    return Xs, Ys


def block_cca(X, Y, perms, seed=0) -> dict:
    """First canonical correlation between blocks X and Y + permutation p-value.

    perms: number of row-shuffles of Y used to build the null. The p-value is
    (1 + #{perm r >= observed r}) / (1 + perms)."""
    Xs, Ys = _prep(X, Y)
    n, p, q = Xs.shape[0], Xs.shape[1], Ys.shape[1]
    k = min(p, q)
    cca = CCA(n_components=1, max_iter=1000).fit(Xs, Ys)
    r_obs = _first_canonical_corr(cca, Xs, Ys)

    rng = np.random.RandomState(seed)
    count = 0
    for _ in range(perms):
        perm = rng.permutation(n)
        m = CCA(n_components=1, max_iter=1000).fit(Xs, Ys[perm])
        if _first_canonical_corr(m, Xs, Ys[perm]) >= r_obs:
            count += 1
    p_value = (1 + count) / (1 + perms)

    pls = PLSCanonical(n_components=1, max_iter=1000).fit(Xs, Ys)
    r_pls = _first_canonical_corr(pls, Xs, Ys)

    return {"r_canonical": r_obs, "p_perm": p_value, "n": n,
            "n_components": k, "r_pls": r_pls,
            "x_loadings": cca.x_loadings_[:, 0].tolist(),
            "y_loadings": cca.y_loadings_[:, 0].tolist()}
