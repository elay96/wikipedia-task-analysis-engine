import numpy as np
import cca


def test_block_cca_detects_strong_shared_signal():
    rng = np.random.RandomState(1)
    z = rng.randn(80, 1)  # shared latent factor
    X = np.hstack([z + 0.2 * rng.randn(80, 1), rng.randn(80, 2)])
    Y = np.hstack([z + 0.2 * rng.randn(80, 1), rng.randn(80, 2)])
    res = cca.block_cca(X, Y, perms=200)
    assert res["r_canonical"] > 0.7
    assert res["p_perm"] < 0.05
    assert res["n"] == 80


def test_block_cca_null_when_blocks_independent():
    rng = np.random.RandomState(2)
    X = rng.randn(80, 3)
    Y = rng.randn(80, 3)
    res = cca.block_cca(X, Y, perms=200)
    # Independent blocks: observed r should sit inside the permutation null.
    assert res["p_perm"] > 0.05


def test_block_cca_imputes_nan():
    rng = np.random.RandomState(3)
    X = rng.randn(60, 2)
    Y = rng.randn(60, 2)
    X[0, 0] = np.nan
    res = cca.block_cca(X, Y, perms=50)
    assert not np.isnan(res["r_canonical"])
