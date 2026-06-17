import numpy as np
import pandas as pd
import creativity_model as cm


def test_zscore_centers_and_scales():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    z = cm.zscore(s)
    assert abs(z.mean()) < 1e-9
    assert abs(z.std(ddof=0) - 1.0) < 1e-9


def test_zscore_constant_is_zero():
    z = cm.zscore(pd.Series([7.0, 7.0, 7.0]))
    assert (z == 0).all()


def test_build_composites_singleton_equals_zscore():
    df = pd.DataFrame({
        "participant_id": [1, 2, 3, 4],
        "Curiosity - Score": [1.0, 2.0, 3.0, 4.0],
        "GF - Score": [4.0, 3.0, 2.0, 1.0],
        "Verbal Fluency - Forward Flow": [0.0, 1.0, 2.0, 3.0],
        "AUT Broom - Number of Answers": [1.0, 1.0, 1.0, 1.0],
        "AUT Belt - Number of Answers": [1.0, 2.0, 3.0, 4.0],
        "AQT Pencil - Number of Answers": [1.0, 2.0, 3.0, 4.0],
        "AQT Pillow - Number of Answers": [1.0, 2.0, 3.0, 4.0],
        "Verbal Fluency - Number of Answers": [1.0, 2.0, 3.0, 4.0],
        "AUT Belt - Originality": [1.0, 2.0, 3.0, 4.0],
        "AUT Broom - Originality": [1.0, 2.0, 3.0, 4.0],
        "AQT Pencil - Originality": [1.0, 2.0, 3.0, 4.0],
        "AQT Pillow - Originality": [1.0, 2.0, 3.0, 4.0],
        "AQT Complexity Score": [1.0, 2.0, 3.0, 4.0],
    })
    out = cm.build_composites(df)
    assert list(out["participant_id"]) == [1, 2, 3, 4]
    assert set(cm.PREDICTORS) <= set(out.columns)
    expected = cm.zscore(df["Curiosity - Score"])
    assert np.allclose(out["curiosity"].to_numpy(), expected.to_numpy())


def test_build_composites_fluency_is_mean_of_zscores():
    df = pd.DataFrame({
        "participant_id": [1, 2, 3, 4],
        "Curiosity - Score": [1.0, 2.0, 3.0, 4.0],
        "GF - Score": [1.0, 2.0, 3.0, 4.0],
        "Verbal Fluency - Forward Flow": [1.0, 2.0, 3.0, 4.0],
        "AUT Broom - Number of Answers": [1.0, 2.0, 3.0, 4.0],
        "AUT Belt - Number of Answers": [1.0, 2.0, 3.0, 4.0],
        "AQT Pencil - Number of Answers": [1.0, 2.0, 3.0, 4.0],
        "AQT Pillow - Number of Answers": [1.0, 2.0, 3.0, 4.0],
        "Verbal Fluency - Number of Answers": [1.0, 2.0, 3.0, 4.0],
        "AUT Belt - Originality": [1.0, 2.0, 3.0, 4.0],
        "AUT Broom - Originality": [1.0, 2.0, 3.0, 4.0],
        "AQT Pencil - Originality": [1.0, 2.0, 3.0, 4.0],
        "AQT Pillow - Originality": [1.0, 2.0, 3.0, 4.0],
        "AQT Complexity Score": [1.0, 2.0, 3.0, 4.0],
    })
    out = cm.build_composites(df)
    # all five fluency constituents identical after z-scoring -> mean equals that z-score
    expected = cm.zscore(df["AUT Belt - Number of Answers"])
    assert np.allclose(out["dt_fluency"].to_numpy(), expected.to_numpy())


def test_regress_recovers_strong_predictor():
    rng = np.random.RandomState(0)
    n = 200
    signal = rng.randn(n)
    X = np.column_stack([signal, rng.randn(n), rng.randn(n)])
    y = 2.0 * signal + 0.1 * rng.randn(n)
    res = cm.regress_with_ci(X, y, ["a", "b", "c"], n_boot=300, n_perm=300)
    assert res["n"] == 200
    assert res["beta_ci"][0][0] > 0  # signal predictor CI excludes zero
    assert res["r2_full"] > 0.9
    assert res["r2_cv"] > 0.8
    assert res["p_perm"] < 0.05


def test_regress_null_when_no_signal():
    rng = np.random.RandomState(1)
    X = rng.randn(120, 3)
    y = rng.randn(120)
    res = cm.regress_with_ci(X, y, ["a", "b", "c"], n_boot=200, n_perm=300)
    assert res["p_perm"] > 0.05


def test_regress_drops_nan_rows():
    rng = np.random.RandomState(2)
    X = rng.randn(50, 2)
    y = rng.randn(50)
    y[0] = np.nan
    X[1, 0] = np.nan
    res = cm.regress_with_ci(X, y, ["a", "b"], n_boot=50, n_perm=50)
    assert res["n"] == 48


def test_convergent_validity_strong_positive():
    rng = np.random.RandomState(0)
    x = rng.randn(100)
    y = x + 0.3 * rng.randn(100)
    res = cm.convergent_validity(x, y)
    assert res["n"] == 100
    assert res["pearson_r"] > 0.8
    assert res["pearson_ci"][0] > 0
    assert res["spearman_rho"] > 0.7


def test_convergent_validity_pairwise_complete():
    x = [1.0, 2.0, 3.0, np.nan, 5.0]
    y = [1.0, 2.0, np.nan, 4.0, 5.0]
    res = cm.convergent_validity(x, y)
    assert res["n"] == 3
    assert np.isnan(res["pearson_ci"][0])


def test_pca_reduce_shape_and_variance():
    rng = np.random.RandomState(0)
    latent = rng.randn(80, 1)
    cols = {f"m{i}": latent[:, 0] + 0.05 * rng.randn(80) for i in range(5)}
    df = pd.DataFrame(cols)
    comps, evr = cm.pca_reduce(df, list(cols), n_components=3)
    assert comps.shape == (80, 3)
    assert evr[0] > 0.9


def test_pca_reduce_imputes_nan():
    rng = np.random.RandomState(1)
    df = pd.DataFrame(rng.randn(40, 4), columns=list("abcd"))
    df.iloc[0, 0] = np.nan
    comps, evr = cm.pca_reduce(df, list("abcd"), n_components=2)
    assert comps.shape == (40, 2)
    assert not np.isnan(comps).any()
