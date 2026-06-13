import numpy as np
import pandas as pd
import analysis


def test_spearman_matrix_perfect_monotonic():
    df = pd.DataFrame({"m1": [1, 2, 3, 4, 5], "b1": [2, 4, 6, 8, 10]})
    out = analysis.spearman_matrix(df, ["m1"], ["b1"])
    row = out.iloc[0]
    assert row["measure"] == "m1" and row["feature"] == "b1"
    assert abs(row["rho"] - 1.0) < 1e-9
    assert "p_FDR" in out.columns and "fdr_significant" in out.columns


def test_spearman_matrix_handles_nan_pairs():
    df = pd.DataFrame({"m1": [1, 2, np.nan, 4], "b1": [1, np.nan, 3, 4]})
    out = analysis.spearman_matrix(df, ["m1"], ["b1"])
    assert len(out) == 1  # uses pairwise-complete (n=2 here)


def test_fdr_flag_threshold():
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "m1": list(range(20)), "m2": list(range(20)),
        "b1": list(range(20)), "b2": list(rng.rand(20)),
    })
    out = analysis.spearman_matrix(df, ["m1", "m2"], ["b1", "b2"])
    assert out["fdr_significant"].dtype == bool
    # m1 vs b1 is perfectly monotonic -> must survive FDR
    perfect = out[(out["measure"] == "m1") & (out["feature"] == "b1")].iloc[0]
    assert perfect["fdr_significant"]
