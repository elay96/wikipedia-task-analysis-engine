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
