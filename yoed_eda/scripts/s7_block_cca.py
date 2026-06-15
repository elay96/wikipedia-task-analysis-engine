"""S7: canonical correlation between the creativity block and the browsing block.

One multivariate question ("is there any shared structure between the two
blocks?") instead of 195 pairwise tests. Reports the first canonical
correlation with a permutation p-value, a PLS cross-check, and the variable
loadings that define the shared axis."""
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import cca
from s4_creativity_behavior import FEATURES, MEASURES

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUTDIR = HERE.parent / "output"
FIGDIR = OUTDIR / "figures"

DYN_FEATURES = ["dyn_slope", "dyn_early_late_delta"]
N_PERM = 2000


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    participants = pd.read_csv(DATA / "participants.csv")
    feats = pd.read_csv(OUTDIR / "participant_features.csv")
    df = participants.merge(feats, on="participant_id", how="inner")

    x_cols = [m for m in MEASURES if m in df.columns]
    y_cols = [f for f in FEATURES + DYN_FEATURES if f in df.columns]
    X = df[x_cols].apply(pd.to_numeric, errors="coerce").values
    Y = df[y_cols].apply(pd.to_numeric, errors="coerce").values

    res = cca.block_cca(X, Y, perms=N_PERM)
    res["x_cols"] = x_cols
    res["y_cols"] = y_cols
    (OUTDIR / "cca_results.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"creativity block: {len(x_cols)} vars | browsing block: {len(y_cols)} vars | n={res['n']}")
    print(f"first canonical r = {res['r_canonical']:.3f} | permutation p = {res['p_perm']:.4f}")
    print(f"PLS first-component r = {res['r_pls']:.3f}")

    # Loadings bar chart: which variables define the shared axis.
    fig, axes = plt.subplots(1, 2, figsize=(11, 0.32 * max(len(x_cols), len(y_cols)) + 2))
    for ax, cols, load, title, color in [
        (axes[0], x_cols, res["x_loadings"], "Creativity loadings", "#E45756"),
        (axes[1], y_cols, res["y_loadings"], "Browsing loadings", "#4C78A8"),
    ]:
        order = sorted(range(len(cols)), key=lambda i: load[i])
        ax.barh([cols[i] for i in order], [load[i] for i in order], color=color)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="y", labelsize=7)
    fig.suptitle(f"First canonical variate (r={res['r_canonical']:.2f}, "
                 f"perm p={res['p_perm']:.3f})", fontsize=11)
    fig.tight_layout(); fig.savefig(FIGDIR / "cca_loadings.png", dpi=150); plt.close(fig)
    print(f"figures -> {FIGDIR}")


if __name__ == "__main__":
    main()
