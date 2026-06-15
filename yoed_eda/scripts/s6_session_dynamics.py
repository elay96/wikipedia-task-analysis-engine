"""S6: correlate within-session explore->exploit dynamics with creativity.

Dynamics features (from s3): dyn_slope (negative = converging/exploit over the
session) and dyn_early_late_delta. Only participants with enough steps have
non-nan dynamics; pairwise-complete Spearman + BH-FDR handles the rest."""
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import analysis
from s4_creativity_behavior import MEASURES

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUTDIR = HERE.parent / "output"
FIGDIR = OUTDIR / "figures"

DYN_FEATURES = ["dyn_slope", "dyn_early_late_delta"]


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    participants = pd.read_csv(DATA / "participants.csv")
    feats = pd.read_csv(OUTDIR / "participant_features.csv")
    df = participants.merge(feats, on="participant_id", how="inner")

    measures = [m for m in MEASURES if m in df.columns]
    features = [f for f in DYN_FEATURES if f in df.columns]
    corr = analysis.spearman_matrix(df, measures, features)
    corr.to_csv(OUTDIR / "dynamics_correlations.csv", index=False)

    n_with_dyn = int(df["dyn_slope"].notna().sum())
    n_sig = int(corr["fdr_significant"].sum())
    print(f"participants with dynamics: {n_with_dyn}/{len(df)}")
    print(f"dynamics correlations: {len(corr)} | FDR-significant: {n_sig}")

    # Distribution of dyn_slope (negative bars = converging sessions).
    slopes = pd.to_numeric(df["dyn_slope"], errors="coerce").dropna()
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.hist(slopes, bins=20, color="#4C78A8", alpha=0.85)
    ax.axvline(0, color="black", lw=1)
    ax.set_xlabel("dyn_slope (semantic step distance over session)")
    ax.set_ylabel("participants")
    ax.set_title(f"median={slopes.median():.3f}, n={len(slopes)}", fontsize=9)
    fig.tight_layout(); fig.savefig(FIGDIR / "dyn_slope_hist.png", dpi=150); plt.close(fig)

    # Top dynamics pairs by |rho|.
    top = corr.dropna(subset=["rho"]).reindex(
        corr["rho"].abs().sort_values(ascending=False).index).head(4)
    for _, r in top.iterrows():
        x = pd.to_numeric(df[r["measure"]], errors="coerce")
        y = pd.to_numeric(df[r["feature"]], errors="coerce")
        fig, ax = plt.subplots(figsize=(4, 3.2))
        ax.scatter(x, y, s=18, alpha=0.7)
        ax.set_xlabel(r["measure"], fontsize=8); ax.set_ylabel(r["feature"], fontsize=8)
        ax.set_title(f"rho={r['rho']:.2f}, p_FDR={r['p_FDR']:.3f}", fontsize=9)
        fig.tight_layout()
        safe = f"dyn_{r['measure']}__{r['feature']}".replace(" ", "_").replace("/", "-")
        fig.savefig(FIGDIR / f"scatter_{safe}.png", dpi=150); plt.close(fig)

    print(f"figures -> {FIGDIR}")


if __name__ == "__main__":
    main()
