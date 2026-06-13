"""S4: correlate creativity/cognition measures with browsing features."""
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import analysis

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUTDIR = HERE.parent / "output"
FIGDIR = OUTDIR / "figures"

MEASURES = [
    "Verbal Fluency - Number of Answers", "Verbal Fluency - Forward Flow",
    "AUT Broom - Number of Answers", "AUT Belt - Number of Answers",
    "AUT Belt - Originality", "AUT Broom - Originality",
    "AQT Pencil - Number of Answers", "AQT Pillow - Number of Answers",
    "AQT Pencil - Originality", "AQT Pillow - Originality",
    "AQT Complexity Score", "Curiosity - Score", "GF - Score",
]
FEATURES = [
    "mean_step_distance", "var_step_distance", "forward_flow",
    "bh_score", "clustering", "global_efficiency", "char_path_length",
    "n_pages", "n_unique_pages", "n_searches", "revisit_rate",
    "search_vs_link_ratio", "mean_dwell", "var_dwell", "path_breadth",
]


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    participants = pd.read_csv(DATA / "participants.csv")
    feats = pd.read_csv(OUTDIR / "participant_features.csv")
    df = participants.merge(feats, on="participant_id", how="inner")

    measures = [m for m in MEASURES if m in df.columns]
    features = [f for f in FEATURES if f in df.columns]
    corr = analysis.spearman_matrix(df, measures, features)
    corr.to_csv(OUTDIR / "spearman_correlations.csv", index=False)

    n_sig = int(corr["fdr_significant"].sum())
    print(f"correlations: {len(corr)} | FDR-significant: {n_sig}")

    # Heatmap of rho.
    pivot = corr.pivot(index="measure", columns="feature", values="rho").reindex(index=measures, columns=features)
    fig, ax = plt.subplots(figsize=(0.7 * len(features) + 4, 0.5 * len(measures) + 3))
    im = ax.imshow(pivot.values, cmap="RdBu_r", vmin=-0.6, vmax=0.6, aspect="auto")
    ax.set_xticks(range(len(features))); ax.set_xticklabels(features, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(measures))); ax.set_yticklabels(measures, fontsize=8)
    for _, r in corr.iterrows():
        if r["fdr_significant"]:
            ax.text(features.index(r["feature"]), measures.index(r["measure"]), "*",
                    ha="center", va="center", color="black", fontsize=12)
    fig.colorbar(im, ax=ax, label="Spearman rho")
    fig.tight_layout(); fig.savefig(FIGDIR / "corr_heatmap.png", dpi=150); plt.close(fig)

    # Scatter for each FDR-significant pair (or top-6 by |rho| if none).
    sig = corr[corr["fdr_significant"]]
    pairs = sig if len(sig) else corr.dropna(subset=["rho"]).reindex(corr["rho"].abs().sort_values(ascending=False).index).head(6)
    for _, r in pairs.iterrows():
        x = pd.to_numeric(df[r["measure"]], errors="coerce"); y = pd.to_numeric(df[r["feature"]], errors="coerce")
        fig, ax = plt.subplots(figsize=(4, 3.2))
        ax.scatter(x, y, s=18, alpha=0.7)
        ax.set_xlabel(r["measure"], fontsize=8); ax.set_ylabel(r["feature"], fontsize=8)
        ax.set_title(f"rho={r['rho']:.2f}, p_FDR={r['p_FDR']:.3f}", fontsize=9)
        fig.tight_layout()
        safe = f"{r['measure']}__{r['feature']}".replace(" ", "_").replace("/", "-")
        fig.savefig(FIGDIR / f"scatter_{safe}.png", dpi=150); plt.close(fig)

    print(f"figures -> {FIGDIR}")


if __name__ == "__main__":
    main()
