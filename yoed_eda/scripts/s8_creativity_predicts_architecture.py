# s8_creativity_predicts_architecture.py
"""S8: does trait creativity predict browsing architecture?

Tests the dissociation (creativity -> forward_flow/Dancer, not bh_score/BH) and
the Verbal-Fluency-Forward-Flow -> browsing-forward_flow convergent validity,
with a PCA robustness check. See
docs/superpowers/specs/2026-06-17-creativity-predicts-browsing-architecture-design.md"""
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import creativity_model as cm
from s4_creativity_behavior import MEASURES

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUTDIR = HERE.parent / "output"
FIGDIR = OUTDIR / "figures"
OUTCOMES = ["forward_flow", "bh_score"]
VFF_COL = "Verbal Fluency - Forward Flow"
OUTCOME_COLORS = {"forward_flow": "#4C78A8", "bh_score": "#E45756"}


def _plot_effects(primary: dict, path: Path) -> None:
    preds = primary[OUTCOMES[0]]["predictors"]
    y = np.arange(len(preds))
    fig, ax = plt.subplots(figsize=(7, 0.6 * len(preds) + 2))
    for offset, oc in zip((-0.15, 0.15), OUTCOMES):
        r = primary[oc]
        betas = r["betas"]
        err = [[b - c[0] for b, c in zip(betas, r["beta_ci"])],
               [c[1] - b for b, c in zip(betas, r["beta_ci"])]]
        ax.errorbar(betas, y + offset, xerr=err, fmt="o", capsize=3,
                    color=OUTCOME_COLORS[oc], label=oc)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(preds)
    ax.set_xlabel("standardized beta (95% CI)")
    ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def _plot_convergent(df: pd.DataFrame, conv: dict, path: Path) -> None:
    x = pd.to_numeric(df[VFF_COL], errors="coerce")
    y = pd.to_numeric(df["forward_flow"], errors="coerce")
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.scatter(x, y, s=18, alpha=0.7)
    ax.set_xlabel("Verbal Fluency - Forward Flow (words)")
    ax.set_ylabel("forward_flow (browsing)")
    ax.set_title(f"r={conv['pearson_r']:.2f}, "
                 f"CI=[{conv['pearson_ci'][0]:.2f},{conv['pearson_ci'][1]:.2f}], "
                 f"n={conv['n']}", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def main() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    participants = pd.read_csv(DATA / "participants.csv")
    feats = pd.read_csv(OUTDIR / "participant_features.csv")
    df = participants.merge(feats, on="participant_id", how="inner")
    df = df.merge(cm.build_composites(participants), on="participant_id", how="inner")

    X = df[cm.PREDICTORS].to_numpy(dtype=float)
    primary = {oc: cm.regress_with_ci(X, df[oc].to_numpy(dtype=float), cm.PREDICTORS)
               for oc in OUTCOMES}
    conv = cm.convergent_validity(df[VFF_COL], df["forward_flow"])

    measures = [m for m in MEASURES if m in df.columns]
    pcs, evr = cm.pca_reduce(df, measures, n_components=3)
    pc_names = [f"PC{i + 1}" for i in range(pcs.shape[1])]
    robustness = {oc: cm.regress_with_ci(pcs, df[oc].to_numpy(dtype=float), pc_names)
                  for oc in OUTCOMES}

    results = {"primary": primary, "convergent_validity": conv,
               "robustness_pca": robustness, "pca_explained_variance": evr}
    (OUTDIR / "creativity_architecture.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    _plot_effects(primary, FIGDIR / "creativity_effects.png")
    _plot_convergent(df, conv, FIGDIR / "convergent_forward_flow.png")

    for oc in OUTCOMES:
        r = primary[oc]
        print(f"{oc}: R2_cv={r['r2_cv']:.3f} R2_full={r['r2_full']:.3f} "
              f"p_perm={r['p_perm']:.4f} n={r['n']}")
    print(f"convergent VFF->forward_flow: r={conv['pearson_r']:.3f} "
          f"CI={conv['pearson_ci']} n={conv['n']}")
    print(f"results -> {OUTDIR / 'creativity_architecture.json'}")


if __name__ == "__main__":
    main()
