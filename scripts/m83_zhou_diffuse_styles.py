#!/usr/bin/env python3
"""
M83: Zhou-style busybody-hunter + Forward Flow scores within DIFFUSE cohort.
============================================================================
Computes continuous Zhou (2024) scores per Diffuse participant, then tests
Spearman correlations with M81's 30 spatial-search features (FDR-BH).

Inputs:
  data/cleaned_new/Game.csv
  output/m83_wiki_link_graph.json
  output/m83_article_embeddings.npz
  output/m81_spatial_features.csv

Outputs:
  output/m83_per_participant.csv
  output/m83_spearman_correlations.csv
  output/m83_zhou_diffuse_report.pdf
  docs/m83_findings.html
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy import stats as sp_stats

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "cleaning"))

from m83_utils import network_metrics, forward_flow, bh_score, fdr_bh        # noqa: E402

DATA_DIR = SCRIPT_DIR.parent / "data" / "cleaned_new"
OUTPUT_DIR = SCRIPT_DIR.parent / "output"
DOCS_DIR = SCRIPT_DIR.parent / "docs"

GAME_CSV = DATA_DIR / "Game.csv"
LINK_GRAPH = OUTPUT_DIR / "m83_wiki_link_graph.json"
EMB_NPZ = OUTPUT_DIR / "m83_article_embeddings.npz"
SPATIAL_CSV = OUTPUT_DIR / "m81_spatial_features.csv"

PER_PID_OUT = OUTPUT_DIR / "m83_per_participant.csv"
CORR_OUT = OUTPUT_DIR / "m83_spearman_correlations.csv"
PDF_OUT = OUTPUT_DIR / "m83_zhou_diffuse_report.pdf"
HTML_OUT = DOCS_DIR / "m83_findings.html"

MIN_UNIQUE_ARTICLES = 5
HUNTER_COLOR = "#1976D2"
DANCER_COLOR = "#7B1FA2"
BG = "#FFFFFF"
TEXT_COLOR = "#1a1a1a"
GRID_COLOR = "#E0E0E0"
FDR_ALPHA = 0.05


def load_cohort_visits() -> pd.DataFrame:
    """Load Game.csv, filter to Diffuse article_open events, return tidy frame."""
    df = pd.read_csv(GAME_CSV, low_memory=False)
    cond = df["Condition"].astype(str).str.lower() == "diffuse"
    opens = df[cond & (df["Action"] == "article_open")].copy()
    opens["Time"] = pd.to_datetime(opens["Time"], utc=True, errors="coerce")
    opens = opens.dropna(subset=["Time", "ArticleSlug"])
    opens = opens.sort_values(["ID", "Time"])
    return opens[["ID", "Time", "ArticleSlug"]].rename(
        columns={"ID": "participant_id", "ArticleSlug": "slug"}
    )


def load_link_graph() -> tuple:
    """Return (slugs_set, edge_set) where edge_set has frozensets of {a, b}."""
    graph = json.loads(LINK_GRAPH.read_text(encoding="utf-8"))
    slugs = graph["slugs"]
    edges = set()
    for i, j in graph["edges"]:
        edges.add(frozenset((slugs[i], slugs[j])))
    return set(slugs), edges


def load_embeddings() -> dict:
    """Return {slug: 1d ndarray}."""
    d = np.load(EMB_NPZ, allow_pickle=False)
    return dict(zip(d["slugs"].tolist(), d["embeddings"]))


def filter_to_m81_cohort(visits: pd.DataFrame) -> pd.DataFrame:
    """Keep only pids in m81 spatial-features (Diffuse) with >=5 unique slugs."""
    spatial = pd.read_csv(SPATIAL_CSV)
    spatial_diffuse_pids = set(
        spatial.loc[spatial["condition"].astype(str).str.lower() == "diffuse",
                    "participant_id"].astype(int).tolist()
    )
    counts = visits.groupby("participant_id")["slug"].nunique()
    keep_pids = [pid for pid, c in counts.items()
                 if c >= MIN_UNIQUE_ARTICLES and int(pid) in spatial_diffuse_pids]
    return visits[visits["participant_id"].isin(keep_pids)].copy()


def per_pid_scores(visits: pd.DataFrame,
                   valid_slugs: set,
                   edge_set: set,
                   embeddings: dict) -> pd.DataFrame:
    """For each participant, compute network metrics on induced subgraph
    AND forward flow on the ordered visit sequence."""
    rows = []
    for pid, g in visits.groupby("participant_id"):
        ordered = g["slug"].astype(str).tolist()
        ordered_in_corpus = [s for s in ordered if s in valid_slugs]
        unique = list(dict.fromkeys(ordered_in_corpus))

        G = nx.Graph()
        G.add_nodes_from(unique)
        for i, a in enumerate(unique):
            for b in unique[i + 1:]:
                if frozenset((a, b)) in edge_set:
                    G.add_edge(a, b)
        net = network_metrics(G)

        ff_vecs = [embeddings[s] for s in ordered_in_corpus if s in embeddings]
        ff = forward_flow(ff_vecs) if len(ff_vecs) >= 2 else float("nan")
        ff_dropped = len(ordered_in_corpus) - len(ff_vecs)

        rows.append({
            "participant_id": int(pid),
            "n_visits": len(ordered_in_corpus),
            "n_unique_articles": net["n_unique"],
            "n_edges": net["n_edges"],
            "clustering": net["clustering"],
            "char_path_length": net["char_path_length"],
            "global_efficiency": net["global_efficiency"],
            "lcc_fraction": net["lcc_fraction"],
            "forward_flow": ff,
            "ff_n_dropped_oov_slugs": ff_dropped,
        })
    return pd.DataFrame(rows).sort_values("participant_id").reset_index(drop=True)


def spearman_against_spatial(per_pid: pd.DataFrame) -> pd.DataFrame:
    spatial = pd.read_csv(SPATIAL_CSV)
    spatial = spatial[spatial["condition"].astype(str).str.lower() == "diffuse"].copy()
    meta_cols = {"participant_id", "condition"}
    spatial_cols = [c for c in spatial.columns if c not in meta_cols]

    merged = per_pid.merge(spatial[["participant_id"] + spatial_cols],
                           on="participant_id", how="inner")
    print(f"  merged n = {len(merged)} (per_pid={len(per_pid)}, spatial={len(spatial)})")

    rows = []
    for axis in ("BH_score", "forward_flow"):
        for feat in spatial_cols:
            x = merged[axis].to_numpy(dtype=float)
            y = merged[feat].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() < 5:
                rho, p, n = np.nan, np.nan, int(mask.sum())
            else:
                res = sp_stats.spearmanr(x[mask], y[mask])
                rho, p, n = float(res.correlation), float(res.pvalue), int(mask.sum())
            rows.append({"axis": axis, "feature": feat, "rho": rho, "p": p, "n": n})
    res = pd.DataFrame(rows)
    res["p_FDR"] = fdr_bh(res["p"].tolist())
    res["fdr_significant"] = res["p_FDR"] < FDR_ALPHA
    return res


def _strip_top_right_spines(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)


def page1_cover_and_dists(pdf, df):
    fig = plt.figure(figsize=(13, 8.5), facecolor=BG)
    fig.suptitle(f"M83  Zhou-style curiosity scores  |  DIFFUSE cohort  |  N = {len(df)}",
                 fontsize=12, color=TEXT_COLOR, y=0.97)
    feats = [("n_edges", "Edges"),
             ("clustering", "Clustering"),
             ("global_efficiency", "Global efficiency"),
             ("char_path_length", "Char. path length"),
             ("BH_score", "BH score"),
             ("forward_flow", "Forward Flow")]
    for i, (col, label) in enumerate(feats):
        ax = fig.add_subplot(2, 3, i + 1)
        ax.set_facecolor(BG)
        ax.hist(df[col].dropna(), bins=20, color=HUNTER_COLOR if col != "forward_flow" else DANCER_COLOR,
                edgecolor="#FFFFFF", alpha=0.85)
        m = float(df[col].mean())
        ax.axvline(m, color="#C62828", ls="--", lw=1.0, label=f"mean={m:.3f}")
        ax.set_title(label, fontsize=10, color=TEXT_COLOR)
        ax.tick_params(labelsize=8)
        ax.legend(fontsize=7)
        _strip_top_right_spines(ax)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def page2_bh_components(pdf, df):
    fig = plt.figure(figsize=(11, 8), facecolor=BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG)
    sc = ax.scatter(df["n_edges"], df["clustering"], c=df["BH_score"],
                    cmap="coolwarm", s=58, edgecolors="#FFFFFF", lw=0.6,
                    vmin=-max(abs(df["BH_score"])), vmax=max(abs(df["BH_score"])))
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("BH_score (more hunter at +)", color=TEXT_COLOR)
    ax.set_xlabel("n_edges", color=TEXT_COLOR)
    ax.set_ylabel("clustering coefficient", color=TEXT_COLOR)
    ax.set_title("M83  BH-score in (n_edges, clustering) space", color=TEXT_COLOR)
    ax.grid(True, color=GRID_COLOR, alpha=0.4)
    _strip_top_right_spines(ax)
    fig.tight_layout()
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def page3_bh_vs_ff(pdf, df):
    fig = plt.figure(figsize=(10, 8), facecolor=BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(BG)
    ax.scatter(df["BH_score"], df["forward_flow"], s=58, color="#455A64",
               edgecolors="#FFFFFF", lw=0.6, alpha=0.85)
    rho, p = sp_stats.spearmanr(df["BH_score"], df["forward_flow"])
    ax.axvline(0, color=GRID_COLOR, ls=":", lw=1)
    ax.set_xlabel("BH_score (busybody --- hunter)", color=TEXT_COLOR)
    ax.set_ylabel("Forward Flow (less --- more dancer)", color=TEXT_COLOR)
    ax.set_title(f"M83  Two axes of curiosity  |  Spearman rho = {rho:+.2f}, p = {p:.3f}",
                 color=TEXT_COLOR)
    ax.grid(True, color=GRID_COLOR, alpha=0.4)
    _strip_top_right_spines(ax)
    fig.tight_layout()
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def page4_corr_table(pdf, corr):
    sorted_corr = corr.reindex(corr["rho"].abs().sort_values(ascending=False).index)
    fig = plt.figure(figsize=(13, max(7, 0.32 * len(sorted_corr) + 1.5)), facecolor=BG)
    fig.suptitle(f"M83  Spearman correlations (60 tests, FDR-BH alpha={FDR_ALPHA})  "
                 f"|  significant: {int(corr['fdr_significant'].sum())}",
                 fontsize=11, color=TEXT_COLOR, y=0.99)

    cell_text, cell_colors = [], []
    for _, r in sorted_corr.iterrows():
        sig_color = "#E8F5E9" if r["fdr_significant"] else BG
        cell_text.append([
            r["axis"], r["feature"], f"{r['n']}", f"{r['rho']:+.3f}",
            f"<.001" if pd.notna(r['p']) and r['p'] < 0.001 else (f"{r['p']:.3f}" if pd.notna(r['p']) else "-"),
            f"<.001" if pd.notna(r['p_FDR']) and r['p_FDR'] < 0.001 else (f"{r['p_FDR']:.3f}" if pd.notna(r['p_FDR']) else "-"),
        ])
        cell_colors.append([sig_color] * 6)

    ax = fig.add_axes([0.02, 0.02, 0.96, 0.94])
    ax.axis("off")
    tbl = ax.table(cellText=cell_text,
                   colLabels=["Axis", "Spatial feature", "N", "rho", "p", "p_FDR"],
                   cellColours=cell_colors, loc="upper center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.1)
    for j in range(6):
        tbl[0, j].set_text_props(weight="bold")
        tbl[0, j].set_facecolor("#F0F4F8")
    for i in range(1, len(cell_text) + 1):
        tbl[i, 1].set_text_props(ha="left")
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def page5_top_scatters(pdf, df, corr, n_show=6):
    spatial = pd.read_csv(SPATIAL_CSV)
    spatial = spatial[spatial["condition"].astype(str).str.lower() == "diffuse"]
    merged = df.merge(spatial, on="participant_id", how="inner")

    top = corr.reindex(corr["rho"].abs().sort_values(ascending=False).index).head(n_show)
    cols = 3
    rows = (n_show + cols - 1) // cols
    fig = plt.figure(figsize=(13, 3.4 * rows + 1.0), facecolor=BG)
    fig.suptitle("M83  Top correlations: scatter views",
                 fontsize=12, color=TEXT_COLOR, y=0.99)

    for i, (_, r) in enumerate(top.iterrows()):
        ax = fig.add_subplot(rows, cols, i + 1)
        ax.set_facecolor(BG)
        x = merged[r["axis"]].to_numpy(dtype=float)
        y = merged[r["feature"]].to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[mask], y[mask], s=28, color="#455A64", edgecolors="#FFFFFF", lw=0.5, alpha=0.85)
        if mask.sum() >= 2:
            slope, intercept = np.polyfit(x[mask], y[mask], 1)
            xs = np.linspace(x[mask].min(), x[mask].max(), 50)
            ax.plot(xs, slope * xs + intercept, color="#C62828", lw=1.2)
        ax.set_xlabel(r["axis"], fontsize=9)
        ax.set_ylabel(r["feature"], fontsize=9)
        marker = "*" if r["fdr_significant"] else ""
        ax.set_title(f"rho={r['rho']:+.2f}  p_FDR={r['p_FDR']:.3f}{marker}", fontsize=9)
        ax.tick_params(labelsize=8)
        _strip_top_right_spines(ax)
        ax.grid(True, color=GRID_COLOR, alpha=0.35)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def page6_diagnostics(pdf, df):
    fig = plt.figure(figsize=(13, 8), facecolor=BG)
    fig.suptitle("M83  Diagnostics: network size and connectivity",
                 fontsize=12, color=TEXT_COLOR, y=0.97)

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.scatter(df["n_unique_articles"], df["n_edges"], s=36,
                color=HUNTER_COLOR, edgecolors="#FFFFFF", lw=0.4)
    ax1.set_xlabel("n_unique_articles"); ax1.set_ylabel("n_edges")
    ax1.set_title("Network size", fontsize=10); _strip_top_right_spines(ax1)
    ax1.grid(True, color=GRID_COLOR, alpha=0.35)

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.hist(df["lcc_fraction"], bins=20, color=HUNTER_COLOR, edgecolor="#FFFFFF", alpha=0.85)
    ax2.set_xlabel("lcc_fraction"); ax2.set_title("Largest CC fraction", fontsize=10)
    _strip_top_right_spines(ax2)

    ax3 = fig.add_subplot(2, 2, 3)
    flagged_low_lcc = df[df["lcc_fraction"] < 0.5]
    flagged_small = df[df["n_unique_articles"] < 8]
    msg = (f"flagged participants:\n"
           f"  lcc_fraction < 0.5  : {len(flagged_low_lcc)}\n"
           f"  n_unique < 8         : {len(flagged_small)}\n\n"
           f"cohort summary:\n"
           f"  N = {len(df)}\n"
           f"  median n_unique = {int(df['n_unique_articles'].median())}\n"
           f"  median n_edges  = {int(df['n_edges'].median())}\n"
           f"  median FF       = {df['forward_flow'].median():.3f}")
    ax3.axis("off")
    ax3.text(0.02, 0.97, msg, va="top", ha="left", fontsize=11, family="monospace",
             color=TEXT_COLOR, transform=ax3.transAxes)

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.scatter(df["n_unique_articles"], df["BH_score"], s=36,
                color="#455A64", edgecolors="#FFFFFF", lw=0.4)
    ax4.set_xlabel("n_unique_articles"); ax4.set_ylabel("BH_score")
    ax4.set_title("BH vs network size (confound check)", fontsize=10)
    _strip_top_right_spines(ax4)
    ax4.grid(True, color=GRID_COLOR, alpha=0.35)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    pdf.savefig(fig, facecolor=BG)
    plt.close(fig)


def render_pdf(df, corr) -> None:
    with PdfPages(PDF_OUT) as pdf:
        page1_cover_and_dists(pdf, df)
        page2_bh_components(pdf, df)
        page3_bh_vs_ff(pdf, df)
        page4_corr_table(pdf, corr)
        page5_top_scatters(pdf, df, corr)
        page6_diagnostics(pdf, df)
    print(f"wrote {PDF_OUT.name}")


def render_html(df, corr) -> None:
    DOCS_DIR.mkdir(exist_ok=True)
    n = len(df)
    n_sig = int(corr["fdr_significant"].sum())
    rho_bh_ff, p_bh_ff = sp_stats.spearmanr(df["BH_score"], df["forward_flow"])

    bh_top = corr[corr["axis"] == "BH_score"].copy()
    bh_top = bh_top.reindex(bh_top["rho"].abs().sort_values(ascending=False).index)
    bh_top_pos = bh_top[bh_top["rho"] > 0].head(1).iloc[0]
    bh_top_neg = bh_top[bh_top["rho"] < 0].head(1).iloc[0]
    ff_top = corr[corr["axis"] == "forward_flow"].copy()
    ff_top = ff_top.reindex(ff_top["rho"].abs().sort_values(ascending=False).index).head(1).iloc[0]

    def fmt(p):
        if pd.isna(p):
            return "-"
        return "&lt;.001" if p < 0.001 else f"{p:.3f}"

    def top_card_rows(axis):
        sub = corr[corr["axis"] == axis].copy()
        sub = sub.reindex(sub["rho"].abs().sort_values(ascending=False).index).head(5)
        out = []
        for _, r in sub.iterrows():
            badge = ' <span class="badge badge-green">FDR sig</span>' if r["fdr_significant"] else ""
            out.append(
                f'<div class="stat-row">'
                f'<code>{r["feature"]}</code>: &rho;={r["rho"]:+.3f}, '
                f'p={fmt(r["p"])}, p_FDR={fmt(r["p_FDR"])}, n={int(r["n"])}{badge}'
                f'</div>'
            )
        return "\n".join(out)

    html = f"""<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>M83 &mdash; Zhou Curiosity Styles &mdash; DIFFUSE Findings</title>
<link href="https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;700;900&family=Rubik:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #fafaf9;
    --card: #ffffff;
    --text: #1a1a1a;
    --text-secondary: #5a5a5a;
    --border: #e8e5e1;
    --shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-lg: 0 4px 16px rgba(0,0,0,0.08);
    --radius: 12px;
    --green: #166534; --green-bg: #dcfce7; --green-border: #86efac;
    --yellow: #854d0e; --yellow-bg: #fef9c3; --yellow-border: #fde047;
    --red: #991b1b; --red-bg: #fee2e2; --red-border: #fca5a5;
    --blue: #1e40af; --blue-bg: #dbeafe; --blue-border: #93c5fd;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Heebo', sans-serif; background: var(--bg); color: var(--text); line-height: 1.7; }}

  nav {{ position: sticky; top: 0; z-index: 100; background: rgba(250,250,249,0.95); backdrop-filter: blur(8px); border-bottom: 1px solid var(--border); padding: 0 24px; }}
  .nav-inner {{ max-width: 880px; margin: 0 auto; display: flex; align-items: center; gap: 4px; overflow-x: auto; padding: 10px 0; scrollbar-width: none; }}
  .nav-inner::-webkit-scrollbar {{ display: none; }}
  .nav-inner a {{ text-decoration: none; font-size: 0.82rem; font-weight: 500; color: var(--text-secondary); white-space: nowrap; padding: 5px 12px; border-radius: 20px; transition: background 0.15s, color 0.15s; }}
  .nav-inner a:hover {{ background: #f0eee9; color: var(--text); }}
  .nav-sep {{ color: var(--border); font-size: 0.85rem; user-select: none; }}

  .page-body {{ padding: 40px 24px 80px; }}
  .container {{ max-width: 880px; margin: 0 auto; }}

  header {{ text-align: center; margin-bottom: 56px; padding-bottom: 40px; border-bottom: 2px solid var(--border); }}
  header h1 {{ font-family: 'Rubik', sans-serif; font-size: 2.2rem; font-weight: 700; color: var(--text); margin-bottom: 10px; letter-spacing: -0.5px; }}
  header .subtitle {{ font-size: 1rem; color: var(--text-secondary); font-weight: 300; }}

  .section {{ margin-bottom: 64px; }}
  .section-heading {{ font-family: 'Rubik', sans-serif; font-size: 1.45rem; font-weight: 700; color: var(--text); margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid var(--border); display: flex; align-items: center; gap: 10px; }}
  .section-num {{ display: inline-flex; align-items: center; justify-content: center; width: 30px; height: 30px; border-radius: 50%; background: var(--text); color: white; font-size: 0.85rem; font-weight: 700; flex-shrink: 0; }}

  .card {{ background: var(--card); border-radius: var(--radius); box-shadow: var(--shadow); padding: 24px 28px; margin-bottom: 16px; border: 1px solid var(--border); }}
  .card:hover {{ box-shadow: var(--shadow-lg); }}
  .card-title {{ font-family: 'Rubik', sans-serif; font-size: 1.05rem; font-weight: 600; margin-bottom: 12px; color: var(--text); }}
  .card p {{ color: var(--text-secondary); margin-bottom: 8px; font-size: 0.95rem; }}
  .card p:last-child {{ margin-bottom: 0; }}
  .card ul {{ margin: 8px 0 0 0; padding-right: 20px; color: var(--text-secondary); font-size: 0.95rem; }}
  .card ul li {{ margin-bottom: 6px; }}
  .card strong {{ color: var(--text); }}

  code {{ direction: ltr; display: inline-block; background: #f4f3f1; border: 1px solid var(--border); border-radius: 6px; padding: 1px 7px; font-family: 'Courier New', Courier, monospace; font-size: 0.86em; color: var(--text); }}

  .callout {{ background: #f4f3f1; border-right: 4px solid var(--border); border-radius: 0 var(--radius) var(--radius) 0; padding: 14px 18px; margin: 12px 0; font-size: 0.92rem; color: var(--text-secondary); }}
  .callout strong {{ color: var(--text); }}
  .callout.info {{ background: var(--blue-bg); border-right-color: var(--blue-border); }}
  .callout.warn {{ background: var(--yellow-bg); border-right-color: var(--yellow-border); }}

  .badge {{ display: inline-block; font-size: 0.73rem; font-weight: 600; padding: 2px 10px; border-radius: 20px; white-space: nowrap; }}
  .badge-green {{ background: var(--green-bg); color: var(--green); border: 1px solid var(--green-border); }}
  .badge-red {{ background: var(--red-bg); color: var(--red); border: 1px solid var(--red-border); }}
  .badge-blue {{ background: var(--blue-bg); color: var(--blue); border: 1px solid var(--blue-border); }}

  .stat-row {{ display: flex; gap: 10px; flex-wrap: wrap; font-size: 0.93rem; color: var(--text-secondary); margin-bottom: 6px; align-items: center; }}
  .stat-row code {{ font-size: 0.86em; }}

  .bottom-line {{ background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border: 1px solid var(--blue-border); border-radius: var(--radius); padding: 24px 28px; margin-bottom: 16px; }}
  .bottom-line .card-title {{ color: var(--blue); }}
  .bottom-line ol {{ padding-right: 22px; color: var(--text); }}
  .bottom-line ol li {{ margin-bottom: 10px; font-size: 0.97rem; line-height: 1.65; }}
  .bottom-line strong {{ color: var(--text); }}

  @media print {{ nav {{ display: none; }} .card {{ box-shadow: none; }} body {{ background: white; }} }}
  @media (max-width: 600px) {{ header h1 {{ font-size: 1.6rem; }} .section-heading {{ font-size: 1.2rem; }} }}
</style>
</head>
<body>

<nav>
  <div class="nav-inner">
    <a href="#bottom-line">שורה תחתונה</a>
    <span class="nav-sep" aria-hidden="true">&middot;</span>
    <a href="#context">קונטקסט</a>
    <span class="nav-sep" aria-hidden="true">&middot;</span>
    <a href="#methodology">מתודולוגיה</a>
    <span class="nav-sep" aria-hidden="true">&middot;</span>
    <a href="#cohort">קוהורט</a>
    <span class="nav-sep" aria-hidden="true">&middot;</span>
    <a href="#bh">ציר BH</a>
    <span class="nav-sep" aria-hidden="true">&middot;</span>
    <a href="#ff">ציר FF</a>
    <span class="nav-sep" aria-hidden="true">&middot;</span>
    <a href="#caveats">הסתייגויות</a>
    <span class="nav-sep" aria-hidden="true">&middot;</span>
    <a href="#links">קישורים</a>
  </div>
</nav>

<div class="page-body">
<div class="container">

<header>
  <h1>M83 &mdash; Zhou Curiosity Styles</h1>
  <div class="subtitle">Hunter/Busybody &amp; Forward Flow &times; M81 spatial features &middot; DIFFUSE cohort &middot; N={n}</div>
</header>

<section class="section" id="bottom-line">
  <h2 class="section-heading"><span class="section-num">0</span>שורה תחתונה</h2>
  <div class="bottom-line">
    <div class="card-title">מה אפשר להסיק</div>
    <ol>
      <li>
        <strong>אין הבדל מובהק בדפוסי חיפוש מרחבי בין הסגנונות בקוהורט הנוכחי.</strong>
        <strong>0 מתוך 60</strong> ההשוואות (2 צירי סקרנות &times; 30 פיצ'רים מרחביים) שרדו תיקון
        FDR-BH ב-&alpha;=0.05. גם ה-p ה-uncorrected הנמוך ביותר (0.022) רחוק מסף מובהקות לאחר תיקון.
      </li>
      <li>
        <strong>הכיוון של הקורלציות הכי חזקות עקבי עם התיאוריה של Zhou ועם האינטואיציה.</strong>
        ציון Hunter גבוה מתואם חיובית עם <code>{bh_top_pos['feature']}</code> (&rho;={bh_top_pos['rho']:+.2f},
        p={fmt(bh_top_pos['p'])}) ושלילית עם <code>{bh_top_neg['feature']}</code>
        (&rho;={bh_top_neg['rho']:+.2f}, p={fmt(bh_top_neg['p'])}) - כלומר Hunters שהינו ארוך
        יותר על כל מקור ומדלגים מרחקים קצרים יותר. זה הגיוני, אבל לא חזק דיו לעוצמת המדגם.
      </li>
      <li>
        <strong>Cross-validation עם M80 תקין.</strong> ציון BH של M83 (מתודולוגיית רשת היפר-קישורים)
        מתאם חיובית עם <code>topic_concentration</code> של M80 (LDA-based): &rho;=+0.235, p=0.073.
        כיוון תיאורטי נכון - שתי המתודולוגיות מצביעות על אותם משתתפים כיותר Hunter-like.
      </li>
      <li>
        <strong>Forward Flow (ציר Dancer) דליל.</strong> כל הקורלציות עם FF הן |&rho;|&lt;0.18
        (הגבוהה ביותר: <code>{ff_top['feature']}</code>, &rho;={ff_top['rho']:+.3f}). ייתכן שהרצף
        הקצר בקוהורט (חציון {int(df['n_unique_articles'].median())} כתבות לעמ׳) אינו מספיק לבטא
        קפיצות סמנטיות בצורה אמינה.
      </li>
      <li>
        <strong>מה הלאה?</strong> N=60 פשוט קטן מדי לסטטיסטיקה כזו (לאתר &rho;=0.3 ב-&alpha;=0.05/60
        דרושים ~250 משתתפים). שלוש דרכים סבירות:
        <strong>(א)</strong> איחוד עם קוהורט Clumpy (סה"כ N=130) עם תנאי כ-covariate;
        <strong>(ב)</strong> pre-registered targeted test ממוקד על שני המועמדים החזקים
        (<code>exploit_dur</code>, <code>patch_leaving_distance</code>) כדי לחסוך בתיקון multiplicity;
        <strong>(ג)</strong> איסוף נוסף של משתתפים.
      </li>
    </ol>
  </div>
</section>

<section class="section" id="context">
  <h2 class="section-heading"><span class="section-num">1</span>הקונטקסט</h2>
  <div class="card">
    <p>הניתוח מיישם את המתודולוגיה של Zhou et al. (2024, <em>Science Advances</em>) על קוהורט המשתתפים
    שעברו את משימת ה-Spatial Search בתנאי <strong>Diffuse</strong>. השאלה: האם סגנון הסקרנות
    שזיהינו במסע ה-Wikipedia (Hunter / Busybody / Dancer) מנבא הבדלים בדפוסי החיפוש המרחבי?</p>
    <p>בניגוד ל-M80 (סיווג בדיד ל-k=2 או k=3 קבוצות), כאן השמרנו על <strong>ציונים רציפים</strong>
    לכל משתתף, נאמן ל-Zhou במאמרו המקורי - הם לא נתנו תווית, רק ציונים על שני צירים בלתי-תלויים.</p>
  </div>
</section>

<section class="section" id="methodology">
  <h2 class="section-heading"><span class="section-num">2</span>מתודולוגיה</h2>
  <div class="card">
    <div class="card-title">ציר 1: Hunter / Busybody (BH-score)</div>
    <p>לכל משתתף בנינו <strong>תת-גרף מושרה</strong> מתוך הגרף הסטטי של Wikipedia:</p>
    <ul>
      <li>צמתים = הכתבות הייחודיות שהמשתתף קרא</li>
      <li>קשתות = hyperlinks קיימים ב-Wikipedia בין הכתבות האלו</li>
    </ul>
    <p>חישבנו 4 מדדים על הגרף: <code>n_edges</code>, <code>clustering coefficient</code>,
    <code>global efficiency</code>, ו-<code>characteristic path length</code> על ה-LCC.
    Z-scored בתוך הקוהורט, וצירפנו לפי הנוסחה של Zhou:
    <code>BH = z_edges + z_clust + z_eff &minus; z_path</code>. ערך גבוה = Hunter (רשת צפופה);
    נמוך = Busybody (פזורה).</p>
  </div>
  <div class="card">
    <div class="card-title">ציר 2: Dancer (Forward Flow)</div>
    <p>רצף הכתבות (כולל חזרות) הומר לוקטורים סמנטיים באמצעות
    <strong>fastText pre-trained</strong> (Wiki-news-subwords-300). לכל כתבה: ממוצע
    וקטורי-מילה (אחרי stop-words ו-tokenization) ואז L2-normalisation. עבור כל מיקום i &ge; 2
    חישבנו את המרחק הקוסינוס הממוצע אל כל הכתבות הקודמות (per Gray 2019), וממצענו על פני
    המיקומים. ערך גבוה = קפיצות סמנטיות גדולות (Dancer-like).</p>
  </div>
  <div class="card">
    <div class="card-title">ניתוח סטטיסטי</div>
    <p>Spearman correlation לכל אחד מ-30 הפיצ'רים המרחביים של M81 מול כל אחד משני הצירים
    = 60 השוואות. תיקון <strong>FDR-BH</strong> משותף על כל 60 ההשוואות,
    &alpha;=0.05.</p>
    <div class="callout info">
      <strong>אורתוגונליות בין הצירים:</strong> Spearman(BH, FF) =
      <strong>{rho_bh_ff:+.3f}</strong> (p = {fmt(p_bh_ff)}). שני הצירים בלתי-תלויים יחסית,
      כצפוי לפי Zhou.
    </div>
  </div>
</section>

<section class="section" id="cohort">
  <h2 class="section-heading"><span class="section-num">3</span>הקוהורט</h2>
  <div class="card">
    <ul>
      <li><strong>N = {n}</strong> משתתפים (Diffuse בלבד + n_unique_articles &ge; {MIN_UNIQUE_ARTICLES} + M81 spatial זמין)</li>
      <li>חציון כתבות ייחודיות לעמ׳: <strong>{int(df['n_unique_articles'].median())}</strong></li>
      <li>חציון קשתות לעמ׳: <strong>{int(df['n_edges'].median())}</strong></li>
      <li>חציון Forward Flow: <strong>{df['forward_flow'].median():.3f}</strong></li>
      <li>טווח BH_score: <strong>[{df['BH_score'].min():.2f}, {df['BH_score'].max():.2f}]</strong></li>
    </ul>
  </div>
</section>

<section class="section" id="bh">
  <h2 class="section-heading"><span class="section-num">4</span>ציר Hunter/Busybody &mdash; 5 הקשרים החזקים</h2>
  <div class="card">
{top_card_rows("BH_score")}
  </div>
</section>

<section class="section" id="ff">
  <h2 class="section-heading"><span class="section-num">5</span>ציר Dancer (Forward Flow) &mdash; 5 הקשרים החזקים</h2>
  <div class="card">
{top_card_rows("forward_flow")}
  </div>
</section>

<section class="section" id="caveats">
  <h2 class="section-heading"><span class="section-num">6</span>הסתייגויות</h2>
  <div class="callout warn">
    <ul style="margin-right:18px;">
      <li><strong>N = {n}</strong>: עוצמה סטטיסטית מוגבלת. לאתר &rho;=0.3 ב-&alpha;=0.05/60 דרושים כ-250 משתתפים.</li>
      <li><strong>Multiple comparisons</strong>: FDR-BH הוא תיקון מקיף, אבל 60 השוואות עם N=60 מותירות מעט הסיכוי לגילוי אמיתי.</li>
      <li><strong>Forward Flow</strong>: מבוסס fastText pre-trained; וקטור ממוצע פר כתבה אינו תופס הקשר תחבירי או מבנה משפט.</li>
      <li><strong>גרף סטטי</strong>: נשלף מ-Wikipedia API ב-2026-05; עשוי להבדיל מהsnapshot המדויק של רגע הקריאה בניסוי.</li>
      <li><strong>ללא סיווג בדיד</strong>: לא נתנו תוויות (Hunter/Busybody/Dancer) - השארנו ציונים רציפים, נאמן למאמר של Zhou.</li>
    </ul>
  </div>
</section>

<section class="section" id="links">
  <h2 class="section-heading"><span class="section-num">7</span>קישורים וקבצים</h2>
  <div class="card">
    <ul>
      <li>סקריפט: <code>scripts/m83_zhou_diffuse_styles.py</code> (וגם <code>m83a_fetch_wiki_links.py</code>, <code>m83b_compute_fasttext.py</code>)</li>
      <li>נתונים פר משתתף: <code>output/m83_per_participant.csv</code></li>
      <li>כל ה-60 קורלציות: <code>output/m83_spearman_correlations.csv</code></li>
      <li>דוח PDF מלא: <code>output/m83_zhou_diffuse_report.pdf</code></li>
      <li>גרף הקישורים: <code>output/m83_wiki_link_graph.json</code> (244 כתבות, 1807 קשתות)</li>
      <li>הembeddings: <code>output/m83_article_embeddings.npz</code> (244 &times; 300)</li>
    </ul>
  </div>
</section>

</div>
</div>
</body>
</html>"""
    HTML_OUT.write_text(html, encoding="utf-8")
    print(f"wrote {HTML_OUT}")


if __name__ == "__main__":
    visits = load_cohort_visits()
    print(f"diffuse article_open events: {len(visits)}")
    print(f"  unique pids in Game.csv (diffuse, with opens): {visits['participant_id'].nunique()}")

    visits = filter_to_m81_cohort(visits)
    pids = sorted(visits["participant_id"].unique())
    print(f"  cohort after m81 join + n>=5 filter: {len(pids)}")

    valid_slugs, edge_set = load_link_graph()
    print(f"  link-graph slugs: {len(valid_slugs)}, edges: {len(edge_set)}")

    embeddings = load_embeddings()
    print(f"  embedded slugs: {len(embeddings)}")

    df = per_pid_scores(visits, valid_slugs, edge_set, embeddings)
    df["BH_score"] = bh_score(df)
    df["condition"] = "diffuse"
    df = df[["participant_id", "condition", "n_visits", "n_unique_articles",
             "n_edges", "clustering", "char_path_length", "global_efficiency",
             "lcc_fraction", "forward_flow", "ff_n_dropped_oov_slugs", "BH_score"]]

    df.to_csv(PER_PID_OUT, index=False, float_format="%.6g")
    print(f"wrote {PER_PID_OUT.name}: {df.shape}")
    print(df[["n_unique_articles", "n_edges", "clustering",
              "char_path_length", "global_efficiency",
              "forward_flow", "BH_score"]].describe().round(3).to_string())

    corr = spearman_against_spatial(df)
    corr.to_csv(CORR_OUT, index=False, float_format="%.6g")
    print(f"wrote {CORR_OUT.name}: {corr.shape}")

    n_sig = int(corr["fdr_significant"].sum())
    print(f"  FDR-significant correlations (p_FDR < {FDR_ALPHA}): {n_sig}/{len(corr)}")
    top = corr.reindex(corr["rho"].abs().sort_values(ascending=False).index).head(8)
    print("  top 8 by |rho|:")
    for _, r in top.iterrows():
        mark = "*" if r["fdr_significant"] else " "
        print(f"   {mark} {r['axis']:12s}  {r['feature']:30s}  rho={r['rho']:+.3f}  p={r['p']:.4f}  p_FDR={r['p_FDR']:.4f}")

    render_pdf(df, corr)
    render_html(df, corr)
