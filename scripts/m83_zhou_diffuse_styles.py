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

    def fmt(p):
        if pd.isna(p):
            return "-"
        return "<.001" if p < 0.001 else f"{p:.3f}"

    def top_rows(axis):
        sub = corr[corr["axis"] == axis].copy()
        sub = sub.reindex(sub["rho"].abs().sort_values(ascending=False).index).head(5)
        out = []
        for _, r in sub.iterrows():
            sig = ' style="background:#E8F5E9;font-weight:600"' if r["fdr_significant"] else ""
            out.append(
                f'<tr{sig}><td><code>{r["feature"]}</code></td>'
                f'<td>{r["rho"]:+.3f}</td>'
                f'<td>{fmt(r["p"])}</td>'
                f'<td>{fmt(r["p_FDR"])}</td>'
                f'<td>{int(r["n"])}</td></tr>'
            )
        return "\n".join(out)

    html = f"""<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
<meta charset="UTF-8">
<title>M83 - ציוני סקרנות של Zhou על קוהורט DIFFUSE</title>
<style>
  body {{ font-family: -apple-system, "Segoe UI", Arial, sans-serif;
         background:#FFFFFF; color:#1a1a1a; line-height:1.65;
         max-width:1000px; margin:32px auto; padding:0 24px; }}
  h1 {{ color:#1a1a1a; border-bottom:2px solid #1976D2; padding-bottom:8px; }}
  h2 {{ color:#1976D2; margin-top:36px;
       border-bottom:1px solid #E0E0E0; padding-bottom:4px; }}
  code, pre {{ direction:ltr; text-align:left; background:#F5F5F5;
              border:1px solid #E0E0E0; border-radius:4px;
              font-family: Consolas, monospace; }}
  code {{ padding:2px 6px; }}
  pre {{ padding:12px; overflow-x:auto; }}
  table {{ border-collapse:collapse; margin:12px 0; width:100%; direction:ltr; }}
  th, td {{ border:1px solid #CCC; padding:8px 12px; text-align:left; }}
  th {{ background:#F0F4F8; }}
  tr:nth-child(even) td {{ background:#FAFAFA; }}
  .info {{ background:#E3F2FD; border-right:4px solid #1976D2;
          padding:12px 16px; margin:12px 0; border-radius:4px; }}
  .meta {{ color:#666; font-size:0.9em; margin-bottom:24px; }}
  .caveat {{ background:#FFF3E0; border-right:4px solid #FB8C00;
            padding:12px 16px; margin:12px 0; border-radius:4px; }}
</style>
</head>
<body>
<h1>M83 - ציוני Hunter/Busybody + Forward Flow על קוהורט DIFFUSE</h1>
<div class="meta">
  Scripts: <code>scripts/m83a_fetch_wiki_links.py</code>,
  <code>scripts/m83b_compute_fasttext.py</code>,
  <code>scripts/m83_zhou_diffuse_styles.py</code> &middot;
  N = {n} משתתפים &middot; FDR-BH alpha = {FDR_ALPHA}
</div>

<h2>1. תקציר</h2>
<p>
  הניתוח מיישם את המתודולוגיה של Zhou et al. (2024) על קוהורט המשתתפים שעברו את משימת
  ה-Spatial Search בתנאי <strong>Diffuse</strong>. לכל משתתף חושב ציון רציף של Hunter/Busybody
  (BH-score, מצרף 4 מדדי רשת על הגרף הסטטי של Wikipedia) וציון Forward Flow (מרחק קוסינוס סמנטי
  ממוצע באמצעות fastText). שני הצירים נבחנו במתאם Spearman מול 30 הפיצ'רים המרחביים מ-M81,
  עם תיקון FDR-BH על כל 60 ההשוואות.
</p>

<h2>2. מתודולוגיה</h2>
<p>
  <strong>ציר 1 (Hunter/Busybody):</strong> לכל משתתף בנינו תת-גרף מושרה מהגרף הסטטי של
  Wikipedia: צמתים = הכתבות הייחודיות שהמשתתף קרא; קשתות = hyperlinks קיימים ב-Wikipedia.
  מתוך הגרף חישבנו 4 מדדים (edges, clustering coefficient, global efficiency, characteristic
  path length על ה-LCC), z-scored בתוך הקוהורט, וצירפנו ל-BH_score לפי הנוסחה של Zhou:
  <code>z_edges + z_clust + z_eff - z_path</code>. ערך גבוה &rarr; Hunter (רשת צפופה); נמוך &rarr;
  Busybody (פזורה).
</p>
<p>
  <strong>ציר 2 (Dancer/Forward Flow):</strong> רצף הכתבות (כולל חזרות) הומר לוקטורים מסוג
  fastText (mean over in-vocab tokens, L2-normalised). לכל מיקום i &ge; 2 חישבנו את המרחק הקוסינוס
  הממוצע אל כל הכתבות הקודמות, וממצענו על פני המיקומים. ערך גבוה &rarr; קפיצות סמנטיות גדולות
  (Dancer-like).
</p>
<div class="info">
  שני הצירים בלתי-תלויים יחסית: Spearman(BH, FF) = {rho_bh_ff:+.3f} (p = {fmt(p_bh_ff)}).
</div>

<h2>3. הקוהורט</h2>
<table>
<tr><th>מאפיין</th><th>ערך</th></tr>
<tr><td>N (Diffuse + n_unique &ge; {MIN_UNIQUE_ARTICLES} + M81 spatial available)</td><td>{n}</td></tr>
<tr><td>חציון כתבות ייחודיות לעמ׳</td><td>{int(df['n_unique_articles'].median())}</td></tr>
<tr><td>חציון קשתות לעמ׳</td><td>{int(df['n_edges'].median())}</td></tr>
<tr><td>חציון Forward Flow</td><td>{df['forward_flow'].median():.3f}</td></tr>
<tr><td>טווח BH_score</td><td>[{df['BH_score'].min():.2f}, {df['BH_score'].max():.2f}]</td></tr>
</table>

<h2>4. ציר Hunter/Busybody - 5 הקשרים החזקים</h2>
<table>
<tr><th>פיצ'ר מרחבי</th><th>rho</th><th>p</th><th>p_FDR</th><th>N</th></tr>
{top_rows("BH_score")}
</table>

<h2>5. ציר Dancer (Forward Flow) - 5 הקשרים החזקים</h2>
<table>
<tr><th>פיצ'ר מרחבי</th><th>rho</th><th>p</th><th>p_FDR</th><th>N</th></tr>
{top_rows("forward_flow")}
</table>

<h2>6. סיכום ופרשנות</h2>
<p>
  מתוך 60 ההשוואות, <strong>{n_sig}</strong> שרדו תיקון FDR-BH ב-&alpha;={FDR_ALPHA}. שורות מודגשות
  בירוק בטבלאות מעלה הן ה-FDR-significant. הפרשנות הסיבתית מצומצמת בגלל המדגם הקטן והמתאם בלבד;
  הניתוח מציע מועמדים לבחינה ממוקדת.
</p>

<h2>7. הסתייגויות</h2>
<div class="caveat">
  <ul>
    <li><strong>N = {n}</strong>: עוצמה סטטיסטית מוגבלת; קורלציות מתחת ל-|0.3| עשויות לא להגיע למובהקות.</li>
    <li><strong>מיון רב</strong>: FDR-BH מקיף אבל 60 השוואות מתוך אותו מדגם עדיין רגישות לטעויות סוג I.</li>
    <li><strong>Forward Flow</strong>: מבוסס fastText pre-trained; וקטור ממוצע פר כתבה אינו תופס הקשר תחבירי.</li>
    <li><strong>גרף סטטי</strong>: נשלף ע"י Wikipedia API ועשוי להבדיל מ-snapshot המדויק של רגע הקריאה.</li>
    <li><strong>סיווג בדיד</strong>: לא נעשה. במאמר Zhou הציונים רציפים, ונאמן למאמר השארנו אותם כך.</li>
  </ul>
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
