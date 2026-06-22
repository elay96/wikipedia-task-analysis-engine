"""S11: single-page figure of the correlations significant BEFORE correction.

Shows only the raw-significant (p<0.05, uncorrected) browsing x creativity
Spearman correlations, sorted by p, as a forest plot with Fisher 95% CIs.
Visual style mirrors the thesis-spatial-cfg W-report (eyebrow, headline card,
numbered figure caption, how-to card, footer). English by design - the data
labels are English and matplotlib has no RTL shaping here.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import _bootstrap  # noqa: F401
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
OUTDIR = HERE.parent / "output"

RAW_ALPHA = 0.05
NEGLIGIBLE = 0.1  # |rho| below this shaded as a negligible effect
HEADER_BAND = "#f4f4f3"

# Short inclusion-criterion note per feature (drives the N column).
FEATURE_CRITERION = {
    "mean_step_distance": "≥2 semantic vectors", "var_step_distance": "≥2 semantic vectors",
    "forward_flow": "≥2 semantic vectors",
    "dyn_slope": "≥3 semantic steps", "dyn_early_late_delta": "≥3 semantic steps",
    "revisit_rate": "n_pages > 0", "search_vs_link_ratio": "n_pages > 0",
    "mean_dwell": "valid dwell", "var_dwell": "valid dwell",
}

# Palette copied from the W-report style.
INK = "#1f2937"; INK_STRONG = "#111827"; MUTED = "#6b7280"; FAINT = "#9ca3af"
HAIRLINE = "#e5e7eb"; ACCENT_BLUE = "#2563eb"; ACCENT_BLUE_FILL = "#f5f8ff"
NEUTRAL_ACCENT = "#94a3b8"; NEUTRAL_FILL = "#f8f8f7"
POS_COLOR = "#1e40af"  # positive rho
NEG_COLOR = "#b91c1c"  # negative rho


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": "#444", "axes.labelcolor": "#222",
        "xtick.color": "#444", "ytick.color": "#444", "pdf.fonttype": 42,
    })


def fisher_ci(rho: float, n: int, conf: float = 0.95) -> tuple[float, float]:
    """Approximate CI for Spearman rho via Fisher z with the Fieller SE."""
    if n is None or n < 5 or abs(rho) >= 1:
        return rho, rho
    z = np.arctanh(rho)
    se = np.sqrt(1.06 / (n - 3))
    crit = 1.959963984540054 * se
    return float(np.tanh(z - crit)), float(np.tanh(z + crit))


def short_measure(m: str) -> str:
    return (m.replace(" - Number of Answers", " #")
             .replace(" - Originality", " Orig")
             .replace(" - Forward Flow", " FF")
             .replace(" - Score", "")
             .replace("Verbal Fluency", "VF"))


def _card(ax, accent, fill):
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=fill, edgecolor=HAIRLINE,
                           linewidth=1.0, transform=ax.transAxes, clip_on=False, zorder=0))
    ax.add_patch(Rectangle((0, 0), 0.009, 1, facecolor=accent, edgecolor="none",
                           transform=ax.transAxes, clip_on=False, zorder=1))


def draw_header(fig, title, subtitle, meta, page_label):
    ax = fig.add_axes([0.07, 0.905, 0.86, 0.08]); ax.axis("off")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0, 1.0, f"WIKIPEDIA BROWSING × CREATIVITY     ·     {page_label.upper()}",
            fontsize=7.8, fontweight="bold", color=ACCENT_BLUE, transform=ax.transAxes, va="top")
    ax.text(0, 0.70, title, fontsize=14.5, fontweight="bold", color=INK_STRONG,
            transform=ax.transAxes, va="top")
    ax.text(0, 0.26, subtitle, fontsize=10, color=MUTED, transform=ax.transAxes, va="top")
    ax.text(0, 0.0, meta, fontsize=8.3, color=FAINT, transform=ax.transAxes, va="top")
    ax.plot([0, 1], [-0.18, -0.18], color=HAIRLINE, lw=1.0, transform=ax.transAxes, clip_on=False)


def draw_callout(fig, rect, title, body, footnote, accent=ACCENT_BLUE, fill=ACCENT_BLUE_FILL):
    ax = fig.add_axes(rect); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _card(ax, accent, fill)
    tx = 0.03
    max_frac = 0.96 - tx
    fig_h_pt = fig.get_size_inches()[1] * 72.0

    def lf(pt):
        return pt / (rect[3] * fig_h_pt)

    cursor = 1.0 - lf(14)
    ax.text(tx, cursor, title, fontsize=10.3, fontweight="bold", color=accent,
            transform=ax.transAxes, va="top")
    cursor -= lf(17)
    for line in _wrap(fig, ax, body, 9.3, max_frac):
        ax.text(tx, cursor, line, fontsize=9.3, color=INK, transform=ax.transAxes, va="top")
        cursor -= lf(12.6)
    if footnote:
        cursor -= lf(3)
        for line in _wrap(fig, ax, footnote, 8.3, max_frac):
            ax.text(tx, cursor, line, fontsize=8.3, color=MUTED, style="italic",
                    transform=ax.transAxes, va="top")
            cursor -= lf(11.2)


def draw_howto(fig, rect, title, lines):
    ax = fig.add_axes(rect); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    _card(ax, NEUTRAL_ACCENT, NEUTRAL_FILL)
    tx = 0.03; leading = 0.0175 / rect[3]
    ax.text(tx, 0.85, title, fontsize=9.6, fontweight="bold", color=INK_STRONG,
            transform=ax.transAxes, va="top")
    cursor = 0.85 - 2.05 * leading
    for line in lines:
        ax.text(tx, cursor, line, fontsize=8.3, color="#4b5563",
                transform=ax.transAxes, va="top")
        cursor -= leading


def _text_width_frac(fig, ax, text, fontsize, **kw):
    """Width of `text` as a fraction of `ax`'s width, measured with the renderer."""
    try:
        renderer = fig.canvas.get_renderer()
    except AttributeError:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        renderer = FigureCanvasAgg(fig).get_renderer()
    probe = ax.text(0, 0, text, fontsize=fontsize, transform=ax.transAxes, **kw)
    w = probe.get_window_extent(renderer).width
    probe.remove()
    return w / ax.get_window_extent(renderer).width


def draw_caption(fig, rect, prefix, heading, sub):
    ax = fig.add_axes(rect); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0, 0.92, prefix, fontsize=12, fontweight="bold", color=ACCENT_BLUE,
            transform=ax.transAxes, va="top")
    offset = _text_width_frac(fig, ax, prefix, 12, fontweight="bold") + 0.012
    ax.text(offset, 0.92, heading, fontsize=12, fontweight="bold", color=INK_STRONG,
            transform=ax.transAxes, va="top")
    ax.text(0, 0.16, sub, fontsize=9, color=MUTED, style="italic",
            transform=ax.transAxes, va="top")


def draw_footer(fig, source_name):
    ax = fig.add_axes([0.07, 0.025, 0.86, 0.02]); ax.axis("off")
    ax.text(0.5, 0.5,
            f"Generated {date.today().isoformat()}     ·     yoed_eda     ·     source: {source_name}",
            fontsize=7, color=FAINT, ha="center", transform=ax.transAxes)


def _wrap(fig, ax, text, fontsize, max_frac, **kw):
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cur and _text_width_frac(fig, ax, trial, fontsize, **kw) > max_frac:
            lines.append(cur); cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def draw_glossary(fig, rect, groups, name_size=8.0):
    """Render grouped definition lists into `rect`.

    `groups` = [(group_title, [(var, description), ...]), ...]. Layout adapts to
    the variable-name width: short names sit inline to the left of the wrapped
    description; long names (e.g. full measure labels) stack above it so they
    never collide with the text.
    """
    ax = fig.add_axes(rect); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig_h_pt = fig.get_size_inches()[1] * 72.0

    def lf(pt):
        return pt / (rect[3] * fig_h_pt)

    max_name = 0.0
    for _, items in groups:
        for var, _ in items:
            max_name = max(max_name, _text_width_frac(fig, ax, var, name_size,
                                                      family="monospace", fontweight="bold"))
    inline = max_name + 0.02 <= 0.34
    desc_x = (max_name + 0.02) if inline else 0.025

    cursor = 1.0
    for gi, (title, items) in enumerate(groups):
        if gi:
            cursor -= lf(7)
        ax.text(0.0, cursor, title, fontsize=9.4, fontweight="bold",
                color=ACCENT_BLUE, transform=ax.transAxes, va="top")
        cursor -= lf(15.5)
        for var, desc in items:
            ax.text(0.0, cursor, var, fontsize=name_size, family="monospace",
                    fontweight="bold", color=INK_STRONG, transform=ax.transAxes, va="top")
            if not inline:
                cursor -= lf(11.0)
            for line in _wrap(fig, ax, desc, 8.3, 1.0 - desc_x):
                ax.text(desc_x, cursor, line, fontsize=8.3, color="#374151",
                        transform=ax.transAxes, va="top")
                cursor -= lf(11.3)
            cursor -= lf(3.5)


def draw_forest(fig, rows, rect):
    n = len(rows)
    plot_rows = rows[::-1]  # smallest-p row on top
    extent = max(abs(r["lo"]) for r in rows) if rows else 0.4
    extent = max(extent, max(abs(r["hi"]) for r in rows), max(abs(r["rho"]) for r in rows))
    extent *= 1.15

    ax = fig.add_axes(rect)
    ax.axvspan(-NEGLIGIBLE, NEGLIGIBLE, color="#f1f0ee", zorder=0)
    ax.axvline(0, color="#333", lw=0.9, zorder=1)
    for i in range(n - 1):
        ax.axhline(i + 0.5, color=HAIRLINE, lw=0.5, zorder=0)

    for i, r in enumerate(plot_rows):
        color = POS_COLOR if r["rho"] >= 0 else NEG_COLOR
        ax.plot([r["lo"], r["hi"]], [i, i], color=color, lw=1.8,
                solid_capstyle="round", zorder=2)
        ax.plot(r["rho"], i, "o", markersize=5.5, markerfacecolor=color,
                markeredgecolor=color, markeredgewidth=1.1, zorder=3)
        ax.text(1.02, i, f"ρ={r['rho']:+.2f}   p={r['p']:.3f}   n={r['n']}",
                transform=ax.get_yaxis_transform(), fontsize=7.6,
                family="monospace", color=INK, va="center", ha="left")

    ax.set_yticks(range(n))
    ax.set_yticklabels([r["label"] for r in plot_rows], fontsize=7.7)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlim(-extent, extent)
    ax.set_xlabel("Spearman ρ   (browsing feature vs creativity measure)",
                  fontsize=9.5, labelpad=7)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", length=0)
    ax.set_axisbelow(True)
    ax.grid(axis="x", linestyle=":", color="#ddd", alpha=0.6)
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", markersize=6,
               markerfacecolor=POS_COLOR, markeredgecolor=POS_COLOR, label="positive ρ"),
        Line2D([0], [0], marker="o", linestyle="none", markersize=6,
               markerfacecolor=NEG_COLOR, markeredgecolor=NEG_COLOR, label="negative ρ"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01),
              ncol=2, frameon=False, fontsize=8.5, handletextpad=0.35, columnspacing=1.8)


def _fmt(v, nd=2):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    a = abs(v)
    if a != 0 and (a >= 1e4 or a < 1e-3):
        mant, exp = f"{v:.1e}".split("e")
        return f"{mant}e{int(exp)}"
    out = f"{v:.{nd}f}"
    return "0.00" if out in ("-0.00", "0.00") else out


def draw_desc_table(fig, desc, rect):
    ax = fig.add_axes(rect); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    headers = ["Feature", "N", "Mean", "SD", "Min", "Median", "Max", "Inclusion criterion"]
    col_x =   [0.005,     0.275, 0.345, 0.430, 0.515, 0.600,   0.690, 0.780]
    aligns =  ["left",   "right","right","right","right","right","right","left"]
    n = len(desc)
    header_h = 0.045
    row_h = (1.0 - header_h) / n

    ax.add_patch(Rectangle((0, 1.0 - header_h), 1.0, header_h, facecolor=HEADER_BAND,
                           edgecolor="none", transform=ax.transAxes, zorder=0))
    for x, h, a in zip(col_x, headers, aligns):
        ax.text(x, 1.0 - header_h / 2, h, fontsize=7.6, fontweight="bold",
                color=MUTED, transform=ax.transAxes, ha=a, va="center")

    for i, d in enumerate(desc):
        top = 1.0 - header_h - i * row_h
        if i % 2 == 1:
            ax.add_patch(Rectangle((0, top - row_h), 1.0, row_h, facecolor=NEUTRAL_FILL,
                                   edgecolor="none", transform=ax.transAxes, zorder=0))
        y = top - row_h / 2
        full = d["N"] == desc[0]["N_max"]
        ax.text(col_x[0], y, d["feature"], fontsize=7.6, family="monospace",
                color=INK_STRONG, transform=ax.transAxes, va="center")
        ax.text(col_x[1], y, str(d["N"]), fontsize=7.6, family="monospace",
                color=INK if full else NEG_COLOR, fontweight="normal" if full else "bold",
                ha="right", transform=ax.transAxes, va="center")
        for cx, key, nd in ((col_x[2], "mean", 2), (col_x[3], "sd", 2),
                            (col_x[4], "min", 2), (col_x[5], "median", 2),
                            (col_x[6], "max", 2)):
            ax.text(cx, y, _fmt(d[key], nd), fontsize=7.6, family="monospace",
                    color="#374151", ha="right", transform=ax.transAxes, va="center")
        crit = d["criterion"]
        ax.text(col_x[7], y, crit, fontsize=7.2,
                color=MUTED if crit == "—" else "#374151",
                transform=ax.transAxes, va="center")

    for yy in (1.0 - header_h, 1.0 - header_h - n * row_h):
        ax.plot([0, 1], [yy, yy], color=HAIRLINE, lw=1.0, transform=ax.transAxes, clip_on=False)


def _render_table_page(pdf, feats, n_part):
    feature_cols = [c for c in feats.columns if c != "participant_id"]
    desc = []
    Ns = [int(pd.to_numeric(feats[c], errors="coerce").notna().sum()) for c in feature_cols]
    n_max = max(Ns)
    for c in feature_cols:
        s = pd.to_numeric(feats[c], errors="coerce")
        desc.append({
            "feature": c, "N": int(s.notna().sum()), "N_max": n_max,
            "mean": s.mean(), "sd": s.std(), "min": s.min(),
            "median": s.median(), "max": s.max(),
            "criterion": FEATURE_CRITERION.get(c, "—"),
        })
    n_reduced = sum(1 for d in desc if d["N"] < n_max)

    fig = plt.figure(figsize=(8.5, 11)); fig.patch.set_facecolor("white")
    draw_header(
        fig, "Per-participant browsing metrics: descriptive statistics",
        "Every behavioural feature computed per participant, with its available N",
        f"N = {n_part} participants   ·   {len(feature_cols)} features   ·   "
        f"{n_reduced} features lose participants to inclusion criteria",
        "Page 3 of 4")
    draw_callout(
        fig, [0.07, 0.765, 0.86, 0.105], "Headline",
        f"All {len(feature_cols)} metrics are shown for every participant (N = {n_part}).",
        "Counts and graph features keep all 107; semantic features need ≥2 vectors "
        "(N=100) and session dynamics need ≥3 steps (N=81). A red, bold N marks a reduced sample.")
    draw_caption(
        fig, [0.07, 0.715, 0.86, 0.045], "Table 1.",
        "Descriptive statistics for every per-participant metric",
        "Mean, SD and range across participants; the N column exposes the inclusion criteria.")
    draw_desc_table(fig, desc, [0.05, 0.165, 0.90, 0.50])
    draw_howto(fig, [0.07, 0.055, 0.86, 0.085], "How to read", [
        "Each row is one browsing feature computed once per participant.",
        "N = participants with a valid value; a red bold N means the criterion dropped some.",
        "Full per-participant values are in participant_features.csv.",
    ])
    draw_footer(fig, "participant_features.csv")
    pdf.savefig(fig, dpi=200); plt.close(fig)


def _render_figure_page(pdf, corr):
    src_name = "spearman_correlations.csv"
    sig = corr[corr["p"] < RAW_ALPHA].sort_values("p").reset_index(drop=True)
    rows = []
    for _, r in sig.iterrows():
        lo, hi = fisher_ci(float(r["rho"]), int(r["n"]))
        rows.append({
            "label": f"{short_measure(r['measure'])}  —  {r['feature']}",
            "rho": float(r["rho"]), "p": float(r["p"]), "n": int(r["n"]),
            "lo": lo, "hi": hi,
        })
    n_tests = len(corr)
    expected = round(RAW_ALPHA * n_tests, 1)

    fig = plt.figure(figsize=(8.5, 11)); fig.patch.set_facecolor("white")
    draw_header(
        fig, "Browsing-creativity correlations significant before correction",
        "Spearman ρ, p < .05 uncorrected, sorted by p (smallest on top)",
        f"N = {sig['n'].min()}–{sig['n'].max()} per test   ·   "
        f"{n_tests} tests total   ·   0 survive BH-FDR",
        "Page 4 of 4")
    draw_callout(
        fig, [0.07, 0.765, 0.86, 0.105], "Headline",
        f"{len(sig)} of {n_tests} raw tests reach p < .05; chance alone expects ~{expected}.",
        "All rows below are uncorrected. After BH-FDR, Bonferroni, or per-family "
        "FDR, none remain significant.")
    draw_caption(
        fig, [0.07, 0.715, 0.86, 0.045], "Figure 1.",
        "Effect size of each pre-correction hit (Spearman ρ, 95% CI)",
        "Positive ρ = more of the feature goes with a higher creativity score.")
    draw_forest(fig, rows, [0.30, 0.215, 0.40, 0.46])
    draw_howto(fig, [0.07, 0.055, 0.86, 0.115], "How to read", [
        "Dot = Spearman ρ.   Bar = its 95% CI (Fisher z, Fieller SE).",
        "Line at 0 = no association.   Shaded band = a negligible effect (|ρ| < 0.1).",
        "Blue = positive association; red = negative.   All rows are p < .05 uncorrected.",
        "None survive multiple-comparison correction, so treat these as exploratory leads.",
    ])
    draw_footer(fig, src_name)
    pdf.savefig(fig, dpi=200); plt.close(fig)
    return len(sig)


BROWSING_GLOSSARY = [
    ("Counts & searches", [
        ("n_pages", "Total article pages the participant opened while browsing."),
        ("n_unique_pages", "Distinct articles opened (repeat opens counted once)."),
        ("n_searches", "Number of search queries the participant typed."),
        ("revisit_rate", "Share of opens that were repeats: (n_pages − n_unique_pages) / n_pages."),
        ("search_vs_link_ratio", "Searches per page opened; high = navigates by searching, low = by links."),
    ]),
    ("Dwell — time on page", [
        ("mean_dwell", "Mean time per page in ms, derived from gaps between consecutive opens."),
        ("var_dwell", "Variance of those per-page dwell times."),
    ]),
    ("Semantic movement — from page-text embeddings", [
        ("mean_step_distance", "Mean cosine distance between consecutive pages' text; bigger = larger topical jumps."),
        ("var_step_distance", "Variance of the step distances (how uneven the jumps are)."),
        ("forward_flow", "Mean semantic distance from each page back to all earlier pages (Gray 2019); higher = steadily diverging."),
    ]),
    ("Session dynamics — within-session trend", [
        ("dyn_slope", "Trend of step distance over the session; negative = jumps shrink (explore→exploit), positive = grow."),
        ("dyn_early_late_delta", "Mean step distance in the second half minus the first half."),
        ("dyn_n_steps", "Number of semantic steps the participant contributed."),
    ]),
    ("Browsing graph — visited-page network (Zhou 2024)", [
        ("n_unique", "Nodes in the visited-page graph (unique articles with cached text)."),
        ("n_edges", "Edges: a link from one visited page to another visited page."),
        ("clustering", "Average clustering coefficient — how interlinked the neighbours are."),
        ("char_path_length", "Mean shortest-path length within the largest connected component."),
        ("global_efficiency", "Average inverse shortest-path — how efficiently the network connects."),
        ("lcc_fraction", "Fraction of visited pages in the largest connected component."),
        ("path_breadth", "Number of distinct Wikipedia categories the visited pages span."),
        ("bh_score", "Busybody–Hunter composite (within-cohort z of edges+clustering+efficiency − path length); higher = more hunter-like."),
    ]),
]

CREATIVITY_GLOSSARY = [
    ("Divergent thinking — fluency (number of answers)", [
        ("AUT Broom - Number of Answers", "Alternative Uses Task: count of unusual uses generated for a broom."),
        ("AUT Belt - Number of Answers", "Alternative Uses Task, object = belt: count of answers."),
        ("AQT Pencil - Number of Answers", "Alternative-uses-style task, object = pencil: count of answers."),
        ("AQT Pillow - Number of Answers", "Alternative-uses-style task, object = pillow: count of answers."),
        ("Verbal Fluency - Number of Answers", "Count of words produced in the verbal fluency task."),
    ]),
    ("Divergent thinking — originality / quality", [
        ("AUT Belt - Originality", "Rated originality of the belt-use answers."),
        ("AUT Broom - Originality", "Rated originality of the broom-use answers."),
        ("AQT Pencil - Originality", "Rated originality of the pencil answers."),
        ("AQT Pillow - Originality", "Rated originality of the pillow answers."),
        ("AQT Complexity Score", "Complexity / elaboration score of the AQT answers."),
    ]),
    ("Semantic forward flow — verbal", [
        ("Verbal Fluency - Forward Flow", "Forward flow on the generated words (Gray 2019): same construct as browsing forward_flow, in word space."),
    ]),
    ("Curiosity — self-report", [
        ("Curiosity - Score", "Self-reported curiosity questionnaire score."),
    ]),
    ("Cognitive control", [
        ("GF - Score", "Fluid-intelligence (Gf) score; included as a control for general cognitive ability."),
    ]),
]


def _render_glossary_browsing(pdf, page_label):
    fig = plt.figure(figsize=(8.5, 11)); fig.patch.set_facecolor("white")
    draw_header(
        fig, "What each variable means — browsing features",
        "Computed per participant from the Wikipedia free-browsing task",
        "21 behavioural features   ·   source: the browsing task (not the questionnaire)",
        page_label)
    draw_caption(
        fig, [0.07, 0.815, 0.86, 0.045], "Legend A.",
        "Browsing features and how each is derived",
        "All features below come from each participant's own browsing trace.")
    draw_glossary(fig, [0.07, 0.075, 0.86, 0.70], BROWSING_GLOSSARY)
    draw_footer(fig, "features.py · s3_semantic_features.py")
    pdf.savefig(fig, dpi=200); plt.close(fig)


def _render_glossary_creativity(pdf, page_label):
    fig = plt.figure(figsize=(8.5, 11)); fig.patch.set_facecolor("white")
    draw_header(
        fig, "What each variable means — creativity & cognition",
        "External measures from the test battery, scored outside the browsing task",
        "13 measures   ·   source: the creativity/cognition battery (questionnaire & tests)",
        page_label)
    draw_caption(
        fig, [0.07, 0.815, 0.86, 0.045], "Legend B.",
        "Creativity and cognition measures and where each comes from",
        "These are the person-level traits correlated against the browsing features.")
    draw_glossary(fig, [0.07, 0.17, 0.86, 0.60], CREATIVITY_GLOSSARY)
    draw_callout(
        fig, [0.07, 0.05, 0.86, 0.105], "How they group in the analysis",
        "Fluency and originality form divergent-thinking composites; verbal forward "
        "flow, curiosity and Gf enter on their own.",
        "Composites follow creativity_model.py.")
    draw_footer(fig, "participants.csv")
    pdf.savefig(fig, dpi=200); plt.close(fig)


def main() -> None:
    corr = pd.read_csv(OUTDIR / "spearman_correlations.csv")
    feats = pd.read_csv(OUTDIR / "participant_features.csv")
    n_part = len(feats)

    setup_style()
    out = OUTDIR / "yoed_metrics_and_hits.pdf"
    with PdfPages(out) as pdf:
        _render_glossary_browsing(pdf, "Page 1 of 4")
        _render_glossary_creativity(pdf, "Page 2 of 4")
        _render_table_page(pdf, feats, n_part)
        n_sig = _render_figure_page(pdf, corr)
    print(f"PDF -> {out}  (4 pages; Table: {n_part} participants; Figure: {n_sig} correlations)")


if __name__ == "__main__":
    main()
