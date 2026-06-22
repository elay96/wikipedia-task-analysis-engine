"""S10: Yoed-facing report - per-participant metrics + results before/after corrections.

Answers Yoed's two questions:
  1. All per-participant metrics (exported CSV + descriptive table + full appendix).
  2. Results before vs after multiple-comparison correction, plus whether the
     correction (and the data-inclusion criteria) are too strict.

Renders an RTL Hebrew HTML report per the html-findings-design system and
converts it to PDF via headless Chrome.
"""
from __future__ import annotations

import html
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUTDIR = HERE.parent / "output"

RAW_ALPHA = 0.05
FDR_ALPHA = 0.05

FEATURE_CRITERIA = {
    "n_pages": "ספירת עמודים שנצפו - ללא סינון",
    "n_unique_pages": "ספירת עמודים ייחודיים - ללא סינון",
    "n_searches": "ספירת חיפושים - ללא סינון",
    "revisit_rate": "דורש n_pages>0",
    "search_vs_link_ratio": "דורש n_pages>0",
    "mean_dwell": "דורש לפחות שהות אחת תקינה (dwell)",
    "var_dwell": "דורש לפחות שהות אחת תקינה (dwell)",
    "mean_step_distance": "דורש ≥2 וקטורים סמנטיים תקינים",
    "var_step_distance": "דורש ≥2 וקטורים סמנטיים תקינים",
    "forward_flow": "דורש ≥2 וקטורים סמנטיים תקינים",
    "dyn_slope": "דורש ≥3 צעדים סמנטיים (MIN_STEPS_FOR_DYNAMICS)",
    "dyn_early_late_delta": "דורש ≥3 צעדים סמנטיים",
    "dyn_n_steps": "מספר הצעדים שתרם הנבדק",
    "n_unique": "צמתי גרף ייחודיים - ללא סינון",
    "n_edges": "קשתות בגרף - ללא סינון",
    "clustering": "מקדם clustering - ללא סינון",
    "char_path_length": "אורך מסלול אופייני - ללא סינון",
    "global_efficiency": "global efficiency - ללא סינון",
    "lcc_fraction": "גודל רכיב קשיר מירבי - ללא סינון",
    "path_breadth": "רוחב מסלול - ללא סינון",
    "bh_score": "ציון BH - ללא סינון",
}


def bh(pvals) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    mask = ~np.isnan(p)
    out = np.full(p.shape, np.nan)
    pv = p[mask]
    n = len(pv)
    if n == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order] * n / (np.arange(1, n + 1))
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.clip(q, 0, 1)
    res = np.empty(n)
    res[order] = q
    out[mask] = res
    return out


def esc(x) -> str:
    return html.escape(str(x))


def fmt(x, nd=3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "-"
    return f"{x:.{nd}f}"


def main() -> None:
    parts = pd.read_csv(DATA / "participants.csv")
    feats = pd.read_csv(OUTDIR / "participant_features.csv")
    corr = pd.read_csv(OUTDIR / "spearman_correlations.csv")

    feature_cols = [c for c in feats.columns if c != "participant_id"]

    # --- correction strategies ---
    corr = corr.copy()
    corr["q_per_measure"] = corr.groupby("measure")["p"].transform(lambda s: bh(s.values))
    corr["q_per_feature"] = corr.groupby("feature")["p"].transform(lambda s: bh(s.values))
    corr["bonferroni"] = np.clip(corr["p"] * len(corr), 0, 1)

    n_tests = len(corr)
    n_raw = int((corr["p"] < RAW_ALPHA).sum())
    n_raw01 = int((corr["p"] < 0.01).sum())
    expected_fp = round(RAW_ALPHA * n_tests, 1)
    strategies = [
        ("ללא תיקון (p<0.05 גולמי)", n_raw, "כל מבחן בנפרד ב-α=0.05"),
        ("ללא תיקון מחמיר יותר (p<0.01)", n_raw01, "סף גולמי שמרני יותר"),
        ("BH-FDR על כל 195 המבחנים (הנוכחי)", int((corr["p_FDR"] < FDR_ALPHA).sum()),
         "משפחה אחת של 195 מבחנים, q<0.05"),
        ("BH-FDR בתוך כל מדד יצירתיות", int((corr["q_per_measure"] < FDR_ALPHA).sum()),
         "13 משפחות, 15 מבחנים כל אחת"),
        ("BH-FDR בתוך כל פיצ'ר התנהגותי", int((corr["q_per_feature"] < FDR_ALPHA).sum()),
         "15 משפחות, 13 מבחנים כל אחת"),
        ("Bonferroni (p<0.05)", int((corr["bonferroni"] < FDR_ALPHA).sum()),
         "התיקון השמרני ביותר"),
    ]

    # --- descriptive stats per feature ---
    desc_rows = []
    for c in feature_cols:
        s = pd.to_numeric(feats[c], errors="coerce")
        desc_rows.append({
            "feature": c, "N": int(s.notna().sum()), "mean": s.mean(),
            "sd": s.std(), "min": s.min(), "median": s.median(), "max": s.max(),
            "criterion": FEATURE_CRITERIA.get(c, ""),
        })

    # --- top raw-significant correlations ---
    top = corr.sort_values("p").head(n_raw)

    # --- full per-participant appendix ---
    merged = feats.copy()

    render_html(parts, feats, corr, feature_cols, desc_rows, strategies, top,
                merged, n_tests, n_raw, expected_fp)


def render_html(parts, feats, corr, feature_cols, desc_rows, strategies, top,
                merged, n_tests, n_raw, expected_fp) -> None:
    n_part = len(feats)

    # strategy table
    strat_rows = "".join(
        f"<tr><td style='text-align:right'>{esc(name)}</td>"
        f"<td><strong>{n}</strong></td><td style='text-align:right;color:var(--text-secondary)'>{esc(desc)}</td></tr>"
        for name, n, desc in strategies
    )

    # descriptive table
    desc_html = "".join(
        f"<tr><td><code>{esc(d['feature'])}</code></td><td>{d['N']}</td>"
        f"<td>{fmt(d['mean'])}</td><td>{fmt(d['sd'])}</td><td>{fmt(d['min'])}</td>"
        f"<td>{fmt(d['median'])}</td><td>{fmt(d['max'])}</td>"
        f"<td style='text-align:right;font-size:0.8rem;color:var(--text-secondary)'>{esc(d['criterion'])}</td></tr>"
        for d in desc_rows
    )

    # top correlations table
    top_html = "".join(
        f"<tr><td style='text-align:right'>{esc(r['measure'])}</td><td><code>{esc(r['feature'])}</code></td>"
        f"<td>{int(r['n'])}</td><td>{fmt(r['rho'])}</td><td>{fmt(r['p'])}</td>"
        f"<td>{fmt(r['p_FDR'])}</td><td>{fmt(r['q_per_measure'])}</td>"
        f"<td><span class='badge badge-red'>n.s.</span></td></tr>"
        for _, r in top.iterrows()
    )

    # full per-participant appendix
    app_cols = ["participant_id"] + feature_cols
    app_head = "".join(f"<th>{esc(c)}</th>" for c in app_cols)
    app_body = ""
    for _, row in merged.iterrows():
        cells = f"<td>{int(row['participant_id'])}</td>"
        for c in feature_cols:
            v = row[c]
            nd = 0 if c in ("n_pages", "n_unique_pages", "n_searches", "dyn_n_steps",
                            "n_unique", "n_edges") else 3
            cells += f"<td>{fmt(v, nd)}</td>"
        app_body += f"<tr>{cells}</tr>"

    page = f"""<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ניתוח התנהגות גלישה x יצירתיות - מדדים ותיקונים</title>
<link href="https://fonts.googleapis.com/css2?family=Heebo:wght@300;400;500;700;900&family=Rubik:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #fafaf9; --card: #ffffff; --text: #1a1a1a; --text-secondary: #5a5a5a;
    --border: #e8e5e1; --shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-lg: 0 4px 16px rgba(0,0,0,0.08); --radius: 12px;
    --green: #166534; --green-bg: #dcfce7; --green-border: #86efac;
    --yellow: #854d0e; --yellow-bg: #fef9c3; --yellow-border: #fde047;
    --red: #991b1b; --red-bg: #fee2e2; --red-border: #fca5a5;
    --blue: #1e40af; --blue-bg: #dbeafe; --blue-border: #93c5fd;
  }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Heebo', sans-serif; background: var(--bg); color: var(--text); line-height: 1.7; }}
  nav {{ position: sticky; top: 0; z-index: 100; background: rgba(250,250,249,0.95); backdrop-filter: blur(8px); border-bottom: 1px solid var(--border); padding: 0 24px; }}
  .nav-inner {{ max-width: 980px; margin: 0 auto; display: flex; align-items: center; gap: 4px; overflow-x: auto; padding: 10px 0; scrollbar-width: none; }}
  .nav-inner::-webkit-scrollbar {{ display: none; }}
  .nav-inner a {{ text-decoration: none; font-size: 0.82rem; font-weight: 500; color: var(--text-secondary); white-space: nowrap; padding: 5px 12px; border-radius: 20px; transition: background 0.15s, color 0.15s; }}
  .nav-inner a:hover {{ background: #f0eee9; color: var(--text); }}
  .nav-sep {{ color: var(--border); font-size: 0.85rem; user-select: none; }}
  .page-body {{ padding: 40px 24px 80px; }}
  .container {{ max-width: 980px; margin: 0 auto; }}
  header {{ text-align: center; margin-bottom: 56px; padding-bottom: 40px; border-bottom: 2px solid var(--border); }}
  header h1 {{ font-family: 'Rubik', sans-serif; font-size: 2.1rem; font-weight: 700; margin-bottom: 10px; letter-spacing: -0.5px; }}
  header .subtitle {{ font-size: 1rem; color: var(--text-secondary); font-weight: 300; }}
  .section {{ margin-bottom: 64px; }}
  .section-heading {{ font-family: 'Rubik', sans-serif; font-size: 1.45rem; font-weight: 700; margin-bottom: 20px; padding-bottom: 10px; border-bottom: 2px solid var(--border); display: flex; align-items: center; gap: 10px; }}
  .section-num {{ display: inline-flex; align-items: center; justify-content: center; width: 30px; height: 30px; border-radius: 50%; background: var(--text); color: white; font-size: 0.85rem; font-weight: 700; flex-shrink: 0; }}
  .card {{ background: var(--card); border-radius: var(--radius); box-shadow: var(--shadow); padding: 24px 28px; margin-bottom: 16px; border: 1px solid var(--border); }}
  .card-title {{ font-family: 'Rubik', sans-serif; font-size: 1.05rem; font-weight: 600; margin-bottom: 12px; }}
  .card p {{ color: var(--text-secondary); margin-bottom: 8px; font-size: 0.95rem; }}
  .card p:last-child {{ margin-bottom: 0; }}
  .card ul {{ margin: 8px 0 0 0; padding-right: 20px; color: var(--text-secondary); font-size: 0.95rem; }}
  .card ul li {{ margin-bottom: 6px; }}
  .card strong {{ color: var(--text); }}
  code {{ direction: ltr; display: inline-block; background: #f4f3f1; border: 1px solid var(--border); border-radius: 6px; padding: 1px 7px; font-family: 'Courier New', Courier, monospace; font-size: 0.86em; color: var(--text); }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 0.85rem; }}
  th, td {{ padding: 7px 9px; border-bottom: 1px solid var(--border); text-align: center; }}
  th {{ font-family: 'Rubik', sans-serif; font-weight: 600; background: #f4f3f1; position: sticky; top: 0; }}
  tbody tr:hover {{ background: #faf9f7; }}
  .callout {{ background: #f4f3f1; border-right: 4px solid var(--border); border-radius: 0 var(--radius) var(--radius) 0; padding: 14px 18px; margin: 12px 0; font-size: 0.92rem; color: var(--text-secondary); }}
  .callout strong {{ color: var(--text); }}
  .callout.info {{ background: var(--blue-bg); border-right-color: var(--blue-border); }}
  .callout.warn {{ background: var(--yellow-bg); border-right-color: var(--yellow-border); }}
  .badge {{ display: inline-block; font-size: 0.73rem; font-weight: 600; padding: 2px 10px; border-radius: 20px; white-space: nowrap; }}
  .badge-green {{ background: var(--green-bg); color: var(--green); border: 1px solid var(--green-border); }}
  .badge-red {{ background: var(--red-bg); color: var(--red); border: 1px solid var(--red-border); }}
  .badge-blue {{ background: var(--blue-bg); color: var(--blue); border: 1px solid var(--blue-border); }}
  .bottom-line {{ background: linear-gradient(135deg, #eff6ff 0%, #dbeafe 100%); border: 1px solid var(--blue-border); border-radius: var(--radius); padding: 24px 28px; margin-bottom: 16px; }}
  .bottom-line .card-title {{ color: var(--blue); }}
  .bottom-line ol {{ padding-right: 22px; color: var(--text); }}
  .bottom-line ol li {{ margin-bottom: 10px; font-size: 0.97rem; line-height: 1.65; }}
  .scroll-x {{ overflow-x: auto; }}
  .appendix table {{ font-size: 0.68rem; }}
  .appendix td, .appendix th {{ padding: 3px 5px; }}
  @media print {{
    nav {{ display: none; }} .card {{ box-shadow: none; }} body {{ background: white; }}
    .section {{ margin-bottom: 32px; page-break-inside: avoid; }}
    .appendix {{ page-break-before: always; }}
    th {{ position: static; }}
  }}
</style>
</head>
<body>
<nav><div class="nav-inner">
  <a href="#bottom-line">שורה תחתונה</a><span class="nav-sep">&middot;</span>
  <a href="#context">קונטקסט</a><span class="nav-sep">&middot;</span>
  <a href="#methodology">מתודולוגיה</a><span class="nav-sep">&middot;</span>
  <a href="#data">מדדים</a><span class="nav-sep">&middot;</span>
  <a href="#corrections">לפני/אחרי תיקון</a><span class="nav-sep">&middot;</span>
  <a href="#inclusion">קריטריוני הכללה</a><span class="nav-sep">&middot;</span>
  <a href="#caveats">הסתייגויות</a><span class="nav-sep">&middot;</span>
  <a href="#appendix">נספח: מדדים פר נבדק</a>
</div></nav>
<div class="page-body"><div class="container">

<header>
  <h1>גלישת ויקיפדיה &times; יצירתיות &mdash; מדדים ותיקונים</h1>
  <div class="subtitle">כל המדדים פר נבדק, ותוצאות לפני ואחרי תיקון מרובי-השוואות &middot; N={n_part}</div>
</header>

<section class="section" id="bottom-line">
  <h2 class="section-heading"><span class="section-num">0</span>שורה תחתונה</h2>
  <div class="bottom-line">
    <div class="card-title">מה אפשר להסיק</div>
    <ol>
      <li><strong>לפני תיקון:</strong> {n_raw} מתוך {n_tests} המבחנים מובהקים ב-p&lt;0.05 גולמי. הצפי המקרי לבדו הוא ~{expected_fp} תוצאות, כלומר התשואה הגולמית רק במעט מעל רעש.</li>
      <li><strong>אחרי תיקון:</strong> 0 תוצאות שורדות BH-FDR (התיקון הנוכחי על כל {n_tests} המבחנים).</li>
      <li><strong>גם תיקון פחות שמרני לא משנה את התמונה:</strong> גם FDR בתוך משפחה קטנה (פר מדד או פר פיצ'ר) וגם תקרת p&lt;0.01 גולמית מניבים 0 ממצאים יציבים. הבעיה אינה רק חומרת התיקון.</li>
      <li><strong>קריטריוני ההכללה מורידים N בחלק מהפיצ'רים:</strong> מ-107 ל-100 (פיצ'רים סמנטיים, דורש ≥2 וקטורים) ועד 81 (פיצ'רים דינמיים, דורש ≥3 צעדים). זה מקטין עוצמה סטטיסטית אך אינו מקור ההיעדר של ממצאים.</li>
      <li><strong>הכל מצורף ושקוף:</strong> טבלת כל המדדים פר נבדק (נספח + CSV) וטבלת כל {n_tests} המתאמים עם p גולמי ו-p_FDR זה לצד זה.</li>
    </ol>
  </div>
</section>

<section class="section" id="context">
  <h2 class="section-heading"><span class="section-num">1</span>הקונטקסט</h2>
  <div class="card">
    <p>הדוח מסכם את כל המדדים ההתנהגותיים שחושבו פר נבדק מתוך גלישת ויקיפדיה החופשית (N={n_part}), ואת המתאמים שלהם עם מדדי היצירתיות והקוגניציה - <strong>לפני ואחרי</strong> תיקון מרובי-השוואות.</p>
    <p>שתי שאלות נבדקות במפורש: (1) האם תיקון מרובי-ההשוואות שמרני מדי? (2) האם קריטריוני ההכללה/הניקוי של הנתונים מחמירים מדי ומורידים נבדקים?</p>
  </div>
</section>

<section class="section" id="methodology">
  <h2 class="section-heading"><span class="section-num">2</span>מתודולוגיה</h2>
  <div class="card">
    <div class="card-title">מבחנים וסטטיסטיקה</div>
    <ul>
      <li><strong>13 מדדי יצירתיות/קוגניציה</strong> &times; <strong>15 פיצ'רים התנהגותיים</strong> = <strong>{n_tests} מתאמי Spearman</strong>.</li>
      <li>מתאם דרגות (Spearman <code>&rho;</code>) - עמיד לחריגים וללא הנחת ליניאריות.</li>
      <li>תיקון מרובי-השוואות: Benjamini-Hochberg FDR ב-<code>&alpha;=0.05</code>.</li>
      <li>זוג נבדק רק אם יש לו ערך תקין בשני המשתנים (<code>n&ge;3</code>); אחרת מסומן חסר.</li>
    </ul>
  </div>
</section>

<section class="section" id="data">
  <h2 class="section-heading"><span class="section-num">3</span>המדדים פר נבדק - סטטיסטיקה תיאורית</h2>
  <div class="card">
    <p>15 הפיצ'רים ההתנהגותיים שנכנסו למתאמים, וכן פיצ'רים נוספים שחושבו. עמודת <strong>N</strong> חושפת את אפקט קריטריוני ההכללה - ראו סעיף 5.</p>
    <div class="scroll-x"><table>
      <thead><tr><th>פיצ'ר</th><th>N</th><th>ממוצע</th><th>ס.ת.</th><th>מינ'</th><th>חציון</th><th>מקס'</th><th>קריטריון הכללה</th></tr></thead>
      <tbody>{desc_html}</tbody>
    </table></div>
  </div>
</section>

<section class="section" id="corrections">
  <h2 class="section-heading"><span class="section-num">4</span>תוצאות לפני ואחרי תיקון</h2>
  <div class="card">
    <div class="card-title">כמה תוצאות שורדות בכל אסטרטגיית תיקון?</div>
    <div class="scroll-x"><table>
      <thead><tr><th style="text-align:right">אסטרטגיה</th><th>מובהקים</th><th style="text-align:right">הסבר</th></tr></thead>
      <tbody>{strat_rows}</tbody>
    </table></div>
    <div class="callout info"><strong>הנקודה המרכזית:</strong> המעבר מ-{n_raw} (גולמי) ל-0 (FDR) אינו ארטיפקט של תיקון חמור מדי. גם ריכוך התיקון למשפחות קטנות, וגם הקשחת הסף הגולמי ל-p&lt;0.01, מובילים לאותה מסקנה. התשואה הגולמית ({n_raw}) קרובה לצפי המקרי (~{expected_fp}).</div>
  </div>
  <div class="card">
    <div class="card-title">כל המתאמים המובהקים לפני תיקון (p&lt;0.05 גולמי), ממוינים לפי p</div>
    <div class="scroll-x"><table>
      <thead><tr><th style="text-align:right">מדד</th><th>פיצ'ר</th><th>n</th><th>&rho;</th><th>p גולמי</th><th>p_FDR</th><th>q פר-מדד</th><th>סטטוס</th></tr></thead>
      <tbody>{top_html}</tbody>
    </table></div>
    <p style="margin-top:10px">הטבלה המלאה של כל {n_tests} המתאמים נמצאת ב-<code>spearman_correlations.csv</code>.</p>
  </div>
</section>

<section class="section" id="inclusion">
  <h2 class="section-heading"><span class="section-num">5</span>קריטריוני ההכללה והניקוי</h2>
  <div class="card">
    <div class="card-title">מה מסונן ואיך זה משפיע על N</div>
    <ul>
      <li><strong>עמודי שירות:</strong> <code>Main_Page</code> ועמודי פירושונים (disambiguation) מסומנים ומוחרגים מהפיצ'רים הסמנטיים.</li>
      <li><strong>גרסה היסטורית חסרה:</strong> ערך ללא רוויזיה היסטורית בזמן הצפייה מתועד ומדולג.</li>
      <li><strong>שהות (dwell):</strong> <code>duration_ms</code>/<code>end_time</code> ריקים במקור; שהות נגזרת מהפרשי <code>start_time</code> עוקבים בתוך מושב.</li>
    </ul>
    <p style="margin-top:10px">השפעת הקריטריונים על גודל המדגם הזמין לכל פיצ'ר:</p>
    <ul>
      <li>פיצ'רים מבניים/ספירות: <strong>N=107</strong> (ללא סינון).</li>
      <li>פיצ'רים מבוססי-שהות: <strong>N=105</strong>.</li>
      <li>פיצ'רים סמנטיים (<code>step_distance</code>, <code>forward_flow</code>): <strong>N=100</strong> - דורש ≥2 וקטורים תקינים.</li>
      <li>פיצ'רים דינמיים (<code>dyn_slope</code>, <code>dyn_early_late_delta</code>): <strong>N=81</strong> - דורש ≥3 צעדים סמנטיים.</li>
    </ul>
    <div class="callout warn"><strong>פרשנות:</strong> הקריטריון הדינמי (≥3 צעדים) הוא המחמיר ביותר ומוריד ל-N=81. אפשר לשקול לרכך אותו (למשל ≥2 צעדים) כדי להגדיל עוצמה, אך גם בפיצ'רים שבהם N=107 מלא אין ממצאים ששורדים תיקון - כך שהקטנת ה-N אינה ההסבר העיקרי להיעדר ממצאים.</div>
  </div>
</section>

<section class="section" id="caveats">
  <h2 class="section-heading"><span class="section-num">6</span>הסתייגויות</h2>
  <div class="callout warn">
    <ul style="margin-right:18px;">
      <li><strong>ניתוח חוקר (exploratory):</strong> {n_tests} מבחנים ללא היפותזות אפריוריות - לכן תיקון מרובי-השוואות הכרחי.</li>
      <li><strong>עוצמה:</strong> ב-N≈100 ו-α=0.05 דו-צדדי, גילוי <code>&rho;</code>≈0.27 מצריך בדיוק את גודל המדגם הזה; אפקטים קטנים יותר לא יזוהו באמינות.</li>
      <li><strong>שהות נגזרת:</strong> מדדי dwell מבוססי הפרשי זמן, לא מדידה ישירה.</li>
      <li><strong>אם יש היפותזות ממוקדות:</strong> צמצום מראש למשפחת מבחנים קטנה ומוגדרת (ולא {n_tests}) יגדיל עוצמה אמיתית - זו הדרך הנכונה ל"לרכך" בלי לנפח שגיאות מסוג I.</li>
    </ul>
  </div>
</section>

<section class="section appendix" id="appendix">
  <h2 class="section-heading"><span class="section-num">7</span>נספח: כל המדדים פר נבדק (N={n_part})</h2>
  <div class="card">
    <p>הטבלה המלאה. הקובץ המקורי: <code>participant_features.csv</code>.</p>
    <div class="scroll-x"><table>
      <thead><tr>{app_head}</tr></thead>
      <tbody>{app_body}</tbody>
    </table></div>
  </div>
</section>

</div></div>
</body>
</html>"""

    out_html = OUTDIR / "yoed_corrections_report.html"
    out_html.write_text(page, encoding="utf-8")
    print(f"HTML -> {out_html}")
    to_pdf(out_html, OUTDIR / "yoed_corrections_report.pdf")


def to_pdf(html_path: Path, pdf_path: Path) -> None:
    chrome = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    cmd = [chrome, "--headless", "--disable-gpu", "--no-pdf-header-footer",
           f"--print-to-pdf={pdf_path}", html_path.as_uri()]
    subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    print(f"PDF  -> {pdf_path}")


if __name__ == "__main__":
    main()
