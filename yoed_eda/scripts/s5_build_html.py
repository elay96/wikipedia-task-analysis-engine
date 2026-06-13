"""s5_build_html.py - Build Hebrew HTML findings page for YOED EDA."""

import base64
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "output"
FIG_DIR = OUT_DIR / "figures"
CORR_CSV = OUT_DIR / "spearman_correlations.csv"
HTML_OUT = OUT_DIR / "yoed_eda_findings.html"


def img_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def read_correlations() -> list[dict]:
    rows = []
    with open(CORR_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def top15_table(rows: list[dict]) -> str:
    sorted_rows = sorted(rows, key=lambda r: abs(float(r["rho"])), reverse=True)
    top = sorted_rows[:15]

    def rho_fmt(v: str) -> str:
        return f"{float(v):.3f}"

    def p_fmt(v: str) -> str:
        val = float(v)
        if val < 0.001:
            return "&lt;0.001"
        return f"{val:.3f}"

    def sig_badge(r: dict) -> str:
        if r["fdr_significant"].strip().lower() == "true":
            return '<span class="badge badge-green">FDR sig</span>'
        return '<span class="badge badge-red">n.s.</span>'

    rows_html = "\n".join(
        f"<tr>"
        f"<td>{r['measure']}</td>"
        f"<td><code>{r['feature']}</code></td>"
        f"<td style='text-align:center'>{rho_fmt(r['rho'])}</td>"
        f"<td style='text-align:center'>{p_fmt(r['p'])}</td>"
        f"<td style='text-align:center'>{p_fmt(r['p_FDR'])}</td>"
        f"<td style='text-align:center'>{r['n']}</td>"
        f"<td style='text-align:center'>{sig_badge(r)}</td>"
        f"</tr>"
        for r in top
    )

    return f"""
<table style="width:100%;border-collapse:collapse;font-size:0.88rem;direction:ltr;">
  <thead>
    <tr style="background:#f4f3f1;font-family:'Rubik',sans-serif;font-weight:600;">
      <th style="padding:8px 10px;text-align:right;border-bottom:2px solid #e8e5e1;">Measure</th>
      <th style="padding:8px 10px;text-align:right;border-bottom:2px solid #e8e5e1;">Feature</th>
      <th style="padding:8px 10px;text-align:center;border-bottom:2px solid #e8e5e1;">&rho;</th>
      <th style="padding:8px 10px;text-align:center;border-bottom:2px solid #e8e5e1;">p</th>
      <th style="padding:8px 10px;text-align:center;border-bottom:2px solid #e8e5e1;">p_FDR</th>
      <th style="padding:8px 10px;text-align:center;border-bottom:2px solid #e8e5e1;">n</th>
      <th style="padding:8px 10px;text-align:center;border-bottom:2px solid #e8e5e1;">FDR</th>
    </tr>
  </thead>
  <tbody>
{rows_html}
  </tbody>
</table>"""


def scatter_grid(figs: list[tuple[str, str]]) -> str:
    items = []
    for path, caption in figs:
        b64 = img_b64(Path(path))
        items.append(
            f'<div style="flex:1 1 320px;">'
            f'<img src="data:image/png;base64,{b64}" '
            f'style="width:100%;border-radius:8px;border:1px solid #e8e5e1;" alt="{caption}">'
            f'<p style="text-align:center;font-size:0.8rem;color:#5a5a5a;margin-top:6px;">{caption}</p>'
            f"</div>"
        )
    return (
        '<div style="display:flex;flex-wrap:wrap;gap:16px;margin-top:16px;">'
        + "".join(items)
        + "</div>"
    )


def build_html() -> None:
    rows = read_correlations()
    n_sig = sum(1 for r in rows if r["fdr_significant"].strip().lower() == "true")
    print(f"n_sig (FDR-significant correlations): {n_sig}")

    heatmap_b64 = img_b64(FIG_DIR / "corr_heatmap.png")

    scatter_figs = [
        (
            str(FIG_DIR / "scatter_AUT_Belt_-_Number_of_Answers__forward_flow.png"),
            "AUT Belt - #Answers vs. forward_flow",
        ),
        (
            str(FIG_DIR / "scatter_AUT_Belt_-_Number_of_Answers__var_step_distance.png"),
            "AUT Belt - #Answers vs. var_step_distance",
        ),
        (
            str(FIG_DIR / "scatter_AUT_Broom_-_Number_of_Answers__char_path_length.png"),
            "AUT Broom - #Answers vs. char_path_length",
        ),
        (
            str(FIG_DIR / "scatter_AUT_Broom_-_Number_of_Answers__global_efficiency.png"),
            "AUT Broom - #Answers vs. global_efficiency",
        ),
        (
            str(FIG_DIR / "scatter_AUT_Broom_-_Originality__var_step_distance.png"),
            "AUT Broom - Originality vs. var_step_distance",
        ),
        (
            str(FIG_DIR / "scatter_Verbal_Fluency_-_Forward_Flow__char_path_length.png"),
            "Verbal Fluency - Forward Flow vs. char_path_length",
        ),
    ]

    table_html = top15_table(rows)
    scatter_html = scatter_grid(scatter_figs)

    html = f"""<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>YOED EDA &mdash; ממצאים</title>
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
  .sub-div {{ margin-top: 10px; padding-right: 14px; border-right: 3px solid var(--yellow-border); }}
  .sub-div strong {{ color: var(--text); }}

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
    <a href="#experiment">הניסוי והדאטה</a>
    <span class="nav-sep" aria-hidden="true">&middot;</span>
    <a href="#reconstruction">שחזור דפים</a>
    <span class="nav-sep" aria-hidden="true">&middot;</span>
    <a href="#features">מאפייני גלישה</a>
    <span class="nav-sep" aria-hidden="true">&middot;</span>
    <a href="#correlations">קשר יצירתיות</a>
    <span class="nav-sep" aria-hidden="true">&middot;</span>
    <a href="#caveats">הסתייגויות</a>
  </div>
</nav>

<div class="page-body">
<div class="container">

<header>
  <h1>YOED EDA &mdash; ממצאים</h1>
  <div class="subtitle">N=107 משתתפים &middot; 195 קורלציות Spearman &middot; 0 שרדו FDR</div>
</header>

<!-- ===== SECTION 0: BOTTOM LINE ===== -->
<section class="section" id="bottom-line">
  <h2 class="section-heading"><span class="section-num">0</span>שורה תחתונה</h2>
  <div class="bottom-line">
    <div class="card-title">מה אפשר להסיק</div>
    <ol>
      <li><strong>תוצאת null אחרי תיקון ריבוי השוואות.</strong>
        מבין 195 קורלציות (13 מדדי יצירתיות/קוגניציה x 15 מאפייני גלישה), {n_sig} שרדו תיקון BH-FDR (&alpha;=0.05). הדאטה אינו מספק עדות לקשר מובהק בין יצירתיות לדפוס הגלישה בויקיפדיה.</li>
      <li><strong>המגמות הגולמיות החזקות ביותר מכוונות ל-AUT.</strong>
        מדדי AUT (חשיבה מתבדרת - Belt/Broom) מראים את הקורלציות הגולמיות הגבוהות ביותר עם פיזור סמנטי (<code>var_step_distance</code>, <code>forward_flow</code>) ואורך מסלול ברשת (<code>char_path_length</code>). כלומר: יותר תשובות AUT = מסע גלישה מגוון ורחב יותר - אך לא מובהק לאחר תיקון.</li>
      <li><strong>כוח סטטיסטי מוגבל.</strong>
        N=107 עם 195 השוואות מגביל את הכוח לאיתור אפקטים קטנים. תוצאת ה-null עקבית עם גודל המדגם ולא בהכרח משקפת היעדר קשר אמיתי.</li>
      <li><strong>הדאטה עשיר ואיכותי.</strong>
        99.2% שחזור היסטורי, 712 ביקורים תוכניים, 469 גרסאות ייחודיות - תשתית מוצקה לניתוחים עתידיים עם N גדול יותר.</li>
    </ol>
  </div>
</section>

<!-- ===== SECTION 1: EXPERIMENT ===== -->
<section class="section" id="experiment">
  <h2 class="section-heading"><span class="section-num">1</span>הניסוי והדאטה</h2>
  <div class="card">
    <div class="card-title">משתתפים וסביבה</div>
    <p>107 משתתפים השתתפו בניסוי. כל אחד בחר שאלה לחקור וגלש בחופשיות במראת ויקיפדיה (פרוקסי מקוון).</p>
    <ul>
      <li><strong>453 דפים ייחודיים</strong> נבקרו בסך הכול.</li>
      <li><strong>836 ביקורי דף</strong> ו-<strong>599 חיפושים</strong> נרשמו.</li>
      <li><strong>טווח תאריכים:</strong> 18/11/2025 - 31/12/2025.</li>
    </ul>
  </div>
  <div class="card">
    <div class="card-title">שלושה רבדי נתונים</div>
    <ul>
      <li><strong>(א) מדדי יצירתיות/קוגניציה (13 מדדים):</strong> Verbal Fluency (כמות + Forward Flow), AUT Broom/Belt (כמות + מקוריות), AQT Pencil/Pillow (כמות + מקוריות + Complexity), Curiosity, GF.</li>
      <li><strong>(ב) התנהגות גלישה:</strong> 836 ביקורי דף, 599 חיפושים, 15 מאפיינים מחושבים לכל משתתף.</li>
      <li><strong>(ג) דיווח עצמי:</strong> שאלת המשתתף, ידע קודם, למידה נתפסת, תגליות, ומשוב AI.</li>
    </ul>
  </div>
</section>

<!-- ===== SECTION 2: RECONSTRUCTION ===== -->
<section class="section" id="reconstruction">
  <h2 class="section-heading"><span class="section-num">2</span>שחזור הדפים (היסטורי)</h2>
  <div class="card">
    <div class="card-title">שיטת שחזור גרסאות</div>
    <p>לכל ביקור אותרה גרסת הדף (revision) שהייתה חיה ברגע הביקור המדויק, דרך MediaWiki API
      (<code>rvstart=timestamp, rvdir=older</code>).</p>
  </div>
  <div class="card">
    <div class="card-title">סטטיסטיקת שחזור</div>
    <ul>
      <li><strong>712</strong> ביקורים "תוכניים" לאחר סינון <strong>124</strong> דפי ניווט (Main_Page ודפי פירושונים).</li>
      <li><strong>706/712 (99.2%)</strong> גרסאות אותרו בהצלחה.</li>
      <li><strong>469</strong> גרסאות ייחודיות נמשכו (dedup לפי <code>revid</code>).</li>
      <li><strong>6</strong> לא אותרו - דפי Special/ערך שנמחק.</li>
      <li><strong>5</strong> כשלי extract - namespace של Category/File.</li>
    </ul>
    <p>לכל גרסה נשמרו: טקסט מלא, לינקים יוצאים, וקטגוריות.</p>
  </div>
</section>

<!-- ===== SECTION 3: FEATURES ===== -->
<section class="section" id="features">
  <h2 class="section-heading"><span class="section-num">3</span>מאפייני הגלישה (15 מאפיינים למשתתף)</h2>
  <div class="card">
    <div class="card-title">סמנטיים - מ-fastText embeddings על הטקסט</div>
    <ul>
      <li><strong><code>mean_step_distance</code>, <code>var_step_distance</code>:</strong> מרחק סמנטי בין דפים עוקבים = רוחב חקירה. שונות גבוהה = מעבר בין נושאים מרוחקים.</li>
      <li><strong><code>forward_flow</code>:</strong> מרחק מצטבר מכל הדפים הקודמים - מדד לחדשנות/חקירה כוללת.</li>
    </ul>
  </div>
  <div class="card">
    <div class="card-title">מבנה רשת - Zhou Hunter/Busybody</div>
    <ul>
      <li><strong><code>bh_score</code>:</strong> ציון Hunter-Busybody על תת-גרף הדפים שבוקרו.</li>
      <li><strong><code>clustering</code>:</strong> מקדם קלאסטרינג - כמה "שכנויות" הדפים שייכים לאשכולות צפופים.</li>
      <li><strong><code>global_efficiency</code>:</strong> יעילות גלובלית של תת-גרף הניווט.</li>
      <li><strong><code>char_path_length</code>:</strong> אורך מסלול ממוצע בין דפים - מרחק במרחב הרשת.</li>
    </ul>
  </div>
  <div class="card">
    <div class="card-title">מבניים - ספירות והתנהגות</div>
    <ul>
      <li><strong><code>n_pages</code>, <code>n_unique_pages</code>:</strong> סך ביקורים ודפים ייחודיים. חציון: 6 (טווח 0-19).</li>
      <li><strong><code>n_searches</code>:</strong> מספר חיפושים. חציון: 5.</li>
      <li><strong><code>revisit_rate</code>:</strong> שיעור ביקורים חוזרים.</li>
      <li><strong><code>search_vs_link_ratio</code>:</strong> יחס חיפוש/ניווט-דרך-לינק.</li>
      <li><strong><code>mean_dwell</code>, <code>var_dwell</code>:</strong> זמן שהייה ממוצע ושונות. חציון שהייה ~41 שניות.</li>
      <li><strong><code>path_breadth</code>:</strong> מספר קטגוריות ויקיפדיה ייחודיות שנבקרו. ממוצע ~29.</li>
    </ul>
  </div>
</section>

<!-- ===== SECTION 4: CORRELATIONS ===== -->
<section class="section" id="correlations">
  <h2 class="section-heading"><span class="section-num">4</span>קשר יצירתיות&harr;התנהגות גלישה</h2>
  <div class="card">
    <div class="card-title">שיטה</div>
    <p>Spearman לכל זוג מדד&times;מאפיין, תיקון Benjamini-Hochberg FDR על כל 195 ההשוואות יחד (&alpha;=0.05), בעקביות עם ניתוחי ה-M הקודמים.</p>
    <div class="callout info">
      <strong>תוצאה:</strong> {n_sig} מובהקים אחרי FDR מתוך 195 קורלציות.
    </div>
  </div>
  <div class="card">
    <div class="card-title">מפת חום - &rho; לכל זוג מדד&times;מאפיין</div>
    <p style="font-size:0.85rem;color:#5a5a5a;margin-bottom:12px;">* מסמן מובהקות FDR אם קיים. אף תא לא קיבל *.</p>
    <img src="data:image/png;base64,{heatmap_b64}"
         style="width:100%;border-radius:8px;border:1px solid #e8e5e1;"
         alt="Correlation heatmap">
  </div>
  <div class="card">
    <div class="card-title">TOP 15 קורלציות לפי |&rho;|</div>
    <p style="font-size:0.85rem;color:#5a5a5a;margin-bottom:12px;">נקראו מ-<code>spearman_correlations.csv</code>. כל הערכים n.s. לאחר FDR.</p>
    {table_html}
  </div>
  <div class="card">
    <div class="card-title">Scatter plots - האפקטים הגולמיים החזקים ביותר</div>
    <p>המגמות הלא-מובהקות מכוונות לכך שמדדי AUT (חשיבה מתבדרת) הולכים עם פיזור סמנטי גדול יותר
       במסע הגלישה: יותר תשובות AUT Belt = <code>forward_flow</code> גבוה יותר ו-<code>var_step_distance</code> גדול יותר;
       AUT Broom = מסלול רשת ארוך יותר (<code>char_path_length</code>) ויעילות גבוהה יותר (<code>global_efficiency</code>).
       כל האפקטים נשארו לא-מובהקים לאחר תיקון FDR.</p>
    {scatter_html}
  </div>
</section>

<!-- ===== SECTION 5: CAVEATS ===== -->
<section class="section" id="caveats">
  <h2 class="section-heading"><span class="section-num">5</span>הערות איכות-דאטה ומגבלות</h2>
  <div class="callout warn">
    <ul style="margin-right:18px;">
      <li><strong>זמן שהייה מוערך:</strong> <code>duration_ms</code>/<code>end_time</code> היו ריקים במקור; זמן השהייה נגזר מהפרש <code>start_time</code> בין דפים עוקבים (לדף האחרון: <code>ended_at</code>). חציון ~41 שניות, ללא ערכים שליליים.</li>
      <li><strong>דפי ניווט הוצאו:</strong> 124 דפי ניווט (Main_Page/פירושונים) הוצאו מהניתוח הסמנטי.</li>
      <li><strong>6 דפים ללא גרסה היסטורית</strong> הושמטו (דפי Special/ערכים שנמחקו).</li>
      <li><strong>כוח סטטיסטי מוגבל:</strong> N=107 עם 195 השוואות - כוח מוגבל לאיתור אפקטים קטנים. תוצאת ה-null עקבית עם זאת ואינה מוכיחה היעדר קשר.</li>
      <li><strong>קטגוריות ויקיפדיה:</strong> נמשכו מהגרסה הנוכחית - קירוב יציב לגרסה ההיסטורית אך לא זהה לה.</li>
    </ul>
  </div>
</section>

</div>
</div>
</body>
</html>"""

    HTML_OUT.write_text(html, encoding="utf-8")
    print(f"Output: {HTML_OUT}")


if __name__ == "__main__":
    build_html()
