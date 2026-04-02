# Methodology Review HTML Page — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single self-contained HTML file (`docs/methodology_review.html`) in Hebrew RTL that explains the methodological debate around explore/exploit metrics, compares approaches, recommends MANOVA with 2 PCA DVs, and includes a draft email to 3 advisors.

**Architecture:** One HTML file with inline CSS. Google Fonts link for Heebo/Rubik (matching existing `measures_guide.html` style) — degrades gracefully to system sans-serif if offline. 7 sections with sticky top nav. Light theme. Print-friendly via `@media print`.

**Tech Stack:** HTML5, CSS3, no JavaScript required.

---

## File Structure

```
docs/
└── methodology_review.html    # CREATE — the single deliverable
```

---

### Task 1: Create the HTML file with base structure and CSS

**Files:**
- Create: `docs/methodology_review.html`

- [ ] **Step 1: Write the complete HTML file**

Write `docs/methodology_review.html` with all 7 sections. The file structure:

```html
<!DOCTYPE html>
<html lang="he" dir="rtl">
<head>
  <!-- meta charset, viewport, title -->
  <!-- Google Fonts: Heebo + Rubik (matching measures_guide.html) -->
  <!-- All CSS inline in <style> -->
</head>
<body>
  <nav><!-- sticky top nav with 7 section links --></nav>
  <div class="container">
    <header><!-- title + subtitle --></header>
    <section id="intro"><!-- Section 1: Introduction --></section>
    <section id="metrics"><!-- Section 2: Metrics Map table --></section>
    <section id="problems"><!-- Section 3: Three Problems --></section>
    <section id="approaches"><!-- Section 4: Approach Comparison --></section>
    <section id="decision"><!-- Section 5: Decision Framework --></section>
    <section id="recommendation"><!-- Section 6: Recommendation --></section>
    <section id="email"><!-- Section 7: Draft Email --></section>
  </div>
</body>
</html>
```

**CSS design system** (matching `measures_guide.html` conventions):

```css
:root {
  --explore: #2b6cb0;
  --exploit: #c05621;
  --bg: #fafaf9;
  --card: #ffffff;
  --text: #1a1a1a;
  --text-secondary: #5a5a5a;
  --border: #e8e5e1;
  --shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
  --radius: 12px;
  --green: #166534; --green-bg: #dcfce7;
  --yellow: #854d0e; --yellow-bg: #fef9c3;
  --red: #991b1b; --red-bg: #fee2e2;
}
```

Key CSS features:
- Sticky top nav (`position: sticky; top: 0; z-index: 100`)
- `.container` max-width 880px centered
- `.card` for content blocks (white bg, shadow, rounded corners)
- `.badge` for status labels (green/yellow/red variants)
- Tables: bordered, alternating row colors, RTL aligned
- `.flowchart` monospace block (direction: ltr for ASCII art)
- `@media print` — hide nav, remove shadows, force white bg

**Section contents — all in Hebrew except where noted:**

**Section 1 — מבוא:**
- One paragraph: the project studies explore/exploit behavior in Wikipedia browsing
- One paragraph: experimental setup — manipulation before Wikipedia task, between-subjects design
- One paragraph: why methodology matters — need to define DV before pre-registration, advisors disagree on approach

**Section 2 — מפת מדדים:**
Table with columns: #, שם מדד, מה מודד, סוג, ציר, סטטוס

Rows (key metrics only — 10 rows):

| # | שם | מה מודד | סוג | ציר | סטטוס |
|---|-----|---------|-----|------|-------|
| M2 | סף 60 שניות | דף מעל 60s = Exploit | בינארי | זמן | ✅ |
| M4 | Type vs Paste | הבחנה בין כתיבה להדבקה | קטגוריאלי | כתיבה | ✅ |
| M15 | Median פר נבדק | דף מעל Median אישי = Exploit | בינארי | זמן | ✅ |
| M16 | LSA סמנטי | מרחק cosine בין דפים (TF-IDF) | רציף | סמנטיקה | ✅ |
| M18 | כתיבה בינארית | יש/אין כתיבה בדף | בינארי | כתיבה | ✅ |
| M20 | LSA + Type/Paste | שילוב: סמנטיקה AND כתיבה | משולב | סמנטיקה+כתיבה | ✅ |
| M26 | Topic Modeling | LDA — מרחק נושאי (JSD) | רציף | סמנטיקה | ✅ |
| M27 | LSA + Median זמן | שילוב: סמנטיקה AND זמן-median | משולב | סמנטיקה+זמן | ✅ |
| M28 | LSA + 60s | שילוב: סמנטיקה AND 60s | משולב | סמנטיקה+זמן | ✅ |
| M29 | PCA גולמי | PCA על 3 אותות רציפים | רציף | PCA | ✅ |

**Section 3 — הבעיה:**
Three `.card` blocks, each with heading + badge + bullet list:

Card 1: "שילוב שרירותי (M27, M28)" — badge-red "בעייתי"
- AND לוגי — Exploit רק אם שני המדדים מסכימים
- אין הצדקה למשקל שווה בין הצירים
- הסף שרירותי (למה 60 שניות? למה Median?)
- איבוד מידע: רציף → בינארי → שילוב

Card 2: "PCA על סיווגים בינאריים" — badge-red "בעייתי"
- מעגליות — M27/M28 בנויים מ-M2/M15/M16/M18, מתואמים בהכרח
- PCA מניח משתנים רציפים ונורמליים, לא 0/1
- לא עונה על שאלת construct validity

Card 3: "PCA על אותות גולמיים — M29" — badge-yellow "לגיטימי, אבל..."
- תוצאה: PC1=55.7% (engagement), PC2=33.3% (semantic breadth)
- שני מימדים משמעותיים, לא אחד
- הציר הסמנטי חלקית בלתי תלוי מציר ה-engagement
- **זה לא כישלון — זה ממצא!** עקבי עם Zhou et al. 2024

**Section 4 — השוואת גישות:**
Table with columns: גישה, יתרון, חסרון, מתאימה כש..., המלצה

6 rows as defined in spec. Row "PCA — 2 DVs" highlighted green. Use badge classes for recommendation column.

**Section 5 — מסגרת קבלת החלטות:**
ASCII flowchart in `.flowchart` div (direction: ltr):

```
PC1 explains >70% variance?
  │
  ├── YES → Use PC1 alone as DV
  │         (unified metric is legitimate)
  │
  └── NO  → Are PC1 + PC2 both significant?
              │
              ├── YES → MANOVA with 2 DVs     ◄── אתה כאן
              │         (PC1=55.7%, PC2=33.3%)
              │
              └── NO  → Consider Factor Analysis
                        or single representative metric
```

Below the flowchart, a `.card` with additional criteria:
- מתאם גבוה + בסיס תיאורטי → שילוב אולי עובד
- CFA — לבדוק אם פקטור אחד מתאים
- N קטן מדי ל-MANOVA → PC1 לבד כניתוח משני

**Section 6 — ההמלצה:**
Green-highlighted `.card` with:

Numbered analysis plan (5 steps):
1. חלץ 3 אותות גולמיים (זמן, מרחק נושאי JSD, כמות כתיבה)
2. PCA על standardized signals
3. ציון ממוצע PC1 + PC2 פר נבדק
4. MANOVA: condition כ-IV, PC1 + PC2 כ-DVs
5. Follow-up: t-tests נפרדים על כל PC

Then "איך זה פותר את הוויכוח" with 3 advisor cards:
- כנרת: analysis plan ברור, PC1 ניתן להצגה כ"מדד engagement"
- יועד: אין שילוב שרירותי, PCA מגדיר משקלות מנתונים, עקבי עם Zhou 2024
- יובל: PC1 ממפה ל-explore/exploit dynamics, PC2 מוסיף ממד סמנטי

**Section 7 — טיוטת מייל:**
A `.card` styled as an email. In Hebrew. Concise and diplomatic. Structure:

- **נושא:** עדכון מתודולוגי — מדד Explore/Exploit למטלת הוויקיפדיה
- **פתיחה:** שלום כנרת, יובל ויועד (1 line)
- **רקע:** המחשבה המקורית — מדד אחיד משילוב (2-3 lines)
- **מה נבנה:** M27, M28 (שילובים), M29 PCA על גולמיים (2 lines)
- **הבעיה:** שילוב שרירותי + PCA מראה 2 מימדים לא אחד (3 lines)
- **ממצאי PCA:** PC1=55.7% engagement, PC2=33.3% semantic breadth (2 lines)
- **המלצה:** MANOVA עם PC1+PC2 כ-DVs, condition כ-IV (3 lines with bullets)
- **הנימוק:** עקבי עם Zhou et al. 2024, מבוסס-נתונים, פותר את חוסר ההסכמה (2 lines)
- **צעדים הבאים:** הגשת pre-registration עם analysis plan זה (1 line)
- **סיום:** אשמח לשמוע את דעתכם (1 line)

Total email: ~20 lines. Minimal, clear, no academic fluff.

- [ ] **Step 2: Verify the file opens correctly in browser**

Open `docs/methodology_review.html` in browser:
- Verify RTL layout renders correctly
- Verify nav links scroll to correct sections
- Verify tables render with proper alignment and colors
- Verify flowchart displays correctly (LTR inside RTL page)
- Verify print preview looks clean (Ctrl+P)

- [ ] **Step 3: Commit**

```bash
git add docs/methodology_review.html
git commit -m "docs: add methodology review page with recommendation and email draft"
```

---

### Task 2: Add .superpowers to .gitignore

**Files:**
- Modify: `.gitignore`

- [ ] **Step 1: Add .superpowers directory to .gitignore**

Append to `.gitignore`:

```
# Brainstorming session files
.superpowers/
```

- [ ] **Step 2: Commit**

```bash
git add .gitignore
git commit -m "chore: add .superpowers to gitignore"
```
