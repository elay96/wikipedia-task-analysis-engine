# Yoed EDA - Wikipedia Browsing & Creativity (Design Spec)

**Date:** 2026-06-13
**Owner:** Elay Guez
**Data source:** Prof. Yoed Kenett (secondary advisor) - `wiki_behavior_107_data.csv`
**Working dir:** `yoed_eda/` (new folder under repo root)

## 1. Background & Goal

Yoed Kenett shared a behavioral dataset from a Wikipedia free-browsing experiment (N=107). Each participant
chose a question to explore, then browsed a proxied Wikipedia mirror (`/proxy/wiki/wiki/{Article}`) while page
visits and search queries were logged. The dataset also contains per-participant creativity/cognition measures
(the kind Kenett studies: semantic-network creativity).

**Goal:** Reconstruct the exact Wikipedia pages participants saw (historical revision at visit time), derive
semantic + structural browsing features, and test whether the creativity/cognition measures relate to browsing
behavior - in the style of the existing M-analysis pipeline (Spearman + BH-FDR, semantic distances, topic
concentration, Hunter/Busybody curiosity styles, optional PCA composite DV).

## 2. Dataset (verified)

- **107 participants**, 30 columns, one row per participant.
- **Creativity/cognition measures (13):** Verbal Fluency (Number of Answers, Forward Flow), AUT Broom/Belt
  (Number of Answers, Originality), AQT Pencil/Pillow (Number of Answers, Originality), AQT Complexity Score,
  Curiosity Score, GF Score.
- **Session meta:** session_id, started_at, ended_at, total_pages_visited, total_searches, completion_code.
- **Reflective free-text:** user_question, interest_motivation, perceived_learning, new_questions,
  pre_knowledge, topic_familiarity, discoveries, ai_feedback.
- **Behavior (nested JSON):**
  - `page_visits_rows_json` - 836 total visits. Fields: id, session_id, url, title, entry_method
    (all `navigation`), start_time (epoch ms), end_time (**all NaN**), duration_ms (**all NaN**), previous_page.
  - `search_queries_rows_json` - query, timestamp (epoch ms), results_count (NaN).
- **URLs:** every visit is `/proxy/wiki/wiki/{Article_Title}` -> maps to `https://en.wikipedia.org/wiki/{Article_Title}`.
- **Unique articles:** 453. **Visit date range:** 2025-11-18 to 2025-12-31.

### Data-quality facts to handle & document
- `duration_ms` / `end_time` are entirely empty -> dwell must be derived from consecutive `start_time` within a
  session; the last page per session uses `ended_at` as its end.
- `Main_Page` and disambiguation pages flagged separately (navigational, not topical).
- Some historical revisions may be missing (very new/deleted articles) -> log and skip from semantic features.

## 3. Approach

**Approach A (chosen): extend the existing M-pipeline conventions.** Self-contained scripts under `yoed_eda/`
that reuse the project's established methods - BH-FDR + Spearman (m78/m79), Hunter/Busybody styles (m80),
fasttext embeddings (m83b) - for consistency and comparability with the thesis. A new historical-revision
fetcher is written from scratch (existing m83a fetches the *current* revision, not historical).

## 4. Folder Structure

```
yoed_eda/
├── data/
│   ├── raw/wiki_behavior_107_data.csv      # copy of original
│   ├── participants.csv                      # participant-level (107 rows)
│   ├── visits.csv                            # visit-level (836 rows, + derived dwell)
│   └── searches.csv                          # query-level
├── cache/wiki/                               # one JSON per resolved revid (no refetch)
├── scripts/
│   ├── s1_parse_flatten.py
│   ├── s2_reconstruct_pages.py
│   ├── s3_semantic_features.py
│   ├── s4_creativity_behavior.py
│   └── s5_build_html.py
├── output/
│   ├── figures/*.png
│   └── yoed_eda_findings.html
└── README.md
```

## 5. Pipeline Stages

### s1 - Parse & flatten
- Read CSV; explode the two JSON columns into tidy long tables.
- `visits.csv`: ordered per session by start_time; compute `dwell_ms` = next visit start_time - this start_time;
  last page uses `ended_at`. Add `article` (decoded title), `is_main_page`, `is_disambiguation` flags.
- `searches.csv`: query, timestamp, participant_id, session_id.
- `participants.csv`: the 13 measures + session meta + reflective text fields.

### s2 - Reconstruct pages (historical)
- For each visit, query MediaWiki API:
  `action=query&prop=revisions&titles={article}&rvstart={visit_ts}&rvdir=older&rvlimit=1&rvprop=ids|timestamp|content`
  (+ a parse/extract call for plain text, categories, outgoing links).
- Resolve to the revision live at the visit moment; **dedup by resolved revid** so shared revisions fetch once.
- Cache each revid's content/categories/links as JSON in `cache/wiki/`.
- Polite: descriptive User-Agent, ~1 req/sec, retry with backoff. Log articles with no historical revision.

### s3 - Semantic features (per participant)
Embeddings: fasttext over each page's plain-text extract (m83b style).
- `mean_step_distance`, `var_step_distance` - semantic distance between consecutive pages (exploration breadth).
- `topic_concentration` - overall semantic dispersion of the journey (m80 style).
- `hunter_busybody_score` - Zhou curiosity style over the visited-page graph (m80).
Structural:
- `n_pages`, `n_unique_pages`, `n_searches`, `revisit_rate`, `search_vs_link_ratio`,
  `mean_dwell`, `var_dwell`, `path_breadth` (unique Wikipedia categories touched).

### s4 - Creativity <-> behavior analysis (M style)
- Spearman correlation matrix: 13 creativity/cognition measures x browsing features.
- BH-FDR correction across all comparisons; flag q < .05.
- Scatter plots for significant pairs.
- Optional: PCA composite of behavior features as a single DV (m32/m50 style).

### s5 - HTML findings page
- Follow the `html-findings-design` skill: Heebo + Rubik, RTL Hebrew, light mode, bottom-line box first,
  numbered cards, all PNG figures embedded.
- Document data-quality notes (derived dwell, flagged pages, missing revisions).

## 6. Out of Scope (YAGNI)
- Modern transformer embeddings (kept as a possible later upgrade; not in v1).
- NLP modeling of the reflective free-text beyond descriptive summary.
- Statistical inference beyond correlation + FDR (no regression/SEM) unless requested later.

## 7. Success Criteria
- All 836 visits flattened with derived dwell; 453 articles reconstructed at the correct historical revision
  (or logged as missing).
- A reproducible 5-script pipeline runnable end to end.
- An HTML findings page reporting the creativity<->behavior correlations with FDR, ready to share with Yoed.
