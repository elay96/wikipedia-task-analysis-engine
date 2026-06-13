# Yoed EDA - Wikipedia Browsing x Creativity

EDA of Prof. Yoed Kenett's N=107 Wikipedia free-browsing dataset.
Reconstructs the historical revision each participant saw, derives semantic +
structural browsing features, correlates them with creativity/cognition
measures (Spearman + BH-FDR).

## Run order
1. `python yoed_eda/scripts/s1_parse_flatten.py`     # CSV -> participants/visits/searches
2. `python yoed_eda/scripts/s2_reconstruct_pages.py`  # fetch historical revisions (live API)
3. `python yoed_eda/scripts/s3_semantic_features.py`  # embeddings + per-participant features
4. `python yoed_eda/scripts/s4_creativity_behavior.py`# Spearman + BH-FDR + figures
5. `python yoed_eda/scripts/s5_build_html.py`         # findings HTML

## Tests
`python -m pytest yoed_eda/scripts/test_*.py -v`

## Data-quality notes
- `duration_ms` / `end_time` are empty in the source; dwell is derived from
  consecutive `start_time` within a session (last page uses `ended_at`).
- `Main_Page` and disambiguation pages are flagged and excluded from semantic features.
- Articles with no historical revision at visit time are logged and skipped.
