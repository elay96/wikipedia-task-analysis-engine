# M83 - Zhou-style curiosity scores (Hunter/Busybody/Dancer) within DIFFUSE cohort

**Date:** 2026-05-12
**Author:** Elay (designed with Claude)
**Status:** Draft - awaiting user approval

## 1. Goal

Compute continuous Zhou-style curiosity scores (busybody-hunter score + forward flow) for participants in the DIFFUSE spatial-task condition, and test whether either score correlates with the 30 spatial-search features from M81. The aim is to find which (if any) spatial-foraging behaviours covary with naturalistic Wikipedia browsing style.

## 2. Background

Zhou et al. (2024, *Science Advances*) characterise knowledge networks built by Wikipedia readers along two near-orthogonal axes:

- **Busybody-Hunter (BH) score** - aggregated from 4 network metrics computed on the induced subgraph of visited articles in Wikipedia's hyperlink graph: number of edges, clustering coefficient, global efficiency, characteristic path length. High BH = Hunter (tight, connected); low BH = Busybody (loose, dispersed).
- **Forward Flow (FF)** - mean cosine *semantic* distance from each visited article to all previously-visited articles, using pre-trained fastText word embeddings. High FF = Dancer-like creative leaps between disparate topics.

In the paper these are **continuous scores**, not discrete classifications. We follow that convention.

The existing M80 pipeline approximates Hunter/Busybody using a simpler LDA-topic-concentration metric and KMeans (k=2). M83 supersedes that approach with the network-based and semantic-distance metrics directly from Zhou (2024) and stays on the continuous scale.

## 3. Scope

- **Cohort:** participants in `Condition == 'diffuse'` (Wikipedia Game.csv) who have `>= 5` unique article visits AND a row in `output/m81_spatial_features.csv`. Expected N ~63-67.
- **Wikipedia data:** 459-article curated corpus already in the project. Static hyperlink graph and fastText embeddings will be fetched only for articles that any Diffuse participant visited (~300-400).
- **Spatial DVs:** all 30 features in `output/m81_spatial_features.csv` (rewards, path length, coverage entropy, transitions, exploit-time, Levy alpha, FPT, etc.).
- **Out of scope:** Clumpy participants, discrete Hunter/Busybody/Dancer classification, M80's LDA features, individual-trial-level analysis.

## 4. Architecture

Three scripts, one analysis output set:

```
scripts/
  m83a_fetch_wiki_links.py    # one-time asset builder (Wikipedia REST API)
  m83b_compute_fasttext.py    # one-time asset builder (fastText embeddings)
  m83_zhou_diffuse_styles.py  # main analysis

output/
  m83_wiki_link_graph.json        # static hyperlink adjacency (asset)
  m83_article_embeddings.npz      # fastText article embeddings (asset)
  m83_per_participant.csv         # BH-score, FF, 4 metrics, per-pid
  m83_spearman_correlations.csv   # 60 rows: axis x feature, r, p, p_FDR
  m83_zhou_diffuse_report.pdf     # 6-page graphical report

docs/
  m83_findings.html               # Hebrew RTL light-mode report
```

## 5. Component specs

### 5.1 `m83a_fetch_wiki_links.py` - static hyperlink graph

**Purpose:** binary undirected adjacency between all articles that any Diffuse participant visited, reflecting Wikipedia's actual hyperlink structure.

**Steps:**

1. Load `data/cleaned_new/Game.csv`, filter to `Condition == 'diffuse'`, collect `target_slugs = unique(ArticleSlug)` from `article_open` events.
2. For each slug, call Wikipedia REST API:
   `GET https://en.wikipedia.org/w/api.php?action=query&prop=links&titles={title}&pllimit=max&plnamespace=0&format=json`
   Use the stored `revid` to fetch the snapshot consistent with `articles.jsonl` (where supported, otherwise use current as fallback and log the divergence).
3. Resolve Wikipedia redirects so that link targets map back to slugs in `target_slugs`. If a redirect target is not in `target_slugs`, drop the link (we only care about edges within our corpus).
4. Build binary undirected adjacency: `edge[A,B] = 1 if A has a link to B OR B has a link to A`.
5. Save `output/m83_wiki_link_graph.json` with `{slugs, edges (i<j pairs), scope: "diffuse_visited_only", fetched_at, n_total_edges}`.
6. Print sanity stats: degree mean/median/max, fraction of nodes with degree 0, count of articles whose API call failed.

**Rate limiting / robustness:**
- `time.sleep(0.5)` between API calls (~5 minutes for ~400 articles).
- Retry x3 on transient failures.
- Resumable: write partial progress to `output/m83_wiki_link_graph_partial.jsonl`; on rerun, skip slugs already there.
- If final fetch fails for a slug, the slug stays in the graph but as an isolated node (incoming edges from others still count).

**Sanity bounds:** expect 1k-15k undirected edges for ~400 academic-topic Wikipedia articles. Outside that range, print warning.

### 5.2 `m83b_compute_fasttext.py` - article embeddings

**Purpose:** 300-d vector per article, used to compute pairwise cosine distance for Forward Flow.

**Steps:**

1. Load fastText pre-trained vectors. Prefer a local pre-downloaded `cc.en.300.vec.gz` (path configurable, e.g. `~/.cache/fasttext/cc.en.300.vec.gz`). If absent, fall back to `gensim.downloader.load('fasttext-wiki-news-subwords-300')` (~1GB, downloaded once and cached by gensim).
2. Load `data/cleaned_new/articles.jsonl`; filter to articles in `target_slugs` from 5.1.
3. For each article:
   - Tokenize content using the same regex/stopword rule as `scripts/compute_similarity.py` (`[a-z]{2,}`, English stopwords).
   - For each token in fastText vocab, retrieve its vector. Skip OOV tokens.
   - `embedding = mean of vectors`. If < 5 in-vocab tokens, warn and store zero vector (will produce `D=0` against any vector; handled in M83 main as flagged).
   - L2-normalize.
4. Save `output/m83_article_embeddings.npz` with `slugs`, `embeddings` (n x 300), `oov_rate_per_article`.

**Resource notes:** fastText model is large (~1GB resident). Loaded once, used to compute ~400 embeddings, then discarded. The output `.npz` is small (~1MB) and is committed to repo. fastText model itself is gitignored.

### 5.3 `m83_zhou_diffuse_styles.py` - main analysis

**Step 5.3.1: Build cohort**

```python
df = load Game.csv
diffuse = df[df.Condition == 'diffuse']
opens = diffuse[diffuse.Action == 'article_open'].sort_values(['ID', 'Time'])
visits_by_pid = opens.groupby('ID').ArticleSlug.apply(list)  # ordered, with revisits
# filter: >= 5 unique articles AND in m81 spatial features
m81 = pd.read_csv('output/m81_spatial_features.csv')
m81_diffuse_pids = set(m81[m81.condition.str.lower() == 'diffuse'].participant_id)
cohort_pids = [pid for pid in visits_by_pid.index
               if len(set(visits_by_pid[pid])) >= 5 and pid in m81_diffuse_pids]
```

Expected `len(cohort_pids) ~ 63-67`.

**Step 5.3.2: Per-participant network metrics**

For each `pid in cohort_pids`:

```python
visited = list(set(visits_by_pid[pid]))       # unique articles
A_full = load m83_wiki_link_graph             # sparse adjacency
A_sub  = A_full[visited, visited]              # induced subgraph
G = nx.Graph(A_sub)

n_unique = len(visited)
n_edges  = G.number_of_edges()
density  = nx.density(G)                       # informational; not in BH-score
clustering = nx.average_clustering(G)
global_eff = nx.global_efficiency(G)

# characteristic path length: largest connected component
ccs = sorted(nx.connected_components(G), key=len, reverse=True)
lcc = G.subgraph(ccs[0]).copy()
lcc_fraction = len(lcc) / n_unique
char_path = nx.average_shortest_path_length(lcc) if len(lcc) >= 2 else 0.0
```

**Step 5.3.3: Per-participant Forward Flow**

```python
ordered = visits_by_pid[pid]                   # full ordered sequence, incl. revisits
emb = load m83_article_embeddings
vectors = [emb[slug] for slug in ordered if slug in emb]   # drop any slug not in embedding asset (warn)
N = len(vectors)
if N < 2:
    forward_flow = NaN
else:
    per_position_FF = []
    for i in range(1, N):                       # i = 1..N-1 (0-indexed); position 2..N (1-indexed)
        prev_vectors = vectors[:i]               # all previous
        dists = [1 - cosine_sim(vectors[i], v) for v in prev_vectors]
        per_position_FF.append(mean(dists))
    forward_flow = mean(per_position_FF)
```

This matches the Forward Flow definition (Gray 2019, used by Zhou): average across positions i >= 2 of the mean cosine distance from page i to all previous pages.

**Step 5.3.4: Aggregate to DataFrame**

Columns: `participant_id, condition, n_visits, n_unique_articles, n_edges, density, clustering, char_path_length, global_efficiency, lcc_fraction, forward_flow`.

**Step 5.3.5: Compute BH score (within Diffuse cohort)**

```python
z_edges = zscore(df.n_edges)
z_clust = zscore(df.clustering)
z_eff   = zscore(df.global_efficiency)
z_path  = zscore(df.char_path_length)
df['BH_score'] = z_edges + z_clust + z_eff - z_path  # Zhou eq, hunter-positive
```

Z-scoring is within the Diffuse cohort (not against any external baseline) - the score is relative position among Diffuse peers.

**Step 5.3.6: Spearman correlations + FDR**

```python
spatial = pd.read_csv('output/m81_spatial_features.csv')
spatial_cols = [c for c in spatial.columns if c not in ('participant_id', 'condition')]
merged = df.merge(spatial[['participant_id'] + spatial_cols], on='participant_id', how='inner')

rows = []
for axis in ['BH_score', 'forward_flow']:
    for feat in spatial_cols:
        rho, p = scipy.stats.spearmanr(merged[axis], merged[feat], nan_policy='omit')
        rows.append({'axis': axis, 'feature': feat, 'rho': rho, 'p': p, 'n': merged[[axis, feat]].dropna().shape[0]})

res = pd.DataFrame(rows)
res['p_FDR'] = fdr_bh(res['p'], alpha=0.05)
res['fdr_significant'] = res['p_FDR'] < 0.05
res.to_csv('output/m83_spearman_correlations.csv', index=False)
```

FDR-BH is applied across **all 60** tests (2 axes x 30 features) jointly.

### 5.4 PDF report (6 pages)

1. **Cover + cohort summary** - N, age/gender if available, distribution histograms of the 4 raw metrics + FF.
2. **BH-score scatter** - `n_edges` vs `clustering`, points coloured by BH-score (continuous gradient).
3. **2D space** - `BH_score` (x) vs `forward_flow` (y); mirrors Zhou Fig 8A inset.
4. **Spearman table** - all 60 correlations sorted by `|rho|`, FDR-significant rows highlighted `#E8F5E9`.
5. **Top scatters** - 6 strongest correlations as xy-scatter + linear-regression line + `rho`/`p_FDR` annotation.
6. **Diagnostics** - `n_visits` vs network size; flag participants with `lcc_fraction < 0.5` or `n_unique < 8`.

### 5.5 Hebrew HTML findings (`docs/m83_findings.html`)

RTL, light-mode, styled to match `docs/m82_findings.html`. Sections:

1. Title + 3-4 sentence abstract.
2. Methodology - Zhou framework, 4 metrics, FF formula, what was operationalised.
3. Cohort - N, descriptives.
4. Hunter/Busybody axis - correlation table, top 5 findings.
5. Dancer axis (FF) - same.
6. Summary + interpretation - which spatial-foraging patterns covary with which Wikipedia-style axis.
7. Caveats - small N, multiple comparisons, fastText/TF-IDF alternative, choice of static hyperlink scope.

## 6. Edge cases and decisions

| Issue | Decision | Rationale |
|---|---|---|
| Disconnected subgraph for char_path_length | Use largest connected component (LCC). Record `lcc_fraction`. | Paper's formula assumes connectivity; LCC is NetworkX convention. |
| Revisits in Forward Flow | Include all `article_open` events in chronological order; same-slug pairs yield D=0. | Paper: "sequence of pages a reader has browsed" - includes revisits. |
| `n_unique < 5` for a participant | Exclude from cohort. | Matches M80's filter; network metrics meaningless for tiny graphs. |
| OOV-only article (zero embedding) | Include in graph computation; warn for FF. D against zero vector is 1.0 - mark as flagged in CSV. | Avoids hiding articles silently. |
| API failure for an article | Keep slug as isolated node; log. | Doesn't break analysis; downstream metrics handle isolates. |
| Z-scoring across or within cohort | Within Diffuse cohort. | Question is about within-condition variation; no external reference. |
| Multiple comparison correction | FDR-BH across all 60 tests jointly. | Standard for exploratory correlation panels; not over-conservative. |

## 7. Data flow

```
data/cleaned_new/Game.csv ----+
                              |
                              v
                     m83a_fetch_wiki_links.py
                              |
                              v
                     output/m83_wiki_link_graph.json --+
                                                       |
data/cleaned_new/articles.jsonl --+                    |
                                  |                    |
                                  v                    |
                       m83b_compute_fasttext.py        |
                                  |                    |
                                  v                    |
                       output/m83_article_embeddings.npz
                                  |                    |
                                  +--------+-----------+
                                           |
output/m81_spatial_features.csv -----------+
                                           |
                                           v
                            m83_zhou_diffuse_styles.py
                                           |
                                           v
              +---------------+------------+-------------------+
              |               |            |                   |
              v               v            v                   v
        per_participant   spearman_     report.pdf      docs/m83_findings.html
            .csv          corrs.csv
```

## 8. Validation

- Sanity check that BH-score distribution within Diffuse cohort spans both signs and has no extreme outliers (>4 SD).
- Sanity check that Forward Flow is bounded in `[0, 2]` (cosine distance range for L2-normalized vectors).
- Cross-check vs. M80: BH-score and M80's `topic_concentration` should correlate positively (since both proxy Hunter-likeness), though not perfectly. Print Spearman in the PDF diagnostics page.
- Cross-check vs. Zhou Fig 2 marginal distributions: our `clustering` and `global_efficiency` histograms should roughly resemble theirs (right-skewed for clustering, left-skewed for global efficiency).

## 9. Out-of-spec / future work

- Extending the analysis to Clumpy or to all participants would require re-running `m83a` with a wider `target_slugs` set; the rest of the pipeline is condition-agnostic.
- The choice of fastText over TF-IDF for Forward Flow is debatable for small academic corpora; a robustness check using TF-IDF can be added later by swapping the embedding source.
- Discrete classification (Hunter/Busybody/Dancer labels) is intentionally NOT done; could be added later via tertile splits or kmeans on (BH_score, forward_flow).

## 10. Acceptance criteria

The analysis is considered complete when:

- `output/m83_per_participant.csv` exists with one row per cohort participant and all listed columns populated (no all-NaN columns).
- `output/m83_spearman_correlations.csv` has 60 rows, all with valid `rho` and `p_FDR`.
- `output/m83_zhou_diffuse_report.pdf` opens with all 6 pages rendered.
- `docs/m83_findings.html` opens in a browser, displays RTL Hebrew text correctly, light mode, and references the actual N and any FDR-significant findings.
- All scripts can be re-run idempotently (assets cache on disk).
