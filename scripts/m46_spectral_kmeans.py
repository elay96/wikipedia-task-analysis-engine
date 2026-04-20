import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from sentence_transformers import SentenceTransformer
import umap
from sklearn.cluster import KMeans
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import silhouette_score

# --- Paths ---
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / 'data'
OUTPUT_DIR = ROOT / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Dark theme colors ---
BG_COLOR = '#0d1117'
TEXT_COLOR = '#e6edf3'
LABEL_COLOR = '#c9d1d9'
GRID_COLOR = '#21262d'
BORDER_COLOR = '#30363d'
MUTED_COLOR = '#8b949e'

TOPIC_PALETTE = [
    '#4FC3F7', '#81C784', '#FFB74D', '#F06292', '#CE93D8',
    '#80DEEA', '#FFCC80', '#A5D6A7', '#EF9A9A', '#B0BEC5',
    '#FFF176', '#90CAF9', '#C5E1A5', '#FFAB91', '#80CBC4',
]

UMAP_REFERENCE_SILHOUETTE = 0.5809

# ============================================================
# 1. Load data
# ============================================================
print("Loading wiki_texts.json ...")
with open(DATA_DIR / 'cleaned' / 'wiki_texts.json', encoding='utf-8') as f:
    wiki = json.load(f)

slugs = sorted(wiki.keys())
texts = [wiki[s] for s in slugs]
print(f"  {len(slugs)} articles loaded")

# ============================================================
# 2. BERT embeddings
# ============================================================
print("\nEncoding with sentence-transformers (all-MiniLM-L6-v2) ...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(texts, show_progress_bar=True)
print(f"  Embedding shape: {embeddings.shape}")

# ============================================================
# 3. Baseline: raw BERT + K-means (no dim reduction)
# ============================================================
K_RANGE = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
N_COMPONENTS_RANGE = [2, 5, 10, 20, 50]

print("\n--- Baseline: Raw BERT + K-means ---")
baseline_scores = {}
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)
    score = silhouette_score(embeddings, labels)
    baseline_scores[k] = score
    print(f"  k={k:2d}  silhouette={score:.4f}")

best_baseline_k = max(baseline_scores, key=baseline_scores.get)
best_baseline_score = baseline_scores[best_baseline_k]
print(f"\n  Best baseline: k={best_baseline_k}, silhouette={best_baseline_score:.4f}")

# ============================================================
# 4. Grid search: Spectral Embedding + K-means
# ============================================================
print("\n--- Grid search: Spectral Embedding + K-means ---")
grid_scores = {}  # (n_components, k) -> silhouette

t0 = time.time()
for n_comp in N_COMPONENTS_RANGE:
    print(f"\n  n_components={n_comp} - computing spectral embedding ...")
    t_spec = time.time()
    se = SpectralEmbedding(n_components=n_comp, random_state=42, n_neighbors=10)
    reduced = se.fit_transform(embeddings)
    print(f"    Spectral embedding done in {time.time() - t_spec:.1f}s, shape={reduced.shape}")

    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(reduced)
        score = silhouette_score(reduced, labels)
        grid_scores[(n_comp, k)] = score

total_time = time.time() - t0
print(f"\n  Grid search completed in {total_time:.1f}s")

# ============================================================
# 5. Print full grid
# ============================================================
print("\n--- Silhouette score grid (rows=n_components, cols=k) ---")
header = "n_comp\\k  " + "  ".join(f"{k:5d}" for k in K_RANGE)
print(header)
for n_comp in N_COMPONENTS_RANGE:
    row = f"{n_comp:8d}  " + "  ".join(f"{grid_scores[(n_comp, k)]:.3f}" for k in K_RANGE)
    print(row)

# ============================================================
# 6. Find optimal combination
# ============================================================
best_combo = max(grid_scores, key=grid_scores.get)
best_n_comp, best_k = best_combo
best_spectral_score = grid_scores[best_combo]

print(f"\n--- Top 5 best (n_components, k) combinations ---")
sorted_combos = sorted(grid_scores.items(), key=lambda x: x[1], reverse=True)
for (nc, k), score in sorted_combos[:5]:
    print(f"  n_components={nc:2d}, k={k:2d}  silhouette={score:.4f}")

print(f"\n--- Comparison ---")
print(f"  Raw BERT + K-means (best k={best_baseline_k}): silhouette={best_baseline_score:.4f}")
print(f"  Spectral + K-means (n_comp={best_n_comp}, k={best_k}): silhouette={best_spectral_score:.4f}")
print(f"  UMAP + K-means (from M42, reference):          silhouette={UMAP_REFERENCE_SILHOUETTE:.4f}")

# ============================================================
# 7. Best clustering labels for saving + visualization
# ============================================================
print(f"\nRecomputing best spectral clustering (n_components={best_n_comp}, k={best_k}) ...")
se_best = SpectralEmbedding(n_components=best_n_comp, random_state=42, n_neighbors=10)
reduced_best = se_best.fit_transform(embeddings)
km_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
best_labels = km_best.fit_predict(reduced_best)

# ============================================================
# 8. Save results
# ============================================================
out = {
    "method": "spectral_kmeans_direct",
    "n_components": int(best_n_comp),
    "n_clusters": int(best_k),
    "silhouette_score": float(best_spectral_score),
    "slugs": slugs,
    "topic_assignments": {slug: int(best_labels[i]) for i, slug in enumerate(slugs)},
}
out_path = DATA_DIR / 'cleaned' / 'bert_spectral_direct.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(out, f, indent=2)
print(f"Saved to {out_path}")

# ============================================================
# 9. UMAP 2D for visualization only
# ============================================================
print("\nComputing UMAP 2D for visualization (not used in clustering) ...")
reducer = umap.UMAP(n_components=2, n_neighbors=15, random_state=42)
umap_2d = reducer.fit_transform(embeddings)

# ============================================================
# 10. Visualization
# ============================================================
print("\nRendering visualization ...")

# Build score matrix for heatmap
score_matrix = np.array(
    [[grid_scores[(nc, k)] for k in K_RANGE] for nc in N_COMPONENTS_RANGE]
)

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
fig.patch.set_facecolor(BG_COLOR)
for ax in axes:
    ax.set_facecolor(BG_COLOR)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER_COLOR)

# ---- Panel 1: Heatmap ----
ax1 = axes[0]
im = ax1.imshow(score_matrix, cmap='YlOrRd', aspect='auto')

# Annotate cells
vmin, vmax = score_matrix.min(), score_matrix.max()
for ri, nc in enumerate(N_COMPONENTS_RANGE):
    for ci, k in enumerate(K_RANGE):
        val = score_matrix[ri, ci]
        norm_val = (val - vmin) / (vmax - vmin + 1e-9)
        text_col = '#0d1117' if norm_val > 0.5 else TEXT_COLOR
        is_best = (nc == best_n_comp and k == best_k)
        weight = 'bold' if is_best else 'normal'
        label = f"{val:.3f}" + (" *" if is_best else "")
        ax1.text(ci, ri, label, ha='center', va='center',
                 color=text_col, fontsize=7, fontweight=weight)

ax1.set_xticks(range(len(K_RANGE)))
ax1.set_xticklabels([str(k) for k in K_RANGE], color=LABEL_COLOR, fontsize=9)
ax1.set_yticks(range(len(N_COMPONENTS_RANGE)))
ax1.set_yticklabels([str(nc) for nc in N_COMPONENTS_RANGE], color=LABEL_COLOR, fontsize=9)
ax1.set_xlabel("K (number of clusters)", color=LABEL_COLOR, fontsize=10)
ax1.set_ylabel("Spectral embedding dimensions", color=LABEL_COLOR, fontsize=10)
ax1.set_title("Silhouette Score Grid\n(Spectral Embedding + K-means)", color=TEXT_COLOR, fontsize=11)
ax1.tick_params(colors=LABEL_COLOR)
cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cbar.ax.yaxis.set_tick_params(color=LABEL_COLOR)
plt.setp(cbar.ax.yaxis.get_ticklabels(), color=LABEL_COLOR)
cbar.set_label("Silhouette", color=LABEL_COLOR, fontsize=9)

# ---- Panel 2: UMAP 2D scatter ----
ax2 = axes[1]
for cluster_id in range(best_k):
    mask = best_labels == cluster_id
    color = TOPIC_PALETTE[cluster_id % len(TOPIC_PALETTE)]
    ax2.scatter(
        umap_2d[mask, 0], umap_2d[mask, 1],
        c=color, s=40, alpha=0.85, label=f"Cluster {cluster_id}",
        edgecolors='none'
    )
ax2.set_title(
    f"UMAP Visualization of Best Clustering\n"
    f"(n_comp={best_n_comp}, k={best_k}, sil={best_spectral_score:.3f})\n"
    f"Visualization only - clustering done without UMAP",
    color=TEXT_COLOR, fontsize=10
)
ax2.set_xlabel("UMAP-1", color=LABEL_COLOR)
ax2.set_ylabel("UMAP-2", color=LABEL_COLOR)
ax2.tick_params(colors=LABEL_COLOR)
if best_k <= 12:
    legend = ax2.legend(
        fontsize=7, framealpha=0.2, facecolor=BG_COLOR,
        labelcolor=LABEL_COLOR, loc='best', ncol=2
    )

# ---- Panel 3: Comparison bar chart ----
ax3 = axes[2]
methods = [
    f"Raw BERT\n+ K-means\n(k={best_baseline_k})",
    f"Spectral\n+ K-means\n(n={best_n_comp}, k={best_k})",
    "UMAP\n+ K-means\n(M42 ref)",
]
scores = [best_baseline_score, best_spectral_score, UMAP_REFERENCE_SILHOUETTE]
colors = [TOPIC_PALETTE[0], TOPIC_PALETTE[2], TOPIC_PALETTE[3]]

bars = ax3.bar(range(3), scores, color=colors, width=0.5, edgecolor=BORDER_COLOR)
ax3.set_xticks(range(3))
ax3.set_xticklabels(methods, color=LABEL_COLOR, fontsize=9)
ax3.set_ylabel("Silhouette Score", color=LABEL_COLOR)
ax3.set_ylim(0, max(scores) * 1.25)
ax3.tick_params(colors=LABEL_COLOR)
ax3.set_title("Method Comparison\n(Silhouette Score)", color=TEXT_COLOR, fontsize=11)
ax3.yaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
ax3.grid(axis='y', color=GRID_COLOR, linestyle='--', linewidth=0.5)
ax3.set_axisbelow(True)

for bar, score in zip(bars, scores):
    ax3.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.005,
        f"{score:.4f}",
        ha='center', va='bottom', color=TEXT_COLOR, fontsize=10, fontweight='bold'
    )

fig.suptitle(
    "Spectral Embedding + K-means on Raw BERT Embeddings (No UMAP in Clustering)",
    color=TEXT_COLOR, fontsize=13, y=1.01
)
plt.tight_layout()
out_png = OUTPUT_DIR / 'm46_spectral_kmeans.png'
plt.savefig(out_png, dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
print(f"Saved visualization to {out_png}")
print("\nDone.")
