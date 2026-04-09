import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from sentence_transformers import SentenceTransformer
import umap
from sklearn.cluster import KMeans, DBSCAN
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

# ============================================================
# 1. Load data
# ============================================================
print("Loading wiki_texts.json ...")
with open(DATA_DIR / 'wiki_texts.json', encoding='utf-8') as f:
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
print(f"  Embeddings shape: {embeddings.shape}")

# ============================================================
# 3. UMAP reduction (5D for clustering)
# ============================================================
print("\nRunning UMAP (5D) ...")
reducer = umap.UMAP(n_components=5, n_neighbors=15, random_state=42)
reduced = reducer.fit_transform(embeddings)
print(f"  Reduced shape: {reduced.shape}")

# UMAP 2D for visualization
print("Running UMAP (2D) for visualization ...")
reducer_2d = umap.UMAP(n_components=2, n_neighbors=15, random_state=42)
coords_2d = reducer_2d.fit_transform(embeddings)

# ============================================================
# Experiment 1: K-means + Silhouette
# ============================================================
print("\n" + "="*60)
print("EXPERIMENT 1: K-means + Silhouette")
print("="*60)

K_RANGE = range(2, 21)
sil_scores = {}

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(reduced)
    score = silhouette_score(reduced, labels)
    sil_scores[k] = score
    print(f"  k={k:2d}  silhouette={score:.4f}")

optimal_k = max(sil_scores, key=sil_scores.get)
optimal_sil = sil_scores[optimal_k]
print(f"\n  Optimal k = {optimal_k}  (silhouette = {optimal_sil:.4f})")

km_best = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = km_best.fit_predict(reduced)

# ============================================================
# Experiment 2: DBSCAN with varying eps
# ============================================================
print("\n" + "="*60)
print("EXPERIMENT 2: DBSCAN with varying eps")
print("="*60)

EPS_VALUES = [0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
dbscan_results = []

for eps in EPS_VALUES:
    db = DBSCAN(eps=eps, min_samples=3)
    labels = db.fit_predict(reduced)
    n_outliers = int(np.sum(labels == -1))
    unique_clusters = set(labels) - {-1}
    n_clusters = len(unique_clusters)

    if n_clusters >= 2 and (labels != -1).sum() > n_clusters:
        mask = labels != -1
        sil = silhouette_score(reduced[mask], labels[mask])
    else:
        sil = float('nan')

    dbscan_results.append({
        'eps': eps,
        'n_clusters': n_clusters,
        'n_outliers': n_outliers,
        'silhouette': sil,
        'labels': labels.tolist(),
    })
    print(f"  eps={eps:.2f}  clusters={n_clusters:2d}  outliers={n_outliers:3d}  silhouette={sil:.4f}")

# Best DBSCAN: most clusters with valid silhouette, fewest outliers
valid_dbscan = [r for r in dbscan_results if not np.isnan(r['silhouette'])]
if valid_dbscan:
    best_dbscan = max(valid_dbscan, key=lambda r: (r['silhouette']))
else:
    best_dbscan = dbscan_results[-1]

print(f"\n  Best DBSCAN: eps={best_dbscan['eps']}  clusters={best_dbscan['n_clusters']}  "
      f"outliers={best_dbscan['n_outliers']}  silhouette={best_dbscan['silhouette']:.4f}")

# ============================================================
# Experiment 3: Spectral Embedding + K-means
# ============================================================
print("\n" + "="*60)
print("EXPERIMENT 3: Spectral Embedding + K-means")
print("="*60)

print(f"  Applying SpectralEmbedding (n_components=10) ...")
spec = SpectralEmbedding(n_components=10, random_state=42)
spectral_reduced = spec.fit_transform(reduced)

km_spectral = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
spectral_labels = km_spectral.fit_predict(spectral_reduced)
spectral_sil = silhouette_score(spectral_reduced, spectral_labels)
print(f"  k={optimal_k}  silhouette={spectral_sil:.4f}  (vs K-means alone: {optimal_sil:.4f})")

# ============================================================
# Summary table
# ============================================================
print("\n" + "="*60)
print("SUMMARY COMPARISON")
print("="*60)
print(f"  {'Method':<30} {'N clusters':>10} {'Silhouette':>12} {'Outliers':>10}")
print(f"  {'-'*64}")
print(f"  {'K-means (optimal k)':<30} {optimal_k:>10} {optimal_sil:>12.4f} {'0':>10}")
print(f"  {'DBSCAN (best eps)':<30} {best_dbscan['n_clusters']:>10} {best_dbscan['silhouette']:>12.4f} {best_dbscan['n_outliers']:>10}")
print(f"  {'Spectral + K-means':<30} {optimal_k:>10} {spectral_sil:>12.4f} {'0':>10}")

# ============================================================
# Save bertopic_kmeans.json
# ============================================================
topic_assignments = {slug: int(label) for slug, label in zip(slugs, kmeans_labels)}
output_data = {
    "method": "kmeans",
    "n_clusters": int(optimal_k),
    "silhouette_score": float(optimal_sil),
    "slugs": sorted(slugs),
    "topic_assignments": topic_assignments,
}
out_path = DATA_DIR / 'bertopic_kmeans.json'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output_data, f, indent=2)
print(f"\nSaved: {out_path}")

# ============================================================
# Visualization
# ============================================================
print("\nGenerating visualization ...")

plt.rcParams.update({
    'figure.facecolor': BG_COLOR,
    'axes.facecolor': BG_COLOR,
    'text.color': TEXT_COLOR,
    'axes.labelcolor': LABEL_COLOR,
    'xtick.color': LABEL_COLOR,
    'ytick.color': LABEL_COLOR,
    'axes.edgecolor': BORDER_COLOR,
    'grid.color': GRID_COLOR,
    'font.family': 'sans-serif',
})

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.patch.set_facecolor(BG_COLOR)
fig.suptitle('BERTopic Clustering Experiments', color=TEXT_COLOR, fontsize=15, fontweight='bold', y=1.01)

# --- Panel 1: Silhouette vs k ---
ax1 = axes[0]
ks = list(sil_scores.keys())
scores = list(sil_scores.values())
ax1.plot(ks, scores, color='#58a6ff', linewidth=2, marker='o', markersize=4)
ax1.plot(optimal_k, optimal_sil, 'o', color='#f85149', markersize=10, zorder=5,
         label=f'Optimal k={optimal_k} ({optimal_sil:.3f})')
ax1.set_xlabel('Number of clusters (k)', color=LABEL_COLOR)
ax1.set_ylabel('Silhouette score', color=LABEL_COLOR)
ax1.set_title('K-means: Silhouette vs k', color=TEXT_COLOR, fontsize=12)
ax1.grid(True, color=GRID_COLOR, linestyle='--', alpha=0.5)
ax1.legend(facecolor=BG_COLOR, edgecolor=BORDER_COLOR, labelcolor=TEXT_COLOR, fontsize=9)
ax1.set_xticks(range(2, 21, 2))

# --- Panel 2: UMAP 2D colored by K-means clusters ---
ax2 = axes[1]
cmap = plt.cm.get_cmap('tab20', optimal_k)
scatter = ax2.scatter(
    coords_2d[:, 0], coords_2d[:, 1],
    c=kmeans_labels, cmap='tab20', vmin=0, vmax=optimal_k - 1,
    s=40, alpha=0.85, edgecolors='none'
)
ax2.set_xlabel('UMAP 1', color=LABEL_COLOR)
ax2.set_ylabel('UMAP 2', color=LABEL_COLOR)
ax2.set_title(f'UMAP 2D - K-means (k={optimal_k})', color=TEXT_COLOR, fontsize=12)
plt.colorbar(scatter, ax=ax2, label='Cluster', fraction=0.046, pad=0.04)

# --- Panel 3: Comparison table ---
ax3 = axes[2]
ax3.axis('off')
ax3.set_facecolor(BG_COLOR)

table_data = [
    ['Method', 'N clusters', 'Silhouette', 'Outliers'],
    ['K-means', str(optimal_k), f'{optimal_sil:.4f}', '0'],
    ['DBSCAN', str(best_dbscan['n_clusters']), f"{best_dbscan['silhouette']:.4f}", str(best_dbscan['n_outliers'])],
    ['Spectral+KM', str(optimal_k), f'{spectral_sil:.4f}', '0'],
]

col_widths = [0.32, 0.22, 0.25, 0.21]
row_height = 0.14
x_starts = [0.02]
for w in col_widths[:-1]:
    x_starts.append(x_starts[-1] + w)

for row_idx, row in enumerate(table_data):
    y = 0.82 - row_idx * row_height
    is_header = row_idx == 0

    # Row background
    bg = BG_COLOR if not is_header else GRID_COLOR
    rect = mpatches.FancyBboxPatch(
        (0.01, y - 0.06), 0.98, row_height - 0.01,
        boxstyle='round,pad=0.005',
        facecolor=bg, edgecolor=BORDER_COLOR, linewidth=0.5,
        transform=ax3.transAxes
    )
    ax3.add_patch(rect)

    for col_idx, (cell, x) in enumerate(zip(row, x_starts)):
        color = TEXT_COLOR if is_header else LABEL_COLOR
        weight = 'bold' if is_header else 'normal'
        ha = 'left' if col_idx == 0 else 'center'
        xpos = x + 0.01 if col_idx == 0 else x + col_widths[col_idx] / 2
        ax3.text(xpos, y - 0.01, cell, transform=ax3.transAxes,
                 color=color, fontsize=10, fontweight=weight,
                 ha=ha, va='center')

ax3.set_title('Method Comparison', color=TEXT_COLOR, fontsize=12, pad=10)

plt.tight_layout()
viz_path = OUTPUT_DIR / 'bertopic_clustering_experiments.png'
plt.savefig(viz_path, dpi=150, bbox_inches='tight', facecolor=BG_COLOR)
plt.close()
print(f"Saved: {viz_path}")
print("\nDone.")
