#!/usr/bin/env python3
"""
M11: PCA & Cosine Distance Analysis of the Semantic Knowledge Space
=====================================================================
Publication-quality multi-panel figure for senior researcher review.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from helpers import load_trials, OUTPUT_DIR

DATA_DIR = Path(__file__).parent / '..' / 'data'
SIM_PATH = DATA_DIR / 'similarity_matrix.json'

N_CLUSTERS = 10
CLUSTER_NAMES = [
    'Culture & Global.', 'Biodiversity', 'Brain & Cognition',
    'Race & Discrim.', 'Brain Biology', 'Evolution Core',
    'Evo. Mechanisms', 'Income & Ineq.', 'Art History', 'Art Theory',
]
CLUSTER_COLORS = [
    '#AED581', '#4FC3F7', '#CE93D8', '#EF5350',
    '#81D4FA', '#FFB74D', '#A1887F', '#F48FB1',
    '#FFD54F', '#FF8A65',
]


def main():
    with open(SIM_PATH, 'r', encoding='utf-8') as f:
        sim_data = json.load(f)
    slugs = sim_data['slugs']
    similarities = sim_data['similarities']

    n = len(slugs)
    slug_idx = {s: i for i, s in enumerate(slugs)}
    mat = np.zeros((n, n))
    for key, val in similarities.items():
        a, b = key.split('|||')
        i, j = slug_idx[a], slug_idx[b]
        mat[i, j] = val
        mat[j, i] = val
    np.fill_diagonal(mat, 1.0)

    # Cosine distance
    dist = 1.0 - mat

    # PCA
    X = mat.copy()
    X_centered = X - X.mean(axis=0)
    cov = np.cov(X_centered, rowvar=True)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx_sort = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx_sort]
    eigenvectors = eigenvectors[:, idx_sort]

    total_var = eigenvalues.sum()
    var_explained = eigenvalues / total_var
    cumulative = np.cumsum(var_explained)

    # Project onto PC1, PC2
    pc_scores = X_centered @ eigenvectors[:, :2]

    # K-means on PC scores for cluster coloring
    rng = np.random.RandomState(42)
    centers = pc_scores[rng.choice(n, N_CLUSTERS, replace=False)].copy()
    labels = np.zeros(n, dtype=int)
    for _ in range(100):
        dists_k = np.linalg.norm(pc_scores[:, None] - centers[None, :], axis=2)
        new_labels = np.argmin(dists_k, axis=1)
        if np.all(new_labels == labels):
            break
        labels = new_labels
        for j in range(N_CLUSTERS):
            m = labels == j
            if m.sum() > 0:
                centers[j] = pc_scores[m].mean(axis=0)

    # Sort articles by cluster for heatmap
    cluster_order = np.argsort(labels)
    sorted_labels = labels[cluster_order]

    # Visit counts
    trials = load_trials()
    visit_count = np.zeros(n)
    for tr in trials:
        for pv in tr['page_visits']:
            idx_v = slug_idx.get(pv['title'])
            if idx_v is not None:
                visit_count[idx_v] += 1

    # Transition pairs
    trans_sims = []
    for tr in trials:
        pvs = tr['page_visits']
        for i in range(1, len(pvs)):
            fi = slug_idx.get(pvs[i - 1]['title'])
            ti = slug_idx.get(pvs[i]['title'])
            if fi is not None and ti is not None:
                trans_sims.append(mat[fi, ti])
    trans_sims = np.array(trans_sims)

    upper_tri = mat[np.triu_indices(n, k=1)]

    # ========== FIGURE ==========
    fig = plt.figure(figsize=(24, 20))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3,
                          height_ratios=[1, 1.3, 1])

    ax_scree = fig.add_subplot(gs[0, 0])
    ax_cumul = fig.add_subplot(gs[0, 1])
    ax_dist_hist = fig.add_subplot(gs[0, 2])
    ax_biplot = fig.add_subplot(gs[1, 0:2])
    ax_heatmap = fig.add_subplot(gs[1, 2])
    ax_within_between = fig.add_subplot(gs[2, 0])
    ax_nav_vs_rand = fig.add_subplot(gs[2, 1])
    ax_eigenspectrum = fig.add_subplot(gs[2, 2])

    fig.suptitle(
        'M11: Dimensionality & Semantic Distance Analysis\n'
        f'{n} Wikipedia articles · tf-idf cosine similarity · {N_CLUSTERS} clusters (k-means on PCA)',
        fontsize=15, fontweight='bold', y=0.98)

    # ===== A: SCREE PLOT =====
    k_show = 30
    bars = ax_scree.bar(range(1, k_show + 1), var_explained[:k_show] * 100,
                        color='#5C6BC0', edgecolor='white', linewidth=0.3, alpha=0.85)
    bars[0].set_color('#283593')
    bars[1].set_color('#283593')

    ax_scree.set_xlabel('Principal Component', fontsize=10)
    ax_scree.set_ylabel('Variance Explained (%)', fontsize=10)
    ax_scree.set_title('A. Scree Plot (first 30 components)', fontsize=11, fontweight='bold')
    ax_scree.set_xlim(0.3, k_show + 0.7)
    ax_scree.spines['top'].set_visible(False)
    ax_scree.spines['right'].set_visible(False)

    ax_scree.annotate(f'PC1: {var_explained[0]*100:.1f}%',
                      xy=(1, var_explained[0]*100), xytext=(5, var_explained[0]*100 + 1),
                      fontsize=8, fontweight='bold', color='#283593',
                      arrowprops=dict(arrowstyle='->', color='#283593', lw=1))
    ax_scree.annotate(f'PC2: {var_explained[1]*100:.1f}%',
                      xy=(2, var_explained[1]*100), xytext=(6, var_explained[1]*100 + 0.5),
                      fontsize=8, fontweight='bold', color='#283593',
                      arrowprops=dict(arrowstyle='->', color='#283593', lw=1))

    # ===== B: CUMULATIVE VARIANCE =====
    ax_cumul.plot(range(1, n + 1), cumulative * 100, color='#283593', linewidth=2)
    ax_cumul.fill_between(range(1, n + 1), cumulative * 100, alpha=0.1, color='#5C6BC0')

    for thresh, style, color in [(50, '--', '#999'), (90, '--', '#E53935'), (95, ':', '#E53935')]:
        k = np.searchsorted(cumulative, thresh / 100) + 1
        ax_cumul.axhline(y=thresh, color=color, linestyle=style, linewidth=0.8, alpha=0.6)
        ax_cumul.axvline(x=k, color=color, linestyle=style, linewidth=0.8, alpha=0.6)
        ax_cumul.plot(k, thresh, 'o', color=color, markersize=6, zorder=5)
        ax_cumul.annotate(f'{thresh}% → {k} PCs',
                          xy=(k, thresh), xytext=(k + 8, thresh - 4),
                          fontsize=8, fontweight='bold', color=color,
                          arrowprops=dict(arrowstyle='->', color=color, lw=1))

    ax_cumul.set_xlabel('Number of Components', fontsize=10)
    ax_cumul.set_ylabel('Cumulative Variance (%)', fontsize=10)
    ax_cumul.set_title('B. Cumulative Variance Explained', fontsize=11, fontweight='bold')
    ax_cumul.set_xlim(1, n)
    ax_cumul.set_ylim(0, 102)
    ax_cumul.spines['top'].set_visible(False)
    ax_cumul.spines['right'].set_visible(False)

    # ===== C: DISTANCE DISTRIBUTION =====
    upper_dist = dist[np.triu_indices(n, k=1)]
    ax_dist_hist.hist(upper_dist, bins=60, color='#78909C', edgecolor='white',
                      linewidth=0.3, alpha=0.85, density=True, label='All pairs')

    trans_dists = 1.0 - trans_sims
    ax_dist_hist.hist(trans_dists, bins=30, color='#E53935', edgecolor='white',
                      linewidth=0.3, alpha=0.6, density=True, label='Navigation pairs')

    ax_dist_hist.axvline(x=upper_dist.mean(), color='#455A64', linestyle='--',
                         linewidth=1.5, label=f'All mean: {upper_dist.mean():.3f}')
    ax_dist_hist.axvline(x=trans_dists.mean(), color='#C62828', linestyle='--',
                         linewidth=1.5, label=f'Nav mean: {trans_dists.mean():.3f}')

    ax_dist_hist.set_xlabel('Cosine Distance (1 − similarity)', fontsize=10)
    ax_dist_hist.set_ylabel('Density', fontsize=10)
    ax_dist_hist.set_title('C. Distance Distribution: All Pairs vs. Navigation', fontsize=11, fontweight='bold')
    ax_dist_hist.legend(fontsize=8, framealpha=0.9)
    ax_dist_hist.spines['top'].set_visible(False)
    ax_dist_hist.spines['right'].set_visible(False)

    # ===== D: PCA BIPLOT =====
    for cl in range(N_CLUSTERS):
        mask = labels == cl
        sizes = 20 + visit_count[mask] * 8
        sizes = np.minimum(sizes, 200)
        ax_biplot.scatter(pc_scores[mask, 0], pc_scores[mask, 1],
                          c=CLUSTER_COLORS[cl], s=sizes,
                          alpha=0.75, edgecolors='white', linewidths=0.4,
                          label=f'{CLUSTER_NAMES[cl]} ({mask.sum()})', zorder=3)

    # Label top visited articles
    top_idx = np.argsort(visit_count)[-12:]
    for idx_node in top_idx:
        if visit_count[idx_node] > 0:
            label = slugs[idx_node].replace('_', ' ')
            if len(label) > 22:
                label = label[:19] + '...'
            ax_biplot.annotate(
                label, (pc_scores[idx_node, 0], pc_scores[idx_node, 1]),
                fontsize=6, color='#222', fontweight='bold',
                textcoords='offset points', xytext=(5, 5),
                bbox=dict(boxstyle='round,pad=0.15', facecolor='white',
                          alpha=0.75, edgecolor='none'))

    ax_biplot.set_xlabel(f'PC1 ({var_explained[0]*100:.1f}% variance)', fontsize=10, fontweight='bold')
    ax_biplot.set_ylabel(f'PC2 ({var_explained[1]*100:.1f}% variance)', fontsize=10, fontweight='bold')
    ax_biplot.set_title(
        f'D. PCA Projection (PC1 × PC2 = {(var_explained[0]+var_explained[1])*100:.1f}% variance)\n'
        f'Node size ∝ visit frequency · Color = k-means cluster',
        fontsize=11, fontweight='bold')
    ax_biplot.axhline(y=0, color='#ccc', linewidth=0.5)
    ax_biplot.axvline(x=0, color='#ccc', linewidth=0.5)
    ax_biplot.grid(True, alpha=0.1)
    ax_biplot.legend(fontsize=6.5, loc='upper right', framealpha=0.9, ncol=2)
    ax_biplot.spines['top'].set_visible(False)
    ax_biplot.spines['right'].set_visible(False)

    # ===== E: COSINE DISTANCE HEATMAP (cluster-sorted) =====
    sorted_dist = dist[np.ix_(cluster_order, cluster_order)]

    cmap = LinearSegmentedColormap.from_list('custom',
        ['#1A237E', '#283593', '#5C6BC0', '#9FA8DA', '#E8EAF6', '#FFF9C4', '#FFD54F', '#FF8F00', '#E65100'])
    im = ax_heatmap.imshow(sorted_dist, cmap=cmap, aspect='equal', interpolation='nearest',
                           vmin=0, vmax=1)

    # Cluster boundaries
    boundaries = []
    prev_cl = sorted_labels[0]
    for i, cl in enumerate(sorted_labels):
        if cl != prev_cl:
            boundaries.append(i)
            prev_cl = cl
    for b in boundaries:
        ax_heatmap.axhline(y=b - 0.5, color='white', linewidth=0.8, alpha=0.8)
        ax_heatmap.axvline(x=b - 0.5, color='white', linewidth=0.8, alpha=0.8)

    # Cluster tick labels
    cluster_mids = []
    cluster_tick_labels = []
    start = 0
    for i, b in enumerate(boundaries + [n]):
        mid = (start + b) / 2
        cl = sorted_labels[start]
        cluster_mids.append(mid)
        cluster_tick_labels.append(CLUSTER_NAMES[cl])
        start = b

    ax_heatmap.set_xticks(cluster_mids)
    ax_heatmap.set_xticklabels(cluster_tick_labels, fontsize=5.5, rotation=55, ha='right')
    ax_heatmap.set_yticks(cluster_mids)
    ax_heatmap.set_yticklabels(cluster_tick_labels, fontsize=5.5)
    ax_heatmap.set_title('E. Cosine Distance Matrix\n(sorted by cluster)', fontsize=11, fontweight='bold')

    cbar = plt.colorbar(im, ax=ax_heatmap, shrink=0.8, pad=0.02)
    cbar.set_label('Cosine Distance', fontsize=9)

    # ===== F: WITHIN vs BETWEEN CLUSTER DISTANCES =====
    within_dists = []
    between_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                within_dists.append(dist[i, j])
            else:
                between_dists.append(dist[i, j])
    within_dists = np.array(within_dists)
    between_dists = np.array(between_dists)

    bp = ax_within_between.boxplot(
        [within_dists, between_dists, trans_dists],
        labels=['Within\nCluster', 'Between\nClusters', 'Navigation\nPairs'],
        patch_artist=True, widths=0.5,
        medianprops=dict(color='#C62828', linewidth=2),
        flierprops=dict(markersize=2, alpha=0.3))

    box_colors = ['#A5D6A7', '#EF9A9A', '#90CAF9']
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax_within_between.set_ylabel('Cosine Distance', fontsize=10)
    ax_within_between.set_title('F. Distance by Relationship Type', fontsize=11, fontweight='bold')
    ax_within_between.spines['top'].set_visible(False)
    ax_within_between.spines['right'].set_visible(False)

    # Stats annotation
    stats_text = (
        f'Within:  M={within_dists.mean():.3f}, SD={within_dists.std():.3f}\n'
        f'Between: M={between_dists.mean():.3f}, SD={between_dists.std():.3f}\n'
        f'Navig.:  M={trans_dists.mean():.3f}, SD={trans_dists.std():.3f}'
    )
    ax_within_between.text(0.97, 0.97, stats_text, transform=ax_within_between.transAxes,
                           fontsize=7.5, va='top', ha='right', family='monospace',
                           bbox=dict(boxstyle='round,pad=0.4', facecolor='#F5F5F5',
                                     alpha=0.9, edgecolor='#ccc'))

    # ===== G: NAVIGATION vs RANDOM (effect size) =====
    # Bootstrap CI for navigation mean
    n_boot = 5000
    boot_nav = np.array([np.mean(np.random.choice(trans_sims, len(trans_sims), replace=True))
                         for _ in range(n_boot)])
    boot_rand = np.array([np.mean(np.random.choice(upper_tri, len(trans_sims), replace=True))
                          for _ in range(n_boot)])

    ci_nav = np.percentile(boot_nav, [2.5, 97.5])
    ci_rand = np.percentile(boot_rand, [2.5, 97.5])

    bar_x = [0, 1]
    bar_vals = [upper_tri.mean(), trans_sims.mean()]
    bar_colors = ['#90A4AE', '#42A5F5']
    bars = ax_nav_vs_rand.bar(bar_x, bar_vals, color=bar_colors, edgecolor='white',
                               width=0.45, linewidth=1)

    # Error bars (95% CI)
    ax_nav_vs_rand.errorbar(0, upper_tri.mean(),
                            yerr=[[upper_tri.mean() - ci_rand[0]], [ci_rand[1] - upper_tri.mean()]],
                            fmt='none', color='#333', capsize=6, linewidth=1.5)
    ax_nav_vs_rand.errorbar(1, trans_sims.mean(),
                            yerr=[[trans_sims.mean() - ci_nav[0]], [ci_nav[1] - trans_sims.mean()]],
                            fmt='none', color='#333', capsize=6, linewidth=1.5)

    ax_nav_vs_rand.set_xticks(bar_x)
    ax_nav_vs_rand.set_xticklabels(['Random Pairs\n(baseline)', 'Navigation Pairs\n(observed)'],
                                    fontsize=9, fontweight='bold')
    ax_nav_vs_rand.set_ylabel('Mean Cosine Similarity', fontsize=10)

    ratio = trans_sims.mean() / upper_tri.mean()
    cohens_d = (trans_sims.mean() - upper_tri.mean()) / np.sqrt(
        (upper_tri.std()**2 + trans_sims.std()**2) / 2)

    ax_nav_vs_rand.set_title("G. Navigation Similarity vs. Random Baseline", fontsize=11, fontweight='bold')

    for i, (v, ci) in enumerate(zip(bar_vals, [ci_rand, ci_nav])):
        ax_nav_vs_rand.text(i, v + 0.003, f'{v:.4f}\n[{ci[0]:.4f}, {ci[1]:.4f}]',
                            ha='center', fontsize=8, fontweight='bold', color='#333')

    ax_nav_vs_rand.annotate(
        f"×{ratio:.1f}\nCohen's d = {cohens_d:.2f}",
        xy=(1, trans_sims.mean() * 0.85), xytext=(0.5, trans_sims.mean() * 0.7),
        fontsize=14, fontweight='bold', color='#B71C1C', ha='center',
        arrowprops=dict(arrowstyle='->', color='#B71C1C', lw=2),
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF9C4',
                  alpha=0.95, edgecolor='#B71C1C', linewidth=1.5))

    ax_nav_vs_rand.set_ylim(0, trans_sims.mean() * 1.35)
    ax_nav_vs_rand.spines['top'].set_visible(False)
    ax_nav_vs_rand.spines['right'].set_visible(False)

    # ===== H: EIGENSPECTRUM (log scale) =====
    ax_eigenspectrum.semilogy(range(1, n + 1), eigenvalues, 'o-', color='#283593',
                              markersize=3, linewidth=1, alpha=0.8)

    # Mark the "elbow" — where eigenvalues drop below noise floor
    noise_floor = np.median(eigenvalues[n//2:])
    ax_eigenspectrum.axhline(y=noise_floor, color='#E53935', linestyle='--',
                             linewidth=1, alpha=0.7,
                             label=f'Noise floor: {noise_floor:.4f}')

    k_eff = np.sum(eigenvalues > noise_floor * 2)
    ax_eigenspectrum.axvline(x=k_eff, color='#FF9800', linestyle=':', linewidth=1.5,
                             label=f'Effective dim ≈ {k_eff}')

    ax_eigenspectrum.set_xlabel('Component Index', fontsize=10)
    ax_eigenspectrum.set_ylabel('Eigenvalue (log scale)', fontsize=10)
    ax_eigenspectrum.set_title('H. Eigenspectrum (log scale)\nEffective Dimensionality',
                               fontsize=11, fontweight='bold')
    ax_eigenspectrum.legend(fontsize=8, framealpha=0.9)
    ax_eigenspectrum.spines['top'].set_visible(False)
    ax_eigenspectrum.spines['right'].set_visible(False)
    ax_eigenspectrum.set_xlim(0, n + 1)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outpath = OUTPUT_DIR / 'm11_pca_distance.png'
    plt.savefig(outpath, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Saved: {outpath}')

    # Print summary stats for paper
    print(f'\n=== Summary Statistics ===')
    print(f'Articles: {n}')
    print(f'Similarity pairs: {len(similarities)}')
    print(f'PC1: {var_explained[0]*100:.1f}%, PC2: {var_explained[1]*100:.1f}%')
    print(f'Components for 90%: {np.searchsorted(cumulative, 0.90)+1}')
    print(f'Components for 95%: {np.searchsorted(cumulative, 0.95)+1}')
    print(f'Effective dimensionality: ~{k_eff}')
    print(f'Mean cosine dist (all pairs): {upper_dist.mean():.4f} (SD={upper_dist.std():.4f})')
    print(f'Mean cosine dist (navigation): {trans_dists.mean():.4f} (SD={trans_dists.std():.4f})')
    print(f'Mean cosine dist (within cluster): {within_dists.mean():.4f}')
    print(f'Mean cosine dist (between cluster): {between_dists.mean():.4f}')
    print(f'Navigation/random ratio: {ratio:.2f}x')
    print(f"Cohen's d: {cohens_d:.2f}")
    print(f'Navigation 95% CI: [{ci_nav[0]:.4f}, {ci_nav[1]:.4f}]')


if __name__ == '__main__':
    main()
