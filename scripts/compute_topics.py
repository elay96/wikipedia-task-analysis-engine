#!/usr/bin/env python3
"""
Compute LDA topic model for Wikipedia articles.
================================================
Reads wiki_texts.json, trains LDA, computes pairwise Jensen-Shannon
divergence, outputs topic_model.json.
"""

import json
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

DATA_DIR = Path(__file__).parent / '..' / 'data'
CLEANED_DIR = DATA_DIR / 'cleaned'
INPUT_PATH = CLEANED_DIR / 'wiki_texts.json'
OUTPUT_PATH = CLEANED_DIR / 'topic_model.json'

N_TOPICS = 10
N_TOP_WORDS = 15
RANDOM_STATE = 42


def jsd(p, q):
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log2(p / m + 1e-12))
    kl_qm = np.sum(q * np.log2(q / m + 1e-12))
    return float(np.sqrt(0.5 * (kl_pm + kl_qm)))


def main():
    print("[compute_topics] Loading wiki texts...")
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        wiki = json.load(f)

    slugs = sorted(wiki.keys())
    texts = [wiki[s] for s in slugs]
    print(f"  {len(slugs)} articles")

    print("  Vectorizing (CountVectorizer)...")
    vectorizer = CountVectorizer(
        max_df=0.95,
        min_df=2,
        stop_words='english',
        max_features=5000,
    )
    doc_term_matrix = vectorizer.fit_transform(texts)
    print(f"  Doc-term matrix: {doc_term_matrix.shape}")

    print("  Training LDA...")
    lda = LatentDirichletAllocation(
        n_components=N_TOPICS,
        random_state=RANDOM_STATE,
        max_iter=50,
        learning_method='batch',
    )
    lda.fit(doc_term_matrix)

    print("  Extracting topic distributions...")
    topic_dist_matrix = lda.transform(doc_term_matrix)  # shape: (n_docs, n_topics)

    feature_names = vectorizer.get_feature_names_out()
    topic_words = {}
    for topic_idx, component in enumerate(lda.components_):
        top_indices = component.argsort()[:-N_TOP_WORDS - 1:-1]
        topic_words[str(topic_idx)] = [feature_names[i] for i in top_indices]

    topic_distributions = {
        slug: topic_dist_matrix[i].tolist()
        for i, slug in enumerate(slugs)
    }

    print("  Computing pairwise Jensen-Shannon divergences...")
    distances = {}
    n = len(slugs)
    for i in range(n):
        for j in range(i + 1, n):
            d = jsd(topic_dist_matrix[i], topic_dist_matrix[j])
            key = f"{slugs[i]}|||{slugs[j]}"
            distances[key] = round(d, 6)
        if (i + 1) % 20 == 0:
            print(f"    {i + 1}/{n} rows done")

    print(f"  {len(distances)} pairs computed")

    vals = list(distances.values())
    vals_sorted = sorted(vals)
    mean = sum(vals) / len(vals)
    median = vals_sorted[len(vals_sorted) // 2]

    print(f"  Mean JSD: {mean:.4f}")
    print(f"  Median JSD: {median:.4f}")
    print(f"  Range: [{vals_sorted[0]:.4f}, {vals_sorted[-1]:.4f}]")

    result = {
        "n_topics": N_TOPICS,
        "slugs": slugs,
        "topic_words": topic_words,
        "topic_distributions": topic_distributions,
        "distances": distances,
        "stats": {
            "mean": round(mean, 6),
            "median": round(median, 6),
            "std": round(float(np.std(vals)), 6),
            "min": round(vals_sorted[0], 6),
            "max": round(vals_sorted[-1], 6),
            "n_pairs": len(distances),
        },
    }

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"  Saved: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
