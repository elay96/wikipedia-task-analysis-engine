#!/usr/bin/env python3
"""
Compute tf-idf cosine similarity matrix for Wikipedia articles.
===============================================================
Reads wiki_texts.json, computes tf-idf vectors, outputs similarity_matrix.json.
Uses only stdlib + math — no sklearn.
"""

import json
import math
import re
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).parent / '..' / 'data'
CLEANED_DIR = DATA_DIR / 'cleaned'
INPUT_PATH = CLEANED_DIR / 'wiki_texts.json'
OUTPUT_PATH = CLEANED_DIR / 'similarity_matrix.json'

STOP_WORDS = frozenset(
    "a an the and or but in on at to for of is it this that with as by from "
    "are was were be been being have has had do does did will would shall should "
    "may might can could not no nor so if then than also about up its he she they "
    "we you i me him her us them my our your his their which who whom what when "
    "where how all each every both few more most other some such only own same "
    "into over after before between through during above below out off again "
    "further once here there these those am just very too any much many".split()
)

MIN_WORD_LEN = 2
MIN_DOC_FREQ = 2


def tokenize(text):
    return [w for w in re.findall(r'[a-z]{2,}', text.lower())
            if w not in STOP_WORDS and len(w) >= MIN_WORD_LEN]


def compute_tfidf(docs):
    """Return list of {term: tfidf_weight} dicts, one per doc."""
    n_docs = len(docs)

    # Document frequency
    doc_freq = Counter()
    tf_per_doc = []
    for tokens in docs:
        tf = Counter(tokens)
        tf_per_doc.append(tf)
        doc_freq.update(tf.keys())

    # Filter rare terms
    valid_terms = {t for t, df in doc_freq.items() if df >= MIN_DOC_FREQ}

    # IDF
    idf = {t: math.log(n_docs / df) for t, df in doc_freq.items() if t in valid_terms}

    # TF-IDF vectors
    tfidf_vecs = []
    for tf in tf_per_doc:
        total = sum(tf.values())
        vec = {}
        for term, count in tf.items():
            if term in idf:
                vec[term] = (count / total) * idf[term]
        tfidf_vecs.append(vec)

    return tfidf_vecs


def cosine_sim(v1, v2):
    common = set(v1.keys()) & set(v2.keys())
    if not common:
        return 0.0
    dot = sum(v1[t] * v2[t] for t in common)
    mag1 = math.sqrt(sum(val ** 2 for val in v1.values()))
    mag2 = math.sqrt(sum(val ** 2 for val in v2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def main():
    print("[compute_similarity] Loading wiki texts...")
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        wiki = json.load(f)

    slugs = sorted(wiki.keys())
    print(f"  {len(slugs)} articles")

    print("  Tokenizing...")
    tokenized = [tokenize(wiki[s]) for s in slugs]
    total_tokens = sum(len(t) for t in tokenized)
    print(f"  {total_tokens:,} tokens total")

    print("  Computing tf-idf...")
    tfidf_vecs = compute_tfidf(tokenized)

    print("  Computing pairwise cosine similarities...")
    similarities = {}
    n = len(slugs)
    for i in range(n):
        for j in range(i + 1, n):
            sim = cosine_sim(tfidf_vecs[i], tfidf_vecs[j])
            key = f"{slugs[i]}|||{slugs[j]}"
            similarities[key] = round(sim, 6)
        if (i + 1) % 20 == 0:
            print(f"    {i + 1}/{n} rows done")

    print(f"  {len(similarities)} pairs computed")

    # Stats
    vals = list(similarities.values())
    avg = sum(vals) / len(vals)
    vals_sorted = sorted(vals)
    median = vals_sorted[len(vals_sorted) // 2]
    print(f"  Mean similarity: {avg:.4f}")
    print(f"  Median similarity: {median:.4f}")
    print(f"  Range: [{vals_sorted[0]:.4f}, {vals_sorted[-1]:.4f}]")

    result = {
        "slugs": slugs,
        "similarities": similarities,
        "stats": {
            "mean": round(avg, 6),
            "median": round(median, 6),
            "min": round(vals_sorted[0], 6),
            "max": round(vals_sorted[-1], 6),
            "n_pairs": len(similarities),
        }
    }

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"  Saved: {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
