#!/usr/bin/env python3
"""
Compute BERTopic model for Wikipedia articles.
===============================================
Reads wiki_texts.json, trains BERTopic, outputs bertopic_model.json
with article_title -> topic_id mapping.
"""

import json
import numpy as np
from pathlib import Path
from bertopic import BERTopic

DATA_DIR = Path(__file__).parent / '..' / 'data'
CLEANED_DIR = DATA_DIR / 'cleaned'
INPUT_PATH = CLEANED_DIR / 'wiki_texts.json'
OUTPUT_PATH = CLEANED_DIR / 'bertopic_model.json'


def main():
    print("[compute_bertopic] Loading wiki texts...")
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        wiki = json.load(f)

    slugs = sorted(wiki.keys())
    texts = [wiki[s] for s in slugs]
    print(f"  {len(slugs)} articles")

    print("  Training BERTopic...")
    model = BERTopic(verbose=True)
    topics, probs = model.fit_transform(texts)

    topic_info = model.get_topic_info()
    n_topics = len(topic_info) - 1  # exclude -1 (outlier topic)
    print(f"  Found {n_topics} topics (+ outlier topic -1)")

    topic_assignments = {}
    for slug, topic_id in zip(slugs, topics):
        topic_assignments[slug] = int(topic_id)

    topic_words = {}
    for topic_id in sorted(set(topics)):
        if topic_id == -1:
            topic_words[str(topic_id)] = ["outlier"]
            continue
        words = model.get_topic(topic_id)
        topic_words[str(topic_id)] = [w for w, _ in words[:15]]

    result = {
        "n_topics": n_topics,
        "slugs": slugs,
        "topic_assignments": topic_assignments,
        "topic_words": topic_words,
    }

    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"  Saved: {OUTPUT_PATH}")

    topic_counts = {}
    for t in topics:
        topic_counts[t] = topic_counts.get(t, 0) + 1
    print("\n  Topic distribution:")
    for t in sorted(topic_counts.keys()):
        print(f"    Topic {t}: {topic_counts[t]} articles")


if __name__ == '__main__':
    main()
