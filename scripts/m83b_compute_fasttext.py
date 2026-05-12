#!/usr/bin/env python3
"""
M83b: fastText article embeddings for Diffuse-visited articles.
================================================================
Loads pre-trained fastText vectors (local .vec.gz preferred; gensim
downloader fallback), tokenises each article (same regex as
compute_similarity.py), averages in-vocab word vectors, L2-normalises,
and saves an .npz keyed by slug.

Inputs:
  data/cleaned_new/articles.jsonl
  output/m83_wiki_link_graph.json     (to get the slug list)

Output:
  output/m83_article_embeddings.npz   keys: 'slugs', 'embeddings', 'oov_rate'
"""
from __future__ import annotations

import gzip
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

DATA_DIR = SCRIPT_DIR.parent / "data" / "cleaned_new"
OUTPUT_DIR = SCRIPT_DIR.parent / "output"
ARTICLES_JSONL = DATA_DIR / "articles.jsonl"
LINK_GRAPH = OUTPUT_DIR / "m83_wiki_link_graph.json"
EMB_OUT = OUTPUT_DIR / "m83_article_embeddings.npz"

LOCAL_FASTTEXT = Path(os.path.expanduser("~/.cache/fasttext/cc.en.300.vec.gz"))
GENSIM_MODEL = "fasttext-wiki-news-subwords-300"

STOP_WORDS = frozenset(
    "a an the and or but in on at to for of is it this that with as by from "
    "are was were be been being have has had do does did will would shall should "
    "may might can could not no nor so if then than also about up its he she they "
    "we you i me him her us them my our your his their which who whom what when "
    "where how all each every both few more most other some such only own same "
    "into over after before between through during above below out off again "
    "further once here there these those am just very too any much many".split()
)

TOKEN_RE = re.compile(r"[a-z]{2,}")


def tokenize(text: str) -> list:
    return [w for w in TOKEN_RE.findall(text.lower())
            if w not in STOP_WORDS and len(w) >= 2]


def load_fasttext():
    """Return an object with `.has_index_for(word)` / `.get_vector(word)`,
    or a dict-of-word-to-ndarray loaded from the local .vec.gz fallback.

    Tries local file first (faster); falls back to gensim.downloader.
    """
    if LOCAL_FASTTEXT.exists():
        print(f"  loading local fastText: {LOCAL_FASTTEXT}")
        vectors: dict = {}
        with gzip.open(LOCAL_FASTTEXT, "rt", encoding="utf-8") as f:
            header = f.readline().split()
            n_words, dim = int(header[0]), int(header[1])
            print(f"  vocab {n_words}, dim {dim}")
            for line in f:
                parts = line.rstrip().split(" ")
                word = parts[0]
                vec = np.asarray(parts[1:], dtype=np.float32)
                if vec.shape[0] == dim:
                    vectors[word] = vec
        return {"kind": "dict", "data": vectors, "dim": dim}
    else:
        print(f"  downloading via gensim: {GENSIM_MODEL}")
        import gensim.downloader as gd
        kv = gd.load(GENSIM_MODEL)
        return {"kind": "kv", "data": kv, "dim": int(kv.vector_size)}


def get_vector(ft, word: str):
    if ft["kind"] == "dict":
        return ft["data"].get(word)
    kv = ft["data"]
    if word in kv:
        return np.asarray(kv[word], dtype=np.float32)
    return None


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"loading {LINK_GRAPH}...")
    graph = json.loads(LINK_GRAPH.read_text(encoding="utf-8"))
    target_slugs = set(graph["slugs"])
    print(f"  target slugs: {len(target_slugs)}")

    print(f"loading {ARTICLES_JSONL}...")
    articles_by_slug: dict = {}
    with ARTICLES_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            slug = row.get("article_slug")
            if slug in target_slugs:
                articles_by_slug[slug] = row.get("content", "")
    print(f"  matched articles: {len(articles_by_slug)} / {len(target_slugs)}")

    ft = load_fasttext()
    dim = ft["dim"]

    slugs_out: list = []
    embs: list = []
    oov_rates: list = []

    for slug in sorted(target_slugs):
        content = articles_by_slug.get(slug, "")
        tokens = tokenize(content)
        if not tokens:
            slugs_out.append(slug)
            embs.append(np.zeros(dim, dtype=np.float32))
            oov_rates.append(1.0)
            continue
        vecs = []
        oov = 0
        for t in tokens:
            v = get_vector(ft, t)
            if v is None:
                oov += 1
            else:
                vecs.append(v)
        if not vecs:
            slugs_out.append(slug)
            embs.append(np.zeros(dim, dtype=np.float32))
            oov_rates.append(1.0)
            continue
        mean = np.mean(np.stack(vecs), axis=0)
        norm = np.linalg.norm(mean)
        if norm > 0:
            mean = mean / norm
        slugs_out.append(slug)
        embs.append(mean.astype(np.float32))
        oov_rates.append(oov / len(tokens))

    E = np.stack(embs)
    np.savez_compressed(EMB_OUT,
                        slugs=np.asarray(slugs_out),
                        embeddings=E,
                        oov_rate=np.asarray(oov_rates, dtype=np.float32))
    print(f"wrote {EMB_OUT.name}: {E.shape}")
    print(f"  mean OOV rate: {np.mean(oov_rates):.3f} (median {np.median(oov_rates):.3f})")
    n_zero = int(np.sum(np.linalg.norm(E, axis=1) == 0))
    print(f"  zero-vector articles (likely empty content): {n_zero}")


if __name__ == "__main__":
    main()
