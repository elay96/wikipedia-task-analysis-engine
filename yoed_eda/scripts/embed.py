"""fastText embeddings for reconstructed article extracts.

Loads pre-trained fastText vectors (local cc.en.300.vec.gz preferred, gensim
fallback), tokenises each extract, averages in-vocab vectors, L2-normalises.
Mirrors scripts/m83b_compute_fasttext.py so embeddings are comparable."""
from __future__ import annotations

import gzip
import os
import re
from pathlib import Path

import numpy as np

LOCAL_FASTTEXT = Path(os.path.expanduser("~/.cache/fasttext/cc.en.300.vec.gz"))
GENSIM_MODEL = "fasttext-wiki-news-subwords-300"
TOKEN_RE = re.compile(r"[a-z]{2,}")
STOP_WORDS = frozenset(
    "a an the and or but in on at to for of is it this that with as by from "
    "are was were be been being have has had do does did will would shall should "
    "may might can could not no nor so if then than also about up its he she they "
    "we you i me him her us them my our your his their which who whom what when "
    "where how all each every both few more most other some such only own same "
    "into over after before between through during above below out off again "
    "further once here there these those am just very too any much many".split()
)


def tokenize(text: str) -> list:
    return [t for t in TOKEN_RE.findall((text or "").lower()) if t not in STOP_WORDS]


def load_vectors(limit: int | None = None) -> dict:
    """Return {word: np.ndarray}. Prefers local .vec.gz, else gensim download."""
    if LOCAL_FASTTEXT.exists():
        vecs = {}
        with gzip.open(LOCAL_FASTTEXT, "rt", encoding="utf-8") as f:
            next(f)  # header line "n d"
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                parts = line.rstrip().split(" ")
                vecs[parts[0]] = np.asarray(parts[1:], dtype=np.float32)
        return vecs
    import gensim.downloader as api
    kv = api.load(GENSIM_MODEL)
    return {w: kv[w] for w in kv.index_to_key}


def embed_text(text: str, vectors: dict, dim: int = 300):
    """Mean of in-vocab token vectors, L2-normalised. None if no tokens hit."""
    toks = tokenize(text)
    mats = [vectors[t] for t in toks if t in vectors]
    if not mats:
        return None
    v = np.mean(np.stack(mats), axis=0)
    n = np.linalg.norm(v)
    return (v / n) if n > 0 else v
