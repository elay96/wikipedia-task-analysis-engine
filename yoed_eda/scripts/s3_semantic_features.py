"""S3: build per-participant feature table from visits + cached pages."""
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd

import embed
import features as feat
from m83_utils import bh_score, network_metrics, title_to_slug

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
CACHE = HERE.parent / "cache" / "wiki"
OUT = HERE.parent / "output" / "participant_features.csv"


def load_pages() -> dict:
    """slug -> {extract, outlinks(slugs), categories} (first revid per slug)."""
    pages = {}
    for fp in CACHE.glob("*.json"):
        p = json.loads(fp.read_text(encoding="utf-8"))
        slug = p.get("slug")
        if slug and slug not in pages:
            pages[slug] = {
                "extract": p.get("extract", ""),
                "outlinks": [title_to_slug(t) for t in p.get("outlinks", [])],
                "categories": p.get("categories", []),
            }
    return pages


def main() -> None:
    visits = pd.read_csv(DATA / "visits.csv")
    searches = pd.read_csv(DATA / "searches.csv")
    pages = load_pages()

    # Authorized optimization: build vectors dict from corpus tokens only,
    # instead of materializing the full ~1M-word gensim vocab into a Python dict.
    import gensim.downloader as api
    print("loading fastText (gensim cached model)...")
    kv = api.load(embed.GENSIM_MODEL)
    corpus_tokens = set()
    for p in pages.values():
        corpus_tokens.update(embed.tokenize(p["extract"]))
    vectors = {t: kv[t] for t in corpus_tokens if t in kv}
    print(f"corpus vocab: {len(corpus_tokens)} tokens, {len(vectors)} in model")

    emb_cache = {s: embed.embed_text(p["extract"], vectors) for s, p in pages.items()}

    rows = []
    for pid, g in visits.sort_values(["participant_id", "order_in_session"]).groupby("participant_id"):
        topical = g[~g["is_main_page"] & ~g["is_disambiguation"]]
        slugs = [s for s in topical["article"].tolist() if s in pages]
        seq_vecs = [emb_cache[s] for s in slugs if emb_cache.get(s) is not None]
        n_searches = int((searches["participant_id"] == pid).sum())

        visit_dicts = topical[["article", "dwell_ms"]].to_dict("records")
        cats = {c for s in set(slugs) for c in pages[s]["categories"]}
        G = feat.visited_subgraph(slugs, {s: pages[s]["outlinks"] for s in set(slugs)})

        row = {"participant_id": pid}
        row.update(feat.structural_features(visit_dicts, n_searches))
        row.update(feat.semantic_features(seq_vecs))
        row.update(network_metrics(G))
        row["path_breadth"] = len(cats)
        rows.append(row)

    df = pd.DataFrame(rows)
    # Hunter/Busybody score across participants (within-cohort z-score; m80/m83 style).
    df["bh_score"] = bh_score(df)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"participant_features: {df.shape} -> {OUT}")


if __name__ == "__main__":
    main()
