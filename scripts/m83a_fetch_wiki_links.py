#!/usr/bin/env python3
"""
M83a: Build the static hyperlink subgraph for Diffuse-visited articles.
========================================================================
For each ArticleSlug that any Diffuse participant opened, fetch Wikipedia's
main-namespace outlinks (paginated), keep only edges to slugs already in
the target set, and save the union as an undirected binary adjacency.

Inputs:
  data/cleaned_new/Game.csv

Output:
  output/m83_wiki_link_graph.json   (final)
  output/m83_wiki_link_graph_partial.jsonl  (resume-on-rerun checkpoint)

Sanity stats printed at end:
  degree mean/median/max, fraction with degree 0, n_failed.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "cleaning"))

from m83_utils import collect_diffuse_slugs, title_to_slug                    # noqa: E402
from api_client import build_session, fetch_outlinks                          # noqa: E402

DATA_DIR = SCRIPT_DIR.parent / "data" / "cleaned_new"
OUTPUT_DIR = SCRIPT_DIR.parent / "output"
GAME_CSV = DATA_DIR / "Game.csv"
PARTIAL_OUT = OUTPUT_DIR / "m83_wiki_link_graph_partial.jsonl"
FINAL_OUT = OUTPUT_DIR / "m83_wiki_link_graph.json"


def load_partial() -> dict:
    """Return {slug: {'links': [...], 'status': ...}} from any prior run."""
    if not PARTIAL_OUT.exists():
        return {}
    out = {}
    with PARTIAL_OUT.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            out[row["slug"]] = row
    return out


def append_partial(row: dict) -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    with PARTIAL_OUT.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"loading {GAME_CSV}...")
    df = pd.read_csv(GAME_CSV, low_memory=False)
    target = sorted(collect_diffuse_slugs(df))
    print(f"  Diffuse-visited slugs: {len(target)}")

    cache = load_partial()
    print(f"  cached from prior run: {len(cache)}")

    session = build_session()
    n_done = 0
    n_failed = 0
    for i, slug in enumerate(target, 1):
        if slug in cache:
            continue
        if not slug:
            continue
        title = slug.replace("_", " ")
        result = fetch_outlinks(session, title)
        row = {"slug": slug, "status": result["status"],
               "links": result["links"], "error": result["error"]}
        append_partial(row)
        cache[slug] = row
        if result["status"] == "ok":
            n_done += 1
        else:
            n_failed += 1
        if i % 25 == 0 or i == len(target):
            print(f"  {i}/{len(target)} fetched | ok-this-run={n_done} | fail-this-run={n_failed}")

    target_set = set(target)
    edges = set()
    for slug, row in cache.items():
        if slug not in target_set:
            continue
        if row["status"] != "ok":
            continue
        for link_title in row["links"]:
            tgt = title_to_slug(link_title)
            if tgt and tgt in target_set and tgt != slug:
                a, b = sorted([slug, tgt])
                edges.add((a, b))

    slug_to_idx = {s: i for i, s in enumerate(target)}
    edge_pairs = [[slug_to_idx[a], slug_to_idx[b]] for a, b in sorted(edges)]
    payload = {
        "slugs": target,
        "n_articles": len(target),
        "edges": edge_pairs,
        "n_total_edges": len(edge_pairs),
        "scope": "diffuse_visited_only",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "n_failed_articles": sum(1 for r in cache.values() if r["status"] != "ok"),
    }
    with FINAL_OUT.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"wrote {FINAL_OUT.name}")

    degree = [0] * len(target)
    for a, b in edges:
        degree[slug_to_idx[a]] += 1
        degree[slug_to_idx[b]] += 1
    degree_arr = sorted(degree)
    if degree_arr:
        med = degree_arr[len(degree_arr) // 2]
        mean = sum(degree_arr) / len(degree_arr)
        mx = max(degree_arr)
        zero_frac = sum(1 for d in degree_arr if d == 0) / len(degree_arr)
        print(f"  degree: mean={mean:.1f} median={med} max={mx} zero_frac={zero_frac:.2f}")
        print(f"  total undirected edges: {len(edges)}")
        if len(edges) < 300 or len(edges) > 30000:
            print(f"  WARNING: edge count {len(edges)} is outside the expected 300-30000 band.")


if __name__ == "__main__":
    main()
