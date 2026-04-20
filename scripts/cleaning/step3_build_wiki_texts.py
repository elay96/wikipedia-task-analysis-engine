#!/usr/bin/env python3
"""Stage 3: build slug-keyed wiki_texts.json from articles.jsonl.

For each article_slug we pick the content from the revid with the latest
(timestamp, revid) - this keeps wiki_texts.json in sync with whatever the
most recent participant saw for that slug.

Usage:
    py scripts/cleaning/step3_build_wiki_texts.py
    py scripts/cleaning/step3_build_wiki_texts.py --jsonl X --out Y
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_JSONL = PROJECT_ROOT / "data" / "cleaned" / "articles.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "data" / "cleaned" / "wiki_texts.json"


def build_wiki_texts(jsonl_path: Path) -> dict:
    slug_to_records: dict[str, list] = defaultdict(list)
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            slug = rec.get("article_slug")
            content = rec.get("content") or ""
            if not slug or not content.strip():
                continue
            slug_to_records[slug].append(rec)

    wiki_texts = {}
    for slug, recs in slug_to_records.items():
        recs.sort(key=lambda r: (r.get("timestamp") or "", int(r.get("revid") or 0)))
        wiki_texts[slug] = recs[-1]["content"]
    return wiki_texts


def _main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Build wiki_texts.json from articles.jsonl.")
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    if not args.jsonl.exists():
        print(f"[step3] ERROR: {args.jsonl} does not exist", file=sys.stderr)
        return 1

    wiki_texts = build_wiki_texts(args.jsonl)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        json.dump(wiki_texts, f, ensure_ascii=False)

    lens = [len(v) for v in wiki_texts.values()]
    if lens:
        median = sorted(lens)[len(lens) // 2]
        print(f"[step3] wrote {len(wiki_texts)} slugs -> {args.out}")
        print(f"[step3] content chars: min={min(lens)} median={median} max={max(lens)}")
    else:
        print(f"[step3] WARNING: 0 slugs written", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(_main())
