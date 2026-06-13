"""S2: for each visit resolve the revision live at visit time, then fetch and
cache extract + outlinks + categories once per unique revid.

Resumable: revids already cached are skipped; visit_revisions.csv is rewritten
fully each run from visits.csv + the resolver."""
import json
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd

import reconstruct
from api_client import (build_session, fetch_extract_by_revid, fetch_outlinks,
                        fetch_revision_at)

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
CACHE = HERE.parent / "cache" / "wiki"
VISITS = DATA / "visits.csv"
VISIT_REV_OUT = DATA / "visit_revisions.csv"


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    visits = pd.read_csv(VISITS)
    # Skip navigational pages for reconstruction.
    topical = visits[~visits["is_main_page"] & ~visits["is_disambiguation"]].copy()

    session = build_session()

    # Resolve revid per (slug, visit timestamp).
    rows = []
    for _, v in topical.iterrows():
        slug = v["article"]
        iso = reconstruct.epoch_ms_to_iso(v["start_time_ms"])
        if not slug or iso is None:
            rows.append({"visit_id": v["visit_id"], "participant_id": v["participant_id"],
                         "article": slug, "start_iso": iso,
                         "revid": None, "rev_status": "no_timestamp"})
            continue
        res = fetch_revision_at(session, slug, iso)
        rows.append({"visit_id": v["visit_id"], "participant_id": v["participant_id"],
                     "article": slug, "start_iso": iso,
                     "revid": res["revid"], "rev_status": res["status"]})
    visit_rev = pd.DataFrame(rows)
    visit_rev.to_csv(VISIT_REV_OUT, index=False)

    # Fetch content once per unique resolved revid.
    revids = sorted({int(r) for r in visit_rev["revid"].dropna().unique()})
    slug_by_revid = (visit_rev.dropna(subset=["revid"])
                     .drop_duplicates("revid").set_index("revid")["article"].to_dict())
    n_new = n_fail = 0
    for revid in revids:
        cp = reconstruct.cache_path(CACHE, revid)
        if cp.exists():
            continue
        ex = fetch_extract_by_revid(session, revid)
        slug = slug_by_revid.get(revid, "")
        links = fetch_outlinks(session, slug) if slug else {"links": []}
        cats = reconstruct.fetch_categories(session, slug) if slug else []
        payload = {"revid": revid, "slug": slug,
                   "extract": ex.get("content") or "", "extract_status": ex["status"],
                   "outlinks": links.get("links", []), "categories": cats}
        cp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        if ex["status"] == "ok":
            n_new += 1
        else:
            n_fail += 1

    ok = (visit_rev["rev_status"] == "ok").sum()
    print(f"visits resolved: {len(visit_rev)} | revision ok: {ok} | "
          f"not_found/err: {len(visit_rev) - ok}")
    print(f"unique revids: {len(revids)} | newly fetched: {n_new} | extract failures: {n_fail}")


if __name__ == "__main__":
    main()
