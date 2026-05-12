"""M83 pure-logic helpers: slug<->title, network metrics, forward flow,
BH score, FDR-BH. Imported by m83a, m83b, and m83 main scripts. Fully
unit-tested in scripts/cleaning/test_m83_utils.py.
"""
from __future__ import annotations

from typing import Sequence
import numpy as np


def title_to_slug(title: str) -> str:
    """Wikipedia title (with spaces, possibly anchor) -> URL slug."""
    if not title:
        return ""
    head = title.split("#", 1)[0]
    return head.strip().replace(" ", "_")


def slug_to_title(slug: str) -> str:
    """URL slug -> human title (underscores -> spaces)."""
    return slug.replace("_", " ")
