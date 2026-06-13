"""Put the repo's scripts/ and scripts/cleaning/ on sys.path so yoed_eda
modules can import the shared api_client and m83_utils helpers."""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts"
CLEANING = SCRIPTS / "cleaning"
for p in (SCRIPTS, CLEANING):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
