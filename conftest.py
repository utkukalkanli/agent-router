import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for p in (ROOT / "scripts", ROOT / "scripts" / "routers"):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
