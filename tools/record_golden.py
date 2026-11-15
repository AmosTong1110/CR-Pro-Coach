"""Record (or check) the golden engine hashes.

    python tools/record_golden.py            # compare with tests/golden/engine_hashes.json, print speed
    python tools/record_golden.py --write    # re-record after an intentional gameplay change
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend_engine.simulator.card_db import CardDB  # noqa: E402
from backend_engine.simulator.golden import scenarios  # noqa: E402

OUT = ROOT / "tests" / "golden" / "engine_hashes.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    db = CardDB.load()
    old = json.loads(OUT.read_text()) if OUT.exists() else {}
    new, changed = {}, []
    t_all = time.perf_counter()
    for name, fn in scenarios(db).items():
        t = time.perf_counter()
        new[name] = fn()
        dt = time.perf_counter() - t
        status = "same" if old.get(name) == new[name] else ("NEW" if name not in old else "CHANGED")
        if status != "same":
            changed.append(name)
        print(f"{name:<18} {len(new[name]):>3} samples  {dt:6.2f}s  {status}")
    print(f"total {time.perf_counter() - t_all:.1f}s; changed: {changed or 'none'}")
    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(new, indent=1) + "\n")
        print(f"wrote {OUT}")
    elif changed:
        sys.exit(1)


if __name__ == "__main__":
    main()
