"""Compare two adaptive_kv benchmark JSON outputs (token parity + metric deltas)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def _index_runs(path: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: dict[tuple[str, str, int], dict[str, Any]] = {}
    for r in data.get("runs", []):
        key = (r["scenario"], r["baseline"], r.get("repeat", 0))
        out[key] = r
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Diff two benchmark JSON files")
    p.add_argument("left", type=Path)
    p.add_argument("right", type=Path)
    ns = p.parse_args()
    L = _index_runs(ns.left)
    R = _index_runs(ns.right)
    keys = sorted(set(L) | set(R))
    mismatches = 0
    for k in keys:
        a, b = L.get(k), R.get(k)
        if a is None or b is None:
            print(f"missing run {k}: left={a is not None} right={b is not None}")
            mismatches += 1
            continue
        ta, tb = a.get("token_ids"), b.get("token_ids")
        if ta != tb:
            print(f"token_ids differ {k}: len {len(ta or [])} vs {len(tb or [])}")
            mismatches += 1
        ca = (a.get("metrics") or {}).get("counters") or {}
        cb = (b.get("metrics") or {}).get("counters") or {}
        keys_c = sorted(set(ca) | set(cb))
        for name in keys_c:
            va, vb = ca.get(name, 0), cb.get(name, 0)
            if va != vb:
                print(f"counter {name} {k}: {va} -> {vb}")
    if mismatches:
        sys.exit(1)
    print("compare: no token_id mismatches for shared keys")


if __name__ == "__main__":
    main()
