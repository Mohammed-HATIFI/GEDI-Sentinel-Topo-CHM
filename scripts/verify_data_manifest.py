#!/usr/bin/env python
"""Verify every repository-included file recorded in data/manifest.csv."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    failures = []
    with (DATA / "manifest.csv").open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["status"] != "included":
                continue
            path = DATA / row["relative_path"]
            if not path.is_file():
                failures.append(f"missing: {row['relative_path']}")
                continue
            if int(row["size_bytes"]) != path.stat().st_size:
                failures.append(f"size mismatch: {row['relative_path']}")
            if row["sha256"] != sha256(path):
                failures.append(f"SHA256 mismatch: {row['relative_path']}")
    if failures:
        print("DATA MANIFEST CHECK FAILED")
        for failure in failures:
            print(f" - {failure}")
        return 1
    print("DATA MANIFEST CHECK PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
