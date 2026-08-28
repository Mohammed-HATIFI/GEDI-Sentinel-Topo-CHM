#!/usr/bin/env python
"""Build a checksum manifest from an explicitly named release directory."""
from __future__ import annotations
import argparse
import csv
import hashlib
from pathlib import Path


def sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    rows = []
    for path in sorted(p for p in args.root.rglob("*") if p.is_file()):
        rows.append({"logical_id": path.stem, "relative_path": path.relative_to(args.root).as_posix(), "sha256": sha256(path), "size_bytes": path.stat().st_size})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["logical_id", "relative_path", "sha256", "size_bytes"])
        writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
