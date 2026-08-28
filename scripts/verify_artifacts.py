#!/usr/bin/env python
"""Verify SHA-256 identities in a JSON checkpoint manifest."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    failures = []
    for name, record in manifest["models"].items():
        path = args.root / record["filename"]
        if not path.is_file():
            failures.append(f"{name}: missing {path}")
        elif sha256(path) != record["sha256"]:
            failures.append(f"{name}: SHA-256 mismatch")
        else:
            print(f"OK {name}: {path.name}")
    if failures:
        raise SystemExit("\n".join(failures))


if __name__ == "__main__":
    main()
