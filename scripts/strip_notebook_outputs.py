#!/usr/bin/env python
"""Create deterministic output-stripped notebook copies for Git releases."""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def strip(source: Path, destination: Path):
    notebook = json.loads(source.read_text(encoding="utf-8"))
    for cell in notebook.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["outputs"] = []
            cell["execution_count"] = None
    notebook.setdefault("metadata", {}).pop("widgets", None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    strip(args.source, args.destination)


if __name__ == "__main__":
    main()
