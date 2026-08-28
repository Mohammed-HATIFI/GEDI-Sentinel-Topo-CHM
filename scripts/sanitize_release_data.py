#!/usr/bin/env python
"""Remove machine-local paths from the public processed-data release."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GEDI = ROOT / "data" / "processed" / "gedi"
LOCAL_PATH = re.compile(r"^[A-Za-z]:[\\/]")


def sanitize_json_value(value):
    if isinstance(value, dict):
        return {key: sanitize_json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_json_value(item) for item in value]
    if isinstance(value, str) and LOCAL_PATH.match(value):
        return "<LOCAL_PATH_REDACTED>"
    return value


def main() -> None:
    for path in sorted(GEDI.glob("*/sample_catalog_step05.csv")):
        frame = pd.read_csv(path)
        removable = [column for column in ("x_path", "source_shard") if column in frame]
        frame = frame.drop(columns=removable)
        frame.to_csv(path, index=False)
        print(f"Sanitized {path.relative_to(ROOT)}; removed={removable}")

    for path in sorted(GEDI.glob("*/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload = sanitize_json_value(payload)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"Sanitized {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
