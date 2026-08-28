from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class SequenceRecord:
    split: str
    patch_key: str
    years: tuple[int, ...]
    months: tuple[int, ...]
    sample_ids: tuple[str, ...]
    x_paths: tuple[str, ...]


def load_catalogs(experiment_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    experiment_root = Path(experiment_root)
    samples = pd.read_csv(experiment_root / "sample_catalog_step05.csv", low_memory=False)
    shots = pd.read_csv(experiment_root / "shot_catalog_step05.csv.gz", low_memory=False)
    required_samples = {"split", "patch_key", "sample_id", "year", "month", "x_path"}
    required_shots = {"sample_id", "rh95", "local_row", "local_col", "aux_shot_uid"}
    if missing := required_samples.difference(samples.columns):
        raise KeyError(f"Missing sample columns: {sorted(missing)}")
    if missing := required_shots.difference(shots.columns):
        raise KeyError(f"Missing shot columns: {sorted(missing)}")
    return samples, shots


def build_complete_sequences(
    samples: pd.DataFrame,
    *,
    split: str,
    years: Sequence[int],
    anchor_month: int = 7,
    max_month_gap: int = 1,
) -> List[SequenceRecord]:
    """Select one temporally comparable sample per patch and year.

    Exact ECHOSAT regression uses a complete, evenly spaced annual vector.  We
    therefore keep only patches covering every requested year and never fill or
    interpolate a missing year.
    """
    years = tuple(int(v) for v in years)
    frame = samples[samples["split"].astype(str).str.lower().eq(str(split).lower())].copy()
    frame = frame[frame["year"].isin(years)].copy()
    frame["month_gap"] = (pd.to_numeric(frame["month"]) - int(anchor_month)).abs()
    frame = frame[frame["month_gap"] <= int(max_month_gap)].copy()
    sort_cols = ["patch_key", "year", "month_gap"]
    ascending = [True, True, True]
    if "n_unique_gedi" in frame.columns:
        sort_cols.append("n_unique_gedi")
        ascending.append(False)
    selected = frame.sort_values(sort_cols, ascending=ascending).drop_duplicates(["patch_key", "year"])

    records: List[SequenceRecord] = []
    for patch_key, group in selected.groupby("patch_key", sort=True):
        by_year = group.set_index(group["year"].astype(int))
        if not set(years).issubset(set(by_year.index.tolist())):
            continue
        ordered = by_year.loc[list(years)]
        records.append(
            SequenceRecord(
                split=str(split),
                patch_key=str(patch_key),
                years=years,
                months=tuple(int(v) for v in ordered["month"].tolist()),
                sample_ids=tuple(str(v) for v in ordered["sample_id"].tolist()),
                x_paths=tuple(str(v) for v in ordered["x_path"].tolist()),
            )
        )
    return records


def build_sliding_sequences(
    samples: pd.DataFrame,
    *,
    split: str,
    all_years: Sequence[int],
    window_length: int,
    anchor_month: int = 7,
    max_month_gap: int = 1,
) -> List[SequenceRecord]:
    """Build every complete consecutive annual window without imputation."""
    all_years = tuple(sorted(int(v) for v in all_years))
    window_length = int(window_length)
    if window_length < 2 or window_length > len(all_years):
        raise ValueError(f"Invalid window_length={window_length} for years={all_years}")

    frame = samples[samples["split"].astype(str).str.lower().eq(str(split).lower())].copy()
    frame = frame[frame["year"].isin(all_years)].copy()
    frame["month_gap"] = (pd.to_numeric(frame["month"]) - int(anchor_month)).abs()
    frame = frame[frame["month_gap"] <= int(max_month_gap)].copy()
    sort_cols = ["patch_key", "year", "month_gap"]
    ascending = [True, True, True]
    if "n_unique_gedi" in frame.columns:
        sort_cols.append("n_unique_gedi")
        ascending.append(False)
    selected = frame.sort_values(sort_cols, ascending=ascending).drop_duplicates(["patch_key", "year"])

    records: List[SequenceRecord] = []
    first_year, last_year = min(all_years), max(all_years)
    for patch_key, group in selected.groupby("patch_key", sort=True):
        by_year = group.set_index(group["year"].astype(int))
        for start in range(first_year, last_year - window_length + 2):
            years = tuple(range(start, start + window_length))
            if not set(years).issubset(set(by_year.index.tolist())):
                continue
            ordered = by_year.loc[list(years)]
            records.append(
                SequenceRecord(
                    split=str(split),
                    patch_key=f"{patch_key}|Y{years[0]}-{years[-1]}",
                    years=years,
                    months=tuple(int(v) for v in ordered["month"].tolist()),
                    sample_ids=tuple(str(v) for v in ordered["sample_id"].tolist()),
                    x_paths=tuple(str(v) for v in ordered["x_path"].tolist()),
                )
            )
    return records


def build_same_month_sequences(
    samples: pd.DataFrame,
    *,
    split: str,
    all_years: Sequence[int],
    window_length: int = 3,
    leaf_on_months: Sequence[int] = (5, 6, 7, 8, 9),
) -> List[SequenceRecord]:
    """Build consecutive annual sequences at an identical leaf-on month.

    This is the radiometrically safer alternative to mixing May, June, July,
    August and September inside one annual trajectory.  Every returned record
    is therefore, for example, May->May->May or August->August->August.  No
    within-year median and no temporal interpolation are applied.
    """
    all_years = tuple(sorted(int(v) for v in all_years))
    window_length = int(window_length)
    months = tuple(sorted({int(v) for v in leaf_on_months}))
    if window_length < 2 or window_length > len(all_years):
        raise ValueError(f"Invalid window_length={window_length} for years={all_years}")
    if not months:
        raise ValueError("leaf_on_months must contain at least one month")

    frame = samples[
        samples["split"].astype(str).str.lower().eq(str(split).lower())
    ].copy()
    frame["year"] = pd.to_numeric(frame["year"], errors="coerce")
    frame["month"] = pd.to_numeric(frame["month"], errors="coerce")
    frame = frame[frame["year"].isin(all_years) & frame["month"].isin(months)].copy()

    # A patch/year/month should normally be unique.  If catalog duplicates are
    # present, keep the occurrence with the largest GEDI support deterministically.
    sort_cols = ["patch_key", "month", "year"]
    ascending = [True, True, True]
    if "n_unique_gedi" in frame.columns:
        sort_cols.append("n_unique_gedi")
        ascending.append(False)
    selected = frame.sort_values(sort_cols, ascending=ascending).drop_duplicates(
        ["patch_key", "year", "month"]
    )

    records: List[SequenceRecord] = []
    first_year, last_year = min(all_years), max(all_years)
    for (patch_key, month), group in selected.groupby(["patch_key", "month"], sort=True):
        by_year = group.set_index(group["year"].astype(int))
        for start in range(first_year, last_year - window_length + 2):
            years = tuple(range(start, start + window_length))
            if not set(years).issubset(set(by_year.index.tolist())):
                continue
            ordered = by_year.loc[list(years)]
            records.append(
                SequenceRecord(
                    split=str(split),
                    patch_key=f"{patch_key}|M{int(month):02d}|Y{years[0]}-{years[-1]}",
                    years=years,
                    months=tuple(int(month) for _ in years),
                    sample_ids=tuple(str(v) for v in ordered["sample_id"].tolist()),
                    x_paths=tuple(str(v) for v in ordered["x_path"].tolist()),
                )
            )
    return records


def sequence_support_table(records_by_split: Dict[str, Sequence[SequenceRecord]]) -> pd.DataFrame:
    rows = []
    for split, records in records_by_split.items():
        for record in records:
            rows.append(
                {
                    "split": split,
                    "patch_key": record.patch_key,
                    "n_years": len(record.years),
                    "years": ",".join(map(str, record.years)),
                    "months": ",".join(f"{m:02d}" for m in record.months),
                }
            )
    return pd.DataFrame(rows)


class B4SequenceCropDataset(Dataset):
    """Random aligned crops from complete multi-year B4 patch sequences."""

    def __init__(
        self,
        records: Sequence[SequenceRecord],
        shots: pd.DataFrame,
        *,
        crop_size: int = 96,
        samples_per_epoch: int = 256,
        drop_channels: Sequence[int] = (12, 13),
        seed: int = 42,
        center_on_gedi: bool = False,
        balanced_height_anchors: bool = False,
        height_bins: Sequence[float] = (0, 2.5, 5, 8, 10, 15, 20, 30, 40, 42),
    ):
        if not records:
            raise ValueError("No complete annual sequence is available.")
        self.records = list(records)
        self.crop_size = int(crop_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.drop_channels = tuple(sorted(set(int(v) for v in drop_channels)))
        self.seed = int(seed)
        self.center_on_gedi = bool(center_on_gedi)
        self.balanced_height_anchors = bool(balanced_height_anchors)
        self.height_bins = np.asarray(tuple(float(v) for v in height_bins), dtype=np.float64)
        if self.balanced_height_anchors and len(self.height_bins) < 2:
            raise ValueError("height_bins must have at least two edges")
        self.epoch = 0
        self.shots = {
            str(sample_id): group.copy()
            for sample_id, group in shots.groupby(shots["sample_id"].astype(str), sort=False)
        }

        self.record_anchors = {}
        if self.center_on_gedi:
            for record in self.records:
                anchors = []
                for sample_id in record.sample_ids:
                    group = self.shots.get(str(sample_id))
                    if group is None:
                        continue
                    rows = pd.to_numeric(group["local_row"], errors="coerce")
                    cols = pd.to_numeric(group["local_col"], errors="coerce")
                    values = pd.to_numeric(group["rh95"], errors="coerce")
                    valid = rows.notna() & cols.notna() & values.notna() & (values > 0)
                    anchors.extend(
                        zip(
                            rows[valid].astype(int),
                            cols[valid].astype(int),
                            values[valid].astype(float),
                        )
                    )
                self.record_anchors[record.patch_key] = tuple(anchors)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _rng(self, index: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + 1_000_003 * self.epoch + int(index))

    def __getitem__(self, index: int):
        rng = self._rng(index)
        record = self.records[int(rng.integers(0, len(self.records)))]
        first = np.load(record.x_paths[0], mmap_mode="r", allow_pickle=False)
        height, width, _ = first.shape
        if min(height, width) < self.crop_size:
            raise ValueError(f"Crop {self.crop_size} exceeds input shape {first.shape}")
        anchors = self.record_anchors.get(record.patch_key, ())
        if self.center_on_gedi and anchors:
            if self.balanced_height_anchors:
                by_bin = {}
                for anchor in anchors:
                    bin_idx = int(np.digitize(float(anchor[2]), self.height_bins[1:-1], right=False))
                    by_bin.setdefault(bin_idx, []).append(anchor)
                available = tuple(sorted(by_bin))
                chosen_bin = available[int(rng.integers(0, len(available)))]
                pool = by_bin[chosen_bin]
                anchor_r, anchor_c, _ = pool[int(rng.integers(0, len(pool)))]
            else:
                anchor_r, anchor_c, _ = anchors[int(rng.integers(0, len(anchors)))]
            row0 = int(np.clip(anchor_r - rng.integers(0, self.crop_size), 0, height - self.crop_size))
            col0 = int(np.clip(anchor_c - rng.integers(0, self.crop_size), 0, width - self.crop_size))
        else:
            row0 = int(rng.integers(0, height - self.crop_size + 1))
            col0 = int(rng.integers(0, width - self.crop_size + 1))

        cubes = []
        target = np.zeros((len(record.years), self.crop_size, self.crop_size), dtype=np.float32)
        keep_channels = None
        for year_idx, (sample_id, x_path) in enumerate(zip(record.sample_ids, record.x_paths)):
            x = np.load(x_path, mmap_mode="r", allow_pickle=False)
            if keep_channels is None:
                keep_channels = [c for c in range(x.shape[-1]) if c not in self.drop_channels]
            crop = np.asarray(
                x[row0 : row0 + self.crop_size, col0 : col0 + self.crop_size, keep_channels],
                dtype=np.float32,
            )
            cubes.append(np.moveaxis(crop, -1, 0))

            group = self.shots.get(str(sample_id))
            if group is None:
                continue
            rows = pd.to_numeric(group["local_row"], errors="coerce").to_numpy()
            cols = pd.to_numeric(group["local_col"], errors="coerce").to_numpy()
            values = pd.to_numeric(group["rh95"], errors="coerce").to_numpy()
            valid = (
                np.isfinite(rows)
                & np.isfinite(cols)
                & np.isfinite(values)
                & (rows >= row0)
                & (rows < row0 + self.crop_size)
                & (cols >= col0)
                & (cols < col0 + self.crop_size)
                & (values > 0)
            )
            for rr, cc, value in zip(rows[valid].astype(int), cols[valid].astype(int), values[valid]):
                tr = rr - row0
                tc = cc - col0
                target[year_idx, tr, tc] = max(float(target[year_idx, tr, tc]), float(value))

        meta = {
            "patch_key": record.patch_key,
            "row0": row0,
            "col0": col0,
            "years": record.years,
            "months": record.months,
        }
        return torch.from_numpy(np.stack(cubes)), torch.from_numpy(target), meta
