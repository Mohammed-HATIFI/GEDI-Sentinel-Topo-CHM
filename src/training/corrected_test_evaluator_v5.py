from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


def _absolute_delta_column(shots: pd.DataFrame) -> pd.Series:
    if "aux_abs_temporal_delta_days" in shots.columns:
        return pd.to_numeric(shots["aux_abs_temporal_delta_days"], errors="coerce").abs()
    if "abs_temporal_delta_days" in shots.columns:
        return pd.to_numeric(shots["abs_temporal_delta_days"], errors="coerce").abs()
    if "aux_temporal_delta_days" in shots.columns:
        return pd.to_numeric(shots["aux_temporal_delta_days"], errors="coerce").abs()
    raise KeyError(
        "The shot catalogue must contain aux_abs_temporal_delta_days, "
        "abs_temporal_delta_days, or aux_temporal_delta_days."
    )


def _load_full_patch(path: str | Path, keep_channels: Sequence[int]) -> torch.Tensor:
    array = np.load(str(path), mmap_mode="r", allow_pickle=False)
    crop = np.asarray(array[..., list(keep_channels)], dtype=np.float32)
    tensor = torch.from_numpy(np.moveaxis(crop, -1, 0))
    pad_h = (16 - tensor.shape[-2] % 16) % 16
    pad_w = (16 - tensor.shape[-1] % 16) % 16
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="replicate")
    return tensor


@torch.inference_mode()
def evaluate_full_patch_temporal_nearest(
    *,
    model,
    records,
    shots: pd.DataFrame,
    device: str,
    split: str | None = None,
    drop_channels: Sequence[int] = (12, 13),
    min_height: float = 2.5,
    max_height: float = 45.0,
    progress_every: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """STEP07-compatible full-patch inference without target leakage.

    All candidate occurrences are retained first.  A single row per GEDI shot
    is then selected using only acquisition metadata:

    1. minimum absolute GEDI-to-Sentinel temporal distance;
    2. sequence position closest to the centre of the temporal window;
    3. deterministic sample/window/order tie breakers.

    Neither RH95 nor a prediction error is used by the selection rule.
    Full 512-pixel patches are evaluated so the frozen B4 reference receives
    the same spatial context as the canonical STEP07 evaluator.
    """
    if not records:
        raise ValueError("No sequence record was supplied.")

    record_splits = {
        str(getattr(record, "split", "")).strip().lower()
        for record in records
        if str(getattr(record, "split", "")).strip()
    }
    if split is None:
        if len(record_splits) == 1:
            active_split = next(iter(record_splits))
        elif not record_splits:
            # Backward compatibility for older TEST-only SequenceRecord objects.
            active_split = "test"
        else:
            raise ValueError(
                f"Records contain multiple splits {sorted(record_splits)}; pass split explicitly."
            )
    else:
        active_split = str(split).strip().lower()
        if not active_split:
            raise ValueError("split must be a non-empty string")
        if record_splits and record_splits != {active_split}:
            raise ValueError(
                f"Requested split={active_split!r}, but records contain {sorted(record_splits)}."
            )

    model.to(device).eval()
    frame_shots = shots.copy()
    frame_shots["_abs_temporal_delta_days_v5"] = _absolute_delta_column(frame_shots)
    if "split" in frame_shots.columns:
        frame_shots = frame_shots[
            frame_shots["split"].astype(str).str.strip().str.lower().eq(active_split)
        ].copy()

    required = {
        "sample_id",
        "rh95",
        "local_row",
        "local_col",
        "aux_shot_uid",
        "_abs_temporal_delta_days_v5",
    }
    missing = required.difference(frame_shots.columns)
    if missing:
        raise KeyError(f"Missing shot columns for corrected {active_split}: {sorted(missing)}")

    shot_groups = {
        str(sample_id): group.copy()
        for sample_id, group in frame_shots.groupby(
            frame_shots["sample_id"].astype(str), sort=False
        )
    }

    first_path = Path(records[0].x_paths[0])
    first = np.load(str(first_path), mmap_mode="r", allow_pickle=False)
    height, width, channels = first.shape
    drop = set(int(v) for v in drop_channels)
    keep = [index for index in range(channels) if index not in drop]

    rows: list[dict[str, object]] = []
    candidate_order = 0
    total = len(records)
    for record_index, record in enumerate(records):
        cubes = [_load_full_patch(path, keep) for path in record.x_paths]
        sequence = torch.stack(cubes, dim=0).unsqueeze(0).to(device)
        output = model(sequence)[0].float().cpu().numpy()
        output = output[..., :height, :width]
        del sequence, cubes

        sequence_centre = (len(record.years) - 1) / 2.0
        for year_index, (year, month, sample_id) in enumerate(
            zip(record.years, record.months, record.sample_ids)
        ):
            group = shot_groups.get(str(sample_id))
            if group is None:
                continue
            rr = pd.to_numeric(group["local_row"], errors="coerce")
            cc = pd.to_numeric(group["local_col"], errors="coerce")
            yy = pd.to_numeric(group["rh95"], errors="coerce")
            dd = pd.to_numeric(
                group["_abs_temporal_delta_days_v5"], errors="coerce"
            )
            valid = (
                rr.notna()
                & cc.notna()
                & yy.notna()
                & dd.notna()
                & yy.between(float(min_height), float(max_height), inclusive="both")
                & rr.ge(0)
                & rr.lt(height)
                & cc.ge(0)
                & cc.lt(width)
            )
            for index in group.index[valid]:
                local_r = int(rr.loc[index])
                local_c = int(cc.loc[index])
                rows.append(
                    {
                        "split": active_split,
                        "patch_key": str(record.patch_key),
                        "sample_id": str(sample_id),
                        "aux_shot_uid": str(group.loc[index, "aux_shot_uid"]),
                        "year": int(year),
                        "month": int(month),
                        "rh95": float(yy.loc[index]),
                        "abs_temporal_delta_days": float(dd.loc[index]),
                        "sequence_start_year": int(record.years[0]),
                        "sequence_end_year": int(record.years[-1]),
                        "sequence_position": int(year_index),
                        "sequence_center_distance": float(
                            abs(float(year_index) - sequence_centre)
                        ),
                        "candidate_order": int(candidate_order),
                        "pred_off_reference": float(
                            output[0, year_index, local_r, local_c]
                        ),
                        "pred_on_growthloss": float(
                            output[1, year_index, local_r, local_c]
                        ),
                    }
                )
                candidate_order += 1

        if progress_every and (
            (record_index + 1) % int(progress_every) == 0
            or record_index + 1 == total
        ):
            print(
                f"[V5 {active_split.upper()}] full-patch sequence {record_index + 1:03d}/{total:03d} "
                f"| candidates={len(rows):,}",
                flush=True,
            )
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()

    raw = pd.DataFrame(rows)
    if raw.empty:
        record_sample_ids = {
            str(sample_id)
            for record in records
            for sample_id in record.sample_ids
        }
        matching_sample_count = len(record_sample_ids.intersection(shot_groups))
        raise RuntimeError(
            f"Corrected {active_split.upper()} produced no valid candidate occurrence "
            f"(records={len(records)}, split_shots={len(frame_shots)}, "
            f"record_samples_with_shots={matching_sample_count})."
        )

    selection_columns = [
        "aux_shot_uid",
        "abs_temporal_delta_days",
        "sequence_center_distance",
        "sample_id",
        "sequence_start_year",
        "candidate_order",
    ]
    nearest = (
        raw.sort_values(selection_columns, kind="mergesort")
        .drop_duplicates("aux_shot_uid", keep="first")
        .reset_index(drop=True)
    )
    nearest["selection_rule"] = (
        "min_abs_temporal_delta_then_central_sequence_then_stable_order"
    )
    if not nearest["aux_shot_uid"].is_unique:
        raise AssertionError(
            f"Corrected {active_split.upper()} must contain one row per GEDI shot."
        )
    return raw.reset_index(drop=True), nearest


def canonical_step07_unique_nearest(
    path: str | Path,
    *,
    min_height: float = 2.5,
    max_height: float = 45.0,
) -> pd.DataFrame:
    """Load the official B4 STEP07 table using its published selection rule."""
    frame = pd.read_csv(Path(path), low_memory=False)
    required = {
        "aux_shot_uid",
        "abs_temporal_delta_days",
        "rh95",
        "prediction_original_coords",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing STEP07 columns: {sorted(missing)}")
    frame = frame[
        pd.to_numeric(frame["rh95"], errors="coerce").between(
            float(min_height), float(max_height), inclusive="both"
        )
    ].copy()
    frame["_delta"] = pd.to_numeric(
        frame["abs_temporal_delta_days"], errors="coerce"
    ).fillna(np.inf)
    frame["_row"] = np.arange(len(frame))
    return (
        frame.sort_values(["aux_shot_uid", "_delta", "_row"], kind="mergesort")
        .drop_duplicates("aux_shot_uid", keep="first")
        .reset_index(drop=True)
    )
