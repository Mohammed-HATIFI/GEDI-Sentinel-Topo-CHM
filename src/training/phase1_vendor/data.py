from __future__ import annotations

import gc
import hashlib
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from training.common import MAX_AUX_K, get_split_dir


# ======================================================================================
# 1) SHARDS HELPERS
# ======================================================================================
def list_shards(folder: Path, prefix: str = "shard_") -> List[Path]:
    """
    Return sorted NPZ shards from a folder.

    Robust behavior:
      - primary legacy experiment format: shard_*.npz
      - builder/materialized fallback:    part_*.npz
      - final fallback: any *.npz file
    """
    folder = Path(folder)
    if not folder.exists():
        return []

    primary = sorted(folder.glob(f"{prefix}*.npz"))
    if primary:
        return primary

    # Compatibility with 04_patch_builder.py local outputs.
    fallback_part = sorted(folder.glob("part_*.npz"))
    if fallback_part:
        return fallback_part

    # Last-resort fallback for already-materialized NPZ shards with a custom prefix.
    return sorted(folder.glob("*.npz"))


def list_experiment_shards(experiment_root: Path, split: str) -> List[Path]:
    """
    Convenience wrapper for experiment shards.

    Preferred convention:
      experiments/<exp_name>/<split>/shard_*.npz

    Compatibility fallback:
      experiments/<exp_name>/<split>/part_*.npz
    """
    return list_shards(get_split_dir(experiment_root, split), prefix="shard_")


def count_samples(shards: List[Path], y_key: str = "y") -> int:
    """
    Count total N across shards.
    Priority:
      - y
      - target
      - X
    """
    tot = 0
    for p in shards:
        with np.load(p, allow_pickle=False) as z:
            if y_key in z.files:
                tot += int(z[y_key].shape[0])
            elif "target" in z.files:
                tot += int(z["target"].shape[0])
            elif "X" in z.files:
                tot += int(z["X"].shape[0])
            else:
                raise KeyError(f"[count_samples] shard sans '{y_key}', 'target' ni 'X': {p}")
    return int(tot)


def _extract_valid_sparse_targets_from_npz(z: Any) -> np.ndarray:
    """
    Returns ALL valid GEDI targets from one shard.

    Priority:
      1) y_sparse + y_mask  (pixel-wise rasterized sparse supervision)
      2) y + y_mask         (current 04_patch_builder raster sparse supervision)
      3) aux_y + aux_mask / aux_rows>=0
      4) fallback to y / target if sparse labels are absent
    """
    if "y_sparse" in z.files and "y_mask" in z.files:
        y_sparse = np.asarray(z["y_sparse"], dtype=np.float32)
        y_mask = np.asarray(z["y_mask"]).astype(bool)

        if y_sparse.shape == y_mask.shape:
            vals = y_sparse[y_mask]
        else:
            vals = y_sparse[np.isfinite(y_sparse)]

        vals = np.asarray(vals, dtype=np.float32)
        vals = vals[np.isfinite(vals)]
        return vals

    if "y" in z.files and "y_mask" in z.files:
        y = np.asarray(z["y"], dtype=np.float32)
        y_mask = np.asarray(z["y_mask"]).astype(bool)

        if y.ndim == 4 and y.shape[1] == 1:
            y = y[:, 0]
        if y_mask.ndim == 4 and y_mask.shape[1] == 1:
            y_mask = y_mask[:, 0]

        if y.shape == y_mask.shape:
            vals = y[y_mask]
        else:
            vals = y[np.isfinite(y)]

        vals = np.asarray(vals, dtype=np.float32)
        vals = vals[np.isfinite(vals)]
        return vals

    if "aux_y" in z.files:
        aux_y = np.asarray(z["aux_y"], dtype=np.float32)
        if aux_y.ndim == 1:
            aux_y = aux_y[None, :]

        if "aux_mask" in z.files:
            aux_mask = np.asarray(z["aux_mask"]).astype(bool)
            if aux_mask.shape == aux_y.shape:
                vals = aux_y[aux_mask]
            else:
                vals = aux_y[np.isfinite(aux_y)]
        elif "aux_rows" in z.files:
            aux_rows = np.asarray(z["aux_rows"])
            if aux_rows.shape == aux_y.shape:
                vals = aux_y[(aux_rows >= 0) & np.isfinite(aux_y)]
            else:
                vals = aux_y[np.isfinite(aux_y)]
        else:
            vals = aux_y[np.isfinite(aux_y)]

        vals = np.asarray(vals, dtype=np.float32)
        vals = vals[np.isfinite(vals)]
        return vals

    if "y" in z.files:
        y = np.asarray(z["y"], dtype=np.float32).reshape(-1)
        return y[np.isfinite(y)]

    if "target" in z.files:
        y = np.asarray(z["target"], dtype=np.float32).reshape(-1)
        return y[np.isfinite(y)]

    return np.empty((0,), dtype=np.float32)


def count_sparse_targets(shards: List[Path]) -> int:
    """
    Counts ALL valid GEDI sparse labels stored in raster sparse supervision
    (y_sparse/y_mask or y/y_mask) or aux_y across shards.
    """
    total = 0
    for p in shards:
        try:
            with np.load(p, allow_pickle=False) as z:
                total += int(_extract_valid_sparse_targets_from_npz(z).size)
        except Exception:
            continue
    return total


def _finite_stats(x: np.ndarray) -> Dict[str, float]:
    x = np.asarray(x, dtype=np.float32)
    fin = np.isfinite(x)
    if not fin.any():
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "nan%": 100.0,
        }
    v = x[fin]
    return {
        "n": float(v.size),
        "mean": float(v.mean()),
        "std": float(v.std()),
        "min": float(v.min()),
        "max": float(v.max()),
        "nan%": float(100.0 * (1.0 - fin.mean())),
    }


def preflight_patches(
    patch_root: Path,
    n_shards: int = 2,
    n_samples: int = 8,
    max_aux_k: int = MAX_AUX_K,
) -> None:
    """
    Sanity checks on an experiment root containing train/val/test.
    """
    patch_root = Path(patch_root)
    print("\n" + "=" * 110)
    print("PREFLIGHT (PATCHES) — sanity checks")
    print("=" * 110)

    for split in ["train", "val", "test"]:
        folder = patch_root / split
        shards = list_shards(folder)
        tot = count_samples(shards) if shards else 0
        sparse_tot = count_sparse_targets(shards) if shards else 0

        print(
            f"\n[SPLIT] {split}: dir={folder} | "
            f"n_shards={len(shards)} | samples={tot} | sparse_targets={sparse_tot}"
        )

        for sp in shards[: int(n_shards)]:
            with np.load(sp, allow_pickle=False) as z:
                if "X" not in z.files:
                    raise RuntimeError(f"[PREFLIGHT] {sp} missing 'X'")

                X = z["X"]
                y = z["y"] if "y" in z.files else (z["target"] if "target" in z.files else None)

                print(f"  - {sp.name}")
                print(f"    X={tuple(X.shape)} {X.dtype} | y={None if y is None else tuple(y.shape)}")

                stX = _finite_stats(X[: int(n_samples)])
                print(
                    f"    X stats: mean={stX['mean']:.3f} std={stX['std']:.3f} "
                    f"min={stX['min']:.3f} max={stX['max']:.3f} nan%={stX['nan%']:.2f}"
                )

                n_sparse = int(_extract_valid_sparse_targets_from_npz(z).size)

                if "y_sparse" in z.files and "y_mask" in z.files:
                    ys = np.asarray(z["y_sparse"], dtype=np.float32)
                    ym = np.asarray(z["y_mask"]).astype(bool)
                    yc = np.asarray(z["y_count"], dtype=np.uint16) if "y_count" in z.files else None
                    print(
                        f"    raster_sparse: y_sparse={tuple(ys.shape)} | y_mask={tuple(ym.shape)} | "
                        f"y_count={None if yc is None else tuple(yc.shape)} | sparse_targets={n_sparse}"
                    )
                elif "y" in z.files and "y_mask" in z.files and np.asarray(z["y"]).ndim >= 3:
                    ys = np.asarray(z["y"], dtype=np.float32)
                    ym = np.asarray(z["y_mask"]).astype(bool)
                    yn = np.asarray(z["y_n_valid_pixels"], dtype=np.int32) if "y_n_valid_pixels" in z.files else None
                    print(
                        f"    raster_sparse(builder): y={tuple(ys.shape)} | y_mask={tuple(ym.shape)} | "
                        f"y_n_valid_pixels={None if yn is None else tuple(yn.shape)} | sparse_targets={n_sparse}"
                    )

                if all(k in z.files for k in ("aux_rows", "aux_cols", "aux_y")):
                    K = int(z["aux_rows"].shape[1])
                    ok = "✅" if K == int(max_aux_k) else "⚠️ (pad/trunc later)"
                    meta_keys = [
                        k for k in (
                            "aux_shot_uid",
                            "aux_shot_id",
                            "aux_gedi_date",
                            "aux_gedi_ordinal",
                            "aux_temporal_delta_days",
                            "aux_abs_temporal_delta_days",
                            "aux_lon",
                            "aux_lat",
                            "aux_source_rowid",
                        )
                        if k in z.files
                    ]
                    print(
                        f"    aux: K_shard={K} | MAX_AUX_K={max_aux_k} {ok} | "
                        f"aux_mask={'aux_mask' in z.files} | sparse_targets={n_sparse} | "
                        f"gedi_unique_meta={meta_keys}"
                    )


def preflight_experiment(
    experiment_root: Path,
    n_shards: int = 2,
    n_samples: int = 8,
    max_aux_k: int = MAX_AUX_K,
) -> None:
    """
    Wrapper for final experiments.
    """
    preflight_patches(
        Path(experiment_root),
        n_shards=n_shards,
        n_samples=n_samples,
        max_aux_k=max_aux_k,
    )


# ======================================================================================
# 2) AUGMENTATION — SAR-safe flips only
# ======================================================================================
def augment_patch_with_aux(
    x: torch.Tensor,        # (B, C, H, W)
    aux_rows: torch.Tensor, # (B, K) long
    aux_cols: torch.Tensor, # (B, K) long
    aux_mask: Optional[torch.Tensor] = None,
    *,
    p_hflip: float = 0.5,
    p_vflip: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    SAR-safe augmentation used during training.

    Applies ONLY horizontal / vertical flips, and updates GEDI sparse-point
    coordinates coherently.

    Why no 90° rotation:
      - Sentinel-1 ascending / descending channels are directional.
      - PALSAR HH/HV are directional as well.
      - Rotations would alter the semantics of directional SAR signatures.

    Invalid coordinates (-1) are preserved.
    If aux_mask is provided, coordinates masked-out there are also preserved.
    """
    if x.ndim != 4:
        raise RuntimeError(f"augment_patch_with_aux expects x=(B,C,H,W), got {tuple(x.shape)}")
    if aux_rows.ndim != 2 or aux_cols.ndim != 2:
        raise RuntimeError(
            f"augment_patch_with_aux expects aux_rows/aux_cols=(B,K), got "
            f"{tuple(aux_rows.shape)} / {tuple(aux_cols.shape)}"
        )

    _, _, H, W = x.shape

    valid = (aux_rows >= 0) & (aux_cols >= 0)
    if aux_mask is not None:
        valid = valid & aux_mask.bool()

    if float(p_hflip) > 0.0 and torch.rand(1).item() < float(p_hflip):
        x = torch.flip(x, dims=[-1])
        aux_cols = torch.where(valid, (W - 1) - aux_cols, aux_cols)

    if float(p_vflip) > 0.0 and torch.rand(1).item() < float(p_vflip):
        x = torch.flip(x, dims=[-2])
        aux_rows = torch.where(valid, (H - 1) - aux_rows, aux_rows)

    return x, aux_rows, aux_cols


_augment_patch_with_aux = augment_patch_with_aux


# ======================================================================================
# 3) Layout helpers
# ======================================================================================
def _to_chw(Xi: np.ndarray, expected_channels: int) -> torch.Tensor:
    """
    Convert one sample Xi to torch tensor in CHW format.

    Supports:
      - (C,H,W)
      - (H,W,C)
    """
    Xi = np.asarray(Xi)
    if Xi.ndim != 3:
        raise RuntimeError(f"[DATA] Xi must be 3D, got shape={Xi.shape}")

    if int(Xi.shape[0]) == int(expected_channels):
        return torch.from_numpy(np.ascontiguousarray(Xi)).float()

    if int(Xi.shape[-1]) == int(expected_channels):
        return torch.from_numpy(np.ascontiguousarray(np.transpose(Xi, (2, 0, 1)))).float()

    raise RuntimeError(
        f"[DATA] Cannot infer layout for Xi shape={Xi.shape} with expected_channels={expected_channels}. "
        "Expected (C,H,W) or (H,W,C)."
    )


def _extract_scalar_target_for_sample(z: Any, i: int) -> float:
    """
    Return one scalar compatibility target for sample i.

    Priority:
      1) target or scalar y already present
      2) y_patch_median written by current 04_patch_builder
      3) median over valid raster sparse pixels from y_sparse/y_mask or y/y_mask
      4) NaN if nothing meaningful is available
    """
    if "target" in z.files:
        arr = np.asarray(z["target"], dtype=np.float32).reshape(-1)
        return float(arr[i]) if i < arr.shape[0] else float("nan")

    if "y" in z.files:
        y = np.asarray(z["y"], dtype=np.float32)
        if y.ndim <= 2:
            arr = y.reshape(y.shape[0], -1)[:, 0]
            return float(arr[i]) if i < arr.shape[0] else float("nan")

    if "y_patch_median" in z.files:
        arr = np.asarray(z["y_patch_median"], dtype=np.float32).reshape(-1)
        return float(arr[i]) if i < arr.shape[0] else float("nan")

    if "y_sparse" in z.files and "y_mask" in z.files:
        ys = np.asarray(z["y_sparse"], dtype=np.float32)
        ym = np.asarray(z["y_mask"]).astype(bool)
        vals = ys[i][ym[i]] if ys.shape == ym.shape else ys[i][np.isfinite(ys[i])]
        vals = np.asarray(vals, dtype=np.float32)
        vals = vals[np.isfinite(vals)]
        return float(np.median(vals)) if vals.size > 0 else float("nan")

    if "y" in z.files and "y_mask" in z.files:
        y = np.asarray(z["y"], dtype=np.float32)
        ym = np.asarray(z["y_mask"]).astype(bool)
        if y.ndim == 4 and y.shape[1] == 1:
            y = y[:, 0]
        if ym.ndim == 4 and ym.shape[1] == 1:
            ym = ym[:, 0]
        vals = y[i][ym[i]] if y.shape == ym.shape else y[i][np.isfinite(y[i])]
        vals = np.asarray(vals, dtype=np.float32)
        vals = vals[np.isfinite(vals)]
        return float(np.median(vals)) if vals.size > 0 else float("nan")

    return float("nan")



# ======================================================================================
# 4) GEDI unique metadata helpers
# ======================================================================================
def _stable_string_to_uint63(value: Any) -> int:
    """
    Stable 63-bit non-zero integer id from a GEDI shot id string.

    Why not Python hash()? Python randomizes hashes between processes. A stable
    hash is required so validation/test grouping is reproducible.
    """
    if value is None:
        return 0
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "null", "<na>"}:
        return 0
    h = hashlib.blake2b(s.encode("utf-8", errors="ignore"), digest_size=8).digest()
    uid = int.from_bytes(h, byteorder="little", signed=False) & ((1 << 63) - 1)
    return int(uid if uid > 0 else 1)


def _shot_id_array_to_uid(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    flat = arr.reshape(-1)
    out = np.fromiter((_stable_string_to_uint63(x) for x in flat), dtype=np.int64, count=flat.size)
    return out.reshape(arr.shape)


def _read_aux_meta_array(
    z: Any,
    key: str,
    *,
    dtype: np.dtype,
    default_value: float | int,
    n: int,
    max_aux_k: int,
) -> np.ndarray:
    """
    Read one optional aux metadata array and pad/truncate it to (N, max_aux_k).

    Missing metadata never breaks training; it returns a filled array. The
    unique-GEDI metrics later require aux_shot_uid > 0, so missing shot ids
    simply disable those metrics with NaN rather than crashing.
    """
    out = np.full((int(n), int(max_aux_k)), default_value, dtype=dtype)
    if key not in z.files:
        return out

    arr = np.asarray(z[key])
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        return out

    n_use = min(int(n), int(arr.shape[0]))
    k_use = min(int(max_aux_k), int(arr.shape[1]))
    if n_use <= 0 or k_use <= 0:
        return out

    try:
        out[:n_use, :k_use] = arr[:n_use, :k_use].astype(dtype, copy=False)
    except Exception:
        # Handles object/string arrays with invalid numeric values.
        out[:n_use, :k_use] = np.asarray(arr[:n_use, :k_use], dtype=dtype)
    return out


def _read_aux_shot_uid(z: Any, *, n: int, max_aux_k: int) -> np.ndarray:
    """
    Prefer numeric aux_shot_uid when present. Otherwise derive it from aux_shot_id.
    """
    if "aux_shot_uid" in z.files:
        return _read_aux_meta_array(
            z,
            "aux_shot_uid",
            dtype=np.int64,
            default_value=0,
            n=n,
            max_aux_k=max_aux_k,
        )

    out = np.zeros((int(n), int(max_aux_k)), dtype=np.int64)
    if "aux_shot_id" not in z.files:
        return out

    arr = np.asarray(z["aux_shot_id"])
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        return out

    n_use = min(int(n), int(arr.shape[0]))
    k_use = min(int(max_aux_k), int(arr.shape[1]))
    if n_use <= 0 or k_use <= 0:
        return out

    out[:n_use, :k_use] = _shot_id_array_to_uid(arr[:n_use, :k_use])
    return out


# ======================================================================================
# 4) PatchShardIterable (reads experiment NPZ shards)
# Yields:
#   (x, y, aux_rows, aux_cols, aux_y, aux_mask, y_sparse, y_mask, y_count)
# ======================================================================================
class PatchShardIterable(IterableDataset):
    """
    Iterate over NPZ shards produced by the final experiment builder.

    Expected fields:
      - X: (N,H,W,C) or (N,C,H,W)
      - target: (N,) or scalar y                          compatibility scalar target
      - y_sparse + y_mask                                 pixel-wise sparse target map (legacy experiment format)
      - y + y_mask                                        pixel-wise sparse target map (current 04_patch_builder format)
      - y_count:  (N,H,W) optional                        number of GEDI shots / pixel
      - y_n_valid_pixels: (N,) optional                   builder diagnostic only
      - aux_rows/aux_cols/aux_y: (N,K)                    legacy sparse supervision points
      - aux_mask: (N,K) optional mask (1=valid)
      - aux_shot_id or aux_shot_uid: (N,K)                GEDI shot identity for unique-shot metrics
      - aux_gedi_ordinal: (N,K) optional                  GEDI acquisition ordinal/date
      - aux_temporal_delta_days / aux_abs_temporal_delta_days: (N,K) optional
      - aux_source_rowid: (N,K) optional                  original row id diagnostic

    Yields:
      (x, y, aux_rows, aux_cols, aux_y, aux_mask, y_sparse, y_mask, y_count, meta)

    meta is a dict of tensors:
      aux_shot_uid, aux_gedi_ordinal, aux_temporal_delta_days,
      aux_abs_temporal_delta_days, aux_source_rowid
    """

    def __init__(
        self,
        shard_paths: List[Path],
        *,
        seed: int,
        shuffle_shards: bool,
        shuffle_within: bool,
        in_ch: int,
        max_aux_k: int = MAX_AUX_K,
        drop_channels: Optional[List[int]] = None,
        use_original_coords: bool = False,
    ):
        super().__init__()
        self.shard_paths = [Path(p) for p in shard_paths]
        self.seed = int(seed)
        self.shuffle_shards = bool(shuffle_shards)
        self.shuffle_within = bool(shuffle_within)
        self.in_ch = int(in_ch)
        self.max_aux_k = int(max_aux_k)
        self.drop_channels = sorted(int(c) for c in (drop_channels or []))
        self.drop_channels_set = set(self.drop_channels)
        self.use_original_coords = bool(use_original_coords)  # kept for compatibility only
        self.raw_in_ch = int(self.in_ch + len(self.drop_channels))

    def _apply_drop_channels(self, xt: torch.Tensor) -> torch.Tensor:
        if not self.drop_channels:
            if int(xt.shape[0]) != int(self.in_ch):
                raise RuntimeError(
                    f"[DATA] Tensor has {int(xt.shape[0])} channels but in_ch={self.in_ch} "
                    "and no drop_channels provided."
                )
            return xt

        c_total = int(xt.shape[0])
        bad = [c for c in self.drop_channels if c < 0 or c >= c_total]
        if bad:
            raise RuntimeError(
                f"[DATA] Invalid drop_channels={self.drop_channels} for tensor with {c_total} channels"
            )

        keep = [c for c in range(c_total) if c not in self.drop_channels_set]
        xt = xt[keep]

        if int(xt.shape[0]) != int(self.in_ch):
            raise RuntimeError(
                f"[DATA] After dropping channels {self.drop_channels}, "
                f"got {int(xt.shape[0])} channels but expected in_ch={self.in_ch}"
            )
        return xt

    def __iter__(self) -> Iterator[
        Tuple[
            torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
            torch.Tensor, torch.Tensor, torch.Tensor,
            Dict[str, torch.Tensor],
        ]
    ]:
        info = get_worker_info()
        worker_id = int(info.id) if info is not None else 0
        num_workers = int(info.num_workers) if info is not None else 1

        all_shards = list(self.shard_paths)
        rng_global = np.random.default_rng(self.seed)

        # Global deterministic shuffle first, then worker partition.
        if self.shuffle_shards and len(all_shards) > 1:
            perm = rng_global.permutation(len(all_shards))
            all_shards = [all_shards[i] for i in perm]

        shards = all_shards[worker_id::num_workers] if num_workers > 1 else all_shards
        rng = np.random.default_rng(self.seed + worker_id * 1337)

        for sp in shards:
            try:
                with np.load(sp, allow_pickle=True) as z:
                    if "X" not in z.files:
                        raise RuntimeError(f"[DATA] {sp} missing 'X'")
                    X = z["X"]

                    if ("target" not in z.files) and ("y" not in z.files):
                        raise RuntimeError(f"[DATA] {sp} missing 'y'/'target'")

                    has_aux = all(k in z.files for k in ("aux_rows", "aux_cols", "aux_y"))
                    aux_rows_np = np.asarray(z["aux_rows"], dtype=np.int16) if has_aux else None
                    aux_cols_np = np.asarray(z["aux_cols"], dtype=np.int16) if has_aux else None
                    aux_y_np = np.asarray(z["aux_y"], dtype=np.float32) if has_aux else None
                    aux_mask_np = (
                        np.asarray(z["aux_mask"], dtype=np.uint8)
                        if (has_aux and "aux_mask" in z.files)
                        else None
                    )

                    has_raster_sparse = False
                    y_sparse_np = None
                    y_mask_np = None
                    y_count_np = None

                    n = int(X.shape[0])
                    aux_shot_uid_np = _read_aux_shot_uid(z, n=n, max_aux_k=self.max_aux_k)
                    aux_gedi_ordinal_np = _read_aux_meta_array(
                        z, "aux_gedi_ordinal", dtype=np.int64, default_value=0, n=n, max_aux_k=self.max_aux_k
                    )
                    aux_temporal_delta_days_np = _read_aux_meta_array(
                        z, "aux_temporal_delta_days", dtype=np.float32, default_value=np.nan, n=n, max_aux_k=self.max_aux_k
                    )
                    aux_abs_temporal_delta_days_np = _read_aux_meta_array(
                        z, "aux_abs_temporal_delta_days", dtype=np.float32, default_value=np.nan, n=n, max_aux_k=self.max_aux_k
                    )
                    aux_source_rowid_np = _read_aux_meta_array(
                        z, "aux_source_rowid", dtype=np.int64, default_value=-1, n=n, max_aux_k=self.max_aux_k
                    )

                    if "y_sparse" in z.files and "y_mask" in z.files:
                        has_raster_sparse = True
                        y_sparse_np = np.asarray(z["y_sparse"], dtype=np.float32)
                        y_mask_np = np.asarray(z["y_mask"]).astype(bool)
                        y_count_np = np.asarray(z["y_count"], dtype=np.uint16) if "y_count" in z.files else None
                    elif "y" in z.files and "y_mask" in z.files:
                        # Current 04_patch_builder format: y is the sparse raster map.
                        y_sparse_np = np.asarray(z["y"], dtype=np.float32)
                        y_mask_np = np.asarray(z["y_mask"]).astype(bool)
                        if y_sparse_np.ndim == 4 and y_sparse_np.shape[1] == 1:
                            y_sparse_np = y_sparse_np[:, 0]
                        if y_mask_np.ndim == 4 and y_mask_np.shape[1] == 1:
                            y_mask_np = y_mask_np[:, 0]
                        has_raster_sparse = True
                        y_count_np = None

                    if has_raster_sparse and y_sparse_np is not None and int(y_sparse_np.shape[0]) != n:
                        raise RuntimeError(f"[DATA] {sp}: X has N={n} but y_sparse has shape={y_sparse_np.shape}")

                    order = np.arange(n, dtype=np.int64)
                    if self.shuffle_within and n > 1:
                        rng.shuffle(order)

                    for i in order:
                        Xi = np.asarray(X[i])
                        xt = _to_chw(Xi, self.raw_in_ch)
                        xt = self._apply_drop_channels(xt)

                        yi = torch.tensor(_extract_scalar_target_for_sample(z, int(i)), dtype=torch.float32)

                        aux_r = torch.full((self.max_aux_k,), -1, dtype=torch.long)
                        aux_c = torch.full((self.max_aux_k,), -1, dtype=torch.long)
                        aux_yy = torch.full((self.max_aux_k,), float("nan"), dtype=torch.float32)
                        aux_m = torch.zeros((self.max_aux_k,), dtype=torch.uint8)

                        aux_shot_uid = torch.zeros((self.max_aux_k,), dtype=torch.long)
                        aux_gedi_ordinal = torch.zeros((self.max_aux_k,), dtype=torch.long)
                        aux_temporal_delta_days = torch.full((self.max_aux_k,), float("nan"), dtype=torch.float32)
                        aux_abs_temporal_delta_days = torch.full((self.max_aux_k,), float("nan"), dtype=torch.float32)
                        aux_source_rowid = torch.full((self.max_aux_k,), -1, dtype=torch.long)

                        if has_aux and aux_rows_np is not None and aux_cols_np is not None and aux_y_np is not None:
                            K_shard = int(aux_rows_np.shape[1])
                            K_use = min(K_shard, self.max_aux_k)

                            rr = aux_rows_np[i, :K_use].astype(np.int64, copy=False)
                            cc = aux_cols_np[i, :K_use].astype(np.int64, copy=False)
                            yy = aux_y_np[i, :K_use].astype(np.float32, copy=False)

                            if aux_mask_np is not None:
                                m = aux_mask_np[i, :K_use].astype(np.uint8, copy=False)
                            else:
                                m = np.isfinite(yy).astype(np.uint8)

                            aux_r[:K_use] = torch.from_numpy(rr)
                            aux_c[:K_use] = torch.from_numpy(cc)
                            aux_yy[:K_use] = torch.from_numpy(yy)
                            aux_m[:K_use] = torch.from_numpy(m)

                            aux_shot_uid[:K_use] = torch.from_numpy(aux_shot_uid_np[i, :K_use].astype(np.int64, copy=False))
                            aux_gedi_ordinal[:K_use] = torch.from_numpy(aux_gedi_ordinal_np[i, :K_use].astype(np.int64, copy=False))
                            aux_temporal_delta_days[:K_use] = torch.from_numpy(aux_temporal_delta_days_np[i, :K_use].astype(np.float32, copy=False))
                            aux_abs_temporal_delta_days[:K_use] = torch.from_numpy(aux_abs_temporal_delta_days_np[i, :K_use].astype(np.float32, copy=False))
                            aux_source_rowid[:K_use] = torch.from_numpy(aux_source_rowid_np[i, :K_use].astype(np.int64, copy=False))

                        if has_raster_sparse and y_sparse_np is not None and y_mask_np is not None:
                            ys = torch.from_numpy(np.ascontiguousarray(y_sparse_np[i])).float()
                            ym = torch.from_numpy(np.ascontiguousarray(y_mask_np[i])).bool()
                            if y_count_np is not None:
                                yc = torch.from_numpy(np.ascontiguousarray(y_count_np[i])).float()
                            else:
                                yc = torch.zeros_like(ys, dtype=torch.float32)
                        else:
                            _, H, W = xt.shape
                            ys = torch.full((H, W), float("nan"), dtype=torch.float32)
                            ym = torch.zeros((H, W), dtype=torch.bool)
                            yc = torch.zeros((H, W), dtype=torch.float32)

                        meta = {
                            "aux_shot_uid": aux_shot_uid,
                            "aux_gedi_ordinal": aux_gedi_ordinal,
                            "aux_temporal_delta_days": aux_temporal_delta_days,
                            "aux_abs_temporal_delta_days": aux_abs_temporal_delta_days,
                            "aux_source_rowid": aux_source_rowid,
                        }

                        yield xt, yi, aux_r, aux_c, aux_yy, aux_m, ys, ym, yc, meta

            except Exception as e:
                raise RuntimeError(f"[PatchShardIterable] Échec lecture shard: {sp}") from e

            gc.collect()
