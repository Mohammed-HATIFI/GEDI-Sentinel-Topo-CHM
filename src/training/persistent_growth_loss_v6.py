from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import torch

from .ablation_training import load_official_growth_loss, sha256_file


SUPPORTED_RULES = {
    "official_echosat",
    "persistent_running_max",
    "persistent_running_max_robust",
}


def _constrained_robust_line(
    values: torch.Tensor,
    *,
    slope_min: float,
    slope_max: float,
    max_intercept: float,
) -> torch.Tensor:
    """Fit a no-gradient robust line to ``[B,Y,H,W]`` annual values.

    Three or more points use a Theil-Sen median slope and median intercept.
    Exactly two points intentionally use their constant midpoint (the usual
    two-value median) so that a free slope cannot interpolate two noisy annual maps. One point is returned
    unchanged and is subsequently assigned zero temporal confidence.
    """
    batch, years, height, width = values.shape
    if years <= 1:
        return values.clone()
    if years == 2:
        level = values.mean(dim=1, keepdim=True)
        return level.expand(batch, years, height, width).clone()

    pairwise_slopes = []
    for left in range(years - 1):
        for right in range(left + 1, years):
            pairwise_slopes.append(
                (values[:, right] - values[:, left]) / float(right - left)
            )
    slope = torch.stack(pairwise_slopes, dim=1).median(dim=1).values
    slope = torch.clamp(slope, min=float(slope_min), max=float(slope_max))
    x = torch.arange(years, device=values.device, dtype=values.dtype).view(
        1, years, 1, 1
    )
    intercept = (values - slope.unsqueeze(1) * x).median(dim=1).values
    intercept = torch.clamp(intercept, min=0.0, max=float(max_intercept))
    return intercept.unsqueeze(1) + slope.unsqueeze(1) * x


def robust_pseudo_target(
    reference_sequence: torch.Tensor,
    disturbance_index: torch.Tensor,
    *,
    slope_min: float = 0.0,
    slope_max: float = 2.0,
    max_intercept_after_disturbance: float = 100.0,
    min_segment_points: int = 2,
    short_segment_confidence: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build robust pseudo-targets and per-pixel confidence for T>=4.

    Stable pixels receive one robust fit over all years with confidence 1.
    A disturbed pixel is fitted piecewise only when both sides contain at
    least two observations. A 2+3 or 3+2 split receives confidence 0.5. If a
    side has fewer than two points, confidence is zero and the temporal loss
    does not supervise that pixel.
    """
    if reference_sequence.ndim != 4:
        raise ValueError("reference_sequence must have shape [B,Y,H,W]")
    if disturbance_index.shape != (
        reference_sequence.shape[0],
        reference_sequence.shape[2],
        reference_sequence.shape[3],
    ):
        raise ValueError("disturbance_index must have shape [B,H,W]")

    with torch.no_grad():
        _, years, _, _ = reference_sequence.shape
        fitted = _constrained_robust_line(
            reference_sequence,
            slope_min=slope_min,
            slope_max=slope_max,
            max_intercept=max_intercept_after_disturbance,
        )
        confidence = torch.ones_like(reference_sequence[:, 0])

        for split in range(1, years):
            mask = disturbance_index == split
            if not bool(mask.any()):
                continue
            n_before = int(split)
            n_after = int(years - split)
            if n_before < int(min_segment_points) or n_after < int(min_segment_points):
                confidence = torch.where(mask, torch.zeros_like(confidence), confidence)
                continue

            before = _constrained_robust_line(
                reference_sequence[:, :split],
                slope_min=slope_min,
                slope_max=slope_max,
                max_intercept=max_intercept_after_disturbance,
            )
            after = _constrained_robust_line(
                reference_sequence[:, split:],
                slope_min=slope_min,
                slope_max=slope_max,
                max_intercept=max_intercept_after_disturbance,
            )
            candidate = torch.cat([before, after], dim=1)
            fitted = torch.where(mask.unsqueeze(1), candidate, fitted)
            split_confidence = (
                1.0
                if n_before >= 3 and n_after >= 3
                else float(short_segment_confidence)
            )
            confidence = torch.where(
                mask,
                torch.full_like(confidence, split_confidence),
                confidence,
            )
        return fitted, confidence


def persistent_loss_flags(
    reference_sequence: torch.Tensor,
    *,
    drop_threshold_m: float = 5.0,
    consecutive_flags: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return running maximum, yearly loss flags, and persistent-window flags.

    Parameters
    ----------
    reference_sequence:
        Frozen B4 CHM sequence with shape ``[B, Y, H, W]``.
    drop_threshold_m:
        A yearly value is flagged when it is at least this far below the
        maximum of all preceding years.
    consecutive_flags:
        Number K of consecutive yearly loss flags required for a persistent
        event. K=2 reproduces the original persistent-pair rule.

    Notes
    -----
    The first year cannot be a loss flag because it has no previous year.
    ``persistent_windows[:, t]`` is true when all K flags starting at t are
    true. The returned event index is therefore the start of the persistent
    run, which is also the piecewise-regression split used downstream.
    No GEDI target is used to build this mask.
    """
    if reference_sequence.ndim != 4:
        raise ValueError(
            "reference_sequence must have shape [B,Y,H,W], got "
            f"{tuple(reference_sequence.shape)}"
        )
    if float(drop_threshold_m) <= 0:
        raise ValueError("drop_threshold_m must be positive")
    required = int(consecutive_flags)
    if required != consecutive_flags or required < 1:
        raise ValueError("consecutive_flags must be a positive integer")

    batch, years, height, width = reference_sequence.shape
    running_before = torch.empty_like(reference_sequence)
    running_before[:, 0] = reference_sequence[:, 0]
    if years > 1:
        running_before[:, 1:] = torch.cummax(
            reference_sequence[:, :-1], dim=1
        ).values

    loss_flags = (reference_sequence - running_before) <= -float(drop_threshold_m)
    loss_flags[:, 0] = False
    if years >= required:
        persistent_windows = loss_flags.unfold(1, required, 1).all(dim=-1)
    else:
        persistent_windows = torch.zeros(
            (batch, 0, height, width),
            dtype=torch.bool,
            device=reference_sequence.device,
        )
    return running_before, loss_flags, persistent_windows


def persistent_disturbance_index(
    reference_sequence: torch.Tensor,
    *,
    drop_threshold_m: float = 5.0,
    consecutive_flags: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the first persistent-loss year, with 0 reserved for no event."""
    with torch.no_grad():
        _, _, windows = persistent_loss_flags(
            reference_sequence,
            drop_threshold_m=drop_threshold_m,
            consecutive_flags=consecutive_flags,
        )
        batch, _, height, width = reference_sequence.shape
        if windows.shape[1] == 0:
            index = torch.zeros(
                (batch, height, width),
                dtype=torch.long,
                device=reference_sequence.device,
            )
        else:
            has_event = windows.any(dim=1)
            index = windows.to(torch.int64).argmax(dim=1)
            index = torch.where(has_event, index, torch.zeros_like(index))
        # The figure defines a pixel-wise mask, so no 3x3 surround dilation is
        # applied. The official parent currently ignores this second output.
        surround = torch.zeros_like(index)
        return index, surround


def build_growth_loss(
    *,
    official_repo: Path,
    disturbance_rule: str,
    persistent_drop_m: float = 5.0,
    persistent_required_consecutive_flags: int = 2,
    disturbance_indicator: float = -1.0,
    slope_min: float = 0.0,
    slope_max: float = 2.0,
    full_disturbance_window: bool = True,
    use_l2: bool = True,
    max_intercept_after_disturbance: float = 100.0,
    disturbance_factor: float = 1.0,
    no_disturbance_factor: float = 1.0,
    slope_no_disturbance: float = -0.0,
):
    """Instantiate the official loss with either its native or persistent mask."""
    disturbance_rule = str(disturbance_rule).strip().lower()
    if disturbance_rule not in SUPPORTED_RULES:
        raise ValueError(
            f"Unsupported disturbance_rule={disturbance_rule!r}; "
            f"choose one of {sorted(SUPPORTED_RULES)}"
        )

    required_flags = int(persistent_required_consecutive_flags)
    if required_flags != persistent_required_consecutive_flags or required_flags < 1:
        raise ValueError("persistent_required_consecutive_flags must be a positive integer")

    OfficialGrowthLoss, official_source = load_official_growth_loss(official_repo)

    if disturbance_rule == "official_echosat":
        LossClass = OfficialGrowthLoss
    elif disturbance_rule == "persistent_running_max":
        threshold = float(persistent_drop_m)

        class PersistentRunningMaxGrowthLoss(OfficialGrowthLoss):
            def get_disturbance_idx(self, out0):
                return persistent_disturbance_index(
                    out0,
                    drop_threshold_m=threshold,
                    consecutive_flags=required_flags,
                )

        LossClass = PersistentRunningMaxGrowthLoss
    else:
        threshold = float(persistent_drop_m)

        class PersistentRobustGrowthLoss(OfficialGrowthLoss):
            def get_disturbance_idx(self, out0):
                return persistent_disturbance_index(
                    out0,
                    drop_threshold_m=threshold,
                    consecutive_flags=required_flags,
                )

            def get_pseudo_target_and_confidence(self, out0):
                disturbance_idx, _ = self.get_disturbance_idx(out0)
                return robust_pseudo_target(
                    out0,
                    disturbance_idx,
                    slope_min=self.slope_min,
                    slope_max=self.slope_max,
                    max_intercept_after_disturbance=self.max_intercept_after_disturbance,
                    min_segment_points=2,
                    short_segment_confidence=0.5,
                )

            def forward(self, out, target):
                del target  # GEDI is never used to construct the pseudo-target.
                prediction = out[:, 1]
                fitted, confidence = self.get_pseudo_target_and_confidence(out[:, 0])
                residual = fitted - prediction
                per_year = residual.square() if self.use_l2 else residual.abs()
                per_pixel = per_year.mean(dim=1)
                weighted = per_pixel * confidence
                return weighted.sum() / confidence.sum().clamp_min(1.0)

        LossClass = PersistentRobustGrowthLoss

    loss = LossClass(
        disturbance_indicator=disturbance_indicator,
        slope_min=slope_min,
        slope_max=slope_max,
        full_disturbance_window=full_disturbance_window,
        use_l2=use_l2,
        max_intercept_after_disturbance=max_intercept_after_disturbance,
        disturbance_factor=disturbance_factor,
        no_disturbance_factor=no_disturbance_factor,
        slope_no_disturbance=slope_no_disturbance,
    )
    adapter_source = Path(__file__).resolve()
    provenance: Dict[str, object] = {
        "disturbance_rule": disturbance_rule,
        "official_growth_loss_source": str(Path(official_source).resolve()),
        "official_growth_loss_sha256": sha256_file(official_source),
        "persistent_adapter_source": str(adapter_source),
        "persistent_adapter_sha256": sha256_file(adapter_source),
        "persistent_drop_m": float(persistent_drop_m),
        "persistent_required_consecutive_flags": required_flags,
        "persistent_running_max_excludes_current_year": True,
        "persistent_spatial_dilation": False,
        "gedi_filter_at_10m": False,
    }
    if disturbance_rule == "persistent_running_max_robust":
        provenance.update(
            {
                "pseudo_target_fit": "constrained_theil_sen",
                "stable_min_points": 3,
                "disturbed_min_points_per_segment": 2,
                "two_point_segment_fit": "constant_midpoint",
                "short_segment_confidence": 0.5,
                "invalid_segment_confidence": 0.0,
            }
        )
    return loss, provenance


def mask_summary(
    reference_sequence: torch.Tensor,
    *,
    drop_threshold_m: float = 5.0,
    consecutive_flags: int = 2,
) -> Dict[str, float]:
    """Small read-only diagnostic used by the notebook preflight."""
    _, flags, windows = persistent_loss_flags(
        reference_sequence,
        drop_threshold_m=drop_threshold_m,
        consecutive_flags=consecutive_flags,
    )
    persistent = windows.any(dim=1) if windows.shape[1] else torch.zeros_like(flags[:, 0])
    return {
        "yearly_loss_flag_fraction": float(flags.float().mean().detach().cpu()),
        "persistent_pixel_fraction": float(persistent.float().mean().detach().cpu()),
        "persistent_pixel_count": float(persistent.sum().detach().cpu()),
        "pixel_count": float(persistent.numel()),
    }
