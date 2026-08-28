from __future__ import annotations

from typing import Dict

import torch
from torch import nn


AOI_CHANNEL_INDEX = 12
AOI_THRESHOLD = 0.5


def prepare_aoi_masked_modules(engine) -> Dict[str, object]:
    """Return the retained Phase-2 modules with AOI-masked temporal loss.

    The original source modules and completed checkpoints remain untouched.  A third,
    non-trainable model output carries the AOI support from input channel 12 to the
    temporal loss.  The two scientific prediction heads retain their original indices.
    """
    modules = engine.import_growth_modules()
    BaseModel = modules["B4EchoSatTwoHead"]

    class B4EchoSatTwoHeadAOI(BaseModel):
        def forward(self, sequence: torch.Tensor) -> torch.Tensor:
            out = super().forward(sequence)  # [B,2,T,H,W]
            aoi = sequence[:, :, AOI_CHANNEL_INDEX]
            if not bool(torch.isfinite(aoi).all()):
                raise RuntimeError("Non-finite AOI support encountered in Phase 2 input")
            aoi = (aoi > AOI_THRESHOLD).to(dtype=out.dtype)
            return torch.cat([out, aoi.unsqueeze(1)], dim=1)  # [B,3,T,H,W]

    from src import persistent_growth_loss_v6 as base_loss_module
    from src import persistent_training_v6 as training_module

    original_builder = base_loss_module.build_growth_loss

    def build_growth_loss_aoi_masked(**kwargs):
        base_loss, provenance = original_builder(**kwargs)

        class AOIMaskedTemporalLoss(nn.Module):
            def __init__(self, base):
                super().__init__()
                self.base = base

            def forward(self, out, target):
                if out.ndim != 5 or out.shape[1] < 3:
                    raise ValueError(f"Expected AOI-carrying output [B,3,T,H,W], got {tuple(out.shape)}")
                prediction = out[:, 1]
                # Delegate the breakpoint and fitted-trajectory construction to the
                # retained loss.  Only its final spatial reduction is changed.
                fitted, disturbance_mask, slope_no_disturbance = self.base.get_regression(out, target)
                aoi = out[:, 2] > AOI_THRESHOLD
                finite = torch.isfinite(prediction) & torch.isfinite(fitted)
                valid = aoi & finite
                per_year = (fitted - prediction).square() if self.base.use_l2 else (fitted - prediction).abs()
                factor = torch.ones_like(disturbance_mask, dtype=per_year.dtype)
                if self.base.no_disturbance_factor != 1:
                    factor = torch.where(
                        slope_no_disturbance < self.base.slope_no_disturbance,
                        factor * self.base.no_disturbance_factor, factor,
                    )
                if self.base.disturbance_factor != 1:
                    factor = torch.where(
                        disturbance_mask, factor * self.base.disturbance_factor, factor,
                    )
                weights = valid.to(per_year.dtype) * factor.unsqueeze(1)
                return (per_year * weights).sum() / weights.sum().clamp_min(1.0)

        provenance = dict(provenance)
        provenance.update({
            "temporal_support": "AOI_masked",
            "aoi_channel_index_zero_based": AOI_CHANNEL_INDEX,
            "aoi_threshold": AOI_THRESHOLD,
            "normalisation": "valid_AOI_confidence_weighted_pixel_years",
        })
        return AOIMaskedTemporalLoss(base_loss), provenance

    training_module.build_growth_loss = build_growth_loss_aoi_masked
    modules["B4EchoSatTwoHead"] = B4EchoSatTwoHeadAOI
    modules["train"] = training_module.train_phase1_direct_temporal_stage
    modules["aoi_channel_index"] = AOI_CHANNEL_INDEX
    modules["aoi_threshold"] = AOI_THRESHOLD
    return modules
