from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from source_snapshot.b4_model_snapshot import CanopyHyTecModel


def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if state and all(str(k).startswith("module.") for k in state):
        return {str(k)[7:]: v for k, v in state.items()}
    return {str(k): v for k, v in state.items()}


def load_b4_reference(
    checkpoint_path: Path,
    *,
    n_channels: int = 15,
    base_ch: int = 64,
    dropout: float = 0.15,
    device: str | torch.device = "cpu",
) -> CanopyHyTecModel:
    """Instantiate the exact B4 U-Net and load the frozen Phase-1 checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    model = CanopyHyTecModel(
        n_channels=int(n_channels),
        n_classes=1,
        dropout=float(dropout),
        base_ch=int(base_ch),
        use_attention=True,
    )
    raw = torch.load(str(checkpoint_path), map_location="cpu")
    state = raw.get("model", raw.get("state_dict", raw)) if isinstance(raw, dict) else raw
    incompatible = model.load_state_dict(_strip_module_prefix(state), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    model.to(device)
    model.eval()
    return model


def _match_spatial(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] != ref.shape[-2:]:
        x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
    return x


class B4EchoSatTwoHead(nn.Module):
    """ECHOSAT two-head protocol adapted to the pretrained B4 decoder features.

    The B4 reference model is fully frozen.  Its original 1x1 reference head is
    retained.  The trainable prediction head is the same three-layer Conv3D head
    used in the official ECHOSAT repository.

    Input:  ``[B, Y, C, H, W]``
    Output: ``[B, 2, Y, H, W]`` where head 0 is reference and head 1 prediction.
    """

    def __init__(
        self,
        reference: CanopyHyTecModel,
        *,
        head_mode: str = "absolute",
        residual_scale: float = 1.0,
    ):
        super().__init__()
        if head_mode not in {"absolute", "residual"}:
            raise ValueError(f"head_mode must be 'absolute' or 'residual', got {head_mode!r}")
        self.head_mode = str(head_mode)
        self.residual_scale = float(residual_scale)
        self.reference = reference
        for parameter in self.reference.parameters():
            parameter.requires_grad = False
        self.reference.eval()

        channels = int(reference.base_ch)
        if channels % 4:
            raise ValueError(f"ECHOSAT GroupNorm requires base_ch divisible by 4, got {channels}")
        self.prediction_head = nn.Sequential(
            nn.Conv3d(
                channels,
                channels,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
                padding_mode="replicate",
            ),
            nn.GroupNorm(4, channels),
            nn.ReLU(),
            nn.Conv3d(
                channels,
                channels,
                kernel_size=(3, 3, 3),
                padding=(1, 1, 1),
                padding_mode="replicate",
            ),
            nn.GroupNorm(4, channels),
            nn.ReLU(),
            nn.Conv3d(channels, 1, kernel_size=(1, 1, 1)),
        )
        if self.head_mode == "residual":
            # Start exactly from B4. The first optimizer step updates the final
            # layer; subsequent steps propagate into the preceding Conv3D blocks.
            nn.init.zeros_(self.prediction_head[-1].weight)
            nn.init.zeros_(self.prediction_head[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.reference.eval()
        return self

    @torch.no_grad()
    def _reference_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        ref = self.reference
        x1 = ref.inc(x)
        x2 = ref.down1(x1)
        x3 = ref.down2(x2)
        x4 = ref.down3(x3)
        x5 = ref.down4(x4)

        d5 = _match_spatial(ref.up1(x5), x4)
        d5 = ref.convup1(torch.cat([ref._gate(ref.att1, d5, x4), d5], dim=1))
        d4 = _match_spatial(ref.up2(d5), x3)
        d4 = ref.convup2(torch.cat([ref._gate(ref.att2, d4, x3), d4], dim=1))
        d3 = _match_spatial(ref.up3(d4), x2)
        d3 = ref.convup3(torch.cat([ref._gate(ref.att3, d3, x2), d3], dim=1))
        d2 = _match_spatial(ref.up4(d3), x1)
        d2 = ref.convup4(torch.cat([ref._gate(ref.att4, d2, x1), d2], dim=1))
        z_ref = ref.act(ref.outc(d2)).squeeze(1)
        return d2.detach(), z_ref.detach()

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 5:
            raise ValueError(f"Expected [B,Y,C,H,W], got {tuple(sequence.shape)}")
        _, years, _, _, _ = sequence.shape
        features = []
        references = []
        for year_idx in range(years):
            feat, z_ref = self._reference_features(sequence[:, year_idx])
            features.append(feat)
            references.append(z_ref)
        feature_cube = torch.stack(features, dim=2)  # [B,E,Y,H,W]
        z_ref = torch.stack(references, dim=1)       # [B,Y,H,W]
        correction = self.prediction_head(feature_cube).squeeze(1)
        if self.head_mode == "residual":
            z_pred = z_ref + self.residual_scale * correction
        else:
            z_pred = correction
        return torch.stack([z_ref, z_pred], dim=1)

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return self.prediction_head.parameters()

    def trainable_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
