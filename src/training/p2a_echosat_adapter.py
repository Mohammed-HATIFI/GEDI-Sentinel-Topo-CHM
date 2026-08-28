from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from source_snapshot.b4_model_snapshot import CanopyHyTecModel


def _strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if state and all(str(key).startswith("module.") for key in state):
        return {str(key)[7:]: value for key, value in state.items()}
    return {str(key): value for key, value in state.items()}


def _match_spatial(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    if x.shape[-2:] != reference.shape[-2:]:
        x = F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)
    return x


class AntiShrinkTransferReference(nn.Module):
    """Exact P2A transfer head used by Last_Ablation - Copie.ipynb.

    The parameter names intentionally match the checkpoints written by
    06_train_growthloss_catalog.py (``base_model.*``, ``residual_head.*``,
    ``gate_head.*``, ``high_focus_head.*`` and the four scalar parameters).
    """

    def __init__(
        self,
        base_model: CanopyHyTecModel,
        *,
        hidden_ch: int = 32,
        gate_bias: float = -2.0,
        high_threshold: float = 20.0,
        init_high_gain: float = 0.0,
        init_global_scale: float = 1.0,
        init_residual_scale: float = 1.0,
        freeze_base: bool = False,
    ):
        super().__init__()
        self.base_model = base_model
        channels = int(base_model.base_ch)
        hidden_ch = int(hidden_ch)
        self.residual_head = nn.Sequential(
            nn.Conv2d(channels, hidden_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, 1, kernel_size=1),
        )
        self.gate_head = nn.Sequential(
            nn.Conv2d(channels, hidden_ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, 1, kernel_size=1),
        )
        self.high_focus_head = nn.Sequential(
            nn.Conv2d(channels, hidden_ch, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_ch, 1, kernel_size=1),
        )
        nn.init.constant_(self.gate_head[-1].bias, float(gate_bias))
        self.global_scale = nn.Parameter(torch.tensor(float(init_global_scale)))
        self.global_bias = nn.Parameter(torch.tensor(0.0))
        self.residual_scale = nn.Parameter(torch.tensor(float(init_residual_scale)))
        self.high_gain = nn.Parameter(torch.tensor(float(init_high_gain)))
        self.high_threshold = float(high_threshold)
        if freeze_base:
            for parameter in self.base_model.parameters():
                parameter.requires_grad = False

    def predict_from_decoder_features(
        self, decoder_features: torch.Tensor, base_prediction: torch.Tensor
    ) -> torch.Tensor:
        residual = self.residual_head(decoder_features).squeeze(1)
        gate = torch.sigmoid(self.gate_head(decoder_features)).squeeze(1)
        high_focus = torch.sigmoid(self.high_focus_head(decoder_features)).squeeze(1)
        high_base = F.softplus(base_prediction - self.high_threshold)
        correction = (
            self.residual_scale * gate * residual
            + self.high_gain * high_focus * high_base
        )
        return self.global_scale * base_prediction + self.global_bias + correction

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features, base_prediction = _decode_base_features(self.base_model, x)
        return self.predict_from_decoder_features(features, base_prediction).unsqueeze(1)


@torch.no_grad()
def _decode_base_features(
    reference: CanopyHyTecModel, x: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    x1 = reference.inc(x)
    x2 = reference.down1(x1)
    x3 = reference.down2(x2)
    x4 = reference.down3(x3)
    x5 = reference.down4(x4)

    d5 = _match_spatial(reference.up1(x5), x4)
    d5 = reference.convup1(torch.cat([reference._gate(reference.att1, d5, x4), d5], dim=1))
    d4 = _match_spatial(reference.up2(d5), x3)
    d4 = reference.convup2(torch.cat([reference._gate(reference.att2, d4, x3), d4], dim=1))
    d3 = _match_spatial(reference.up3(d4), x2)
    d3 = reference.convup3(torch.cat([reference._gate(reference.att3, d3, x2), d3], dim=1))
    d2 = _match_spatial(reference.up4(d3), x1)
    d2 = reference.convup4(torch.cat([reference._gate(reference.att4, d2, x1), d2], dim=1))
    base_prediction = reference.act(reference.outc(d2)).squeeze(1)
    return d2.detach(), base_prediction.detach()


def load_p2a_reference(
    checkpoint_path: Path,
    *,
    n_channels: int = 15,
    base_ch: int = 64,
    dropout: float = 0.15,
    hidden_ch: int = 32,
    gate_bias: float = -2.0,
    high_threshold: float = 20.0,
    device: str | torch.device = "cpu",
) -> AntiShrinkTransferReference:
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    base_model = CanopyHyTecModel(
        n_channels=int(n_channels),
        n_classes=1,
        dropout=float(dropout),
        base_ch=int(base_ch),
        use_attention=True,
    )
    reference = AntiShrinkTransferReference(
        base_model,
        hidden_ch=int(hidden_ch),
        gate_bias=float(gate_bias),
        high_threshold=float(high_threshold),
        init_high_gain=0.0,
        init_global_scale=1.0,
        init_residual_scale=1.0,
        freeze_base=False,
    )
    raw = torch.load(str(checkpoint_path), map_location="cpu")
    state = raw.get("model", raw.get("state_dict", raw)) if isinstance(raw, dict) else raw
    incompatible = reference.load_state_dict(_strip_module_prefix(state), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"P2A checkpoint mismatch: missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    reference.to(device)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad = False
    return reference


class P2AEchoSatTwoHead(nn.Module):
    """Frozen P2A reference followed by a fresh ECHOSAT Conv3D residual head."""

    def __init__(
        self,
        reference: AntiShrinkTransferReference,
        *,
        head_mode: str = "residual",
        residual_scale: float = 1.0,
    ):
        super().__init__()
        if head_mode not in {"absolute", "residual"}:
            raise ValueError(f"Invalid head_mode={head_mode!r}")
        self.reference = reference
        self.head_mode = str(head_mode)
        self.residual_scale = float(residual_scale)
        for parameter in self.reference.parameters():
            parameter.requires_grad = False
        self.reference.eval()

        channels = int(reference.base_model.base_ch)
        if channels % 4:
            raise ValueError(f"ECHOSAT GroupNorm needs base_ch divisible by 4, got {channels}")
        self.prediction_head = nn.Sequential(
            nn.Conv3d(channels, channels, 3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(4, channels),
            nn.ReLU(),
            nn.Conv3d(channels, channels, 3, padding=1, padding_mode="replicate"),
            nn.GroupNorm(4, channels),
            nn.ReLU(),
            nn.Conv3d(channels, 1, 1),
        )
        if self.head_mode == "residual":
            nn.init.zeros_(self.prediction_head[-1].weight)
            nn.init.zeros_(self.prediction_head[-1].bias)

    def train(self, mode: bool = True):
        super().train(mode)
        self.reference.eval()
        return self

    @torch.no_grad()
    def _reference_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features, base_prediction = _decode_base_features(self.reference.base_model, x)
        p2a_prediction = self.reference.predict_from_decoder_features(
            features, base_prediction
        )
        return features.detach(), p2a_prediction.detach()

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 5:
            raise ValueError(f"Expected [B,Y,C,H,W], got {tuple(sequence.shape)}")
        features = []
        references = []
        for year_idx in range(sequence.shape[1]):
            feat, z_ref = self._reference_features(sequence[:, year_idx])
            features.append(feat)
            references.append(z_ref)
        feature_cube = torch.stack(features, dim=2)
        z_ref = torch.stack(references, dim=1)
        correction = self.prediction_head(feature_cube).squeeze(1)
        z_pred = (
            z_ref + self.residual_scale * correction
            if self.head_mode == "residual"
            else correction
        )
        return torch.stack([z_ref, z_pred], dim=1)

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        return self.prediction_head.parameters()

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)
