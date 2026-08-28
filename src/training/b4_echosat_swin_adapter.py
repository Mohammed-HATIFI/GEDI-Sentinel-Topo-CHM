from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Dict, Tuple

import torch
import torch.nn as nn

from .b4_echosat_adapter import B4EchoSatTwoHead


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_official_swin_module(official_repo: Path) -> Tuple[ModuleType, Path, str]:
    """Load EchoSat's vendored Swin implementation without altering sys.path."""
    source = (
        Path(official_repo)
        / "fine-tuning"
        / "models"
        / "swin_video_unet.py"
    ).resolve()
    if not source.exists():
        raise FileNotFoundError(source)
    source_hash = sha256_file(source)
    module_name = f"echosat_official_swin_{source_hash[:16]}"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load official EchoSat Swin source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "BasicLayer3D"):
        raise AttributeError(f"BasicLayer3D missing from {source}")
    return module, source, source_hash


class EchoSatOfficialSwinPredictionHead(nn.Module):
    """Matched Swin-3D temporal head operating on frozen B4 decoder features.

    Input and output follow the Conv3D-head contract used by
    :class:`B4EchoSatTwoHead`:

    - input: ``[B, E, T, H, W]``
    - output: ``[B, 1, T, H, W]``

    The Swin blocks come directly from the vendored official EchoSat source.
    The final projection is zero initialized so residual mode starts exactly at
    the frozen B4 reference prediction.
    """

    def __init__(
        self,
        *,
        official_repo: Path,
        channels: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        window_size: Tuple[int, int, int] = (2, 6, 6),
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        if int(channels) % int(num_heads):
            raise ValueError(
                f"channels={channels} must be divisible by num_heads={num_heads}"
            )
        official, source, source_hash = load_official_swin_module(official_repo)
        self.channels = int(channels)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.window_size = tuple(int(value) for value in window_size)
        self.official_source = str(source)
        self.official_source_sha256 = source_hash

        self.swin = official.BasicLayer3D(
            dim=self.channels,
            depth=self.depth,
            num_heads=self.num_heads,
            window_size=self.window_size,
            mlp_ratio=float(mlp_ratio),
            qkv_bias=True,
            drop=float(drop),
            attn_drop=float(attn_drop),
            drop_path=float(drop_path),
            norm_layer=nn.LayerNorm,
            downsample=False,
        )
        self.norm = nn.LayerNorm(self.channels)
        self.output_projection = nn.Linear(self.channels, 1)
        nn.init.zeros_(self.output_projection.weight)
        nn.init.zeros_(self.output_projection.bias)

    def forward(self, feature_cube: torch.Tensor) -> torch.Tensor:
        if feature_cube.ndim != 5:
            raise ValueError(
                f"Expected feature cube [B,E,T,H,W], got {tuple(feature_cube.shape)}"
            )
        if int(feature_cube.shape[1]) != self.channels:
            raise ValueError(
                f"Expected E={self.channels}, got E={int(feature_cube.shape[1])}"
            )
        x = feature_cube.permute(0, 2, 3, 4, 1).contiguous()
        x, _ = self.swin(x)
        x = self.output_projection(self.norm(x))
        return x.permute(0, 4, 1, 2, 3).contiguous()

    def provenance(self) -> Dict[str, object]:
        return {
            "head_type": "EchoSatOfficialSwinPredictionHead",
            "official_swin_source": self.official_source,
            "official_swin_source_sha256": self.official_source_sha256,
            "channels": self.channels,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "window_size": list(self.window_size),
            "output_projection_zero_initialized": True,
        }


def build_b4_echosat_swin_two_head(
    reference,
    *,
    official_repo: Path,
    head_mode: str = "residual",
    residual_scale: float = 1.0,
    depth: int = 2,
    num_heads: int = 4,
    window_size: Tuple[int, int, int] = (2, 6, 6),
) -> Tuple[B4EchoSatTwoHead, Dict[str, object]]:
    """Create frozen-B4 two-head model with an official EchoSat Swin head."""
    model = B4EchoSatTwoHead(
        reference,
        head_mode=head_mode,
        residual_scale=float(residual_scale),
    )
    model.prediction_head = EchoSatOfficialSwinPredictionHead(
        official_repo=official_repo,
        channels=int(reference.base_ch),
        depth=int(depth),
        num_heads=int(num_heads),
        window_size=window_size,
    )
    return model, model.prediction_head.provenance()
