from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================================
# 1) BASIC BLOCKS
# ======================================================================================
class DoubleConv(nn.Module):
    """Two 3x3 conv blocks with BN/ReLU and optional dropout between the 2 convs."""

    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        if in_ch <= 0 or out_ch <= 0:
            raise ValueError(f"in_ch and out_ch must be > 0, got in_ch={in_ch}, out_ch={out_ch}")
        if not (0.0 <= float(dropout) <= 1.0):
            raise ValueError(f"dropout must be in [0,1], got {dropout}")

        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if float(dropout) > 0.0:
            layers.append(nn.Dropout2d(p=float(dropout)))
        layers += [
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionGate(nn.Module):
    """Standard attention gate used on skip connections."""

    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        if min(F_g, F_l, F_int) <= 0:
            raise ValueError(f"AttentionGate channels must be > 0, got F_g={F_g}, F_l={F_l}, F_int={F_int}")

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if g.shape[-2:] != x.shape[-2:]:
            g = F.interpolate(g, size=x.shape[-2:], mode="bilinear", align_corners=False)
        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(psi)
        return x * psi


# ======================================================================================
# 2) HELPERS
# ======================================================================================
def _match_spatial(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Resize x to ref spatial size only when needed."""
    if x.shape[-2:] != ref.shape[-2:]:
        x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
    return x


# ======================================================================================
# 3) ATTENTION U-NET — MAIN MODEL
# ======================================================================================
class CanopyHyTecModel(nn.Module):
    """
    Attention U-Net for dense canopy-height regression (RH95).

    Notes
    -----
    - Sparse GEDI supervision is handled outside the model via aux_rows / aux_cols / aux_y / aux_mask.
    - Default base_ch is set to 64 to allow a 1024-channel bottleneck (64 -> 128 -> 256 -> 512 -> 1024),
      but you should still control it from the run block for fair ablations.
    - Output activation is Softplus to keep predictions positive and smooth.
    """

    def __init__(
        self,
        n_channels: int = 8,
        n_classes: int = 1,
        dropout: float = 0.25,
        base_ch: int = 64,
        use_attention: bool = True,
    ):
        super().__init__()
        self.n_channels = int(n_channels)
        self.n_classes = int(n_classes)
        self.dropout = float(dropout)
        self.base_ch = int(base_ch)
        self.use_attention = bool(use_attention)

        if self.n_channels <= 0:
            raise ValueError(f"n_channels must be > 0, got {self.n_channels}")
        if self.n_classes <= 0:
            raise ValueError(f"n_classes must be > 0, got {self.n_classes}")
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError(f"dropout must be in [0,1], got {self.dropout}")
        if self.base_ch <= 0:
            raise ValueError(f"base_ch must be > 0, got {self.base_ch}")

        b = self.base_ch
        self.bottleneck_channels = b * 16

        # Encoder
        self.inc = DoubleConv(self.n_channels, b, dropout=0.0)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(b, b * 2, dropout=self.dropout))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(b * 2, b * 4, dropout=self.dropout))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(b * 4, b * 8, dropout=self.dropout))
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(b * 8, b * 16, dropout=min(self.dropout * 2.0, 0.4)),
        )

        # Decoder
        self.up1 = nn.ConvTranspose2d(b * 16, b * 8, kernel_size=2, stride=2)
        self.att1 = AttentionGate(b * 8, b * 8, b * 4)
        self.convup1 = DoubleConv(b * 16, b * 8, dropout=self.dropout)

        self.up2 = nn.ConvTranspose2d(b * 8, b * 4, kernel_size=2, stride=2)
        self.att2 = AttentionGate(b * 4, b * 4, b * 2)
        self.convup2 = DoubleConv(b * 8, b * 4, dropout=self.dropout)

        self.up3 = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.att3 = AttentionGate(b * 2, b * 2, max(1, b))
        self.convup3 = DoubleConv(b * 4, b * 2, dropout=self.dropout)

        self.up4 = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.att4 = AttentionGate(b, b, max(1, b // 2))
        self.convup4 = DoubleConv(b * 2, b, dropout=0.0)

        self.outc = nn.Conv2d(b, self.n_classes, kernel_size=1)
        self.act = nn.Softplus(beta=5.0)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _gate(self, att: AttentionGate, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if self.use_attention:
            return att(g, x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)       # b
        x2 = self.down1(x1)    # 2b
        x3 = self.down2(x2)    # 4b
        x4 = self.down3(x3)    # 8b
        x5 = self.down4(x4)    # 16b

        d5 = self.up1(x5)
        d5 = _match_spatial(d5, x4)
        x4a = self._gate(self.att1, d5, x4)
        d5 = self.convup1(torch.cat([x4a, d5], dim=1))

        d4 = self.up2(d5)
        d4 = _match_spatial(d4, x3)
        x3a = self._gate(self.att2, d4, x3)
        d4 = self.convup2(torch.cat([x3a, d4], dim=1))

        d3 = self.up3(d4)
        d3 = _match_spatial(d3, x2)
        x2a = self._gate(self.att3, d3, x2)
        d3 = self.convup3(torch.cat([x2a, d3], dim=1))

        d2 = self.up4(d3)
        d2 = _match_spatial(d2, x1)
        x1a = self._gate(self.att4, d2, x1)
        d2 = self.convup4(torch.cat([x1a, d2], dim=1))

        out = self.outc(d2)
        return self.act(out)


# ======================================================================================
# 4) TRANSFER LEARNING — EFFICIENTNET-B4 ENCODER
# ======================================================================================
def _adapt_first_conv_weights(pretrained_weight: torch.Tensor, n_channels: int) -> torch.Tensor:
    out_ch, in_ch, _, _ = pretrained_weight.shape
    if in_ch != 3:
        raise ValueError(f"Expected RGB conv with 3 input channels, got {in_ch}")
    if n_channels <= 0:
        raise ValueError(f"n_channels must be > 0, got {n_channels}")

    n_repeat = math.ceil(n_channels / 3)
    repeated = pretrained_weight.repeat(1, n_repeat, 1, 1)
    adapted = repeated[:, :n_channels, :, :]
    adapted = adapted * (3.0 / max(1, n_channels))
    return adapted.contiguous()


class CanopyTLModel(nn.Module):
    """EfficientNet-B4 encoder + lightweight U-Net-like decoder."""

    def __init__(
        self,
        n_channels: int = 8,
        pretrained: bool = True,
        freeze_encoder_epochs: int = 5,
        dropout: float = 0.25,
        timm_model_name: str = "efficientnet_b4",
    ):
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError("timm is required for CanopyTLModel. Install with: pip install timm") from e

        self.n_channels = int(n_channels)
        self.freeze_encoder_epochs = int(freeze_encoder_epochs)
        self.dropout = float(dropout)
        self.timm_model_name = str(timm_model_name or "efficientnet_b4")
        self._current_epoch = 0

        if self.n_channels <= 0:
            raise ValueError(f"n_channels must be > 0, got {self.n_channels}")
        if not (0.0 <= self.dropout <= 1.0):
            raise ValueError(f"dropout must be in [0,1], got {self.dropout}")

        self.encoder = timm.create_model(
            self.timm_model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3, 4),
            in_chans=3,
        )

        old_conv = self.encoder.conv_stem
        with torch.no_grad():
            adapted_w = _adapt_first_conv_weights(old_conv.weight.data, self.n_channels)

        new_conv = nn.Conv2d(
            self.n_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None),
        )
        new_conv.weight = nn.Parameter(adapted_w)
        if old_conv.bias is not None:
            new_conv.bias = nn.Parameter(old_conv.bias.data.clone())
        self.encoder.conv_stem = new_conv

        enc_chs = [info["num_chs"] for info in self.encoder.feature_info]
        c1, c2, c3, c4 = enc_chs

        self.up1 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.att1 = AttentionGate(c3, c3, max(1, c3 // 2))
        self.dec1 = DoubleConv(c3 * 2, c3, dropout=self.dropout)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.att2 = AttentionGate(c2, c2, max(1, c2 // 2))
        self.dec2 = DoubleConv(c2 * 2, c2, dropout=self.dropout)

        self.up3 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.att3 = AttentionGate(c1, c1, max(1, c1 // 2))
        self.dec3 = DoubleConv(c1 * 2, c1, dropout=self.dropout)

        self.up4 = nn.ConvTranspose2d(c1, 32, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(32, 32, dropout=0.0)

        self.up5 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec5 = DoubleConv(16, 16, dropout=0.0)

        self.outc = nn.Conv2d(16, 1, kernel_size=1)
        self.act = nn.Softplus(beta=5.0)

        if self.freeze_encoder_epochs > 0:
            self._freeze_encoder()

    def _freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def _unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True

    def on_epoch_start(self, epoch: int):
        self._current_epoch = int(epoch)
        if self.freeze_encoder_epochs > 0 and int(epoch) == self.freeze_encoder_epochs + 1:
            self._unfreeze_encoder()
            print(f"[TL] Epoch {epoch}: EfficientNet-B4 encoder unfrozen ✅", flush=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        f1, f2, f3, f4 = feats

        d = self.up1(f4)
        d = _match_spatial(d, f3)
        d = self.dec1(torch.cat([self.att1(d, f3), d], dim=1))

        d = self.up2(d)
        d = _match_spatial(d, f2)
        d = self.dec2(torch.cat([self.att2(d, f2), d], dim=1))

        d = self.up3(d)
        d = _match_spatial(d, f1)
        d = self.dec3(torch.cat([self.att3(d, f1), d], dim=1))

        d = self.dec4(self.up4(d))
        d = self.dec5(self.up5(d))
        d = _match_spatial(d, x)

        return self.act(self.outc(d))


# ======================================================================================
# 5) OPTIONAL TRANSFER LEARNING — DINOv3
# ======================================================================================
class CanopyDinoV3TLModel(nn.Module):
    """Optional DINOv3 ViT-L/16 transfer-learning model."""

    def __init__(
        self,
        n_channels: int = 8,
        pretrained: bool = True,
        pretrained_source: Optional[str] = None,
        freeze_encoder_epochs: int = 5,
        dropout: float = 0.25,
        local_files_only: bool = False,
        patch_size: int = 16,
    ):
        super().__init__()
        try:
            from transformers import AutoConfig, AutoModel
        except ImportError as e:
            raise ImportError("transformers is required for CanopyDinoV3TLModel. Install with: pip install transformers") from e

        self.n_channels = int(n_channels)
        self.freeze_encoder_epochs = int(freeze_encoder_epochs)
        self.dropout = float(dropout)
        self.patch_size = int(patch_size)
        self._current_epoch = 0

        if self.n_channels <= 0:
            raise ValueError(f"n_channels must be > 0, got {self.n_channels}")
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be > 0, got {self.patch_size}")

        source = pretrained_source or "facebook/dinov3-vitl16-pretrain-sat493m"

        self.input_adapter = nn.Sequential(
            nn.Conv2d(self.n_channels, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1),
        )

        if pretrained:
            self.encoder = AutoModel.from_pretrained(source, local_files_only=bool(local_files_only))
        else:
            cfg = AutoConfig.from_pretrained(source, local_files_only=bool(local_files_only))
            self.encoder = AutoModel.from_config(cfg)

        hidden_size = int(getattr(self.encoder.config, "hidden_size", 1024))
        self.num_register_tokens = int(getattr(self.encoder.config, "num_register_tokens", 0))

        self.proj = nn.Sequential(
            nn.Conv2d(hidden_size, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 128, dropout=self.dropout)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(64, 64, dropout=self.dropout)
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(32, 32, dropout=self.dropout)
        self.up4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(16, 16, dropout=0.0)

        self.outc = nn.Conv2d(16, 1, kernel_size=1)
        self.act = nn.Softplus(beta=5.0)

        if self.freeze_encoder_epochs > 0:
            self._freeze_encoder()

    def _freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def _unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True

    def on_epoch_start(self, epoch: int):
        self._current_epoch = int(epoch)
        if self.freeze_encoder_epochs > 0 and int(epoch) == self.freeze_encoder_epochs + 1:
            self._unfreeze_encoder()
            print(f"[TL] Epoch {epoch}: DINOv3 encoder unfrozen ✅", flush=True)

    def _tokens_to_map(self, patch_tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        ph = h // self.patch_size
        pw = w // self.patch_size
        b, n, c = patch_tokens.shape
        expected = ph * pw
        if n != expected:
            raise RuntimeError(
                f"DINO token count mismatch: got {n}, expected {expected} for HxW={h}x{w} and patch={self.patch_size}"
            )
        fmap = patch_tokens.transpose(1, 2).contiguous().view(b, c, ph, pw)
        return fmap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        if (h % self.patch_size) != 0 or (w % self.patch_size) != 0:
            raise ValueError(
                f"Input spatial size must be divisible by patch_size={self.patch_size}. Got HxW={h}x{w}"
            )

        x_rgb = self.input_adapter(x)
        outputs = self.encoder(pixel_values=x_rgb, return_dict=True)
        tokens = outputs.last_hidden_state

        start = 1 + self.num_register_tokens
        patch_tokens = tokens[:, start:, :]

        fmap = self._tokens_to_map(patch_tokens, h, w)
        d = self.proj(fmap)
        d = self.dec1(self.up1(d))
        d = self.dec2(self.up2(d))
        d = self.dec3(self.up3(d))
        d = self.dec4(self.up4(d))
        d = _match_spatial(d, x)

        return self.act(self.outc(d))




# ======================================================================================
# 6) EXTERNAL FOUNDATION BACKBONES — GENERIC DENSE CHM ADAPTERS
# ======================================================================================
class CanopyHFViTTLModel(nn.Module):
    """Generic Hugging Face vision-transformer adapter for dense CHM regression.

    The adapter is intentionally conservative: it maps the active CHM input stack
    to 3 channels, runs a pretrained HF encoder, converts patch tokens/features to
    a feature map, and decodes to a one-channel canopy-height map.

    Native model-specific heads may perform better, but this wrapper lets us run a
    fair first benchmark inside the existing GEDI/STEP07 pipeline.
    """

    DEFAULT_SOURCE: Optional[str] = None
    MODEL_FAMILY: str = "hf_external"

    def __init__(
        self,
        n_channels: int = 8,
        pretrained: bool = True,
        pretrained_source: Optional[str] = None,
        freeze_encoder_epochs: int = 5,
        dropout: float = 0.25,
        local_files_only: bool = False,
        patch_size: int = 16,
        trust_remote_code: bool = True,
    ):
        super().__init__()
        try:
            from transformers import AutoConfig, AutoModel
        except ImportError as e:
            raise ImportError(
                f"transformers is required for {self.__class__.__name__}. "
                "Install with: pip install transformers"
            ) from e

        self.n_channels = int(n_channels)
        self.freeze_encoder_epochs = int(freeze_encoder_epochs)
        self.dropout = float(dropout)
        self.patch_size = int(patch_size)
        self.local_files_only = bool(local_files_only)
        self.trust_remote_code = bool(trust_remote_code)
        self._current_epoch = 0

        if self.n_channels <= 0:
            raise ValueError(f"n_channels must be > 0, got {self.n_channels}")
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be > 0, got {self.patch_size}")

        source = pretrained_source or self.DEFAULT_SOURCE
        if not source:
            raise ValueError(
                f"{self.__class__.__name__} needs --pretrained-source because no safe default source is configured."
            )
        self.pretrained_source = str(source)

        self.input_adapter = nn.Sequential(
            nn.Conv2d(self.n_channels, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, kernel_size=1),
        )

        if pretrained:
            self.encoder = AutoModel.from_pretrained(
                self.pretrained_source,
                local_files_only=self.local_files_only,
                trust_remote_code=self.trust_remote_code,
            )
        else:
            cfg = AutoConfig.from_pretrained(
                self.pretrained_source,
                local_files_only=self.local_files_only,
                trust_remote_code=self.trust_remote_code,
            )
            self.encoder = AutoModel.from_config(cfg, trust_remote_code=self.trust_remote_code)

        cfg = getattr(self.encoder, "config", None)
        hidden_size = self._infer_hidden_size(cfg)
        self.num_register_tokens = int(getattr(cfg, "num_register_tokens", 0)) if cfg is not None else 0

        self.proj = nn.Sequential(
            nn.Conv2d(hidden_size, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 128, dropout=self.dropout)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(64, 64, dropout=self.dropout)
        self.up3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(32, 32, dropout=self.dropout)
        self.up4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(16, 16, dropout=0.0)
        self.outc = nn.Conv2d(16, 1, kernel_size=1)
        self.act = nn.Softplus(beta=5.0)

        if self.freeze_encoder_epochs > 0:
            self._freeze_encoder()

    @staticmethod
    def _infer_hidden_size(cfg) -> int:
        if cfg is None:
            return 768
        for name in ("hidden_size", "embed_dim", "encoder_embed_dim", "d_model", "dim"):
            value = getattr(cfg, name, None)
            if value is not None:
                try:
                    return int(value)
                except Exception:
                    pass
        return 768

    def _freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False

    def _unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True

    def on_epoch_start(self, epoch: int):
        self._current_epoch = int(epoch)
        if self.freeze_encoder_epochs > 0 and int(epoch) == self.freeze_encoder_epochs + 1:
            self._unfreeze_encoder()
            print(
                f"[TL] Epoch {epoch}: {self.MODEL_FAMILY} encoder unfrozen",
                flush=True,
            )

    def _call_encoder(self, x_rgb: torch.Tensor):
        try:
            return self.encoder(pixel_values=x_rgb, return_dict=True)
        except TypeError:
            try:
                return self.encoder(x_rgb, return_dict=True)
            except TypeError:
                return self.encoder(x_rgb)

    @staticmethod
    def _first_tensor(obj):
        if torch.is_tensor(obj):
            return obj
        if hasattr(obj, "last_hidden_state") and torch.is_tensor(obj.last_hidden_state):
            return obj.last_hidden_state
        if hasattr(obj, "feature_maps"):
            maps = getattr(obj, "feature_maps")
            if isinstance(maps, (list, tuple)) and maps:
                for item in reversed(maps):
                    if torch.is_tensor(item):
                        return item
        if hasattr(obj, "hidden_states"):
            states = getattr(obj, "hidden_states")
            if isinstance(states, (list, tuple)) and states:
                for item in reversed(states):
                    if torch.is_tensor(item):
                        return item
        if isinstance(obj, dict):
            for key in ("last_hidden_state", "features", "feature_map", "x"):
                value = obj.get(key)
                if torch.is_tensor(value):
                    return value
            for value in obj.values():
                if torch.is_tensor(value):
                    return value
        if isinstance(obj, (list, tuple)):
            for value in obj:
                t = CanopyHFViTTLModel._first_tensor(value)
                if torch.is_tensor(t):
                    return t
        return None

    def _tokens_to_map(self, tokens: torch.Tensor, h: int, w: int) -> torch.Tensor:
        if tokens.ndim != 3:
            raise RuntimeError(f"Expected token tensor [B,N,C], got shape={tuple(tokens.shape)}")
        b, n, c = tokens.shape
        ph = max(1, int(math.ceil(h / self.patch_size)))
        pw = max(1, int(math.ceil(w / self.patch_size)))
        expected = ph * pw

        if n >= expected:
            patch_tokens = tokens[:, -expected:, :]
            return patch_tokens.transpose(1, 2).contiguous().view(b, c, ph, pw)

        side = int(math.sqrt(n))
        if side * side == n:
            return tokens.transpose(1, 2).contiguous().view(b, c, side, side)

        usable = int(math.sqrt(n)) ** 2
        if usable <= 0:
            raise RuntimeError(f"Cannot reshape token sequence of length {n} into a spatial map")
        side = int(math.sqrt(usable))
        patch_tokens = tokens[:, -usable:, :]
        return patch_tokens.transpose(1, 2).contiguous().view(b, c, side, side)

    def _output_to_map(self, outputs, h: int, w: int) -> torch.Tensor:
        feat = self._first_tensor(outputs)
        if feat is None:
            raise RuntimeError(f"{self.MODEL_FAMILY} encoder did not return a tensor feature")
        if feat.ndim == 4:
            return feat
        if feat.ndim == 3:
            return self._tokens_to_map(feat, h, w)
        raise RuntimeError(f"Unsupported {self.MODEL_FAMILY} feature shape: {tuple(feat.shape)}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        x_rgb = self.input_adapter(x)
        outputs = self._call_encoder(x_rgb)
        fmap = self._output_to_map(outputs, h, w)

        d = self.proj(fmap)
        d = self.dec1(self.up1(d))
        d = self.dec2(self.up2(d))
        d = self.dec3(self.up3(d))
        d = self.dec4(self.up4(d))
        d = _match_spatial(d, x)
        return self.act(self.outc(d))


class CanopyDOFATLModel(CanopyHFViTTLModel):
    MODEL_FAMILY = "DOFA"
    DEFAULT_SOURCE = "MVRL/DOFA-base"


class CanopyClayTLModel(CanopyHFViTTLModel):
    MODEL_FAMILY = "Clay"
    DEFAULT_SOURCE = "made-with-clay/Clay"


class CanopyTerraMindTLModel(CanopyHFViTTLModel):
    MODEL_FAMILY = "TerraMind"
    DEFAULT_SOURCE = "ibm-nasa-geospatial/TerraMind-1.0-base"


class CanopyScaleMAETLModel(CanopyHFViTTLModel):
    MODEL_FAMILY = "Scale-MAE"
    DEFAULT_SOURCE = None


class CanopyPrithviTLModel(CanopyHFViTTLModel):
    MODEL_FAMILY = "Prithvi"
    DEFAULT_SOURCE = "ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL"


class CanopySatlasTLModel(CanopyHFViTTLModel):
    MODEL_FAMILY = "Satlas"
    DEFAULT_SOURCE = "allenai/satlaspretrain"


# ======================================================================================
# 6) HY-TEC LOSS CONFIG CONTAINER
# ======================================================================================
class HyTecLossV6(nn.Module):
    """
    Lightweight container for hybrid-loss hyperparameters and buffers.

    IMPORTANT:
    The actual sparse GEDI loss is computed in trainloop.py, which reads these attributes:
      - alpha_reg
      - beta_ord
      - ord_thresholds
      - reg_bin_edges
      - reg_bin_weights
      - ord_pos_weight
      - temperature
      - huber_beta
      - logits_clip
      - eps
      - patch_weight_mode
      - patch_count_cap
      - gamma_cls
      - cls_bin_weights
    """

    def __init__(
        self,
        alpha_reg: float = 1.0,
        beta_ord: float = 0.2,
        ord_thresholds: Sequence[float] = (3.0, 10.0, 20.0, 30.0),
        temperature: float = 4.0,
        huber_beta: float = 1.0,
        reg_bin_edges: Sequence[float] = (3.0, 10.0, 20.0, 30.0),
        reg_bin_weights: Sequence[float] = (1.0, 1.0, 1.0, 1.0, 1.0),
        ord_pos_weight: Sequence[float] | float = (1.0, 1.0, 1.0, 1.0),
        logits_clip: float = 15.0,
        eps: float = 1e-6,
        patch_weight_mode: str = "sqrt_n",
        patch_count_cap: float = 16.0,
        gamma_cls: float = 0.0,
        cls_bin_weights: Optional[Sequence[float]] = None,
    ):
        super().__init__()

        self.alpha_reg = float(alpha_reg)
        self.beta_ord = float(beta_ord)
        self.temperature = float(temperature)
        self.huber_beta = float(huber_beta)
        self.logits_clip = float(logits_clip)
        self.eps = float(eps)
        self.patch_weight_mode = str(patch_weight_mode)
        self.patch_count_cap = float(patch_count_cap)

        if self.alpha_reg < 0.0:
            raise ValueError(f"alpha_reg must be >= 0, got {alpha_reg}")
        if self.beta_ord < 0.0:
            raise ValueError(f"beta_ord must be >= 0, got {beta_ord}")
        if self.temperature <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if self.huber_beta <= 0.0:
            raise ValueError(f"huber_beta must be > 0, got {huber_beta}")
        if self.logits_clip <= 0.0:
            raise ValueError(f"logits_clip must be > 0, got {logits_clip}")
        if self.eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")

        if self.patch_weight_mode not in {"equal", "sqrt_n", "count", "clipped_count"}:
            raise ValueError(
                f"patch_weight_mode must be one of ['equal', 'sqrt_n', 'count', 'clipped_count'], got {self.patch_weight_mode}"
            )
        if self.patch_count_cap <= 0.0:
            raise ValueError(f"patch_count_cap must be > 0, got {patch_count_cap}")

        thr = torch.tensor(list(ord_thresholds), dtype=torch.float32)
        if thr.numel() < 1:
            raise ValueError("ord_thresholds must contain at least one threshold")
        if bool(torch.any(thr[1:] <= thr[:-1])):
            raise ValueError(f"ord_thresholds must be strictly increasing, got {tuple(float(x) for x in thr)}")
        self.register_buffer("ord_thresholds", thr)

        edges = tuple(float(x) for x in reg_bin_edges)
        weights = tuple(float(x) for x in reg_bin_weights)
        if len(weights) != len(edges) + 1:
            raise ValueError(
                f"reg_bin_weights must have len(reg_bin_edges)+1. Got edges={edges} weights={weights}"
            )
        if any(edges[i] >= edges[i + 1] for i in range(len(edges) - 1)):
            raise ValueError(f"reg_bin_edges must be strictly increasing. Got: {edges}")

        self.reg_bin_edges = edges
        self.reg_bin_weights = weights
        self.reg_crit = nn.SmoothL1Loss(reduction="none", beta=self.huber_beta)

        K = int(thr.numel())
        if isinstance(ord_pos_weight, (list, tuple, np.ndarray)):
            if len(ord_pos_weight) != K:
                raise ValueError(f"ord_pos_weight must have length {K}, got {len(ord_pos_weight)}")
            pw = torch.tensor(list(ord_pos_weight), dtype=torch.float32)
        else:
            pw = torch.full((K,), float(ord_pos_weight), dtype=torch.float32)
        self.register_buffer("ord_pos_weight", pw)

        if float(gamma_cls) < 0.0:
            raise ValueError(f"gamma_cls must be >= 0, got {gamma_cls}")
        self.gamma_cls = float(gamma_cls)

        n_classes = len(self.reg_bin_edges) + 1
        if cls_bin_weights is None:
            cbw = torch.tensor(list(self.reg_bin_weights), dtype=torch.float32)
        else:
            cbw = torch.tensor(list(cls_bin_weights), dtype=torch.float32)

        if int(cbw.numel()) != int(n_classes):
            raise ValueError(
                f"cls_bin_weights must have {n_classes} values (one per class), got {cbw.numel()}"
            )
        self.register_buffer("cls_bin_weights", cbw)

    def extra_repr(self) -> str:
        return (
            f"alpha_reg={self.alpha_reg}, beta_ord={self.beta_ord}, "
            f"temperature={self.temperature}, huber_beta={self.huber_beta}, "
            f"patch_weight_mode='{self.patch_weight_mode}', patch_count_cap={self.patch_count_cap}, "
            f"n_ord={int(self.ord_thresholds.numel())}, n_classes={len(self.reg_bin_edges)+1}"
        )


# ======================================================================================
# 7) COMPATIBILITY ALIASES
# ======================================================================================
MaamouraHyTecModel = CanopyHyTecModel
MaamouraTLModel = CanopyTLModel
MaamouraDinoV3TLModel = CanopyDinoV3TLModel
MaamouraDOFATLModel = CanopyDOFATLModel
MaamouraSatlasTLModel = CanopySatlasTLModel
MaamouraClayTLModel = CanopyClayTLModel
MaamouraTerraMindTLModel = CanopyTerraMindTLModel
MaamouraScaleMAETLModel = CanopyScaleMAETLModel
MaamouraPrithviTLModel = CanopyPrithviTLModel

ArganHyTecModel = CanopyHyTecModel
ArganTLModel = CanopyTLModel
ArganDinoV3TLModel = CanopyDinoV3TLModel
ArganDOFATLModel = CanopyDOFATLModel
ArganSatlasTLModel = CanopySatlasTLModel
ArganClayTLModel = CanopyClayTLModel
ArganTerraMindTLModel = CanopyTerraMindTLModel
ArganScaleMAETLModel = CanopyScaleMAETLModel
ArganPrithviTLModel = CanopyPrithviTLModel


__all__ = [
    "DoubleConv",
    "AttentionGate",
    "CanopyHyTecModel",
    "CanopyTLModel",
    "CanopyDinoV3TLModel",
    "CanopyPrithviTLModel",
    "CanopyScaleMAETLModel",
    "CanopyTerraMindTLModel",
    "CanopyClayTLModel",
    "CanopySatlasTLModel",
    "CanopyDOFATLModel",
    "CanopyHFViTTLModel",
    "MaamouraHyTecModel",
    "MaamouraTLModel",
    "MaamouraDinoV3TLModel",
    "ArganHyTecModel",
    "ArganTLModel",
    "ArganDinoV3TLModel",
    "ArganPrithviTLModel",
    "ArganScaleMAETLModel",
    "ArganTerraMindTLModel",
    "ArganClayTLModel",
    "ArganSatlasTLModel",
    "ArganDOFATLModel",
    "MaamouraPrithviTLModel",
    "MaamouraScaleMAETLModel",
    "MaamouraTerraMindTLModel",
    "MaamouraClayTLModel",
    "MaamouraSatlasTLModel",
    "MaamouraDOFATLModel",
    "HyTecLossV6",
]
