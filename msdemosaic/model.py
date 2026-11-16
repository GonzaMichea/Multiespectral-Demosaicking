"""Small U-Net for multispectral demosaicking (< 2 M parameters)."""
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_BANDS = 16

# What the network is fed (the mosaic Y and the one-hot mask M are always the call arguments):
#   "wb"          [WB(Y, M), Y*M, M], 48 channels. Default for every network. The weighted
#                 bilinear estimate does the pattern-dependent interpolation, so the network
#                 only has to refine it; with the other inputs the pattern-agnostic network
#                 learns several times slower (see docs/revision-2026-11-16.md).
#   "mosaic_mask" [Y, M], 17 channels. The input of the original proposal; kept for ablation.
#   "mosaic"      Y only. A conventional network tied to one pattern; kept as a baseline.
INPUT_MODES = ("wb", "mosaic_mask", "mosaic")
_IN_CHANNELS = {"wb": 3 * NUM_BANDS, "mosaic_mask": 1 + NUM_BANDS, "mosaic": 1}

_WB_1D = (1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0)


def weighted_bilinear(sparse: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Torch version of `msdemosaic.wb.wb` (normalised 7x7 triangular convolution).

    sparse = Y * M and mask, both (B, 16, H, W). Measured samples keep their value.
    """
    bands = sparse.shape[1]
    f = torch.tensor(_WB_1D, dtype=sparse.dtype, device=sparse.device)
    kernel = torch.outer(f, f)[None, None].repeat(bands, 1, 1, 1)
    num = F.conv2d(sparse, kernel, padding=3, groups=bands)
    den = F.conv2d(mask, kernel, padding=3, groups=bands).clamp_min(1e-8)
    return torch.where(mask > 0, sparse, num / den)


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Two-level U-Net: forward(y (B, 1, H, W), m (B, 16, H, W)) -> cube (B, 16, H, W)."""

    def __init__(self, input_mode: str = "wb", base: int = 48):
        super().__init__()
        if input_mode not in INPUT_MODES:
            raise ValueError(f"unknown input mode {input_mode!r}; expected one of {INPUT_MODES}")
        self.input_mode = input_mode
        self.base = base
        self.enc1 = _block(_IN_CHANNELS[input_mode], base)
        self.enc2 = _block(base, 2 * base)
        self.bottleneck = _block(2 * base, 4 * base)
        self.up2 = nn.ConvTranspose2d(4 * base, 2 * base, 2, stride=2)
        self.dec2 = _block(4 * base, 2 * base)
        self.up1 = nn.ConvTranspose2d(2 * base, base, 2, stride=2)
        self.dec1 = _block(2 * base, base)
        self.out = nn.Conv2d(base, NUM_BANDS, 1)

    def _network_input(self, y: torch.Tensor, m: torch.Tensor | None) -> torch.Tensor:
        if self.input_mode == "mosaic":
            return y
        if m is None:
            raise ValueError(f"input mode {self.input_mode!r} needs the mask")
        if self.input_mode == "mosaic_mask":
            return torch.cat([y, m], dim=1)
        sparse = y * m
        return torch.cat([weighted_bilinear(sparse, m), sparse, m], dim=1)

    def forward(self, y: torch.Tensor, m: torch.Tensor | None = None) -> torch.Tensor:
        x = self._network_input(y, m)
        # Pad to a multiple of 4 (two poolings) and crop back, so any H, W works.
        H, W = x.shape[-2:]
        pad_h, pad_w = (-H) % 4, (-W) % 4
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        b = self.bottleneck(F.max_pool2d(e2, 2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)[..., :H, :W]


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_checkpoint(path, model: UNet, *, train_pattern: str, iters: int, val_psnr: float) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "input_mode": model.input_mode,
            "base": model.base,
            "train_pattern": train_pattern,
            "iters": iters,
            "val_psnr": float(val_psnr),
        },
        path,
    )


def load_method(ckpt_path, device: torch.device | None = None) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    """Wrap a checkpoint as method(mosaic (H, W), mask (H, W, 16)) -> recon (H, W, 16)."""
    device = device or pick_device()
    ck = torch.load(ckpt_path, map_location="cpu")
    model = UNet(input_mode=ck["input_mode"], base=ck["base"])
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()

    def method(mosaic: np.ndarray, mask: np.ndarray) -> np.ndarray:
        y = torch.from_numpy(np.ascontiguousarray(mosaic, dtype=np.float32))[None, None].to(device)
        m = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32)).permute(2, 0, 1)[None].to(device)
        with torch.no_grad():
            out = model(y, m)
        return out[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)

    return method
