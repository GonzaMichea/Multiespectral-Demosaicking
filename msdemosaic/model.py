"""Small U-Net for multispectral demosaicking (< 2 M parameters)."""
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_BANDS = 16


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Two-level U-Net.

    use_mask=True: input is [Y, M] (1 + 16 channels), the pattern-agnostic setup.
    use_mask=False: input is the mosaic Y only (fixed-pattern baseline).
    """

    def __init__(self, use_mask: bool, base: int = 48):
        super().__init__()
        self.use_mask = use_mask
        self.base = base
        c_in = 1 + NUM_BANDS if use_mask else 1
        self.enc1 = _block(c_in, base)
        self.enc2 = _block(base, 2 * base)
        self.bottleneck = _block(2 * base, 4 * base)
        self.up2 = nn.ConvTranspose2d(4 * base, 2 * base, 2, stride=2)
        self.dec2 = _block(4 * base, 2 * base)
        self.up1 = nn.ConvTranspose2d(2 * base, base, 2, stride=2)
        self.dec1 = _block(2 * base, base)
        self.out = nn.Conv2d(base, NUM_BANDS, 1)

    def forward(self, y: torch.Tensor, m: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_mask:
            if m is None:
                raise ValueError("this network was built with use_mask=True and needs the mask")
            x = torch.cat([y, m], dim=1)
        else:
            x = y
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
            "use_mask": model.use_mask,
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
    model = UNet(use_mask=ck["use_mask"], base=ck["base"])
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()

    def method(mosaic: np.ndarray, mask: np.ndarray) -> np.ndarray:
        y = torch.from_numpy(np.ascontiguousarray(mosaic, dtype=np.float32))[None, None].to(device)
        m = None
        if model.use_mask:
            m = torch.from_numpy(np.ascontiguousarray(mask, dtype=np.float32)).permute(2, 0, 1)[None].to(device)
        with torch.no_grad():
            out = model(y, m)
        return out[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)

    return method
