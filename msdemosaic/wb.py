"""Weighted bilinear (WB) demosaicking, Brauers & Aach 2006.

Generalised to arbitrary patterns with normalised convolution: each band's
sparse samples are blurred with a 7x7 triangular kernel and divided by the
blurred sampling mask. For 4x4 periodic patterns this is the classic WB.
"""
import numpy as np
from scipy.ndimage import convolve

_F1 = np.array([1, 2, 3, 4, 3, 2, 1], dtype=np.float64)
_KERNEL = np.outer(_F1, _F1)


def wb(mosaic: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """mosaic (H, W), mask (H, W, B) one-hot -> recon (H, W, B) float32."""
    recon = np.empty(mask.shape, dtype=np.float32)
    for k in range(mask.shape[-1]):
        m = mask[..., k].astype(np.float64)
        num = convolve(mosaic * m, _KERNEL, mode="constant")
        den = convolve(m, _KERNEL, mode="constant")
        recon[..., k] = num / np.maximum(den, 1e-8)
    measured = mask == 1
    recon[measured] = np.broadcast_to(mosaic[..., None], mask.shape)[measured]
    return recon
