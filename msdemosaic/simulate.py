"""Capture model: Y = sum_k X_k * C_k (ideal filters, no noise)."""
import numpy as np

from .patterns import pattern_to_mask


def simulate(cube: np.ndarray, pattern: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Simulate an MSFA capture.

    cube: (H, W, 16) float32 in [0, 1]; pattern: (H, W) int band indices.
    Returns (mosaic (H, W) float32, mask (H, W, 16) float32 one-hot).
    """
    if cube.shape[:2] != pattern.shape:
        raise ValueError(f"cube {cube.shape} and pattern {pattern.shape} do not match")
    mask = pattern_to_mask(pattern)
    mosaic = np.take_along_axis(cube, pattern[..., None], axis=-1)[..., 0].astype(np.float32)
    return mosaic, mask
