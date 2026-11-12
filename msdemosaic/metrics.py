"""Reconstruction metrics on cubes of shape (H, W, B) in [0, 1]."""
import numpy as np
from skimage.metrics import structural_similarity

_MSE_FLOOR = 1e-10  # caps PSNR at 100 dB
_EPS = 1e-8


def _psnr_from_mse(mse):
    return 10.0 * np.log10(1.0 / np.maximum(mse, _MSE_FLOOR))


def psnr(recon: np.ndarray, cube: np.ndarray) -> float:
    diff = recon.astype(np.float64) - cube.astype(np.float64)
    return float(_psnr_from_mse(np.mean(diff**2)))


def psnr_per_band(recon: np.ndarray, cube: np.ndarray) -> np.ndarray:
    diff = recon.astype(np.float64) - cube.astype(np.float64)
    return _psnr_from_mse(np.mean(diff**2, axis=(0, 1)))


def ssim(recon: np.ndarray, cube: np.ndarray) -> float:
    return float(
        np.mean(
            [
                structural_similarity(cube[..., k], recon[..., k], data_range=1.0)
                for k in range(cube.shape[-1])
            ]
        )
    )


def sam(recon: np.ndarray, cube: np.ndarray) -> float:
    """Spectral angle mapper in degrees, averaged over pixels."""
    a = recon.astype(np.float64)
    b = cube.astype(np.float64)
    dot = np.sum(a * b, axis=-1)
    # eps inside each norm keeps all-zero (black) pixels at angle 0 instead of NaN/90.
    na = np.sqrt(np.sum(a * a, axis=-1) + _EPS)
    nb = np.sqrt(np.sum(b * b, axis=-1) + _EPS)
    cos = np.clip((dot + _EPS) / (na * nb), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)).mean())


def evaluate(recon: np.ndarray, cube: np.ndarray) -> dict:
    recon = np.clip(recon, 0.0, 1.0)
    return {"psnr": psnr(recon, cube), "ssim": ssim(recon, cube), "sam": sam(recon, cube)}
