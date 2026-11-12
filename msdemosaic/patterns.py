"""MSFA patterns: band-index maps of shape (H, W) with values in 0..15.

Every pattern is anchored at the origin (0, 0) of the image or patch it is
generated for, so train patches and full test images share the same phase.
"""
import numpy as np

NUM_BANDS = 16
TEST_PATTERNS = ("imec", "osp", "btes", "random")

# 4x4 kernels, 0-indexed (the slides list them 1-indexed).
_KERNELS = {
    "imec": [[6, 7, 5, 4], [14, 15, 13, 12], [10, 11, 9, 8], [2, 3, 1, 0]],
    "btes": [[0, 12, 3, 14], [8, 4, 10, 6], [2, 15, 1, 13], [11, 7, 9, 5]],
    "random": [[10, 2, 13, 5], [0, 8, 15, 3], [12, 6, 1, 11], [4, 14, 7, 9]],
}


def _tile(kernel: np.ndarray, H: int, W: int) -> np.ndarray:
    reps = (-(-H // 4), -(-W // 4))
    return np.tile(kernel, reps)[:H, :W].astype(np.int64)


def make_pattern(name: str, H: int, W: int) -> np.ndarray:
    """Return one of the four test patterns as an (H, W) int64 band-index map."""
    if name in _KERNELS:
        return _tile(np.array(_KERNELS[name]), H, W)
    if name == "osp":
        # Optimal sphere packing lattice (Diaz et al. 2023), a=7, b=4.
        i, j = np.mgrid[0:H, 0:W]
        return ((7 * (i + 1) + 4 * (j + 1)) % NUM_BANDS).astype(np.int64)
    raise ValueError(f"unknown pattern {name!r}; expected one of {TEST_PATTERNS}")


def sample_training_pattern(rng: np.random.Generator, H: int, W: int) -> np.ndarray:
    """Sample a random balanced pattern for training the pattern-agnostic network.

    With probability 0.5 a random 4x4 kernel is tiled ("tile" mode); otherwise
    every aligned 4x4 block gets an independent permutation ("block" mode).
    Tiled kernels equal to a held-out test kernel are rejected.
    """
    if H % 4 or W % 4:
        raise ValueError("H and W must be multiples of 4")
    if rng.random() < 0.5:
        forbidden = [np.array(k) for k in _KERNELS.values()]
        while True:
            kernel = rng.permutation(NUM_BANDS).reshape(4, 4)
            if not any((kernel == f).all() for f in forbidden):
                return _tile(kernel, H, W)
    nh, nw = H // 4, W // 4
    blocks = rng.permuted(np.tile(np.arange(NUM_BANDS), (nh * nw, 1)), axis=1)
    return blocks.reshape(nh, nw, 4, 4).transpose(0, 2, 1, 3).reshape(H, W).astype(np.int64)


def pattern_to_mask(pattern: np.ndarray) -> np.ndarray:
    """One-hot mask (H, W, 16), float32."""
    return np.eye(NUM_BANDS, dtype=np.float32)[pattern]
