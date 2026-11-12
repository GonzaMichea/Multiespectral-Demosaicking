"""Datasets: CAVE (31-band PNG stacks) and ARAD-1K (.mat), reduced to 16 bands."""
import re
from pathlib import Path

import imageio.v3 as imageio
import numpy as np

BAND_IDX = list(range(0, 31, 2))  # 16 of 31 bands: 400, 420, ..., 700 nm
_CAVE_BAND = re.compile(r"_ms_(\d{2})\.png$")


def select_bands(cube31: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(cube31[..., BAND_IDX])


def load_cave_scene(scene_dir: Path) -> np.ndarray:
    """Load one CAVE scene as (512, 512, 31) float32 in [0, 1]."""
    bands = {}
    for path in Path(scene_dir).rglob("*.png"):
        match = _CAVE_BAND.search(path.name)
        if match:
            bands[int(match.group(1))] = path
    if sorted(bands) != list(range(1, 32)):
        raise ValueError(f"{scene_dir}: expected bands 01..31, found {sorted(bands)}")
    planes = []
    for b in range(1, 32):
        img = imageio.imread(bands[b])
        if img.ndim == 3:  # a few CAVE files are stored as RGB(A)
            img = img[..., 0]
        scale = 65535.0 if img.dtype == np.uint16 else 255.0
        planes.append(img.astype(np.float32) / scale)
    cube = np.stack(planes, axis=-1)
    if cube.shape != (512, 512, 31):
        raise ValueError(f"{scene_dir}: unexpected shape {cube.shape}")
    return cube


def load_arad_scene(mat_path: Path) -> np.ndarray:
    """Load one ARAD-1K scene as (482, 512, 31) float32 in [0, 1]."""
    import h5py

    with h5py.File(mat_path, "r") as f:
        cube = np.array(f["cube"], dtype=np.float32)
    # MATLAB v7.3 files come back with reversed axes: (31, W, H) -> (H, W, 31).
    if cube.shape[0] == 31:
        cube = cube.transpose(2, 1, 0)
    if cube.shape != (482, 512, 31):
        raise ValueError(f"{mat_path}: unexpected shape {cube.shape}")
    return np.clip(cube, 0.0, 1.0)


def _cave_dirs(data_root: Path) -> list[Path]:
    root = Path(data_root) / "CAVE"
    dirs = sorted(d for d in root.glob("*_ms") if d.is_dir())
    if len(dirs) != 32:
        raise FileNotFoundError(f"expected 32 CAVE scenes in {root}, found {len(dirs)}")
    return dirs


def list_scenes(data_root: Path, data_mode: str) -> dict[str, list[Path]]:
    """Scene paths per split.

    "arad": train/val on ARAD-1K (220/30), test on the 32 CAVE scenes.
    "cave-split": CAVE only, fixed shuffle (seed 0) into 21/3/8.
    """
    cave = _cave_dirs(data_root)
    if data_mode == "cave-split":
        order = np.random.default_rng(0).permutation(32)
        shuffled = [cave[i] for i in order]
        return {"train": shuffled[:21], "val": shuffled[21:24], "test": shuffled[24:]}
    if data_mode == "arad":
        spec = Path(data_root) / "ARAD_1K" / "Train_Spec"
        mats = sorted(spec.glob("*.mat"))[:250]
        if len(mats) < 250:
            raise FileNotFoundError(f"expected >= 250 .mat files in {spec}, found {len(mats)}")
        return {"train": mats[:220], "val": mats[220:], "test": cave}
    raise ValueError(f"unknown data mode {data_mode!r}; expected 'arad' or 'cave-split'")


def load_cube16(path: Path, cache_dir: Path) -> np.ndarray:
    """Load a scene as a 16-band float32 cube, cached on disk as float16."""
    path, cache_dir = Path(path), Path(cache_dir)
    cached = cache_dir / f"{path.stem}.npy"
    if cached.exists():
        return np.load(cached).astype(np.float32)
    cube31 = load_cave_scene(path) if path.is_dir() else load_arad_scene(path)
    cube = select_bands(cube31)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cached, cube.astype(np.float16))
    return cube.astype(np.float16).astype(np.float32)


def sample_patch(cubes: list[np.ndarray], rng: np.random.Generator, patch: int = 64) -> np.ndarray:
    """Random crop of a random scene with random flips and 90-degree rotations."""
    cube = cubes[rng.integers(len(cubes))]
    H, W = cube.shape[:2]
    i = rng.integers(H - patch + 1)
    j = rng.integers(W - patch + 1)
    p = cube[i : i + patch, j : j + patch]
    if rng.random() < 0.5:
        p = p[::-1]
    if rng.random() < 0.5:
        p = p[:, ::-1]
    p = np.rot90(p, k=int(rng.integers(4)))
    return np.ascontiguousarray(p, dtype=np.float32)
