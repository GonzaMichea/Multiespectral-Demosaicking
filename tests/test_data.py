import imageio.v3 as imageio
import numpy as np
import pytest

from msdemosaic.data import list_scenes, load_cave_scene, load_cube16, sample_patch, select_bands


def _fake_cave(root, name, dtype=np.uint16, rgba=False, nested=True):
    d = root / f"{name}_ms" / f"{name}_ms" if nested else root / f"{name}_ms"
    d.mkdir(parents=True)
    for b in range(1, 32):
        img = np.full((512, 512), b * 1000 if dtype == np.uint16 else b, dtype)
        if rgba:
            img = np.stack([img] * 4, -1)
        imageio.imwrite(d / f"{name}_ms_{b:02d}.png", img)
    (d / "Thumbs.db").write_bytes(b"x")
    imageio.imwrite(d / f"{name}_RGB.bmp", np.zeros((4, 4, 3), np.uint8))
    return root / f"{name}_ms"


def test_cave_loader_16bit_nested(tmp_path):
    cube = load_cave_scene(_fake_cave(tmp_path, "toy"))
    assert cube.shape == (512, 512, 31) and cube.dtype == np.float32
    assert cube[0, 0, 0] == pytest.approx(1000 / 65535) and cube[0, 0, 30] == pytest.approx(31000 / 65535)


def test_cave_loader_8bit_rgba(tmp_path):
    cube = load_cave_scene(_fake_cave(tmp_path, "toy", dtype=np.uint8, rgba=True, nested=False))
    assert cube.shape == (512, 512, 31) and cube[0, 0, 30] == pytest.approx(31 / 255)


def test_cave_loader_missing_band(tmp_path):
    d = _fake_cave(tmp_path, "toy")
    next(d.rglob("*_ms_07.png")).unlink()
    with pytest.raises(ValueError):
        load_cave_scene(d)


def test_select_bands():
    cube = np.arange(31, dtype=np.float32)[None, None, :] * np.ones((2, 2, 1), np.float32)
    assert select_bands(cube)[0, 0].tolist() == list(range(0, 31, 2))


def test_cave_split_is_deterministic_and_disjoint(tmp_path):
    for i in range(32):
        (tmp_path / "CAVE" / f"s{i:02d}_ms").mkdir(parents=True)
    a, b = list_scenes(tmp_path, "cave-split"), list_scenes(tmp_path, "cave-split")
    assert a == b and [len(a[k]) for k in ("train", "val", "test")] == [21, 3, 8]
    assert len(set(a["train"]) | set(a["val"]) | set(a["test"])) == 32


def test_arad_mode_split(tmp_path):
    for i in range(32):
        (tmp_path / "CAVE" / f"s{i:02d}_ms").mkdir(parents=True)
    spec = tmp_path / "ARAD_1K" / "Train_Spec"
    spec.mkdir(parents=True)
    for i in range(300):
        (spec / f"ARAD_1K_{i:04d}.mat").touch()
    s = list_scenes(tmp_path, "arad")
    assert [len(s[k]) for k in ("train", "val", "test")] == [220, 30, 32]
    assert not set(s["train"]) & set(s["val"])


def test_list_scenes_missing_data_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_scenes(tmp_path, "cave-split")


def test_load_cube16_caches(tmp_path):
    scene = _fake_cave(tmp_path, "toy")
    cache = tmp_path / "cache"
    a = load_cube16(scene, cache)
    assert a.shape == (512, 512, 16) and a.dtype == np.float32
    assert a[0, 0, 1] == pytest.approx(3000 / 65535, abs=1e-4)
    assert (cache / "toy_ms.npy").exists()
    next(scene.rglob("*_ms_07.png")).unlink()  # second call must come from cache
    assert np.array_equal(load_cube16(scene, cache), a)


def test_sample_patch():
    cubes = [np.random.default_rng(0).random((100, 120, 16), dtype=np.float32)]
    p = sample_patch(cubes, np.random.default_rng(1), 64)
    assert p.shape == (64, 64, 16) and p.dtype == np.float32 and p.flags["C_CONTIGUOUS"]
