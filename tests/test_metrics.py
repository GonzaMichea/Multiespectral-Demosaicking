import numpy as np
import pytest

from msdemosaic.metrics import evaluate, psnr, psnr_per_band, sam


def test_psnr_known_value():
    cube = np.full((32, 32, 16), 0.5, np.float32)
    assert psnr(cube + 0.1, cube) == pytest.approx(20.0, abs=1e-3)


def test_identical():
    cube = np.random.default_rng(0).random((32, 32, 16), dtype=np.float32)
    m = evaluate(cube, cube)
    assert set(m) == {"psnr", "ssim", "sam"}
    assert m["psnr"] == pytest.approx(100.0) and m["ssim"] == pytest.approx(1.0) and m["sam"] < 0.1


def test_sam_scale_invariant_and_orthogonal():
    cube = np.random.default_rng(0).random((8, 8, 16), dtype=np.float32) + 0.1
    assert sam(0.5 * cube, cube) < 0.1
    a = np.zeros((8, 8, 16), np.float32)
    a[..., 0] = 1
    b = np.zeros((8, 8, 16), np.float32)
    b[..., 1] = 1
    assert sam(a, b) == pytest.approx(90.0, abs=1e-3)


def test_sam_black_pixels_no_nan():
    assert np.isfinite(sam(np.zeros((8, 8, 16), np.float32), np.zeros((8, 8, 16), np.float32)))


def test_evaluate_clips():
    cube = np.ones((32, 32, 16), np.float32)
    assert evaluate(cube + 5.0, cube)["psnr"] == pytest.approx(100.0)


def test_psnr_per_band_shape():
    cube = np.random.default_rng(0).random((32, 32, 16), dtype=np.float32)
    assert psnr_per_band(cube * 0.9, cube).shape == (16,)
