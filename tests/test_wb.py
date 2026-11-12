import numpy as np
import pytest

from msdemosaic.metrics import psnr
from msdemosaic.patterns import TEST_PATTERNS, make_pattern, sample_training_pattern
from msdemosaic.simulate import simulate
from msdemosaic.wb import wb


@pytest.mark.parametrize("name", TEST_PATTERNS)
def test_wb_reproduces_flat_spectra(name):
    cube = np.ones((32, 32, 16), np.float32) * np.linspace(0.1, 0.9, 16, dtype=np.float32)
    recon = wb(*simulate(cube, make_pattern(name, 32, 32)))
    assert recon.shape == cube.shape and recon.dtype == np.float32
    assert np.allclose(recon, cube, atol=1e-5)


def test_wb_handles_block_random_patterns():
    cube = np.ones((32, 32, 16), np.float32) * np.linspace(0.1, 0.9, 16, dtype=np.float32)
    rng = np.random.default_rng(3)
    for _ in range(10):
        recon = wb(*simulate(cube, sample_training_pattern(rng, 32, 32)))
        assert np.allclose(recon, cube, atol=1e-5)


def test_wb_keeps_measured_pixels_and_beats_zero_fill():
    yy, xx = np.mgrid[0:64, 0:64] / 64
    cube = np.stack([0.5 + 0.4 * np.sin(2 * np.pi * (yy + xx) + k / 3) for k in range(16)], -1).astype(np.float32)
    mosaic, mask = simulate(cube, make_pattern("imec", 64, 64))
    recon = wb(mosaic, mask)
    assert np.allclose(recon[mask == 1], cube[mask == 1])
    assert psnr(recon, cube) > 30
