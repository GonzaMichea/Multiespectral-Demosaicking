import numpy as np
import pytest

from msdemosaic.patterns import TEST_PATTERNS, make_pattern, sample_training_pattern


def test_imec_kernel():
    assert make_pattern("imec", 8, 8)[:4, :4].tolist() == [[6, 7, 5, 4], [14, 15, 13, 12], [10, 11, 9, 8], [2, 3, 1, 0]]
    assert (make_pattern("imec", 8, 8)[4:, 4:] == make_pattern("imec", 8, 8)[:4, :4]).all()


def test_osp_formula():
    P = make_pattern("osp", 32, 32)
    assert P[0, 0] == 11 and P[1, 0] == 2 and P[0, 1] == 15
    assert (P[16:, :] == P[:16, :]).all()


@pytest.mark.parametrize("name", TEST_PATTERNS)
def test_balanced(name):
    P = make_pattern(name, 64, 64)
    assert P.dtype == np.int64 and np.bincount(P.ravel(), minlength=16).tolist() == [256] * 16


def test_training_sampler_balanced_and_disjoint():
    rng = np.random.default_rng(0)
    forbidden = [make_pattern(n, 4, 4).tolist() for n in ("imec", "btes", "random")]
    tiled = 0
    for _ in range(200):
        P = sample_training_pattern(rng, 64, 64)
        assert np.bincount(P.ravel(), minlength=16).tolist() == [256] * 16
        is_tile = (P == np.tile(P[:4, :4], (16, 16))).all()
        tiled += is_tile
        if is_tile:
            assert P[:4, :4].tolist() not in forbidden
    assert 60 < tiled < 140  # both modes are used


def test_unknown_pattern_raises():
    with pytest.raises(ValueError):
        make_pattern("bayer", 8, 8)
