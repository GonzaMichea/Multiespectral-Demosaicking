import numpy as np

from msdemosaic.patterns import make_pattern
from msdemosaic.simulate import simulate


def test_simulate():
    rng = np.random.default_rng(0)
    cube = rng.random((16, 16, 16), dtype=np.float32)
    P = make_pattern("osp", 16, 16)
    mosaic, mask = simulate(cube, P)
    assert mosaic.shape == (16, 16) and mask.shape == (16, 16, 16)
    assert mosaic.dtype == np.float32 and mask.dtype == np.float32
    assert (mask.sum(-1) == 1).all()
    assert mosaic[3, 5] == cube[3, 5, P[3, 5]]
    assert np.allclose((cube * mask).sum(-1), mosaic)
