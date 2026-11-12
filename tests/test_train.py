import numpy as np
import pytest
import torch

from msdemosaic.patterns import make_pattern
from msdemosaic.train import make_batch, train


def _smooth_cubes(n=4):
    yy, xx = np.mgrid[0:96, 0:96] / 96
    return [np.stack([0.5 + 0.4 * np.sin(2 * np.pi * (yy * (i + 1) + xx) + k / 3) for k in range(16)], -1).astype(np.float32) for i in range(n)]


def test_make_batch_fixed_uses_anchored_pattern():
    y, m, x = make_batch(_smooth_cubes(), np.random.default_rng(0), "imec", batch=4)
    assert y.shape == (4, 1, 64, 64) and m.shape == (4, 16, 64, 64) and x.shape == (4, 16, 64, 64)
    assert (m[0].argmax(0).numpy() == make_pattern("imec", 64, 64)).all()
    assert torch.allclose((x * m).sum(1, keepdim=True), y)


def test_make_batch_random_changes_between_batches():
    rng = np.random.default_rng(0)
    m1 = make_batch(_smooth_cubes(), rng, "random", batch=2)[1]
    m2 = make_batch(_smooth_cubes(), rng, "random", batch=2)[1]
    assert (m1[0] == m1[1]).all() and not (m1[0] == m2[0]).all()


@pytest.mark.parametrize("pattern,use_mask", [("imec", False), ("random", True)])
def test_train_smoke(tmp_path, pattern, use_mask):
    cubes = _smooth_cubes()
    out = tmp_path / "m.pt"
    best = train(cubes[:3], cubes[3:], train_pattern=pattern, iters=150, out=out, batch=8, val_every=50, device=torch.device("cpu"))
    ck = torch.load(out, map_location="cpu")
    assert ck["use_mask"] is use_mask and ck["train_pattern"] == pattern
    assert np.isfinite(best) and ck["val_psnr"] == pytest.approx(best)
    log = np.genfromtxt(out.with_suffix(".log.csv"), delimiter=",", names=True)
    assert log["train_l1"][-1] < log["train_l1"][0]
