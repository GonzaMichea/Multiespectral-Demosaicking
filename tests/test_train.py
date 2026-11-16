import numpy as np
import pytest
import torch

from msdemosaic.patterns import make_pattern
from msdemosaic.train import TrainingPaused, make_batch, train


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


@pytest.mark.parametrize("pattern,input_mode", [("imec", "wb"), ("random", "wb"), ("imec", "mosaic"), ("random", "mosaic_mask")])
def test_train_smoke(tmp_path, pattern, input_mode):
    cubes = _smooth_cubes()
    out = tmp_path / "m.pt"
    best = train(cubes[:3], cubes[3:], train_pattern=pattern, input_mode=input_mode, iters=150, out=out, batch=8,
                 val_every=50, device=torch.device("cpu"))
    ck = torch.load(out, map_location="cpu")
    assert ck["input_mode"] == input_mode and ck["train_pattern"] == pattern
    assert np.isfinite(best) and ck["val_psnr"] == pytest.approx(best)
    log = np.genfromtxt(out.with_suffix(".log.csv"), delimiter=",", names=True)
    assert log["train_l1"][-1] < log["train_l1"][0]


def _weights(path):
    return torch.load(path, map_location="cpu")["state_dict"]


def test_resume_matches_uninterrupted_run(tmp_path):
    cubes = _smooth_cubes()
    kw = dict(train_pattern="random", iters=60, batch=4, val_every=20, device=torch.device("cpu"))
    full = tmp_path / "full.pt"
    train(cubes[:3], cubes[3:], out=full, **kw)

    part = tmp_path / "part.pt"
    with pytest.raises(TrainingPaused):
        train(cubes[:3], cubes[3:], out=part, max_iters_this_run=20, **kw)
    assert part.with_suffix(".state.pt").exists()
    best = train(cubes[:3], cubes[3:], out=part, **kw)  # resumes at iteration 20

    a, b = _weights(full), _weights(part)
    assert all(torch.equal(a[k], b[k]) for k in a)
    assert best == pytest.approx(torch.load(full, map_location="cpu")["val_psnr"])
    log = np.genfromtxt(part.with_suffix(".log.csv"), delimiter=",", names=True)
    assert log["iter"].tolist() == [20, 40, 60]
    assert not part.with_suffix(".state.pt").exists()  # removed once training completes


def test_resume_rejects_state_from_different_run(tmp_path):
    cubes = _smooth_cubes()
    out = tmp_path / "m.pt"
    with pytest.raises(TrainingPaused):
        train(cubes[:3], cubes[3:], train_pattern="imec", iters=40, out=out, batch=4, val_every=20,
              max_iters_this_run=20, device=torch.device("cpu"))
    with pytest.raises(ValueError):
        train(cubes[:3], cubes[3:], train_pattern="imec", iters=80, out=out, batch=4, val_every=20,
              device=torch.device("cpu"))


def test_fixed_networks_get_the_same_input_as_the_agnostic_one(tmp_path):
    cubes = _smooth_cubes()
    for pattern in ("imec", "random"):
        out = tmp_path / f"{pattern}.pt"
        train(cubes[:3], cubes[3:], train_pattern=pattern, iters=10, out=out, batch=4, val_every=10, device=torch.device("cpu"))
        assert torch.load(out, map_location="cpu")["input_mode"] == "wb"


def test_maskless_network_cannot_train_on_random_patterns(tmp_path):
    cubes = _smooth_cubes()
    with pytest.raises(ValueError):
        train(cubes[:3], cubes[3:], train_pattern="random", input_mode="mosaic", iters=10, out=tmp_path / "m.pt",
              batch=4, val_every=10, device=torch.device("cpu"))


def test_resume_rejects_state_with_different_input_mode(tmp_path):
    cubes = _smooth_cubes()
    kw = dict(train_pattern="imec", iters=40, out=tmp_path / "m.pt", batch=4, val_every=20, device=torch.device("cpu"))
    with pytest.raises(TrainingPaused):
        train(cubes[:3], cubes[3:], input_mode="wb", max_iters_this_run=20, **kw)
    with pytest.raises(ValueError):
        train(cubes[:3], cubes[3:], input_mode="mosaic_mask", **kw)
