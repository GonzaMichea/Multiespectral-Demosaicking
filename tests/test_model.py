import numpy as np
import pytest
import torch

from msdemosaic.model import INPUT_MODES, UNet, load_method, save_checkpoint, weighted_bilinear
from msdemosaic.patterns import TEST_PATTERNS, make_pattern, pattern_to_mask, sample_training_pattern
from msdemosaic.simulate import simulate
from msdemosaic.wb import wb


def _mask(batch, H, W, name="imec"):
    m = torch.from_numpy(pattern_to_mask(make_pattern(name, H, W))).permute(2, 0, 1)
    return m[None].expand(batch, -1, -1, -1).contiguous()


def test_input_modes():
    assert INPUT_MODES == ("wb", "mosaic_mask", "mosaic") and UNet().input_mode == "wb"
    with pytest.raises(ValueError):
        UNet(input_mode="rgb")


@pytest.mark.parametrize("mode", INPUT_MODES)
def test_param_budget_and_shape(mode):
    net = UNet(input_mode=mode)
    assert sum(p.numel() for p in net.parameters()) < 2_000_000
    assert net(torch.rand(2, 1, 64, 64), _mask(2, 64, 64)).shape == (2, 16, 64, 64)


@pytest.mark.parametrize("mode", INPUT_MODES)
def test_arbitrary_size(mode):
    net = UNet(input_mode=mode)
    assert net(torch.rand(1, 1, 482, 510), _mask(1, 482, 510)).shape == (1, 16, 482, 510)


@pytest.mark.parametrize("mode", ["wb", "mosaic_mask"])
def test_mask_required(mode):
    with pytest.raises(ValueError):
        UNet(input_mode=mode)(torch.rand(1, 1, 16, 16))


def test_mosaic_mode_ignores_the_mask():
    net = UNet(input_mode="mosaic").eval()
    y = torch.rand(1, 1, 32, 32)
    with torch.no_grad():
        assert torch.equal(net(y), net(y, _mask(1, 32, 32, "osp")))


def _np_and_torch_wb(pattern):
    cube = np.random.default_rng(0).random((*pattern.shape, 16), dtype=np.float32)
    mosaic, mask = simulate(cube, pattern)
    y = torch.from_numpy(mosaic)[None, None]
    m = torch.from_numpy(mask).permute(2, 0, 1)[None]
    return wb(mosaic, mask), weighted_bilinear(y * m, m)[0].permute(1, 2, 0).numpy()


@pytest.mark.parametrize("name", TEST_PATTERNS)
def test_weighted_bilinear_matches_numpy_wb(name):
    ref, out = _np_and_torch_wb(make_pattern(name, 30, 34))  # size not a multiple of 4
    assert out.dtype == np.float32 and np.allclose(out, ref, atol=1e-5)


def test_weighted_bilinear_matches_numpy_wb_on_training_patterns():
    rng = np.random.default_rng(1)
    for _ in range(6):
        ref, out = _np_and_torch_wb(sample_training_pattern(rng, 32, 32))
        assert np.allclose(out, ref, atol=1e-5)


def test_wb_mode_feeds_wb_estimate_sparse_samples_and_mask():
    net = UNet(input_mode="wb")
    seen = {}
    net.enc1.register_forward_pre_hook(lambda module, args: seen.__setitem__("x", args[0]))
    y, m = torch.rand(2, 1, 32, 32), _mask(2, 32, 32, "osp")
    net(y, m)
    x = seen["x"]
    assert x.shape == (2, 48, 32, 32)
    assert torch.allclose(x[:, :16], weighted_bilinear(y * m, m))
    assert torch.equal(x[:, 16:32], y * m) and torch.equal(x[:, 32:], m)


@pytest.mark.parametrize("mode", INPUT_MODES)
def test_checkpoint_roundtrip_as_method(tmp_path, mode):
    net = UNet(input_mode=mode).eval()
    save_checkpoint(tmp_path / "a.pt", net, train_pattern="random", iters=1, val_psnr=0.0)
    assert torch.load(tmp_path / "a.pt", map_location="cpu")["input_mode"] == mode
    method = load_method(tmp_path / "a.pt", device=torch.device("cpu"))
    cube = np.random.default_rng(0).random((32, 32, 16), dtype=np.float32)
    mosaic, mask = simulate(cube, make_pattern("btes", 32, 32))
    recon = method(mosaic, mask)
    assert recon.shape == (32, 32, 16) and recon.dtype == np.float32 and np.isfinite(recon).all()
    with torch.no_grad():
        direct = net(torch.from_numpy(mosaic)[None, None], torch.from_numpy(mask).permute(2, 0, 1)[None])
    assert np.allclose(recon, direct[0].permute(1, 2, 0).numpy(), atol=1e-5)
