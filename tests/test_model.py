import numpy as np
import pytest
import torch

from msdemosaic.model import UNet, load_method, save_checkpoint
from msdemosaic.patterns import make_pattern
from msdemosaic.simulate import simulate


@pytest.mark.parametrize("use_mask", [True, False])
def test_param_budget_and_shape(use_mask):
    net = UNet(use_mask=use_mask)
    assert sum(p.numel() for p in net.parameters()) < 2_000_000
    y, m = torch.rand(2, 1, 64, 64), torch.rand(2, 16, 64, 64)
    assert net(y, m if use_mask else None).shape == (2, 16, 64, 64)


def test_arbitrary_size():
    net = UNet(use_mask=True)
    assert net(torch.rand(1, 1, 482, 510), torch.rand(1, 16, 482, 510)).shape == (1, 16, 482, 510)


def test_mask_required_when_use_mask():
    with pytest.raises(ValueError):
        UNet(use_mask=True)(torch.rand(1, 1, 16, 16))


def test_checkpoint_roundtrip_as_method(tmp_path):
    net = UNet(use_mask=True)
    save_checkpoint(tmp_path / "a.pt", net, train_pattern="random", iters=1, val_psnr=0.0)
    method = load_method(tmp_path / "a.pt", device=torch.device("cpu"))
    cube = np.random.default_rng(0).random((32, 32, 16), dtype=np.float32)
    mosaic, mask = simulate(cube, make_pattern("btes", 32, 32))
    recon = method(mosaic, mask)
    assert recon.shape == (32, 32, 16) and recon.dtype == np.float32 and np.isfinite(recon).all()
    with torch.no_grad():
        direct = net(torch.from_numpy(mosaic)[None, None], torch.from_numpy(mask).permute(2, 0, 1)[None])
    assert np.allclose(recon, direct[0].permute(1, 2, 0).numpy(), atol=1e-5)
