"""Short diagnostic training runs used in the 16 Nov 2026 review (see docs/revision-2026-11-16.md).

Trains small variants for a few thousand iterations and scores them on the
VALIDATION scenes only (full image and interior), so the test scenes stay untouched.

    .venv/bin/python scripts/pilot.py . <out-dir> 2000 A_agnostic_YM,B_fixed_imec_nomask

Run names are the keys of SPECS below. Results accumulate in <out-dir>/results.json.
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

from msdemosaic.data import list_scenes, load_cube16
from msdemosaic.metrics import psnr
from msdemosaic.model import UNet, pick_device
from msdemosaic.patterns import TEST_PATTERNS, make_pattern
from msdemosaic.simulate import simulate
from msdemosaic.train import make_batch, _make_val_set, _validate
from msdemosaic.wb import wb

ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2]); ITERS = int(sys.argv[3]); RUNS = sys.argv[4].split(",")
dev = pick_device()
scenes = list_scenes(ROOT / "data", "cave-split")
train_cubes = [load_cube16(p, ROOT / "data/cache") for p in scenes["train"]]
val_cubes = {p.stem: load_cube16(p, ROOT / "data/cache") for p in scenes["val"]}

_f = torch.tensor([1., 2, 3, 4, 3, 2, 1])
_K = torch.outer(_f, _f)[None, None].repeat(16, 1, 1, 1)

def wb_torch(sparse, m):
    k = _K.to(sparse.device)
    num = F.conv2d(sparse, k, padding=3, groups=16)
    den = F.conv2d(m, k, padding=3, groups=16).clamp_min(1e-8)
    return (num / den) * (1 - m) + sparse

class SparseResUNet(UNet):
    """Input [Y*M, M] (32 ch); output = WB(Y, M) + residual."""
    def __init__(self, base=48):
        super().__init__(use_mask=True, base=base)
        self.enc1[0] = nn.Conv2d(32, base, 3, padding=1)
    def forward(self, y, m=None):
        sparse = y * m
        x = torch.cat([sparse, m], 1)
        H, W = x.shape[-2:]
        ph, pw = (-H) % 4, (-W) % 4
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        e1 = self.enc1(x); e2 = self.enc2(F.max_pool2d(e1, 2)); b = self.bottleneck(F.max_pool2d(e2, 2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], 1)); d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return wb_torch(sparse, m) + self.out(d1)[..., :H, :W]

def build(kind):
    if kind == "nomask": return UNet(use_mask=False)
    if kind == "mask": return UNet(use_mask=True)
    if kind == "sparse_res": return SparseResUNet()
    raise ValueError(kind)

def run_training(name, train_pattern, kind):
    torch.manual_seed(0); rng = np.random.default_rng(0)
    model = build(kind).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ITERS, eta_min=1e-5)
    val_set = _make_val_set(list(val_cubes.values()), train_pattern)
    curve, t0 = [], time.time()
    for it in range(1, ITERS + 1):
        y, m, x = (t.to(dev) for t in make_batch(train_cubes, rng, train_pattern, 32))
        loss = F.l1_loss(model(y, m if model.use_mask else None), x)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        if it % 500 == 0:
            v = _validate(model, val_set, dev); curve.append((it, round(v, 2)))
            print(f"  [{name}] it {it} val-patch {v:.2f} dB  {it/(time.time()-t0):.1f} it/s", flush=True)
    return model.eval(), curve

@torch.no_grad()
def recon(model, mosaic, mask):
    y = torch.from_numpy(mosaic)[None, None].to(dev)
    m = torch.from_numpy(mask).permute(2, 0, 1)[None].to(dev)
    out = model(y, m if model.use_mask else None)
    return np.clip(out[0].permute(1, 2, 0).cpu().numpy(), 0, 1)

INT = (slice(64, 448), slice(64, 448))
def score(fn, patterns):
    """Mean PSNR over val scenes: full image and interior (>= 64 px from every border)."""
    res = {}
    for pname, pfun in patterns.items():
        full, inner = [], []
        for cube in val_cubes.values():
            mosaic, mask = simulate(cube, pfun(*cube.shape[:2]))
            r = fn(mosaic, mask)
            full.append(psnr(r, cube)); inner.append(psnr(r[INT], cube[INT]))
        res[pname] = (round(float(np.mean(full)), 2), round(float(np.mean(inner)), 2))
    return res

PATS = {n: (lambda H, W, n=n: make_pattern(n, H, W)) for n in TEST_PATTERNS}
PATS["osp_shift4rows"] = lambda H, W: np.roll(make_pattern("osp", H, W), -4, axis=0)

SPECS = {
    "C_fixed_osp_nomask": ("osp", "nomask"),
    "D_fixed_osp_mask": ("osp", "mask"),
    "A_agnostic_YM": ("random", "mask"),
    "B_fixed_imec_nomask": ("imec", "nomask"),
    "E_agnostic_sparse_res": ("random", "sparse_res"),
    "F_fixed_imec_sparse_res": ("imec", "sparse_res"),
    "G_fixed_imec_mask": ("imec", "mask"),
}
results = {}
res_path = OUT / "results.json"
if res_path.exists():
    results = json.loads(res_path.read_text())
if "wb" not in results:
    results["wb"] = {"scores": score(lambda y, m: np.clip(wb(y, m), 0, 1), PATS)}
    print("WB (val scenes) full/interior:", results["wb"]["scores"], flush=True)
for name in RUNS:
    tp, kind = SPECS[name]
    print(f"== {name}: train_pattern={tp} input={kind} iters={ITERS}", flush=True)
    model, curve = run_training(name, tp, kind)
    sc = score(lambda y, m: recon(model, y, m), PATS)
    results[name] = {"curve": curve, "scores": sc, "iters": ITERS}
    print(f"   {name} full/interior PSNR on val scenes: {sc}", flush=True)
    res_path.write_text(json.dumps(results, indent=1))
print("PILOT DONE", flush=True)
