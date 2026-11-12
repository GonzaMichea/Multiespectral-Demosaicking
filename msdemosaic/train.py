"""Train a demosaicking U-Net.

    python -m msdemosaic.train --pattern imec   --data-mode cave-split --out checkpoints/fixed_imec.pt
    python -m msdemosaic.train --pattern osp    --data-mode cave-split --out checkpoints/fixed_osp.pt
    python -m msdemosaic.train --pattern random --data-mode cave-split --out checkpoints/agnostic.pt

--pattern imec/osp trains a fixed-pattern network (mosaic only).
--pattern random trains the pattern-agnostic network (mosaic + mask), drawing a
new random balanced pattern for every batch.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .data import sample_patch
from .model import UNet, pick_device, save_checkpoint
from .patterns import make_pattern, pattern_to_mask, sample_training_pattern

VAL_PATCHES = 256
VAL_PATTERN_SEED = 1234
NUM_VAL_PATTERNS = 8


def _to_tensors(patches: np.ndarray, pattern: np.ndarray):
    """patches (B, P, P, 16), pattern (P, P) -> y (B,1,P,P), m (B,16,P,P), x (B,16,P,P)."""
    x = torch.from_numpy(patches).permute(0, 3, 1, 2).contiguous()
    m = torch.from_numpy(pattern_to_mask(pattern)).permute(2, 0, 1)[None].expand(len(patches), -1, -1, -1).contiguous()
    y = (x * m).sum(dim=1, keepdim=True)
    return y, m, x


def make_batch(cubes, rng: np.random.Generator, train_pattern: str, batch: int = 32, patch: int = 64):
    """One training batch; a single pattern is shared by the whole batch."""
    if train_pattern == "random":
        pattern = sample_training_pattern(rng, patch, patch)
    else:
        pattern = make_pattern(train_pattern, patch, patch)
    patches = np.stack([sample_patch(cubes, rng, patch) for _ in range(batch)])
    return _to_tensors(patches, pattern)


def _make_val_set(cubes_val, train_pattern: str, patch: int = 64):
    """Fixed validation batches: list of (y, m, x), one per validation pattern."""
    rng = np.random.default_rng(0)
    patches = np.stack([sample_patch(cubes_val, rng, patch) for _ in range(VAL_PATCHES)])
    if train_pattern == "random":
        prng = np.random.default_rng(VAL_PATTERN_SEED)
        patterns = [sample_training_pattern(prng, patch, patch) for _ in range(NUM_VAL_PATTERNS)]
    else:
        patterns = [make_pattern(train_pattern, patch, patch)]
    chunks = np.array_split(np.arange(VAL_PATCHES), len(patterns))
    return [_to_tensors(patches[idx], pat) for idx, pat in zip(chunks, patterns)]


@torch.no_grad()
def _validate(model: UNet, val_set, device) -> float:
    model.eval()
    sq_err, count = 0.0, 0
    for y, m, x in val_set:
        for s in range(0, len(x), 64):
            yb, mb, xb = (t[s : s + 64].to(device) for t in (y, m, x))
            pred = model(yb, mb if model.use_mask else None).clamp(0, 1)
            sq_err += F.mse_loss(pred, xb, reduction="sum").item()
            count += xb.numel()
    model.train()
    return float(10 * np.log10(1.0 / max(sq_err / count, 1e-10)))


def train(cubes_train, cubes_val, *, train_pattern: str, iters: int, out, batch: int = 32,
          lr: float = 1e-3, val_every: int = 1000, seed: int = 0, device=None) -> float:
    """Train and save the best checkpoint (by validation PSNR) to `out`. Returns that PSNR."""
    out = Path(out)
    device = device or pick_device()
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    model = UNet(use_mask=(train_pattern == "random")).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iters, eta_min=1e-5)
    val_set = _make_val_set(cubes_val, train_pattern)

    out.parent.mkdir(parents=True, exist_ok=True)
    log_path = out.with_suffix(".log.csv")
    log_path.write_text("iter,train_l1,val_psnr\n")

    best, running, start = -np.inf, [], time.time()
    for it in range(1, iters + 1):
        y, m, x = (t.to(device) for t in make_batch(cubes_train, rng, train_pattern, batch))
        loss = F.l1_loss(model(y, m if model.use_mask else None), x)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running.append(loss.item())

        if it % val_every == 0 or it == iters:
            val_psnr = _validate(model, val_set, device)
            train_l1 = float(np.mean(running))
            running = []
            with open(log_path, "a") as f:
                f.write(f"{it},{train_l1:.6f},{val_psnr:.4f}\n")
            if val_psnr > best:
                best = val_psnr
                save_checkpoint(out, model, train_pattern=train_pattern, iters=it, val_psnr=val_psnr)
            rate = it / (time.time() - start)
            print(f"iter {it:6d}/{iters}  l1 {train_l1:.5f}  val {val_psnr:.2f} dB  best {best:.2f}  {rate:.1f} it/s", flush=True)
    return float(best)


def main() -> None:
    from .data import list_scenes, load_cube16

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pattern", choices=("imec", "osp", "random"), required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--data-mode", choices=("arad", "cave-split"), required=True)
    parser.add_argument("--iters", type=int, default=40000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--val-every", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    scenes = list_scenes(args.data_root, args.data_mode)
    cache = args.data_root / "cache"
    cubes_train = [load_cube16(p, cache) for p in scenes["train"]]
    cubes_val = [load_cube16(p, cache) for p in scenes["val"]]
    print(f"{len(cubes_train)} train / {len(cubes_val)} val scenes, device {pick_device()}")
    best = train(cubes_train, cubes_val, train_pattern=args.pattern, iters=args.iters, out=args.out,
                 batch=args.batch, val_every=args.val_every, seed=args.seed)
    print(f"best val PSNR {best:.2f} dB -> {args.out}")


if __name__ == "__main__":
    main()
