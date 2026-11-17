"""Result figures from the evaluation CSVs and the trained checkpoints.

    python -m msdemosaic.figures --data-root data --data-mode cave-split \
        --checkpoints checkpoints --results results

Writes PDF and PNG files to <results>/figs/. Methods without results or
checkpoints are skipped.
"""
import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .evaluate_all import METHOD_ORDER, load_methods, summarize
from .patterns import TEST_PATTERNS, make_pattern
from .simulate import simulate

COLORS = {
    "wb": "#7f7f7f",
    "conventional_imec": "#9db8d9",
    "fixed_imec": "#2a5596",
    "fixed_osp": "#e8822c",
    "agnostic": "#9b1822",
}
LABELS = {
    "wb": "WB",
    "conventional_imec": "U-Net no mask (IMEC)",
    "fixed_imec": "U-Net fixed-IMEC",
    "fixed_osp": "U-Net fixed-OSP",
    "agnostic": "U-Net agnostic",
}
PATTERN_LABELS = {"imec": "IMEC", "osp": "OSP", "btes": "BTES", "random": "Random"}
WAVELENGTHS = 400 + 20 * np.arange(16)  # nm
RGB_BANDS = (13, 7, 3)  # ~660 / 540 / 460 nm
CROP = (slice(192, 320), slice(192, 320))
SPECTRA_PIXELS = ((40, 40), (90, 90))  # inside the crop


def _read_csv(path: Path) -> list[dict]:
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for key in ("psnr", "ssim", "sam"):
            if key in r:
                r[key] = float(r[key])
        if "band" in r:
            r["band"] = int(r["band"])
    return rows


def _save(fig, out: Path, name: str) -> None:
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def _methods_in(rows) -> list[str]:
    present = {r["method"] for r in rows}
    return [m for m in METHOD_ORDER if m in present]


def plot_psnr_bars(rows: list[dict], out: Path) -> None:
    summary = {(r["method"], r["pattern"]): r for r in summarize(rows)}
    methods = _methods_in(rows)
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    width = 0.8 / len(methods)
    x = np.arange(len(TEST_PATTERNS))
    for i, m in enumerate(methods):
        means = [summary[(m, p)]["psnr_mean"] for p in TEST_PATTERNS]
        stds = [summary[(m, p)]["psnr_std"] for p in TEST_PATTERNS]
        ax.bar(x + (i - (len(methods) - 1) / 2) * width, means, width, yerr=stds, capsize=2,
               color=COLORS[m], label=LABELS[m], error_kw={"linewidth": 0.8})
    ax.set_xticks(x, [PATTERN_LABELS[p] for p in TEST_PATTERNS])
    ax.set_xlabel("Test pattern")
    ax.set_ylabel("PSNR (dB)")
    ax.set_ylim(bottom=10)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.25), frameon=False)
    _save(fig, out, "psnr_bars")


def plot_psnr_per_band(band_rows: list[dict], out: Path) -> None:
    methods = _methods_in(band_rows)
    fig, axes = plt.subplots(1, 4, figsize=(11, 2.8), sharey=True)
    for ax, pattern in zip(axes, TEST_PATTERNS):
        for m in methods:
            values = np.zeros(16)
            counts = np.zeros(16)
            for r in band_rows:
                if r["method"] == m and r["pattern"] == pattern:
                    values[r["band"]] += r["psnr"]
                    counts[r["band"]] += 1
            ax.plot(WAVELENGTHS, values / counts, marker="o", markersize=3, color=COLORS[m], label=LABELS[m])
        ax.set_title(PATTERN_LABELS[pattern], fontsize=10)
        ax.set_xlabel("Wavelength (nm)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("PSNR (dB)")
    axes[0].legend(fontsize=7, frameon=False)
    _save(fig, out, "psnr_per_band")


def _rgb(cube: np.ndarray) -> np.ndarray:
    rgb = cube[..., list(RGB_BANDS)]
    return np.clip(rgb / max(rgb.max(), 1e-6), 0, 1) ** (1 / 2.2)


def plot_visual(cube: np.ndarray, recons: dict[str, np.ndarray], pattern: str, out: Path) -> None:
    """False-colour crops (top) and mean absolute error maps (bottom) per method."""
    ref = cube[CROP]
    scale = max(ref[..., list(RGB_BANDS)].max(), 1e-6)
    n = len(recons) + 1
    # Narrow panels and two-line titles keep the text readable when the figure is
    # scaled down to a poster or paper column (six panels side by side).
    fig, axes = plt.subplots(2, n, figsize=(1.8 * n, 4.2), gridspec_kw={"wspace": 0.08, "hspace": 0.3})
    axes[0, 0].imshow(_rgb(ref))
    axes[0, 0].set_title("Reference", fontsize=12)
    axes[1, 0].axis("off")
    for col, (m, recon) in enumerate(recons.items(), start=1):
        crop = np.clip(recon[CROP], 0, 1)
        rgb = np.clip(crop[..., list(RGB_BANDS)] / scale, 0, 1) ** (1 / 2.2)
        axes[0, col].imshow(rgb)
        axes[0, col].set_title(LABELS[m].replace("U-Net ", "U-Net\n"), fontsize=12)
        err = np.abs(crop - ref).mean(-1)
        im = axes[1, col].imshow(err, cmap="magma", vmin=0, vmax=0.15)
        axes[1, col].set_title(f"MAE {err.mean():.4f}", fontsize=11)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    # The colour bar takes its room from both rows, so each error map stays under its
    # reconstruction; shrink/anchor place it next to the bottom row only.
    colorbar = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, shrink=0.42, anchor=(0.0, 0.0))
    colorbar.set_label("Mean abs. error", fontsize=11)
    colorbar.ax.tick_params(labelsize=10)
    fig.suptitle(f"Pattern: {PATTERN_LABELS[pattern]}", fontsize=12, y=1.04)
    _save(fig, out, f"visual_{pattern}")


def plot_spectra(cube: np.ndarray, recons: dict[str, np.ndarray], out: Path) -> None:
    fig, axes = plt.subplots(1, len(SPECTRA_PIXELS), figsize=(7.5, 2.8), sharey=True)
    for ax, (i, j) in zip(axes, SPECTRA_PIXELS):
        ii, jj = CROP[0].start + i, CROP[1].start + j
        ax.plot(WAVELENGTHS, cube[ii, jj], color="black", linewidth=2, label="Reference")
        for m, recon in recons.items():
            ax.plot(WAVELENGTHS, np.clip(recon[ii, jj], 0, 1), color=COLORS[m], marker="o", markersize=3,
                    linewidth=1, label=LABELS[m])
        ax.set_title(f"Pixel ({ii}, {jj}), OSP pattern", fontsize=9)
        ax.set_xlabel("Wavelength (nm)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Reflectance")
    axes[0].legend(fontsize=7, frameon=False)
    _save(fig, out, "spectra")


def plot_training_curves(checkpoints: Path, out: Path) -> None:
    logs = {m: Path(checkpoints) / f"{m}.log.csv" for m in METHOD_ORDER[1:]}
    logs = {m: p for m, p in logs.items() if p.exists()}
    if not logs:
        return
    fig, ax = plt.subplots(figsize=(5, 3))
    for m, path in logs.items():
        data = np.genfromtxt(path, delimiter=",", names=True)
        ax.plot(np.atleast_1d(data["iter"]), np.atleast_1d(data["val_psnr"]), color=COLORS[m], label=LABELS[m])
    # Thousands on the x axis: the default five-digit tick labels overlap at this width.
    ax.xaxis.set_major_formatter(lambda value, _pos: f"{value / 1000:g}k")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Validation PSNR (dB)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    _save(fig, out, "training_curves")


def main() -> None:
    from .data import list_scenes, load_cube16

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--data-mode", choices=("arad", "cave-split"), required=True)
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--scene", default="beads_ms", help="test scene for the visual figures (falls back to the first test scene)")
    args = parser.parse_args()

    out = args.results / "figs"
    out.mkdir(parents=True, exist_ok=True)
    plot_psnr_bars(_read_csv(args.results / "per_scene.csv"), out)
    plot_psnr_per_band(_read_csv(args.results / "per_band.csv"), out)
    plot_training_curves(args.checkpoints, out)

    scenes = list_scenes(args.data_root, args.data_mode)["test"]
    scene = next((p for p in scenes if p.stem == args.scene), scenes[0])
    cube = load_cube16(scene, args.data_root / "cache")
    methods = load_methods(args.checkpoints)
    for pattern in ("imec", "osp"):
        mosaic, mask = simulate(cube, make_pattern(pattern, *cube.shape[:2]))
        recons = {m: fn(mosaic, mask) for m, fn in methods.items()}
        plot_visual(cube, recons, pattern, out)
        if pattern == "osp":
            plot_spectra(cube, recons, out)
    print(f"Figures for scene {scene.stem} written to {out}")


if __name__ == "__main__":
    main()
