"""Evaluate every method on every test pattern and scene; write tables.

Usage:
    python -m msdemosaic.evaluate_all --data-root data --data-mode cave-split \
        --checkpoints checkpoints --out results
"""
import argparse
import csv
import json
from pathlib import Path
from typing import Callable

import numpy as np

from .metrics import evaluate, psnr_per_band
from .patterns import TEST_PATTERNS, make_pattern
from .simulate import simulate
from .wb import wb

METHOD_ORDER = ("wb", "fixed_imec", "fixed_osp", "agnostic")
METRICS = ("psnr", "ssim", "sam")
# Pattern each fixed network was trained on.
_SEEN = {"fixed_imec": "imec", "fixed_osp": "osp"}


def run_grid(
    methods: dict[str, Callable], cubes: dict[str, np.ndarray], patterns=TEST_PATTERNS
) -> tuple[list[dict], list[dict]]:
    """Run methods x patterns x scenes. Returns (per-scene rows, per-band rows)."""
    rows, band_rows = [], []
    for scene, cube in cubes.items():
        for pattern_name in patterns:
            mosaic, mask = simulate(cube, make_pattern(pattern_name, *cube.shape[:2]))
            for method_name, method in methods.items():
                recon = np.clip(method(mosaic, mask), 0.0, 1.0)
                key = {"method": method_name, "pattern": pattern_name, "scene": scene}
                rows.append({**key, **evaluate(recon, cube)})
                for band, value in enumerate(psnr_per_band(recon, cube)):
                    band_rows.append({**key, "band": band, "psnr": float(value)})
    return rows, band_rows


def summarize(rows: list[dict]) -> list[dict]:
    """Mean and std over scenes for each (method, pattern)."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["method"], r["pattern"]), []).append(r)
    summary = []
    for (method, pattern), group in groups.items():
        entry = {"method": method, "pattern": pattern}
        for metric in METRICS:
            values = np.array([g[metric] for g in group])
            entry[f"{metric}_mean"] = float(values.mean())
            entry[f"{metric}_std"] = float(values.std())
        summary.append(entry)
    return summary


def hypotheses(summary: list[dict]) -> dict:
    """H1-H3 figures from mean PSNR. Entries whose methods are missing are None."""
    p = {(r["method"], r["pattern"]): r["psnr_mean"] for r in summary}
    methods = {m for m, _ in p}

    def drop(fixed):
        if fixed not in methods:
            return None
        seen = _SEEN[fixed]
        unseen = [p[(fixed, pat)] for pat in TEST_PATTERNS if pat != seen]
        return p[(fixed, seen)] - float(np.mean(unseen))

    out = {"h1_drop_db": drop("fixed_imec"), "h1_drop_db_osp": drop("fixed_osp")}
    fixed_present = [f for f in _SEEN if f in methods]
    if "agnostic" not in methods or not fixed_present:
        out.update(h2_gap_db=None, h2_holds=None, h3_margin_db=None, h3_holds=None)
        return out
    gaps = {_SEEN[f]: p[(f, _SEEN[f])] - p[("agnostic", _SEEN[f])] for f in fixed_present}
    margins = {
        f"{f}@{pat}": p[("agnostic", pat)] - p[(f, pat)]
        for f in fixed_present
        for pat in TEST_PATTERNS
        if pat != _SEEN[f]
    }
    out.update(
        h2_gap_db=gaps,
        h2_holds=all(g < 1.0 for g in gaps.values()),
        h3_margin_db=margins,
        h3_holds=all(m > 0 for m in margins.values()),
    )
    return out


def _cell(summary_index, method, pattern, metric):
    r = summary_index.get((method, pattern))
    if r is None:
        return "--"
    digits = 4 if metric == "ssim" else 2
    return f"{r[f'{metric}_mean']:.{digits}f} ± {r[f'{metric}_std']:.{digits}f}"


def write_tables(summary: list[dict], out: Path) -> None:
    index = {(r["method"], r["pattern"]): r for r in summary}
    methods = [m for m in METHOD_ORDER if any(k[0] == m for k in index)]
    labels = {"psnr": "PSNR (dB)", "ssim": "SSIM", "sam": "SAM (deg)"}

    md = []
    for metric in METRICS:
        md.append(f"### {labels[metric]}\n")
        md.append("| Method | " + " | ".join(TEST_PATTERNS) + " |")
        md.append("|---|" + "---|" * len(TEST_PATTERNS))
        for m in methods:
            md.append(f"| {m} | " + " | ".join(_cell(index, m, pat, metric) for pat in TEST_PATTERNS) + " |")
        md.append("")
    (out / "table.md").write_text("\n".join(md))

    tex = ["\\begin{tabular}{ll" + "c" * len(TEST_PATTERNS) + "}", "\\hline",
           "Metric & Method & " + " & ".join(pat.upper() if pat != "random" else "Random" for pat in TEST_PATTERNS) + " \\\\",
           "\\hline"]
    for metric in METRICS:
        for i, m in enumerate(methods):
            cells = " & ".join(_cell(index, m, pat, metric).replace("±", "$\\pm$") for pat in TEST_PATTERNS)
            head = labels[metric] if i == 0 else ""
            tex.append(f"{head} & {m.replace('_', '-')} & {cells} \\\\")
        tex.append("\\hline")
    tex.append("\\end{tabular}")
    (out / "table.tex").write_text("\n".join(tex) + "\n")


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_methods(checkpoints: Path) -> dict[str, Callable]:
    """WB plus every trained network found in the checkpoint directory."""
    from .model import load_method

    methods: dict[str, Callable] = {"wb": wb}
    for name in METHOD_ORDER[1:]:
        ckpt = Path(checkpoints) / f"{name}.pt"
        if ckpt.exists():
            methods[name] = load_method(ckpt)
        else:
            print(f"[warn] {ckpt} not found; skipping {name}")
    return methods


def main() -> None:
    from .data import list_scenes, load_cube16

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--data-mode", choices=("arad", "cave-split"), required=True)
    parser.add_argument("--checkpoints", type=Path, default=Path("checkpoints"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    scenes = list_scenes(args.data_root, args.data_mode)["test"]
    cubes = {p.stem: load_cube16(p, args.data_root / "cache") for p in scenes}
    methods = load_methods(args.checkpoints)
    print(f"Evaluating {list(methods)} on {len(cubes)} scenes x {len(TEST_PATTERNS)} patterns")

    rows, band_rows = run_grid(methods, cubes)
    summary = summarize(rows)
    _write_csv(args.out / "per_scene.csv", rows)
    _write_csv(args.out / "per_band.csv", band_rows)
    _write_csv(args.out / "summary.csv", summary)
    write_tables(summary, args.out)
    hyp = hypotheses(summary)
    hyp["data_mode"] = args.data_mode
    hyp["num_test_scenes"] = len(cubes)
    (args.out / "hypotheses.json").write_text(json.dumps(hyp, indent=2) + "\n")
    print((args.out / "table.md").read_text())
    print(json.dumps(hyp, indent=2))


if __name__ == "__main__":
    main()
