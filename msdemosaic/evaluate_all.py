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

# "conventional_imec" is the optional mask-less network (--input mosaic) trained on IMEC only.
METHOD_ORDER = ("wb", "conventional_imec", "fixed_imec", "fixed_osp", "agnostic")
METRICS = ("psnr", "ssim", "sam")
# Pattern each single-pattern network was trained on.
_TRAINED_ON = {"conventional_imec": "imec", "fixed_imec": "imec", "fixed_osp": "osp"}
# Fixed networks with the same input as the agnostic one: the pairs compared in H2 and H3.
_SEEN = {"fixed_imec": "imec", "fixed_osp": "osp"}
# H2 threshold as presented to the course (plan slides): agnostic within 2 dB of each
# fixed network on that network's own pattern. The written proposal had a stricter
# 1 dB goal; per-scene counts for both are kept in the paired statistics.
H2_THRESHOLD_DB = 2.0


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


def _sample_std(values) -> float:
    """Sample standard deviation (ddof=1); 0 when there is a single value."""
    values = np.asarray(values, dtype=np.float64)
    return float(values.std(ddof=1)) if values.size > 1 else 0.0


def summarize(rows: list[dict]) -> list[dict]:
    """Mean and sample std over scenes for each (method, pattern)."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r["method"], r["pattern"]), []).append(r)
    summary = []
    for (method, pattern), group in groups.items():
        entry = {"method": method, "pattern": pattern}
        for metric in METRICS:
            values = np.array([g[metric] for g in group])
            entry[f"{metric}_mean"] = float(values.mean())
            entry[f"{metric}_std"] = _sample_std(values)
        summary.append(entry)
    return summary


def hypotheses(summary: list[dict]) -> dict:
    """H1-H3 figures from mean PSNR. Entries whose methods are missing are None."""
    p = {(r["method"], r["pattern"]): r["psnr_mean"] for r in summary}
    methods = {m for m, _ in p}

    def drop(fixed):
        if fixed not in methods:
            return None
        seen = _TRAINED_ON[fixed]
        unseen = [p[(fixed, pat)] for pat in TEST_PATTERNS if pat != seen]
        return p[(fixed, seen)] - float(np.mean(unseen))

    out = {"h1_drop_db": drop("fixed_imec"), "h1_drop_db_osp": drop("fixed_osp"),
           "h1_drop_db_conventional": drop("conventional_imec")}
    fixed_present = [f for f in _SEEN if f in methods]
    if "agnostic" not in methods or not fixed_present:
        out.update(h2_threshold_db=H2_THRESHOLD_DB, h2_gap_db=None, h2_holds=None,
                   h3_margin_db=None, h3_wb_margin_db=None, h3_holds=None)
        return out
    gaps = {_SEEN[f]: p[(f, _SEEN[f])] - p[("agnostic", _SEEN[f])] for f in fixed_present}
    margins = {
        f"{f}@{pat}": p[("agnostic", pat)] - p[(f, pat)]
        for f in fixed_present
        for pat in TEST_PATTERNS
        if pat != _SEEN[f]
    }
    # H3 also compares against the classical interpolation, on every test pattern
    # (none of them is seen by the agnostic network).
    wb_margins = ({pat: p[("agnostic", pat)] - p[("wb", pat)] for pat in TEST_PATTERNS}
                  if "wb" in methods else None)
    out.update(
        h2_threshold_db=H2_THRESHOLD_DB,
        h2_gap_db=gaps,
        h2_holds=all(g < H2_THRESHOLD_DB for g in gaps.values()),
        h3_margin_db=margins,
        h3_wb_margin_db=wb_margins,
        h3_holds=all(m > 0 for m in margins.values()) and all(m > 0 for m in (wb_margins or {}).values()),
    )
    return out


def paired_differences(rows: list[dict]) -> dict:
    """Per-scene PSNR differences behind H2 and H3 (paired by scene).

    h2[pattern]: fixed - agnostic on the pattern the fixed network was trained on.
    h3["fixed@pattern"]: agnostic - fixed on a pattern the fixed network never saw.
    h3_wb[pattern]: agnostic - WB on every test pattern.
    Each entry has mean, sample std and n over scenes. Scene-to-scene PSNR spread is
    several dB, so these paired numbers are far tighter than differences of means.
    """
    psnr = {(r["method"], r["pattern"], r["scene"]): r["psnr"] for r in rows}
    methods = {m for m, _, _ in psnr}
    scenes = sorted({s for _, _, s in psnr})
    out: dict = {"h2": {}, "h3": {}, "h3_wb": {}}
    if "agnostic" not in methods:
        return out

    def stats(diffs):
        return {"mean": float(diffs.mean()), "std": _sample_std(diffs), "n": int(diffs.size),
                "min": float(diffs.min()), "max": float(diffs.max())}

    if "wb" in methods:
        for pattern in TEST_PATTERNS:
            diffs = np.array([psnr[("agnostic", pattern, s)] - psnr[("wb", pattern, s)] for s in scenes])
            out["h3_wb"][pattern] = {**stats(diffs), "scenes_agnostic_better": int((diffs > 0).sum())}
    for fixed, seen in _SEEN.items():
        if fixed not in methods:
            continue
        for pattern in TEST_PATTERNS:
            sign = 1.0 if pattern == seen else -1.0  # H2: fixed - agnostic; H3: agnostic - fixed
            diffs = np.array([sign * (psnr[(fixed, pattern, s)] - psnr[("agnostic", pattern, s)]) for s in scenes])
            entry = stats(diffs)
            if pattern == seen:
                entry["scenes_within_threshold"] = int((diffs < H2_THRESHOLD_DB).sum())
                entry["scenes_within_1db"] = int((diffs < 1.0).sum())
                out["h2"][pattern] = entry
            else:
                entry["scenes_agnostic_better"] = int((diffs > 0).sum())
                out["h3"][f"{fixed}@{pattern}"] = entry
    return out


def training_info(checkpoints: Path) -> dict:
    """Iterations trained and best validation PSNR of every finished network, from its log."""
    info = {}
    for name in METHOD_ORDER[1:]:
        log = Path(checkpoints) / f"{name}.log.csv"
        if not log.exists():
            continue
        with open(log) as f:
            rows = list(csv.DictReader(f))
        if rows:
            info[name] = {"iters": max(int(float(r["iter"])) for r in rows),
                          "best_val_psnr": max(float(r["val_psnr"]) for r in rows)}
    return info


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
    hyp["paired"] = paired_differences(rows)
    hyp["training"] = training_info(args.checkpoints)
    hyp["data_mode"] = args.data_mode
    hyp["num_test_scenes"] = len(cubes)
    (args.out / "hypotheses.json").write_text(json.dumps(hyp, indent=2) + "\n")
    print((args.out / "table.md").read_text())
    print(json.dumps(hyp, indent=2))


if __name__ == "__main__":
    main()
