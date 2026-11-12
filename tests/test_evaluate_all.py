import numpy as np
import pytest

from msdemosaic.evaluate_all import hypotheses, run_grid, summarize
from msdemosaic.wb import wb


def test_run_grid_and_summarize():
    rng = np.random.default_rng(0)
    cubes = {f"s{i}": rng.random((32, 32, 16), dtype=np.float32) for i in range(3)}
    rows, band_rows = run_grid({"wb": wb, "oracle_free_zero": lambda y, m: np.zeros(m.shape, np.float32)}, cubes)
    assert len(rows) == 2 * 4 * 3 and len(band_rows) == 2 * 4 * 3 * 16
    s = summarize(rows)
    assert len(s) == 8
    get = lambda meth, pat: next(r for r in s if r["method"] == meth and r["pattern"] == pat)
    assert get("wb", "imec")["psnr_mean"] > get("oracle_free_zero", "imec")["psnr_mean"]
    assert get("wb", "imec")["psnr_std"] >= 0


def test_hypotheses_from_synthetic_summary():
    def row(m, p, v):
        return {"method": m, "pattern": p, "psnr_mean": v}

    s = [row("fixed_imec", "imec", 40), row("fixed_imec", "osp", 25), row("fixed_imec", "btes", 26), row("fixed_imec", "random", 24),
         row("fixed_osp", "osp", 41), row("fixed_osp", "imec", 25), row("fixed_osp", "btes", 25), row("fixed_osp", "random", 25),
         *[row("agnostic", p, 39.5) for p in ("imec", "osp", "btes", "random")]]
    h = hypotheses(s)
    assert h["h1_drop_db"] == pytest.approx(15.0)
    assert h["h2_gap_db"] == {"imec": pytest.approx(0.5), "osp": pytest.approx(1.5)} and h["h2_holds"] is False
    assert h["h3_holds"] is True and len(h["h3_margin_db"]) == 6
    assert hypotheses([r for r in s if r["method"] != "agnostic"])["h2_gap_db"] is None


def test_hypotheses_partial_only_fixed_imec():
    s = [{"method": "fixed_imec", "pattern": p, "psnr_mean": v} for p, v in
         (("imec", 40), ("osp", 30), ("btes", 30), ("random", 30))]
    h = hypotheses(s)
    assert h["h1_drop_db"] == pytest.approx(10.0) and h["h1_drop_db_osp"] is None
    assert h["h2_holds"] is None and h["h3_holds"] is None
