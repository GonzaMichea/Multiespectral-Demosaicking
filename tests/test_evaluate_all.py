import numpy as np
import pytest

from msdemosaic.evaluate_all import METHOD_ORDER, hypotheses, paired_differences, run_grid, summarize, training_info
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


def _row(m, p, scene, v):
    return {"method": m, "pattern": p, "scene": scene, "psnr": v, "ssim": 0.9, "sam": 5.0}


def test_summarize_uses_sample_std():
    s = summarize([_row("wb", "imec", "a", 30.0), _row("wb", "imec", "b", 34.0)])[0]
    assert s["psnr_mean"] == pytest.approx(32.0)
    assert s["psnr_std"] == pytest.approx(np.std([30.0, 34.0], ddof=1))  # 2.83, not the population value 2.0
    assert s["ssim_std"] == 0.0


def test_summarize_single_scene_has_zero_std():
    assert summarize([_row("wb", "imec", "a", 30.0)])[0]["psnr_std"] == 0.0


def test_paired_differences_per_scene():
    rows = []
    for scene, fixed_seen, agn in (("a", 40.0, 39.5), ("b", 38.0, 36.5), ("c", 36.0, 36.0)):
        rows += [_row("fixed_imec", "imec", scene, fixed_seen), _row("agnostic", "imec", scene, agn)]
        for pat in ("osp", "btes", "random"):
            rows += [_row("fixed_imec", pat, scene, 22.0), _row("agnostic", pat, scene, agn)]
    p = paired_differences(rows)
    h2 = p["h2"]["imec"]  # fixed - agnostic on the fixed network's own pattern
    assert h2["mean"] == pytest.approx((0.5 + 1.5 + 0.0) / 3)
    assert h2["std"] == pytest.approx(np.std([0.5, 1.5, 0.0], ddof=1))
    assert h2["n"] == 3 and h2["scenes_within_1db"] == 2
    h3 = p["h3"]["fixed_imec@osp"]  # agnostic - fixed on a pattern the fixed network never saw
    assert h3["mean"] == pytest.approx(np.mean([17.5, 14.5, 14.0])) and h3["scenes_agnostic_better"] == 3
    assert "osp" not in p["h2"]  # no fixed_osp rows -> no entry
    assert paired_differences([r for r in rows if r["method"] != "agnostic"]) == {"h2": {}, "h3": {}}



def test_conventional_network_is_reported_in_h1_only():
    s = [{"method": m, "pattern": p, "psnr_mean": v} for m, vals in {
        "conventional_imec": (35, 23, 22, 24), "fixed_imec": (36, 33, 32, 32), "agnostic": (35, 35, 35, 35),
    }.items() for p, v in zip(("imec", "osp", "btes", "random"), vals)]
    h = hypotheses(s)
    assert h["h1_drop_db_conventional"] == pytest.approx(35 - 23.0)
    assert h["h1_drop_db"] == pytest.approx(36 - 97 / 3)
    assert set(h["h2_gap_db"]) == {"imec"} and len(h["h3_margin_db"]) == 3  # only the same-input fixed networks
    assert hypotheses([r for r in s if r["method"] == "agnostic"])["h1_drop_db_conventional"] is None


def test_method_order_lists_the_conventional_baseline_before_the_fixed_networks():
    assert METHOD_ORDER == ("wb", "conventional_imec", "fixed_imec", "fixed_osp", "agnostic")


def test_training_info_reads_iterations_from_the_logs(tmp_path):
    (tmp_path / "agnostic.log.csv").write_text("iter,train_l1,val_psnr\n1000,0.02,31.5\n2000,0.01,33.25\n3000,0.01,33.0\n")
    (tmp_path / "fixed_imec.log.csv").write_text("iter,train_l1,val_psnr\n500,0.02,30.0\n")
    (tmp_path / "_running_fixed_osp.log.csv").write_text("iter,train_l1,val_psnr\n500,0.02,30.0\n")  # unfinished: ignored
    info = training_info(tmp_path)
    assert info == {"agnostic": {"iters": 3000, "best_val_psnr": 33.25}, "fixed_imec": {"iters": 500, "best_val_psnr": 30.0}}
    assert training_info(tmp_path / "missing") == {}
