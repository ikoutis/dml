"""The recovery tool's index grids must mirror the sbatch array maps and
produce collision-free run_ids across the whole suite (this test class is
what caught/guards the R1-vs-M3 duplicate-cell hazard)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from incomplete import GRIDS, compress, run_id_for  # noqa: E402

EXPECTED_SIZES = {"r1": 60, "m1": 60, "m2": 20, "m3": 25, "m4": 15,
                  "m5": 5, "m6": 30, "m1b": 10, "m7": 80}
# base suite (experiments.md §6); m1b [D-007] and m7 [D-011] are follow-ups.
_BASE_SUITE = ("r1", "m1", "m2", "m3", "m4", "m5", "m6")


def test_grid_sizes_match_sbatch_arrays():
    assert {e: n for e, (_, n, _, _) in GRIDS.items()} == EXPECTED_SIZES
    # base suite (experiments.md §6); m1b is a [D-007] follow-up on top.
    assert sum(EXPECTED_SIZES[e] for e in _BASE_SUITE) == 215


def test_run_ids_unique_across_suite():
    seen = {}
    for exp, (grid, n, _, _) in GRIDS.items():
        for i in range(n):
            rid = run_id_for(grid(i))
            assert rid not in seen, \
                f"{exp}[{i}] collides with {seen[rid]}: {rid}"
            seen[rid] = f"{exp}[{i}]"


def test_spot_checks():
    r1 = GRIDS["r1"][0]
    assert run_id_for(r1(0)) == "resnet32x2_cifar100_K2_indep_seed0001_d001"
    assert run_id_for(r1(59)) == \
        "wrn28x10x1-mobilenetx1_cifar100_K2_dml_seed0005_d001"
    m1 = GRIDS["m1"][0]
    assert run_id_for(m1(59)) == \
        "resnet32x8_cifar100_K8_mwmd1_noise40_seed0005_d001"
    m3 = GRIDS["m3"][0]
    assert run_id_for(m3(0)) == "resnet32x2_cifar100_K2_mwmd1_seed0001_d001"
    m6 = GRIDS["m6"][0]
    assert run_id_for(m6(5)) == \
        "wrn28x10x4-resnet32x4_cifar100_K8_mwmt1k0_seed0001_d001"


def test_compress():
    assert compress([0, 1, 2, 5, 7, 8]) == "0-2,5,7-8"
    assert compress([3]) == "3"
    assert compress([]) == ""
    assert compress([9, 1, 0]) == "0-1,9"
