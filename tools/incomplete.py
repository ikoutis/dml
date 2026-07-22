"""Print the SLURM array indices of an experiment's incomplete runs, as an
--array expression, so recovery after ANY interruption (cancelled array,
node failure, lost requeue) is one line:

    IDS=$(python tools/incomplete.py m1)
    [ -n "$IDS" ] && sbatch --array=$IDS slurm/m1_headline_k8.sbatch

A run is complete when its metrics CSV has reached the final epoch
(--epochs, default 200 -> epoch 199). Runs that self-requeued at the 72 h
wall, were preempted, or died mid-way are reported and simply resume from
their checkpoints when resubmitted.

The index grids below MIRROR the slurm/*.sbatch array maps — if a script's
map changes, change it here too (tests/test_suite_grids.py pins sizes and
run_id uniqueness across the whole suite).

    python tools/incomplete.py m1 --list      # per-index status table
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.run_experiment import auto_arm_label, build_parser, compose_run_id  # noqa: E402

HETERO = "wrn28x10:4,resnet32:4"
TAG = ["--run_tag", "d001"]


def _r1(i):
    cohorts = ["resnet32:2", "mobilenet:1,resnet32:1", "mobilenet:2",
               "wrn28x10:1,resnet32:1", "wrn28x10:2", "wrn28x10:1,mobilenet:1"]
    arms = ["indep", "dml"]
    return (["--cohort", cohorts[i // 10], "--arm", arms[(i // 5) % 2],
             "--seed", str(i % 5 + 1)] + TAG)


def _m1(i):
    regimes = [("resnet32:8", "0.0"), (HETERO, "0.0"), ("resnet32:8", "0.4")]
    arms = [["--arm", "indep"], ["--arm", "dml"],
            ["--arm", "matched", "--match_weight", "random"],
            ["--arm", "matched", "--match_weight", "disagreement"]]
    cohort, noise = regimes[i // 20]
    return (["--cohort", cohort, "--label_noise_rate", noise,
             "--seed", str(i % 5 + 1)] + arms[(i // 5) % 4] + TAG)


def _m2(i):
    arms = [["--arm", "matched", "--match_weight", "disagreement",
             "--k_matchings", "2"],
            ["--arm", "matched", "--match_weight", "disagreement",
             "--k_matchings", "3"],
            ["--arm", "matched", "--match_weight", "disagreement",
             "--k_anneal", "0:3,60:2,120:1"],
            ["--arm", "matched", "--match_weight", "disagreement",
             "--k_matchings", "2", "--peel_weighting", "uniform"]]
    return (["--cohort", "resnet32:8", "--seed", str(i % 5 + 1)]
            + arms[i // 5] + TAG)


def _m3(i):
    seed = str(i % 5 + 1)
    if i < 5:
        k, arm_idx = 2, 1
    else:
        j = i - 5
        k = 4 if j // 10 == 0 else 12
        arm_idx = (j // 5) % 2
    arm = (["--arm", "dml"] if arm_idx == 0 else
           ["--arm", "matched", "--match_weight", "disagreement"])
    return ["--cohort", f"resnet32:{k}", "--seed", seed] + arm + TAG


def _m4(i):
    arms = [["--arm", "matched", "--match_weight", "disagreement",
             "--rematch_every_epochs", "0"],
            ["--arm", "matched", "--match_weight", "disagreement",
             "--recency_lambda", "0.5"],
            ["--arm", "matched", "--match_weight", "disagreement",
             "--recency_lambda", "1.0"]]
    return (["--cohort", HETERO, "--seed", str(i % 5 + 1)]
            + arms[i // 5] + TAG)


def _m5(i):
    return (["--cohort", "resnet32:8", "--arm", "dml", "--target", "ensemble",
             "--seed", str(i + 1)] + TAG)


def _m6(i):
    regimes = [(HETERO, "0.0"), ("resnet32:8", "0.4")]
    arms = [["--arm", "matched", "--match_weight", "teachable",
             "--kappa", "1.0"],
            ["--arm", "matched", "--match_weight", "teachable",
             "--kappa", "0.0", "--arm_label", "mwmt1k0"],
            ["--arm", "matched", "--match_weight", "accgap"]]
    cohort, noise = regimes[i // 15]
    return (["--cohort", cohort, "--label_noise_rate", noise,
             "--seed", str(i % 5 + 1)] + arms[(i // 5) % 3] + TAG)


def _m1b(i):  # [D-007] follow-up: per-class matched arm, clean + noise
    noise = "0.0" if i < 5 else "0.4"
    return (["--cohort", "resnet32:8", "--arm", "matched",
             "--match_weight", "perclass", "--label_noise_rate", noise,
             "--seed", str(i % 5 + 1)] + TAG)


def _m7(i):  # [D-011] Tier 1: fixed communication topologies, K=12, clean+noise
    # Own run_tag d011: the dml/mwmd1 K=12 cells would otherwise collide with
    # M3's cohort-scaling cells (also resnet32:12 dml/mwmd1). Self-contained.
    # Indices 70-79: [D-012] connectivity probe (disconnected cliques, clean).
    if i >= 70:
        graph = "clusters:2" if i < 75 else "clusters:4"
        return (["--cohort", "resnet32:12", "--arm", "topology",
                 "--graph", graph, "--seed", str(i % 5 + 1),
                 "--run_tag", "d011"])
    noise = "0.0" if i < 35 else "0.4"
    j = i % 35
    arms = [["--arm", "indep"],
            ["--arm", "matched", "--match_weight", "disagreement",
             "--k_matchings", "1"],
            ["--arm", "topology", "--graph", "ring"],
            ["--arm", "matched", "--match_weight", "disagreement",
             "--k_matchings", "2"],
            ["--arm", "topology", "--graph", "prism"],
            ["--arm", "topology", "--graph", "rregular:3"],
            ["--arm", "dml"]]
    return (["--cohort", "resnet32:12", "--label_noise_rate", noise,
             "--seed", str(j % 5 + 1)] + arms[j // 5] + ["--run_tag", "d011"])


def _m7l(i):  # [D-013] weak-learner probe: lenet:12 ladder + clusters, clean
    arms = [["--arm", "indep"],
            ["--arm", "matched", "--match_weight", "disagreement",
             "--k_matchings", "1"],
            ["--arm", "topology", "--graph", "ring"],
            ["--arm", "matched", "--match_weight", "disagreement",
             "--k_matchings", "2"],
            ["--arm", "topology", "--graph", "prism"],
            ["--arm", "topology", "--graph", "rregular:3"],
            ["--arm", "topology", "--graph", "clusters:2"],
            ["--arm", "topology", "--graph", "clusters:4"],
            ["--arm", "dml"]]
    return (["--cohort", "lenet:12", "--seed", str(i % 5 + 1)]
            + arms[i // 5] + ["--run_tag", "d013"])


def _m9(i):  # [D-015] controlled epidemiology: one zombie, resnet32:12, clean
    arms = [["--arm", "topology", "--graph", "ring"],
            ["--arm", "topology", "--graph", "prism"],
            ["--arm", "topology", "--graph", "rregular:3"],
            ["--arm", "topology", "--graph", "clusters:4"],
            ["--arm", "matched", "--match_weight", "random",
             "--k_matchings", "1"],
            ["--arm", "matched", "--match_weight", "random",
             "--k_matchings", "2"],
            ["--arm", "dml"]]
    return (["--cohort", "resnet32:12", "--zombie_slot", "0",
             "--seed", str(i % 5 + 1)] + arms[i // 5]
            + ["--run_tag", "d015"])


# exp -> (index->argv, n_tasks, output_dir, sbatch script)
GRIDS = {
    "r1": (_r1, 60, "results/suite/r1_pairs", "slurm/r1_pairs.sbatch"),
    "m1": (_m1, 60, "results/suite/m1_headline", "slurm/m1_headline_k8.sbatch"),
    "m2": (_m2, 20, "results/suite/m2_degree_dial", "slurm/m2_degree_dial.sbatch"),
    "m3": (_m3, 25, "results/suite/m3_cohort_scaling", "slurm/m3_cohort_scaling.sbatch"),
    "m4": (_m4, 15, "results/suite/m4_rotation", "slurm/m4_rotation.sbatch"),
    "m5": (_m5, 5, "results/suite/m5_target_structure", "slurm/m5_target_structure.sbatch"),
    "m6": (_m6, 30, "results/suite/m6_weight_signal", "slurm/m6_weight_signal.sbatch"),
    "m1b": (_m1b, 10, "results/suite/m1_headline", "slurm/m1b_perclass.sbatch"),
    "m7": (_m7, 80, "results/suite/m7_topology", "slurm/m7_topology.sbatch"),
    "m7l": (_m7l, 45, "results/suite/m7l_lenet", "slurm/m7l_lenet.sbatch"),
    "m9": (_m9, 35, "results/suite/m9_zombie", "slurm/m9_zombie.sbatch"),
}


def run_id_for(argv) -> str:
    args = build_parser().parse_args(argv)
    return compose_run_id(args, args.arm_label or auto_arm_label(args))


def last_epoch(metrics_path: str):
    if not os.path.exists(metrics_path):
        return None
    last = None
    with open(metrics_path, newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("epoch") not in (None, ""):
                last = int(float(row["epoch"]))
    return last


def compress(indices) -> str:
    """[0,1,2,5,7,8] -> '0-2,5,7-8' (SLURM --array syntax)."""
    out, i = [], 0
    idx = sorted(indices)
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and idx[j + 1] == idx[j] + 1:
            j += 1
        out.append(str(idx[i]) if i == j else f"{idx[i]}-{idx[j]}")
        i = j + 1
    return ",".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("exp", choices=sorted(GRIDS))
    ap.add_argument("--results_dir", default=None,
                    help="Override the experiment's output dir")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--list", action="store_true",
                    help="Print a per-index status table to stderr")
    args = ap.parse_args()

    grid, n_tasks, out_dir, script = GRIDS[args.exp]
    out_dir = args.results_dir or out_dir
    target = args.epochs - 1

    missing = []
    for i in range(n_tasks):
        rid = run_id_for(grid(i))
        ep = last_epoch(os.path.join(out_dir, rid + "_metrics.csv"))
        done = ep is not None and ep >= target
        if not done:
            missing.append(i)
        if args.list:
            state = ("complete" if done else
                     f"at epoch {ep}" if ep is not None else "not started")
            print(f"  [{i:3d}] {rid}: {state}", file=sys.stderr)

    if missing:
        print(f"[{args.exp}] {len(missing)}/{n_tasks} incomplete -> "
              f"sbatch --array=$(python tools/incomplete.py {args.exp}) "
              f"{script}", file=sys.stderr)
        print(compress(missing))
    else:
        print(f"[{args.exp}] all {n_tasks} runs complete.", file=sys.stderr)


if __name__ == "__main__":
    main()
