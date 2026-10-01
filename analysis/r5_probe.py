"""R5b gradient probe: how far the KD gradient each model actually uses is from
the dense gradient, throughout training, for every teacher-sampling scheme.

Each probe row (one per probed update, model 0) gives
  rel_err = ||g_used - g_dense||^2 / ||g_dense||^2
  pred    = the sampled-peers proposition's expected rel_err for a FRESH
            uniform d-subset at the same parameters,
            (1/d) (K-1-d)/(K-2) tr(Sigma) / ||g_dense||^2
  updates_since_draw = updates since the teacher sets were drawn.

If teacher sets drawn once per epoch, or peeled matchings, behaved like a
fresh uniform draw at every step, mean rel_err / mean pred would be ~1 and
flat across the epoch. Ratios are of means (the ratio of an average error to
its average prediction), with seed-free bootstrap CIs over probe points.

    python analysis/r5_probe.py [--dir results/suite/r5_probe]
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd

BINS = [(0, 1), (1, 235), (235, 470), (470, 10 ** 9)]
BIN_NAMES = ["at_draw", "early", "middle", "late"]


def arm_of(run_id: str) -> str:
    return run_id.split("_K12_")[1].split("_seed")[0]


def describe(arm: str):
    if arm.startswith("unif") and arm.endswith("-step"):
        return "uniform, every update", int(arm[4:-5])
    if arm.startswith("unif"):
        return "uniform, every epoch", int(arm[4:])
    if arm.startswith("rand"):
        return "peeled matching, every epoch", int(arm[4:])
    return arm, -1


def ratio_ci(err, pred, rng, n_boot=2000):
    err, pred = np.asarray(err), np.asarray(pred)
    r = err.mean() / pred.mean()
    idx = rng.integers(0, len(err), size=(n_boot, len(err)))
    boots = err[idx].mean(1) / pred[idx].mean(1)
    return r, np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/suite/r5_probe")
    ap.add_argument("--skip_epochs", type=int, default=1,
                    help="drop the first epochs (random-init transient)")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    frames = [pd.read_csv(f) for f in
              sorted(glob.glob(os.path.join(args.dir, "*_gradprobe.csv")))]
    if not frames:
        raise SystemExit(f"no *_gradprobe.csv under {args.dir}")
    df = pd.concat(frames, ignore_index=True)
    df = df[df["epoch"] >= args.skip_epochs]
    df["arm"] = df["run_id"].map(arm_of)

    rows = []
    for arm, g in df.groupby("arm"):
        scheme, d = describe(arm)
        r, lo, hi = ratio_ci(g["rel_err"], g["pred_rel_var_uniform"], rng)
        rec = {"scheme": scheme, "d": d, "n_probes": len(g),
               "rel_err": g["rel_err"].mean(),
               "pred": g["pred_rel_var_uniform"].mean(),
               "ratio": r, "ratio_lo": lo, "ratio_hi": hi,
               "cos": g["cos"].mean()}
        for (a, b), name in zip(BINS, BIN_NAMES):
            gb = g[(g["updates_since_draw"] >= a) & (g["updates_since_draw"] < b)]
            rec[f"ratio_{name}"] = (gb["rel_err"].mean()
                                    / gb["pred_rel_var_uniform"].mean()
                                    if len(gb) else np.nan)
            rec[f"n_{name}"] = len(gb)
        rows.append(rec)
    out = pd.DataFrame(rows).sort_values(["scheme", "d"])

    print(f"Probe points from epoch {args.skip_epochs} on; ratio = mean "
          f"observed / mean predicted (fresh uniform draw), 95% bootstrap CI\n")
    print(f"{'scheme':<30}{'d':>3}{'n':>6}{'rel_err':>9}{'pred':>8}"
          f"{'ratio':>7}  {'95% CI':<15}{'cos':>6}   ratio by time since draw")
    for r in out.itertuples():
        by_time = "  ".join(
            f"{n}={getattr(r, 'ratio_' + n):.2f}"
            for n in BIN_NAMES if getattr(r, 'n_' + n) > 0)
        print(f"{r.scheme:<30}{r.d:>3}{r.n_probes:>6}{r.rel_err:>9.3f}"
              f"{r.pred:>8.3f}{r.ratio:>7.2f}  [{r.ratio_lo:.2f},{r.ratio_hi:.2f}]"
              f"{'':<2}{r.cos:>6.3f}   {by_time}")
    if args.csv:
        out.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
