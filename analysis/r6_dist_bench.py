"""R6: measured communication in the multi-node benchmark (src/dist_trainer.py).

Reads results/suite/r6_dist_bench/*_dist.csv (one row per epoch, written by
rank 0 with every rank's timings) and reports, from epoch 1 on (epoch 0
carries cuDNN autotuning and the cross-rank batch checks):

* per-update compute and communication time for each mode and network;
* the straggler test: rank 5 sleeps S ms per update before exchanging.
  A global collective makes every other rank wait ~S; pairwise exchange
  makes only the straggler's partner wait, so the other ranks average ~S/11.
  Reported: mean and max communication time over the non-straggler ranks,
  their excess over the S=0 run, and the slowest-rank epoch time.

    python analysis/r6_dist_bench.py [--dir results/suite/r6_dist_bench]
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd


def load(directory, skip):
    rows = []
    for f in sorted(glob.glob(os.path.join(directory, "*_dist.csv"))):
        df = pd.read_csv(f)
        df = df[df["epoch"] >= skip]
        if df.empty:
            continue
        rid = df["run_id"].iloc[0]
        net = "tcp" if rid.endswith("d021-tcp") else "ib"
        length = "full" if rid.endswith("_d021") else "short"
        K = int(df["K"].iloc[0])
        strag = int(df["straggler_rank"].iloc[0])
        rank_cols = [f"rank_{r:02d}_comm_ms" for r in range(K)]
        others = [c for r, c in enumerate(rank_cols) if r != strag]
        rows.append({
            "run_id": rid, "mode": df["mode"].iloc[0], "net": net,
            "length": length, "S_ms": float(df["straggler_ms"].iloc[0]),
            "epochs": len(df),
            "compute_ms": df["compute_ms"].mean(),
            "comm_ms": df["comm_ms"].mean(),
            "comm_p95_ms": df["comm_p95_ms"].mean(),
            "others_comm_mean": df[others].to_numpy().mean(),
            "others_comm_max": df[others].to_numpy().max(axis=1).mean(),
            "epoch_s": df["epoch_train_s"].mean(),
            "acc_final": df["avg_test_acc"].iloc[-1],
            "ens_final": df["ensemble_test_acc"].iloc[-1],
            "wall_s": df["wall_s"].iloc[-1],
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/suite/r6_dist_bench")
    ap.add_argument("--skip", type=int, default=1)
    args = ap.parse_args()
    df = load(args.dir, args.skip)
    if df.empty:
        raise SystemExit(f"no *_dist.csv with epochs >= {args.skip} in {args.dir}")

    print("== Per-update cost, no straggler (ms) ==")
    base = df[(df["S_ms"] == 0)]
    for r in base.sort_values(["net", "length", "mode"]).itertuples():
        frac = r.comm_ms / (r.compute_ms + r.comm_ms) if r.mode != "indep" else 0
        print(f"  {r.mode:<6}{r.net:<4}{r.length:<6} compute {r.compute_ms:7.2f}"
              f"  comm {r.comm_ms:7.3f} (p95 {r.comm_p95_ms:6.3f})"
              f"  comm share {100 * frac:5.1f}%  epoch {r.epoch_s:6.1f}s")

    print("\n== Straggler: rank 5 sleeps S ms/update (IB, short runs) ==")
    st = df[(df["net"] == "ib") & (df["length"] == "short")]
    for mode in ("dense", "p2p"):
        m = st[st["mode"] == mode].sort_values("S_ms")
        if m.empty:
            continue
        ref = m[m["S_ms"] == 0]
        ref_mean = ref["others_comm_mean"].iloc[0] if len(ref) else np.nan
        ref_epoch = ref["epoch_s"].iloc[0] if len(ref) else np.nan
        print(f"  {mode}:")
        for r in m.itertuples():
            print(f"    S={r.S_ms:4.0f}  other ranks wait {r.others_comm_mean:7.2f} ms"
                  f" (excess {r.others_comm_mean - ref_mean:+7.2f}, "
                  f"= {(r.others_comm_mean - ref_mean) / r.S_ms if r.S_ms else 0:4.2f} S)"
                  f"  worst other rank {r.others_comm_max:7.2f} ms"
                  f"  slowest-rank epoch {r.epoch_s:6.1f}s"
                  f" ({r.epoch_s - ref_epoch:+6.1f})")

    full = df[df["length"] == "full"]
    if not full.empty:
        print("\n== 200-epoch runs ==")
        for r in full.sort_values("mode").itertuples():
            print(f"  {r.mode:<6} final acc {100 * r.acc_final:.2f}"
                  f"  ens {100 * r.ens_final:.2f}  wall {r.wall_s / 3600:.2f} h")


if __name__ == "__main__":
    main()
