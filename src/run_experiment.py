"""CLI entry point for the reduced-communication DML experiment suite.

Examples (see dev-communication/experiments.md for the full suite):

  # R1 replication cell: the paper's ResNet-32/ResNet-32 DML pair
  python -m src.run_experiment --cohort resnet32:2 --arm dml --seed 1 \
      --output_dir results/r1_pairs

  # M1 headline cell: MWM-matched mutual learning, K=8 heterogeneous
  python -m src.run_experiment --cohort wrn28x10:4,resnet32:4 --arm matched \
      --match_weight disagreement --seed 1 --output_dir results/m1_hetero

  # Smoke test (CPU, tiny):
  python -m src.run_experiment --cohort resnet20:4 --arm matched \
      --epochs 2 --batch_size 32 --n_valid 500 --limit_train 512 \
      --device cpu --output_dir results/smoke --verbose
"""

from __future__ import annotations

import argparse
import random
import sys
import time

import numpy as np
import torch

from .cohort import build_cohort, cohort_slug
from .data import load_data, make_loaders, num_classes
from .mutual_trainer import MutualTrainer, TrainerConfig


def str2bool(v: str) -> bool:
    return str(v).lower() in ("true", "1", "yes")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Reduced-communication DML")
    # data
    p.add_argument("--dataset", default="cifar100",
                   choices=["cifar100", "cifar10"])
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--download", action="store_true")
    p.add_argument("--n_valid", type=int, default=5000)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--label_noise_rate", type=float, default=0.0)
    p.add_argument("--noise_seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--eval_batch_size", type=int, default=500)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--limit_train", type=int, default=0,
                   help="Use only the first N training examples (smoke tests)")
    # cohort & arm
    p.add_argument("--cohort", default="resnet32:2",
                   help="e.g. 'resnet32:8' or 'wrn28x10:4,resnet32:4'")
    p.add_argument("--arm", default="dml",
                   choices=["indep", "dml", "matched", "topology"])
    p.add_argument("--target", default="peers", choices=["peers", "ensemble"],
                   help="dml arm only: 'ensemble' = DML_e (averaged target)")
    p.add_argument("--arm_label", default="",
                   help="Override the auto-generated arm label in run_id")
    # matched-arm knobs
    p.add_argument("--k_matchings", type=int, default=1)
    p.add_argument("--k_anneal", default="",
                   help="degree annealing schedule, e.g. '0:3,60:2,120:1'")
    p.add_argument("--match_weight", default="disagreement",
                   choices=["disagreement", "teachable", "accgap", "random",
                            "perclass", "errorfield"])
    p.add_argument("--kappa", type=float, default=1.0)
    p.add_argument("--rematch_every_epochs", type=int, default=1,
                   help="0 = static: match once at epoch 0 and freeze")
    p.add_argument("--recency_lambda", type=float, default=0.0)
    p.add_argument("--recency_gamma", type=float, default=0.5)
    p.add_argument("--peel_weighting", default="weight",
                   choices=["weight", "uniform"])
    p.add_argument("--graph", default="complete",
                   help="'complete', 'ring', 'prism', 'latticeK4', "
                        "'rregular:d', or 'clusters:m' (disconnected "
                        "m-cliques)")
    p.add_argument("--graph_seed", type=int, default=0)
    p.add_argument("--zombie_slot", type=int, default=-1,
                   help="[D-015] index of an implanted dead (uniform-logit, "
                        "never-training) cohort member; -1 = none")
    # optimization (DML-paper defaults)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=5e-4)
    p.add_argument("--nesterov", type=str2bool, default=True)
    p.add_argument("--lr_step", type=int, default=60)
    p.add_argument("--lr_gamma", type=float, default=0.1)
    p.add_argument("--kd_T", type=float, default=1.0)
    # bookkeeping
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--device", default="cuda")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--output_dir", default="results/dev")
    p.add_argument("--run_tag", default="",
                   help="dev-communication entry id, e.g. d001")
    p.add_argument("--checkpoint_every", type=int, default=25)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p


_WEIGHT_CODE = {"disagreement": "mwmd", "teachable": "mwmt",
                "accgap": "gap", "random": "rand",
                "perclass": "mwmpc", "errorfield": "mwmef"}


def auto_arm_label(args) -> str:
    zomb = f"-zomb{args.zombie_slot}" if args.zombie_slot >= 0 else ""
    if args.arm == "indep":
        return "indep"
    if args.arm == "dml":
        return ("dmle" if args.target == "ensemble" else "dml") + zomb
    if args.arm == "topology":
        # e.g. topo-ring, topo-prism, topo-rregular3 (':' stripped)
        return "topo-" + args.graph.replace(":", "") + zomb
    label = f"{_WEIGHT_CODE[args.match_weight]}{args.k_matchings}"
    if args.match_weight == "teachable" and args.kappa != 1.0:
        label += f"-k{args.kappa:g}"
    if args.k_anneal:
        from .mutual_trainer import parse_k_anneal
        label += "-anneal" + "".join(str(k) for _, k
                                     in parse_k_anneal(args.k_anneal))
    if args.peel_weighting == "uniform" and (args.k_matchings > 1
                                             or args.k_anneal):
        label += "-unif"
    if args.rematch_every_epochs == 0:
        label += "-static"
    elif args.rematch_every_epochs > 1:
        label += f"-re{args.rematch_every_epochs}"
    if args.recency_lambda > 0:
        label += f"-rec{args.recency_lambda:g}"
    if args.graph != "complete":
        label += f"-{args.graph.replace(':', '')}"
    if args.kd_T != 1.0:
        label += f"-T{args.kd_T:g}"
    return label + zomb


def compose_run_id(args, arm_label: str) -> str:
    parts = [cohort_slug(args.cohort), args.dataset,
             f"K{len_cohort(args.cohort)}", arm_label]
    if args.label_noise_rate > 0:
        parts.append(f"noise{int(round(args.label_noise_rate * 100))}")
    parts.append(f"seed{args.seed:04d}")
    if args.run_tag:
        parts.append(args.run_tag)
    return "_".join(parts)


def len_cohort(spec: str) -> int:
    from .cohort import parse_cohort_spec
    return len(parse_cohort_spec(spec))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.n_valid < 1:
        parser.error("--n_valid must be >= 1 (the matcher and the "
                     "policy-facing metrics evaluate on the validation set)")

    # Global seeding: init + batch order shared across arms at equal seed.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True

    device = args.device if (args.device == "cpu"
                             or torch.cuda.is_available()) else "cpu"
    if device != args.device:
        print(f"[warn] requested --device {args.device} but CUDA is "
              f"unavailable; falling back to CPU.", flush=True)

    train_set, valid_set, test_set, info = load_data(
        args.dataset, args.data_dir, n_valid=args.n_valid,
        split_seed=args.split_seed, label_noise_rate=args.label_noise_rate,
        noise_seed=args.noise_seed, download=args.download)
    if args.limit_train > 0:
        train_set = torch.utils.data.Subset(
            train_set, list(range(min(args.limit_train, len(train_set)))))
        info["n_train"] = len(train_set)
    train_loader, valid_loader, test_loader = make_loaders(
        train_set, valid_set, test_set, args.batch_size,
        args.eval_batch_size, args.seed, num_workers=args.num_workers,
        pin_memory=(device != "cpu"))

    n_cls = num_classes(args.dataset)
    slots = build_cohort(args.cohort, n_cls, device)
    K = len(slots)

    arm_label = args.arm_label or auto_arm_label(args)
    run_id = compose_run_id(args, arm_label)

    static_row = {
        "run_id": run_id, "run_tag": args.run_tag,
        "cohort": args.cohort, "dataset": args.dataset, "K": K,
        "arm": args.arm, "arm_label": arm_label, "target": args.target,
        "seed": args.seed, "epochs": args.epochs,
        "batch_size": args.batch_size, "base_lr": args.lr,
        "momentum": args.momentum, "weight_decay": args.weight_decay,
        "nesterov": args.nesterov, "graph_seed": args.graph_seed,
        "zombie_slot": args.zombie_slot,
        "lr_step": args.lr_step, "lr_gamma": args.lr_gamma,
        "kd_T": args.kd_T, "k_matchings": args.k_matchings,
        "k_anneal": args.k_anneal, "match_weight": args.match_weight,
        "kappa": args.kappa,
        "rematch_every_epochs": args.rematch_every_epochs,
        "recency_lambda": args.recency_lambda,
        "recency_gamma": args.recency_gamma,
        "peel_weighting": args.peel_weighting, "graph": args.graph,
        "label_noise_rate": args.label_noise_rate,
        "actual_noise_rate": round(info["actual_noise_rate"], 6),
        "noise_seed": args.noise_seed, "split_seed": args.split_seed,
        "n_train": info["n_train"], "n_valid": info["n_valid"],
        "n_test": info["n_test"],
        "cohort_params": sum(s.n_params for s in slots),
    }

    cfg = TrainerConfig(
        run_id=run_id, arm=args.arm, arm_label=arm_label, target=args.target,
        epochs=args.epochs, lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay, nesterov=args.nesterov,
        lr_step=args.lr_step, lr_gamma=args.lr_gamma, kd_T=args.kd_T,
        k_matchings=args.k_matchings, k_anneal=args.k_anneal,
        match_weight=args.match_weight, kappa=args.kappa,
        rematch_every_epochs=args.rematch_every_epochs,
        recency_lambda=args.recency_lambda,
        recency_gamma=args.recency_gamma,
        peel_weighting=args.peel_weighting, graph=args.graph,
        graph_seed=args.graph_seed, zombie_slot=args.zombie_slot,
        seed=args.seed, device=device,
        output_dir=args.output_dir, checkpoint_every=args.checkpoint_every,
        resume=args.resume, verbose=args.verbose, trap_usr1=True,
        static_row=static_row)

    if args.verbose:
        print(f"[*] run_id: {run_id}")
        for s in slots:
            print(f"[*]   {s.name}: {s.arch} ({s.n_params:,} params)")

    tic = time.time()
    trainer = MutualTrainer(cfg, slots, train_loader, valid_loader,
                            test_loader, n_cls, info["n_valid"])
    final = trainer.train()
    elapsed = time.time() - tic

    if trainer.preempted:
        # Exit 85 tells the sbatch script to `scontrol requeue` this array
        # task; the requeued run resumes from the checkpoint just written.
        sys.exit(85)

    print(f"RESULT | {run_id} | avg_test_acc={final.get('avg_test_acc', float('nan')):.6f} "
          f"| ensemble_test_acc={final.get('ensemble_test_acc', float('nan')):.6f} "
          f"| comm_bytes_cum={final.get('comm_bytes_cum', 0):.3e} "
          f"| elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
