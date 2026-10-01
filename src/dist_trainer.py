"""Multi-node mutual learning with real communication: one model per process.

The suite trainer (mutual_trainer.py) runs a whole cohort in one process and
only *accounts* for communication. This trainer places each cohort member in
its own process (one GPU each, possibly on different nodes) and exchanges
posteriors over torch.distributed, so communication time, synchronization
and stragglers can be measured instead of modeled.

Modes (per update, after every model's forward pass on the shared batch):
* ``indep`` — no exchange; the compute-only baseline.
* ``dense`` — exact dense DML through the aggregate form: one all-reduce of
  the posterior sum, then p_bar_{-i} = (sum - p_i)/(K-1) locally.
* ``p2p``   — degree-d teacher sampling: each model swaps posteriors with its
  partners in d edge-disjoint random perfect matchings (isend/irecv only, no
  collective), redrawn once per epoch or every update from a seed shared by
  all ranks, so no coordination traffic is needed.

Every rank reads the same augmented batch (same seed, same loader); the first
batches are checksummed across ranks to verify it. Updates are simultaneous,
as in the suite. Timing per update, with CUDA synchronization between
phases: ``compute_ms`` (forward + loss/backward/step), ``comm_ms`` (the
exchange, including any wait for slower peers) and ``straggle_ms`` (an
injected delay on --straggler_rank, placed before the exchange to mimic a
slow device).

Launch one process per model, e.g. ``srun python -m src.dist_trainer ...``.
RANK/WORLD_SIZE/LOCAL_RANK are read from torchrun- or SLURM-style variables;
MASTER_ADDR/MASTER_PORT must be set. Writes ``{run_id}_dist.csv`` (one row
per epoch, from rank 0).
"""

from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .metrics import CsvWriter

MODES = ("indep", "dense", "p2p")


@dataclass
class DistConfig:
    mode: str = "dense"
    degree: int = 1
    rematch: str = "epoch"          # p2p: 'epoch' | 'step'
    epochs: int = 200
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    nesterov: bool = True
    lr_step: int = 60
    lr_gamma: float = 0.1
    kd_T: float = 1.0
    seed: int = 1
    straggler_rank: int = -1
    straggler_ms: float = 0.0
    check_batches: int = 2          # cross-rank batch checksums at start
    run_id: str = "dist"
    output_dir: str = "results/dist"


def draw_matchings(rng: np.random.Generator, K: int, d: int) -> List[List[int]]:
    """d edge-disjoint perfect matchings of K (even) nodes, as partner arrays.

    Uses the round-robin 1-factorization of the complete graph on a random
    relabelling of the nodes and picks d of its K-1 rounds at random. Every
    rank calls this with an identically seeded generator, so all ranks agree
    on the matchings without exchanging anything.
    """
    if K % 2:
        raise ValueError("p2p mode needs an even number of ranks")
    perm = rng.permutation(K)
    rounds = rng.choice(K - 1, size=d, replace=False)
    out = []
    for r in rounds:
        partner = [-1] * K
        # Circle method: fix position K-1, rotate the rest.
        a, b = int(perm[K - 1]), int(perm[r])
        partner[a], partner[b] = b, a
        for k in range(1, K // 2):
            u = int(perm[(r + k) % (K - 1)])
            v = int(perm[(r - k) % (K - 1)])
            partner[u], partner[v] = v, u
        out.append(partner)
    return out


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class DistMutualTrainer:
    def __init__(self, cfg: DistConfig, model: nn.Module, train_loader,
                 device, num_classes: int,
                 evaluate: Optional[Callable] = None):
        if cfg.mode not in MODES:
            raise ValueError(f"Unknown mode '{cfg.mode}'")
        if cfg.rematch not in ("epoch", "step"):
            raise ValueError(f"Unknown rematch '{cfg.rematch}'")
        self.cfg = cfg
        self.rank = dist.get_rank()
        self.K = dist.get_world_size()
        if cfg.mode == "p2p" and not 1 <= cfg.degree <= self.K - 1:
            raise ValueError(f"degree {cfg.degree} out of range for K={self.K}")
        self.model = model
        self.train_loader = train_loader
        self.device = device
        self.num_classes = num_classes
        self.evaluate = evaluate
        self.opt = torch.optim.SGD(model.parameters(), lr=cfg.lr,
                                   momentum=cfg.momentum,
                                   weight_decay=cfg.weight_decay,
                                   nesterov=cfg.nesterov)
        self.sched = torch.optim.lr_scheduler.StepLR(
            self.opt, step_size=cfg.lr_step, gamma=cfg.lr_gamma)
        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_kl = nn.KLDivLoss(reduction="batchmean")
        # Shared by every rank: identical seeds give identical matchings.
        self.match_rng = np.random.default_rng(cfg.seed * 7919 + 3)
        self.partners: List[List[int]] = []
        self.csv = None
        if self.rank == 0:
            os.makedirs(cfg.output_dir, exist_ok=True)
            self.csv = CsvWriter(os.path.join(cfg.output_dir,
                                              f"{cfg.run_id}_dist.csv"))
            self.csv.truncate_to_epoch(-1)

    # ------------------------------------------------------------------
    def _check_batch(self, x, y):
        s = torch.tensor([float(x.double().sum()), float(y.double().sum())],
                         dtype=torch.float64, device=self.device)
        gathered = [torch.zeros_like(s) for _ in range(self.K)]
        dist.all_gather(gathered, s)
        if any(not torch.equal(g, gathered[0]) for g in gathered):
            raise RuntimeError("ranks received different batches; the "
                               "shared-input assumption is violated")

    def _exchange(self, p: torch.Tensor) -> Optional[torch.Tensor]:
        """Return the teacher target for this rank (None in indep mode)."""
        if self.cfg.mode == "indep":
            return None
        if self.cfg.mode == "dense":
            total = p.clone()
            dist.all_reduce(total)
            return (total - p) / (self.K - 1)
        recv = [torch.empty_like(p) for _ in self.partners]
        ops = []
        for buf, partner in zip(recv, self.partners):
            j = partner[self.rank]
            ops.append(dist.P2POp(dist.isend, p, j))
            ops.append(dist.P2POp(dist.irecv, buf, j))
        for req in dist.batch_isend_irecv(ops):
            req.wait()
        return torch.stack(recv).mean(dim=0)

    def payload_bytes_per_update(self, batch: int) -> float:
        """Posterior bytes this rank receives per update (fp32)."""
        n = batch * self.num_classes * 4
        if self.cfg.mode == "dense":
            return 2.0 * (self.K - 1) / self.K * n   # ring all-reduce model
        if self.cfg.mode == "p2p":
            return float(self.cfg.degree * n)
        return 0.0

    # ------------------------------------------------------------------
    def train_one_epoch(self, epoch: int) -> dict:
        cfg, dev, T = self.cfg, self.device, self.cfg.kd_T
        self.model.train()
        if cfg.mode == "p2p" and cfg.rematch == "epoch":
            self.partners = draw_matchings(self.match_rng, self.K, cfg.degree)
        comp, comm, strag = [], [], []
        loss_sum, n_batches, n_seen = 0.0, 0, 0
        _sync(dev)
        t_epoch = time.perf_counter()
        for b, (x, y) in enumerate(self.train_loader):
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            if epoch == 0 and b < cfg.check_batches:
                self._check_batch(x, y)
            if cfg.mode == "p2p" and cfg.rematch == "step":
                self.partners = draw_matchings(self.match_rng, self.K,
                                               cfg.degree)
            _sync(dev)
            t0 = time.perf_counter()
            logits = self.model(x)
            p = F.softmax(logits.detach() / T, dim=1).contiguous()
            _sync(dev)
            t1 = time.perf_counter()
            if self.rank == cfg.straggler_rank and cfg.straggler_ms > 0:
                time.sleep(cfg.straggler_ms / 1000.0)
            t2 = time.perf_counter()
            target = self._exchange(p)
            _sync(dev)
            t3 = time.perf_counter()
            loss = self.loss_ce(logits, y)
            if target is not None:
                loss = loss + self.loss_kl(F.log_softmax(logits / T, dim=1),
                                           target) * (T * T)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            _sync(dev)
            t4 = time.perf_counter()
            comp.append((t1 - t0 + t4 - t3) * 1e3)
            comm.append((t3 - t2) * 1e3)
            strag.append((t2 - t1) * 1e3)
            loss_sum += float(loss.item())
            n_batches += 1
            n_seen += int(y.size(0))
        epoch_s = time.perf_counter() - t_epoch
        self.sched.step()
        comm_a = np.asarray(comm)
        return {
            "epoch_train_s": epoch_s,
            "compute_ms": float(np.mean(comp)),
            "comm_ms": float(comm_a.mean()),
            "comm_p50_ms": float(np.percentile(comm_a, 50)),
            "comm_p95_ms": float(np.percentile(comm_a, 95)),
            "straggle_ms": float(np.mean(strag)),
            "train_loss": loss_sum / max(n_batches, 1),
            "n_updates": n_batches,
            "payload_bytes": self.payload_bytes_per_update(1) * n_seen,
        }

    def _gather(self, values: List[float]) -> np.ndarray:
        t = torch.tensor(values, dtype=torch.float64, device=self.device)
        out = [torch.zeros_like(t) for _ in range(self.K)]
        dist.all_gather(out, t)
        return torch.stack(out).cpu().numpy()

    def train(self) -> None:
        cfg = self.cfg
        keys = ["epoch_train_s", "compute_ms", "comm_ms", "comm_p50_ms",
                "comm_p95_ms", "straggle_ms", "train_loss", "payload_bytes"]
        wall0 = time.perf_counter()
        for epoch in range(cfg.epochs):
            stats = self.train_one_epoch(epoch)
            acc, ens = (self.evaluate(self.model) if self.evaluate
                        else (float("nan"), float("nan")))
            per_rank = self._gather([stats[k] for k in keys] + [acc])
            if self.rank != 0:
                continue
            row = {"run_id": cfg.run_id, "mode": cfg.mode,
                   "degree": cfg.degree if cfg.mode == "p2p" else
                   (self.K - 1 if cfg.mode == "dense" else 0),
                   "rematch": cfg.rematch, "K": self.K, "seed": cfg.seed,
                   "straggler_rank": cfg.straggler_rank,
                   "straggler_ms": cfg.straggler_ms, "epoch": epoch,
                   "n_updates": stats["n_updates"],
                   "wall_s": time.perf_counter() - wall0,
                   # The slowest rank sets the pace of a synchronous epoch.
                   "epoch_train_s": float(per_rank[:, 0].max())}
            for c, k in enumerate(keys[1:], start=1):
                row[k] = float(per_rank[:, c].mean())
            row["comm_ms_max_rank"] = float(per_rank[:, 2].max())
            row["avg_test_acc"] = float(per_rank[:, -1].mean())
            row["ensemble_test_acc"] = ens
            for r in range(self.K):
                row[f"rank_{r:02d}_comm_ms"] = float(per_rank[r, 2])
            self.csv.write(row)
            print(f"[{cfg.run_id}] epoch {epoch + 1}/{cfg.epochs} "
                  f"acc={row['avg_test_acc']:.4f} "
                  f"comp={row['compute_ms']:.2f}ms "
                  f"comm={row['comm_ms']:.3f}ms "
                  f"epoch={row['epoch_train_s']:.1f}s", flush=True)
        if self.csv is not None:
            self.csv.close()


def make_evaluator(test_loader, device, num_classes: int):
    """Own-model test accuracy, plus the cohort ensemble via one all-reduce
    of the summed test posteriors (outside the timed training loop)."""
    def evaluate(model):
        model.eval()
        probs, ys = [], []
        with torch.no_grad():
            for x, y in test_loader:
                probs.append(F.softmax(model(x.to(device)), dim=1))
                ys.append(y.to(device))
        P, Y = torch.cat(probs), torch.cat(ys)
        acc = float((P.argmax(1) == Y).float().mean())
        dist.all_reduce(P)
        ens = float((P.argmax(1) == Y).float().mean())
        model.train()
        return acc, ens
    return evaluate


# ----------------------------------------------------------------------------
def _env_int(*names, default=None):
    for n in names:
        if n in os.environ:
            return int(os.environ[n])
    return default


def main() -> None:
    from .cohort import cohort_slug, parse_cohort_spec
    from .data import load_data, make_loaders, num_classes
    from .models import build_model

    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--mode", default="dense", choices=MODES)
    p.add_argument("--degree", type=int, default=1)
    p.add_argument("--rematch", default="epoch", choices=["epoch", "step"])
    p.add_argument("--cohort", default="resnet32:12",
                   help="one architecture per rank, e.g. resnet32:12")
    p.add_argument("--dataset", default="cifar100")
    p.add_argument("--data_dir", default="data")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--eval_batch_size", type=int, default=500)
    p.add_argument("--n_valid", type=int, default=5000)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--straggler_rank", type=int, default=-1)
    p.add_argument("--straggler_ms", type=float, default=0.0)
    p.add_argument("--backend", default="nccl", choices=["nccl", "gloo"])
    p.add_argument("--run_tag", default="")
    p.add_argument("--output_dir", default="results/dist")
    args = p.parse_args()

    rank = _env_int("RANK", "SLURM_PROCID")
    world = _env_int("WORLD_SIZE", "SLURM_NTASKS")
    local = _env_int("LOCAL_RANK", "SLURM_LOCALID", default=0)
    if rank is None or world is None:
        raise SystemExit("RANK/WORLD_SIZE (or SLURM_PROCID/SLURM_NTASKS) "
                         "must be set; launch one process per model")
    archs = parse_cohort_spec(args.cohort)
    if len(archs) != world:
        raise SystemExit(f"--cohort has {len(archs)} models but "
                         f"WORLD_SIZE={world}")
    use_cuda = args.backend == "nccl"
    device = torch.device(f"cuda:{local}" if use_cuda else "cpu")
    if use_cuda:
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
    dist.init_process_group(args.backend, rank=rank, world_size=world)

    # Identical global seeding on every rank: identical loader order and
    # augmentation. Every rank builds the whole cohort so model i starts from
    # the same initialization regardless of which rank holds it.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    train_set, _, test_set, _ = load_data(
        args.dataset, args.data_dir, n_valid=args.n_valid,
        split_seed=args.split_seed)
    train_loader, _, test_loader = make_loaders(
        train_set, train_set, test_set, args.batch_size,
        args.eval_batch_size, args.seed, num_workers=args.num_workers,
        pin_memory=use_cuda)
    n_cls = num_classes(args.dataset)
    models = [build_model(a, num_classes=n_cls) for a in archs]
    model = models[rank].to(device)
    del models

    label = {"indep": "indep", "dense": "dense",
             "p2p": f"p2p{args.degree}"
                    + ("-step" if args.rematch == "step" else "")}[args.mode]
    if args.straggler_rank >= 0 and args.straggler_ms > 0:
        label += f"-strag{args.straggler_rank}x{args.straggler_ms:g}ms"
    run_id = "_".join(x for x in [cohort_slug(args.cohort), args.dataset,
                                  f"K{world}", label,
                                  f"seed{args.seed:04d}", args.run_tag] if x)
    cfg = DistConfig(mode=args.mode, degree=args.degree, rematch=args.rematch,
                     epochs=args.epochs, lr=args.lr, seed=args.seed,
                     straggler_rank=args.straggler_rank,
                     straggler_ms=args.straggler_ms, run_id=run_id,
                     output_dir=args.output_dir)
    trainer = DistMutualTrainer(cfg, model, train_loader, device, n_cls,
                                evaluate=make_evaluator(test_loader, device,
                                                        n_cls))
    if rank == 0:
        print(f"[*] {run_id}: K={world}, backend={args.backend}, "
              f"{len(train_loader)} updates/epoch", flush=True)
    trainer.train()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
