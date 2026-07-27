"""Cohort trainer for deep mutual learning and its reduced-communication
(matched) variants.

Arms (dev-communication/experiments.md §1)
------------------------------------------
* ``indep``   — K models trained side by side on CE only (no coupling). Same
  init, data order, and step count as the coupled arms at equal seed.
* ``dml``     — dense DML (Zhang et al., CVPR 2018): every model mimics all
  K-1 peers, L_i = CE(z_i, y) + (1/(K-1)) * sum_j KL(p_j || p_i).
  With ``--target ensemble`` this becomes DML_e: a single KL to the peers'
  averaged posterior (the paper's Eq. 11 variant, expected WORSE — §3.6).
* ``matched`` — per-refresh matching-based mimicry: each model mimics the
  k partners assigned by (peeled) maximum-weight matching on the chosen
  edge-weight signal. k=1 random == Def-KT-style unselective pairing;
  k=1 + disagreement/teachable weights == MWM matched mutual learning.

Update rule (declared design decision #1, from knowledge-diffusion
dev-communication/ideas.md 2026-07-17 notes): SIMULTANEOUS. For every batch,
all K models' logits are computed (train mode, one forward per model) BEFORE
any model takes its gradient step; mimicry targets are the detached pre-step
logits. No model ever trains on a peer's already-updated weights within a
batch. This holds identically in every arm. (The original DML paper describes
alternating/sequential updates; the root-level legacy trainer.py is
effectively simultaneous. We standardize on simultaneous everywhere.)

Total mimicry mass is 1 in every coupled arm (dense averages K-1 KLs;
matched arms normalize their k alphas to sum to 1), so arms differ in the
STRUCTURE of the mimicry signal and in communication, not in loss magnitude.

Communication accounting (analytic, logged per epoch; §1.4 of the design):
* logit floats sent per model = (#partners this model teaches) x (examples
  seen this epoch) x (num classes). Dense/DML_e: K-1 partners; matched: k.
* matcher overhead per model = n_valid hard predictions (ints) per refresh
  that uses measured weights (disagreement/teachable/accgap); 0 for random
  matching, static epochs, dense, and indep.
* bytes = 4 x floats + 2 x ints (fp32 logits, int16 class ids).
"""

from __future__ import annotations

import os
import random
import signal
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import matching as mt
from .cohort import Slot
from .metrics import (CsvWriter, evaluate_cohort, mean_pairwise_disagreement,
                      mean_pairwise_error_correlation)

ARMS = ("indep", "dml", "matched", "topology")


class _ZombieModel(nn.Module):
    """A dead cohort member for the controlled-epidemiology arm ([D-015]).

    Emits exactly-zero logits — the uniform posterior, which is what an
    actually-collapsed model converges to ([D-014]) — everywhere it is
    observed: as a KD teacher, in eval, in the ensemble (where a uniform
    posterior shifts every class equally and so never changes the argmax).
    Never trains. The single dummy parameter keeps the per-slot optimizer/
    scheduler/checkpoint plumbing uniform; its grad stays None, so SGD
    (including weight decay) never touches it.
    """

    def __init__(self, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x.new_zeros(x.shape[0], self.num_classes)


@dataclass
class TrainerConfig:
    run_id: str
    arm: str = "dml"
    arm_label: str = ""
    target: str = "peers"           # dml only: 'peers' | 'ensemble' (DML_e)
    epochs: int = 200
    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    nesterov: bool = True
    lr_step: int = 60
    lr_gamma: float = 0.1
    kd_T: float = 1.0               # DML paper uses T=1 (no temperature)
    # matched-arm knobs
    k_matchings: int = 1
    k_anneal: str = ""              # e.g. "0:3,60:2,120:1"
    match_weight: str = "disagreement"
    kappa: float = 1.0
    rematch_every_epochs: int = 1   # 0 = static (match once at epoch 0)
    recency_lambda: float = 0.0
    recency_gamma: float = 0.5
    peel_weighting: str = "weight"  # 'weight' | 'uniform'
    graph: str = "complete"
    graph_seed: int = 0
    zombie_slot: int = -1           # [D-015]: index of the implanted dead
                                    # model (-1 = none). See _ZombieModel.
    # [D-018] temporally sparse communication. Distil on `comm_on` of every
    # `comm_block` updates (0/0 = every update, the historical behaviour);
    # `kd_scale` multiplies the KD term on active updates, so a schedule can
    # be run at matched time-integrated distillation dose.
    comm_on: int = 0
    comm_block: int = 0
    kd_scale: float = 1.0
    # How to charge the communication ledger. 'p2p' bills `degree` posterior
    # streams per update (point-to-point exchange, the historical behaviour);
    # 'allreduce' bills 2(K-1)/K streams, the ring-all-reduce cost of the
    # aggregate form, which is what exact dense DML actually needs.
    comm_accounting: str = "p2p"
    # bookkeeping
    seed: int = 1
    device: str = "cuda"
    output_dir: str = "results/dev"
    checkpoint_every: int = 25
    resume: bool = False
    verbose: bool = False
    # SLURM wall-clock resilience: when True, SIGUSR1 (sent by
    # `#SBATCH --signal=B:USR1@...` ahead of the 72 h wall and forwarded by
    # the sbatch script) makes the trainer checkpoint at the end of the
    # current epoch and stop with .preempted = True, so the CLI can exit 85
    # and the script can requeue the array task.
    trap_usr1: bool = False
    static_row: Dict = field(default_factory=dict)  # run-identity CSV columns


def parse_k_anneal(spec: str) -> List[Tuple[int, int]]:
    """'0:3,60:2,120:1' -> [(0, 3), (60, 2), (120, 1)], sorted by epoch."""
    if not spec:
        return []
    out = []
    for part in spec.split(","):
        e, k = part.strip().split(":")
        out.append((int(e), int(k)))
    return sorted(out)


class MutualTrainer:
    def __init__(self, cfg: TrainerConfig, slots: List[Slot],
                 train_loader, valid_loader, test_loader,
                 num_classes: int, n_valid: int):
        if cfg.arm not in ARMS:
            raise ValueError(f"Unknown arm '{cfg.arm}'. Arms: {ARMS}")
        self.cfg = cfg
        self.slots = slots
        self.K = len(slots)
        self.models = [s.model for s in slots]
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.num_classes = num_classes
        self.n_valid = n_valid
        self.device = torch.device(cfg.device)

        # [D-015] controlled epidemiology: replace one slot with a dead
        # model AFTER the whole cohort is built, so the healthy slots keep
        # bit-identical inits with the no-zombie anchor runs at equal seed
        # (the replacement consumes no RNG). arch = 'zombie' automatically
        # keeps it out of the per-architecture mean columns.
        if cfg.zombie_slot >= 0:
            z = cfg.zombie_slot
            if z >= self.K:
                raise ValueError(f"zombie_slot {z} out of range for K={self.K}")
            if cfg.arm == "indep":
                raise ValueError("zombie in an indep cohort is a no-op; "
                                 "use the no-zombie anchors instead")
            if cfg.arm == "matched" and cfg.match_weight != "random":
                # Measured weights would glue the matcher to the zombie (it
                # maximally disagrees with everyone), confounding exposure.
                raise ValueError("zombie + measured match weights is "
                                 "confounded; use --match_weight random "
                                 "([D-015])")
            zm = _ZombieModel(num_classes).to(self.device)
            self.slots[z].model = zm
            self.slots[z].arch = "zombie"
            self.slots[z].n_params = 0
            self.models[z] = zm

        if cfg.arm == "matched":
            if self.K % 2 != 0:
                raise ValueError("matched arms need an even cohort size K")
            ks = [cfg.k_matchings] + [k for _, k in parse_k_anneal(cfg.k_anneal)]
            for k in ks:
                if not 1 <= k <= self.K - 1:
                    raise ValueError(
                        f"k={k} out of range [1, K-1={self.K - 1}] "
                        f"(k_matchings/k_anneal)")
                if k > self.K // 2:
                    print(f"[warn] k={k} > K/2={self.K // 2}: peel "
                          f"feasibility is no longer guaranteed (Dirac "
                          f"bound) and refresh may raise.", flush=True)

        self.loss_ce = nn.CrossEntropyLoss()
        self.loss_kl = nn.KLDivLoss(reduction="batchmean")

        self.optimizers = [
            torch.optim.SGD(m.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                            weight_decay=cfg.weight_decay,
                            nesterov=cfg.nesterov)
            for m in self.models
        ]
        self.schedulers = [
            torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.lr_step,
                                            gamma=cfg.lr_gamma)
            for opt in self.optimizers
        ]

        # Matcher machinery. The matcher RNG is separate from the training
        # RNG so arms at equal seed share init and batch order.
        self.match_rng = np.random.default_rng(cfg.seed * 100003 + 17)
        self.graph_mask = (None if cfg.graph == "complete"
                           else mt.build_graph_mask(cfg.graph, self.K,
                                                    cfg.graph_seed))
        self.recency = mt.RecencyState(K=self.K, gamma=cfg.recency_gamma)
        self.k_schedule = parse_k_anneal(cfg.k_anneal)
        self.teachers: List[List[Tuple[int, float]]] = [[] for _ in range(self.K)]
        self.current_matchings: List[List[mt.Pair]] = []
        self.static_matched = False
        self.cached_val: Optional[Dict] = None  # last val evaluation

        # DML_e (target='ensemble') at K=2 coincides with plain DML — the
        # ensemble of the OTHERS is the single peer — so it uses the peers
        # path there.
        if cfg.arm == "dml" and (cfg.target == "peers" or self.K == 2):
            w = 1.0 / max(self.K - 1, 1)
            self.teachers = [[(j, w) for j in range(self.K) if j != i]
                             for i in range(self.K)]

        # Static-topology arm (M7): each model distills from ALL its fixed
        # graph neighbours (averaged-neighbour target, alpha = 1/degree),
        # set once and never refreshed — decentralized, no coordinator, no
        # matcher probes. The graph is fixed by cfg.graph (+ graph_seed for
        # the random-regular expanders).
        self.graph_degree = 0
        if cfg.arm == "topology":
            if self.graph_mask is None:
                raise ValueError("topology arm needs --graph != complete "
                                 "(use ring/prism/rregular:d/latticeK4/"
                                 "clusters:m)")
            neigh = mt.graph_neighbors(self.graph_mask)
            degs = {len(n) for n in neigh}
            self.teachers = [[(j, 1.0 / len(n)) for j in n] if n else []
                             for n in neigh]
            self.graph_degree = self.graph_mask.sum(axis=1).max()
            self._spectral_gap = mt.graph_spectral_gap(self.graph_mask)
            if cfg.verbose:
                print(f"[*] topology '{cfg.graph}': degrees={sorted(degs)}, "
                      f"spectral_gap={self._spectral_gap:.4f}", flush=True)

        # Cumulative per-model communication counters (uniform across the
        # cohort — perfect matchings and dense coupling are degree-regular).
        self.comm_floats_cum = 0.0
        self.comm_matcher_ints_cum = 0.0

        os.makedirs(cfg.output_dir, exist_ok=True)
        base = os.path.join(cfg.output_dir, cfg.run_id)
        self.metrics_csv = CsvWriter(base + "_metrics.csv")
        self.matches_csv = CsvWriter(base + "_matches.csv")
        self.ckpt_path = base + "_ckpt.pt"
        self.start_epoch = 0
        self._preempt_requested = False
        self.preempted = False

    # ------------------------------------------------------------------
    # Matching refresh
    # ------------------------------------------------------------------
    def _current_k(self, epoch: int) -> int:
        k = self.cfg.k_matchings
        for e, kk in self.k_schedule:
            if epoch >= e:
                k = kk
        return k

    def _refresh_due(self, epoch: int) -> bool:
        if self.cfg.arm != "matched":
            return False
        if self.cfg.rematch_every_epochs == 0:
            return not self.static_matched
        return epoch % self.cfg.rematch_every_epochs == 0

    def _val_preds_for_matcher(self) -> Dict:
        if self.cached_val is None:
            self.cached_val = evaluate_cohort(self.models, self.valid_loader,
                                              self.device)
        return self.cached_val

    def _refresh_matching(self, epoch: int) -> None:
        cfg = self.cfg
        k = self._current_k(epoch)
        needs_preds = cfg.match_weight != "random"
        preds, y = None, None
        if needs_preds:
            ev = self._val_preds_for_matcher()
            preds, y = ev["preds"], ev["y"]
            # Matcher overhead: every model ships its n_valid hard preds.
            self.comm_matcher_ints_cum += self.n_valid
            self._matcher_ints_epoch = float(self.n_valid)
        else:
            self._matcher_ints_epoch = 0.0

        W = mt.edge_weights(cfg.match_weight, preds, y, cfg.kappa, self.K,
                            self.match_rng)
        Wp = self.recency.penalize(W, cfg.recency_lambda, self.graph_mask)
        matchings = mt.peel_matchings(Wp, k, self.graph_mask)
        self.recency.update(matchings)
        self.current_matchings = matchings
        # Alphas come from the RAW weights (the penalty only steers
        # selection); dense normalization to mass 1 happens inside.
        self.teachers = mt.matchings_to_teachers(matchings, W, self.K,
                                                 cfg.peel_weighting)
        if cfg.rematch_every_epochs == 0:
            self.static_matched = True

        solver = "exact" if self.K <= mt._EXACT_MAX_K else "greedy"
        run_tag = cfg.static_row.get("run_tag", "")
        for layer, M in enumerate(matchings):
            for i, j in M:
                self.matches_csv.write({
                    "run_id": cfg.run_id, "run_tag": run_tag,
                    "epoch": epoch, "layer": layer,
                    "i": i, "j": j,
                    "arch_i": self.slots[i].arch, "arch_j": self.slots[j].arch,
                    "weight": f"{W[i, j]:.6f}",
                    "penalized_weight": f"{Wp[i, j]:.6f}",
                    "weight_mode": cfg.match_weight, "solver": solver,
                    "k": k,
                })

    # ------------------------------------------------------------------
    # Communication schedule ([D-018])
    # ------------------------------------------------------------------
    def _comm_active(self, batch_idx: int) -> bool:
        """Is this update one of the `comm_on`-in-`comm_block` distilling ones?

        Uses the standard even-spacing rule ``(t * on) % block < on``, which
        for on=6, block=11 selects updates 1, 3, 5, 7, 9, 11 of each block and
        yields exactly `on` active updates per block whenever gcd(on, block)=1.
        The index is the batch position within the epoch, so the schedule is
        deterministic and unaffected by resume.
        """
        on, block = self.cfg.comm_on, self.cfg.comm_block
        if not block or on >= block:
            return True
        return (batch_idx * on) % block < on

    def _stream_multiplier(self) -> float:
        """Posterior streams billed per communicating update, per model."""
        if self.cfg.comm_accounting == "allreduce":
            return 0.0 if self.K < 2 else 2.0 * (self.K - 1) / self.K
        return float(self._degree())

    # ------------------------------------------------------------------
    # One epoch of simultaneous mutual training
    # ------------------------------------------------------------------
    def _train_one_epoch(self, epoch: int) -> Dict:
        cfg = self.cfg
        T = cfg.kd_T
        for m in self.models:
            m.train()
        sums = {"loss": np.zeros(self.K), "ce": np.zeros(self.K),
                "kd": np.zeros(self.K)}
        n_batches = 0
        n_seen = 0
        n_seen_comm = 0
        n_batches_comm = 0
        for x, y in self.train_loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            # Simultaneous update: all forwards precede all steps.
            outputs = [m(x) for m in self.models]
            detached = [o.detach() for o in outputs]

            # [D-018]: on non-distilling updates every model trains on CE
            # alone and nothing is exchanged.
            comm_active = self._comm_active(n_batches)

            if (comm_active and cfg.arm == "dml"
                    and cfg.target == "ensemble" and self.K > 2):
                probs = torch.stack([F.softmax(d / T, dim=1)
                                     for d in detached])
                probs_sum = probs.sum(dim=0)

            for i in range(self.K):
                if i == cfg.zombie_slot:
                    continue    # the dead model never trains
                ce = self.loss_ce(outputs[i], y)
                kd = outputs[i].new_zeros(())
                if not comm_active:
                    pass    # CE-only update
                elif cfg.arm == "dml" and cfg.target == "ensemble" and self.K > 2:
                    # DML_e: single KL to the mean posterior of the OTHERS.
                    p_ens = (probs_sum - probs[i]) / (self.K - 1)
                    kd = self.loss_kl(F.log_softmax(outputs[i] / T, dim=1),
                                      p_ens) * (T * T)
                else:
                    for j, alpha in self.teachers[i]:
                        kd = kd + alpha * self.loss_kl(
                            F.log_softmax(outputs[i] / T, dim=1),
                            F.softmax(detached[j] / T, dim=1)) * (T * T)
                if comm_active and cfg.kd_scale != 1.0:
                    kd = kd * cfg.kd_scale
                loss = ce + kd
                self.optimizers[i].zero_grad()
                loss.backward()
                self.optimizers[i].step()

                sums["loss"][i] += float(loss.item())
                sums["ce"][i] += float(ce.item())
                sums["kd"][i] += float(kd.item())
            n_batches += 1
            n_seen += int(y.size(0))
            if comm_active:
                n_batches_comm += 1
                n_seen_comm += int(y.size(0))

        for key in sums:
            sums[key] = sums[key] / max(n_batches, 1)
        sums["n_seen"] = n_seen
        sums["n_seen_comm"] = n_seen_comm
        sums["comm_duty"] = n_batches_comm / max(n_batches, 1)
        return sums

    # ------------------------------------------------------------------
    # Communication accounting
    # ------------------------------------------------------------------
    def _degree(self) -> int:
        """Partners each model exchanges per-batch logits with."""
        if self.cfg.arm == "indep":
            return 0
        if self.cfg.arm == "dml":
            return self.K - 1
        if self.cfg.arm == "topology":
            return int(self.graph_degree)
        return len(self.current_matchings)

    def _comm_epoch(self, n_seen_comm: int) -> Dict[str, float]:
        """Bytes exchanged this epoch, billed only on communicating updates."""
        floats = self._stream_multiplier() * n_seen_comm * self.num_classes
        ints = getattr(self, "_matcher_ints_epoch", 0.0)
        self.comm_floats_cum += floats
        self._matcher_ints_epoch = 0.0
        bytes_epoch = 4.0 * floats + 2.0 * ints
        bytes_cum = 4.0 * self.comm_floats_cum + 2.0 * self.comm_matcher_ints_cum
        return {
            "comm_logit_floats_epoch": floats,
            "comm_matcher_ints_epoch": ints,
            "comm_bytes_epoch": bytes_epoch,
            "comm_bytes_cum": bytes_cum,
            "comm_bytes_cum_cohort": bytes_cum * self.K,
        }

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def _save_checkpoint(self, epoch: int) -> None:
        state = {
            "epoch": epoch,
            "models": [m.state_dict() for m in self.models],
            "optimizers": [o.state_dict() for o in self.optimizers],
            "schedulers": [s.state_dict() for s in self.schedulers],
            "torch_rng": torch.get_rng_state(),
            # The train loader's shuffle generator advances one permutation
            # per epoch; without it, a resumed run would replay the batch
            # order from epoch 0 instead of continuing the sequence.
            "loader_rng": (self.train_loader.generator.get_state()
                           if getattr(self.train_loader, "generator", None)
                           is not None else None),
            "cuda_rng": (torch.cuda.get_rng_state_all()
                         if torch.cuda.is_available() else None),
            "py_rng": random.getstate(),
            "np_rng": np.random.get_state(),
            "match_rng": self.match_rng.bit_generator.state,
            "recency": self.recency.state_dict(),
            "teachers": self.teachers,
            "current_matchings": self.current_matchings,
            "static_matched": self.static_matched,
            "comm_floats_cum": self.comm_floats_cum,
            "comm_matcher_ints_cum": self.comm_matcher_ints_cum,
        }
        tmp = self.ckpt_path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, self.ckpt_path)

    def _load_checkpoint(self) -> None:
        state = torch.load(self.ckpt_path, map_location=self.device,
                           weights_only=False)
        for m, sd in zip(self.models, state["models"]):
            m.load_state_dict(sd)
        for o, sd in zip(self.optimizers, state["optimizers"]):
            o.load_state_dict(sd)
        for s, sd in zip(self.schedulers, state["schedulers"]):
            s.load_state_dict(sd)
        torch.set_rng_state(state["torch_rng"].cpu())
        if state.get("loader_rng") is not None and \
                getattr(self.train_loader, "generator", None) is not None:
            self.train_loader.generator.set_state(state["loader_rng"].cpu())
        if state["cuda_rng"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([t.cpu() for t in state["cuda_rng"]])
        random.setstate(state["py_rng"])
        np.random.set_state(state["np_rng"])
        self.match_rng.bit_generator.state = state["match_rng"]
        self.recency = mt.RecencyState.from_state_dict(state["recency"])
        self.teachers = state["teachers"]
        self.current_matchings = state["current_matchings"]
        self.static_matched = state["static_matched"]
        self.comm_floats_cum = state["comm_floats_cum"]
        self.comm_matcher_ints_cum = state["comm_matcher_ints_cum"]
        self.start_epoch = state["epoch"] + 1
        self.metrics_csv.truncate_to_epoch(state["epoch"])
        self.matches_csv.truncate_to_epoch(state["epoch"])
        if self.cfg.verbose:
            print(f"[*] Resumed from {self.ckpt_path} at epoch "
                  f"{self.start_epoch}")

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def train(self) -> Dict:
        cfg = self.cfg
        if cfg.trap_usr1 and hasattr(signal, "SIGUSR1"):
            signal.signal(signal.SIGUSR1,
                          lambda signum, frame: setattr(
                              self, "_preempt_requested", True))
        if cfg.resume and os.path.exists(self.ckpt_path):
            self._load_checkpoint()
        if self.start_epoch >= cfg.epochs:
            # Checkpoint says the run already finished. If the metrics CSV
            # is nonetheless incomplete (e.g. rows lost to the [D-006]
            # orphaned-writer incident), resume CANNOT restore them — the
            # loop below has nothing to train. Deleting the checkpoint
            # forces a clean retrain.
            print(f"WARNING | {cfg.run_id} | checkpoint is at epoch "
                  f"{self.start_epoch - 1} (>= --epochs {cfg.epochs}): "
                  f"nothing to do. If the metrics CSV is incomplete, delete "
                  f"{self.ckpt_path} and rerun.", flush=True)
        if self.start_epoch == 0:
            # Fresh start (including --requeue'd jobs preempted before their
            # first checkpoint): clear any stale rows so the CSVs never carry
            # duplicate epochs.
            self.metrics_csv.truncate_to_epoch(-1)
            self.matches_csv.truncate_to_epoch(-1)

        final = {}
        for epoch in range(self.start_epoch, cfg.epochs):
            tic = time.time()
            if self._refresh_due(epoch):
                self._refresh_matching(epoch)

            # lr from the first HEALTHY slot; the zombie's optimizer never
            # steps, so its scheduler is skipped (silences the step-order
            # warning and keeps its meaningless lr out of the log).
            lr_slot = (1 if cfg.zombie_slot == 0 else 0)
            lr_this_epoch = self.optimizers[lr_slot].param_groups[0]["lr"]
            train_stats = self._train_one_epoch(epoch)
            for i, s in enumerate(self.schedulers):
                if i != cfg.zombie_slot:
                    s.step()

            val = evaluate_cohort(self.models, self.valid_loader, self.device)
            test = evaluate_cohort(self.models, self.test_loader, self.device)
            self.cached_val = val
            comm = self._comm_epoch(train_stats["n_seen_comm"])

            row = dict(cfg.static_row)
            row.update({
                "epoch": epoch,
                "lr": lr_this_epoch,
                # The degree actually in effect this epoch (matchings may
                # lag the k_anneal schedule when rematch_every_epochs > 1).
                "k_current": self._degree(),
                # [D-018]: realized fraction of updates that distilled, so the
                # byte-matching can be checked from the CSV rather than assumed.
                "comm_duty": round(train_stats["comm_duty"], 6),
                "epoch_seconds": round(time.time() - tic, 2),
                "avg_test_acc": float(np.mean(test["accs"])),
                "std_test_acc": float(np.std(test["accs"])),
                "min_test_acc": float(np.min(test["accs"])),
                "max_test_acc": float(np.max(test["accs"])),
                "ensemble_test_acc": test["ensemble_acc"],
                "ens_minus_avg_test": test["ensemble_acc"]
                                      - float(np.mean(test["accs"])),
                "avg_val_acc": float(np.mean(val["accs"])),
                "ensemble_val_acc": val["ensemble_acc"],
                "disagreement_val": mean_pairwise_disagreement(val["preds"]),
                "rho_val": mean_pairwise_error_correlation(val["preds"],
                                                           val["y"]),
                "disagreement_test": mean_pairwise_disagreement(test["preds"]),
                "rho_test": mean_pairwise_error_correlation(test["preds"],
                                                            test["y"]),
            })
            # Per-architecture means (informative for heterogeneous cohorts).
            for arch in sorted({s.arch for s in self.slots}):
                idx = [s.index for s in self.slots if s.arch == arch]
                row[f"avg_test_acc_{arch}"] = float(np.mean(test["accs"][idx]))
            if cfg.zombie_slot >= 0:
                # Healthy-only counterparts, comparable to the no-zombie
                # anchors. The cohort avg/disagreement/rho columns above
                # include the dead member (constant predictor) and are NOT
                # reconstructible post-hoc; the ensemble is argmax-invariant
                # to a uniform member, so it needs no counterpart.
                h = [i for i in range(self.K) if i != cfg.zombie_slot]
                row["avg_test_acc_healthy"] = float(np.mean(test["accs"][h]))
                row["avg_val_acc_healthy"] = float(np.mean(val["accs"][h]))
                row["disagreement_test_healthy"] = \
                    mean_pairwise_disagreement(test["preds"][h])
                row["rho_test_healthy"] = \
                    mean_pairwise_error_correlation(test["preds"][h],
                                                    test["y"])
                row["disagreement_val_healthy"] = \
                    mean_pairwise_disagreement(val["preds"][h])
                row["rho_val_healthy"] = \
                    mean_pairwise_error_correlation(val["preds"][h], val["y"])
            for s in self.slots:
                i = s.index
                row[f"model_{i:02d}_arch"] = s.arch
                row[f"model_{i:02d}_test_acc"] = float(test["accs"][i])
                row[f"model_{i:02d}_val_acc"] = float(val["accs"][i])
                row[f"model_{i:02d}_train_loss"] = float(train_stats["loss"][i])
                row[f"model_{i:02d}_ce_loss"] = float(train_stats["ce"][i])
                row[f"model_{i:02d}_kd_loss"] = float(train_stats["kd"][i])
            row.update(comm)
            self.metrics_csv.write(row)
            final = row

            if cfg.verbose:
                print(f"[{cfg.run_id}] epoch {epoch + 1}/{cfg.epochs} "
                      f"avg_test={row['avg_test_acc']:.4f} "
                      f"ens_test={row['ensemble_test_acc']:.4f} "
                      f"({row['epoch_seconds']}s)", flush=True)

            if (epoch + 1) % cfg.checkpoint_every == 0 or \
                    epoch == cfg.epochs - 1:
                self._save_checkpoint(epoch)

            if self._preempt_requested:
                # Wall clock approaching (SIGUSR1): make the state on disk
                # complete for this epoch, then hand control back so the
                # SLURM script can requeue the array task; the requeued run
                # resumes from exactly here via --resume.
                self._save_checkpoint(epoch)
                self.preempted = True
                print(f"PREEMPT | {cfg.run_id} | checkpointed at epoch "
                      f"{epoch}; stopping for requeue", flush=True)
                break

        self.metrics_csv.close()
        self.matches_csv.close()
        return final
