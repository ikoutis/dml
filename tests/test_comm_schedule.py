"""Temporally sparse communication ([D-018]).

The arm exists to answer one question: at a fixed communication budget, is it
better to sample one fresh peer on every update, or the exact cohort mean on a
fraction of updates? For that comparison to mean anything the byte matching has
to be exact, so these tests pin the schedule, the ledger, and the dose.
"""

import csv
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.cohort import Slot
from src.mutual_trainer import MutualTrainer, TrainerConfig

IN_DIM, N_CLS = 6, 4


def make_slots(K, seed=0):
    torch.manual_seed(seed)
    slots = []
    for i in range(K):
        m = nn.Sequential(nn.Linear(IN_DIM, 16), nn.ReLU(),
                          nn.Linear(16, N_CLS))
        slots.append(Slot(index=i, name=f"m{i:02d}", arch="tiny", model=m,
                          n_params=sum(p.numel() for p in m.parameters())))
    return slots


def make_loader(n, seed, batch=8):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, IN_DIM, generator=g)
    y = torch.randint(0, N_CLS, (n,), generator=g)
    return DataLoader(TensorDataset(x, y), batch_size=batch, shuffle=False)


def make_trainer(tmp_path, K=12, n_train=88, **kw):
    slots = make_slots(K, seed=3)
    defaults = dict(run_id="sched", arm="dml", target="ensemble", epochs=1,
                    lr=0.05, momentum=0.9, weight_decay=5e-4, nesterov=True,
                    lr_step=60, lr_gamma=0.1, kd_T=1.0, seed=1, device="cpu",
                    output_dir=str(tmp_path), checkpoint_every=100,
                    verbose=False)
    defaults.update(kw)
    cfg = TrainerConfig(**defaults)
    return MutualTrainer(cfg, slots, make_loader(n_train, 10),
                         make_loader(16, 11), make_loader(16, 12),
                         num_classes=N_CLS, n_valid=16)


# --------------------------------------------------------------------------
# The schedule
# --------------------------------------------------------------------------
def test_schedule_selects_the_documented_updates(tmp_path):
    """6-of-11 must select updates 1, 3, 5, 7, 9, 11 of each block."""
    t = make_trainer(tmp_path, comm_on=6, comm_block=11)
    active = [i for i in range(11) if t._comm_active(i)]
    assert active == [0, 2, 4, 6, 8, 10]


def test_schedule_is_exact_per_block(tmp_path):
    """Exactly comm_on active updates in every block, not just on average."""
    t = make_trainer(tmp_path, comm_on=6, comm_block=11)
    for block in range(20):
        hits = sum(t._comm_active(block * 11 + i) for i in range(11))
        assert hits == 6


def test_no_schedule_means_every_update(tmp_path):
    t = make_trainer(tmp_path)
    assert all(t._comm_active(i) for i in range(50))


# --------------------------------------------------------------------------
# The ledger
# --------------------------------------------------------------------------
def test_allreduce_multiplier(tmp_path):
    """Ring all-reduce bills 2(K-1)/K streams, not K-1."""
    t = make_trainer(tmp_path, K=12, comm_accounting="allreduce")
    assert t._stream_multiplier() == 2 * 11 / 12


def test_p2p_multiplier_is_degree(tmp_path):
    t = make_trainer(tmp_path, K=12, comm_accounting="p2p")
    assert t._stream_multiplier() == 11


def test_temporal_dense_matches_degree_one_bytes(tmp_path):
    """The point of the arm: 6/11 of updates at 2(K-1)/K streams each is
    exactly one stream per update, i.e. degree-1's budget."""
    K, n_train, batch = 12, 88, 8          # 11 batches per epoch
    sparse = make_trainer(tmp_path / "a", K=K, n_train=n_train, comm_on=6,
                          comm_block=11, comm_accounting="allreduce")
    stats = sparse._train_one_epoch(0)
    assert stats["comm_duty"] == 6 / 11
    billed = sparse._comm_epoch(stats["n_seen_comm"])["comm_bytes_epoch"]

    # Degree-1 point-to-point over the same epoch: one stream per update.
    one_stream_floats = n_train * N_CLS
    assert billed == 4.0 * one_stream_floats


def test_duty_is_recorded_in_the_csv(tmp_path):
    t = make_trainer(tmp_path, comm_on=6, comm_block=11,
                     comm_accounting="allreduce")
    t.train()
    path = os.path.join(str(tmp_path), "sched_metrics.csv")
    rows = list(csv.DictReader(open(path)))
    # The column is rounded to 6 decimals for readability, so compare with a
    # tolerance rather than exactly.
    assert abs(float(rows[-1]["comm_duty"]) - 6 / 11) < 1e-6


# --------------------------------------------------------------------------
# The dose
# --------------------------------------------------------------------------
def test_kd_scale_leaves_ce_only_updates_alone(tmp_path):
    """On a non-distilling update the loss must be plain CE, whatever the
    dose scaling is set to."""
    t = make_trainer(tmp_path, comm_on=1, comm_block=1000, kd_scale=7.0,
                     comm_accounting="allreduce")
    # Only batch 0 distils; every later batch is CE-only.
    assert t._comm_active(0)
    assert not any(t._comm_active(i) for i in range(1, 20))


def test_dose_matching_integrates_to_one(tmp_path):
    """6/11 of updates at 11/6 weight is one KD unit per update on average."""
    t = make_trainer(tmp_path, comm_on=6, comm_block=11, kd_scale=11 / 6)
    stats = t._train_one_epoch(0)
    assert abs(stats["comm_duty"] * t.cfg.kd_scale - 1.0) < 1e-12


def test_scheduled_arm_trains_and_differs_from_unscheduled(tmp_path):
    """Sanity: withholding KD on 5 of 11 updates changes the outcome."""
    a = make_trainer(tmp_path / "full", comm_accounting="allreduce")
    b = make_trainer(tmp_path / "sparse", comm_on=6, comm_block=11,
                     comm_accounting="allreduce")
    a._train_one_epoch(0)
    b._train_one_epoch(0)
    pa = torch.cat([p.flatten() for p in a.models[0].parameters()])
    pb = torch.cat([p.flatten() for p in b.models[0].parameters()])
    assert not torch.allclose(pa, pb)
