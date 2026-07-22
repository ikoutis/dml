"""Trainer tests (CPU, tiny synthetic data, tiny MLPs).

The load-bearing test is the PARITY test: the trainer's one-epoch result must
be bit-close to a manual reference implementing the declared update rule —
simultaneous forwards, detached teacher logits, per-model SGD steps. A
sequential-update or non-detached implementation fails it.
"""

import copy
import csv
import os

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.cohort import Slot
from src.mutual_trainer import MutualTrainer, TrainerConfig, parse_k_anneal

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


def base_cfg(tmp_path, **kw):
    defaults = dict(run_id="test_run", arm="dml", epochs=1, lr=0.05,
                    momentum=0.9, weight_decay=5e-4, nesterov=True,
                    lr_step=60, lr_gamma=0.1, kd_T=1.0, seed=1, device="cpu",
                    output_dir=str(tmp_path), checkpoint_every=100,
                    verbose=False)
    defaults.update(kw)
    return TrainerConfig(**defaults)


def make_trainer(tmp_path, K=3, **kw):
    slots = make_slots(K, seed=3)
    train_loader = make_loader(32, seed=10)
    valid_loader = make_loader(16, seed=11)
    test_loader = make_loader(16, seed=12)
    cfg = base_cfg(tmp_path, **kw)
    return MutualTrainer(cfg, slots, train_loader, valid_loader, test_loader,
                         num_classes=N_CLS, n_valid=16), slots


def manual_epoch(models, loader, teachers, lr=0.05, dmle=False):
    """Reference: simultaneous updates with detached teachers."""
    loss_ce = nn.CrossEntropyLoss()
    loss_kl = nn.KLDivLoss(reduction="batchmean")
    opts = [torch.optim.SGD(m.parameters(), lr=lr, momentum=0.9,
                            weight_decay=5e-4, nesterov=True) for m in models]
    K = len(models)
    for m in models:
        m.train()
    for x, y in loader:
        outs = [m(x) for m in models]
        det = [o.detach() for o in outs]
        for i in range(K):
            loss = loss_ce(outs[i], y)
            if dmle:
                p_ens = torch.stack([F.softmax(det[j], dim=1)
                                     for j in range(K) if j != i]).mean(0)
                loss = loss + loss_kl(F.log_softmax(outs[i], dim=1), p_ens)
            else:
                for j, alpha in teachers[i]:
                    loss = loss + alpha * loss_kl(
                        F.log_softmax(outs[i], dim=1),
                        F.softmax(det[j], dim=1))
            opts[i].zero_grad()
            loss.backward()
            opts[i].step()


def assert_params_close(models_a, models_b):
    for ma, mb in zip(models_a, models_b):
        for pa, pb in zip(ma.parameters(), mb.parameters()):
            assert torch.allclose(pa, pb, atol=1e-6), \
                "trainer diverged from the declared update rule"


class TestParity:
    def test_dense_dml_matches_manual(self, tmp_path):
        trainer, slots = make_trainer(tmp_path, K=3, arm="dml")
        ref = [copy.deepcopy(s.model) for s in slots]
        K = 3
        teachers = [[(j, 1.0 / (K - 1)) for j in range(K) if j != i]
                    for i in range(K)]
        trainer._train_one_epoch(0)
        manual_epoch(ref, make_loader(32, seed=10), teachers)
        assert_params_close([s.model for s in slots], ref)

    def test_matched_k1_matches_manual(self, tmp_path):
        trainer, slots = make_trainer(tmp_path, K=4, arm="matched",
                                      match_weight="random")
        teachers = [[(1, 1.0)], [(0, 1.0)], [(3, 1.0)], [(2, 1.0)]]
        trainer.teachers = teachers
        ref = [copy.deepcopy(s.model) for s in slots]
        trainer._train_one_epoch(0)
        manual_epoch(ref, make_loader(32, seed=10), teachers)
        assert_params_close([s.model for s in slots], ref)

    def test_dmle_matches_manual(self, tmp_path):
        trainer, slots = make_trainer(tmp_path, K=3, arm="dml",
                                      target="ensemble")
        ref = [copy.deepcopy(s.model) for s in slots]
        trainer._train_one_epoch(0)
        manual_epoch(ref, make_loader(32, seed=10), None, dmle=True)
        assert_params_close([s.model for s in slots], ref)

    def test_indep_kd_is_zero(self, tmp_path):
        trainer, _ = make_trainer(tmp_path, K=3, arm="indep")
        stats = trainer._train_one_epoch(0)
        assert np.allclose(stats["kd"], 0.0)
        assert np.allclose(stats["loss"], stats["ce"])


class TestDetachment:
    def test_teacher_receives_no_gradient(self):
        torch.manual_seed(0)
        m0 = nn.Linear(IN_DIM, N_CLS)
        m1 = nn.Linear(IN_DIM, N_CLS)
        x = torch.randn(8, IN_DIM)
        y = torch.randint(0, N_CLS, (8,))
        out0, out1 = m0(x), m1(x)
        loss = nn.CrossEntropyLoss()(out0, y) + \
            nn.KLDivLoss(reduction="batchmean")(
                F.log_softmax(out0, dim=1), F.softmax(out1.detach(), dim=1))
        loss.backward()
        assert all(p.grad is None for p in m1.parameters())
        assert all(p.grad is not None for p in m0.parameters())


class TestCommAccounting:
    def test_dense(self, tmp_path):
        trainer, _ = make_trainer(tmp_path, K=4, arm="dml", epochs=1)
        trainer.train()
        # degree K-1 = 3; 32 examples; 4 classes
        assert trainer.comm_floats_cum == pytest.approx(3 * 32 * N_CLS)
        assert trainer.comm_matcher_ints_cum == 0

    def test_indep(self, tmp_path):
        trainer, _ = make_trainer(tmp_path, K=4, arm="indep", epochs=2)
        trainer.train()
        assert trainer.comm_floats_cum == 0
        assert trainer.comm_matcher_ints_cum == 0

    def test_matched_measured_weight(self, tmp_path):
        trainer, _ = make_trainer(tmp_path, K=4, arm="matched",
                                  match_weight="disagreement", epochs=2,
                                  rematch_every_epochs=1)
        trainer.train()
        # degree 1, two epochs of 32 examples
        assert trainer.comm_floats_cum == pytest.approx(2 * 1 * 32 * N_CLS)
        # matcher ships n_valid=16 hard preds per model per refresh (2x)
        assert trainer.comm_matcher_ints_cum == pytest.approx(2 * 16)

    def test_matched_random_weight_no_matcher_cost(self, tmp_path):
        trainer, _ = make_trainer(tmp_path, K=4, arm="matched",
                                  match_weight="random", epochs=1)
        trainer.train()
        assert trainer.comm_matcher_ints_cum == 0

    def test_static_matches_once(self, tmp_path):
        trainer, _ = make_trainer(tmp_path, K=4, arm="matched",
                                  match_weight="disagreement", epochs=3,
                                  rematch_every_epochs=0)
        trainer.train()
        assert trainer.comm_matcher_ints_cum == pytest.approx(16)  # once
        with open(os.path.join(str(tmp_path), "test_run_matches.csv")) as fh:
            rows = list(csv.DictReader(fh))
        assert {r["epoch"] for r in rows} == {"0"}  # one refresh only

    def test_peel2_degree(self, tmp_path):
        trainer, _ = make_trainer(tmp_path, K=4, arm="matched",
                                  match_weight="disagreement",
                                  k_matchings=2, epochs=1)
        trainer.train()
        assert trainer.comm_floats_cum == pytest.approx(2 * 32 * N_CLS)


class TestEndToEnd:
    def test_matched_run_writes_csvs(self, tmp_path):
        trainer, _ = make_trainer(tmp_path, K=4, arm="matched",
                                  match_weight="disagreement", epochs=2,
                                  static_row={"run_id": "test_run"})
        final = trainer.train()
        with open(os.path.join(str(tmp_path), "test_run_metrics.csv")) as fh:
            rows = list(csv.DictReader(fh))
        assert [r["epoch"] for r in rows] == ["0", "1"]
        for col in ("avg_test_acc", "ensemble_test_acc", "rho_val",
                    "disagreement_val", "comm_bytes_cum", "k_current"):
            assert col in rows[0]
        assert final["k_current"] == 1
        with open(os.path.join(str(tmp_path), "test_run_matches.csv")) as fh:
            match_rows = list(csv.DictReader(fh))
        # 2 refreshes x (K/2 = 2 pairs) x k=1 layers
        assert len(match_rows) == 4

    def test_resume_continues(self, tmp_path):
        kw = dict(K=4, arm="matched", match_weight="disagreement",
                  checkpoint_every=1)
        trainer, _ = make_trainer(tmp_path, epochs=2, **kw)
        trainer.train()
        trainer2, _ = make_trainer(tmp_path, epochs=4, resume=True, **kw)
        trainer2.train()
        assert trainer2.start_epoch == 2
        with open(os.path.join(str(tmp_path), "test_run_metrics.csv")) as fh:
            rows = list(csv.DictReader(fh))
        assert [r["epoch"] for r in rows] == ["0", "1", "2", "3"]

    def test_fresh_start_clears_stale_csv(self, tmp_path):
        # A --requeue'd job that never checkpointed restarts from epoch 0;
        # the surviving CSV rows must be cleared, not appended to.
        kw = dict(K=4, arm="matched", match_weight="disagreement")
        trainer, _ = make_trainer(tmp_path, epochs=2, **kw)
        trainer.train()
        trainer2, _ = make_trainer(tmp_path, epochs=2, **kw)  # no resume
        trainer2.train()
        with open(os.path.join(str(tmp_path), "test_run_metrics.csv")) as fh:
            rows = list(csv.DictReader(fh))
        assert [r["epoch"] for r in rows] == ["0", "1"]

    def test_k_anneal_out_of_range_raises(self, tmp_path):
        with pytest.raises(ValueError):
            make_trainer(tmp_path, K=4, arm="matched",
                         match_weight="disagreement", k_anneal="0:5")

    def test_k_anneal_schedule(self, tmp_path):
        trainer, _ = make_trainer(tmp_path, K=4, arm="matched",
                                  match_weight="disagreement",
                                  k_anneal="0:2,2:1", epochs=3)
        trainer.train()
        with open(os.path.join(str(tmp_path), "test_run_metrics.csv")) as fh:
            rows = list(csv.DictReader(fh))
        assert [r["k_current"] for r in rows] == ["2", "2", "1"]

    def test_odd_K_matched_raises(self, tmp_path):
        with pytest.raises(ValueError):
            make_trainer(tmp_path, K=3, arm="matched")


class TestPreemption:
    """The 72 h wall-clock path: SIGUSR1 -> finish the epoch in flight,
    checkpoint, stop with .preempted = True; a --resume run completes."""

    def test_usr1_checkpoints_stops_and_resumes(self, tmp_path):
        import signal as sig

        kw = dict(K=4, arm="matched", match_weight="disagreement",
                  checkpoint_every=100)
        trainer, _ = make_trainer(tmp_path, epochs=5, trap_usr1=True, **kw)
        orig = trainer._train_one_epoch

        def wrapped(epoch):
            out = orig(epoch)
            if epoch == 1:  # the signal lands mid-run
                sig.raise_signal(sig.SIGUSR1)
            return out

        trainer._train_one_epoch = wrapped
        trainer.train()
        assert trainer.preempted
        assert os.path.exists(os.path.join(str(tmp_path),
                                           "test_run_ckpt.pt"))
        with open(os.path.join(str(tmp_path), "test_run_metrics.csv")) as fh:
            rows = list(csv.DictReader(fh))
        assert [r["epoch"] for r in rows] == ["0", "1"]

        trainer2, _ = make_trainer(tmp_path, epochs=5, resume=True, **kw)
        trainer2.train()
        assert trainer2.start_epoch == 2
        assert not trainer2.preempted
        with open(os.path.join(str(tmp_path), "test_run_metrics.csv")) as fh:
            rows = list(csv.DictReader(fh))
        assert [r["epoch"] for r in rows] == ["0", "1", "2", "3", "4"]


class TestTopology:
    """Static-topology arm: each model distills from ALL its fixed graph
    neighbours, set once, never refreshed."""

    def make(self, tmp_path, K, graph, **kw):
        slots = make_slots(K, seed=3)
        tl = make_loader(32, seed=10)
        vl = make_loader(16, seed=11)
        cfg = base_cfg(tmp_path, arm="topology", graph=graph, **kw)
        return MutualTrainer(cfg, slots, tl, vl, vl, num_classes=N_CLS,
                             n_valid=16)

    def test_teachers_are_graph_neighbours(self, tmp_path):
        t = self.make(tmp_path, 6, "ring")   # 0-1-2-3-4-5-0
        # every node has its two cycle neighbours, alpha = 1/2, symmetric
        for i in range(6):
            partners = {j for j, _ in t.teachers[i]}
            assert partners == {(i - 1) % 6, (i + 1) % 6}
            assert all(abs(a - 0.5) < 1e-9 for _, a in t.teachers[i])
        assert t._degree() == 2

    def test_prism_degree_and_static(self, tmp_path):
        t = self.make(tmp_path, 12, "prism")
        assert t._degree() == 3
        assert not t._refresh_due(0)          # topology never refreshes
        before = [list(x) for x in t.teachers]
        t.train()
        assert [list(x) for x in t.teachers] == before  # unchanged

    def test_comm_scales_with_degree_no_matcher_cost(self, tmp_path):
        t = self.make(tmp_path, 12, "rregular:3", epochs=1)
        t.train()
        assert t.comm_floats_cum == pytest.approx(3 * 32 * N_CLS)
        assert t.comm_matcher_ints_cum == 0   # fixed graph => no probes

    def test_requires_graph(self, tmp_path):
        with pytest.raises(ValueError):
            self.make(tmp_path, 6, "complete")   # topology needs a real graph

    def test_disconnected_clusters(self, tmp_path):
        # [D-012] connectivity probe: clusters:2 on K=6 -> three frozen
        # pairs; each model's sole teacher is its pair-mate, alpha = 1.
        t = self.make(tmp_path, 6, "clusters:2", epochs=1)
        for i in range(6):
            mate = i + 1 if i % 2 == 0 else i - 1
            assert t.teachers[i] == [(mate, 1.0)]
        assert t._degree() == 1
        t.train()   # trains end-to-end despite gap 0 (that's the point)


class TestZombie:
    """[D-015] controlled epidemiology: one implanted dead model that emits
    exactly-uniform logits and never trains."""

    def make(self, tmp_path, K=6, **kw):
        slots = make_slots(K, seed=3)
        tl = make_loader(32, seed=10)
        vl = make_loader(16, seed=11)
        cfg = base_cfg(tmp_path, zombie_slot=0, **kw)
        return MutualTrainer(cfg, slots, tl, vl, vl, num_classes=N_CLS,
                             n_valid=16)

    def test_zombie_emits_uniform_and_never_trains(self, tmp_path):
        t = self.make(tmp_path, arm="topology", graph="ring")
        x = torch.randn(5, IN_DIM)
        out = t.models[0](x)
        assert out.shape == (5, N_CLS)
        assert torch.all(out == 0)                      # exact uniform logits
        before = [p.clone() for p in t.models[0].parameters()]
        final = t.train()
        after = list(t.models[0].parameters())
        assert all(torch.equal(a, b) for a, b in zip(before, after))
        assert final["model_00_train_loss"] == 0.0      # never stepped
        assert t.slots[0].arch == "zombie"

    def test_healthy_slots_keep_anchor_inits(self, tmp_path):
        # Replacing slot 0 must not perturb the other slots' inits — the
        # per-model pairing with no-zombie anchor runs depends on it.
        ref = make_slots(6, seed=3)
        t = self.make(tmp_path, arm="topology", graph="ring")
        for i in range(1, 6):
            for p_ref, p_z in zip(ref[i].model.parameters(),
                                  t.models[i].parameters()):
                assert torch.equal(p_ref, p_z)

    def test_neighbours_receive_uniform_teacher(self, tmp_path):
        # Ring 0-1-2-3-4-5-0: models 1 and 5 each have the zombie as one of
        # two teachers; their kd_loss must include a KL-to-uniform term.
        t = self.make(tmp_path, arm="topology", graph="ring")
        assert (0, 0.5) in [tuple(p) for p in t.teachers[1]]
        assert (0, 0.5) in [tuple(p) for p in t.teachers[5]]
        final = t.train()
        assert final["model_01_kd_loss"] > 0
        assert "avg_test_acc_healthy" in final
        healthy = [final[f"model_{i:02d}_test_acc"] for i in range(1, 6)]
        assert final["avg_test_acc_healthy"] == pytest.approx(
            float(np.mean(healthy)))

    def test_ensemble_argmax_invariant_to_zombie(self, tmp_path):
        # A uniform member shifts every class's mean probability equally, so
        # the ensemble prediction with the zombie must equal the healthy-only
        # ensemble prediction.
        from src.metrics import evaluate_cohort
        t = self.make(tmp_path, arm="topology", graph="ring")
        ev_all = evaluate_cohort(t.models, t.valid_loader, t.device)
        ev_healthy = evaluate_cohort(t.models[1:], t.valid_loader, t.device)
        assert ev_all["ensemble_acc"] == pytest.approx(
            ev_healthy["ensemble_acc"])

    def test_matched_arm_with_zombie(self, tmp_path):
        # The zombie participates in random matching like anyone else; the
        # run completes and someone is paired with it each refresh.
        t = self.make(tmp_path, arm="matched", match_weight="random")
        t.train()
        partners = {j for M in t.current_matchings for (a, j) in M
                    if a == 0} | {a for M in t.current_matchings
                                  for (a, j) in M if j == 0}
        assert len(partners) >= 1

    def test_indep_zombie_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            self.make(tmp_path, arm="indep")

    def test_zombie_resume_roundtrip(self, tmp_path):
        t = self.make(tmp_path, arm="topology", graph="ring", epochs=2,
                      checkpoint_every=1)
        t.train()
        t2 = self.make(tmp_path, arm="topology", graph="ring", epochs=2,
                       checkpoint_every=1, resume=True)
        t2.train()
        assert t2.start_epoch == 2   # resumed past the end, no retrain


def test_parse_k_anneal():
    assert parse_k_anneal("") == []
    assert parse_k_anneal("120:1,0:3,60:2") == [(0, 3), (60, 2), (120, 1)]
