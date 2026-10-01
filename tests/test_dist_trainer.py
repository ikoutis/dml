"""Distributed trainer tests: real multi-process runs on the gloo backend.

The parity tests check that one process per model, exchanging posteriors
over torch.distributed, produces the same parameters as the single-process
suite trainer running the same update rule.
"""

import csv
import os
import sys

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from src.dist_trainer import DistConfig, DistMutualTrainer, draw_matchings
from src.mutual_trainer import MutualTrainer
from tests.test_trainer import (N_CLS, assert_params_close, base_cfg,
                                make_loader, make_slots)

K = 4
LR = 0.05

# Multi-process gloo hangs at init on Windows; these run on Linux (Wulver).
needs_linux = pytest.mark.skipif(sys.platform == "win32",
                                 reason="multi-process gloo needs Linux")


def _worker(rank, world, init_file, out_dir, cfg_kw, loader_seeds):
    dist.init_process_group(
        "gloo", init_method="file:///" + init_file.replace("\\", "/"),
        rank=rank, world_size=world)
    try:
        model = make_slots(world, seed=3)[rank].model
        loader = make_loader(32, seed=loader_seeds[rank])
        cfg = DistConfig(lr=LR, epochs=1, seed=1, run_id="t",
                         output_dir=out_dir, **cfg_kw)
        trainer = DistMutualTrainer(cfg, model, loader, torch.device("cpu"),
                                    N_CLS)
        trainer.train()
        torch.save(model.state_dict(), os.path.join(out_dir, f"r{rank}.pt"))
    finally:
        dist.destroy_process_group()


def run_dist(tmp_path, loader_seeds=None, **cfg_kw):
    init_file = str(tmp_path / "init")
    out_dir = str(tmp_path / "out")
    os.makedirs(out_dir, exist_ok=True)
    mp.spawn(_worker, args=(K, init_file, out_dir, cfg_kw,
                            loader_seeds or [10] * K),
             nprocs=K, join=True)
    models = make_slots(K, seed=3)
    for r, s in enumerate(models):
        s.model.load_state_dict(torch.load(os.path.join(out_dir, f"r{r}.pt")))
    return [s.model for s in models], out_dir


def single_process(tmp_path, teachers=None, **kw):
    slots = make_slots(K, seed=3)
    trainer = MutualTrainer(base_cfg(tmp_path, **kw), slots,
                            make_loader(32, seed=10), make_loader(16, seed=11),
                            make_loader(16, seed=12), num_classes=N_CLS,
                            n_valid=16)
    if teachers is not None:
        trainer.teachers = teachers
    trainer._train_one_epoch(0)
    return [s.model for s in slots]


def teachers_from(partners_list):
    d = len(partners_list)
    return [[(p[i], 1.0 / d) for p in partners_list] for i in range(K)]


class TestDrawMatchings:
    @pytest.mark.parametrize("Kn,d", [(4, 1), (4, 3), (12, 1), (12, 5),
                                      (12, 11)])
    def test_perfect_and_disjoint(self, Kn, d):
        rng = np.random.default_rng(0)
        for _ in range(20):
            ms = draw_matchings(rng, Kn, d)
            assert len(ms) == d
            edges = set()
            for partner in ms:
                assert sorted(partner) == list(range(Kn))
                for i, j in enumerate(partner):
                    assert j != i and partner[j] == i
                    edges.add(frozenset((i, j)))
            assert len(edges) == d * Kn // 2

    def test_odd_K_raises(self):
        with pytest.raises(ValueError):
            draw_matchings(np.random.default_rng(0), 5, 1)


@needs_linux
class TestParity:
    def test_dense_matches_single_process(self, tmp_path):
        got, _ = run_dist(tmp_path / "d", mode="dense")
        ref = single_process(tmp_path / "s", arm="dml", target="ensemble")
        assert_params_close(got, ref)

    @pytest.mark.parametrize("d", [1, 2])
    def test_p2p_matches_single_process(self, tmp_path, d):
        got, _ = run_dist(tmp_path / "d", mode="p2p", degree=d)
        partners = draw_matchings(np.random.default_rng(1 * 7919 + 3), K, d)
        ref = single_process(tmp_path / "s", arm="matched",
                             match_weight="random",
                             teachers=teachers_from(partners))
        assert_params_close(got, ref)

    def test_indep_matches_single_process(self, tmp_path):
        got, _ = run_dist(tmp_path / "d", mode="indep")
        ref = single_process(tmp_path / "s", arm="indep")
        assert_params_close(got, ref)


@needs_linux
class TestRuntime:
    def test_straggler_and_csv(self, tmp_path):
        _, out = run_dist(tmp_path, mode="dense", straggler_rank=1,
                          straggler_ms=20)
        with open(os.path.join(out, "t_dist.csv")) as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == 1
        r = rows[0]
        assert int(r["n_updates"]) == 4
        # Only rank 1 sleeps, so the cohort mean is about 20/K ms per update,
        # and the other ranks wait for it inside the all-reduce.
        assert 20 / K * 0.8 < float(r["straggle_ms"]) < 20 / K * 3
        others = [float(r[f"rank_{i:02d}_comm_ms"]) for i in (0, 2, 3)]
        assert min(others) > 10

    def test_mismatched_batches_raise(self, tmp_path):
        with pytest.raises(Exception, match="different batches"):
            run_dist(tmp_path, loader_seeds=[10, 10, 11, 10], mode="dense")
