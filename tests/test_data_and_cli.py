"""Tests for label noise, cohort parsing, arm labels, and run ids
(no dataset download needed)."""

import numpy as np
import pytest

from src.cohort import cohort_slug, parse_cohort_spec
from src.data import apply_symmetric_noise, num_classes
from src.run_experiment import auto_arm_label, build_parser, compose_run_id


class TestNoise:
    def test_rate_and_wrongness(self):
        n, n_cls = 5000, 10
        rng = np.random.RandomState(0)
        targets = list(rng.randint(0, n_cls, size=n))
        orig = list(targets)
        idx = np.arange(n)
        flipped = apply_symmetric_noise(targets, idx, 0.4, n_cls, noise_seed=42)
        assert 0.36 < flipped / n < 0.44
        for t, o in zip(targets, orig):
            if t != o:
                assert 0 <= t < n_cls
        # flipped labels are never the original class
        n_diff = sum(1 for t, o in zip(targets, orig) if t != o)
        assert n_diff == flipped

    def test_deterministic(self):
        t1 = list(range(10)) * 10
        t2 = list(t1)
        idx = np.arange(len(t1))
        apply_symmetric_noise(t1, idx, 0.5, 10, noise_seed=7)
        apply_symmetric_noise(t2, idx, 0.5, 10, noise_seed=7)
        assert t1 == t2

    def test_scope(self):
        targets = [0] * 100
        apply_symmetric_noise(targets, np.arange(50), 1.0, 10, noise_seed=1)
        assert all(t != 0 for t in targets[:50])
        assert all(t == 0 for t in targets[50:])

    def test_zero_rate(self):
        targets = [1, 2, 3]
        assert apply_symmetric_noise(targets, np.arange(3), 0.0, 10, 1) == 0
        assert targets == [1, 2, 3]


class TestCohortSpec:
    def test_parse(self):
        assert parse_cohort_spec("resnet32:8") == ["resnet32"] * 8
        assert parse_cohort_spec("wrn28x10:2,resnet32:2") == \
            ["wrn28x10", "wrn28x10", "resnet32", "resnet32"]
        assert parse_cohort_spec("mobilenet") == ["mobilenet"]

    def test_slug(self):
        assert cohort_slug("resnet32:8") == "resnet32x8"
        assert cohort_slug("wrn28x10:4,resnet32:4") == \
            "wrn28x10x4-resnet32x4"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            parse_cohort_spec("")


class TestArmLabels:
    def parse(self, argv):
        return build_parser().parse_args(argv)

    def test_baselines(self):
        assert auto_arm_label(self.parse(["--arm", "indep"])) == "indep"
        assert auto_arm_label(self.parse(["--arm", "dml"])) == "dml"
        assert auto_arm_label(self.parse(
            ["--arm", "dml", "--target", "ensemble"])) == "dmle"

    def test_matched_variants(self):
        a = self.parse(["--arm", "matched"])
        assert auto_arm_label(a) == "mwmd1"
        a = self.parse(["--arm", "matched", "--match_weight", "random"])
        assert auto_arm_label(a) == "rand1"
        a = self.parse(["--arm", "matched", "--k_matchings", "2"])
        assert auto_arm_label(a) == "mwmd2"
        a = self.parse(["--arm", "matched", "--rematch_every_epochs", "0"])
        assert auto_arm_label(a) == "mwmd1-static"
        a = self.parse(["--arm", "matched", "--recency_lambda", "0.5"])
        assert auto_arm_label(a) == "mwmd1-rec0.5"
        a = self.parse(["--arm", "matched", "--match_weight", "teachable"])
        assert auto_arm_label(a) == "mwmt1"
        a = self.parse(["--arm", "matched", "--graph", "rregular:3"])
        assert auto_arm_label(a) == "mwmd1-rregular3"
        a = self.parse(["--arm", "matched", "--k_anneal", "0:3,120:1"])
        assert auto_arm_label(a) == "mwmd1-anneal31"
        a = self.parse(["--arm", "matched", "--k_anneal", "0:3,60:2,120:1"])
        assert auto_arm_label(a) == "mwmd1-anneal321"
        a = self.parse(["--arm", "matched", "--match_weight", "teachable",
                        "--kappa", "0"])
        assert auto_arm_label(a) == "mwmt1-k0"
        a = self.parse(["--arm", "matched", "--kd_T", "4"])
        assert auto_arm_label(a) == "mwmd1-T4"

    def test_run_id(self):
        a = self.parse(["--arm", "matched", "--cohort", "resnet32:8",
                        "--seed", "3", "--run_tag", "d001"])
        rid = compose_run_id(a, auto_arm_label(a))
        assert rid == "resnet32x8_cifar100_K8_mwmd1_seed0003_d001"

    def test_run_id_noise(self):
        a = self.parse(["--arm", "dml", "--cohort", "resnet32:8",
                        "--label_noise_rate", "0.4"])
        rid = compose_run_id(a, auto_arm_label(a))
        assert rid == "resnet32x8_cifar100_K8_dml_noise40_seed0001"


def test_num_classes():
    assert num_classes("cifar100") == 100
    assert num_classes("cifar10") == 10
