"""Unit tests for the matching engine (CPU only, no torch needed)."""

import itertools

import numpy as np
import pytest

from src import matching as mt


def brute_force_mwpm(W, mask=None):
    """Reference maximum-weight perfect matching by full enumeration."""
    K = W.shape[0]
    best_val, best = -np.inf, None

    def all_matchings(nodes):
        if not nodes:
            yield []
            return
        i = nodes[0]
        for idx in range(1, len(nodes)):
            j = nodes[idx]
            rest = nodes[1:idx] + nodes[idx + 1:]
            for m in all_matchings(rest):
                yield [(i, j)] + m

    for m in all_matchings(list(range(K))):
        if mask is not None and any(not mask[i, j] for i, j in m):
            continue
        v = sum(W[i, j] for i, j in m)
        if v > best_val:
            best_val, best = v, m
    return best_val, best


def sym_random(K, rng):
    W = rng.uniform(size=(K, K))
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0)
    return W


class TestExactSolver:
    @pytest.mark.parametrize("K", [2, 4, 6, 8])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_matches_brute_force(self, K, seed):
        rng = np.random.RandomState(seed)
        W = sym_random(K, rng)
        pairs = mt.max_weight_perfect_matching(W)
        got = sum(W[i, j] for i, j in pairs)
        want, _ = brute_force_mwpm(W)
        assert got == pytest.approx(want)
        covered = sorted(itertools.chain.from_iterable(pairs))
        assert covered == list(range(K))

    def test_negative_weights_ok(self):
        rng = np.random.RandomState(3)
        W = sym_random(6, rng) - 5.0  # all-negative weights still legal edges
        np.fill_diagonal(W, 0)
        pairs = mt.max_weight_perfect_matching(W)
        want, _ = brute_force_mwpm(W)
        assert sum(W[i, j] for i, j in pairs) == pytest.approx(want)

    def test_masked_matches_brute_force(self):
        rng = np.random.RandomState(4)
        K = 6
        W = sym_random(K, rng)
        mask = mt.build_graph_mask("ring", K)
        pairs = mt.max_weight_perfect_matching(W, mask)
        for i, j in pairs:
            assert mask[i, j]
        want, _ = brute_force_mwpm(W, mask)
        assert sum(W[i, j] for i, j in pairs) == pytest.approx(want)

    def test_odd_K_raises(self):
        with pytest.raises(ValueError):
            mt.max_weight_perfect_matching(np.zeros((5, 5)))

    def test_infeasible_mask_raises(self):
        K = 4
        mask = np.zeros((K, K), dtype=bool)
        mask[0, 1] = mask[1, 0] = True  # 2 and 3 isolated
        with pytest.raises(RuntimeError):
            mt.max_weight_perfect_matching(np.ones((K, K)), mask)


class TestGreedy:
    @pytest.mark.parametrize("seed", range(5))
    def test_half_approximation(self, seed):
        rng = np.random.RandomState(seed)
        K = 10
        W = sym_random(K, rng)
        greedy_val = sum(W[i, j] for i, j in mt.greedy_matching(W))
        exact_val = sum(W[i, j]
                        for i, j in mt.max_weight_perfect_matching(W))
        assert greedy_val >= 0.5 * exact_val - 1e-12


class TestPeel:
    def test_disjoint_perfect_decreasing(self):
        rng = np.random.RandomState(7)
        K, k = 8, 3
        W = sym_random(K, rng)
        layers = mt.peel_matchings(W, k)
        assert len(layers) == k
        seen = set()
        totals = []
        for M in layers:
            covered = sorted(itertools.chain.from_iterable(M))
            assert covered == list(range(K))  # each layer perfect
            for e in M:
                assert e not in seen  # edge-disjoint
                seen.add(e)
            totals.append(sum(W[i, j] for i, j in M))
        assert totals == sorted(totals, reverse=True)

    def test_k1_is_mwpm(self):
        rng = np.random.RandomState(8)
        W = sym_random(6, rng)
        assert mt.peel_matchings(W, 1)[0] == \
            mt.max_weight_perfect_matching(W)


class TestRandomMatching:
    def test_valid_and_seeded(self):
        rng1 = np.random.default_rng(5)
        rng2 = np.random.default_rng(5)
        m1 = mt.random_perfect_matching(8, rng1)
        m2 = mt.random_perfect_matching(8, rng2)
        assert m1 == m2
        covered = sorted(itertools.chain.from_iterable(m1))
        assert covered == list(range(8))

    def test_within_ring(self):
        rng = np.random.default_rng(6)
        mask = mt.build_graph_mask("ring", 6)
        m = mt.random_perfect_matching(6, rng, mask)
        for i, j in m:
            assert mask[i, j]


class TestWeights:
    def test_disagreement(self):
        preds = np.array([[0, 0, 1, 1],
                          [0, 1, 1, 0],
                          [0, 0, 1, 1]])
        D = mt.pairwise_disagreement(preds)
        assert D[0, 1] == pytest.approx(0.5)
        assert D[0, 2] == pytest.approx(0.0)
        assert D == pytest.approx(D.T)

    def test_teachable_hand_computed(self):
        y = np.array([0, 0, 0, 0])
        # model 0 right on {0,1,2}; model 1 right on {0,3}
        preds = np.array([[0, 0, 0, 9],
                          [0, 9, 9, 0]])
        # mentor = model 0 (acc .75 vs .5); m = P(0 right & 1 wrong) = 2/4
        # u = P(0 wrong & 1 right) = 1/4; w = m - kappa*u
        W = mt.teachable_weights(preds, y, kappa=1.0)
        assert W[0, 1] == pytest.approx(0.5 - 0.25)
        W2 = mt.teachable_weights(preds, y, kappa=0.0)
        assert W2[0, 1] == pytest.approx(0.5)

    def test_accgap(self):
        y = np.array([0, 0, 0, 0])
        preds = np.array([[0, 0, 0, 9],
                          [0, 9, 9, 0]])
        W = mt.accgap_weights(preds, y)
        assert W[0, 1] == pytest.approx(0.25)

    def test_random_mode_needs_no_preds(self):
        rng = np.random.default_rng(0)
        W = mt.edge_weights("random", None, None, 1.0, 6, rng)
        assert W.shape == (6, 6)
        assert W == pytest.approx(W.T)

    def test_perclass_distance(self):
        # 2 classes; model 0 perfect on class 0, wrong on class 1;
        # model 1 the mirror; model 2 == model 0. y = [0,0,1,1].
        y = np.array([0, 0, 1, 1])
        preds = np.array([[0, 0, 0, 0],   # acc vec [1, 0]
                          [1, 1, 1, 1],   # acc vec [0, 1]
                          [0, 0, 0, 0]])  # acc vec [1, 0] (== model 0)
        W = mt.perclass_distance_weights(preds, y, n_cls=2)
        assert W == pytest.approx(W.T)
        # models 0 and 2 have identical profiles -> distance 0
        assert W[0, 2] == pytest.approx(0.0)
        # models 0 and 1 are maximally complementary: ||[1,0]-[0,1]|| / sqrt(2) = 1
        assert W[0, 1] == pytest.approx(1.0)
        assert W[0, 1] > W[0, 2]

    def test_errorfield_distance(self):
        y = np.array([0, 0, 0, 0])
        # model 0 errs on {2,3}; model 1 errs on {0,1}: disjoint -> all 4 differ
        preds = np.array([[0, 0, 9, 9],
                          [9, 9, 0, 0],
                          [0, 0, 9, 9]])  # == model 0: identical error field
        W = mt.errorfield_distance_weights(preds, y)
        assert W == pytest.approx(W.T)
        assert W[0, 1] == pytest.approx(1.0)   # exactly one errs on every example
        assert W[0, 2] == pytest.approx(0.0)   # identical error fields

    def test_new_modes_via_edge_weights(self):
        rng = np.random.default_rng(0)
        y = np.array([0, 1, 0, 1])
        preds = np.array([[0, 1, 0, 1], [0, 0, 0, 1], [1, 1, 0, 1], [0, 1, 1, 1]])
        for mode in ("perclass", "errorfield"):
            W = mt.edge_weights(mode, preds, y, 1.0, 4, rng)
            assert W.shape == (4, 4)
            assert W == pytest.approx(W.T)
            assert np.allclose(np.diag(W), 0.0)


class TestRecency:
    def test_penalty_breaks_repeated_pair(self):
        # Strong static preference for {0,1},{2,3}; with a large recency
        # penalty the matcher must eventually rotate away from it.
        W = np.array([[0.0, 1.0, 0.1, 0.1],
                      [1.0, 0.0, 0.1, 0.1],
                      [0.1, 0.1, 0.0, 1.0],
                      [0.1, 0.1, 1.0, 0.0]])
        rec = mt.RecencyState(K=4, gamma=1.0)
        first = mt.max_weight_perfect_matching(rec.penalize(W, 5.0))
        assert (0, 1) in first
        rec.update([first])
        second = mt.max_weight_perfect_matching(rec.penalize(W, 5.0))
        assert (0, 1) not in second

    def test_lambda_zero_noop(self):
        W = np.ones((4, 4))
        rec = mt.RecencyState(K=4)
        rec.update([[(0, 1), (2, 3)]])
        assert rec.penalize(W, 0.0) is W

    def test_state_roundtrip(self):
        rec = mt.RecencyState(K=4, gamma=0.7)
        rec.update([[(0, 1), (2, 3)]])
        rec2 = mt.RecencyState.from_state_dict(rec.state_dict())
        assert np.allclose(rec.r, rec2.r)
        assert rec2.gamma == 0.7


class TestTeachers:
    def test_alphas_sum_to_one(self):
        rng = np.random.RandomState(9)
        W = sym_random(6, rng)
        layers = mt.peel_matchings(W, 2)
        for mode in ("weight", "uniform"):
            teachers = mt.matchings_to_teachers(layers, W, 6, mode)
            for ts in teachers:
                assert len(ts) == 2
                assert sum(a for _, a in ts) == pytest.approx(1.0)

    def test_weight_mode_prefers_heavier_edge(self):
        W = np.zeros((4, 4))
        W[0, 1] = W[1, 0] = 0.9
        W[0, 2] = W[2, 0] = 0.1
        W[2, 3] = W[3, 2] = 0.8
        W[1, 3] = W[3, 1] = 0.2
        layers = [[(0, 1), (2, 3)], [(0, 2), (1, 3)]]
        teachers = mt.matchings_to_teachers(layers, W, 4, "weight")
        alphas0 = dict(teachers[0])
        assert alphas0[1] > alphas0[2]

    def test_uniform_mode(self):
        layers = [[(0, 1), (2, 3)]]
        teachers = mt.matchings_to_teachers(layers, np.ones((4, 4)), 4,
                                            "uniform")
        assert teachers[0] == [(1, 1.0)]


class TestGraphs:
    def test_ring_degree_two(self):
        m = mt.build_graph_mask("ring", 8)
        assert (m.sum(axis=0) == 2).all()
        assert not m.diagonal().any()
        assert (m == m.T).all()

    def test_rregular(self):
        m = mt.build_graph_mask("rregular:3", 8, seed=1)
        assert (m.sum(axis=0) == 3).all()
        assert not m.diagonal().any()
        assert (m == m.T).all()

    def test_complete(self):
        m = mt.build_graph_mask("complete", 5)
        assert m.sum() == 5 * 4

    def test_cycle_alias(self):
        assert (mt.build_graph_mask("cycle", 8)
                == mt.build_graph_mask("ring", 8)).all()

    def test_prism_is_3regular(self):
        m = mt.build_graph_mask("prism", 12)
        assert (m.sum(axis=0) == 3).all()
        assert (m == m.T).all()
        assert not m.diagonal().any()
        with pytest.raises(ValueError):
            mt.build_graph_mask("prism", 11)  # needs even K

    def test_latticeK4_is_4regular(self):
        m = mt.build_graph_mask("latticeK4", 12)
        assert (m.sum(axis=0) == 4).all()
        assert (m == m.T).all()

    def test_graph_neighbors(self):
        m = mt.build_graph_mask("ring", 4)   # 0-1-2-3-0
        nb = mt.graph_neighbors(m)
        assert sorted(nb[0]) == [1, 3]
        assert sorted(nb[1]) == [0, 2]

    def test_spectral_gap_expander_beats_cycle(self):
        # An expander must have a strictly larger spectral gap than a cycle.
        gap_ring = mt.graph_spectral_gap(mt.build_graph_mask("ring", 12))
        gap_prism = mt.graph_spectral_gap(mt.build_graph_mask("prism", 12))
        gap_rand = mt.graph_spectral_gap(
            mt.build_graph_mask("rregular:3", 12, seed=1))
        assert gap_rand > gap_prism > gap_ring >= 0
        # the expander mixes markedly better than the cycle at equal-ish size
        assert gap_rand > 2 * gap_ring
        # and the cycle's gap shrinks with N (1 - cos(2pi/N)), motivating
        # larger cohorts for a clean expansion separation
        assert (mt.graph_spectral_gap(mt.build_graph_mask("ring", 24))
                < mt.graph_spectral_gap(mt.build_graph_mask("ring", 12)))
