"""Matching engine: edge weights, maximum-weight perfect matchings, peeled
k-matchings, recency penalties, and communication-graph restrictions.

This is the DML-side implementation of the pairing rule developed in the
knowledge-diffusion repo (src/policies/mwm.py there; ideas in its
dev-communication/ideas.md, entry 2026-07-16). Differences from the KD-repo
version: the cohort is Oracle-less and symmetric (mutual pairs, no
mentor/mentee roles), the exact solver is a bitmask DP (exact up to K=16),
and it adds edge peeling (Idea 2c), a recency penalty (rotation protection),
and restriction to a fixed communication graph (Idea 2b).

Edge-weight signals (computed from hard predictions on the validation set)
---------------------------------------------------------------------------
* ``disagreement`` (default; label-free, symmetric — the benign-cohort v1
  weight of Idea 2b):      d_ij = P(pred_i != pred_j)
* ``teachable`` (labels needed; the MWM-D weight): for the pair {i, j} with
  mentor = the higher-validation-accuracy member,
      w_ij = m_ij - kappa * u_ij,
  where m_ij = P(mentor right AND mentee wrong), u_ij = P(mentor wrong AND
  mentee right). In mutual (symmetric-KD) sessions the pair-summed
  first-order gain cancels, so this is a second-order signal like d_ij.
* ``accgap``: |acc_i - acc_j| — the first-order shadow; by the modularity
  result it should NOT beat random. Included as a falsification arm.
* ``random``: weights drawn i.i.d. uniform — a fresh random perfect matching
  every refresh.

All weights are symmetric; matchings are over an even number of models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

Pair = Tuple[int, int]

WEIGHT_MODES = ("disagreement", "teachable", "accgap", "random",
                "perclass", "errorfield")

_EXACT_MAX_K = 16  # bitmask DP: 2^16 * 16^2 ~ 17M ops, < 1 s


# ---------------------------------------------------------------------------
# Edge weights
# ---------------------------------------------------------------------------

def pairwise_disagreement(preds: np.ndarray) -> np.ndarray:
    """preds: (K, n) hard predictions. Returns symmetric (K, K) d_ij."""
    K = preds.shape[0]
    D = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            d = float(np.mean(preds[i] != preds[j]))
            D[i, j] = D[j, i] = d
    return D


def teachable_weights(preds: np.ndarray, y: np.ndarray,
                      kappa: float = 1.0) -> np.ndarray:
    """MWM-D weights. preds: (K, n) hard predictions, y: (n,) clean labels."""
    correct = preds == y[None, :]
    acc = correct.mean(axis=1)
    K = preds.shape[0]
    W = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            mentor, mentee = (i, j) if acc[i] >= acc[j] else (j, i)
            m = float(np.mean(correct[mentor] & ~correct[mentee]))
            u = float(np.mean(~correct[mentor] & correct[mentee]))
            W[i, j] = W[j, i] = m - kappa * u
    return W


def accgap_weights(preds: np.ndarray, y: np.ndarray) -> np.ndarray:
    acc = (preds == y[None, :]).mean(axis=1)
    return np.abs(acc[:, None] - acc[None, :])


def perclass_accuracy_vectors(preds: np.ndarray, y: np.ndarray,
                              n_cls: int) -> np.ndarray:
    """Per-class competence profile a_i in R^n_cls: a_i[c] = accuracy of model i
    on the validation examples whose TRUE label is c. Shape (K, n_cls).
    Classes absent from the val set get 0 (they contribute equally to every
    pair, so they do not bias distances)."""
    K = preds.shape[0]
    A = np.zeros((K, n_cls))
    for c in range(n_cls):
        m = (y == c)
        if m.any():
            A[:, c] = (preds[:, m] == c).mean(axis=1)
    return A


def perclass_distance_weights(preds: np.ndarray, y: np.ndarray,
                              n_cls: int) -> np.ndarray:
    """Multidimensional edge weight: Euclidean distance between per-class
    accuracy profiles, w_ij = ||a_i - a_j||_2 (KD ideas.md granularity ladder;
    the per-CLASS aggregation of the error field). Large when two models are
    good at DIFFERENT classes — complementary competence a scalar disagreement
    can miss when two models disagree equally overall but on different classes.
    Normalized by sqrt(n_cls) so the scale is comparable across datasets."""
    A = perclass_accuracy_vectors(preds, y, n_cls)
    K = preds.shape[0]
    W = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            W[i, j] = W[j, i] = float(
                np.linalg.norm(A[i] - A[j]) / np.sqrt(n_cls))
    return W


def errorfield_distance_weights(preds: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-example error-field distance: with e_i[x] = 1[pred_i(x) != y(x)],
    w_ij = P(exactly one of i, j errs) = mean(e_i XOR e_j) (the symmetric
    teachable mass m_ij + u_ij; the finest-grain version of the KD error-field
    overlap rho_ij). Large when the two models are wrong on DIFFERENT examples
    — disjoint error fields, the pair the theory calls most valuable."""
    err = (preds != y[None, :])
    K = preds.shape[0]
    W = np.zeros((K, K))
    for i in range(K):
        for j in range(i + 1, K):
            W[i, j] = W[j, i] = float(np.mean(err[i] != err[j]))
    return W


def edge_weights(mode: str, preds: Optional[np.ndarray],
                 y: Optional[np.ndarray], kappa: float,
                 K: int, rng: np.random.Generator) -> np.ndarray:
    if mode == "random":
        W = rng.uniform(size=(K, K))
        W = (W + W.T) / 2
        np.fill_diagonal(W, 0.0)
        return W
    if preds is None:
        raise ValueError(f"weight mode '{mode}' needs validation predictions")
    if mode == "disagreement":
        return pairwise_disagreement(preds)
    if mode == "teachable":
        if y is None:
            raise ValueError("teachable weights need validation labels")
        return teachable_weights(preds, y, kappa)
    if mode == "accgap":
        if y is None:
            raise ValueError("accgap weights need validation labels")
        return accgap_weights(preds, y)
    if mode == "perclass":
        if y is None:
            raise ValueError("perclass weights need validation labels")
        n_cls = int(y.max()) + 1
        return perclass_distance_weights(preds, y, n_cls)
    if mode == "errorfield":
        if y is None:
            raise ValueError("errorfield weights need validation labels")
        return errorfield_distance_weights(preds, y)
    raise ValueError(f"Unknown weight mode '{mode}'.")


# ---------------------------------------------------------------------------
# Communication graphs (Idea 2b: matching restricted to E(G))
# ---------------------------------------------------------------------------

def build_graph_mask(kind: str, K: int, seed: int = 0) -> np.ndarray:
    """Boolean (K, K) adjacency; True = edge allowed. 'complete', 'ring', or
    'rregular:d' (random d-regular via the configuration model, seeded)."""
    mask = np.zeros((K, K), dtype=bool)
    if kind == "complete":
        mask[:] = True
        np.fill_diagonal(mask, False)
        return mask
    if kind == "ring":
        for i in range(K):
            mask[i, (i + 1) % K] = mask[(i + 1) % K, i] = True
        return mask
    if kind.startswith("rregular:"):
        d = int(kind.split(":", 1)[1])
        if d >= K or (d * K) % 2 != 0:
            raise ValueError(f"No {d}-regular graph on {K} nodes.")
        rng = np.random.RandomState(seed)
        for _attempt in range(1000):
            stubs = np.repeat(np.arange(K), d)
            rng.shuffle(stubs)
            m = np.zeros((K, K), dtype=bool)
            ok = True
            for a, b in zip(stubs[0::2], stubs[1::2]):
                if a == b or m[a, b]:
                    ok = False
                    break
                m[a, b] = m[b, a] = True
            if not ok:
                continue
            # A d-regular graph need not admit a perfect matching (e.g. a
            # disconnected component with an odd cut) — resample until the
            # mask can actually host matched arms.
            if K % 2 == 0:
                try:
                    max_weight_perfect_matching(np.ones((K, K)), m)
                except RuntimeError:
                    continue
            return m
        raise RuntimeError(f"Could not sample a perfectly-matchable "
                           f"{d}-regular graph on {K} nodes.")
    raise ValueError(f"Unknown graph '{kind}'.")


# ---------------------------------------------------------------------------
# Solvers
# ---------------------------------------------------------------------------

_NEG = -1e18  # forbidden-edge weight


def _masked(W: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    Wm = W.astype(float).copy()
    if mask is not None:
        Wm[~mask] = _NEG
    np.fill_diagonal(Wm, _NEG)
    return Wm


def max_weight_perfect_matching(W: np.ndarray,
                                mask: Optional[np.ndarray] = None
                                ) -> List[Pair]:
    """Maximum-weight PERFECT matching. Exact bitmask DP for K <= 16;
    greedy (locally-dominant edges, the Preis/Hoepman 1/2-approximation
    order) above. For K <= 16, raises iff no perfect matching exists within
    the mask. For K > 16 the greedy fallback builds a maximal matching,
    which on a masked (sparse) graph may fail to be perfect even when a
    perfect matching exists — it then raises conservatively. Dense cohorts
    (mask=None) are always perfectly matchable by greedy."""
    K = W.shape[0]
    if K % 2 != 0:
        raise ValueError(f"Perfect matching needs even K, got {K}.")
    Wm = _masked(W, mask)
    if K > _EXACT_MAX_K:
        pairs = greedy_matching(W, mask)
        if len(pairs) < K // 2:
            raise RuntimeError("Greedy matching could not perfectly match "
                               "the cohort within the graph.")
        return pairs

    full = (1 << K) - 1
    dp = np.full(1 << K, -np.inf)
    dp[0] = 0.0
    choice: dict = {}
    for s in range(1 << K):
        if dp[s] == -np.inf:
            continue
        # Lowest unmatched vertex — canonical, halves the state space.
        rem = full & ~s
        if rem == 0:
            continue
        i = (rem & -rem).bit_length() - 1
        for j in range(i + 1, K):
            if not (s >> j) & 1:
                if Wm[i, j] <= _NEG / 2:
                    continue
                ns = s | (1 << i) | (1 << j)
                nv = dp[s] + Wm[i, j]
                if nv > dp[ns]:
                    dp[ns] = nv
                    choice[ns] = (s, i, j)
    if dp[full] == -np.inf:
        raise RuntimeError("No perfect matching exists within the graph mask.")
    pairs: List[Pair] = []
    s = full
    while s:
        s, i, j = choice[s]
        pairs.append((i, j))
    return sorted(pairs)


def greedy_matching(W: np.ndarray, mask: Optional[np.ndarray] = None
                    ) -> List[Pair]:
    """Sorted-greedy matching = the outcome of iterated locally-dominant
    edge selection (Preis/Hoepman) for static weights; 1/2-approximation."""
    K = W.shape[0]
    Wm = _masked(W, mask)
    edges = sorted(((Wm[i, j], i, j) for i in range(K) for j in range(i + 1, K)
                    if Wm[i, j] > _NEG / 2), reverse=True)
    used: set = set()
    pairs: List[Pair] = []
    for _, i, j in edges:
        if i not in used and j not in used:
            pairs.append((i, j))
            used.update((i, j))
    return sorted(pairs)


def random_perfect_matching(K: int, rng: np.random.Generator,
                            mask: Optional[np.ndarray] = None) -> List[Pair]:
    """Uniform random perfect matching (complete graph) or a random perfect
    matching within the mask (max-weight matching on i.i.d. random weights,
    which for restricted graphs is random but not exactly uniform)."""
    if mask is None:
        order = rng.permutation(K)
        return sorted(tuple(sorted((int(order[2 * i]), int(order[2 * i + 1]))))
                      for i in range(K // 2))
    W = rng.uniform(size=(K, K))
    W = (W + W.T) / 2
    return max_weight_perfect_matching(W, mask)


def peel_matchings(W: np.ndarray, k: int,
                   mask: Optional[np.ndarray] = None) -> List[List[Pair]]:
    """Peel k edge-disjoint maximum-weight perfect matchings, best first
    (Idea 2c: the greedy maximum-weight k-matching, already edge-colored by
    weight rank).

    Feasibility: on the complete graph, after peeling j matchings every
    vertex retains degree K-1-j, so by Dirac's theorem a perfect matching
    is GUARANTEED to remain while K-1-j >= K/2, i.e. for k <= K/2. Beyond
    that (or on masked graphs) greedy peeling can strand the residual graph
    and this raises RuntimeError. The suite uses k <= 3 at K = 8."""
    K = W.shape[0]
    if k < 1:
        raise ValueError("k must be >= 1")
    avail = np.ones((K, K), dtype=bool) if mask is None else mask.copy()
    np.fill_diagonal(avail, False)
    matchings: List[List[Pair]] = []
    for _ in range(k):
        M = max_weight_perfect_matching(W, avail)
        matchings.append(M)
        for i, j in M:
            avail[i, j] = avail[j, i] = False
    return matchings


# ---------------------------------------------------------------------------
# Recency penalty (rotation protection)
# ---------------------------------------------------------------------------

@dataclass
class RecencyState:
    """Exponentially decayed count of recent pairings.

    On every refresh: r <- gamma * r, then r_ij += 1 for each matched pair.
    The penalized weight is  w'_ij = w_ij - lam * r_ij * scale,  where scale
    is the mean |w| over allowed edges — this keeps lam unitless across
    weight modes. lam = 0 disables the penalty (r is still tracked, so the
    matches.csv diagnostics are comparable across arms).
    """
    K: int
    gamma: float = 0.5
    r: np.ndarray = field(default=None)

    def __post_init__(self):
        if self.r is None:
            self.r = np.zeros((self.K, self.K))

    def penalize(self, W: np.ndarray, lam: float,
                 mask: Optional[np.ndarray] = None) -> np.ndarray:
        if lam == 0.0:
            return W
        allowed = np.ones_like(W, dtype=bool) if mask is None else mask
        allowed = allowed & ~np.eye(self.K, dtype=bool)
        scale = float(np.mean(np.abs(W[allowed]))) if allowed.any() else 0.0
        return W - lam * scale * self.r

    def update(self, matchings: Sequence[Sequence[Pair]]) -> None:
        self.r *= self.gamma
        for M in matchings:
            for i, j in M:
                self.r[i, j] += 1.0
                self.r[j, i] += 1.0

    def state_dict(self):
        return {"K": self.K, "gamma": self.gamma, "r": self.r.copy()}

    @classmethod
    def from_state_dict(cls, d):
        return cls(K=int(d["K"]), gamma=float(d["gamma"]), r=d["r"].copy())


# ---------------------------------------------------------------------------
# Per-model teacher lists from matchings
# ---------------------------------------------------------------------------

def matchings_to_teachers(matchings: List[List[Pair]], W: np.ndarray,
                          K: int, peel_weighting: str = "weight"
                          ) -> List[List[Tuple[int, float]]]:
    """Convert k edge-disjoint matchings into per-model teacher lists
    [(teacher_idx, alpha)], with alphas normalized to sum to 1 per model.

    'uniform': alpha = 1/k for every matched teacher.
    'weight' (Idea 2c design point (i)): alpha_l proportional to
    max(w(M_l(i)), 0) + eps, so the lighter peeled matching contributes
    less and the terms degrade gracefully toward k=1 as weights deplete.
    """
    eps = 1e-8
    teachers: List[List[Tuple[int, float]]] = [[] for _ in range(K)]
    for M in matchings:
        for i, j in M:
            teachers[i].append((j, float(W[i, j])))
            teachers[j].append((i, float(W[i, j])))
    out: List[List[Tuple[int, float]]] = []
    for i in range(K):
        ts = teachers[i]
        if not ts:
            out.append([])
            continue
        if peel_weighting == "uniform":
            alphas = np.full(len(ts), 1.0 / len(ts))
        elif peel_weighting == "weight":
            raw = np.array([max(w, 0.0) + eps for _, w in ts])
            alphas = raw / raw.sum()
        else:
            raise ValueError(f"Unknown peel_weighting '{peel_weighting}'.")
        out.append([(j, float(a)) for (j, _), a in zip(ts, alphas)])
    return out
