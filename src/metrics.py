"""Evaluation and CSV logging for the mutual-learning cohort.

Reporting conventions (dev-communication/experiments.md §4):
* Individual accuracy: top-1 on the 10k test set, per model, plus the mean
  over the cohort (and per-architecture means for heterogeneous cohorts).
* Ensemble accuracy: argmax of the MEAN POSTERIOR (average of softmax
  outputs) over the cohort — the paper's ensembling convention.
* Diversity readouts: mean pairwise disagreement P(pred_i != pred_j) and
  mean pairwise error correlation rho (Pearson correlation of the per-example
  error indicators), computed on both the validation set (policy-facing) and
  the test set (reporting).
"""

from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Cohort evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_cohort(models: List[torch.nn.Module], loader, device) -> Dict:
    """One pass of every model over ``loader``.

    Returns {'preds': (K, n) int array, 'y': (n,), 'accs': (K,),
    'ensemble_acc': float}. Ensemble = argmax of the mean posterior.
    """
    for m in models:
        m.eval()
    K = len(models)
    all_preds = [[] for _ in range(K)]
    all_y = []
    ens_correct = 0
    n_total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y_dev = y.to(device, non_blocking=True)
        probs_sum = None
        for i, m in enumerate(models):
            logits = m(x)
            probs = F.softmax(logits, dim=1)
            probs_sum = probs if probs_sum is None else probs_sum + probs
            all_preds[i].append(logits.argmax(dim=1).cpu().numpy())
        ens_correct += int((probs_sum.argmax(dim=1) == y_dev).sum().item())
        n_total += y.size(0)
        all_y.append(y.numpy())
    preds = np.stack([np.concatenate(p) for p in all_preds])
    y = np.concatenate(all_y)
    accs = (preds == y[None, :]).mean(axis=1)
    return {
        "preds": preds,
        "y": y,
        "accs": accs,
        "ensemble_acc": ens_correct / max(n_total, 1),
    }


def mean_pairwise_disagreement(preds: np.ndarray) -> float:
    K = preds.shape[0]
    if K < 2:
        return float("nan")
    total, cnt = 0.0, 0
    for i in range(K):
        for j in range(i + 1, K):
            total += float(np.mean(preds[i] != preds[j]))
            cnt += 1
    return total / cnt


def mean_pairwise_error_correlation(preds: np.ndarray, y: np.ndarray) -> float:
    """Mean Pearson correlation of error indicators over all pairs.
    Pairs where either member has zero error variance are skipped."""
    K = preds.shape[0]
    if K < 2:
        return float("nan")
    errors = (preds != y[None, :]).astype(float)
    stds = errors.std(axis=1)
    vals = []
    for i in range(K):
        for j in range(i + 1, K):
            if stds[i] < 1e-12 or stds[j] < 1e-12:
                continue
            c = np.corrcoef(errors[i], errors[j])[0, 1]
            if np.isfinite(c):
                vals.append(float(c))
    return float(np.mean(vals)) if vals else float("nan")


# ---------------------------------------------------------------------------
# Streaming CSV writer (resume-aware)
# ---------------------------------------------------------------------------

class CsvWriter:
    """Append-oriented CSV writer. The header is fixed by the first row ever
    written; later rows must carry the same keys. ``truncate_to_epoch``
    supports checkpoint resume: it drops rows with epoch > the resumed epoch.

    Each write REOPENS the file (open-append-close) instead of holding a
    long-lived handle. This matters on a live results directory: git
    operations (stash apply, checkout) replace files by inode, and a held
    handle then appends into the orphaned old inode — rows silently vanish
    from the visible file (observed in production, [D-006]). Reopening by
    path per write survives any concurrent file replacement; at one write
    per epoch the cost is nil."""

    def __init__(self, path: str):
        self.path = path
        self.fieldnames: Optional[List[str]] = None
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, newline="") as fh:
                reader = csv.reader(fh)
                try:
                    self.fieldnames = next(reader)
                except StopIteration:
                    self.fieldnames = None

    def truncate_to_epoch(self, epoch: int, epoch_col: str = "epoch") -> None:
        """Keep only rows with int(row[epoch_col]) <= epoch."""
        if self.fieldnames is None or not os.path.exists(self.path):
            return
        with open(self.path, newline="") as fh:
            rows = list(csv.DictReader(fh))
        kept = [r for r in rows if r.get(epoch_col) not in (None, "")
                and int(float(r[epoch_col])) <= epoch]
        with open(self.path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(kept)

    def write(self, row: Dict) -> None:
        if self.fieldnames is None:
            self.fieldnames = list(row.keys())
        need_header = (not os.path.exists(self.path)
                       or os.path.getsize(self.path) == 0)
        with open(self.path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.fieldnames)
            if need_header:
                writer.writeheader()
            writer.writerow(row)

    def close(self) -> None:
        pass  # kept for API compatibility; no handle is held anymore
