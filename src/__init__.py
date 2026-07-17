"""Reduced-communication deep mutual learning (matched mutual learning, MML).

Extension of the DML baseline in this repository (root-level trainer.py,
faithful to Zhang et al., CVPR 2018) with per-round matching-based peer
selection, peeled k-matchings, and communication accounting.

Design document: dev-communication/experiments.md
Origin of the ideas: knowledge-diffusion repo, dev-communication/ideas.md,
entry "2026-07-16 — Matched & sparse mutual learning (DML x MWM-D x topology)".
"""
