"""Ladder-mined deck distributions (the ALL-DECKS design).

Two consumers:

* GENERALIST self-play: `--decks ladder_all` -> UNIFORM over all mined lists (breadth; no meta
  snapshot baked in; per-ARCHETYPE density stays healthy because lists share cores).
* FINETUNE opponents: a DESIGNED counts file (F_now / F_counter / F_wide, e.g.
  scripts/build_ft_*.py) -- `counts` is plain per-deck sampling mass, normalized here.
  `mixed_weights(counts, uniform_mix)` alpha-blends it with uniform as late-competition
  shift insurance; designed mixtures already encode their intended spread, so ft jobs run
  `--opponent-uniform-mix 0.0` and the file IS the distribution.

Rating-band blend REMOVED (user order 2026-08-03: opponent pools are always designed
mixtures now). Legacy files with [N, 3] per-band counts (climb/contested/summit -- raw pool
files like ladder_decks_v16.json, or the [c,c,c] triplication older ft builders wrote) load
fine: rows are summed to a scalar mass. NB for raw pool files that changes semantics vs the
retired 0.2/0.3/0.5 band blend (raw play-count proportions, no summit up-weighting) -- do
not point --ladder-decks-file at a raw pool for finetune opponents; build a mixture file.
"""
from __future__ import annotations

import json

import numpy as np


def load_ladder_decks(path):
    """-> (decks: list[list[int]], counts: np.ndarray [N] per-deck sampling mass).
    Accepts scalar counts, [N, 1], or legacy [N, 3] per-band rows (summed)."""
    d = json.load(open(path))
    decks = [list(x) for x in d["decks"]]
    counts = np.asarray(d["counts"], dtype=np.float64)
    if counts.ndim == 2:
        counts = counts.sum(axis=1)
    assert counts.shape == (len(decks),), f"counts shape {counts.shape} vs {len(decks)} decks"
    assert all(len(x) == 60 for x in decks)
    return decks, counts


def mixed_weights(counts, uniform_mix=0.5):
    """(1-uniform_mix)*counts/sum + uniform_mix*uniform -- the finetune opponent distribution."""
    counts = np.asarray(counts, dtype=np.float64)
    n = len(counts)
    tot = counts.sum()
    meta = counts / tot if tot > 0 else np.full(n, 1.0 / n)
    return (1.0 - uniform_mix) * meta + uniform_mix / n
