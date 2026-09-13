"""READ-TIME matchup-matrix report (the log is full-granularity; ALL aggregation happens here).

Input: a run dir containing matchup_r*.jsonl (written by train_selfplay every iteration:
{"step": S, "cells": {"i:j": [w0, w1, draws, truncs]}} -- i/j = pool indices, i is seat-0's deck;
-1 = the finetune ship deck) + the pool the indices refer to (--ladder-file or --pool name).

Distortion handling (user design 2026-07-09):
  * Distortion 1 (cell N ~ p(A)p(B)): family-cluster aggregation (--cluster-t, union-find on card
    overlap) + Wilson CIs + a min-N gate. Granularity preserved in the LOG; collapse only here.
  * Distortion 2 (exposure-skew: the net pilots popular decks better, overstating edges vs
    rare decks): per-deck exposure counts + [EXP-SKEW] flag when sides differ >EXPOSURE_SKEW x,
    and a TREND column (first-half vs second-half wr) -- a DECLINING edge as the rare side's
    exposure grows = exposure-inflated cell, flagged [DECLINING].

  python scripts/matchup_report.py RUN_DIR --ladder-file ladder_decks_v4.json \\
      [--cluster-t 40] [--min-n 30] [--focus IDX|name-substr] [--top 25]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EXPOSURE_SKEW = 10.0


def wilson(w, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = w / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    s = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - s) / d, (c + s) / d)


def load_events(run_dir):
    """[(step, i, j, [w0, w1, dr, tr])] merged across ranks, step-sorted; + headers."""
    ev, headers = [], []
    for f in sorted(glob.glob(os.path.join(run_dir, "matchup_r*.jsonl"))):
        for line in open(f):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "header" in d:
                headers.append(d["header"])
                continue
            for k, c in d["cells"].items():
                i, j = k.split(":")
                ev.append((int(d["step"]), int(i), int(j), c))
    ev.sort(key=lambda x: x[0])
    return ev, headers


def cluster_families(decks, t):
    """union-find over card overlap >= t -> deck_idx -> family root; -1 stays -1 (ship deck)."""
    from collections import Counter as C
    cs = [C(d) for d in decks]
    par = list(range(len(decks)))

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for a in range(len(decks)):
        for b in range(a + 1, len(decks)):
            if sum((cs[a] & cs[b]).values()) >= t:
                par[find(a)] = find(b)
    return {i: find(i) for i in range(len(decks))}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir")
    p.add_argument("--ladder-file", default=None, help="ladder_decks json the indices refer to")
    p.add_argument("--pool", default=None, help="resolve_deck_pool name instead of a ladder file")
    p.add_argument("--cluster-t", type=int, default=0, help="family overlap threshold (0 = per-list)")
    p.add_argument("--min-n", type=int, default=30)
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--focus", default=None, help="only cells involving this deck index (int)")
    a = p.parse_args()

    ev, headers = load_events(a.run_dir)
    if not ev:
        raise SystemExit("no matchup events found")
    print(f"[mm] {len(ev)} window-cells, steps {ev[0][0]:,}..{ev[-1][0]:,}; headers={headers[:1]}")

    decks = None
    if a.ladder_file:
        from rl.decks_ladder import load_ladder_decks
        decks, _ = load_ladder_decks(a.ladder_file)
    elif a.pool:
        from rl.train_selfplay import resolve_deck_pool
        decks = resolve_deck_pool(a.pool)

    fam = None
    if a.cluster_t and decks:
        fam = cluster_families(decks, a.cluster_t)
        fam[-1] = -1
        print(f"[mm] clustered {len(decks)} decks -> {len(set(fam.values()))} families (t={a.cluster_t})")

    def key(i, j):
        if fam is not None:
            i, j = fam.get(i, i), fam.get(j, j)
        return (i, j) if i <= j else (j, i)          # unordered pair; counts re-oriented below

    mid = ev[len(ev) // 2][0]                        # median step = time-axis split point
    agg = defaultdict(lambda: [0, 0, 0, 0])          # (a,b) -> [wins_a, wins_b, draws, truncs]
    half = {0: defaultdict(lambda: [0, 0]), 1: defaultdict(lambda: [0, 0])}   # wr trend halves
    expo = Counter()                                 # deck/family -> total episodes involved
    for step, i, j, (w0, w1, dr, tr) in ev:
        ia, ib = (fam.get(i, i), fam.get(j, j)) if fam is not None else (i, j)
        flip = ia > ib
        A, B = (ib, ia) if flip else (ia, ib)
        wa, wb = (w1, w0) if flip else (w0, w1)
        c = agg[(A, B)]
        c[0] += wa; c[1] += wb; c[2] += dr; c[3] += tr
        h = half[0 if step <= mid else 1][(A, B)]
        h[0] += wa; h[1] += wa + wb
        n = w0 + w1 + dr + tr
        expo[ia] += n; expo[ib] += n

    def name(x):
        return "SHIP" if x == -1 else f"d{x}"

    rows = []
    for (A, B), (wa, wb, dr, tr) in agg.items():
        if a.focus is not None and int(a.focus) not in (A, B):
            continue
        n = wa + wb
        if n < a.min_n:
            continue
        wr = wa / n
        lo, hi = wilson(wa, n)
        h0, h1 = half[0].get((A, B)), half[1].get((A, B))
        trend = (h1[0] / h1[1] - h0[0] / h0[1]) if (h0 and h1 and h0[1] >= 10 and h1[1] >= 10) else None
        skew = max(expo[A], 1) / max(expo[B], 1)
        flags = []
        if skew > EXPOSURE_SKEW or skew < 1 / EXPOSURE_SKEW:
            flags.append("EXP-SKEW")
        if trend is not None and abs(trend) > 0.05:
            flags.append("DECLINING" if (trend < 0) == (wr > 0.5) else "RISING")
        rows.append((abs(wr - 0.5), A, B, wr, lo, hi, n, dr + tr, expo[A], expo[B], trend, flags))

    rows.sort(key=lambda r: -r[0])
    print(f"\n{'A':>6} {'B':>6} {'wr(A)':>6} {'95% CI':>13} {'n':>7} {'d+t':>5} "
          f"{'exp(A)':>8} {'exp(B)':>8} {'trend':>6}  flags")
    for _, A, B, wr, lo, hi, n, dt, ea, eb, tr_, fl in rows[:a.top]:
        t = f"{tr_:+.3f}" if tr_ is not None else "   --"
        print(f"{name(A):>6} {name(B):>6} {wr:6.3f} [{lo:.3f},{hi:.3f}] {n:7d} {dt:5d} "
              f"{ea:8d} {eb:8d} {t:>6}  {','.join(fl)}")


if __name__ == "__main__":
    main()
