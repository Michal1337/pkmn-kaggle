"""DECK RANKING vs the live meta AND a wide static prior (the ship-selection instrument).

Motivation: raw training winrates are a function of OUR tf-idf SAMPLING -- a deck can look strong
merely because the pool over-samples what it beats. This scores every deck twice over the same
matchup cells, changing only the weights:

  EV_kaggle  opponents weighted by ACTUAL 3-day ladder play share. Each played list is
             represented by its tf-idf neighbourhood (cos >= T) with each neighbour's games
             weighted by sim**KPOW, so the played build dominates its own estimate instead of
             being averaged with weaker cousins (a flat neighbourhood flatters us ~9pts on
             grimmsnarl, whose played lists sit at percentile 97-100 of their neighbours).
             No exact-match needed: unseen lists still map onto whatever resembles them.
  EV_uniform every VIABLE deck counts once (fixed/static/wide). Games are importance-reweighted
             by (1/n_viable)/pool_weight_j to undo the sampler; pool weights are clipped so the
             importance weights stay bounded. The viability FLOOR matters: without it 18% of the
             mass sits on sub-0.40 decks, and a density correction is WORSE (corr(density,wr)
             = +0.58, i.e. isolated decks are weak, so 1/density loads 55% of mass onto junk).
  EV_blend   0.5 * each.

Also reports SPREAD = kaggle - uniform (positive = favoured by today's meta, negative = better
against a wide field) and TAIL = winrate over the worst decile of covered meta mass (a deck can
average well and still have an exploitable hole).

Loop-prone lists (>5% truncation, the shell stall bug) are excluded on both sides.

  deck_rank_blend.py [COS_T] [MIN_CELL] [TOPN] [VIABILITY_FLOOR] [SIM_POWER]
  defaults:          0.85    30         40     0.45              30
"""
import csv
import glob
import json
import math
import sys
from collections import Counter, defaultdict

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
G = "data"
RUN = f"{G}/runs/gen20m_v6"
T = float(sys.argv[1]) if len(sys.argv) > 1 else 0.85
MINC = int(sys.argv[2]) if len(sys.argv) > 2 else 30
TOPN = int(sys.argv[3]) if len(sys.argv) > 3 else 40
FLOOR = float(sys.argv[4]) if len(sys.argv) > 4 else 0.45
KPOW = float(sys.argv[5]) if len(sys.argv) > 5 else 30.0
# argv[6] = meta-shares file (default rolling 3d); argv[7] = sort column b|k|u
# (replaces the blend_v16_1d.py fork -- 1-day pure-kaggle view = `... mined_1d_X.json k`)
SHARES = sys.argv[6] if len(sys.argv) > 6 else f"{G}/episodes/mined_3d_shares.json"
SORTK = sys.argv[7] if len(sys.argv) > 7 else "b"
assert SORTK in ("b", "k", "u"), "sort column must be b|k|u"

pool = json.load(open(f"{G}/ladder_decks_v16.json"))["decks"]
N = len(pool)
pw = np.load(f"{G}/pool_v16_weights_5050.npy")
pw = pw / pw.sum()
names = {}
for r in csv.DictReader(open("EN_Card_Data.csv", encoding="utf-8-sig")):
    try:
        names[int(r["Card ID"])] = r["Card Name"].strip()
    except (ValueError, KeyError):
        pass


def label(i, k=4):
    c = Counter(int(x) for x in pool[i])
    out = []
    for cid, n in c.most_common():
        nm = names.get(cid, str(cid))
        if "Energy" in nm:
            continue
        out.append(f"{n}x {nm}")
        if len(out) >= k:
            break
    return ", ".join(out)


live = pw > 0   # viability gate applied after the winrates are known (see below)

# ---- tf-idf for the meta neighbourhoods ----
vocab = sorted({c for d in pool for c in d})
col = {c: k for k, c in enumerate(vocab)}
M = np.zeros((N, len(vocab)), dtype=np.float32)
for i, d in enumerate(pool):
    for c in d:
        M[i, col[c]] += 1.0
idf = np.log(N / np.maximum((M > 0).sum(axis=0), 1))
M *= idf
M /= np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)

m = json.load(open(SHARES))
META = []
for d, cnt in zip(m["decks"], m["counts"]):
    v = np.zeros(len(vocab), dtype=np.float32)
    for c in d:
        if int(c) in col:
            v[col[int(c)]] += 1.0
    v *= idf
    nrm = np.linalg.norm(v)
    if nrm > 1e-9:
        META.append((v / nrm, float(sum(cnt))))
tot = sum(p for _, p in META)

# ---- cells ----
SELF = defaultdict(lambda: [0, 0, 0])
PAIR = defaultdict(lambda: [0, 0])
for fp in sorted(glob.glob(f"{RUN}/matchup_r*.jsonl")):
    with open(fp) as f:
        for line in f:
            if '"cells"' not in line:
                continue
            for pr, v in json.loads(line)["cells"].items():
                a, b = pr.split(":")
                a, b = int(a), int(b)
                if not (0 <= a < N) or not (0 <= b < N):
                    continue
                dec = v[0] + v[1]
                tr = v[3] if len(v) > 3 else 0
                for idx, w in ((a, v[0]), (b, v[1])):
                    s = SELF[idx]; s[0] += w; s[1] += dec; s[2] += dec + tr
                if dec:
                    c = PAIR[(a, b)]; c[0] += v[0]; c[1] += dec
                    c = PAIR[(b, a)]; c[0] += v[1]; c[1] += dec
loopy = {i for i, (w, d, g) in SELF.items() if g >= 150 and (g - d) / g > 0.05}

NBR, W = [], []
for vec, p in META:
    sim = M @ vec
    idx = [int(j) for j in np.where(sim >= T)[0] if j not in loopy]
    if idx:
        sw = np.power(np.clip(np.array([sim[j] for j in idx]), 0, 1), KPOW)
        sw = sw / sw.sum()
        NBR.append(list(zip(idx, sw.tolist()))); W.append(p / tot)
eff = [1.0 / sum(w * w for _, w in nb) for nb in NBR]      # effective #decks per neighbourhood
print(f"similarity weighting sim**{KPOW}: effective neighbours "
      f"min {min(eff):.1f} | median {np.median(eff):.1f} | max {max(eff):.1f} "
      f"(raw sizes {min(len(n) for n in NBR)}..{max(len(n) for n in NBR)})")

# per-candidate: uniform (importance-reweighted over ALL opponents) + kaggle (neighbourhoods)
byc = defaultdict(list)
for (a, b), (w, n) in PAIR.items():
    byc[a].append((b, w, n))

# ---- viability gate: a wide prior over decks someone could PLAUSIBLY play ----
selfwr = np.full(N, np.nan)
for i, (w_, d_, g_) in SELF.items():
    if 0 <= i < N and d_ >= 500:
        selfwr[i] = w_ / d_
viable = live & ~np.isnan(selfwr) & (selfwr >= FLOOR)
imp = np.zeros(N)
imp[viable] = (1.0 / viable.sum()) / pw[viable]
print(f"viability floor {FLOOR:.2f}: {int(viable.sum())} of {int(live.sum())} live decks kept "
      f"| dropped mean-wr {np.nanmean(selfwr[live & ~viable]):.3f} "
      f"| kept mean-wr {np.nanmean(selfwr[viable]):.3f}")
print(f"importance weights {imp[viable].min():.3f}..{imp[viable].max():.3f}")

rows = []
for cand, (w, dec, g) in SELF.items():
    if dec < 800 or cand in loopy or cand >= N:
        continue
    num = den = 0.0
    for b, cw, cn in byc.get(cand, ()):
        if cn and imp[b] > 0:
            num += imp[b] * cw; den += imp[b] * cn
    if den < 500:
        continue
    ev_u = num / den
    ev_k = cov = 0.0
    cells = []
    for k, share in enumerate(W):
        cw = cn = 0.0
        raw = 0
        for j, sw in NBR[k]:
            if j == cand:
                continue
            c = PAIR.get((cand, j))
            if c:
                cw += sw * c[0]; cn += sw * c[1]; raw += c[1]
        if raw >= MINC and cn > 0:
            r = cw / cn
            ev_k += share * r; cov += share
            cells.append((r, share))
    if cov < 0.85:
        continue
    ev_k /= cov
    cells.sort()
    acc = 0.0
    tail_num = tail_den = 0.0
    for r, s in cells:                       # worst decile of covered meta mass
        take = min(s, 0.10 * cov - acc)
        if take <= 0:
            break
        tail_num += take * r; tail_den += take; acc += take
    rows.append({"i": cand, "k": ev_k, "u": ev_u, "b": 0.5 * ev_k + 0.5 * ev_u,
                 "tail": tail_num / max(tail_den, 1e-9), "cov": cov})
rows.sort(key=lambda r: -r[SORTK])
print(f"\ndecks scored: {len(rows)}   BLEND = 0.5*kaggle3d + 0.5*uniform-over-decks")
print(f"\n{'#':>3} {'deck':>7} {'BLEND':>6} {'kaggle':>7} {'unif':>6} {'spread':>7} {'tail10':>7}  cards")
for n, r in enumerate(rows[:TOPN], 1):
    print(f"{n:>3} d{r['i']:<6} {r['b']:.4f} {r['k']:7.4f} {r['u']:6.4f} {r['k']-r['u']:+7.3f} "
          f"{r['tail']:7.3f}  {label(r['i'])}")

rank = {r["i"]: n + 1 for n, r in enumerate(rows)}
by = {r["i"]: r for r in rows}
print("\nWATCHLIST (blend | kaggle | uniform | spread | worst-decile):")
for i in (1836, 3161, 1444, 2195, 149, 4043, 1145, 4050, 697, 2092, 4757, 817, 268):
    r = by.get(i)
    if r:
        print(f"  d{i:<5} rank {rank[i]:>4}/{len(rows)}  {r['b']:.4f} | {r['k']:.4f} | "
              f"{r['u']:.4f} | {r['k']-r['u']:+.3f} | {r['tail']:.3f}")
    else:
        print(f"  d{i:<5} not scoreable")
