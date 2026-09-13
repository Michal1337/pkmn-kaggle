"""Canonical v9+ pool-weight rebuild (runs on the cluster; GPFS paths below).

Pool layout (append-only; drops are weight-0 zeroing, indices NEVER move):
  idx < 2253            v8 Kaggle-lineage pool (episode-mined + scout + curated)
  2253..4059            majors lists (harvest_majors.py, distinct decks)
  4060..4113            SALVAGED hand-mapped top-16 majors (apply_handmap.py)
  4114..4727            grassroots top-8 x>=2 appearance adds (from limitless_full)

Tiers (user spec 2026-07-14):
  KAGGLE 75%     = idx<2253 or in mined_jul14.json (freshly episode-mined)
  HUMAN-META 25% = majors with BEST placing <=128 across ALL events
                   (majors_keep_idx.json, built by scripts/majors_provenance.py --
                    the original harvest kept only first-seen provenance, which
                    mislabeled netdecked winners) + grassroots adds (grass8_idx.json)

Kernel: plain tf-idf cosine density (user: "1 over tf-idf", no squaring),
w = 1/mean-similarity, clip [0.2/N, 8/N], tier masses pinned exactly.
"""
import json

import numpy as np

GRP = "data"
v9 = json.load(open(f"{GRP}/ladder_decks_v9.json"))
mined = json.load(open(f"{GRP}/mined_jul14.json"))
decks = v9["decks"]
N = len(decks)
assert len(v9["counts"]) == N, "counts/decks desync -- decks_ladder.load() would crash"

mined_set = {tuple(sorted(d)) for d in mined["decks"]}
incumbent = np.array([(i < 2253) or (tuple(sorted(d)) in mined_set) for i, d in enumerate(decks)])

grass8 = {int(k) for k in json.load(open(f"{GRP}/grass8_idx.json"))}
keep_idx = set(json.load(open(f"{GRP}/majors_keep_idx.json"))) | grass8
human = np.array([(i in keep_idx) and not incumbent[i] for i in range(N)])
print(f"human-meta tier: {int(human.sum())} "
      f"(majors {len(keep_idx) - len(grass8)} + grassroots {len(grass8)})")

vocab = sorted({c for d in decks for c in d})
col = {c: i for i, c in enumerate(vocab)}
M = np.zeros((N, len(vocab)))
for i, d in enumerate(decks):
    for c in d:
        M[i, col[c]] += 1.0
idf = np.log(N / (M > 0).sum(axis=0))
M *= idf
M /= np.linalg.norm(M, axis=1, keepdims=True)
S = M @ M.T
np.fill_diagonal(S, 0.0)
w = 1.0 / S.mean(axis=1)
w[~incumbent & ~human] = 0.0
w /= w.sum()
live = w > 0
w[live] = np.clip(w[live], 0.2 / N, 8.0 / N)
for mask, t in ((incumbent, 0.75), (human, 0.25)):
    w[mask] *= t / w[mask].sum()
assert abs(w.sum() - 1.0) < 1e-9
np.save(f"{GRP}/pool_v9_weights.npy", w)
u = 1.0 / N
a, b = w[incumbent], w[human]
print(f"kaggle:     n={a.size} mass={a.sum():.3f} med={np.median(a)/u:.2f}x")
print(f"human-meta: n={b.size} mass={b.sum():.3f} med={np.median(b)/u:.2f}x "
      f"games/deck/1B={np.median(b)*14e6:.0f}")
print("WROTE pool_v9_weights.npy")
