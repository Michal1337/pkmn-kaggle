"""Mine ALL distinct decks from the daily episode zips -> ladder_decks.json for rl/decks_ladder.

Per exact 60-card list: play counts per rating band (episode avg_score from each zip's
manifest.csv; bands <1150 / 1150-1250 / 1250+). EXACT-list dedup only (variants are deck-space
augmentation -- never collapse near-duplicates). Long tail pruned at --min-games total.

  python scripts/mine_ladder_decks.py OUT.json ZIP1 [ZIP2 ...] [--min-games 10] [--workers 16]
"""
import argparse
import collections
import csv
import io
import json
import zipfile
from multiprocessing import Pool


def band(score):
    return 0 if score < 1150 else (1 if score < 1250 else 2)


def _job(arg):
    zp, pairs, recent = arg
    out = []
    try:
        z = zipfile.ZipFile(zp)
    except Exception:
        return out
    for nm, sc in pairs:
        try:
            d = json.loads(z.read(nm))
        except Exception:
            continue
        decks = {}
        for step in d.get("steps", []):
            for pi, ag in enumerate(step):
                a = ag.get("action")
                if isinstance(a, list) and len(a) == 60 and all(isinstance(x, int) for x in a) \
                        and pi not in decks:
                    decks[pi] = tuple(sorted(a))   # ORDER-CANONICAL: a deck shuffles in play, so
                    #  the same 60 cards in a different logged order is the SAME deck. v1 keyed the
                    #  raw ordered tuple -> ~1.28x order-inflation AND split each deck's plays across
                    #  orders so fragmented decks got pruned below --min-games; sorting merges the
                    #  counts BEFORE the prune (recovers real decks + true uniform distribution).
            if len(decks) == 2:
                break
        for dk in decks.values():
            out.append((dk, band(sc), recent))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out")
    p.add_argument("zips", nargs="+")
    p.add_argument("--min-games", type=int, default=10)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--union-pools", type=str, default=None,
                   help="comma resolve_deck_pool names (e.g. 'top50,good') to UNION in after mining: "
                        "curated decks (aichi/official/Limitless/meta2) genuinely absent from the "
                        "ladder get added with ZERO band counts = uniform-pool only, no meta weight "
                        "(shift-insurance). Order-canonicalized, so already-present decks are skipped.")
    # EVIDENCE-UNION keep rules (2026-07-09): raw volume conflates rare-because-bad with
    # rare-because-NEW and rare-because-ELITE (the kyurem/slowking blind spot: a 2-game summit
    # lock deck dethroned our #1). Keep a deck if established (min-games) OR elite-endorsed
    # (summit-band games) OR emerging (games in the last N day-zips).
    p.add_argument("--keep-summit", type=int, default=0,
                   help="also keep decks with >= this many summit-band (1250+) games (0 = off)")
    p.add_argument("--recent-days", type=int, default=0,
                   help="treat the last N zips (sorted by name = date) as 'recent' (0 = off)")
    p.add_argument("--recent-min", type=int, default=3,
                   help="also keep decks with >= this many games in the recent window")
    a = p.parse_args()

    zips_sorted = sorted(a.zips)
    recent_set = set(zips_sorted[-a.recent_days:]) if a.recent_days > 0 else set()
    tasks = []
    for zp in a.zips:
        z = zipfile.ZipFile(zp)
        jn = {n.split("/")[-1][:-5]: n for n in z.namelist() if n.endswith(".json")}
        rows = list(csv.DictReader(io.StringIO(z.read("manifest.csv").decode())))
        pairs = [(jn[r["episode_id"]], float(r["avg_score"])) for r in rows if r["episode_id"] in jn]
        for i in range(0, len(pairs), 400):
            tasks.append((zp, pairs[i:i + 400], zp in recent_set))
    print(f"[mine] {len(a.zips)} zips ({len(recent_set)} recent), "
          f"{sum(len(t[1]) for t in tasks)} episodes", flush=True)

    counts = collections.defaultdict(lambda: [0, 0, 0])
    recent_counts = collections.defaultdict(int)
    with Pool(a.workers) as pool:
        for out in pool.imap_unordered(_job, tasks):
            for dk, b, rec in out:
                counts[dk][b] += 1
                if rec:
                    recent_counts[dk] += 1

    total_lists = len(counts)
    kept = {dk: c for dk, c in counts.items() if sum(c) >= a.min_games}
    n_est = len(kept)
    n_sum = n_rec = 0
    for dk, c in counts.items():
        if dk in kept:
            continue
        if a.keep_summit and c[2] >= a.keep_summit:
            kept[dk] = c; n_sum += 1
        elif recent_set and recent_counts.get(dk, 0) >= a.recent_min:
            kept[dk] = c; n_rec += 1
    if a.keep_summit or recent_set:
        print(f"[keep-rules] established(>={a.min_games}g)={n_est} "
              f"+ summit(>={a.keep_summit})={n_sum} + recent({a.recent_days}d>={a.recent_min}g)={n_rec}",
              flush=True)
    n_mined = len(kept)

    added = 0
    if a.union_pools:
        import sys
        sys.path.insert(0, ".")
        from rl.train_selfplay import resolve_deck_pool
        for nm in a.union_pools.split(","):
            for d in resolve_deck_pool(nm.strip()):
                t = tuple(sorted(d))              # canonical form -> skip if the ladder already has it
                if t not in kept:
                    kept[t] = [0, 0, 0]           # zero counts = uniform-pool only, no meta weight
                    added += 1
        print(f"[union] +{added} curated decks from '{a.union_pools}' not on the ladder "
              f"(zero-count = shift-insurance, uniform-only)", flush=True)

    decks = sorted(kept, key=lambda dk: -sum(kept[dk]))   # ladder decks first (by plays), curated last
    payload = {"version": "mine_ladder_decks-v3-evidence-union", "min_games": a.min_games,
               "keep_summit": a.keep_summit, "recent_days": a.recent_days, "recent_min": a.recent_min,
               "bands": ["climb<1150", "contested1150-1250", "summit1250+"],
               "n_source_lists": total_lists, "n_mined_kept": n_mined, "n_union_added": added,
               "union_pools": a.union_pools,
               "decks": [list(dk) for dk in decks],
               "counts": [kept[dk] for dk in decks]}
    json.dump(payload, open(a.out, "w"))
    tot = sum(sum(c) for c in kept.values())
    print(f"[mine] wrote {a.out}: {len(decks)} decks ({n_mined} mined + {added} curated; of "
          f"{total_lists} order-canonical lists, min_games={a.min_games}), {tot} deck-plays "
          f"(bands: {[sum(c[b] for c in kept.values()) for b in range(3)]})", flush=True)


if __name__ == "__main__":
    main()
