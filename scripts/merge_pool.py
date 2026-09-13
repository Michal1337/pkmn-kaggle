"""Merge harvested curated decks (harvest_limitless.py output) into a ladder pool file.
Curated decks enter with ZERO band counts = uniform-only (shift-insurance, no meta weight),
order-canonical dedup against the existing pool.

  python scripts/merge_pool.py BASE_POOL.json HARVEST.json OUT.json
"""
import json
import sys

base = json.load(open(sys.argv[1]))
harv = json.load(open(sys.argv[2]))

have = {tuple(sorted(d)) for d in base["decks"]}
added, dup = 0, 0
for d in harv["decks"]:
    k = tuple(sorted(d))
    if k in have:
        dup += 1
        continue
    have.add(k)
    base["decks"].append(list(d))
    base["counts"].append([0, 0, 0])
    added += 1

base["version"] = base.get("version", "?") + "+limitless"
base["n_limitless_added"] = added
json.dump(base, open(sys.argv[3], "w"))
print(f"[merge] base {len(base['decks']) - added} + harvest {len(harv['decks'])} "
      f"({dup} already present) -> {sys.argv[3]}: {len(base['decks'])} decks (+{added} curated)")
