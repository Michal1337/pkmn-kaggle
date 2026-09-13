"""ENGINE-validate pool decks before training: every deck must survive a real battle_start
(catches 4-copy violations from mapping collisions, missing basics, any illegality -- one bad
deck in the uniform pool is a mid-run crash lottery). Validates decks whose band counts are all
zero (the curated/harvested ones; ladder decks are engine-proven by definition) unless --all.

  PYTHONPATH=. python scripts/validate_pool_decks.py POOL.json [--all]
"""
import argparse
import json
import sys

sys.path.insert(0, ".")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pool")
    p.add_argument("--all", action="store_true")
    a = p.parse_args()

    d = json.load(open(a.pool))
    decks = [list(x) for x in d["decks"]]
    counts = d["counts"]
    idx = [i for i in range(len(decks)) if a.all or sum(counts[i]) == 0]
    print(f"[validate] {len(idx)} decks to engine-check (of {len(decks)})", flush=True)

    from rl.env_selfplay import TwoSidedSelfPlayEnv
    ref = decks[0]                                    # engine-proven ladder deck as the opponent
    bad = []
    env = None
    for n, i in enumerate(idx):
        try:
            env = TwoSidedSelfPlayEnv(decks=[decks[i], ref], seed=123)
            # force the candidate on seat0 and the ref on seat1 via the sample hook
            env._sample_decks = lambda dk=decks[i]: ((dk, ref), setattr(env, "_deck_idx", (0, 1)))[0]
            env.reset()
            env.close()
        except Exception as e:
            bad.append((i, str(e)[:120]))
            try:
                env.close()
            except Exception:
                pass
        if (n + 1) % 50 == 0:
            print(f"  {n + 1}/{len(idx)} checked, {len(bad)} bad", flush=True)
    if bad:
        print(f"[validate] {len(bad)} BAD decks -> writing bad_decks.json", flush=True)
        for i, e in bad[:10]:
            print(f"    idx {i}: {e}", flush=True)
        json.dump([i for i, _ in bad], open("bad_decks.json", "w"))
        sys.exit(1)
    print("[validate] ALL DECKS ENGINE-OK", flush=True)


if __name__ == "__main__":
    main()
