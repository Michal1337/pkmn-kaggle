"""Direct checkpoint-vs-checkpoint head-to-head, lockstep-batched (rl/gates_fast.py mirror mode).

Answers the question the clone-anchored field eval cannot: did checkpoint A actually learn
anything over checkpoint B? Both nets pilot the SAME deck each game (mirror), greedy argmax,
seat balanced by fixed-seed flip -- so there is no frozen-opponent EV ceiling, no meta-mix
weighting, and no deck-pair dilution. If A's h2h edge over B is zero, A learned nothing B
did not already know; if it is large while the field series was flat, the field instrument
was compressing real improvement.

wr is reported from A's side. Truncations (turn-cap) are excluded from the denominator.

  PYTHONPATH=. PKMN_TURN_CAP=200 python scripts/h2h_ckpt.py CKPT_A CKPT_B GAMES \\
      [--pool-json P] [--pool-weights W] [--sample N] [--seed 777] [--rank 0 --world 1]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

torch.set_grad_enabled(False)

from rl.card_features import get_card_table
from rl.policy import build_token_net
from rl.encoding import TokenEncoder
from rl.gates import field_winrate_fast

ap = argparse.ArgumentParser()
ap.add_argument("ckpt_a"); ap.add_argument("ckpt_b"); ap.add_argument("games", type=int)
ap.add_argument("--pool-json", default="data/census_elite_v1.json")
ap.add_argument("--pool-weights", default="data/census_elite_v1_w.npy")
ap.add_argument("--sample", type=int, default=1553)
ap.add_argument("--seed", type=int, default=777)
ap.add_argument("--rank", type=int, default=0)
ap.add_argument("--world", type=int, default=1)
ap.add_argument("--live", type=int, default=128)
ap.add_argument("--progress-every", type=int, default=250)
ap.add_argument("--out", default="")
ap.add_argument("--tag", default="")
a = ap.parse_args()

ct = get_card_table()
enc = TokenEncoder(ct)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if dev.type != "cuda":
    print("WARNING: no CUDA -- ~50x slower", flush=True)


def _load(p):
    ck = torch.load(p, map_location="cpu")
    net = build_token_net(ct, ck["net_config"]); net.load_state_dict(ck["net"])
    net.eval().to(dev)
    return net, int(ck.get("global_step", -1))


net_a, step_a = _load(a.ckpt_a)
net_b, step_b = _load(a.ckpt_b)

pj = json.load(open(a.pool_json))
w = np.load(a.pool_weights); w = w / w.sum()
rng = np.random.default_rng(a.seed)
idx = rng.choice(len(pj["decks"]), size=a.sample, replace=False, p=w)
pool = [pj["decks"][i] for i in idx]

_t0 = time.time()


def _prog(fin, tot, wins, decided, draws):
    el = time.time() - _t0
    print(f"[prog] {a.tag or 'h2h'} r{a.rank}/{a.world} {fin}/{tot} ({fin/tot:.0%}) "
          f"{fin/max(el,1e-6):.1f} games/s  wrA={wins/max(decided,1):.4f} draws={draws}", flush=True)


fi = field_winrate_fast(net_a, net_b, pool, pool, [1.0], a.games, enc, dev,
                        seed=4321, rank=a.rank, world=a.world, live_n=a.live,
                        progress_every=a.progress_every, progress_cb=_prog, mirror=True)
wr = fi["wins"] / max(fi["decided"], 1)
print(f"H2H a={a.ckpt_a} (step {step_a}) vs b={a.ckpt_b} (step {step_b}) games={a.games} "
      f"sample={a.sample} rank={a.rank}/{a.world} turn_cap={os.environ.get('PKMN_TURN_CAP','0')} "
      f"wrA={wr:.4f} ({fi['wins']}/{fi['decided']} draws={fi['draws']} errs={fi['errs']})", flush=True)
if fi["errs"] > 0.02 * max(fi["n"], 1):
    sys.exit(f"ABORT: {fi['errs']}/{fi['n']} games errored")

if a.out:
    rec = {"tag": a.tag, "a": a.ckpt_a, "b": a.ckpt_b, "step_a": step_a, "step_b": step_b,
           "games": a.games, "sample": a.sample, "seed": a.seed, "rank": a.rank, "world": a.world,
           "turn_cap": int(os.environ.get("PKMN_TURN_CAP", "0") or "0"),
           "wins": fi["wins"], "decided": fi["decided"], "draws": fi["draws"]}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"{a.tag}_r{a.rank}of{a.world}.json"), "w") as fh:
        json.dump(rec, fh)
