"""Offline ft-gate audit: a finetune checkpoint pilots the FT DECK against the frozen base
checkpoint piloting the mined opponent pool (rl/gates_fast.py, non-mirror).

Why this exists: the inline ft_wr of the two ft arms is NOT comparable -- single-net's
opponent is the frozen base, two-net's is the live adapting B-net, so two-net's inline
number is depressed by construction. This driver measures every arm against the SAME
frozen opponent on the SAME deck mix, which is the number that predicts ladder EV.

  PYTHONPATH=. PKMN_TURN_CAP=200 python scripts/ft_gate_ckpt.py CKPT BASE GAMES \
      [--deck-csv D] [--opp-json P] [--opp-weights counts|uniform] [--rank 0 --world 1]

wr is reported with truncations dropped (historical convention) AND scored as losses
(ladder-truthful) -- a divergence between the two means the deck is stalling.
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch

torch.set_grad_enabled(False)

from rl.card_features import get_card_table
from rl.policy import build_token_net
from rl.encoding import TokenEncoder
from rl.gates import field_winrate_fast

G = "data"
ap = argparse.ArgumentParser()
ap.add_argument("ckpt"); ap.add_argument("base"); ap.add_argument("games", type=int)
ap.add_argument("--deck-csv", default=f"{G}/deck_d1145.csv")
ap.add_argument("--opp-json", default=f"{G}/ft_opp_d1145.json")
ap.add_argument("--opp-weights", choices=["counts", "uniform"], default="counts")
ap.add_argument("--base-key", default="net",
                help="state-dict key to load from BASE (net_b = a two-net ckpt's opponent net)")
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


_ADDITIVE_ZERO_INIT_KEYS = ("decktop_emb",)   # v2.3 features that are exact no-ops at zero


def _load(p, key="net"):
    ck = torch.load(p, map_location="cpu")
    sd = ck.get(key)
    if sd is None:
        raise SystemExit(f"{p} has no '{key}' state dict (keys: {[k for k in ck if 'net' in k]})")
    net = build_token_net(ct, ck["net_config"])
    # v2.2 ckpts (e.g. the crown) predate decktop_emb; it is zero-initialised and enters the
    # card stream multiplicatively, so an absent key leaves the net numerically identical to
    # what it was when trained. Any OTHER mismatch is a real arch error and still raises.
    missing, unexpected = net.load_state_dict(sd, strict=False)
    bad = [k for k in list(missing) + list(unexpected) if k not in _ADDITIVE_ZERO_INIT_KEYS]
    if bad:
        raise SystemExit(f"{p}: incompatible state dict (missing/unexpected: {bad})")
    if missing:
        print(f"[compat] {os.path.basename(p)}: {list(missing)} absent -> zero (v2.2 ckpt)", flush=True)
    net.eval().to(dev)
    return net, int(ck.get("global_step", -1))


net, step = _load(a.ckpt)
base, bstep = _load(a.base, a.base_key)

deck = [int(r[0]) for r in csv.reader(open(a.deck_csv)) if r and r[0].strip()]
assert len(deck) == 60, f"deck csv has {len(deck)} cards"
pj = json.load(open(a.opp_json))
opp = [[int(x) for x in d] for d in pj["decks"]]
if a.opp_weights == "counts" and "counts" in pj:
    w = np.array([(sum(c) if isinstance(c, list) else float(c)) for c in pj["counts"]], float)
else:
    w = np.ones(len(opp), float)
w = w / w.sum()

_t0 = time.time()


def _prog(fin, tot, wins, decided, draws):
    el = time.time() - _t0
    print(f"[prog] {a.tag or 'ftgate'} r{a.rank}/{a.world} {fin}/{tot} ({fin/tot:.0%}) "
          f"{fin/max(el,1e-6):.1f} games/s  wr={wins/max(decided,1):.4f} draws={draws}", flush=True)


fi = field_winrate_fast(net, base, [deck], opp, list(w), a.games, enc, dev,
                        seed=4321, rank=a.rank, world=a.world, live_n=a.live,
                        progress_every=a.progress_every, progress_cb=_prog, mirror=False)
wr = fi["wins"] / max(fi["decided"], 1)
wrl = fi["wins"] / max(fi["decided"] + fi["draws"], 1)      # truncations scored as losses
print(f"FTGATE ckpt={a.ckpt} (step {step}) base={a.base} (step {bstep}) games={a.games} "
      f"opp={len(opp)}@{a.opp_weights} turn_cap={os.environ.get('PKMN_TURN_CAP','0')} "
      f"wr={wr:.4f} wr_loss={wrl:.4f} ({fi['wins']}/{fi['decided']} draws={fi['draws']} "
      f"errs={fi['errs']})", flush=True)

if a.out:
    rec = {"tag": a.tag, "ckpt": a.ckpt, "base": a.base, "step": step, "games": a.games,
           "opp_weights": a.opp_weights, "seed": a.seed, "rank": a.rank, "world": a.world,
           "turn_cap": int(os.environ.get("PKMN_TURN_CAP", "0") or "0"),
           "wins": fi["wins"], "decided": fi["decided"], "draws": fi["draws"]}
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, f"{a.tag}_r{a.rank}of{a.world}.json"), "w") as fh:
        json.dump(rec, fh)
