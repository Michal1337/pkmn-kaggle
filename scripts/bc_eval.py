"""Eval-only BC scorer: top-1/top-3 val accuracy of an existing checkpoint on ANY corpus dir,
using bc_train's exact val protocol (game-level tail split, per-batch tensors, would_ko zeroing).
Calibration anchor between corpora: e.g. score the old-corpus 5M net on the new corpus's val tail
so old and new runs' numbers become comparable.

  PYTHONPATH=. python scripts/bc_eval.py CKPT NPY_DIR [--val-frac 0.1] [--batch 1024]
"""
import argparse

import numpy as np
import torch

torch.set_grad_enabled(False)

from rl.card_features import get_card_table
from rl.encoding import TokenEncoder
from rl.enc_constants import OPT_WK
from rl.policy import build_token_net
from scripts.bc_train import read_rows

WK_LO, WK_HI = OPT_WK, OPT_WK + 3

p = argparse.ArgumentParser()
p.add_argument("ckpt")
p.add_argument("data", help="npy-dir corpus")
p.add_argument("--val-frac", type=float, default=0.1)
p.add_argument("--batch", type=int, default=1024)
p.add_argument("--zero-wouldko", action=argparse.BooleanOptionalAction, default=True)
a = p.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ct = get_card_table()
int_keys = set(TokenEncoder(ct).int_keys)

ck = torch.load(a.ckpt, map_location="cpu")
net = build_token_net(ct, ck["net_config"])
net.load_state_dict(ck["net"])
net.eval().to(dev)

import os
d = {f[:-4]: np.load(os.path.join(a.data, f), mmap_mode="r")
     for f in sorted(os.listdir(a.data)) if f.endswith(".npy")}
N = int(d["__labels__"].shape[0])
nval = max(1, int(N * a.val_frac))
v0 = N - nval
labels = read_rows(d["__labels__"], v0, N)
keys = [k for k in d if k not in ("__labels__", "__is_attack__", "__group__")]

use_bf16 = dev.type == "cuda"
c1 = c3 = tot = 0
# DECK-TOP SUBSET (2026-08-09, KS-BC): accuracy restricted to decisions where the tracker KNOWS
# the top of the deck (any nonzero self_deck_top). This is the targeted metric for the deck-top
# feature: overall top1 can rise while stacker play stays unlearned.
dt1 = dt3 = dtot = 0
_has_dt = "self_deck_top" in d
SLAB = 131072
for s in range(v0, N, SLAB):
    e = min(N, s + SLAB)
    arrs = {k: read_rows(d[k], s, e) for k in keys}
    for i in range(0, e - s, a.batch):
        ob = {k: torch.as_tensor(np.asarray(arrs[k][i:i + a.batch]),
                                 dtype=(torch.long if k in int_keys else torch.float32),
                                 device=dev) for k in keys}
        if a.zero_wouldko:
            ob["opt_attr"][..., WK_LO:WK_HI] = 0.0
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            lg = net.logits_value(ob)[0]
        lg = lg.float()
        yb = torch.as_tensor(labels[s - v0 + i:s - v0 + i + len(lg)], dtype=torch.long, device=dev)
        _t1 = lg.argmax(1) == yb
        _t3 = (lg.topk(3, 1).indices == yb[:, None]).any(1)
        c1 += int(_t1.sum()); c3 += int(_t3.sum()); tot += len(yb)
        if _has_dt:
            dtm = ob["self_deck_top"].amax(1) > 0
            dt1 += int(_t1[dtm].sum()); dt3 += int(_t3[dtm].sum()); dtot += int(dtm.sum())
print(f"BC_EVAL ckpt={a.ckpt} data={a.data} val_rows={tot} "
      f"top1={c1 / tot:.4f} top3={c3 / tot:.4f}"
      + (f" | KNOWN-TOP subset: rows={dtot} ({dtot / max(tot, 1):.1%}) "
         f"top1={dt1 / max(dtot, 1):.4f} top3={dt3 / max(dtot, 1):.4f}" if _has_dt else ""),
      flush=True)
