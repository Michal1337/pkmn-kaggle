# -*- coding: utf-8 -*-
"""One shard of a MCTS-vs-greedy mirror h2h: SAME net + SAME deck, A=mcts B=greedy, alternating seats.
Prints 'RES aw d bw' (A=mcts wins / draws / B=greedy wins).
  python mcts_h2h_shard.py CKPT DECK_NAME NGAMES N_SIMS N_DET"""
import sys, os, warnings, logging
warnings.filterwarnings("ignore"); logging.disable(logging.CRITICAL)
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from rl.decks_train import TRAIN_TOP15
from rl.encoding import TokenEncoder, GameTracker, AbilityTracker
from rl.policy import build_token_net
from rl.card_features import get_card_table
from rl import search_agent as SA
from kaggle_environments import make

CKPT = sys.argv[1]; DECK_NAME = sys.argv[2]; NGAMES = int(sys.argv[3])
NSIMS = int(sys.argv[4]) if len(sys.argv) > 4 else 40
NDET = int(sys.argv[5]) if len(sys.argv) > 5 else 2
torch.set_num_threads(1)
if DECK_NAME.startswith("csv:"):
    DECK = [int(x) for x in open(DECK_NAME[4:]) if x.strip()]
else:
    from rl.decks_train import TRAIN_TOP50
    DECK = TRAIN_TOP50.get(DECK_NAME) or TRAIN_TOP15[DECK_NAME]

ck = torch.load(CKPT, map_location="cpu"); cards = get_card_table()
enc = TokenEncoder(cards)
net = build_token_net(cards, ck["net_config"]); net.load_state_dict(ck["net"]); net.eval()
WK = bool(ck.get("net_config", {}).get("would_ko"))


def make_agent(mode):
    tr, ab = GameTracker(), AbilityTracker()
    def ai(obs):
        sel = obs.get("select")
        if sel is None:
            tr.reset(); ab.reset(); return [int(c) for c in DECK]
        tr.update(obs); ab.note_turn((obs.get("current") or {}).get("turn"))
        if mode == "mcts":
            pick = SA.mcts_select(obs, net, enc, DECK, tr, ab.slots, n_sims=NSIMS, n_det=NDET)
        else:
            if WK:
                SA.annotate_would_ko(obs, DECK, enc)
            pick = SA._net_greedy_select(obs, net, enc, DECK, tr, ab.slots)
        ab.record(sel, pick); return pick
    return ai


env = make("cabt", debug=False)
aw = d = bw = 0
for g in range(NGAMES):
    a_p0 = (g % 2 == 0)                              # balanced seats within the shard (NGAMES even)
    A, B = make_agent("mcts"), make_agent("greedy")
    env.reset(); env.run([A, B] if a_p0 else [B, A])
    r0, r1 = env.state[0]["reward"], env.state[1]["reward"]
    ra, rb = (r0, r1) if a_p0 else (r1, r0)
    if ra is None or rb is None:
        continue
    if ra == rb: d += 1
    elif ra > rb: aw += 1
    else: bw += 1
print(f"RES {aw} {d} {bw}", flush=True)
