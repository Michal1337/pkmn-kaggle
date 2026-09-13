"""v2.2 precise opponent-hand eviction -- behavioral guards.

Tracker semantics (rl/encoding.py GameTracker, engine-verified against ApiJson.h/CardMove.h):
  * NO TTL: a known-in-hand belief persists across turns until an observed exit names its serial
    (type-10 Play / public type-6 move / visible-zone ground truth) or the public hand size
    contradicts the belief count (``_enforce_hand_count``).
  * A blind (identity-redacted) type-7 hand-exit is COUNTED, not acted on: beliefs survive but
    lose their CERTAINTY (``hand_beliefs_for`` flag / ``opp_hand_flag``); a belief stamped after
    the exit is certain again.
  * Forced eviction (count contradiction) demotes uncertain-then-oldest first.

Encoder (v2.2): ``opp_hand_flag[i]`` = 1 iff slot i is a KNOWN belief with no blind exit observed
since its stamp; 0 on uncertain-known, UNK fill, and padding. The net adds a zero-init
``hand_certain_emb`` on the opp_hand stream only.

Run:  python -m pytest tests/test_v22_hand.py -v
"""
import random

import numpy as np
import torch

from rl.encoding import GameTracker, TokenEncoder, MAX_HAND
from rl.card_features import get_card_table
from rl.policy import build_token_net, obs_to_tensors


# ---------------------------------------------------------------- tracker (synthetic obs) ----

def _obs(turn, me=0, logs=(), hand0=6, hand1=6):
    """Minimal decision obs: only the fields GameTracker.update reads."""
    return {
        "current": {
            "yourIndex": me, "turn": turn, "stadium": [],
            "players": [
                {"handCount": hand0, "discard": [], "active": [], "bench": []},
                {"handCount": hand1, "discard": [], "active": [], "bench": []},
            ],
        },
        "logs": list(logs),
    }


def _reveal(cid, ser, pi=1):
    """Public deck->hand move (e.g. a search): creates a known-in-hand belief."""
    return {"type": 6, "playerIndex": pi, "cardId": cid, "serial": ser,
            "fromArea": 1, "toArea": 2}


def _blind_exit(pi=1):
    """Identity-redacted hand-exit (type-7 MoveCardReverse, fromArea==HAND)."""
    return {"type": 7, "playerIndex": pi, "fromArea": 2, "toArea": 1}


def _play(cid, ser, pi=1):
    return {"type": 10, "playerIndex": pi, "cardId": cid, "serial": ser}


def test_belief_persists_across_turns():
    """The old 1-turn TTL is gone: with no exit evidence a belief lives for many turns."""
    tr = GameTracker()
    tr.update(_obs(1, logs=[_reveal(101, 9001), _reveal(102, 9002)]))
    assert tr.hand_ids_for(1) == [101, 102]
    for t in (3, 5, 9, 15):
        tr.update(_obs(t))
    assert tr.hand_ids_for(1) == [101, 102], "belief must survive turns without exit evidence"
    assert [f for _, f in tr.hand_beliefs_for(1)] == [1.0, 1.0]


def test_play_evicts_exact_serial():
    tr = GameTracker()
    tr.update(_obs(1, logs=[_reveal(101, 9001), _reveal(102, 9002)]))
    tr.update(_obs(2, logs=[_play(101, 9001)]))
    assert tr.hand_ids_for(1) == [102]


def test_public_move_evicts_exact_serial():
    tr = GameTracker()
    tr.update(_obs(1, logs=[_reveal(101, 9001), _reveal(102, 9002)]))
    # public hand->deck (e.g. Iono-class shuffle: type-6 WITH serial)
    tr.update(_obs(2, logs=[{"type": 6, "playerIndex": 1, "cardId": 101, "serial": 9001,
                             "fromArea": 2, "toArea": 1}]))
    assert tr.hand_ids_for(1) == [102]


def test_blind_exit_counts_not_wipes():
    tr = GameTracker()
    tr.update(_obs(1, logs=[_reveal(101, 9001), _reveal(102, 9002)]))
    tr.update(_obs(3, logs=[_blind_exit()]))
    # beliefs KEPT (hand still holds >= 2), certainty gone
    assert tr.hand_ids_for(1) == [101, 102]
    assert [f for _, f in tr.hand_beliefs_for(1)] == [0.0, 0.0]
    assert tr.blind_exits[1] == 1
    # a belief stamped AFTER the exit is certain again (and sorts newest-first)
    tr.update(_obs(5, logs=[_reveal(103, 9003)]))
    assert tr.hand_beliefs_for(1) == [(103, 1.0), (101, 0.0), (102, 0.0)]


def test_hand_count_invariant_forces_oldest_out():
    tr = GameTracker()
    tr.update(_obs(1, logs=[_reveal(101, 9001)]))
    tr.update(_obs(3, logs=[_reveal(102, 9002)]))
    assert tr.hand_ids_for(1) == [102, 101]              # newest first
    tr.update(_obs(5, hand1=1))                          # K=2 > H=1 -> evict the OLDEST
    assert tr.hand_ids_for(1) == [102]
    tr.update(_obs(7, hand1=0))                          # empty hand -> no beliefs can stand
    assert tr.hand_ids_for(1) == []


def test_uncertain_evicted_before_certain():
    tr = GameTracker()
    tr.update(_obs(1, logs=[_reveal(101, 9001)]))
    tr.update(_obs(3, logs=[_blind_exit(), _reveal(102, 9002)]))   # 101 uncertain, 102 certain
    tr.update(_obs(5, hand1=1))
    assert tr.hand_beliefs_for(1) == [(102, 1.0)]


def test_reset_clears_v22_state():
    tr = GameTracker()
    tr.update(_obs(1, logs=[_reveal(101, 9001), _blind_exit()]))
    tr.reset()
    assert tr.blind_exits == {0: 0, 1: 0}
    assert tr.hand_epoch == {0: {}, 1: {}}
    assert tr.hand_beliefs_for(1) == []


# ---------------------------------------------------------------- encoder + net wiring ----

def _first_real_obs(seed=0):
    """One real seat-0 decision obs from the engine (for a structurally-valid encode input)."""
    from kaggle_environments.envs.cabt.cg import game
    from kaggle_environments.envs.cabt.cg.sim import Battle
    from rl.decks_train import TRAIN_TOP50

    deck = list(TRAIN_TOP50.values())[13]
    rng = random.Random(seed)
    if Battle.battle_ptr:
        try:
            game.battle_finish()
        except Exception:
            pass
        Battle.battle_ptr = None
    o, _ = game.battle_start(deck, deck)
    for _ in range(60):
        s = o.get("current") or {}
        if s.get("result", -1) >= 0:
            break
        if s.get("yourIndex", 0) == 0 and (s.get("players") or [{}, {}])[1].get("handCount", 0) >= 3:
            game.battle_finish(); Battle.battle_ptr = None
            return o, deck
        sel = o.get("select") or {}
        opt = sel.get("option") or []
        k = sel.get("maxCount", 1) or 1
        picks = rng.sample(range(len(opt)), min(k, len(opt))) if opt else []
        o = game.battle_select(sorted(set(picks)))
    game.battle_finish(); Battle.battle_ptr = None
    raise AssertionError("no seat-0 obs with opp handCount >= 3 collected")


def _tracker_with_beliefs(raw, deck):
    """Tracker primed with two injected opp beliefs: an OLD uncertain one and a NEW certain one."""
    tr = GameTracker()
    tr.update(raw)
    cid_a, cid_b = deck[0], deck[1]
    tr._cur_turn = 1
    tr._add(1, cid_a, 990001); tr._set_zone(1, 990001, 2)   # epoch 0
    tr.blind_exits[1] += 1                                  # blind exit -> A uncertain
    tr._cur_turn = 2
    tr._add(1, cid_b, 990002); tr._set_zone(1, 990002, 2)   # epoch 1 == current -> certain
    assert tr.hand_beliefs_for(1) == [(cid_b, 1.0), (cid_a, 0.0)]
    return tr, cid_a, cid_b


def test_opp_hand_flag_encoding():
    ct = get_card_table()
    enc = TokenEncoder(ct)
    assert enc.shapes["opp_hand_flag"] == (MAX_HAND,)
    raw, deck = _first_real_obs()
    tr, cid_a, cid_b = _tracker_with_beliefs(raw, deck)
    e = enc.encode(raw, set(), self_deck=deck, tracker=tr)

    hc = int(raw["current"]["players"][1]["handCount"])
    n_present = min(hc, MAX_HAND)
    ids, m, fl = e["opp_hand_id"], e["opp_hand_mask"], e["opp_hand_flag"]
    assert fl.shape == (MAX_HAND,) and fl.dtype == np.float32
    # newest-first: certain belief in slot 0, uncertain in slot 1, then UNK fill to handCount
    assert int(ids[0]) == cid_b and fl[0] == 1.0
    assert int(ids[1]) == cid_a and fl[1] == 0.0
    assert (ids[2:n_present] == enc.UNK).all() and (fl[2:] == 0.0).all()
    assert (m[:n_present] == 1.0).all() and (m[n_present:] == 0.0).all()


def test_flag_wiring_in_net():
    """The flag must reach exactly the opp_hand stream: with a NON-zero certainty embedding,
    flipping a present slot's flag changes the outputs; corrupting a padded slot's flag cannot."""
    torch.manual_seed(0)
    ct = get_card_table()
    enc = TokenEncoder(ct)
    raw, deck = _first_real_obs()
    tr, _, _ = _tracker_with_beliefs(raw, deck)
    e = enc.encode(raw, set(), self_deck=deck, tracker=tr)

    net = build_token_net(ct, {"d_model": 64, "nhead": 4, "nlayers": 2})
    net.eval()
    with torch.no_grad():
        net.hand_certain_emb.normal_(std=0.5)             # zero-init would make this vacuous
    b = {k: v.unsqueeze(0) for k, v in obs_to_tensors(e, "cpu").items()}
    with torch.no_grad():
        l0, v0 = net.logits_value(b)
    legal = b["action_mask"][0] > 0.5

    c = {k: v.clone() for k, v in b.items()}
    c["opp_hand_flag"][0, 0] = 0.0                        # certain -> uncertain on a PRESENT slot
    with torch.no_grad():
        l1, v1 = net.logits_value(c)
    assert (l1[0][legal] - l0[0][legal]).abs().max().item() > 0.0 or \
           (v1 - v0).abs().max().item() > 0.0, "opp_hand_flag is invisible to the net"

    pad = np.flatnonzero(e["opp_hand_mask"] < 0.5)
    if pad.size:
        c2 = {k: v.clone() for k, v in b.items()}
        c2["opp_hand_flag"][0, int(pad[0])] = 1.0         # flag on a PADDED slot: must be inert
        with torch.no_grad():
            l2, v2 = net.logits_value(c2)
        assert (l2[0][legal] - l0[0][legal]).abs().max().item() == 0.0
        assert (v2 - v0).abs().max().item() == 0.0


if __name__ == "__main__":
    test_belief_persists_across_turns()
    test_play_evicts_exact_serial()
    test_public_move_evicts_exact_serial()
    test_blind_exit_counts_not_wipes()
    test_hand_count_invariant_forces_oldest_out()
    test_uncertain_evicted_before_certain()
    test_reset_clears_v22_state()
    test_opp_hand_flag_encoding()
    test_flag_wiring_in_net()
    print("v2.2 hand-eviction tests: PASSED")
