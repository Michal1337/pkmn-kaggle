"""Matchup-matrix logging: (1) the index-based deck sampling consumes the SAME rng stream as the
old choice()-based code (running chains resume byte-identically); (2) terminal infos carry the
deck ids + the collect-side outcome coding matches the terminal rewards."""
import random

import numpy as np

from rl.decks import DECKS
from rl.env_selfplay import TwoSidedSelfPlayEnv


def test_rng_stream_compat():
    # base env: choice(seq) == seq[randrange(len(seq))] on the same Random state
    seq = [[i] * 60 for i in range(37)]
    r1, r2 = random.Random(123), random.Random(123)
    for _ in range(200):
        a = r1.choice(seq)
        i = r2.randrange(len(seq))
        assert seq[i] is a
    # finetune env: choices(seq, w) == choices(range(n), w) index-wise
    w = [x + 1.0 for x in range(37)]
    r1, r2 = random.Random(9), random.Random(9)
    for _ in range(200):
        a = r1.choices(seq, weights=w, k=1)[0]
        i = r2.choices(range(len(seq)), weights=w, k=1)[0]
        assert seq[i] is a


def test_weighted_deck_sampling():
    # zero-weight decks are never drawn; indices recorded for the matchup log
    env = TwoSidedSelfPlayEnv(decks=list(DECKS.values())[:3], seed=5,
                              deck_weights=[0.0, 0.0, 1.0])
    for _ in range(50):
        d0, d1 = env._sample_decks()
        assert env._deck_idx == (2, 2)
        assert d0 is env.decks[2] and d1 is env.decks[2]
    # default (None) keeps the uniform randrange path
    env2 = TwoSidedSelfPlayEnv(decks=list(DECKS.values())[:3], seed=5)
    seen = {env2._sample_decks() and env2._deck_idx for _ in range(60)}
    assert len({i for p in seen for i in p}) == 3


def _outcome(term):
    if term is None:
        return 3
    if term[0] > term[1]:
        return 0
    if term[1] > term[0]:
        return 1
    return 2


def test_terminal_info_carries_matchup():
    env = TwoSidedSelfPlayEnv(decks=list(DECKS.values())[:3], seed=11)
    events = 0
    for _ in range(4):
        enc, _seat, _info = env.reset()
        idx = env._deck_idx
        assert all(0 <= x < 3 for x in idx)
        assert env.deck[0] == list(env.decks[idx[0]]) and env.deck[1] == list(env.decks[idx[1]])
        rng = random.Random(2)
        done, info, steps = False, {}, 0
        while not done and steps < 3000:
            legal = np.flatnonzero(np.asarray(enc["action_mask"]) > 0.5)
            enc, _seat, _r, done, info = env.step(int(rng.choice(legal)))
            steps += 1
        if done and "mm" in info:
            events += 1
            assert tuple(info["mm"]) == idx
            out = _outcome(info.get("terminal"))
            assert out in (0, 1, 2, 3)
            if "terminal" in info:
                t = info["terminal"]
                if out == 0:
                    assert t[0] > t[1]
                elif out == 1:
                    assert t[1] > t[0]
    env.close()
    assert events > 0, "no completed episodes produced matchup events"
