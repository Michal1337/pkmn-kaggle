"""FrozenOpponentEnv (deck-finetune collection): single-agent view invariants.

Checks, over several full episodes with a tiny frozen net:
  * only AGENT decisions surface (surfaced seat == env.agent_seat, constant within an episode)
  * the agent deck sits on env.agent_seat; the opponent deck comes from the weighted list
  * the agent seat RANDOMIZES across episodes (seat-confound cancellation)
  * episodes terminate with agent-perspective terminal rewards in [-1, 1] (margin off)
  * multi-pick buffering and opponent rolls never leak an opponent decision
"""
import os
import tempfile

import numpy as np
import pytest
import torch

from rl.card_features import get_card_table
from rl.decks_train import TRAIN_TOP15
from rl.env_selfplay import FrozenOpponentEnv
from rl.policy import build_token_net


@pytest.fixture(scope="module")
def frozen_ckpt():
    cfg = {"arch": "transformer2", "d_model": 32, "nhead": 2, "nlayers": 1, "ff": 32,
           "static": True, "structured": False, "split_heads": True}
    net = build_token_net(get_card_table(), cfg)
    fd, path = tempfile.mkstemp(suffix=".pt")
    os.close(fd)
    torch.save({"net": net.state_dict(), "net_config": cfg}, path)
    yield path
    os.unlink(path)


def test_asymmetric_deck_env():
    """AsymmetricDeckEnv (GPU-batched finetune collection): ALL decisions surface; info carries
    agent_seat; agent deck sits on agent_seat; both seats act within an episode."""
    from rl.env_selfplay import AsymmetricDeckEnv
    agent_deck = TRAIN_TOP15["k02_alakazam_non_ex"]
    opp_decks = [TRAIN_TOP15["k01_mega_lucario_ex"], TRAIN_TOP15["k05_fezandipiti_ex"]]
    env = AsymmetricDeckEnv(agent_deck, opp_decks, [0.5, 0.5], seed=1)
    rng = np.random.default_rng(1)
    agent_seats_seen = set()
    try:
        for _ep in range(6):
            obs, seat, info = env.reset()
            assert info["agent_seat"] == env.agent_seat
            agent_seats_seen.add(env.agent_seat)
            assert env.deck[env.agent_seat] == list(agent_deck)
            assert env.deck[1 - env.agent_seat] in [list(d) for d in opp_decks]
            seats_in_ep = {seat}
            done = False
            steps = 0
            while not done:
                legal = np.flatnonzero(np.asarray(obs["action_mask"]) > 0.5)
                obs, seat, r, done, info = env.step(int(rng.choice(legal)))
                assert info["agent_seat"] == env.agent_seat
                seats_in_ep.add(seat)
                steps += 1
                assert steps < 5000
            assert seats_in_ep == {0, 1}, "both seats must surface decisions"
        assert agent_seats_seen == {0, 1}, f"agent seat never randomized: {agent_seats_seen}"
    finally:
        env.close()


def test_frozen_opponent_env(frozen_ckpt):
    agent_deck = TRAIN_TOP15["k02_alakazam_non_ex"]
    opp_decks = [TRAIN_TOP15["k01_mega_lucario_ex"], TRAIN_TOP15["k05_fezandipiti_ex"]]
    env = FrozenOpponentEnv(agent_deck, opp_decks, [0.7, 0.3], frozen_ckpt, seed=0)
    rng = np.random.default_rng(0)
    seats_seen = set()
    try:
        for _ep in range(8):
            obs, seat, info = env.reset()
            assert seat == env.agent_seat == info["seat"]
            seats_seen.add(env.agent_seat)
            assert env.deck[env.agent_seat] == list(agent_deck)
            assert env.deck[1 - env.agent_seat] in [list(d) for d in opp_decks]
            done = False
            steps = 0
            r = 0.0
            while not done:
                legal = np.flatnonzero(np.asarray(obs["action_mask"]) > 0.5)
                assert len(legal) > 0
                obs, seat, r, done, _info = env.step(int(rng.choice(legal)))
                assert seat == env.agent_seat          # opponent decisions must never surface
                steps += 1
                assert steps < 3000
            assert -1.0 <= float(r) <= 1.0
        assert seats_seen == {0, 1}, f"agent seat never randomized: {seats_seen}"
    finally:
        env.close()
