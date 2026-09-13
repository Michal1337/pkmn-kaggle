"""Pad / attention-mask isolation guard -- a CRITICAL correctness invariant.

The token encoder emits fixed-width streams (hand/deck/discard/prize/units/options) padded
to constant caps, with a parallel ``*_mask`` marking real vs pad slots. policy.TokenTransformer
turns those masks into the Transformer's ``src_key_padding_mask`` (and drops padded tokens
entirely on the b=1 inference fast-path). If that assembly is wrong -- misaligned toks/pads,
flipped polarity, an all-pad softmax row, or the option gather leaking a padded slot -- the net
silently trains on garbage.

This proves isolation empirically: take real game obs, CORRUPT every PADDED input slot with
random values (leaving the masks untouched), and assert the legal-action logits + value are
BITWISE-unchanged. Padded tokens must not reach any real token, option, or the value.

Run:  python -m pytest tests/test_pad_mask.py -v     (or: python tests/test_pad_mask.py)
"""
import random

import numpy as np
import torch

from rl.encoding import GameTracker, TokenEncoder, MAX_OPTIONS, N_STATE_TOKENS, OPT_PICKED
from rl.card_features import get_card_table
from rl.policy import build_token_net, obs_to_tensors, _CARD_STREAMS


def _collect_obs(n=6, seed=0):
    """Drive one real game and return the agent (seat 0)'s first ``n`` encoded decision obs."""
    from kaggle_environments.envs.cabt.cg import game
    from kaggle_environments.envs.cabt.cg.sim import Battle
    from rl.decks_train import TRAIN_TOP50

    ct = get_card_table()
    enc = TokenEncoder(ct)
    deck = list(TRAIN_TOP50.values())[13]
    rng = random.Random(seed)

    def finish():
        if Battle.battle_ptr:
            try:
                game.battle_finish()
            except Exception:
                pass
            Battle.battle_ptr = None

    out = []
    finish()
    o, _ = game.battle_start(deck, deck)
    tr = GameTracker()
    for _ in range(400):
        s = o.get("current") or {}
        if s.get("result", -1) >= 0:
            break
        if s.get("yourIndex", 0) == 0:
            tr.update(o)
            out.append(enc.encode(o, set(), self_deck=deck, tracker=tr))
            if len(out) >= n:
                break
        sel = o.get("select") or {}
        opt = sel.get("option") or []
        k = sel.get("maxCount", 1) or 1
        picks = rng.sample(range(len(opt)), min(k, len(opt))) if opt else []
        try:
            o = game.battle_select(sorted(set(picks)))
        except Exception:
            break
    finish()
    return out, ct


def _corrupt_pads(b, vocab):
    """Deep-copy ``b`` with every PADDED slot's CONTENT randomized; masks left untouched."""
    c = {k: v.clone() for k, v in b.items()}
    for name, _ in _CARD_STREAMS:
        pad = c[f"{name}_mask"] < 0.5
        c[f"{name}_id"][pad] = torch.randint(1, vocab, c[f"{name}_id"].shape)[pad]
    for side in ("self", "opp"):
        pad = c[f"{side}_unit_mask"] < 0.5
        for key in ("top_id", "preevo_id", "tool_id", "energy_id"):
            t = c[f"{side}_unit_{key}"]
            t[pad] = torch.randint(1, vocab, t.shape)[pad]
        at = c[f"{side}_unit_attr"]
        at[pad] = torch.randn_like(at)[pad]
    # option slots are PADDING iff neither legal nor already-picked (picked slots are visible-but-
    # illegal by design -- corrupting them SHOULD change the output, so they are not corrupted here)
    optpad = ((c["action_mask"][:, :MAX_OPTIONS] < 0.5)
              & (c["opt_attr"][..., OPT_PICKED] < 0.5))
    for key, hi in (("opt_verb", 17), ("opt_attack_id", 2048),
                    ("opt_src_card", vocab), ("opt_tgt_card", vocab),
                    ("opt_src_pos", N_STATE_TOKENS), ("opt_tgt_pos", N_STATE_TOKENS)):
        t = c[key]
        t[optpad] = torch.randint(0, hi, t.shape)[optpad]
    a = c["opt_attr"]
    a[optpad] = torch.randn_like(a)[optpad]
    a[..., OPT_PICKED][optpad] = 0.0   # the picked flag is MASK-like (drives presence): keep it intact
    return c


def _batch(obs_sub):
    return {k: torch.stack([obs_to_tensors(e, "cpu")[k] for e in obs_sub]) for k in obs_sub[0]}


def test_pad_mask_isolation():
    torch.manual_seed(0)
    obs_list, ct = _collect_obs()
    assert len(obs_list) >= 2, f"only collected {len(obs_list)} obs"
    net = build_token_net(ct, {"d_model": 128, "nhead": 4, "nlayers": 3, "static": True,
                               "split_heads": True, "value_categorical": True})
    net.eval()

    # b>1 exercises encoder(src_key_padding_mask); b=1 exercises the drop-padded fast-path.
    for tag, sub in (("b>1", obs_list), ("b=1", obs_list[:1]), ("b=2", obs_list[:2])):
        b = _batch(sub)
        with torch.no_grad():
            l0, v0 = net.logits_value(b)
            l1, v1 = net.logits_value(_corrupt_pads(b, ct.vocab_size))
        legal = b["action_mask"] > 0.5
        dl = (l0[legal] - l1[legal]).abs().max().item()
        dv = (v0 - v1).abs().max().item()
        assert dl == 0.0 and dv == 0.0, f"[{tag}] pad leak: max|d_logit|={dl:.2e} max|d_value|={dv:.2e}"


if __name__ == "__main__":
    test_pad_mask_isolation()
    print("pad/mask isolation: PASSED (padded tokens fully isolated, bitwise-exact)")
