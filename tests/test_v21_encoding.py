"""v2.1 encoding fixes -- behavioral guards for the two information-gap changes.

Fix 1 (drawable flag): the self_deck stream keeps the FULL decklist; ``self_deck_flag[i]`` must be
1 iff decklist copy i is NOT visible in any of our public zones (hand / discard / board incl.
preEvolution+tools+energyCards / own stadium), checked against an INDEPENDENT Counter-based
reference recomputed from the raw obs (not the encoder's own helper -- no circularity). The flag
exists ONLY on the self_deck stream.

Fix 2 (already-picked visibility): encoding with a non-empty ``picked`` set must (a) set the
opt_attr OPT_PICKED flag on exactly those slots, (b) keep them action-masked, and (c) keep their
tokens ATTENDED by the net -- corrupting a picked option's content must change the legal-action
logits (presence proof), while corrupting a truly-absent slot must change nothing (pad proof,
complementing tests/test_pad_mask.py).

Run:  python -m pytest tests/test_v21_encoding.py -v
"""
import random
from collections import Counter

import numpy as np
import torch

from rl.encoding import GameTracker, TokenEncoder, MAX_OPTIONS, OPT_PICKED, SUBMIT_ACTION
from rl.card_features import get_card_table
from rl.policy import build_token_net, obs_to_tensors


def _collect(n=8, seed=0):
    """Drive one real game; return [(raw_obs, encoded, tracker_state)] for seat 0 + the deck."""
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
    for _ in range(600):
        s = o.get("current") or {}
        if s.get("result", -1) >= 0:
            break
        if s.get("yourIndex", 0) == 0:
            tr.update(o)
            out.append((o, enc.encode(o, set(), self_deck=deck, tracker=tr)))
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
    return out, deck, enc, ct


def _reference_flags(deck, raw_obs):
    """Independent multiset reference: per-copy drawable flags (1 = not visible), decklist order."""
    s = raw_obs["current"]
    me = s["yourIndex"]
    mp = s["players"][me]

    def cid(c):
        if c is None:
            return 0
        return (c.get("id") or 0) if isinstance(c, dict) else (c or 0)

    vis = Counter()
    for c in (mp.get("hand") or []):
        if cid(c) > 0:
            vis[cid(c)] += 1
    for c in (mp.get("discard") or []):
        if cid(c) > 0:
            vis[cid(c)] += 1
    for grp in ("active", "bench"):
        for pk in (mp.get(grp) or []):
            if not pk:
                continue
            if pk.get("id"):
                vis[pk["id"]] += 1
            for c in (pk.get("preEvolution") or []) + (pk.get("tools") or []) \
                    + (pk.get("energyCards") or []):
                if cid(c) > 0:
                    vis[cid(c)] += 1
    for c in (s.get("stadium") or []):
        if isinstance(c, dict) and c.get("playerIndex") == me and c.get("id"):
            vis[c["id"]] += 1
    flags = []
    for d in deck:
        if vis.get(d, 0) > 0:
            vis[d] -= 1
            flags.append(0.0)           # this copy is visible somewhere -> not drawable
        else:
            flags.append(1.0)           # still in deck (or face-down in our prizes)
    return flags


def test_deck_drawable_flags():
    obs_list, deck, enc, ct = _collect()
    assert len(obs_list) >= 2, f"only collected {len(obs_list)} obs"
    for raw, e in obs_list:
        # the stream always carries the FULL decklist, in order, fully masked-in
        assert list(e["self_deck_id"]) == list(deck)
        assert (e["self_deck_mask"] == 1.0).all()
        # per-copy drawable flag matches the independent reference
        assert list(e["self_deck_flag"]) == _reference_flags(deck, raw), "drawable-flag mismatch"
    # sanity: the flag actually engages over the game (note: the very first setup decision exposes
    # an empty hand in the obs, so all 60 stay flagged there -- the reference agrees). No
    # monotonicity assert: shuffle-back effects can legitimately re-raise flags.
    sums = [float(e["self_deck_flag"].sum()) for _, e in obs_list]
    assert min(sums) < 60.0, "flag never engaged (no visible own cards over the whole rollout?)"


def _reference_opp_visible(raw_obs):
    """Counter of the opponent's CURRENTLY-VISIBLE card ids (discard / board incl. attachments /
    their stadium). Every such instance is serial-tracked with a visible zone, so the number of
    flag-0 tokens per id in opp_deck must equal exactly this count."""
    s = raw_obs["current"]
    me = s["yourIndex"]
    op = s["players"][1 - me]

    def cid(c):
        if c is None:
            return 0
        return (c.get("id") or 0) if isinstance(c, dict) else (c or 0)

    vis = Counter()
    for c in (op.get("discard") or []):
        if cid(c) > 0:
            vis[cid(c)] += 1
    for grp in ("active", "bench"):
        for pk in (op.get(grp) or []):
            if not pk:
                continue
            if pk.get("id"):
                vis[pk["id"]] += 1
            for c in (pk.get("preEvolution") or []) + (pk.get("tools") or []) \
                    + (pk.get("energyCards") or []):
                if cid(c) > 0:
                    vis[cid(c)] += 1
    for c in (s.get("stadium") or []):
        if isinstance(c, dict) and c.get("playerIndex") == (1 - me) and c.get("id"):
            vis[c["id"]] += 1
    return vis


def test_opp_deck_hidden_flags():
    obs_list, deck, enc, ct = _collect()
    assert len(obs_list) >= 2, f"only collected {len(obs_list)} obs"
    saw_visible = False
    for raw, e in obs_list:
        ids, fl, m = e["opp_deck_id"], e["opp_deck_flag"], e["opp_deck_mask"]
        assert (m == 1.0).all()                                # always 60 present (revealed + UNK)
        assert ((fl == 0.0) | (fl == 1.0)).all()
        assert (fl[ids == enc.UNK] == 1.0).all(), "unrevealed UNK slots must be flagged hidden"
        # per id: #flag-0 tokens == #currently-visible instances of that id in opp's public zones
        vis = _reference_opp_visible(raw)
        real = ids != enc.UNK
        got0 = Counter()
        for i in np.flatnonzero(real & (fl < 0.5)):
            got0[int(ids[i])] += 1
        assert got0 == +vis, f"visible/spent flag mismatch: got {dict(got0)} want {dict(vis)}"
        if got0:
            saw_visible = True
    assert saw_visible, "opp never had a visible card over the rollout?"


def test_picked_flag_and_visibility():
    torch.manual_seed(0)
    obs_list, deck, enc, ct = _collect()
    # find an obs with >= 3 options so picked/legal/absent slots coexist
    raw = next((r for r, _ in obs_list if len((r.get("select") or {}).get("option") or []) >= 3), None)
    assert raw is not None, "no multi-option decision collected"
    n_opt = len(raw["select"]["option"])
    pk = 1                                                     # pick slot 1, keep 0/2 legal

    tr = GameTracker(); tr.update(raw)
    e = enc.encode(raw, {pk}, self_deck=deck, tracker=tr)
    # (a) flag set exactly on the picked slot, (b) picked slot action-masked
    flags = e["opt_attr"][:, OPT_PICKED]
    assert flags[pk] == 1.0 and flags.sum() == 1.0
    assert e["action_mask"][pk] == 0.0
    assert e["cls_scalars"][17] > 0.0                          # buffered-pick count scalar

    net = build_token_net(ct, {"d_model": 64, "nhead": 4, "nlayers": 2})
    net.eval()
    b = {k: v.unsqueeze(0) for k, v in obs_to_tensors(e, "cpu").items()}
    with torch.no_grad():
        l0, v0 = net.logits_value(b)
    legal = b["action_mask"][0] > 0.5

    # (c) corrupting the PICKED option's content must CHANGE the outputs (it is attended) ...
    c = {k: v.clone() for k, v in b.items()}
    c["opt_src_card"][0, pk] = (int(c["opt_src_card"][0, pk]) + 7) % ct.vocab_size or 1
    c["opt_attr"][0, pk, :OPT_PICKED] += 1.0
    with torch.no_grad():
        l1, v1 = net.logits_value(c)
    assert (l1[0][legal] - l0[0][legal]).abs().max().item() > 0.0 or \
           (v1 - v0).abs().max().item() > 0.0, "picked option is invisible to the net"

    # ... while corrupting a truly-ABSENT slot changes nothing (bitwise)
    if n_opt < MAX_OPTIONS:
        dead = n_opt                                           # first slot past the real options
        c2 = {k: v.clone() for k, v in b.items()}
        c2["opt_src_card"][0, dead] = 5
        c2["opt_attr"][0, dead, :OPT_PICKED] = 3.0             # corrupt content, NOT the mask-like flag
        c2["opt_attr"][0, dead, OPT_PICKED + 1:] = 3.0
        with torch.no_grad():
            l2, v2 = net.logits_value(c2)
        assert (l2[0][legal] - l0[0][legal]).abs().max().item() == 0.0
        assert (v2 - v0).abs().max().item() == 0.0

    # picked stays unsampleable: its logit is masked to -inf-ish
    assert l0[0][pk].item() < -1e8


if __name__ == "__main__":
    test_deck_drawable_flags()
    test_picked_flag_and_visibility()
    print("v2.1 encoding tests: PASSED")
