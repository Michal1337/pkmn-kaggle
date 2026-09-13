"""State-token truncation (--state-trunc) -- correctness guards.

Design under test (policy.state_keep_np + TokenTransformer._encode(state_keep=...)):
GATHER-THEN-SLICE. The option src/tgt gather runs on the FULL state sequence (absolute positions
stay valid -- the split_heads-bug class cannot occur), then a batch-uniform keep-list drops state
positions that are padded in EVERY row. Dropping such tokens is EXACT: the encoder has no
positional encoding, masked keys get exactly-zero attention weight, and the mean-pool excludes
pads (the same argument as option truncation and the existing b=1 pad-drop path).

Tiers:
  1. keep-list invariants (header identity, no present token dropped, fillers all-pad,
     shape bucketing) on real engine obs + adversarial masks;
  2. forward equivalence full-vs-sliced (logits/value allclose + argmax identical) across the
     split_heads x static x categorical grid, composed with option truncation;
  3. fuzz: random synthetic obs incl. HOLES in streams (face-down units) and pathological masks;
  4. gradient equivalence through a backward pass (the update path uses the same _encode).

Run:  python -m pytest tests/test_state_trunc.py -v
"""
import random

import numpy as np
import torch

from rl.card_features import get_card_table
from rl.encoding import TokenEncoder
from rl.enc_constants import _TOKEN_LAYOUT, N_STATE_TOKENS, MAX_OPTIONS, N_ACTIONS, OPT_PICKED
from rl.policy import (build_token_net, obs_to_tensors, state_keep_np, bucketed_opt_len_np,
                       _STREAM_MASK_KEY, _STATE_TRUNC_PAD)

torch.manual_seed(0)


# ------------------------------------------------------------------ corpus (real engine obs)
_CORPUS = None


def _corpus(n=24, seed=3):
    """Encoded obs from real games at varied depths (multiple games -> heterogeneous boards)."""
    global _CORPUS
    if _CORPUS is not None:
        return _CORPUS
    from kaggle_environments.envs.cabt.cg import game
    from kaggle_environments.envs.cabt.cg.sim import Battle
    from rl.encoding import GameTracker
    from rl.decks_train import TRAIN_TOP50

    enc = TokenEncoder(get_card_table())
    rng = random.Random(seed)
    out = []
    decks = [list(TRAIN_TOP50.values())[13], list(TRAIN_TOP50.values())[7]]
    for g in range(2):
        if Battle.battle_ptr:
            try:
                game.battle_finish()
            except Exception:
                pass
            Battle.battle_ptr = None
        deck = decks[g]
        o, _ = game.battle_start(deck, deck)
        tr = GameTracker()
        for _ in range(400):
            s = o.get("current") or {}
            if s.get("result", -1) >= 0:
                break
            if s.get("yourIndex", 0) == 0:
                tr.update(o)
                out.append(enc.encode(o, set(), self_deck=deck, tracker=tr))
                if len(out) >= n * (g + 1) // 2:
                    break
            sel = o.get("select") or {}
            opt = sel.get("option") or []
            k = sel.get("maxCount", 1) or 1
            picks = rng.sample(range(len(opt)), min(k, len(opt))) if opt else []
            try:
                o = game.battle_select(sorted(set(picks)))
            except Exception:
                break
        game.battle_finish(); Battle.battle_ptr = None
    assert len(out) >= 8, f"only {len(out)} obs collected"
    _CORPUS = (out, enc)
    return _CORPUS


def _batch(obs_list, device="cpu"):
    return {k: torch.stack([obs_to_tensors(e, device)[k] for e in obs_list]) for k in obs_list[0]}


def _np_batch(obs_list):
    return {k: np.stack([np.asarray(e[k]) for e in obs_list]) for k in obs_list[0]}


def _stream_spans(split_heads):
    """[(name, start, size)] in FINAL-sequence coords (after the split value/submit insertion)."""
    off = 2 if split_heads else 0
    spans, pos = [], 0
    for name, size in _TOKEN_LAYOUT:
        base = pos + (off if pos >= 1 else 0)      # insertion sits AFTER cls (position 0)
        spans.append((name, base, size))
        pos += size
    return spans


def _present_final(npb, split_heads):
    """Boolean [n_final] -- positions present in ANY row, in FINAL coords (header = present)."""
    off = 2 if split_heads else 0
    pres = np.zeros(N_STATE_TOKENS + off, bool)
    pres[:3 + off] = True                          # cls (+value/submit) + sel_type + sel_ctx
    for name, base, size in _stream_spans(split_heads):
        mkey = _STREAM_MASK_KEY.get(name, f"{name}_mask")
        if mkey in npb:
            m = npb[mkey].reshape(-1, size) > 0.5
            pres[base:base + size] = m.any(axis=0)
    return pres


# ------------------------------------------------------------------ tier 1: keep invariants
def test_keep_invariants_real():
    obs_list, enc = _corpus()
    npb = _np_batch(obs_list)
    for split in (False, True):
        off = 2 if split else 0
        sk = state_keep_np(npb, split)
        assert sk is not None, "real mid-game batch must have droppable tokens"
        hdr = 3 + off
        assert list(sk[:hdr]) == list(range(hdr)), "header must be kept first, in order"
        pres = _present_final(npb, split)
        kept = set(int(i) for i in sk)
        missing = [i for i in np.flatnonzero(pres) if i not in kept]
        assert not missing, f"PRESENT positions dropped: {missing[:10]}"
        # fillers (kept but not present) must be padded in EVERY row
        for i in kept - set(np.flatnonzero(pres).tolist()):
            assert not pres[i]
        n_full = N_STATE_TOKENS + off
        assert len(sk) % _STATE_TRUNC_PAD == 0 or len(sk) == n_full
        assert len(set(sk.tolist())) == len(sk), "duplicate keep positions"
        assert sk.max() < n_full


def test_keep_invariants_adversarial():
    obs_list, enc = _corpus()
    npb = _np_batch(obs_list[:6])
    # (a) empty streams: zero out discards + stadium entirely
    a = {k: v.copy() for k, v in npb.items()}
    a["self_discard_mask"][:] = 0.0; a["opp_discard_mask"][:] = 0.0; a["stadium_mask"][:] = 0.0
    # (b) holes: knock a mid-list unit out while keeping a later one present
    a["self_unit_mask"][:, 1] = 0.0
    a["self_unit_mask"][0, 3] = 1.0
    for split in (False, True):
        sk = state_keep_np(a, split)
        pres = _present_final(a, split)
        kept = set(int(i) for i in sk)
        assert all(i in kept for i in np.flatnonzero(pres)), "hole handling dropped a present token"
        # the hole position (unit slot 1) sits BEFORE the last present unit -> must be kept too
        for name, base, size in _stream_spans(split):
            if name == "self_units":
                last = int(np.flatnonzero(a["self_unit_mask"].reshape(-1, size).any(axis=0) > 0)[-1])
                assert all((base + j) in kept for j in range(last + 1)), "last-present slice broke"


# ------------------------------------------------------------------ tier 2: forward equivalence
def _nets(ct):
    for split in (False, True):
        for static in (False, True):
            for cat in (False, True):
                torch.manual_seed(11)
                net = build_token_net(ct, {"d_model": 64, "nhead": 4, "nlayers": 2,
                                           "static": static, "split_heads": split,
                                           "value_categorical": cat})
                net.eval()
                yield net, (split, static, cat)


def test_forward_equivalence_grid():
    obs_list, enc = _corpus()
    ct = enc.cards
    b = _batch(obs_list)
    npb = _np_batch(obs_list)
    ol = bucketed_opt_len_np(npb["action_mask"], npb["opt_attr"])
    for net, cfg in _nets(ct):
        sk_np = state_keep_np(npb, net.split_heads)
        sk = torch.as_tensor(sk_np)
        with torch.no_grad():
            l0, v0 = net.logits_value(b)                                   # full, untruncated
            l1, v1 = net.logits_value(b, opt_len=ol, state_keep=sk)       # both truncations
            l2, v2 = net.logits_value(b, state_keep=sk)                    # state-only
        for lx, vx in ((l1, v1), (l2, v2)):
            assert torch.allclose(l0, lx, atol=1e-5, rtol=1e-5), f"logits diverged {cfg}"
            assert torch.allclose(v0, vx, atol=1e-5, rtol=1e-5), f"value diverged {cfg}"
            assert torch.equal(l0.argmax(-1), lx.argmax(-1)), f"argmax flipped {cfg}"
        legal = b["action_mask"] < 0.5
        assert (l1[legal] <= -1e8).all(), "masked logits leaked"


def test_forward_equivalence_adversarial_masks():
    obs_list, enc = _corpus()
    ct = enc.cards
    npb = _np_batch(obs_list[:6])
    npb["self_discard_mask"][:] = 0.0                 # empty stream -> sliced to length 0
    npb["opp_hand_mask"][:, 5:] = 0.0                 # shortened stream
    npb["self_unit_mask"][:, 1] = 0.0                 # hole
    b = {k: torch.as_tensor(v).clone() for k, v in
         _batch([{kk: npb[kk][i] for kk in npb} for i in range(npb["cls_scalars"].shape[0])]).items()}
    for net, cfg in _nets(ct):
        sk_np = state_keep_np(npb, net.split_heads)
        if sk_np is None:
            continue
        sk = torch.as_tensor(sk_np)
        with torch.no_grad():
            l0, v0 = net.logits_value(b)
            l1, v1 = net.logits_value(b, state_keep=sk)
        assert torch.allclose(l0, l1, atol=1e-5, rtol=1e-5), f"adversarial logits diverged {cfg}"
        assert torch.allclose(v0, v1, atol=1e-5, rtol=1e-5), f"adversarial value diverged {cfg}"


# ------------------------------------------------------------------ tier 3: fuzz (synthetic)
def _rand_obs(enc, rng, B):
    """Random-but-shape-valid obs batch; masks Bernoulli WITH holes; opt pos any state pos or -1."""
    sh, ik, UNK = enc.shapes, enc.int_keys, int(enc.UNK)
    out = {}
    for k, shape in sh.items():
        if k in ik:
            hi = {"select_type": 15, "select_context": 63, "opt_verb": 16,
                  "opt_attack_id": 2047}.get(k)
            if hi is not None:
                out[k] = rng.integers(0, hi + 1, size=(B, *shape)).astype(np.int64)
            elif k in ("opt_src_pos", "opt_tgt_pos"):
                p = rng.integers(-1, N_STATE_TOKENS, size=(B, *shape)).astype(np.int64)
                out[k] = np.where(rng.random((B, *shape)) < 0.5, -1, p)
            else:                                              # card ids
                out[k] = rng.integers(0, UNK + 1, size=(B, *shape)).astype(np.int64)
        elif k.endswith("_mask"):
            out[k] = (rng.random((B, *shape)) < 0.45).astype(np.float32)
        elif k == "action_mask":
            m = (rng.random((B, *shape)) < 0.15).astype(np.float32)
            m[:, MAX_OPTIONS] = 1.0                            # submit legal -> no all-masked row
            out[k] = m
        else:                                                  # float features/flags/attrs
            out[k] = rng.random((B, *shape)).astype(np.float32)
    out["opt_attr"][..., OPT_PICKED] = (rng.random(out["opt_attr"].shape[:-1]) < 0.05)
    return out


def test_fuzz_equivalence():
    obs_list, enc = _corpus()                                   # for enc.shapes only
    ct = enc.cards
    rng = np.random.default_rng(0)
    torch.manual_seed(5)
    nets = [(build_token_net(ct, {"d_model": 32, "nhead": 4, "nlayers": 1,
                                  "split_heads": s}).eval(), s) for s in (False, True)]
    for trial in range(60):
        B = int(rng.integers(1, 7))                            # B=1 exercises _TRUNC_B1 composition
        npb = _rand_obs(enc, rng, B)
        b = {k: torch.as_tensor(v) for k, v in npb.items()}
        ol = bucketed_opt_len_np(npb["action_mask"], npb["opt_attr"])
        for net, split in nets:
            sk_np = state_keep_np(npb, split)
            # keep invariants under fuzz
            pres = _present_final(npb, split)
            if sk_np is not None:
                kept = set(int(i) for i in sk_np)
                assert all(i in kept for i in np.flatnonzero(pres)), f"trial {trial}: present dropped"
            sk = None if sk_np is None else torch.as_tensor(sk_np)
            with torch.no_grad():
                l0, v0 = net.logits_value(b)
                l1, v1 = net.logits_value(b, opt_len=ol, state_keep=sk)
            assert torch.allclose(l0, l1, atol=1e-5, rtol=1e-5), f"trial {trial} split={split}"
            assert torch.allclose(v0, v1, atol=1e-5, rtol=1e-5), f"trial {trial} split={split}"


# ------------------------------------------------------------------ tier 4: gradient equivalence
def test_grad_equivalence():
    obs_list, enc = _corpus()
    ct = enc.cards
    b = _batch(obs_list)
    npb = _np_batch(obs_list)

    def run(net, sk):
        net.train()
        net.zero_grad()
        logits, value = net.logits_value(b, state_keep=sk)
        legal = b["action_mask"] > 0.5
        loss = logits[legal].sum() * 1e-3 + value.sum()
        loss.backward()
        return {n: p.grad.detach().clone() for n, p in net.named_parameters() if p.grad is not None}

    for split in (False, True):
        torch.manual_seed(23)
        net = build_token_net(ct, {"d_model": 64, "nhead": 4, "nlayers": 2, "split_heads": split})
        sk = torch.as_tensor(state_keep_np(npb, split))
        g0 = run(net, None)
        g1 = run(net, sk)
        assert g0.keys() == g1.keys()
        for n in g0:
            assert torch.allclose(g0[n], g1[n], atol=1e-5, rtol=1e-4), f"grad diverged: {n} (split={split})"


def test_collect_rollout_smoke_state_trunc():
    """End-to-end collect loop with state_trunc=True and a REAL TokenTransformer: exercises the
    torch-branch keep (incoming obs), the numpy-branch keep (loop bottom), device threading, and
    the boot get_value -- the exact code path the trainer runs (compile aside)."""
    import rl.train_selfplay as ts
    obs_list, enc = _corpus()
    ct = enc.cards
    torch.manual_seed(9)
    net = build_token_net(ct, {"d_model": 32, "nhead": 4, "nlayers": 1, "split_heads": True})
    net.eval()

    N, T = 3, 4
    shapes = enc.shapes
    base = _np_batch((obs_list * 2)[:N])

    class _Vec:
        def step(self, actions):
            obs = {k: v.copy() for k, v in base.items()}
            return obs, np.zeros(N, np.int64), np.zeros(N, np.float32), np.zeros(N, bool), [{}] * N

    buf = {"obs": {k: torch.zeros((T, N, *shapes[k]),
                                  dtype=(torch.long if k in enc.int_keys else torch.float32))
                   for k in shapes},
           "seat": torch.zeros(T, N, dtype=torch.long), "act": torch.zeros(T, N, dtype=torch.long),
           "logp": torch.zeros(T, N), "val": torch.zeros(T, N),
           "rew": torch.zeros(T, N), "done": torch.zeros(T, N)}
    cur = {k: torch.as_tensor(base[k], dtype=(torch.long if k in enc.int_keys else torch.float32))
           for k in base}
    boot, seat, _, end_obs, _ = ts.collect_rollout(
        net, _Vec(), buf, shapes, enc.int_keys, torch.device("cpu"), T, cur,
        torch.zeros(N, dtype=torch.long), state_trunc=True)
    assert boot.shape == (N,)
    # actions must be legal under the stored masks (the sliced forward produced them)
    for t in range(T):
        m = buf["obs"]["action_mask"][t]
        assert (m[torch.arange(N), buf["act"][t]] > 0.5).all(), "sliced collect sampled illegal action"


if __name__ == "__main__":
    test_keep_invariants_real()
    test_keep_invariants_adversarial()
    test_forward_equivalence_grid()
    test_forward_equivalence_adversarial_masks()
    test_fuzz_equivalence()
    test_grad_equivalence()
    test_collect_rollout_smoke_state_trunc()
    print("state-trunc tests: PASSED")
