"""Shared decision-time search primitives (independent of the net).

Observation-conditioned PIMC determinization (`_determinize`), the branchable-node
test (`_branchable`), and the MCTS node (`_Node`) -- imported by the live v2 MCTS in
search_agent.py. The v1-net MCTS that used to live here (search_select / mcts_* /
the obs_to_tensors-based value + priors) was removed along with the v1 policy; only
these net-agnostic primitives remain.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .decks import DECKS
from .encoding import MAX_OPTIONS


def _load_sample_deck():
    """The engine 'sample' deck (agent/deck.csv) -- part of the meta/training pool but
    not in DECKS. Bundle-safe: alongside this module in a submission, ../agent in repo."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    for p in (os.path.join(here, "deck.csv"), os.path.join(here, "..", "agent", "deck.csv")):
        try:
            if os.path.exists(p):
                return [int(x) for x in open(p) if x.strip()]
        except Exception:
            pass
    return None


# Deck hypotheses for opponent inference: the known meta = official archetypes + the
# engine sample deck (which is also the multi-deck training pool). Covers every deck an
# opponent might play, so inference isn't forced into a same-archetype fallback.
_CANDIDATES = [list(d) for d in DECKS.values()]
_sample_deck = _load_sample_deck()
if _sample_deck:
    _CANDIDATES.append(_sample_deck)


def _cid(c):
    return c.get("id") if isinstance(c, dict) else c


def _observed(pl, stadium, owner, with_hand):
    """Counter of card ids of player `owner` that are publicly visible (or, for our
    own side, also our hand). These are hard constraints: definitely in their 60."""
    cnt = Counter()
    for grp in ("active", "bench"):
        for pk in (pl.get(grp) or []):
            if not pk:
                continue
            cnt[pk["id"]] += 1
            for key in ("preEvolution", "energyCards", "tools"):
                for c in (pk.get(key) or []):
                    cnt[_cid(c)] += 1
    for c in (pl.get("discard") or []):
        cnt[_cid(c)] += 1
    if with_hand:
        for c in (pl.get("hand") or []):
            cnt[_cid(c)] += 1
    for c in (stadium or []):
        if isinstance(c, dict) and c.get("playerIndex") == owner:
            cnt[c["id"]] += 1
    return cnt


def _fit(cards, n, full, rng):
    """Trim/pad a card list to exactly n (pad by sampling the full deck)."""
    cards = list(cards)
    rng.shuffle(cards)
    if len(cards) > n:
        return cards[:n]
    while len(cards) < n and full:
        cards.append(rng.choice(full))
    return cards


def _basic_pokemon(ids, enc):
    """A basic Pokemon id from `ids` (for a face-down opp active), else None."""
    for cid in ids:
        f = enc.cards.features(cid)
        if f is not None and f[0] > 0.5 and f[3] > 0.5:   # category=pokemon, stage=Basic
            return int(cid)
    return None


def _determinize(obs, deck, rng, enc):
    """Observation-conditioned PIMC. The opponent's visible cards (discard, board,
    attachments, their stadium) are hard constraints; we infer which known decklist
    is consistent with them and fill the hidden zones from `inferred_deck - observed`.
    Our own hidden cards (deck/prizes) come from `our_deck - seen - hand`."""
    s = obs["current"]; me = s["yourIndex"]
    mp, op = s["players"][me], s["players"][1 - me]
    stadium = s.get("stadium")

    # --- opponent: infer archetype, then deal the unseen remainder ---
    seen = _observed(op, stadium, 1 - me, with_hand=False)
    cands = [Counter(d) for d in (_CANDIDATES + [list(deck)])]
    consistent = [c for c in cands if all(seen[k] <= c[k] for k in seen)]
    if not consistent:                       # off-meta opponent: best-overlap match
        consistent = [max(cands, key=lambda c: sum(min(seen[k], c[k]) for k in seen))]
    D = rng.choice(consistent)               # ensemble across determinizations
    full_opp = list(D.elements())
    rem = []
    for cid, ct in D.items():
        rem += [cid] * max(0, ct - seen[cid])
    rng.shuffle(rem)

    dC = op["deckCount"]; pC = len(op.get("prize") or []); hC = op.get("handCount", 0)
    face = bool(op.get("active") and op["active"][0] is None)
    rem = _fit(rem, dC + pC + hC + (1 if face else 0), full_opp, rng)
    i = 0
    opp_deck = rem[i:i + dC]; i += dC
    opp_prize = rem[i:i + pC]; i += pC
    opp_hand = rem[i:i + hC]; i += hC
    opp_active = []
    if face:
        opp_active = [_basic_pokemon(rem[i:] + full_opp, enc) or rem[i]]

    # --- our side: deck/prizes are what's left of our deck after seen + hand ---
    seen_me = _observed(mp, stadium, me, with_hand=True)
    Dme = Counter(deck)
    rem_me = []
    for cid, ct in Dme.items():
        rem_me += [cid] * max(0, ct - seen_me[cid])
    rng.shuffle(rem_me)
    mdC = mp["deckCount"]; mpC = len(mp.get("prize") or [])
    rem_me = _fit(rem_me, mdC + mpC, list(deck), rng)

    return dict(
        your_deck=rem_me[:mdC],
        your_prize=rem_me[mdC:mdC + mpC],
        opponent_deck=opp_deck,
        opponent_prize=opp_prize,
        opponent_hand=opp_hand,
        opponent_active=opp_active,
    )


def _branchable(obs_dict):
    """A node we branch on: single-pick with >=2 real options, not terminal."""
    s = obs_dict.get("select")
    return (s is not None and s.get("maxCount", 1) == 1
            and 2 <= len(s["option"]) <= MAX_OPTIONS   # >MAX_OPTIONS: not encodable, net can't score
            and obs_dict["current"]["result"] < 0)


class _Node:
    __slots__ = ("sid", "obs", "me_turn", "term", "tv", "P", "N", "W", "kids", "exp")

    def __init__(self, sid, obs, me):
        self.sid = sid; self.obs = obs
        cur = obs["current"]
        self.term = cur["result"] >= 0
        self.tv = (0.0 if cur["result"] == 2 else (1.0 if cur["result"] == me else -1.0)) if self.term else 0.0
        self.me_turn = cur["yourIndex"] == me
        self.P = None; self.exp = False
        n = len(obs["select"]["option"]) if obs.get("select") else 0
        self.N = np.zeros(n); self.W = np.zeros(n); self.kids = {}


# ===========================================================================
# Decision-time PUCT MCTS + would_ko (merged from the former search_agent.py)
# ===========================================================================
import dataclasses
import random

import numpy as np
import torch

from .encoding import SUBMIT_ACTION
from .encoding import N_ACTIONS

try:    # per-attack is_variable flag for variable-aware would_KO sampling (bundle-safe)
    from .attack_data import ATTACKS as _WK_ATTACKS
except Exception:
    _WK_ATTACKS = {}

WK_NDET_VAR = 10    # determinizations for a VARIABLE-damage attack (coin/conditional) -> KO probability +
                    # expected-prizes + win estimates; fixed-damage attacks are deterministic -> 1 sim is exact.


def _tens(enc, enc_obs, dev=None):
    return {k: torch.as_tensor(np.asarray(v)[None],
                               dtype=(torch.long if k in enc.int_keys else torch.float32),
                               device=dev)
            for k, v in enc_obs.items()}


def _net_dev(net):
    """Device of a net (GPU sweep support 2026-08-02; None = CPU-default, the historic path)."""
    try:
        return next(net.parameters()).device
    except Exception:
        return None


@torch.no_grad()
def _net_greedy_select(obs, net, enc, deck, tracker, ability):
    """Full engine selection from the v2 net (buffered single-pick), no search."""
    sel = obs["select"]; picked: list[int] = []
    for _ in range(sel.get("maxCount", 1) + 1):
        o = _tens(enc, enc.encode(obs, set(picked), self_deck=deck, tracker=tracker, ability_slots=ability), _net_dev(net))
        a = int(net.logits_value(o)[0][0].argmax())
        if a == SUBMIT_ACTION:
            break
        picked.append(a)
        if len(picked) >= sel.get("maxCount", 1):
            break
    return sorted(set(picked))


@torch.no_grad()
def _value(net, enc, obs_dict, me, deck, tracker, ability) -> float:
    cur = obs_dict["current"]
    if cur["result"] >= 0:                                   # terminal
        return 0.0 if cur["result"] == 2 else (1.0 if cur["result"] == me else -1.0)
    o = _tens(enc, enc.encode(obs_dict, set(), self_deck=deck, tracker=tracker, ability_slots=ability), _net_dev(net))
    v = float(net.get_value(o)[0])
    return v if cur["yourIndex"] == me else -v               # net value is for the acting player


@torch.no_grad()
def _priors_value(net, enc, obs_dict, me, deck, tracker, ability):
    o = _tens(enc, enc.encode(obs_dict, set(), self_deck=deck, tracker=tracker, ability_slots=ability), _net_dev(net))
    logits, val = net.logits_value(o)
    n = len(obs_dict["select"]["option"])
    p = torch.softmax(logits[0, :n], -1).cpu().numpy() if n else np.zeros(0)
    v = float(val[0]); cur = obs_dict["current"]
    return p, (v if cur["yourIndex"] == me else -v)


def _advance(api, sid, obs, net, enc, deck, tracker, ability):
    """Step through forced/multi-pick selects (net-greedy) to the next branchable node."""
    while obs["current"]["result"] < 0 and not _branchable(obs):
        st = api.search_step(sid, _net_greedy_select(obs, net, enc, deck, tracker, ability))
        sid = st.searchId; obs = dataclasses.asdict(st.observation)
    return sid, obs


def mcts_visits(obs, net, enc, deck, tracker=None, ability=None,
                n_sims=160, n_det=2, c_puct=1.5, rng=None):
    """PUCT MCTS over the v2 net; returns (root visit counts over options, ok)."""
    rng = rng or random.Random()
    sel = obs.get("select")
    if sel is None or not _branchable(obs) or sel.get("type") != 0:
        return None, False
    from sdk_cg import api
    me = obs["current"]["yourIndex"]
    n_opt = len(sel["option"])
    agg = np.zeros(n_opt)

    def simulate(node):
        if node.term:
            return node.tv
        if not node.exp:
            node.P, v = _priors_value(net, enc, node.obs, me, deck, tracker, ability)
            node.exp = True
            return v
        N, W, P = node.N, node.W, node.P
        sqrtsum = float(np.sqrt(N.sum() + 1e-8))
        Q = np.where(N > 0, W / np.maximum(N, 1), 0.0)
        score = (Q if node.me_turn else -Q) + c_puct * P * sqrtsum / (1.0 + N)
        a = int(score.argmax())
        if a not in node.kids:
            st = api.search_step(node.sid, [a])
            csid, cobs = _advance(api, st.searchId, dataclasses.asdict(st.observation),
                                  net, enc, deck, tracker, ability)
            node.kids[a] = _Node(csid, cobs, me)
        v = simulate(node.kids[a])
        node.N[a] += 1; node.W[a] += v
        return v

    for _ in range(n_det):
        try:
            root_ss = api.search_begin(api.to_observation_class(obs), **_determinize(obs, deck, rng, enc))
        except Exception:
            continue
        root = _Node(root_ss.searchId, dataclasses.asdict(root_ss.observation), me)
        if len(root.N) != n_opt:                              # alignment guard
            try: api.search_end()
            except Exception: pass
            return None, False
        for _ in range(n_sims):
            simulate(root)
        agg += root.N
        try: api.search_release(root_ss.searchId)
        except Exception: pass
    try: api.search_end()
    except Exception: pass
    if agg.sum() == 0:
        return None, False
    return agg, True


def _my_prizes(obs, me) -> int:
    pls = (obs.get("current") or {}).get("players") or []
    return len(pls[me].get("prize") or []) if 0 <= me < len(pls) else 6


def _advance_resolve(api, sid, obs, me):
    """Net-free: step (first-legal) through our remaining sub-selects until the attack resolves
    (turn passes to opp / terminal). Used by the would_KO sim -- no net needed, so it runs in
    env workers during collection."""
    while obs["current"]["result"] < 0 and obs["current"]["yourIndex"] == me:
        sel = obs.get("select") or {}
        opts = sel.get("option") or []
        k = sel.get("maxCount", 1) or 1
        pick = list(range(min(k, len(opts)))) if opts else []
        st = api.search_step(sid, pick)
        sid = st.searchId; obs = dataclasses.asdict(st.observation)
    return sid, obs


def would_ko_flags(obs, deck, enc, n_var=WK_NDET_VAR, rng=None, early_stop=False) -> dict:
    """Engine-accurate would-KO per ATTACK option: simulate the attack 1 ply on the SDK sim and
    report the KO RATE (we take a prize / win). Net-free + minimal -> usable as a TRAINING FEATURE
    per attack-option (abilities/stadium/weakness/variable all resolved by the real engine).
    VARIABLE-aware: a fixed-damage attack is deterministic given the visible board -> 1 sim (exact);
    a VARIABLE (coin/conditional) attack is sampled `n_var` times -> KO probability (the engine
    re-rolls each determinization; manual_coin=False auto-flips off the persistent agent_ptr RNG).
    Returns {option_index: (ko_rate, exp_prizes_taken, win_rate)}; {} if no attack options / not a
    MAIN select. exp_prizes_taken (0..6) grades the prize VALUE of the KO (ex=2, mega-ex=3); win_rate
    = P(this move ENDS the game in my favour: I take the last prize OR the opp can't promote)."""
    sel = obs.get("select")
    if sel is None or sel.get("type") != 0:
        return {}
    opts = sel.get("option") or []
    atk = [i for i, o in enumerate(opts) if o.get("attackId") is not None]
    if not atk:
        return {}
    from sdk_cg import api
    me = obs["current"]["yourIndex"]
    p0 = _my_prizes(obs, me)
    rng = rng or random.Random()
    out = {}
    for a in atk:
        av = _WK_ATTACKS.get(opts[a].get("attackId"))
        ndet = max(1, n_var) if (av and av[1]) else 1     # variable -> sample prob; fixed -> 1 exact
        kos = wins = trials = 0
        prize_sum = 0.0
        seen = set()                                       # distinct per-sim (ko,prizes,won) -> early-stop when unanimous
        for _ in range(ndet):
            try:
                ss = api.search_begin(api.to_observation_class(obs), **_determinize(obs, deck, rng, enc))
            except Exception:
                continue
            try:
                st = api.search_step(ss.searchId, [a])
                _, o2 = _advance_resolve(api, st.searchId, dataclasses.asdict(st.observation), me)
                trials += 1
                cur = o2["current"]
                won = (cur["result"] == me)                # this move ENDS the game in my favour
                took = max(0, p0 - _my_prizes(o2, me))     # prizes I gain this ply (0/1/2/3; ex=2, mega=3)
                ko_bit = 1 if (won or (cur["result"] < 0 and took > 0)) else 0
                kos += ko_bit
                prize_sum += took
                wins += int(won)
                if early_stop: seen.add((ko_bit, took, int(won)))
            except Exception:
                pass
            finally:
                try: api.search_release(ss.searchId)
                except Exception: pass
            # EARLY-STOP: a determinization-invariant attack (all sims identical) is settled after 3
            # confirmations -> skip the remaining n_var sims (97% of variable attacks are deterministic;
            # the ~3% genuine coin-flips disagree early and run the full n_var, so the rate is unbiased there).
            if early_stop and ndet > 1 and trials >= 3 and len(seen) == 1:
                break
        if trials:
            out[a] = (kos / trials, prize_sum / trials, wins / trials)
    try: api.search_end()
    except Exception: pass
    return out


def write_would_ko(obs, flags) -> None:
    """Write the engine-sim consequence onto each attack option in-place (the encoder emits these):
      o['would_ko']        KO/prize-take RATE in [0,1]
      o['would_ko_prizes'] EXPECTED prizes taken (0..6) -- grades a 2-prize ex KO above a 1-prize basic
      o['would_ko_win']    P(this move ENDS the game in my favour) in [0,1]"""
    opts = (obs.get("select") or {}).get("option") or []
    for i, r in flags.items():
        if 0 <= i < len(opts):
            ko, prizes, win = r
            opts[i]["would_ko"] = float(ko)
            opts[i]["would_ko_prizes"] = float(prizes)
            opts[i]["would_ko_win"] = float(win)


def annotate_would_ko(obs, deck, enc, n_var=WK_NDET_VAR, rng=None, early_stop=False) -> dict:
    """Compute would_ko_flags AND write them onto the attack options (the per-option TRAINING
    feature). Call ONCE per real (root) decision -- in the env at collection AND in the inference
    agent when net_config['would_ko'] -> train==test (both use the default n_var). Returns flags."""
    flags = would_ko_flags(obs, deck, enc, n_var=n_var, rng=rng, early_stop=early_stop)
    write_would_ko(obs, flags)
    return flags


@torch.no_grad()
def _priors_value_batch(net, enc, leaves, me, deck, tracker, ability):
    """Batched _priors_value: one stacked forward for all pending leaves.
    leaves: list of obs_dicts (each branchable, non-terminal). Returns [(P, v_signed)]."""
    encs = [enc.encode(o, set(), self_deck=deck, tracker=tracker, ability_slots=ability)
            for o in leaves]
    _dev = _net_dev(net)
    batch = {k: torch.as_tensor(np.stack([np.asarray(e[k]) for e in encs]),
                                dtype=(torch.long if k in enc.int_keys else torch.float32),
                                device=_dev)
             for k in encs[0]}
    logits, vals = net.logits_value(batch)
    out = []
    for i, o in enumerate(leaves):
        n = len(o["select"]["option"])
        p = torch.softmax(logits[i, :n], -1).cpu().numpy() if n else np.zeros(0)
        v = float(vals[i])
        out.append((p, v if o["current"]["yourIndex"] == me else -v))
    return out


def mcts_visits_batched(obs, net, enc, deck, tracker=None, ability=None,
                        n_sims=160, n_det=2, c_puct=1.5, rng=None, leaf_k=8, vloss=1.0):
    """PUCT MCTS with BATCHED leaf evaluation (CPU-submission speedup): per wave, run up to
    `leaf_k` virtual-loss descents pooled ACROSS the n_det determinization trees, evaluate all
    new leaves in ONE stacked net forward, then back up and release the virtual losses. The
    engine stepping and the greedy sub-selects inside _advance stay serial (phase 2 if
    profiling says they dominate). Semantics match mcts_visits up to the standard virtual-loss
    exploration perturbation. Returns (aggregate root visit counts, ok)."""
    rng = rng or random.Random()
    sel = obs.get("select")
    if sel is None or not _branchable(obs) or sel.get("type") != 0:
        return None, False
    from sdk_cg import api
    me = obs["current"]["yourIndex"]
    n_opt = len(sel["option"])

    roots = []
    for _ in range(n_det):
        try:
            ss = api.search_begin(api.to_observation_class(obs), **_determinize(obs, deck, rng, enc))
        except Exception:
            continue
        r = _Node(ss.searchId, dataclasses.asdict(ss.observation), me)
        if len(r.N) != n_opt:                                 # alignment guard
            try: api.search_release(ss.searchId)
            except Exception: pass
            continue
        roots.append([r, n_sims, ss.searchId])                # node, sims_left, sid

    if not roots:
        try: api.search_end()
        except Exception: pass
        return None, False

    def descend(root):
        """One virtual-loss descent. Returns ('leaf', path, node) needing net eval,
        or ('done', path, value) if terminal/backed-up inline."""
        path = []
        node = root
        while True:
            if node.term:
                return "done", path, node.tv
            if not node.exp or node.P is None:      # unexpanded OR reserved-pending (P arrives
                return "leaf", path, node           # after the wave's batched forward)
            N, W, P = node.N, node.W, node.P
            sqrtsum = float(np.sqrt(N.sum() + 1e-8))
            Q = np.where(N > 0, W / np.maximum(N, 1), 0.0)
            score = (Q if node.me_turn else -Q) + c_puct * P * sqrtsum / (1.0 + N)
            a = int(score.argmax())
            node.N[a] += vloss                                # virtual loss (visit-inflate)
            path.append((node, a))
            if a not in node.kids:
                st = api.search_step(node.sid, [a])
                csid, cobs = _advance(api, st.searchId, dataclasses.asdict(st.observation),
                                      net, enc, deck, tracker, ability)
                node.kids[a] = _Node(csid, cobs, me)
            node = node.kids[a]

    def backup(path, v):
        for nd, a in path:
            nd.N[a] += 1.0 - vloss                            # replace virtual with real visit
            nd.W[a] += v

    def unwind(path):
        for nd, a in path:
            nd.N[a] -= vloss

    live = [r for r in roots if r[1] > 0]
    while live:
        pending = []                                          # (rootrec, path, leafnode)
        for rec in live:
            budget = min(rec[1], max(1, leaf_k // len(live)))
            for _ in range(budget):
                kind, path, x = descend(rec[0])
                if kind == "done":
                    backup(path, x)
                    rec[1] -= 1
                elif x.exp:                                   # duplicate leaf in this wave:
                    unwind(path)                              # already queued -> retry next wave
                else:
                    x.exp = True                              # reserve (P set after the forward)
                    pending.append((rec, path, x))
        if pending:
            res = _priors_value_batch(net, enc, [x.obs for _, _, x in pending],
                                      me, deck, tracker, ability)
            for (rec, path, x), (P, v) in zip(pending, res):
                x.P = P
                backup(path, v)
                rec[1] -= 1
        live = [r for r in roots if r[1] > 0]

    agg = np.zeros(n_opt)
    for r, _, sid in roots:
        agg += r.N
        try: api.search_release(sid)
        except Exception: pass
    try: api.search_end()
    except Exception: pass
    if agg.sum() == 0:
        return None, False
    return agg, True


def mcts_select(obs, net, enc, deck, tracker=None, ability=None,
                n_sims=160, n_det=2, c_puct=1.5, rng=None, batched=False, leaf_k=8):
    """PUCT MCTS choice -> selection list[int] (net-greedy fallback / deck step)."""
    if obs.get("select") is None:                             # deck-selection step
        return [int(c) for c in deck]
    fn = mcts_visits_batched if batched else mcts_visits
    kw = dict(leaf_k=leaf_k) if batched else {}
    agg, ok = fn(obs, net, enc, deck, tracker, ability, n_sims, n_det, c_puct, rng, **kw)
    if not ok:
        return _net_greedy_select(obs, net, enc, deck, tracker, ability)
    return [int(agg.argmax())]


def mcts_policy(obs, net, enc, deck, tracker=None, ability=None,
                n_sims=160, n_det=2, c_puct=1.5, rng=None, temp=1.0):
    """For AlphaZero self-play on v2: returns (selection list[int], pi over N_ACTIONS)."""
    agg, ok = mcts_visits(obs, net, enc, deck, tracker, ability, n_sims, n_det, c_puct, rng)
    if not ok:
        return _net_greedy_select(obs, net, enc, deck, tracker, ability), None
    pi = np.zeros(N_ACTIONS, dtype=np.float32)
    if temp and temp != 1.0:
        agg = agg ** (1.0 / temp)
    pi[:len(agg)] = agg / agg.sum()
    a = int(np.random.choice(len(agg), p=pi[:len(agg)])) if temp > 0 else int(agg.argmax())
    return [a], pi
