"""Evaluation gates for the self-play trainer (extracted from train_selfplay for clarity).

* _gate_schedule / gated_winrate -- the C11 mirror promotion gate (fixed schedule, rank-split).
* field_winrate -- the LB-proxy field gate (net's decks vs a frozen field-clone on weighted meta
  decks; see rl/field_mix.py for the canonical mix and scripts/meta_gate.py for the offline twin).
All are stateless module-level functions; the promotion POLICY (parity bar, field-ratchet teacher
promotion in finetune mode) lives in train_selfplay.maybe_promote/_field_gate closures.
"""
from __future__ import annotations


import numpy as np
import torch

from .encoding import SUBMIT_ACTION


@torch.no_grad()
def _net_greedy_pick(net, enc, raw_obs, deck, tracker, ability_slots, device):
    """Greedy full selection for one decision (encode, argmax, loop until SUBMIT/maxCount). Used by the
    C11 gated-promotion h2h for the teacher opponent."""
    sel = raw_obs["select"]
    picked: list[int] = []
    while True:
        e = enc.encode(raw_obs, set(picked), self_deck=deck, tracker=tracker, ability_slots=ability_slots)
        o = {k: torch.as_tensor(np.asarray(v)[None],
                                dtype=(torch.long if k in enc.int_keys else torch.float32), device=device)
             for k, v in e.items()}
        a = int(net.logits_value(o)[0].argmax(-1).item())
        if a == SUBMIT_ACTION:
            break
        picked.append(a)
        if len(picked) >= sel.get("maxCount", 1):
            break
    return sorted(set(picked))


def _gate_schedule(n_decks, n_games, mode):
    """Deterministic ordered-pair schedule for the promotion gate: n_games (agent_deck, opp_deck)
    index pairs in PILOT-SWAP blocks -- each scheduled pair (i, j) is immediately followed by
    (j, i), so per block the current net pilots BOTH decks once and deck strength cancels.
    mirror: j == i (both sides the same deck -- pure piloting probe). cross: j = (i + k) % n with
    the offset k cycling after each full deck round-robin, covering matchups evenly. The schedule
    is a pure function of (n_decks, n_games, mode) -> IDENTICAL every gate, so consecutive gates
    are paired comparisons (matchup mix is never a noise source).

    When the pool outgrows the round-robin (n_games < 2*n_decks, e.g. --decks ladder_all), the
    sequential walk would cover only a head-PREFIX of the pool (biased: ladder_all is sorted by
    play count, while training is uniform). Then: FIXED-SEED uniform draw over ALL decks instead
    -- unbiased w.r.t. the training pool, still a pure function of the same args (identical every
    gate; pilot-swap blocks kept)."""
    n_pairs = (n_games + 1) // 2
    if n_pairs < n_decks:
        rng = np.random.default_rng(97531)
        ii = rng.choice(n_decks, size=n_pairs, replace=False)
        pairs = []
        for i in (int(x) for x in ii):
            j = i if mode == "mirror" else int((i + 1 + rng.integers(n_decks - 1)) % n_decks)
            pairs.append((i, j))
            pairs.append((j, i))
        return pairs[:n_games]
    pairs = []
    i, k = 0, 1
    while len(pairs) < n_games:
        j = i if (mode == "mirror" or n_decks == 1) else (i + k) % n_decks
        pairs.append((i, j))
        pairs.append((j, i))
        i += 1
        if i == n_decks:
            i = 0
            if mode == "cross" and n_decks > 1:
                k = k % (n_decks - 1) + 1
    return pairs[:n_games]


def opp_sequence(opp_weights, n_games, seed):
    """THE weighted opponent sequence for every field/cross gate (consolidation rule 2026-08-03:
    inline ft_wr, cross-gate, frozen-base cells and offline diags must all draw opponents from
    the SAME mixture through this one builder -- they used to quantize independently, so their
    absolute levels answered different questions).

    Largest-remainder counts when every positive-weight deck is representable (round(w*n) >= 1):
    exact proportions, byte-identical to the historical builder (fe10k series unaffected).
    When the pool outgrows the gate (ft mixtures: ~4k decks vs 2k games, round() zeroed the
    entire wide third), fixed-seed SYSTEMATIC resampling over the full CDF instead: high-weight
    decks keep ~exact counts, every tail deck appears with probability w*n. Both branches are a
    pure function of (weights, n_games, seed) -> identical every gate, paired comparisons hold."""
    w = np.asarray(opp_weights, float)
    counts = [int(round(x * n_games)) for x in w]
    if all(c >= 1 for c, x in zip(counts, w) if x > 0):
        while sum(counts) < n_games:
            counts[int(np.argmax(w))] += 1
        seq = [i for i, c in enumerate(counts) for _ in range(c)][:n_games]
        np.random.default_rng(seed).shuffle(seq)
        return seq
    rng = np.random.default_rng(seed)
    cdf = np.cumsum(w / w.sum())
    pts = (np.arange(n_games) + rng.random()) / n_games
    seq = np.searchsorted(cdf, pts).tolist()
    rng.shuffle(seq)
    return seq


def _gate_env(deck_a, deck_b, encoder, seed):
    """A TwoSidedSelfPlayEnv pinned to ONE ordered deck pair (seat0 = deck_a, seat1 = deck_b) --
    the gate drives BOTH seats externally, so the two-sided env (which exposes every decision as
    (obs, seat)) is the right frame for batched net routing. Native encode: ~52us/decision vs
    ~1ms for the Python path, and safe to multi-instance per process (per-instance holders)."""
    from .env_selfplay import TwoSidedSelfPlayEnv

    class _Pinned(TwoSidedSelfPlayEnv):
        def _sample_decks(self):
            self._deck_idx = (0, 1)
            return self.decks[0], self.decks[1]
    return _Pinned(decks=[deck_a, deck_b], encoder=encoder, seed=seed, native_encode=True)


@torch.no_grad()
def gated_winrate(net, teacher, pool, n_games, enc, device, seed=1234, mode="mirror",
                  rank=0, world=1):
    """C11 promotion gate, generalist-aware. Plays the FIXED _gate_schedule (same pairs + same
    seed every gate -> paired comparisons across gates) with the current `net` (greedy) vs the
    `teacher` (greedy). Under DDP every rank plays sched[rank::world] and the caller all-reduces
    the counts. Returns (LOCAL win fraction excl. draws, details dict with wins/decided/draws +
    per-agent-deck W/D/L for <out>/promotion_gate.jsonl).

    LOCKSTEP-BATCHED (2026-07-14, the b=1 gate took ~80s for 200 games/rank while training does
    ~20 games/s/rank): every rank now hosts ALL its games as simultaneous in-process battles
    (multi-battle per process proven 2026-07-13; per-instance holders shipped with it) and each
    barrier routes the acting rows into TWO batched forwards (candidate rows + teacher rows)
    through the NATIVE encoder, instead of one b=1 Python-encode forward per decision. Per-game
    math is identical: same schedule, same greedy argmax per decision, independent battles.
    Seat assignment moved from the env's internal rng to an explicit fixed-seed draw -- still
    identical on every gate (paired comparisons preserved); the one-off re-draw vs the pre-
    lockstep implementation shifts the absolute bar within schedule noise."""
    from .policy import bucketed_opt_len_np
    net.eval(); teacher.eval()
    sched = _gate_schedule(len(pool), n_games, mode)[rank::world]
    cand_seat = np.random.default_rng(seed + rank).integers(0, 2, size=len(sched))
    live = []
    for g, (ia, ib) in enumerate(sched):
        env = _gate_env(pool[ia], pool[ib], enc, seed=int(seed + rank * 100_000 + g))
        obs, seat, _info = env.reset()
        live.append({"env": env, "obs": obs, "seat": int(seat), "g": g, "ia": ia})
    wins = decided = draws = 0
    per_deck: dict[int, list[int]] = {}                      # agent-deck idx -> [w, d, l]
    try:
        for _barrier in range(20_000):                       # hard guard (games cap at max_steps anyway)
            if not live:
                break
            for grp, gnet in (([b for b in live if b["seat"] == cand_seat[b["g"]]], net),
                              ([b for b in live if b["seat"] != cand_seat[b["g"]]], teacher)):
                if not grp:
                    continue
                np_obs = {k: np.stack([np.asarray(b["obs"][k]) for b in grp])
                          for k in grp[0]["obs"]}
                batch = {k: torch.as_tensor(v, dtype=(torch.long if k in enc.int_keys
                                                      else torch.float32), device=device)
                         for k, v in np_obs.items()}
                acts = gnet.logits_value(
                    batch, opt_len=bucketed_opt_len_np(np_obs["action_mask"], np_obs["opt_attr"])
                )[0].argmax(-1).cpu().numpy()
                for b, a in zip(grp, acts):
                    b["act"] = int(a)
            finished = []
            for b in live:
                obs, seat, _r, done, info = b["env"].step(b["act"])
                if not done:
                    b["obs"], b["seat"] = obs, int(seat)
                    continue
                t = info.get("terminal")
                cs = int(cand_seat[b["g"]])
                row = per_deck.setdefault(b["ia"], [0, 0, 0])
                if t is None or t[0] == t[1]:                # truncation/draw: logged, not scored
                    draws += 1; row[1] += 1
                elif t[cs] > t[1 - cs]:
                    wins += 1; decided += 1; row[0] += 1
                else:
                    decided += 1; row[2] += 1
                finished.append(b)
            for b in finished:
                b["env"].close(); live.remove(b)
    finally:
        for b in live:
            b["env"].close()
    wr = wins / max(decided, 1)
    return wr, {"n": len(sched), "wins": wins, "decided": decided, "draws": draws, "mode": mode,
                "per_deck": {int(kk): vv for kk, vv in sorted(per_deck.items())}}


# ==================== lockstep-batched field gate (absorbed from rl/gates_fast.py) ====================
# LOCKSTEP-BATCHED field gate -- the field_winrate twin of the July-14 gated_winrate rework.
# 
# field_winrate plays its schedule ONE GAME AT A TIME with b=1 forwards for BOTH sides (candidate
# argmax + _net_greedy_pick for the clone), which is why a 1,850-game FIELD_BROAD costs ~30 min on
# an H200: the GPU is idle for almost all of it. gated_winrate already solved this exact shape --
# host every game as a simultaneous in-process battle and route each barrier's acting rows through
# TWO batched forwards. This applies the same frame to the field gate, with a bounded live window so
# the game count can be 10,000 without 10,000 concurrent battles.
# 
# Differences from field_winrate that DO move the numbers (so these are separate instruments and
# their series must not be mixed):
# 
#   * TwoSidedSelfPlayEnv, not CabtEnv. It carries the PKMN_TURN_CAP stall guard that CabtEnv has
#     never had -- a looping game ends at the cap instead of burning max_steps=4000 (~20x a normal
#     game). That is both the speedup on loop decks AND the fix for them scoring as free draws.
#   * Seat assignment is an explicit fixed-seed draw. CabtEnv.randomize_side drew from the env's own
#     rng; here the candidate DECK is placed at seat0 or seat1 explicitly. The candidate net always
#     pilots the pool deck -- only the seat moves, which matters because P2 carries a real edge
#     (~0.60 in cabt h2h).
#   * Native encode (as the promotion gate has used in production since July).
# 
# Per-game math is otherwise identical: same deterministic schedule, same greedy argmax per
# decision, independent battles.

import numpy as np
import torch



@torch.no_grad()
def field_winrate_fast(net, field_net, agent_pool, opp_decks, opp_weights, n_games, enc, device,
                       seed=4321, rank=0, world=1, live_n=128, progress_every=0, progress_cb=None,
                       mirror=False):
    """Returns {wins, decided, draws, losses, n}. `draws` counts games that ended with no terminal
    (turn-cap / max_steps): the caller decides whether they are dropped (the historical wr) or
    scored as losses (ladder-truthful -- a 600s timeout loses).

    progress_every>0 calls progress_cb(finished, total, wins, decided, draws) every that many
    finished games -- a shard is thousands of games, so without it the run is a black box until
    it ends."""
    from .policy import bucketed_opt_len_np
    net.eval(); field_net.eval()

    cand = _field_candidates(len(agent_pool), n_games)
    if mirror:
        # h2h piloting probe: BOTH seats play the SAME deck (caller passes opp_decks=agent_pool),
        # so deck strength cancels per game and wr is purely net-vs-net. No weighted opp mix.
        sched = [(c, c) for c in cand][rank::world]
    else:
        seq = opp_sequence(opp_weights, n_games, seed)       # shared builder: see its docstring
        sched = [(cand[g], seq[g]) for g in range(n_games)][rank::world]
    # which SEAT the candidate deck occupies; the candidate net always pilots the pool deck
    flip = np.random.default_rng(seed + 1_000_003 + rank).integers(0, 2, size=max(len(sched), 1))

    def _spawn(gi):
        ia, ib = sched[gi]
        cs = int(flip[gi])
        a, b = agent_pool[ia], opp_decks[ib]
        d0, d1 = (b, a) if cs else (a, b)          # candidate deck goes to seat `cs`
        env = _gate_env(d0, d1, enc, seed=int(seed + rank * 1_000_003 + gi))
        obs, seat, _info = env.reset()
        return {"env": env, "obs": obs, "seat": int(seat), "cs": cs}

    wins = decided = draws = 0
    errs = 0                                        # spawn/step failures, reported not hidden
    live: list[dict] = []
    nxt = 0
    _last_prog = [0]
    try:
        while nxt < len(sched) or live:
            while len(live) < live_n and nxt < len(sched):
                try:
                    live.append(_spawn(nxt))
                except Exception:
                    # A one-off bad deck pair must not kill the shard, but a SYSTEMIC problem
                    # (deck format, missing engine) would otherwise be laundered into "draws"
                    # and produce a plausible-looking wrong winrate. Fail loudly on the first
                    # few games instead.
                    if nxt < 5:
                        raise
                    draws += 1; errs += 1
                nxt += 1
            if not live:
                break
            for grp, gnet in (([b for b in live if b["seat"] == b["cs"]], net),
                              ([b for b in live if b["seat"] != b["cs"]], field_net)):
                if not grp:
                    continue
                np_obs = {k: np.stack([np.asarray(b["obs"][k]) for b in grp]) for k in grp[0]["obs"]}
                batch = {k: torch.as_tensor(v, dtype=(torch.long if k in enc.int_keys
                                                      else torch.float32), device=device)
                         for k, v in np_obs.items()}
                acts = gnet.logits_value(
                    batch, opt_len=bucketed_opt_len_np(np_obs["action_mask"], np_obs["opt_attr"])
                )[0].argmax(-1).cpu().numpy()
                for b, a in zip(grp, acts):
                    b["act"] = int(a)
            finished = []
            for b in live:
                try:
                    obs, seat, _r, done, info = b["env"].step(b["act"])
                except Exception:
                    draws += 1; errs += 1; finished.append(b); continue
                if not done:
                    b["obs"], b["seat"] = obs, int(seat)
                    continue
                t = info.get("terminal")
                cs = b["cs"]
                if t is None or t[cs] == t[1 - cs]:          # turn-cap / max_steps / true draw
                    draws += 1
                elif t[cs] > t[1 - cs]:
                    wins += 1; decided += 1
                else:
                    decided += 1
                finished.append(b)
            for b in finished:
                try:
                    b["env"].close()
                except Exception:
                    pass
                live.remove(b)
            if progress_every and progress_cb is not None:
                fin = decided + draws
                if fin // progress_every > _last_prog[0]:
                    _last_prog[0] = fin // progress_every
                    progress_cb(fin, len(sched), wins, decided, draws)
    finally:
        for b in live:
            try:
                b["env"].close()
            except Exception:
                pass
    return {"wins": wins, "decided": decided, "draws": draws, "errs": errs,
            "losses": decided - wins, "n": len(sched)}


@torch.no_grad()
def ddp_cross_gate(net_x, net_y, x_pool, y_decks, y_weights, enc, device, seed,
                   bar=None, block=250, cap=2000, live_n=128, ddp_rank=0, ddp_world=1):
    """DDP-parallel cross gate, NO early stopping (user orders 2026-08-02: the rank0-serial
    gate stalled training up to ~500s while 3 GPUs idled, and the early-stopped 250-game
    B-side cells were too noisy to read trends from -- with the parallel gate, full caps are
    affordable). Round-based: in round t, DDP rank r plays schedule-block index t*ddp_world+r
    (disjoint shards of ONE fixed cap-game schedule), then all ranks all-reduce the tallies;
    every cell plays its FULL cap. `bar` is accepted for call-compat but unused. ddp_world=1
    degenerates to serial full-cap. Returns identical {wr, se, n, draws, blocks} on every rank."""
    import math as _m
    import torch as _t
    n_blocks = max(1, cap // block)
    wins = dec = drw = 0
    played = 0
    rounds = _m.ceil(n_blocks / max(1, ddp_world))
    use_dist = ddp_world > 1
    if use_dist:
        import torch.distributed as _dist
    for t in range(rounds):
        bi = t * ddp_world + ddp_rank
        w = d = dr = pl = 0
        if bi < n_blocks:
            fi = field_winrate_fast(net_x, net_y, x_pool, y_decks, y_weights, cap, enc, device,
                                    seed=seed, rank=bi, world=n_blocks, live_n=live_n)
            w, d, dr, pl = fi["wins"], fi["decided"], fi["draws"], 1
        if use_dist:
            buf = _t.tensor([w, d, dr, pl], dtype=_t.float64, device=device)
            _dist.all_reduce(buf)
            w, d, dr, pl = (int(x) for x in buf.tolist())
        wins += w; dec += d; drw += dr; played += pl
    p = wins / max(dec, 1)
    return {"wr": p, "n": dec, "draws": drw, "blocks": played,
            "se": _m.sqrt(max(p * (1 - p), 1e-9) / max(dec, 1))}


def sequential_cross_gate(net_x, net_y, x_pool, y_decks, y_weights, enc, device, seed,
                          bar=None, block=250, cap=2000, live_n=64):
    """Early-stopping asymmetric gate for the two-net finetune cross-promotion (user design
    2026-07-29): net_x pilots x_pool, net_y pilots the weighted y_decks mix, seat-balanced.

    Plays `block`-game shards of ONE fixed `cap`-game schedule (rank/world sharding, so shards
    are disjoint) and stops as soon as the running wr separates from `bar` by 2*SE -- most checks
    are nowhere near the bar and resolve in one or two blocks, which is what keeps the 20M-step
    gate cadence at ~2-3% training overhead instead of ~10%. bar=None = precision mode (play
    exactly `cap`; used for the frozen-vs-frozen baseline, which only reruns on promotions).

    Returns {wr, se, n, draws, blocks}; wr is from net_x's side, truncations excluded."""
    import math as _m
    world = max(1, cap // block)
    wins = dec = drw = 0
    blocks = 0
    for r in range(world):
        fi = field_winrate_fast(net_x, net_y, x_pool, y_decks, y_weights, cap, enc, device,
                                seed=seed, rank=r, world=world, live_n=live_n)
        wins += fi["wins"]; dec += fi["decided"]; drw += fi["draws"]; blocks += 1
        if bar is not None and dec > 50:
            p = wins / dec
            se = _m.sqrt(max(p * (1 - p), 1e-9) / dec)
            if abs(p - bar) > 2 * se:
                break
    p = wins / max(dec, 1)
    return {"wr": p, "n": dec, "draws": drw, "blocks": blocks,
            "se": _m.sqrt(max(p * (1 - p), 1e-9) / max(dec, 1))}
