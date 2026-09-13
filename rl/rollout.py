"""Rollout collection + GAE for the two-sided self-play trainer (extracted from
train_selfplay). collect_rollout also implements FINETUNE collection (frozen GPU opponent on the
non-agent seat + amask; see rl/env_finetune.py). make_buffers allocates the rollout buffers.
"""
from __future__ import annotations

import time

import numpy as np
import torch
from torch.distributions import Categorical

from .policy import bucketed_opt_len, bucketed_opt_len_np, state_keep_np


def make_buffers(num_steps, num_envs, shapes, int_keys, device):
    obs_buf = {k: torch.zeros((num_steps, num_envs, *sh),
                              dtype=(torch.long if k in int_keys else torch.float32), device=device)
               for k, sh in shapes.items()}
    return {"obs": obs_buf,
            "act":  torch.zeros((num_steps, num_envs), dtype=torch.long, device=device),
            "logp": torch.zeros((num_steps, num_envs), device=device),
            "rew":  torch.zeros((num_steps, num_envs), device=device),
            "done": torch.zeros((num_steps, num_envs), device=device),
            "val":  torch.zeros((num_steps, num_envs), device=device),
            "seat": torch.zeros((num_steps, num_envs), dtype=torch.long, device=device),
            "amask": torch.ones((num_steps, num_envs), device=device),   # finetune: 1=agent row
            "fmask": torch.zeros((num_steps, num_envs), device=device)}  # frozen-wide: 1=frozen-
            # piloted opponent row (out of BOTH policy losses); all-zero unless the env partition
            # is active, so downstream masks are inert by default


def compute_gae_2s(rew, val, done, seat, boot_val, boot_seat, gamma, lam, trunc=None):
    """NEGAMAX GAE over the interleaved two-seat rollout. rew/val/done/seat: [T, N]; boot_*: [N].
    When the acting seat changes between consecutive decisions, the next state's value is from the
    OTHER player's perspective -> negate it (and negate the GAE carry). done zeros the bootstrap across
    episode boundaries. Returns (advantages, returns) in the same [T, N] layout (each from its own
    transition's acting-seat perspective).

    trunc (optional [T, N] bool-ish): rows that are ANTI-LOOP TRUNCATION terminals (env reward
    -PEN on the looping seat's final decision). With it, the truncation becomes a TWO-SIDED
    terminal: the looper's whole final run of same-seat decisions keeps the natural -PEN-scale
    credit, and the OTHER seat's last decision below that run (the victim of the stall) is
    anchored to an explicit +1 WIN with the negamax mirror severed there -- the -PEN must never
    reach the victim as a +PEN-scale jackpot (the 2026-08-01 production reward-hack). Rows
    further back propagate off these anchors at ordinary +/-1 game scale. None = historical
    symmetric mirror, byte-identical."""
    T, N = rew.shape
    adv = torch.zeros_like(rew)
    lastgae = torch.zeros(N, device=rew.device)
    if trunc is not None:
        tail = torch.zeros(N, dtype=torch.bool, device=rew.device)   # walking up a truncating run
        tseat = torch.zeros(N, dtype=seat.dtype, device=rew.device)  # seat that truncated
    for t in reversed(range(T)):
        if t == T - 1:
            next_val, next_seat = boot_val, boot_seat
        else:
            next_val, next_seat = val[t + 1], seat[t + 1]
        nonterminal = 1.0 - done[t]
        sign = torch.where(seat[t] == next_seat, 1.0, -1.0)   # flip perspective when the acting seat changes
        delta = rew[t] + gamma * sign * next_val * nonterminal - val[t]
        lastgae = delta + gamma * lam * nonterminal * sign * lastgae
        if trunc is not None:
            tr = trunc[t] > 0
            # victim terminal: first opposite-seat row under the truncating run -> +1 win,
            # carry/bootstrap from the looper's stream replaced wholesale by the anchor
            vict = tail & (seat[t] != tseat) & (done[t] < 0.5)
            lastgae = torch.where(vict, rew[t] + gamma * 1.0 - val[t], lastgae)
            # the run extends through the looper's own consecutive decisions; a trunc row
            # (re)starts it; the victim row (or any episode boundary) ends it
            tail = tr | (tail & (seat[t] == tseat) & (done[t] < 0.5))
            tseat = torch.where(tr, seat[t], tseat)
        adv[t] = lastgae
    return adv, adv + val


def _to_tensors(obs_np, int_keys, device):
    return {k: torch.as_tensor(obs_np[k], dtype=(torch.long if k in int_keys else torch.float32), device=device)
            for k in obs_np}


_STEP_T = [0.0]      # TRAIN_PROF: wall spent inside envs.step() (worker engine+encode+IPC+barrier)


def collect_rollout(net, envs, buf, shapes, int_keys, device, num_steps, cur_obs, cur_seat,
                    bf16=False, lv=None, state_trunc=False, opp_net=None, opp_lv=None,
                    agent_seat=None, mm=None, step_hook=None, train_opp=False,
                    frozen_net=None, frozen_envs=None, fw=None, frozen_sample=False, frozen_lv=None,
                    ks_net=None, ks_envs=None, ksq=None, ks_lv=None):
    """Fill `buf` with one two-sided rollout (module-level so it is unit-testable with a fake vec env).

    Stores, per step, the ACTING seat's obs/seat/value/action/logp, the transition's reward, and --
    CRITICALLY -- ``done[step]`` = whether the transition AT step `step` ENDED the episode (NOT the
    CleanRL "obs[step] is a fresh start" convention). This matches ``compute_gae_2s``'s ``1 - done[t]``:
    when transition t is terminal the bootstrap is killed and the NEXT step (an auto-reset, new episode)
    is treated as a fresh non-terminal start. Returns (boot_val, boot_seat, episode_winner_returns,
    end_obs, terms) where ``terms`` = {(step, env): terminal-rewards-dict} for every REAL (non-
    truncated) terminal in the window -- consumed by the value diagnostics (rl/value_diag.py).

    FINETUNE (opp_net + agent_seat set): envs are AsymmetricDeckEnv; where the acting seat is the
    OPPONENT's, the sampled action is overridden by opp_net's greedy pick (via `opp_lv` — the
    compiled full-batch forward — when provided, else one eager batched forward on the subset)
    and buf['amask'] marks the row 0 (policy/entropy losses are masked
    to agent rows in update_net; VALUE trains on all rows -- both-seat value learning is exactly
    what 2-side self-play does, and GAE needs the live net's values everywhere). `agent_seat` is
    updated IN PLACE from infos ('next_agent_seat' crosses auto-reset boundaries).

    FROZEN-WIDE PARTITION (frozen_net + frozen_envs set; two-net only, user 2026-08-06): envs
    flagged in `frozen_envs` (np bool [N], static) have their opponent seat piloted by
    `frozen_net` (greedy, like the single-net path) instead of live B -- B concentrates its
    capacity on the concentrated counter+now stream. Those rows get buf['fmask']=1 and logp=0:
    they are off-policy for BOTH nets (A's pg mask keeps only amask rows; B's mask subtracts
    fmask rows in update_net). A's value stays on them (spectator, single-net convention), so
    A's GAE/value stream over the wide games is unchanged. `fw` (optional list) receives 1/0
    per decided frozen-env episode from the agent's side = free wide-vs-base field telemetry."""
    # train_opp + opp_lv is the GRAPHED two-net path (2026-07-31 sps work): opp_lv is a dense
    # reduce-overhead twin of the LIVE B, refreshed in place per iteration by the caller --
    # sampling stays eager on the selected rows, so B's exploration math is byte-identical to
    # the eager subset branch.
    eprs = []
    terms = {}
    _STEP_T[0] = 0.0
    _seat_np = cur_seat.cpu().numpy() if agent_seat is not None else None   # one D2H per rollout;
    # maintained from seat_np in-loop (agent_seat is a NUMPY array, mutated in place for the caller)
    # option-bucket + state-keep for the INCOMING obs (tensors from the previous rollout's tail):
    # one legacy GPU-sync per rollout; every in-loop plan is then computed CPU-side from the numpy
    # obs (bucketed_opt_len_np / state_keep_np) BEFORE the H2D copy -> zero per-step GPU->CPU syncs.
    _ol = bucketed_opt_len(cur_obs["action_mask"], cur_obs["opt_attr"])
    _sk = None
    if state_trunc:
        _sk_np = state_keep_np(cur_obs, net.split_heads)       # torch branch: one D2H
        _sk = None if _sk_np is None else torch.as_tensor(_sk_np, device=device)
    for step in range(num_steps):
        for k in shapes:
            buf["obs"][k][step] = cur_obs[k]
        buf["seat"][step] = cur_seat
        with torch.no_grad():
            if lv is not None:             # --compile-collect: compiled logits_value + eager sampling
                # bf16 applies HERE too (2026-07-13: the async+varlen path passes lv= and was
                # silently skipping the autocast below -> --bf16-collect was inert in production)
                if bf16:
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        logits, value = lv(cur_obs, opt_len=_ol, state_keep=_sk)
                else:
                    logits, value = lv(cur_obs, opt_len=_ol, state_keep=_sk)   # == get_action_and_value's math
                dist = Categorical(logits=logits)
                action = dist.sample()
                logp = dist.log_prob(action)
            elif bf16:                     # --bf16-collect: fp32 buffers receive the casts on store
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    action, logp, _, value = net.get_action_and_value(cur_obs, opt_len=_ol, state_keep=_sk)
            else:
                action, logp, _, value = net.get_action_and_value(cur_obs, opt_len=_ol, state_keep=_sk)
        if opp_net is not None:                    # FINETUNE: frozen net decides the opponent seat
            amask_np = _seat_np == agent_seat
            buf["amask"][step] = torch.as_tensor(amask_np, dtype=torch.float32, device=device)
            opp_rows_np = ~amask_np
            fr_np = None
            ks_np = None
            if frozen_envs is not None or ks_envs is not None:   # FROZEN STREAMS: split opponent
                # rows by env stream; fmask covers BOTH (two-net B must train on neither)
                fr_np = (opp_rows_np & frozen_envs) if frozen_envs is not None else None
                ks_np = (opp_rows_np & ks_envs) if ks_envs is not None else None
                _fm = np.zeros_like(opp_rows_np)
                if fr_np is not None:
                    _fm |= fr_np
                if ks_np is not None:
                    _fm |= ks_np
                buf["fmask"][step] = torch.as_tensor(_fm, dtype=torch.float32, device=device)
                amask_np = amask_np | _fm          # live-B routing below sees frozen rows as "not
                #  its rows"; buf['amask'] above already stored the TRUE agent mask
            if not amask_np.all():
                # host-built int index instead of a CUDA bool mask: bool indexing runs nonzero()
                # on-GPU with a data-dependent output shape -> an implicit host sync per indexed
                # tensor (~41/step on the collect path; finetune is collect-bound). index_select /
                # index_copy_ over np.flatnonzero = identical values (same ascending rows), 0 syncs.
                idx = torch.as_tensor(np.flatnonzero(~amask_np), device=device)
                with torch.no_grad():
                    if opp_lv is not None:
                        # compiled path: FULL batch (constant rows = the opt_len-bucket shape
                        # space the live collect warms; a per-step row subset would retrace on
                        # every opponent count) + bf16 to match the live collect; opponent rows
                        # are selected on the OUTPUT, which costs argmax-tie epsilons only.
                        # opp_lv is itself cudagraph-backed: its run OVERWRITES the live net's
                        # graph-pool outputs, so materialize them first (the buf stores below
                        # otherwise trip the overwritten-output guard).
                        action, logp, value = action.clone(), logp.clone(), value.clone()
                        if bf16:
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                o_full = opp_lv(cur_obs, opt_len=_ol, state_keep=_sk)
                        else:
                            o_full = opp_lv(cur_obs, opt_len=_ol, state_keep=_sk)
                        if train_opp:
                            # TWO-NET graphed path: same logp/value semantics as the eager
                            # branch below -- B samples (exploration) and stores its OWN
                            # logp/value on its rows; index_select materializes the graph-pool
                            # outputs before the next replay overwrites them.
                            o_dist = Categorical(logits=o_full[0].index_select(0, idx).float())
                            o_pick = o_dist.sample()
                            logp = logp.index_copy(0, idx, o_dist.log_prob(o_pick).to(logp.dtype))
                            value = value.index_copy(
                                0, idx, o_full[1].index_select(0, idx).to(value.dtype))
                        else:
                            o_pick = o_full[0].index_select(0, idx).argmax(-1)
                    else:
                        sub = {k: cur_obs[k].index_select(0, idx) for k in cur_obs}
                        o_out = opp_net.logits_value(sub, opt_len=_ol, state_keep=_sk)
                        if train_opp:
                            # TWO-NET FINETUNE: the opponent net is LIVE. Sample (not argmax --
                            # B is learning and needs exploration) and store B's OWN logp/value
                            # on its rows: logp feeds B's PPO ratio in update_net, value makes
                            # buf['val'] the ACTING net's value everywhere (what negamax GAE
                            # actually wants -- val[t+1] is the next actor's estimate).
                            o_dist = torch.distributions.Categorical(logits=o_out[0].float())
                            o_pick = o_dist.sample()
                            o_logp = o_dist.log_prob(o_pick)
                            action = action.clone()
                            logp = logp.index_copy(0, idx, o_logp.to(logp.dtype))
                            value = value.index_copy(0, idx, o_out[1].to(value.dtype))
                        else:
                            o_pick = o_out[0].argmax(-1)
                            action = action.clone()
                action.index_copy_(0, idx, o_pick)
                if not train_opp:
                    logp = logp.index_fill(0, idx, 0.0)        # off-policy rows: pg-masked in update
            if fr_np is not None and fr_np.any():
                # FROZEN-WIDE rows: eager subset forward through the frozen pilot (~10%% of rows,
                # eager is fine and keeps the graphed path untouched). Greedy by default;
                # frozen_sample=True SAMPLES (specialist streams, e.g. the ship-ft KS stream --
                # a deterministic opponent is weaker and an easier overfit target, 2026-08-09).
                # Value stays A's spectator estimate either way.
                idxf = torch.as_tensor(np.flatnonzero(fr_np), device=device)
                with torch.no_grad():
                    if frozen_lv is not None:
                        # SPEED STACK (2026-08-10): full-batch reduce-overhead graph, rows
                        # selected on the OUTPUT (index_select materializes before the next
                        # graphed call overwrites the pool) -- the eager subset pair was ~half
                        # the collect fwd time at 30% stream share.
                        if bf16:
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                _ff = frozen_lv(cur_obs, opt_len=_ol, state_keep=_sk)
                        else:
                            _ff = frozen_lv(cur_obs, opt_len=_ol, state_keep=_sk)
                        f_out = (_ff[0].index_select(0, idxf),)
                    else:
                        subf = {k: cur_obs[k].index_select(0, idxf) for k in cur_obs}
                        if bf16:
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                f_out = frozen_net.logits_value(subf, opt_len=_ol, state_keep=_sk)
                        else:
                            f_out = frozen_net.logits_value(subf, opt_len=_ol, state_keep=_sk)
                action = action.clone()                        # graph-pool output safety
                if frozen_sample:
                    action.index_copy_(0, idxf,
                                       Categorical(logits=f_out[0].float()).sample())
                else:
                    action.index_copy_(0, idxf, f_out[0].argmax(-1))
                logp = logp.index_fill(0, idxf, 0.0)           # off-policy for BOTH nets
            if ks_np is not None and ks_np.any():
                # KS-STREAM rows: eager subset forward through the KS specialist, SAMPLED
                # (a deterministic specialist is weaker and an easier overfit target).
                idxk = torch.as_tensor(np.flatnonzero(ks_np), device=device)
                with torch.no_grad():
                    if ks_lv is not None:
                        if bf16:
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                _kf = ks_lv(cur_obs, opt_len=_ol, state_keep=_sk)
                        else:
                            _kf = ks_lv(cur_obs, opt_len=_ol, state_keep=_sk)
                        k_out = (_kf[0].index_select(0, idxk),)
                    else:
                        subk = {k: cur_obs[k].index_select(0, idxk) for k in cur_obs}
                        if bf16:
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                k_out = ks_net.logits_value(subk, opt_len=_ol, state_keep=_sk)
                        else:
                            k_out = ks_net.logits_value(subk, opt_len=_ol, state_keep=_sk)
                action = action.clone()                        # graph-pool output safety
                action.index_copy_(0, idxk, Categorical(logits=k_out[0].float()).sample())
                logp = logp.index_fill(0, idxk, 0.0)           # off-policy everywhere
        buf["act"][step] = action; buf["logp"][step] = logp; buf["val"][step] = value
        _s0 = time.time()
        obs_np, seat_np, reward, done, infos = envs.step(action.cpu().numpy())
        _STEP_T[0] += time.time() - _s0
        if agent_seat is not None:                 # track per-env agent seat across auto-resets
            _prev_as = (agent_seat.copy()
                        if ((fw is not None and frozen_envs is not None)
                            or (ksq is not None and ks_envs is not None)) else None)
            _seat_np = np.asarray(seat_np)
            for e, info in enumerate(infos):
                nas = info.get("next_agent_seat")
                if nas is not None:
                    agent_seat[e] = int(nas)
        buf["rew"][step] = torch.as_tensor(reward, device=device)
        buf["done"][step] = torch.as_tensor(done, dtype=torch.float32, device=device)   # THIS step's terminal flag
        _ol = bucketed_opt_len_np(obs_np["action_mask"], obs_np["opt_attr"])   # CPU-side, pre-H2D
        if state_trunc:
            _sk_np = state_keep_np(obs_np, net.split_heads)                     # CPU-side, pre-H2D
            _sk = None if _sk_np is None else torch.as_tensor(_sk_np, device=device)
        cur_obs = _to_tensors(obs_np, int_keys, device)
        cur_seat = torch.as_tensor(seat_np, device=device)
        for e, (d, info) in enumerate(zip(done, infos)):
            if d and "terminal" in info:
                eprs.append(max(info["terminal"].values()))     # winner's reward (~ 1 + victory margin)
                terms[(step, e)] = info["terminal"]             # both seats' rewards -> value diag
                if fw is not None and frozen_envs is not None and frozen_envs[e]:
                    t = info["terminal"]                        # wide-vs-frozen-base outcome, agent side
                    _as = int(_prev_as[e])
                    if t[_as] != t[1 - _as]:
                        fw.append(1 if t[_as] > t[1 - _as] else 0)
                if ksq is not None and ks_envs is not None and ks_envs[e]:
                    t = info["terminal"]                        # agent-vs-KS-specialist outcome
                    _as = int(_prev_as[e])
                    if t[_as] != t[1 - _as]:
                        ksq.append(1 if t[_as] > t[1 - _as] else 0)
            if d and mm is not None and "mm" in info:           # matchup-matrix event (full granularity)
                t = info.get("terminal")
                if t is None:
                    out = 3                                     # truncated (no terminal reward)
                elif t[0] > t[1]:
                    out = 0                                     # seat-0's deck won
                elif t[1] > t[0]:
                    out = 1                                     # seat-1's deck won
                else:
                    out = 2                                     # draw
                mm.append((info["mm"][0], info["mm"][1], out))
        if step_hook is not None:
            # buf rows for `step` are complete here; the SYNC teacher-prepass overlap
            # (train_selfplay) uses this to launch frozen-teacher forwards on a side stream
            # while the loop continues env-stepping. Inert (None) on every other path.
            step_hook(step)
    with torch.no_grad():
        boot_val = net.get_value(cur_obs, opt_len=_ol, state_keep=_sk)   # same obs the plans came from
        if train_opp and agent_seat is not None:
            # two-net: the bootstrap must come from the END obs's ACTOR -- B's value on B's rows
            # (A's value there is a spectator estimate the GAE would then sign-flip incorrectly
            # against B's own val stream).
            _bm_np = _seat_np == agent_seat
            if frozen_envs is not None:
                # frozen-stream envs: end-obs actor may be the FROZEN pilot -- bootstrap from A's
                # spectator value there (single-net convention), never from B (B never plays them)
                _bm_np = _bm_np | frozen_envs
            if ks_envs is not None:
                _bm_np = _bm_np | ks_envs
            _bm = torch.as_tensor(_bm_np, device=boot_val.device)
            b_boot = opp_net.get_value(cur_obs, opt_len=_ol, state_keep=_sk)
            boot_val = torch.where(_bm, boot_val, b_boot.to(boot_val.dtype))
    return boot_val, cur_seat, eprs, cur_obs, terms

