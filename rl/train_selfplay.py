"""Two-sided (all-seats) self-play PPO -- the C9 + C11 pipeline.

A SECOND trainer alongside ``rl.train`` (which is left untouched as the single-sided path). The live
net pilots BOTH seats (pure self-play, no frozen-snapshot rollout opponent); every decision is
collected, tagged with its seat. Returns are computed with a NEGAMAX GAE: because the same net plays
both sides and the value head predicts "outcome for the player to move", consecutive decisions from
DIFFERENT seats are opposite-perspective, so the bootstrap value (and the GAE carry) flip sign
whenever the acting seat changes. This propagates each game's terminal reward to BOTH seats correctly
without retro-assigning the loser's reward.

Carries the value-program features from ``rl.policy``/``rl.train`` (categorical value + CE loss,
value clipping, teacher-student distillation, register tokens, F19 graded reward) and adds:
  * C9  two-sided collection (this whole file);
  * C11 gated teacher promotion: periodically the current net plays an in-process both-orientation
    h2h vs the teacher; the teacher is replaced only when the current net wins >= --promote-winrate.

Run:  python -m rl.train_selfplay --decks top50 --value-categorical --teacher-kl-coef 0.005 \
          --teacher-value-coef 0.005 --promote-winrate 0.55 --num-envs 64 --total-timesteps 18000000
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import math
import os
import threading
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from contextlib import nullcontext
from types import SimpleNamespace

from .card_features import get_card_table
from .encoding import TokenEncoder
from .policy import (build_token_net, bucketed_opt_len, state_keep_np, bucket_for,
                     int32_safe_rows, _DEFAULT_OPT_BUCKETS,
                     MAX_OPTIONS as _P_MAXOPT, OPT_PICKED as _P_OPTPICK)

# --- deck-pool resolver + LR schedule (absorbed from the retired rl/train.py) ---
def resolve_deck_pool(name: str) -> list[list[int]]:
    """Map a --decks value to a list of decks (each side is sampled from this)."""
    sample = load_deck()  # engine sample deck (agent/deck.csv)
    if name == "all":
        return list(DECKS.values()) + [sample]
    if name in ("gen", "all+gen") and not GENERATED:   # import failed -> don't silently shrink the pool
        raise SystemExit("--decks gen/all+gen but rl/decks_generated.py is missing/empty "
                         "(run scripts/build_decks.py)")
    if name == "gen":                                  # 50 generated archetypes
        return list(GENERATED.values())
    if name == "all+gen":                              # official + sample + generated
        return list(DECKS.values()) + [sample] + list(GENERATED.values())
    if name == "official":
        return list(DECKS.values())
    if name == "sample":
        return [sample]
    if name in ("good", "real", "meta"):               # 4 official + real (Limitless + Kaggle-mined) decks
        try:
            from .decks_train import META                # consolidated deck sets (see decks_train.py)
        except Exception:
            META = {}
        return list(DECKS.values()) + list(META.values())
    if name in ("top15", "train15"):                   # top-15 Kaggle archetypes + 7 CL Aichi (decks_train.py)
        from .decks_train import TRAIN_TOP15
        return list(TRAIN_TOP15.values())
    if name in ("top50", "train50"):                   # top-50 Kaggle archetypes + 7 CL Aichi
        from .decks_train import TRAIN_TOP50
        return list(TRAIN_TOP50.values())
    if name in DECKS:
        return [DECKS[name]]
    if name in GENERATED:
        return [GENERATED[name]]
    raise SystemExit(f"unknown --decks '{name}' "
                     "(all|gen|all+gen|official|sample|good|top15|top50|<name>)")


def lr_at(it, decay_iters, base_lr, schedule, warmup_iters, min_ratio):
    """LR for iteration `it` (1-based): linear warmup, then linear/cosine decay over `decay_iters` to
    base_lr*min_ratio, then FLAT at the floor (Orbit-style: decay window decoupled from total budget;
    `decay_step` is clamped so past the window the LR holds at the floor). Pure function of `it` ->
    resume-safe (start_it continues the same curve). Shared by train + train_selfplay."""
    if warmup_iters > 0 and it <= warmup_iters:
        return base_lr * it / warmup_iters
    prog = min(max((it - warmup_iters) / max(1, decay_iters), 0.0), 1.0)   # clamp -> FLAT at floor after window
    decay = 0.5 * (1.0 + math.cos(math.pi * prog)) if schedule == "cosine" else (1.0 - prog)
    return base_lr * (min_ratio + (1.0 - min_ratio) * decay)

from .value_diag import DiagAccumulator
from .vec_env_selfplay import SelfPlayVecEnv


# v2.3 DECK-TOP: nets gain `decktop_emb` (zero-init, so a pre-v2.3 checkpoint is behaviourally
# IDENTICAL once loaded). Loading such a checkpoint must therefore tolerate exactly these missing
# keys -- and nothing else: a genuine arch mismatch (wrong d_model, renamed layer, unexpected key)
# must still raise, which plain strict=False would silently swallow.
_ADDITIVE_ZERO_INIT_KEYS = ("decktop_emb",)


def load_net_compat(net, sd, what=""):
    """load_state_dict, permitting ONLY the known additive zero-init params to be absent."""
    missing, unexpected = net.load_state_dict(sd, strict=False)
    bad_missing = [k for k in missing if not k.endswith(_ADDITIVE_ZERO_INIT_KEYS)]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"checkpoint/arch mismatch{' for ' + what if what else ''}: "
            f"missing={bad_missing} unexpected={list(unexpected)}")
    if missing:
        print(f"[compat] {what or 'net'}: pre-v2.3 checkpoint, zero-init {list(missing)} "
              f"(exact no-op at load)", flush=True)
    return missing


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--total-timesteps", type=int, default=18_000_000)
    p.add_argument("--num-envs", type=int, default=64)
    p.add_argument("--num-steps", type=int, default=256)
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--anneal-lr", action="store_true", default=True)
    p.add_argument("--lr-schedule", choices=["linear", "cosine"], default="cosine")
    p.add_argument("--warmup-steps", type=int, default=2_000_000,
                   help="linear LR warmup over this many GLOBAL env steps (FLAT, budget-invariant: "
                        "warmup exists to settle Adam moments + survive garbage early values, a "
                        "fixed cost -- %%-of-budget over-warms long runs and under-warms short ones). "
                        "Clamped to 20%% of the run")
    p.add_argument("--warmup-frac", type=float, default=0.0,
                   help="DEPRECATED %%-of-decay-window warmup; if >0 it OVERRIDES --warmup-steps "
                        "(keeps old launch lines byte-identical)")
    p.add_argument("--lr-min-ratio", type=float, default=0.1,
                   help="final LR = lr * this (schedule floor). Default 0.1: the old 0.02 floor froze the "
                        "last ~15-20%% of a 100M run (kl->0.0000) at full collection cost")
    p.add_argument("--decay-steps", type=int, default=None,
                   help="LR decays over this many GLOBAL steps then holds FLAT at lr*min_ratio (Orbit-style, "
                        "decoupled from the budget); None = 70%% of --total-timesteps (decaying over the "
                        "full budget wastes the tail -- see lr-min-ratio)")
    p.add_argument("--init-from", type=str, default=None,
                   help="checkpoint .pt to WARM-START net weights from (STRICT load, fresh optimizer/"
                        "LR schedule/step; the teacher starts as a copy) -- e.g. a BC net")
    p.add_argument("--lr-restart", action="store_true",
                   help="on --resume, RESTART the LR schedule from the resume point (fresh warmup+cosine) instead of "
                        "continuing it -> a warm restart to escape a plateau (use with --resume best.pt)")
    p.add_argument("--gamma", type=float, default=0.997,
                   help="0.997 = settled default (100M 4-arm A/B 2026-07-03: fixes gamma=0.99's early-"
                        "game target starvation, ~10-15%% better mid-game value mse; 1.0 equivalent)")
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--num-minibatches", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1,
                   help="accumulate this many consecutive minibatches per optimizer step "
                        "(samples/step = minibatch x world x N; each minibatch's loss is scaled "
                        "1/N so the step's gradient is the EXACT mean over the group, Orbit-style). "
                        "The noise-reduction lever that neither lowers lr below the ~1.5e-5 "
                        "mobility floor nor costs VRAM (bonus: N x fewer allreduces). Adv-norm "
                        "stays per-minibatch. num-minibatches must be divisible by N. "
                        "Default 1 = byte-identical legacy path")
    p.add_argument("--update-epochs", type=int, default=1,
                   help="1 = production default (speed stack; ep4 ~= ep1 at maturity per the epochs/"
                        "target_kl study, and the update dominates wall clock at ep4)")
    p.add_argument("--target-kl", type=float, default=None)
    p.add_argument("--critic-warmup-until", type=int, default=0,
                   help="CRITIC-FIRST RESTART (2026-07-13, restart-shock fix): while global_step "
                        "< this ABSOLUTE step, the update trains the VALUE function only (pg + "
                        "entropy losses zeroed; the teacher-kl/value terms stay and pin the "
                        "policy while the shared trunk moves). Use after a pool injection: the "
                        "value head mispredicts unseen decks -> large WRONG advantages shove the "
                        "policy (the mechanism behind the 5e-5/3e-5/2e-5 restart shocks at 3B). "
                        "Set to injection_base + ~30-50M. 0 = off")
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.01)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--vf-clip", type=float, default=None)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--weight-decay", type=float, default=0.0,
                   help="AdamW weight decay. Default 0.0 == plain Adam (the long-standing exact "
                        "setting). Nonzero is a FRESH-RUN knob (sweep arm sweep_wd001; the Orbit "
                        "Wars winner ran 0.01 on its 200M net) -- never flip mid-lineage")
    p.add_argument("--norm-adv", action="store_true", default=True)
    # net
    p.add_argument("--arch", default="transformer2")
    p.add_argument("--static", action="store_true")
    p.add_argument("--split-heads", action="store_true")
    p.add_argument("--would-ko", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--native-encode", action="store_true",
                   help="JSON-free collection: sdk_cg engine + GetBinaryObs + Cython native encode for BOTH "
                        "seats (byte-identical). Requires --no-would-ko + the built encode_native + "
                        "sdk_cg/libcg_custom.so. Training wall-clock only; inference unchanged.")
    p.add_argument("--value-categorical", action="store_true")
    p.add_argument("--value-atoms", type=int, default=51)
    p.add_argument("--value-vmax", type=float, default=1.0)
    p.add_argument("--bf16", action="store_true",
                   help="run the PPO update forward under bf16 autocast; loss upcast to fp32 (memory-bound update -> ~1.8x)")
    p.add_argument("--bf16-collect", action="store_true",
                   help="ALSO run the collection action-selection forward under bf16 autocast (logp/value "
                        "cast back to fp32 in the buffer). Measured NO-OP on d128 (the collect fwd is "
                        "launch-bound, not compute-bound) -- kept for bigger nets")
    p.add_argument("--state-trunc", action="store_true",
                   help="batch-max STATE-token truncation (gather-then-slice; policy.state_keep_np): "
                        "drop state tokens padded in EVERY row of the batch before the encoder. EXACT "
                        "(same argument as option truncation; no positional encoding). Cuts the "
                        "O(seq^2) attention + per-token FFN over the ~337-token state region to the "
                        "board's actual size (typ. ~40-60%% fewer state tokens mid-game)")
    p.add_argument("--compile-collect", action="store_true",
                   help="torch.compile(mode=reduce-overhead) the COLLECTION forward (logits_value; "
                        "sampling stays eager -> identical math). The b=96 d128 fwd is kernel-launch-"
                        "bound (~100 tiny kernels/step); cudagraph replay collapses it. Requires the "
                        "sync-free collect loop (bucketed_opt_len_np) -- graphs cannot contain syncs")
    p.add_argument("--async-collect", action="store_true",
                   help="overlap collection of the NEXT rollout (own thread + CUDA stream, a frozen "
                        "one-update-stale COPY of the net as the behavior policy) with THIS "
                        "iteration's update. Stored logp/val are the behavior net's, so the PPO "
                        "ratio absorbs the lag as ordinary importance sampling. Pays off when "
                        "update-time >= collect-time (20M+ nets): hides collection entirely")
    p.add_argument("--collect-graphs", action="store_true",
                   help="build DENSE cudagraphed collect twin(s) (varlen NJT is dynamo-untraceable): "
                        "torch.compile(mode=reduce-overhead) -> one cudagraph replay per barrier. "
                        "The collect forward is LAUNCH-bound (51ms/barrier vs ~2ms compute, 2026-07-13); "
                        "graphs collapse it. ASYNC: two double-buffered twins refreshed post-update "
                        "(stale=2). SYNC (2026-07-15): one twin refreshed from the live net every "
                        "iteration (stale=0). Collect numerics already diverge from the update path "
                        "under --bf16-collect (ratio noise); dense-vs-NJT is the same class. All "
                        "graphs are captured serially at startup, before any second thread exists")
    p.add_argument("--obs-shm", action="store_true",
                   help="workers write encoded obs into a shared-memory batch buffer instead of pickling "
                        "over the Pipe (main reads zero-copy) -> cuts collection IPC + np.stack.")
    p.add_argument("--envs-per-worker", type=int, default=1,
                   help="pack K envs per worker PROCESS (engine is multi-battle per process, proven "
                        "2026-07-13; the old 1-battle/process rule was a Python singleton). K=8 turns "
                        "768 procs/124 cores (~6x oversub, step barrier = slowest proc's scheduling "
                        "tail) into ~96 procs (~0.8x): the tail WAS the engine wall. Same envs, same "
                        "batch, same seeds -> no training-math change. Requires --native-encode")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the UPDATE forward (b>1, static shapes) -> ~1.5x update; "
                        "async collection has its own per-twin compile (--collect-graphs).")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--nlayers", type=int, default=3)
    p.add_argument("--ff", type=int, default=256)
    # reward / decks
    p.add_argument("--terminal-margin", type=float, default=0.0)
    p.add_argument("--decks", type=str, default="top50")
    # teacher-student (B6) + gated promotion (C11)
    p.add_argument("--teacher-kl-coef", type=float, default=0.0)
    p.add_argument("--teacher-value-coef", type=float, default=0.0)
    p.add_argument("--promote-strict", action="store_true",
                   help="finetune guarded promotion WITHOUT downward slack: teacher promotes only "
                        "at/above the best rolling ft_wr seen (user 2026-08-09). For curriculum "
                        "legs (per-leg --reset-ft-anchor bounds the stale-bar cost); leave off "
                        "for long stationary fts")
    p.add_argument("--reset-ft-anchor", action="store_true",
                   help="on resume, DISCARD the checkpointed ft_best guarded-promotion peak. "
                        "Guarded promotion assumes a STATIONARY opponent: it holds the teacher "
                        "back unless rolling ft_wr is within ~2sigma of the best ever seen. A "
                        "curriculum leg deliberately raises opponent strength, so ft_wr drops by "
                        "design and an inherited anchor would pin the teacher for the rest of the "
                        "run (the 2026-08-08 ftks1 failure: ft_best=0.485 from the buggy-ft_wr era "
                        "vs a real 0.37, promotion mathematically unreachable). Set at every rung "
                        "change; leave OFF for ordinary same-opponent resumes")
    p.add_argument("--ft-wr-stop", type=float, default=None,
                   help="CURRICULUM early exit: end the leg once the rolling ft_wr reaches this. "
                        "Valid only when the ft opponent IS the thing being measured (one "
                        "--ladder-decks-file deck + a fixed --opponent-ckpt), where inline ft_wr "
                        "== wr-vs-current-rung, already computed free from the collection stream. "
                        "The leg still stops at --total-timesteps if never reached")
    p.add_argument("--ft-wr-stop-hold", type=int, default=3,
                   help="--ft-wr-stop must hold for this many CONSECUTIVE logged iterations "
                        "before the leg ends (the n=2000 rolling window is noisy enough to "
                        "mis-rank checkpoints, so a single crossing is not evidence)")
    p.add_argument("--ft-wr-stop-min-n", type=int, default=2000,
                   help="--ft-wr-stop is ignored until the rolling window holds this many games")
    p.add_argument("--teacher-refresh", type=int, default=5_000_000,
                   help="steps between teacher refresh / promotion attempts. 5M = settled default: at "
                        "1M the 200-game gate ate 64%% of wall clock (172s gate per ~100s of training "
                        "at ~10k sps); 5M + the rank-split gate = ~3%% overhead")
    p.add_argument("--promote-winrate", type=float, default=None,
                   help="C11: replace the teacher only when the current net wins >= this in an in-process "
                        "h2h vs the teacher (None = unconditional time-based refresh)")
    p.add_argument("--promote-games", type=int, default=None,
                   help="C11: games for the promotion h2h; None = max(200, 5*len(pool)) -- generalist-aware "
                        "(the old fixed 80-100 is too noisy once the pool has 20-50 decks)")
    p.add_argument("--gate-decks-file", type=str, default=None,
                   help="json list of POOL INDICES: frozen promotion-gate slate (mirror schedule "
                        "drawn from this subset; training sampling untouched; ignored in ft mode)")
    p.add_argument("--gate-pairs", choices=["mirror", "cross"], default="mirror",
                   help="promotion-gate deck schedule: mirror = same deck both sides (pure piloting "
                        "probe); cross = round-robin cross-matchups in pilot-swap blocks (deck strength "
                        "cancels per block). Threshold guidance: --promote-winrate 0.6 at N>=200 is a "
                        "~3-sigma bar; 0.7 was the specialist/N=100 setting and stalls a generalist teacher")
    # (The in-run FIELD gate was REMOVED 2026-07-09, user call: its ratchet saturates >0.85, its
    # launch-frozen mix goes stale mid-run, and it cost ~10-13% wall-clock. Field-relative
    # measurement is OFFLINE now -- scripts/meta_gate.py snap-audits of periodic ckpts; in-run
    # progress = mirror parity gate + the matchup matrix (+ ft_wr vs the static opponent in
    # finetune mode, read straight from the collection stream).)
    # DECK FINETUNE (post-PPO specialization; see rl/env_finetune.py): the live net always pilots
    # --finetune-deck (random seat per episode); the other seat is a FROZEN net (--opponent-ckpt,
    # GPU-batched in collect) piloting --opponent-decks at observed-meta weights. Collection
    # becomes single-agent (constant surfaced seat -> GAE unchanged). Keep the teacher anchor ON;
    # judge ships ONLY by offline audits (Goodhart: training opponent == the static net).
    p.add_argument("--finetune-deck", type=str, default=None,
                   help="deck name (TRAIN_TOP50/META2) the live net pilots EVERY episode; also becomes "
                        "the gate pool")
    p.add_argument("--opponent-ckpt", type=str, default=None,
                   help="frozen ckpt piloting the opponent seat during finetune collection")
    p.add_argument("--cross-gate-delta", type=float, default=0.03,
                   help="two-net cross-gate promotion margin: promote a net when its live-vs-other-"
                        "teacher wr beats the frozen-vs-frozen baseline by this much (the baseline "
                        "replaces an absolute bar because the asymmetric deck matchup EV != 0.5)")
    p.add_argument("--ft-balance-mode", choices=["off", "hard", "soft", "turns"], default="off",
                   help="TWO-NET train-the-loser balancing (user design 2026-08-02, the fix for "
                        "the winner-snowball instability: the advantaged side farms wins, the "
                        "loser's stream becomes loss-dominated, V->-1, advantages vanish, policy "
                        "decays). Per iteration the rolling live A-vs-B wr (all-reduced across "
                        "ranks) gates updates: wr above the band -> A is gated, below -> B is "
                        "gated. hard = the gated net's ENTIRE loss zeroed; soft = only its "
                        "pg/entropy zeroed (value + teacher terms keep training). turns = "
                        "promotion-driven strict alternation (2026-08-03 v2: one net trains "
                        "per phase; phase ends when the trainee earns its cross-gate "
                        "promotion, --ft-phase-timebox backstop; band ignored). off = "
                        "numerically identical to the historical path")
    p.add_argument("--ft-balance-band", type=str, default="0.45,0.55",
                   help="lo,hi rolling-wr band for --ft-balance-mode hard/soft (inside = both "
                        "train). RETIRED in turns mode (2026-08-03 ceiling audit: the inline wr "
                        "floor is mixture-structural, absolute bands are uncalibratable)")
    p.add_argument("--ft-phase-timebox", type=float, default=120e6,
                   help="turns mode backstop: a phase that cannot EARN its promotion within this "
                        "many steps flips anyway (the opponent keeps facing the last PROMOTED "
                        "snapshot, so unverified drift never becomes the target)")
    p.add_argument("--two-net-finetune", action="store_true",
                   help="finetune with a LIVE opponent net (user design 2026-07-29): --opponent-ckpt "
                        "seeds net B, which TRAINS on the opponent rows (~amask) with its own AdamW -- "
                        "A specializes 'ft-deck vs all', B specializes 'all vs ft-deck'. B samples "
                        "(not greedy); each net's pg/ent/value train on its OWN rows; BOTH nets carry "
                        "teacher anchors (B's = t_b, same kl/value coefs; 2026-08-03 leak-audit order). "
                        "ft_wr becomes the A-vs-B curriculum diagnostic (NON-stationary "
                        "-- judge ships by offline audits only). Sync collect only; eager opp forward.")
    p.add_argument("--ft-teachers-from", type=str, default=None,
                   help="ckpt path: after init/resume, load BOTH teachers (T_A and t_b) from this "
                        "ckpt's net -- 'teachers start as the generalist' (user 2026-08-03). Use on "
                        "two-net (re)starts; standard promotions still refresh them afterwards")
    p.add_argument("--frozen-wide-envs", type=float, default=0.0,
                   help="two-net ft recipe v2 (user 2026-08-06): this fraction of envs samples "
                        "opponents from --frozen-wide-decks piloted by the FROZEN base (greedy, "
                        "single-net semantics); live B pilots only the remaining envs' concentrated "
                        "mixture -- B's whole capacity goes to the adversarial job. Those rows are "
                        "off-policy for both nets (fmask); A still trains policy+value on its own "
                        "rows everywhere. Emits fw_wr (wide-vs-base) telemetry. 0 = off")
    p.add_argument("--frozen-wide-decks", type=str, default=None,
                   help="ladder-decks json for the frozen-wide env stream (e.g. the v2 builder's "
                        "_wide.json); --ladder-decks-file should then be the concentrated mixture")
    p.add_argument("--frozen-wide-ckpt", type=str, default=None,
                   help="OVERRIDE the frozen-wide stream's pilot net (default: the --opponent-ckpt "
                        "base). SHIP-FT KS STREAM (user 2026-08-09): point the stream at a "
                        "SPECIALIST pilot -- e.g. --frozen-wide-decks ks_agent_113.json + this at "
                        "ftks1@380M gives a 10%% KS stream piloted by a competent KS net instead "
                        "of the crown (whose KS play is the 626-vs-1365-LB artifact). Built from "
                        "the ckpt's own net_config; v2.2/v2.3 compat via load_net_compat")
    p.add_argument("--frozen-wide-sample", action="store_true",
                   help="frozen-wide stream SAMPLES its picks instead of greedy argmax. Use for "
                        "specialist-pilot streams: a deterministic opponent is both weaker and "
                        "an easier overfit target")
    # SECOND frozen stream (user 2026-08-10): wide 20%% + KS 10%% simultaneously. Same partition
    # mechanics as frozen-wide; always SAMPLED (it exists for specialist pilots). Emits ks_wr.
    p.add_argument("--frozen-ks-envs", type=float, default=0.0,
                   help="fraction of envs sampling --frozen-ks-decks piloted by --frozen-ks-ckpt "
                        "(sampled picks). Stacks with --frozen-wide-envs; both rows are off-policy "
                        "for A and (two-net) excluded from B via fmask. 0 = off")
    p.add_argument("--frozen-ks-decks", type=str, default=None,
                   help="ladder-decks json for the KS stream (e.g. ks_opp_113_ladder.json)")
    p.add_argument("--frozen-ks-ckpt", type=str, default=None,
                   help="pilot net for the KS stream (the matched KS specialist: kscurr's net for "
                        "the alakazam ft, kscurrdk's for dusknoir)")
    p.add_argument("--ft-start-turn", choices=["A", "B"], default="B",
                   help="turns mode: which side trains FIRST on a fresh start (user 2026-08-06: "
                        "A-first = phase 1 is the proven single-net recipe vs the frozen base and "
                        "the ship artifact improves from step 0; B-first = the historical default, "
                        "sharpen the adversary before A specializes). Resumes keep the "
                        "checkpointed turn state; this only seeds fresh runs")
    p.add_argument("--opponent-decks", type=str, default=None,
                   help="name=weight list for the frozen opponent's decks")
    # ALL-DECKS distributions (rl/decks_ladder.py; file from scripts/mine_ladder_decks.py):
    p.add_argument("--ladder-decks-file", type=str, default=None,
                   help="ladder_decks.json. With --decks ladder_all the GENERALIST pool = UNIFORM "
                        "over all mined lists (breadth, no meta snapshot). In finetune mode (if "
                        "--opponent-decks not given): a DESIGNED mixture file (F_now/F_counter/"
                        "F_wide counts = sampling mass; build_ft_*.py), sampled at "
                        "mixed_weights(counts, --opponent-uniform-mix); run mix 0.0 so the file "
                        "IS the distribution")
    p.add_argument("--compile-mode", type=str, default="default",
                   choices=["default", "max-autotune", "reduce-overhead"],
                   help="torch.compile mode for the UPDATE forward (sps probe 2026-07-10)")
    p.add_argument("--varlen-attn", action="store_true",
                   help="jagged (NJT) flash attention over per-row PACKED tokens instead of "
                        "padded+key-padding-mask (which forces the slow mem-efficient kernel AND "
                        "the int32 row cap). EXACT (no positional encoding; the b=1 path's argument, "
                        "batched; tests/test_varlen_attn.py). Applied to net+collect+teacher "
                        "together; --collect-graphs/--bf16-collect give the collect twins their own "
                        "numerics (behavior-policy data; ratios absorb it). ~2.2x attn (A100 probe)")
    p.add_argument("--sched-anchor-step", type=int, default=-1,
                   help="with --lr-restart: anchor the restarted LR schedule at this ABSOLUTE global "
                        "step instead of the current resume point. Fixes the restart-chain bug "
                        "(2026-07-10): every chain leg / self-heal restart re-anchored at ITS OWN "
                        "resume step, pinning LR at peak forever and pumping entropy (0.65->0.84). "
                        "-1 = use the anchor persisted in the resume ckpt (sched_anchor), falling "
                        "back to the resume point (and persisting it) if absent")
    p.add_argument("--async-serial-iters", type=int, default=3,
                   help="ASYNC: run this many iterations fully SYNCHRONOUS before pipelining. Dynamo "
                        "is not thread-safe: the iter-1 update compile + the first automatic-dynamic "
                        "re-specializations FX-trace on the main thread, and a collector thread "
                        "calling its compiled forward in that window crashes (S3 stud-3 2026-07-10; "
                        "hopper only survived on warm inductor cache). 3 covers auto-dynamic settling")
    p.add_argument("--teacher-prepass-collect", action="store_true",  # sync mode (2026-07-15): chunked onto a side stream during collection (frozen teacher -> exact)
                   help="ASYNC only: run the frozen teacher's no-grad pre-pass over the freshly "
                        "collected buffer INSIDE the collector thread (hides its ~2-3s under the "
                        "update). EXACT: same fp32 forward on the same rows, just scheduled earlier; "
                        "teacher promotion takes a lock so weights can't swap mid-forward. No-op "
                        "without --async-collect / without teacher terms")
    p.add_argument("--deck-weights-file", type=str, default=None,
                   help=".npy of per-pool-deck sampling weights (tf-idf density design 2026-07-10: "
                        "75%% ladder / 25%% curated tier split, inverse idf-cosine-density within "
                        "tier, 0.2x-8x floor/cap). None = uniform sampling (byte-identical rng)")
    p.add_argument("--opponent-uniform-mix", type=float, default=0.5,
                   help="finetune opponent distribution = (1-x)*band-blend-meta + x*UNIFORM over all "
                        "mined decks. The uniform mass is late-competition insurance against a meta "
                        "shift there is no time to adapt to; raise toward 1.0 near the deadline")
    # value diagnostics (rl/value_diag.py): distance-to-terminal buckets of sign-acc/MSE from the
    # rollout buffer itself (free -- no extra games). Full metrics -> <out>/value_diag.jsonl.
    p.add_argument("--value-diag", action=argparse.BooleanOptionalAction, default=True,
                   help="log per-bucket value quality (sign-acc vs outcome by steps-to-terminal)")
    p.add_argument("--diag-every", type=int, default=20,
                   help="iterations per [vdiag] summary window (accumulated in between)")
    # infra
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ddp", action="store_true",
                   help="data-parallel across GPUs (torchrun): each rank runs its own two-sided envs; grads "
                        "all-reduced; teacher promotion decided on rank0 and broadcast. world=1 when off.")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--resume", type=str, default=None,
                   help="latest.pt to FULLY resume: net+optimizer+teacher+global_step (continues the LR schedule)")
    p.add_argument("--out", type=str, default=os.path.join(os.environ.get("HOME", "."), "pkmn_runs", "selfplay"))
    p.add_argument("--save-every", type=int, default=500_000)
    p.add_argument("--log-every", type=int, default=1)
    return p.parse_args()


# Extracted modules (re-exported here so existing imports keep working: tests import
# compute_gae_2s/collect_rollout/_to_tensors/_gate_schedule; offline field tools import
# field_winrate from rl.gates directly).
from .gates import _gate_schedule, _net_greedy_pick, gated_winrate  # noqa: F401
from .rollout import _STEP_T, _to_tensors, collect_rollout, compute_gae_2s, make_buffers  # noqa: F401



def main():
    args = parse_args()
    args.batch_size = args.num_envs * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    if args.grad_accum < 1 or args.num_minibatches % args.grad_accum != 0:
        raise SystemExit(f"--grad-accum ({args.grad_accum}) must be >=1 and divide "
                         f"--num-minibatches ({args.num_minibatches}): a partial trailing group "
                         f"would step on an underweighted gradient")
    if args.value_categorical and args.terminal_margin > 0 and args.value_vmax < 1.0 + args.terminal_margin:
        raise SystemExit(f"--value-vmax ({args.value_vmax}) < 1+terminal_margin ({1.0 + args.terminal_margin}): "
                         f"graded returns would be clamped, miscalibrating the categorical value targets. "
                         f"Set --value-vmax >= {1.0 + args.terminal_margin}.")
    # ---- DDP (data-parallel) setup; world=1 when not --ddp ----
    ddp = args.ddp
    if ddp:
        # 60-min collective timeout (default ~10): rank0 plays the gated-promotion h2h in-process
        # (b=1 forwards, up to --promote-games sequential games) while other ranks block in the
        # teacher-broadcast -- a slow gate must surface as slow, not as an opaque NCCL watchdog kill.
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(minutes=60))
        rank, world = dist.get_rank(), dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        rank, world, local_rank = 0, 1, 0
        device = torch.device(args.device)
    is_main = rank == 0
    if is_main:
        os.makedirs(args.out, exist_ok=True)
        print(f"[cfg] {vars(args)}", flush=True)
        if ddp:
            print(f"[ddp] world={world} -> effective batch {args.batch_size * world}", flush=True)
    torch.manual_seed(args.seed + rank); np.random.seed(args.seed + rank)   # per-rank -> diverse envs
    torch.backends.cuda.matmul.allow_tf32 = True     # TF32 tensor cores for fp32 matmuls (Ampere+/Blackwell):
    torch.backends.cudnn.allow_tf32 = True           # large matmul speedup at negligible training-quality cost

    ct = get_card_table()
    enc = TokenEncoder(ct)
    net_config = {"arch": args.arch, "d_model": args.d_model, "nhead": args.nhead,
                  "nlayers": args.nlayers, "ff": args.ff, "static": args.static,
                  "would_ko": args.would_ko, "split_heads": args.split_heads,
                  "value_categorical": args.value_categorical,
                  "value_atoms": args.value_atoms, "value_vmax": args.value_vmax}
    # GENERALIST pool: named pool, or 'ladder_all' = UNIFORM over every mined ladder list
    if args.decks == "ladder_all":
        if not args.ladder_decks_file:
            raise SystemExit("--decks ladder_all needs --ladder-decks-file (mine_ladder_decks.py)")
        from .decks_ladder import load_ladder_decks
        pool, _lcounts = load_ladder_decks(args.ladder_decks_file)
        if is_main:
            print(f"[decks] ladder_all -> {len(pool)} mined lists, UNIFORM (no meta snapshot)", flush=True)
    else:
        pool = resolve_deck_pool(args.decks)
    # DECK-FINETUNE mode: fixed agent deck vs a frozen opponent on weighted meta decks. The pool
    # collapses to the ship deck.
    frozen_opp = None
    if args.finetune_deck or args.opponent_ckpt:
        if not (args.finetune_deck and args.opponent_ckpt):
            raise SystemExit("finetune mode needs BOTH --finetune-deck and --opponent-ckpt")
    if args.two_net_finetune:
        if not args.opponent_ckpt:
            raise SystemExit("--two-net-finetune requires finetune mode (--finetune-deck + --opponent-ckpt)")
        if args.async_collect:
            raise SystemExit("--two-net-finetune: sync collect only (pilot scope)")
        if args.value_categorical or args.vf_clip is not None:
            raise SystemExit("--two-net-finetune: plain MSE value only (pilot scope)")
    # NB the construction below must run for BOTH ft modes -- a 2026-07-30 edit accidentally
    # nested it under the two-net validation, which silently degraded single-net finetune to
    # generalist self-play (no [finetune] banner, no ft_wr, pool stayed the ladder file).
    if args.finetune_deck:
        from .decks_train import TRAIN_TOP50 as _NAMED

        def _resolve_deck(nm):
            """named deck, or 'csv:<path>' = a 60-int ship-deck csv (e.g. deck_d1145.csv)."""
            nm = nm.strip()
            if nm.startswith("csv:"):
                import csv as _csv
                with open(nm[4:]) as f:
                    deck = [int(x) for row in _csv.reader(f) for x in row if x.strip()]
                assert len(deck) == 60, f"{nm}: {len(deck)} cards, want 60"
                return deck
            return _NAMED[nm]

        def _resolve_deck_set(nm):
            """'json:<path>' -> a deck SET for the agent side (KS-family finetune, user
            2026-08-08). File: {"decks": [[60 ints], ...], "weights": [...], "ids": [...]}.
            `ids` are real pool indices so the matchup log stays per-deck readable -- we need
            that to build the step-2 counter map with a COMPETENT pilot. Returns None for the
            ordinary single-deck specs so back-compat is exact."""
            nm = nm.strip()
            if not nm.startswith("json:"):
                return None
            j = json.load(open(nm[5:]))
            ds = [[int(x) for x in d] for d in j["decks"]]
            assert ds, f"{nm}: empty deck set"
            for d in ds:
                assert len(d) == 60, f"{nm}: a deck has {len(d)} cards, want 60"
            ws = [float(x) for x in (j.get("weights") or [1.0] * len(ds))]
            ids = [int(x) for x in (j.get("ids") or [-1] * len(ds))]
            assert len(ws) == len(ds) and len(ids) == len(ds), f"{nm}: weights/ids length mismatch"
            return ds, ws, ids

        def _parse_named(spec):
            ds, ws = [], []
            for kv in spec.split(","):
                nm, w = kv.split("=")
                ds.append(_resolve_deck(nm)); ws.append(float(w))
            return ds, ws

        # opponent source priority: explicit list > designed mixture file
        if args.opponent_decks:
            fo_decks, fo_w = _parse_named(args.opponent_decks)
            src = f"{len(fo_decks)} weighted meta decks (--opponent-decks)"
        elif args.ladder_decks_file:
            # designed F_now/F_counter/F_wide counts file; uniform_mix 0.0 = the file IS the mix
            from .decks_ladder import load_ladder_decks, mixed_weights
            fo_decks, _cnt = load_ladder_decks(args.ladder_decks_file)
            fo_w = mixed_weights(_cnt, args.opponent_uniform_mix).tolist()
            src = f"{len(fo_decks)} mixture decks (uniform_mix={args.opponent_uniform_mix})"
        else:
            raise SystemExit("finetune mode needs --opponent-decks or --ladder-decks-file")
        _aset = _resolve_deck_set(args.finetune_deck)
        if _aset:
            _ad, _aw, _aid = _aset
            frozen_opp = {"agent_deck": _ad[0], "agent_decks": _ad, "agent_weights": _aw,
                          "agent_ids": _aid, "opp_decks": fo_decks, "opp_weights": fo_w}
        else:
            frozen_opp = {"agent_deck": _resolve_deck(args.finetune_deck),
                          "opp_decks": fo_decks, "opp_weights": fo_w}
        if args.frozen_wide_envs > 0:
            # 2026-08-09: single-net fts may carry a frozen stream too (the ship-ft KS stream);
            # the old two-net-only gate guarded B-leak semantics that single-net doesn't have
            # (fmask is inert in the single-net update -- p_fm is consumed only under opt_b).
            if not args.frozen_wide_decks:
                raise SystemExit("--frozen-wide-envs needs --frozen-wide-decks")
            if args.async_collect:
                raise SystemExit("--frozen-wide-envs is sync-collect only")
        if args.frozen_ks_envs > 0:
            if not (args.frozen_ks_decks and args.frozen_ks_ckpt):
                raise SystemExit("--frozen-ks-envs needs --frozen-ks-decks and --frozen-ks-ckpt")
            if args.async_collect:
                raise SystemExit("--frozen-ks-envs is sync-collect only")
            if args.frozen_wide_envs + args.frozen_ks_envs > 0.5:
                raise SystemExit("frozen streams exceed 50%% of envs -- the conc mixture would starve")
            from .decks_ladder import load_ladder_decks as _lld, mixed_weights as _mw
            _wd, _wcnt = _lld(args.frozen_wide_decks)
            _n_wide = int(round(args.frozen_wide_envs * args.num_envs))
            frozen_opp.update({"wide_decks": _wd, "wide_weights": _mw(_wcnt, 0.0).tolist(),
                               "wide_flags": [i < _n_wide for i in range(args.num_envs)]})
            if is_main:
                print(f"[frozen-wide] {_n_wide}/{args.num_envs} envs sample "
                      f"{args.frozen_wide_decks} ({len(_wd)} decks) piloted by the FROZEN base; "
                      f"live B pilots the other {args.num_envs - _n_wide} envs only", flush=True)
        if args.frozen_ks_envs > 0:
            # SECOND stream: envs [wide .. wide+ks) -- disjoint from the wide block by layout
            from .decks_ladder import load_ladder_decks as _lldk, mixed_weights as _mwk
            _kd, _kcnt = _lldk(args.frozen_ks_decks)
            _n0 = int(round(args.frozen_wide_envs * args.num_envs))
            _n_ks = int(round(args.frozen_ks_envs * args.num_envs))
            frozen_opp.update({"ks_decks": _kd, "ks_weights": _mwk(_kcnt, 0.0).tolist(),
                               "ks_flags": [_n0 <= i < _n0 + _n_ks for i in range(args.num_envs)]})
            if is_main:
                print(f"[frozen-ks] {_n_ks}/{args.num_envs} envs sample {args.frozen_ks_decks} "
                      f"({len(_kd)} decks) piloted by {args.frozen_ks_ckpt} (SAMPLED)", flush=True)
        # deck-SET mode gates on EVERY agent deck (user 2026-08-08 "on all decks"), so n_gate
        # below scales with the set and the gate cannot be dominated by one list.
        pool = frozen_opp.get("agent_decks") or [frozen_opp["agent_deck"]]
        if is_main:
            _what = (f"deck SET '{args.finetune_deck}' ({len(pool)} decks, sampled per episode)"
                     if frozen_opp.get("agent_decks") else f"agent deck '{args.finetune_deck}'")
            print(f"[finetune] {_what} EVERY episode (random seat) vs "
                  f"FROZEN {args.opponent_ckpt} (GPU-batched in collect) on {src}", flush=True)
    # promotion-gate size: explicit --promote-games, else generalist-aware auto (SE ~0.032 at 250)
    n_gate = args.promote_games if args.promote_games else max(200, 5 * len(pool))
    if is_main and frozen_opp is None:
        print(f"[decks] pool='{args.decks}' -> {len(pool)} deck(s); BOTH seats sampled (symmetric self-play)", flush=True)

    deck_weights = None
    if args.deck_weights_file:
        deck_weights = np.load(args.deck_weights_file).tolist()
        if len(deck_weights) != len(pool):
            raise SystemExit(f"--deck-weights-file has {len(deck_weights)} weights "
                             f"but the pool has {len(pool)} decks (stale sidecar?)")
        if is_main:
            print(f"[decks] weighted sampling: {args.deck_weights_file} "
                  f"(range {min(deck_weights)*len(pool):.2f}x..{max(deck_weights)*len(pool):.2f}x uniform)",
                  flush=True)
    # PROMOTION-GATE SLATE (user design 2026-07-24): optional frozen subset of pool indices
    # for the gate's mirror schedule -- concentrates the 800 gate games on decks where skill
    # expresses (the full-pool schedule wasted ~17%% of games on sub-0.45 tail mirrors).
    # Training sampling is UNTOUCHED; tch_wr changes meaning at the deploy boundary.
    gate_pool = pool
    if args.gate_decks_file and frozen_opp is None:
        _gidx = json.load(open(args.gate_decks_file))
        gate_pool = [pool[i] for i in _gidx]
        if is_main:
            print(f"[gate] slate: {len(gate_pool)} decks from {args.gate_decks_file}", flush=True)
    envs = SelfPlayVecEnv(args.num_envs,
                          {"decks": pool, "would_ko": args.would_ko, "terminal_margin": args.terminal_margin,
                           "native_encode": args.native_encode,
                           **({"deck_weights": deck_weights} if deck_weights is not None else {}),
                           **({"frozen_opp": frozen_opp} if frozen_opp else {})},
                          base_seed=args.seed * 1000 + rank * args.num_envs,   # per-rank seed range -> distinct envs
                          obs_shm=args.obs_shm, envs_per_worker=args.envs_per_worker)
    net = build_token_net(ct, net_config).to(device)
    if args.init_from:                                   # WARM-START weights (e.g. a BC net): STRICT
        ick = torch.load(args.init_from, map_location=device)   # load, fresh optimizer/schedule; the
        load_net_compat(net, ick["net"], "init-from")            # teacher below copies these weights
        if is_main:
            print(f"[init] warm-started from {args.init_from} "
                  f"(bc_val_acc={ick.get('bc_val_acc')})", flush=True)
    def _broadcast_net():
        """Rank-0's params+buffers to every rank (identical weights after any load)."""
        for p in net.parameters():
            dist.broadcast(p.data, src=0)
        for b in net.buffers():
            dist.broadcast(b.data, src=0)

    if ddp:                                              # identical initial weights across ranks
        _broadcast_net()
    opt = optim.AdamW(net.parameters(), lr=args.lr, eps=1e-5, weight_decay=args.weight_decay,
                      fused=(device.type == "cuda"))   # default wd=0 -> identical math to Adam; fused kernel
    envs.pin_obs_buffers()                             # DMA H2D for the shm obs (best-effort; needs CUDA up)
    if is_main:
        print(f"[net] params={sum(p.numel() for p in net.parameters()):,}", flush=True)

    if args.varlen_attn:
        net.varlen_attn = True     # teacher/collect/opp nets get the same flag at construction below
        # dynamo CANNOT trace NJT ops (torch 2.11: SymNode.nested_int_coeff AttributeError at
        # compile, S6a 2026-07-10) -> no FULL-graph compile under varlen. Measured: eager+varlen
        # BEATS compiled+padded on the update (~9.5-11.5s vs 15.7s, stud-3 20M). --compile is
        # reinterpreted as FLAT-SEGMENT compile (qkv + post-attn blocks; see policy._encoder_njt):
        # inductor fusion for the FFN-dominated FLOPs with the NJT plumbing kept eager.
        net.varlen_compile = bool(args.compile)
        if args.compile or args.compile_collect:
            args.compile = False
            args.compile_collect = False
            if is_main:
                print("[varlen] full-graph compile OFF (dynamo cannot trace NJT); flat segments "
                      + ("COMPILED (dynamic=True, per-layer graphs)" if net.varlen_compile
                         else "eager"), flush=True)
        if is_main:
            print("[varlen] jagged NJT attention ON (net+teacher"
                  + ("" if args.collect_graphs else "+collect") + ")", flush=True)
    use_teacher = args.teacher_kl_coef > 0.0 or args.teacher_value_coef > 0.0
    teacher_net = None
    t_lock = threading.Lock()      # guards teacher weights vs the collector's pre-pass (async)
    t_epoch = {"n": 0}             # bumped on every teacher swap -> stale collector pre-pass detection
    if use_teacher:
        teacher_net = build_token_net(ct, net_config).to(device)
        teacher_net.load_state_dict(net.state_dict()); teacher_net.eval()
        teacher_net.varlen_attn = args.varlen_attn
        teacher_net.varlen_compile = net.varlen_compile if args.varlen_attn else False
        for tp in teacher_net.parameters():
            tp.requires_grad_(False)
        if is_main:
            print(f"[teacher] B6 on (kl={args.teacher_kl_coef} value={args.teacher_value_coef}); "
                  f"refresh/promote every {args.teacher_refresh} steps"
                  + ("" if args.promote_winrate is None else
                     f"; GATED >= {args.promote_winrate} over {n_gate} {args.gate_pairs} games "
                     f"(fixed schedule)"), flush=True)

    ft_anchor = {"best": -1.0}   # finetune guarded-promotion peak ft_wr (persisted across legs)
    resume_step = 0
    if args.resume:                                         # FULL resume: net + optimizer + teacher + step
        rck = torch.load(args.resume, map_location=device)
        load_net_compat(net, rck["net"], "resume")          # arch mismatch still raises
        if ddp:                                             # keep all ranks identical after the load
            _broadcast_net()
        if rck.get("opt") is not None:
            opt.load_state_dict(rck["opt"])
        resume_step = int(rck.get("global_step", 0))
        if teacher_net is not None:
            tsd = rck.get("teacher")
            load_net_compat(teacher_net, tsd if tsd is not None else net.state_dict(), "teacher")
            teacher_net.eval()
        if args.reset_ft_anchor:                              # curriculum: the objective moved
            if is_main:
                print(f"[ft-anchor] RESET: discarding checkpointed ft_best="
                      f"{rck.get('ft_best')} (new opponent rung)", flush=True)
        else:
            ft_anchor["best"] = float(rck.get("ft_best", -1.0))   # guarded peak survives legs
        if is_main:
            parts = ["net",
                     "opt" if rck.get("opt") is not None else "opt=FRESH",
                     "teacher" if rck.get("teacher") is not None else "teacher=net-clone"]
            print(f"[resume] {args.resume}: {'+'.join(parts)} at step {resume_step}", flush=True)

    # torch.compile the UPDATE forward only (b>1, static minibatch shapes); async collection has
    # its own per-twin compiles. Shares net params by reference -> DDP all-reduce + opt.step() unchanged.
    if args.compile or args.compile_collect or args.collect_graphs:
        # dynamo's recompile budget defaults to 8 PER CODE OBJECT and is SHARED by the update and
        # collect wrappers (both wrap net.logits_value). Our legitimate static shapes -- 4 opt_len
        # buckets x {collect,update} batch sizes x (with --state-trunc) ~11 keep lengths -- exceed
        # it, after which NEW shapes silently fall back to EAGER (correct but slow, no warning).
        # torch>=2.6 names it recompile_limit, older cache_size_limit; the config module
        # raises on unknown attrs (helios torch 2.5.1 aarch64), so set only what exists.
        for _attr in ("recompile_limit", "cache_size_limit"):
            if hasattr(torch._dynamo.config, _attr):
                setattr(torch._dynamo.config, _attr, 64)
    update_lv = net.logits_value
    if args.compile:
        _cm = None if args.compile_mode == "default" else args.compile_mode
        # dynamic stays False: dynamic=True inductor-compiles the update with symbolic shapes and
        # dies outright (InductorError: AssertionError, torch 2.11 + this graph, 2026-07-10). The
        # async thread-race with mid-run re-traces is handled on the COLLECTOR side instead
        # (per-twin eager fallback + --async-serial-iters).
        update_lv = torch.compile(net.logits_value, dynamic=False, mode=_cm)
        if is_main:
            print(f"[compile] update forward compiled (torch.compile dynamic=False mode={args.compile_mode})",
                  flush=True)
    collect_lv = None
    if args.compile_collect:
        # cudagraph-replay the launch-bound b=num_envs collection forward; one graph per opt_len
        # bucket (bounded set). Sampling stays eager on the returned logits (no RNG in the graph).
        collect_lv = torch.compile(net.logits_value, mode="reduce-overhead", dynamic=False)
        if is_main:
            print("[compile] collect forward compiled (mode=reduce-overhead)", flush=True)

    shapes = enc.shapes
    buf = make_buffers(args.num_steps, args.num_envs, shapes, enc.int_keys, device)

    # FINETUNE: frozen opponent net on the GPU (batched picks for the non-agent seat in collect)
    opp_net = None
    opp_lv = None
    agent_seat_np = None
    if frozen_opp is not None:
        ock = torch.load(args.opponent_ckpt, map_location="cpu")
        opp_net = build_token_net(ct, ock["net_config"]).to(device)
        load_net_compat(opp_net, ock["net"], "opponent-ckpt"); opp_net.eval()
        opp_net.varlen_attn = args.varlen_attn
        opp_net.varlen_compile = net.varlen_compile if args.varlen_attn else False
        if not args.two_net_finetune:                       # two-net: B is LIVE (trains on ~amask rows)
            for op_ in opp_net.parameters():
                op_.requires_grad_(False)
        if (args.compile_collect or args.collect_graphs) and not args.two_net_finetune:
            # frozen weights never refresh -> one plain reduce-overhead compile is enough (no
            # twin/refresh machinery); rollout calls it on the FULL batch so shapes stay inside
            # the opt_len-bucket space the live collect already warms (a per-step row subset
            # would retrace on every opponent count). DENSE forward, same as the sync twin:
            # the varlen NJT path is not dynamo-traceable (SymNode nested_int_coeff, the
            # 9fe190a bug class) -- dense-vs-varlen is argmax-epsilon on a frozen opponent.
            opp_net.varlen_attn = False
            opp_net.varlen_compile = False
            _ofn = torch.compile(opp_net.logits_value, mode="reduce-overhead", dynamic=False)
            _ofb = [0]
            def _o_safe(o, opt_len=None, state_keep=None, _c=_ofn, _e=opp_net.logits_value):
                try:
                    return _c(o, opt_len=opt_len, state_keep=state_keep)
                except Exception as ex:
                    _ofb[0] += 1
                    if _ofb[0] <= 3:
                        print(f"[opp-compile] fallback #{_ofb[0]}: {type(ex).__name__}: {str(ex)[:220]}",
                              flush=True)
                    return _e(o, opt_len=opt_len, state_keep=state_keep)
            opp_lv = _o_safe
            if is_main:
                print("[compile] frozen-opponent forward compiled (full-batch, reduce-overhead)", flush=True)

    # TWO-NET FINETUNE: B gets its own optimizer (same recipe as A's) and, on resume, its own
    # saved state. 2026-07-31 sps work: B is now compiled on BOTH hot paths (the pilot ran it
    # eager everywhere and cost 2.1x production per step -- update 13.7s, collect fwd 15.3s on
    # 4xGH200). The "capture-once" worry that kept B eager was wrong: graph replay reads
    # parameter STORAGE, and in-place refresh/opt-step keeps pointers valid -- the same
    # contract the live actor's sync twin has run in production since Jul 22.
    opt_b = None
    update_lv_b = None
    t_b_net = None                                          # B's frozen teacher (cross-gate reference)
    cg_state = {"base": None, "promos_a": 0, "promos_b": 0, "turn": args.ft_start_turn}
    # fresh runs seed the turn from --ft-start-turn; resumes overwrite via the checkpointed
    # cg_state below, so mid-run restarts never reset the cycle
    if args.two_net_finetune:
        opt_b = optim.AdamW(opp_net.parameters(), lr=args.lr, eps=1e-5,
                            weight_decay=args.weight_decay, fused=(device.type == "cuda"))
        # T_B starts as B's init (== --opponent-ckpt weights). Since 2026-08-03 it is BOTH the
        # cross-gate reference AND B's teacher anchor (same kl/value coefs as A's teacher, user
        # order after the ceiling/leak audit: unanchored B gained +3.3pt on counters while
        # LEAKING -4.4pt of inherited wide-third piloting -- each wide deck gets ~0.008% of
        # games, nowhere near maintenance exposure, so drift wins without a restoring force).
        # Promotions keep refreshing it (earned ratchet), exactly the generalist teacher shape.
        t_b_net = build_token_net(ct, ock["net_config"]).to(device)
        t_b_net.load_state_dict(opp_net.state_dict()); t_b_net.eval()
        t_b_net.varlen_attn = args.varlen_attn
        t_b_net.varlen_compile = net.varlen_compile if args.varlen_attn else False
        for _p in t_b_net.parameters():
            _p.requires_grad_(False)
        if args.resume:
            _rb = torch.load(args.resume, map_location=device)
            if _rb.get("net_b") is not None:
                load_net_compat(opp_net, _rb["net_b"], "resume net_b")
                if _rb.get("opt_b") is not None:
                    opt_b.load_state_dict(_rb["opt_b"])
                if _rb.get("t_b") is not None:               # the cross-gate ratchet survives legs
                    load_net_compat(t_b_net, _rb["t_b"], "resume t_b")
                if _rb.get("cg_state") is not None:
                    cg_state.update(_rb["cg_state"])
                    cg_state["base"] = None                  # always re-measure after a restart
                if is_main:
                    print(f"[two-net] resumed net_b+opt_b+t_b from {args.resume} "
                          f"(promos A={cg_state['promos_a']} B={cg_state['promos_b']})", flush=True)
            elif is_main:
                print("[two-net] no net_b in resume ckpt -> B starts from --opponent-ckpt", flush=True)
            del _rb
        if args.ft_teachers_from:
            # both teachers <- the generalist base (user 2026-08-03): overrides whatever the
            # resume restored. T_A/t_b diverge again via earned promotions only.
            _tck = torch.load(args.ft_teachers_from, map_location=device)
            _tsd = _tck.get("net", _tck)
            if teacher_net is not None:
                load_net_compat(teacher_net, _tsd, "teacher refresh")
            t_b_net.load_state_dict(_tsd)
            cg_state["base"] = None                          # stale baseline died with old teachers
            del _tck, _tsd
            if is_main:
                print(f"[two-net] T_A and t_b RESET from {args.ft_teachers_from} "
                      f"(generalist-anchor start)", flush=True)
        update_lv_b = opp_net.logits_value
        if args.compile:
            _cmb = None if args.compile_mode == "default" else args.compile_mode
            update_lv_b = torch.compile(opp_net.logits_value, dynamic=False, mode=_cmb)
            if is_main:
                print("[two-net] B update forward compiled (dynamic=False, same mode as A)", flush=True)
        if is_main:
            print(f"[two-net] LIVE opponent: B trains on ~amask rows, lr={args.lr} (same schedule as A); "
                  f"ft_wr is now the A-vs-B curriculum diagnostic, NOT a promotion signal", flush=True)
    fw_net = None
    fw_envs_np = None
    fw_outcomes = None
    fw_lv = None
    if args.frozen_wide_envs > 0:
        # FROZEN-WIDE pilot: permanently frozen -- distinct from T_A/t_b, which promotions keep
        # refreshing (piloting the stream with a promoted specialist would defeat its purpose).
        # Default = the ORIGINAL base (--opponent-ckpt); --frozen-wide-ckpt overrides it with a
        # specialist (ship-ft KS stream, user 2026-08-09). Rebuilt on every (re)start.
        _fwck = ock
        if args.frozen_wide_ckpt:
            _fwck = torch.load(args.frozen_wide_ckpt, map_location="cpu")
        fw_net = build_token_net(ct, _fwck["net_config"]).to(device)
        load_net_compat(fw_net, _fwck["net"], "frozen-wide"); fw_net.eval()
        fw_net.varlen_attn = args.varlen_attn
        fw_net.varlen_compile = net.varlen_compile if args.varlen_attn else False
        for _p in fw_net.parameters():
            _p.requires_grad_(False)
        fw_envs_np = np.array(frozen_opp["wide_flags"], dtype=bool)
        fw_outcomes = collections.deque(maxlen=2000)   # rolling stream-vs-agent outcomes -> fw_wr
        if args.compile_collect or args.collect_graphs:
            # frozen stream -> same full-batch reduce-overhead treatment as the frozen opponent
            # (2026-08-10 speed fix: eager subset forwards were ~half the collect fwd time)
            fw_net.varlen_attn = False
            fw_net.varlen_compile = False
            _fwfn = torch.compile(fw_net.logits_value, mode="reduce-overhead", dynamic=False)
            _fwfb = [0]
            def _fw_safe(o, opt_len=None, state_keep=None, _c=_fwfn, _e=fw_net.logits_value):
                try:
                    return _c(o, opt_len=opt_len, state_keep=state_keep)
                except Exception as ex:
                    _fwfb[0] += 1
                    if _fwfb[0] <= 3:
                        print(f"[fw-compile] fallback #{_fwfb[0]}: {type(ex).__name__}", flush=True)
                    return _e(o, opt_len=opt_len, state_keep=state_keep)
            fw_lv = _fw_safe
        if is_main and args.frozen_wide_ckpt:
            print(f"[frozen-wide] SPECIALIST pilot {args.frozen_wide_ckpt} "
                  f"(step {_fwck.get('global_step')}) "
                  f"picks={'SAMPLED' if args.frozen_wide_sample else 'greedy'}", flush=True)
        if args.frozen_wide_ckpt:
            del _fwck
    ks_net = None
    ks_envs_np = None
    ks_outcomes = None
    ks_lv = None
    if args.frozen_ks_envs > 0:
        # KS stream pilot: the matched KS specialist, permanently frozen, always SAMPLED
        _kck = torch.load(args.frozen_ks_ckpt, map_location="cpu")
        ks_net = build_token_net(ct, _kck["net_config"]).to(device)
        load_net_compat(ks_net, _kck["net"], "frozen-ks"); ks_net.eval()
        ks_net.varlen_attn = args.varlen_attn
        ks_net.varlen_compile = net.varlen_compile if args.varlen_attn else False
        for _p in ks_net.parameters():
            _p.requires_grad_(False)
        ks_envs_np = np.array(frozen_opp["ks_flags"], dtype=bool)
        ks_outcomes = collections.deque(maxlen=2000)   # rolling agent-vs-KS outcomes -> ks_wr
        del _kck
        if args.compile_collect or args.collect_graphs:
            ks_net.varlen_attn = False
            ks_net.varlen_compile = False
            _ksfn = torch.compile(ks_net.logits_value, mode="reduce-overhead", dynamic=False)
            _ksfb = [0]
            def _ks_safe(o, opt_len=None, state_keep=None, _c=_ksfn, _e=ks_net.logits_value):
                try:
                    return _c(o, opt_len=opt_len, state_keep=state_keep)
                except Exception as ex:
                    _ksfb[0] += 1
                    if _ksfb[0] <= 3:
                        print(f"[ks-compile] fallback #{_ksfb[0]}: {type(ex).__name__}", flush=True)
                    return _e(o, opt_len=opt_len, state_keep=state_keep)
            ks_lv = _ks_safe

    next_obs_np, next_seat_np, _infos0 = envs.reset()
    next_obs = _to_tensors(next_obs_np, enc.int_keys, device)
    next_seat = torch.as_tensor(next_seat_np, device=device)
    if opp_net is not None:
        agent_seat_np = np.array([int(i.get("agent_seat", 0)) for i in _infos0], dtype=np.int64)

    def buffer_state_keep(b_obs):
        """BUFFER-level state-keep (valid for every minibatch subset; one D2H per iteration and
        uniform shapes across minibatches -> at most one compile specialization per iter)."""
        if not args.state_trunc:
            return None
        _sk_np = state_keep_np(b_obs, net.split_heads)
        return None if _sk_np is None else torch.as_tensor(_sk_np, device=device)

    def teacher_prepass(b_obs, b_sk, t_lv=None):
        """Frozen-teacher no-grad forward over the flattened buffer -> t_cache. Called from
        update_net (default) or from the collector thread (--teacher-prepass-collect, where its
        cost hides under the update). Identical math either way: fp32, same rows, same
        stride/bucketing; b_sk is a pure function of b_obs so both call sites derive the same one.
        t_lv overrides the forward (B's teacher t_b_net; eager -- t_b promotions would
        invalidate a compiled graph the same way A's do, and B's prepass runs inline anyway)."""
        tl, tv, tvd = [], [], []
        _tstride = (args.minibatch_size if args.varlen_attn
                    else min(args.minibatch_size, int32_safe_rows(args.nhead)))
        # bf16 autocast (2026-07-13 sps work): the prepass is a full-batch forward that runs in
        # the collector thread and CONTENDS with the update for the GPU -- it was a large slice
        # of the 25s collect-fwd wall. autocast is thread-safe (no dynamo); outputs upcast to
        # fp32 on cat. Both call sites (update inline + collector) share THIS one implementation,
        # so the exactness contract between them is preserved; the teacher targets themselves
        # shift by bf16 rounding only (teacher-kl coef 0.005).
        def _run(lv):
            tl, tv, tvd = [], [], []
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                                 enabled=device.type == "cuda"):
                for s in range(0, args.batch_size, _tstride):
                    cmb = {k: b_obs[k][s:s + _tstride] for k in shapes}
                    cl, cv, cvd = lv(cmb, return_vdist=True,
                                     opt_len=bucketed_opt_len(cmb["action_mask"], cmb["opt_attr"]),
                                     state_keep=b_sk)
                    tl.append(cl.float()); tv.append(cv.float())
                    if cvd is not None:
                        tvd.append(cvd.float())
            return {"logits": torch.cat(tl), "val": torch.cat(tv),
                    "vdist": torch.cat(tvd) if tvd else None}
        if t_lv is not None:
            return _run(t_lv)
        try:
            return _run(teacher_net.logits_value)
        except RuntimeError as e:
            # A teacher promotion (load_state_dict) invalidates dynamo guards; the retrace --
            # which can fire inside the collector thread -- hits the NJT fx-trace wall and
            # killed a leg (sweep2_nobc @20M, 2026-07-14). One-shot fully-eager retry: teacher
            # targets shift by compile-numerics only (same class as bf16 rounding, coef 0.005).
            print(f"[teacher-prepass] compiled fwd failed ({type(e).__name__}: {str(e)[:80]}) "
                  f"-> eager retry", flush=True)
            return _run(torch._dynamo.disable(teacher_net.logits_value))

    # TWO-NET BALANCE GATING scales (user design 2026-08-02): 0-dim CUDA tensors mutated
    # in-place each iteration (stable identity -> no dynamo recompile, DDP graphs identical
    # across ranks/iterations). p*=policy+entropy scale, a*=value+teacher scale, per net.
    # mode=off leaves them at 1.0 forever -> numerically identical to the historical path.
    _bal = {"pa": torch.ones((), device=device), "aa": torch.ones((), device=device),
            "pb": torch.ones((), device=device), "ab": torch.ones((), device=device)}
    _bal_state = {"gate": "none"}
    _bal_lo, _bal_hi = (float(x) for x in args.ft_balance_band.split(","))

    def update_net(advantages, returns, t_pre=None):
        # CRITIC-WARMUP is an UPDATE-level decision: evaluated ONCE here (same post-increment
        # global_step the old per-chunk check read -> byte-identical loss on every iteration,
        # including the boundary one) and consumed by both the chunk loop and the banner.
        critic_only = global_step < args.critic_warmup_until
        # TURNS FAST-PATH (user 2026-08-03, "23s updates for one net"): the frozen side's
        # forward/backward/teacher-prepass is pure waste -- its loss scales are 0, so skipping
        # the computation entirely yields the SAME updates (one semantic change vs the
        # zero-scaled path: the frozen side's Adam moments are PRESERVED during the phase
        # instead of decaying through zero-grad steps -- a true pause). Rank-safe: the turn
        # comes from cg_state (broadcast decisions / global_step timebox), identical everywhere.
        _turns_ft = args.ft_balance_mode == "turns" and opt_b is not None
        _skip_a = _turns_ft and cg_state.get("turn", "B") == "B" and not critic_only
        _skip_b = _turns_ft and cg_state.get("turn", "B") == "A"
        b_obs = {k: buf["obs"][k].reshape((-1, *shapes[k])) for k in shapes}
        b_sk = buffer_state_keep(b_obs)
        b_logp = buf["logp"].reshape(-1); b_actions = buf["act"].reshape(-1)
        b_adv = advantages.reshape(-1); b_returns = returns.reshape(-1); b_vals = buf["val"].reshape(-1)
        # FINETUNE: policy/entropy terms train on AGENT rows only (opponent rows hold the frozen
        # net's off-policy actions); value trains on ALL rows (live net's own values, negamax
        # returns -- both-seat value learning, same as ordinary 2-side self-play).
        b_amask = buf["amask"].reshape(-1) if args.opponent_ckpt else None
        b_fmask = (buf["fmask"].reshape(-1)
                   if (fw_envs_np is not None or ks_envs_np is not None) else None)
        t_cache = None
        if (not _skip_a and teacher_net is not None
                and (args.teacher_kl_coef > 0 or args.teacher_value_coef > 0)):
            t_cache = t_pre if t_pre is not None else teacher_prepass(b_obs, b_sk)
        # TWO-NET: B's teacher targets (t_b_net -- same kl/value anchoring as A, user order
        # 2026-08-03 after the ceiling/leak audit). Inline full-batch forward, NOT hidden in the
        # collector like A's can be -- accepted sps cost on ft runs; masked to B's rows below.
        tb_cache = None
        if (not _skip_b and opt_b is not None and t_b_net is not None
                and (args.teacher_kl_coef > 0 or args.teacher_value_coef > 0)):
            tb_cache = teacher_prepass(b_obs, b_sk, t_lv=t_b_net.logits_value)
        # INT32-SAFE CHUNKED FORWARD (2026-07-09): SDPA's attention indexing overflows when
        # rows*heads*seq^2 >= 2^31 (seq <= 540 incl option slots), capping the minibatch at 1748
        # rows for h4 / 874 for h8. Above the safe row count the minibatch runs in chunks with
        # gradient accumulation in SUM form: full-minibatch normalizers (adv-norm mu/sd, masked
        # denominators) are computed BEFORE chunking, each chunk backwards its globally-normalized
        # loss CONTRIBUTION, so accumulated grads == the big-minibatch grads EXACTLY. chunks==1
        # (all current nets) -> one pass, identical to the unchunked code.
        # varlen NJT never builds the padded rows*heads*seq^2 index, so the int32 cap only
        # binds on the dense path (the docstring's "degenerates to 1 chunk" went stale at the
        # nmb-8 switch: per-rank mb 6,144 vs cap 1,749 = 4 pointless launches per minibatch).
        # PKMN_CSZ overrides for tests. Chunk size is math-neutral (SUM-accum + precomputed
        # full-minibatch normalizers) -- only kernel batching and fp reduction order change.
        _csz_env = os.environ.get("PKMN_CSZ")
        _safe_rows = (int(_csz_env) if _csz_env else
                      (args.minibatch_size if args.varlen_attn else int32_safe_rows(args.nhead)))
        _csz = math.ceil(args.minibatch_size / max(1, math.ceil(args.minibatch_size / _safe_rows)))
        inds = np.arange(args.batch_size); clipfracs, approx_kls = [], []
        last = {"pg": 0.0, "v": 0.0, "ent": 0.0}
        # PERMUTE-ONCE + SLICE (2026-07-13 sps work): the old loop fancy-indexed ~40 obs tensors
        # per minibatch (40x32 gather kernels/update) and forced ~64 GPU syncs via per-minibatch
        # .item() stats -- together the launch-storm behind the 16-18s update wall. Now: one
        # gather per key per EPOCH (identical shuffle RNG -> identical minibatch composition),
        # minibatches are contiguous views, stats stay on-device until ONE sync at return, and
        # the opt-len bucket comes from a per-row precompute (one sync per update, numpy after).
        _pres = b_obs["action_mask"][..., :_P_MAXOPT] > 0.5
        _pres = _pres | (b_obs["opt_attr"][..., _P_OPTPICK] > 0.5)
        _rowlen_np = ((_pres * torch.arange(1, _P_MAXOPT + 1, device=_pres.device)).amax(-1)
                      ).cpu().numpy()                       # per-row furthest present slot+1
        for _epoch in range(args.update_epochs):
            np.random.shuffle(inds)
            perm = torch.as_tensor(inds, device=b_adv.device)
            p_obs = {k: b_obs[k][perm] for k in shapes}
            p_act = b_actions[perm]; p_logp = b_logp[perm]; p_adv = b_adv[perm]
            p_ret = b_returns[perm]; p_val = b_vals[perm]
            p_am = b_amask[perm] if b_amask is not None else None
            p_fm = b_fmask[perm] if b_fmask is not None else None
            p_tl = t_cache["logits"][perm] if t_cache is not None else None
            p_tv = t_cache["val"][perm] if t_cache is not None else None
            p_tvd = (t_cache["vdist"][perm] if (t_cache is not None and t_cache["vdist"] is not None)
                     else None)
            p_tbl = tb_cache["logits"][perm] if tb_cache is not None else None
            p_tbv = tb_cache["val"][perm] if tb_cache is not None else None
            _rl_p = _rowlen_np[inds]
            for s in range(0, args.batch_size, args.minibatch_size):
                mbsl = slice(s, min(s + args.minibatch_size, args.batch_size))
                _mb_i = s // args.minibatch_size
                _ga_first = _mb_i % args.grad_accum == 0          # group start -> zero grads
                _ga_last = _mb_i % args.grad_accum == args.grad_accum - 1   # group end -> step
                n_rows = float(mbsl.stop - mbsl.start)
                # full-minibatch normalizers (data-only, no forward -> exact under chunking)
                adv_full = p_adv[mbsl]
                m_full = msum = None
                if p_am is None:
                    if args.norm_adv:
                        adv_full = (adv_full - adv_full.mean()) / (adv_full.std() + 1e-8)
                else:                                        # FINETUNE: agent-row masking
                    m_full = p_am[mbsl]; msum = m_full.sum().clamp(min=1.0)
                    if args.norm_adv:
                        mu = (adv_full * m_full).sum() / msum
                        sd = (((adv_full - mu) ** 2 * m_full).sum() / msum).sqrt()
                        adv_full = (adv_full - mu) / (sd + 1e-8)
                # TWO-NET: B's advantages normalized over B's OWN rows (same masked-normalizer
                # pattern as A's; the raw p_adv is reused -- adv_full above is already A-normalized)
                mB_full = msum_b = advB_full = None
                if opt_b is not None and m_full is not None:
                    mB_full = 1.0 - m_full
                    if p_fm is not None:
                        # FROZEN-WIDE rows: frozen pilot acted -- off-policy for B too; B's pg/
                        # ent/value/teacher/adv-norm all flow from this mask, so one subtraction
                        # removes the wide stream from B's training entirely
                        mB_full = (mB_full - p_fm[mbsl]).clamp(min=0.0)
                    msum_b = mB_full.sum().clamp(min=1.0)
                    advB_full = p_adv[mbsl]
                    if args.norm_adv:
                        muB = (advB_full * mB_full).sum() / msum_b
                        sdB = (((advB_full - muB) ** 2 * mB_full).sum() / msum_b).sqrt()
                        advB_full = (advB_full - muB) / (sdB + 1e-8)
                if _ga_first:
                    opt.zero_grad()
                    if opt_b is not None:
                        opt_b.zero_grad()
                acc = {"pg": 0.0, "v": 0.0, "ent": 0.0, "clip": 0.0, "kl": 0.0}
                for ci in range(mbsl.start, mbsl.stop, _csz):
                    rows = slice(ci, min(ci + _csz, mbsl.stop))
                    mb_obs = {k: p_obs[k][rows] for k in shapes}
                    _ol = bucket_for(int(_rl_p[rows].max()))  # exact trailing-slot drop, sync-free
                    _lo = rows.start - mbsl.start             # chunk offset (used by BOTH sides)
                    if not _skip_a:                           # turns fast-path: frozen A skipped whole
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                            s_logits, newval, vdist = update_lv(mb_obs, return_vdist=True, opt_len=_ol,
                                                                state_keep=b_sk)
                        if args.bf16:                           # upcast -> loss/ratios/softmax in fp32 (safe)
                            s_logits = s_logits.float(); newval = newval.float()
                            vdist = vdist.float() if vdist is not None else None
                        pdist = Categorical(logits=s_logits)
                        newlogp = pdist.log_prob(p_act[rows]); entropy = pdist.entropy()
                        ratio = (newlogp - p_logp[rows]).exp()
                        adv = adv_full[_lo:_lo + (rows.stop - rows.start)]
                        pgmax = torch.max(-adv * ratio,
                                          -adv * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef))
                        if m_full is None:                       # standard 2-side path (unchanged math)
                            with torch.no_grad():                # stats stay on-device (no sync here)
                                acc["clip"] = acc["clip"] + ((ratio - 1.0).abs() > args.clip_coef).float().sum() / n_rows
                                acc["kl"] = acc["kl"] + ((ratio - 1.0) - (newlogp - p_logp[rows])).sum() / n_rows
                            pg_loss = pgmax.sum() / n_rows
                            ent_loss = entropy.sum() / n_rows
                        else:
                            m = m_full[_lo:_lo + (rows.stop - rows.start)]
                            with torch.no_grad():
                                acc["clip"] = acc["clip"] + ((((ratio - 1.0).abs() > args.clip_coef).float() * m).sum() / msum)
                                acc["kl"] = acc["kl"] + ((((ratio - 1.0) - (newlogp - p_logp[rows])) * m).sum() / msum)
                            pg_loss = (pgmax * m).sum() / msum
                            ent_loss = (entropy * m).sum() / msum
                        if net.value_categorical:
                            vtgt = net.value_target_dist(p_ret[rows])
                            v_loss = -(vtgt * torch.log_softmax(vdist, dim=-1)).sum(-1).sum() / n_rows
                        elif args.vf_clip is not None:
                            vc = p_val[rows] + torch.clamp(newval - p_val[rows], -args.vf_clip, args.vf_clip)
                            v_loss = 0.5 * torch.max((newval - p_ret[rows]) ** 2,
                                                     (vc - p_ret[rows]) ** 2).sum() / n_rows
                        elif mB_full is not None:
                            # TWO-NET: A's value trains on A's OWN rows only (B owns its rows' value
                            # now; the single-net ft's value-on-all-rows would have A regress toward
                            # positions it never acts from under a different policy)
                            v_loss = 0.5 * (((newval - p_ret[rows]) ** 2) * m).sum() / msum
                        else:
                            v_loss = 0.5 * ((newval - p_ret[rows]) ** 2).sum() / n_rows
                        if critic_only:
                            # CRITIC-FIRST WARMUP: value only; pg/ent zeroed. The teacher terms
                            # below stay -> the policy is actively pinned while the trunk learns
                            # the new decks' values. Stats (pg/ent/kl) still tracked for telemetry.
                            loss = _bal["aa"] * (args.vf_coef * v_loss)
                        else:
                            # balance gating: pa scales A's policy/entropy, aa its value (+teacher)
                            loss = _bal["pa"] * (pg_loss - args.ent_coef * ent_loss) \
                                + _bal["aa"] * (args.vf_coef * v_loss)
                        if t_cache is not None:
                            t_logp = torch.log_softmax(p_tl[rows], dim=-1)
                            s_logp = torch.log_softmax(s_logits, dim=-1)
                            kl = (t_logp.exp() * (t_logp - s_logp)).sum(-1).sum() / n_rows
                            with torch.no_grad():        # A teacher-KL telemetry (mirror of b_tkl)
                                acc["a_tkl"] = acc.get("a_tkl", 0.0) + kl.detach()
                            if net.value_categorical:
                                tvp = torch.softmax(p_tvd[rows], dim=-1)
                                vdistill = -(tvp * torch.log_softmax(vdist, dim=-1)).sum(-1).sum() / n_rows
                            else:
                                vdistill = 0.5 * ((newval - p_tv[rows]) ** 2).sum() / n_rows
                            loss = loss + _bal["aa"] * (args.teacher_kl_coef * kl
                                                        + args.teacher_value_coef * vdistill)
                        if args.grad_accum > 1:                  # group mean: each minibatch weighs 1/N
                            loss = loss / args.grad_accum
                        loss.backward()                          # grads ACCUMULATE across chunks (exact)
                    if mB_full is not None and not _skip_b:
                        # TWO-NET: B's pass over the SAME chunk -- compiled forward (mirrors A's
                        # update_lv), losses masked to B's rows, backward into B's params only
                        # (separate graph from A's). Plain MSE value (guarded at startup); no
                        # teacher terms (B's whole job is unconstrained specialization --
                        # revisit if its entropy collapses).
                        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.bf16):
                            bo = update_lv_b(mb_obs, opt_len=_ol, state_keep=b_sk)
                        b_logits, b_newval = bo[0].float(), bo[1].float()
                        bdist = Categorical(logits=b_logits)
                        b_newlogp = bdist.log_prob(p_act[rows])
                        b_ratio = (b_newlogp - p_logp[rows]).exp()
                        mB = mB_full[_lo:_lo + (rows.stop - rows.start)]
                        advB = advB_full[_lo:_lo + (rows.stop - rows.start)]
                        b_pgmax = torch.max(-advB * b_ratio,
                                            -advB * torch.clamp(b_ratio, 1 - args.clip_coef,
                                                                1 + args.clip_coef))
                        b_pg = (b_pgmax * mB).sum() / msum_b
                        b_ent = (bdist.entropy() * mB).sum() / msum_b
                        b_v = 0.5 * (((b_newval - p_ret[rows]) ** 2) * mB).sum() / msum_b
                        b_loss = _bal["pb"] * (b_pg - args.ent_coef * b_ent) \
                            + _bal["ab"] * (args.vf_coef * b_v)
                        if tb_cache is not None:
                            # B's teacher anchor (mirrors A's terms; masked to B's rows -- the
                            # opponent-seat positions, which is exactly where the wide leak lives)
                            tb_logp = torch.log_softmax(p_tbl[rows], dim=-1)
                            b_slogp = torch.log_softmax(b_logits, dim=-1)
                            b_tkl = ((tb_logp.exp() * (tb_logp - b_slogp)).sum(-1) * mB
                                     ).sum() / msum_b
                            b_vdistill = 0.5 * (((b_newval - p_tbv[rows]) ** 2) * mB).sum() / msum_b
                            b_loss = b_loss + _bal["ab"] * (args.teacher_kl_coef * b_tkl
                                                            + args.teacher_value_coef * b_vdistill)
                            with torch.no_grad():
                                acc["b_tkl"] = acc.get("b_tkl", 0.0) + b_tkl.detach()
                        if args.grad_accum > 1:
                            b_loss = b_loss / args.grad_accum
                        b_loss.backward()
                        with torch.no_grad():
                            acc["b_pg"] = acc.get("b_pg", 0.0) + b_pg.detach()
                            acc["b_ent"] = acc.get("b_ent", 0.0) + b_ent.detach()
                            acc["b_v"] = acc.get("b_v", 0.0) + b_v.detach()
                    if not _skip_a:                          # A stats exist only when A computed
                        with torch.no_grad():
                            acc["pg"] = acc["pg"] + pg_loss.detach()
                            acc["v"] = acc["v"] + v_loss.detach()
                            acc["ent"] = acc["ent"] + ent_loss.detach()
                clipfracs.append(acc["clip"]); approx_kls.append(acc["kl"])
                if _ga_last:                                 # group boundary: reduce + clip + step
                    if not _skip_a:
                        if ddp:                              # average grads across ranks -> identical step
                            # FLATTENED (2026-07-13 sps): one fused allreduce instead of ~150 per-param
                            # NCCL calls per step. Same sum, same /world -> exact. With --grad-accum the
                            # reduce happens once per GROUP (accumulated local grads reduce identically).
                            _gs = [p.grad for p in net.parameters() if p.grad is not None]
                            _flat = torch._utils._flatten_dense_tensors(_gs)
                            dist.all_reduce(_flat); _flat /= world
                            for g, gsync in zip(_gs, torch._utils._unflatten_dense_tensors(_flat, _gs)):
                                g.copy_(gsync)
                        nn.utils.clip_grad_norm_(net.parameters(), args.max_grad_norm); opt.step()
                    if opt_b is not None and not _skip_b:    # TWO-NET: B mirrors A's reduce/clip/step
                        if ddp:
                            _gsb = [p.grad for p in opp_net.parameters() if p.grad is not None]
                            _flatb = torch._utils._flatten_dense_tensors(_gsb)
                            dist.all_reduce(_flatb); _flatb /= world
                            for g, gsync in zip(_gsb, torch._utils._unflatten_dense_tensors(_flatb, _gsb)):
                                g.copy_(gsync)
                        nn.utils.clip_grad_norm_(opp_net.parameters(), args.max_grad_norm)
                        opt_b.step()
                last = acc                                   # tensors; converted at return (one sync)
            if args.target_kl is not None and approx_kls:
                nmb = max(1, args.batch_size // args.minibatch_size)
                # one host sync (stack+mean), not one per stat tensor (deferred-stats contract)
                ek = float(torch.stack([torch.as_tensor(x) for x in approx_kls[-nmb:]]).mean())
                if ddp:                                      # all ranks must early-stop together
                    t = torch.tensor(ek, device=device); dist.all_reduce(t); ek = (t / world).item()
                if ek > args.target_kl:
                    break
        # ONE host sync for all deferred stats (they were 0-dim CUDA tensors during the loop)
        clipfracs = [float(x) for x in clipfracs]
        approx_kls = [float(x) for x in approx_kls]
        last = {"pg": float(last["pg"]), "v": float(last["v"]), "ent": float(last["ent"]),
                # B-SIDE VISIBILITY (user 2026-08-03, the turns B-degradation mystery): B's
                # pg/value/entropy losses were computed but never surfaced -- same
                # accumulation semantics as the A-side stats.
                **({"b_pg": float(last.get("b_pg", 0.0)), "b_v": float(last.get("b_v", 0.0)),
                    "b_ent": float(last.get("b_ent", 0.0)),
                    "b_tkl": float(last.get("b_tkl", 0.0)),
                    "a_tkl": float(last.get("a_tkl", 0.0))} if opt_b is not None else {})}
        return clipfracs, approx_kls, last

    def cross_gate_promote(global_step):
        """TWO-NET cross-gate promotion (user design 2026-07-29, every --teacher-refresh steps):
        each net is gated against the OTHER side's frozen teacher, and earns its OWN snapshot --
            A promotes (T_A <- A, + best.pt) when  wr(A  vs T_B) >= baseline + delta
            B promotes (T_B <- B)            when  wr(T_A vs B ) <= baseline - delta
        where baseline = wr(T_A vs T_B), the frozen-frozen matchup EV -- an absolute bar is
        meaningless here because the asymmetric deck matchup EV != 0.5 (measured ~0.60 for the
        ft side at equal skill) and it DRIFTS as both sides strengthen. The baseline is cached
        and re-measured only after a promotion changes a teacher. Both gates run against the
        PRE-promotion teachers (decide first, then apply). Sequential early-stopping keeps the
        typical check at 1-2 blocks. Ship decisions stay with the offline audits -- these gates
        certify progress, they are not the ship signal."""
        # DDP-PARALLEL gating (user order 2026-08-02): every rank plays disjoint shards of the
        # same schedule via ddp_cross_gate -- a full worst-case gate (re-baseline + no early
        # separation) drops from ~500s rank0-serial to ~1/world of that; results are identical
        # on all ranks (all-reduced tallies), so decisions need no broadcast (kept as a guard).
        from .gates import ddp_cross_gate
        delta = args.cross_gate_delta
        if frozen_opp.get("agent_decks") and len(frozen_opp["agent_decks"]) > 1:
            raise SystemExit("--two-net-finetune with a deck SET would cross-gate on only the "
                             "first deck; step 2 needs the gate taught about the set first")
        ft_deck = frozen_opp["agent_deck"]
        od, ow = frozen_opp["opp_decks"], frozen_opp["opp_weights"]
        net.eval(); opp_net.eval()
        decisions = torch.zeros(3, device=device)            # [promote_a, promote_b, new_base]
        _cg_t0 = time.time()
        _rk, _wd = (rank, world) if ddp else (0, 1)
        if is_main:
            print(f"[cross-gate] step={global_step} gating...", flush=True)
        if cg_state["base"] is None:                         # startup or post-promotion re-measure
            b = ddp_cross_gate(teacher_net, t_b_net, [ft_deck], od, ow, enc, device,
                               seed=90210, bar=None, block=250, cap=1000,
                               ddp_rank=_rk, ddp_world=_wd)
            cg_state["base"] = b["wr"]
            if is_main:
                print(f"[cross-gate] baseline T_A-vs-T_B = {b['wr']:.4f} (n={b['n']})", flush=True)
        base = cg_state["base"]
        # TURNS integration (user 2026-08-03): under strict alternation a side that was frozen
        # for the whole window cannot have improved -- test only sides that actually trained
        # since the last gate (skips half the gate games and avoids meaningless re-tests).
        # The trained-flags come from the balance block and are deterministic across ranks
        # (driven by the all-reduced rolling wr), so no broadcast is needed for the skips.
        _test_a = _test_b = True
        if args.ft_balance_mode == "turns":
            _test_a = _bal_state.pop("a_trained", False)
            _test_b = _bal_state.pop("b_trained", False)
        ga = ddp_cross_gate(net, t_b_net, [ft_deck], od, ow, enc, device,
                            seed=90211, bar=base + delta, ddp_rank=_rk, ddp_world=_wd) \
            if _test_a else None
        gb = ddp_cross_gate(teacher_net, opp_net, [ft_deck], od, ow, enc, device,
                            seed=90212, bar=base - delta, ddp_rank=_rk, ddp_world=_wd) \
            if _test_b else None
        pa = ga is not None and ga["wr"] >= base + delta     # A beats B's frozen by delta
        pb = gb is not None and gb["wr"] <= base - delta     # B holds A's frozen BELOW baseline-delta
        # last gate readings (all-reduced -> rank-identical): the timebox-bank floor guard
        if ga is not None:
            cg_state["last_a_wr"] = float(ga["wr"])
        if gb is not None:
            cg_state["last_b_wr"] = float(gb["wr"])
        decisions[0] = float(pa); decisions[1] = float(pb); decisions[2] = base
        if is_main:
            _sa = (f"A-vs-T_B={ga['wr']:.4f}(n={ga['n']},blk={ga['blocks']})"
                   f"{'*PROMOTE*' if pa else ''}") if ga is not None else "A-skip(frozen)"
            _sb = (f"T_A-vs-B={gb['wr']:.4f}(n={gb['n']},blk={gb['blocks']})"
                   f"{'*PROMOTE*' if pb else ''}") if gb is not None else "B-skip(frozen)"
            print(f"[cross-gate] step={global_step} base={base:.4f} {_sa} {_sb} "
                  f"[{time.time() - _cg_t0:.0f}s]", flush=True)
        if ddp:
            dist.broadcast(decisions, src=0)                 # all ranks apply identical decisions
        pa, pb = bool(decisions[0].item()), bool(decisions[1].item())
        if pa:
            with t_lock:
                teacher_net.load_state_dict(net.state_dict()); teacher_net.eval()
                t_epoch["n"] += 1
            cg_state["promos_a"] += 1; cg_state["base"] = None
            if is_main:
                torch.save({"net": net.state_dict(), "args": vars(args), "net_config": net_config,
                            "global_step": global_step, "cross_gate_promo": cg_state["promos_a"]},
                           os.path.join(args.out, "best.pt"))
                print(f"[cross-gate] A PROMOTED (#{cg_state['promos_a']}) -> T_A updated + best.pt",
                      flush=True)
        if pb:
            t_b_net.load_state_dict(opp_net.state_dict()); t_b_net.eval()
            cg_state["promos_b"] += 1; cg_state["base"] = None
            if is_main:
                print(f"[cross-gate] B PROMOTED (#{cg_state['promos_b']}) -> T_B updated", flush=True)
        # TURNS: flip on promotion (user design 2026-08-03 v2) -- the phase ends the moment
        # the TRAINING side banks its verified +delta. A frozen side promoting (possible once
        # right after a timebox flip, from residual pre-flip training) refreshes its teacher
        # but does NOT flip. Deterministic: turn + broadcast decisions are rank-identical.
        if args.ft_balance_mode == "turns":
            _t = cg_state.get("turn", "B")
            if (_t == "A" and pa) or (_t == "B" and pb):
                cg_state["turn"] = "A" if _t == "B" else "B"
                cg_state["phase_start"] = global_step
                if is_main:
                    print(f"[turns] step={global_step} PROMOTION flip -> training "
                          f"{cg_state['turn']}", flush=True)
        net.train()
        return None

    def maybe_promote(global_step):
        """C11: refresh / gated-promote the teacher. On promotion, log it AND save best.pt (the gated-best
        net == the new teacher -- the deployment-relevant artifact, which periodic ckpts can miss)."""
        if teacher_net is None:
            return None
        if args.promote_winrate is None:                    # time-based refresh (no gate)
            with t_lock:                                     # collector pre-pass may be mid-forward
                teacher_net.load_state_dict(net.state_dict()); teacher_net.eval()   # net is DDP-synced -> identical
                t_epoch["n"] += 1
            if is_main:
                print(f"[teacher] step={global_step} time-based refresh", flush=True)
            return None
        # FINETUNE: the mirror h2h is skipped entirely (mirror parity on the ship deck neither
        # targets the objective nor is guaranteed to move with it -- a never-promoting mirror gate
        # would pin the anchor at the init and the KL term would drag the policy back toward
        # generalist play). GUARDED promotion (user design 2026-07-09): promote every
        # --teacher-refresh while the rolling training wr vs the STATIC opponent (ft_wr, free
        # from the collection stream) is within noise of the best seen (margin ~2sigma of the
        # 2000-episode window); HOLD only on a genuine slump, so the anchor stays at the peak and
        # pulls back. A strict new-best ratchet would starve late (bounded metric saturates,
        # max-of-noise best); pure trailing would follow a degradation down.
        if opp_net is not None:
            w, n = float(sum(ft_wr_win)), float(len(ft_wr_win))
            if ddp:                                          # identical decision on every rank
                t = torch.tensor([w, n], device=device)
                dist.all_reduce(t)
                w, n = float(t[0].item()), float(t[1].item())
            wr_now = (w / n) if n >= 200 else None           # too little data -> promote freely
            # --promote-strict (user 2026-08-09): NO downward slack -- promote only at/above the
            # best seen. Kills the slow-decline ratchet leak (0.433->0.424->0.422->0.412, each
            # step inside the 0.02 slack, teacher following the net down by installments). Meant
            # for CURRICULUM legs, where --reset-ft-anchor bounds the stale-outlier-bar cost to
            # one leg; leave OFF for long stationary fts (max-of-noise best starves late there).
            _bar = ft_anchor["best"] - (0.0 if args.promote_strict else 0.02)
            if wr_now is None or wr_now >= _bar:
                with t_lock:
                    teacher_net.load_state_dict(net.state_dict()); teacher_net.eval()
                    t_epoch["n"] += 1
                if is_main:
                    _w = "n/a" if wr_now is None else f"{wr_now:.3f}"
                    print(f"[promote] step={global_step} ft_wr={_w} "
                          f"best={ft_anchor['best']:.3f} -> teacher promoted (guarded)", flush=True)
            elif is_main:
                print(f"[hold] step={global_step} ft_wr={wr_now:.3f} < bar "
                      f"{_bar:.3f} -> teacher held "
                      f"({'strict ratchet' if args.promote_strict else 'slump guard'})", flush=True)
            if wr_now is not None:
                ft_anchor["best"] = max(ft_anchor["best"], wr_now)
            return wr_now
        net.eval()
        # fixed seed + fixed schedule (NOT global_step): every gate replays the same conditions,
        # so gate-to-gate wr movement is the NET's movement (paired comparison). EVERY rank plays
        # its sched[rank::world] slice concurrently (one battle per gate helper here; ranks
        # are separate processes -> gate wall time / world).
        _, gate_info = gated_winrate(net, teacher_net, gate_pool, n_gate, enc, device,
                                     seed=1234, mode=args.gate_pairs, rank=rank, world=world)
        net.train()
        if ddp:                                              # sum the per-rank counts -> all ranks decide alike
            t = torch.tensor([float(gate_info["wins"]), float(gate_info["decided"]),
                              float(gate_info["draws"])], device=device)
            dist.all_reduce(t)
            gate_info["wins"], gate_info["decided"], gate_info["draws"] = \
                int(t[0].item()), int(t[1].item()), int(t[2].item())
            gate_info["n"] = n_gate
            pds = [None] * world                             # merge per-deck W/D/L onto rank0 (jsonl only)
            dist.all_gather_object(pds, gate_info["per_deck"])
            if is_main:
                merged: dict = {}
                for pd in pds:
                    for kk, (w_, d_, l_) in (pd or {}).items():
                        row = merged.setdefault(int(kk), [0, 0, 0])
                        row[0] += w_; row[1] += d_; row[2] += l_
                gate_info["per_deck"] = {kk: merged[kk] for kk in sorted(merged)}
        wr = gate_info["wins"] / max(gate_info["decided"], 1)
        promoted = wr >= args.promote_winrate
        if promoted:
            with t_lock:
                teacher_net.load_state_dict(net.state_dict()); teacher_net.eval()   # identical net -> identical teacher
                t_epoch["n"] += 1
            if is_main:
                torch.save({"net": net.state_dict(), "args": vars(args), "net_config": net_config,
                            "global_step": global_step, "promote_wr": wr},
                           os.path.join(args.out, "best.pt"))
                print(f"[promote] step={global_step} wr={wr:.3f} >= {args.promote_winrate} "
                      f"-> TEACHER UPDATED + saved best.pt", flush=True)
        elif is_main:
            print(f"[gate] step={global_step} wr={wr:.3f} < {args.promote_winrate} -> teacher held", flush=True)
        if is_main:                                          # per-deck W/D/L -> post-hoc regression picture
            with open(os.path.join(args.out, "promotion_gate.jsonl"), "a") as f:
                f.write(json.dumps({"step": int(global_step), "wr": round(float(wr), 4),
                                    "promoted": bool(promoted), **gate_info}) + "\n")
        return wr

    # value diagnostics: rank0's own rollouts are a representative sample; window -> jsonl + line
    vd = DiagAccumulator(args.gamma, os.path.join(args.out, "value_diag.jsonl")) \
        if (args.value_diag and is_main) else None

    global_step = resume_step
    start = time.time()
    _PROF = bool(os.environ.get("TRAIN_PROF"))                       # per-iter collect/update split
    # asymmetric truncation penalty (see the compute_gae_2s call site): OFF unless set ->
    # byte-identical to the historical path. _TRUNC_PEN mirrors the env-side knob.
    _TRUNC_ASYM = os.environ.get("PKMN_TRUNC_ASYM") == "1"
    _TRUNC_PEN = float(os.environ.get("PKMN_TRUNC_PENALTY", "") or "0")
    if _TRUNC_ASYM and is_main:
        print(f"[trunc-asym] ACTIVE: two-sided truncation terminals — looper keeps "
              f"-{_TRUNC_PEN:g} (full-line credit, V learns it); victim's last decision "
              f"anchored to +1 WIN, negamax mirror severed (no +{_TRUNC_PEN:g} jackpot)", flush=True)
    _ct = _ut = 0.0
    num_iters = args.total_timesteps // (args.batch_size * world)   # total_timesteps = GLOBAL budget across ranks
    decay_iters = (args.decay_steps // (args.batch_size * world)) if args.decay_steps \
        else max(1, int(0.7 * num_iters))                           # default: decay over 70% of the budget
    if args.warmup_frac > 0:                                        # deprecated %-mode (old launch lines)
        warmup_iters = int(args.warmup_frac * decay_iters)
    else:                                                           # FLAT warmup (budget-invariant)
        warmup_iters = min(args.warmup_steps // (args.batch_size * world), max(1, num_iters // 5))
    start_it = (global_step // (args.batch_size * world)) + 1        # resume continues the LR schedule + loop here
    # warm-restart -> LR schedule anchored at a FIXED absolute step (NOT each leg's resume point):
    # precedence = --sched-anchor-step > ckpt's persisted sched_anchor > this resume point (legacy).
    # The chosen anchor is persisted in every latest.pt so chain legs / self-heal restarts CONTINUE
    # one cosine instead of re-peaking (the entropy-pump bug, 2026-07-10).
    sched_anchor = 0
    if args.resume and args.lr_restart:
        if args.sched_anchor_step >= 0:
            sched_anchor = int(args.sched_anchor_step)
        else:
            _ck_anchor = rck.get("sched_anchor", -1) if isinstance(rck, dict) else -1
            sched_anchor = int(_ck_anchor) if _ck_anchor >= 0 else int(global_step)
        if is_main:
            print(f"[lr] restarted schedule anchored at step {sched_anchor:,} "
                  f"(resume step {global_step:,})", flush=True)
    sched_offset = sched_anchor // (args.batch_size * world)
    last_promote = (global_step // args.teacher_refresh) * args.teacher_refresh   # avoid an immediate promote on resume
    last_tch_wr = float("nan")              # last gated-promotion winrate vs the teacher (carried forward)
    ft_wr_win: list = []                    # finetune: rolling training-wr-vs-static-opponent window

    # MATCHUP MATRIX (2026-07-09, user design): full-granularity per-episode (deck_i, deck_j,
    # outcome) log with a TIME AXIS (one jsonl line per iteration window, delta counts). All
    # aggregation/clustering/CI/debias happens at READ time (scripts/matchup_report.py) -- never
    # at log time, so no granularity is lost. Per-rank files; merge when reading. Header line
    # records the deck-id space so indices stay interpretable per run.
    mm_events: list = []
    mm_path = os.path.join(args.out, f"matchup_r{rank}.jsonl")
    if is_main or world > 1:
        with open(mm_path, "a") as _f:
            _f.write(json.dumps({"header": {"decks": args.decks,
                                            "ladder_file": args.ladder_decks_file,
                                            "pool_size": len(pool),
                                            "deck_weights_file": args.deck_weights_file,
                                            "finetune_deck": args.finetune_deck,
                                            "resume_step": int(global_step)}}) + "\n")
    # ASYNC COLLECT setup (opt-in; see --async-collect help). The collector thread owns the vec
    # env + its own CUDA stream + a frozen copy of the net (refreshed after every update); the
    # main thread updates on the previously collected buffer. Buffers are double-banked.
    a_pending = None
    if args.async_collect:
        if not args.varlen_attn:
            # the old non-varlen async path (single compiled collect_net + eager-fallback shim)
            # deliberately retained the torn-weights refresh bug the twins below were built to
            # fix, and no supported launch used it -- retired 2026-07-14 cleanup.
            raise SystemExit("--async-collect requires --varlen-attn")
        # DOUBLE-BUFFERED collector twins (2026-07-11, kl-sawtooth root cause): the old single
        # collect net was load_state_dict'ed WHILE the collector thread was mid-forward (refresh
        # lands ~11s into a ~17s rollout) -> occasional torn-weights forwards whose stored logp
        # matches no real policy -> phase-dependent kl/clip sawtooth (reset by gate pauses; a
        # restart's fresh thread alignment made it vanish -- the tell). Now: refresh loads the
        # IDLE twin, the next spawn hands the freshly loaded twin to the thread; weights are
        # never mutated under a running forward. Each twin is a self-contained RECORD (net +
        # forward wrapper + state-trunc setting) decided ONCE here, so the spawn sites can never
        # mispair a net with another twin's cudagraph wrapper (whose weight pointers are baked).
        twins = []
        for _ in range(2):
            _cn = build_token_net(ct, net_config).to(device)
            _cn.load_state_dict(net.state_dict()); _cn.eval()
            # --collect-graphs: twins go DENSE so the WHOLE forward is dynamo-traceable (NJT is
            # not) and can be cudagraph-replayed; teacher/update/opp stay varlen (collect logp is
            # behavior-policy data -- the PPO ratio absorbs the numerical-path delta, same as bf16)
            _cn.varlen_attn = not args.collect_graphs
            _cn.varlen_compile = net.varlen_compile if _cn.varlen_attn else False
            for _p in _cn.parameters():
                _p.requires_grad_(False)
            twins.append(SimpleNamespace(net=_cn, lv=_cn.logits_value,
                                         strunc=args.state_trunc))
        c_active = 0                      # index of the twin the NEXT spawn should use
        if args.collect_graphs:
            # One reduce-overhead compile per twin (params are per-graph static pointers; the
            # in-place load_state_dict refresh keeps them valid). Every (twin, opt-bucket) shape is
            # exercised 3x HERE -- compile, record, replay -- serially on the main thread, because
            # inductor cudagraph capture is not safe against a concurrently-allocating second
            # thread. After this, the collector thread only replays. Autocast mirrors the rollout
            # loop exactly (rollout.py lv-branch bf16): the guard set must match or every live call
            # recompiles. Runtime surprises (an unwarmed shape mid-run) fall back eager per twin.
            # state_trunc is forced OFF on this path: its ~11 data-dependent keep-lengths would
            # multiply the graph count (x opt-buckets x twins ~= 88 captures); full-state is the
            # same math on a FIXED shape (8 graphs), and replay swallows the extra tokens' cost.
            for tw in twins:
                _cfn = torch.compile(tw.net.logits_value, mode="reduce-overhead", dynamic=False)

                def _safe(o, opt_len=None, state_keep=None, _c=_cfn, _e=tw.net.logits_value):
                    try:
                        return _c(o, opt_len=opt_len, state_keep=state_keep)
                    except RuntimeError as e:
                        if ("symbolically trace" not in str(e)
                                and "stream is capturing" not in str(e)):
                            raise
                        return _e(o, opt_len=opt_len, state_keep=state_keep)
                tw.lv = _safe
                tw.strunc = False
            _t0 = time.time()
            with torch.no_grad():
                for tw in twins:
                    for _L in _DEFAULT_OPT_BUCKETS:
                        for _ in range(3):
                            if args.bf16_collect:
                                with torch.autocast("cuda", dtype=torch.bfloat16):
                                    tw.lv(next_obs, opt_len=int(_L), state_keep=None)
                            else:
                                tw.lv(next_obs, opt_len=int(_L), state_keep=None)
                    torch.cuda.synchronize()
            if is_main:
                print(f"[collect-graphs] {2 * len(_DEFAULT_OPT_BUCKETS)} (twin x bucket) graphs "
                      f"captured in {time.time() - _t0:.0f}s (dense twins, "
                      f"buckets={list(_DEFAULT_OPT_BUCKETS)}, bf16={args.bf16_collect})", flush=True)
        a_buf2 = make_buffers(args.num_steps, args.num_envs, shapes, enc.int_keys, device)
        # priority=-1 (2026-07-13 sps work): the collector's many small forwards were starved
        # between the update's large kernels (25s of a 27s rollout was collect-fwd wait);
        # high-priority stream lets select forwards preempt the queue.
        a_stream = torch.cuda.Stream(priority=-1) if device.type == "cuda" else None
        a_state: dict = {}
        a_serial_left = max(0, args.async_serial_iters)  # serial-warmup countdown (see arg help)
        _tpre_on = (args.teacher_prepass_collect and teacher_net is not None
                    and (args.teacher_kl_coef > 0 or args.teacher_value_coef > 0))
        if _tpre_on and is_main:
            print("[async] teacher pre-pass runs in the collector thread (exact; hidden under update)",
                  flush=True)

        def _a_ctx():
            """The collector's high-priority stream context (no-op on CPU)."""
            return torch.cuda.stream(a_stream) if a_stream is not None else nullcontext()

        def _collector(dst, obs0, seat0, mm_list, twin):
            # twin = the double-buffered record this rollout OWNS for its whole duration:
            # net + forward wrapper + state-trunc setting, all decided once at construction.
            try:
                if a_stream is not None:
                    a_stream.wait_stream(torch.cuda.default_stream(device))
                with _a_ctx():
                    r = collect_rollout(twin.net, envs, dst, shapes, enc.int_keys, device,
                                        args.num_steps, obs0, seat0, bf16=args.bf16_collect,
                                        lv=twin.lv, state_trunc=twin.strunc, opp_net=opp_net,
                                        opp_lv=opp_lv, agent_seat=agent_seat_np, mm=mm_list)
                if a_stream is not None:
                    a_stream.synchronize()               # buffers complete before the main thread reads
                a_state["r"] = r
                # staleness diagnostic (2026-07-11, kl-sawtooth hunt): which update generation
                # produced the weights this rollout ran with (stamped at refresh time)
                a_state["src_it"] = getattr(twin.net, "_refresh_it", -1)
                if _tpre_on:
                    # frozen-teacher pre-pass on the buffer just collected (eager, no dynamo ->
                    # thread-safe). t_lock: promotion must not swap teacher weights mid-forward;
                    # tc_ep records WHICH teacher was used (a promotion landing after this makes
                    # the cache stale -> discarded at join, inline fallback for that one iter).
                    tb_obs = {k: dst["obs"][k].reshape((-1, *shapes[k])) for k in shapes}
                    tb_sk = buffer_state_keep(tb_obs)
                    with t_lock:
                        ep0 = t_epoch["n"]
                        with _a_ctx():
                            tc = teacher_prepass(tb_obs, tb_sk)
                        if a_stream is not None:
                            a_stream.synchronize()
                    a_state["tc"] = tc; a_state["tc_ep"] = ep0
            except Exception as e:                       # surfaced at the next join
                a_state["err"] = e

        def _spawn_collector():
            """Launch the NEXT rollout on the active twin -> overlaps this iteration's update."""
            nonlocal a_pending
            a_mm: list = []
            a_state.clear(); a_state["mm"] = a_mm
            a_pending = threading.Thread(target=_collector,
                                         args=(a_buf2, next_obs, next_seat, a_mm,
                                               twins[c_active]), daemon=True)
            a_pending.start()

    # SYNC-PATH SPEED (2026-07-15, ported from the async stack after the sync fork validated the
    # loop shape; both are exact-math-class changes -- collect logp is behavior-policy data and
    # the PPO ratio absorbs the dense/bf16 numerics delta, same contract as the async twins):
    #   (a) --collect-graphs: ONE dense cudagraphed twin, refreshed from the live net every
    #       iteration BEFORE collection (in-place copy keeps the graphs' baked pointers valid;
    #       stale=0 semantics preserved -- the twin IS the current policy).
    #   (b) --teacher-prepass-collect: the teacher is FROZEN during a sync rollout (promotions
    #       happen after the update), so its prepass is chunked onto a LOW-priority side stream
    #       from collect_rollout's step_hook -- the GPU fills env-step gaps with teacher work and
    #       the update no longer pays the inline prepass. state_keep=None + opt_len=None keep the
    #       targets exact (truncation/bucketing are exact-by-masking); only kernel-shape fp noise
    #       differs (same class as the bf16 the teacher already runs under).
    s_twin = None
    b_twin = None
    s_tpre = None
    if not args.async_collect and args.collect_graphs:
        _sn = build_token_net(ct, net_config).to(device)
        _sn.load_state_dict(net.state_dict()); _sn.eval()
        _sn.varlen_attn = False                       # dense: whole forward dynamo-traceable
        _sn.varlen_compile = False
        for _p in _sn.parameters():
            _p.requires_grad_(False)
        _scfn = torch.compile(_sn.logits_value, mode="reduce-overhead", dynamic=False)

        def _s_safe(o, opt_len=None, state_keep=None, _c=_scfn, _e=_sn.logits_value):
            try:
                return _c(o, opt_len=opt_len, state_keep=state_keep)
            except RuntimeError as e:
                if ("symbolically trace" not in str(e) and "stream is capturing" not in str(e)):
                    raise
                return _e(o, opt_len=opt_len, state_keep=state_keep)
        s_twin = SimpleNamespace(net=_sn, lv=_s_safe, strunc=False)
        _t0 = time.time()
        with torch.no_grad():
            for _L in _DEFAULT_OPT_BUCKETS:
                for _ in range(3):                    # compile, record, replay per bucket shape
                    if args.bf16_collect:
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            s_twin.lv(next_obs, opt_len=int(_L), state_keep=None)
                    else:
                        s_twin.lv(next_obs, opt_len=int(_L), state_keep=None)
            torch.cuda.synchronize()
        if is_main:
            print(f"[collect-graphs] SYNC: {len(_DEFAULT_OPT_BUCKETS)} bucket graphs captured in "
                  f"{time.time() - _t0:.0f}s (dense twin, refreshed per iteration, "
                  f"bf16={args.bf16_collect})", flush=True)
        if opp_lv is not None:
            # warm the frozen-opponent graphs over the same bucket shapes (capture once,
            # no refresh -- the weights never change)
            _t0 = time.time()
            with torch.no_grad():
                for _L in _DEFAULT_OPT_BUCKETS:
                    for _ in range(3):
                        if args.bf16_collect:
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                opp_lv(next_obs, opt_len=int(_L), state_keep=None)
                        else:
                            opp_lv(next_obs, opt_len=int(_L), state_keep=None)
                torch.cuda.synchronize()
            if is_main:
                print(f"[collect-graphs] frozen-opponent graphs warmed in {time.time() - _t0:.0f}s",
                      flush=True)
        if args.two_net_finetune and opp_net is not None:
            # TWO-NET B collect twin (2026-07-31 sps): dense reduce-overhead twin of the LIVE B,
            # exactly the actor's sync-twin contract -- refreshed in place per iteration below,
            # full-batch forward in rollout with eager sampling on the selected rows. Replaces
            # the eager per-step subset forward that made ft collect ~1.6x production's.
            _bn = build_token_net(ct, ock["net_config"]).to(device)
            _bn.load_state_dict(opp_net.state_dict()); _bn.eval()
            _bn.varlen_attn = False
            _bn.varlen_compile = False
            for _p in _bn.parameters():
                _p.requires_grad_(False)
            _bcfn = torch.compile(_bn.logits_value, mode="reduce-overhead", dynamic=False)

            def _b_safe(o, opt_len=None, state_keep=None, _c=_bcfn, _e=_bn.logits_value):
                try:
                    return _c(o, opt_len=opt_len, state_keep=state_keep)
                except RuntimeError as e:
                    if ("symbolically trace" not in str(e) and "stream is capturing" not in str(e)):
                        raise
                    return _e(o, opt_len=opt_len, state_keep=state_keep)
            b_twin = SimpleNamespace(net=_bn, lv=_b_safe)
            _t0 = time.time()
            with torch.no_grad():
                for _L in _DEFAULT_OPT_BUCKETS:
                    for _ in range(3):
                        if args.bf16_collect:
                            with torch.autocast("cuda", dtype=torch.bfloat16):
                                b_twin.lv(next_obs, opt_len=int(_L), state_keep=None)
                        else:
                            b_twin.lv(next_obs, opt_len=int(_L), state_keep=None)
                torch.cuda.synchronize()
            opp_lv = b_twin.lv
            if is_main:
                print(f"[collect-graphs] two-net B twin graphs captured in {time.time() - _t0:.0f}s "
                      f"(dense, refreshed per iteration)", flush=True)
    # HELD BACK (2026-07-15 stud-3 arm C): the chunked sync prepass destabilized within 15
    # iterations (kl->0.12, clip->0.37, ent pump, vdiag degraded) -- teacher targets subtly
    # wrong somewhere (ordering/masking/opt_len=None path). Requires an equality test vs the
    # inline teacher_prepass on an identical buffer before it may ship; opt-in via
    # PKMN_SYNC_TPRE=1 for that debugging only. --teacher-prepass-collect in SYNC mode is
    # otherwise a NO-OP (the update computes the prepass inline, exactly as before).
    _s_tpre_on = (not args.async_collect and args.teacher_prepass_collect
                  and teacher_net is not None
                  and (args.teacher_kl_coef > 0 or args.teacher_value_coef > 0)
                  and device.type == "cuda"
                  and os.environ.get("PKMN_SYNC_TPRE") == "1")
    if _s_tpre_on:
        s_stream = torch.cuda.Stream(priority=0)      # LOW priority: never delays collect fwds
        _S_CHUNK = 8                                  # steps per teacher chunk (rows = 8 x envs)
        s_tpre = {"parts": [], "next": 0}

        def _s_prepass_range(t0, t1):
            cmb = {k: buf["obs"][k][t0:t1].reshape((-1, *shapes[k])) for k in shapes}
            # rows [t0:t1] were written on the DEFAULT stream; the side stream must observe
            # those writes before its teacher forward reads them (cross-stream ordering).
            s_stream.wait_stream(torch.cuda.default_stream(device))
            with torch.no_grad(), torch.cuda.stream(s_stream), \
                 torch.autocast("cuda", dtype=torch.bfloat16):
                cl, cv, cvd = teacher_net.logits_value(cmb, return_vdist=True,
                                                       opt_len=None, state_keep=None)
            # RAW side-stream tensors only: the old code called .float() HERE, which launches
            # on the DEFAULT stream with no ordering vs the side-stream forward -> a data race
            # that sporadically fed garbage teacher targets (the 2026-07-15 arm-C kl blowup).
            # All reads of these tensors happen in _s_tpre_finish AFTER s_stream.synchronize().
            s_tpre["parts"].append((cl, cv, cvd))

        def _s_hook(step):
            if s_tpre.get("skip"):                    # turns fast-path: frozen A needs no targets
                return
            if step + 1 - s_tpre["next"] >= _S_CHUNK:
                _s_prepass_range(s_tpre["next"], step + 1)
                s_tpre["next"] = step + 1

        def _s_tpre_finish():
            if s_tpre.get("skip"):
                s_tpre["parts"] = []; s_tpre["next"] = 0
                return None                           # update_net skips A anyway (inline fallback safe)
            if s_tpre["next"] < args.num_steps:       # flush the tail chunk
                _s_prepass_range(s_tpre["next"], args.num_steps)
            s_stream.synchronize()                    # side-stream tensors safe on default stream
            parts = s_tpre["parts"]
            s_tpre["parts"] = []; s_tpre["next"] = 0
            return {"logits": torch.cat([p[0] for p in parts]).float(),
                    "val": torch.cat([p[1] for p in parts]).float(),
                    "vdist": (torch.cat([p[2] for p in parts]).float()
                              if parts and parts[0][2] is not None else None)}
        if is_main:
            print("[sync] teacher pre-pass chunked onto a side stream during collection "
                  f"(chunk={_S_CHUNK} steps; exact targets)", flush=True)

    _cw_was = False
    _stop_hold = [0]                        # consecutive iters over --ft-wr-stop (curriculum leg)
    for it in range(start_it, num_iters + 1):
        if args.anneal_lr:
            if opt_b is not None:                       # TWO-NET: B shares A's lr schedule exactly
                opt_b.param_groups[0]["lr"] = lr_at(it - sched_offset, decay_iters, args.lr,
                                                    args.lr_schedule, warmup_iters, args.lr_min_ratio)
            opt.param_groups[0]["lr"] = lr_at(it - sched_offset, decay_iters, args.lr,
                                              args.lr_schedule, warmup_iters, args.lr_min_ratio)
        if _PROF and device.type == "cuda":
            torch.cuda.synchronize()
        _t0 = time.time()
        t_pre = None                                     # collector-computed teacher cache (async)
        a_stale = -1                                     # rollout weight-age in iters (async diag)
        if not args.async_collect:
            if s_twin is not None:
                # per-iteration refresh: the twin IS the current policy (stale=0); in-place copy
                # keeps the cudagraph wrappers' baked parameter pointers valid.
                s_twin.net.load_state_dict(net.state_dict())
                if b_twin is not None:                       # two-net: B's twin refreshes likewise
                    b_twin.net.load_state_dict(opp_net.state_dict())
                _c_net, _c_lv, _c_st = s_twin.net, s_twin.lv, s_twin.strunc
            else:
                _c_net, _c_lv, _c_st = net, collect_lv, args.state_trunc
            if _s_tpre_on:
                # turns fast-path: don't build A-teacher targets while A is frozen (update_net
                # skips A entirely; a flip mid-window just falls back to one inline prepass)
                s_tpre["skip"] = (args.ft_balance_mode == "turns" and opp_net is not None
                                  and cg_state.get("turn", "B") == "B")
            boot_val, boot_seat, _eprs, next_obs, _terms = collect_rollout(
                _c_net, envs, buf, shapes, enc.int_keys, device, args.num_steps, next_obs, next_seat,
                bf16=args.bf16_collect, lv=_c_lv, state_trunc=_c_st,
                opp_net=opp_net, opp_lv=opp_lv, agent_seat=agent_seat_np, mm=mm_events,
                step_hook=_s_hook if _s_tpre_on else None, train_opp=args.two_net_finetune,
                frozen_net=fw_net, frozen_envs=fw_envs_np, fw=fw_outcomes,
                frozen_sample=args.frozen_wide_sample, frozen_lv=fw_lv,
                ks_net=ks_net, ks_envs=ks_envs_np, ksq=ks_outcomes, ks_lv=ks_lv)
            if _s_tpre_on:
                t_pre = _s_tpre_finish()
                if os.environ.get("PKMN_TPRE_EQ") == "1":
                    # equality audit vs the canonical inline prepass (same buffer): residual
                    # diffs must be kernel-shape fp noise only (opt_len/state_keep=None are
                    # exact-by-masking; both paths bf16). Large or row-sparse-huge diffs = bug.
                    _eq_obs = {k: buf["obs"][k].reshape((-1, *shapes[k])) for k in shapes}
                    _eq_ref = teacher_prepass(_eq_obs, buffer_state_keep(_eq_obs))
                    _dl = (t_pre["logits"] - _eq_ref["logits"]).abs()
                    _dv = (t_pre["val"] - _eq_ref["val"]).abs()
                    _p = torch.softmax(_eq_ref["logits"], -1)
                    _q = torch.log_softmax(t_pre["logits"], -1)
                    _kl = (_p * (torch.log(_p.clamp_min(1e-9)) - _q)).sum(-1)
                    print(f"[tpre-eq] it={it} dlogit max={_dl.max():.4e} p99={torch.quantile(_dl.flatten().float(), 0.99):.4e} "
                          f"dval max={_dv.max():.4e} klrow max={_kl.max():.4e} mean={_kl.mean():.4e}",
                          flush=True)
            next_seat = boot_seat
        else:
            if a_pending is None:                        # serial warmup (and the very first prime)
                # pinned to twins[0] (NOT c_active): preserves the long-standing serial-warmup
                # alternation where odd serial iters collect with 1-update-stale weights -- same
                # staleness class as steady-state async, PPO-valid (stored logp self-consistent).
                boot_val, boot_seat, _eprs, next_obs, _terms = collect_rollout(
                    twins[0].net, envs, buf, shapes, enc.int_keys, device, args.num_steps,
                    next_obs, next_seat, bf16=args.bf16_collect,
                    lv=twins[0].lv, state_trunc=twins[0].strunc,
                    opp_net=opp_net, opp_lv=opp_lv, agent_seat=agent_seat_np,
                    mm=mm_events)
            else:                                        # join rollout collected during last update
                a_pending.join()
                if "err" in a_state:
                    raise a_state["err"]
                boot_val, boot_seat, _eprs, next_obs, _terms = a_state.pop("r")
                mm_events.extend(a_state.pop("mm"))
                t_pre = a_state.pop("tc", None)          # teacher cache for the SAME bank we swap in
                if t_pre is not None and a_state.pop("tc_ep", -1) != t_epoch["n"]:
                    t_pre = None                         # teacher promoted since -> recompute inline
                _src = a_state.pop("src_it", -1)
                a_stale = (it - _src) if _src >= 0 else -1   # rollout weight-age in iters (diag)
                buf, a_buf2 = a_buf2, buf                # filled bank becomes current
            next_seat = boot_seat
            # SERIAL WARMUP (fix for the S3 crash 2026-07-10): dynamo is NOT thread-safe -- the
            # iter-1 update compile (and the first few automatic-dynamic re-specializations) FX-
            # traces on the MAIN thread, and a collector thread merely CALLING its compiled forward
            # during that window dies with "using FX to symbolically trace a dynamo-optimized
            # function" (cold-cache stud-3; hopper survived on warm inductor cache only). So the
            # first --async-serial-iters iterations run fully synchronous; pipelining starts once
            # the update graph has settled. The spawn for iter N+1 happens AFTER update N.
            if a_serial_left <= 0:
                _spawn_collector()                       # NEXT rollout now -> overlaps this update
        global_step += args.batch_size * world
        # banner reads the SAME post-increment predicate update_net acts on (they can never
        # disagree on the boundary iteration -- the old pre-increment banner could)
        _cw = global_step < args.critic_warmup_until
        if is_main and _cw and not _cw_was:
            print(f"[critic-warmup] ACTIVE: value-only updates until step "
                  f"{args.critic_warmup_until:,} (pg+ent zeroed; teacher terms pin the policy)",
                  flush=True)
        if is_main and _cw_was and not _cw:
            print(f"[critic-warmup] done at step {global_step:,} -> full PPO resumes", flush=True)
        _cw_was = _cw
        if mm_events:                       # flush the matchup window (delta counts, time-stamped)
            _cells: dict = {}
            for _i, _j, _o in mm_events:
                _k = f"{_i}:{_j}"
                _c = _cells.setdefault(_k, [0, 0, 0, 0])
                _c[_o] += 1
            with open(mm_path, "a") as _f:
                _f.write(json.dumps({"step": int(global_step), "cells": _cells}) + "\n")
            if opp_net is not None:
                # FINETUNE progress = training wr vs the STATIC stage-1 net (user design 2026-07-09:
                # the collection stream IS that comparison -- no gate games). The AGENT deck id is
                # always NEGATIVE (-1 unnamed, or -(pool_idx+2) in deck-SET mode); outcome 0 =
                # seat-0's deck won. Test the SIGN, not the literal -1: a deck-set run records real
                # deck identities and an `== -1` test silently inverts every seat-0 episode, pinning
                # ft_wr at ~0.5 for ANY true winrate (and corrupting the promotion signal below).
                for _i, _j, _o in mm_events:
                    if _o in (0, 1):
                        ft_wr_win.append(1.0 if (_i < 0) == (_o == 0) else 0.0)
                del ft_wr_win[:-2000]
            mm_events.clear()
        if vd is not None and _terms:
            vd.add(buf["val"].cpu().numpy(), buf["done"].cpu().numpy(),
                   buf["seat"].cpu().numpy(), _terms)
            if it % args.diag_every == 0:
                _, _vline = vd.summary(step=global_step)
                print(f"[vdiag] step={global_step} {_vline}", flush=True)
        # TWO-NET BALANCE GATING (user design 2026-08-02): rolling live A-vs-B wr, all-reduced
        # so every rank takes the same decision; outside the band the winning net's scales are
        # zeroed (hard: everything; soft: policy/entropy only) until balance returns. 400-game
        # warmup before the first gating decision.
        if opp_net is not None and args.ft_balance_mode != "off":
            _blw = float(sum(ft_wr_win)); _bln = float(len(ft_wr_win))
            if ddp:
                _blt = torch.tensor([_blw, _bln], device=device)
                dist.all_reduce(_blt)
                _blw, _bln = _blt.tolist()
            _bgate = "none"
            _bwr = _blw / _bln if _bln > 0 else 0.5
            if args.ft_balance_mode == "turns":
                # PROMOTION-DRIVEN ALTERNATION (user design 2026-08-03 v2): exactly ONE net
                # trains at a time (stationary value targets within a phase -- the v_ce
                # 0.014->0.051 blowup of simultaneous training was the original motivation).
                # B trains first. A phase ends when the TRAINING side EARNS its cross-gate
                # promotion (the flip fires in cross_gate_promote): prove +delta over your
                # own frozen snapshot on the gate mixture, bank it, hand over -- a bounded
                # partial best response, so the alternation cannot cycle deep, and every
                # flip hands the opponent a freshly-PROMOTED frozen target (teacher
                # bookkeeping coherent by construction). Absolute inline bands are retired
                # here: the 2026-08-03 ceiling audit measured the inline floor as mixture-
                # structural (wide third ~0.74 wall; 0.45 unreachable by construction).
                # Backstop: --ft-phase-timebox flips a phase that cannot promote. Turn
                # state lives in cg_state -> CHECKPOINTED, survives legs/restarts.
                _turn = cg_state.setdefault("turn", "B")
                cg_state.setdefault("phase_start", global_step)
                if global_step - cg_state["phase_start"] >= args.ft_phase_timebox:
                    # TIMEBOX-BANK (2026-08-04, from the 680M cells: a timeboxed phase can carry
                    # real progress its teacher never captured -- A gained +2.9pt on the counter
                    # tier while its gate read flat). Sync the trainee's teacher to its LIVE
                    # state iff its last gate shows non-collapse (floor = baseline -/+ delta;
                    # the -10 arm's collapsed-A phase is exactly what the guard excludes). Exam
                    # re-aligns with curriculum for the next phase; best.pt stays promotion-only.
                    _tb_base = cg_state.get("base"); _tb_d = args.cross_gate_delta
                    if _turn == "A":
                        _lw = cg_state.get("last_a_wr")
                        if _tb_base is not None and _lw is not None and _lw >= _tb_base - _tb_d:
                            with t_lock:
                                teacher_net.load_state_dict(net.state_dict()); teacher_net.eval()
                                t_epoch["n"] += 1
                            cg_state["base"] = None
                            if is_main:
                                print(f"[turns] step={global_step} TIMEBOX-BANK: T_A <- live A "
                                      f"(last gate {_lw:.4f} >= floor {_tb_base - _tb_d:.4f})", flush=True)
                    else:
                        _lw = cg_state.get("last_b_wr")
                        if _tb_base is not None and _lw is not None and _lw <= _tb_base + _tb_d:
                            t_b_net.load_state_dict(opp_net.state_dict()); t_b_net.eval()
                            cg_state["base"] = None
                            if is_main:
                                print(f"[turns] step={global_step} TIMEBOX-BANK: T_B <- live B "
                                      f"(last gate {_lw:.4f} <= ceil {_tb_base + _tb_d:.4f})", flush=True)
                    _turn = cg_state["turn"] = "A" if _turn == "B" else "B"
                    cg_state["phase_start"] = global_step
                    if is_main:
                        print(f"[turns] step={global_step} TIMEBOX flip (no promotion in "
                              f"{args.ft_phase_timebox:.0f} steps) -> training {_turn}", flush=True)
                _bgate = "A" if _turn == "B" else "B"          # gate = the FROZEN side
                _bal_state["a_trained"] = _bal_state.get("a_trained", False) or _turn == "A"
                _bal_state["b_trained"] = _bal_state.get("b_trained", False) or _turn == "B"
                _on_a = 1.0 if _turn == "A" else 0.0
                _bal["pa"].fill_(_on_a); _bal["aa"].fill_(_on_a)
                _bal["pb"].fill_(1.0 - _on_a); _bal["ab"].fill_(1.0 - _on_a)
            else:
                if _bln >= 400:
                    _bgate = "A" if _bwr > _bal_hi else ("B" if _bwr < _bal_lo else "none")
                _soft = args.ft_balance_mode == "soft"
                _bal["pa"].fill_(0.0 if _bgate == "A" else 1.0)
                _bal["aa"].fill_(0.0 if (_bgate == "A" and not _soft) else 1.0)
                _bal["pb"].fill_(0.0 if _bgate == "B" else 1.0)
                _bal["ab"].fill_(0.0 if (_bgate == "B" and not _soft) else 1.0)
            if is_main and _bgate != _bal_state["gate"]:
                _tinfo = f" (turn={cg_state['turn']})" if args.ft_balance_mode == "turns" else ""
                print(f"[balance] step={global_step} rolling_wr={_bwr:.3f} "
                      f"(n={int(_bln)}) -> gate={_bgate}{_tinfo}", flush=True)
                _bal_state["gate"] = _bgate
        # ASYMMETRIC TRUNC PENALTY (2026-08-01, post-cliff fix, v2 two-sided rewards): with
        # PKMN_TRUNC_ASYM=1 the anti-loop truncation is a TWO-SIDED terminal — the looping seat
        # keeps its full -PEN reward (its whole final run of decisions inherits -PEN-scale
        # credit, and V learns the true cost), while the victim's last decision is anchored to
        # an explicit +1 WIN inside compute_gae_2s, severing the negamax mirror that turned
        # -PEN into a +PEN victim jackpot (the reward-hack that collapsed production tch_wr).
        # Trunc rows are identified by their unique env reward (-PEN, PEN >= 2).
        _tmask = ((buf["done"] > 0) & (buf["rew"] < -1.5)) if _TRUNC_ASYM else None
        advantages, returns = compute_gae_2s(buf["rew"], buf["val"], buf["done"], buf["seat"],
                                             boot_val, boot_seat, args.gamma, args.gae_lambda,
                                             trunc=_tmask)
        if _PROF and device.type == "cuda":
            torch.cuda.synchronize()
        _t1 = time.time()
        clipfracs, approx_kls, last = update_net(advantages, returns, t_pre)
        if _PROF and device.type == "cuda":
            torch.cuda.synchronize()
        _ct, _ut = _t1 - _t0, time.time() - _t1
        if args.async_collect:                           # next spawn uses the post-THIS-update policy
            # DOUBLE-BUFFER refresh: load the IDLE twin (the in-flight thread owns the other
            # one; weights are never mutated under a running forward), flip for the next spawn.
            # load_state_dict copies IN-PLACE -> the cudagraph wrappers' baked pointers stay valid.
            _idle = 1 - c_active
            twins[_idle].net.load_state_dict(net.state_dict())
            twins[_idle].net._refresh_it = it            # staleness stamp (see a_state["src_it"])
            c_active = _idle
            if a_serial_left > 0:                        # serial warmup: count down; pipeline when settled
                a_serial_left -= 1
                if a_serial_left == 0:
                    if is_main:
                        print(f"[async] serial warmup done (it={it}) -> pipelining", flush=True)
                    with torch.no_grad():                # prewarm BOTH twins' compiles on the MAIN
                        _po = {k: v[:2] for k, v in next_obs.items()}   # thread (never in-thread dynamo)
                        for tw in twins:
                            tw.net.logits_value(_po)
                    _spawn_collector()

        if teacher_net is not None and global_step - last_promote >= args.teacher_refresh:
            wr = (cross_gate_promote(global_step) if args.two_net_finetune
                  else maybe_promote(global_step)); last_promote = global_step
            if wr is not None:
                last_tch_wr = wr

        if is_main and it % args.log_every == 0:
            # sps = THIS iteration's rate (collect+update; excludes gate/ckpt pauses). avg = true
            # whole-process mean incl. those pauses, counting only steps done SINCE RESUME (the old
            # global_step/elapsed read ~900k right after resuming a 100M ckpt).
            sps = int(args.batch_size * world / max(_ct + _ut, 1e-6))
            avg = int((global_step - resume_step) / max(time.time() - start, 1e-6))
            # symmetric two-sided self-play has NO per-iter agent-vs-opponent winrate (both seats are the
            # SAME live net -> ~0.5, useless); the meaningful "winrate" is vs the gated teacher, which
            # updates every --teacher-refresh and is carried forward here so it shows on every line.
            if _PROF and args.async_collect:
                # async: _ct is the JOIN WAIT (rollout ran under the update); per-rollout step time
                # is written by the collector thread for the IN-FLIGHT rollout -> report separately
                prof = f" join={_ct:.1f}s update={_ut:.1f}s (overlap; inflight-step={_STEP_T[0]:.1f}s)"
            elif _PROF:
                prof = (f" collect={_ct:.1f}s[step={_STEP_T[0]:.1f}s fwd={_ct-_STEP_T[0]:.1f}s] update={_ut:.1f}s "
                        f"(c-sps={int(args.batch_size*world/max(_ct,1e-6))})")
            else:
                prof = ""
            _ftw = (f" ft_wr={sum(ft_wr_win)/len(ft_wr_win):.3f}[n={len(ft_wr_win)}]"
                    if ft_wr_win else "")
            _ftw += (f" fw_wr={sum(fw_outcomes)/len(fw_outcomes):.3f}[n={len(fw_outcomes)}]"
                     if fw_outcomes else "")
            _ftw += (f" ks_wr={sum(ks_outcomes)/len(ks_outcomes):.3f}[n={len(ks_outcomes)}]"
                     if ks_outcomes else "")
            _st = f" stale={a_stale}" if (args.async_collect and a_stale >= 0) else ""
            _bstats = (f" b_pg={last['b_pg']:.3f} b_v={last['b_v']:.3f} b_ent={last['b_ent']:.3f}"
                       + (f" b_tkl={last['b_tkl']:.4f}" if last.get("b_tkl") else "")
                       + (f" a_tkl={last['a_tkl']:.4f}" if last.get("a_tkl") else "")
                       if "b_pg" in last else "")
            print(f"step={global_step} it={it}/{num_iters} sps={sps} avg={avg} tch_wr={last_tch_wr:.3f}"
                  f"{_ftw} pg={last['pg']:.3f} v_ce={last['v']:.3f} ent={last['ent']:.3f} "
                  f"clip={np.mean(clipfracs):.3f} kl={np.mean(approx_kls):.4f}{_bstats}{_st}{prof}",
                  flush=True)

        def _save_latest():
            torch.save({"net": net.state_dict(), "args": vars(args), "opt": opt.state_dict(),
                        "teacher": (teacher_net.state_dict() if teacher_net is not None else None),
                        "net_config": net_config, "global_step": global_step,
                        "ft_best": ft_anchor["best"],
                        "sched_anchor": int(sched_offset * args.batch_size * world),
                        **({"net_b": opp_net.state_dict(), "opt_b": opt_b.state_dict(),
                            "t_b": t_b_net.state_dict(), "cg_state": dict(cg_state)}
                           if opt_b is not None else {})},
                       os.path.join(args.out, "latest.pt"))

        if is_main and (global_step % args.save_every < args.batch_size * world or it == num_iters):
            path = os.path.join(args.out, f"ckpt_{global_step}.pt")
            torch.save({"net": net.state_dict(), "args": vars(args),
                        "net_config": net_config, "global_step": global_step,
                        **({"net_b": opp_net.state_dict()} if opt_b is not None else {})}, path)
            _save_latest()
            print(f"[ckpt] saved {path}", flush=True)

        # CURRICULUM early exit: the leg's job is to beat the CURRENT rung, not to burn its whole
        # step budget. All-reduced so every rank leaves the loop on the same iteration.
        if args.ft_wr_stop is not None and ft_wr_win:
            _sw, _sn = float(sum(ft_wr_win)), float(len(ft_wr_win))
            if ddp:
                _stt = torch.tensor([_sw, _sn], device=device)
                dist.all_reduce(_stt)
                _sw, _sn = float(_stt[0].item()), float(_stt[1].item())
            _hit = _sn >= args.ft_wr_stop_min_n and (_sw / _sn) >= args.ft_wr_stop
            _stop_hold[0] = _stop_hold[0] + 1 if _hit else 0
            if _stop_hold[0] >= args.ft_wr_stop_hold:
                if is_main:
                    print(f"[ft-wr-stop] ft_wr={_sw/_sn:.4f} >= {args.ft_wr_stop} held "
                          f"{_stop_hold[0]} iters (n={int(_sn)}) -> LEG COMPLETE at "
                          f"step {global_step:,}", flush=True)
                    _save_latest()
                break

    if a_pending is not None:                            # drain the in-flight rollout before teardown
        a_pending.join()
    envs.close()
    if ddp:
        dist.barrier(); dist.destroy_process_group()
    if is_main:
        print("done.", flush=True)


if __name__ == "__main__":
    main()
