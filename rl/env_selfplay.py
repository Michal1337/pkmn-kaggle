"""Two-sided (all-seats) self-play env for cabt -- the C9 collection variant.

Unlike ``rl.env.CabtEnv`` (single-agent: the opponent is played INSIDE the worker and only the
agent's own decisions are surfaced), this env plays NO internal opponent. It drives the engine and
SURFACES EVERY decision, tagged with the seat (0/1) that must act, so the trainer can collect
transitions for BOTH seats -- the LIVE net pilots both sides (pure self-play, ~2x the matchup
experience per game, which is the point for a multi-deck generalist).

Design (mirrors CabtEnv's engine-driving + per-side trackers, minus the opponent loop):
  * ``reset()  -> (encoded_obs, seat, info)``  : start a battle (both decks sampled from the pool),
    surface the first decision.
  * ``step(a)  -> (encoded_obs, seat, reward, done, info)`` : apply ``a`` for the CURRENT seat
    (buffering multi-select like CabtEnv), advance to the NEXT decision (which may be the same seat
    continuing its turn, or the other seat), and surface it. ``reward`` is the ACTING seat's reward
    (0 mid-game; the seat's terminal outcome on the move that ends the game). The OTHER seat's credit
    is handled by the trainer's NEGAMAX GAE (sign-flip bootstrap when the acting seat changes), so the
    env never needs to retro-assign the loser's reward.
  * per-seat ``GameTracker``/``AbilityTracker`` each fed ONLY their own seat's decision obs (the cabt
    interpreter sends an obs only to the player about to move -> train == test for both seats).

Terminal reward (per seat): +1 winner / -1 loser / 0 draw, optionally scaled by the F19 victory
margin (loser's remaining prizes). An illegal move forfeits that seat.
"""

from __future__ import annotations

import ctypes
import logging
import os
import random

import numpy as np

logging.disable(logging.CRITICAL)

from .encoding import TokenEncoder, SUBMIT_ACTION, build_mask
from .encoding import GameTracker, AbilityTracker
from .card_features import get_card_table
from .env import load_deck, _BinEngine


_FORENSICS_DIR = os.environ.get("PKMN_DECK_FORENSICS_DIR")   # crash forensics (2026-07-10): the
# v2-pool legs die every ~3-5h to a worker-process death (C++ segfault -> pipe EOF; Python-level
# recovery can't catch it) that the old 1,107-deck pool never produced -> suspect = a rare
# mid-game engine crash on one of the 1,077 NEW lists. Each worker appends its (i, j) pool
# indices BEFORE starting every battle; after a crash, the file whose mtime froze first names
# the culprit pair. Pure logging: OFF unless the env var is set.

# STALL-LOOP GUARD (2026-07-25, user design): shell decks exploit truncation=0.0 by looping a
# free ability (ABILITY->ENERGY->CARD, ~50% of their games) -- on the ladder that burns the
# 600s overage clock = a LOSS. Two gated knobs, both OFF unless set (unset = byte-identical):
#   PKMN_TURN_CAP=N       end the episode when the engine's own turnActionCount reaches N
#                         (encoder full-scale turn = 20; legit combos < ~50; the loop burns
#                         1000+). gamma=0.997 puts the whole capped run inside real credit.
#   PKMN_TRUNC_PENALTY=1  truncation pays -1 to the ACTING seat (ladder-truthful timeout loss);
#                         the other seat's + side arrives via the two-sided GAE seat-flip
#                         (zero-sum), and mm logging still records outcome 3 (no "terminal").
_TURN_CAP = int(os.environ.get("PKMN_TURN_CAP", "0") or "0")
# numeric since 2026-08-01 (loop-penalty experiment): "1" = ladder-truthful timeout loss
# (production, unchanged), higher values = anti-loop shock therapy (the fossil-equilibrium
# probe). Unset/empty = 0.0 = the pre-turn-cap draw semantics.
_TRUNC_PENALTY = float(os.environ.get("PKMN_TRUNC_PENALTY", "") or "0")


class TwoSidedSelfPlayEnv:
    """Both seats piloted by the trainer's live net; every decision surfaced with its seat tag."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        decks: list[list[int]] | None = None,    # POOL: BOTH seats sampled per episode (symmetric self-play)
        encoder: TokenEncoder | None = None,
        would_ko: bool = False,
        terminal_margin: float = 0.0,            # F19: scale terminal reward by victory margin
        max_steps: int = 4000,
        native_encode: bool = False,             # JSON-free collection: sdk_cg + GetBinaryObs + native encode (both seats)
        seed: int | None = None,
        deck_weights: list[float] | None = None,  # per-deck sampling weights (tf-idf density design,
        #                                           2026-07-10); None = uniform (byte-identical rng)
    ):
        self.decks = decks or [load_deck()]
        self.deck_weights = None
        if deck_weights is not None:
            assert len(deck_weights) == len(self.decks), \
                f"deck_weights {len(deck_weights)} != decks {len(self.decks)}"
            self.deck_weights = [float(x) for x in deck_weights]
        self.encoder = encoder or TokenEncoder(get_card_table())
        self.would_ko = bool(would_ko)
        self.terminal_margin = float(terminal_margin)
        self.max_steps = max_steps
        self.rng = random.Random(seed)

        self.trackers = [GameTracker(), GameTracker()]
        self.abilities = [AbilityTracker(), AbilityTracker()]
        self.deck: list[list[int] | None] = [None, None]
        self._last_tracked = [None, None]
        self._obs = None
        self._picked: list[int] = []
        self._steps = 0
        self._done = True
        self._forfeit: int | None = None         # a seat that made an illegal move -> it loses
        self._last_enc = None                    # cached encode (placeholder returned on terminal)
        self._native = bool(native_encode)
        self._game = None
        self._Battle = None
        self._wk = None                          # fast native would-ko driver (set only when the gate is on)

    # -- engine handle (lazy so logging is disabled first) -----------------
    def _ensure_engine(self):
        if self._game is not None:
            return
        if self._native:
            if self.would_ko:
                from rl.wk_native import wk_native_enabled
                if not wk_native_enabled():      # gate off -> unchanged behavior
                    raise ValueError("native_encode (JSON-free) is incompatible with would_ko; use --no-would-ko.")
            from rl.env import _native_handles
            B, L = _native_handles()          # per-instance battle holder (K envs per process OK)
            self._Battle, self._lib = B, L
            from rl.native_enc import NativeEncoder
            self._nenc = NativeEncoder(self.encoder)
            self._CAP = 16384
            self._cbuf = (ctypes.c_int * self._CAP)()
            self._cbuf_np = np.ctypeslib.as_array(self._cbuf)
            self._game = _BinEngine(L, B, self._cbuf, self._cbuf_np, self._CAP)
            if self.would_ko:                 # fast engine-side would-ko (PKMN_WK_NATIVE=1)
                from rl.wk_native import WKNative
                self._wk = WKNative(L, self.encoder)
        else:
            from kaggle_environments.envs.cabt.cg import game as g
            from kaggle_environments.envs.cabt.cg.sim import Battle as B
            self._game, self._Battle = g, B

    def _engine(self):
        self._ensure_engine()
        return self._game

    def _state(self):
        return self._obs["current"]

    def _seat(self) -> int:
        return int(self._state()["yourIndex"])

    def _sample_decks(self):
        """Per-episode (seat0, seat1) decks; subclass hook (FrozenOpponentEnv assigns asymmetric
        agent/opponent decks here). Base = symmetric: both seats sampled from the pool
        (uniform, or by `deck_weights` -- the tf-idf density sampling design).
        Records `self._deck_idx` = (pool_idx_seat0, pool_idx_seat1) for the matchup-matrix log.
        NB uniform path: `choice(seq)` == `seq[randrange(len(seq))]` consume the SAME rng
        stream -> episodes are byte-identical to the pre-logging code."""
        if self.deck_weights is not None:
            i0 = self.rng.choices(range(len(self.decks)), weights=self.deck_weights, k=1)[0]
            i1 = self.rng.choices(range(len(self.decks)), weights=self.deck_weights, k=1)[0]
        else:
            i0 = self.rng.randrange(len(self.decks))
            i1 = self.rng.randrange(len(self.decks))
        self._deck_idx = (i0, i1)
        return self.decks[i0], self.decks[i1]

    # -- gym-ish API --------------------------------------------------------
    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng.seed(seed)
        game = self._engine()
        for _attempt in range(32):
            self._safe_finish()
            d0, d1 = self._sample_decks()
            if _FORENSICS_DIR:
                try:
                    with open(os.path.join(_FORENSICS_DIR,
                                           f"worker_deck_{os.getpid()}.txt"), "a") as _f:
                        _f.write(f"{self._deck_idx[0]} {self._deck_idx[1]}\n")
                except Exception:
                    pass
            self.deck = [list(d0), list(d1)]
            obs, start = game.battle_start(d0, d1)
            if obs is None:
                raise RuntimeError(f"battle_start failed: errorPlayer={start.errorPlayer}")
            for t in self.trackers:
                t.reset()
            for a in self.abilities:
                a.reset()
            self._last_tracked = [None, None]
            self._obs = obs
            self._picked = []
            self._steps = 0
            self._done = False
            self._forfeit = None
            if self._state()["result"] < 0:      # a live decision exists (not an instant end)
                break
        seat = self._seat()
        enc = self._encode()
        return enc, seat, {"seat": seat}

    def step(self, action: int):
        if self._done:
            raise RuntimeError("step() after episode end; call reset().")
        info: dict = {}
        seat = self._seat()
        sel = self._obs["select"]
        mask = build_mask(sel, set(self._picked))
        action = int(action)
        if action >= len(mask) or mask[action] == 0:        # guard illegal picks (clamp to first legal)
            legal = [i for i, m in enumerate(mask) if m]
            action = legal[0]
            info["illegal_action"] = True

        submit = action == SUBMIT_ACTION
        if not submit:
            self._picked.append(action)

        if submit or len(self._picked) >= sel["maxCount"]:
            terminated = self._apply(seat, sorted(set(self._picked)))
            self._picked = []
            if terminated:
                self._done = True
                rewards = self._terminal_rewards()
                info["terminal"] = rewards                  # {0: r0, 1: r1} for diagnostics
                info["mm"] = getattr(self, "_deck_idx", (-1, -1))   # matchup-matrix deck ids
                return self._last_enc, seat, rewards[seat], True, info
            self._steps += 1
            if _TURN_CAP:                                   # stall-loop guard: one turn ran away
                cur = self._obs.get("current") or {}
                tac = cur.get("turnActionCount")
                if tac is None and self._native and self._game.last_arr is not None:
                    tac = int(self._game.last_arr[1])       # BinaryObs layout: [turn, turnActionCount, ...]
                if tac is not None and int(tac) >= _TURN_CAP:
                    self._done = True
                    return self._last_enc, seat, (-_TRUNC_PENALTY), True, \
                        {**info, "truncated": True, "trunc_turncap": True,
                         "mm": getattr(self, "_deck_idx", (-1, -1))}
            if self._steps >= self.max_steps:               # truncation (no terminal reward)
                self._done = True
                return self._last_enc, seat, (-_TRUNC_PENALTY), True, \
                    {**info, "truncated": True,
                     "mm": getattr(self, "_deck_idx", (-1, -1))}
            nseat = self._seat()
            return self._encode(), nseat, 0.0, False, info

        # still buffering this multi-select: same decision + seat, updated mask
        return self._encode(), seat, 0.0, False, info

    def close(self):
        self._safe_finish()

    # -- internals ----------------------------------------------------------
    def _encode(self):
        seat = self._seat()
        if self._native:                                     # JSON-free: binary + native tracker/encode
            arr = self._game.last_arr
            if self._obs is not self._last_tracked[seat]:
                self.trackers[seat].update_native(arr)
                self._game.advance_log()                     # consume logIndex[seat] once/decision
                if self._wk is not None:                     # fast would-ko (once/decision, like JSON)
                    from rl.wk_native import fill_wk
                    fill_wk(self._nenc.wk, self._wk.flags(self._Battle.battle_ptr, arr,
                                                          self._obs["select"], self.deck[seat]))
                self._last_tracked[seat] = self._obs
            self.abilities[seat].note_turn(self._state()["turn"])
            enc = self._nenc.encode(arr, self._obs["select"], self.deck[seat],
                                    self.trackers[seat], self.abilities[seat].slots, self._picked)
            self._last_enc = enc
            return enc
        if self._obs is not self._last_tracked[seat]:        # fold this decision's logs into THIS seat's tracker
            self.trackers[seat].update(self._obs)
            if self.would_ko:
                from rl import search_agent as _SA2
                _SA2.annotate_would_ko(self._obs, self.deck[seat], self.encoder)
            self._last_tracked[seat] = self._obs
        self.abilities[seat].note_turn((self._state() or {}).get("turn"))
        enc = self.encoder.encode(self._obs, set(self._picked), self_deck=self.deck[seat],
                                  tracker=self.trackers[seat], ability_slots=self.abilities[seat].slots)
        self._last_enc = enc
        return enc

    def _apply(self, seat: int, indices: list[int]) -> bool:
        """Submit ``seat``'s selection; return True if the game ended (incl. this seat forfeiting)."""
        game = self._engine()
        self.abilities[seat].record(self._obs["select"], indices)
        try:
            self._obs = game.battle_select(indices)
        except Exception:
            self._forfeit = seat                            # illegal move -> this seat loses
            return True
        return self._state()["result"] >= 0

    def _terminal_rewards(self) -> dict[int, float]:
        if self._forfeit is not None:                       # forfeiting seat loses, other wins
            w, l = 1 - self._forfeit, self._forfeit
            return {w: 1.0, l: -1.0}
        s = self._state()
        r = s["result"]
        if r == 2 or r < 0:                                 # draw / unfinished
            return {0: 0.0, 1: 0.0}
        win, lose = r, 1 - r
        if self.terminal_margin <= 0.0:
            return {win: 1.0, lose: -1.0}
        rem = len((s["players"][lose].get("prize") or []))  # loser's remaining prizes = victory margin
        m = self.terminal_margin * (rem / 6.0)
        return {win: 1.0 + m, lose: -(1.0 + m)}

    def _safe_finish(self):
        if self._Battle is None or not self._Battle.battle_ptr:
            return
        try:
            self._engine().battle_finish()
        except Exception:
            pass
        self._Battle.battle_ptr = None


# ==================== deck-finetune view (absorbed from rl/env_finetune.py) ====================
# Single-agent view over the two-sided env with a FROZEN opponent net piloting the other seat.
class AsymmetricDeckEnv(TwoSidedSelfPlayEnv):
    """Two-sided env with ASYMMETRIC per-episode decks: the agent deck on a RANDOM seat, the other
    seat's deck sampled by meta weight. ALL decisions surface (standard two-sided interface) -- the
    LEARNER supplies actions for BOTH seats, using the frozen opponent net (GPU-batched in
    collect_rollout) for the non-agent seat. This supersedes FrozenOpponentEnv for training:
    worker-side CPU forwards of the frozen net measured ~100ms/decision -> ~230 sps (unusable);
    GPU-batching in the learner keeps collection at self-mirror speed. info carries 'agent_seat'
    every step (and the worker forwards 'next_agent_seat' across auto-resets) for loss masking."""

    def __init__(self, agent_deck, opp_decks, opp_weights,
                 agent_decks=None, agent_weights=None, agent_ids=None, **kw):
        """agent_deck: the single fixed deck (back-compat). agent_decks: optional SET sampled per
        episode -- pass agent_weights to bias it, agent_ids to record real pool indices in the
        matchup log (so per-deck results stay readable; -1 means 'unnamed agent deck')."""
        self.agent_decks = ([list(d) for d in agent_decks] if agent_decks
                            else [list(agent_deck)])
        n_a = len(self.agent_decks)
        self.agent_weights = ([float(w) for w in agent_weights] if agent_weights
                              else [1.0] * n_a)
        assert len(self.agent_weights) == n_a, "agent_weights must match agent_decks"
        # The matchup event identifies the AGENT by a NEGATIVE deck id -- train_selfplay's ft_wr
        # (and the teacher-promotion signal it feeds) tests `_i < 0` to decide which seat was ours.
        # Recording a raw pool index here silently inverts ft_wr for every seat-0 episode, which
        # drives it to ~0.5 for ANY true winrate. So encode: -1 for an unnamed deck, else
        # -(pool_idx + 2), which stays negative and is decodable as pool_idx = -v - 2.
        _raw = list(agent_ids) if agent_ids else [-1] * n_a
        self.agent_ids = [-1 if i is None or i < 0 else -(int(i) + 2) for i in _raw]
        assert len(self.agent_ids) == n_a, "agent_ids must match agent_decks"
        # the parent needs every deck the agent can hold (decklist features are per-episode)
        super().__init__(decks=[list(d) for d in self.agent_decks], **kw)
        self.agent_deck = list(self.agent_decks[0])
        self.opp_decks = [list(d) for d in opp_decks]
        self.opp_weights = [float(w) for w in opp_weights]
        self.agent_seat = 0

    def _sample_decks(self):
        self.agent_seat = int(self.rng.random() < 0.5)
        # index-sampled for the matchup log; choices(range(n), w) == choices(seq, w) rng-stream-wise
        oi = self.rng.choices(range(len(self.opp_decks)), weights=self.opp_weights, k=1)[0]
        opp = self.opp_decks[oi]
        if len(self.agent_decks) > 1:          # deck-SET mode: resample the agent deck too
            ai = self.rng.choices(range(len(self.agent_decks)),
                                  weights=self.agent_weights, k=1)[0]
            self.agent_deck = self.agent_decks[ai]
            a_id = self.agent_ids[ai]
        else:
            a_id = self.agent_ids[0]
        # matchup ids: a_id is NEGATIVE (agent marker; -1 unnamed, else -(pool_idx+2)),
        # oi = index into opp_decks
        self._deck_idx = (a_id, oi) if self.agent_seat == 0 else (oi, a_id)
        return (self.agent_deck, opp) if self.agent_seat == 0 else (opp, self.agent_deck)

    def reset(self, seed: int | None = None):
        enc, seat, info = super().reset(seed)
        return enc, seat, {**info, "agent_seat": self.agent_seat}

    def step(self, action: int):
        enc, seat, r, done, info = super().step(action)
        return enc, seat, r, done, {**info, "agent_seat": self.agent_seat}


class FrozenOpponentEnv(TwoSidedSelfPlayEnv):
    def __init__(self, agent_deck, opp_decks, opp_weights, frozen_ckpt, **kw):
        super().__init__(decks=[list(agent_deck)], **kw)
        self.agent_deck = list(agent_deck)
        self.opp_decks = [list(d) for d in opp_decks]
        self.opp_weights = [float(w) for w in opp_weights]
        self.agent_seat = 0

        # frozen opponent net: CPU, single-threaded, built from the ckpt's own net_config
        import torch
        self._t = torch
        torch.set_num_threads(1)
        from .card_features import get_card_table
        from .policy import build_token_net
        ck = torch.load(frozen_ckpt, map_location="cpu")
        self._fnet = build_token_net(get_card_table(), ck["net_config"])
        self._fnet.load_state_dict(ck["net"])
        self._fnet.eval()
        self._int_keys = set(self.encoder.int_keys)

    # -- internals -----------------------------------------------------------
    def _sample_decks(self):
        """Ship deck on a random seat; opponent deck sampled by meta weight."""
        self.agent_seat = int(self.rng.random() < 0.5)
        oi = self.rng.choices(range(len(self.opp_decks)), weights=self.opp_weights, k=1)[0]
        opp = self.opp_decks[oi]
        self._deck_idx = (-1, oi) if self.agent_seat == 0 else (oi, -1)
        return (self.agent_deck, opp) if self.agent_seat == 0 else (opp, self.agent_deck)

    def _opp_pick(self, enc_obs) -> int:
        """Frozen net's greedy action over the (already-encoded) opponent obs."""
        t = self._t
        with t.no_grad():
            o = {k: t.as_tensor(np.asarray(v)[None],
                                dtype=(t.long if k in self._int_keys else t.float32))
                 for k, v in enc_obs.items()}
            logits = self._fnet.logits_value(o)[0][0]
            mask = t.as_tensor(np.asarray(enc_obs["action_mask"]) > 0.5)
            return int(logits.masked_fill(~mask, float("-inf")).argmax().item())

    def _roll_opponent(self, enc, seat):
        """Play frozen-opponent decisions until the agent acts or the episode ends.
        Returns (enc, done, agent_reward, info)."""
        info: dict = {}
        while seat != self.agent_seat:
            enc, seat, _r, done, info = super().step(self._opp_pick(enc))
            if done:
                term = info.get("terminal")
                # truncation: 0 unless the trunc-penalty is on -- then the opponent's -1 flips
                # to our +1 (ladder-truthful: their stall is our timeout win)
                r = float(term[self.agent_seat]) if term else \
                    (-float(_r) if info.get("truncated") else 0.0)
                return enc, True, r, info
        return enc, False, 0.0, info

    # -- gym-ish API (agent's view) ------------------------------------------
    def reset(self, seed: int | None = None):
        for _ in range(32):
            enc, seat, info = super().reset(seed)
            seed = None
            enc, done, _r, _info = self._roll_opponent(enc, seat)
            if not done:                          # normal case: an agent decision is live
                return enc, self.agent_seat, {"seat": self.agent_seat}
        raise RuntimeError("FrozenOpponentEnv: 32 consecutive episodes ended before the agent acted")

    def step(self, action: int):
        enc, seat, r, done, info = super().step(int(action))
        if done:                                  # ended on the agent's own decision
            term = info.get("terminal")
            r = float(term[self.agent_seat]) if term else float(r)
            return enc, self.agent_seat, r, True, info
        enc, done, r, info2 = self._roll_opponent(enc, seat)
        if done:                                  # ended (or truncated) on an opponent decision
            return enc, self.agent_seat, r, True, info2
        return enc, self.agent_seat, 0.0, False, info
