"""FAST would-ko for the native (JSON-free) collection path.

Mirrors ``rl.search_agent.would_ko_flags`` decision-for-decision without any JSON:
  * the serialized redacted state comes from the engine's ``GetSerialObs`` export
    (byte-identical to the JSON obs's ``search_begin_input``),
  * the per-attack 1-ply sim loop (clone -> step option -> first-legal advance ->
    read KO/prizes/win) runs entirely in C++ (``WouldKO`` export, libcg_wk.so),
  * the observation-conditioned PIMC determinization is the SAME ``_determinize``
    code as the JSON path, fed by ``_bin_to_wk_obs`` -- a GetBinaryObs-buffer parse
    that reconstructs exactly the obs fields ``_determinize`` consumes (validated
    byte-identical against the JSON obs in scripts/wko_native_validate.py).

FEATURE-GATED: everything here is inert unless PKMN_WK_NATIVE=1 AND the loaded engine
lib exposes the ``WouldKO`` symbol (i.e. PKMN_ENGINE_LIB points at libcg_wk.so). With
the gate off, every existing code path is untouched (native+would_ko still raises).
"""

from __future__ import annotations

import ctypes
import os
import random


def wk_native_enabled() -> bool:
    """The feature gate: opt-in via env var (the lib symbol is checked at WKNative init)."""
    return os.environ.get("PKMN_WK_NATIVE") == "1"


# ---------------------------------------------------------------------------
# GetBinaryObs buffer -> the minimal obs dict rl.search_agent._determinize reads.
# Layout mirror of native_encode/BinaryObs.h (sections A/B/C/D + E's selectType);
# the same walk as GameTracker.update_native_py / binary_to_obs_py in rl/encoding.py.
# ---------------------------------------------------------------------------
def _bin_to_wk_obs(arr):
    """Returns (obs_dict, select_type_wire). obs_dict carries exactly what _determinize
    consumes: yourIndex; per player active/bench (unit dicts with id + attachment id lists,
    None for a face-down active), discard/hand as bare id lists, prize as [None]*count,
    deckCount/handCount; stadium as [{'id','playerIndex'}]. select_type_wire == the JSON
    obs's select['type'] (engine enum-1 wire offset; 0 == MAIN)."""
    a = arr.tolist() if hasattr(arr, "tolist") else list(arr)
    me = a[2]
    deck_c = [a[9 + p * 12 + 1] for p in (0, 1)]
    hand_c = [a[9 + p * 12 + 2] for p in (0, 1)]
    prize_c = [a[9 + p * 12 + 3] for p in (0, 1)]
    i = 33                                                # A(9) + B(2*12)

    def _rd(i):                                           # cardlist -> [(id, serial, playerIndex)]
        cnt = a[i]; i += 1
        out = []
        for _ in range(cnt if cnt > 0 else 0):
            out.append((a[i], a[i + 1], a[i + 2])); i += 3
        return out, i

    hand_me, i = _rd(i)                                   # C: self hand
    d0, i = _rd(i)                                        #    discard 0
    d1, i = _rd(i)                                        #    discard 1
    stad, i = _rd(i)                                      #    stadium
    units = [[[], []], [[], []]]                          # D: [player][active|bench]
    for pl in range(2):
        for grp in range(2):
            c = a[i]; i += 1
            for _ in range(c):
                present = a[i]; i += 1
                if present == 0:                          # face-down (opp setup active) -> None
                    units[pl][grp].append(None)
                    continue
                pid_c = a[i]; i += 6                      # id, serial, playerIndex, hp, maxHp, appear
                ne = a[i]; i += 1; i += ne                # energies histogram
                ec, i = _rd(i)                            # energyCards, tools, preEvolution
                tl, i = _rd(i)
                pe, i = _rd(i)
                units[pl][grp].append({
                    "id": pid_c,
                    "preEvolution": [c0 for c0, _, _ in pe],
                    "tools": [c0 for c0, _, _ in tl],
                    "energyCards": [c0 for c0, _, _ in ec],
                })
    sel_type = a[i]                                       # E: selectType (wire = enum-1; 0 == MAIN)

    players = []
    for p in range(2):
        pl = {
            "active": units[p][0],
            "bench": units[p][1],
            "discard": [c0 for c0, _, _ in (d0 if p == 0 else d1)],
            "prize": [None] * prize_c[p],
            "deckCount": deck_c[p],
            "handCount": hand_c[p],
        }
        if p == me:
            pl["hand"] = [c0 for c0, _, _ in hand_me]
        players.append(pl)
    stadium = [{"id": c0, "playerIndex": pi} for c0, _, pi in stad]
    return {"current": {"yourIndex": me, "players": players, "stadium": stadium}}, sel_type


# ---------------------------------------------------------------------------
# ctypes plumbing for the wk exports (declared once per lib; agent ptr shared per lib,
# mirroring sdk_cg.api's process-global agent_ptr).
# ---------------------------------------------------------------------------
_AGENTS: dict = {}


def _declare_wk(lib) -> None:
    if getattr(lib, "_wk_declared", False):
        return
    lib.GetSerialObs.restype = ctypes.c_int
    lib.GetSerialObs.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    lib.WouldKO.restype = ctypes.c_int
    lib.WouldKO.argtypes = ([ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
                            + [ctypes.POINTER(ctypes.c_int)] * 6
                            + [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)])
    lib.AgentStart.restype = ctypes.c_void_p
    lib._wk_declared = True


def _agent_for(lib):
    key = id(lib)
    a = _AGENTS.get(key)
    if a is None:
        a = ctypes.c_void_p(lib.AgentStart())
        _AGENTS[key] = a
    return a


def _ci(lst):
    return (ctypes.c_int * max(1, len(lst)))(*lst)


def fill_wk(wk_arr, flags) -> None:
    """Write a flags dict {opt_idx: (ko_rate, exp_prizes, win_rate)} into the NativeEncoder's
    (MAX_OPTIONS, 3) wk matrix (raw prizes 0..6 -- encode_native applies the /6 clamp, exactly
    like the JSON option_row does). Zeroes the matrix first (one matrix per decision)."""
    wk_arr[:] = 0.0
    n = wk_arr.shape[0]
    for i, (ko, prizes, win) in flags.items():
        if 0 <= i < n:
            wk_arr[i, 0] = ko
            wk_arr[i, 1] = prizes
            wk_arr[i, 2] = win


class WKNative:
    """Per-env driver for the fast would-ko path. ``flags`` is a faithful mirror of
    rl.search_agent.would_ko_flags (same determinization code + rng-consumption order,
    same per-sim skip-on-error semantics, same aggregation)."""

    SBUF_CAP = 1 << 16

    def __init__(self, lib, encoder):
        if not hasattr(lib, "WouldKO"):
            raise ValueError(
                "PKMN_WK_NATIVE=1 but the loaded engine lib has no WouldKO export -- "
                "set PKMN_ENGINE_LIB to the libcg_wk.so build (or unset PKMN_WK_NATIVE)."
            )
        _declare_wk(lib)
        self.lib = lib
        self.enc = encoder
        self.agent = _agent_for(lib)
        self._sbuf = ctypes.create_string_buffer(self.SBUF_CAP)
        self._out = (ctypes.c_int * 4)()

    def flags(self, battle_ptr, arr, sel, deck, n_var=None, rng=None, early_stop=False) -> dict:
        """{attack_option_index: (ko_rate, exp_prizes_taken, win_rate)} for the current decision.
        ``arr`` = the GetBinaryObs buffer; ``sel`` = the env's binary_to_obs select dict.
        ``early_stop`` mirrors would_ko_flags': stop a variable attack's sampling once 3 sims
        agree unanimously (default OFF, like the envs' annotate calls)."""
        from .search_agent import _determinize, _WK_ATTACKS, WK_NDET_VAR
        if n_var is None:
            n_var = WK_NDET_VAR
        opts = (sel or {}).get("option") or []
        atk = [i for i, o in enumerate(opts) if o.get("attackId") is not None]
        if not atk:
            return {}
        wk_obs, sel_type = _bin_to_wk_obs(arr)
        if sel_type != 0:                                  # MAIN selects only (mirror reference)
            return {}
        n = self.lib.GetSerialObs(battle_ptr, self._sbuf, self.SBUF_CAP)
        if n <= 0 or n > self.SBUF_CAP:
            return {}
        rng = rng or random.Random()
        out = {}
        for a in atk:
            av = _WK_ATTACKS.get(opts[a].get("attackId"))
            ndet = max(1, n_var) if (av and av[1]) else 1  # variable -> sample prob; fixed -> 1 exact
            kos = wins = trials = 0
            prize_sum = 0.0
            seen = set()                                   # early-stop unanimity (mirror reference)
            for _ in range(ndet):
                # fresh determinization per sim (identical rng-consumption order to the reference)
                f = _determinize(wk_obs, deck, rng, self.enc)
                r = self.lib.WouldKO(
                    self.agent, self._sbuf, n,
                    _ci(f["your_deck"]), _ci(f["your_prize"]),
                    _ci(f["opponent_deck"]), _ci(f["opponent_prize"]),
                    _ci(f["opponent_hand"]), _ci(f["opponent_active"]),
                    a, 1, self._out,
                )
                if r == 0 and self._out[3] > 0:            # a completed sim (nsims=1 -> per-sim ints)
                    kos += self._out[0]
                    wins += self._out[1]
                    prize_sum += self._out[2] / 100.0
                    trials += self._out[3]
                    if early_stop:
                        seen.add((self._out[0], self._out[2] // 100, self._out[1]))
                # (r != 0 / 0 trials mirrors the reference's skip-on-error: contributes nothing)
                if early_stop and ndet > 1 and trials >= 3 and len(seen) == 1:
                    break
            if trials:
                out[a] = (kos / trials, prize_sum / trials, wins / trials)
        return out
