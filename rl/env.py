"""Engine driver: loads the native cabt library (sdk_cg) and exposes _BinEngine — the
one-battle-per-process binary engine handle that rl.env_selfplay builds on — plus load_deck.
(The legacy single-agent CabtEnv wrapper was retired 2026-09-13; see rl/env_selfplay.py.)"""

from __future__ import annotations

import ctypes
import logging
import os
import random

import numpy as np

logging.disable(logging.CRITICAL)  # silence kaggle_environments import chatter

from .encoding import TokenEncoder, SUBMIT_ACTION, build_mask
from .encoding import GameTracker, AbilityTracker, binary_to_obs
from .card_features import get_card_table


_ENGINE_LIBS: dict = {}


def _load_engine_lib(path):
    """Load an ALTERNATIVE engine build (PKMN_ENGINE_LIB override) with the same ctypes
    surface sdk_cg.sim declares on the default lib. Cached per path; GameInitialize once.
    Additive: only reached when the env var is set -- the default path is untouched.
    The Structures are defined INLINE (not imported from sdk_cg.sim): importing sim loads
    AND GameInitialize()s the default lib as a side effect, and the engine's header-static
    tables are STB_GNU_UNIQUE (process-global across dlopen'd builds) -- two loaded engines
    share one CardTable, so the alt lib's init then double-fills it (assert/corruption)."""
    L = _ENGINE_LIBS.get(path)
    if L is None:
        class StartData(ctypes.Structure):
            _fields_ = [("battlePtr", ctypes.c_void_p),
                        ("errorPlayer", ctypes.c_int),
                        ("errorType", ctypes.c_int)]

        class SerialData(ctypes.Structure):
            _fields_ = [("json", ctypes.c_char_p),
                        ("data", ctypes.POINTER(ctypes.c_ubyte)),
                        ("count", ctypes.c_int),
                        ("selectPlayer", ctypes.c_int)]
        L = ctypes.cdll.LoadLibrary(path)
        L.GameInitialize()
        L.BattleStart.restype = StartData
        L.BattleStart.argtypes = [ctypes.POINTER(ctypes.c_int)]
        L.BattleFinish.argtypes = [ctypes.c_void_p]
        L.Select.restype = ctypes.c_int
        L.Select.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        L.GetBattleData.restype = SerialData
        L.GetBattleData.argtypes = [ctypes.c_void_p]
        L.AgentStart.restype = ctypes.c_void_p
        _ENGINE_LIBS[path] = L
    return L


def _native_handles():
    """Per-INSTANCE native-engine handles: the shared ctypes lib (signatures declared once,
    idempotent) + a fresh battle-pointer holder. The engine is multi-battle per process
    (ApiBattleStart = new ApiData(); zero mutable statics -- scripts/test_multibattle.py); the
    old 'one battle per process' constraint was sdk_cg.sim's `Battle` CLASS-attribute singleton,
    which K in-process envs would trample. Shared by CabtEnv and TwoSidedSelfPlayEnv.

    PKMN_ENGINE_LIB (additive override): path to an alternative engine .so (e.g. the wko build
    libcg_wk.so, an export superset). Unset -> the default sdk_cg lib, byte-identical behavior."""
    import types
    lib_path = os.environ.get("PKMN_ENGINE_LIB")
    if lib_path:
        L = _load_engine_lib(lib_path)
    else:
        from sdk_cg.sim import lib as L
    L.GetBinaryObs.restype = ctypes.c_int
    L.GetBinaryObs.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    L.AdvanceLog.restype = ctypes.c_int
    L.AdvanceLog.argtypes = [ctypes.c_void_p]
    return types.SimpleNamespace(battle_ptr=None, obs=None), L


class _BinEngine:
    """Stage-2 JSON-free engine shim. Same battle_start/select/finish interface as sdk_cg.game, but
    steps the engine WITHOUT GetBattleData (no JSON serialize) and returns the light binary_to_obs
    game-logic dict; the raw GetBinaryObs buffer is stashed on ``last_arr`` for the tracker/encode
    (which read it directly). ``advance_log`` consumes the current player's log delta (once/decision)."""

    def __init__(self, lib, Battle, cbuf, cbuf_np, cap):
        self.lib = lib; self.Battle = Battle; self.cbuf = cbuf; self.cbuf_np = cbuf_np
        self.cap = cap; self.last_arr = None

    def _getbin(self):
        n = self.lib.GetBinaryObs(self.Battle.battle_ptr, self.cbuf, self.cap)
        if n > self.cap:
            # WriteBinaryObs stops WRITING at cap but returns the full count -- parsing the
            # truncated buffer would silently corrupt the obs AND the tracker. Raise instead:
            # the vec_env worker recovery turns this into a clean reset episode.
            raise RuntimeError(f"GetBinaryObs overflow: obs needs {n} ints > cap {self.cap}")
        a = np.array(self.cbuf_np[:n], dtype=np.int32)          # copy: stable across the decision's buffering
        self.last_arr = a
        return a

    def battle_start(self, deck0, deck1):
        cards = list(deck0) + list(deck1)
        arg = (ctypes.c_int * len(cards))(*cards)
        sd = self.lib.BattleStart(arg)
        self.Battle.battle_ptr = sd.battlePtr
        if not self.Battle.battle_ptr:
            return None, sd
        return binary_to_obs(self._getbin()), sd

    def battle_select(self, select_list):
        parg = (ctypes.c_int * len(select_list))(*select_list)
        err = self.lib.Select(self.Battle.battle_ptr, parg, len(select_list))
        if err != 0:
            raise ValueError("battle_ptr broken.") if err == 30 else IndexError()
        return binary_to_obs(self._getbin())

    def battle_finish(self):
        self.lib.BattleFinish(self.Battle.battle_ptr)

    def advance_log(self):
        self.lib.AdvanceLog(self.Battle.battle_ptr)

_AGENT_DECK = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "agent", "deck.csv")


def load_deck(path: str = _AGENT_DECK) -> list[int]:
    with open(path) as f:
        return [int(line) for line in f if line.strip()]
