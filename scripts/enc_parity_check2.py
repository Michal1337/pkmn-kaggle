"""FULL-STACK native-vs-python observation parity on live game states. v2 -- the valid one.

v1 compared the native encode against TokenEncoder fed the env's obs dict -- but in native mode
that dict is binary_to_obs()'s LIGHT game-logic stub, so v1 measured the stub, not the encoder.

This version uses the wko-validator trick: the JSON and the binary obs are pulled from the SAME
battle pointer, so they describe the identical position with no cross-engine lockstep. Per
decision:

  native path (production, untouched): binary obs -> update_native() -> NativeEncoder
  shadow path (submission):  lib.GetBattleData(same ptr) -> full JSON -> shadow GameTracker
                             .update(json) -> TokenEncoder

and the two encoded dicts are compared per key. A mismatch means the TRAINING obs and the
SUBMISSION obs disagree about the same position -- the silent way a shipped agent underperforms
its internal numbers. This covers BOTH open audit axes at once: encoder parity (binary vs JSON
construction) and tracker parity (update_native vs update).

Log-cursor note: the JSON is fetched AFTER battle_select's GetBinaryObs (arr already embeds the
pending log delta) and BEFORE the native tracker consumes it -- both trackers therefore see the
same delta regardless of whether GetBattleData and AdvanceLog share a cursor.

Ability slots are shared (both paths derive them from the same select+indices stream), which
removes a non-informative axis from the comparison.

  PYTHONPATH=. PKMN_TURN_CAP=200 python scripts/enc_parity_check2.py [N_GAMES] [SEED]
"""
import collections
import ctypes
import json
import random
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
N_GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 40
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 20260729

from rl.card_features import get_card_table
from rl.encoding import TokenEncoder
from rl.env_selfplay import TwoSidedSelfPlayEnv
from rl.encoding import GameTracker

G = "data"
pool = json.load(open(f"{G}/ladder_decks_v12.json"))["decks"]
rng = random.Random(SEED)
enc = TokenEncoder(get_card_table())

decisions = [0]
exact_bad = collections.Counter()      # key -> mismatching decisions (exact compare)
close_bad = collections.Counter()      # key -> mismatches surviving allclose(1e-6)
examples = []


class AuditEnv(TwoSidedSelfPlayEnv):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self._shadow_tr = [GameTracker(), GameTracker()]
        self._shadow_last = [None, None]
        self._sjson = None
        self._sjson_for = None
        self._gbd_ready = False

    def _fetch_json(self):
        lib = self._game.lib
        if not self._gbd_ready:
            from sdk_cg.sim import SerialData
            lib.GetBattleData.restype = SerialData
            lib.GetBattleData.argtypes = [ctypes.c_void_p]
            self._gbd_ready = True
        sd = lib.GetBattleData(self._Battle.battle_ptr)
        return json.loads(sd.json.decode())

    def _encode(self):
        # full JSON for THIS decision, fetched once per new obs (multi-pick re-encodes reuse it)
        if self._obs is not self._sjson_for:
            self._sjson = self._fetch_json()
            self._sjson_for = self._obs
        nat = super()._encode()                       # production native path, untouched
        seat = self._seat()
        jo = self._sjson
        if self._obs is not self._shadow_last[seat]:  # same once-per-decision gate as production
            self._shadow_tr[seat].update(jo)
            self._shadow_last[seat] = self._obs
        py = self.encoder.encode(jo, set(self._picked), self_deck=self.deck[seat],
                                 tracker=self._shadow_tr[seat],
                                 ability_slots=self.abilities[seat].slots)
        decisions[0] += 1
        for k in nat:
            a, b = np.asarray(nat[k]), np.asarray(py[k])
            if a.shape != b.shape:
                exact_bad[k] += 1; close_bad[k] += 1
                if len(examples) < 6:
                    examples.append(f"dec {decisions[0]} seat {seat} '{k}': SHAPE {a.shape} vs {b.shape}")
                continue
            if not np.array_equal(a, b):
                exact_bad[k] += 1
                if not np.allclose(a.astype(np.float64), b.astype(np.float64), atol=1e-6):
                    close_bad[k] += 1
                    if len(examples) < 6:
                        idx = np.argwhere(a != b)
                        i0 = tuple(idx[0])
                        examples.append(f"dec {decisions[0]} seat {seat} '{k}': {len(idx)} cells, "
                                        f"first at {list(i0)}: nat={a[i0]} py={b[i0]}")
        return nat


games = 0
for g in range(N_GAMES):
    da, db = rng.choice(pool), rng.choice(pool)
    env = AuditEnv(decks=[[int(x) for x in da], [int(x) for x in db]],
                   encoder=enc, seed=SEED + g, native_encode=True)
    try:
        obs, seat, _ = env.reset()
        done, steps = False, 0
        while not done and steps < 4000:
            legal = np.flatnonzero(np.asarray(obs["action_mask"]) > 0)
            a = int(rng.choice(list(legal))) if len(legal) else 0
            obs, seat, _r, done, _info = env.step(a)
            steps += 1
        games += 1
    except Exception as e:
        print(f"[audit] game {g}: {type(e).__name__}: {e}", flush=True)
    finally:
        env.close()

print(f"\nPARITY2 games={games}/{N_GAMES} decisions={decisions[0]:,}")
if exact_bad:
    print("keys with EXACT mismatches (count of decisions):")
    for k, c in exact_bad.most_common():
        tag = "  [REAL: fails allclose too]" if close_bad[k] else "  [float-epsilon only]"
        print(f"  {k:>16}: {c:6,} / {decisions[0]:,}{tag}")
    for e in examples:
        print("  " + e)
else:
    print("ALL KEYS BYTE-IDENTICAL across every decision")
sys.exit(1 if close_bad else 0)
