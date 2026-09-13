"""Reusable native-encode driver, shared by the JSON-free env (learner) and the self-play opponent
(so NEITHER side pays the Python encode). Holds the once-built attack-feature / fx / offset / Tera
matrices + a B=1 BatchEncoder; ``encode`` reads a GetBinaryObs buffer + a tracker -> the token dict
(byte-identical to TokenEncoder.encode -- see native_encode/ validators). Requires the built
``encode_native`` module + ``sdk_cg`` with ``libcg_custom.so`` (GetBinaryObs). Training-only.
"""
from __future__ import annotations

import numpy as np

from .encoding import _ATTACKS, _OFF, MAX_ATTACK, MAX_OPTIONS, build_mask


class NativeEncoder:
    def __init__(self, encoder):
        from . import effect_data
        from encode_native import BatchEncoder
        ct = encoder.cards
        self.unk = int(encoder.UNK)
        V = ct.vocab_size
        tera = np.zeros(V + 1, np.float32)
        for c in range(V + 1):
            f = ct.features(c)
            if f.shape[0] > 40 and f[40] > 0.5:
                tera[c] = 1.0
        self.tera = tera
        af = np.zeros((MAX_ATTACK, 4), np.float32)
        for aid, (dmg, var, cost, eff) in _ATTACKS.items():
            if 0 <= aid < MAX_ATTACK:
                af[aid] = [min(dmg, 350) / 350.0, float(var), min(cost, 5) / 5.0, float(eff)]
        self.af = af
        nfx = effect_data.N_ATTACK_FX
        fx = np.zeros((MAX_ATTACK, nfx), np.float32)
        for aid in range(MAX_ATTACK):
            fx[aid] = effect_data.attack_multihot(aid)
        self.fx = fx
        self.offs = np.array([_OFF["self_units"], _OFF["opp_units"], _OFF["self_hand"], _OFF["opp_hand"],
                              _OFF["self_discard"], _OFF["opp_discard"], _OFF["stadium"], _OFF["effect"]], np.int64)
        self.benc = BatchEncoder(1, nfx)
        self._ei = np.zeros(0, np.int64)
        self._ef = np.zeros(0, np.float32)
        self.wk = np.zeros((MAX_OPTIONS, 3), np.float64)         # would_ko trio per option (stays 0 unless
        #                                                          the PKMN_WK_NATIVE fast path fills it --
        #                                                          rl/wk_native.fill_wk, once per decision)
        # M2-light (2026-07-10): the [0]-row views are STABLE (BatchEncoder's arrays never
        # realloc) -> build the output dict ONCE instead of ~40 dict inserts per decision.
        # Contract unchanged: encode() returns views the caller must consume/copy before the
        # next encode on this env (each env owns its NativeEncoder; the vec worker copies to
        # shm immediately).
        self._out = {k: v[0] for k, v in self.benc.batch.items()}

    def encode(self, arr, sel, self_deck, tracker, ability_slots, picked):
        """arr = int32 GetBinaryObs buffer; sel = binary_to_obs['select']; picked = chosen indices
        (this decision's buffered multi-select set -> the opt_attr already-picked flags, v2.1).
        Returns the token dict (views into the B=1 batch -- consume/copy before the next encode)."""
        n = arr.shape[0]
        opp = 1 - int(arr[2])                                    # yourIndex = buf[2]
        odk = tracker.copies_hidden_for(opp)                     # {cid: (n_copies, n_hidden)} -> opp_deck_flag
        beliefs = tracker.hand_beliefs_for(opp)                  # [(cid, certain)] -> opp_hand_id + _flag (v2.2)
        ohd = [c for c, _ in beliefs]
        ohf = [f for _, f in beliefs]
        dbuf = tracker.opp_def_buff or {}
        if dbuf:
            dser = np.fromiter(dbuf.keys(), np.int64, len(dbuf))
            dred = np.fromiter(dbuf.values(), np.float32, len(dbuf))
        else:
            dser, dred = self._ei, self._ef
        self.benc.encode(arr, n, self.unk, self_deck, tracker.top_ids, odk, ohd, ohf, float(tracker.offense_buff),
                         picked, self.tera, ability_slots, dser, dred,
                         self.offs, self.af, self.fx, self.wk, 0)
        out = self._out
        out["action_mask"] = build_mask(sel, set(picked))
        return out
