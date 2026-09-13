"""Differential fuzz: GameTracker.update(json_obs) vs update_native(binary_buf) must be
byte-identical on EQUIVALENT inputs (incl. dict insertion order, which the belief sort's
stable tie-break depends on). Synthesizes random game-ish states + log deltas, builds both
representations per BinaryObs.h / ApiJson.h, and diffs full tracker state.

Run:  python -m pytest tests/test_tracker_fuzz.py -q     (400 seeded games)
      python tests/test_tracker_fuzz.py [N]              (standalone, default 2000)
"""
import random
import sys

import numpy as np

import rl.encoding as E
from rl.encoding import GameTracker

AREAS = [1, 2, 3, 4, 5, 6, 7, 12]


class _buff_tables:
    """Temporarily inject synthetic buff entries so those tracker branches are exercised;
    restore afterwards (other tests in the same process need the real tables)."""

    def __enter__(self):
        self._d = dict(E.DEFENSE_BUFF_ATTACKS)
        self._o = dict(E.OFFENSE_BUFF_CARDS)
        E.DEFENSE_BUFF_ATTACKS[901] = 30.0
        E.DEFENSE_BUFF_ATTACKS[902] = 60.0
        E.OFFENSE_BUFF_CARDS[801] = 20.0
        E.OFFENSE_BUFF_CARDS[802] = 30.0

    def __exit__(self, *exc):
        E.DEFENSE_BUFF_ATTACKS.clear(); E.DEFENSE_BUFF_ATTACKS.update(self._d)
        E.OFFENSE_BUFF_CARDS.clear(); E.OFFENSE_BUFF_CARDS.update(self._o)


def make_state(rng, serial_pool):
    """Random visible state: per-player discard/active/bench(+attached)/handCount + stadium."""
    st = {"players": [], "stadium": []}
    for p in range(2):
        pl = {
            "discard": [],
            "active": [],
            "bench": [],
            "handCount": rng.randint(0, 8),
        }
        for _ in range(rng.randint(0, 4)):
            pl["discard"].append({"id": rng.randint(1, 60), "serial": rng.choice(serial_pool[p])})
        def mk_pk():
            if rng.random() < 0.15:
                return None                       # facedown / empty
            pk = {"id": rng.randint(1, 60), "serial": rng.choice(serial_pool[p]),
                  "hp": 100, "maxHp": 120, "appearThisTurn": 0,
                  "energies": [rng.randint(0, 11) for _ in range(rng.randint(0, 3))],
                  "preEvolution": [], "tools": [], "energyCards": []}
            for key, kmax in (("preEvolution", 2), ("tools", 2), ("energyCards", 3)):
                for _ in range(rng.randint(0, kmax)):
                    pk[key].append({"id": rng.randint(1, 60), "serial": rng.choice(serial_pool[p])})
            return pk
        if rng.random() < 0.9:
            pk = mk_pk()
            pl["active"] = [pk] if pk is not None else [None]
        for _ in range(rng.randint(0, 3)):
            pl["bench"].append(mk_pk())
        st["players"].append(pl)
    if rng.random() < 0.4:
        st["stadium"] = [{"id": rng.randint(1, 60), "serial": rng.choice(serial_pool[rng.randint(0, 1)]),
                          "playerIndex": rng.randint(0, 1)}]
    return st


def make_logs(rng, serial_pool):
    logs = []
    for _ in range(rng.randint(0, 6)):
        t = rng.choice([6, 6, 6, 7, 10, 10, 11, 12, 13, 14, 15, 15, 8, 4, 2])
        pid = rng.randint(0, 1)
        ser = rng.choice(serial_pool[pid])
        cid = rng.choice([rng.randint(1, 60), 801, 802])
        lg = {"type": t, "playerIndex": pid}
        if t == 6:
            lg.update(cardId=cid, serial=ser,
                      fromArea=rng.choice(AREAS), toArea=rng.choice([2, 2, 2, 1, 3, 6]))
        elif t == 7:
            lg.update(fromArea=rng.choice([2, 2, 1, 3]), toArea=rng.choice([1, 3]))
        elif t == 10:
            lg.update(cardId=cid, serial=ser)
        elif t in (11, 12, 13, 14):
            lg.update(cardId=cid, serial=ser)
        elif t == 15:
            lg.update(cardId=cid, serial=ser, attackId=rng.choice([901, 902, 555]))
        logs.append(lg)
    return logs


def to_binary(me, turn, st, logs):
    """Build the BinaryObs.h int32 buffer for (state, log delta) as seen by `me`."""
    buf = []
    put = buf.append
    # A
    put(turn); put(0); put(me); put(0); put(0); put(0); put(0); put(0); put(-1)
    # B
    for p in range(2):
        pl = st["players"][p]
        put(5)                       # benchCapacity
        put(30)                      # deck.size
        put(pl["handCount"])         # hand.size
        put(6)                       # prize.size
        put(len(pl["discard"]))      # trash.size
        put(len(pl["active"]))       # active.size
        put(len(pl["bench"]))        # bench.size
        for _ in range(5):
            put(0)                   # status
    def put_card(c, owner):
        put(c["id"]); put(c["serial"]); put(c.get("playerIndex", owner))
    def put_list(lst, owner):
        put(len(lst))
        for c in lst:
            put_card(c, owner)
    # C: self hand (tracker ignores; emit empty), discard0, discard1, stadium
    put(0)
    put_list(st["players"][0]["discard"], 0)
    put_list(st["players"][1]["discard"], 1)
    put_list(st["stadium"], 0)
    # D: units
    def put_pk(pk, owner):
        if pk is None:
            put(0); return
        put(1)
        put(pk["id"]); put(pk["serial"]); put(owner)
        put(pk["hp"]); put(pk["maxHp"]); put(pk["appearThisTurn"])
        put(len(pk["energies"]))
        for e in pk["energies"]:
            put(e)
        put_list(pk["energyCards"], owner)   # BinaryObs order: energyCards, tools, preEvolutions
        put_list(pk["tools"], owner)
        put_list(pk["preEvolution"], owner)
    for p in range(2):
        act = st["players"][p]["active"]
        put(len(act))
        for pk in act:
            put_pk(pk, p)
        bn = st["players"][p]["bench"]
        put(len(bn))
        for pk in bn:
            put_pk(pk, p)
    # E: select scalars + options + deck + contextCard + effect
    for _ in range(6):
        put(0)
    put(0)          # nopt
    put(-1)         # deck null
    put(0)          # contextCard null
    put(0)          # effect null
    # F: looking null
    put(-1)
    # G: logs (7 ints each, -1 for absent)
    put(len(logs))
    for lg in logs:
        put(lg["type"]); put(lg.get("playerIndex", -1))
        put(lg.get("cardId", -1)); put(lg.get("serial", -1))
        put(lg.get("fromArea", -1)); put(lg.get("toArea", -1))
        put(lg.get("attackId", -1))
    return np.asarray(buf, dtype=np.int32)


def to_json(me, turn, st, logs):
    players = []
    for p in range(2):
        pl = st["players"][p]
        players.append({
            "discard": list(pl["discard"]),
            "active": list(pl["active"]),
            "bench": list(pl["bench"]),
            "handCount": pl["handCount"],
            "hand": [] if p != me else [],
        })
    return {"current": {"yourIndex": me, "turn": turn, "players": players,
                        "stadium": list(st["stadium"])},
            "logs": list(logs)}


def snap(tr):
    return {
        "serials": {p: {c: sorted(s) for c, s in tr.serials[p].items()} for p in (0, 1)},
        "serials_order": {p: list(tr.serials[p].keys()) for p in (0, 1)},
        "zone": {p: list(tr.zone[p].items()) for p in (0, 1)},          # order-sensitive
        "card_of": {p: list(tr.card_of[p].items()) for p in (0, 1)},
        "zone_turn": {p: list(tr.zone_turn[p].items()) for p in (0, 1)},
        "hand_epoch": {p: list(tr.hand_epoch[p].items()) for p in (0, 1)},
        "blind_exits": dict(tr.blind_exits),
        "def_buff": dict(tr.opp_def_buff),
        "off_buff": tr.offense_buff,
        "hb0": tr.hand_beliefs_for(0), "hb1": tr.hand_beliefs_for(1),
        "ch0": list(tr.copies_hidden_for(0).items()), "ch1": list(tr.copies_hidden_for(1).items()),
    }


def run(seed):
    rng = random.Random(seed)
    serial_pool = [list(range(1, 61)), list(range(61, 121))]   # global-unique serials per player
    tj, tn = GameTracker(), GameTracker()
    turn = 1
    me = rng.randint(0, 1)
    for step in range(rng.randint(3, 25)):
        if rng.random() < 0.4:
            turn += 1
        if rng.random() < 0.3:
            me = 1 - me
        st = make_state(rng, serial_pool)
        logs = make_logs(rng, serial_pool)
        tj.update(to_json(me, turn, st, logs))
        tn.update_native(to_binary(me, turn, st, logs))
        a, b = snap(tj), snap(tn)
        if a != b:
            for k in a:
                if a[k] != b[k]:
                    print(f"seed={seed} step={step} key={k}\n  json  ={a[k]}\n  native={b[k]}")
            return False
    return True


def test_tracker_json_native_parity():
    with _buff_tables():
        for seed in range(400):
            assert run(seed), f"tracker state diverged (seed {seed}; details printed above)"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    bad = 0
    with _buff_tables():
        for seed in range(n):
            if not run(seed):
                bad += 1
                if bad >= 5:
                    break
    print(f"done: {n} games fuzzed, {bad} divergent")


if __name__ == "__main__":
    main()
