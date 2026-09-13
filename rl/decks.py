"""The competition's default starter decks (60 cards each), by name.

Card ids reference EN_Card_Data.csv. Each deck is written as (card_id, count)
pairs for readability and expanded to a flat 60-card list. Run
``python -m rl.decks`` to validate counts/legality against the card DB and
(re)write decks/<name>.csv.

NOTE: these are the OFFICIAL default decks. They are NOT identical to the
engine's bundled sample deck (kaggle_environments .../cabt.py), which is the
same Mega Abomasnow archetype but a different list. ``agent/deck.csv`` currently
holds that bundled sample, not ``mega_abomasnow`` below.
"""

from __future__ import annotations

# (card_id, count) per deck. Comments are card names for human reference.
_DECKS_SPEC = {
    "mega_abomasnow": [
        (721, 2),   # Kyogre
        (722, 4),   # Snover
        (723, 4),   # Mega Abomasnow ex
        (1121, 4),  # Ultra Ball
        (1126, 1),  # Precious Trolley
        (1192, 4),  # Carmine
        (1227, 4),  # Lillie's Determination
        (1262, 3),  # Surfing Beach
        (3, 34),    # Basic {W} Energy
    ],
    "dragapult": [
        (119, 4),   # Dreepy
        (120, 4),   # Drakloak
        (121, 3),   # Dragapult ex
        (140, 1),   # Fezandipiti ex
        (184, 1),   # Latias ex
        (235, 2),   # Budew
        (1071, 1),  # Meowth ex
        (1079, 2),  # Rare Candy
        (1080, 1),  # Unfair Stamp
        (1086, 4),  # Buddy-Buddy Poffin
        (1097, 2),  # Night Stretcher
        (1120, 4),  # Crushing Hammer
        (1121, 4),  # Ultra Ball
        (1152, 3),  # Poke Pad
        (1156, 1),  # Lucky Helmet
        (1182, 3),  # Boss's Orders
        (1198, 4),  # Crispin
        (1210, 2),  # Brock's Scouting
        (1227, 4),  # Lillie's Determination
        (1256, 2),  # Team Rocket's Watchtower
        (2, 4),     # Basic {R} Energy
        (5, 4),     # Basic {P} Energy
    ],
    "iono": [
        (265, 3),   # Iono's Voltorb
        (268, 3),   # Iono's Tadbulb
        (269, 3),   # Iono's Bellibolt ex
        (270, 3),   # Iono's Wattrel
        (271, 3),   # Iono's Kilowattrel
        (1086, 3),  # Buddy-Buddy Poffin
        (1097, 2),  # Night Stretcher
        (1110, 1),  # Max Rod
        (1118, 1),  # Energy Retrieval
        (1121, 3),  # Ultra Ball
        (1152, 2),  # Poke Pad
        (1227, 4),  # Lillie's Determination
        (1233, 4),  # Canari
        (1254, 3),  # Levincia
        (4, 22),    # Basic {L} Energy
    ],
    "mega_lucario": [
        (673, 2),   # Makuhita
        (674, 2),   # Hariyama
        (675, 2),   # Lunatone
        (676, 3),   # Solrock
        (677, 3),   # Riolu
        (678, 4),   # Mega Lucario ex
        (1102, 4),  # Dusk Ball
        (1123, 2),  # Switch
        (1141, 4),  # Premium Power Pro
        (1142, 4),  # Fighting Gong
        (1152, 4),  # Poke Pad
        (1159, 1),  # Hero's Cape
        (1182, 2),  # Boss's Orders
        (1192, 4),  # Carmine
        (1227, 4),  # Lillie's Determination
        (1252, 2),  # Gravity Mountain
        (6, 13),    # Basic {F} Energy
    ],
}


def deck(name: str) -> list[int]:
    """Expand a named deck spec into a flat list of 60 card ids."""
    out: list[int] = []
    for cid, n in _DECKS_SPEC[name]:
        out += [cid] * n
    return out


DECKS: dict[str, list[int]] = {name: deck(name) for name in _DECKS_SPEC}
DECK_NAMES = list(DECKS)


if __name__ == "__main__":
    import csv
    import os

    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rows = list(csv.DictReader(open(os.path.join(ROOT, "EN_Card_Data.csv"), encoding="utf-8")))
    NAME, STAGE = {}, {}
    for r in rows:
        cid = r["Card ID"].strip()
        if cid and cid not in NAME:
            NAME[int(cid)] = r["Card Name"]
            STAGE[int(cid)] = r["Stage (Pokémon)/Type (Energy and Trainer)"].strip()

    os.makedirs(os.path.join(ROOT, "decks"), exist_ok=True)
    ok = True
    for name, spec in _DECKS_SPEC.items():
        cards = deck(name)
        print(f"\n=== {name} ({len(cards)} cards) ===")
        if len(cards) != 60:
            print(f"  !! NOT 60 CARDS ({len(cards)})"); ok = False
        for cid, n in spec:
            cname = NAME.get(cid, "??? UNKNOWN ID")
            stage = STAGE.get(cid, "")
            is_basic_energy = stage == "Basic Energy"
            flag = ""
            if cid not in NAME:
                flag = "  <-- UNKNOWN CARD ID"; ok = False
            elif n > 4 and not is_basic_energy:
                flag = f"  <-- ILLEGAL: {n} copies of non-basic-energy"; ok = False
            print(f"  {n:2d}x #{cid:<5} {cname}{flag}")
        # write csv (one id per line)
        path = os.path.join(ROOT, "decks", f"{name}.csv")
        with open(path, "w") as f:
            f.write("\n".join(str(c) for c in cards) + "\n")
    print(f"\nwrote decks/*.csv  |  all valid: {ok}")
