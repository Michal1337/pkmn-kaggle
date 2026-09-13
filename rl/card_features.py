"""Static per-card feature table built from EN_Card_Data.csv.

The cabt engine identifies every card by an integer ``id`` only; the observation
never carries names, HP, attacks, etc. This module turns the competition's card
list into a fixed-width numeric feature vector per card id, so the policy can
reason about *what a card is* instead of memorising opaque ids.

Two ways to consume identity downstream:
  * ``FEATURES[id]``  -> dense static features (HP, type, stage, cost, ...). Fully
    numeric, generalises across cards, needs no embedding layer.
  * ``id`` itself     -> index into an ``nn.Embedding(vocab_size, d)`` if you want
    learned identity on top of the static features.

Everything is derived once and cached. Card id 0 is reserved as PAD (all zeros).
"""

from __future__ import annotations

import csv
import os
import re
from functools import lru_cache

import numpy as np

# Energy type order matches the engine/visualizer string 'CGRWLPFDM A'
# (verified: a Pokemon's energies=[3] corresponds to {W} Water).
ENERGY_ORDER = "CGRWLPFDM A"  # index 9 (space) is unused but kept for alignment
N_ENERGY = len(ENERGY_ORDER)  # 11
_ENERGY_IDX = {c: i for i, c in enumerate(ENERGY_ORDER)}
_ENERGY_IDX["竜"] = _ENERGY_IDX["A"]  # dragon kanji -> 'A'

# High-level card category.
CATEGORIES = ["pokemon", "trainer", "energy"]
# Fine-grained kind from the "Stage/Type" column.
STAGES = [
    "Basic Pokémon", "Stage 1 Pokémon", "Stage 2 Pokémon",
    "Item", "Pokémon Tool", "Supporter", "Stadium",
    "Basic Energy", "Special Energy",
]
RULES = ["n/a", "ACE SPEC", "Pokémon ex", "Mega Pokémon ex"]
SPECIAL_TAGS = ["Ancient", "Future", "Tera", "Trainer's"]  # substring match in Category

# Normalisation constants (rough maxima from the data; only affect scale).
HP_MAX = 400.0
DMG_MAX = 400.0
RETREAT_MAX = 4.0
COST_MAX = 5.0
ATTACKS_MAX = 3.0

DEFAULT_CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "EN_Card_Data.csv")


def _onehot(value, vocab) -> list[float]:
    v = [0.0] * len(vocab)
    if value in vocab:
        v[vocab.index(value)] = 1.0
    return v


def _energy_onehot(type_field: str) -> list[float]:
    """Type column like '{W}' or '竜' -> 11-dim one-hot (multi marks summed)."""
    v = [0.0] * N_ENERGY
    for sym in re.findall(r"\{(.*?)\}", type_field or ""):
        if sym in _ENERGY_IDX:
            v[_ENERGY_IDX[sym]] += 1.0
    if (type_field or "").strip() == "竜":
        v[_ENERGY_IDX["A"]] = 1.0
    return v


def _parse_cost(cost: str) -> list[float]:
    """Attack cost string like '{D}●●' -> 11-dim energy histogram.

    '●' is a colorless requirement (-> 'C', index 0); '{X}' is a typed one.
    """
    v = [0.0] * N_ENERGY
    if not cost or cost == "n/a":
        return v
    for sym in re.findall(r"\{(.*?)\}", cost):
        if sym in _ENERGY_IDX:
            v[_ENERGY_IDX[sym]] += 1.0
    v[_ENERGY_IDX["C"]] += cost.count("●")
    return v


def _parse_int(s: str) -> int:
    m = re.search(r"\d+", s or "")
    return int(m.group()) if m else 0


def _build(csv_path: str):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    # Group rows by card id (Pokemon may have several attack rows).
    by_id: dict[int, list[dict]] = {}
    for r in rows:
        cid = _parse_int(r["Card ID"])
        by_id.setdefault(cid, []).append(r)

    max_id = max(by_id) if by_id else 0
    vocab_size = max_id + 1

    # Build one feature row per card id.
    feats: dict[int, list[float]] = {}
    names: dict[int, str] = {}
    for cid, group in by_id.items():
        head = group[0]
        stage = (head["Stage (Pokémon)/Type (Energy and Trainer)"] or "").strip()
        rule = (head["Rule"] or "n/a").strip()
        category = (head["Category"] or "").strip()
        names[cid] = head["Card Name"]

        if "Pokémon" in stage:
            cat = "pokemon"
        elif "Energy" in stage:
            cat = "energy"
        else:
            cat = "trainer"

        # Aggregate attacks across this card's rows.
        damages = [_parse_int(g["Damage"]) for g in group if (g["Move Name"] or "").strip() not in ("", "n/a")]
        costs = [_parse_cost(g["Cost"]) for g in group if (g["Cost"] or "n/a") != "n/a"]
        n_attacks = len(damages)
        max_dmg = max(damages) if damages else 0
        cost_sum = np.sum(costs, axis=0) if costs else np.zeros(N_ENERGY)
        max_cost_total = max((sum(c) for c in costs), default=0.0)

        row: list[float] = []
        row += _onehot(cat, CATEGORIES)                          # 3
        row += _onehot(stage, STAGES)                            # 9
        row += _energy_onehot(head["Type"])                      # 11  (pokemon/energy type)
        row += _energy_onehot(head["Weakness"])                  # 11
        row += _onehot(rule, RULES)                              # 4
        row += [1.0 if any(t in category for t in [tag]) else 0.0 for tag in SPECIAL_TAGS]  # 4
        row.append(_parse_int(head["HP"]) / HP_MAX)              # 1
        row.append(_parse_int(head["Retreat"]) / RETREAT_MAX)    # 1
        row.append(1.0 if (head["Resistance (Type)"] or "").strip() not in ("", "n/a") else 0.0)  # 1
        row.append(n_attacks / ATTACKS_MAX)                      # 1
        row.append(max_dmg / DMG_MAX)                            # 1
        row.append(max_cost_total / COST_MAX)                    # 1
        row += (cost_sum / COST_MAX).tolist()                    # 11  (summed attack cost histogram)
        feats[cid] = row

    feat_dim = len(next(iter(feats.values())))
    matrix = np.zeros((vocab_size, feat_dim), dtype=np.float32)
    for cid, row in feats.items():
        matrix[cid] = row
    return matrix, names, vocab_size, feat_dim


class CardTable:
    """Lazily-built, cached lookup of static card features."""

    def __init__(self, csv_path: str = DEFAULT_CSV):
        self.csv_path = csv_path
        self.matrix, self.names, self.vocab_size, self.feat_dim = _build(csv_path)

    def features(self, card_id) -> np.ndarray:
        """Static feature vector for a card id (PAD/unknown -> zeros)."""
        if card_id is None or card_id < 0 or card_id >= self.vocab_size:
            return np.zeros(self.feat_dim, dtype=np.float32)
        return self.matrix[card_id]

    def name(self, card_id) -> str:
        return self.names.get(card_id, f"#{card_id}")


@lru_cache(maxsize=4)
def get_card_table(csv_path: str = DEFAULT_CSV) -> CardTable:
    return CardTable(csv_path)


if __name__ == "__main__":
    t = get_card_table()
    print(f"vocab_size={t.vocab_size}  feat_dim={t.feat_dim}")
    for cid in (3, 723, 1262):
        print(f"  {cid:4d} {t.name(cid):24s} feat[:6]={t.features(cid)[:6]}")
