"""Transient turn-scoped buff tables (GENERATED from sdk_cg.api.all_attack() + EN_Card_Data.csv).

DEFENSE_BUFF_ATTACKS: attackId -> damage reduction granted to the ATTACKER's own Pokemon
during the opponent's next turn (200 == "prevent all damage"). Detected from ATTACK logs
(carry attackId + serial) -> set on that unit's encoding unit_attr[23], active the turn after.

OFFENSE_BUFF_CARDS: Trainer/Item/Stadium cardId -> "this turn" extra attack damage. Detected
from our own PLAY logs (cardId) -> a CLS scalar. Pokemon ability/attack-based buffs are
EXCLUDED (abilities emit no log; playing a Pokemon to bench != using its buff).
"""

DEFENSE_BUFF_ATTACKS = {
    75: 200,
    78: 30,
    100: 50,
    186: 30,
    233: 200,
    244: 200,
    261: 200,
    349: 200,
    411: 60,
    416: 30,
    505: 200,
    570: 50,
    584: 10,
    595: 200,
    684: 200,
    780: 30,
    788: 200,
    790: 200,
    849: 40,
    860: 200,
    896: 10,
    897: 20,
    943: 30,
    986: 200,
    1047: 30,
    1054: 200,
    1064: 200,
    1204: 60,
    1205: 200,
    1212: 200,
    1266: 200,
    1279: 20,
    1309: 200,
    1327: 200,
    1382: 200,
    1426: 50,
    1470: 200,
    1525: 30,
}

OFFENSE_BUFF_CARDS = {
    1141: 30,   # Premium Power Pro
    1191: 30,   # Kieran
    1211: 40,   # Black Belt’s Training
}
