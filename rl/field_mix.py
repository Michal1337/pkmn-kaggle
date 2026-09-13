"""THE canonical field mix -- the single opponent-deck distribution every field-relative
instrument must use (in-run field gates, scripts/field_eval_broad_fast.py, meta_gate weighted
summaries, finetune gates), so all "field wr" numbers land on ONE scale.

Rules:
  * Numbers are comparable ONLY within the same VERSION. Log the version with every score.
  * Refresh DELIBERATELY (new version constant from a fresh rating-banded scout), never edit in
    place: a running curve keeps its launch-time mix (frozen yardstick), offline re-scoring can
    re-anchor old ckpts to the new version via field_eval_broad_fast.
  * Weights = band-blend 0.2*climb(<1150) + 0.3*contested(1150-1250) + 0.5*summit(1250+) from
    episodes/scout_elo.py -- the mixture a top submission's rating is actually decided against.

History:
  V1 (implicit, gen5m in-run gate): pooled allN counts -- archaludon-heavy. gen5m's curve stays V1.
  V2 2026-07-07 (scout of 06-28..07-02, bands 2026-07-05): grimmsnarl-court-weighted; first used
     by ft_aichi2. == FIELD_MIX_V2 below.
  V3 2026-07-09 (META SHIFT: ladder scores compressed, 1250+ band EMPTY; grimmsnarl/archaludon
     collapsed to ~1% of the new 1150-1250 top band). Weights = OPPONENT-EXPOSURE of our live
     alakazam sub (loss_mine 07-06..08, 1,197 games -- the field the crown actually faces), on
     the CURRENT exact lists (META3, incl. Yushin Ito's fez+grimmsnarl that farms us at 0.45).
"""

FIELD_MIX_VERSION = "V3_2026_07_09"

FIELD_MIX = {
    "meta3_cynthia_garchomp": 0.26,
    "meta3_fez_grimmsnarl_yushin": 0.15,
    "meta3_kanga_multi_ragingbolt": 0.11,
    "meta3_fez": 0.10,
    "meta3_kangaskhan": 0.09,
    "meta2_grimmsnarl_ex": 0.09,
    "k02_alakazam_non_ex": 0.09,
    "meta3_fez_latias_kanga_meowth": 0.06,
    "meta2_archaludon_ex": 0.03,
    "k07_mega_starmie_ex": 0.02,
}

FIELD_MIX_V2 = {   # frozen record (curves launched under V2 keep this yardstick)
    "meta2_grimmsnarl_ex": 0.23,
    "meta2_archaludon_ex": 0.21,
    "meta2_fez_grimmsnarl_ex": 0.10,
    "k02_alakazam_non_ex": 0.09,
    "k05_fezandipiti_ex": 0.08,
    "k07_mega_starmie_ex": 0.08,
    "k01_mega_lucario_ex": 0.07,
    "meta2_cornerstone_ogerpon_ex": 0.05,
    "meta2_chandelure_non_ex": 0.04,
    "meta2_cynthia_garchomp_ex": 0.02,
    "k12_team_rocket_s_mewtwo_ex": 0.02,
    "k15_barbaracle_non_ex": 0.01,
}


def as_arg() -> str:
    """--field-decks / --opponent-decks / meta_gate_agg weight-string form."""
    return ",".join(f"{k}={v}" for k, v in FIELD_MIX.items())


if __name__ == "__main__":
    print(FIELD_MIX_VERSION)
    print(as_arg())
