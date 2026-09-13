# rl/ — cabt self-play PPO (the shipped pipeline)

Two-sided self-play PPO over the cabt engine with a token-transformer pointer
policy (~20M params), BC warm-start, teacher-KL distillation with gated
promotion, and a per-ship-deck two-net finetune stage. Observation/action
design: `STATE_ACTION_SPACE.md`.

## Modules (one file, one role)

| file | role |
|---|---|
| `enc_constants.py` | shape/layout constants for encoders + token model — single source of truth (exported into submission bundles) |
| `encoding.py` | obs dict → fixed-shape arrays: `TokenEncoder` (v2 token stream), `GameTracker`/`AbilityTracker` (reveal + ability memory; the train==test path) |
| `_encfast.pyx` | Cython fast path for the two hottest `encode()` helpers; byte-identical to the Python fallbacks |
| `native_enc.py` | driver for the C native encoder (`native_encode/`) — JSON-free binary-obs path used by learner + frozen opponents |
| `wk_native.py` | fast would-KO annotation for the native collection path |
| `card_features.py` | `EN_Card_Data.csv` → static per-card feature table + embedding vocab |
| `attack_data.py` | generated per-attack properties (damage, cost, effect flags) |
| `buff_data.py` | generated turn-scoped buff tables |
| `effect_data.py` | generated frozen effect-category multi-hots |
| `env.py` | engine driver: native library loading + `_BinEngine` (one battle per process) + `load_deck` |
| `env_selfplay.py` | two-sided all-seats self-play env (production collection) + `FrozenOpponentEnv` (deck-finetune view over it) |
| `vec_env_selfplay.py` | subprocess vector env: one battle per worker process, shared-memory obs, frozen wide/KS opponent streams |
| `policy.py` | v2 token-transformer actor-critic: pointer net over per-decision options, split heads, varlen/NJT attention |
| `train_selfplay.py` | THE production trainer: two-sided PPO + GAE, teacher-KL + gated promotion, two-net finetune (`--two-net-finetune`, balance modes incl. `turns`), asym truncation terminals, DDP; also hosts `resolve_deck_pool` + `lr_at` |
| `rollout.py` | rollout collection + GAE for the two-sided trainer (CUDA-graphs collect, teacher prepass cache) |
| `gates.py` | promotion-gate schedules + the lockstep-batched field gate (`field_winrate_fast`) + DDP cross-gate — the eval workhorse |
| `value_diag.py` | value-head diagnostics: distance-to-terminal decomposition |
| `decks.py` | competition starter decks |
| `decks_train.py` | consolidated named deck sets: mined Kaggle archetypes, human top-cut lists (Aichi/NAIC), live-ladder metas → `TRAIN_TOP15` / `TRAIN_TOP50` |
| `decks_ladder.py` | ladder-mined pool loader (the all-decks training distribution; json + weight sidecars) |
| `field_mix.py` | the canonical field-mix opponent distribution used by field-relative instruments |
| `search_agent.py` | decision-time search: greedy select, would-KO annotate, PUCT + PIMC-determinized MCTS (`mcts2` export backend) |

## Invariants guarded by tests (`../tests/`)

pad/attention-mask isolation, tracker JSON↔native differential fuzz, v2.1/v2.2
encoding behavior, varlen == padded encoder equality, state truncation, int32-safe
chunked update, matchup-log rng-stream stability, gate schedules, self-play and
finetune env regressions, value-diag bucketing.
