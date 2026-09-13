# State & action space (cabt RL)

How `rl/encoding.py` + `rl/env.py` turn the cabt engine into an RL problem.
Grounded in measured engine behaviour (40+ self-play games) and the 1267-card
`EN_Card_Data.csv`. Dimensions are constants at the top of `encoding.py`.

> **v2 (post-audit).** Active Pokemon is now encoded as its own vector with bench
> pooled separately (was: active+bench pooled together, hiding which is active);
> deck-search option cards are resolved via `sel["deck"]` (were blind); `MAX_OPTIONS`
> 64→96, `MAX_HAND` 15→20, per-Pokemon counts clipped. **This changes obs/net
> dims — old checkpoints are incompatible; retrain from scratch.**
>
> **v2.1 (2026-07-02, information-gap fixes).** (1) The `self_deck` stream keeps the
> FULL decklist and gains a per-copy DRAWABLE flag (`self_deck_flag`): 1 iff that
> copy is not visible in our public zones = still in deck or face-down in our
> prizes — see `encoding.decklist_drawable_flags`; the net adds
> `flag * drawable_emb` to the deck tokens. (2) The `opp_deck` stream gains the
> analogous per-copy HIDDEN flag (`opp_deck_flag`, own `opp_drawable_emb` param):
> 1 iff that revealed copy is not currently visible in a public zone (tracker
> belief — `GameTracker.copies_hidden_for`; hidden copies emitted first within a
> cid) or the slot is unrevealed UNK — i.e. plausibly still to come; 0 = spent
> (their discard/board/stadium). Flags exist ONLY on the two deck streams. (3)
> `opt_attr` gained an already-picked flag (col `OPT_PICKED`=14, `OPT_STRUCT`
> 14→15): multi-select picks stay visible-but-illegal so later picks/submit see
> the set built so far (net presence = legal OR picked). Mirrored in
> `encode_native.pyx` (revalidate on rebuild). **Old checkpoints + prebuilt BC
> datasets are incompatible; retrain.** (Parts of this doc below still describe the
> pre-token v1 layout — the token layout's source of truth is `enc_constants.py` +
> `encoding.py`.)

## The core difficulty

The engine asks a *different question* each step: a `select` with a
variable-length list of legal `option`s, and you return indices. Option index
*i* has no fixed meaning, and the list length varies (1–57 observed). So:

- **Identity, not position.** Every option is scored from its own feature row;
  the policy is a per-option scorer, not a fixed-index head.
- **Masking.** Illegal/padded slots are masked out every step.
- **Multi-select via buffering.** Most decisions pick 1, but some pick up to
  `maxCount` (2–3 seen). The env buffers single picks and submits the set when
  complete, so every RL action stays one masked `Discrete` choice.

## Action space

`Discrete(N_ACTIONS)` where `N_ACTIONS = MAX_OPTIONS + 1 = 97`.

| action | meaning |
|---|---|
| `0 .. MAX_OPTIONS-1` (0–95) | pick option *i* from `select["option"]` |
| `MAX_OPTIONS` (96) = `SUBMIT_ACTION` | submit the buffered set (end an optional/multi selection early) |

Masking rule (`build_mask`):
- option *i* legal ⟺ it exists, fits `MAX_OPTIONS`, and isn't already buffered, and `len(picked) < maxCount`;
- submit legal ⟺ `minCount ≤ len(picked) < maxCount`;
- once `len(picked) == maxCount` the env auto-submits (no agent step wasted);
- the mask is never all-zero.

Get the current mask with `env.action_masks()` (CleanRL/MaskablePPO convention).

> `MAX_OPTIONS = 96` covers the observed max of ~69. If a selection ever exceeds
> 96 options the extra slots are dropped (and unreachable) — bump the constant if
> you see it. The action is always a valid engine option index by construction.

## Observation space

`obs` is a `dict` of fixed-shape numpy arrays (float32, except `*_id` int64).
Encoded from the **acting player's** perspective (`yourIndex` = "self").
Card identity is dual-channel: a **static feature array** (from the CSV) plus a
raw **id** for an `nn.Embedding` hook.

| key | shape | contents |
|---|---|---|
| `scalars` | (14,) | turn, turnActionCount, first-player flag, supporter/stadium/energy/retreat-used flags, `remainDamageCounter`, `remainEnergyCost`, min/max count, buffered-pick count, #options, stadium-present |
| `select_type` | (1,) int | SelectType (embed) |
| `select_context` | (1,) int | SelectContext (embed) |
| `self_player`, `opp_player` | (10,) | handCount, deckCount, #prize, #discard, benchMax, 5 status flags (poison/burn/sleep/paralyze/confuse) |
| `self_active_dyn`, `opp_active_dyn` | (1, 18) | the ACTIVE Pokemon (its own vector, not pooled): present, hp ratio, maxHp, 11-d energy histogram (clipped), #energy, #tools, appearedThisTurn, #preEvolution |
| `self_active_static`, `opp_active_static` | (1, 59) | active Pokemon static card features |
| `self_active_id`, `opp_active_id` | (1,) int | active card id (embed hook) |
| `self_bench_dyn`, `opp_bench_dyn` | (5, 18) | bench Pokemon (pooled), same per-slot features |
| `self_bench_static`, `opp_bench_static` | (5, 59) | bench static card features |
| `self_bench_id`, `opp_bench_id` | (5,) int | bench card ids (embed hook) |
| `hand_static` | (20, 59) | static features of own hand cards (opponent hand is hidden) |
| `hand_id` | (20,) int | own hand card ids |
| `hand_mask` | (20,) | which hand slots are filled |
| `self_discard_agg`, `opp_discard_agg` | (59,) | mean static features over the discard pile (set summary) |
| `stadium_static` | (59,) | static features of the stadium card |
| `stadium_id` | (1,) int | stadium card id |
| `opt_dyn` | (96, 28) | per option: 16-d type one-hot + normalized refs (area, index, playerIndex, inPlayArea/Index, energyIndex, count, number) + has-card/attackId/serial flags + attackId scale |
| `opt_card_static` | (96, 59) | static features of an option's referenced card (incl. deck-search cards resolved via `sel["deck"][index]`; zeros if none) |
| `opt_card_id` | (96,) int | referenced card id per option (embed hook) |
| `action_mask` | (97,) | current legal-action mask |

### Static card features (59-d, per card id)

From `rl/card_features.py`: category one-hot (pokemon/trainer/energy), stage
one-hot (9: Basic/Stage1/Stage2/Item/Tool/Supporter/Stadium/Basic-/Special-
Energy), 11-d type, 11-d weakness, rule one-hot (none/ACE-SPEC/ex/Mega-ex), 4
special-tag flags (Ancient/Future/Tera/Trainer's-Pokemon), HP, retreat,
has-resistance, #attacks, max damage, max single-attack cost, 11-d summed
attack-cost histogram. Card id 0 = PAD (zeros). Energy order is `CGRWLPFDM A`.

## Reward

Terminal: **+1 win / −1 loss / 0 draw** (from `current.result`). Forfeit
(illegal submit) = −1. Optional dense shaping via `reward_shaping(prev, new,
agent_idx)`; `prize_diff_shaping(scale)` rewards net prize cards taken.

## Episode / opponent

- The env drives the engine directly (`battle_start`/`battle_select`) with fixed
  decks, so there is **no deck-selection step** and `select` is never None.
- The **opponent plays inside `step`**; the agent only sees its own turns.
  Default opponent is uniform-random-legal; pass `opponent_fn(raw_obs, rng)` to
  swap in a scripted bot or a frozen policy snapshot for **self-play**.
- `randomize_side=True` alternates which player the agent controls (cancels
  first-player bias).

## Hard constraint: one battle per process

The native engine uses a single global battle pointer. **Do not run two
`CabtEnv`s in one process.** For parallelism use subprocess vector envs
(e.g. SB3 `SubprocVecEnv` / a CleanRL multiprocessing harness); `env.close()`
frees the native battle.
