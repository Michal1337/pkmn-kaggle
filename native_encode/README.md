# Native encode (the shipped JSON-free collection path)

STATUS (final): shipped and byte-identical-validated against the Python encoder
(`scripts/enc_parity_check2.py`; the m3_* validation harnesses that proved it during
development were retired at competition end — see git history). The dual encode paths
(this C encoder and `rl/encoding.py`) MUST stay mirrored if either is ever changed.

Goal: move the per-decision **collection** work off Python. Profiled on the training CPU (H100 node,
OMP=1), collection per decision breaks down as:

| stage | µs | share |
|-------|----|-------|
| engine `Select` (C++ game step) | 14 | **1.5%** |
| JSON serialize (`GetBattleData`) | 166 | 18% |
| JSON parse (`json.loads`) | 60 | 6% |
| `GameTracker.update` (Python) | 34 | 4% |
| **`encode()` (Python)** | 666 | **71%** |

The engine is a *myth* bottleneck (1.5%). The cost is JSON round-trip (**24%**) + Python `encode()`
(**71%**). Native-path ceiling ≈ **4–7× collection** (~2× SPS, Amdahl-bounded by the GPU update).

## Architecture (B): binary-obs export + Python tracker + native encode

- **Engine**: a `GetBinaryObs` export serializes the redacted per-player obs (same fields as
  `ApiJson`/`ToJson`, `playerIndex = state.selectPlayer`) into a flat int/float buffer instead of a
  JSON string. See `BinaryObs.h` (our code; `#include "BinaryObs.h"` added to the engine's Export.cpp).
- **Python**: `GameTracker` stays (cross-turn, log-based, only ~4%).
- **Native encode** (planned, Cython): reads the binary buffer + the tracker outputs → the token
  arrays, kept **byte-identical** to `rl/encoding.py::TokenEncoder.encode`.

## Hard constraints

- **train == test**: Kaggle inference runs the OFFICIAL engine + JSON + the Python `encode()`. The
  native path is **training-collection only** and must stay byte-identical to the Python encode
  forever (two-implementation maintenance). A self-built engine is fine for training (same source ⇒
  same rules; RNG/games need not match). Can't ship a modified engine to Kaggle.
- **Wall-clock only** — raises throughput, not the leaderboard ceiling.

## Status

- **Feasibility**: engine compiles on cluster gcc-13/C++20 (`g++ -std=c++20 -shared Export.cpp`).
- **M1 DONE**: custom `.so` drives full games via `sdk_cg`, correct obs schema.
- **M2 DONE**: `GetBinaryObs` serializes the full redacted per-player obs (sections A–F: scalars,
  counts/status, card lists, units, select+options, `looking`) into an int buffer — **5.2 µs** vs
  **205.9 µs** JSON round-trip (40× on the export step).
- **M3 DONE (correctness)**: Cython `encode_native` over the binary buffer is byte-identical to the
  Python encode across **all keys** — narrow (single deck, ~2.9k obs), hardened (**57 decks / 40k obs /
  0 mismatches**, cap-overflow scanner clean), AND the **would_ko trio** (see below).
- **would_ko fix (adversarial-audit catch)**: `opt_attr[11,12,13]` = would_ko / would_ko_prizes/6 /
  would_ko_win are a SEPARATE Python engine-sim (`search_agent.annotate_would_ko`) written onto the
  option dicts before `encode` reads them — NOT in the binary buffer. The native path takes them as a
  `wk` [MAX_OPTIONS,3] **float64** arg (float32 loses the /6-divide precision) and writes cols 11–13.
  The original validator never called `annotate_would_ko`, so both sides read 0 and the gap hid; the
  validator now annotates and verifies (421/2448 obs, 1466 nonzero cells). With `--would-ko` default
  True, a native path missing this would train/serve on a wrong array.
- **M3 DONE (speedup, typed)**: parse into fixed C stack arrays + typed memoryviews (no Python dicts),
  fx multi-hot as a vectorized numpy gather (the 128×NFX inner loop cost ~80 µs on a Python-global
  lookup) → encode-only **~65 µs** (`encode_state` dict path) vs Python **~605 µs** = **~9×**;
  end-to-end collection-encode **100 µs** vs **841 µs** = **8.41×** (H100 training node, OMP=1, 4,000 obs).
- **M3 DONE (batch-write fast path)**: `class BatchEncoder(B, NFX)` preallocates the `[B,…]` batch
  arrays + their memoryviews ONCE; `.encode(…, row)` writes obs `row` straight in (no per-obs alloc, no
  `np.stack`). Byte-identical to `encode_state`; **19.6 µs/obs vs 49.1 µs = 2.5×** faster than the dict
  path. This is the training-collection entry point.

- **v2.1 encoding change (2026-07-02) — REBUILD + REVALIDATE REQUIRED.** Three mirrored changes:
  (1) the `self_deck` stream keeps the FULL decklist and gains a NEW per-copy DRAWABLE flag output
  `self_deck_flag` (1 iff the copy is not visible in our zones = deck-or-prized; C-side visible-zone
  count in `_fill_views`, Python side `encoding.decklist_drawable_flags`); (2) the `opp_deck` stream
  gains the analogous HIDDEN flag `opp_deck_flag` — the glue/validators now pass
  `tracker.copies_hidden_for(opp)` ({cid: (n, n_hidden)}) where they passed `copies_for`, expanded
  in-C hidden-first per cid, UNK fill = 1; (3) `opt_attr` gained an already-picked flag col 14
  (`OPT_STRUCT` 14→15, fx cols shift +1; `encode_state`/`BatchEncoder.encode` now take the picked
  INDEX LIST where they took `picked_count`). The compiled `.so` on the cluster is STALE until
  rebuilt; re-prove parity (`scripts/enc_parity_check2.py`; historically `m3_full_validate.py` +
  `m3_hard_validate.py`, retired — see git history) before any `--native-encode` run. Old
  checkpoints/BC datasets are incompatible (opt_attr width 35→36 + two new obs keys + two new net
  params) — retrain from scratch.

## Build (training node, a training node)

    # engine source expected at ~/engine_src (the competition C++ source + this BinaryObs.h)
    cp native_encode/BinaryObs.h ~/engine_src/
    cd ~/engine_src
    grep -q BinaryObs.h Export.cpp || sed -i '/#include "All.h"/a #include "BinaryObs.h"' Export.cpp
    g++ -std=c++20 -fPIC -O2 -shared Export.cpp -o libcg_custom.so   # -O3/LTO if deployed

## M3 DONE — correct + fast (typed)

`encode_native.pyx` reads the binary buffer and builds **every** token key — CLS scalars, all card
streams (deck/prize/hand/discard/stadium, self+opp, arg/tracker/UNK sourced), unit tokens, and the
full option loop (opt_attr/verb/attack_id + pointer src/tgt resolution incl. section-F `looking`) —
**byte-identical** to `TokenEncoder.encode` over **~2,940 obs/run, 0 mismatches, all keys**. Build:
`PYTHONPATH=<cython-dir> python setup_native.py build_ext --inplace` (same model as setup_encfast).

The buffer is parsed into **fixed-size C stack arrays** (card lists `[HCAP]`, per-pokemon fields
`pk_*[NPK][…]`, option params `opt_*[OCAP]`) and written straight into the output numpy arrays via
typed memoryviews. **No intermediate Python dicts/lists, no per-option `np.zeros`, no `dict.get()`.**
Option src/tgt pointer resolution is inlined on typed ints (a `NIL` sentinel replaces the "absent
JSON field" `None`). Caps are safe upper bounds for a 60-card game; reads always advance the cursor
fully and clamp only the *store*, so an over-cap list can never desync the parse.

### Measured per-decision cost (hopper training CPU, OMP=1, 4,000 obs)

The training fast path per obs = `GetBinaryObs` (**4.6 µs**) + `prep` (**~5 µs**, tracker `copies_for`
+ `hand_ids_for`) + **`BatchEncoder.encode` (~12 µs)** ≈ **~22 µs**, vs the Python path ~840 µs
(JSON serialize+parse ~195 + `encode` ~645) = **~38×** collection-encode.

**Speed levers (both requested, both measured):**
1. **Batch-write (`BatchEncoder`) — DONE, 3.4× on the encode.** Preallocate the `[B,…]` batch +
   memoryviews once; write obs `row` straight in. `encode_state` (fresh dict) ~42 µs →
   `BatchEncoder.encode` **~12 µs**. Erases the ~21 µs/obs allocation + per-call memoryview
   acquisition (and the cross-env `np.stack` copy at integration). This is THE training entry point.
1b. **fx-fill skip (DONE, ~9 µs):** the fx multi-hot was applied via a numpy fancy-index gather
   `fx[opt_attack_id]` — a per-obs temp-array alloc + two copies (~9 µs, invisible until batch-write
   removed the other allocs). `fx[0]` (no-attack multihot) is all-zeros, so after the memset only the
   FEW attack rows need writing — a typed loop over `oaid>0` rows (OS hoisted to a C local). Dropped
   `BatchEncoder` 21 → **12 µs**. (Lesson: numpy per-call overhead on small arrays is ~µs each — in a
   ~12 µs typed encode, every `np.*` call in the hot path is a tax; keep it pure-C.)
2. **prep opp-deck expand in Cython — DONE, but NET NEUTRAL.** The `copies_for` `{cid:cnt}` dict is
   now expanded to the opp-deck stream inside the encode (dict API) instead of a Python list-comp in
   `prep`. Measured wash: it *relocates* ~2 µs (prep leaner, encode ~2 µs heavier), total unchanged —
   because the expand still iterates Python objects under the GIL, and prep is otherwise dominated by
   the tracker methods `copies_for`/`hand_ids_for` (~4 µs) which can't move without rewriting the
   tracker (which the Python encode shares anyway). Kept for cleaner integration glue, not for speed.
3. parse+fills are already typed C — diminishing returns beyond here.

SPS gain is Amdahl-bounded by the GPU (PPO) update — measure the live collection:update ratio to
translate. Training-wall-clock only (Kaggle inference still runs the official engine + JSON + the
Python encode); the two encodes are kept byte-identical, so **every `encoding.py` change must be
mirrored here and re-proven** (parity harness: `scripts/enc_parity_check2.py`).

## Stage-1 integration (`CabtEnv --native-encode`) — DONE, measured

Wired into the training env (`rl/env.py`, behind `native_encode=True` / `train.py --native-encode`):
in native mode `CabtEnv` runs the **sdk_cg** engine (`libcg_custom.so`) — `battle_start`/`select` still
produce the JSON obs for the game logic (tracker / mask / would_ko / opponent), and `_encode` calls
`GetBinaryObs` + a B=1 `BatchEncoder` instead of the Python encode (+ `action_mask` built in Python).
**Byte-identical in-env** (proven over a rollout incl. multi-select buffering with non-empty `picked`
+ would_ko, via `native_validate=True`; env-level harness retired, see git history).

**Measured (H100 training node, OMP=1, random opponent):** learner encode **~95 µs vs ~728 µs Python = 7.7×** in
the env context (deeper games than the isolated bench). But the per-**step** win is JSON-floor-bound:

| regime | native step | implied python-encode step | collection |
|--------|-----------:|---------------------------:|-----------:|
| `--no-would-ko` | ~708 µs | ~1341 µs | **~1.9×** |
| would_ko (default) | ~4016 µs | ~4649 µs | ~1.16× |

The floor is **the JSON round-trip is paid per `battle_select` — agent AND every opponent turn — so
several per step** (bigger than a single encode). would_ko adds a per-decision engine-sim that
dominates when on. Caveats: measured with a **random** opponent; real self-play adds the *opponent's*
Python encode (native only speeds the learner), shrinking the win further. Net: Stage 1 is a real but
modest ~1.1–1.9× depending on config; **the JSON round-trip is the real remaining lever (Stage 2:
add a logs section to the binary obs + port GameTracker to read it, dropping JSON entirely → ~14×).**
