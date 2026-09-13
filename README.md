# PTCG AI Battle — self-play RL agent (final: 12th / 6,807 teams)

Agent + training pipeline for the Kaggle
[Pokémon TCG AI Battle Challenge](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)
(2026-06 → 2026-08-16). Two agents play full Pokémon TCG on the competition's **cabt**
engine; a submission is a `.tar.gz` with `main.py` (an `agent(obs)` policy) and a fixed
60-card `deck.csv`, rated on a live ladder.

**Final result: rank 12 of 6,807** (score 1186.4). Final portfolio: an Alakazam control
deck and a Dragapult/Dusknoir disruption deck, each piloted by a deck-specialized
finetune of the same 20M-parameter generalist.

## Approach (short version)

1. **Engine-native environment** — the competition ships the C++ engine; we drive it
   directly (one battle per process, `sdk_cg/`), with a C observation encoder
   (`native_encode/`) byte-identical to the Python one. Train == test, always.
2. **Behavior-cloning warm start** — 12M decisions mined from public ladder replays;
   BC gives ~2× step-compression for the RL that follows.
3. **Self-play PPO at scale** — two-sided (both seats learn), token-transformer pointer
   policy over the per-decision option list, teacher-KL distillation with gated
   promotion, asymmetric truncation terminals (anti-stall), ~11B env steps for the
   generalist ("crown") at up to 16.5k steps/s on 4×H200.
4. **Real-deck opponent pool** — ~4.9k exact decklists mined from ladder episodes +
   human tournament top-cuts, popularity-weighted; the meta was re-scouted daily.
5. **Per-deck finetunes** — the ship lever: a two-net finetune (agent net vs a live
   counter-adapting opponent net, balance-gated) specializes the crown for each ship
   deck; final consolidation phase trains the agent side alone vs a frozen adversary.
6. **Instrument discipline** — frozen-reference eval batteries (≥1,600 games/cell,
   replicated seeds), opponent-decomposed ladder probes, and a strict "orderings
   within one instrument only" rule decided every ship choice.

## Layout

```
rl/               training pipeline (see rl/README.md — one line per module)
native_encode/    C encoder (JSON-free collection path; see its README)
scripts/          pipeline scripts: BC, pool mining, eval gates, export, meta scout
sdk_cg/           engine bindings (the engine binaries themselves are proprietary
                  and not included -- get them from the competition page)
tests/            invariant guards (46 tests: encoding parity, masks, envs, gates)
EN_Card_Data.csv  card database (1,267 post-rotation cards)
WRITEUP.md        the competition writeup, with figures
```

## Setup

```bash
pip install torch numpy  # torch per your CUDA; then:
pip install --no-deps "git+https://github.com/Kaggle/kaggle-environments.git"
python setup_encfast.py build_ext --inplace          # optional Cython fast path
# native encoder (optional, big collection speedup): see native_encode/README.md
```

Train (single node): `PYTHONPATH=. python -m rl.train_selfplay --ddp ...` — see
`rl/train_selfplay.py`'s argparse for the full recipe; the key production parameters
are listed in `WRITEUP.md`.

Export a submission: `python scripts/export_rl_submission.py --ckpt <ckpt.pt>
--backend transformer2 --deck <deck.csv> --out sub.tar.gz`.

## Principles that kept winning

1. Train == test, always — one tracker/encoder path, parity-proven.
2. Trust the LB > field instruments > mirrors; no relative instrument can see mutual
   degradation — keep a frozen external reference in every experiment.
3. Measure at power; single points are wobbles until two agree.
4. Every reward change is an incentive change — ask who gets paid, on both sides of
   the zero-sum mirror, before shipping.
5. Adversarial training needs an active balancer — unbalanced self-play destroys the
   loser and then the signal.
6. The meta is adaptive and you are the target — the moat is piloting quality,
   adaptation latency, and what the field has never seen you play.
