"""Regression guards for the TWO-SIDED self-play pipeline (rl.env_selfplay / rl.train_selfplay).

A bug here silently corrupts every downstream experiment, so these are deliberately decisive:

  * test_negamax_gae               -- the per-seat sign-flip GAE on hand-checked rollouts.
  * test_collect_done_convention   -- runs the REAL collect_rollout against a scripted fake env that
        spans an episode boundary, and asserts (a) done[step] is THIS step's terminal flag and (b) the
        GAE returns isolate the two episodes. This is the guard for the collection<->GAE done-convention
        mismatch (terminal transitions must NOT bootstrap into the next game).
  * test_two_sided_env_matches_inference_replay[seat] -- drives a real game and proves the env's
        interleaved per-seat tracking (GameTracker reveals/buffs, AbilityTracker, would_ko, shared
        encoder) is byte-identical to a fresh ISOLATED single-seat inference replay -> train == test.

Run:  python -m pytest tests/test_selfplay.py -v     (or: python tests/test_selfplay.py)
"""
import numpy as np
import torch

from rl.encoding import TokenEncoder, GameTracker, AbilityTracker
from rl.card_features import get_card_table
from rl.train_selfplay import compute_gae_2s, collect_rollout, _to_tensors


# --------------------------------------------------------------------------- GAE
def _gae(seats, rews, dones, vals=None, boot_seat=0, boot_val=0.0, g=1.0, lam=1.0):
    T = len(seats)
    seat = torch.tensor(seats).long().view(T, 1)
    rew = torch.tensor(rews).float().view(T, 1)
    done = torch.tensor(dones).float().view(T, 1)
    val = torch.zeros(T, 1) if vals is None else torch.tensor(vals).float().view(T, 1)
    _, ret = compute_gae_2s(rew, val, done, seat,
                            torch.tensor([boot_val]).float(), torch.tensor([boot_seat]).long(), g, lam)
    return [round(x, 4) for x in ret.view(-1).tolist()]


def test_negamax_gae():
    # done[t] == "transition t ended the episode". value=0 -> returns are Monte-Carlo, per acting seat.
    assert _gae([0, 1, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]) == [-1, 1, -1, 1]          # seat1 wins
    assert _gae([0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1]) == [1, 1, -1, 1]            # multi-turn, seat0 wins
    assert _gae([0, 1, 0, 1, 0], [0, 1, 0, 0, 1], [0, 1, 0, 0, 1]) == [-1, 1, 1, -1, 1]  # two episodes
    assert _gae([0, 1, 0, 1], [0, 0, 0, 1.5], [0, 0, 0, 1]) == [-1.5, 1.5, -1.5, 1.5]    # F19 margin propagates


def _asym_gae(seats, rews, dones, g=1.0, lam=1.0):
    """PKMN_TRUNC_ASYM v2 (two-sided truncation terminals): same call as production --
    trunc mask = done & rew < -1.5 passed into compute_gae_2s. Returns (adv, ret)."""
    T = len(seats)
    seat = torch.tensor(seats).long().view(T, 1)
    rew = torch.tensor(rews).float().view(T, 1)
    done = torch.tensor(dones).float().view(T, 1)
    val = torch.zeros(T, 1)
    tmask = (done > 0) & (rew < -1.5)
    adv, ret = compute_gae_2s(rew, val, done, seat,
                              torch.tensor([0.0]), torch.tensor([0]).long(), g, lam,
                              trunc=tmask)
    return [round(x, 4) for x in adv.view(-1).tolist()], [round(x, 4) for x in ret.view(-1).tolist()]


def test_trunc_asym():
    # SYMMETRIC -10 (the 2026-08-01 production cliff): the negamax mirror hands the truncation
    # VICTIM (seat 0) a +10-scale return -- the reward-hack the two-sided terminal removes.
    assert _gae([0, 1, 0, 1], [0, 0, 0, -10], [0, 0, 0, 1]) == [10, -10, 10, -10]

    # TWO-SIDED: looper (seat 1) keeps the full -10 on its truncating decision; the victim's
    # last decision (t=2) is an explicit +1 WIN; earlier rows propagate at ordinary game scale
    # off those anchors (a normal game from the victim's perspective).
    adv, ret = _asym_gae([0, 1, 0, 1], [0, 0, 0, -10], [0, 0, 0, 1])
    assert adv == [1, -1, 1, -10] and ret == [1, -1, 1, -10]

    # a TURN is a run of same-seat decisions: the looper's whole final run inherits the
    # -10-scale credit (whole-line discouragement); the victim anchor sits at the seat
    # boundary BELOW the run, not merely at the row before the cap.
    adv, ret = _asym_gae([0, 0, 1, 1, 1], [0, 0, 0, 0, -10], [0, 0, 0, 0, 1])
    assert adv == [1, 1, -10, -10, -10] and ret == [1, 1, -10, -10, -10]

    # ordinary terminals are untouched (mask misses -1 and +1) -> byte-identical to _gae
    adv, ret = _asym_gae([0, 1, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1])
    assert adv == [-1, 1, -1, 1] and ret == [-1, 1, -1, 1]
    adv, ret = _asym_gae([0, 1, 0, 1], [0, 0, 0, -1], [0, 0, 0, 1])
    assert adv == [1, -1, 1, -1] and ret == [1, -1, 1, -1]

    # multi-episode buffer: the two-sided terminal stays confined to its own episode
    # (ep1 = seat1 win at t=1; ep2 = seat0 truncates at t=4, victim seat1 anchored +1 at t=3)
    adv, ret = _asym_gae([0, 1, 0, 1, 0], [0, 1, 0, 0, -10], [0, 1, 0, 0, 1])
    assert adv == [-1, 1, -1, 1, -10] and ret == [-1, 1, -1, 1, -10]


# ---------------------------------------------------------- collection done convention
class _FakeNet:
    """value==0 everywhere -> returns are pure Monte-Carlo (so the GAE output is hand-checkable)."""
    value_categorical = False

    def get_action_and_value(self, obs, opt_len=None, state_keep=None):
        n = obs["action_mask"].shape[0]
        return (torch.zeros(n, dtype=torch.long), torch.zeros(n), torch.zeros(n), torch.zeros(n))

    def get_value(self, obs, opt_len=None, state_keep=None):
        return torch.zeros(obs["action_mask"].shape[0])


class _FakeVec:
    """Scripts (acting_seat, reward, done) per step for ONE env. reset()/step() return zero-obs; the
    step AFTER a done is already the next episode's first decision (mirrors the real auto-reset)."""
    def __init__(self, shapes, int_keys, script):
        self.shapes, self.int_keys, self.script, self.t = shapes, int_keys, script, 0

    def _obs(self):
        return {k: np.zeros((1, *sh), dtype=(np.int64 if k in self.int_keys else np.float32))
                for k, sh in self.shapes.items()}

    def reset(self):
        self.t = 0
        return self._obs(), np.array([self.script[0][0]], dtype=np.int64), [{}]

    def step(self, _actions):
        seat, rew, done = self.script[self.t]
        self.t += 1
        nxt = self.script[self.t][0] if self.t < len(self.script) else seat
        info = {"terminal": {0: rew, 1: -rew}} if done else {}
        return (self._obs(), np.array([nxt], dtype=np.int64),
                np.array([rew], dtype=np.float32), np.array([done], dtype=np.bool_), [info])


def test_collect_done_convention():
    enc = TokenEncoder(get_card_table())
    shapes = enc.shapes
    # ep1: seats[0,1], seat1 wins@step1 ; ep2 (auto-reset): seats[0,1,0], seat0 wins@step4.
    script = [(0, 0.0, False), (1, 1.0, True), (0, 0.0, False), (1, 0.0, False), (0, 1.0, True)]
    T = len(script)
    buf = {"obs": {k: torch.zeros((T, 1, *sh), dtype=(torch.long if k in enc.int_keys else torch.float32))
                   for k, sh in shapes.items()},
           "act": torch.zeros((T, 1), dtype=torch.long), "logp": torch.zeros((T, 1)),
           "rew": torch.zeros((T, 1)), "done": torch.zeros((T, 1)),
           "val": torch.zeros((T, 1)), "seat": torch.zeros((T, 1), dtype=torch.long)}
    vec = _FakeVec(shapes, enc.int_keys, script)
    obs_np, seat_np, _ = vec.reset()
    cur_obs = _to_tensors(obs_np, enc.int_keys, "cpu")
    cur_seat = torch.as_tensor(seat_np)
    boot_val, boot_seat, _eprs, _end, terms = collect_rollout(_FakeNet(), vec, buf, shapes, enc.int_keys,
                                                              torch.device("cpu"), T, cur_obs, cur_seat)
    # terminal capture for the value diagnostics: both scripted terminals, keyed (step, env)
    assert set(terms.keys()) == {(1, 0), (4, 0)}, terms.keys()
    assert terms[(1, 0)] == {0: 1.0, 1: -1.0}
    # (a) done[step] is THIS step's terminal flag (NOT shifted to the next step)
    assert buf["done"].view(-1).tolist() == [0, 1, 0, 0, 1], buf["done"].view(-1).tolist()
    assert buf["seat"].view(-1).tolist() == [0, 1, 0, 1, 0], buf["seat"].view(-1).tolist()
    assert buf["rew"].view(-1).tolist() == [0, 1, 0, 0, 1], buf["rew"].view(-1).tolist()
    # (b) the GAE isolates the two episodes (no bootstrap across the boundary)
    _, ret = compute_gae_2s(buf["rew"], buf["val"], buf["done"], buf["seat"], boot_val, boot_seat, 1.0, 1.0)
    assert [round(x, 4) for x in ret.view(-1).tolist()] == [-1, 1, 1, -1, 1], ret.view(-1).tolist()


# ------------------------------------------------------- env tracking == inference replay
def _differential_replay(seat_under_test, would_ko, seed=11, max_steps=4000):
    """Drive a real game; replay `seat_under_test`'s stream through a FRESH isolated tracker set and
    return the number of (encodings, key-mismatches)."""
    import random
    from rl.decks import DECKS
    from rl.env_selfplay import TwoSidedSelfPlayEnv
    encoder = TokenEncoder(get_card_table())

    class Rec(TwoSidedSelfPlayEnv):
        def __init__(self, *a, **k):
            super().__init__(*a, **k); self.log = []
        def _encode(self):
            s = self._seat(); enc = super()._encode()
            if s == seat_under_test:
                assert self._obs["current"]["yourIndex"] == seat_under_test
                self.log.append(("enc", id(self._obs), tuple(self._picked), self._obs,
                                 {k: np.asarray(enc[k]).copy() for k in enc}))
            return enc
        def _apply(self, s, indices):
            if s == seat_under_test:
                self.log.append(("rec", self._obs["select"], list(indices)))
            return super()._apply(s, indices)

    env = Rec(decks=list(DECKS.values())[:2], would_ko=would_ko, seed=seed)
    enc, _seat, _info = env.reset()
    deck = list(env.deck[seat_under_test])
    rng = random.Random(1); steps = 0; done = False
    while not done and steps < max_steps:
        legal = np.flatnonzero(np.asarray(enc["action_mask"]) > 0.5)
        enc, _seat, _r, done, _info = env.step(int(rng.choice(legal))); steps += 1
    env.close()

    ftr, fab = GameTracker(), AbilityTracker(); last = None; n = mism = 0
    for e in env.log:
        if e[0] == "enc":
            _, oid, picked, obs, ee = e
            if oid != last:
                ftr.update(obs); last = oid          # obs already would_ko-annotated by env; do NOT re-annotate
            fab.note_turn(obs["current"]["turn"])
            re = encoder.encode(obs, set(picked), self_deck=deck, tracker=ftr, ability_slots=fab.slots)
            n += 1
            mism += sum(0 if np.array_equal(np.asarray(re[k]), ee[k]) else 1 for k in ee)
        else:
            fab.record(e[1], e[2])
    return n, mism


def test_two_sided_env_matches_inference_replay():
    for seat in (0, 1):
        for wko in (False, True):
            n, mism = _differential_replay(seat, wko)
            assert n > 0 and mism == 0, f"seat={seat} would_ko={wko}: {n} encodings, {mism} mismatches"


def test_gate_schedule():
    """Generalist promotion-gate schedule: deterministic, pilot-swap blocks, balanced coverage."""
    from collections import Counter
    from rl.train_selfplay import _gate_schedule

    assert _gate_schedule(50, 250, "mirror") == _gate_schedule(50, 250, "mirror")  # pure function
    s = _gate_schedule(50, 250, "mirror")
    assert len(s) == 250 and all(a == b for a, b in s)          # mirror: same deck both sides
    per = Counter(a for a, _ in s)
    assert max(per.values()) - min(per.values()) <= 2           # near-even deck coverage

    s = _gate_schedule(20, 200, "cross")
    for i in range(0, len(s) - 1, 2):                           # (i,j) immediately followed by (j,i)
        assert s[i] == (s[i + 1][1], s[i + 1][0])
    pilots = Counter(a for a, _ in s)
    assert min(pilots.values()) == max(pilots.values()) == 10   # every deck piloted equally

    assert _gate_schedule(1, 200, "mirror") == [(0, 0)] * 200   # specialist degenerate case
    assert _gate_schedule(1, 200, "cross") == [(0, 0)] * 200


# ------------------------------------------------- two-net graphed-vs-eager collect equivalence
class _RowVec:
    """N envs; a float obs feature carries a per-env id so a row-dependent fake net can prove the
    full-batch path selects the right rows. Seats alternate per (step, env); no terminals."""
    def __init__(self, shapes, int_keys, n, steps, fkey):
        self.shapes, self.int_keys, self.n, self.steps, self.fkey, self.t = shapes, int_keys, n, steps, fkey, 0

    def _obs(self):
        o = {k: np.zeros((self.n, *sh), dtype=(np.int64 if k in self.int_keys else np.float32))
             for k, sh in self.shapes.items()}
        f = o[self.fkey].reshape(self.n, -1)
        f[:, 0] = np.arange(self.n) + 1.0
        f[:, 1] = float(self.t)
        return o

    def _seats(self):
        return np.array([(self.t + e) % 2 for e in range(self.n)], dtype=np.int64)

    def reset(self):
        self.t = 0
        return self._obs(), self._seats(), [{}] * self.n

    def step(self, _actions):
        self.t += 1
        return (self._obs(), self._seats(), np.zeros(self.n, dtype=np.float32),
                np.zeros(self.n, dtype=np.bool_), [{}] * self.n)


class _RowNet:
    """Deterministic logits/value derived from the per-row obs id -> row-content-dependent, so the
    eager SUBSET forward and the full-batch forward + index_select agree iff the row plumbing is
    right. `scale` differentiates A from B."""
    value_categorical = False

    def __init__(self, fkey, n_opts, scale):
        self.fkey, self.n_opts, self.scale = fkey, n_opts, scale

    def logits_value(self, obs, opt_len=None, state_keep=None):
        f = obs[self.fkey].reshape(obs[self.fkey].shape[0], -1)
        v = f[:, 0] * self.scale + f[:, 1]
        logits = v[:, None] * torch.linspace(-1.0, 1.0, self.n_opts)[None, :] * 0.37
        return logits, v * 0.01

    def get_action_and_value(self, obs, opt_len=None, state_keep=None):
        logits, val = self.logits_value(obs)
        dist = torch.distributions.Categorical(logits=logits)
        a = dist.sample()
        return a, dist.log_prob(a), dist.entropy(), val

    def get_value(self, obs, opt_len=None, state_keep=None):
        return self.logits_value(obs)[1]


def test_two_net_graphed_collect_equivalence():
    """The 2026-07-31 sps rework routes the LIVE B through a full-batch compiled forward
    (rollout train_opp + opp_lv). Same rng => byte-identical act/logp/val/amask vs the eager
    per-step subset path it replaces."""
    enc = TokenEncoder(get_card_table())
    shapes = enc.shapes
    fkey = next(k for k in shapes if k not in enc.int_keys
                and int(np.prod(shapes[k])) >= 2)
    N, T = 4, 6
    n_opts = int(shapes["action_mask"][0])
    A = _RowNet(fkey, n_opts, scale=1.0)
    B = _RowNet(fkey, n_opts, scale=-1.3)

    def run(opp_lv):
        vec = _RowVec(shapes, enc.int_keys, N, T, fkey)
        buf = {"obs": {k: torch.zeros((T, N, *sh), dtype=(torch.long if k in enc.int_keys else torch.float32))
                       for k, sh in shapes.items()},
               "act": torch.zeros((T, N), dtype=torch.long), "logp": torch.zeros((T, N)),
               "rew": torch.zeros((T, N)), "done": torch.zeros((T, N)),
               "val": torch.zeros((T, N)), "seat": torch.zeros((T, N), dtype=torch.long),
               "amask": torch.zeros((T, N))}
        obs_np, seat_np, _ = vec.reset()
        cur_obs = _to_tensors(obs_np, enc.int_keys, "cpu")
        cur_seat = torch.as_tensor(seat_np)
        aseat = np.zeros(N, dtype=np.int64)                     # agent pilots seat 0 everywhere
        torch.manual_seed(1234)
        bv, bs, _e, _o, _t = collect_rollout(A, vec, buf, shapes, enc.int_keys, torch.device("cpu"),
                                             T, cur_obs, cur_seat, opp_net=B, opp_lv=opp_lv,
                                             agent_seat=aseat, train_opp=True)
        return buf, bv

    b_eager, bv_eager = run(None)
    b_graph, bv_graph = run(lambda o, opt_len=None, state_keep=None: B.logits_value(o))
    for k in ("act", "logp", "val", "amask", "seat"):
        assert torch.equal(b_eager[k], b_graph[k]), f"{k} diverged"
    assert torch.equal(bv_eager, bv_graph)
    # sanity: the exercise is real -- some rows were B's, and B actually sampled non-uniformly
    assert 0.0 < b_eager["amask"].mean().item() < 1.0


if __name__ == "__main__":
    test_negamax_gae(); print("test_negamax_gae OK")
    test_collect_done_convention(); print("test_collect_done_convention OK")
    test_two_sided_env_matches_inference_replay(); print("test_two_sided_env_matches_inference_replay OK")
    test_gate_schedule(); print("test_gate_schedule OK")
    test_two_net_graphed_collect_equivalence(); print("test_two_net_graphed_collect_equivalence OK")
    print("\nALL SELF-PLAY REGRESSION TESTS PASS")
