"""Unit test for rl/value_diag.py -- the distance-to-terminal bucketing over a rollout buffer.

Hand-crafted [T=6, N=2] buffer with known episode boundaries, seats, values and outcomes; every
bucket count / sign-acc / MSE is verified against hand-computed numbers, including:
  * backward walk stops at the previous episode's done (never crosses episodes),
  * outcome is resolved PER ACTING SEAT from the terminal dict (negamax perspective),
  * draws (outcome 0) count for MSE but are excluded from sign-acc,
  * truncated episodes (no entry in ``terms``) are skipped entirely.
"""
import numpy as np

from rl.value_diag import diag_from_buffer, BUCKET_NAMES, DiagAccumulator


def _mk():
    T, N = 6, 2
    val = np.zeros((T, N), np.float32)
    done = np.zeros((T, N), np.float32)
    seat = np.zeros((T, N), np.int64)
    return val, done, seat


def test_buckets_and_walk():
    val, done, seat = _mk()
    # env 0: one episode ending at t=4 (seat pattern 0,1,0,1,0), seat0 wins.
    seat[:, 0] = [0, 1, 0, 1, 0, 0]
    done[4, 0] = 1.0
    val[:, 0] = [0.5, -0.5, 0.5, 0.4, 0.9, 0.0]      # correct sign everywhere for seat outcome
    # env 1: episode A ends t=1 (draw), episode B ends t=4; t=5 unfinished (ignored).
    seat[:, 1] = [0, 1, 0, 1, 0, 1]
    done[1, 1] = 1.0
    done[4, 1] = 1.0
    val[:, 1] = [0.2, 0.3, -0.6, 0.7, -0.8, 0.0]
    terms = {
        (4, 0): {0: 1.0, 1: -1.0},                    # env0: seat0 wins
        (1, 1): {0: 0.0, 1: 0.0},                     # env1 ep A: draw
        (4, 1): {0: -1.0, 1: 1.0},                    # env1 ep B: seat1 wins
    }
    out = diag_from_buffer(val, done, seat, terms, gamma=0.99)

    # env0 walk: k=1..5 (t=4..0); env1 epA: k=1..2 (t=1..0); env1 epB: k=1..3 (t=4..2, stops at done[1,1])
    # bucket "1-2" (k<=2): env0 k1,k2 + env1A k1,k2 + env1B k1,k2 -> n=6
    # bucket "3-4" (k in 3..4): env0 k3,k4 + env1B k3 -> n=3
    # bucket "5-8": env0 k5 -> n=1
    n_by = {BUCKET_NAMES[i]: row[0] for i, row in out.items()}
    assert n_by == {"1-2": 6, "3-4": 3, "5-8": 1}, n_by

    # sign-acc: draws (env1 epA, 2 samples) excluded from the denominator.
    # bucket "1-2" signed samples: env0 k1 (v=.9, seat0, r=+1 OK), k2 (v=.4, seat1, r=-1 WRONG),
    #   env1B k1 (v=-.8, seat0, r=-1 OK), k2 (v=.7, seat1, r=+1 OK) -> 3/4
    b12 = out[0]
    assert b12[2] == 4 and b12[1] == 3
    # bucket "3-4": env0 k3 (v=.5, seat0,+1 OK), k4 (v=-.5, seat1,-1 OK), env1B k3 (v=-.6, seat0,-1 OK) -> 3/3
    b34 = out[1]
    assert b34[2] == 3 and b34[1] == 3

    # mse_out spot-check, bucket "5-8": env0 k5, v=0.5, r(seat0)=+1 -> (0.5-1)^2 = 0.25
    b58 = out[2]
    assert abs(b58[3] / b58[0] - 0.25) < 1e-6
    # mse_disc spot-check: same sample vs gamma^4 * 1
    want = (0.5 - 0.99 ** 4) ** 2
    assert abs(b58[4] / b58[0] - want) < 1e-6


def test_truncation_skipped_and_accumulator():
    val, done, seat = _mk()
    done[3, 0] = 1.0                                  # truncated episode: done but NOT in terms
    out = diag_from_buffer(val, done, seat, {}, gamma=0.99)
    assert out == {}

    acc = DiagAccumulator(gamma=1.0)
    val[:, 0] = 1.0
    seat[:, 0] = 0
    acc.add(val, done, seat, {(3, 0): {0: 1.0, 1: -1.0}})
    acc.add(val, done, seat, {(3, 0): {0: 1.0, 1: -1.0}})
    rows, line = acc.summary(step=123)
    assert rows["1-2"]["n"] == 4 and rows["3-4"]["n"] == 4    # two windows accumulated
    assert rows["1-2"]["sign_acc"] == 1.0 and rows["1-2"]["mse_out"] == 0.0
    assert "1-2:sa=1.00" in line
    assert acc.b == {}                                # summary resets the window


if __name__ == "__main__":
    test_buckets_and_walk()
    test_truncation_skipped_and_accumulator()
    print("value_diag tests: PASSED")
