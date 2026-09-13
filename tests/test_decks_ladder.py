"""rl/decks_ladder: loader (scalar + legacy band-shape counts) + uniform-mix math."""
import json
import os
import tempfile

import numpy as np

from rl.decks_ladder import load_ladder_decks, mixed_weights


def test_mixed_weights_limits():
    counts = np.array([10.0, 0.0, 30.0])
    meta = counts / counts.sum()
    assert np.allclose(mixed_weights(counts, uniform_mix=0.0), meta)
    assert np.allclose(mixed_weights(counts, uniform_mix=1.0), [1 / 3] * 3)
    mid = mixed_weights(counts, uniform_mix=0.5)
    assert abs(mid.sum() - 1.0) < 1e-12
    assert np.allclose(mid, 0.5 * meta + 0.5 / 3)


def test_mixed_weights_zero_total():
    # all-zero counts (e.g. a fully retired file) degrade to uniform, not NaN
    assert np.allclose(mixed_weights(np.zeros(4), uniform_mix=0.0), [0.25] * 4)


def _roundtrip(counts_payload, expect):
    decks = [list(range(60)), [7] * 60]
    payload = {"version": "t", "decks": decks, "counts": counts_payload}
    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        json.dump(payload, open(path, "w"))
        d, c = load_ladder_decks(path)
        assert d == decks
        assert c.shape == (2,) and np.allclose(c, expect)
    finally:
        os.unlink(path)


def test_load_scalar_counts():
    _roundtrip([3, 14], [3.0, 14.0])


def test_load_legacy_band_counts_summed():
    # legacy [N, 3] per-band rows (raw pool files, [c,c,c] triplicated ft files) sum per deck
    _roundtrip([[3, 2, 1], [0, 5, 9]], [6.0, 14.0])
