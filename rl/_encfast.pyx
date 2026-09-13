# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""Cython fast path for the hottest encode() array-fill helpers (byte-identical to the pure-Python
versions in rl/encoding.py; those remain the fallback when this module isn't compiled).

Profiled: encode() is ~75% of collection CPU and construction-bound. These two helpers are the
cleanest self-contained, array-write-heavy pieces (id-streams + energy histogram). Typed
memoryview writes replace numpy scalar-assignment overhead; _card_id is inlined.

Build (on a node with numpy, e.g. via scripts/build_encfast.sh):
    pip install --target=<dir> Cython && PYTHONPATH=<dir> python setup_encfast.py build_ext --inplace
The resulting .so imports with NO Cython at runtime.
"""
import numpy as np
cimport numpy as cnp


cdef inline long long _cid(object c):
    # mirror rl.encoding._card_id EXACTLY: None->0; dict-> (c['id'] or 0); else c or 0
    if c is None:
        return 0
    if isinstance(c, dict):
        return (<object>(c.get("id") or 0))
    return (<object>(c or 0))


def id_list(cards, int n, long long unk, int fill_unk):
    """== TokenEncoder._id_list: cards (list of dict/int/None) -> (ids[n] int64, mask[n] float32).
    Fills real card ids then ``fill_unk`` UNK ids, both masked 1.0; the rest stay 0 (padding)."""
    cdef cnp.ndarray[cnp.int64_t, ndim=1] ids = np.zeros(n, dtype=np.int64)
    cdef cnp.ndarray[cnp.float32_t, ndim=1] mask = np.zeros(n, dtype=np.float32)
    cdef long long[:] idv = ids
    cdef float[:] mv = mask
    cdef int i = 0, k
    cdef object c
    if cards is not None:
        for c in cards:
            if i >= n:
                break
            idv[i] = _cid(c)
            mv[i] = 1.0
            i += 1
    for k in range(fill_unk):
        if i >= n:
            break
        idv[i] = unk
        mv[i] = 1.0
        i += 1
    return ids, mask


def energy_hist12(energies):
    """== TokenEncoder._energy_hist12: 12-bin float32 histogram over int energy types in [0,12)."""
    cdef cnp.ndarray[cnp.float32_t, ndim=1] v = np.zeros(12, dtype=np.float32)
    cdef float[:] vv = v
    cdef object e
    cdef long ei
    if energies is not None:
        for e in energies:
            if isinstance(e, int):
                ei = e
                if 0 <= ei < 12:
                    vv[ei] += 1.0
    return v
