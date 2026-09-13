"""Scalable BC-dataset build from ZIPPED Kaggle replay archives.

Streams episode json straight out of .zip files (no full unzip), processes them in parallel, and
reuses the OFF-BY-ONE-FIXED rows_from_episode from build_bc_dataset.py (so the label fix + the
self-validating tripwire apply unchanged). Writes one compressed .npz and reports the masked rate.

  python scripts/build_bc_from_zips.py OUT.npz ZIP1 [ZIP2 ...]
Env: BC_WORKERS (default 16), BC_CAP_EPS (default 0 = all episodes in the given zips).
"""
import os
import sys
import json
import zipfile
from multiprocessing import Pool

import numpy as np

# import the FIXED rows_from_episode + shared encoder from the sibling script (scripts/ on path).
# build_bc_dataset reads sys.argv AT IMPORT (MAX_EPS=int(sys.argv[3])), so hide our zip args during import.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_argv = sys.argv
sys.argv = [_argv[0]]
import build_bc_dataset as B   # B.rows_from_episode (off-by-one fixed), B.enc
sys.argv = _argv

OUT = sys.argv[1]
ZIPS = sys.argv[2:]
WORKERS = int(os.environ.get("BC_WORKERS", "16"))
CAP = int(os.environ.get("BC_CAP_EPS", "0"))


def _job(arg):
    """(zip_path, member) -> (rows, labels, attack) for that episode; never raises (bad episode -> [])."""
    zp, name = arg
    try:
        with zipfile.ZipFile(zp) as z:
            ep = json.loads(z.read(name))
        rows, labs, atk = [], [], []
        for r, l, ia in B.rows_from_episode(ep):
            rows.append(r); labs.append(l); atk.append(ia)
        return rows, labs, atk
    except Exception:
        return [], [], []


def main():
    tasks = []
    for zp in ZIPS:
        with zipfile.ZipFile(zp) as z:
            tasks.extend((zp, n) for n in z.namelist() if n.endswith(".json"))
    if CAP:
        tasks = tasks[:CAP]
    print(f"[bc-zips] {len(tasks)} episodes from {len(ZIPS)} zip(s); workers={WORKERS}", flush=True)

    # apply_async + per-task get(timeout) instead of imap_unordered: a would_ko sim can infinite-loop
    # in the C engine on a pathological episode (no exception raised), and imap's blocking iteration
    # would hang the WHOLE build forever. Here a hung episode just times out and is skipped.
    rows, labels, attack, done, skipped = [], [], [], 0, 0
    TIMEOUT = float(os.environ.get("BC_EP_TIMEOUT", "60"))
    with Pool(WORKERS) as p:
        asyncs = [p.apply_async(_job, (t,)) for t in tasks]
        for a in asyncs:
            try:
                r, l, ia = a.get(timeout=TIMEOUT)
            except Exception:
                r, l, ia = [], [], []; skipped += 1     # hung/failed episode -> skip, keep going
            rows.extend(r); labels.extend(l); attack.extend(ia); done += 1
            if done % 1000 == 0:
                print(f"[bc-zips]   {done}/{len(tasks)} eps -> {len(rows)} rows (skipped {skipped})", flush=True)
    if skipped:
        print(f"[bc-zips] skipped {skipped} hung/failed episodes (timeout {TIMEOUT}s each)", flush=True)
    print(f"[bc-zips] {len(rows)} rows from {len(tasks)} episodes", flush=True)
    if not rows:
        print("[bc-zips] NO ROWS"); return

    int_keys = set(B.enc.int_keys)
    out = {}
    for k in rows[0].keys():
        dt = np.int32 if (k in int_keys or k == "__group__") else np.float32
        out[k] = np.stack([r[k] for r in rows]).astype(dt)
    out["__labels__"] = np.array(labels, dtype=np.int64)
    out["__is_attack__"] = np.array(attack, dtype=np.int8)
    # UNCOMPRESSED: high-entropy float32 features compress poorly, so savez_compressed spends
    # 15-20 min of zlib for little gain at ~1M rows. Plain savez writes in ~1-2 min (disk is ample).
    # OUT without a .npz suffix -> a DIRECTORY of per-key .npy files (bc_train's memmap mode: lazy
    # paging + ONE shared page-cache copy for all DDP ranks; skips the npz_to_npydir conversion).
    if OUT.endswith(".npz"):
        os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
        np.savez(OUT, **out)
    else:
        os.makedirs(OUT, exist_ok=True)
        for k, arr in out.items():
            np.save(os.path.join(OUT, k + ".npy"), arr)

    # self-check: with the off-by-one fix every kept label is legal -> masked rate MUST be 0
    am = out["action_mask"]; lab = out["__labels__"]; n = len(lab)
    masked = float((am[np.arange(n), lab] < 0.5).mean())
    # DEDUP self-check: the COLLAPSED label (its canonical) must still be legal under the DEDUP'd mask
    # (non-canonical legal dups zeroed). dup_legal_indices picks the first-LEGAL canonical, so this is 0.
    dmasked = -1.0
    if "__group__" in out:
        grp = out["__group__"]; A = am.shape[1]
        dmask = am * (grp == np.arange(A)[None, :])
        dlab = grp[np.arange(n), lab]
        dmasked = float((dmask[np.arange(n), dlab] < 0.5).mean())
    print(f"[bc-zips] wrote {OUT}: {n} rows, masked_rate={masked:.6f} dedup_masked_rate={dmasked:.6f} "
          f"(both MUST be 0.0); attack_rows={int(np.sum(attack))} wouldko={'on' if B.WOULD_KO else 'off'}", flush=True)


if __name__ == "__main__":
    main()
