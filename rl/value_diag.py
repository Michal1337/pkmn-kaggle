"""Value-head diagnostics: distance-to-terminal decomposition of value quality.

FREE at training time: reuses the PPO rollout buffer, no extra games. For every episode that ENDS
with a real terminal outcome inside the rollout window, every step of that episode inside the
window has a known steps-to-terminal ``k`` and a known final outcome for its ACTING seat (the
terminal info carries both seats' rewards; negamax perspective). Bucket by k and measure:

  * ``sign_acc``  : fraction sign(V(s)) == sign(outcome for the acting seat), draws excluded.
                    GAMMA-FREE -> comparable across gamma arms. THE decision metric.
  * ``mse_out``   : mean (V - outcome)^2 vs the RAW +/-1 outcome. Gamma-free, comparable.
  * ``mse_disc``  : mean (V - gamma^(k-1) * outcome)^2 -- each arm's OWN regression scale.
                    Do NOT compare across gammas (target distributions differ).
  * ``mean_absV`` : output-scale drift / "is the head stuck at 0" indicator.

Expected healthy shape: near-terminal buckets -> sign_acc ~1 / mse_out ~0 (lethal boards are
visible); early-game buckets pinned near outcome variance (sign_acc ~0.5-0.6). A flat & high
NEAR-TERMINAL bucket = broken/underpowered head; a flat EARLY bucket = the gamma/variance story
(see the value-head investigation, 2026-07-02).

Truncated episodes (no ``terminal`` info) and episode segments whose end lies outside the buffer
window are skipped -- their outcome is unknown.
"""
from __future__ import annotations

import json

# steps-to-terminal bucket upper edges (k=1 == the decision that ended the game)
BUCKETS = (2, 4, 8, 16, 32, 64, 128, 1 << 30)


def _bname(i: int) -> str:
    lo = 1 if i == 0 else BUCKETS[i - 1] + 1
    return f"{lo}-{BUCKETS[i]}" if BUCKETS[i] < (1 << 30) else f">{BUCKETS[i - 1]}"


BUCKET_NAMES = tuple(_bname(i) for i in range(len(BUCKETS)))


def _bidx(k: int) -> int:
    for i, edge in enumerate(BUCKETS):
        if k <= edge:
            return i
    return len(BUCKETS) - 1


def diag_from_buffer(val, done, seat, terms, gamma):
    """Per-bucket sums from one rollout buffer.

    val/done/seat: [T, N] numpy arrays (the trainer's buf tensors, moved to cpu);
    terms: {(t_step, env): {seat: reward}} for every REAL terminal (non-truncated) done in the
    window (captured by collect_rollout). Walks each terminal's episode backward inside the
    window; steps-to-terminal k = 1 at the terminal transition.

    Returns {bucket_idx: [n, sign_ok, sign_n, se_out, se_disc, sum_absV]}.
    """
    out: dict[int, list] = {}
    for (t, e), rr in terms.items():
        s, k = int(t), 1
        while s >= 0:
            r = float(rr.get(int(seat[s, e]), 0.0))
            v = float(val[s, e])
            b = out.setdefault(_bidx(k), [0, 0, 0, 0.0, 0.0, 0.0])
            b[0] += 1
            if r != 0.0:                       # draws carry no sign information
                b[2] += 1
                if (v > 0) == (r > 0):
                    b[1] += 1
            b[3] += (v - r) ** 2
            b[4] += (v - (gamma ** (k - 1)) * r) ** 2
            b[5] += abs(v)
            s -= 1
            if s < 0 or done[s, e] > 0.5:      # crossed into the previous episode -> stop
                break
            k += 1
    return out


class DiagAccumulator:
    """Accumulates diag_from_buffer sums across iterations; emits a compact log line + a JSONL
    record per summary window (full per-bucket metrics live in the JSONL, the line is a glance)."""

    def __init__(self, gamma: float, path: str | None = None):
        self.gamma = float(gamma)
        self.path = path
        self.reset()

    def reset(self):
        self.b: dict[int, list] = {}

    def add(self, val, done, seat, terms):
        for i, row in diag_from_buffer(val, done, seat, terms, self.gamma).items():
            acc = self.b.setdefault(i, [0, 0, 0, 0.0, 0.0, 0.0])
            for j in range(6):
                acc[j] += row[j]

    def summary(self, step=None, reset: bool = True):
        """-> (rows dict, one-line string); appends a JSONL record when a path is set."""
        rows = {}
        for i in sorted(self.b):
            n, sok, sn, seo, sed, sav = self.b[i]
            rows[BUCKET_NAMES[i]] = {
                "n": n,
                "sign_acc": (sok / sn) if sn else None,
                "mse_out": seo / n,
                "mse_disc": sed / n,
                "mean_absV": sav / n,
            }
        if self.path and rows:
            with open(self.path, "a") as f:
                f.write(json.dumps({"step": step, "gamma": self.gamma, "buckets": rows}) + "\n")
        parts = []
        for name, m in rows.items():
            sa = f"{m['sign_acc']:.2f}" if m["sign_acc"] is not None else "--"
            parts.append(f"{name}:sa={sa}|v={m['mean_absV']:.2f}")
        line = " ".join(parts)
        if reset:
            self.reset()
        return rows, line
