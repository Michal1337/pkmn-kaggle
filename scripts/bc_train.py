"""Behavioral-cloning trainer for a given v2 architecture -> reports val top-1 accuracy.

Cheap ARCHITECTURE SELECTOR: supervised CE on expert (winning-side) decisions, no RL. Run with
different --emb-dim/--static/... and compare best_val_acc to pick the architecture.

  python scripts/bc_train.py <data.npz> --emb-dim 48            # baseline
  python scripts/bc_train.py <data.npz> --emb-dim 48 --static   # static features
  python scripts/bc_train.py <data.npz> --emb-dim 128           # bigger embedding

DDP (auto-detected from torchrun env; no flag):
  torchrun --nproc_per_node=2 scripts/bc_train.py <data.npz> ...
Gradients are all-reduce-averaged, so effective batch = --batch * world at identical math;
validation/saves/prints run on rank0 only. Sharding: .npz (in-RAM) mode shards a SHARED per-epoch
row permutation (identical seeds -> disjoint equal shards); .npy-DIR (memmap) mode shards SLABS --
see batches_slab. MEMORY: memmap mode streams ~3 slabs (~30GB) + a rank0 val cache, so the 479GB
bc_v22_full trains inside a 200G allocation on any node.

would_KO A/B (data must be built with BC_WOULD_KO=1 so opt_attr's trio is populated):
  python scripts/bc_train.py <wk.npz>                    # would_ko ON
  python scripts/bc_train.py <wk.npz> --zero-wouldko     # would_ko OFF (same rows, trio zeroed)
Reports overall val_acc plus ATTACK-decision and KO-AVAILABLE subset accuracies (where would_ko
bites). The KO-available subset is derived from the would_ko-rate column BEFORE any zeroing, so
both arms score the identical row subset.
"""
import argparse
import datetime
import os
import queue
import threading
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from rl.card_features import get_card_table
from rl.encoding import TokenEncoder
from rl.enc_constants import OPT_WK
from rl.policy import build_token_net

WK_LO, WK_HI = OPT_WK, OPT_WK + 3   # opt_attr cols [WK_LO:WK_HI] = would_ko (rate / exp-prizes / P-win)


def read_rows(arr, start, stop):
    """Rows [start:stop) of a np.load(mmap_mode='r') array via BUFFERED SEQUENTIAL file reads
    (np.fromfile at the right offset). Touching the mmap instead faults 4KB pages synchronously --
    measured ~10-20MB/s/node on GPFS vs ~GB/s streaming, i.e. hours-per-epoch vs minutes on the
    479GB set. In-RAM arrays just slice-copy."""
    if not isinstance(arr, np.memmap):
        return np.asarray(arr[start:stop])
    rowel = int(np.prod(arr.shape[1:], dtype=np.int64))
    flat = np.fromfile(arr.filename, dtype=arr.dtype, count=(stop - start) * rowel,
                       offset=arr.offset + start * rowel * arr.dtype.itemsize)
    return flat.reshape((stop - start,) + arr.shape[1:])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data", help=".npz file, or a DIRECTORY of per-key .npy files (memmap mode)")
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--nlayers", type=int, default=3)
    p.add_argument("--ff", type=int, default=256)
    p.add_argument("--static", action="store_true")
    p.add_argument("--split-heads", action="store_true",
                   help="dedicated value/submit tokens (match the production RL recipe -> warm-start compatible)")
    p.add_argument("--structured", action="store_true", help="verb-conditioned action head")
    p.add_argument("--zero-wouldko", action="store_true", help="zero the would_ko trio (nowk arm)")
    p.add_argument("--dedup", action="store_true",
                   help="collapse provably-interchangeable single-pick options (mask non-canonical "
                        "legal dups + relabel pick to its canonical) using the dataset's __group__")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=5e-4,
                   help="peak LR (with warmup+cosine below). 5e-4 is safe for d128 AND d256/L5 -- the "
                        "old constant 1e-3 collapsed the 5M net (flat val 0.41); d256 needed 1e-4 flat")
    p.add_argument("--lr-schedule", choices=["cosine", "linear", "none"], default="cosine")
    p.add_argument("--warmup-steps", type=int, default=1500,
                   help="linear LR warmup over this many OPTIMIZER steps (flat, dataset-size-invariant; "
                        "clamped to 20%% of the run). Stabilizes big nets early (the d256/L5 collapse)")
    p.add_argument("--warmup-frac", type=float, default=0.0,
                   help="DEPRECATED %%-of-total warmup; >0 overrides --warmup-steps")
    p.add_argument("--lr-min-ratio", type=float, default=0.1, help="final LR = lr * this (schedule floor)")
    p.add_argument("--max-grad-norm", type=float, default=1.0, help="grad clipping (0 = off)")
    p.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True,
                   help="bf16-autocast forward, fp32 loss (same recipe as the RL update; ~1.8x)")
    p.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True,
                   help="torch.compile the forward (CUDA only; static shapes)")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--slab-rows", type=int, default=262144,
                   help="memmap-mode streaming slab size in rows (~40KB/row -> ~10GB/slab; a slab "
                        "spans ~1.8k episodes so within-slab shuffling keeps batches decorrelated)")
    p.add_argument("--init-from", type=str, default=None,
                   help="BC-FINETUNE: warm-start from an RL ckpt; the net is built from the ckpt's "
                        "net_config (arch flags ignored) and loaded STRICT")
    p.add_argument("--lr-new", type=float, default=None,
                   help="separate lr for zero-history params (decktop_emb); scheduled at a "
                        "constant ratio to --lr. None = single lr for everything")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    # DDP: auto-detected from the torchrun env; single-process runs are untouched (world == 1)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    ddp = world > 1
    if ddp:
        backend = "nccl" if torch.cuda.is_available() else "gloo"   # gloo -> CPU smoke-testable
        dist.init_process_group(backend=backend, timeout=datetime.timedelta(minutes=60))
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            a.device = f"cuda:{local_rank}"
        else:
            a.device = "cpu"
    is_main = rank == 0

    torch.manual_seed(a.seed); np.random.seed(a.seed)
    ct = get_card_table(); enc = TokenEncoder(ct)
    int_keys = set(enc.int_keys)
    dev = torch.device(a.device)

    # data: a .npz (decompressed fully into RAM; random row gathers are fine there) or a DIRECTORY
    # of per-key .npy files (numpy MEMMAP; produce with scripts/npz_to_npydir.py). Memmap arrays are
    # NEVER gathered row-wise -- all bulk access goes through read_rows (see above for why) and
    # training runs in slab-streaming order (batches_slab). Tensor + dtype conversion happens PER
    # BATCH (materializing full int64 tensors up front doubled the id memory).
    mmapped = os.path.isdir(a.data)
    if mmapped:
        d = {f[:-4]: np.load(os.path.join(a.data, f), mmap_mode="r")
             for f in sorted(os.listdir(a.data)) if f.endswith(".npy")}
    else:
        z = np.load(a.data)
        d = {k: z[k] for k in z.files}
    N = int(d["__labels__"].shape[0])
    labels = read_rows(d["__labels__"], 0, N)
    has_group = "__group__" in d
    if a.dedup and not has_group:
        raise SystemExit("--dedup needs a dataset built with __group__ (rebuild with the updated builder)")
    keys = [k for k in d if k not in ("__labels__", "__is_attack__", "__group__")]
    obs_np = {k: d[k] for k in keys}
    group_np = d.get("__group__")

    # ---- DEDUP arm: collapse provably-interchangeable single-pick options (validated in option_dedup) ----
    # group[i] = the first-LEGAL canonical option index for option i (identity for unique options and
    # for every option of a multi-pick select, where dedup is disabled). Arm B masks the non-canonical
    # legal duplicates out of the action-mask (PER BATCH, in batches()) and relabels the expert pick
    # to its canonical, so the softmax never splits probability across identical options.
    if a.dedup:
        for _s in range(0, N, 1 << 20):                           # chunked: group_np may be a memmap
            _e = min(N, _s + (1 << 20))
            _g = read_rows(group_np, _s, _e)
            labels[_s:_e] = _g[np.arange(_e - _s), labels[_s:_e]]  # relabel expert pick -> legal canonical
    y = torch.as_tensor(labels, dtype=torch.long)
    is_attack = (torch.as_tensor(read_rows(d["__is_attack__"], 0, N), dtype=torch.long).bool()
                 if "__is_attack__" in d else torch.zeros(N, dtype=torch.bool))

    nval = max(1, int(N * a.val_frac))
    # GAME-LEVEL holdout: rows are stored grouped by episode (build order), so the contiguous TAIL is
    # ~whole held-out games -> NO same-game train/val leakage. (A random ROW split leaks badly: a game's
    # ~78 near-identical consecutive decisions scatter across train+val, inflating val_acc.)
    v0 = N - nval
    idx = np.arange(N)
    vi, ti = idx[v0:], idx[:v0]

    # VAL CACHE, rank0 only (val/metrics run there): ONE sequential streaming read of the val tail
    # (~40KB/row -> ~48GB on the full 12M-row set), reused by every epoch's eval. Also the source of
    # the would_ko subset metrics (RAW trio, before any zeroing).
    val_np = gv_np = None
    if is_main:
        _t0 = time.time()
        val_np = {k: read_rows(obs_np[k], v0, N) for k in keys}
        gv_np = read_rows(group_np, v0, N) if group_np is not None else None
        oa_v = val_np["opt_attr"]                          # (nval, MAX_OPTIONS, W)
        wk_present = float(np.abs(oa_v[..., WK_LO:WK_HI]).sum())
        vi_atk = is_attack[torch.as_tensor(vi)]
        vi_ko = torch.as_tensor((oa_v[..., WK_LO] >= 0.5).any(axis=1))
        print(f"[bc-train] val cache: {sum(x.nbytes for x in val_np.values()) / 1e9:.1f} GB "
              f"in {time.time() - _t0:.0f}s", flush=True)

    # SLAB grid over the train range [0, v0): memmap mode shuffles + shards CONTIGUOUS slabs, never
    # scattered rows (see read_rows). In-RAM (.npz) mode keeps the old full row permutation.
    slab_bounds = [(s, min(s + a.slab_rows, v0)) for s in range(0, v0, a.slab_rows)]
    if mmapped and len(slab_bounds) < world:
        raise SystemExit(f"--slab-rows {a.slab_rows} leaves {len(slab_bounds)} slab(s) for {world} "
                         f"ranks; lower --slab-rows")

    def epoch_plan(ep):
        """Shared-seed slab shuffle -> ranks own DISJOINT slab subsets. Returns (this rank's slabs,
        lockstep step budget). The budget is the MIN of the per-rank batch counts, computed
        identically on every rank from the shared shuffle (unequal step counts would deadlock the
        grad all-reduce); ranks holding more batches skip their few final ones."""
        g = torch.Generator().manual_seed(a.seed * 100_000 + ep)
        so = torch.randperm(len(slab_bounds), generator=g).tolist()
        per_rank = [[slab_bounds[i] for i in so[r::world]] for r in range(world)]
        nsteps = min(sum((e0 - s0 + a.batch - 1) // a.batch for s0, e0 in sl) for sl in per_rank)
        return per_rank[rank], nsteps

    if a.init_from:
        # BC-FINETUNE mode (2026-08-09, KS-BC): warm-start from an RL checkpoint. The net is built
        # from the CKPT's own net_config (arch flags on the command line are ignored -- a silent
        # arch mismatch would otherwise fail strict load or, worse, half-load), and the load is
        # STRICT: the source here is v2.3 like the current code, so any mismatch is a real error.
        _ick = torch.load(a.init_from, map_location="cpu")
        cfg = _ick["net_config"]
        net = build_token_net(ct, cfg).to(dev)
        net.load_state_dict(_ick["net"])
        print(f"[bc-train] INIT-FROM {a.init_from} (step {_ick.get('global_step')}) "
              f"cfg={cfg}", flush=True)
        del _ick
    else:
        cfg = {"arch": "transformer2", "d_model": a.d_model,
               "nhead": a.nhead, "nlayers": a.nlayers, "ff": a.ff, "static": a.static,
               "structured": a.structured, "split_heads": a.split_heads}
        net = build_token_net(ct, cfg).to(dev)
    if ddp:                                        # identical initial weights across ranks
        for pp in net.parameters():
            dist.broadcast(pp.data, src=0)
        for bb in net.buffers():
            dist.broadcast(bb.data, src=0)
    # --lr-new: separate (usually hotter) lr for named zero-history params -- decktop_emb never
    # received gradient pressure during RL (the ablation shows it inert), so it can absorb a
    # larger step while the mature trunk stays at the gentle finetune lr.
    _new_names = [n for n, _ in net.named_parameters() if n in ("decktop_emb",)]
    if a.lr_new is not None and _new_names:
        _new = {n for n in _new_names}
        opt = torch.optim.Adam(
            [{"params": [p for n, p in net.named_parameters() if n not in _new], "lr": a.lr,
              "initial_ratio": 1.0},
             {"params": [p for n, p in net.named_parameters() if n in _new], "lr": a.lr_new,
              "initial_ratio": a.lr_new / a.lr}],
            lr=a.lr)
        print(f"[bc-train] param-group lr: base={a.lr} new={a.lr_new} for {_new_names}", flush=True)
    else:
        opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    # speed stack (mirrors the RL trainers): TF32 matmuls, bf16-autocast forward with fp32 loss,
    # torch.compile'd forward (static shapes: fixed batch + full MAX_OPTIONS; last ragged batch and
    # the val remainder add a couple of specializations -> bump the dynamo budget like train_selfplay)
    use_cuda = dev.type == "cuda"
    if use_cuda:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    fwd = net.logits_value
    if a.compile and use_cuda:
        torch._dynamo.config.recompile_limit = 64
        torch._dynamo.config.cache_size_limit = 64
        fwd = torch.compile(net.logits_value, dynamic=False)
        if is_main:
            print("[compile] forward compiled (torch.compile dynamic=False)", flush=True)
    amp = torch.autocast("cuda", dtype=torch.bfloat16, enabled=(a.bf16 and use_cuda))
    # per-STEP warmup + cosine decay to lr*min_ratio (reuses the RL trainer's schedule; the old
    # constant-LR training both collapsed d256/L5 at 1e-3 and left late epochs noisy at full LR)
    from rl.train_selfplay import lr_at
    if mmapped:
        total_steps = max(1, sum(epoch_plan(ep)[1] for ep in range(a.epochs)))
    else:
        shard_len = (len(ti) - len(ti) % world) // world if ddp else len(ti)
        steps_per_ep = (shard_len + a.batch - 1) // a.batch
        total_steps = max(1, a.epochs * steps_per_ep)
    warmup_steps = (int(a.warmup_frac * total_steps) if a.warmup_frac > 0
                    else min(a.warmup_steps, max(1, total_steps // 5)))
    gstep = 0
    lossf = nn.CrossEntropyLoss()
    nparams = sum(pp.numel() for pp in net.parameters())
    tag = (f"d{a.d_model}L{a.nlayers}h{a.nhead}{'+static' if a.static else ''}"
           f"{'+split' if a.split_heads else ''}"
           f"{'+struct' if a.structured else ''}{'+NOWK' if a.zero_wouldko else '+WK'}"
           f"{'+DEDUP' if a.dedup else '+RAW'}")
    if is_main:
        print(f"[bc-train] {tag} params={nparams:,} N={N} train={len(ti)} val={len(vi)} dev={dev} "
              f"world={world} eff_batch={a.batch * world} "
              f"wk_present={wk_present:.1f} val_atk={int(vi_atk.sum())} val_ko={int(vi_ko.sum())}", flush=True)

    def batches(arrs, grp, base, order, bs):
        """Yield (obs, label) batches. arrs = IN-RAM per-key arrays whose row i is GLOBAL row
        base+i; order = LOCAL row indices (memmaps must be materialized via read_rows first --
        never index a memmap here). Converts dtype + moves to device per batch; dedup masking and
        would_ko zeroing are applied HERE (keeps the resident footprint at one batch, and a
        read-only mmap couldn't be edited in place anyway)."""
        for i in range(0, len(order), bs):
            b = order[i:i + bs]
            ob = {k: torch.as_tensor(np.asarray(arrs[k][b]),
                                     dtype=(torch.long if k in int_keys else torch.float32),
                                     device=dev) for k in keys}
            if a.dedup:
                gb = torch.as_tensor(np.asarray(grp[b]), dtype=torch.long, device=dev)
                canon = (gb == torch.arange(gb.shape[1], device=dev)[None, :]).float()
                ob["action_mask"] = ob["action_mask"] * canon     # drop non-canonical legal dups
            if a.zero_wouldko:                                    # nowk arm: feature off, rows kept
                ob["opt_attr"][..., WK_LO:WK_HI] = 0.0
            yield ob, y[torch.as_tensor(base + b)].to(dev)

    def slab_stream(slabs):
        """Prefetch generator: a daemon thread sequential-reads the NEXT slab (read_rows releases
        the GIL during I/O) while the GPU trains on the current one. maxsize=1 -> at most ~3 slabs
        resident (in use / queued / being read)."""
        q = queue.Queue(maxsize=1)

        def work():
            for s0, e0 in slabs:
                sd = {k: read_rows(obs_np[k], s0, e0) for k in keys}
                sg = read_rows(group_np, s0, e0) if (a.dedup and group_np is not None) else None
                q.put((s0, e0, sd, sg))
            q.put(None)

        threading.Thread(target=work, daemon=True).start()
        while True:
            item = q.get()
            if item is None:
                return
            yield item

    gv = torch.as_tensor(gv_np, dtype=torch.long) if gv_np is not None else None
    yv_group = gv[torch.arange(len(vi)), y[torch.as_tensor(vi)]] if gv is not None else None

    def train_step(ob, yb):
        nonlocal gstep
        gstep += 1
        if a.lr_schedule != "none":
            _base = lr_at(gstep, total_steps, a.lr,
                          a.lr_schedule, warmup_steps, a.lr_min_ratio)
            # scale EVERY param group by the same schedule factor, so --lr-new keeps its ratio
            for _pg in opt.param_groups:
                _pg["lr"] = _base * (_pg.get("initial_ratio") or 1.0)
        with amp:
            logits = fwd(ob)[0]
        loss = lossf(logits.float(), yb)                          # loss/softmax in fp32 (bf16-safe)
        opt.zero_grad(); loss.backward()
        if ddp:                                           # average grads -> identical step on all ranks
            for pp in net.parameters():
                if pp.grad is not None:
                    dist.all_reduce(pp.grad)
                    pp.grad /= world
        if a.max_grad_norm > 0:                           # clip AFTER averaging (identical everywhere)
            nn.utils.clip_grad_norm_(net.parameters(), a.max_grad_norm)
        opt.step()

    best = best3 = best_atk = best_ko = best_eq = 0.0
    for ep in range(a.epochs):
        net.train(); ep_t0 = time.time()
        if mmapped:
            # SLAB-STREAMING epoch: shuffled slab order (shared seed -> disjoint rank shards),
            # sequential slab reads overlapped with compute, rows shuffled WITHIN each slab.
            mine, nsteps = epoch_plan(ep)
            done = 0
            stream = slab_stream(mine)
            for s0, e0, sd, sg in stream:
                perm = np.random.default_rng([a.seed, ep, s0]).permutation(e0 - s0)
                for ob, yb in batches(sd, sg, s0, perm, a.batch):
                    if done >= nsteps:                    # lockstep budget reached (see epoch_plan)
                        break
                    done += 1
                    train_step(ob, yb)
            for _ in stream:                              # drain so the prefetch thread exits
                pass                                      # (it blocks on q.put otherwise)
        else:
            # in-RAM epoch: SHARED full row permutation (same generator seed on every rank) ->
            # ranks take disjoint equal-size shards in lockstep; the < world remainder is dropped
            # so every rank runs the SAME step count (unequal counts deadlock the all-reduce).
            g = torch.Generator().manual_seed(a.seed * 100_000 + ep)
            order = ti[torch.randperm(len(ti), generator=g).numpy()]
            if ddp:
                order = order[:len(order) - len(order) % world][rank::world]
            for ob, yb in batches(obs_np, group_np, 0, order, a.batch):
                train_step(ob, yb)
        if not is_main:
            continue                                       # val/saves/prints: rank0 only (ranks stay in
        net.eval(); preds = []; am_all = []; vloss = 0.0; tot = 0   # lockstep via the all-reduce above)
        with torch.no_grad():
            # unshuffled + from the rank0 val cache -> preds align to vi order
            for ob, yb in batches(val_np, gv_np, v0, np.arange(nval), a.batch):
                with amp:
                    lg = fwd(ob)[0]
                lg = lg.float()
                vloss += float(lossf(lg, yb)) * len(yb); tot += len(yb)
                top3 = (lg.topk(3, 1).indices == yb[:, None]).any(1)
                preds.append(torch.stack([(lg.argmax(1) == yb), top3], 1).cpu())
                am_all.append(lg.argmax(1).cpu())
        pr = torch.cat(preds); c1, c3 = pr[:, 0], pr[:, 1]   # (val, 2): [correct@1, correct@3]
        acc = float(c1.float().mean()); t3 = float(c3.float().mean())
        atk = float(c1[vi_atk].float().mean()) if int(vi_atk.sum()) else 0.0
        ko = float(c1[vi_ko].float().mean()) if int(vi_ko.sum()) else 0.0
        # effect-equivalence-aware acc (FAIR A/B metric): argmax and expert pick map to the SAME
        # signature group. For arm A this credits picking an interchangeable copy; for arm B (dups
        # already masked) it coincides with raw acc. Identical computation -> directly comparable.
        eq = acc
        if gv is not None:
            am_all = torch.cat(am_all)
            eq = float((gv[torch.arange(len(vi)), am_all] == yv_group).float().mean())
        if a.out and acc > best:                 # save BEST-val weights, rolling (was: FINAL weights
            torch.save({"net": net.state_dict(), "net_config": cfg,   # once at the end -- late val
                        "bc_val_acc": acc, "epoch": ep}, a.out)       # regression would ship a worse net,
        best = max(best, acc); best3 = max(best3, t3); best_atk = max(best_atk, atk)   # and no ckpt existed mid-run)
        best_ko = max(best_ko, ko); best_eq = max(best_eq, eq)
        print(f"[bc-train] {tag} ep{ep} val_acc={acc:.4f} equiv_acc={eq:.4f} top3={t3:.4f} "
              f"atk_acc={atk:.4f} ko_acc={ko:.4f} val_loss={vloss / max(tot, 1):.4f} "
              f"t={time.time() - ep_t0:.0f}s", flush=True)
    if is_main:
        print(f"[bc-train] RESULT {tag}: best_val_acc={best:.4f} best_equiv_acc={best_eq:.4f} best_top3={best3:.4f} "
              f"best_atk_acc={best_atk:.4f} best_ko_acc={best_ko:.4f} params={nparams:,}", flush=True)
    if ddp:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()
