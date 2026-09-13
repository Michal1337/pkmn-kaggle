"""Subprocess vector env for the TWO-SIDED self-play env (rl.env_selfplay).

Much leaner than ``rl.vec_env.SubprocVecEnv``: there is NO internal opponent (the trainer's live net
pilots both seats), so there is no opponent net, no weight broadcast, no inference server. Each worker
just drives K ``TwoSidedSelfPlayEnv``s and returns, per step, each env's next decision's (obs, seat,
reward, done). Auto-resets on episode end (like the single-sided worker).

K-ENVS-PER-WORKER (2026-07-13 engine sprint): the engine is fully multi-battle per process
(ApiBattleStart = new ApiData(), zero mutable statics -- proven by scripts/test_multibattle.py; the
old one-battle-per-process rule was sdk_cg's Python class-attr singleton, now per-instance). With
one env per process, 768 workers on 124 cores meant ~6x oversubscription and every step barrier
waited on the SLOWEST process's scheduling tail -- that tail WAS the engine wall (~5.4s/iter on
hopper-2, 16.5s joins on the stud-3 finetune). ``envs_per_worker=K`` packs K envs per process,
stepped sequentially per barrier: procs ~= cores, no tail, K-fold fewer pipe messages. Default 1 =
the exact previous topology. K>1 requires native_encode (the JSON fallback path still uses the
kaggle_environments module-global engine handle).

step(actions) -> (obs_stacked, seats[N], rewards[N], dones[N], infos): the trainer batches the obs,
runs the live net to pick an action for whatever seat is acting in each env, and feeds them back.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from multiprocessing import shared_memory

import numpy as np


# absorbed from the retired rl/vec_env.py:
def _attach_shm(name):
    """Attach to a main-owned shared-memory block. Unregister from THIS process's
    resource_tracker so worker exit doesn't try to unlink a block the main owns."""
    shm = shared_memory.SharedMemory(name=name)
    try:
        resource_tracker.unregister(shm._name, "shared_memory")
    except Exception:
        pass
    return shm



def _build_env(encoder, seed, env_kwargs, local_ix=0):
    """TwoSidedSelfPlayEnv, or FrozenOpponentEnv when env_kwargs carries a 'frozen_opp' config
    (deck-finetune mode: fixed agent deck vs a frozen net piloting weighted meta decks).
    FROZEN-WIDE partition: when frozen_opp carries 'wide_flags' (per-env bools, sliced per
    worker by SelfPlayVecEnv), flagged envs sample opponents from the wide deck list instead
    of the concentrated mixture -- the trainer pilots those seats with the frozen base."""
    kw = dict(env_kwargs)
    fo = kw.pop("frozen_opp", None)
    if fo:
        from rl.env_selfplay import AsymmetricDeckEnv
        kw.pop("decks", None)                     # decks come from the finetune config
        wf = fo.get("wide_flags")
        ks = fo.get("ks_flags")                   # SECOND frozen stream (2026-08-10): specialist-
        # piloted deck list (the ship-ft KS stream) -- same partition mechanics as wide.
        # agent side may be a deck SET (KS-family finetune) -- None-valued keys are the
        # single-deck default inside AsymmetricDeckEnv, so this is a no-op for ship finetunes.
        _aset = {"agent_decks": fo.get("agent_decks"), "agent_weights": fo.get("agent_weights"),
                 "agent_ids": fo.get("agent_ids")}
        if ks is not None and ks[local_ix]:
            return AsymmetricDeckEnv(fo["agent_deck"], fo["ks_decks"], fo["ks_weights"],
                                     encoder=encoder, seed=seed, **_aset, **kw)
        if wf is not None and wf[local_ix]:
            return AsymmetricDeckEnv(fo["agent_deck"], fo["wide_decks"], fo["wide_weights"],
                                     encoder=encoder, seed=seed, **_aset, **kw)
        return AsymmetricDeckEnv(fo["agent_deck"], fo["opp_decks"], fo["opp_weights"],
                                 encoder=encoder, seed=seed, **_aset, **kw)
    from rl.env_selfplay import TwoSidedSelfPlayEnv
    return TwoSidedSelfPlayEnv(encoder=encoder, seed=seed, **kw)


def _ref_spec(env_kwargs):
    """Per-key encoded-obs (shape, dtype) from one throwaway TwoSidedSelfPlayEnv reset -- shapes are
    fixed (the encoder pads to constants), so one sample defines the shared batch buffers. Finetune
    mode probes shapes with the BASE env (encoder-constant) -- no frozen-net load in the main."""
    from rl.env_selfplay import TwoSidedSelfPlayEnv
    from rl.encoding import TokenEncoder
    from rl.card_features import get_card_table
    kw = dict(env_kwargs)
    fo = kw.pop("frozen_opp", None)
    if fo:
        kw["decks"] = fo.get("agent_decks") or [fo["agent_deck"]]
    env = TwoSidedSelfPlayEnv(encoder=TokenEncoder(get_card_table()), seed=0, **kw)
    try:
        obs, _, _ = env.reset()
    finally:
        env.close()
    return {k: (tuple(v.shape), v.dtype) for k, v in obs.items()}


def _worker(remote, parent_remote, env_kwargs, seeds, obs_shm=None):
    """Hosts len(seeds) envs (global ids idx0..idx0+K-1), stepped sequentially per barrier.
    Pipe protocol (LISTS, one message per barrier):
      ("reset", None)      -> [per-env reset payload]
      ("step", [aK])       -> [per-env step payload]      (auto-reset + in-env recovery per env)
      ("close", None)      -> True
    With obs_shm the encoded obs go to this worker's rows of the shared batch buffer and the pipe
    carries only the light tuples; the main's recv barriers until the rows are written."""
    parent_remote.close()
    import logging
    logging.disable(logging.CRITICAL)
    from rl.encoding import TokenEncoder
    from rl.card_features import get_card_table

    enc = TokenEncoder(get_card_table())
    envs = [_build_env(enc, s, env_kwargs, local_ix=j) for j, s in enumerate(seeds)]
    K = len(envs)

    _write_obs = None
    if obs_shm is not None:
        _idx0, _n = obs_shm["idx0"], obs_shm["n"]
        _oshm, _obufs = [], {}
        for k, (shp, dt) in obs_shm["spec"].items():
            s = _attach_shm(obs_shm["names"][k])          # don't register for unlink (main owns it)
            _oshm.append(s)
            _obufs[k] = np.ndarray((_n, *shp), dtype=dt, buffer=s.buf)
        _zero_obs = {k: np.zeros(shp, dtype=dt) for k, (shp, dt) in obs_shm["spec"].items()}

        def _write_obs(j, o, _obufs=_obufs, _idx0=_idx0, _zero=_zero_obs):
            if o is None:                                 # recovery fell back to a None _last_enc ->
                o = _zero                                 # write zeros (neutral boundary), never crash
            for k in _obufs:
                _obufs[k][_idx0 + j] = o[k]

    def _step_one(j, action, last_obs):
        env = envs[j]
        try:
            obs, seat, r, done, info = env.step(action)
            if done:
                info = {**info, "done": True}
                obs, seat, rinfo = env.reset()           # auto-reset -> next episode's first decision
                if "agent_seat" in rinfo:                # asymmetric mode: the NEW episode's agent
                    info["next_agent_seat"] = rinfo["agent_seat"]   # seat (obs/seat are new-episode)
        except Exception:
            # a bad engine state must not kill the worker -> recover with a neutral boundary
            try:
                obs, seat, rinfo = env.reset()
                info = {"recovered": True}
                if "agent_seat" in rinfo:
                    info["next_agent_seat"] = rinfo["agent_seat"]
            except Exception:
                obs, seat, info = env._last_enc, 0, {"recovered": True}
            r, done = 0.0, True
        if obs is None:                                   # doubly-failed recovery (_last_enc unset): zero
            obs = {k: np.zeros_like(v) for k, v in last_obs.items()}  # obs -- the non-shm _stack
        return obs, int(seat), float(r), bool(done), info

    try:
        last_obs = [None] * K                             # last successfully-produced obs (shape donors)
        while True:
            cmd, data = remote.recv()
            if cmd == "reset":
                out = []
                for j, env in enumerate(envs):
                    obs, seat, info = env.reset()
                    last_obs[j] = obs
                    if _write_obs is not None:
                        _write_obs(j, obs); out.append((seat, info))
                    else:
                        out.append((obs, seat, info))
                remote.send(out)
            elif cmd == "step":
                out = []
                for j, a in enumerate(data):
                    obs, seat, r, done, info = _step_one(j, a, last_obs[j])
                    last_obs[j] = obs
                    if _write_obs is not None:
                        _write_obs(j, obs); out.append((seat, r, done, info))
                    else:
                        out.append((obs, seat, r, done, info))
                remote.send(out)
            elif cmd == "close":
                for env in envs:
                    env.close()
                remote.send(True)
                break
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        for env in envs:
            try:
                env.close()
            except Exception:
                pass


class SelfPlayVecEnv:
    def __init__(self, num_envs: int, env_kwargs: dict, base_seed: int = 0,
                 start_method: str | None = None, obs_shm: bool = False,
                 envs_per_worker: int = 1):
        self.num_envs = num_envs
        self._K = max(1, int(envs_per_worker))
        if self._K > 1 and not env_kwargs.get("native_encode"):
            raise ValueError("envs_per_worker > 1 requires native_encode (the JSON path still "
                             "drives the kaggle_environments module-global engine handle)")
        self._ctx = mp.get_context(start_method or "spawn")
        self._env_kwargs = env_kwargs
        self._base_seed = base_seed
        self._respawns = 0            # workers replaced after an engine-abort death (telemetry)
        self._send_failed = set()     # WORKER indices whose command send hit a dead pipe this barrier

        # worker w owns global env ids [starts[w], starts[w] + counts[w])
        self._starts = list(range(0, num_envs, self._K))
        self._counts = [min(self._K, num_envs - s) for s in self._starts]
        self.num_workers = len(self._starts)
        if self._K > 1:
            print(f"[vec] envs-per-worker={self._K}: {num_envs} envs in {self.num_workers} "
                  f"worker processes", flush=True)

        # Learner-obs shared-memory batch (see _worker): zero-copy read instead of pickle + np.stack.
        self._obs_shm = bool(obs_shm)
        self._shm = []
        self._obs_shms = [None] * self.num_workers
        if self._obs_shm:
            lspec = _ref_spec(env_kwargs)
            lnames = {}
            self._learner_obs_bufs = {}
            for k, (shp, dt) in lspec.items():
                nbytes = max(num_envs * int(np.prod(shp)) * np.dtype(dt).itemsize, 1)
                shm = shared_memory.SharedMemory(create=True, size=nbytes)
                self._shm.append(shm); lnames[k] = shm.name
                self._learner_obs_bufs[k] = np.ndarray((num_envs, *shp), dtype=dt, buffer=shm.buf)
            for w in range(self.num_workers):
                self._obs_shms[w] = {"idx0": self._starts[w], "n": num_envs,
                                     "names": lnames, "spec": lspec}

        self.remotes = [None] * self.num_workers
        self.procs = [None] * self.num_workers
        for w in range(self.num_workers):
            self._spawn(w, self._seeds_for(w, 0))

    def _seeds_for(self, w, respawns):
        """Per-env seed = base + global_env_id (+ num_envs * respawns on a respawn -- never replay
        the dead worker's deck-sampling sequence from the top). Identical to the 1-env-per-worker
        formula for K=1."""
        i0, c = self._starts[w], self._counts[w]
        return [self._base_seed + gid + self.num_envs * respawns for gid in range(i0, i0 + c)]

    def _spawn(self, w, seeds):
        ek = self._env_kwargs
        fo = ek.get("frozen_opp") if isinstance(ek, dict) else None
        if fo and (fo.get("wide_flags") is not None or fo.get("ks_flags") is not None):
            # FROZEN STREAMS: slice the global per-env flags down to this worker's K envs so
            # _build_env's local_ix indexes correctly (worker w owns gids starts[w]..+counts[w])
            i0, c = self._starts[w], self._counts[w]
            fo2 = dict(fo)
            for _k in ("wide_flags", "ks_flags"):
                if fo.get(_k) is not None:
                    fo2[_k] = list(fo[_k][i0:i0 + c])
            ek = dict(ek); ek["frozen_opp"] = fo2
        wr, r = self._ctx.Pipe()
        p = self._ctx.Process(target=_worker, args=(wr, r, ek, seeds, self._obs_shms[w]),
                              daemon=True)
        p.start()
        wr.close()
        self.remotes[w] = r
        self.procs[w] = p

    def _respawn(self, w):
        """Replace a dead worker (the engine's C++ 'invalid index' abort kills the whole process;
        Python in the worker cannot catch it) with a fresh process + its K envs, and return its
        first reset payload (a LIST). The dead worker's in-flight episodes are simply dropped. If
        the replacement dies during its own reset the EOFError propagates -- a systemic failure
        should still kill the leg (the sbatch wrapper restarts it)."""
        pid = self.procs[w].pid if self.procs[w] is not None else -1
        print(f"[vec] WORKER DIED: worker={w} envs={self._starts[w]}.."
              f"{self._starts[w] + self._counts[w] - 1} pid={pid} -> respawning "
              f"(forensics/worker_deck_{pid}.txt has its in-flight battles)", flush=True)
        _fd = os.environ.get("PKMN_DECK_FORENSICS_DIR")
        if _fd:
            try:
                with open(os.path.join(_fd, f"DEAD_{pid}.marker"), "w") as _f:
                    _f.write(f"worker={w} idx0={self._starts[w]} count={self._counts[w]} "
                             f"pid={pid} respawn={self._respawns + 1}\n")
            except Exception:
                pass
        try:
            self.remotes[w].close()
        except Exception:
            pass
        try:
            self.procs[w].join(timeout=5)
        except Exception:
            pass
        self._respawns += 1
        self._spawn(w, self._seeds_for(w, self._respawns))
        self.remotes[w].send(("reset", None))
        payload = self.remotes[w].recv()
        print(f"[vec] respawned worker {w} (total respawns: {self._respawns})", flush=True)
        return payload

    def _stack(self, obs_list):
        return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}

    def _recv_all(self, mode="step"):
        """recv from every worker (each yields a LIST of per-env payloads, flattened in env order);
        a dead worker (engine C++ 'invalid index' abort -> process SIGABRT -> pipe EOF) is
        RESPAWNED in place instead of killing the whole trainer (2026-07-12: root cause is an
        engine bug -- ToolCountProc's TR-energy eviction indexes the wrong player's energy list --
        reachable by pool decks we keep on purpose, user call). For a 'step' barrier the
        replacement's reset payloads are synthesized as neutral terminals (r=0, done=True, fresh
        first obs): the exact contract of the worker's own in-env recovery path, which the trainer
        already handles; the lost episodes are dropped."""
        out = []
        for w in range(self.num_workers):
            if w in self._send_failed:
                self._send_failed.discard(w)
                payload = self._respawn(w)
            else:
                try:
                    out.extend(self.remotes[w].recv())
                    continue
                except (EOFError, ConnectionResetError, OSError):
                    # EOFError = clean pipe close; ConnectionResetError = the abort raced the recv
                    payload = self._respawn(w)
            if mode == "reset":
                out.extend(payload)
                continue
            # step-mode: synthesize neutral terminals from the reset payloads, which are
            # (seat, info) under obs_shm and (obs, seat, info) otherwise -- the leading obs
            # (if any) passes through unchanged.
            for item in payload:
                *head, seat, rinfo = item
                info = {"respawned": True}
                if "agent_seat" in rinfo:
                    info["next_agent_seat"] = rinfo["agent_seat"]
                out.append((*head, int(seat), 0.0, True, info))
        return out

    def reset(self):
        for w, r in enumerate(self.remotes):
            try:
                r.send(("reset", None))
            except (BrokenPipeError, OSError):
                self._send_failed.add(w)
        if self._obs_shm:
            seats, infos = zip(*self._recv_all(mode="reset"))
            return dict(self._learner_obs_bufs), np.asarray(seats, dtype=np.int64), list(infos)
        obs, seats, infos = zip(*self._recv_all(mode="reset"))
        return self._stack(obs), np.asarray(seats, dtype=np.int64), list(infos)

    def step(self, actions):
        acts = np.asarray(actions).tolist()           # one bulk conversion, not num_envs int() calls
        for w in range(self.num_workers):
            i0, c = self._starts[w], self._counts[w]
            try:
                self.remotes[w].send(("step", acts[i0:i0 + c]))
            except (BrokenPipeError, OSError):
                self._send_failed.add(w)
        if self._obs_shm:
            seats, rews, dones, infos = zip(*self._recv_all())
            return (dict(self._learner_obs_bufs),
                    np.asarray(seats, dtype=np.int64),
                    np.asarray(rews, dtype=np.float32),
                    np.asarray(dones, dtype=np.bool_),
                    list(infos))
        obs, seats, rews, dones, infos = zip(*self._recv_all())
        return (self._stack(obs),
                np.asarray(seats, dtype=np.int64),
                np.asarray(rews, dtype=np.float32),
                np.asarray(dones, dtype=np.bool_),
                list(infos))

    def pin_obs_buffers(self):
        """Page-lock (cudaHostRegister) the learner-obs shm buffers so per-step H2D copies use DMA.
        Best-effort: no-op if not obs_shm / no CUDA. Call once after CUDA is initialized."""
        self._pinned = []
        if not self._obs_shm:
            return
        try:
            import torch
            if not torch.cuda.is_available():
                return
            cudart = torch.cuda.cudart()
            for arr in self._learner_obs_bufs.values():
                cudart.cudaHostRegister(arr.ctypes.data, arr.nbytes, 0)   # 0 = cudaHostRegisterDefault
                self._pinned.append(arr.ctypes.data)
            import sys; print(f"[pin-obs] page-locked {len(self._pinned)} shm obs buffers", file=sys.stderr, flush=True)
        except Exception as e:
            import sys; print(f"[pin-obs] skipped ({type(e).__name__}: {e})", file=sys.stderr, flush=True)
            self._pinned = []

    def close(self):
        for r in self.remotes:
            try:
                r.send(("close", None)); r.recv()
            except Exception:
                pass
        for _ptr in getattr(self, "_pinned", []):
            try:
                import torch; torch.cuda.cudart().cudaHostUnregister(_ptr)
            except Exception:
                pass
        for shm in self._shm:
            try:
                shm.close(); shm.unlink()
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
