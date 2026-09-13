"""Token-based Transformer actor-critic for the cabt action space (v2).

Additive sibling of ``rl/policy.py``. Consumes the token streams emitted by
``rl/encoding.TokenEncoder`` and exposes the SAME interface as the existing nets
(``logits_value`` / ``get_value`` / ``get_action_and_value``) so it is drop-in.

Design (matches encoding's streams):
  * one shared card embedding ``nn.Embedding(vocab_size+1, d_model, padding_idx=0)`` (used directly)
    -- index 0 = EMPTY/pad, index ``vocab_size`` = UNK (hidden card);
  * a per-token-type embedding (CLS, the card-list streams, the unit stream, the
    option stream) added to every token;
  * each card-list token = card_emb(id) + type_emb;
  * each in-play unit token = card_emb(top) + sum card_emb(preevo) + sum
    card_emb(tool) + unit_attr_proj(attr[23]) + type_emb;
  * each option token = card_emb(src) + card_emb(tgt) + opt_attr_proj(attr) +
    type_emb;
  * the CLS token = cls_param + scalar_proj(cls_scalars) + select-type/ctx emb;
  * one ``nn.TransformerEncoder`` over the whole sequence with a
    ``src_key_padding_mask`` built from each stream's pad mask (options use the
    legal-action mask). Value reads the CLS output; the policy pointer-scores the
    option-token outputs and concatenates a CLS-derived SUBMIT logit, then masks.
"""

from __future__ import annotations

import os

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

# b=1 padded-token truncation (exact; see _encode). Escape hatch for A/B benchmarking only.
_TRUNC_B1 = os.environ.get("V2_NO_TRUNC") != "1"

from .card_features import CardTable
from . import effect_data           # N_ATTACK_FX (opt_attr width) + per-card ability/trainer multi-hots (static matrix)
from .encoding import (
    MAX_OPTIONS, N_ACTIONS, OPT_STRUCT, OPT_PICKED, N_OPT_TYPES, MAX_ATTACK,
    N_SELECT_TYPES, N_SELECT_CTX,
    UNIT_ATTR, G,
)
from .enc_constants import _TOKEN_LAYOUT, N_STATE_TOKENS

# int64 id tensors (everything else float32). Must match TokenEncoder.int_keys.
_INT_KEYS = {
    "select_type", "select_context", "effect_id",
    "self_deck_id", "opp_deck_id",
    "self_prize_id", "opp_prize_id",
    "self_hand_id", "opp_hand_id",
    "self_discard_id", "opp_discard_id",
    "stadium_id",
    "self_unit_top_id", "self_unit_preevo_id", "self_unit_tool_id", "self_unit_energy_id",
    "opp_unit_top_id", "opp_unit_preevo_id", "opp_unit_tool_id", "opp_unit_energy_id",
    "opt_src_pos", "opt_tgt_pos",
    "opt_src_card", "opt_tgt_card",
    "opt_verb", "opt_attack_id",
}

# Token-type ids for the type/zone embedding. Each card-list STREAM gets its own
# type so the net can tell a hand card from a discard card etc.
(_T_CLS,
 _T_SELF_DECK, _T_OPP_DECK,
 _T_SELF_PRIZE, _T_OPP_PRIZE,
 _T_SELF_HAND, _T_OPP_HAND,
 _T_SELF_DISC, _T_OPP_DISC,
 _T_STADIUM,
 _T_SELF_ACTIVE, _T_SELF_BENCH,
 _T_OPP_ACTIVE, _T_OPP_BENCH,
 _T_OPT, _T_EFFECT,
 _T_SEL_TYPE, _T_SEL_CTX) = range(18)   # select-type / select-context now their own tokens (off CLS)
_T_CARD_SYNTH = 18   # synthesized card token (attached energy/tool, deck-search pick) -- APPENDED so
                     # ids 0..17 are unchanged; gives synth tokens the zone marker real cards have
_N_TTYPES = 19

N_SCRATCH = 4        # hardcoded register/workspace tokens (cf. "ViTs Need Registers"); permanent arch feature

# (key, type-id) for the flat card-list streams (id + matching *_mask).
_CARD_STREAMS = [
    ("self_deck", _T_SELF_DECK), ("opp_deck", _T_OPP_DECK),
    ("self_prize", _T_SELF_PRIZE), ("opp_prize", _T_OPP_PRIZE),
    ("self_hand", _T_SELF_HAND), ("opp_hand", _T_OPP_HAND),
    ("self_discard", _T_SELF_DISC), ("opp_discard", _T_OPP_DISC),
    ("stadium", _T_STADIUM),
    ("effect", _T_EFFECT),       # source-effect tokens: the card driving this select + contextCard
]


_DEFAULT_OPT_BUCKETS = (32, 64, 128, MAX_OPTIONS)   # keep 128: the pre-v2.2 cap stays the common
                                                    # worst case; only a real >128 menu pays 192
_OPT_TRUNC_ON = os.environ.get("OPT_TRUNC", "1") != "0"    # OPT_TRUNC=0 -> full 128 (A/B baseline)


def bucket_for(m: int, buckets=_DEFAULT_OPT_BUCKETS) -> int:
    """Smallest bucket >= m (MAX_OPTIONS when trunc is off or m exceeds every bucket). The ONE
    bucket-rounding rule -- shared by bucketed_opt_len / bucketed_opt_len_np / the update loop's
    per-row precompute, so compiled-graph shapes can never drift between the three."""
    if not _OPT_TRUNC_ON:
        return MAX_OPTIONS
    for b in buckets:
        if m <= b:
            return b
    return MAX_OPTIONS


def int32_safe_rows(nhead: int, worst_seq: int = 540) -> int:
    """Padded-SDPA int32 index cap: rows x nhead x seq^2 < 2^31 (with 5%% headroom). Governs the
    update-chunk size and the teacher-prepass stride on the NON-varlen path; under --varlen-attn
    the NJT flash kernels never build the padded index and the cap is physically irrelevant (the
    chunking then just degenerates to 1 chunk at current shapes)."""
    return max(1, int(0.95 * 2**31) // (nhead * worst_seq * worst_seq))


def bucketed_opt_len(action_mask, opt_attr=None, buckets=_DEFAULT_OPT_BUCKETS) -> int:
    """Smallest bucket >= the furthest PRESENT option index+1 across the batch, for b>1 option-token
    truncation. `action_mask`: [..., N_ACTIONS]. Present = legal OR already-picked (pass `opt_attr`
    so visible-but-illegal picked slots are never truncated away). Option slots beyond the returned
    length are present for NO row, so dropping their (already-masked) tokens before the encoder is
    EXACT. Bucketing keeps the number of distinct sliced shapes small (torch.compile stays
    specialized). One GPU->CPU sync; call OUTSIDE any torch.compile'd region and pass the result in
    as `opt_len`."""
    if not _OPT_TRUNC_ON:
        return MAX_OPTIONS
    present = action_mask[..., :MAX_OPTIONS] > 0.5
    if opt_attr is not None:                       # include already-picked (visible-but-illegal) slots
        present = present | (opt_attr[..., OPT_PICKED] > 0.5)
    cols = present.reshape(-1, present.shape[-1]).any(dim=0)   # [MAX_OPTIONS] present in ANY row
    idx = torch.nonzero(cols)
    if idx.numel() == 0:
        return buckets[0]
    return bucket_for(int(idx.max().item()) + 1, buckets)


def bucketed_opt_len_np(action_mask, opt_attr=None, buckets=_DEFAULT_OPT_BUCKETS) -> int:
    """bucketed_opt_len computed from the NUMPY obs (identical result, same buckets/switch) --
    for the collection loop, where the numpy batch is in hand BEFORE the H2D copy: computing the
    bucket CPU-side removes a per-step GPU->CPU sync (256+ per rollout)."""
    if not _OPT_TRUNC_ON:
        return MAX_OPTIONS
    present = np.asarray(action_mask)[..., :MAX_OPTIONS] > 0.5
    if opt_attr is not None:
        present = present | (np.asarray(opt_attr)[..., OPT_PICKED] > 0.5)
    idx = np.flatnonzero(present.reshape(-1, present.shape[-1]).any(axis=0))
    if idx.size == 0:
        return buckets[0]
    return bucket_for(int(idx[-1]) + 1, buckets)


# ---- state-token truncation (batch-max, GATHER-THEN-SLICE) ---------------------------------
# The option src/tgt gather in _encode runs on the FULL state sequence; the kept-token slice is
# applied AFTER it, so absolute option positions never need remapping (the failure class of the
# old split_heads +2 bug cannot occur). Dropping a state token is EXACT when it is padded in
# EVERY row of the batch: the encoder has no positional encoding (type embeddings only), masked
# keys get exactly-zero attention weight, and the mean-pool already excludes pads -- the same
# argument (and the same last-present-index semantics) as the option truncation above. Streams
# can contain HOLES (a face-down bench Pokemon is an empty slot mid-list), which last-present
# handles and a present-count would not.
_STATE_TRUNC_PAD = 32     # kept length rounded up to this (bounds compile/cudagraph shape count)
_STREAM_MASK_KEY = {"self_units": "self_unit_mask", "opp_units": "opp_unit_mask"}


def state_keep_np(obs, split_heads: bool, pad_to: int = _STATE_TRUNC_PAD):
    """Batch-uniform keep-list over the FINAL state sequence (i.e. AFTER the split_heads
    value/submit insertion at positions 1/2): header + each stream's first `last-present+1`
    positions, padded up to a multiple of `pad_to` with dropped (all-rows-padded) positions as
    inert fillers so distinct compiled shapes stay few. Accepts the numpy obs batch (collection,
    pre-H2D -> no GPU sync) OR torch tensors (update path -> one small D2H). Returns int64
    positions, or None when nothing can be dropped (caller then skips slicing entirely)."""
    off = 2 if split_heads else 0
    torch_in = torch.is_tensor(next(iter(obs.values())))
    cols_by = {}
    if torch_in:                                   # one concatenated D2H for all streams
        chunks, names = [], []
        for name, size in _TOKEN_LAYOUT:
            mkey = _STREAM_MASK_KEY.get(name, f"{name}_mask")
            if mkey in obs:
                chunks.append((obs[mkey] > 0.5).reshape(-1, size).any(dim=0))
                names.append(name)
        flat = torch.cat(chunks).cpu().numpy()
        p = 0
        for name, size in _TOKEN_LAYOUT:
            if name in names:
                cols_by[name] = flat[p:p + size]; p += size
    else:
        for name, size in _TOKEN_LAYOUT:
            mkey = _STREAM_MASK_KEY.get(name, f"{name}_mask")
            if mkey in obs:
                cols_by[name] = (np.asarray(obs[mkey]) > 0.5).reshape(-1, size).any(axis=0)
    keep = [0] + ([1, 2] if split_heads else [])   # cls (+ value/submit): never padded
    drop = []
    pos = 1                                        # _TOKEN_LAYOUT coords (pre-insertion)
    for name, size in _TOKEN_LAYOUT[1:]:
        base = pos + off
        if name not in cols_by:                    # sel_type / sel_ctx: single, never padded
            keep.extend(range(base, base + size))
        else:
            nz = np.flatnonzero(cols_by[name])
            last = int(nz[-1]) + 1 if nz.size else 0
            keep.extend(range(base, base + last))
            drop.extend(range(base + last, base + size))
        pos += size
    if not drop:
        return None
    n_full = N_STATE_TOKENS + off
    target = min(n_full, -(-len(keep) // pad_to) * pad_to)
    keep.extend(drop[:max(0, target - len(keep))])  # fillers: padded in EVERY row -> inert
    return np.asarray(keep, dtype=np.int64)


def obs_to_tensors(obs: dict, device) -> dict:
    """encoding stack-of-arrays obs -> torch tensors on device (no batch dim)."""
    out = {}
    for k, v in obs.items():
        if k in _INT_KEYS:
            out[k] = torch.as_tensor(np.asarray(v), dtype=torch.long, device=device)
        else:
            out[k] = torch.as_tensor(np.asarray(v), dtype=torch.float32, device=device)
    return out


def _njt_qkv(l, x, D, H, hd):
    """FLAT segment of the varlen layer (module-level so torch.compile guards on the layer
    identity give one stable graph per layer): packed in_proj -> per-head q/k/v on [TOT, D]."""
    w, bias = l.self_attn.in_proj_weight, l.self_attn.in_proj_bias
    q = nn.functional.linear(x, w[:D], bias[:D]).view(-1, H, hd)
    k = nn.functional.linear(x, w[D:2 * D], bias[D:2 * D]).view(-1, H, hd)
    v = nn.functional.linear(x, w[2 * D:], bias[2 * D:]).view(-1, H, hd)
    return q, k, v


def _njt_post(l, x, a):
    """FLAT post-attention segment: out_proj + post-norm residual FFN (TransformerEncoderLayer
    norm_first=False, dropout=0 semantics), all on [TOT, D] regular tensors."""
    sa = nn.functional.linear(a, l.self_attn.out_proj.weight, l.self_attn.out_proj.bias)
    x = l.norm1(x + sa)
    return l.norm2(x + l.linear2(l.activation(l.linear1(x))))


class TokenTransformer(nn.Module):
    """Token-set state -> Transformer encoder -> pointer policy + CLS value.

    Interface: logits_value / get_value / get_action_and_value.
    """

    def __init__(self, vocab_size: int, d_model: int = 128,
                 nhead: int = 4, nlayers: int = 2, dropout: float = 0.0,
                 card_feat=None, structured: bool = False,
                 split_heads: bool = False,
                 value_categorical: bool = False, value_atoms: int = 51, value_vmax: float = 1.0,
                 opt_struct: int = OPT_STRUCT + effect_data.N_ATTACK_FX):
        super().__init__()
        self.d = d_model
        self.UNK = vocab_size                                  # hidden-card id index
        # +1 so index `vocab_size` (UNK) is a real, learnable row; 0 stays pad. Card identity is a
        # full d_model embedding used DIRECTLY -- no bottleneck projection (the emb-dim A/B showed the
        # full-width embedding beats the old 48->d Linear at maturity).
        self.card_emb = nn.Embedding(vocab_size + 1, d_model, padding_idx=0)
        # frozen STATIC per-card features (printed/constant: type/weakness/stage/base-maxHp/cost/...),
        # gathered by id. Complements the LEARNED card_emb AND the engine's DYNAMIC per-instance attrs
        # (effective in-play maxHp/energy/conditions live in unit_attr; here it is the BASE printed maxHp).
        # UNK row (index vocab_size) = zeros. card_feat=None -> disabled (old emb-only behavior).
        if card_feat is not None:
            _f = np.zeros((vocab_size + 1, card_feat.shape[1]), dtype=np.float32)
            _f[:card_feat.shape[0]] = card_feat
            self.register_buffer("card_feat", torch.from_numpy(_f), persistent=False)
            self.static_proj = nn.Linear(card_feat.shape[1], d_model)
        else:
            self.card_feat = None
            self.static_proj = None
        # STRUCTURED action head: verb-conditioned option scoring. The option's OptionType
        # (verb: PLAY/ATTACH/EVOLVE/ATTACK/ABILITY/...) is the first 16 dims of opt_attr; give
        # each verb its own scoring query+bias so ATTACH options share a scoring direction
        # (the action-semantics analogue of static card features). Shared opt_head kept as a
        # fallback for rare verbs. None -> plain pointer (old behavior).
        self.structured = structured
        if structured:
            self.type_query = nn.Embedding(N_OPT_TYPES, d_model)  # OptionType (0..16) -> scoring query
            self.type_bias = nn.Embedding(N_OPT_TYPES, 1)
        self.type_emb = nn.Embedding(_N_TTYPES, d_model)
        self.sel_type_emb = nn.Embedding(N_SELECT_TYPES, d_model)
        self.sel_ctx_emb = nn.Embedding(N_SELECT_CTX, d_model)
        self.cls = nn.Parameter(torch.zeros(d_model))
        # split_heads: give SUBMIT and VALUE their OWN dedicated tokens (each a learned query) instead
        # of both reading the shared CLS. Each attends over the whole board through the encoder and
        # specializes -- value is no longer a compromise with submit, and its token's deep learned
        # aggregation subsumes the mean-pool. CLS stays as the scalar-carrying CONTEXT token the
        # others attend to. Old ckpts lack these params -> split_heads OFF keeps them load-compatible.
        self.split_heads = split_heads
        if split_heads:
            self.value_tok = nn.Parameter(torch.zeros(d_model))   # dedicated value query
            self.submit_tok = nn.Parameter(torch.zeros(d_model))  # dedicated submit query
        # register / scratch tokens: learned tokens with NO input and NO head reading them -- a shared
        # global workspace the self-attention can use to compute/aggregate without hijacking real tokens
        # (cf. "Vision Transformers Need Registers"). HARDCODED always-on (N_SCRATCH), a permanent part
        # of the architecture, not a knob. Inserted between the state and option tokens so cls/value/
        # submit stay at indices 0/1/2 and option-gather positions are unaffected. Never padded.
        self.scratch_tokens = N_SCRATCH
        self.scratch = nn.Parameter(torch.zeros(N_SCRATCH, d_model))

        self.unit_attr_proj = nn.Linear(UNIT_ATTR, d_model)    # unit attr
        # v2.1 DRAWABLE marker: added to a self_deck token iff its decklist copy is still unseen
        # (in deck or face-down in our prizes) -- self_deck_flag. Zero-init -> no-op until learned.
        self.drawable_emb = nn.Parameter(torch.zeros(d_model))
        # v2.3 KNOWN-TOP-OF-DECK depth marker (self deck stream only): 1.0 top / .66 / .33.
        # Zero-init -> exact no-op until learned, so every pre-v2.3 checkpoint loads and behaves
        # identically (load with strict=False; the param simply starts at zero).
        self.decktop_emb = nn.Parameter(torch.zeros(d_model))
        # v2.1 opp HIDDEN marker (separate param -- belief-quality, different semantics): added to
        # an opp_deck token iff that revealed copy is not currently visible (plausibly still to
        # come) or the slot is unrevealed UNK -- opp_deck_flag.
        self.opp_drawable_emb = nn.Parameter(torch.zeros(d_model))
        # v2.2 opp-hand CERTAINTY marker: added to a KNOWN opp_hand token iff its belief is airtight
        # (no identity-redacted hand-exit observed since it was stamped) -- opp_hand_flag. UNK and
        # pad slots carry 0. Only the two deck streams + opp_hand carry a flag input.
        self.hand_certain_emb = nn.Parameter(torch.zeros(d_model))
        # An option's source/target each = the EXACT pre-encoder token of the card/unit it points at
        # (gathered by position in _encode), through its own d->d projection. Distinct maps so src vs
        # tgt stay separable. That gathered token already carries the card's identity/features and,
        # for board refs, the full unit state; opt_attr carries the per-option structural distinctions.
        self.opt_src_proj = nn.Linear(d_model, d_model)
        self.opt_tgt_proj = nn.Linear(d_model, d_model)
        self.opt_attr_proj = nn.Linear(opt_struct, d_model)    # option structural feats (no verb one-hot); opt_struct>OPT_STRUCT in the fx fork
        self.opt_verb_emb = nn.Embedding(N_OPT_TYPES, d_model)  # OptionType (verb) -> its own learned vector
        self.attack_emb = nn.Embedding(MAX_ATTACK, d_model, padding_idx=0)  # attackId -> learned identity (0=no-attack, zero)
        # (the acting/acted-on Pokemon's state now enters each option as that unit's FULL token,
        #  gathered by slot in _encode -- no dedicated projection needed.)
        self.scalar_proj = nn.Linear(G, d_model)               # CLS scalars

        ff = 4 * d_model                                       # FFN width = 4x d_model (conventional)
        layer = nn.TransformerEncoderLayer(d_model, nhead, ff, batch_first=True, dropout=dropout)
        # enable_nested_tensor=False: the nested-tensor fast path engages only under eval/no_grad (and
        # with a padding mask), so it makes the SAME net compute slightly differently between rollout
        # collection (no_grad) and the PPO update (grad) -> a biased importance ratio; and between the
        # eval teacher and the train student -> a spurious distillation gradient. Forcing the standard
        # path makes collection == update == teacher numerically identical.
        self.encoder = nn.TransformerEncoder(layer, nlayers, enable_nested_tensor=False)
        # VARLEN attention (2026-07-10, opt-in via trainer --varlen-attn): run the SAME encoder
        # weights over per-row PACKED tokens (jagged NJT + flash SDPA) instead of padded+mask.
        # The batched generalization of the exact b=1 truncation in _encode (no positional
        # encoding -> dropping pad tokens is EXACT); escapes the key-padding-mask kernel fallback
        # (~2.2x attn fwd+bwd) AND the rows*heads*seq^2 int32 cap. The trainer sets this on net,
        # collect_net AND teacher_net together (collection == update == teacher stays one path).
        self.varlen_attn = False
        self.varlen_compile = False    # + inductor on the FLAT segments (see _encoder_njt)
        self._vlc_qkv = None
        self._vlc_post = None

        self.opt_head = nn.Linear(d_model, 1)
        self.submit_head = nn.Linear(d_model, 1)               # reads submit-token (split) or CLS (else)
        # value-head INPUT width: the value token alone (split) or CLS ++ mean-pool (base; a single
        # token under-summarizes the board, and value is the demonstrated LB ceiling).
        v_in = d_model if split_heads else 2 * d_model
        # CATEGORICAL (distributional) value: instead of a scalar regressed with MSE, emit logits over a
        # fixed atom support and train with cross-entropy on the (two-hot-projected) return. Classification
        # value heads are better-conditioned under noisy / non-stationary targets ("Stop Regressing",
        # Farebrother 2023) -- aimed at the generalist's value REGRESSION. get_value still returns the
        # scalar expectation, so GAE / the rest of the stack is unchanged. value_vmax bounds the support
        # (raise it above 1.0 if a graded terminal reward pushes returns past +/-1).
        self.value_categorical = bool(value_categorical)
        if self.value_categorical:
            self.value_atoms = int(value_atoms)
            self.value_vmax = float(value_vmax)
            self.value_head = nn.Linear(v_in, self.value_atoms)
            self.register_buffer("atom_support",
                                 torch.linspace(-self.value_vmax, self.value_vmax, self.value_atoms),
                                 persistent=False)
        else:
            self.value_head = nn.Linear(v_in, 1)

    # -- token builders -----------------------------------------------------
    def _type(self, B, K, t, device):
        return self.type_emb(torch.full((B, K), t, dtype=torch.long, device=device))

    def _static(self, ids):
        """static_proj(card_feat[ids]) per card token; 0 when static features are disabled."""
        if self.static_proj is None:
            return 0
        return self.static_proj(self.card_feat[ids])

    def _card_stream(self, ids, t, device):
        """ids [B,K] -> token [B,K,d] = card_emb(id) + type_emb + static."""
        B, K = ids.shape
        return self.card_emb(ids) + self._type(B, K, t, device) + self._static(ids)

    def _unit_stream(self, top_id, preevo_id, tool_id, energy_id, attr, active_t, bench_t, device):
        """unit tokens [B,U,d] = (emb(top)+sum emb(preevo)+sum emb(tool)+sum emb(energy))
        + unit_attr_proj(attr) + type_emb. Row 0 = ACTIVE (own type), rows 1.. = BENCH
        (so the net distinguishes the battling Pokemon from the bench; bench stays symmetric).
        The energy id-bag carries SPECIAL-energy identity (Double Turbo/Team Rocket/...),
        complementing the color histogram + count already in attr."""
        B, U = top_id.shape
        idbag = (self.card_emb(top_id)
                 + self.card_emb(preevo_id).sum(dim=2)
                 + self.card_emb(tool_id).sum(dim=2)
                 + self.card_emb(energy_id).sum(dim=2))         # [B,U,d]
        types = torch.full((B, U), bench_t, dtype=torch.long, device=device)
        types[:, 0] = active_t                                  # slot 0 is the Active Pokemon
        return (idbag + self.unit_attr_proj(attr)
                + self.type_emb(types) + self._static(top_id))

    def _opt_stream(self, src_tok, tgt_tok, attr, verb, attack_id, device):
        """Option tokens [B,K,d]. src_tok/tgt_tok ARE the gathered pre-encoder tokens of the
        card/unit each option references as source/target (its view IS that exact token), each
        through its own projection; summed with the per-option structural attr, a learned
        per-OptionType (verb) embedding, a learned per-ATTACK embedding (attack_emb; 0=no-attack ->
        zero via padding_idx, so it distinguishes multiple attacks of the same Pokemon), and the
        option-type-token marker. Card identity/features/unit-state all live in the gathered tokens."""
        B, K = attr.shape[0], attr.shape[1]
        return (self.opt_src_proj(src_tok) + self.opt_tgt_proj(tgt_tok)
                + self.opt_attr_proj(attr) + self.opt_verb_emb(verb)
                + self.attack_emb(attack_id)
                + self._type(B, K, _T_OPT, device))

    # -- core ---------------------------------------------------------------
    def _encode(self, o: dict, opt_len: int | None = None, state_keep=None):
        dev = o["cls_scalars"].device
        B = o["cls_scalars"].shape[0]
        toks, pads = [], []
        _nopad = lambda: pads.append(torch.zeros(B, 1, dtype=torch.bool, device=dev))

        # CLS token = global scalars. Without split_heads its output feeds value + submit directly;
        # with split_heads it is the scalar-carrying CONTEXT token the dedicated heads attend to.
        # Never padded.
        toks.append(self.cls.expand(B, 1, self.d)
                    + self.scalar_proj(o["cls_scalars"]).unsqueeze(1)
                    + self._type(B, 1, _T_CLS, dev)); _nopad()
        # dedicated VALUE (index 1) + SUBMIT (index 2) tokens: each a learned query, marked as a
        # summary token (_T_CLS); value also gets the scalars directly (prize counts ARE the value
        # signal -- a short residual path on top of its board attention). Never padded.
        if self.split_heads:
            toks.append(self.value_tok.expand(B, 1, self.d)
                        + self.scalar_proj(o["cls_scalars"]).unsqueeze(1)
                        + self._type(B, 1, _T_CLS, dev)); _nopad()
            toks.append(self.submit_tok.expand(B, 1, self.d)
                        + self._type(B, 1, _T_CLS, dev)); _nopad()
        # select-TYPE (kind of decision) and select-CONTEXT (its purpose) as SEPARATE tokens:
        # each = its categorical embedding + a token-type marker. The transformer broadcasts this
        # framing to every option via attention. Never padded.
        toks.append(self.sel_type_emb(o["select_type"].squeeze(-1)).unsqueeze(1)
                    + self._type(B, 1, _T_SEL_TYPE, dev)); _nopad()
        toks.append(self.sel_ctx_emb(o["select_context"].squeeze(-1)).unsqueeze(1)
                    + self._type(B, 1, _T_SEL_CTX, dev)); _nopad()

        # flat card-list streams (pad mask from each stream's *_mask)
        for name, t in _CARD_STREAMS:
            tok = self._card_stream(o[f"{name}_id"], t, dev)
            if name == "self_deck":                 # v2.1 drawable marker (deck streams ONLY)
                tok = tok + o["self_deck_flag"].unsqueeze(-1) * self.drawable_emb
                if "self_deck_top" in o:            # v2.3 known-top depth (absent on old obs)
                    tok = tok + o["self_deck_top"].unsqueeze(-1) * self.decktop_emb
            elif name == "opp_deck":                # v2.1 hidden/still-to-come marker (belief)
                tok = tok + o["opp_deck_flag"].unsqueeze(-1) * self.opp_drawable_emb
            elif name == "opp_hand":                # v2.2 certainty marker (known-belief tokens)
                tok = tok + o["opp_hand_flag"].unsqueeze(-1) * self.hand_certain_emb
            toks.append(tok)
            pads.append(o[f"{name}_mask"] < 0.5)

        # in-play unit streams (active + bench), both sides
        for side, (at, bt) in (("self", (_T_SELF_ACTIVE, _T_SELF_BENCH)),
                               ("opp", (_T_OPP_ACTIVE, _T_OPP_BENCH))):
            toks.append(self._unit_stream(
                o[f"{side}_unit_top_id"], o[f"{side}_unit_preevo_id"],
                o[f"{side}_unit_tool_id"], o[f"{side}_unit_energy_id"],
                o[f"{side}_unit_attr"], at, bt, dev))
            pads.append(o[f"{side}_unit_mask"] < 0.5)

        # all STATE tokens so far (order matches encoding._TOKEN_LAYOUT); options index into these.
        state_seq = torch.cat(toks, dim=1)                     # [B, N_STATE, d]

        # option tokens: each gathers the EXACT pre-encoder token of the card/unit it references as
        # source/target (global position from encode; -1 -> zero), so the option's view IS that token.
        # A referenced card with NO token (an attached energy/tool) instead carries its id, from which
        # we SYNTHESIZE that card's token (same form as a card-list token: emb + static).
        def _resolve(pos, card):
            safe = pos.clamp(min=0)
            tok = torch.gather(state_seq, 1, safe.unsqueeze(-1).expand(-1, -1, self.d))
            tok = tok * (pos >= 0).unsqueeze(-1).to(tok.dtype)
            need = ((pos < 0) & (card > 0)).unsqueeze(-1).to(tok.dtype)
            synth = (self.card_emb(card) + self._static(card)
                     + self._type(card.shape[0], card.shape[1], _T_CARD_SYNTH, card.device))
            return tok + need * synth
        # split_heads inserts value_tok(idx1)+submit_tok(idx2) into state_seq BEFORE sel_type, which
        # shifts every state token from sel_type onward by +2 vs encoding._OFF -- and _OFF is exactly
        # what builds opt_src_pos/opt_tgt_pos (it reserves NO slot for value/submit). Compensate here so
        # each option gathers the intended card/unit token; -1 (no board token -> synth path) is left as-is.
        _shift = 2 if self.split_heads else 0     # == #tokens inserted between cls and sel_type
        _src_pos = torch.where(o["opt_src_pos"] >= 0, o["opt_src_pos"] + _shift, o["opt_src_pos"])
        _tgt_pos = torch.where(o["opt_tgt_pos"] >= 0, o["opt_tgt_pos"] + _shift, o["opt_tgt_pos"])
        opt_tok = self._opt_stream(_resolve(_src_pos, o["opt_src_card"]),
                                   _resolve(_tgt_pos, o["opt_tgt_card"]),
                                   o["opt_attr"], o["opt_verb"], o["opt_attack_id"], dev)
        # present = legal OR already-picked (v2.1): a picked option stays ATTENDED (its token carries
        # what's in the buffered multi-select set) while build_mask keeps it illegal to re-pick.
        opt_present = ((o["action_mask"][..., :MAX_OPTIONS] > 0.5)
                       | (o["opt_attr"][..., OPT_PICKED] > 0.5))
        if opt_len is not None and opt_len < opt_tok.shape[1]:   # b>1 batch-max option truncation:
            opt_tok = opt_tok[:, :opt_len]                        # drop trailing option slots legal for
            opt_present = opt_present[:, :opt_len]                # NO row (masked keys -> EXACT, not approx)
        n_opt = opt_tok.shape[1]

        # batch-max STATE truncation (--state-trunc): keep only `state_keep` positions. Runs AFTER
        # the option gather (which used the full state_seq -> absolute positions stay valid), and
        # every dropped position is padded in EVERY row (see state_keep_np) -> EXACT, same argument
        # as the option truncation. Header (cls/value/submit) is always keep[0:3] in order, so the
        # enc[:, 0/1/2] reads below stay correct; filler positions carry their pad=True bits.
        pad_state = torch.cat(pads, dim=1)
        if state_keep is not None:
            state_seq = state_seq.index_select(1, state_keep)
            pad_state = pad_state.index_select(1, state_keep)

        # scratch/register tokens go BETWEEN the state tokens and the option tokens: this keeps the
        # option-gather positions (which index into state_seq) and the trailing `-n_opt:` option slice
        # both correct, and keeps cls/value/submit at indices 0/1/2. Never padded.
        if self.scratch_tokens:
            scr = (self.scratch.unsqueeze(0).expand(B, self.scratch_tokens, self.d)
                   + self._type(B, self.scratch_tokens, _T_CLS, dev))
            seq = torch.cat([state_seq, scr, opt_tok], dim=1)
            pad = torch.cat([pad_state,
                             torch.zeros(B, self.scratch_tokens, dtype=torch.bool, device=dev),
                             ~opt_present], dim=1)
        else:
            seq = torch.cat([state_seq, opt_tok], dim=1)
            pad = torch.cat([pad_state, ~opt_present], dim=1)
        # b=1 (the per-worker opponent + any single-obs inference) is dominated by the
        # O(seq^2) attention over a mostly-PADDED sequence (128 option slots, usually
        # <20 legal). The encoder has no positional encoding (only type embeddings), so
        # attention is permutation-invariant in the keys -> dropping padded tokens before
        # the encoder is EXACT, not an approximation. Cuts the CPU forward several-fold.
        if seq.shape[0] == 1 and _TRUNC_B1:
            keep = (~pad[0]).nonzero(as_tuple=False).squeeze(1)   # present token indices
            enc_t = self.encoder(seq[:, keep])                    # no pad mask: all present
            enc = seq.new_zeros(seq.shape)
            enc[:, keep] = enc_t                                  # scatter back; padded slots
            #                                                       stay 0 but get action-masked
            extra = (enc[:, 1], enc[:, 2]) if self.split_heads else None  # value_out, submit_out (never padded -> kept)
            return enc[:, 0], enc[:, -n_opt:], enc_t.mean(dim=1), extra  # pooled = mean over PRESENT tokens
        # A fully-padded row would NaN the softmax inside attention. The CLS token
        # is never padded, so no row is all-pad -> safe. (Belt-and-braces: encoder
        # rows still attend to CLS.)
        if self.varlen_attn and seq.is_cuda:
            enc = self._encoder_njt(seq, pad)
        else:
            enc = self.encoder(seq, src_key_padding_mask=pad)
        cls_out = enc[:, 0]                    # CLS / context
        opt_out = enc[:, -n_opt:]              # the trailing option tokens
        present = (~pad).unsqueeze(-1).to(enc.dtype)              # [B,K,1] mean-pool over non-pad tokens
        pooled = (enc * present).sum(dim=1) / present.sum(dim=1).clamp(min=1.0)
        extra = (enc[:, 1], enc[:, 2]) if self.split_heads else None  # (value_out, submit_out)
        return cls_out, opt_out, pooled, extra

    def _encoder_njt(self, seq, pad):
        """Jagged (varlen) encoder forward: pack each row's PRESENT tokens, run the encoder's OWN
        layer weights manually (packed-qkv linears -> jagged flash SDPA -> out_proj -> post-norm
        FFN, exactly nn.TransformerEncoderLayer's norm_first=False, dropout=0 semantics), scatter
        the outputs back to the padded grid. Pad slots come back 0 (like the b=1 path) and are
        never read: cls/value/submit are always present, absent option slots are action-masked,
        and the mean-pool weights by ~pad. Per-token ops (linears/norms/FFN) run on the flat
        [TOT, D] values; NJT exists only around SDPA (the composition the A100 probe validated).
        FLAT-SEGMENT COMPILE (varlen_compile, 2026-07-10): dynamo cannot trace the NJT ops, but
        the qkv projections and the post-attention block are plain [TOT, D] tensor programs ->
        compiled with dynamic=True (one graph per layer-module identity) while the jagged
        construction + SDPA stay eager. Restores inductor fusion for the FFN-dominated FLOPs in
        BOTH the update and the collect forward. Transient dynamo thread-race errors (async
        collector vs main-thread tracing) fall back to the eager twins, same as a_lv_safe."""
        B, S, D = seq.shape
        keep = ~pad
        lens = keep.sum(1)
        offs = nn.functional.pad(lens.cumsum(0), (1, 0))
        x = seq[keep]                                        # [TOT, D]
        H = self.encoder.layers[0].self_attn.num_heads
        hd = D // H
        qkv_fn, post_fn = _njt_qkv, _njt_post
        if self.varlen_compile and seq.is_cuda:
            if self._vlc_qkv is None:
                self._vlc_qkv = torch.compile(_njt_qkv, dynamic=True)
                self._vlc_post = torch.compile(_njt_post, dynamic=True)
            qkv_fn, post_fn = self._vlc_qkv, self._vlc_post
        for l in self.encoder.layers:
            if self.training and (l.dropout1.p > 0 or l.self_attn.dropout > 0):
                raise RuntimeError("varlen_attn assumes dropout=0 (matches the production config)")
            try:
                q, k, v = qkv_fn(l, x, D, H, hd)
            except RuntimeError as e:
                if "symbolically trace" not in str(e):
                    raise
                q, k, v = _njt_qkv(l, x, D, H, hd)
            jq = torch.nested.nested_tensor_from_jagged(q, offsets=offs).transpose(1, 2)
            jk = torch.nested.nested_tensor_from_jagged(k, offsets=offs).transpose(1, 2)
            jv = torch.nested.nested_tensor_from_jagged(v, offsets=offs).transpose(1, 2)
            a = nn.functional.scaled_dot_product_attention(jq, jk, jv)
            a = a.transpose(1, 2).values().reshape(-1, D)    # back to flat per-token
            try:
                x = post_fn(l, x, a)
            except RuntimeError as e:
                if "symbolically trace" not in str(e):
                    raise
                x = _njt_post(l, x, a)
        enc = seq.new_zeros(B, S, D)
        enc[keep] = x
        return enc

    def _value_input(self, cls_out, pooled, extra):
        """The tensor fed to value_head: the dedicated value token (split) or CLS ++ mean-pool (base)."""
        if self.split_heads:
            return extra[0]
        return torch.cat([cls_out, pooled], dim=-1)

    def _value_out(self, vin):
        """-> (value_scalar, value_logits). value_logits is None for the scalar head; for the
        categorical head it is the per-atom logits (for the CE loss) and the scalar is the
        softmax-weighted atom expectation (the API GAE / inference consume)."""
        if self.value_categorical:
            vlogits = self.value_head(vin)                      # [B, atoms]
            probs = torch.softmax(vlogits, dim=-1)
            return (probs * self.atom_support).sum(-1), vlogits
        return self.value_head(vin).squeeze(-1), None

    def logits_value(self, o: dict, return_vdist: bool = False, opt_len: int | None = None,
                     state_keep=None):
        cls_out, opt_out, pooled, extra = self._encode(o, opt_len=opt_len, state_keep=state_keep)
        n = opt_out.shape[1]                                   # == opt_len if truncated, else MAX_OPTIONS
        if self.structured:                                    # verb-conditioned scoring + shared fallback
            verb = o["opt_verb"][:, :n]                         # [B, n] OptionType per (kept) option
            opt_logits = ((opt_out * self.type_query(verb)).sum(-1)
                          + self.type_bias(verb).squeeze(-1)
                          + self.opt_head(opt_out).squeeze(-1))
        else:
            opt_logits = self.opt_head(opt_out).squeeze(-1)    # [B, n]
        if n < MAX_OPTIONS:                                    # restore full width; dropped slots -> -inf
            opt_logits = nn.functional.pad(opt_logits, (0, MAX_OPTIONS - n), value=-1e9)
        submit_src = extra[1] if self.split_heads else cls_out  # dedicated submit token (split) or CLS
        submit_logit = self.submit_head(submit_src)            # [B, 1]
        value, vdist = self._value_out(self._value_input(cls_out, pooled, extra))
        logits = torch.cat([opt_logits, submit_logit], dim=-1)  # [B, N_ACTIONS]
        logits = logits.masked_fill(o["action_mask"] < 0.5, -1e9)
        if return_vdist:
            return logits, value, vdist
        return logits, value

    def get_value(self, o: dict, opt_len: int | None = None, state_keep=None):
        cls_out, _, pooled, extra = self._encode(o, opt_len=opt_len, state_keep=state_keep)
        return self._value_out(self._value_input(cls_out, pooled, extra))[0]

    def get_action_and_value(self, o: dict, action=None, return_vdist: bool = False,
                             opt_len: int | None = None, state_keep=None):
        logits, value, vdist = self.logits_value(o, return_vdist=True, opt_len=opt_len,
                                                 state_keep=state_keep)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        out = (action, dist.log_prob(action), dist.entropy(), value)
        return (*out, vdist) if return_vdist else out

    # -- categorical-value helpers (used by the trainer's CE loss) ----------
    def value_target_dist(self, returns):
        """Two-hot projection of scalar `returns` [N] onto the atom support -> soft targets [N, atoms]
        (the standard C51 projection). CE(value_logits, this) is the categorical value loss. vmax bounds
        the support; returns outside [-vmax, vmax] are clamped (raise --value-vmax if a graded reward
        pushes returns past +/-1)."""
        atoms = self.atom_support.numel()
        delta = (2.0 * self.value_vmax) / (atoms - 1)
        r = returns.clamp(-self.value_vmax, self.value_vmax)
        b = (r + self.value_vmax) / delta                      # [N] position in [0, atoms-1]
        lo = b.floor().long().clamp(0, atoms - 1)
        hi = (lo + 1).clamp(0, atoms - 1)
        w_hi = (b - lo.to(r.dtype)).clamp(0.0, 1.0)            # fractional part -> upper atom
        rows = torch.arange(returns.shape[0], device=r.device)
        tgt = returns.new_zeros(returns.shape[0], atoms)
        tgt[rows, lo] = 1.0 - w_hi                              # lo==hi at the top atom -> all mass here
        tgt[rows, hi] = tgt[rows, hi] + w_hi
        return tgt


def build_token_net(card_table: CardTable, net_config: dict | None = None) -> TokenTransformer:
    """Construct a TokenTransformer sized to ``card_table`` (vocab_size).

    ``net_config`` keys are constructor kwargs (d_model, nhead, nlayers,
    ff, dropout); unknown keys (e.g. an 'arch' tag, or a legacy 'emb_dim') are ignored.
    """
    cfg = dict(net_config or {})
    cfg.pop("arch", None)
    cfg.pop("emb_dim", None)                               # removed: card_emb is always d_model now (no projection)
    cfg.pop("ff", None)                                    # FFN width derived as 4*d_model
    use_static = cfg.pop("static", False)                  # opt-in static card features; old ckpts have no key -> OFF (load-compatible)
    use_structured = cfg.pop("structured", False)          # opt-in verb-conditioned action head
    use_split = cfg.pop("split_heads", False)              # opt-in: dedicated submit+value tokens (else both read CLS)
    cfg.pop("would_ko", None)                              # metadata flag (env/inference annotate); not a net arg
    opt_struct = cfg.pop("opt_struct", OPT_STRUCT + effect_data.N_ATTACK_FX)   # opt_attr carries the per-attack effect multi-hot
    feat = card_table.matrix if use_static else None
    if feat is not None:                                   # append per-card ability+trainer effect multi-hots to the NET static matrix
        extra = np.asarray([effect_data.ability_multihot(i) + effect_data.trainer_multihot(i)
                            for i in range(feat.shape[0])], dtype=np.float32)
        feat = np.concatenate([np.asarray(feat, dtype=np.float32), extra], axis=1)
    return TokenTransformer(card_table.vocab_size, card_feat=feat, structured=use_structured,
                            split_heads=use_split, opt_struct=opt_struct, **cfg)


if __name__ == "__main__":
    # quick smoke: build the net, encode + batch 3 real obs, forward.
    import os
    import pickle

    from .encoding import TokenEncoder

    enc = TokenEncoder()
    net = build_token_net(enc.cards, {})
    pool_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "notes", "vcalib_pool.pkl")
    pool = pickle.load(open(pool_path, "rb"))
    outs = [enc.encode(pool[i]["root"]) for i in range(3)]
    batch = {k: torch.stack([obs_to_tensors(e, "cpu")[k] for e in outs]) for k in outs[0]}
    logits, value = net.logits_value(batch)
    a, lp, ent, v = net.get_action_and_value(batch)
    gv = net.get_value(batch)
    print("logits", tuple(logits.shape), "value", tuple(value.shape),
          "action", tuple(a.shape), "get_value", tuple(gv.shape))
    # every sampled action must be legal
    legal = batch["action_mask"].gather(1, a[:, None]).squeeze(1)
    assert (legal > 0.5).all(), "sampled an illegal action"
    n_params = sum(p.numel() for p in net.parameters())
    print(f"params={n_params:,}  N_ACTIONS={N_ACTIONS}  OK")
