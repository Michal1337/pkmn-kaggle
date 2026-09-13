# cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True
"""M3 (typed): native encode over the binary obs buffer (BinaryObs.h / GetBinaryObs).

Byte-identical to rl/encoding.py TokenEncoder.encode across EVERY token key. Fully typed: the buffer
is parsed into fixed-size C stack arrays (card lists, per-pokemon fields, option params) and written
straight into the output arrays via memoryviews -- no intermediate Python dicts/lists, no per-option
np.zeros, no dict .get(). Option src/tgt pointer resolution is inlined on typed ints.

Two entry points share one fill core (`_fill_views`):
  * `encode_state(...)`  -> allocates a fresh dict of arrays and returns it (validation / standalone).
  * `class BatchEncoder(B, NFX)` -> preallocates the [B, ...] batch arrays AND acquires their
    memoryviews ONCE; `.encode(..., row)` writes obs `row` straight into the batch (no per-obs alloc,
    no np.stack). This is the training-collection fast path.

Constants MUST match rl/enc_constants.py. Caps (HCAP/ENCAP/...) are safe upper bounds for a 60-card
game; reads always advance the buffer cursor fully and clamp only the STORE, so an over-cap list can
never desync the parse (it would only drop entries a 60-card game can't produce).

Passed besides the obs buffer: self_deck_ids (the FULL decklist, emitted in full; the in-C
visible-zone subtraction fills the per-copy DRAWABLE flag `self_deck_flag`, v2.1), opp_deck_counts
(tracker.copies_hidden_for dict {cid: (n_copies, n_hidden)}, expanded to the opp-deck stream +
`opp_deck_flag` in Cython -- lever 2), opp_hand_ids + opp_hand_flags
(tracker.hand_beliefs_for split into parallel lists -> opp_hand stream + the v2.2 per-belief
CERTAINTY flag `opp_hand_flag`), offense_buff, picked_idx (this decision's buffered multi-select indices ->
cv[17] count + the opt_attr col-14 already-picked flags, v2.1; None == empty), tera_bits (per-cardid
Tera flag), ability_slots (self unit slots), def_serials/def_reductions (tracker.opp_def_buff parallel
arrays), offs (8 _OFF stream offsets), af (attack_feats [MAX_ATTACK,4]), fx (attack_multihot
[MAX_ATTACK,NFX]).
"""
import numpy as np
cimport numpy as cnp
from libc.string cimport memset

MAX_HAND = 30
MAX_DISCARD = 60
DECK_SIZE = 60
N_PRIZE = 6
N_STADIUM = 2
N_BENCH = 8
N_PREEVO = 2
MAX_TOP = 3        # v2.3 deck-top depth cap; MUST equal GameTracker.MAX_TOP
N_TOOLS = 4
N_ENERGY_CARDS = 4
UNIT_ATTR = 24
N_ENERGY_BINS = 12
N_SELECT_TYPES = 16
N_SELECT_CTX = 64
MAX_OPTIONS = 192    # v2.2: 128 -> 192 (mirrors enc_constants.py; see the int32-seq note there)
N_OPT_TYPES = 17
MAX_ATTACK = 2048
OPT_STRUCT = 15      # v2.1: +1 already-picked flag at col 14 (would_ko trio stays [11:14])
_CNT = 15.0

# storage caps (safe upper bounds; a 60-card game can't exceed these) -- cdef enum => compile-time
# C constants usable as fixed C-array dimensions (DEF is deprecated in Cython 3).
cdef enum:
    HCAP = 128      # hand / discard / deck / looking
    SCAP = 8        # stadium
    NPK = 18        # 2 players * (1 active + 8 bench)
    ENCAP = 32      # energies per pokemon
    ECCAP = 16      # energy cards per pokemon
    TLCAP = 8       # tools per pokemon
    PECAP = 4       # pre-evolutions per pokemon
    OCAP = 192      # options (== MAX_OPTIONS)
    VISCAP = 2048   # visible-count array bound for the remaining-deck subtraction (card ids < ~1300)


cdef inline double fmin1(double x):
    return x if x < 1.0 else 1.0


cdef inline void z1f(float[::1] v):
    if v.shape[0] > 0: memset(&v[0], 0, <size_t>v.shape[0] * 4)
cdef inline void z1l(long long[::1] v):
    if v.shape[0] > 0: memset(&v[0], 0, <size_t>v.shape[0] * 8)
cdef inline void z2f(float[:, ::1] v):
    if v.shape[0] > 0: memset(&v[0, 0], 0, <size_t>(v.shape[0] * v.shape[1]) * 4)
cdef inline void z2l(long long[:, ::1] v):
    if v.shape[0] > 0: memset(&v[0, 0], 0, <size_t>(v.shape[0] * v.shape[1]) * 8)
cdef inline void neg1l(long long[::1] v):   # int64 all-0xFF bytes == -1 (np.full(-1) equivalent)
    if v.shape[0] > 0: memset(&v[0], 255, <size_t>v.shape[0] * 8)
cdef inline void cl1(long long[::1] v, long long unk):   # card-id clamp to [0, UNK] (np.clip mirror)
    cdef Py_ssize_t i
    for i in range(v.shape[0]):
        if v[i] < 0: v[i] = 0
        elif v[i] > unk: v[i] = unk
cdef inline void cl2(long long[:, ::1] v, long long unk):
    cdef Py_ssize_t i, j
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if v[i, j] < 0: v[i, j] = 0
            elif v[i, j] > unk: v[i, j] = unk


cdef int c_board_slot(int area, int index):
    if area == 4: return 0
    if area == 5 and 0 <= index < 8: return 1 + index      # N_BENCH
    return -1


cdef long long c_unit_pos(int area, int index, bint owner_self, long long[::1] offs):
    cdef int sl = c_board_slot(area, index)
    if sl < 0: return -1
    return (offs[0] if owner_self else offs[1]) + sl


cdef long long c_ref_pos(int area, int index, bint owner_self, long long[::1] offs):
    cdef int sl = c_board_slot(area, index)
    if sl >= 0: return (offs[0] if owner_self else offs[1]) + sl
    if index < 0: return -1
    if area == 2:
        if index >= 30: return -1                          # MAX_HAND
        return (offs[2] if owner_self else offs[3]) + index
    if area == 3:
        if index >= 60: return -1                          # MAX_DISCARD
        return (offs[4] if owner_self else offs[5]) + index
    return -1


cdef int c_read_cardlist(int[::1] b, int i, long long* dst, int cap, int* out_cnt):
    """count-prefixed list of (id,serial,playerIndex) triples -> ids into dst[:cap]; advance fully.
    cnt<0 (e.g. deck/looking == null) -> out_cnt=-1, only the count int consumed."""
    cdef int cnt = b[i], k
    i += 1
    for k in range(cnt):
        if k < cap: dst[k] = b[i]
        i += 3
    out_cnt[0] = cnt
    return i


cdef void _sil(long long[::1] outid, float[::1] outm, list ids, int n, int unk, int fill_unk):
    """Python-list-sourced stream into pre-zeroed views (self/opp deck, prizes, opp_hand)."""
    cdef int i = 0, k
    cdef long long cid
    for cid in ids:
        if i >= n: break
        outid[i] = cid; outm[i] = 1.0; i += 1
    for k in range(fill_unk):
        if i >= n: break
        outid[i] = unk; outm[i] = 1.0; i += 1


cdef void _sic(long long[::1] outid, float[::1] outm, long long* src, int srcn, int n, int unk, int fill_unk):
    """C-array-sourced stream into pre-zeroed views (self hand, self/opp discard)."""
    cdef int i = 0, k
    for k in range(srcn):
        if i >= n: break
        outid[i] = src[k]; outm[i] = 1.0; i += 1
    for k in range(fill_unk):
        if i >= n: break
        outid[i] = unk; outm[i] = 1.0; i += 1


cdef _fill_views(int[::1] buf, int nbuf, int unk,
                 list self_deck_ids, list top_ids, dict opp_deck_counts, list opp_hand_ids, list opp_hand_flags,
                 float offense_buff, object picked_idx,
                 float[::1] tera_bits, object ability_slots,
                 long long[::1] def_serials, float[::1] def_reductions,
                 long long[::1] offs, float[:, ::1] af, float[:, ::1] fx, double[:, ::1] wk,
                 # ---- 45 output row-views (already the single-obs shape) ----
                 float[::1] cv, long long[::1] stype_v, long long[::1] sctx_v,
                 long long[::1] effid_v, float[::1] effm_v,
                 long long[::1] sdeck_id, float[::1] sdeck_m, float[::1] sdeck_fl, float[::1] sdeck_top,
                 long long[::1] odeck_id, float[::1] odeck_m, float[::1] odeck_fl,
                 long long[::1] sprize_id, float[::1] sprize_m, long long[::1] oprize_id, float[::1] oprize_m,
                 long long[::1] shand_id, float[::1] shand_m, long long[::1] ohand_id, float[::1] ohand_m,
                 float[::1] ohand_fl,
                 long long[::1] sdisc_id, float[::1] sdisc_m, long long[::1] odisc_id, float[::1] odisc_m,
                 long long[::1] sid, float[::1] smask,
                 long long[::1] s_top, long long[:, ::1] s_pre, long long[:, ::1] s_tool,
                 long long[:, ::1] s_en, float[:, ::1] s_attr, float[::1] s_mask,
                 long long[::1] o_top, long long[:, ::1] o_pre, long long[:, ::1] o_tool,
                 long long[:, ::1] o_en, float[:, ::1] o_attr, float[::1] o_mask,
                 float[:, ::1] oattr, long long[::1] overb, long long[::1] oaid,
                 long long[::1] ospos, long long[::1] otpos, long long[::1] oscard, long long[::1] otcard):
    cdef int i, j, k, q, oi, slot, di, si, pl, grp, c2, fj, pi
    cdef int n_top, td_cid
    cdef int turn, tac, me, first, supp, stadp, eatt, retr, opp
    cdef int cnt, ne, present, t, area, index, owner, pk_idx
    cdef int nopt, n_o, ctx_id = 0, eff_id_v = 0, has_eff
    cdef int st_type, st_ctx, minC, maxC, rdc, rec
    cdef int p0, p1, p2, p3, p4, aid, jj, pidx_v, ceil_v, e
    cdef int id_v, ser_v, hp_v, mhp_v, app_v, nkn, src_cid
    cdef int od_i, od_r, od_cnt, od_hid, OS, fx0z, N_FX
    cdef long long od_cid
    cdef int picked_count = 0, poi, cid_v, rem_i
    cdef int vis_cnt[VISCAP]
    cdef int NIL = -2000000000
    cdef bint owner_self, store
    cdef long long sp, pos, tp
    cdef long long stad_pos = -1
    cdef int deckC[2]
    cdef int handC[2]
    cdef int prizeC[2]
    cdef int pois[2]
    cdef int burn[2]
    cdef int asle[2]
    cdef int para[2]
    cdef int conf[2]
    cdef long long hand_a[HCAP]
    cdef long long d0_a[HCAP]
    cdef long long d1_a[HCAP]
    cdef long long deck_a[HCAP]
    cdef long long look_a[HCAP]
    cdef long long stad_a[SCAP]
    cdef int hand_n, d0_n, d1_n, deck_n, look_n, stad_n = 0
    cdef long long pk_id[NPK]
    cdef long long pk_serial[NPK]
    cdef int pk_present[NPK]
    cdef int pk_hp[NPK]
    cdef int pk_maxhp[NPK]
    cdef int pk_appear[NPK]
    cdef int pk_nen[NPK]
    cdef int pk_nec[NPK]
    cdef int pk_ntl[NPK]
    cdef int pk_npe[NPK]
    cdef long long pk_en[NPK][ENCAP]
    cdef long long pk_ec[NPK][ECCAP]
    cdef long long pk_tl[NPK][TLCAP]
    cdef long long pk_pe[NPK][PECAP]
    cdef int opt_ty[OCAP]
    cdef int opt_p0[OCAP]
    cdef int opt_p1[OCAP]
    cdef int opt_p2[OCAP]
    cdef int opt_p3[OCAP]
    cdef int opt_p4[OCAP]
    cdef long long[::1] u_top
    cdef long long[:, ::1] u_pre, u_tool, u_en
    cdef float[:, ::1] u_attr
    cdef float[::1] u_mask
    cdef double mhp

    # ---- clear all output views (batch buffers may be reused across rollouts) ----
    z1f(cv); z1l(stype_v); z1l(sctx_v); z1l(effid_v); z1f(effm_v)
    z1l(sdeck_id); z1f(sdeck_m); z1f(sdeck_fl); z1f(sdeck_top)
    z1l(odeck_id); z1f(odeck_m); z1f(odeck_fl)
    z1l(sprize_id); z1f(sprize_m); z1l(oprize_id); z1f(oprize_m)
    z1l(shand_id); z1f(shand_m); z1l(ohand_id); z1f(ohand_m); z1f(ohand_fl)
    z1l(sdisc_id); z1f(sdisc_m); z1l(odisc_id); z1f(odisc_m)
    z1l(sid); z1f(smask)
    z1l(s_top); z2l(s_pre); z2l(s_tool); z2l(s_en); z2f(s_attr); z1f(s_mask)
    z1l(o_top); z2l(o_pre); z2l(o_tool); z2l(o_en); z2f(o_attr); z1f(o_mask)
    z2f(oattr); z1l(overb); z1l(oaid); neg1l(ospos); neg1l(otpos); z1l(oscard); z1l(otcard)

    for q in range(NPK):
        pk_present[q] = 0; pk_nen[q] = 0; pk_nec[q] = 0; pk_ntl[q] = 0; pk_npe[q] = 0

    # --- A: scalars ---
    turn = buf[0]; tac = buf[1]; me = buf[2]; first = buf[3]
    supp = buf[4]; stadp = buf[5]; eatt = buf[6]; retr = buf[7]
    opp = 1 - me
    i = 9
    # --- B: per-player counts + status ---
    for pl in range(2):
        deckC[pl] = buf[i+1]; handC[pl] = buf[i+2]; prizeC[pl] = buf[i+3]
        pois[pl] = buf[i+7]; burn[pl] = buf[i+8]; asle[pl] = buf[i+9]; para[pl] = buf[i+10]; conf[pl] = buf[i+11]
        i += 12
    # --- C: card lists (self hand, discard0, discard1, stadium) ---
    i = c_read_cardlist(buf, i, &hand_a[0], HCAP, &hand_n)
    i = c_read_cardlist(buf, i, &d0_a[0], HCAP, &d0_n)
    i = c_read_cardlist(buf, i, &d1_a[0], HCAP, &d1_n)
    cnt = buf[i]; i += 1; stad_n = cnt
    for k in range(cnt):
        slot = 0 if buf[i+2] == me else 1
        sid[slot] = buf[i]; smask[slot] = 1.0
        if k < SCAP: stad_a[k] = buf[i]
        i += 3

    cv[0] = fmin1(turn / 50.0); cv[1] = fmin1(tac / 20.0)
    cv[2] = 1.0 if first == me else 0.0
    cv[3] = 1.0 if supp else 0.0; cv[4] = 1.0 if stadp else 0.0
    cv[5] = 1.0 if eatt else 0.0; cv[6] = 1.0 if retr else 0.0
    cv[7] = deckC[me] / 60.0; cv[8] = deckC[opp] / 60.0
    cv[9] = prizeC[me] / 6.0; cv[10] = prizeC[opp] / 6.0
    cv[11] = fmin1(handC[me] / 12.0); cv[12] = fmin1(handC[opp] / 12.0)

    # --- D: units (p0 active/bench, p1 active/bench) into per-pokemon C arrays ---
    for pl in range(2):
        for grp in range(2):                                   # 0 active, 1 bench
            c2 = buf[i]; i += 1
            for j in range(c2):
                present = buf[i]; i += 1
                pk_idx = pl*9 + (0 if grp == 0 else 1 + j)
                if pk_idx >= pl*9 + 9: pk_idx = pl*9 + 8       # clamp (unreachable in a real game)
                store = (present != 0)
                if not store:
                    continue                                    # facedown -> None (pk_present stays 0)
                id_v = buf[i]; ser_v = buf[i+1]; hp_v = buf[i+3]; mhp_v = buf[i+4]; app_v = buf[i+5]; i += 6
                ne = buf[i]; i += 1
                pk_present[pk_idx] = 1; pk_id[pk_idx] = id_v; pk_serial[pk_idx] = ser_v
                pk_hp[pk_idx] = hp_v; pk_maxhp[pk_idx] = mhp_v; pk_appear[pk_idx] = app_v; pk_nen[pk_idx] = ne
                for q in range(ne):
                    if q < ENCAP: pk_en[pk_idx][q] = buf[i]
                    i += 1
                i = c_read_cardlist(buf, i, &pk_ec[pk_idx][0], ECCAP, &pk_nec[pk_idx])
                i = c_read_cardlist(buf, i, &pk_tl[pk_idx][0], TLCAP, &pk_ntl[pk_idx])
                i = c_read_cardlist(buf, i, &pk_pe[pk_idx][0], PECAP, &pk_npe[pk_idx])

    # build unit tokens straight from the pokemon arrays
    for si in range(2):
        if si == 0:
            pi = me; u_top = s_top; u_pre = s_pre; u_tool = s_tool; u_en = s_en; u_attr = s_attr; u_mask = s_mask
        else:
            pi = opp; u_top = o_top; u_pre = o_pre; u_tool = o_tool; u_en = o_en; u_attr = o_attr; u_mask = o_mask
        for slot in range(1 + N_BENCH):
            pk_idx = pi*9 + slot
            if pk_present[pk_idx] == 0:
                continue
            u_top[slot] = pk_id[pk_idx]; u_mask[slot] = 1.0
            for q in range(pk_npe[pk_idx] if pk_npe[pk_idx] < N_PREEVO else N_PREEVO): u_pre[slot, q] = pk_pe[pk_idx][q]
            for q in range(pk_ntl[pk_idx] if pk_ntl[pk_idx] < N_TOOLS else N_TOOLS): u_tool[slot, q] = pk_tl[pk_idx][q]
            for q in range(pk_nec[pk_idx] if pk_nec[pk_idx] < N_ENERGY_CARDS else N_ENERGY_CARDS): u_en[slot, q] = pk_ec[pk_idx][q]
            mhp = pk_maxhp[pk_idx]
            u_attr[slot, 0] = (pk_hp[pk_idx] / mhp) if mhp > 0 else 0.0
            u_attr[slot, 1] = mhp / 500.0
            for q in range(pk_nen[pk_idx] if pk_nen[pk_idx] < ENCAP else ENCAP):
                e = pk_en[pk_idx][q]
                if 0 <= e < N_ENERGY_BINS: u_attr[slot, 2 + e] += 1.0
            for q in range(N_ENERGY_BINS):
                u_attr[slot, 2 + q] = fmin1(u_attr[slot, 2 + q] / 4.0)
            u_attr[slot, 14] = fmin1(pk_nen[pk_idx] / 12.0)
            if slot == 0:                                       # active-only status
                u_attr[slot, 15] = 1.0 if pois[pi] else 0.0
                u_attr[slot, 16] = 1.0 if burn[pi] else 0.0
                u_attr[slot, 17] = 1.0 if asle[pi] else 0.0
                u_attr[slot, 18] = 1.0 if para[pi] else 0.0
                u_attr[slot, 19] = 1.0 if conf[pi] else 0.0
            u_attr[slot, 20] = 1.0 if pk_appear[pk_idx] else 0.0
            id_v = <int>pk_id[pk_idx]
            if 0 <= id_v < tera_bits.shape[0] and tera_bits[id_v] > 0.5: u_attr[slot, 21] = 1.0
        if si == 0 and ability_slots:
            for slot in ability_slots:
                if 0 <= slot < 1 + N_BENCH and u_mask[slot] > 0.5: u_attr[slot, 22] = 1.0
        for slot in range(1 + N_BENCH):
            if u_mask[slot] > 0.5:
                ser_v = <int>pk_serial[pi*9 + slot]
                for di in range(def_serials.shape[0]):
                    if def_serials[di] == ser_v:
                        u_attr[slot, 23] = (def_reductions[di] if def_reductions[di] < 200.0 else 200.0) / 200.0
                        break

    # --- E: select-level fields + options ---
    st_type = buf[i]; st_ctx = buf[i+1]; minC = buf[i+2]; maxC = buf[i+3]
    rdc = buf[i+4]; rec = buf[i+5]; i += 6
    nopt = buf[i]; i += 1
    for j in range(nopt):
        if j < MAX_OPTIONS:
            opt_ty[j] = buf[i]; opt_p0[j] = buf[i+1]; opt_p1[j] = buf[i+2]
            opt_p2[j] = buf[i+3]; opt_p3[j] = buf[i+4]; opt_p4[j] = buf[i+5]
        i += 6
    i = c_read_cardlist(buf, i, &deck_a[0], HCAP, &deck_n)      # deck-search subset (deck_n=-1 -> null)
    if buf[i]:
        ctx_id = buf[i+1]; i += 4
    else:
        i += 1
    if buf[i]:
        eff_id_v = buf[i+1]; i += 4
    else:
        i += 1
    # --- F: looking (zone_card area 12) ---
    i = c_read_cardlist(buf, i, &look_a[0], HCAP, &look_n)

    if picked_idx is not None:
        picked_count = len(picked_idx)
    cv[13] = fmin1(rdc / 20.0); cv[14] = rec / 5.0
    cv[15] = fmin1(minC / 5.0); cv[16] = fmin1(maxC / 5.0)
    cv[17] = fmin1(picked_count / 5.0); cv[18] = fmin1(offense_buff / 100.0)
    stype_v[0] = min(st_type, N_SELECT_TYPES - 1)
    sctx_v[0] = min(st_ctx, N_SELECT_CTX - 1)
    if eff_id_v: effid_v[0] = eff_id_v; effm_v[0] = 1.0
    if ctx_id: effid_v[1] = ctx_id; effm_v[1] = 1.0

    # --- streams ---
    # self deck stream = the FULL decklist + a per-copy DRAWABLE flag (v2.1): flag=1 iff that copy
    # is not visible in any of our public zones (hand / own discard / own units incl. preevo+tools+
    # energyCards / own stadium) == still in deck or face-down in our prizes. MIRRORS
    # encoding.decklist_drawable_flags EXACTLY: count-based (zone enumeration order irrelevant),
    # same per-unit windows (PECAP/TLCAP/ECCAP), decklist walk order, visible copies consume
    # duplicates front-first. `looking` cards stay flag=1 (still deck-owned).
    memset(&vis_cnt[0], 0, sizeof(vis_cnt))
    for k in range(hand_n if hand_n < HCAP else HCAP):
        cid_v = <int>hand_a[k]
        if 0 < cid_v < VISCAP: vis_cnt[cid_v] += 1
    if me == 0:
        for k in range(d0_n if d0_n < HCAP else HCAP):
            cid_v = <int>d0_a[k]
            if 0 < cid_v < VISCAP: vis_cnt[cid_v] += 1
    else:
        for k in range(d1_n if d1_n < HCAP else HCAP):
            cid_v = <int>d1_a[k]
            if 0 < cid_v < VISCAP: vis_cnt[cid_v] += 1
    for slot in range(1 + N_BENCH):
        pk_idx = me*9 + slot
        if pk_present[pk_idx] == 0:
            continue
        cid_v = <int>pk_id[pk_idx]
        if 0 < cid_v < VISCAP: vis_cnt[cid_v] += 1
        for q in range(pk_npe[pk_idx] if pk_npe[pk_idx] < PECAP else PECAP):
            cid_v = <int>pk_pe[pk_idx][q]
            if 0 < cid_v < VISCAP: vis_cnt[cid_v] += 1
        for q in range(pk_ntl[pk_idx] if pk_ntl[pk_idx] < TLCAP else TLCAP):
            cid_v = <int>pk_tl[pk_idx][q]
            if 0 < cid_v < VISCAP: vis_cnt[cid_v] += 1
        for q in range(pk_nec[pk_idx] if pk_nec[pk_idx] < ECCAP else ECCAP):
            cid_v = <int>pk_ec[pk_idx][q]
            if 0 < cid_v < VISCAP: vis_cnt[cid_v] += 1
    if smask[0] > 0.5:                                         # slot 0 == self-owned stadium
        cid_v = <int>sid[0]
        if 0 < cid_v < VISCAP: vis_cnt[cid_v] += 1
    _sil(sdeck_id, sdeck_m, self_deck_ids, DECK_SIZE, unk, 0)  # the FULL decklist, in order
    rem_i = 0                                                  # decklist position
    for cid_obj in self_deck_ids:
        if rem_i >= DECK_SIZE:
            break
        cid_v = cid_obj
        if 0 < cid_v < VISCAP and vis_cnt[cid_v] > 0:
            vis_cnt[cid_v] -= 1                                # visible copy -> flag stays 0
        else:
            sdeck_fl[rem_i] = 1.0                              # still in deck (or prized)
        rem_i += 1
    # v2.3 self_deck_top: KNOWN top-of-deck depth code, mirrors GameTracker.top_flags -- the id is
    # assigned to the FIRST still-drawable decklist slot holding it (copies are interchangeable);
    # a nonzero entry doubles as the "slot taken" marker (== the python `used` set).
    if top_ids is not None:
        n_top = len(top_ids)
        if n_top > MAX_TOP:
            n_top = MAX_TOP
        for k in range(n_top):
            td_cid = <int>top_ids[k]
            rem_i = 0
            for cid_obj in self_deck_ids:
                if rem_i >= DECK_SIZE:
                    break
                cid_v = cid_obj
                if cid_v == td_cid and sdeck_fl[rem_i] > 0.0 and sdeck_top[rem_i] == 0.0:
                    sdeck_top[rem_i] = <float>(MAX_TOP - k) / <float>MAX_TOP
                    break
                rem_i += 1
    # opp deck: expand copies_hidden_for {cid: (n_copies, n_hidden)} in Cython, then UNK-fill to 60.
    # v2.1 opp_deck_flag: hidden copies FIRST within a cid get flag 1 (not currently visible ->
    # plausibly still to come), visible/spent copies 0; unrevealed UNK fill = 1. MIRRORS the
    # Python expansion in encoding.encode ([1]*h + [0]*(n-h), same dict iteration order).
    od_i = 0
    for od_cid, od_val in opp_deck_counts.items():
        od_cnt = <int>od_val[0]; od_hid = <int>od_val[1]
        for od_r in range(od_cnt):
            if od_i >= DECK_SIZE: break
            odeck_id[od_i] = od_cid; odeck_m[od_i] = 1.0
            if od_r < od_hid: odeck_fl[od_i] = 1.0
            od_i += 1
        if od_i >= DECK_SIZE: break
    while od_i < DECK_SIZE:
        odeck_id[od_i] = unk; odeck_m[od_i] = 1.0; odeck_fl[od_i] = 1.0; od_i += 1
    _sil(sprize_id, sprize_m, [], N_PRIZE, unk, min(prizeC[me], N_PRIZE))
    _sil(oprize_id, oprize_m, [], N_PRIZE, unk, min(prizeC[opp], N_PRIZE))
    _sic(shand_id, shand_m, &hand_a[0], hand_n, MAX_HAND, unk, 0)
    nkn = min(len(opp_hand_ids), handC[opp])
    if nkn > MAX_HAND: nkn = MAX_HAND
    _sil(ohand_id, ohand_m, opp_hand_ids[:nkn], MAX_HAND, unk, max(0, min(handC[opp], MAX_HAND) - nkn))
    for j in range(nkn):                                   # v2.2 per-belief certainty flag
        ohand_fl[j] = <float>opp_hand_flags[j]
    if me == 0:
        _sic(sdisc_id, sdisc_m, &d0_a[0], d0_n, MAX_DISCARD, unk, 0)
        _sic(odisc_id, odisc_m, &d1_a[0], d1_n, MAX_DISCARD, unk, 0)
    else:
        _sic(sdisc_id, sdisc_m, &d1_a[0], d1_n, MAX_DISCARD, unk, 0)
        _sic(odisc_id, odisc_m, &d0_a[0], d0_n, MAX_DISCARD, unk, 0)

    # --- option loop (opt_attr/verb/attack_id/src_pos/tgt_pos/src_card/tgt_card + fx), fully inlined ---
    n_o = min(nopt, MAX_OPTIONS)
    has_eff = 1 if eff_id_v else 0
    if smask[0] > 0.5: stad_pos = offs[6] + 0
    elif smask[1] > 0.5: stad_pos = offs[6] + 1
    ceil_v = maxC if maxC > rdc else rdc
    if ceil_v < 1: ceil_v = 1
    for oi in range(n_o):
        t = opt_ty[oi]
        p0 = opt_p0[oi]; p1 = opt_p1[oi]; p2 = opt_p2[oi]; p3 = opt_p3[oi]; p4 = opt_p4[oi]
        overb[oi] = t if t < N_OPT_TYPES - 1 else N_OPT_TYPES - 1
        # ---- struct row [0..13] ----
        if t == 6:
            oattr[oi, 0] = fmin1(p4 / 5.0)                      # count
        elif t == 0:
            oattr[oi, 1] = fmin1(p0 / <double>ceil_v)          # number (rescaled by max(maxC,rdc))
        elif t == 13:
            aid = p0
            if aid > 0:
                oaid[oi] = aid if aid < MAX_ATTACK - 1 else MAX_ATTACK - 1
                if aid < af.shape[0]:
                    oattr[oi, 2] = af[aid, 0]; oattr[oi, 3] = af[aid, 1]
                    oattr[oi, 4] = af[aid, 2]; oattr[oi, 5] = af[aid, 3]
        elif t == 16:
            if 0 <= p0 < 5: oattr[oi, 6 + p0] = 1.0
        # ---- would_ko trio [11,12,13] (engine-sim result passed in via wk; 0 when --no-would-ko) ----
        oattr[oi, 11] = wk[oi, 0]
        oattr[oi, 12] = fmin1(wk[oi, 1] / 6.0)
        oattr[oi, 13] = wk[oi, 2]
        # ---- src/tgt pointer resolution ----
        if t == 3 or t == 4 or t == 5 or t == 6:
            pidx_v = p2
        else:
            pidx_v = NIL
        owner_self = (pidx_v == NIL) or (pidx_v == me)
        owner = me if pidx_v == NIL else pidx_v
        if t == 4 or t == 5 or t == 6:
            area = p0; index = p1
            sp = c_unit_pos(area, index, owner_self, offs)
            if sp >= 0: ospos[oi] = sp
            pk_idx = -1
            if area == 4:
                pk_idx = owner*9 + 0
            elif area == 5 and 0 <= index < N_BENCH:
                pk_idx = owner*9 + 1 + index
            if pk_idx >= 0 and pk_present[pk_idx]:
                jj = p3
                if t == 4:
                    if 0 <= jj < pk_ntl[pk_idx]: otcard[oi] = pk_tl[pk_idx][jj]
                else:
                    if 0 <= jj < pk_nec[pk_idx]: otcard[oi] = pk_ec[pk_idx][jj]
        elif t == 13:
            ospos[oi] = offs[0]; otpos[oi] = offs[1]
        else:
            if t == 3 or t == 8 or t == 9 or t == 10 or t == 11:
                area = p0
            elif t == 7:
                area = 2
            else:
                area = NIL
            if t == 3 or t == 8 or t == 9 or t == 10 or t == 11:
                index = p1
            elif t == 7:
                index = p0
            else:
                index = NIL
            if area == 7:
                pos = stad_pos
            else:
                pos = c_ref_pos(area, index, owner_self, offs)
            if pos >= 0:
                ospos[oi] = pos
            else:
                src_cid = 0
                if t == 15:
                    src_cid = p0                                # cardId
                elif index >= 0:
                    if area == 1:
                        if 0 <= index < deck_n: src_cid = <int>deck_a[index]
                    elif area == 7:
                        if index < stad_n: src_cid = <int>stad_a[index]
                    elif area == 12:
                        if look_n > 0 and index < look_n:
                            src_cid = <int>look_a[index]
                            if src_cid < 0: src_cid = 0
                    elif area == 2:
                        if owner == me and index < hand_n: src_cid = <int>hand_a[index]
                    elif area == 3:
                        if owner == 0:
                            if index < d0_n: src_cid = <int>d0_a[index]
                        else:
                            if index < d1_n: src_cid = <int>d1_a[index]
                    elif area == 4:
                        if index == 0 and pk_present[owner*9 + 0]: src_cid = <int>pk_id[owner*9 + 0]
                    elif area == 5:
                        if 0 <= index < N_BENCH and pk_present[owner*9 + 1 + index]:
                            src_cid = <int>pk_id[owner*9 + 1 + index]
                if src_cid != 0:
                    oscard[oi] = src_cid
                elif has_eff:
                    ospos[oi] = offs[7]
            if t == 8 or t == 9:
                tp = c_unit_pos(p2, p3, owner_self, offs)
                if tp >= 0: otpos[oi] = tp
    # already-picked flag [col 14] (v2.1): options in this decision's buffered multi-select set stay
    # VISIBLE to the net (present in attention) but action-masked. Mirrors encoding.encode exactly
    # (any picked index < MAX_OPTIONS is flagged, independent of n_o).
    if picked_idx is not None:
        for poi in picked_idx:
            if 0 <= poi < MAX_OPTIONS:
                oattr[poi, 14] = 1.0
    # fx multi-hot [cols OPT_STRUCT:]: rows with oaid==0 use fx[0]. fx[0] is all-zeros (no-attack
    # multihot), so the memset already set those rows -> write ONLY the few attack rows (oaid>0).
    # OS hoisted to a C local so the inner write is fully typed (the Python-global lookup was ~80us).
    OS = OPT_STRUCT
    N_FX = fx.shape[1]
    fx0z = 1
    for fj in range(N_FX):
        if fx[0, fj] != 0.0:
            fx0z = 0; break
    if fx0z:
        for oi in range(n_o):                                  # padded/non-attack rows already 0 (=fx[0])
            aid = <int>oaid[oi]
            if aid != 0:
                for fj in range(N_FX): oattr[oi, OS + fj] = fx[aid, fj]
    else:                                                      # safety fallback (fx[0] not zero): full copy
        for oi in range(MAX_OPTIONS):
            aid = <int>oaid[oi]
            for fj in range(N_FX): oattr[oi, OS + fj] = fx[aid, fj]

    # ---- card-id clamp to [0, UNK] -- mirrors encoding.py's np.clip over int_keys minus
    # _NON_CARD_INT_KEYS: an id absent from the card CSV (engine-DB skew) must map to the
    # learnable UNK row here too, instead of overflowing card_emb (crash / silent divergence).
    # No-op with a matched DB (all ids already in range) -> byte-identity preserved.
    cl1(sdeck_id, unk); cl1(odeck_id, unk); cl1(sprize_id, unk); cl1(oprize_id, unk)
    cl1(shand_id, unk); cl1(ohand_id, unk); cl1(sdisc_id, unk); cl1(odisc_id, unk)
    cl1(sid, unk); cl1(effid_v, unk); cl1(oscard, unk); cl1(otcard, unk)
    cl1(s_top, unk); cl1(o_top, unk)
    cl2(s_pre, unk); cl2(s_tool, unk); cl2(s_en, unk)
    cl2(o_pre, unk); cl2(o_tool, unk); cl2(o_en, unk)


def encode_state(int[::1] buf, int nbuf, int unk,
                 list self_deck_ids, list top_ids, dict opp_deck_counts, list opp_hand_ids, list opp_hand_flags,
                 float offense_buff, object picked_idx,
                 float[::1] tera_bits, object ability_slots,
                 long long[::1] def_serials, float[::1] def_reductions,
                 long long[::1] offs, float[:, ::1] af, float[:, ::1] fx, double[:, ::1] would_ko):
    """Standalone / validation path: allocate a fresh dict of arrays, fill, return it."""
    cdef int NFX = fx.shape[1]
    out = {}
    out["cls_scalars"] = np.empty(19, np.float32)
    out["select_type"] = np.empty(1, np.int64); out["select_context"] = np.empty(1, np.int64)
    out["effect_id"] = np.empty(2, np.int64); out["effect_mask"] = np.empty(2, np.float32)
    for pref, sz in (("self_deck", 60), ("opp_deck", 60), ("self_prize", 6), ("opp_prize", 6),
                     ("self_hand", 30), ("opp_hand", 30), ("self_discard", 60), ("opp_discard", 60)):
        out[pref + "_id"] = np.empty(sz, np.int64); out[pref + "_mask"] = np.empty(sz, np.float32)
    out["self_deck_flag"] = np.empty(60, np.float32)           # v2.1 per-copy drawable flag
    out["self_deck_top"] = np.empty(60, np.float32)            # v2.3 per-copy known-top depth
    out["opp_deck_flag"] = np.empty(60, np.float32)            # v2.1 per-copy hidden/still-to-come flag
    out["opp_hand_flag"] = np.empty(30, np.float32)            # v2.2 known-belief certainty flag
    out["stadium_id"] = np.empty(2, np.int64); out["stadium_mask"] = np.empty(2, np.float32)
    for side in ("self", "opp"):
        out[side + "_unit_top_id"] = np.empty(1 + N_BENCH, np.int64)
        out[side + "_unit_preevo_id"] = np.empty((1 + N_BENCH, N_PREEVO), np.int64)
        out[side + "_unit_tool_id"] = np.empty((1 + N_BENCH, N_TOOLS), np.int64)
        out[side + "_unit_energy_id"] = np.empty((1 + N_BENCH, N_ENERGY_CARDS), np.int64)
        out[side + "_unit_attr"] = np.empty((1 + N_BENCH, UNIT_ATTR), np.float32)
        out[side + "_unit_mask"] = np.empty(1 + N_BENCH, np.float32)
    out["opt_attr"] = np.empty((MAX_OPTIONS, OPT_STRUCT + NFX), np.float32)
    out["opt_verb"] = np.empty(MAX_OPTIONS, np.int64); out["opt_attack_id"] = np.empty(MAX_OPTIONS, np.int64)
    out["opt_src_pos"] = np.empty(MAX_OPTIONS, np.int64); out["opt_tgt_pos"] = np.empty(MAX_OPTIONS, np.int64)
    out["opt_src_card"] = np.empty(MAX_OPTIONS, np.int64); out["opt_tgt_card"] = np.empty(MAX_OPTIONS, np.int64)
    _fill_views(buf, nbuf, unk, self_deck_ids, top_ids, opp_deck_counts, opp_hand_ids, opp_hand_flags, offense_buff, picked_idx,
                tera_bits, ability_slots, def_serials, def_reductions, offs, af, fx, would_ko, out["cls_scalars"], out["select_type"], out["select_context"], out["effect_id"], out["effect_mask"],
                out["self_deck_id"], out["self_deck_mask"], out["self_deck_flag"], out["self_deck_top"],
                out["opp_deck_id"], out["opp_deck_mask"], out["opp_deck_flag"],
                out["self_prize_id"], out["self_prize_mask"], out["opp_prize_id"], out["opp_prize_mask"],
                out["self_hand_id"], out["self_hand_mask"], out["opp_hand_id"], out["opp_hand_mask"],
                out["opp_hand_flag"],
                out["self_discard_id"], out["self_discard_mask"], out["opp_discard_id"], out["opp_discard_mask"],
                out["stadium_id"], out["stadium_mask"],
                out["self_unit_top_id"], out["self_unit_preevo_id"], out["self_unit_tool_id"],
                out["self_unit_energy_id"], out["self_unit_attr"], out["self_unit_mask"],
                out["opp_unit_top_id"], out["opp_unit_preevo_id"], out["opp_unit_tool_id"],
                out["opp_unit_energy_id"], out["opp_unit_attr"], out["opp_unit_mask"],
                out["opt_attr"], out["opt_verb"], out["opt_attack_id"],
                out["opt_src_pos"], out["opt_tgt_pos"], out["opt_src_card"], out["opt_tgt_card"])
    return out


cdef class BatchEncoder:
    """Training-collection fast path. Preallocate the [B, ...] batch arrays + their memoryviews ONCE;
    `.encode(..., row)` writes obs `row` straight into the batch (no per-obs alloc, no np.stack).
    Read the stacked result from `.batch` after filling rows [0, nrows)."""
    cdef public dict batch
    cdef public int B
    cdef float[:, ::1] cls_b
    cdef long long[:, ::1] stype_b, sctx_b, effid_b
    cdef float[:, ::1] effm_b
    cdef long long[:, ::1] sdeck_id_b, odeck_id_b, sprize_id_b, oprize_id_b, shand_id_b, ohand_id_b, sdisc_id_b, odisc_id_b
    cdef float[:, ::1] sdeck_m_b, sdeck_fl_b, sdeck_top_b, odeck_m_b, odeck_fl_b, sprize_m_b, oprize_m_b, shand_m_b, ohand_m_b, ohand_fl_b, sdisc_m_b, odisc_m_b
    cdef long long[:, ::1] sid_b
    cdef float[:, ::1] smask_b
    cdef long long[:, ::1] s_top_b, o_top_b
    cdef long long[:, :, ::1] s_pre_b, s_tool_b, s_en_b, o_pre_b, o_tool_b, o_en_b
    cdef float[:, :, ::1] s_attr_b, o_attr_b
    cdef float[:, ::1] s_mask_b, o_mask_b
    cdef float[:, :, ::1] oattr_b
    cdef long long[:, ::1] overb_b, oaid_b, ospos_b, otpos_b, oscard_b, otcard_b

    def __init__(self, int B, int NFX):
        self.B = B
        self.batch = {}
        cdef dict d = self.batch
        def mk(key, shape, dt):
            a = np.zeros(shape, dt); d[key] = a; return a
        self.cls_b = mk("cls_scalars", (B, 19), np.float32)
        self.stype_b = mk("select_type", (B, 1), np.int64); self.sctx_b = mk("select_context", (B, 1), np.int64)
        self.effid_b = mk("effect_id", (B, 2), np.int64); self.effm_b = mk("effect_mask", (B, 2), np.float32)
        self.sdeck_id_b = mk("self_deck_id", (B, 60), np.int64); self.sdeck_m_b = mk("self_deck_mask", (B, 60), np.float32)
        self.sdeck_fl_b = mk("self_deck_flag", (B, 60), np.float32)
        self.sdeck_top_b = mk("self_deck_top", (B, 60), np.float32)
        self.odeck_id_b = mk("opp_deck_id", (B, 60), np.int64); self.odeck_m_b = mk("opp_deck_mask", (B, 60), np.float32)
        self.odeck_fl_b = mk("opp_deck_flag", (B, 60), np.float32)
        self.sprize_id_b = mk("self_prize_id", (B, 6), np.int64); self.sprize_m_b = mk("self_prize_mask", (B, 6), np.float32)
        self.oprize_id_b = mk("opp_prize_id", (B, 6), np.int64); self.oprize_m_b = mk("opp_prize_mask", (B, 6), np.float32)
        self.shand_id_b = mk("self_hand_id", (B, 30), np.int64); self.shand_m_b = mk("self_hand_mask", (B, 30), np.float32)
        self.ohand_id_b = mk("opp_hand_id", (B, 30), np.int64); self.ohand_m_b = mk("opp_hand_mask", (B, 30), np.float32)
        self.ohand_fl_b = mk("opp_hand_flag", (B, 30), np.float32)   # v2.2 certainty flag
        self.sdisc_id_b = mk("self_discard_id", (B, 60), np.int64); self.sdisc_m_b = mk("self_discard_mask", (B, 60), np.float32)
        self.odisc_id_b = mk("opp_discard_id", (B, 60), np.int64); self.odisc_m_b = mk("opp_discard_mask", (B, 60), np.float32)
        self.sid_b = mk("stadium_id", (B, 2), np.int64); self.smask_b = mk("stadium_mask", (B, 2), np.float32)
        self.s_top_b = mk("self_unit_top_id", (B, 9), np.int64)
        self.s_pre_b = mk("self_unit_preevo_id", (B, 9, 2), np.int64)
        self.s_tool_b = mk("self_unit_tool_id", (B, 9, 4), np.int64)
        self.s_en_b = mk("self_unit_energy_id", (B, 9, 4), np.int64)
        self.s_attr_b = mk("self_unit_attr", (B, 9, 24), np.float32)
        self.s_mask_b = mk("self_unit_mask", (B, 9), np.float32)
        self.o_top_b = mk("opp_unit_top_id", (B, 9), np.int64)
        self.o_pre_b = mk("opp_unit_preevo_id", (B, 9, 2), np.int64)
        self.o_tool_b = mk("opp_unit_tool_id", (B, 9, 4), np.int64)
        self.o_en_b = mk("opp_unit_energy_id", (B, 9, 4), np.int64)
        self.o_attr_b = mk("opp_unit_attr", (B, 9, 24), np.float32)
        self.o_mask_b = mk("opp_unit_mask", (B, 9), np.float32)
        self.oattr_b = mk("opt_attr", (B, MAX_OPTIONS, OPT_STRUCT + NFX), np.float32)
        self.overb_b = mk("opt_verb", (B, MAX_OPTIONS), np.int64)
        self.oaid_b = mk("opt_attack_id", (B, MAX_OPTIONS), np.int64)
        self.ospos_b = mk("opt_src_pos", (B, MAX_OPTIONS), np.int64)
        self.otpos_b = mk("opt_tgt_pos", (B, MAX_OPTIONS), np.int64)
        self.oscard_b = mk("opt_src_card", (B, MAX_OPTIONS), np.int64)
        self.otcard_b = mk("opt_tgt_card", (B, MAX_OPTIONS), np.int64)

    def encode(self, int[::1] buf, int nbuf, int unk,
               list self_deck_ids, list top_ids, dict opp_deck_counts, list opp_hand_ids, list opp_hand_flags,
               float offense_buff, object picked_idx,
               float[::1] tera_bits, object ability_slots,
               long long[::1] def_serials, float[::1] def_reductions,
               long long[::1] offs, float[:, ::1] af, float[:, ::1] fx, double[:, ::1] would_ko, int row):
        _fill_views(buf, nbuf, unk, self_deck_ids, top_ids, opp_deck_counts, opp_hand_ids, opp_hand_flags, offense_buff, picked_idx,
                    tera_bits, ability_slots, def_serials, def_reductions, offs, af, fx, would_ko,
                    self.cls_b[row], self.stype_b[row], self.sctx_b[row], self.effid_b[row], self.effm_b[row],
                    self.sdeck_id_b[row], self.sdeck_m_b[row], self.sdeck_fl_b[row], self.sdeck_top_b[row],
                    self.odeck_id_b[row], self.odeck_m_b[row], self.odeck_fl_b[row],
                    self.sprize_id_b[row], self.sprize_m_b[row], self.oprize_id_b[row], self.oprize_m_b[row],
                    self.shand_id_b[row], self.shand_m_b[row], self.ohand_id_b[row], self.ohand_m_b[row],
                    self.ohand_fl_b[row],
                    self.sdisc_id_b[row], self.sdisc_m_b[row], self.odisc_id_b[row], self.odisc_m_b[row],
                    self.sid_b[row], self.smask_b[row],
                    self.s_top_b[row], self.s_pre_b[row], self.s_tool_b[row], self.s_en_b[row], self.s_attr_b[row], self.s_mask_b[row],
                    self.o_top_b[row], self.o_pre_b[row], self.o_tool_b[row], self.o_en_b[row], self.o_attr_b[row], self.o_mask_b[row],
                    self.oattr_b[row], self.overb_b[row], self.oaid_b[row],
                    self.ospos_b[row], self.otpos_b[row], self.oscard_b[row], self.otcard_b[row])


# ===========================================================================
# M4 (2026-07-10): native GameTracker.update_native + binary_to_obs.
# Collect-step profile (175us/dec): tracker walk 63us + binary_to_obs parse ~25us were the two
# biggest PYTHON slices left on the native path. Both are ported here 1:1 -- the tracker STATE
# stays the tracker's own Python dicts (serials/zone/card_of/zone_turn/hand_epoch), only the
# buffer walk + the belief-update inner ops run compiled. MUST stay mirrored with
# rl/encoding.py GameTracker.update_native / binary_to_obs (validated by m3_tracker_validate).
# ===========================================================================

cdef inline void _t_add(dict serials, dict card_of, object cid, object ser):
    # mirrors GameTracker._add for a KNOWN-valid pi: `if cid and ser is not None`
    cdef object s = serials.get(cid)
    if s is None:
        s = set()
        serials[cid] = s
    (<set>s).add(ser)
    card_of[ser] = cid


cdef inline void _t_setzone(dict zone, dict zt, dict he, object ser, int area,
                            object cur_turn, object be):
    # mirrors GameTracker._set_zone body (pi/None checks done by the caller)
    zone[ser] = area
    if area == 2:
        zt[ser] = cur_turn
        he[ser] = be
    else:
        if ser in zt:
            del zt[ser]
        if ser in he:
            del he[ser]


cdef enum:
    ACAP = 64            # per-unit attached-list cap (energyCards/tools/preEvo <= game limits)


cdef int _rd_into(int[::1] b, int i, int* ids, int* sers, int* pis, int* out_n):
    # cardlist -> parallel C arrays (cap ACAP; cursor always fully advances)
    cdef int cnt = b[i], k, n = 0
    i += 1
    if cnt > 0:
        for k in range(cnt):
            if n < ACAP:
                ids[n] = b[i]; sers[n] = b[i + 1]; pis[n] = b[i + 2]
                n += 1
            i += 3
    out_n[0] = n
    return i


def tracker_update_native(tr, int[::1] b, dict def_buff_attacks, dict off_buff_cards):
    """Compiled twin of GameTracker.update_native(arr). Same walk order, same belief rules,
    same call sequence into the tracker's dicts -> identical end state."""
    cdef int turn = b[0], me = b[2], opp = 1 - me
    cdef int i = 9, p, grp, c, k, n, present, ne, area
    cdef int t, pid, cid, ser, fA, tA, aid
    cdef int ids[ACAP]
    cdef int sers[ACAP]
    cdef int pis[ACAP]
    cdef int nlist
    tr._cur_turn = turn
    if tr._buff_turn is None or <object>turn != tr._buff_turn:
        tr._buff_turn = turn
        tr.opp_def_buff = {}
        tr.offense_buff = 0.0
    cdef list serials = [tr.serials[0], tr.serials[1]]
    cdef list zone = [tr.zone[0], tr.zone[1]]
    cdef list card_of = [tr.card_of[0], tr.card_of[1]]
    cdef list zone_turn = [tr.zone_turn[0], tr.zone_turn[1]]
    cdef list hand_epoch = [tr.hand_epoch[0], tr.hand_epoch[1]]
    cdef dict blind_exits = tr.blind_exits
    cdef list top_ids = tr.top_ids                           # v2.3 deck-top (self side)
    cdef object cur_turn = tr._cur_turn
    cdef dict opp_def_buff
    cdef object mag, ob

    i += 24                                                  # B: 2 x 12 per-player scalars
    cdef int cnt
    cnt = b[i]; i += 1 + (cnt * 3 if cnt > 0 else 0)         # C: self hand (tracker-unused)
    cdef int i_d0 = i
    cnt = b[i]; i += 1 + (cnt * 3 if cnt > 0 else 0)
    cdef int i_d1 = i
    cnt = b[i]; i += 1 + (cnt * 3 if cnt > 0 else 0)
    cdef int i_st = i
    cnt = b[i]; i += 1 + (cnt * 3 if cnt > 0 else 0)
    cdef int i_units = i
    for p in range(2):                                       # D: units (skip walk; re-walked below)
        for grp in range(2):
            c = b[i]; i += 1
            for k in range(c):
                present = b[i]; i += 1
                if present == 0:
                    continue
                i += 6
                ne = b[i]; i += 1; i += ne
                cnt = b[i]; i += 1 + (cnt * 3 if cnt > 0 else 0)
                cnt = b[i]; i += 1 + (cnt * 3 if cnt > 0 else 0)
                cnt = b[i]; i += 1 + (cnt * 3 if cnt > 0 else 0)
    i += 6                                                   # E: select scalars
    cnt = b[i]; i += 1; i += cnt * 6                         # options
    cnt = b[i]; i += 1
    if cnt >= 0:
        i += cnt * 3                                         # deck
    i += 4 if b[i] else 1                                    # contextCard
    i += 4 if b[i] else 1                                    # effect
    cnt = b[i]; i += 1
    if cnt >= 0:
        i += cnt * 3                                         # F: looking
    cdef int lg_n = b[i]
    i += 1

    # (1) reveal logs -- same order/branch structure as the Python twin (incl. its elif chain)
    for k in range(lg_n):
        t = b[i]; pid = b[i+1]; cid = b[i+2]; ser = b[i+3]
        fA = b[i+4]; tA = b[i+5]; aid = b[i+6]
        i += 7
        if t == 6 or (10 <= t <= 15):                        # _REVEAL_TYPES = {6,10..15}
            if cid != 0 and cid != -1 and ser != -1 and (pid == 0 or pid == 1):
                _t_add(<dict>serials[pid], <dict>card_of[pid], cid, ser)
        if t == 6:
            if (pid == 0 or pid == 1) and ser != -1 and tA != -1:
                _t_setzone(<dict>zone[pid], <dict>zone_turn[pid], <dict>hand_epoch[pid],
                           ser, tA, cur_turn, blind_exits[pid])
        if t == 7 and fA == 2:
            if pid == 0 or pid == 1:
                blind_exits[pid] = <object>blind_exits[pid] + 1
        if pid == me:                                        # v2.3 deck-top: GameTracker._update_top
            if t == 6 or t == 7:
                if fA == 1:
                    del top_ids[:]                           # non-draw exit from our deck
                elif tA == 1:
                    if t == 7 or cid <= 0:
                        del top_ids[:]                       # redacted move into our deck
                    else:
                        top_ids.insert(0, cid)
                        del top_ids[MAX_TOP:]
            elif t == 4:                                     # Draw: off the top, id visible to us
                if len(top_ids) > 0:
                    if cid > 0 and cid != <int>top_ids[0]:
                        tr.top_desync = <object>tr.top_desync + 1
                        del top_ids[:]                       # desync -> distrust the stack
                    else:
                        tr.top_hits = <object>tr.top_hits + 1
                        del top_ids[0]
            elif t == 0 or t == 5 or t == 13:
                del top_ids[:]                               # Shuffle / redacted draw / Devolve
        if t == 10:
            if (pid == 0 or pid == 1) and ser != -1 and (<dict>zone[pid]).get(ser) == 2:
                (<dict>zone[pid])[ser] = 1
                (<dict>zone_turn[pid]).pop(ser, None)
                (<dict>hand_epoch[pid]).pop(ser, None)
        if t == 15 and pid == opp and ser != -1:
            mag = def_buff_attacks.get(None if aid == -1 else aid)
            if mag:
                opp_def_buff = tr.opp_def_buff
                opp_def_buff[ser] = mag
        elif t == 10 and pid == me:
            ob = off_buff_cards.get(None if cid == -1 else cid)
            if ob:
                tr.offense_buff = tr.offense_buff + ob

    # (2) currently-visible public cards (ground truth; raw ints, NO -1->None mapping -- the
    #     Python twin passes them straight to _add/_set_zone, whose only guards are pi/cid/None)
    i = _rd_into(b, i_d0, ids, sers, pis, &nlist)
    for k in range(nlist):
        if ids[k] != 0:
            _t_add(<dict>serials[0], <dict>card_of[0], ids[k], sers[k])
        _t_setzone(<dict>zone[0], <dict>zone_turn[0], <dict>hand_epoch[0],
                   sers[k], 3, cur_turn, blind_exits[0])
    i = _rd_into(b, i_d1, ids, sers, pis, &nlist)
    for k in range(nlist):
        if ids[k] != 0:
            _t_add(<dict>serials[1], <dict>card_of[1], ids[k], sers[k])
        _t_setzone(<dict>zone[1], <dict>zone_turn[1], <dict>hand_epoch[1],
                   sers[k], 3, cur_turn, blind_exits[1])
    # units: attached lists are read ec, tl, pe but PROCESSED pe+tl+ec (list-concat order of the twin)
    cdef int pid_c, pser
    cdef int ec_i[ACAP]
    cdef int ec_s[ACAP]
    cdef int ec_p[ACAP]
    cdef int tl_i[ACAP]
    cdef int tl_s[ACAP]
    cdef int tl_p[ACAP]
    cdef int pe_i[ACAP]
    cdef int pe_s[ACAP]
    cdef int pe_p[ACAP]
    cdef int n_ec, n_tl, n_pe
    i = i_units
    for p in range(2):
        for grp in range(2):
            area = 4 if grp == 0 else 5
            c = b[i]; i += 1
            for k in range(c):
                present = b[i]; i += 1
                if present == 0:
                    continue
                pid_c = b[i]; pser = b[i+1]; i += 6
                ne = b[i]; i += 1; i += ne
                i = _rd_into(b, i, ec_i, ec_s, ec_p, &n_ec)
                i = _rd_into(b, i, tl_i, tl_s, tl_p, &n_tl)
                i = _rd_into(b, i, pe_i, pe_s, pe_p, &n_pe)
                if pid_c != 0:
                    _t_add(<dict>serials[p], <dict>card_of[p], pid_c, pser)
                _t_setzone(<dict>zone[p], <dict>zone_turn[p], <dict>hand_epoch[p],
                           pser, area, cur_turn, blind_exits[p])
                for n in range(n_pe):
                    if pe_i[n] != 0:
                        _t_add(<dict>serials[p], <dict>card_of[p], pe_i[n], pe_s[n])
                    _t_setzone(<dict>zone[p], <dict>zone_turn[p], <dict>hand_epoch[p],
                               pe_s[n], area, cur_turn, blind_exits[p])
                for n in range(n_tl):
                    if tl_i[n] != 0:
                        _t_add(<dict>serials[p], <dict>card_of[p], tl_i[n], tl_s[n])
                    _t_setzone(<dict>zone[p], <dict>zone_turn[p], <dict>hand_epoch[p],
                               tl_s[n], area, cur_turn, blind_exits[p])
                for n in range(n_ec):
                    if ec_i[n] != 0:
                        _t_add(<dict>serials[p], <dict>card_of[p], ec_i[n], ec_s[n])
                    _t_setzone(<dict>zone[p], <dict>zone_turn[p], <dict>hand_epoch[p],
                               ec_s[n], area, cur_turn, blind_exits[p])
    i = _rd_into(b, i_st, ids, sers, pis, &nlist)            # stadium: owner = entry playerIndex
    for k in range(nlist):
        if (pis[k] == 0 or pis[k] == 1):
            if ids[k] != 0:
                _t_add(<dict>serials[pis[k]], <dict>card_of[pis[k]], ids[k], sers[k])
            _t_setzone(<dict>zone[pis[k]], <dict>zone_turn[pis[k]], <dict>hand_epoch[pis[k]],
                       sers[k], 7, cur_turn, blind_exits[pis[k]])

    # (3) hand-count invariant (rare contradiction branch stays Python)
    tr._enforce_hand_count(0, b[9 + 2])
    tr._enforce_hand_count(1, b[9 + 12 + 2])


def binary_to_obs_native(int[::1] b):
    """Compiled twin of rl/encoding.binary_to_obs: the MINIMAL control-flow obs dict."""
    cdef int turn = b[0], me = b[2], result = b[8]
    cdef int i = 9, p, k, cnt, ne, present, c, u, l3
    cdef int p0 = b[9 + 3], p1 = b[9 + 12 + 3]
    i += 24
    for p in range(4):                                       # C: hand, discard0, discard1, stadium
        cnt = b[i]; i += 1
        if cnt > 0:
            i += cnt * 3
    for p in range(2):                                       # D: units
        for k in range(2):
            c = b[i]; i += 1
            for u in range(c):
                present = b[i]; i += 1
                if present == 0:
                    continue
                i += 6
                ne = b[i]; i += 1; i += ne
                for l3 in range(3):
                    cnt = b[i]; i += 1
                    if cnt > 0:
                        i += cnt * 3
    cdef int minC = b[i + 2], maxC = b[i + 3]
    i += 6
    cdef int nopt = b[i]
    i += 1
    cdef list opts = []
    cdef dict o
    cdef int t
    for k in range(nopt):
        t = b[i]
        o = {"type": t}
        if t == 0:
            o["number"] = b[i+1]
        elif t == 3:
            o["area"] = b[i+1]; o["index"] = b[i+2]; o["playerIndex"] = b[i+3]
        elif t == 4:
            o["area"] = b[i+1]; o["index"] = b[i+2]; o["playerIndex"] = b[i+3]; o["toolIndex"] = b[i+4]
        elif t == 5:
            o["area"] = b[i+1]; o["index"] = b[i+2]; o["playerIndex"] = b[i+3]; o["energyIndex"] = b[i+4]
        elif t == 6:
            o["area"] = b[i+1]; o["index"] = b[i+2]; o["playerIndex"] = b[i+3]
            o["energyIndex"] = b[i+4]; o["count"] = b[i+5]
        elif t == 7:
            o["index"] = b[i+1]
        elif t == 8 or t == 9:
            o["area"] = b[i+1]; o["index"] = b[i+2]; o["inPlayArea"] = b[i+3]; o["inPlayIndex"] = b[i+4]
        elif t == 10 or t == 11:
            o["area"] = b[i+1]; o["index"] = b[i+2]
        elif t == 13:
            o["attackId"] = b[i+1]
        elif t == 15:
            o["cardId"] = b[i+1]; o["serial"] = b[i+2]
        elif t == 16:
            o["specialConditionType"] = b[i+1]
        opts.append(o)
        i += 6
    return {
        "current": {"result": result, "yourIndex": me, "turn": turn,
                    "players": [{"prize": [None] * p0}, {"prize": [None] * p1}]},
        "select": {"option": opts, "maxCount": maxC, "minCount": minC},
    }


def copies_hidden_native(dict serials_pi, dict zone_pi):
    """Compiled twin of GameTracker.copies_hidden_for's body for one player: {cid: (n, hidden)}
    where hidden counts serials whose last-known zone is NOT public (DISCARD/ACTIVE/BENCH/STADIUM
    = 3/4/5/7). Same dict/set iteration order as the Python twin -> identical output dict order
    (the opp_deck stream expansion is order-sensitive)."""
    cdef dict out = {}
    cdef object cid, sset, ser, z
    cdef int h
    for cid, sset in serials_pi.items():
        h = 0
        for ser in <set>sset:
            z = zone_pi.get(ser)
            if not (z == 3 or z == 4 or z == 5 or z == 7):
                h += 1
        out[cid] = (len(<set>sset), h)
    return out
