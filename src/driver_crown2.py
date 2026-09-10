"""CROWN2 driver: exact bit-level codec.

Stack (encoder decisions in Python via proven probe modules; streaming decode + bit
packing in C libcrown2.so):
  Q-family   (probe_b4 b/c/g/j): sign-flipped LOCO-365 keys + auto-K quantile groups
             (WIDE-K) + per-group best-of-E16 (probe_b5_c) + per-group Huff/Golomb/rANS
             3-way backend (probe_b5_d) + Golomb-only shift adapters (B8 NOTE re-probe
             under Golomb: bias is Huffman-translation-invariant, Golomb-visible).
  GRID-family (probe_b8 M-THR): 8 codec-constant energy grids, zero map side, same
             E16 + 3-way backends. Per-channel MDL family gate Q-vs-GRID.
  C decoders mirror probe_b5_d.prove_recon scalar sim (E16 formulas bit-verified vs
  predictors_X) with zero-border rules; see crown2.c header + CROWN2_FORMAT.md.

Border convention (encoder==decoder, causal): zero border; row0 copies left;
col0 copies top (NOT edge-replicate). Alphabet +-1024 assert-loud, never clip.
Runs: dropped (CROWN ablative). numpy+PIL+gcc+ctypes only, no torch.
"""
import ctypes, math, time, sys, os
import numpy as np
from PIL import Image
sys.path.insert(0, "/tmp/opencode/autocompress/csrc")
sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
os.chdir("/tmp/opencode/autocompress/experiments")
import probe_b4_a as A
from probe_b5_c import predictors_X, prepX, E16
from probe_b5_d import golomb_best
from driver import canon_tables

LIB = ctypes.CDLL("/tmp/opencode/autocompress/csrc/libcrown2.so")
c_u8 = ctypes.c_uint8; c_i16 = ctypes.c_int16; c_u16 = ctypes.c_uint16
c_i32 = ctypes.c_int32; c_u32 = ctypes.c_uint32; c_i64 = ctypes.c_int64
LIB.pack_syms.argtypes = [ctypes.POINTER(c_i16), ctypes.POINTER(c_u8), ctypes.c_int,
    ctypes.POINTER(c_u32), ctypes.POINTER(c_u8), ctypes.POINTER(c_u8), ctypes.c_size_t]
LIB.pack_syms.restype = ctypes.c_size_t
LIB.golomb_pack.argtypes = [ctypes.POINTER(c_i16), ctypes.POINTER(c_u8), ctypes.POINTER(c_u8),
    ctypes.c_int, ctypes.POINTER(c_u8), ctypes.c_size_t]
LIB.golomb_pack.restype = ctypes.c_size_t
LIB.rans_norm.argtypes = [ctypes.POINTER(c_i64), ctypes.c_int, ctypes.POINTER(c_i32),
    ctypes.POINTER(c_u16), ctypes.POINTER(c_i32)]
LIB.rans_norm.restype = None
LIB.rans_encode.argtypes = [ctypes.POINTER(c_i16), ctypes.c_int, ctypes.POINTER(c_i32),
    ctypes.POINTER(c_u16), ctypes.POINTER(c_i32), ctypes.POINTER(c_u8)]
LIB.rans_encode.restype = ctypes.c_size_t
LIB.rans_slots.argtypes = [ctypes.POINTER(c_u16), ctypes.POINTER(c_i32), ctypes.c_int,
    ctypes.POINTER(c_i32)]
LIB.rans_decode.argtypes = [ctypes.POINTER(c_u8), ctypes.c_size_t, ctypes.c_int,
    ctypes.POINTER(c_i32), ctypes.POINTER(c_u16), ctypes.POINTER(c_i32), ctypes.c_int,
    ctypes.POINTER(c_i32), ctypes.POINTER(c_i16)]
LIB.rans_decode.restype = ctypes.c_int
LIB.crown2_decode_ch.argtypes = [ctypes.POINTER(c_u8), ctypes.c_size_t,
    ctypes.POINTER(c_u8), ctypes.c_size_t, ctypes.POINTER(c_i16), ctypes.POINTER(c_i32),
    ctypes.c_int, ctypes.POINTER(c_u32), ctypes.POINTER(c_u8), ctypes.POINTER(c_u8),
    ctypes.POINTER(c_u8), ctypes.POINTER(c_u8), ctypes.POINTER(c_u8), ctypes.POINTER(ctypes.c_int8),
    ctypes.POINTER(c_u8), ctypes.POINTER(c_u8), ctypes.c_int, ctypes.POINTER(c_i32),
    ctypes.POINTER(c_i16), ctypes.POINTER(c_u8), ctypes.POINTER(c_u8), ctypes.c_int,
    ctypes.POINTER(c_i32), ctypes.POINTER(c_i16), ctypes.c_int, ctypes.c_int]
LIB.crown2_decode_ch.restype = ctypes.c_int
LIB.ycocg_inv.argtypes = [ctypes.POINTER(c_i16), ctypes.c_int, ctypes.c_int, ctypes.c_char_p]

EXPID = {n: i for i, n in enumerate(E16)}
WIDE = (2, 3, 4, 6, 9, 12, 18, 27, 36, 48, 64)
WIDE_IDX = {k: i for i, k in enumerate(WIDE)}
GRIDS = {
    0: np.array([4, 12, 28, 60, 120], dtype=np.int64),
    1: np.array([2, 6, 16, 40, 100], dtype=np.int64),
    2: np.array([8, 24, 64, 160, 400], dtype=np.int64),
    3: np.array([1, 3, 8, 24, 80], dtype=np.int64),
    4: np.array([3, 9, 20, 48, 110], dtype=np.int64),
    5: np.array([6, 18, 44, 100, 220], dtype=np.int64),
    6: np.array([1, 2, 5, 16, 48], dtype=np.int64),
    7: np.array([5, 15, 36, 80, 160], dtype=np.int64),
}
RES_MAX = 1024

IMAGES = [
    "/tmp/opencode/autocompress/experiments/real_photos/kodim01.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim02.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim05.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim07.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim13.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim19.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim23.png",
]
JXL_E3 = {"kodim01.png": 3.36, "kodim02.png": 3.06, "kodim05.png": 3.51,
          "kodim07.png": 2.73, "kodim13.png": 3.91, "kodim19.png": 3.22, "kodim23.png": 2.81}


def check_range(arr, where):
    m = int(np.abs(arr).max()) if arr.size else 0
    assert m <= RES_MAX, f"ALPHABET VIOLATION {where}: |r|max={m} > 1024 (abort, never clip)"


def huff_cost(g):
    """(total incl 16+A*24 table, data, A) for int array g; empty -> zeros."""
    g = np.asarray(g).reshape(-1)
    if g.size == 0:
        return 0, 0, 0
    uv, cn = np.unique(g, return_counts=True)
    d = A.huff_bits(cn.tolist())
    return d + 16 + len(cn) * 24, d, len(cn)


def golomb_cost(g):
    """Best (data+4k-side, k, adopted_bias, bias_side) with G-bias gate d in [-4,3]."""
    g = np.asarray(g, dtype=np.int64).reshape(-1)
    if g.size == 0:
        return 0, 0, 0, 0
    gd0, k0 = golomb_best(g)
    best = (gd0 + 4, k0, 0)
    for d in range(-4, 4):
        gd, k = golomb_best(g - d)
        if gd + 4 + 3 < best[0]:
            best = (gd + 4 + 3, k, d)
    tot, k, d = best
    return tot, k, d, (3 if d != 0 else 0)


def rans_cost(g):
    """Exact rANS bits incl 16+A*32 table + 64b count/paylen framing (decode-verified)."""
    g = np.ascontiguousarray(np.asarray(g).reshape(-1).astype(np.int16))
    N = g.size
    if N == 0:
        return 0, None
    uv = np.unique(g)
    syms = np.array([int(v) + 1024 for v in uv], dtype=np.int32)
    Ad = len(syms)
    hist = np.zeros(2049, dtype=np.int64)
    for v in uv:
        hist[int(v) + 1024] = int((g == v).sum())
    freq = np.zeros(Ad, dtype=np.uint16)
    cum = np.zeros(Ad, dtype=np.int32)
    LIB.rans_norm(hist.ctypes.data_as(ctypes.POINTER(c_i64)), Ad,
                  syms.ctypes.data_as(ctypes.POINTER(c_i32)),
                  freq.ctypes.data_as(ctypes.POINTER(c_u16)),
                  cum.ctypes.data_as(ctypes.POINTER(c_i32)))
    lut = np.full(2049, -1, dtype=np.int32)
    for i, s in enumerate(syms):
        lut[s] = i
    out = np.zeros(N * 3 + 16, dtype=np.uint8)
    n = LIB.rans_encode(g.ctypes.data_as(ctypes.POINTER(c_i16)), N,
                        lut.ctypes.data_as(ctypes.POINTER(c_i32)),
                        freq.ctypes.data_as(ctypes.POINTER(c_u16)),
                        cum.ctypes.data_as(ctypes.POINTER(c_i32)),
                        out.ctypes.data_as(ctypes.POINTER(c_u8)))
    assert n > 0
    slot = np.zeros(1 << 14, dtype=np.int32)
    LIB.rans_slots(freq.ctypes.data_as(ctypes.POINTER(c_u16)),
                   cum.ctypes.data_as(ctypes.POINTER(c_i32)), Ad,
                   slot.ctypes.data_as(ctypes.POINTER(c_i32)))
    dec = np.zeros(N, dtype=np.int16)
    rc = LIB.rans_decode(out.ctypes.data_as(ctypes.POINTER(c_u8)), n, N,
                         syms.ctypes.data_as(ctypes.POINTER(c_i32)),
                         freq.ctypes.data_as(ctypes.POINTER(c_u16)),
                         cum.ctypes.data_as(ctypes.POINTER(c_i32)), Ad,
                         slot.ctypes.data_as(ctypes.POINTER(c_i32)),
                         dec.ctypes.data_as(ctypes.POINTER(c_i16)))
    assert rc == 0 and np.array_equal(dec, g), "rANS roundtrip FAIL"
    return 16 + Ad * 32 + 64 + n * 8, (bytes(out[:n]), syms, freq, cum)


def quantile_groups(key, rfM, K):
    """Greedy equal-pixel quantile partition (probe_b4_j exact semantics)."""
    uk, cn = np.unique(key, return_counts=True)
    ma = np.array([np.abs(rfM[key == v]).mean() for v in uk])
    order = np.argsort(ma, kind="stable")
    uks, cns = uk[order], cn[order]
    tot = cns.sum()
    tgt = tot / K
    groups, cur, acc = [], [], 0
    for u, c in zip(uks, cns):
        cur.append(u)
        acc += c
        if acc >= tgt and len(groups) < K - 1:
            groups.append(cur)
            cur, acc = [], 0
    groups.append(cur)
    return groups


def solve_groups(group_vals, pred_bits=4, back_bits=2, use_rans=True):
    """Per-group joint (expert x backend) exact select.

    group_vals: list of dicts {expert_name: int array}. Returns (total_bits, wins)
    where wins[g] = (cost, expert, backend 0/1/2, k, bias, rans_blob_or_None).
    H/G costs exact; R exact via rans_cost on the winning expert only (probe_b5_d).
    Empty groups -> (0, expert0, H, 0, 0, None), cost 0, no table shipped.
    """
    total = 0
    wins = []
    for gd in group_vals:
        bw = None
        for n, g in gd.items():
            g = np.asarray(g).reshape(-1)
            if g.size == 0:
                ht, hk, hd = 0, 0, 0
                gt, gk, gdlt = 0, 0, 0
            else:
                ht, _, _ = huff_cost(g)
                gt, gk, gdlt, _ = golomb_cost(g)
            if bw is None or ht < bw[0]:
                bw = [ht, n, 0, 0, 0]
            if gt < bw[0]:
                bw = [gt, n, 1, gk, gdlt]
        cost2, bn, bb, bk, bd = bw
        if use_rans:
            g = np.asarray(gd[bn]).reshape(-1)
            if g.size:
                rt, blob = rans_cost(g)
                if rt < cost2:
                    bw = [rt, bn, 2, 0, 0]
                    wins.append((rt, bn, 2, 0, 0, blob))
                    total += rt
                    continue
        # re-resolve G k/bias for winner (cheap, exact)
        g = np.asarray(gd[bn]).reshape(-1)
        if bw[2] == 1 and g.size:
            gt, gk, gdlt, _ = golomb_cost(g)
            bw = [gt, bn, 1, gk, gdlt]
        wins.append((bw[0], bw[1], bw[2], bw[3], bw[4], None))
        total += bw[0]
    return total, wins


def encode_channel_q(ch, D, RF, topk=3):
    """Q-family: WIDE-K search (2-way) -> top-k 3-way finalize. Returns plan dict."""
    key, s = D["key"], D["s"]
    rfM = RF["MED"]
    uk_all = np.unique(key)
    cands = []
    for K in WIDE:
        groups = quantile_groups(key, rfM, K)
        gbits = math.ceil(math.log2(K))
        mapside = 16 + 736 + len(uk_all) * gbits  # hdr16(fake, real=16b below) + mask + gids
        gv = []
        for gkeys in groups:
            sel = np.isin(key, np.array(gkeys))
            gv.append({n: RF[n][sel] for n in E16})
        sub, wins2 = solve_groups(gv, use_rans=False)
        cands.append((mapside + len(groups) * 6 + sub, K, groups, wins2))
    cands.sort(key=lambda t: t[0])
    best = None
    for _, K, groups, _ in cands[:topk]:
        gv = []
        for gkeys in groups:
            sel = np.isin(key, np.array(gkeys))
            gv.append({n: RF[n][sel] for n in E16})
        sub, wins = solve_groups(gv, use_rans=True)
        gbits = math.ceil(math.log2(K))
        mapside = 16 + 736 + len(uk_all) * gbits
        # real framing extras: kvals 4b/G-group, dbias 3b/shifted-G-group, R framing counted in rans_cost
        xtra = sum(4 for w in wins if w[2] == 1) + sum(3 for w in wins if w[2] == 1 and w[4] != 0)
        t = mapside + len(groups) * 6 + xtra + sub
        if best is None or t < best[0]:
            best = (t, K, groups, wins)
    return {"family": 0, "K": best[1], "groups": best[2], "wins": best[3], "grid": 0}


def encode_channel_grid(ch, D, RU):
    """GRID-family: 8 energy grids (2-way select), 3-way finalize on winner."""
    a, b, c = D["a"], D["b"], D["c"]
    e = (np.abs(a - c) + np.abs(b - c)).astype(np.int64)
    best = None
    for gi, thr in GRIDS.items():
        gg = np.digitize(e.ravel(), thr, right=True)
        gv = []
        for c_ in range(6):
            sel = (gg == c_).reshape(e.shape)
            gv.append({n: RU[n][sel] for n in E16})
        sub, wins2 = solve_groups(gv, use_rans=False)
        t = 16 + 8 + len(gv) * 6 + sub  # hdr + occmask + meta
        if best is None or t < best[0]:
            best = (t, gi, gg.reshape(e.shape), wins2)
    _, gi, gg, _ = best
    gv = []
    for c_ in range(6):
        sel = (gg == c_)
        gv.append({n: RU[n][sel] for n in E16})
    sub, wins = solve_groups(gv, use_rans=True)
    xtra = sum(4 for w in wins if w[2] == 1) + sum(3 for w in wins if w[2] == 1 and w[4] != 0)
    t = 16 + 8 + len(gv) * 6 + xtra + sub
    return {"family": 1, "K": 6, "groups": None, "wins": wins, "grid": gi, "gmap": gg}


class BitWriter:
    def __init__(self):
        self.acc = 0
        self.nb = 0
        self.out = bytearray()

    def put(self, v, n):
        self.acc = (self.acc << n) | (v & ((1 << n) - 1))
        self.nb += n
        while self.nb >= 8:
            self.nb -= 8
            self.out.append((self.acc >> self.nb) & 0xFF)
            self.acc &= ((1 << self.nb) - 1) if self.nb else 0

    def flush(self):
        if self.nb:
            self.out.append((self.acc << (8 - self.nb)) & 0xFF)
            self.acc, self.nb = 0, 0
        return bytes(self.out)


def assemble_channel(plan, D, ch, H, W):
    """Build exact channel bytes + decode-side arrays. Returns (blob, meta)."""
    N = H * W
    fam = plan["family"]
    wins = plan["wins"]
    ng = len(wins)
    # per-pixel expanded group + residual + sign
    if fam == 0:
        key, s = D["key"], D["s"]
        cmap = np.zeros(729, dtype=np.uint8)
        for gi, gkeys in enumerate(plan["groups"]):
            for u in gkeys:
                cmap[int(u)] = gi
        expmap = cmap[key.reshape(-1)].reshape(H, W)
        P = D["P"]
        resplane = np.zeros((H, W), dtype=np.int32)
        for gi, w in enumerate(wins):
            _, bn, _, _, _, _ = w
            m = (expmap == gi)
            r = (ch - P[bn]).astype(np.int32)
            resplane[m] = r[m]
        check_range(resplane, "Q residuals")
        symplane = (s.astype(np.int32) * resplane)
        check_range(symplane, "Q flipped")
        signplane = s.astype(np.int8)
        uk = np.unique(key)
        gbits = math.ceil(math.log2(plan["K"]))
    else:
        a, b, c = D["a"], D["b"], D["c"]
        gg = plan["gmap"]
        expmap = gg
        P = D["P"]
        resplane = np.zeros((H, W), dtype=np.int32)
        for gi, w in enumerate(wins):
            _, bn, _, _, _, _ = w
            m = (expmap == gi)
            r = (ch - P[bn]).astype(np.int32)
            resplane[m] = r[m]
        check_range(resplane, "GRID residuals")
        symplane = resplane
        signplane = np.ones((H, W), dtype=np.int8)
        uk, gbits = None, 0
    back = np.array([w[2] for w in wins], dtype=np.uint8)
    # force empty groups to backend H (no table, no payload)
    counts = np.array([(expmap == gi).sum() for gi in range(ng)])
    for gi in range(ng):
        if counts[gi] == 0:
            back[gi] = 0
    # ---- headers ----
    blob = bytearray()
    kidx = WIDE_IDX[plan["K"]] if fam == 0 else 15
    blob.append((fam & 1) | ((plan["grid"] & 7) << 1) | ((kidx & 15) << 4))
    blob.append(ng & 0xFF)
    gsel = np.arange(ng * 2, dtype=np.uint8).reshape(ng, 2)
    gsel[:, 1] = np.arange(ng, dtype=np.uint8)  # IG off: identity
    if fam == 0:
        mask = np.zeros(729, dtype=np.uint8)
        mask[uk] = 1
        blob += np.packbits(mask).tobytes()  # 92B
        assert len(blob) == 2 + 92
        bw = BitWriter()
        for u in sorted(uk.tolist()):
            bw.put(int(cmap[int(u)]), gbits)
        blob += bw.flush()
    else:
        occ = 0
        for gi in range(ng):
            if counts[gi] > 0:
                occ |= (1 << gi)
        blob.append(occ & 0xFF)
    # per-group meta nibbles
    bw = BitWriter()
    for gi, w in enumerate(wins):
        bw.put(EXPID[w[1]], 4)
        bw.put(int(back[gi]), 2)
    blob += bw.flush()
    # kvals (4b) + dbias (3b) for G groups in expanded order
    bwk, bwb = BitWriter(), BitWriter()
    for gi, w in enumerate(wins):
        if back[gi] == 1:
            bwk.put(int(w[3]), 4)
            bwb.put(int(w[4]) & 7, 3)
    blob += bwk.flush() + bwb.flush()
    # ---- payloads ----
    cd = np.zeros((ng * 2049,), np.uint32)
    ln = np.zeros((ng * 2049,), np.uint8)
    flat_exp = expmap.reshape(-1)
    flat_sym = symplane.reshape(-1)
    hmask = back[flat_exp] == 0
    hsyms = flat_sym[hmask].astype(np.int16)
    hkeys = flat_exp[hmask].astype(np.uint8)
    for gi, w in enumerate(wins):
        if back[gi] == 0 and counts[gi] > 0:
            g = flat_sym[flat_exp == gi]
            vals, cn = np.unique(g, return_counts=True)
            harr = np.zeros(2049, dtype=np.int64)
            for v, cc in zip(vals.tolist(), cn.tolist()):
                harr[int(v) + 1024] = int(cc)
            codes = canon_tables(harr)
            for sym, (cc_, ll_) in codes.items():
                cd[gi * 2049 + sym] = cc_
                ln[gi * 2049 + sym] = ll_
            # table
            tbl = bytearray()
            tbl += len(codes).to_bytes(2, "little")
            for sym in sorted(codes):
                tbl += ((sym - 1024) & 0xFFFF).to_bytes(2, "little")
                tbl.append(codes[sym][1])
            blob += bytes(tbl)
    # H payload via C
    if hsyms.size:
        cap = (hsyms.size * 32) // 8 + 1024
        buf = (c_u8 * cap)()
        n = LIB.pack_syms(hsyms.ctypes.data_as(ctypes.POINTER(c_i16)),
                          hkeys.ctypes.data_as(ctypes.POINTER(c_u8)), hsyms.size,
                          cd.ctypes.data_as(ctypes.POINTER(c_u32)),
                          ln.ctypes.data_as(ctypes.POINTER(c_u8)), buf, cap)
        assert n > 0
        blob += int(n).to_bytes(4, "little") + bytes(buf[:n])
    else:
        blob += (0).to_bytes(4, "little")
    # G payload via C (transmitted t = v - d)
    gmask = back[flat_exp] == 1
    kbyexp = np.zeros((ng,), np.uint8)
    dbexp = np.zeros((ng,), np.int8)
    for gi, w in enumerate(wins):
        if back[gi] == 1:
            kbyexp[gi] = w[3]
            dbexp[gi] = w[4]
    if gmask.sum():
        gpix_exp = flat_exp[gmask]
        gvals = flat_sym[gmask].astype(np.int32) - dbexp[gpix_exp].astype(np.int32)
        check_range(gvals, "G transmitted")
        gvals = gvals.astype(np.int16)
        gkeys = gpix_exp.astype(np.uint8)
        cap = int(gvals.size * 300 + 64)  # worst case k=0,q~2048 -> 257B/symbol
        buf = (c_u8 * cap)()
        n = LIB.golomb_pack(gvals.ctypes.data_as(ctypes.POINTER(c_i16)),
                            gkeys.ctypes.data_as(ctypes.POINTER(c_u8)),
                            kbyexp.ctypes.data_as(ctypes.POINTER(c_u8)),
                            gvals.size, buf, cap)
        assert n > 0
        blob += int(n).to_bytes(4, "little") + bytes(buf[:n])
    else:
        blob += (0).to_bytes(4, "little")
    # R groups: count + table + payload
    rsyms_list, goff = [], [0]
    for gi, w in enumerate(wins):
        if back[gi] == 2:
            g = flat_sym[flat_exp == gi].astype(np.int16)
            assert g.size > 0
            pay, syms32, freq, cum = w[5]
            blob += int(g.size).to_bytes(4, "little")
            blob += int(len(syms32)).to_bytes(2, "little")
            for i, sm in enumerate(syms32.tolist()):
                blob += ((sm - 1024) & 0xFFFF).to_bytes(2, "little")
                blob += int(freq[i]).to_bytes(2, "little")
            blob += int(len(pay)).to_bytes(4, "little") + pay
            rsyms_list.append(g)
            goff.append(goff[-1] + g.size)
        else:
            goff.append(goff[-1])
    meta = {"ng": ng, "fam": fam, "back": back, "wins": wins, "cd": cd, "ln": ln,
            "rsyms": np.concatenate(rsyms_list).astype(np.int16) if rsyms_list else np.zeros(0, np.int16),
            "goff": np.array(goff, dtype=np.int32), "gsel": gsel,
            "cmap": cmap if fam == 0 else np.zeros(729, np.uint8),
            "grid": GRIDS[plan["grid"]] if fam == 1 else np.zeros(5, np.int64),
            "kbyexp": kbyexp, "dbexp": dbexp}
    return bytes(blob), meta


def encode_image(rgb):
    H, W, _ = rgb.shape
    N = H * W
    Y, Co, Cg = A.ycocg_fwd(rgb)
    out = bytearray()
    out += b"C2" + H.to_bytes(2, "little") + W.to_bytes(2, "little") + bytes([1, 0])
    metas = []
    for ch in (Y, Co, Cg):
        D = prepX(ch)
        RF = {n: (D["s"] * (ch - D["P"][n]).astype(np.int32)).astype(np.int32) for n in E16}
        for n in E16:
            check_range(RF[n], f"flip {n}")
        RU = {n: (ch - D["P"][n]).astype(np.int32) for n in E16}
        for n in E16:
            check_range(RU[n], f"plain {n}")
        pq = encode_channel_q(ch, D, RF)
        pg = encode_channel_grid(ch, D, RU)
        # exact real-byte compare (assemble both, take smaller) — honest MDL gate
        bq, mq = assemble_channel(pq, D, ch, H, W)
        bg, mg = assemble_channel(pg, D, ch, H, W)
        if len(bg) < len(bq):  # fam bit already inside hdr byte; direct compare
            out += bg
            metas.append(mg)
            fam = "G"
        else:
            out += bq
            metas.append(mq)
            fam = "Q"
        metas[-1]["famname"] = fam
    return bytes(out), metas


def decode_image(blob, H, W):
    assert blob[:2] == b"C2"
    N = H * W
    p = 8
    planes = []
    ch_metas = []
    for _ in range(3):
        b0, ng = blob[p], blob[p + 1]
        p += 2
        fam = b0 & 1
        grid = (b0 >> 1) & 7
        cmap = np.zeros(729, dtype=np.uint8)
        grid_thr = np.zeros(5, dtype=np.int32)
        occ = 0xFF
        if fam == 0:
            mask = np.unpackbits(np.frombuffer(blob[p:p + 92], dtype=np.uint8))[:729]
            p += 92
            gbits = 0
            # need K to know gbits: stored in kidx nibble
            kidx = (b0 >> 4) & 15
            K = WIDE[kidx]
            gbits = math.ceil(math.log2(K))
            na = int(mask.sum())
            nbytes = (na * gbits + 7) // 8
            raw = blob[p:p + nbytes]
            p += nbytes
            acc, nb, qp = 0, 0, 0
            uk = np.where(mask)[0]
            for u in uk:
                while nb < gbits:
                    acc = (acc << 8) | raw[qp]
                    qp += 1
                    nb += 8
                nb -= gbits
                cmap[u] = (acc >> nb) & ((1 << gbits) - 1)
                acc &= ((1 << nb) - 1) if nb else 0
        else:
            occ = blob[p]
            p += 1
            grid_thr = np.array(GRIDS[grid], dtype=np.int32)
        # meta nibbles
        nbytes = (ng * 6 + 7) // 8
        raw = blob[p:p + nbytes]
        p += nbytes
        predid = np.zeros(128, dtype=np.uint8)
        backend = np.zeros(128, dtype=np.uint8)
        acc, nb, qp = 0, 0, 0
        for gi in range(ng):
            while nb < 6:
                acc = (acc << 8) | raw[qp]
                qp += 1
                nb += 8
            nb -= 6
            v = (acc >> nb) & 0x3F
            acc &= ((1 << nb) - 1) if nb else 0
            predid[gi] = (v >> 2) & 15
            backend[gi] = v & 3
        # kvals + dbias for G groups
        nG = int((backend[:ng] == 1).sum())
        kvals = np.zeros(128, dtype=np.uint8)
        dbias = np.zeros(128, dtype=np.int8)
        nbytes = (nG * 4 + 7) // 8
        raw = blob[p:p + nbytes] if nbytes else b""
        p += nbytes
        acc, nb, qp = 0, 0, 0
        for gi in range(ng):
            if backend[gi] == 1:
                while nb < 4:
                    acc = (acc << 8) | raw[qp]
                    qp += 1
                    nb += 8
                nb -= 4
                kvals[gi] = (acc >> nb) & 15
                acc &= ((1 << nb) - 1) if nb else 0
        nbytes = (nG * 3 + 7) // 8
        raw = blob[p:p + nbytes] if nbytes else b""
        p += nbytes
        acc, nb, qp = 0, 0, 0
        for gi in range(ng):
            if backend[gi] == 1:
                while nb < 3:
                    acc = (acc << 8) | raw[qp]
                    qp += 1
                    nb += 8
                nb -= 3
                v = (acc >> nb) & 7
                acc &= ((1 << nb) - 1) if nb else 0
                dbias[gi] = v - 8 if v >= 4 else v
        # occupancy per expanded group
        if fam == 0:
            nonempty = np.zeros(ng, dtype=bool)
            for u in np.where(mask)[0]:
                nonempty[cmap[u]] = True
        else:
            nonempty = np.array([(occ >> gi) & 1 for gi in range(ng)], dtype=bool)
        cd = np.zeros((128 * 2049,), np.uint32)
        ln = np.zeros((128 * 2049,), np.uint8)
        for gi in range(ng):
            if backend[gi] == 0 and nonempty[gi]:
                A_ = int.from_bytes(blob[p:p + 2], "little")
                p += 2
                syms = []
                for _ in range(A_):
                    sv = int.from_bytes(blob[p:p + 2], "little")
                    ll = blob[p + 2]
                    p += 3
                    s = (sv + 1024) & 0xFFFF
                    ln[gi * 2049 + s] = ll
                    syms.append(s)
                syms.sort(key=lambda s: (ln[gi * 2049 + s], s))
                code, prev = 0, 0
                for s in syms:
                    L = int(ln[gi * 2049 + s])
                    code <<= (L - prev)
                    cd[gi * 2049 + s] = code
                    code += 1
                    prev = L
        hn = int.from_bytes(blob[p:p + 4], "little")
        p += 4
        hpay = blob[p:p + hn]
        p += hn
        gn_ = int.from_bytes(blob[p:p + 4], "little")
        p += 4
        gpay = blob[p:p + gn_]
        p += gn_
        # R groups
        rlist, goff = [], [0]
        for gi in range(ng):
            if backend[gi] == 2:
                cnt = int.from_bytes(blob[p:p + 4], "little")
                p += 4
                A_ = int.from_bytes(blob[p:p + 2], "little")
                p += 2
                syms32 = np.zeros(A_, np.int32)
                freq = np.zeros(A_, np.uint16)
                cum = np.zeros(A_, np.int32)
                c = 0
                for i in range(A_):
                    sv = int.from_bytes(blob[p:p + 2], "little")
                    f = int.from_bytes(blob[p + 2:p + 4], "little")
                    p += 4
                    syms32[i] = (sv + 1024) & 0xFFFF
                    freq[i] = f
                    cum[i] = c
                    c += f
                n_ = int.from_bytes(blob[p:p + 4], "little")
                p += 4
                pay = bytes(blob[p:p + n_])
                p += n_
                slots = np.zeros(16384, np.int32)
                LIB.rans_slots(freq.ctypes.data_as(ctypes.POINTER(c_u16)),
                               cum.ctypes.data_as(ctypes.POINTER(c_i32)), A_,
                               slots.ctypes.data_as(ctypes.POINTER(c_i32)))
                dec = np.zeros(cnt, dtype=np.int16)
                bb = (c_u8 * n_)(*pay) if n_ else (c_u8 * 0)()
                rc = LIB.rans_decode(bb, n_, cnt,
                                     syms32.ctypes.data_as(ctypes.POINTER(c_i32)),
                                     freq.ctypes.data_as(ctypes.POINTER(c_u16)),
                                     cum.ctypes.data_as(ctypes.POINTER(c_i32)), A_,
                                     slots.ctypes.data_as(ctypes.POINTER(c_i32)),
                                     dec.ctypes.data_as(ctypes.POINTER(c_i16)))
                assert rc == 0, (gi, rc)
                rlist.append(dec)
                goff.append(goff[-1] + cnt)
            else:
                goff.append(goff[-1])
        rsyms = np.concatenate(rlist).astype(np.int16) if rlist else np.zeros(0, np.int16)
        goff = np.array(goff, dtype=np.int32)
        hasn = np.zeros(128, dtype=np.uint8)
        for gi in range(ng):
            hasn[gi] = 1 if (backend[gi] == 0 and nonempty[gi]) else 0
        gsel = np.arange(256, dtype=np.uint8).reshape(128, 2)
        for gi in range(ng):
            gsel[gi, 0] = gi
            gsel[gi, 1] = gi
        plane = np.zeros(N, dtype=np.int16)
        hb = (c_u8 * len(hpay))(*hpay) if len(hpay) else (c_u8 * 0)()
        gb = (c_u8 * len(gpay))(*gpay) if len(gpay) else (c_u8 * 0)()
        rs = rsyms.ctypes.data_as(ctypes.POINTER(c_i16)) if rsyms.size else None
        rc = LIB.crown2_decode_ch(
            hb, len(hpay), gb, len(gpay), rs, goff.ctypes.data_as(ctypes.POINTER(c_i32)), ng,
            cd.ctypes.data_as(ctypes.POINTER(c_u32)), ln.ctypes.data_as(ctypes.POINTER(c_u8)),
            cmap.ctypes.data_as(ctypes.POINTER(c_u8)), predid.ctypes.data_as(ctypes.POINTER(c_u8)),
            backend.ctypes.data_as(ctypes.POINTER(c_u8)), kvals.ctypes.data_as(ctypes.POINTER(c_u8)),
            dbias.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),
            gsel.ctypes.data_as(ctypes.POINTER(c_u8)), hasn.ctypes.data_as(ctypes.POINTER(c_u8)),
            fam, grid_thr.ctypes.data_as(ctypes.POINTER(c_i32)),
            None, None, None, 0, None,
            plane.ctypes.data_as(ctypes.POINTER(c_i16)), H, W)
        assert rc == 0, rc
        planes.append(plane)
        ch_metas.append({"ng": ng, "fam": fam})
    yc = np.zeros(3 * N, dtype=np.int16)
    yc[0 * N:1 * N] = planes[0].reshape(-1)
    yc[1 * N:2 * N] = planes[1].reshape(-1)
    yc[2 * N:3 * N] = planes[2].reshape(-1)
    out = bytearray(N * 3)
    LIB.ycocg_inv(yc.ctypes.data_as(ctypes.POINTER(c_i16)), H, W,
                  (ctypes.c_char * len(out)).from_buffer(out))
    assert p == len(blob), (p, len(blob))
    return bytes(out), ch_metas


def wilcoxon_exact(diffs):
    """Exact two-sided Wilcoxon signed-rank p for n<=20 (enumerate 2^n). Zeros dropped."""
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n == 0:
        return 1.0, 0
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        avg = (i + 1 + j + 1) / 2
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    wobs = sum(r for r, x in zip(ranks, d) if x > 0)
    tot = sum(ranks)
    ge = 0
    for m in range(1 << n):
        w = sum(ranks[i] for i in range(n) if (m >> i) & 1)
        if abs(w - tot / 2) >= abs(wobs - tot / 2) - 1e-12:
            ge += 1
    return ge / (1 << n), wobs


if __name__ == "__main__":
    print("driver_crown2 scaffold ok")
