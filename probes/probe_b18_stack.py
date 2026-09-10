"""probe_b18_stack.py — B18 STACK weapon on the exact CROWN3 arm (new file only).

Codec "B18" = CROWN3 + three gated weapons (all decoder-safe, all side counted):
  W-A: per-image WAVG granularity GS in {16,32} (GS chosen by fast exact proxy,
       +1b/image; CROWN3 fixed GS=32 and dropped b17's size-sel).
  W-B: LMS5-T0 / LMS5-T3 per-pixel adaptive-FIR experts (18th/19th; 5b ids
       already hold <=32; ZERO side; recon-only state, deterministic).
  W-C: ERR4 Q-key = LOCO-365 x prev-LEFT-MEDres-bin (decoder-derivable from
       causal recon; map over <=2916-key space, counted exactly).
GRID family unchanged (CROWN3 verbatim path). Per-channel Q-vs-GRID by exact
blob bytes. RCT min-over-{C6,C27,C12} like CROWN3. New magic b"B8" framing
(documented below); channel blobs counted as ACTUAL Python-built bytes.

Invariants: YCoCg-R assert round-trips; causal-recon-only conditioning;
alphabet |r|<=1024 assert-loud; bpp = total_bits/(H*W*3). numpy+PIL+ctypes only.

B8 wire format (probe-level, Python decode in this file):
  b"B8" + H(u16) + W(u16) + ver(u8=1) + rctid(u8: 0=C6,1=C27,2=C12) + gsbit(u8)
  per channel (Y,Co,Cg order):
    Q: b"Q" + K(u8) + ng(u8) + ncell-mask(365B) + keymap + W-side + meta7b +
       ktbl + dtbl + H-sect + G-sect + R-sect   [same section order as CROWN3]
    G: CROWN3 GRID channel blob verbatim (C3.assemble_channel) with 1B tag b"G"
"""

import ctypes
import math
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, "/tmp/opencode/autocompress/csrc")
sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
os.chdir("/tmp/opencode/autocompress/experiments")
import driver_crown3 as C3
import probe_b17_rctw as B17
from probe_b5_c import prepX, E16, predictors_X
from probe_b4_b import nbhd as nbhd_base
from probe_b4_c import loco_ctx365
from driver import canon_tables

HAPRE = C3.HAPRE
C3LIB = C3.C3
c_u8, c_i8, c_i16, c_u16, c_i32, c_u32, c_i64 = (
    C3.c_u8, C3.c_i8, C3.c_i16, C3.c_u16, C3.c_i32, C3.c_u32, C3.c_i64)

E19 = E16 + ["WAVG", "LMS5_T0", "LMS5_T3"]
EXPID = {n: i for i, n in enumerate(E19)}
DUMP_GROUPS = None  # set to a path to append per-channel Q-group dumps (pickle)
WIDE = C3.WIDE
WIDE_IDX = C3.WIDE_IDX
RCTS = C3.RCTS
NKEY = 729 * 4  # ERR4 key space
MASKB = (NKEY + 7) // 8  # 365
RES_MAX = 1024

IMAGES = C3.IMAGES


def check_range(arr, where):
    m = int(np.abs(arr).max()) if arr.size else 0
    assert m <= RES_MAX, f"ALPHABET VIOLATION {where}: |r|max={m}"


def lms_pred_plane(plane, thr):
    """Integer sign-sign LMS 5-tap {L,T,TL,TR,MED}, sum-16 renorm. Returns PRED."""
    p = plane.astype(np.int32)
    H, W = p.shape
    a, b, c, d, Ww, NNe, NE = nbhd_base(plane)
    M = B17.med_pred(plane)
    L, T, TL, TR, MD = a.ravel(), b.ravel(), c.ravel(), d.ravel(), M.ravel()
    y = p.ravel()
    out = np.empty_like(y)
    w = [3, 3, 3, 3, 4]
    for n in range(y.size):
        l, t, tl, tr, md = int(L[n]), int(T[n]), int(TL[n]), int(TR[n]), int(MD[n])
        pr = (w[0] * l + w[1] * t + w[2] * tl + w[3] * tr + w[4] * md + 8) // 16
        out[n] = pr
        e = int(y[n]) - pr
        if abs(e) > thr:
            se = 1 if e > 0 else -1
            m = (l + t + tl + tr + md + 2) // 5
            xs = (l, t, tl, tr, md)
            for i in range(5):
                di = xs[i] - m
                if di > 0:
                    w[i] += se
                elif di < 0:
                    w[i] -= se
            for i in range(5):
                w[i] = min(20, max(-8, w[i]))
            s = sum(w)
            while s > 16:
                j = max(range(5), key=lambda i: (w[i], -i))
                w[j] -= 1
                s -= 1
            while s < 16:
                j = min(range(5), key=lambda i: (w[i], i))
                w[j] += 1
                s += 1
    return out.reshape(H, W)


def err4_key(D, ch):
    """LOCO-365 key x prev-LEFT MED-residual bin (decoder-derivable)."""
    key0, s = D["key"], D["s"]
    M = B17.med_pred(ch)
    R = (s * (ch - M).astype(np.int32)).astype(np.int32)
    RL = np.zeros_like(R)
    RL[:, 1:] = np.abs(R[:, :-1])
    prevL = np.minimum(RL // 3, 3).astype(np.int64)
    return (key0.astype(np.int64) * 4 + prevL), s


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


class BitReader:
    def __init__(self, buf):
        self.buf = buf
        self.qp = 0
        self.acc = 0
        self.nb = 0

    def get(self, n):
        while self.nb < n:
            self.acc = (self.acc << 8) | self.buf[self.qp]
            self.qp += 1
            self.nb += 8
        self.nb -= n
        v = (self.acc >> self.nb) & ((1 << n) - 1)
        self.acc &= ((1 << self.nb) - 1) if self.nb else 0
        return v


def fit_wavg(planes, H, W, gs):
    return C3.fit_weighted(planes, H, W) if gs == 32 else B17_wavg(planes, H, W, gs)


def B17_wavg(planes, H, W, GS):
    out_res, use_all, wall_all, side, bm, ng = B17.weighted_fit(planes, H, W, GS)
    chs = []
    for c in range(3):
        check_range(out_res[c], f"WAVG{GS} res ch{c}")
        use = np.array(use_all[c], dtype=np.uint8)
        w = np.zeros((ng, 4), dtype=np.int8)
        for g, wq in enumerate(wall_all[c]):
            if wq is not None:
                w[g] = wq.astype(np.int8)
        chs.append({"res": out_res[c].astype(np.int32), "use": use, "w": w})
    nbh = math.ceil(H / GS)
    nbw = math.ceil(W / GS)
    return chs, bm, nbh, nbw, ng, side


def proxy_gs_choice(rgb):
    """Fast exact proxy (b17-(b) frame, global Huffman): argmin GS in {16,32}."""
    H, W, _ = rgb.shape
    best = None
    for rname, perm, t in RCTS:
        yc = B17.rct_fwd(rgb, perm, t)
        planes = [yc[:, :, c] for c in range(3)]
        npx = H * W * 3
        for GS in (16, 32):
            out_res, _, _, side, _, _ = B17.weighted_fit(planes, H, W, GS)
            tb = side + 1 + sum(
                B17.exact_plane_bits(out_res[c])[0] for c in range(3)) + 64 + 2
            if best is None or tb < best[0]:
                best = (tb, GS)
    return best[1]


def encode_channel_q(ch, D, RF, key, gs_use, gs_w, H, W, topk=3,
                       nkey=NKEY, maskB=MASKB):
    """Q-family over ERR4 key. Returns (blob, meta) with ACTUAL bytes."""
    N = H * W
    rfM = RF["MED"]
    uk_all = np.unique(key)
    cands = []
    for K in WIDE:
        groups = C3.quantile_groups(key, rfM, K)
        gbits = math.ceil(math.log2(K))
        mapside = 16 + maskB * 8 + len(uk_all) * gbits
        gv = []
        for gkeys in groups:
            sel = np.isin(key, np.array(gkeys))
            gv.append({n: RF[n][sel] for n in RF})
        sub, wins2 = C3.solve_groups(gv, use_rans=False)
        cands.append((mapside + len(groups) * 6 + sub, K, groups, wins2))
    cands.sort(key=lambda t: t[0])
    best = None
    for _, K, groups, _ in cands[:topk]:
        gv = []
        for gkeys in groups:
            sel = np.isin(key, np.array(gkeys))
            gv.append({n: RF[n][sel] for n in RF})
        sub, wins = C3.solve_groups(gv, use_rans=True)
        gbits = math.ceil(math.log2(K))
        mapside = 16 + maskB * 8 + len(uk_all) * gbits
        xtra = sum(4 for w in wins if w[2] == 1) + sum(3 for w in wins if w[2] == 1 and w[4] != 0)
        t = mapside + len(groups) * 6 + xtra + sub
        if best is None or t < best[0]:
            best = (t, K, groups, wins)
    _, K, groups, wins = best
    ng = len(wins)
    # symbol plane
    cmap = np.zeros(nkey, dtype=np.int64)
    for gi, gkeys in enumerate(groups):
        for u in gkeys:
            cmap[int(u)] = gi
    expmap = cmap[key.reshape(-1)].reshape(H, W)
    P = D["P"]
    resplane = np.zeros((H, W), dtype=np.int32)
    for gi, w in enumerate(wins):
        _, bn, _, _, _, _ = w
        m = (expmap == gi)
        if bn == "WAVG":
            r = gs_use["res"]
        elif bn == "LMS5_T0":
            r = gs_use["lms0"]
        elif bn == "LMS5_T3":
            r = gs_use["lms3"]
        else:
            r = (ch - P[bn]).astype(np.int32)
        resplane[m] = r[m]
    check_range(resplane, "Q residuals")
    s = D["s"]
    symplane = (s.astype(np.int32) * resplane)
    check_range(symplane, "Q flipped")
    back = np.array([w[2] for w in wins], dtype=np.uint8)
    counts = np.array([(expmap == gi).sum() for gi in range(ng)])
    for gi in range(ng):
        if counts[gi] == 0:
            back[gi] = 0
    blob = bytearray()
    blob += b"Q"
    blob.append(WIDE_IDX[K] & 0xFF)
    blob.append(ng & 0xFF)
    blob.append(1 if nkey == NKEY else 0)
    mask = np.zeros(nkey, dtype=np.uint8)
    mask[uk_all] = 1
    blob += np.packbits(mask).tobytes()
    assert len(blob) == 4 + maskB
    bw = BitWriter()
    gbits = math.ceil(math.log2(K))
    for u in sorted(uk_all.tolist()):
        bw.put(int(cmap[int(u)]), gbits)
    blob += bw.flush()
    # W-side (GS-parameterized)
    ngG = gs_w["ng"]
    bw = BitWriter()
    for g in range(ngG):
        bw.put(int(gs_w["use"][g]), 1)
    blob += bw.flush()
    bw = BitWriter()
    for g in range(ngG):
        if gs_w["use"][g]:
            for t_ in range(4):
                bw.put(int(gs_w["w"][g, t_]) & 31, 5)
    blob += bw.flush()
    # per-group meta 7b
    bw = BitWriter()
    for gi, w in enumerate(wins):
        bw.put(EXPID[w[1]], 5)
        bw.put(int(back[gi]), 2)
    blob += bw.flush()
    bwk, bwb = BitWriter(), BitWriter()
    for gi, w in enumerate(wins):
        if back[gi] == 1:
            bwk.put(int(w[3]), 4)
            bwb.put(int(w[4]) & 7, 3)
    blob += bwk.flush() + bwb.flush()
    kbyexp = np.zeros((ng,), np.uint8)
    dbexp = np.zeros((ng,), np.int8)
    for gi, w in enumerate(wins):
        if back[gi] == 1:
            kbyexp[gi] = w[3]
            dbexp[gi] = w[4]
    # H section (mirror C3 packing)
    cd = np.zeros((ng * 2049,), np.uint32)
    ln = np.zeros((ng * 2049,), np.uint8)
    flat_exp = expmap.reshape(-1)
    flat_sym = symplane.reshape(-1)
    hmask = back[flat_exp] == 0
    hsyms = flat_sym[hmask].astype(np.int16)
    hkeys = flat_exp[hmask].astype(np.int32)
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
            tbl = bytearray()
            tbl += len(codes).to_bytes(2, "little")
            for sym in sorted(codes):
                tbl += ((sym - 1024) & 0xFFFF).to_bytes(2, "little")
                tbl.append(codes[sym][1])
            blob += bytes(tbl)
    if hsyms.size:
        cap = (hsyms.size * 32) // 8 + 1024
        buf = (c_u8 * cap)()
        hk = hkeys.astype(np.uint8)
        n = HAPRE.pack_syms(hsyms.ctypes.data_as(ctypes.POINTER(c_i16)),
                            hk.ctypes.data_as(ctypes.POINTER(c_u8)), hsyms.size,
                            cd.ctypes.data_as(ctypes.POINTER(c_u32)),
                            ln.ctypes.data_as(ctypes.POINTER(c_u8)), buf, cap)
        assert n > 0
        blob += int(n).to_bytes(4, "little") + bytes(buf[:n])
    else:
        blob += (0).to_bytes(4, "little")
    # G section (vendored crown3 golomb pack)
    gmask = back[flat_exp] == 1
    if gmask.sum():
        gpix_exp = flat_exp[gmask]
        gvals = flat_sym[gmask].astype(np.int32) - dbexp[gpix_exp].astype(np.int32)
        check_range(gvals, "G transmitted")
        gvals = gvals.astype(np.int16)
        gkeys = gpix_exp.astype(np.uint8)
        cap = int(gvals.size * 300 + 64)
        buf = (c_u8 * cap)()
        n = C3LIB.golomb_pack(gvals.ctypes.data_as(ctypes.POINTER(c_i16)),
                              gkeys.ctypes.data_as(ctypes.POINTER(c_u8)),
                              kbyexp.ctypes.data_as(ctypes.POINTER(c_u8)),
                              gvals.size, buf, cap)
        assert n > 0
        blob += int(n).to_bytes(4, "little") + bytes(buf[:n])
    else:
        blob += (0).to_bytes(4, "little")
    # R section
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
    meta = {"ng": ng, "K": K, "groups": groups, "wins": wins, "back": back,
            "cmap": cmap, "kbyexp": kbyexp, "dbexp": dbexp,
            "cd": cd, "ln": ln, "counts": counts}
    if DUMP_GROUPS is not None:
        import pickle as _pk
        dump = {"K": K, "ng": ng, "groups": [np.array(gk) for gk in groups],
                "wins": [(w[0], w[1], w[2], w[3], w[4]) for w in wins],
                "back": back.copy(), "counts": counts.copy(),
                "key": key.copy(), "expmap": expmap.copy(),
                "symplane": symplane.copy()}
        with open(DUMP_GROUPS, "ab") as f:
            _pk.dump(dump, f)
    return bytes(blob), meta


def encode_image_rct(rgb, rname, perm, t, gs, use_lms=True, use_err4=True):
    H, W, _ = rgb.shape
    yc = B17.rct_fwd(rgb, perm, t)
    assert np.array_equal(B17.rct_inv(yc, perm, t), rgb), "RCT round-trip FAIL"
    planes = [yc[:, :, c] for c in range(3)]
    wchs, bm, nbh, nbw, ngG, wside = fit_wavg(planes, H, W, gs)
    BANK = list(E19) if use_lms else list(E16 + ["WAVG"])
    if use_lms:
        lms0 = [lms_pred_plane(pl, 0) for pl in planes]
        lms3 = [lms_pred_plane(pl, 3) for pl in planes]
    else:
        lms0 = lms3 = [None, None, None]
    out = bytearray()
    out += b"B8" + H.to_bytes(2, "little") + W.to_bytes(2, "little")
    out += bytes([1, C3.RCTID[rname], 0 if gs == 16 else 1])
    infos = []
    for ci, ch in enumerate(planes):
        D = prepX(ch)
        if use_err4:
            key, _ = err4_key(D, ch)
            nkey, maskB = NKEY, MASKB
        else:
            key, nkey, maskB = D["key"].astype(np.int64), 729, 92
        P = D["P"]
        RF = {n: (D["s"] * (ch - P[n]).astype(np.int32)).astype(np.int32) for n in E16}
        RF["WAVG"] = (D["s"] * wchs[ci]["res"]).astype(np.int32)
        if use_lms:
            RF["LMS5_T0"] = (D["s"] * (ch - lms0[ci]).astype(np.int32)).astype(np.int32)
            RF["LMS5_T3"] = (D["s"] * (ch - lms3[ci]).astype(np.int32)).astype(np.int32)
        for n in RF:
            check_range(RF[n], f"flip {rname} ch{ci} {n}")
        RU = {n: (ch - P[n]).astype(np.int32) for n in E16}
        RU["WAVG"] = wchs[ci]["res"]
        # NOTE: GRID reuses CROWN3's assembler (17-expert EXPID); LMS experts
        # are Q-family-only. GRID RU stays E17 (strength-safe, crash-free).
        RUGRID = dict(RU)
        gs_use = {"res": wchs[ci]["res"],
                  "lms0": (ch - lms0[ci]).astype(np.int32) if use_lms else None,
                  "lms3": (ch - lms3[ci]).astype(np.int32) if use_lms else None}
        gs_w = {"use": wchs[ci]["use"], "w": wchs[ci]["w"], "ng": ngG, "nbw": nbw}
        bq, mq = encode_channel_q(ch, D, RF, key, gs_use, gs_w, H, W,
                                  nkey=nkey, maskB=maskB)
        pg = C3.encode_channel_grid(ch, D, RUGRID)
        bg, mg = C3.assemble_channel(pg, D, ch, wchs[ci], H, W, nbw, ngG)
        if len(bg) < len(bq):
            out += b"G" + bg
            infos.append({"fam": "G", "nbytes": len(bg) + 1})
        else:
            out += bq  # bq already starts with its own b"Q" tag (no double tag)
            infos.append({"fam": "Q", "nbytes": len(bq),
                          "picks": {w[1]: 0 for w in mq["wins"]}})
            for w in mq["wins"]:
                infos[-1]["picks"][w[1]] = infos[-1]["picks"].get(w[1], 0) + 1
    return bytes(out), {"rct": rname, "gs": gs, "chs": infos}


def encode_image(rgb, rcts=None, gs=None):
    if gs is None:
        gs = proxy_gs_choice(rgb)
    cands = []
    for (rname, perm, t) in (rcts or RCTS):
        blob, info = encode_image_rct(rgb, rname, perm, t, gs)
        cands.append((len(blob), blob, info))
    cands.sort(key=lambda t_: t_[0])
    # +1b GS-side counted in bpp by caller (1 bit/image)
    return cands[0][1], {"winner": cands[0][2], "gs": gs,
                         "losers": [(c[2]["rct"], len(c[1])) for c in cands[1:]]}


def loco_q_s(g):
    a = abs(g)
    if g == 0:
        return 0
    s = 1 if g > 0 else -1
    return s * (1 if a <= 2 else (2 if a <= 7 else (3 if a <= 21 else 4)))


def e16_scalar(name, a, b, c, d, Ww, NNe, NE):
    if name == "MED":
        return min(a, b) if c >= max(a, b) else (max(a, b) if c <= min(a, b) else a + b - c)
    if name == "TOP":
        return b
    if name == "LEFT":
        return a
    if name == "PAETH":
        p = a + b - c
        apa, apb, apc = abs(p - a), abs(p - b), abs(p - c)
        return a if (apa <= apb and apa <= apc) else (b if apb <= apc else c)
    if name == "GRAD":
        return (a + b) // 2 + (b - c) // 4
    if name in ("GAP80", "GAP32", "GAP16", "GAP"):
        thr = {"GAP80": 80, "GAP32": 32, "GAP16": 16, "GAP": 80}[name]
        gh = abs(a - Ww) + abs(b - c) + abs(b - NE)
        gv = abs(a - c) + abs(b - NNe) + abs(NE - b)
        return a if gv - gh > thr else (b if gv - gh < -thr else (a + b) // 2 + (NE - c) // 4)
    if name == "DG":
        return (c + d) // 2
    if name == "AVG_AB":
        return (a + b) // 2
    if name == "PLANE":
        return a + b - c
    if name == "AC":
        return (a + c) // 2
    if name == "C":
        return c
    if name == "BC":
        return (b + c) // 2
    if name == "D":
        return d
    if name == "AVG3":
        return (a + b + c) // 3
    raise ValueError(name)


def lms_step(w, xs, m, y_true, thr):
    """One fused LMS predict+update. Returns (pred, w). Mutates w list."""
    pr = (w[0] * xs[0] + w[1] * xs[1] + w[2] * xs[2] + w[3] * xs[3] + w[4] * xs[4] + 8) // 16
    e = y_true - pr
    if abs(e) > thr:
        se = 1 if e > 0 else -1
        for i in range(5):
            di = xs[i] - m
            if di > 0:
                w[i] += se
            elif di < 0:
                w[i] -= se
        for i in range(5):
            w[i] = min(20, max(-8, w[i]))
        s = sum(w)
        while s > 16:
            j = max(range(5), key=lambda i: (w[i], -i))
            w[j] -= 1
            s -= 1
        while s < 16:
            j = min(range(5), key=lambda i: (w[i], i))
            w[j] += 1
            s += 1
    return pr, w


def parse_sections(blob, p, layout, ngG):
    """Parse one channel (Q or GRID-CROWN3 layout). Returns (meta, p)."""
    if layout == "Q":
        assert blob[p:p + 1] == b"Q"
        p += 1
        K = WIDE[blob[p]]
        ng = blob[p + 1]
        nkf = blob[p + 2]
        p += 3
        nkey = NKEY if nkf == 1 else 729
        maskB = (nkey + 7) // 8
        gbits = math.ceil(math.log2(K))
        mask = np.unpackbits(np.frombuffer(blob[p:p + maskB], dtype=np.uint8))[:nkey]
        p += maskB
        uk = np.where(mask)[0]
        cmap = np.zeros(nkey, dtype=np.int64)
        br = BitReader(blob[p:p + (len(uk) * gbits + 7) // 8])
        for u in uk:
            cmap[u] = br.get(gbits)
        p += (len(uk) * gbits + 7) // 8
        grid_thr = None
        fam = 0
    else:
        # p points at CROWN3 GRID channel header (b0,ng); 1B tag consumed by caller.
        b0, ng = blob[p], blob[p + 1]
        p += 2
        fam = b0 & 1
        assert fam == 1
        grid = (b0 >> 1) & 7
        grid_thr = np.array(C3.GRIDS[grid], dtype=np.int64)
        K, cmap, uk = 6, None, None
    wuse = np.zeros(ngG, dtype=np.uint8)
    br = BitReader(blob[p:p + (ngG + 7) // 8])
    for g in range(ngG):
        wuse[g] = br.get(1)
    p += (ngG + 7) // 8
    nused = int(wuse.sum())
    ww = np.zeros((ngG, 4), dtype=np.int8)
    br = BitReader(blob[p:p + (nused * 20 + 7) // 8])
    for g in range(ngG):
        if wuse[g]:
            for t_ in range(4):
                v = br.get(5)
                ww[g, t_] = v - 32 if v >= 16 else v
    p += (nused * 20 + 7) // 8
    if layout == "G":
        occ = blob[p]
        p += 1
        nonempty = np.array([(occ >> gi) & 1 for gi in range(ng)], dtype=bool)
    else:
        nonempty = np.zeros(ng, dtype=bool)
        for u in uk:
            nonempty[cmap[u]] = True
    predid = np.zeros(ng, dtype=np.uint8)
    backend = np.zeros(ng, dtype=np.uint8)
    br = BitReader(blob[p:p + (ng * 7 + 7) // 8])
    for gi in range(ng):
        v = br.get(7)
        predid[gi] = (v >> 2) & 31
        backend[gi] = v & 3
        assert predid[gi] <= 18 and backend[gi] <= 2
    p += (ng * 7 + 7) // 8
    nG = int((backend[:ng] == 1).sum())
    kvals = np.zeros(ng, dtype=np.uint8)
    dbias = np.zeros(ng, dtype=np.int8)
    br = BitReader(blob[p:p + (nG * 4 + 7) // 8])
    for gi in range(ng):
        if backend[gi] == 1:
            kvals[gi] = br.get(4)
    p += (nG * 4 + 7) // 8
    br = BitReader(blob[p:p + (nG * 3 + 7) // 8])
    for gi in range(ng):
        if backend[gi] == 1:
            v = br.get(3)
            dbias[gi] = v - 8 if v >= 4 else v
    p += (nG * 3 + 7) // 8
    trees = {}
    for gi in range(ng):
        if backend[gi] == 0 and nonempty[gi]:
            A_ = int.from_bytes(blob[p:p + 2], "little")
            p += 2
            lens = {}
            for _ in range(A_):
                sv = int.from_bytes(blob[p:p + 2], "little")
                ll = blob[p + 2]
                p += 3
                lens[(sv + 1024) & 0xFFFF] = ll
            # build decode tree
            ch0, ch1, svv = {}, {}, {}
            nn = 1
            order = sorted(lens, key=lambda s: (lens[s], s))
            code, prev = 0, 0
            cmap2 = {}
            for s in order:
                code <<= (lens[s] - prev)
                prev = lens[s]
                cmap2[s] = (code, lens[s])
                code += 1
            # tree insert
            nodes = [[-1, -1]]
            val = [-99999]
            for s, (cc_, ll_) in cmap2.items():
                node = 0
                for bi in range(ll_ - 1, -1, -1):
                    bit = (cc_ >> bi) & 1
                    if nodes[node][bit] == -1:
                        nodes[node][bit] = len(nodes)
                        nodes.append([-1, -1])
                        val.append(-99999)
                    node = nodes[node][bit]
                val[node] = s - 1024
            trees[gi] = (nodes, val)
    hn = int.from_bytes(blob[p:p + 4], "little")
    p += 4
    hpay = blob[p:p + hn]
    p += hn
    gn_ = int.from_bytes(blob[p:p + 4], "little")
    p += 4
    gpay = blob[p:p + gn_]
    p += gn_
    rclust = {}
    rptr = {}
    for gi in range(ng):
        if backend[gi] == 2:
            cnt = int.from_bytes(blob[p:p + 4], "little")
            p += 4
            A_ = int.from_bytes(blob[p:p + 2], "little")
            p += 2
            syms32 = np.zeros(A_, np.int32)
            freq = np.zeros(A_, np.uint16)
            cum = np.zeros(A_, np.int32)
            cc = 0
            for i in range(A_):
                sv = int.from_bytes(blob[p:p + 2], "little")
                f = int.from_bytes(blob[p + 2:p + 4], "little")
                p += 4
                syms32[i] = (sv + 1024) & 0xFFFF
                freq[i] = f
                cum[i] = cc
                cc += f
            n_ = int.from_bytes(blob[p:p + 4], "little")
            p += 4
            pay = bytes(blob[p:p + n_])
            p += n_
            slots = np.zeros(16384, np.int32)
            HAPRE.rans_slots(freq.ctypes.data_as(ctypes.POINTER(c_u16)),
                             cum.ctypes.data_as(ctypes.POINTER(c_i32)), A_,
                             slots.ctypes.data_as(ctypes.POINTER(c_i32)))
            dec = np.zeros(cnt, dtype=np.int16)
            bb = (c_u8 * n_)(*pay) if n_ else (c_u8 * 0)()
            rc = HAPRE.rans_decode(bb, n_, cnt,
                                   syms32.ctypes.data_as(ctypes.POINTER(c_i32)),
                                   freq.ctypes.data_as(ctypes.POINTER(c_u16)),
                                   cum.ctypes.data_as(ctypes.POINTER(c_i32)), A_,
                                   slots.ctypes.data_as(ctypes.POINTER(c_i32)),
                                   dec.ctypes.data_as(ctypes.POINTER(c_i16)))
            assert rc == 0
            rclust[gi] = dec
            rptr[gi] = 0
    return {"layout": layout, "K": K, "ng": ng, "cmap": cmap, "grid_thr": grid_thr,
            "nkey": nkey if layout == "Q" else 0,
            "wuse": wuse, "ww": ww, "predid": predid, "backend": backend,
            "kvals": kvals, "dbias": dbias, "nonempty": nonempty,
            "trees": trees, "hpay": hpay, "gpay": gpay, "rclust": rclust,
            "rptr": rptr}, p


class PayReader:
    def __init__(self, buf):
        self.buf = buf
        self.bp = 0

    def bit(self):
        byte = self.buf[self.bp >> 3]
        b = (byte >> (7 - (self.bp & 7))) & 1
        self.bp += 1
        return b


def decode_channel(meta, H, W, gs):
    """Full raster-order Python decode of one channel -> int32 plane."""
    nbw = math.ceil(W / gs)
    recon = np.zeros((H, W), dtype=np.int32)
    hr = PayReader(meta["hpay"])
    gr = PayReader(meta["gpay"])
    w0 = [3, 3, 3, 3, 4]
    w3 = [3, 3, 3, 3, 4]
    ENAMES = E19
    for i in range(H):
        for j in range(W):
            lv = recon[i, j - 1] if j > 0 else (recon[i - 1, 0] if i > 0 else 0)
            tv = recon[i - 1, j] if i > 0 else (recon[i, j - 1] if j > 0 else 0)
            # b4_b-style (E16/keys/GRID): TL copies L at borders; TR copies L on row0.
            tl = recon[i - 1, j - 1] if (i > 0 and j > 0) else lv
            # B17-style (WAVG/LMS-tap/err4-MED): TL/TR zero at borders.
            tlb = recon[i - 1, j - 1] if (i > 0 and j > 0) else 0
            if i == 0:
                tr = lv
            elif j + 1 >= W:
                tr = tv
            else:
                tr = recon[i - 1, j + 1]
            trB = recon[i - 1, j + 1] if (i > 0 and j + 1 < W) else 0
            if j >= 2:
                ww2 = recon[i, j - 2]
            else:
                ww2 = lv
            if i >= 2:
                nn2 = recon[i - 2, j]
            else:
                nn2 = tv
            a, b, c, d, Ww, NNe, NE = lv, tv, tl, tr, ww2, nn2, tr
            if meta["layout"] == "Q":
                q1, q2, q3 = loco_q_s(b - c), loco_q_s(c - a), loco_q_s(d - b)
                neg = (q1 < 0) or (q1 == 0 and q2 < 0) or (q1 == 0 and q2 == 0 and q3 < 0)
                s = -1 if neg else 1
                if neg:
                    m1, m2, m3 = -q1, -q2, -q3
                else:
                    m1, m2, m3 = q1, q2, q3
                key0 = m1 * 81 + m2 * 9 + m3
                if meta["nkey"] == NKEY:
                    if j > 0:
                        la = recon[i, j - 2] if j >= 2 else (recon[i - 1, 0] if i > 0 else 0)
                        lt = recon[i - 1, j - 1] if i > 0 else (recon[i, j - 2] if j >= 2 else 0)
                        ltlB = recon[i - 1, j - 2] if (i > 0 and j >= 2) else 0
                        cLP = recon[i - 1, j - 2] if (i > 0 and j >= 2) else la
                        dLP = la if i == 0 else recon[i - 1, j]
                        mn, mx = (la, lt) if la < lt else (lt, la)
                        medL = mn if ltlB >= mx else (mx if ltlB <= mn else la + lt - ltlB)
                        lq1 = loco_q_s(lt - cLP)
                        lq2 = loco_q_s(cLP - la)
                        lq3 = loco_q_s(dLP - lt)
                        lneg = (lq1 < 0) or (lq1 == 0 and lq2 < 0) or (lq1 == 0 and lq2 == 0 and lq3 < 0)
                        sL = -1 if lneg else 1
                        rL = sL * (int(recon[i, j - 1]) - medL)
                        key4 = key0 * 4 + min(abs(int(rL)) // 3, 3)
                    else:
                        key4 = key0 * 4
                else:
                    key4 = key0
                g = int(meta["cmap"][key4])
            else:
                s = 1
                e = abs(a - c) + abs(b - c)
                g = 0
                for t_ in meta["grid_thr"]:
                    if t_ < e:
                        g += 1
            x = int(meta["predid"][g])
            bb = int(meta["backend"][g])
            if bb == 0:
                nodes, val = meta["trees"][g]
                node = 0
                while True:
                    bit = hr.bit()
                    node = nodes[node][bit]
                    assert node != -1
                    if val[node] != -99999:
                        sym = val[node]
                        break
            elif bb == 1:
                q = 0
                while gr.bit() == 0:
                    q += 1
                k = int(meta["kvals"][g])
                rem = 0
                for _ in range(k):
                    rem = (rem << 1) | gr.bit()
                M = (q << k) | rem
                sym = (M // 2) if (M % 2 == 0) else (-(M + 1) // 2)
                sym += int(meta["dbias"][g])
            else:
                arr = meta["rclust"][g]
                sym = int(arr[meta["rptr"][g]])
                meta["rptr"][g] += 1
            res = s * sym if meta["layout"] == "Q" else sym
            en = ENAMES[x]
            mnB, mxB = (a, b) if a < b else (b, a)
            medB17 = mnB if tlb >= mxB else (mxB if tlb <= mnB else a + b - tlb)
            if en == "WAVG":
                bg = (i // gs) * nbw + (j // gs)
                if meta["wuse"][bg]:
                    w = [int(v) for v in meta["ww"][bg]]
                    pr = (w[0] * a + w[1] * b + w[2] * tlb + w[3] * trB + 8) // 16
                else:
                    mn, mx = (a, b) if a < b else (b, a)
                    pr = mn if c >= mx else (mx if c <= mn else a + b - c)
            elif en == "LMS5_T0":
                xs = [a, b, c, d, medB17]
                m = (a + b + c + d + xs[4] + 2) // 5
                # predict with current state (update after recon known)
                pr = (w0[0] * xs[0] + w0[1] * xs[1] + w0[2] * xs[2] + w0[3] * xs[3] + w0[4] * xs[4] + 8) // 16
            elif en == "LMS5_T3":
                xs = [a, b, c, d, medB17]
                pr = (w3[0] * xs[0] + w3[1] * xs[1] + w3[2] * xs[2] + w3[3] * xs[3] + w3[4] * xs[4] + 8) // 16
            else:
                pr = e16_scalar(en, a, b, c, d, Ww, NNe, NE)
            recon[i, j] = pr + res
            # advance LMS states with (recon, per-arm preds)
            for wv, thr in ((w0, 0), (w3, 3)):
                xs = [a, b, c, d, medB17]
                m = (a + b + c + d + xs[4] + 2) // 5
                lms_step(wv, xs, m, int(recon[i, j]), thr)
    return recon


def decode_image(blob):
    assert blob[:2] == b"B8", blob[:2]
    H = int.from_bytes(blob[2:4], "little")
    W = int.from_bytes(blob[4:6], "little")
    assert blob[6] == 1
    rct = blob[7]
    gs = 16 if blob[8] == 0 else 32
    ngG = math.ceil(H / gs) * math.ceil(W / gs)
    p = 9
    planes = []
    for _ in range(3):
        layout = "Q" if blob[p:p + 1] == b"Q" else "G"
        meta, p = parse_sections(blob, p + 1 if layout == "G" else p, layout, ngG)
        # NOTE: GRID blobs from C3.assemble_channel include their own 2B header
        # (b0,ng) right after our 1B tag; parse_sections G-branch expects p at b0.
        planes.append(decode_channel(meta, H, W, gs))
    assert p == len(blob), (p, len(blob))
    p0 = np.ascontiguousarray(planes[0].reshape(H, W).astype(np.int16))
    p1 = np.ascontiguousarray(planes[1].reshape(H, W).astype(np.int16))
    p2 = np.ascontiguousarray(planes[2].reshape(H, W).astype(np.int16))
    out = bytearray(H * W * 3)
    rc = C3LIB.crown3_rct_inv(p0.ctypes.data_as(ctypes.POINTER(c_i16)),
                              p1.ctypes.data_as(ctypes.POINTER(c_i16)),
                              p2.ctypes.data_as(ctypes.POINTER(c_i16)), H, W, rct,
                              (ctypes.c_char * len(out)).from_buffer(out))
    assert rc == 0, f"RCT inv range FAIL rc={rc}"
    return bytes(out), {"rct": RCTS[rct][0], "gs": gs}


def decode_q(blob, p, H, W, nbw, ngG, lms_thr_pair):
    raise NotImplementedError("replaced by parse_sections/decode_channel/decode_image")


if __name__ == "__main__":
    print("probe_b18_stack scaffold ok")
