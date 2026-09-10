"""CROWN5 driver: CROWN4 predictors/groups + GLOBAL raster-order adaptive entropy.

Stack (NEW files only; crown4.c / libhapre.so / drivers READ-ONLY, reused by import):
  Front end VERBATIM CROWN4 (driver_crown4.py imports, not reimplementation):
    GLOBAL best-RCT over {C6,C27,C12} (2b flags, homogeneous) [CROWN4 §global header]
    E19 bank (E16 + WAVG G32 LS-4tap MDL>21 + LMS5_T0/T3 recon-only sum-16) [CROWN4]
    Q-family sign-flipped LOCO-365 + WIDE-K quantile vs GRID M-THR energy,
    per-channel choice by EXACT adaptive bytes (CROWN4 candidates, new gate).
  Entropy (NEW, probe_b19_adaptive.py-verbatim states, zero tables):
    acodec 0 = A-act: 16-ctx (4 act x 4 pos) backward-adaptive binary range coder
      (16-bit E1/E2/E3, c0=c1=1, MAXT=128 halving) on Exp-Golomb prefix + raw suffix.
    acodec 1 = C-gol: 4-act-ctx adaptive Golomb-k (N=1/A=4 init, RESET=64 halving,
      k from (N<<k)>=A, raster order) with CROWN polarity (q ZEROS + 1 + k rem).
    Per-channel best-of {A-act, C-gol} by exact bytes (+8b acodec side counted).
    Killed (probe): B-fwd transmitted init (+0.0003), D-run (-1.31% loses to C on
      texture), A-pos unconditioned (-0.34% only). Suffix refinement deferred.

Decoder-safety: every adaptation state (c0/c1 per of 16 ctx; N/A/k per of 4 ctx;
  activity bin; MED/WAVG/LMS preds; LOCO sign) is a pure function of already-decoded
  recon/pixels (+ static ATHR/RESET/init known both sides). Streaming C decode
  (libcrown5.so crown5_decode_ch) mirrors Python integer updates in raster order;
  round-trip decode==original asserted per image; framing p==len asserted loudly.
Alphabet +-1024 assert-loud, never clip. numpy+PIL+gcc+ctypes only, no torch.
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
import probe_b17_rctw as B17
import probe_b19_adaptive as AD
import probe_b18_stack as S18
import driver_crown4 as C4
from probe_b5_c import prepX

C5 = ctypes.CDLL("/tmp/opencode/autocompress/csrc/libcrown5.so")
c_u8 = ctypes.c_uint8
c_i8 = ctypes.c_int8
c_i16 = ctypes.c_int16
c_u16 = ctypes.c_uint16
c_i32 = ctypes.c_int32
c_u32 = ctypes.c_uint32

C5.crown5_decode_ch.argtypes = [ctypes.POINTER(c_u8), ctypes.c_size_t,
                                ctypes.POINTER(c_u8), ctypes.c_size_t,
                                ctypes.POINTER(c_u8), ctypes.c_size_t,
                                ctypes.POINTER(c_u8), ctypes.POINTER(c_u8),
                                ctypes.POINTER(c_u8), ctypes.POINTER(c_i8), ctypes.c_int,
                                ctypes.c_int, ctypes.POINTER(c_i32), ctypes.c_int,
                                ctypes.POINTER(c_i16), ctypes.c_int, ctypes.c_int, ctypes.c_int]
C5.crown5_decode_ch.restype = ctypes.c_int
C5.crown5_rct_inv.argtypes = [ctypes.POINTER(c_i16), ctypes.POINTER(c_i16), ctypes.POINTER(c_i16),
                              ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
C5.crown5_rct_inv.restype = ctypes.c_int

E19 = C4.E19
E17 = C4.E17
EXPID = C4.EXPID
WIDE = C4.WIDE
WIDE_IDX = C4.WIDE_IDX
GRIDS = C4.GRIDS
RES_MAX = 1024
GS = 32
RCTS = C4.RCTS
RCTID = C4.RCTID

IMAGES = C4.IMAGES
JXL_E3 = C4.JXL_E3
CROWN4_BAR = 3.1993
CROWN4_PER = {"kodim01.png": 3.2991, "kodim02.png": 2.9854, "kodim05.png": 3.5917,
              "kodim07.png": 2.6790, "kodim13.png": 3.9525, "kodim19.png": 3.1510,
              "kodim23.png": 2.7364}


def check_range(arr, where):
    m = int(np.abs(np.asarray(arr)).max()) if np.asarray(arr).size else 0
    assert m <= RES_MAX, f"ALPHABET VIOLATION {where}: |r|max={m} > 1024 (abort, never clip)"


class BitWriter:
    def __init__(self):
        self.acc = 0
        self.nb = 0
        self.out = bytearray()

    def put(self, v, n):
        if n <= 0:
            return
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


def symplane_from_plan(plan, D, ch, wch):
    """Recompute CROWN4-verbatim symplane/expmap/resplane (no static payloads).

    Mirrors driver_crown4.assemble_channel first half exactly (same P/wch/lms,
    same sign-flip, same range asserts). ch: (H,W) int32 original (=recon).
    wch: dict with res/use/w (+lms0/lms3 for Q). Returns (symplane, expmap, resplane).
    """
    H, W = ch.shape
    fam = plan["family"]
    wins = plan["wins"]
    ng = len(wins)
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
            if bn == "WAVG":
                r = wch["res"]
            elif bn == "LMS5_T0":
                r = wch["lms0"]
            elif bn == "LMS5_T3":
                r = wch["lms3"]
            else:
                r = (ch - P[bn]).astype(np.int32)
            resplane[m] = r[m]
        check_range(resplane, "C5 Q residuals")
        symplane = (s.astype(np.int32) * resplane)
        check_range(symplane, "C5 Q flipped")
    else:
        gg = plan["gmap"]
        expmap = gg
        P = D["P"]
        resplane = np.zeros((H, W), dtype=np.int32)
        for gi, w in enumerate(wins):
            _, bn, _, _, _, _ = w
            m = (expmap == gi)
            if bn == "WAVG":
                r = wch["res"]
            else:
                r = (ch - P[bn]).astype(np.int32)
            resplane[m] = r[m]
        check_range(resplane, "C5 GRID residuals")
        symplane = resplane
    return symplane.astype(np.int32), expmap, resplane


def golomb_adaptive_encode(sym_flat, act_flat, reset=64):
    """Real C-gol bitstream (CROWN polarity) + analytic cross-check.

    sym_flat/act_flat: flat sequences length N (ints). Returns (pay_bytes, nbits).
    Mirrors probe_b19_run.verify_golomb_stream put() but bulk-unary for speed;
    asserts analytic (AD.golomb_adaptive_bits) == emitted bits (exactness debt paid).
    """
    st = AD.GolombState(AD.NACT, reset=reset)
    bw = BitWriter()
    nbits = 0
    for rv, ev in zip(sym_flat, act_flat):
        ctx = AD.act_bin(ev)
        k = st.k(ctx)
        M = 2 * abs(int(rv)) - (1 if int(rv) > 0 else 0)
        q = M >> k
        rem = M & ((1 << k) - 1) if k else 0
        if q:
            bw.put(0, int(q))
        bw.put(1, 1)
        if k:
            bw.put(int(rem), int(k))
        nbits += int(q) + 1 + int(k)
        st.upd(ctx, abs(int(rv)))
    pay = bw.flush()
    ana = AD.golomb_adaptive_bits(list(sym_flat), list(act_flat), reset=reset, use_act=True)
    assert nbits == ana, f"C-gol analytic/stream MISMATCH {nbits} vs {ana} (abort)"
    assert len(pay) * 8 - nbits < 8 and len(pay) * 8 >= nbits, "padding invariant"
    return pay, nbits


def channel_header_bytes(plan, wch, H, W, nbw, ng32):
    """Build channel prefix bytes up to per-group predid meta (no payload).

    Returns (prefix, cmap_or_zeros, grid_thr, predid_arr, ng, fam).
    """
    fam = plan["family"]
    wins = plan["wins"]
    ng = len(wins)
    assert 1 <= ng <= 64, ng
    blob = bytearray()
    kidx = WIDE_IDX[plan["K"]] if fam == 0 else 15
    blob.append((fam & 1) | ((plan["grid"] & 7) << 1) | ((kidx & 15) << 4))
    blob.append(ng & 0xFF)
    blob.append(0)  # placeholder acodec, patched by caller
    bw = BitWriter()
    for g in range(ng32):
        bw.put(int(wch["use"][g]), 1)
    blob += bw.flush()
    bw = BitWriter()
    for g in range(ng32):
        if wch["use"][g]:
            for t in range(4):
                bw.put(int(wch["w"][g, t]) & 31, 5)
    blob += bw.flush()
    if fam == 0:
        key = None
        cmap = np.zeros(729, dtype=np.uint8)
        for gi, gkeys in enumerate(plan["groups"]):
            for u in gkeys:
                cmap[int(u)] = gi
        # active-key mask via C4 convention: keys present in D["key"]; caller passes uk
        grid_thr = np.zeros(5, dtype=np.int32)
    else:
        cmap = np.zeros(729, dtype=np.uint8)
        grid_thr = np.array(GRIDS[plan["grid"]], dtype=np.int32)
    predid = np.zeros(ng, dtype=np.uint8)
    for gi, w in enumerate(wins):
        pid = EXPID[w[1]]
        assert pid <= 18, pid
        predid[gi] = pid
    return blob, cmap, grid_thr, predid, ng, fam


def assemble_channel_adaptive(plan, D, ch, wch, H, W, nbw, ng32, uk_keys=None):
    """Full CROWN5 channel assembly with per-channel best-of {A-act, C-gol}.

    Returns (blob, meta) where meta holds decode-side arrays + diagnostics.
    Byte-exact: prefix + map + 5b predids + adaptive payload (lengths counted).
    """
    symplane, expmap, _ = symplane_from_plan(plan, D, ch, wch)
    actplane = AD.activity_plane(ch)
    check_range(symplane, "C5 symplane pre-encode")
    fam = plan["family"]
    wins = plan["wins"]
    ng = len(wins)
    # ---- A-act real encode (probe-verbatim) ----
    aenc = AD.cabac_encode_plane(symplane, actplane, use_act=True)
    arith, raw = aenc["arith"], aenc["raw"]
    # ---- C-gol analytic for selection (fast), real stream only if wins ----
    sym_flat = symplane.reshape(-1).tolist()
    act_flat = actplane.reshape(-1).tolist()
    cg_ana = AD.golomb_adaptive_bits(sym_flat, act_flat, reset=64, use_act=True)
    # framing-aware comparison on real bytes (C-gol pad estimated ceil)
    a_pay = 4 + len(arith) + 4 + len(raw)
    c_pay_est = 4 + (cg_ana + 7) // 8
    # full prefix sizes (needed for exact gate)
    # build prefix skeleton to measure header (map+meta) identically both paths
    pre, cmap0, grid_thr0, predid0, ng0, fam0 = channel_header_bytes(plan, wch, H, W, nbw, ng32)
    # map bytes
    if fam == 0:
        key = D["key"]
        uk = np.unique(key)
        gbits = math.ceil(math.log2(plan["K"]))
        maplen = 92 + (len(uk) * gbits + 7) // 8
    else:
        maplen = 1
    metalen = (ng * 5 + 7) // 8
    hdrlen = len(pre) + maplen + metalen
    a_tot = hdrlen + a_pay
    c_tot = hdrlen + c_pay_est
    if a_tot <= c_tot:
        acodec = 0
        pay = int(len(arith)).to_bytes(4, "little") + arith + int(len(raw)).to_bytes(4, "little") + raw
        cg_bits = None
    else:
        acodec = 1
        pay_bytes, cg_bits = golomb_adaptive_encode(sym_flat, act_flat, reset=64)
        pay = int(len(pay_bytes)).to_bytes(4, "little") + pay_bytes
    # ---- assemble ----
    blob = bytearray()
    kidx = WIDE_IDX[plan["K"]] if fam == 0 else 15
    blob.append((fam & 1) | ((plan["grid"] & 7) << 1) | ((kidx & 15) << 4))
    blob.append(ng & 0xFF)
    blob.append(acodec & 0xFF)
    bw = BitWriter()
    for g in range(ng32):
        bw.put(int(wch["use"][g]), 1)
    blob += bw.flush()
    bw = BitWriter()
    for g in range(ng32):
        if wch["use"][g]:
            for t in range(4):
                bw.put(int(wch["w"][g, t]) & 31, 5)
    blob += bw.flush()
    if fam == 0:
        key = D["key"]
        uk = np.unique(key)
        mask = np.zeros(729, dtype=np.uint8)
        mask[uk] = 1
        blob += np.packbits(mask).tobytes()
        assert len(blob) == 3 + (ng32 + 7) // 8 + ((int(wch["use"].sum()) * 20 + 7) // 8) + 92
        cmap = np.zeros(729, dtype=np.uint8)
        for gi, gkeys in enumerate(plan["groups"]):
            for u in gkeys:
                cmap[int(u)] = gi
        gbits = math.ceil(math.log2(plan["K"]))
        bw = BitWriter()
        for u in sorted(uk.tolist()):
            bw.put(int(cmap[int(u)]), gbits)
        blob += bw.flush()
        grid_thr = np.zeros(5, dtype=np.int64)
    else:
        occ = 0
        for gi in range(ng):
            if (expmap == gi).sum() > 0:
                occ |= (1 << gi)
        blob.append(occ & 0xFF)
        cmap = np.zeros(729, dtype=np.uint8)
        grid_thr = np.array(GRIDS[plan["grid"]], dtype=np.int64)
    bw = BitWriter()
    for gi, w in enumerate(wins):
        bw.put(int(EXPID[w[1]]), 5)
    blob += bw.flush()
    blob += pay
    # decode-side meta
    predid = np.zeros(128, dtype=np.uint8)
    for gi, w in enumerate(wins):
        predid[gi] = EXPID[w[1]]
    meta = {"ng": ng, "fam": fam, "acodec": acodec, "predid": predid,
            "cmap": cmap, "grid": np.array(grid_thr, dtype=np.int32),
            "wuse": wch["use"].astype(np.uint8), "ww": np.ascontiguousarray(wch["w"]),
            "nbw": nbw, "ng32": ng32,
            "arith": arith, "raw": raw, "cg_bits": cg_bits,
            "a_pay": a_pay, "c_pay_est": c_pay_est, "nbytes": len(blob)}
    return bytes(blob), meta


def encode_image_rct(rgb, rname, perm, t):
    H, W, _ = rgb.shape
    yc = B17.rct_fwd(rgb, perm, t)
    assert np.array_equal(B17.rct_inv(yc, perm, t), rgb), f"RCT {rname} round-trip FAIL"
    planes = [yc[:, :, c] for c in range(3)]
    wchs, bm, nbh, nbw, ng32, wside = C4.fit_weighted(planes, H, W)
    lms0 = [C4.lms_pred_plane(pl, 0) for pl in planes]
    lms3 = [C4.lms_pred_plane(pl, 3) for pl in planes]
    out = bytearray()
    out += b"C5" + H.to_bytes(2, "little") + W.to_bytes(2, "little") + bytes([1, RCTID[rname]])
    infos = []
    for ci, ch in enumerate(planes):
        D = prepX(ch)
        RF = {n: (D["s"] * (ch - D["P"][n]).astype(np.int32)).astype(np.int32) for n in C4.E16}
        RF["WAVG"] = (D["s"] * wchs[ci]["res"]).astype(np.int32)
        RF["LMS5_T0"] = (D["s"] * (ch - lms0[ci]).astype(np.int32)).astype(np.int32)
        RF["LMS5_T3"] = (D["s"] * (ch - lms3[ci]).astype(np.int32)).astype(np.int32)
        for n in E19:
            check_range(RF[n], f"flip {rname} ch{ci} {n}")
        RU = {n: (ch - D["P"][n]).astype(np.int32) for n in C4.E16}
        RU["WAVG"] = wchs[ci]["res"]
        for n in E17:
            check_range(RU[n], f"plain {rname} ch{ci} {n}")
        wch_q = dict(wchs[ci])
        wch_q["lms0"] = (ch - lms0[ci]).astype(np.int32)
        wch_q["lms3"] = (ch - lms3[ci]).astype(np.int32)
        check_range(wch_q["lms0"], f"lms0 {rname} ch{ci}")
        check_range(wch_q["lms3"], f"lms3 {rname} ch{ci}")
        pq = C4.encode_channel_q(ch, D, RF)
        pg = C4.encode_channel_grid(ch, D, RU)
        # ---- 4 candidates: Q-static (CROWN4-verbatim), Q-adapt, G-static, G-adapt ----
        # static via imported CROWN4 assembler (read-only reuse, exact bytes)
        bq_st, mq_st = C4.assemble_channel(pq, D, ch, wch_q, H, W, nbw, ng32)
        bg_st, mg_st = C4.assemble_channel(pg, D, ch, wchs[ci], H, W, nbw, ng32)
        bq_st_u = bq_st[:2] + bytes([2]) + bq_st[2:]  # insert choice=2 (static)
        bg_st_u = bg_st[:2] + bytes([2]) + bg_st[2:]
        bq_ad, mq_ad = assemble_channel_adaptive(pq, D, ch, wch_q, H, W, nbw, ng32)
        bg_ad, mg_ad = assemble_channel_adaptive(pg, D, ch, wchs[ci], H, W, nbw, ng32)
        cands = [("Q", 2, bq_st_u, None), ("G", 2, bg_st_u, None),
                 ("Q", mq_ad["acodec"], bq_ad, mq_ad), ("G", mg_ad["acodec"], bg_ad, mg_ad)]
        cands.sort(key=lambda t_: len(t_[2]))
        fam_w, chc_w, blob_w, meta_ad = cands[0]
        out += blob_w
        infos.append({"fam": fam_w, "choice": chc_w, "meta": meta_ad, "nbytes": len(blob_w),
                      "qst": len(bq_st_u), "gst": len(bg_st_u),
                      "qad": len(bq_ad), "gad": len(bg_ad)})
    return bytes(out), {"rct": rname, "chs": infos, "ng32": ng32, "nbw": nbw,
                        "wuse": [int(w["use"].sum()) for w in wchs]}


def encode_image(rgb):
    cands = []
    for (rname, perm, t) in RCTS:
        t0 = time.perf_counter()
        blob, info = encode_image_rct(rgb, rname, perm, t)
        info["ms"] = (time.perf_counter() - t0) * 1000
        cands.append((len(blob), blob, info))
    cands.sort(key=lambda t_: t_[0])
    return cands[0][1], {"winner": cands[0][2],
                         "losers": [(c[2]["rct"], len(c[1]), c[2]["ms"]) for c in cands[1:]]}


def decode_image(blob):
    H = int.from_bytes(blob[2:4], "little")
    W = int.from_bytes(blob[4:6], "little")
    assert blob[:2] == b"C5", blob[:2]
    assert blob[6] == 1, blob[6]
    rct = blob[7] & 3
    assert rct <= 2, f"RCT id VIOLATION {rct} (abort)"
    N = H * W
    p = 8
    planes = []
    for _ in range(3):
        b0, ng, choice = blob[p], blob[p + 1], blob[p + 2]
        assert choice <= 2, f"CHOICE VIOLATION {choice} (abort)"
        p += 3
        fam = b0 & 1
        grid = (b0 >> 1) & 7
        assert fam in (0, 1) and grid <= 7, (fam, grid)
        assert 1 <= ng <= 64, f"NG VIOLATION {ng} (abort)"
        cmap = np.zeros(729, dtype=np.uint8)
        grid_thr = np.zeros(5, dtype=np.int32)
        occ = 0xFF
        mask = None
        K = None
        gbits = 0
        nbw = math.ceil(W / GS)
        nbh = math.ceil(H / GS)
        ng32 = nbh * nbw
        nbytes = (ng32 + 7) // 8
        raw = blob[p:p + nbytes]
        assert len(raw) == nbytes, "FRAMING VIOLATION wuse (abort)"
        p += nbytes
        wuse = np.zeros(ng32, dtype=np.uint8)
        acc, nb, qp = 0, 0, 0
        for g in range(ng32):
            while nb < 1:
                assert qp < len(raw), "FRAMING VIOLATION wuse bits (abort)"
                acc = (acc << 8) | raw[qp]
                qp += 1
                nb += 8
            nb -= 1
            wuse[g] = (acc >> nb) & 1
            acc &= ((1 << nb) - 1) if nb else 0
        nused = int(wuse.sum())
        nbytes = (nused * 20 + 7) // 8
        raw = blob[p:p + nbytes] if nbytes else b""
        assert len(raw) == nbytes, "FRAMING VIOLATION ww (abort)"
        p += nbytes
        ww = np.zeros((ng32, 4), dtype=np.int8)
        acc, nb, qp = 0, 0, 0
        for g in range(ng32):
            if wuse[g]:
                for tt in range(4):
                    while nb < 5:
                        assert qp < len(raw), "FRAMING VIOLATION ww bits (abort)"
                        acc = (acc << 8) | raw[qp]
                        qp += 1
                        nb += 8
                    nb -= 5
                    v = (acc >> nb) & 31
                    acc &= ((1 << nb) - 1) if nb else 0
                    ww[g, tt] = v - 32 if v >= 16 else v
        if fam == 0:
            assert p + 92 <= len(blob), "FRAMING VIOLATION qmask (abort)"
            mask = np.unpackbits(np.frombuffer(blob[p:p + 92], dtype=np.uint8))[:729]
            p += 92
            kidx = (b0 >> 4) & 15
            assert kidx in WIDE_IDX.values(), f"KIDX VIOLATION {kidx} (abort)"
            K = WIDE[kidx]
            gbits = math.ceil(math.log2(K))
            na = int(mask.sum())
            nbytes = (na * gbits + 7) // 8
            assert p + nbytes <= len(blob), "FRAMING VIOLATION qids (abort)"
            raw = blob[p:p + nbytes]
            p += nbytes
            acc, nb, qp = 0, 0, 0
            uk = np.where(mask)[0]
            for u in uk:
                while nb < gbits:
                    assert qp < len(raw), "FRAMING VIOLATION qid bits (abort)"
                    acc = (acc << 8) | raw[qp]
                    qp += 1
                    nb += 8
                nb -= gbits
                cmap[u] = (acc >> nb) & ((1 << gbits) - 1)
                acc &= ((1 << nb) - 1) if nb else 0
        else:
            assert p < len(blob), "FRAMING VIOLATION occ (abort)"
            occ = blob[p]
            p += 1
            grid_thr = np.array(GRIDS[grid], dtype=np.int32)
        if choice in (0, 1):
            acodec = choice
            nbytes = (ng * 5 + 7) // 8
            assert p + nbytes <= len(blob), "FRAMING VIOLATION predid (abort)"
            raw = blob[p:p + nbytes]
            p += nbytes
            predid = np.zeros(128, dtype=np.uint8)
            acc, nb, qp = 0, 0, 0
            for gi in range(ng):
                while nb < 5:
                    assert qp < len(raw), "FRAMING VIOLATION predid bits (abort)"
                    acc = (acc << 8) | raw[qp]
                    qp += 1
                    nb += 8
                nb -= 5
                v = (acc >> nb) & 31
                acc &= ((1 << nb) - 1) if nb else 0
                predid[gi] = v
                assert v <= 18, f"PREDID VIOLATION {v} (abort)"
            if acodec == 0:
                assert p + 4 <= len(blob), "FRAMING VIOLATION alen (abort)"
                alen = int.from_bytes(blob[p:p + 4], "little")
                p += 4
                assert p + alen <= len(blob), "FRAMING VIOLATION arith (abort)"
                abuf = blob[p:p + alen]
                p += alen
                assert p + 4 <= len(blob), "FRAMING VIOLATION rlen (abort)"
                rlen = int.from_bytes(blob[p:p + 4], "little")
                p += 4
                assert p + rlen <= len(blob), "FRAMING VIOLATION raw (abort)"
                rbuf = blob[p:p + rlen]
                p += rlen
                gbuf, glen = b"", 0
            else:
                assert p + 4 <= len(blob), "FRAMING VIOLATION paylen (abort)"
                glen = int.from_bytes(blob[p:p + 4], "little")
                p += 4
                assert p + glen <= len(blob), "FRAMING VIOLATION pay (abort)"
                gbuf = blob[p:p + glen]
                p += glen
                abuf, alen, rbuf, rlen = b"", 0, b"", 0
            plane = np.zeros(N, dtype=np.int16)
            ab = (c_u8 * len(abuf))(*abuf) if len(abuf) else (c_u8 * 0)()
            rb = (c_u8 * len(rbuf))(*rbuf) if len(rbuf) else (c_u8 * 0)()
            gb = (c_u8 * len(gbuf))(*gbuf) if len(gbuf) else (c_u8 * 0)()
            rc = C5.crown5_decode_ch(
                ab, len(abuf), rb, len(rbuf), gb, len(gbuf),
                cmap.ctypes.data_as(ctypes.POINTER(c_u8)),
                predid.ctypes.data_as(ctypes.POINTER(c_u8)),
                wuse.ctypes.data_as(ctypes.POINTER(c_u8)),
                ww.ctypes.data_as(ctypes.POINTER(c_i8)), nbw,
                fam, grid_thr.ctypes.data_as(ctypes.POINTER(c_i32)), acodec,
                plane.ctypes.data_as(ctypes.POINTER(c_i16)), H, W, ng)
            assert rc == 0, f"CROWN5 adaptive channel decode FAIL rc={rc} (loud)"
            planes.append(plane)
        else:
            # ---- static CROWN4-verbatim channel suffix (choice==2) ----
            nbytes = (ng * 7 + 7) // 8
            assert p + nbytes <= len(blob), "FRAMING VIOLATION meta7 (abort)"
            raw = blob[p:p + nbytes]
            p += nbytes
            predid = np.zeros(128, dtype=np.uint8)
            backend = np.zeros(128, dtype=np.uint8)
            acc, nb, qp = 0, 0, 0
            for gi in range(ng):
                while nb < 7:
                    assert qp < len(raw), "FRAMING VIOLATION meta7 bits (abort)"
                    acc = (acc << 8) | raw[qp]
                    qp += 1
                    nb += 8
                nb -= 7
                v = (acc >> nb) & 0x7F
                acc &= ((1 << nb) - 1) if nb else 0
                predid[gi] = (v >> 2) & 31
                backend[gi] = v & 3
                assert predid[gi] <= 18 and backend[gi] <= 2, (gi, predid[gi], backend[gi])
            nG = int((backend[:ng] == 1).sum())
            kvals = np.zeros(128, dtype=np.uint8)
            dbias = np.zeros(128, dtype=np.int8)
            nbytes = (nG * 4 + 7) // 8
            raw = blob[p:p + nbytes] if nbytes else b""
            assert len(raw) == nbytes, "FRAMING VIOLATION kvals (abort)"
            p += nbytes
            acc, nb, qp = 0, 0, 0
            for gi in range(ng):
                if backend[gi] == 1:
                    while nb < 4:
                        assert qp < len(raw), "FRAMING VIOLATION kval bits (abort)"
                        acc = (acc << 8) | raw[qp]
                        qp += 1
                        nb += 8
                    nb -= 4
                    kvals[gi] = (acc >> nb) & 15
                    acc &= ((1 << nb) - 1) if nb else 0
            nbytes = (nG * 3 + 7) // 8
            raw = blob[p:p + nbytes] if nbytes else b""
            assert len(raw) == nbytes, "FRAMING VIOLATION dbias (abort)"
            p += nbytes
            acc, nb, qp = 0, 0, 0
            for gi in range(ng):
                if backend[gi] == 1:
                    while nb < 3:
                        assert qp < len(raw), "FRAMING VIOLATION dbias bits (abort)"
                        acc = (acc << 8) | raw[qp]
                        qp += 1
                        nb += 8
                    nb -= 3
                    v = (acc >> nb) & 7
                    acc &= ((1 << nb) - 1) if nb else 0
                    dbias[gi] = v - 8 if v >= 4 else v
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
                    assert p + 2 <= len(blob), "FRAMING VIOLATION H-A (abort)"
                    A_ = int.from_bytes(blob[p:p + 2], "little")
                    p += 2
                    syms = []
                    for _ in range(A_):
                        assert p + 3 <= len(blob), "FRAMING VIOLATION H-sym (abort)"
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
            assert p + 4 <= len(blob), "FRAMING VIOLATION hn (abort)"
            hn = int.from_bytes(blob[p:p + 4], "little")
            p += 4
            assert p + hn <= len(blob), "FRAMING VIOLATION hpay (abort)"
            hpay = blob[p:p + hn]
            p += hn
            assert p + 4 <= len(blob), "FRAMING VIOLATION gn (abort)"
            gn_ = int.from_bytes(blob[p:p + 4], "little")
            p += 4
            assert p + gn_ <= len(blob), "FRAMING VIOLATION gpay (abort)"
            gpay = blob[p:p + gn_]
            p += gn_
            HAPRE = C4.HAPRE
            LIBC4 = C4.C4
            rlist, goff = [], [0]
            for gi in range(ng):
                if backend[gi] == 2:
                    assert p + 4 <= len(blob), "FRAMING VIOLATION R-cnt (abort)"
                    cnt = int.from_bytes(blob[p:p + 4], "little")
                    p += 4
                    assert p + 2 <= len(blob), "FRAMING VIOLATION R-A (abort)"
                    A_ = int.from_bytes(blob[p:p + 2], "little")
                    p += 2
                    syms32 = np.zeros(A_, np.int32)
                    freq = np.zeros(A_, np.uint16)
                    cum = np.zeros(A_, np.int32)
                    c = 0
                    for i in range(A_):
                        assert p + 4 <= len(blob), "FRAMING VIOLATION R-sym (abort)"
                        sv = int.from_bytes(blob[p:p + 2], "little")
                        f = int.from_bytes(blob[p + 2:p + 4], "little")
                        p += 4
                        syms32[i] = (sv + 1024) & 0xFFFF
                        freq[i] = f
                        cum[i] = c
                        c += f
                    assert p + 4 <= len(blob), "FRAMING VIOLATION R-n (abort)"
                    n_ = int.from_bytes(blob[p:p + 4], "little")
                    p += 4
                    assert p + n_ <= len(blob), "FRAMING VIOLATION R-pay (abort)"
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
            plane = np.zeros(N, dtype=np.int16)
            hb = (c_u8 * len(hpay))(*hpay) if len(hpay) else (c_u8 * 0)()
            gb = (c_u8 * len(gpay))(*gpay) if len(gpay) else (c_u8 * 0)()
            rs = rsyms.ctypes.data_as(ctypes.POINTER(c_i16)) if rsyms.size else None
            kvals8 = kvals.astype(np.uint8)
            rc = LIBC4.crown4_decode_ch(
                hb, len(hpay), gb, len(gpay), rs, goff.ctypes.data_as(ctypes.POINTER(c_i32)), ng,
                cd.ctypes.data_as(ctypes.POINTER(c_u32)), ln.ctypes.data_as(ctypes.POINTER(c_u8)),
                cmap.ctypes.data_as(ctypes.POINTER(c_u8)), predid.ctypes.data_as(ctypes.POINTER(c_u8)),
                backend.ctypes.data_as(ctypes.POINTER(c_u8)), kvals8.ctypes.data_as(ctypes.POINTER(c_u8)),
                dbias.ctypes.data_as(ctypes.POINTER(c_i8)), hasn.ctypes.data_as(ctypes.POINTER(c_u8)),
                wuse.ctypes.data_as(ctypes.POINTER(c_u8)), ww.ctypes.data_as(ctypes.POINTER(c_i8)), nbw,
                fam, grid_thr.ctypes.data_as(ctypes.POINTER(c_i32)),
                plane.ctypes.data_as(ctypes.POINTER(c_i16)), H, W)
            assert rc == 0, f"CROWN5 static channel decode FAIL rc={rc} (loud)"
            planes.append(plane)
    p0 = np.ascontiguousarray(planes[0].reshape(H, W))
    p1 = np.ascontiguousarray(planes[1].reshape(H, W))
    p2 = np.ascontiguousarray(planes[2].reshape(H, W))
    out = bytearray(N * 3)
    rc = C5.crown5_rct_inv(p0.ctypes.data_as(ctypes.POINTER(c_i16)),
                           p1.ctypes.data_as(ctypes.POINTER(c_i16)),
                           p2.ctypes.data_as(ctypes.POINTER(c_i16)), H, W, rct,
                           (ctypes.c_char * len(out)).from_buffer(out))
    assert rc == 0, f"RCT inv range FAIL rc={rc} (loud)"
    assert p == len(blob), f"FRAMING VIOLATION tail p={p} len={len(blob)} (abort)"
    return bytes(out), {"rct": RCTS[rct][0]}


def wilcoxon_exact(diffs):
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


def selftest_tiny():
    import numpy as _np
    _np.random.seed(0)
    tiny = (_np.zeros((8, 8, 3), dtype=_np.uint8))
    tiny[:, :] = _np.array([12, 200, 33], dtype=_np.uint8)
    tiny[0, 0] = [255, 0, 255]
    blob, _ = encode_image(tiny)
    out, _ = decode_image(blob)
    assert _np.array_equal(_np.frombuffer(out, dtype=_np.uint8).reshape(8, 8, 3), tiny), \
        "CROWN5 tiny round-trip FAIL (loud)"
    # random blocks incl saturated edges (alphabet/border stress)
    rng = _np.random.default_rng(1)
    for trial in range(2):
        px = rng.integers(0, 256, size=(16, 16, 3)).astype(_np.uint8)
        px[0, :] = 255
        px[:, 0] = 0
        blob, _ = encode_image(px)
        out, _ = decode_image(blob)
        assert _np.array_equal(_np.frombuffer(out, dtype=_np.uint8).reshape(16, 16, 3), px), \
            f"CROWN5 random round-trip FAIL trial {trial} (loud)"
    print("[crown5 selftest_tiny PASS: 8x8 + 2x16x16 random-edge round-trips ok]", flush=True)


if __name__ == "__main__":
    import json as _json
    selftest_tiny()
    _t_all = time.perf_counter()
    _res = {}
    _bpps = []
    for _path in IMAGES:
        _fn = _path.split("/")[-1]
        _px = np.array(Image.open(_path).convert("RGB"))
        _H, _W, _ = _px.shape
        _npx = _H * _W * 3
        _t0 = time.perf_counter()
        _blob, _info = encode_image(_px)
        _enc_ms = (time.perf_counter() - _t0) * 1000
        _t1 = time.perf_counter()
        _out, _di = decode_image(_blob)
        _dec_ms = (time.perf_counter() - _t1) * 1000
        assert np.array_equal(
            np.frombuffer(_out, dtype=np.uint8).reshape(_H, _W, 3), _px), \
            f"ROUND-TRIP FAIL {_fn} (loud, abort)"
        _bpp = len(_blob) * 8 / _npx
        _chs = _info["winner"]["chs"]
        _res[_fn] = {"bpp": round(float(_bpp), 4), "bytes": len(_blob),
                     "rct": _info["winner"]["rct"],
                     "fams": [c["fam"] for c in _chs],
                     "choices": [int(c["choice"]) for c in _chs],
                     "nbytes": [int(c["nbytes"]) for c in _chs],
                     "qst": [int(c["qst"]) for c in _chs],
                     "gst": [int(c["gst"]) for c in _chs],
                     "qad": [int(c["qad"]) for c in _chs],
                     "gad": [int(c["gad"]) for c in _chs],
                     "enc_ms": round(_enc_ms), "dec_ms": round(_dec_ms)}
        _bpps.append(_bpp)
        _d4 = (_bpp - CROWN4_BAR) / CROWN4_BAR * 100
        _dJ = (_bpp - JXL_E3[_fn]) / JXL_E3[_fn] * 100
        _dP = (_bpp - CROWN4_PER[_fn]) / CROWN4_PER[_fn] * 100
        print(f"{_fn}: C5={_bpp:.4f} vs CR4ex-avg {CROWN4_BAR} ({_d4:+.2f}% vs avg) "
              f"vs JXL {_dJ:+.2f}% vs CR4-per {_dP:+.2f}% "
              f"RCT={_res[_fn]['rct']} fams={_res[_fn]['fams']} "
              f"ch={_res[_fn]['choices']} RT-PASS "
              f"[enc {_enc_ms/1000:.0f}s dec {_dec_ms:.0f}ms]", flush=True)
    import numpy as _np3
    _avg = float(_np3.mean(_bpps))
    print(f"AVG={_avg:.4f} vs CROWN4-exact {CROWN4_BAR} "
          f"({(_avg-CROWN4_BAR)/CROWN4_BAR*100:+.3f}%) | "
          f"probe-transfer 3.1138 est | JXL-e3 3.2291", flush=True)
    _diffs = [_res[f]["bpp"] - JXL_E3[f] for f in
              ["kodim01.png", "kodim02.png", "kodim05.png", "kodim07.png",
               "kodim13.png", "kodim19.png", "kodim23.png"]]
    _p, _w = wilcoxon_exact(_diffs)
    _wins = sum(1 for d in _diffs if d < 0)
    print(f"vs JXL-e3: {_wins}W-{7-_wins}L p={_p:.3f} "
          f"{'BOSS-5 KO (p<.05)' if (_p < 0.05 and _wins >= 6) else 'Boss-5 STANDS'}", flush=True)
    _d4a = [_res[f]["bpp"] - CROWN4_PER[f] for f in
            ["kodim01.png", "kodim02.png", "kodim05.png", "kodim07.png",
             "kodim13.png", "kodim19.png", "kodim23.png"]]
    _p4, _w4 = wilcoxon_exact(_d4a)
    _wins4 = sum(1 for d in _d4a if d < 0)
    print(f"vs CROWN4-exact: {_wins4}W-{7-_wins4}L p={_p4:.3f}", flush=True)
    print(f"TOTAL {(time.perf_counter()-_t_all)/60:.1f} min", flush=True)
    print(_json.dumps(_res, indent=1))
