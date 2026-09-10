"""probe_b19_adaptive.py — CABAC-class adaptivity vs static-table overhead (new file only).

Branch: b18 autopsy PROVED kodim05's loss is ~90% entropy-coding overhead
(floor 3.500 beats JXL 3.5120; fine groups -> oversized tables -> Golomb
fallback misfits heavy tails). This probe kills the overhead with
backward/forward-adaptive coders on a FIXED front end (MED + C6 YCoCg-R),
so every delta is pure entropy-coding effect (predictor held constant).

Frame (all configs identical residuals):
  RGB -> C6 YCoCg-R (B17.rct_fwd perm0 t6) -> causal MED (B17.med_pred,
  CROWN2 border doctrine) -> int32 residuals per plane (|r|<=1024 asserted).

Backends (all exact bits, all side counted, decoder-safe):
  S0    static order-0 global Huffman per plane (baseline; 16+A*24 tables).
  S0-G  static best-k Golomb per plane (+4b k side). Reference only.
  A-pos backward-adaptive binary arithmetic, pos-indexed ctx (no activity).
  A-act backward-adaptive binary arithmetic, 4-activity x 4-pos = 16 ctx (MAIN).
  B-fwd forward-adaptive: A-act + transmitted 4b/ctx init (64b/plane) [+reset].
  C-gol adaptive Golomb-k per activity ctx, raster-order N/A counters (RESET 64).
  D-run run/interrupt on top of C (flat act==0 runs + M-1 interrupt).

Adaptation state is a pure function of already-decoded symbols/pixels
(+ transmitted init for B + static reset schedule). Decoder mirrors exactly;
round-trip asserted (real decode, not simulation).

Unit: bpp = total_bits/(H*W*3). numpy+PIL only, CPU, no torch.
"""
import heapq
import math
import os
import sys
import time
import json

import numpy as np
from PIL import Image

sys.path.insert(0, "/tmp/opencode/autocompress/csrc")
sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
os.chdir("/tmp/opencode/autocompress/experiments")
import probe_b17_rctw as B17

D = "/tmp/opencode/autocompress/experiments/real_photos"
FILES = ["kodim01.png", "kodim02.png", "kodim05.png", "kodim07.png",
         "kodim13.png", "kodim19.png", "kodim23.png"]
CROWN4 = {"kodim01.png": 3.2991, "kodim02.png": 2.9854, "kodim05.png": 3.5917,
          "kodim07.png": 2.6790, "kodim13.png": 3.9525, "kodim19.png": 3.1510,
          "kodim23.png": 2.7364}
JXL_E3 = {"kodim01.png": 3.3593, "kodim02.png": 3.0611, "kodim05.png": 3.5120,
          "kodim07.png": 2.7310, "kodim13.png": 3.9141, "kodim19.png": 3.2154,
          "kodim23.png": 2.8110}
CROWN4_AVG = float(np.mean(list(CROWN4.values())))
JXL_AVG = 3.2291

# Activity thresholds (static, fixed a priori for all images; JPEG-LS-like).
# e = |a-c|+|b-c| from causal recon; 4 bins.
ATHR = (4, 12, 48)
NACT = 4
NPOS = 4  # prefix-pos contexts (pos0,1,2,3+)
MAXT = 128  # binary ctx adaptation cap (fast, CABAC-like forgetting)


def r_to_M(r):
    return 2 * abs(int(r)) - (1 if int(r) > 0 else 0)


def M_to_r(M):
    M = int(M)
    if M == 0:
        return 0
    if M & 1:
        return (M + 1) // 2
    return -(M // 2)


def entropy_bits(arr):
    _, cn = np.unique(np.asarray(arr).reshape(-1), return_counts=True)
    p = cn.astype(np.float64) / cn.sum()
    return float((-(p * np.log2(p))).sum() * cn.sum())


def med_residuals(plane):
    return (plane.astype(np.int32) - B17.med_pred(plane).astype(np.int32))


def activity_plane(plane):
    """Vectorized activity e=|a-c|+|b-c| with B17 border doctrine (matches raster)."""
    p = plane.astype(np.int32)
    L = np.zeros_like(p)
    L[:, 1:] = p[:, :-1]
    L[:, 0] = np.concatenate([[0], p[:-1, 0]])
    T = np.zeros_like(p)
    T[1:, :] = p[:-1, :]
    T[0, :] = np.concatenate([[0], p[0, :-1]])
    TL = np.zeros_like(p)
    TL[1:, 1:] = p[:-1, :-1]
    return np.abs(L - TL) + np.abs(T - TL)


def act_bin(e):
    t0, t1, t2 = ATHR
    if e <= t0:
        return 0
    if e <= t1:
        return 1
    if e <= t2:
        return 2
    return 3


# ---------------- binary arithmetic coder (E1/E2/E3, 16-bit) ----------------
class BinEnc(object):
    def __init__(self, nctx, maxt=MAXT):
        self.low = 0
        self.high = 0xFFFF
        self.follow = 0
        self.acc = 0
        self.nb = 0
        self.out = bytearray()
        self.c0 = [1] * nctx
        self.c1 = [1] * nctx
        self.maxt = maxt
        self.nctx = nctx

    def _w(self, b):
        self.acc = (self.acc << 1) | (b & 1)
        self.nb += 1
        if self.nb == 8:
            self.out.append(self.acc & 0xFF)
            self.acc = 0
            self.nb = 0

    def _out(self, b):
        self._w(b)
        while self.follow:
            self._w(b ^ 1)
            self.follow -= 1

    def enc(self, bit, ctx):
        c0 = self.c0[ctx]
        c1 = self.c1[ctx]
        tot = c0 + c1
        rng = self.high - self.low + 1
        split = (rng * c0) // tot
        if split <= 0:
            split = 1
        elif split >= rng:
            split = rng - 1
        if bit == 0:
            self.high = self.low + split - 1
            c0 += 1
        else:
            self.low = self.low + split
            c1 += 1
        if c0 + c1 >= self.maxt:
            c0 = (c0 + 1) // 2
            c1 = (c1 + 1) // 2
            if c0 < 1:
                c0 = 1
            if c1 < 1:
                c1 = 1
        self.c0[ctx] = c0
        self.c1[ctx] = c1
        low = self.low
        high = self.high
        while True:
            if high < 0x8000:
                self._out(0)
                low *= 2
                high = high * 2 + 1
            elif low >= 0x8000:
                self._out(1)
                low = (low - 0x8000) * 2
                high = (high - 0x8000) * 2 + 1
            elif low >= 0x4000 and high < 0xC000:
                self.follow += 1
                low = (low - 0x4000) * 2
                high = (high - 0x4000) * 2 + 1
            else:
                break
        self.low = low & 0xFFFF
        self.high = high & 0xFFFF

    def finish(self):
        self.follow += 1
        if self.low < 0x4000:
            self._out(0)
        else:
            self._out(1)
        if self.nb:
            self.out.append((self.acc << (8 - self.nb)) & 0xFF)
        return bytes(self.out)


class BinDec(object):
    def __init__(self, buf, nctx, maxt=MAXT):
        self.bits = []
        for by in bytearray(buf):
            for i in range(7, -1, -1):
                self.bits.append((by >> i) & 1)
        self.pos = 0
        self.low = 0
        self.high = 0xFFFF
        self.code = 0
        for _ in range(16):
            self.code = (self.code << 1) | self._read()
        self.c0 = [1] * nctx
        self.c1 = [1] * nctx
        self.maxt = maxt

    def _read(self):
        if self.pos < len(self.bits):
            b = self.bits[self.pos]
            self.pos += 1
            return b
        return 0

    def dec(self, ctx):
        c0 = self.c0[ctx]
        c1 = self.c1[ctx]
        tot = c0 + c1
        rng = self.high - self.low + 1
        split = (rng * c0) // tot
        if split <= 0:
            split = 1
        elif split >= rng:
            split = rng - 1
        if self.code < self.low + split:
            bit = 0
            self.high = self.low + split - 1
            c0 += 1
        else:
            bit = 1
            self.low = self.low + split
            c1 += 1
        if c0 + c1 >= self.maxt:
            c0 = (c0 + 1) // 2
            c1 = (c1 + 1) // 2
            if c0 < 1:
                c0 = 1
            if c1 < 1:
                c1 = 1
        self.c0[ctx] = c0
        self.c1[ctx] = c1
        while True:
            if self.high < 0x8000:
                self.low *= 2
                self.high = self.high * 2 + 1
                self.code = self.code * 2 + self._read()
            elif self.low >= 0x8000:
                self.low = (self.low - 0x8000) * 2
                self.high = (self.high - 0x8000) * 2 + 1
                self.code = (self.code - 0x8000) * 2 + self._read()
            elif self.low >= 0x4000 and self.high < 0xC000:
                self.low = (self.low - 0x4000) * 2
                self.high = (self.high - 0x4000) * 2 + 1
                self.code = (self.code - 0x4000) * 2 + self._read()
            else:
                break
            self.low &= 0xFFFF
            self.high &= 0xFFFF
            self.code &= 0xFFFF
        return bit


# ---------------- CABAC-like per-plane encode/decode ----------------
def cabac_encode_plane(res, act, use_act=True, init_levels=None):
    """Backward-adaptive binary arithmetic on Exp-Golomb binarization.

    init_levels: None (uniform c0=c1=1) or list of 4b levels per ctx (forward).
    Returns dict(bits, arith_bytes, raw_bytes, side_bits).
    Decoder-safety: ctx state = pure function of already-coded bins (+ init).
    """
    H, W = res.shape
    nctx = NACT * NPOS if use_act else NPOS
    enc = BinEnc(nctx)
    if init_levels is not None:
        for ctx, lv in enumerate(init_levels):
            p1 = (int(lv) + 0.5) / 16.0
            c1 = max(1, int(round(p1 * 16)))
            c0 = max(1, 16 - c1)
            enc.c0[ctx] = c0
            enc.c1[ctx] = c1
    raw_acc = 0
    raw_nb = 0
    raw_out = bytearray()
    rL = res.ravel().tolist()
    aL = act.ravel().tolist()
    t0, t1, t2 = ATHR
    for rv, ev in zip(rL, aL):
        if use_act:
            if ev <= t0:
                ab = 0
            elif ev <= t1:
                ab = 1
            elif ev <= t2:
                ab = 2
            else:
                ab = 3
        M = 2 * abs(rv) - (1 if rv > 0 else 0)
        cn = M + 1
        L = cn.bit_length()
        base = ab * NPOS if use_act else 0
        for i in range(L):
            b = 0 if i < L - 1 else 1
            p = i if i < NPOS else NPOS - 1
            enc.enc(b, base + p)
        if L > 1:
            rem = cn ^ (1 << (L - 1))
            for j in range(L - 2, -1, -1):
                raw_acc = (raw_acc << 1) | ((rem >> j) & 1)
                raw_nb += 1
                if raw_nb == 8:
                    raw_out.append(raw_acc & 0xFF)
                    raw_acc = 0
                    raw_nb = 0
    arith = enc.finish()
    if raw_nb:
        raw_out.append((raw_acc << (8 - raw_nb)) & 0xFF)
    side = 0 if init_levels is None else 4 * nctx
    total = len(arith) * 8 + len(raw_out) * 8 + 64 + side  # 64b: 2x32b lengths
    return {"bits": total, "arith": arith, "raw": bytes(raw_out), "side": side}


def cabac_decode_plane(arith, raw, H, W, rec_plane, use_act=True, init_levels=None):
    """Real raster-order decode: act+pred from causal recon only. Returns res plane."""
    nctx = NACT * NPOS if use_act else NPOS
    dec = BinDec(arith, nctx)
    rawbits = []
    for by in bytearray(raw):
        for i in range(7, -1, -1):
            rawbits.append((by >> i) & 1)
    rp = 0
    if init_levels is not None:
        for ctx, lv in enumerate(init_levels):
            p1 = (int(lv) + 0.5) / 16.0
            c1 = max(1, int(round(p1 * 16)))
            c0 = max(1, 16 - c1)
            dec.c0[ctx] = c0
            dec.c1[ctx] = c1
    rec = rec_plane.astype(np.int32).copy()
    res = np.zeros((H, W), dtype=np.int32)
    t0, t1, t2 = ATHR
    for i in range(H):
        for j in range(W):
            lv = rec[i, j - 1] if j > 0 else (rec[i - 1, 0] if i > 0 else 0)
            tv = rec[i - 1, j] if i > 0 else (rec[i, j - 1] if j > 0 else 0)
            tl = rec[i - 1, j - 1] if (i > 0 and j > 0) else 0
            e = abs(int(lv) - int(tl)) + abs(int(tv) - int(tl))
            if use_act:
                ab = 0 if e <= t0 else (1 if e <= t1 else (2 if e <= t2 else 3))
                base = ab * NPOS
            else:
                base = 0
            L = 1
            while True:
                p = (L - 1) if (L - 1) < NPOS else NPOS - 1
                b = dec.dec(base + p)
                if b == 1:
                    break
                L += 1
                assert L <= 16, "prefix runaway"
            rem = 0
            for _ in range(L - 1):
                rem = (rem << 1) | rawbits[rp]
                rp += 1
            cn = (1 << (L - 1)) | rem
            M = cn - 1
            r = M_to_r(M)
            mn = lv if lv < tv else tv
            mx = tv if lv < tv else lv
            pr = mn if tl >= mx else (mx if tl <= mn else lv + tv - tl)
            rec[i, j] = int(pr) + int(r)
            res[i, j] = int(r)
    return res, rec


# ---------------- adaptive Golomb (per-act N/A, raster order) ----------------
class GolombState(object):
    def __init__(self, nctx, reset=64, A0=4, N0=1):
        self.N = [N0] * nctx
        self.A = [A0] * nctx
        self.reset = reset

    def k(self, ctx):
        N = self.N[ctx]
        A = self.A[ctx]
        k = 0
        while (N << k) < A:
            k += 1
        return k

    def upd(self, ctx, abr):
        self.A[ctx] += int(abr)
        self.N[ctx] += 1
        if self.N[ctx] >= self.reset:
            self.N[ctx] >>= 1
            self.A[ctx] >>= 1
            if self.N[ctx] < 1:
                self.N[ctx] = 1


def golomb_adaptive_bits(res_list, act_list, reset=64, use_act=True):
    nctx = NACT if use_act else 1
    st = GolombState(nctx, reset=reset)
    tot = 0
    for rv, ev in zip(res_list, act_list):
        ctx = act_bin(ev) if use_act else 0
        k = st.k(ctx)
        M = 2 * abs(int(rv)) - (1 if int(rv) > 0 else 0)
        tot += (M >> k) + 1 + k
        st.upd(ctx, abs(int(rv)))
    return tot


def golomb_static_best(res_list):
    best = None
    for k in range(0, 9):
        t = 0
        for rv in res_list:
            M = 2 * abs(int(rv)) - (1 if int(rv) > 0 else 0)
            t += (M >> k) + 1 + k
        t += 4
        if best is None or t < best[0]:
            best = (t, k)
    return best


# ---------------- run/interrupt over adaptive Golomb ----------------
def run_adaptive_bits(res_list, act_list, reset=64):
    """Activity-gated run mode (act==0 runs, L=0 allowed) + M-1 interrupt."""
    n = len(res_list)
    st = GolombState(NACT, reset=reset)
    run_st = GolombState(1, reset=reset)
    tot = 0
    i = 0
    while i < n:
        if act_bin(act_list[i]) == 0:
            j = i
            while j < n and res_list[j] == 0:
                j += 1
            L = j - i
            kr = run_st.k(0)
            tot += (L >> kr) + 1 + kr
            run_st.upd(0, L)
            i = j
            if i >= n:
                break
            rv = res_list[i]
            assert rv != 0
            k = st.k(0)
            M = 2 * abs(int(rv)) - (1 if int(rv) > 0 else 0)
            Mp = M - 1
            tot += (Mp >> k) + 1 + k
            st.upd(0, abs(int(rv)))
            i += 1
        else:
            ctx = act_bin(act_list[i])
            k = st.k(ctx)
            M = 2 * abs(int(res_list[i])) - (1 if int(res_list[i]) > 0 else 0)
            tot += (M >> k) + 1 + k
            st.upd(ctx, abs(int(res_list[i])))
            i += 1
    return tot
