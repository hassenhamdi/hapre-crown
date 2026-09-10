"""CROWN6 driver: exact bit-level codec = CROWN4 line + per-context micro-MLP expert.

Stack (encoder decisions in Python via proven probe modules; streaming decode in
NEW libcrown6.so; packing/rANS via READ-ONLY libhapre.so):
  Front end: GLOBAL best-RCT over {C6,C27,C12} per image (2 bits, homogeneous
             planes; probe_b17 M1/M4 verbatim).
  Predict:   E16 bank (probe_b5_c) + WAVG 17th (probe_b17 G32 LS 4-tap) +
             LMS5_T0 18th + LMS5_T3 19th (probe_b18_stack verbatim, ZERO side) +
             MLP 20th expert (id 19): per-channel gated assembly of per-energy-ctx
             12feat->8->1 tanh micro-MLPs (probe_b21 committed config: L1 loss,
             LS-init, it200/lr0.003, adaptive-int16 {256..4096}, double MDL gate).
             Non-winning/empty contexts fall back to edge-MED (probe verbatim).
  Group:     CROWN4 Q-family (sign-flipped LOCO-365 + WIDE-K quantile, E20 search)
             vs GRID-family (M-THR energy grids, E18 = E17+MLP), per-channel
             exact-assembled-byte MDL gate (MLP-variant vs no-MLP-variant).
  Backend:   per-group Huff/Golomb/rANS 3-way + Golomb-only G-bias adapters.
             Golomb unary polarity: q ZEROS + one 1 + k-bit remainder (vendored
             crown6_golomb_pack; libhapre era-variant NOT wire-compatible).

MLP wire forward (float64, FIXED order, bit-exact numpy==C by construction):
  dequant w = int16 / S exactly (S = 2^k power of two -> exact in binary FP);
  pre[j] = b1[j]; for i in 0..11: pre[j] += (x[i]/128.0)*W1[j][i];
  h[j] = LUT-tanh(pre[j]) via FROZEN table (src/crown6_tanhlut.h, gen_tanhlut.py):
    rr=|pre|*16384 (exact); rr>131072 -> +/-1.0;
    else idx=floor(rr+0.5), h=sign(pre)*T[idx]  [T[i]=tanh(i/16384)];
  y = b2; for j in 0..7: y += W2[j]*h[j];
  pred = floor(y*128.0 + 0.5)  (half-up; DIFFERS from torch .round half-even
  on exact .5 values -- quantified in xcheck, encoder ALWAYS uses this path).
  The frozen LUT resolves the mandated-cross-check finding: numpy vectorized
  tanh vs C libm tanh differ by 1 ulp on some inputs (e.g. pre=19.06), which
  flipped int-rounded outputs at knife-edge .5 boundaries. Identical constants
  on both sides => h bits identical by construction (xcheck proves 0 mismatch).
  Features x[12] are ZERO-border causal taps (c6_causal_planes: tap = plane
  value or 0 outside; interior bit-identical to probe_b21's edge-replicate
  features, but edge-replication peeks at undecoded pixels on row 0 / col 0
  and broke streaming decode -- deviation [D6]). Energy ctx from causal L/T/TL;
  thresholds float64.

Training: probe_b21_perctx imports reused (train_net, adaptive_quantize,
ctx_of_plane, features_targets, med_pred_plane, hbits). Nets are trained per
candidate-RCT planes (MLP statistics are plane-distribution-specific; YCoCg-R
nets would mismatch RCT planes). Per-unit RNG seeding (deterministic under
.npz cache hits). Gate bit-costs use the WIRE forward (not torch) so gates
reflect transmitted predictions exactly.
Deviations from probe_b21 (all documented here):
  [D1] planes are RCT{C6,C27,C12} (not YCoCg-R) -- required by CROWN line.
  [D2] per-unit torch seeding (cache-determinism); hyperparams identical.
  [D3] gate/prediction forward is wire-float64, not torch-float32 (xcheck
       quantifies torch-vs-wire int disagreement; C matches wire exactly).
  [D4] thresholds stored float64 (probe ledger assumed int16; quantiles can be
       fractional -- float64 is exact, bytes counted honestly).
  [D5] expert id 19 (keeps WAVG/LMS; "17th family" counts E16+MLP only).
  [D6] MLP features/fallback-MED use ZERO-border causal taps, not probe_b21's
       edge-replication (which peeks at undecoded pixels on row 0/col 0 --
       causality bug-class; forced-MLP streaming test diverged at pixel
       (0,0)). Interior pixels bit-identical to probe; border-only change.

No existing files touched (this file + crown6.c + CROWN6_FORMAT.md + .npz are new).
Toolchain: numpy+PIL+gcc+ctypes+torch-CPU only.
"""
import ctypes
import hashlib
import math
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, "/tmp/opencode/autocompress/src")
sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
# (removed: import-time os.chdir side effect; use absolute paths)
import probe_b4_a as A
import probe_b17_rctw as B17
import probe_b18_stack as S18
import probe_b21_perctx as B21
import torch
from probe_b5_c import predictors_X, prepX, E16
from probe_b5_d import golomb_best
from driver import canon_tables

torch.set_num_threads(12)

HAPRE = ctypes.CDLL("/tmp/opencode/autocompress/src/libhapre.so")
C6 = ctypes.CDLL("/tmp/opencode/autocompress/src/libcrown6.so")
c_u8 = ctypes.c_uint8
c_i8 = ctypes.c_int8
c_i16 = ctypes.c_int16
c_u16 = ctypes.c_uint16
c_i32 = ctypes.c_int32
c_u32 = ctypes.c_uint32
c_i64 = ctypes.c_int64
c_f64 = ctypes.c_double

HAPRE.pack_syms.argtypes = [ctypes.POINTER(c_i16), ctypes.POINTER(c_u8), ctypes.c_int,
                            ctypes.POINTER(c_u32), ctypes.POINTER(c_u8),
                            ctypes.POINTER(c_u8), ctypes.c_size_t]
HAPRE.pack_syms.restype = ctypes.c_size_t
C6.golomb_pack = C6.crown6_golomb_pack
C6.golomb_pack.argtypes = [ctypes.POINTER(c_i16), ctypes.POINTER(c_u8), ctypes.POINTER(c_u8),
                           ctypes.c_int, ctypes.POINTER(c_u8), ctypes.c_size_t]
C6.golomb_pack.restype = ctypes.c_size_t
HAPRE.rans_norm.argtypes = [ctypes.POINTER(c_i64), ctypes.c_int, ctypes.POINTER(c_i32),
                            ctypes.POINTER(c_u16), ctypes.POINTER(c_i32)]
HAPRE.rans_norm.restype = None
HAPRE.rans_encode.argtypes = [ctypes.POINTER(c_i16), ctypes.c_int, ctypes.POINTER(c_i32),
                              ctypes.POINTER(c_u16), ctypes.POINTER(c_i32), ctypes.POINTER(c_u8)]
HAPRE.rans_encode.restype = ctypes.c_size_t
HAPRE.rans_slots.argtypes = [ctypes.POINTER(c_u16), ctypes.POINTER(c_i32), ctypes.c_int,
                             ctypes.POINTER(c_i32)]
HAPRE.rans_decode.argtypes = [ctypes.POINTER(c_u8), ctypes.c_size_t, ctypes.c_int,
                              ctypes.POINTER(c_i32), ctypes.POINTER(c_u16), ctypes.POINTER(c_i32),
                              ctypes.c_int, ctypes.POINTER(c_i32), ctypes.POINTER(c_i16)]
HAPRE.rans_decode.restype = ctypes.c_int
C6.crown6_decode_ch.argtypes = [ctypes.POINTER(c_u8), ctypes.c_size_t,
                                ctypes.POINTER(c_u8), ctypes.c_size_t,
                                ctypes.POINTER(c_i16), ctypes.POINTER(c_i32), ctypes.c_int,
                                ctypes.POINTER(c_u32), ctypes.POINTER(c_u8),
                                ctypes.POINTER(c_u8), ctypes.POINTER(c_u8), ctypes.POINTER(c_u8),
                                ctypes.POINTER(c_u8), ctypes.POINTER(c_i8), ctypes.POINTER(c_u8),
                                ctypes.POINTER(c_u8), ctypes.POINTER(c_i8), ctypes.c_int,
                                ctypes.c_int, ctypes.POINTER(c_i32),
                                ctypes.POINTER(c_i16), ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.POINTER(c_f64), ctypes.POINTER(c_u8),
                                ctypes.POINTER(c_f64)]
C6.crown6_decode_ch.restype = ctypes.c_int
C6.crown6_rct_inv.argtypes = [ctypes.POINTER(c_i16), ctypes.POINTER(c_i16), ctypes.POINTER(c_i16),
                              ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
C6.crown6_rct_inv.restype = ctypes.c_int
C6.crown6_mlp_predict_batch.argtypes = [ctypes.POINTER(c_i32), ctypes.c_int,
                                        ctypes.POINTER(c_f64), ctypes.POINTER(c_f64),
                                        ctypes.POINTER(c_f64), c_f64,
                                        ctypes.POINTER(c_i32)]
C6.crown6_mlp_predict_batch.restype = None

E19 = E16 + ["WAVG", "LMS5_T0", "LMS5_T3"]
E20 = E19 + ["MLP"]
E17 = E16 + ["WAVG"]
E18 = E17 + ["MLP"]
EXPID = {n: i for i, n in enumerate(E20)}
MLP_ID = 19
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
GS = 32
RCTS = [("C6", 0, 6), ("C27", 3, 6), ("C12", 1, 5)]
RCTID = {n: i for i, (n, _, _) in enumerate(RCTS)}

# ---- MLP committed config (probe_b21) ----
MLP_H = 8
MLP_FEAT = 12
MLP_ITERS = 200
MLP_LR = 0.003
MLP_NCTX = 9
MLP_NPAR = 113
MLP_SCALES = (256, 512, 1024, 2048, 4096)
WDIR = "/tmp/opencode/autocompress/experiments/crown6_weights"

IMAGES = [    "/tmp/opencode/autocompress/experiments/real_photos/kodim01.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim02.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim05.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim07.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim13.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim19.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim23.png",
]
JXL_E3 = {"kodim01.png": 3.3593, "kodim02.png": 3.0611, "kodim05.png": 3.5120,
          "kodim07.png": 2.7310, "kodim13.png": 3.9141, "kodim19.png": 3.2154,
          "kodim23.png": 2.8110}
CROWN4_BAR = 3.1993
FORCE_MLP = os.environ.get("CROWN6_FORCE_MLP") == "1"  # debug: prefer MLP variant
_debug_fail = []  # debug: partial-plane store on decode failure


def unit_seed(*parts):
    h = hashlib.sha256(("|".join(map(str, parts))).encode()).hexdigest()[:8]
    return int(h, 16)


def check_range(arr, where):
    m = int(np.abs(np.asarray(arr)).max()) if np.asarray(arr).size else 0
    assert m <= RES_MAX, f"ALPHABET VIOLATION {where}: |r|max={m} > 1024 (abort, never clip)"


# ================= causal MLP features (ZERO-border; encoder==decoder) =================
# probe_b21's edge-replicate features peek at UNDECODED pixels on row 0 / col 0
# (L/T/TR replicate the current/future pixel) -- a causality violation caught by
# the forced-MLP streaming test (first divergence at pixel (0,0)). The MLP path
# therefore uses zero-border causal taps (interior pixels are bit-identical to
# the probe; only border taps differ). Deviation [D6], see FORMAT.md.

def c6_causal_planes(P):
    """Zero-border causal neighbor planes. tap(di,dj)[i,j] = P[i+di,j+dj] or 0."""
    Pp = np.pad(np.asarray(P, dtype=np.int32), ((3, 0), (3, 2)), mode="constant")
    H, W = P.shape
    sl = lambda di, dj: Pp[3 + di:3 + di + H, 3 + dj:3 + dj + W]
    a = sl(0, -1); b = sl(-1, 0); c = sl(-1, -1); d = sl(-1, +1)
    Ww = sl(0, -2); NNe = sl(-2, 0)
    L3 = sl(0, -3); T3 = sl(-3, 0); TL2 = sl(-2, -2); TR2 = sl(-2, +2)
    return a, b, c, d, Ww, NNe, L3, T3, TL2, TR2


def c6_features_targets(P, nfeat=12):
    assert nfeat == 12
    a, b, c, d, Ww, NNe, L3, T3, TL2, TR2 = c6_causal_planes(P)
    F = np.stack([a, b, c, d, Ww, NNe, (a + b) // 2, np.abs(a - b),
                  L3, T3, TL2, TR2], -1)
    return F, P.reshape(-1).astype(np.float32), (a, b, c)


def c6_med_plane(P):
    a, b, c, *_ = c6_causal_planes(P)
    return np.where(c >= np.maximum(a, b), np.minimum(a, b),
                    np.where(c <= np.minimum(a, b), np.maximum(a, b), a + b - c))


def c6_ctx_of_plane(P):
    a, b, c, *_ = c6_causal_planes(P)
    E = np.abs(a - b) + np.abs(a - c) + np.abs(b - c)
    qs = np.quantile(E.reshape(-1).astype(np.float64), np.linspace(0, 1, MLP_NCTX + 1)[1:-1])
    ctx = np.searchsorted(qs, E.reshape(-1)).reshape(E.shape).astype(np.int32)
    return np.clip(ctx, 0, MLP_NCTX - 1), qs.astype(np.float64), E


# ================= MLP wire forward (numpy side of the wire spec) =================
import math as _math
_C6_TANH = np.array([_math.tanh(i / 16384.0) for i in range(8 * 16384 + 1)],
                    dtype=np.float64)  # same formula as crown6_tanhlut.h

def dequant_net(qw, scale):
    """qw: (113,) int16 -> dict of float64 arrays. Order: W1(96) b1(8) W2(8) b2(1)."""
    w = qw.astype(np.float64) / float(scale)  # exact: scale is a power of two
    return {"W1": w[0:96].reshape(8, 12), "b1": w[96:104],
            "W2": w[104:112], "b2": float(w[112])}


def mlp_forward_wire(F, q):
    """Wire-spec forward, fixed accumulation order (matches crown6.c bit-exactly).

    F: (N,12) int32 edge-features. q: dequant dict. Returns (N,) int32.
    """
    F = np.asarray(F)
    N = F.shape[0]
    W1, b1, W2, b2 = q["W1"], q["b1"], q["W2"], q["b2"]
    pre = np.broadcast_to(b1, (N, 8)).copy()
    for i in range(12):
        pre += (F[:, i].astype(np.float64) / 128.0)[:, None] * W1[:, i][None, :]
    rr = np.abs(pre) * 16384.0
    big = rr > 131072.0
    idx = np.floor(np.minimum(rr, 131072.0) + 0.5).astype(np.int64)
    sgn = np.where(pre < 0.0, -1.0, 1.0)
    h = sgn * _C6_TANH[idx]
    h = np.where(big, sgn, h)
    y = np.full(N, b2, dtype=np.float64)
    for j in range(8):
        y += h[:, j] * W2[j]
    return np.floor(y * 128.0 + 0.5).astype(np.int32)


def c_mlp_predict_batch(F, q):
    """Same predictions via libcrown6 (for xcheck)."""
    F = np.ascontiguousarray(np.asarray(F, dtype=np.int32))
    N = F.shape[0]
    W1 = np.ascontiguousarray(q["W1"], dtype=np.float64)
    b1 = np.ascontiguousarray(q["b1"], dtype=np.float64)
    W2 = np.ascontiguousarray(q["W2"], dtype=np.float64)
    out = np.zeros(N, dtype=np.int32)
    C6.crown6_mlp_predict_batch(
        F.ctypes.data_as(ctypes.POINTER(c_i32)), N,
        W1.ctypes.data_as(ctypes.POINTER(c_f64)),
        b1.ctypes.data_as(ctypes.POINTER(c_f64)),
        W2.ctypes.data_as(ctypes.POINTER(c_f64)), c_f64(q["b2"]),
        out.ctypes.data_as(ctypes.POINTER(c_i32)))
    return out


# ================= per-channel MLP training (probe_b21 config, wire gates) =================

def train_channel_mlps(P, tag):
    """Train+gated per-ctx nets on one int32 plane. Returns netinfo dict.

    netinfo: {qs:(8,)f8, win:(9,)u8, scid:(9,)u8 (scale idx, 255 absent),
              qw:(9,113)i16, pred:(H,W)i32 gated pred plane, side_bits:int,
              has_nets:bool, nwin:int}
    Gate bit-costs use the WIRE forward (== transmitted predictions).
    """
    H, W = P.shape
    F, y, _ = c6_features_targets(P, MLP_FEAT)
    Ff = F.reshape(-1, MLP_FEAT)
    ctx, qs, E = c6_ctx_of_plane(P)
    cf = ctx.reshape(-1)
    med_plane = c6_med_plane(P)
    med_res = (P - med_plane).reshape(-1)
    win = np.zeros(MLP_NCTX, dtype=np.uint8)
    scid = np.full(MLP_NCTX, 255, dtype=np.uint8)
    qw = np.zeros((MLP_NCTX, MLP_NPAR), dtype=np.int16)
    pred_vec = med_plane.reshape(-1).copy()
    side = 0
    for k in range(MLP_NCTX):
        m = (cf == k)
        nk = int(m.sum())
        if nk < 100:
            side += 1  # fallback flag still transmitted (probe parity)
            continue
        mb_ctx = B21.hbits(med_res[m])
        torch.manual_seed(unit_seed("crown6", tag, k, MLP_H, MLP_FEAT, MLP_ITERS, MLP_LR))
        net, _ = B21.train_net(Ff[m], y[m], MLP_FEAT, MLP_H, MLP_ITERS, MLP_LR)
        neta, scale = B21.adaptive_quantize(net)
        si = MLP_SCALES.index(scale)
        ma = B21.maxabs_of(neta)
        assert ma * scale <= 32767.0 + 1e-9, f"quant clip risk {tag} ctx{k}"
        with torch.no_grad():
            p = {n: v.numpy() for n, v in neta.named_parameters()}
        qi = np.concatenate([p["fc1.weight"].reshape(-1), p["fc1.bias"].reshape(-1),
                             p["fc2.weight"].reshape(-1), p["fc2.bias"].reshape(-1)])
        qi16 = np.round(qi * scale).astype(np.int64)
        assert qi16.max() <= 32767 and qi16.min() >= -32768, f"int16 clip {tag} ctx{k}"
        q = dequant_net(qi16.astype(np.int16), scale)
        pv = mlp_forward_wire(Ff[m], q)
        qb = B21.hbits(P.reshape(-1)[m].astype(np.int32) - pv)
        if qb + MLP_NPAR * 16 + 1 + 3 < mb_ctx:  # L1 double-gate (probe verbatim)
            win[k] = 1
            scid[k] = si
            qw[k] = qi16.astype(np.int16)
            pred_vec[m] = pv
            side += MLP_NPAR * 16 + 1 + 3
        else:
            side += 1
    asm_q = B21.hbits(P.reshape(-1).astype(np.int32) - pred_vec)
    mb = B21.hbits(med_res)
    thr_bits = 8 * 64
    ch_q_total = asm_q + side + thr_bits
    if ch_q_total >= mb or int(win.sum()) == 0:  # L2 channel gate (probe verbatim)
        return {"qs": np.asarray(qs, dtype=np.float64), "win": np.zeros(9, np.uint8),
                "scid": np.full(9, 255, np.uint8), "qw": np.zeros((9, 113), np.int16),
                "pred": med_plane.copy(), "side_bits": 0, "has_nets": False,
                "nwin": 0, "fallback": True}
    return {"qs": np.asarray(qs, dtype=np.float64), "win": win, "scid": scid, "qw": qw,
            "pred": pred_vec.reshape(H, W).copy(), "side_bits": side + thr_bits,
            "has_nets": True, "nwin": int(win.sum()), "fallback": False}


def mlp_side_bytes(ni):
    """Exact wire bytes of MLP side info (thresholds + scale ids + weights)."""
    assert ni["has_nets"]
    b = bytearray()
    b += np.ascontiguousarray(ni["qs"], dtype=np.float64).tobytes()
    b += bytes(ni["scid"].tolist())
    for k in range(MLP_NCTX):
        if ni["win"][k]:
            b += np.ascontiguousarray(ni["qw"][k], dtype=np.int16).tobytes()
    return bytes(b)


def save_npz(fn, rct, ch, ni):
    os.makedirs(WDIR, exist_ok=True)
    path = os.path.join(WDIR, f"crown6_{fn}.npz")
    d = {}
    if os.path.exists(path):
        with np.load(path) as z:
            for k in z.files:
                d[k] = z[k]
    p = f"{rct}/ch{ch}"
    d[f"{p}/qs"] = ni["qs"]
    d[f"{p}/win"] = ni["win"]
    d[f"{p}/scid"] = ni["scid"]
    d[f"{p}/qw"] = ni["qw"]
    np.savez(path, **d)


def load_npz(fn, rct, ch, P):
    path = os.path.join(WDIR, f"crown6_{fn}.npz")
    if not os.path.exists(path):
        return None
    with np.load(path) as z:
        p = f"{rct}/ch{ch}"
        if f"{p}/qs" not in z.files:
            return None
        ni = {"qs": z[f"{p}/qs"], "win": z[f"{p}/win"], "scid": z[f"{p}/scid"],
              "qw": z[f"{p}/qw"]}
    # rebuild gated pred plane with WIRE forward (== what training stored)
    H, W = P.shape
    F, _, _ = c6_features_targets(P, MLP_FEAT)
    Ff = F.reshape(-1, MLP_FEAT)
    ctx, qs_now, _ = c6_ctx_of_plane(P)
    if not np.array_equal(np.asarray(ni["qs"]), np.asarray(qs_now)):
        return None  # thresholds must reproduce (guards stale caches)
    med_plane = c6_med_plane(P)
    pred_vec = med_plane.reshape(-1).copy()
    cf = ctx.reshape(-1)
    nwin = 0
    for k in range(MLP_NCTX):
        if ni["win"][k]:
            scale = MLP_SCALES[int(ni["scid"][k])]
            q = dequant_net(ni["qw"][k], scale)
            m = (cf == k)
            pred_vec[m] = mlp_forward_wire(Ff[m], q)
            nwin += 1
    side = nwin * (MLP_NPAR * 16 + 1 + 3) + (MLP_NCTX - nwin) * 1 + 8 * 64
    ni.update({"pred": pred_vec.reshape(H, W).copy(), "side_bits": side,
               "has_nets": nwin > 0, "nwin": nwin, "fallback": nwin == 0})
    return ni


# ================= exact-cost helpers (CROWN4-verbatim semantics) =================

def huff_cost(g):
    g = np.asarray(g).reshape(-1)
    if g.size == 0:
        return 0, 0, 0
    uv, cn = np.unique(g, return_counts=True)
    d = A.huff_bits(cn.tolist())
    return d + 16 + len(cn) * 24, d, len(cn)


def golomb_cost(g):
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
    HAPRE.rans_norm(hist.ctypes.data_as(ctypes.POINTER(c_i64)), Ad,
                    syms.ctypes.data_as(ctypes.POINTER(c_i32)),
                    freq.ctypes.data_as(ctypes.POINTER(c_u16)),
                    cum.ctypes.data_as(ctypes.POINTER(c_i32)))
    lut = np.full(2049, -1, dtype=np.int32)
    for i, s in enumerate(syms):
        lut[s] = i
    out = np.zeros(N * 3 + 16, dtype=np.uint8)
    n = HAPRE.rans_encode(g.ctypes.data_as(ctypes.POINTER(c_i16)), N,
                          lut.ctypes.data_as(ctypes.POINTER(c_i32)),
                          freq.ctypes.data_as(ctypes.POINTER(c_u16)),
                          cum.ctypes.data_as(ctypes.POINTER(c_i32)),
                          out.ctypes.data_as(ctypes.POINTER(c_u8)))
    assert n > 0
    slot = np.zeros(1 << 14, dtype=np.int32)
    HAPRE.rans_slots(freq.ctypes.data_as(ctypes.POINTER(c_u16)),
                     cum.ctypes.data_as(ctypes.POINTER(c_i32)), Ad,
                     slot.ctypes.data_as(ctypes.POINTER(c_i32)))
    dec = np.zeros(N, dtype=np.int16)
    rc = HAPRE.rans_decode(out.ctypes.data_as(ctypes.POINTER(c_u8)), n, N,
                           syms.ctypes.data_as(ctypes.POINTER(c_i32)),
                           freq.ctypes.data_as(ctypes.POINTER(c_u16)),
                           cum.ctypes.data_as(ctypes.POINTER(c_i32)), Ad,
                           slot.ctypes.data_as(ctypes.POINTER(c_i32)),
                           dec.ctypes.data_as(ctypes.POINTER(c_i16)))
    assert rc == 0 and np.array_equal(dec, g), "rANS roundtrip FAIL"
    return 16 + Ad * 32 + 64 + n * 8, (bytes(out[:n]), syms, freq, cum)


def quantile_groups(key, rfM, K):
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


def best_hg(gd, skip=()):
    """Best (H/G) over experts in gd dict. Returns [bits, name, backend, k, d]."""
    bw = None
    for n, g in gd.items():
        if n in skip:
            continue
        g = np.asarray(g).reshape(-1)
        if g.size == 0:
            ht, gt, gk, gdlt = 0, 0, 0, 0
        else:
            ht, _, _ = huff_cost(g)
            gt, gk, gdlt, _ = golomb_cost(g)
        if bw is None or ht < bw[0]:
            bw = [ht, n, 0, 0, 0]
        if gt < bw[0]:
            bw = [gt, n, 1, gk, gdlt]
    return bw


def solve_groups_dual(group_vals, use_rans=True):
    """Dual-track exact select: with-MLP vs without-MLP.

    group_vals: list per group of dict name->col (includes 'MLP' iff nets exist).
    Returns (tot_all, wins_all, tot_no, wins_no); win = (bits, name, be, k, d, blob).
    Cost semantics identical to driver_crown4.solve_groups.
    """
    tot_all, tot_no = 0, 0
    wins_all, wins_no = [], []
    for gd in group_vals:
        tracks = []
        for skip in ((), ("MLP",)):
            bw = best_hg(gd, skip)
            cost2, bn, bb, bk, bd = bw
            blob = None
            if use_rans:
                g = np.asarray(gd[bn]).reshape(-1)
                if g.size:
                    rt, bl = rans_cost(g)
                    if rt < cost2:
                        bw = [rt, bn, 2, 0, 0]
                        blob = bl
            g = np.asarray(gd[bw[1]]).reshape(-1)
            if bw[2] == 1 and g.size:
                gt, gk, gdlt, _ = golomb_cost(g)
                bw = [gt, bw[1], 1, gk, gdlt]
            tracks.append((bw[0], bw[1], bw[2], bw[3], bw[4], blob if bw[2] == 2 else None))
        (ta, ba) = (tracks[0][0], tracks[0])
        (tn, bn_) = (tracks[1][0], tracks[1])
        tot_all += ta
        tot_no += tn
        wins_all.append(ba)
        wins_no.append(bn_)
    return tot_all, wins_all, tot_no, wins_no


def fit_weighted(planes, H, W):
    out_res, use_all, wall_all, side, bm, ng = B17.weighted_fit(planes, H, W, GS)
    nbh = math.ceil(H / GS)
    nbw = math.ceil(W / GS)
    assert ng == nbh * nbw, (ng, nbh, nbw)
    chs = []
    for c in range(3):
        check_range(out_res[c], f"WAVG res ch{c}")
        use = np.array(use_all[c], dtype=np.uint8)
        w = np.zeros((ng, 4), dtype=np.int8)
        for g, wq in enumerate(wall_all[c]):
            if wq is not None:
                w[g] = wq.astype(np.int8)
        chs.append({"res": out_res[c].astype(np.int32), "use": use, "w": w})
    return chs, bm, nbh, nbw, ng, side


def lms_pred_plane(plane, thr):
    return S18.lms_pred_plane(plane, thr)


def encode_channel_q(ch, D, RF, has_mlp):
    key, s = D["key"], D["s"]
    rfM = RF["MED"]
    uk_all = np.unique(key)
    cands = []
    for K in WIDE:
        groups = quantile_groups(key, rfM, K)
        gbits = math.ceil(math.log2(K))
        mapside = 16 + 736 + len(uk_all) * gbits
        gv = []
        for gkeys in groups:
            sel = np.isin(key, np.array(gkeys))
            gv.append({n: RF[n][sel] for n in (E20 if has_mlp else E19)})
        sub, _, _, _ = solve_groups_dual(gv, use_rans=False)
        cands.append((mapside + len(groups) * 6 + sub, K, groups))
    cands.sort(key=lambda t: t[0])
    best = None
    for _, K, groups in cands[:3]:
        gv = []
        for gkeys in groups:
            sel = np.isin(key, np.array(gkeys))
            gv.append({n: RF[n][sel] for n in (E20 if has_mlp else E19)})
        sa, wa, sn, wn = solve_groups_dual(gv, use_rans=True)
        if best is None or min(sa, sn) < best[0]:
            best = (min(sa, sn), K, groups, wa, wn)
    _, K, groups, wa, wn = best
    return {"family": 0, "K": K, "groups": groups, "wins_all": wa, "wins_no": wn, "grid": 0}


def encode_channel_grid(ch, D, RU, has_mlp):
    a, b, c = D["a"], D["b"], D["c"]
    e = (np.abs(a - c) + np.abs(b - c)).astype(np.int64)
    names = E18 if has_mlp else E17
    best = None
    for gi, thr in GRIDS.items():
        gg = np.digitize(e.ravel(), thr, right=True)
        gv = []
        for c_ in range(6):
            sel = (gg == c_).reshape(e.shape)
            gv.append({n: RU[n][sel] for n in names})
        sa, _, sn, _ = solve_groups_dual(gv, use_rans=False)
        t = 16 + 8 + len(gv) * 6 + min(sa, sn)
        if best is None or t < best[0]:
            best = (t, gi, gg.reshape(e.shape))
    _, gi, gg = best
    gv = []
    for c_ in range(6):
        sel = (gg == c_)
        gv.append({n: RU[n][sel] for n in names})
    sa, wa, sn, wn = solve_groups_dual(gv, use_rans=True)
    return {"family": 1, "K": 6, "groups": None, "wins_all": wa, "wins_no": wn,
            "grid": gi, "gmap": gg}


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


def assemble_channel(plan, D, ch, wch, mlp_res, mlp_side, use_mlp, H, W, nbw, ng32):
    """Build exact channel bytes for one dual-track choice. Returns (blob, nmlp_groups)."""
    N = H * W
    fam = plan["family"]
    wins = plan["wins_all"] if use_mlp else plan["wins_no"]
    ng = len(wins)
    if use_mlp:
        assert mlp_side is not None
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
            elif bn == "MLP":
                r = mlp_res
            else:
                r = (ch - P[bn]).astype(np.int32)
            resplane[m] = r[m]
        check_range(resplane, "Q residuals")
        symplane = (s.astype(np.int32) * resplane)
        check_range(symplane, "Q flipped")
        uk = np.unique(key)
        gbits = math.ceil(math.log2(plan["K"]))
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
            elif bn == "MLP":
                r = mlp_res
            else:
                r = (ch - P[bn]).astype(np.int32)
            resplane[m] = r[m]
        check_range(resplane, "GRID residuals")
        symplane = resplane
        uk, gbits = None, 0
    back = np.array([w[2] for w in wins], dtype=np.uint8)
    counts = np.array([(expmap == gi).sum() for gi in range(ng)])
    for gi in range(ng):
        if counts[gi] == 0:
            back[gi] = 0
    nmlp = int(sum(1 for gi, w in enumerate(wins) if w[1] == "MLP" and counts[gi] > 0))
    if use_mlp:
        assert nmlp > 0, "MLP variant assembled with zero MLP groups (wastes side bytes)"
    blob = bytearray()
    kidx = WIDE_IDX[plan["K"]] if fam == 0 else 15
    blob.append((fam & 1) | ((plan["grid"] & 7) << 1) | ((kidx & 15) << 4))
    blob.append(ng & 0xFF)
    blob.append(0x01 if use_mlp else 0x00)  # MLP header byte (NEW vs CROWN4)
    if use_mlp:
        blob += mlp_side
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
        mask = np.zeros(729, dtype=np.uint8)
        mask[uk] = 1
        blob += np.packbits(mask).tobytes()
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
            tbl = bytearray()
            tbl += len(codes).to_bytes(2, "little")
            for sym in sorted(codes):
                tbl += ((sym - 1024) & 0xFFFF).to_bytes(2, "little")
                tbl.append(codes[sym][1])
            blob += bytes(tbl)
    if hsyms.size:
        cap = (hsyms.size * 32) // 8 + 1024
        buf = (c_u8 * cap)()
        n = HAPRE.pack_syms(hsyms.ctypes.data_as(ctypes.POINTER(c_i16)),
                            hkeys.ctypes.data_as(ctypes.POINTER(c_u8)), hsyms.size,
                            cd.ctypes.data_as(ctypes.POINTER(c_u32)),
                            ln.ctypes.data_as(ctypes.POINTER(c_u8)), buf, cap)
        assert n > 0
        blob += int(n).to_bytes(4, "little") + bytes(buf[:n])
    else:
        blob += (0).to_bytes(4, "little")
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
        cap = int(gvals.size * 300 + 64)
        buf = (c_u8 * cap)()
        n = C6.golomb_pack(gvals.ctypes.data_as(ctypes.POINTER(c_i16)),
                           gkeys.ctypes.data_as(ctypes.POINTER(c_u8)),
                           kbyexp.ctypes.data_as(ctypes.POINTER(c_u8)),
                           gvals.size, buf, cap)
        assert n > 0
        blob += int(n).to_bytes(4, "little") + bytes(buf[:n])
    else:
        blob += (0).to_bytes(4, "little")
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
    return bytes(blob), nmlp


def encode_image_rct(rgb, rname, perm, t, fn, train=True):
    H, W, _ = rgb.shape
    yc = B17.rct_fwd(rgb, perm, t)
    assert np.array_equal(B17.rct_inv(yc, perm, t), rgb), f"RCT {rname} round-trip FAIL"
    planes = [yc[:, :, c] for c in range(3)]
    wchs, bm, nbh, nbw, ng32, wside = fit_weighted(planes, H, W)
    lms0 = [lms_pred_plane(pl, 0) for pl in planes]
    lms3 = [lms_pred_plane(pl, 3) for pl in planes]
    # ---- per-channel MLP nets (train or load .npz) ----
    nis = []
    t_train = 0.0
    for ci, ch in enumerate(planes):
        ni = None if train else load_npz(fn, rname, ci, ch)
        if ni is None:
            t0 = time.perf_counter()
            ni = train_channel_mlps(ch, f"{fn}|{rname}|ch{ci}")
            t_train += time.perf_counter() - t0
            save_npz(fn, rname, ci, ni)
        nis.append(ni)
    out = bytearray()
    out += b"C6" + H.to_bytes(2, "little") + W.to_bytes(2, "little") + bytes([1, RCTID[rname]])
    infos = []
    for ci, ch in enumerate(planes):
        ni = nis[ci]
        has_mlp = bool(ni["has_nets"])
        mlp_res = (ch - ni["pred"]).astype(np.int32) if has_mlp else None
        if has_mlp:
            check_range(mlp_res, f"MLP plain {rname} ch{ci}")
        mlp_side = mlp_side_bytes(ni) if has_mlp else None
        D = prepX(ch)
        RF = {n: (D["s"] * (ch - D["P"][n]).astype(np.int32)).astype(np.int32) for n in E16}
        RF["WAVG"] = (D["s"] * wchs[ci]["res"]).astype(np.int32)
        RF["LMS5_T0"] = (D["s"] * (ch - lms0[ci]).astype(np.int32)).astype(np.int32)
        RF["LMS5_T3"] = (D["s"] * (ch - lms3[ci]).astype(np.int32)).astype(np.int32)
        if has_mlp:
            RF["MLP"] = (D["s"] * mlp_res).astype(np.int32)
            check_range(RF["MLP"], f"MLP flip {rname} ch{ci}")
        for n in E16:
            check_range(RF[n], f"flip {rname} ch{ci} {n}")
        RU = {n: (ch - D["P"][n]).astype(np.int32) for n in E16}
        RU["WAVG"] = wchs[ci]["res"]
        if has_mlp:
            RU["MLP"] = mlp_res
        for n in E17:
            check_range(RU[n], f"plain {rname} ch{ci} {n}")
        wch_q = dict(wchs[ci])
        wch_q["lms0"] = (ch - lms0[ci]).astype(np.int32)
        wch_q["lms3"] = (ch - lms3[ci]).astype(np.int32)
        check_range(wch_q["lms0"], f"lms0 {rname} ch{ci}")
        check_range(wch_q["lms3"], f"lms3 {rname} ch{ci}")
        pq = encode_channel_q(ch, D, RF, has_mlp)
        pg = encode_channel_grid(ch, D, RU, has_mlp)
        # dual-track exact assembly: MLP-variant vs no-MLP-variant, min bytes wins
        cands = []
        for plan, fam in ((pq, "Q"), (pg, "G")):
            variants = []
            bq1, n1 = assemble_channel(plan, D, ch, wch_q if fam == "Q" else wchs[ci],
                                      mlp_res, mlp_side, False, H, W, nbw, ng32)
            variants.append((len(bq1), bq1, False, 0))
            if has_mlp and any(w[1] == "MLP" for w in plan["wins_all"]):
                bq2, n2 = assemble_channel(plan, D, ch, wch_q if fam == "Q" else wchs[ci],
                                          mlp_res, mlp_side, True, H, W, nbw, ng32)
                variants.append((len(bq2), bq2, True, n2))
            variants.sort(key=lambda t_: t_[0])
            if FORCE_MLP:
                for v in variants:
                    if v[2]:
                        variants = [v]
                        break
            cands.append((variants[0][0], variants[0][1], fam, plan, variants[0][2], variants[0][3]))
        cands.sort(key=lambda t_: t_[0])
        _, bb, fam, plan, used_mlp, nmlp = cands[0]
        out += bb
        picks = {}
        for w in (plan["wins_all"] if used_mlp else plan["wins_no"]):
            picks[w[1]] = picks.get(w[1], 0) + 1
        infos.append({"fam": fam, "plan": plan, "nbytes": len(bb), "mlp": used_mlp,
                      "picks": picks, "nwin": nis[ci]["nwin"]})
    return bytes(out), {"rct": rname, "chs": infos, "ng32": ng32, "nbw": nbw,
                        "wuse": [int(w["use"].sum()) for w in wchs],
                        "t_train": t_train}


def encode_image(rgb, fn, train=True):
    cands = []
    for (rname, perm, t) in RCTS:
        t0 = time.perf_counter()
        blob, info = encode_image_rct(rgb, rname, perm, t, fn, train)
        info["ms"] = (time.perf_counter() - t0) * 1000
        cands.append((len(blob), blob, info))
    cands.sort(key=lambda t_: t_[0])
    return cands[0][1], {"winner": cands[0][2],
                         "losers": [(c[2]["rct"], len(c[1]), c[2]["ms"]) for c in cands[1:]]}


def decode_image(blob):
    H = int.from_bytes(blob[2:4], "little")
    W = int.from_bytes(blob[4:6], "little")
    assert blob[:2] == b"C6", blob[:2]
    assert blob[6] == 1, blob[6]
    rct = blob[7] & 3
    assert rct <= 2, rct
    N = H * W
    p = 8
    planes = []
    for _ in range(3):
        b0, ng = blob[p], blob[p + 1]
        mlp_flag = blob[p + 2]
        p += 3
        assert mlp_flag in (0, 1), mlp_flag
        fam = b0 & 1
        grid = (b0 >> 1) & 7
        cmap = np.zeros(729, dtype=np.uint8)
        grid_thr = np.zeros(5, dtype=np.int32)
        occ = 0xFF
        mlp_thr = np.zeros(8, dtype=np.float64)
        mlp_win = np.zeros(9, dtype=np.uint8)
        mlp_w = np.zeros(9 * 113, dtype=np.float64)
        if mlp_flag:
            mlp_thr = np.frombuffer(bytearray(blob[p:p + 64]), dtype=np.float64).copy()
            p += 64
            scid = list(blob[p:p + 9])
            p += 9
            for k in range(9):
                if scid[k] != 255:
                    assert scid[k] < len(MLP_SCALES), scid[k]
                    scale = MLP_SCALES[scid[k]]
                    qw = np.frombuffer(bytearray(blob[p:p + 226]), dtype=np.int16).copy()
                    p += 226
                    mlp_win[k] = 1
                    mlp_w[k * 113:(k + 1) * 113] = qw.astype(np.float64) / float(scale)
            assert int(mlp_win.sum()) > 0, "mlp_present with zero nets (framing)"
        nbw = math.ceil(W / GS)
        nbh = math.ceil(H / GS)
        ng32 = nbh * nbw
        nbytes = (ng32 + 7) // 8
        raw = blob[p:p + nbytes]
        p += nbytes
        wuse = np.zeros(ng32, dtype=np.uint8)
        acc, nb, qp = 0, 0, 0
        for g in range(ng32):
            while nb < 1:
                acc = (acc << 8) | raw[qp]
                qp += 1
                nb += 8
            nb -= 1
            wuse[g] = (acc >> nb) & 1
            acc &= ((1 << nb) - 1) if nb else 0
        nused = int(wuse.sum())
        nbytes = (nused * 20 + 7) // 8
        raw = blob[p:p + nbytes] if nbytes else b""
        p += nbytes
        ww = np.zeros((ng32, 4), dtype=np.int8)
        acc, nb, qp = 0, 0, 0
        for g in range(ng32):
            if wuse[g]:
                for t in range(4):
                    while nb < 5:
                        acc = (acc << 8) | raw[qp]
                        qp += 1
                        nb += 8
                    nb -= 5
                    v = (acc >> nb) & 31
                    acc &= ((1 << nb) - 1) if nb else 0
                    ww[g, t] = v - 32 if v >= 16 else v
        if fam == 0:
            mask = np.unpackbits(np.frombuffer(blob[p:p + 92], dtype=np.uint8))[:729]
            p += 92
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
        nbytes = (ng * 7 + 7) // 8
        raw = blob[p:p + nbytes]
        p += nbytes
        predid = np.zeros(128, dtype=np.uint8)
        backend = np.zeros(128, dtype=np.uint8)
        acc, nb, qp = 0, 0, 0
        for gi in range(ng):
            while nb < 7:
                acc = (acc << 8) | raw[qp]
                qp += 1
                nb += 8
            nb -= 7
            v = (acc >> nb) & 0x7F
            acc &= ((1 << nb) - 1) if nb else 0
            predid[gi] = (v >> 2) & 31
            backend[gi] = v & 3
            assert predid[gi] <= 19 and backend[gi] <= 2, (gi, predid[gi], backend[gi])
            if predid[gi] == MLP_ID:
                assert mlp_flag == 1, f"MLP predid without mlp_present (ch group {gi})"
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
        mlp_thr = np.ascontiguousarray(mlp_thr, dtype=np.float64)
        mlp_win = np.ascontiguousarray(mlp_win, dtype=np.uint8)
        mlp_w = np.ascontiguousarray(mlp_w, dtype=np.float64)
        rc = C6.crown6_decode_ch(
            hb, len(hpay), gb, len(gpay), rs, goff.ctypes.data_as(ctypes.POINTER(c_i32)), ng,
            cd.ctypes.data_as(ctypes.POINTER(c_u32)), ln.ctypes.data_as(ctypes.POINTER(c_u8)),
            cmap.ctypes.data_as(ctypes.POINTER(c_u8)), predid.ctypes.data_as(ctypes.POINTER(c_u8)),
            backend.ctypes.data_as(ctypes.POINTER(c_u8)), kvals.ctypes.data_as(ctypes.POINTER(c_u8)),
            dbias.ctypes.data_as(ctypes.POINTER(c_i8)), hasn.ctypes.data_as(ctypes.POINTER(c_u8)),
            wuse.ctypes.data_as(ctypes.POINTER(c_u8)), ww.ctypes.data_as(ctypes.POINTER(c_i8)), nbw,
            fam, grid_thr.ctypes.data_as(ctypes.POINTER(c_i32)),
            plane.ctypes.data_as(ctypes.POINTER(c_i16)), H, W,
            int(mlp_flag), mlp_thr.ctypes.data_as(ctypes.POINTER(c_f64)),
            mlp_win.ctypes.data_as(ctypes.POINTER(c_u8)),
            mlp_w.ctypes.data_as(ctypes.POINTER(c_f64)))
        if rc != 0:
            _debug_fail.append({"ch": len(planes), "rc": rc, "plane": plane.copy(),
                                "fam": fam, "mlp": int(mlp_flag), "H": H, "W": W,
                                "predid": predid[:ng].copy(),
                                "backend": backend[:ng].copy()})
        assert rc == 0, (rc, len(planes))
        planes.append(plane)
    p0 = np.ascontiguousarray(planes[0].reshape(H, W))
    p1 = np.ascontiguousarray(planes[1].reshape(H, W))
    p2 = np.ascontiguousarray(planes[2].reshape(H, W))
    out = bytearray(N * 3)
    rc = C6.crown6_rct_inv(p0.ctypes.data_as(ctypes.POINTER(c_i16)),
                           p1.ctypes.data_as(ctypes.POINTER(c_i16)),
                           p2.ctypes.data_as(ctypes.POINTER(c_i16)), H, W, rct,
                           (ctypes.c_char * len(out)).from_buffer(out))
    assert rc == 0, f"RCT inv range FAIL rc={rc}"
    assert p == len(blob), (p, len(blob))
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


def xcheck_torch_vs_wire(images, train=True):
    """MANDATORY cross-check 1: torch-float32 vs wire-float64 int-rounded preds.

    For every winning (image, RCT, channel, ctx) net: compare B21.predict_plane
    (torch, .round half-even) vs mlp_forward_wire on the net's subset.
    Returns (total_pixels, mismatches). Mismatches do NOT affect correctness
    (encoder uses wire path); they are documented torch-float divergence.
    """
    tot, mis = 0, 0
    maxabs = 0.0
    for path in images:
        fn = path.split("/")[-1]
        rgb = np.array(Image.open(path).convert("RGB"))
        for (rname, perm, t) in RCTS:
            yc = B17.rct_fwd(rgb, perm, t)
            planes = [yc[:, :, c] for c in range(3)]
            for ci, P in enumerate(planes):
                ni = None if train else load_npz(fn, rname, ci, P)
                if ni is None:
                    ni = train_channel_mlps(P, f"{fn}|{rname}|ch{ci}")
                    save_npz(fn, rname, ci, ni)
                if not ni["has_nets"]:
                    continue
                F, _, _ = c6_features_targets(P, MLP_FEAT)
                Ff = F.reshape(-1, MLP_FEAT)
                ctx, _, _ = c6_ctx_of_plane(P)
                cf = ctx.reshape(-1)
                for k in range(MLP_NCTX):
                    if not ni["win"][k]:
                        continue
                    scale = MLP_SCALES[int(ni["scid"][k])]
                    m = (cf == k)
                    qw = ni["qw"][k]
                    q = dequant_net(qw, scale)
                    pw = mlp_forward_wire(Ff[m], q)
                    # rebuild torch net from stored int16 (== adaptive_quantize output)
                    import copy as _copy
                    net = B21.TinyMLP(MLP_FEAT, MLP_H)
                    with torch.no_grad():
                        w = qw.astype(np.float32) / float(scale)
                        net.fc1.weight.copy_(torch.from_numpy(w[0:96].reshape(8, 12)))
                        net.fc1.bias.copy_(torch.from_numpy(w[96:104]))
                        net.fc2.weight.copy_(torch.from_numpy(w[104:112].reshape(1, 8)))
                        net.fc2.bias.copy_(torch.from_numpy(w[112:113]))
                    pt = B21.predict_plane(net, Ff[m])
                    d = np.abs(pw.astype(np.int32) - pt.astype(np.int32))
                    tot += int(m.sum())
                    mis += int((d > 0).sum())
                    if d.max() > 0:
                        maxabs = max(maxabs, float(d.max()))
    print(f"[xcheck torch-vs-wire] pixels={tot} mismatches={mis} maxabs={maxabs}", flush=True)
    return tot, mis


def xcheck_c_vs_wire(images):
    """MANDATORY cross-check 2: libcrown6 batch forward vs numpy wire forward.

    Bit-identity required (0 mismatches) on every wire-evaluated pixel
    (all winning subsets, all images/RCTs/channels). Any mismatch BLOCKS
    integration (decode would diverge).
    """
    tot, mis = 0, 0
    for path in images:
        fn = path.split("/")[-1]
        rgb = np.array(Image.open(path).convert("RGB"))
        for (rname, perm, t) in RCTS:
            yc = B17.rct_fwd(rgb, perm, t)
            planes = [yc[:, :, c] for c in range(3)]
            for ci, P in enumerate(planes):
                ni = load_npz(fn, rname, ci, P)
                assert ni is not None, f"missing .npz for {fn} {rname} ch{ci} (train first)"
                if not ni["has_nets"]:
                    continue
                F, _, _ = c6_features_targets(P, MLP_FEAT)
                Ff = F.reshape(-1, MLP_FEAT)
                ctx, _, _ = c6_ctx_of_plane(P)
                cf = ctx.reshape(-1)
                for k in range(MLP_NCTX):
                    if not ni["win"][k]:
                        continue
                    scale = MLP_SCALES[int(ni["scid"][k])]
                    m = (cf == k)
                    q = dequant_net(ni["qw"][k], scale)
                    pw = mlp_forward_wire(Ff[m], q)
                    pc = c_mlp_predict_batch(Ff[m], q)
                    d = np.abs(pw - pc)
                    tot += int(m.sum())
                    mis += int((d > 0).sum())
                    assert int((d > 0).sum()) == 0, \
                        f"C-vs-wire DIVERGENCE {fn} {rname} ch{ci} ctx{k}: {(d>0).sum()} px"
    print(f"[xcheck C-vs-wire] pixels={tot} mismatches={mis} -> {'PASS' if mis==0 else 'FAIL'}",
          flush=True)
    return tot, mis


if __name__ == "__main__":
    import argparse as _ap
    import json as _json
    _p = _ap.ArgumentParser()
    _p.add_argument("--images", nargs="*", default=None)
    _p.add_argument("--xcheck", action="store_true",
                    help="run mandatory cross-checks (torch-vs-wire, C-vs-wire), no encode")
    _p.add_argument("--train-only", action="store_true")
    _p.add_argument("--no-train", action="store_true", help="use .npz cache only, never train")
    _a = _p.parse_args()
    _paths = [_s for _s in IMAGES if (_a.images is None or _s.split('/')[-1] in _a.images)]
    if _a.xcheck:
        _t, _m = xcheck_torch_vs_wire(_paths, train=not _a.no_train)
        _t2, _m2 = xcheck_c_vs_wire(_paths)
        print(_json.dumps({"torch_vs_wire": {"px": _t, "mis": _m},
                           "c_vs_wire": {"px": _t2, "mis": _m2}}))
        sys.exit(0 if _m2 == 0 else 1)
    _t_all = time.perf_counter()
    _res = {}
    _bpps = []
    for _path in _paths:
        _fn = _path.split("/")[-1]
        _px = np.array(Image.open(_path).convert("RGB"))
        _H, _W, _ = _px.shape
        _npx = _H * _W * 3
        if _a.train_only:
            for (rname, perm, t) in RCTS:
                _yc = B17.rct_fwd(_px, perm, t)
                for _ci in range(3):
                    _ni = load_npz(_fn, rname, _ci, _yc[:, :, _ci])
                    if _ni is None:
                        _t0 = time.perf_counter()
                        _ni = train_channel_mlps(_yc[:, :, _ci], f"{_fn}|{rname}|ch{_ci}")
                        print(f"{_fn} {rname} ch{_ci}: trained "
                              f"nwins={_ni['nwin']} fallback={_ni['fallback']} "
                              f"{time.perf_counter()-_t0:.0f}s", flush=True)
                        save_npz(_fn, rname, _ci, _ni)
            continue
        _t0 = time.perf_counter()
        _blob, _info = encode_image(_px, _fn, train=not _a.no_train)
        _enc_s = time.perf_counter() - _t0
        _t1 = time.perf_counter()
        _out, _di = decode_image(_blob)
        _dec_ms = (time.perf_counter() - _t1) * 1000
        assert np.array_equal(np.frombuffer(_out, dtype=np.uint8).reshape(_H, _W, 3), _px), \
            f"ROUND-TRIP FAIL {_fn}"
        _bpp = len(_blob) * 8 / _npx
        _ttr = float(_info["winner"].get("t_train", 0.0))
        _res[_fn] = {"bpp": round(float(_bpp), 4), "bytes": len(_blob),
                     "rct": _info["winner"]["rct"],
                     "fams": [c["fam"] for c in _info["winner"]["chs"]],
                     "mlp": [c["mlp"] for c in _info["winner"]["chs"]],
                     "picks": [c.get("picks", {}) for c in _info["winner"]["chs"]],
                     "train_s": round(_ttr), "enc_s": round(_enc_s),
                     "dec_ms": round(_dec_ms, 1)}
        _bpps.append(_bpp)
        _d4 = (_bpp - CROWN4_BAR) / CROWN4_BAR * 100
        _dJ = (_bpp - JXL_E3[_fn]) / JXL_E3[_fn] * 100
        print(f"{_fn}: C6={_bpp:.4f} vs CR4ex-avg {CROWN4_BAR} ({_d4:+.2f}% vs avg) "
              f"vs JXL {_dJ:+.2f}% "
              f"RCT={_res[_fn]['rct']} fams={_res[_fn]['fams']} mlp={_res[_fn]['mlp']} "
              f"picks={_res[_fn]['picks']} RT-PASS "
              f"[train {_ttr:.0f}s enc-assy {_enc_s-_ttr:.0f}s dec {_dec_ms:.0f}ms]", flush=True)
    if not _a.train_only:
        _avg = float(np.mean(_bpps))
        print(f"AVG={_avg:.4f} vs CROWN4-exact {CROWN4_BAR} "
              f"({(_avg-CROWN4_BAR)/CROWN4_BAR*100:+.3f}%)")
        _diffs = [_res[f]["bpp"] - JXL_E3[f] for f in
                  ["kodim01.png", "kodim02.png", "kodim05.png", "kodim07.png",
                   "kodim13.png", "kodim19.png", "kodim23.png"] if f in _res]
        if len(_diffs) == 7:
            _p, _w = wilcoxon_exact(_diffs)
            _wins = sum(1 for d in _diffs if d < 0)
            print(f"vs JXL-e3: {_wins}W-{7-_wins}L p={_p:.3f} "
                  f"{'BOSS-5 KO (p<.05, wins>=6)' if (_p < 0.05 and _wins >= 6) else 'Boss-5 STANDS'}")
        print(f"TOTAL {(time.perf_counter()-_t_all)/60:.1f} min")
        print(_json.dumps(_res, indent=1))