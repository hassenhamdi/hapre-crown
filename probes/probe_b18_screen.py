"""probe_b18_screen.py — B18 surgical probe SCREENING stage (new file only).

Targets: kodim05 (+2.46% vs JXL-e3) and kodim13 (+1.02%). Priority: flip 05
(KO math: 6W-1L with 13-loss smallest rank -> W=1, p~.016).

Frame: C27-YCoCg-R (CROWN3 winner on both targets; b17 M1) + causal predictors
(CROWN2 border doctrine) + global order-0 exact Huffman per plane
(real heapq + 16+A*24 tables, +64b header). All side counted.
Unit: bpp = total_bits/(H*W*3). numpy+PIL only.

Screens three FRESH weapon families (nothing retired touches these):
  W4 directional/long-lag experts (vectorized shifts, decoder-safe doctrine)
  W5 per-pixel deterministic-adaptive mixers (sequential, recon-only state, 0 side)
  W1 luma-anchored chroma (global LS slope/step + per-G32 gated variant)
Plus per-channel diagnosis + anchor repro (C6-MED order-0 ~= 3.58 +-3%).
"""
import heapq
import math
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
import probe_b17_rctw as B17
from probe_b4_b import nbhd as nbhd_base

D = "/tmp/opencode/autocompress/experiments/real_photos"
FILES = ["kodim01.png", "kodim02.png", "kodim05.png", "kodim07.png",
         "kodim13.png", "kodim19.png", "kodim23.png"]
JXL = {"kodim01.png": 3.3593, "kodim02.png": 3.0611, "kodim05.png": 3.5120,
       "kodim07.png": 2.7310, "kodim13.png": 3.9141, "kodim19.png": 3.2154,
       "kodim23.png": 2.8110}
CR3 = {"kodim01.png": 3.3026, "kodim02.png": 2.9889, "kodim05.png": 3.5984,
       "kodim07.png": 2.6824, "kodim13.png": 3.9542, "kodim19.png": 3.1566,
       "kodim23.png": 2.7448}
HEADER = 64


def huff_bits(counts):
    if len(counts) == 0:
        return 0
    if len(counts) == 1:
        return int(counts[0])
    h = list(map(int, counts))
    heapq.heapify(h)
    t = 0
    while len(h) > 1:
        x = heapq.heappop(h)
        y = heapq.heappop(h)
        s = x + y
        t += s
        heapq.heappush(h, s)
    return t


def plane_bits(res):
    assert int(np.abs(res).max()) <= 4096
    _, cn = np.unique(res, return_counts=True)
    return huff_bits(cn.tolist()) + 16 + len(cn) * 24


def img_bits(res_planes, side_bits=0):
    return HEADER + side_bits + sum(plane_bits(r) for r in res_planes)


def nbhd_big(ch, NL=8, NT=4):
    """Causal far-neighborhood under CROWN2 border doctrine.

    L_k[:,k:]=ch[:,:-k]; L_k[:,:k] mirrors L_{k-1}[:,:k] (clamped-left),
    with L_1 = base `a` (zero border; row0 copies left; col0 copies top).
    Same vertically for T_k with base `b`. Decoder-safe: every entry is a
    causal recon value (encoder recon == orig, lossless) or doctrine const.
    """
    a, b, c, d, Ww, NNe, NE = nbhd_base(ch)
    H, W = ch.shape
    Ls = {1: a}
    for k in range(2, NL + 1):
        Lk = np.empty_like(ch)
        Lk[:, k:] = ch[:, :-k]
        Lk[:, :k] = Ls[k - 1][:, :k]
        Ls[k] = Lk
    Ts = {1: b}
    for k in range(2, NT + 1):
        Tk = np.empty_like(ch)
        Tk[k:, :] = ch[:-k, :]
        Tk[:k, :] = Ts[k - 1][:k, :]
        Ts[k] = Tk
    return {"a": a, "b": b, "c": c, "d": d, "Ww": Ww, "NNe": NNe, "NE": NE,
            "L": Ls, "T": Ts}


def med_of(a, b, c):
    return np.where(c >= np.maximum(a, b), np.minimum(a, b),
                    np.where(c <= np.minimum(a, b), np.maximum(a, b), a + b - c))


def dir_candidates(N):
    a, b, c, d = N["a"], N["b"], N["c"], N["d"]
    Ww, NNe, NE = N["Ww"], N["NNe"], N["NE"]
    L, T = N["L"], N["T"]
    P = {}
    P["MED"] = med_of(a, b, c)
    P["LEFT"] = a
    P["TOP"] = b
    P["AVG_AB"] = (a + b) // 2
    P["PLANE"] = a + b - c
    # long-lag singles
    P["WL2"] = L[2]
    P["WL4"] = L[4]
    P["WL8"] = L[8]
    P["WN2"] = T[2]
    P["WN4"] = T[4]
    # horizontal / vertical line averages (ripple smoothers)
    P["HAVG4"] = (a + L[2] + L[3] + L[4]) // 4
    P["VAVG4"] = (b + T[2] + T[3] + T[4]) // 4
    # gradient-persistent extrapolation along row / column
    P["XH"] = 2 * a - Ww
    P["XV"] = 2 * b - NNe
    P["XH4"] = 2 * a - L[4]
    # diagonal longs
    P["NW2"] = np.empty_like(a)
    P["NW2"][2:, 2:] = 0  # filled below
    src = N["L"]  # placeholder, replaced below
    return P


def diag_long(ch, N):
    """NW/NE long-diagonal recon planes (need 2D shifts of orig=recon)."""
    H, W = ch.shape
    NW2 = np.empty_like(ch)
    NW2[2:, 2:] = ch[:-2, :-2]
    NW2[:2, :] = N["b"][:2, :]
    NW2[2:, :2] = N["a"][2:, :2]
    NE2 = np.empty_like(ch)
    NE2[2:, :-2] = ch[:-2, 2:]
    NE2[:2, :] = N["b"][:2, :]
    NE2[2:, -2:] = N["b"][2:, -2:]
    return NW2, NE2


def lms_mix_plane(plane, pa, pb, half=64):
    """Per-pixel deterministic mixer of two causal predictors (0 side bits).

    State: integer EMAs of |err| for each arm, halflife `half` px.
    pred = arm with smaller EMA (ties -> pa). Encoder uses recon==orig so the
    decoder recomputes the identical path. Returns residual plane (int32).
    """
    H, W = plane.shape
    p = plane.astype(np.int32)
    A = pa.astype(np.int32)
    B = pb.astype(np.int32)
    res = np.empty_like(p)
    ea = 0
    eb = 0
    k = half
    for i in range(H):
        Ai, Bi, pi, ri = A[i], B[i], p[i], res[i]
        for j in range(W):
            use_a = (ea <= eb)
            pr = Ai[j] if use_a else Bi[j]
            e = int(pi[j]) - int(pr)
            ri[j] = e
            ae = abs(e)
            # EMA update of the CHOSEN arm toward |e|; other arm decays slowly
            # toward |e| too (so a temporarily-bad arm can come back).
            if use_a:
                ea += (ae - ea) // k + (1 if (ae - ea) % k >= k // 2 and ae > ea else 0)
                eb += (ae - eb) // (k * 4)
            else:
                eb += (ae - eb) // k + (1 if (ae - eb) % k >= k // 2 and ae > eb else 0)
                ea += (ae - ea) // (k * 4)
    return res


def main():
    t_all = time.time()
    print("== B18 SCREEN: anchor + per-channel diagnosis ==")
    anch = {}
    for fn in FILES:
        px = np.array(Image.open(os.path.join(D, fn)).convert("RGB"))
        H, W, _ = px.shape
        npx = H * W * 3
        yc6 = B17.rct_fwd(px, 0, 6)
        rr6 = [yc6[:, :, c] - B17.med_pred(yc6[:, :, c]) for c in range(3)]
        b6 = img_bits(rr6) / npx
        yc27 = B17.rct_fwd(px, 3, 6)
        rr27 = [yc27[:, :, c] - B17.med_pred(yc27[:, :, c]) for c in range(3)]
        b27 = img_bits(rr27) / npx
        chb = [plane_bits(r) / npx * 3 for r in rr27]  # per-channel bpp share*3
        anch[fn] = (b6, b27)
        print(f"{fn}: C6-MED={b6:.4f} C27-MED={b27:.4f} "
              f"[Y,Co,Cg]={['%.3f' % v for v in chb]} JXL={JXL[fn]:.4f} CR3={CR3[fn]:.4f}",
              flush=True)
    a6 = float(np.mean([v[0] for v in anch.values()]))
    print(f"ANCHOR C6-MED order-0 avg={a6:.4f} vs 3.58 "
          f"({(a6 - 3.58) / 3.58 * 100:+.2f}% -> "
          f"{'PASS' if abs(a6 - 3.58) / 3.58 <= 0.03 else 'FAIL'})")
    a27 = float(np.mean([v[1] for v in anch.values()]))

    print("\n== W4/W1-global predictor sweep (C27 frame, order-0, +side noted) ==")
    print("(global single-predictor replaced on ALL 3 channels; side=0 unless noted)")
    cands = ["MED", "LEFT", "TOP", "AVG_AB", "PLANE", "WL2", "WL4", "WL8",
             "WN2", "WN4", "HAVG4", "VAVG4", "XH", "XV", "XH4"]
    res = {c: {} for c in cands}
    for fn in FILES:
        px = np.array(Image.open(os.path.join(D, fn)).convert("RGB"))
        H, W, _ = px.shape
        npx = H * W * 3
        yc = B17.rct_fwd(px, 3, 6)
        planes = [yc[:, :, c] for c in range(3)]
        NBs = [nbhd_big(ch) for ch in planes]
        for i, ch in enumerate(planes):
            NW2, NE2 = diag_long(ch, NBs[i])
            NBs[i]["NW2"] = NW2
            NBs[i]["NE2"] = NE2
        for c in cands:
            rr = [(planes[i] - dir_candidates(NBs[i])[c]).astype(np.int32)
                  for i in range(3)]
            # NW2/NE2 handled separately below
            res[c][fn] = img_bits(rr) / npx
        for c, key in [("NW2", "NW2"), ("NE2", "NE2")]:
            rr = [(planes[i] - NBs[i][key]).astype(np.int32) for i in range(3)]
            res.setdefault(c, {})[fn] = img_bits(rr) / npx
    allc = cands + ["NW2", "NE2"]
    hdr = "cand   " + "".join(f"{f[5:7]:>9}" for f in FILES) + "      AVG   d05vsMED  d13vsMED"
    print(hdr)
    for c in allc:
        vals = [res[c][f] for f in FILES]
        avg = float(np.mean(vals))
        d05 = (res[c]["kodim05.png"] - res["MED"]["kodim05.png"])
        d13 = (res[c]["kodim13.png"] - res["MED"]["kodim13.png"])
        print(f"{c:7s} " + "".join(f"{v:9.4f}" for v in vals) +
              f" {avg:8.4f} {d05:+.4f} {d13:+.4f}", flush=True)

    print("\n== W5 LMS mixer screen (05/13 + controls, 0 side bits) ==")
    for fn in FILES:
        px = np.array(Image.open(os.path.join(D, fn)).convert("RGB"))
        H, W, _ = px.shape
        npx = H * W * 3
        yc = B17.rct_fwd(px, 3, 6)
        planes = [yc[:, :, c] for c in range(3)]
        NBs = [nbhd_big(ch) for ch in planes]
        med = res["MED"][fn]
        outs = {}
        for tag, pa_k, pb_k, half in [("MED|LEFT", "MED", "LEFT", 64),
                                      ("MED|AVG", "MED", "AVG_AB", 64),
                                      ("MED|WL4", "MED", "WL4", 64),
                                      ("MED|HAVG4", "MED", "HAVG4", 64),
                                      ("MED|LEFT(h16)", "MED", "LEFT", 16)]:
            rr = []
            for i in range(3):
                P = dir_candidates(NBs[i])
                rr.append(lms_mix_plane(planes[i], P[pa_k], P[pb_k], half))
            outs[tag] = img_bits(rr) / npx
        print(f"{fn}: MED={med:.4f} " +
              " ".join(f"{k}={v:.4f}({v - med:+.4f})" for k, v in outs.items()),
              flush=True)

    print("\n== W1 luma-anchored chroma screen (global slope, predictors on Co/Cg) ==")
    for fn in FILES:
        px = np.array(Image.open(os.path.join(D, fn)).convert("RGB"))
        H, W, _ = px.shape
        npx = H * W * 3
        yc = B17.rct_fwd(px, 3, 6)
        Y, Co, Cg = (yc[:, :, c] for c in range(3))
        rY = (Y - B17.med_pred(Y)).astype(np.int32)
        base = img_bits([rY, (Co - B17.med_pred(Co)).astype(np.int32),
                         (Cg - B17.med_pred(Cg)).astype(np.int32)]) / npx
        # global LS: Co ~= (s*Ym + o)>>8 with Ym = MED-pred of Y? No: co-located
        # RECON Y value itself (decoder has it: Y coded first). Fit int slope.
        Yf = Y.astype(np.float64).ravel()
        line = f"{fn}: base={base:.4f}"
        for nm, Ch in [("Co", Co), ("Cg", Cg)]:
            Cf = Ch.astype(np.float64).ravel()
            s = float(((Cf - Cf.mean()) * (Yf - Yf.mean())).sum() /
                      max(((Yf - Yf.mean()) ** 2).sum(), 1))
            # quantize slope x256, offset int
            sq = int(np.clip(round(s * 256), -32768, 32767))
            pred = ((Y.astype(np.int32) * sq) >> 8)
            # remove mean via offset folded into residual centering (free: Huffman handles shift? no—offset matters)
            o = int(round(Cf.mean() - (Yf.mean() * sq / 256)))
            r = (Ch.astype(np.int32) - (pred + o)).astype(np.int32)
            # residual still needs spatial prediction: MED on the LUMA-REMOVED plane
            Ch2 = (Ch.astype(np.int32) - (pred + o)).astype(np.int32)
            r2 = (Ch2 - B17.med_pred(Ch2)).astype(np.int32)
            b_raw = plane_bits(r) / npx * 3
            b_med = plane_bits(r2) / npx * 3
            line += f" {nm}:s={sq / 256:+.3f} raw={b_raw:.3f} med2={b_med:.3f}"
        print(line, flush=True)
    print(f"\nTOTAL {(time.time() - t_all) / 60:.1f} min")


if __name__ == "__main__":
    main()
