"""probe_b18_ctx.py — B18 S2 error-feedback / cross-channel CONTEXT screen.

Fixed expert (MED sign-flipped, isolates grouping effect), K=12 quantile
groups on key variants, per-group order-0 exact Huffman + honest map/id/table
ledger (same as probe_b18_group). Keys:
  LOCO   : loco_ctx365 (baseline)
  ERR4   : LOCO x prevres-bin(|MEDres at LEFT|, 4 bins)   [JXL WP-error analog]
  ERR4T  : LOCO x prevres-bin(|MEDres at TOP|, 4 bins)
  YMAG   : chroma only: LOCO x co-located-YMEDabs-bin (4 bins) [Attack-2 analog]
  ERR4Y  : Y-channel ERR4, chroma YMAG (combined)
numpy+PIL only. bpp=bits/(H*W*3).
"""
import math
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
import probe_b4_a as A
from probe_b4_b import nbhd as nbhd_base
from probe_b4_c import loco_ctx365
from probe_b17_rctw import rct_fwd, med_pred

D = "/tmp/opencode/autocompress/experiments/real_photos"
FILES = ["kodim01.png", "kodim02.png", "kodim05.png", "kodim07.png",
         "kodim13.png", "kodim19.png", "kodim23.png"]
JXL = {"kodim01.png": 3.3593, "kodim02.png": 3.0611, "kodim05.png": 3.5120,
       "kodim07.png": 2.7310, "kodim13.png": 3.9141, "kodim19.png": 3.2154,
       "kodim23.png": 2.8110}
K = 12


def qgroups(key, rfM, K):
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


def gbits_cost(g):
    if g.size == 0:
        return 0
    _, cn = np.unique(g, return_counts=True)
    return A.huff_bits(cn.tolist()) + 16 + len(cn) * 24


def eval_key(px, mode):
    yc = rct_fwd(px, 3, 6)
    H, W, _ = px.shape
    total = 64
    for ci in range(3):
        ch = yc[:, :, ci]
        a, b, c, d, Ww, NNe, NE = nbhd_base(ch)
        key0, s = loco_ctx365(a, b, c, d)
        M = med_pred(ch)
        R = (s * (ch - M).astype(np.int32)).astype(np.int32)
        Yp = yc[:, :, 0]
        aY, bY, cY, dY, *_ = nbhd_base(Yp)
        keyY, sY = loco_ctx365(aY, bY, cY, dY)
        MY = med_pred(Yp)
        RY = np.abs(sY * (Yp - MY).astype(np.int32)).astype(np.int32)
        RL = np.zeros_like(R)
        RL[:, 1:] = np.abs(R[:, :-1])
        RT = np.zeros_like(R)
        RT[1:, :] = np.abs(R[:-1, :])
        prevL = np.minimum(RL // 3, 3).astype(np.int64)
        prevT = np.minimum(RT // 3, 3).astype(np.int64)
        ybin = np.minimum(RY // 4, 3).astype(np.int64)
        if mode == "LOCO":
            key = key0.astype(np.int64)
        elif mode == "ERR4":
            key = key0.astype(np.int64) * 4 + prevL
        elif mode == "ERR4T":
            key = key0.astype(np.int64) * 4 + prevT
        elif mode == "YMAG":
            if ci == 0:
                key = key0.astype(np.int64)
            else:
                key = key0.astype(np.int64) * 4 + ybin
        elif mode == "ERR4Y":
            if ci == 0:
                key = key0.astype(np.int64) * 4 + prevL
            else:
                key = key0.astype(np.int64) * 4 + ybin
        else:
            raise ValueError(mode)
        rfM = np.abs(R)
        groups = qgroups(key, rfM, K)
        uk = np.unique(key)
        gbits = math.ceil(math.log2(K))
        total += 16 + 736 + len(uk) * gbits
        total += len(groups) * 2  # 4-way backend-ish? No: single backend -> 0 id bits; keep map only
        for gkeys in groups:
            sel = np.isin(key, np.array(gkeys))
            total += gbits_cost(R[sel])
    return total


def main():
    t0 = time.time()
    modes = ["LOCO", "ERR4", "ERR4T", "YMAG", "ERR4Y"]
    R = {m: {} for m in modes}
    for f in FILES:
        px = np.array(Image.open(os.path.join(D, f)).convert("RGB"))
        H, W, _ = px.shape
        npx = H * W * 3
        for m in modes:
            R[m][f] = eval_key(px, m) / npx
        print(f"{f}: " + " ".join(f"{m}={R[m][f]:.4f}" for m in modes) +
              f" JXL={JXL[f]:.4f}", flush=True)
    print("\ndelta vs LOCO:")
    for f in FILES:
        print(f"{f}: " + " ".join(f"{m}={R[m][f] - R['LOCO'][f]:+.4f}" for m in modes[1:]))
    av = {m: float(np.mean(list(R[m].values()))) for m in modes}
    print("AVG:", {m: round(v, 4) for m, v in av.items()})
    print(f"TOTAL {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
