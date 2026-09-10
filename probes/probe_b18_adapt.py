"""probe_b18_adapt.py — adaptive-k Golomb (LOCO-I N/A rule) standalone screen.

Per-group (LOCO-quantile K=64, MED sign-flipped, raster order) exact-bit compare:
  static-G (best k + best d, +4+3 side) vs adaptive-k (N/A rule, RESET N0,
  static d reused, +0+3 side).
Tunes N0/init on 05/13 + controls. numpy only (loops), PIL images.
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
from probe_b5_d import golomb_best
from probe_b17_rctw import rct_fwd, med_pred

D = "/tmp/opencode/autocompress/experiments/real_photos"
FILES = ["kodim01.png", "kodim02.png", "kodim05.png", "kodim07.png",
         "kodim13.png", "kodim19.png", "kodim23.png"]
JXL = {"kodim01.png": 3.3593, "kodim02.png": 3.0611, "kodim05.png": 3.5120,
       "kodim07.png": 2.7310, "kodim13.png": 3.9141, "kodim19.png": 3.2154,
       "kodim23.png": 2.8110}
K = 64


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


def static_g_bits(g):
    g = np.asarray(g, dtype=np.int64).reshape(-1)
    if g.size == 0:
        return 0, 0, 0
    gd0, k0 = golomb_best(g)
    best = (gd0 + 4, k0, 0)
    for d in range(-4, 4):
        gd, k = golomb_best(g - d)
        if gd + 4 + 3 < best[0]:
            best = (gd + 4 + 3, k, d)
    return best


def adapt_bits(order_vals, d, n0, n_init=2, a_init=2):
    N, Av = n_init, a_init
    tot = 0
    for v in order_vals:
        k = 0
        while k < 12 and (N << k) < Av:
            k += 1
        M = v - d
        M = M * 2 if M >= 0 else -M * 2 - 1
        tot += (M >> k) + 1 + k
        N += 1
        Av += M if M >= 0 else -M
        if N >= n0:
            N >>= 1
            Av >>= 1
    return tot


def main():
    t0 = time.time()
    for n0 in [32, 64, 128]:
        print(f"== N0={n0} ==")
        for fn in FILES:
            px = np.array(Image.open(os.path.join(D, fn)).convert("RGB"))
            H, W, _ = px.shape
            npx = H * W * 3
            yc = rct_fwd(px, 3, 6)
            ts = ta = 0
            for ci in range(3):
                ch = yc[:, :, ci]
                a, b, c, d, Ww, NNe, NE = nbhd_base(ch)
                key, s = loco_ctx365(a, b, c, d)
                R = (s * (ch - med_pred(ch)).astype(np.int32)).astype(np.int32)
                for gkeys in qgroups(key, np.abs(R), K):
                    sel = np.isin(key, np.array(gkeys))
                    g = R[sel]
                    if g.size == 0:
                        continue
                    sb, _, dd = static_g_bits(g)
                    # raster order of group symbols:
                    idx = np.where(sel.reshape(-1))[0]
                    ab = adapt_bits(g.reshape(-1), dd, n0)
                    ts += sb
                    ta += ab + (3 if dd != 0 else 0)
            print(f"N0={n0} {fn}: staticG={ts / npx:.4f} adaptG={ta / npx:.4f} "
                  f"({(ta - ts) / npx:+.4f}) JXL={JXL[fn]:.4f}", flush=True)
    print(f"TOTAL {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
