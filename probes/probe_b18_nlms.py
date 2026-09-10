"""probe_b18_nlms.py — magnitude-scaled NLMS screen (new file).

Sign-sign LMS (S3) is crude (+-1 steps). True JXL-Weighted scales updates by
error/x magnitudes. NLMS: w += mu*err*xc/||xc||^2 with integer arithmetic,
plus leakage. Stencil 5-tap {L,T,TL,TR,MED}, mu in {1/8,1/32}, leakage on/off.
Frame: C27 + global order-0 exact Huffman + 64b. 05/13 + controls.
"""
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
from probe_b17_rctw import rct_fwd, med_pred
from probe_b4_b import nbhd as nbhd_base
from probe_b18_screen import img_bits

D = "/tmp/opencode/autocompress/experiments/real_photos"
FILES = ["kodim01.png", "kodim02.png", "kodim05.png", "kodim07.png",
         "kodim13.png", "kodim19.png", "kodim23.png"]
JXL = {"kodim01.png": 3.3593, "kodim02.png": 3.0611, "kodim05.png": 3.5120,
       "kodim07.png": 2.7310, "kodim13.png": 3.9141, "kodim19.png": 3.2154,
       "kodim23.png": 2.8110}


def nlms_plane(plane, mu_shift=3, leak=0):
    """mu = 1/2^mu_shift; leak: w[i] -= w[i]>>leak each step (0=off).
    Integer NLMS, sum NOT constrained (DC tracked by MED tap)."""
    p = plane.astype(np.int32)
    H, W = p.shape
    a, b, c, d, Ww, NNe, NE = nbhd_base(plane)
    M = med_pred(plane)
    L, T, TL, TR, MD = (v.ravel() for v in (a, b, c, d, M))
    y = p.ravel()
    w = [0, 0, 0, 0, 16]
    res = np.empty_like(y)
    for n in range(y.size):
        xs = (int(L[n]), int(T[n]), int(TL[n]), int(TR[n]), int(MD[n]))
        pr = (w[0] * xs[0] + w[1] * xs[1] + w[2] * xs[2] + w[3] * xs[3] + w[4] * xs[4]) >> 4
        e = int(y[n]) - pr
        res[n] = e
        if e != 0:
            norm = xs[0] * xs[0] + xs[1] * xs[1] + xs[2] * xs[2] + xs[3] * xs[3] + xs[4] * xs[4]
            norm = (norm >> 8) + 1
            step = (abs(e) << 4) // (norm << mu_shift) + 1
            se = 1 if e > 0 else -1
            m = (xs[0] + xs[1] + xs[2] + xs[3] + xs[4] + 2) // 5
            for i in range(5):
                di = xs[i] - m
                if di > 0:
                    w[i] += se * ((step * di) // 256 + 1)
                elif di < 0:
                    w[i] -= se * ((step * -di) // 256 + 1)
                if leak:
                    w[i] -= w[i] >> leak
                w[i] = min(64, max(-64, w[i]))
    return res.reshape(H, W)


def main():
    t0 = time.time()
    cfgs = [("nlms-s3", 3, 0), ("nlms-s5", 5, 0), ("nlms-s3L", 3, 8)]
    for fn in FILES:
        px = np.array(Image.open(os.path.join(D, fn)).convert("RGB"))
        H, W, _ = px.shape
        npx = H * W * 3
        yc = rct_fwd(px, 3, 6)
        planes = [yc[:, :, c] for c in range(3)]
        base = img_bits([(pl - med_pred(pl)).astype(np.int32) for pl in planes]) / npx
        line = f"{fn}: MED={base:.4f}"
        for tag, ms, lk in cfgs:
            rr = [nlms_plane(pl, ms, lk) for pl in planes]
            v = img_bits(rr) / npx
            line += f" {tag}={v:.4f}({v - base:+.4f})"
        print(line + f" JXL={JXL[fn]:.4f}", flush=True)
    print(f"TOTAL {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
