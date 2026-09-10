"""probe_b18_lms.py — B18 S3 per-pixel adaptive-FIR screen (JXL-Weighted analog).

Our WAVG = block-fixed LS weights. JXL Weighted = per-pixel self-correcting.
This screens a TRUE integer sign-sign LMS FIR with sum-preserving renorm
(state = weights only, recon-derived -> decoder recomputes identically,
ZERO side bits). Stencils: 4-tap {L,T,TL,TR} and 5-tap {+MED}.
Update: w_i += sign(err)*sign(x_i - m) if |err|>THR; renorm sum to 16
(lowest-index tie-break); clamp [-8,20]. Causal recon only.
Frame: C27 + global order-0 exact Huffman per plane + 64b header.
numpy+PIL only.
"""
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
from probe_b17_rctw import rct_fwd, med_pred
from probe_b4_b import nbhd as nbhd_base
from probe_b18_screen import plane_bits, img_bits, HEADER

D = "/tmp/opencode/autocompress/experiments/real_photos"
FILES = ["kodim01.png", "kodim02.png", "kodim05.png", "kodim07.png",
         "kodim13.png", "kodim19.png", "kodim23.png"]
JXL = {"kodim01.png": 3.3593, "kodim02.png": 3.0611, "kodim05.png": 3.5120,
       "kodim07.png": 2.7310, "kodim13.png": 3.9141, "kodim19.png": 3.2154,
       "kodim23.png": 2.8110}


def lms_plane(plane, taps="4t", thr=0):
    p = plane.astype(np.int32)
    H, W = p.shape
    a, b, c, d, Ww, NNe, NE = nbhd_base(plane)
    M = med_pred(plane)
    L = a.ravel()
    T = b.ravel()
    TL = c.ravel()
    TR = d.ravel()
    MD = M.ravel()
    y = p.ravel()
    if taps == "4t":
        NX = 4
    else:
        NX = 5
    w = [4, 4, 4, 4, 0][:NX]
    if NX == 5:
        w = [3, 3, 3, 3, 4]
    res = np.empty_like(y)
    for n in range(y.size):
        l, t, tl, tr = int(L[n]), int(T[n]), int(TL[n]), int(TR[n])
        if NX == 4:
            pr = (w[0] * l + w[1] * t + w[2] * tl + w[3] * tr + 8) // 16
            m = (l + t + tl + tr + 2) // 4
            xs = (l, t, tl, tr)
        else:
            md = int(MD[n])
            pr = (w[0] * l + w[1] * t + w[2] * tl + w[3] * tr + w[4] * md + 8) // 16
            m = (l + t + tl + tr + md + 2) // 5
            xs = (l, t, tl, tr, md)
        e = int(y[n]) - pr
        res[n] = e
        ae = abs(e)
        if ae > thr:
            se = 1 if e > 0 else -1
            for i in range(NX):
                d_i = xs[i] - m
                if d_i > 0:
                    w[i] += se
                elif d_i < 0:
                    w[i] -= se
            # clamp + sum-preserving renorm to 16
            for i in range(NX):
                if w[i] < -8:
                    w[i] = -8
                elif w[i] > 20:
                    w[i] = 20
            s = sum(w)
            while s > 16:
                j = max(range(NX), key=lambda i: (w[i], -i))
                w[j] -= 1
                s -= 1
            while s < 16:
                j = min(range(NX), key=lambda i: (w[i], i))
                w[j] += 1
                s += 1
    return res.reshape(H, W)


def main():
    t0 = time.time()
    cfgs = [("lms4t-T0", "4t", 0), ("lms4t-T3", "4t", 3),
            ("lms5t-T0", "5t", 0), ("lms5t-T3", "5t", 3)]
    R = {}
    for f in FILES:
        px = np.array(Image.open(os.path.join(D, f)).convert("RGB"))
        H, W, _ = px.shape
        npx = H * W * 3
        yc = rct_fwd(px, 3, 6)
        planes = [yc[:, :, c] for c in range(3)]
        base = img_bits([(pl - med_pred(pl)).astype(np.int32) for pl in planes]) / npx
        line = f"{f}: MED={base:.4f}"
        for tag, taps, thr in cfgs:
            rr = [lms_plane(pl, taps, thr) for pl in planes]
            v = img_bits(rr) / npx
            R.setdefault(tag, {})[f] = v
            line += f" {tag}={v:.4f}({v - base:+.4f})"
        print(line + f" JXL={JXL[f]:.4f}", flush=True)
    print("AVG:", {t: round(float(np.mean(list(R[t].values()))), 4) for t, _, _ in cfgs})
    print(f"TOTAL {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
