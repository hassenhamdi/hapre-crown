"""probe_b18_oracle.py — predictor-headroom bound: per-pixel oracle over ~40
predictors (untransmittable) -> entropy floor of best-achievable STATIC
prediction on 05/13/01 (C27 frame). If oracle floor >> current entropy,
a real expert is missing; if ~=, predictor space exhausted.
Fast vectorized. numpy+PIL only.
"""
import math
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
from probe_b17_rctw import rct_fwd, med_pred
from probe_b4_b import nbhd as nbhd_base
from probe_b4_c import loco_ctx365
from probe_b5_c import predictors_X
from probe_b18_screen import nbhd_big
from probe_b18_micro import micro_preds

D = "/tmp/opencode/autocompress/experiments/real_photos"


def entropy_of(v):
    _, cn = np.unique(v, return_counts=True)
    p = cn.astype(np.float64) / cn.sum()
    return float((-(p * np.log2(p))).sum() * cn.sum())


def main():
    t0 = time.time()
    for fn in ["kodim05.png", "kodim13.png", "kodim01.png"]:
        px = np.array(Image.open(os.path.join(D, fn)).convert("RGB"))
        H, W, _ = px.shape
        npx = H * W * 3
        yc = rct_fwd(px, 3, 6)
        tot_oracle = 0
        tot_med = 0
        for ci in range(3):
            ch = yc[:, :, ci]
            a, b, c, d, Ww, NNe, NE = nbhd_base(ch)
            P = dict(predictors_X(a, b, c, d, Ww, NNe, NE))
            NB = nbhd_big(ch)
            L, T = NB["L"], NB["T"]
            P["WL2"] = L[2]
            P["WL4"] = L[4]
            P["WN2"] = T[2]
            P["HAVG4"] = (a + L[2] + L[3] + L[4]) // 4
            P["XH"] = 2 * a - Ww
            P["XV"] = 2 * b - NNe
            P.update(micro_preds(a, b, c, d, Ww, NNe, NE))
            key, s = loco_ctx365(a, b, c, d)
            R = {n: (s * (ch - P[n]).astype(np.int32)) for n in P}
            names = list(P.keys())
            stack = np.stack([np.abs(R[n]) for n in names], axis=0)
            best = np.argmin(stack, axis=0)
            # oracle residual = sign-flipped residual of per-pixel winner
            orr = np.choose(best, [R[n] for n in names])
            tot_oracle += entropy_of(orr)
            tot_med += entropy_of(R["MED"])
        print(f"{fn}: MEDent={tot_med / npx:.4f} ORACLE40ent={tot_oracle / npx:.4f} "
              f"(d={(tot_oracle - tot_med) / npx:+.4f})", flush=True)
    print(f"TOTAL {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
