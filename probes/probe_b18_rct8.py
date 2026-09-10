"""probe_b18_rct8.py — 8-RCT grouped-E16 screen: is C27 optimal in E16 frame?

Extends probe_b18_group.eval_bank_on_image over b17's full 8-bank
(C6 C27 C13 C12 C0 C20 C34 C3). K=12, map+4bids+tables exact. ~3 min.
"""
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
from probe_b17_rctw import BANK, BANK_NAMES, rct_fwd
from probe_b4_b import nbhd as nbhd_base
from probe_b4_c import loco_ctx365
from probe_b5_c import predictors_X, E16
import probe_b4_a as A
from probe_b18_group import quantile_groups, group_bits
import math

D = "/tmp/opencode/autocompress/experiments/real_photos"
FILES = ["kodim01.png", "kodim02.png", "kodim05.png", "kodim07.png",
         "kodim13.png", "kodim19.png", "kodim23.png"]
K = 12


def eval_rct(px, bi):
    from probe_b17_rctw import BANK as B
    perm, t = B[bi]
    yc = rct_fwd(px, perm, t)
    H, W, _ = px.shape
    total = 64 + 3  # header + 3b global RCT id
    for ci in range(3):
        ch = yc[:, :, ci]
        a, b, c, d, Ww, NNe, NE = nbhd_base(ch)
        P = predictors_X(a, b, c, d, Ww, NNe, NE)
        key, s = loco_ctx365(a, b, c, d)
        RF = {n: (s * (ch - P[n]).astype(np.int32)).astype(np.int32) for n in E16}
        groups = quantile_groups(key, RF["MED"], K)
        uk = np.unique(key)
        total += 16 + 736 + len(uk) * math.ceil(math.log2(K))
        total += len(groups) * 4
        for gkeys in groups:
            sel = np.isin(key, np.array(gkeys))
            be = None
            for n in E16:
                g = RF[n][sel]
                dd = A.huff_bits(np.unique(g, return_counts=True)[1].tolist()) if g.size else 0
                if be is None or dd < be[0]:
                    be = (dd, n)
            total += group_bits(RF[be[1]][sel])
    return total


def main():
    t0 = time.time()
    print("img: " + " ".join(f"{n:>9}" for n in BANK_NAMES))
    for f in FILES:
        px = np.array(Image.open(os.path.join(D, f)).convert("RGB"))
        H, W, _ = px.shape
        npx = H * W * 3
        vals = [eval_rct(px, bi) / npx for bi in range(len(BANK))]
        best = int(np.argmin(vals))
        print(f"{f}: " + " ".join(f"{v:9.4f}" for v in vals) +
              f"  BEST={BANK_NAMES[best]}", flush=True)
    print(f"TOTAL {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
