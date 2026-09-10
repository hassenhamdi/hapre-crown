"""probe_b18_micro.py — micro-predictor screen + merge-headroom dump (new file).

A. Micro experts (vectorized, grouped-E16 K=12 gate, exact bits, all 7):
   MEDIAN5 (median of {L,T,TL,TR,C?} causal 5), GAP8, GAP4, AVG4.
B. Merge-headroom dump (05/13, C27, GS32+LMS frame): per-group count vectors
   + bits saved to /tmp/b18_groups_05.npz etc. for offline greedy-merge sim.
"""
import math
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
os.chdir("/tmp/opencode/autocompress/experiments")
import probe_b4_a as A
from probe_b4_b import nbhd as nbhd_base
from probe_b4_c import loco_ctx365
from probe_b5_c import predictors_X, E16
from probe_b17_rctw import rct_fwd, med_pred
from probe_b18_group import quantile_groups, group_bits

D = "/tmp/opencode/autocompress/experiments/real_photos"
FILES = ["kodim01.png", "kodim02.png", "kodim05.png", "kodim07.png",
         "kodim13.png", "kodim19.png", "kodim23.png"]
JXL = {"kodim01.png": 3.3593, "kodim02.png": 3.0611, "kodim05.png": 3.5120,
       "kodim07.png": 2.7310, "kodim13.png": 3.9141, "kodim19.png": 3.2154,
       "kodim23.png": 2.8110}
K = 12


def micro_preds(a, b, c, d, Ww, NNe, NE):
    P = {}
    stack = np.stack([a, b, c, d,
                      np.zeros_like(a)], axis=0)
    # median of {L,T,TL,TR} (4 values -> average of middle two, integer)
    s = np.sort(np.stack([a, b, c, d], axis=0), axis=0)
    P["MEDIAN4"] = (s[1] + s[2]) // 2
    s5 = np.sort(np.stack([a, b, c, d, (a + b) // 2], axis=0), axis=0)
    P["MEDIAN5"] = s5[2]
    gh = np.abs(a - Ww) + np.abs(b - c) + np.abs(b - NE)
    gv = np.abs(a - c) + np.abs(b - NNe) + abs(NE - b) if False else np.abs(a - c) + np.abs(b - NNe) + np.abs(NE - b)
    for thr, nm in [(8, "GAP8"), (4, "GAP4")]:
        P[nm] = np.where(gv - gh > thr, a, np.where(gv - gh < -thr, b, (a + b) // 2 + (NE - c) // 4))
    P["AVG4"] = (a + b + c + d) // 4
    return P


def eval_bank(px, extra):
    yc = rct_fwd(px, 3, 6)
    H, W, _ = px.shape
    total = 64
    picks = {}
    for ci in range(3):
        ch = yc[:, :, ci]
        a, b, c, d, Ww, NNe, NE = nbhd_base(ch)
        P = dict(predictors_X(a, b, c, d, Ww, NNe, NE))
        bank = list(E16)
        if extra:
            P.update(micro_preds(a, b, c, d, Ww, NNe, NE))
            bank += ["MEDIAN4", "MEDIAN5", "GAP8", "GAP4", "AVG4"]
        key, s = loco_ctx365(a, b, c, d)
        RF = {n: (s * (ch - P[n]).astype(np.int32)).astype(np.int32) for n in bank}
        groups = quantile_groups(key, RF["MED"], K)
        uk = np.unique(key)
        total += 16 + 736 + len(uk) * math.ceil(math.log2(K))
        total += len(groups) * math.ceil(math.log2(len(bank)))
        for gkeys in groups:
            sel = np.isin(key, np.array(gkeys))
            be = None
            for n in bank:
                g = RF[n][sel]
                dd = A.huff_bits(np.unique(g, return_counts=True)[1].tolist()) if g.size else 0
                if be is None or dd < be[0]:
                    be = (dd, n)
            total += group_bits(RF[be[1]][sel])
            picks[be[1]] = picks.get(be[1], 0) + 1
    return total, picks


def main():
    t0 = time.time()
    print("== micro-predictor grouped screen ==")
    for fn in FILES:
        px = np.array(Image.open(os.path.join(D, fn)).convert("RGB"))
        H, W, _ = px.shape
        npx = H * W * 3
        b0, _ = eval_bank(px, False)
        b1, pk = eval_bank(px, True)
        new = {k: v for k, v in pk.items() if k not in E16}
        print(f"{fn}: E16={b0 / npx:.4f} +micro={b1 / npx:.4f} ({(b1 - b0) / npx:+.4f}) "
              f"picks={new if new else '-'} JXL={JXL[fn]:.4f}", flush=True)
    print(f"TOTAL {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
