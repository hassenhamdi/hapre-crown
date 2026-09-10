"""probe_b18_bilat.py — chroma-bilateral predictor screen (new file only).

Mechanism: at color edges (coincide with luma edges), causal chroma
predictors ring/overshoot; in flat-chroma texture, smoothing beats MED.
BILAT-H/T: average causal chroma neighbors {L,T}(/TL,TR) whose LUMA is
close to luma-proxy MED(Y) (|dY|<=T); fallback MED if none qualify.
Decoder-safe: Y coded first (CROWN3 order), all inputs causal recon.
Screen: grouped-E16 + BILAT experts on Co/Cg only (luma bank unchanged),
K=12, exact gate incl. 5b ids (17->~21 experts). All 7. numpy+PIL only.
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
from probe_b5_c import predictors_X, E16
import probe_b4_a as A
from probe_b18_group import quantile_groups, group_bits

D = "/tmp/opencode/autocompress/experiments/real_photos"
FILES = ["kodim01.png", "kodim02.png", "kodim05.png", "kodim07.png",
         "kodim13.png", "kodim19.png", "kodim23.png"]
JXL = {"kodim01.png": 3.3593, "kodim02.png": 3.0611, "kodim05.png": 3.5120,
       "kodim07.png": 2.7310, "kodim13.png": 3.9141, "kodim19.png": 3.2154,
       "kodim23.png": 2.8110}
K = 12


def bilat_preds(chroma, Y, T):
    """Returns dict of bilateral chroma predictions (int32 planes)."""
    ac, bc, cc, dc, Wwc, NNec, NEc = nbhd_base(chroma)
    aY, bY, cY, dY, WwY, NNeY, NEY = nbhd_base(Y)
    Ym = med_pred(Y)  # luma proxy (decoder-safe: Y recon known)
    out = {}
    # neighbor luma values aligned with chroma stencil positions
    LY, TY, TLY, TRY = aY, bY, cY, dY
    for tag, members in [("B2", [("L", ac, LY), ("T", bc, TY)]),
                         ("B4", [("L", ac, LY), ("T", bc, TY),
                                 ("TL", cc, TLY), ("TR", dc, TRY)])]:
        num = np.zeros_like(chroma)
        den = np.zeros_like(chroma)
        for _, Cv, Yv in members:
            ok = (np.abs(Yv.astype(np.int32) - Ym.astype(np.int32)) <= T)
            num += np.where(ok, Cv, 0)
            den += np.where(ok, 1, 0)
        M = med_pred(chroma)
        avg = np.where(den > 0, num // np.maximum(den, 1), M)
        out[f"BL{tag}T{T}"] = avg
    return out


def eval_img(px, use_bilat, T=16):
    yc = rct_fwd(px, 3, 6)
    H, W, _ = px.shape
    Y = yc[:, :, 0]
    total = 64
    allpicks = {}
    for ci in range(3):
        ch = yc[:, :, ci]
        a, b, c, d, Ww, NNe, NE = nbhd_base(ch)
        P = dict(predictors_X(a, b, c, d, Ww, NNe, NE))
        bank = list(E16)
        if use_bilat and ci in (1, 2):
            XB = bilat_preds(ch, Y, T)
            P.update(XB)
            bank += list(XB.keys())
        key, s = loco_ctx365(a, b, c, d)
        RF = {n: (s * (ch - P[n]).astype(np.int32)).astype(np.int32) for n in bank}
        groups = quantile_groups(key, RF["MED"], K)
        uk = np.unique(key)
        total += 16 + 736 + len(uk) * math.ceil(math.log2(K))
        idb = math.ceil(math.log2(len(bank)))
        total += len(groups) * idb
        for gkeys in groups:
            sel = np.isin(key, np.array(gkeys))
            be = None
            for n in bank:
                g = RF[n][sel]
                dd = A.huff_bits(np.unique(g, return_counts=True)[1].tolist()) if g.size else 0
                if be is None or dd < be[0]:
                    be = (dd, n)
            total += group_bits(RF[be[1]][sel])
            allpicks[be[1]] = allpicks.get(be[1], 0) + 1
    return total, allpicks


def main():
    t0 = time.time()
    for T in [8, 16, 32]:
        print(f"== BILAT T={T} ==")
        for f in FILES:
            px = np.array(Image.open(os.path.join(D, f)).convert("RGB"))
            H, W, _ = px.shape
            npx = H * W * 3
            b0, _ = eval_img(px, False)
            b1, pk = eval_img(px, True, T)
            new = {k: v for k, v in pk.items() if k.startswith("BL")}
            print(f"T={T} {f}: base={b0 / npx:.4f} bilat={b1 / npx:.4f} "
                  f"({(b1 - b0) / npx:+.4f}) picks={new if new else '-'} "
                  f"JXL={JXL[f]:.4f}", flush=True)
    print(f"TOTAL {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
