"""probe_b10_diag.py — mechanism diagnostics for the magnitude-VQ win.

(a) PYR-R4 exact component split (radius stream/table vs fixed shell-position
    bits vs tail) averaged over 7 images, vs scalar baseline — names the tax
    that kills the pyramid variant.
(b) 4-way magnitude total correlation TC = sum H(Mi) - H(M1..M4) on 2x2 quads
    (infinite-precision headroom for quad-VQ) + clipped (T=4) joint stats:
    P(all-zero quad), P(in-cap), ESC-block tax.
numpy+PIL only, CPU.
"""
import math
import os

import numpy as np
from PIL import Image

import probe_b10_magvq as P

D = P.D
FILES = P.FILES
CHN = ["Y", "Co", "Cg"]


def emp_H(counts):
    n = sum(counts.values())
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def main():
    pyr = {"rstream": 0, "rtab": 0, "posbits": 0, "tstream": 0, "ttab": 0,
           "base": 0}
    tc = {c: [] for c in CHN}
    p0q = {c: [] for c in CHN}
    pincap = {c: [] for c in CHN}
    esctax = {c: [] for c in CHN}  # extra bits paid per escaped block vs scalar
    for fn in FILES:
        px = np.array(Image.open(os.path.join(D, fn)).convert("RGB"))
        H, W, _ = px.shape
        yc = P.fwd(px)
        for c in range(3):
            res = yc[:, :, c] - P.med_pred(yc[:, :, c])
            bc = P.baseline_channel(res)
            pyr["base"] += bc["bits"]
            e = P.pyramid_channel(res, 4)
            for k in ("rstream", "rtab", "posbits", "tstream", "ttab"):
                pyr[k] += e[k]
            # 4-way magnitude TC on 2x2 quads
            b = res.reshape(H // 2, 2, W // 2, 2).transpose(0, 2, 1, 3).reshape(-1, 4)
            m = np.abs(b)
            n = m.shape[0]
            Hs = [emp_H(dict(zip(*np.unique(m[:, i], return_counts=True)))) for i in range(4)]
            uu, cc = np.unique(m, axis=0, return_counts=True)
            Hj = -sum((x / n) * math.log2(x / n) for x in cc)
            tc[CHN[c]].append(sum(Hs) - Hj)
            p0q[CHN[c]].append(float((m.sum(axis=1) == 0).mean()))
            T = 4
            incap = m.max(axis=1) <= T
            pincap[CHN[c]].append(float(incap.mean()))
            # ESC tax: bits an escaped block costs under VQ (ESC symbol +
            # 4 scalar tail lengths) minus what scalar Huffman charges the
            # same 4 values. Uses realized lengths from both tables.
            q = P.quad_mag_channel(res, T)
            JL, TL, BL = q["JL"], q["TL"], bc["L"]
            lesc = JL[q["ESC"]]
            tax = 0
            nesc = 0
            for row, ok in zip(b.tolist(), incap.tolist()):
                if not ok:
                    vq = lesc + sum(TL[int(v)] for v in row)
                    sc = sum(BL[int(v)] for v in row)
                    tax += vq - sc
                    nesc += 1
            esctax[CHN[c]].append(tax / max(nesc, 1))
    print("PYR-R4 7-image totals (bits): base=%d rstream=%d rtab=%d posbits=%d "
          "tstream+ttab=%d" % (pyr["base"], pyr["rstream"], pyr["rtab"],
                               pyr["posbits"], pyr["tstream"] + pyr["ttab"]))
    print("  radius stream+table share: %.1f%%, fixed-position share: %.1f%%" %
          ((pyr["rstream"] + pyr["rtab"]) / pyr["base"] * 100,
           pyr["posbits"] / pyr["base"] * 100))
    for c in CHN:
        print(f"  TC4 {c}: {float(np.mean(tc[c])):.4f} b/block "
              f"({float(np.mean(tc[c])) / 4:.4f} b/residual)")
    for c in CHN:
        print(f"  P(all-zero quad) {c}: {float(np.mean(p0q[c])):.3f} | "
              f"P(in-cap T=4) {c}: {float(np.mean(pincap[c])):.3f} | "
              f"ESC-block tax {c}: {float(np.mean(esctax[c])):.2f} b/block")


if __name__ == "__main__":
    main()
