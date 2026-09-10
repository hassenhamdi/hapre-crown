"""probe_b18_mix.py — parametric mixture-Huffman vs Golomb on dumped G-groups.

Texture groups pick Golomb (table-free) but suffer shape-misfit (0.0266 on
05-ch0). Idea: Huffman tables REBUILT from a parametric mixture (scale s per
group + shape id) = tableless-Huffman: same ~7b side as Golomb, better shape.
Screen: 2-Laplacian mixture {w0 Lap(s) + w1 Lap(R*s)}, (w0,R) grid x s grid,
exact Huffman data + 7b side vs Golomb-exact. Offline on dumps. Fast.
"""
import math
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
import probe_b4_a as A
from probe_b18_xtables import huff_lengths

SHAPES = [(0.85, 6.0), (0.9, 8.0), (0.95, 10.0), (1.0, 1.0)]  # (w0, R); last=Laplacian


def mix_pmf(M, s, w0, R):
    b0 = max(s, 0.5)
    b1 = max(R * s, 0.5)
    xs = np.arange(-M, M + 1, dtype=np.float64)
    p = w0 * np.exp(-np.abs(xs) / b0) / (2 * b0) + (1 - w0) * np.exp(-np.abs(xs) / b1) / (2 * b1)
    p /= p.sum()
    return p


def mix_huff_bits(vals, M, s, w0, R):
    vals = np.asarray(vals, dtype=np.int64)
    if np.abs(vals).max() > M:
        return None  # out of range for this table
    p = mix_pmf(M, s, w0, R)
    # canonical lengths from pmf via heapq on symbols present in pmf support...
    # full support 2M+1 symbols may be big; restrict: lengths via Shannon+1 cap?
    # EXACT: Huffman on full support counts = round(p*N_big)? Use lengths from
    # heapq over support with weights p (float) — valid Huffman instance.
    import heapq
    sup = list(range(-M, M + 1))
    heap = [(float(p[i]), j, [sup[j]]) for j, i in enumerate(range(len(sup)))]
    heapq.heapify(heap)
    depth = {s_: 0 for s_ in sup}
    nxt = len(sup)
    while len(heap) > 1:
        c1, _, m1 = heapq.heappop(heap)
        c2, _, m2 = heapq.heappop(heap)
        for x in m1:
            depth[x] += 1
        for x in m2:
            depth[x] += 1
        heapq.heappush(heap, (c1 + c2, nxt, m1 + m2))
        nxt += 1
    tot = sum(depth[int(v)] for v in vals.tolist())
    return tot


def main():
    t0 = time.time()
    for fn in ["kodim05.png", "kodim13.png"]:
        recs = []
        with open(f"/tmp/b18_dump_{fn[:7]}.pkl", "rb") as f:
            while True:
                try:
                    recs.append(pickle.load(f))
                except EOFError:
                    break
        print(f"== {fn} ==")
        npx = 512 * 768 * 3
        tb = ta = 0
        for ci, r in enumerate(recs):
            symplane, expmap, ng = r["symplane"], r["expmap"], r["ng"]
            wins = r["wins"]
            sb = sa = 0
            for gi in range(ng):
                w = wins[gi]
                if w[2] != 1:
                    continue
                g = symplane[expmap == gi].astype(np.int64).reshape(-1)
                sb += w[0]
                M = int(np.abs(g).max())
                best = None
                ma = float(np.abs(g).mean())
                for s in [ma * f for f in (0.5, 0.7, 1.0, 1.4, 2.0)]:
                    for (w0, R) in SHAPES:
                        hb = mix_huff_bits(g, M, s, w0, R)
                        if hb is None:
                            continue
                        cand = hb + 7  # s-grid(3b)+shape(2b)+d(2b-ish); d folded: use w[4] side as-is
                        if best is None or cand < best:
                            best = cand
                sa += (best + (3 if w[4] != 0 else 0)) if best is not None else w[0]
            print(f"  ch{ci}: Golomb={sb:.0f} mixHuff={sa:.0f} d={sa - sb:+.0f} "
                  f"({(sa - sb) / npx:+.4f}bpp)", flush=True)
            tb += sb
            ta += sa
        print(f"  TOTAL Golomb-part: d={(ta - tb) / npx:+.4f} bpp", flush=True)
    print(f"TOTAL {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
