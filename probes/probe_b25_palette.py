"""probe_b25: palette oracle (flag-free escape design) on Kodak-7.

Design under test (per image, gated vs CROWN6 bytes, 1 flag bit):
  top-N frequent RGB colors -> palette (side N*24b) + single stream over
  alphabet {ESC} U {idx0..idxN-1} with exact Huffman cost (16+(A)*24 table,
  A = distinct symbols used) + ESC pixels raw 24b each.
No hit/miss flags (ESC is an alphabet symbol); decoder-trivial.
N sweep incl. full-U (exact, zero escapes). All counts exact (heapq hbits).

Gate: oracle(02) < FLIF-02 (358,041 B) -> design C wire integration
(per-image PAL alternative, 1b gate). Also report full-7 vs CROWN6.

Usage: python3 probes/probe_b25_palette.py [--out JSON]
"""
import sys
import os
import json
import heapq
import numpy as np
from PIL import Image

D = "/tmp/opencode/autocompress/experiments/real_photos"
IMGS = ["kodim01.png", "kodim02.png", "kodim05.png", "kodim07.png",
        "kodim13.png", "kodim19.png", "kodim23.png"]
CROWN6 = {"kodim01.png": 486346, "kodim02.png": 440219, "kodim05.png": 527981,
          "kodim07.png": 395158, "kodim13.png": 582929, "kodim19.png": 464631,
          "kodim23.png": 403508}
FLIF = {"kodim01.png": 475578, "kodim02.png": 358041, "kodim05.png": 497770,
        "kodim07.png": 356699, "kodim13.png": 545751, "kodim19.png": 446097,
        "kodim23.png": 385544}
NSWEEP = (256, 512, 1024, 2048, 4096, 8192, 16384)


def hbits_counts(counts):
    counts = [int(c) for c in counts if c > 0]
    if len(counts) == 1:
        return counts[0] * 1 + 16 + 24
    H = [(c, s) for s, c in enumerate(counts)]
    heapq.heapify(H)
    par = {}
    nxt = 1 << 28
    H2 = H[:]
    while len(H2) > 1:
        a, sa = heapq.heappop(H2)
        b, sb = heapq.heappop(H2)
        nn = nxt
        nxt += 1
        par[sa] = (nn, 0)
        par[sb] = (nn, 1)
        heapq.heappush(H2, (a + b, nn))
    tot = 0
    for s, c in enumerate(counts):
        dd = 0
        n = s
        while n in par:
            n = par[n][0]
            dd += 1
        tot += c * dd
    return tot + 16 + len(counts) * 24


def oracle(px, Ns):
    """px: (H,W,3) uint8. Returns {N: total_bits} + full-U entry."""
    flat = px.reshape(-1, 3)
    cols, inv, cn = np.unique(flat, axis=0, return_inverse=True,
                              return_counts=True)
    U = len(cols)
    porder = np.argsort(-cn, kind="stable")  # rank -> col idx, freq desc
    c2r = np.empty(U, dtype=np.int64)
    c2r[porder] = np.arange(U)
    ranks = c2r[inv]
    Npx = flat.shape[0]
    out = {"U": int(U)}
    cands = [N for N in Ns if N < U] + [U]
    for N in cands:
        hit = ranks < N
        n_hit = int(hit.sum())
        n_miss = Npx - n_hit
        # alphabet: ESC + used indices (top-N hit ranks present)
        present = np.unique(ranks[hit]) if n_hit else np.array([], dtype=np.int64)
        A = len(present) + (1 if n_miss else 0)
        counts = [n_miss] + [int((ranks[hit] == r).sum()) for r in present] \
            if n_miss else [int((ranks[hit] == r).sum()) for r in present]
        stream = hbits_counts(counts)
        total = N * 24 + stream + n_miss * 24 + 1  # palette + stream + escapes + gate flag
        out[int(N)] = {"bits": int(total), "hit_rate": round(n_hit / Npx, 4),
                       "n_miss": int(n_miss), "A": int(A)}
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="*", default=IMGS)
    ap.add_argument("--out", default="/tmp/opencode/autocompress/probes/probe_b25_nums.json")
    ap = ap.parse_args()
    res = {}
    for fn in ap.images:
        rgb = np.array(Image.open(os.path.join(D, fn)).convert("RGB"))
        o = oracle(rgb, NSWEEP)
        U = o.pop("U")
        bestN = min(o, key=lambda N: o[N]["bits"])
        bb = o[bestN]["bits"] // 8 + (1 if o[bestN]["bits"] % 8 else 0)
        res[fn] = {"U": U, "bestN": bestN, "bestBits": o[bestN]["bits"],
                   "bestBytes": bb, "crown6": CROWN6[fn], "flif": FLIF[fn],
                   "vsCrown6": bb - CROWN6[fn], "vsFlif": bb - FLIF[fn],
                   "sweep": o}
        print(f"{fn} U={U} bestN={bestN} bytes={bb} "
              f"vsCROWN6={bb-CROWN6[fn]:+d} vsFLIF={bb-FLIF[fn]:+d} "
              f"hit={o[bestN]['hit_rate']}", flush=True)
    with open(ap.out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote " + ap.out, flush=True)


if __name__ == "__main__":
    main()
