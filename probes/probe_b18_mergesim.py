"""probe_b18_mergesim.py — offline greedy-merge headroom on dumped Q-groups.

Loads per-channel Q-group dumps (symbols per group via symplane+expmap),
greedy agglomerative merging on EXACT Huffman totals (data + 16+A*24 tables
+ ng*ceil(log2 C) map + 8b C header), gated vs unmerged. Reports per-channel
and total gain. Decides whether to build W-D (decode support).
"""
import math
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
import probe_b4_a as A

NCH = {"kodim05.png": 3, "kodim13.png": 3}


def huff_total(counts):
    """counts: dict sym->n. Returns data+table bits."""
    if not counts:
        return 0
    cn = list(counts.values())
    return A.huff_bits(cn) + 16 + len(cn) * 24


def add_counts(a, b):
    c = dict(a)
    for k, v in b.items():
        c[k] = c.get(k, 0) + v
    return c


def greedy(symplane, expmap, ng_orig):
    groups = []
    for gi in range(ng_orig):
        vals, cn = np.unique(symplane[expmap == gi], return_counts=True)
        if len(vals):
            groups.append({int(v): int(n) for v, n in zip(vals.tolist(), cn.tolist())})
    spent = [huff_total(g) for g in groups]
    base = sum(spent)
    C = len(groups)
    if C <= 1:
        return base, base, 0
    Alloc = list(range(C))

    def mapbits(nc):
        return ng_orig * math.ceil(math.log2(nc)) + 8 if nc > 1 else 0

    total = base + mapbits(C)
    while len(Alloc) > 1:
        L = len(Alloc)
        best = None
        for ii in range(L):
            for jj in range(ii + 1, L):
                i, j = Alloc[ii], Alloc[jj]
                m = add_counts(groups[i], groups[j])
                d = huff_total(m) - spent[i] - spent[j]
                d += mapbits(L - 1) - mapbits(L)
                if best is None or d < best[0]:
                    best = (d, i, j, m)
        if best is None or best[0] >= 0:
            break
        d, i, j, m = best
        total += d
        # merge j into i
        groups[i] = m
        spent[i] = huff_total(m)
        Alloc.remove(j)
    return base, total, len(Alloc)


def main():
    for fn in ["kodim05.png", "kodim13.png"]:
        path = f"/tmp/b18_dump_{fn[:7]}.pkl"
        if not os.path.exists(path):
            print(f"missing {path} (run dump first)")
            continue
        recs = []
        with open(path, "rb") as f:
            while True:
                try:
                    recs.append(pickle.load(f))
                except EOFError:
                    break
        print(f"== {fn}: {len(recs)} channel dumps ==")
        tot_b = tot_m = 0
        for ci, r in enumerate(recs):
            b, m, nc = greedy(r["symplane"], r["expmap"], r["ng"])
            print(f"  ch{ci}: K={r['K']} ng={r['ng']} unmergedH={b} mergedH={m} "
                  f"(d={m - b:+.0f}, clusters={nc})", flush=True)
            tot_b += b
            tot_m += m
        npx = 512 * 768 * 3
        print(f"  TOTAL: d={(tot_m - tot_b) / npx:+.4f} bpp", flush=True)


if __name__ == "__main__":
    main()
