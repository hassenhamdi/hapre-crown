"""probe_b18_xtables.py — cross-image GLOBAL Huffman tables (offline, dumps).

Survey TOP-1 never tried: 'learn the tables, ship the tables'. Online merges
(b12/b13) share tables WITHIN an image; this shares ACROSS images (codec-fixed
tables, amortized cost) — breaks the texture trilemma (table-free + flexible):
per-group pick min(current mixed H/G/R, global-Huffman + table-id side),
tables counted once globally (/7 amortized). Gated exact. LOO honesty check.
"""
import math
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
import probe_b4_a as A

NIMG = 7


def huff_data(counts):
    cn = list(counts.values())
    return A.huff_bits(cn) if cn else 0


def table_bits(nsym):
    return 16 + nsym * 24


def add_counts(a, b):
    c = dict(a)
    for k, v in b.items():
        c[k] = c.get(k, 0) + v
    return c


def load_atoms(fns):
    atoms = []  # (img, ch, counts dict, current_bits, vec|None)
    for fn in fns:
        path = f"/tmp/b18_dump_{fn[:7]}.pkl"
        recs = []
        with open(path, "rb") as f:
            while True:
                try:
                    recs.append(pickle.load(f))
                except EOFError:
                    break
        for ci, r in enumerate(recs):
            symplane, expmap, ng = r["symplane"], r["expmap"], r["ng"]
            wins = r["wins"]
            for gi in range(ng):
                vals, cn = np.unique(symplane[expmap == gi], return_counts=True)
                if not len(vals):
                    continue
                counts = {int(v): int(n) for v, n in zip(vals.tolist(), cn.tolist())}
                atoms.append({"img": fn, "ch": ci, "gi": gi, "counts": counts,
                              "cur": wins[gi][0],
                              "vec": symplane[expmap == gi].reshape(-1)})
    return atoms


def lloyd(atoms, K, iters=15, seed=0):
    rng = np.random.default_rng(seed)
    # init: K biggest atoms
    order = sorted(range(len(atoms)), key=lambda i: -sum(atoms[i]["counts"].values()))
    centers = [dict(atoms[i]["counts"]) for i in order[:K]]
    assign = [0] * len(atoms)
    for _ in range(iters):
        changed = 0
        # precompute center count-lookups
        for ai, a in enumerate(atoms):
            best, bk = None, 0
            for ki, c in enumerate(centers):
                # data bits of a under Huffman table of c: need lengths of c
                # approx by cross-entropy with c's MLE + Huffman overhead factor:
                # EXACT-ish: lengths from c, data = sum_a n*len
                tot_c = sum(c.values())
                # Huffman lengths of c via heapq depths (approx by Shannon+0.1?)
                # Use exact: build lengths once per iter per center
                pass
            assign[ai] = bk
        break
    return centers, assign


def huff_lengths(counts):
    """Exact Huffman code lengths via heapq (symbol->len)."""
    import heapq
    syms = list(counts.keys())
    if len(syms) == 1:
        return {syms[0]: 1}
    heap = [(counts[s], s) for s in syms]
    heapq.heapify(heap)
    depth = {s: 0 for s in syms}
    nxt = 0
    # need tie-break independence: use (count, tiebreak, members)
    heap = [(counts[s], i, [s]) for i, s in enumerate(syms)]
    heapq.heapify(heap)
    nxt = len(syms)
    while len(heap) > 1:
        c1, _, m1 = heapq.heappop(heap)
        c2, _, m2 = heapq.heappop(heap)
        for s in m1:
            depth[s] += 1
        for s in m2:
            depth[s] += 1
        heapq.heappush(heap, (c1 + c2, nxt, m1 + m2))
        nxt += 1
    return depth


def lloyd_exact(atoms, K, iters=12):
    order = sorted(range(len(atoms)), key=lambda i: -sum(atoms[i]["counts"].values()))
    centers = [dict(atoms[i]["counts"]) for i in order[:K]]
    assign = [0] * len(atoms)
    for it in range(iters):
        lens = [huff_lengths(c) for c in centers]
        maxlen = [max(l.values()) if l else 1 for l in lens]
        changed = 0
        for ai, a in enumerate(atoms):
            best, bk = None, 0
            for ki in range(K):
                L = lens[ki]
                ML = maxlen[ki]
                # data = sum n*len (missing symbols -> escape ML+8, rare; penalize)
                d = 0
                for s, n in a["counts"].items():
                    d += n * L.get(s, ML + 8)
                if best is None or d < best:
                    best, bk = d, ki
            if bk != assign[ai]:
                changed += 1
            assign[ai] = bk
        # update
        newc = [dict() for _ in range(K)]
        for ai, a in enumerate(atoms):
            newc[assign[ai]] = add_counts(newc[assign[ai]], a["counts"])
        # drop empty centers (reinit to biggest unfit atom) — keep simple: keep
        centers = [c if c else centers[i] for i, c in enumerate(newc)]
        if changed == 0:
            break
    lens = [huff_lengths(c) for c in centers]
    return centers, assign, lens


def evaluate(fns, K, loo=None):
    atoms = load_atoms(fns)
    train = [a for a in atoms if a["img"] != loo]
    test = atoms if loo is None else [a for a in atoms if a["img"] == loo]
    centers, assign_tr, lens = lloyd_exact(train, K)
    idb = math.ceil(math.log2(K))
    # table cost amortized
    tab = sum(table_bits(len(c)) for c in centers) / NIMG
    npx = 512 * 768 * 3
    # assign test atoms (gated per-atom: min(cur, global+id) + 1b choice flag/group)
    tot_cur = tot_new = 0
    per_img = {}
    maxlen = [max(l.values()) if l else 1 for l in lens]
    ngroups_test = 0
    for a in test:
        cur = a["cur"]
        best = None
        for ki in range(K):
            L = lens[ki]
            d = 0
            ok = True
            for s, n in a["counts"].items():
                if s not in L:
                    ok = False
                    break
                d += n * L[s]
            if not ok:
                continue
            if best is None or d < best:
                best = d
        new = (best + idb) if best is not None else float("inf")
        pick = min(cur, new)
        tot_cur += cur
        tot_new += pick
        ngroups_test += 1
        k = (a["img"], a["ch"])
        if k not in per_img:
            per_img[k] = [0, 0]
        per_img[k][0] += cur
        per_img[k][1] += pick
    # +1b choice flag per test group
    tot_new += ngroups_test
    return tot_cur, tot_new, tab, per_img, K


def main():
    t0 = time.time()
    fns = ["kodim05.png", "kodim13.png"]
    for K in [8, 16, 32]:
        cur, new, tab, per, _ = evaluate(fns, K)
        npx = 512 * 768 * 3
        # NOTE: cur/new here sum over BOTH images; mapside/xtra/group-id parts
        # cancel (identical grouping); only sub-bit (wins) compared + tables.
        print(f"K={K}: sub_cur={cur:.0f} sub_new={new:.0f} tables_amort={tab:.0f} "
              f"net={(new + tab - cur) / npx:+.4f} bpp/img "
              f"(train+test on 05+13: optimistic)", flush=True)
    cur, new, tab, per, _ = evaluate(fns, 16, loo="kodim05.png")
    npx = 512 * 768 * 3
    # LOO: test atoms = 05 only; tables trained w/o 05
    t05cur = sum(v[0] for k, v in per.items() if k[0] == "kodim05.png")
    t05new = sum(v[1] for k, v in per.items() if k[0] == "kodim05.png")
    print(f"LOO-05 K=16: 05sub_cur={t05cur:.0f} 05sub_new={t05new:.0f} "
          f"tables_amort(1/7)={tab:.0f} net05={(t05new + tab - t05cur) / npx:+.4f} bpp",
          flush=True)
    print(f"TOTAL {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
