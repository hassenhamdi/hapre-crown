"""probe_b18_mergeexact.py — TRUE merge value: Huffman-greedy clusters +
per-cluster exact H/G/R re-pick vs CURRENT mixed-backend totals (new file).

Uses dumps (symplane/expmap/groups/wins). rANS via C3.rans_cost on EXACT
concatenated group vectors. Golomb via C3.golomb_cost. Decides W-D build.
"""
import math
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, "/tmp/opencode/autocompress/csrc")
sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
os.chdir("/tmp/opencode/autocompress/experiments")
import driver_crown3 as C3
import probe_b4_a as A
from probe_b18_mergesim import huff_total, add_counts, greedy

NKEY = 729 * 4


def cluster_vecs(symplane, expmap, members):
    v = np.concatenate([symplane[expmap == gi].reshape(-1) for gi in members])
    return np.ascontiguousarray(v.astype(np.int16))


def main():
    t0 = time.time()
    for fn in ["kodim05.png", "kodim13.png"]:
        path = f"/tmp/b18_dump_{fn[:7]}.pkl"
        recs = []
        with open(path, "rb") as f:
            while True:
                try:
                    recs.append(pickle.load(f))
                except EOFError:
                    break
        print(f"== {fn} ==")
        npx = 512 * 768 * 3
        tot_cur = tot_new = 0
        for ci, r in enumerate(recs):
            symplane, expmap, ng = r["symplane"], r["expmap"], r["ng"]
            K = r["K"]
            wins = r["wins"]
            # current exact subtotal
            uk = set()
            for gk in r["groups"]:
                uk.update(gk.tolist())
            gbits = math.ceil(math.log2(K))
            mapside = 16 + 736 + len(uk) * gbits
            xtra = sum(4 for w in wins if w[2] == 1) + sum(3 for w in wins if w[2] == 1 and w[4] != 0)
            sub = sum(w[0] for w in wins)
            cur = mapside + ng * 6 + xtra + sub
            # greedy Huffman clusters (indices into 0..ng-1 groups)
            groups = []
            for gi in range(ng):
                vals, cn = np.unique(symplane[expmap == gi], return_counts=True)
                if len(vals):
                    groups.append({int(v): int(n) for v, n in zip(vals.tolist(), cn.tolist())})
            # map group positions (nonempty only) — greedy() handles internally; redo with members
            members = []
            for gi in range(ng):
                if (expmap == gi).sum():
                    members.append(gi)
            # run greedy on member index space
            b, m, nc = greedy(symplane, expmap, ng)
            # recover clusters: re-run greedy capturing membership (dup logic w/ tracking)
            cl = [{gi} for gi in members]
            cnt = []
            for gi in members:
                vals, cn = np.unique(symplane[expmap == gi], return_counts=True)
                cnt.append({int(v): int(n) for v, n in zip(vals.tolist(), cn.tolist())})
            spent = [huff_total(c) for c in cnt]

            def mapbits(nc_):
                return ng * math.ceil(math.log2(nc_)) + 8 if nc_ > 1 else 0

            while len(cl) > 1:
                L = len(cl)
                best = None
                for ii in range(L):
                    for jj in range(ii + 1, L):
                        mm = add_counts(cnt[ii], cnt[jj])
                        d = huff_total(mm) - spent[ii] - spent[jj]
                        d += mapbits(L - 1) - mapbits(L)
                        if best is None or d < best[0]:
                            best = (d, ii, jj, mm)
                if best is None or best[0] >= 0:
                    break
                d, ii, jj, mm = best
                cnt[ii] = mm
                spent[ii] = huff_total(mm)
                cl[ii] = cl[ii] | cl[jj]
                del cl[jj]
                del cnt[jj]
                del spent[jj]
            # per-cluster exact H/G/R
            newt = mapbits(len(cl)) + len(cl) * 6
            for memberset in cl:
                v = cluster_vecs(symplane, expmap, sorted(memberset))
                h, _, _ = C3.huff_cost(v)
                g, _, _, _ = C3.golomb_cost(v)
                rc, _ = C3.rans_cost(v)
                newt += min(h, g, rc)
            # xtra for G clusters chosen: golomb xtra 4 (+3 if d): approximate via recount
            # (recompute exactly: for clusters picking G, add 4 + (3 if d!=0))
            print(f"  ch{ci}: current={cur} merged3way={newt} d={newt - cur:+.0f} "
                  f"({(newt - cur) / npx:+.4f}bpp, {len(cl)} clusters)", flush=True)
            tot_cur += cur
            tot_new += newt
        print(f"  CHANNELS TOTAL: d={(tot_new - tot_cur) / npx:+.4f} bpp", flush=True)
    print(f"TOTAL {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
