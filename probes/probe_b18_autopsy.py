"""probe_b18_autopsy.py — exact-arm bit autopsy on 05/13/01 (new file only).

Reuses driver_crown3 machinery by import (fit_weighted, prepX, quantile_groups,
huff/golomb/rans cost fns, E17/RF/RU construction) and logs, per channel and
per group: expert, backend, N, alphabet A, chosen bits, Shannon entropy of the
group residual, Huffman-data-vs-entropy gap, rANS-vs-chosen gap.
Answers: is the 05/13 loss residual-entropy (prediction) or coding-gap (Huffman
rounding/tables/backend-side)? Single RCT C27 (CROWN3 winner on 05; 13/01 run
under C27 too for comparability + note).
"""
import math
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, "/tmp/opencode/autocompress/csrc")
sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
os.chdir("/tmp/opencode/autocompress/experiments")
import driver_crown3 as C3
import probe_b17_rctw as B17
from probe_b5_c import prepX, E16

D = "/tmp/opencode/autocompress/experiments/real_photos"


def entropy_of(g):
    _, cn = np.unique(np.asarray(g).reshape(-1), return_counts=True)
    p = cn.astype(np.float64) / cn.sum()
    return float((-(p * np.log2(p))).sum() * cn.sum())


def autopsy(path, rname="C27", perm=3, t=6):
    px = np.array(Image.open(path).convert("RGB"))
    fn = path.split("/")[-1]
    H, W, _ = px.shape
    npx = H * W * 3
    yc = B17.rct_fwd(px, perm, t)
    planes = [yc[:, :, c] for c in range(3)]
    wchs, bm, nbh, nbw, ng32, wside = C3.fit_weighted(planes, H, W)
    print(f"== {fn} {rname}: H={H} W={W} wside={wside}b ({wside / npx:.4f}bpp) "
          f"wuse={[int(w['use'].sum()) for w in wchs]}", flush=True)
    tot_bits = 64 + 8  # header + rct id (approx: 2b, count 8)
    for ci, ch in enumerate(planes):
        DD = prepX(ch)
        RF = {n: (DD["s"] * (ch - DD["P"][n]).astype(np.int32)).astype(np.int32)
              for n in E16}
        RF["WAVG"] = (DD["s"] * wchs[ci]["res"]).astype(np.int32)
        RU = {n: (ch - DD["P"][n]).astype(np.int32) for n in E16}
        RU["WAVG"] = wchs[ci]["res"]
        for fam_name, plan in [("Q", C3.encode_channel_q(ch, DD, RF)),
                               ("G", C3.encode_channel_grid(ch, DD, RU))]:
            # recompute exact subtotal for plan: mapside + groups*6 + xtra + sum wins
            wins = plan["wins"]
            ng = len(wins)
            if plan["family"] == 0:
                uk = np.unique(DD["key"])
                gbits = math.ceil(math.log2(plan["K"]))
                mapside = 16 + 736 + len(uk) * gbits
            else:
                mapside = 16 + 8
            xtra = sum(4 for w in wins if w[2] == 1) + \
                sum(3 for w in wins if w[2] == 1 and w[4] != 0)
            sub = sum(w[0] for w in wins)
            wside_ch = (ng32 + 7) // 8 * 8 + sum(int(wchs[ci]["use"].sum()) * 20 for _ in [0])
            tbits = mapside + ng * 6 + xtra + sub
            # per-group detail with entropy
            det = []
            agg = {"H": [0, 0, 0], "G": [0, 0, 0], "R": [0, 0, 0]}  # bits, entropy, n
            for gi, w in enumerate(wins):
                cost, bn, bb = w[0], w[1], w[2]
                # recover group mask
                if plan["family"] == 0:
                    cmap = {}
                    for gix, gkeys in enumerate(plan["groups"]):
                        for u in gkeys:
                            cmap[int(u)] = gix
                    expmap = np.array([cmap[int(u)] for u in DD["key"].reshape(-1)]).reshape(H, W)
                    m = (expmap == gi)
                else:
                    m = (plan["gmap"] == gi)
                if bn == "WAVG":
                    r = (DD["s"] * wchs[ci]["res"])[m]
                else:
                    r = (DD["s"] * (ch - DD["P"][bn]).astype(np.int32))[m]
                ent = entropy_of(r) if r.size else 0
                N = int(m.sum())
                A_ = len(np.unique(r)) if r.size else 0
                nm = "H" if bb == 0 else ("G" if bb == 1 else "R")
                agg[nm][0] += cost
                agg[nm][1] += ent
                agg[nm][2] += N
                det.append((N, A_, bn, nm, cost, ent,
                            cost - ent if N else 0))
            print(f"  ch{ci} fam={fam_name} K={plan['K']} ng={ng} "
                  f"mapside={mapside} ({mapside / npx:.4f}) xtra={xtra} "
                  f"sub={sub} ({sub / npx:.4f}) T={tbits} ({tbits / npx:.4f})",
                  flush=True)
            for nm in "HGR":
                b, e, n = agg[nm]
                if n:
                    print(f"    be={nm}: N={n} bits={b} ({b / npx:.4f}) "
                          f"ent={e:.0f} gap={b - e:.0f} ({(b - e) / npx:.4f})",
                          flush=True)
            det.sort(key=lambda r_: -r_[6])
            print("    top-5 gaps (N,A,expert,be,cost,ent,gap):",
                  [(d[0], d[1], d[2], d[3], int(d[4]), int(d[5]), int(d[6]))
                   for d in det[:5]], flush=True)
    print("", flush=True)


if __name__ == "__main__":
    t0 = time.time()
    for f in ["kodim05.png", "kodim13.png", "kodim01.png"]:
        autopsy(os.path.join(D, f))
    print(f"TOTAL {(time.time() - t0) / 60:.1f} min")
