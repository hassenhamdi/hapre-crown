"""probe_b18_spatial.py — B18 S4/S5 spatial-coherence grouping screen.

Hypothesis H-SPATIAL: key-scattered quantile groups mix same-energy pixels
from different texture regions (nonstationary statistics); JXL's spatial
groups (256x256 + per-group histograms) separate them. Spatial cell id is
FREE (decoder knows coordinates) — only key->subgroup map is transmitted.

Configs (fixed MED sign-flipped expert, per-channel, K-ledger honest):
  KEY12  : global LOCO-key quantile K=12 (baseline)
  SPC4   : 2x2 cells x per-cell quantile Kc=3  (12 groups; per-cell maps)
  SPC4s  : 2x2 cells x SHARED key->sub map (Kc=3 subs; 12 histograms; 1 map)
  SPC16  : 4x4 cells x per-cell quantile Kc=3  (48 groups; per-cell maps)
  ONE    : K=1 whole-channel single group (table-overhead floor probe)
Exact Huffman data + 16+A*24 tables + map ledger + 64b header.
numpy+PIL only. bpp=bits/(H*W*3).
"""
import math
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
import probe_b4_a as A
from probe_b4_b import nbhd as nbhd_base
from probe_b4_c import loco_ctx365
from probe_b17_rctw import rct_fwd, med_pred

D = "/tmp/opencode/autocompress/experiments/real_photos"
FILES = ["kodim01.png", "kodim02.png", "kodim05.png", "kodim07.png",
         "kodim13.png", "kodim19.png", "kodim23.png"]
JXL = {"kodim01.png": 3.3593, "kodim02.png": 3.0611, "kodim05.png": 3.5120,
       "kodim07.png": 2.7310, "kodim13.png": 3.9141, "kodim19.png": 3.2154,
       "kodim23.png": 2.8110}


def qgroups_counts(keys_list, rfM_list, Ks):
    """Per-cell quantile groups. Returns list (per cell) of groups."""
    out = []
    for key, rfM, K in zip(keys_list, rfM_list, Ks):
        uk, cn = np.unique(key, return_counts=True)
        ma = np.array([np.abs(rfM[key == v]).mean() for v in uk])
        order = np.argsort(ma, kind="stable")
        uks, cns = uk[order], cn[order]
        tot = cns.sum()
        tgt = tot / K
        groups, cur, acc = [], [], 0
        for u, c in zip(uks, cns):
            cur.append(u)
            acc += c
            if acc >= tgt and len(groups) < K - 1:
                groups.append(cur)
                cur, acc = [], 0
        groups.append(cur)
        out.append((groups, uk))
    return out


def gcost(g):
    if g.size == 0:
        return 0
    _, cn = np.unique(g, return_counts=True)
    return A.huff_bits(cn.tolist()) + 16 + len(cn) * 24


def cell_id_map(H, W, ncy, ncx):
    m = np.zeros((H, W), dtype=np.int32)
    for i in range(ncy):
        for j in range(ncx):
            m[i * H // ncy:(i + 1) * H // ncy,
              j * W // ncx:(j + 1) * W // ncx] = i * ncx + j
    return m


def eval_cfg(px, mode):
    yc = rct_fwd(px, 3, 6)
    H, W, _ = px.shape
    total = 64
    for ci in range(3):
        ch = yc[:, :, ci]
        a, b, c, d, Ww, NNe, NE = nbhd_base(ch)
        key0, s = loco_ctx365(a, b, c, d)
        R = (s * (ch - med_pred(ch)).astype(np.int32)).astype(np.int32)
        rfM = np.abs(R)
        if mode == "KEY12":
            q = qgroups_counts([key0], [rfM], [12])[0]
            groups, uk = q
            total += 16 + 736 + len(uk) * 4
            for gkeys in groups:
                total += gcost(R[np.isin(key0, np.array(gkeys))])
        elif mode == "ONE":
            total += 16 + 8  # trivial header, no map
            total += gcost(R)
        elif mode in ("SPC4", "SPC16"):
            ncy = ncx = 2 if mode == "SPC4" else 4
            cm = cell_id_map(H, W, ncy, ncx)
            nc = ncy * ncx
            Kc = 3
            per = []
            for cc in range(nc):
                m = (cm == cc)
                per.append((key0[m], rfM[m]))
            q = qgroups_counts([p[0] for p in per], [p[1] for p in per],
                               [Kc] * nc)
            for cc in range(nc):
                groups, uk = q[cc]
                m = (cm == cc)
                Rc = R[m]
                kc = key0[m]
                total += 16 + 736 + len(uk) * 2  # per-cell map (Kc=3 -> 2 bits)
                for gkeys in groups:
                    total += gcost(Rc[np.isin(kc, np.array(gkeys))])
        elif mode == "SPC4s":
            # shared key->sub map (3 subs from GLOBAL quantile), per-cell histograms
            g = qgroups_counts([key0], [rfM], [3])[0][0]
            sub = np.zeros_like(key0, dtype=np.int32)
            for si, gkeys in enumerate(g):
                sub[np.isin(key0, np.array(gkeys))] = si
            uk = np.unique(key0)
            total += 16 + 736 + len(uk) * 2  # ONE shared map
            cm = cell_id_map(H, W, 2, 2)
            for cc in range(4):
                m = (cm == cc)
                for si in range(3):
                    total += gcost(R[m & (sub == si)])
        else:
            raise ValueError(mode)
    return total


def main():
    t0 = time.time()
    modes = ["KEY12", "ONE", "SPC4", "SPC4s", "SPC16"]
    R = {m: {} for m in modes}
    for f in FILES:
        px = np.array(Image.open(os.path.join(D, f)).convert("RGB"))
        H, W, _ = px.shape
        npx = H * W * 3
        for m in modes:
            R[m][f] = eval_cfg(px, m) / npx
        print(f"{f}: " + " ".join(f"{m}={R[m][f]:.4f}" for m in modes) +
              f" JXL={JXL[f]:.4f}", flush=True)
    print("\ndelta vs KEY12:")
    for f in FILES:
        print(f"{f}: " + " ".join(f"{m}={R[m][f] - R['KEY12'][f]:+.4f}" for m in modes[1:]))
    av = {m: float(np.mean(list(R[m].values()))) for m in modes}
    print("AVG:", {m: round(v, 4) for m, v in av.items()})
    print(f"TOTAL {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
