"""probe_b18_group.py — B18 GROUPED screen (new file only).

CROWN3-frame approximation: per-channel LOCO-365 quantile groups (K=12,
map side counted) + per-group best-of-bank by EXACT Huffman data bits
+ honest ceil(log2 B) predid bits/group + real tables per group.
Compares bank E16 (CROWN3's spatial experts, WAVG excluded here) vs
E16 + lag/directional/cross-channel experts. All side counted.
bpp = total_bits/(H*W*3). numpy+PIL only.
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
from probe_b5_c import predictors_X
from probe_b18_screen import nbhd_big, diag_long

D = "/tmp/opencode/autocompress/experiments/real_photos"
FILES = ["kodim01.png", "kodim02.png", "kodim05.png", "kodim07.png",
         "kodim13.png", "kodim19.png", "kodim23.png"]
JXL = {"kodim01.png": 3.3593, "kodim02.png": 3.0611, "kodim05.png": 3.5120,
       "kodim07.png": 2.7310, "kodim13.png": 3.9141, "kodim19.png": 3.2154,
       "kodim23.png": 2.8110}
E16 = ["MED", "TOP", "LEFT", "PAETH", "GRAD", "GAP80", "DG", "AVG_AB", "PLANE",
       "GAP32", "AC", "C", "BC", "D", "GAP16", "AVG3"]
K = 12


def quantile_groups(key, rfM, K):
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
    return groups


def group_bits(g):
    if g.size == 0:
        return 0
    _, cn = np.unique(g, return_counts=True)
    return A.huff_bits(cn.tolist()) + 16 + len(cn) * 24


def eval_bank_on_image(px, bank_extra_fn=None, K=K, verbose_picks=False):
    """bank_extra_fn(ch, N, Yplane) -> dict name->pred-plane. Returns total bits + picks."""
    from probe_b17_rctw import rct_fwd
    yc = rct_fwd(px, 3, 6)  # C27
    H, W, _ = px.shape
    total = 64
    picks = {}
    for ci in range(3):
        ch = yc[:, :, ci]
        a, b, c, d, Ww, NNe, NE = nbhd_base(ch)
        P = predictors_X(a, b, c, d, Ww, NNe, NE)
        # sign-flip convention from b5: RF = s*(ch-P)
        from probe_b4_c import loco_ctx365
        key, s = loco_ctx365(a, b, c, d)
        RF = {n: (s * (ch - P[n]).astype(np.int32)).astype(np.int32) for n in E16}
        bank = list(E16)
        if bank_extra_fn is not None:
            Yp = yc[:, :, 0]
            X = bank_extra_fn(ch, nbhd_big(ch), Yp, ci, s)
            for n, pred in X.items():
                RF[n] = (s * (ch - pred).astype(np.int32)).astype(np.int32)
                bank.append(n)
        rfM = RF["MED"]
        groups = quantile_groups(key, rfM, K)
        uk = np.unique(key)
        gbits = math.ceil(math.log2(K))
        total += 16 + 736 + len(uk) * gbits  # map side (driver_crown3 ledger)
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
            g = RF[be[1]][sel]
            total += group_bits(g)
            picks[be[1]] = picks.get(be[1], 0) + 1
    return total, picks


def lag_experts(ch, N, Yp, ci, s):
    L, T = N["L"], N["T"]
    out = {}
    for k in [2, 3, 4, 6, 8, 12, 16, 24]:
        if k in T:
            out[f"WN{k}"] = T[k]
    for k in [2, 3, 4, 6, 8]:
        if k in L:
            out[f"WL{k}"] = L[k]
    out["HAVG4"] = (N["a"] + L[2] + L[3] + L[4]) // 4
    out["VAVG4"] = (N["b"] + T[2] + T[3] + T[4]) // 4
    out["XH"] = 2 * N["a"] - N["Ww"]
    out["XV"] = 2 * N["b"] - N["NNe"]
    return out


def lag_plus_xch(ch, N, Yp, ci, s):
    out = lag_experts(ch, N, Yp, ci, s)
    if ci in (1, 2):
        # per-image global-slope-removed variants as extra experts is weak;
        # instead: Y-anchored blend experts with FIXED small slopes (0 side,
        # gate selects): pred = MED(ch) + ((Y-Y_MED)*t)//16 for t in set
        from probe_b17_rctw import med_pred
        Ym = med_pred(Yp)
        dY = (Yp.astype(np.int32) - Ym).astype(np.int32)
        M = med_pred(ch)
        for t in [-4, -2, -1, 1, 2, 4]:
            out[f"YX{t}"] = M + (dY * t) // 16
    return out


def main():
    t0 = time.time()
    cfgs = [("E16", None), ("E16+LAG", lag_experts), ("E16+LAG+XCH", lag_plus_xch)]
    print(f"== B18 GROUPED screen K={K} (map+ids+tables exact) ==")
    results = {}
    for tag, fn in cfgs:
        results[tag] = {}
        tot = 0
        dn = 0
        for f in FILES:
            px = np.array(Image.open(os.path.join(D, f)).convert("RGB"))
            H, W, _ = px.shape
            bits, picks = eval_bank_on_image(px, fn)
            bpp = bits / (H * W * 3)
            results[tag][f] = bpp
            tot += bits
            dn += H * W * 3
            xp = {k: v for k, v in picks.items() if k not in E16}
            print(f"{tag} {f}: {bpp:.4f} newpicks={xp if xp else '-'}_"
                  f"JXL={JXL[f]:.4f}", flush=True)
        print(f"==> {tag} AVG={tot / dn:.4f}", flush=True)
    print("\n== deltas (bpp) vs E16-screen ==")
    print("img       " + "".join(f"{t:>16}" for t, _ in cfgs[1:]))
    for f in FILES:
        row = f"{f}: "
        for tag, _ in cfgs[1:]:
            row += f"{results[tag][f] - results['E16'][f]:+16.4f}"
        print(row + f"   JXL={JXL[f]:.4f}")
    avgs = {t: float(np.mean(list(results[t].values()))) for t, _ in cfgs}
    print("AVG: " + str({t: round(v, 4) for t, v in avgs.items()}))
    print(f"\nTOTAL {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
