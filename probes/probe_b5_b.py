"""probe_b5_b: per-group backend choice — Golomb-Rice vs Huffman (E=6 fixed).
Exact bits: Huff = sum(count*len)+16+A*24; Golomb = sum((M>>k)+1+k) best k 0..12, +4b k side, +1b choice/group.
Clusters on MED flip meanabs (same as CROWN), per-group joint best (expert x backend).
numpy+PIL only. No existing files modified.
"""
import sys, math
import numpy as np
from PIL import Image
sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
import probe_b4_a as A
from probe_b4_g import prep, KSET2
from probe_b4_f import MOE6

IMAGES = [
    "/tmp/opencode/autocompress/experiments/real_photos/kodim01.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim02.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim05.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim07.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim13.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim19.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim23.png",
]
HEADER = A.HEADER

def golomb_best(g):
    """Exact Rice bits for int array g. Returns (data_bits, best_k). Mapping: M=2v (v>=0) else -2v-1."""
    g = np.asarray(g).reshape(-1).astype(np.int64)
    N = g.size
    if N == 0:
        return 0, 0
    M = np.where(g >= 0, 2*g, -2*g-1).astype(np.int64)
    best = None
    for k in range(13):
        tot = int(np.sum(M >> np.int64(k)) + N*(1+k))
        if best is None or tot < best[0]:
            best = (tot, k)
    return best

def huff_total(g):
    uv, cn = np.unique(g, return_counts=True)
    d = A.huff_bits(cn.tolist())
    tb = d + 16 + len(cn)*24
    return tb, d, len(cn)

def eval_img_backend(path, mode="huff"):
    """mode: huff (CROWN baseline) | mix2 (huff vs golomb, 1b choice)"""
    img = np.array(Image.open(path).convert("RGB"))
    H, W, _ = img.shape; dn = H*W*3
    Y, Co, Cg = A.ycocg_fwd(img); chs = [Y, Co, Cg]
    DD = [prep(ch) for ch in chs]
    t = HEADER; detail = []
    for D, kch in zip(DD, chs):
        P = D["P"]; key = D["key"]; s = D["s"]
        # precompute flipped residuals per expert
        RF = {}
        for n in MOE6:
            r = (kch - P[n]).astype(np.int32)
            RF[n] = (s*r).astype(np.int32)
        rfM = RF["MED"]
        bestK = None
        for K in KSET2:
            uk, cn = np.unique(key, return_counts=True)
            ma = np.array([np.abs(rfM[key==v]).mean() for v in uk])
            order = np.argsort(ma, kind="stable")
            uks = uk[order]; cns = cn[order]
            tot = cns.sum(); tgt = tot/K; groups = []; cur = []; acc = 0
            for u, c in zip(uks, cns):
                cur.append(u); acc += c
                if acc >= tgt and len(groups) < K-1:
                    groups.append(cur); cur = []; acc = 0
            groups.append(cur)
            mapbits = len(uk)*math.ceil(math.log2(K)) + 4 + 8
            if mode == "huff":
                tt = mapbits + len(groups)*3
                for gkeys in groups:
                    sel = np.isin(key, np.array(gkeys))
                    be = None
                    for n in MOE6:
                        g = RF[n][sel]
                        if g.size == 0: d_ = 0
                        else: d_ = A.huff_bits(np.unique(g, return_counts=True)[1].tolist())
                        if be is None or d_ < be[0]: be = (d_, n)
                    # table of winner
                    g = RF[be[1]][sel]
                    if g.size: tb, _, _ = A.stream_bits(g.reshape(-1)); tt += tb
            elif mode == "mix2":
                tt = mapbits + len(groups)*(3+1)  # 3b predid + 1b backend choice
                for gkeys in groups:
                    sel = np.isin(key, np.array(gkeys))
                    be = None
                    for n in MOE6:
                        g = RF[n][sel]
                        if g.size == 0:
                            ht = 0; gt = 0; gk = 0
                        else:
                            ht, _, _ = huff_total(g.reshape(-1))
                            gd, gk = golomb_best(g)
                            gt = gd + 4  # +k side
                        # joint pick (table/k incl, choice flag already in tt)
                        if be is None or ht < be[0]: be = (ht, n, "H", gk)
                        if be is None or gt < be[0]: be = (gt, n, "G", gk)
                        # careful: need joint over (n,backend); do explicit:
                    # redo joint properly:
                    best = None
                    for n in MOE6:
                        g = RF[n][sel]
                        if g.size == 0:
                            ht = 0; gt = 0; gk = 0
                        else:
                            ht, _, _ = huff_total(g.reshape(-1))
                            gd, gk2 = golomb_best(g)
                            gt = gd + 4
                        # huff option
                        if best is None or ht < best[0]: best = (ht, n, "H", gk2 if g.size else 0)
                        if best is None or gt < best[0]: best = (gt, n, "G", gk2 if g.size else 0)
                    tt += best[0]
            else:
                raise ValueError(mode)
            if bestK is None or tt < bestK[0]:
                bestK = (tt, K, groups)
        t += bestK[0]
        detail.append((bestK[1], len(bestK[2])))
    return t/dn, detail

def eval_img_breakdown(path):
    """For analysis: count how many groups prefer Golomb, bits saved, alphabet sizes."""
    img = np.array(Image.open(path).convert("RGB"))
    H, W, _ = img.shape
    Y, Co, Cg = A.ycocg_fwd(img); chs = [Y, Co, Cg]
    DD = [prep(ch) for ch in chs]
    stats = []
    for D, kch in zip(DD, chs):
        P = D["P"]; key = D["key"]; s = D["s"]
        RF = {}
        for n in MOE6:
            RF[n] = (s*(kch-P[n]).astype(np.int32)).astype(np.int32)
        rfM = RF["MED"]
        # pick best K under mix2 (reuse eval logic: find bestK)
        bestK = None; bestGroups = None
        for K in KSET2:
            uk, cn = np.unique(key, return_counts=True)
            ma = np.array([np.abs(rfM[key==v]).mean() for v in uk])
            order = np.argsort(ma, kind="stable")
            uks = uk[order]; cns = cn[order]
            tot = cns.sum(); tgt = tot/K; groups = []; cur = []; acc = 0
            for u, c in zip(uks, cns):
                cur.append(u); acc += c
                if acc >= tgt and len(groups) < K-1:
                    groups.append(cur); cur = []; acc = 0
            groups.append(cur)
            mapbits = len(uk)*math.ceil(math.log2(K)) + 4 + 8
            tt = mapbits + len(groups)*4
            for gkeys in groups:
                sel = np.isin(key, np.array(gkeys))
                best = None
                for n in MOE6:
                    g = RF[n][sel]
                    if g.size == 0: ht = 0; gt = 0; gk = 0
                    else:
                        ht, _, _ = huff_total(g.reshape(-1)); gd, gk = golomb_best(g); gt = gd+4
                    if best is None or ht < best[0]: best = (ht, n, "H", gk)
                    if gt < best[0]: best = (gt, n, "G", gk)
                tt += best[0]
            if bestK is None or tt < bestK: bestK = tt; bestGroups = (K, groups)
        K, groups = bestGroups
        for gkeys in groups:
            sel = np.isin(key, np.array(gkeys))
            # best predictor by huff data (for reference) then backend
            be = None
            for n in MOE6:
                g = RF[n][sel]
                d_ = A.huff_bits(np.unique(g, return_counts=True)[1].tolist()) if g.size else 0
                if be is None or d_ < be[0]: be = (d_, n)
            bn = be[1]
            g = RF[bn][sel]
            if g.size == 0: continue
            ht, _, Ah = huff_total(g.reshape(-1)); gd, gk = golomb_best(g); gt = gd+4
            win = "G" if gt < ht else "H"
            stats.append(dict(n=g.size, A=Ah, k=gk, ht=ht, gt=gt, win=win))
    return stats

def main():
    import time
    for mode in ("huff", "mix2"):
        t0 = time.time()
        tot = 0; dnt = 0
        print(f"== mode {mode} ==")
        for path in IMAGES:
            nm = path.split("/")[-1]
            bpp, det = eval_img_backend(path, mode)
            img = np.array(Image.open(path).convert("RGB")); H, W, _ = img.shape; d = H*W*3
            dnt += d; tot += bpp*d
            print(f"{mode} {nm}: {bpp:.4f} groups={det}", flush=True)
        print(f"==> {mode} AVG={tot/dnt:.4f} [{time.time()-t0:.0f}s]", flush=True)
    # breakdown for mix2
    print("== mix2 breakdown (groups preferring Golomb) ==")
    for path in IMAGES:
        nm = path.split("/")[-1]
        st = eval_img_breakdown(path)
        ng = len(st); nG = sum(1 for x in st if x["win"]=="G")
        save = sum(x["ht"]-x["gt"] for x in st if x["win"]=="G")
        cost = sum(x["gt"]-x["ht"] for x in st if x["win"]=="H")
        # alphabet of G-winners
        import numpy as np2
        Alves = sorted([x["A"] for x in st if x["win"]=="G"])[:10]
        print(f"{nm}: groups={ng} Golomb-wins={nG} save={save}b extra_on_H={cost}b tinyA_sample={Alves}", flush=True)

if __name__ == "__main__":
    main()
