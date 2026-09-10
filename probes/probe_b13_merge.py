"""probe_b13_merge: transfer b12 greedy histogram merging onto CROWN-huff frame.

(a) CROWN-huff reproduction: LOCO-365 keys + autoK quantile (KSET2) + dp hill-climb
    refine + per-group best-of-MOE6 predictor, Huffman tables. Pinned refs:
    01:3.4340 02:3.0931 05:3.7394 07:2.8630 13:4.1010 19:3.2758 23:2.8940 (avg 3.3429).
(b) + image-wide cross-channel greedy-best-first table merging (Huffman exact costs),
    merge-map side counted (ng*ceil(log2 C) + 8b C header), gated per image vs (a).
(c) + per-merged-cluster exact H-vs-R backend choice (libhapre.so M=14, decode-asserted,
    1b choice/cluster), gated vs (b).
(c2 stretch) rANS-proxy (entropy+16+A*32) greedy partition + exact H/R upgrade.

UNIT: bpp = total_bits/(H*W*3). Exact counting everywhere.
Decoder-safety: merged tables + map transmitted; keys re-derived from causal recon only.

New file only (probe_b13_ prefix). Reuses proven modules by import.
numpy+PIL+ctypes only, CPU, no torch.
"""
import sys
import os
import math
import heapq
import time

sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
sys.path.insert(0, "/tmp/opencode/autocompress/csrc")
os.chdir("/tmp/opencode/autocompress/experiments")

import numpy as np
from PIL import Image

import probe_b4_a as A
from probe_b4_g import prep, KSET2
from probe_b4_f import MOE6
from probe_b4_h import rans_stream_bits
from dp_refine import refine

IMAGES = A.IMAGES
HEADER = A.HEADER
JXL_E3 = {
    "kodim01.png": 3.3593,
    "kodim02.png": 3.0611,
    "kodim05.png": 3.5120,
    "kodim07.png": 2.7310,
    "kodim13.png": 3.9141,
    "kodim19.png": 3.2154,
    "kodim23.png": 2.8110,
}
PIN_A = {
    "kodim01.png": 3.4340,
    "kodim02.png": 3.0931,
    "kodim05.png": 3.7394,
    "kodim07.png": 2.8630,
    "kodim13.png": 4.1010,
    "kodim19.png": 3.2758,
    "kodim23.png": 2.8940,
}


def huff_bits_list(counts):
    c = [int(x) for x in counts]
    if len(c) == 0:
        return 0
    if len(c) == 1:
        return int(c[0]) * 1
    h = c[:]
    heapq.heapify(h)
    t = 0
    while len(h) > 1:
        x = heapq.heappop(h)
        y = heapq.heappop(h)
        s = x + y
        t += s
        heapq.heappush(h, s)
    return int(t)


def entropy_bits_list(counts):
    c = np.asarray(list(counts), dtype=np.float64)
    n = c.sum()
    if n == 0 or len(c) <= 1:
        return 0.0
    p = c / n
    return float((-(p * np.log2(p))).sum() * n)


def merge_map_bits(ng, C):
    if C <= 1:
        return 0
    return int(ng * math.ceil(math.log2(C)))


def auto_groups_exact(key, rf, KSET):
    best = None
    for K in KSET:
        uk, cn = np.unique(key, return_counts=True)
        ma = np.array([np.abs(rf[key == v]).mean() for v in uk])
        order = np.argsort(ma, kind="stable")
        uks = uk[order]
        cns = cn[order]
        tot = cns.sum()
        tgt = tot / K
        groups = []
        cur = []
        acc = 0
        for u, c in zip(uks, cns):
            cur.append(u)
            acc += c
            if acc >= tgt and len(groups) < K - 1:
                groups.append(cur)
                cur = []
                acc = 0
        groups.append(cur)
        mapbits = len(uk) * math.ceil(math.log2(K)) + 4 + 8
        tt = mapbits
        for gkeys in groups:
            g = rf[np.isin(key, np.array(gkeys))]
            if g.size:
                tb, _, _ = A.stream_bits(g.reshape(-1))
                tt += tb
        if best is None or tt < best:
            best = tt
            bK = K
            bG = groups
            bmap = mapbits
    return bK, bG, best, bmap


def build_image(path):
    """Returns dict with per-channel grouping + per-group chosen residuals."""
    img = np.array(Image.open(path).convert("RGB"))
    H, W, _ = img.shape
    dn = H * W * 3
    # YCoCg-R round-trip assert
    Y, Co, Cg = A.ycocg_fwd(img)
    t = Y - (Cg // 2)
    B = t - (Co // 2)
    G = Cg + t
    R = Co + B
    assert np.array_equal(R, img[:, :, 0].astype(np.int32))
    assert np.array_equal(G, img[:, :, 1].astype(np.int32))
    assert np.array_equal(B, img[:, :, 2].astype(np.int32))
    chs = [Y, Co, Cg]
    DD = [prep(ch) for ch in chs]
    channels = []
    total_a = HEADER
    atoms = []  # image-wide merge atoms
    for ci, (D, kch) in enumerate(zip(DD, chs)):
        P = D["P"]
        key = D["key"]
        s = D["s"]
        resM = (kch - P["MED"]).astype(np.int32)
        rfM = (s * resM).astype(np.int32)
        bK, G, _, bmap = auto_groups_exact(key, rfM, KSET2)
        G, _ = refine(key, rfM, G, bK)
        # per-group best predictor among MOE6 by exact Huffman data bits
        gpred = []
        gvals = []
        gdata = []
        gtab = []
        gA = []
        for gkeys in G:
            sel = np.isin(key, np.array(gkeys))
            best = None
            bn = MOE6[0]
            for n in MOE6:
                r = (kch - P[n]).astype(np.int32)
                rf = (s * r).astype(np.int32)
                g = rf[sel]
                d_ = huff_bits_list(np.unique(g, return_counts=True)[1].tolist()) if g.size else 0
                if best is None or d_ < best:
                    best = d_
                    bn = n
            gpred.append(bn)
            r = (kch - P[bn]).astype(np.int32)
            rf = (s * r).astype(np.int32)
            g = rf[sel]
            gvals.append(np.ascontiguousarray(g.reshape(-1).astype(np.int32)))
            if g.size:
                uv, cn = np.unique(g, return_counts=True)
                d = huff_bits_list(cn.tolist())
                a = len(cn)
            else:
                d, a = 0, 0
            gdata.append(d)
            gA.append(a)
            gtab.append(16 + a * 24 if g.size else 0)
        ch_total = bmap + len(G) * 3 + sum(d + t_ for d, t_ in zip(gdata, gtab))
        total_a += ch_total
        uk = np.unique(key)
        channels.append({
            "K": bK, "G": G, "bmap": bmap, "gpred": gpred, "gvals": gvals,
            "gdata": gdata, "gtab": gtab, "gA": gA, "nactive": len(uk),
            "key": key, "s": s, "P": P, "ch": kch,
        })
        base = len(atoms)
        for gi in range(len(G)):
            uv, cn = np.unique(gvals[gi], return_counts=True)
            cnt = {int(v): int(c) for v, c in zip(uv.tolist(), cn.tolist())} if gvals[gi].size else {}
            atoms.append({
                "ch": ci, "gi": gi, "pred": gpred[gi],
                "vals": gvals[gi], "cnt": cnt,
                "data": gdata[gi], "tab": gtab[gi],
            })
    return {"img": img, "dn": dn, "H": H, "W": W, "channels": channels,
            "total_a": total_a, "atoms": atoms, "path": path}


def greedy_merge_atoms(atoms, backend="huff"):
    """Greedy-best-first agglomeration over atoms (image-wide, cross-channel).

    backend huff: data=huff_bits, tab=16+A*24. rans-proxy: data=entropy, tab=16+A*32.
    Map side: merge_map_bits(ng,C)+8b C header counted in totals; pair gains use map delta.
    Returns dict(total with map+8, C, part (list of atom-idx lists), gains, nmerge, clusters).
    total here = merged tables+data + map + 8 (EXCLUDES base sides; caller adds them).
    """
    ng = len(atoms)
    if ng == 0:
        return {"total": 8, "C": 0, "part": [], "gains": [], "nmerge": 0, "clusters": {}}

    def cost_of(cnt):
        if len(cnt) == 0:
            return 0, 0, (16 if backend == "huff" else 16)
        vals = list(cnt.values())
        if backend == "huff":
            d = huff_bits_list(vals)
            a = len(vals)
            return d, a, 16 + a * 24
        else:
            d = entropy_bits_list(vals)
            a = len(vals)
            return d, a, 16 + a * 32

    cl = {}
    for i, at in enumerate(atoms):
        d, a, tb = cost_of(at["cnt"])
        cl[i] = {"members": [i], "cnt": dict(at["cnt"]), "data": d, "tab": tb}
    alive = set(cl.keys())
    nxt = ng

    def tot_now():
        C = len(alive)
        s = sum(cl[i]["data"] + cl[i]["tab"] for i in alive)
        return s + merge_map_bits(ng, C) + 8, C

    def merge_dicts(a, b):
        if len(a) < len(b):
            a, b = b, a
        m = dict(a)
        for k, v in b.items():
            m[k] = m.get(k, 0) + v
        return m

    def pair_gain(i, j, C):
        mcnt = merge_dicts(cl[i]["cnt"], cl[j]["cnt"])
        mdata, _a, mtab = cost_of(mcnt)
        gain = (cl[i]["data"] + cl[i]["tab"] + cl[j]["data"] + cl[j]["tab"]
                + merge_map_bits(ng, C)) - (mdata + mtab + merge_map_bits(ng, C - 1))
        return gain, mdata, mtab, mcnt

    C = len(alive)
    ids = sorted(alive)
    cache = {}
    for x in range(len(ids)):
        for y in range(x + 1, len(ids)):
            i, j = ids[x], ids[y]
            cache[(i, j)] = pair_gain(i, j, C)
    gains = []
    nmerge = 0
    while True:
        C = len(alive)
        if C <= 1:
            break
        best_k, best_g = None, 0.0
        for k, v in cache.items():
            if k[0] in alive and k[1] in alive and v[0] > best_g:
                best_g, best_k = v[0], k
        if best_k is None or best_g <= 0.0:
            break
        i, j = best_k
        _g, mdata, mtab, mcnt = cache[best_k]
        alive.discard(i)
        alive.discard(j)
        for k in [k for k in cache if k[0] in (i, j) or k[1] in (i, j)]:
            del cache[k]
        cl[nxt] = {"members": cl[i]["members"] + cl[j]["members"],
                   "cnt": mcnt, "data": mdata, "tab": mtab}
        alive.add(nxt)
        C2 = len(alive)
        for o in alive:
            if o == nxt:
                continue
            a1, b1 = (o, nxt) if o < nxt else (nxt, o)
            cache[(a1, b1)] = pair_gain(a1, b1, C2)
        gains.append(float(best_g))
        nxt += 1
        nmerge += 1
    tot, Cfin = tot_now()
    part = [cl[i]["members"] for i in sorted(alive)]
    return {"total": tot, "C": Cfin, "part": part, "gains": gains,
            "nmerge": nmerge, "clusters": {i: cl[i] for i in alive}}


def huffman_lengths(sym_counts):
    syms = list(sym_counts.keys())
    if len(syms) == 0:
        return {}
    if len(syms) == 1:
        return {syms[0]: 1}
    import itertools
    ctr = itertools.count()
    h = [(int(sym_counts[s]), next(ctr), ("leaf", s)) for s in syms]
    heapq.heapify(h)
    while len(h) > 1:
        c1, _, n1 = heapq.heappop(h)
        c2, _, n2 = heapq.heappop(h)
        heapq.heappush(h, (c1 + c2, next(ctr), ("in", n1, n2)))
    root = h[0][2]
    lengths = {}

    def walk(node, d):
        if node[0] == "leaf":
            lengths[node[1]] = max(1, d)
        else:
            walk(node[1], d + 1)
            walk(node[2], d + 1)
    walk(root, 0)
    return lengths


def main():
    t_all = time.time()
    print("== probe_b13 merge-transfer onto CROWN-huff frame ==")
    rows = []
    sumA = sumB = sumC = sumC2 = 0
    sumDn = 0
    tot_ng = tot_Cb = tot_Cc = 0
    tot_merges_b = 0
    gain_b_sum = 0.0
    detail = {}
    for path in IMAGES:
        nm = path.split("/")[-1]
        bi = build_image(path)
        dn = bi["dn"]
        atoms = bi["atoms"]
        ng = len(atoms)
        # ---- (a) ----
        tA = bi["total_a"]
        bppA = tA / dn
        pin = PIN_A[nm]
        ok = abs(bppA - pin) / pin <= 0.01
        print(f"  (a) {nm}: repro={bppA:.4f} pin={pin:.4f} "
              f"d={bppA - pin:+.4f} [{('PASS' if ok else 'FAIL')}] ng={ng} "
              f"K={[c['K'] for c in bi['channels']]} "
              f"nG={[len(c['G']) for c in bi['channels']]}", flush=True)
        # base sides (everything except per-group tables+data)
        base_sides = HEADER + sum(c["bmap"] + len(c["G"]) * 3 for c in bi["channels"])
        unmerged_td = sum(c["gdata"][i] + c["gtab"][i]
                          for c in bi["channels"] for i in range(len(c["G"])))
        assert tA == base_sides + unmerged_td
        # ---- (b) huffman greedy ----
        g = greedy_merge_atoms(atoms, backend="huff")
        merged_total_b = base_sides + g["total"]
        if merged_total_b < tA:
            tB = merged_total_b
            Cb = g["C"]
            part_b = g["part"]
            nb_merge = g["nmerge"]
            gains_b = g["gains"]
            used_merge_b = True
        else:
            tB = tA
            Cb = ng
            part_b = [[i] for i in range(ng)]
            nb_merge = 0
            gains_b = []
            used_merge_b = False
        bppB = tB / dn
        # merged data/table/map split for (b)
        if used_merge_b:
            mdata_b = sum(g["clusters"][i]["data"] for i in g["clusters"])
            mtab_b = sum(g["clusters"][i]["tab"] for i in g["clusters"])
            mmap_b = merge_map_bits(ng, Cb) + 8
        else:
            mdata_b = sum(a["data"] for a in atoms)
            mtab_b = sum(a["tab"] for a in atoms)
            mmap_b = 0
        # ---- (c) per-merged-cluster exact H-vs-R (on (b) partition) ----
        # build vals per cluster in partition part_b
        cl_vals = []
        for members in part_b:
            if len(members) == 1:
                cl_vals.append(atoms[members[0]]["vals"])
            else:
                cl_vals.append(np.concatenate([atoms[m]["vals"] for m in members]))
        ht_list, rt_list = [], []
        nR = 0
        for v in cl_vals:
            if v.size == 0:
                ht_list.append(0)
                rt_list.append(None)
                continue
            uv, cn = np.unique(v, return_counts=True)
            ht = huff_bits_list(cn.tolist()) + 16 + len(cn) * 24
            ht_list.append(ht)
            rt, _ = rans_stream_bits(v)
            rt_list.append(rt)
        # choose per cluster (both pay 1b choice) then gate overall vs (b)
        Cc = Cb
        upgrade = sum(min(h, r) for h, r in zip(ht_list, rt_list)) if cl_vals else 0
        tC_cand = base_sides + mmap_b + Cc * 1 + upgrade
        if tC_cand < tB:
            tC = tC_cand
            picks = ["R" if r < h else "H" for h, r in zip(ht_list, rt_list)]
            nR = sum(1 for p in picks if p == "R")
        else:
            tC = tB
            picks = ["H"] * len(cl_vals)
            nR = 0
        bppC = tC / dn
        # ---- (c2 stretch) rans-proxy partition + exact H/R ----
        g2 = greedy_merge_atoms(atoms, backend="rans")
        merged_proxy_total = base_sides + g2["total"]  # proxy units, not comparable; recompute exact below
        # exact-encode g2 partition with H/R choice
        part2 = g2["part"] if (True) else [[i] for i in range(ng)]
        cl_vals2 = []
        for members in part2:
            if len(members) == 1:
                cl_vals2.append(atoms[members[0]]["vals"])
            else:
                cl_vals2.append(np.concatenate([atoms[m]["vals"] for m in members]))
        ht2, rt2 = [], []
        for v in cl_vals2:
            if v.size == 0:
                ht2.append(0)
                rt2.append(0)
                continue
            uv, cn = np.unique(v, return_counts=True)
            ht2.append(huff_bits_list(cn.tolist()) + 16 + len(cn) * 24)
            rtb, _ = rans_stream_bits(v)
            rt2.append(rtb)
        C2 = g2["C"]
        mmap2 = merge_map_bits(ng, C2) + 8
        tC2_cand = base_sides + mmap2 + C2 * 1 + sum(min(h, r) for h, r in zip(ht2, rt2))
        # gate vs (b): only keep proxy partition if its exact cost beats (b); else fall back to (c)
        if tC2_cand < tC:
            tC2 = tC2_cand
            C2used = C2
            part2used = part2
            note2 = "proxy-wins"
        else:
            tC2 = tC
            C2used = Cc
            part2used = part_b
            note2 = "huff-part-wins"
        bppC2 = tC2 / dn
        rows.append((nm, dn, tA, tB, tC, tC2, ng, Cb, Cc, C2used, nb_merge, nR,
                     sum(gains_b), mdata_b, mtab_b, mmap_b, picks.count("R") if isinstance(picks, list) else 0))
        sumA += tA
        sumB += tB
        sumC += tC
        sumC2 += tC2
        sumDn += dn
        tot_ng += ng
        tot_Cb += Cb
        tot_Cc += Cc
        tot_merges_b += nb_merge
        gain_b_sum += sum(gains_b)
        detail[nm] = {"part_b": part_b, "picks": picks, "bi": bi, "atoms": atoms,
                      "used_merge": used_merge_b, "g": g, "part2": part2used,
                      "note2": note2}
        print(f"  (b) {nm}: bpp={bppB:.4f} dA={bppB - bppA:+.4f} "
              f"({100 * (tB / tA - 1):+.2f}%) tab {ng}->{Cb} m={nb_merge} "
              f"{'MERGED' if used_merge_b else 'NOMERGE'}", flush=True)
        print(f"  (c) {nm}: bpp={bppC:.4f} dA={bppC - bppA:+.4f} dB={bppC - bppB:+.4f} "
              f"R={nR}/{len(cl_vals)} "
              f"(c2 {note2}: bpp={bppC2:.4f} C={C2used})", flush=True)
    avgA, avgB, avgC, avgC2 = sumA / sumDn, sumB / sumDn, sumC / sumDn, sumC2 / sumDn
    print("=" * 100)
    print(f"AVG (a)={avgA:.4f} (pin 3.3429 d={avgA - 3.3429:+.4f}) "
          f"(b)={avgB:.4f} dA={avgB - avgA:+.4f} ({100 * (avgB / avgA - 1):+.2f}%) "
          f"(c)={avgC:.4f} dA={avgC - avgA:+.4f} dB={avgC - avgB:+.4f} "
          f"(c2)={avgC2:.4f}")
    print(f"Tables/image: {tot_ng / 7:.1f} -> (b) {tot_Cb / 7:.1f} (saved {tot_ng - tot_Cb}, "
          f"{tot_merges_b} merges); avg net gain {(gain_b_sum / max(1, tot_merges_b)):.0f} bits/merge")
    # ---- round-trip checks on winning config (c if beats b else b), ALL-7 symbol + YCoCg ----
    print("-- round-trip: canonical symbol ALL-7 on (b) winner maps + YCoCg inverse --")
    ok_all = True
    for path in IMAGES:
        nm = path.split("/")[-1]
        d = detail[nm]
        atoms = d["atoms"]
        part = d["part_b"]
        flat = sorted([m for grp in part for m in grp])
        ok = (flat == list(range(len(atoms))))
        for members in part:
            if len(members) == 1:
                cnt = atoms[members[0]]["cnt"]
            else:
                cnt = {}
                for m in members:
                    for k, v in atoms[m]["cnt"].items():
                        cnt[k] = cnt.get(k, 0) + v
            if len(cnt) == 0:
                continue
            L = huffman_lengths(cnt)
            kr = sum(2.0 ** (-L[s]) for s in L)
            assert abs(kr - 1.0) < 1e-9 or len(L) == 1, (nm, kr)
            # symbol round-trip spot: encode/decode each distinct symbol once + full multiset counts
            enc_order = sorted(cnt.keys(), key=lambda s: (L[s], int(s)))
            enc, dec = {}, {}
            code, prev = 0, 0
            for s in enc_order:
                ln = L[s]
                code <<= (ln - prev)
                enc[s] = (code, ln)
                dec[(code, ln)] = s
                code += 1
                prev = ln
            for s in enc_order:
                cd, ln = enc[s]
                assert dec[(cd, ln)] == s
        # YCoCg inverse already asserted in build; re-assert here
        img = np.array(Image.open(path).convert("RGB"))
        Y, Co, Cg = A.ycocg_fwd(img)
        t = Y - (Cg // 2)
        B = t - (Co // 2)
        G = Cg + t
        R = Co + B
        assert np.array_equal(R, img[:, :, 0].astype(np.int32))
        print(f"  {nm}: partition-cover {'PASS' if ok else 'FAIL'} Kraft+symbol PASS", flush=True)
        ok_all = ok_all and ok
    # literal bitstream on kodim01 first merged cluster (winning (b) partition)
    pick = IMAGES[0]
    nm0 = pick.split("/")[-1]
    d0 = detail[nm0]
    atoms0 = d0["atoms"]
    mem0 = d0["part_b"][0]
    if len(mem0) == 1:
        cnt0 = atoms0[mem0[0]]["cnt"]
        vals0 = atoms0[mem0[0]]["vals"].tolist()
    else:
        cnt0 = {}
        for m in mem0:
            for k, v in atoms0[m]["cnt"].items():
                cnt0[k] = cnt0.get(k, 0) + v
        vals0 = np.concatenate([atoms0[m]["vals"] for m in mem0]).tolist()
    L0 = huffman_lengths(cnt0)
    enc_order = sorted(cnt0.keys(), key=lambda s: (L0[s], int(s)))
    enc, dec = {}, {}
    code, prev = 0, 0
    for s in enc_order:
        ln = L0[s]
        code <<= (ln - prev)
        enc[s] = (code, ln)
        dec[(code, ln)] = s
        code += 1
        prev = ln
    buf, nb, out = 0, 0, bytearray()
    for s in vals0:
        cd, ln = enc[int(s)]
        buf = (buf << ln) | cd
        nb += ln
        while nb >= 8:
            nb -= 8
            out.append((buf >> nb) & 0xFF)
    if nb:
        out.append((buf << (8 - nb)) & 0xFF)
    bits = np.unpackbits(np.frombuffer(bytes(out), dtype=np.uint8))
    acc, al, got = 0, 0, []
    for b in bits.tolist():
        acc = (acc << 1) | b
        al += 1
        if (acc, al) in dec:
            got.append(dec[(acc, al)])
            acc, al = 0, 0
            if len(got) == len(vals0):
                break
    assert got == [int(s) for s in vals0], "bitstream mismatch kodim01 cl0"
    print(f"  {nm0}: literal-bitstream PASS (cl0: {len(vals0)} syms, "
          f"{sum(L0[s] for s in vals0)} bits)", flush=True)
    print(f"TOTAL time {time.time() - t_all:.1f}s; partition-cover ALL-7 "
          f"{'PASS' if ok_all else 'FAIL'}; rANS decodes asserted per R-cluster in (c)/(c2)")
    # stash summary
    with open("/tmp/opencode/autocompress/experiments/probe_b13_summary.txt", "w") as f:
        f.write(f"AVGA={avgA:.4f} AVGB={avgB:.4f} AVGC={avgC:.4f} AVGC2={avgC2:.4f}\n")
        for r in rows:
            f.write("ROW " + " ".join([r[0]] + [f"{x:.4f}" if isinstance(x, float) else str(x) for x in
                                                 [r[2] / r[1], r[3] / r[1], r[4] / r[1], r[5] / r[1]]]
                                + [str(int(x)) for x in r[6:]]) + "\n")


if __name__ == "__main__":
    main()
