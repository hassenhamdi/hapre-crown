"""probe_b12_histshare: JXL-style histogram SHARING honesty test (exploratory branch).

Question: our tables cost ~3-5% of the stream across ~60-100 per-group /
per-context distributions. Does merging similar distributions into shared
tables pay NET after counting (a) extra data bits, (b) saved tables, (c) the
cluster-map side info?

Setup (numpy+PIL only, CPU, no torch):
  - YCoCg-R + causal MED + LOCO-365 contexts (proven helpers imported from
    probe_b4_b.nbhd/predictors and probe_b4_c.loco_ctx365 — exact same code).
  - Atoms = per-channel LOCO-quantile groups (K0=24/ch -> <=72 tables/image,
    the 60-100 regime). Equal-pixel partition by per-key mean|res| (CROWN rule).
  - Rule G (main): greedy-best-first agglomerative merging, all pairs, exact
    Huffman total drives every accept (data=heapq huff bits, table=16+A*24,
    map=nkeys*ceil(log2(C)) per channel, +64b global header).
  - Rule J (second rule): pairs attempted in Jensen-Shannon-distance order
    (static order per pass, repeated passes), exact-total-gated accepts.
  - Winner per image = min(G,J) total.
  - Stretch R: same greedy pass under rANS-proxy costs (data=Shannon entropy,
    table=16+A*32, same map). Entropy proxy is optimistic ~0.5-1%; stated.
  - Baselines: U-RAW (per-key tables, no map — fragmentation showcase),
    U-GRP (quantile groups, with map — the honest delta comparator).
  - Anchors: MED order-0 Huffman (must be ~3.58 avg) + CROWN-lite anchor
    (per-group best-of-7 predictor, K=9) to explain the 3.34 reference
    (3.34 needs predictor selection; MED-only grouped sits ~3.5 — stated).
  - Round-trip: canonical-Huffman symbol round-trip ALL-7 on winner maps +
    full literal bitstream round-trip (pack+unpack+decode) on 2 representative
    images + YCoCg-R inverse assert ALL-7.

UNIT: bpp = total_bits/(H*W*3). Exact counting everywhere.
"""
import heapq
import math
import sys
import time

import numpy as np
from PIL import Image

import probe_b4_a as A
from probe_b4_b import nbhd, predictors
from probe_b4_c import loco_ctx365

IMAGES = A.IMAGES
HEADER = A.HEADER
K0 = 24  # quantile groups per channel -> <=72 tables (60-100 regime)
PNAMES = ["MED", "TOP", "LEFT", "PAETH", "GRAD", "GAP", "DG"]


# ---------------- exact cost primitives ----------------
def huff_bits(counts):
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


def entropy_bits(counts):
    c = np.asarray(list(counts), dtype=np.float64)
    n = c.sum()
    if n == 0 or len(c) <= 1:
        return 0.0
    p = c / n
    return float((-(p * np.log2(p))).sum() * n)


def map_bits(nkeys, nclust):
    if nclust <= 1:
        return 0
    return int(nkeys * math.ceil(math.log2(nclust)))


# ---------------- canonical Huffman (round-trip) ----------------
def huffman_lengths(sym_counts):
    """sym_counts: dict sym->count. Returns dict sym->length (min 1)."""
    syms = list(sym_counts.keys())
    if len(syms) == 0:
        return {}
    if len(syms) == 1:
        return {syms[0]: 1}
    heap = []
    uid = 0
    for s in syms:
        heap.append((int(sym_counts[s]), uid, [s]))
        uid += 1
    heapq.heapify(heap)
    # tree via parent tracking: repeatedly merge leaf-lists; depth = #merges covering leaf
    # simplest: build explicit tree nodes
    heap2 = []
    for s in syms:
        heap2.append([int(sym_counts[s]), uid, {"sym": s}])
        uid += 1
    heapq.heapify(heap2)
    # use counter to avoid dict comparison
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


def canonical_codes(lengths):
    """lengths: dict sym->len. Returns enc {sym:(code,len)}, dec {(code,len):sym}."""
    order = sorted(lengths.keys(), key=lambda s: (lengths[s], int(s)))
    enc, dec = {}, {}
    code, prev = 0, 0
    for s in order:
        ln = lengths[s]
        code <<= (ln - prev)
        enc[s] = (code, ln)
        dec[(code, ln)] = s
        code += 1
        prev = ln
    return enc, dec


# ---------------- per-image prep ----------------
def prep_image(path):
    img = np.array(Image.open(path).convert("RGB"))
    H, W, _ = img.shape
    dn = H * W * 3
    Y, Co, Cg = A.ycocg_fwd(img)
    # YCoCg-R inverse assert (proven round-trip)
    t = Y - (Cg // 2)
    B = t - (Co // 2)
    G = Cg + t
    R = Co + B
    assert np.array_equal(R, img[:, :, 0].astype(np.int32))
    assert np.array_equal(G, img[:, :, 1].astype(np.int32))
    assert np.array_equal(B, img[:, :, 2].astype(np.int32))
    chs = [Y, Co, Cg]
    out = []
    for ch in chs:
        a, b, c, d, Ww, NNe, NE = nbhd(ch)
        P = predictors(a, b, c, d, Ww, NNe, NE)
        med_res = (ch - P["MED"]).astype(np.int32)
        key, _sgn = loco_ctx365(a, b, c, d)
        out.append({"ch": ch, "res": med_res, "key": key, "pred": P,
                    "nbhd": (a, b, c, d)})
    return img, dn, out


def quantile_groups(res, key, K):
    uk, cn = np.unique(key, return_counts=True)
    cntmap = dict(zip([int(v) for v in uk], [int(x) for x in cn]))
    mabs = np.array([float(np.abs(res[key == v]).mean()) for v in uk])
    order = np.argsort(mabs, kind="stable")
    tot = int(cn.sum())
    tgt = tot / float(K)
    groups, cur, acc = [], [], 0
    for idx in order:
        v = int(uk[idx])
        cur.append(v)
        acc += cntmap[v]
        if acc >= tgt and len(groups) < K - 1:
            groups.append(cur)
            cur, acc = [], 0
    groups.append(cur)
    groups = [g for g in groups if len(g) > 0]
    return [int(v) for v in uk], groups  # active keys, groups of keys


def counts_of(res, key, keys):
    mask = np.isin(key, np.asarray(keys))
    g = res[mask]
    if g.size == 0:
        return {}, 0
    _, cn = np.unique(g, return_counts=True)
    syms = np.unique(g)
    return {int(s): int(x) for s, x in zip(syms.tolist(), cn.tolist())}, int(g.size)


def cluster_cost_huff(cnt):
    if len(cnt) == 0:
        return 0, 0, 16
    data = huff_bits(list(cnt.values()))
    alp = len(cnt)
    return data, alp, 16 + alp * 24


def cluster_cost_rans(cnt):
    if len(cnt) == 0:
        return 0.0, 0, 16
    data = entropy_bits(list(cnt.values()))
    alp = len(cnt)
    return data, alp, 16 + alp * 32


def merge_dicts(a, b):
    if len(a) < len(b):
        a, b = b, a
    m = dict(a)
    for k, v in b.items():
        m[k] = m.get(k, 0) + v
    return m


# ---------------- Rule G: greedy best-first (cached pairs) ----------------
def greedy_merge(nkeys, init_groups, res, key, backend="huff"):
    cost_fn = cluster_cost_huff if backend == "huff" else cluster_cost_rans
    cl = {}  # id -> {keys, cnt, data, tab}
    nxt = 0
    for g in init_groups:
        cnt, _n = counts_of(res, key, g)
        data, _alp, tab = cost_fn(cnt)
        cl[nxt] = {"keys": list(g), "cnt": cnt, "data": data, "tab": tab}
        nxt += 1
    alive = set(cl.keys())

    def cur_total():
        C = len(alive)
        s = sum(cl[i]["data"] + cl[i]["tab"] for i in alive)
        return s + map_bits(nkeys, C), C

    # pair cache: (i,j)->(gain, mdata, mtab, mcnt)
    def pair_gain(i, j, C):
        mcnt = merge_dicts(cl[i]["cnt"], cl[j]["cnt"])
        mdata, _a, mtab = cost_fn(mcnt)
        before_map = map_bits(nkeys, C)
        after_map = map_bits(nkeys, C - 1)
        gain = (cl[i]["data"] + cl[i]["tab"] + cl[j]["data"] + cl[j]["tab"]
                + before_map) - (mdata + mtab + after_map)
        return gain, mdata, mtab, mcnt

    ids = sorted(alive)
    C = len(ids)
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
        if best_k is None or best_g <= (0.0 if backend == "rans" else 0):
            break
        i, j = best_k
        _g, mdata, mtab, mcnt = cache[best_k]
        # merge j into new id
        alive.discard(i)
        alive.discard(j)
        # drop cache entries touching i/j
        for k in [k for k in cache if k[0] in (i, j) or k[1] in (i, j)]:
            del cache[k]
        cl[nxt] = {"keys": cl[i]["keys"] + cl[j]["keys"], "cnt": mcnt,
                   "data": mdata, "tab": mtab}
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
    tot, Cfin = cur_total()
    part = [cl[i]["keys"] for i in sorted(alive)]
    return {"total": tot, "C": Cfin, "part": part, "gains": gains,
            "nmerge": nmerge, "clusters": {i: cl[i] for i in alive}}


# ---------------- Rule J: JS-distance order, exact-gated ----------------
def js_dist(cnt_a, n_a, cnt_b, n_b):
    keys = set(cnt_a.keys()) | set(cnt_b.keys())
    tot = 0.0
    for k in keys:
        p = cnt_a.get(k, 0) / n_a if n_a else 0.0
        q = cnt_b.get(k, 0) / n_b if n_b else 0.0
        m = 0.5 * (p + q)
        if p > 0:
            tot += 0.5 * p * math.log2(p / m)
        if q > 0:
            tot += 0.5 * q * math.log2(q / m)
    return tot


def js_order_merge(nkeys, init_groups, res, key, backend="huff", max_passes=10):
    cost_fn = cluster_cost_huff if backend == "huff" else cluster_cost_rans
    # clusters as lists; DSU via alive dict id->{keys,cnt,data,tab,n}
    cl = {}
    for i, g in enumerate(init_groups):
        cnt, n = counts_of(res, key, g)
        data, _a, tab = cost_fn(cnt)
        cl[i] = {"keys": list(g), "cnt": cnt, "data": data, "tab": tab, "n": n}
    alive = set(cl.keys())
    nxt = max(alive) + 1 if alive else 0
    thr = 0.0 if backend == "huff" else 0.0
    gains = []
    nmerge = 0
    for _ in range(max_passes):
        ids = sorted(alive)
        C = len(ids)
        if C <= 1:
            break
        # static JS order for this pass
        pairs = []
        for x in range(len(ids)):
            for y in range(x + 1, len(ids)):
                i, j = ids[x], ids[y]
                d = js_dist(cl[i]["cnt"], cl[i]["n"], cl[j]["cnt"], cl[j]["n"])
                pairs.append((d, i, j))
        pairs.sort(key=lambda t: t[0])
        merged_this_pass = 0
        for _d, i0, j0 in pairs:
            if i0 not in alive or j0 not in alive:
                continue
            Ccur = len(alive)
            if Ccur <= 1:
                break
            mcnt = merge_dicts(cl[i0]["cnt"], cl[j0]["cnt"])
            mdata, _a, mtab = cost_fn(mcnt)
            gain = (cl[i0]["data"] + cl[i0]["tab"] + cl[j0]["data"]
                    + cl[j0]["tab"] + map_bits(nkeys, Ccur)) - (
                        mdata + mtab + map_bits(nkeys, Ccur - 1))
            if gain > thr:
                alive.discard(i0)
                alive.discard(j0)
                cl[nxt] = {"keys": cl[i0]["keys"] + cl[j0]["keys"],
                           "cnt": mcnt, "data": mdata, "tab": mtab,
                           "n": cl[i0]["n"] + cl[j0]["n"]}
                alive.add(nxt)
                nxt += 1
                gains.append(float(gain))
                nmerge += 1
                merged_this_pass += 1
        if merged_this_pass == 0:
            break
    C = len(alive)
    tot = sum(cl[i]["data"] + cl[i]["tab"] for i in alive) + map_bits(nkeys, C)
    part = [cl[i]["keys"] for i in sorted(alive)]
    return {"total": tot, "C": C, "part": part, "gains": gains,
            "nmerge": nmerge, "clusters": {i: cl[i] for i in alive}}


# ---------------- main ----------------
def main():
    t_all = time.time()
    print("== probe_b12 histogram sharing (JXL-style) ==")
    # ---- anchors ----
    print("-- anchors: MED order-0 exact --")
    anchor_rows = []
    tot_bits, tot_dn = 0, 0
    for path in IMAGES:
        nm = path.split("/")[-1]
        img = np.array(Image.open(path).convert("RGB"))
        H, W, _ = img.shape
        dn = H * W * 3
        Y, Co, Cg = A.ycocg_fwd(img)
        t = HEADER
        for ch in (Y, Co, Cg):
            _, res, _ = A.med_pred_ctx9(ch)
            tb, _, _ = A.stream_bits(res.reshape(-1))
            t += tb
        anchor_rows.append((nm, t / dn))
        tot_bits += t
        tot_dn += dn
        print(f"  {nm}: order0={t/dn:.4f}", flush=True)
    avg0 = tot_bits / tot_dn
    print(f"  AVG order0={avg0:.4f} (expect 3.58; "
          f"{'PASS' if 3.58*0.97 <= avg0 <= 3.58*1.03 else 'FAIL'})")
    expect = {"kodim01.png": 3.62, "kodim02.png": 3.33, "kodim05.png": 4.01,
              "kodim07.png": 3.15, "kodim13.png": 4.24, "kodim19.png": 3.51,
              "kodim23.png": 3.19}
    for nm, v in anchor_rows:
        lo, hi = expect[nm] * 0.97, expect[nm] * 1.03
        print(f"    {nm} {v:.4f} vs {expect[nm]:.2f} "
              f"[{'PASS' if lo <= v <= hi else 'FAIL'}]")

    # ---- CROWN-lite anchor (per-group best-of-7, K=9) to explain 3.34 ----
    print("-- anchor2: CROWN-lite (LOCO-quant K=9/ch + per-group best-of-7) --")
    crown_bits, crown_dn = 0, 0
    for path in IMAGES:
        nm = path.split("/")[-1]
        _img, dn, prep = prep_image(path)
        t = HEADER
        for c in prep:
            ch, key = c["ch"], c["key"]
            _akeys, groups = quantile_groups(c["res"], key, 9)
            C = len(groups)
            nkeys = len(np.unique(key))
            t += nkeys * math.ceil(math.log2(C)) if C > 1 else 0
            t += 4 + 8  # K-id + ch header (CROWN convention note)
            for g in groups:
                mask = np.isin(key, np.asarray(g))
                best = None
                bestA = 0
                for pn in PNAMES:
                    r = (ch - c["pred"][pn]).astype(np.int32)[mask]
                    if r.size == 0:
                        dd = 0
                        aa = 0
                    else:
                        _, cn = np.unique(r, return_counts=True)
                        dd = huff_bits(cn.tolist())
                        aa = len(cn)
                    if best is None or dd < best:
                        best, bestA = dd, aa
                t += best + 16 + bestA * 24 + 3  # table + predictor id
        crown_bits += t
        crown_dn += dn
        print(f"  {nm}: crownlite={t/dn:.4f}", flush=True)
    print(f"  AVG crownlite={crown_bits/crown_dn:.4f} (ref LOCO-clustered 3.34; "
          f"MED-only grouped sits higher by construction — see text)")

    # ---- main sharing experiment ----
    print(f"-- main: quantile K0={K0}/ch atoms, greedy-G vs JS-order-J "
          f"(huff), stretch rANS --")
    rows = []
    sumU = sumG = sumJ = sumW = sumR0 = sumR1 = 0
    sumDn = 0
    tbU = tbW = 0
    mgG = mgW = 0
    nmG = nmW = 0
    detail = {}
    for path in IMAGES:
        nm = path.split("/")[-1]
        _img, dn, prep = prep_image(path)
        # unmerged RAW (fragmentation showcase, no map)
        t_raw = HEADER
        nkeys_tot, data_raw, tbl_raw = 0, 0, 0
        for c in prep:
            res, key = c["res"], c["key"]
            for v in np.unique(key):
                g = res[key == v]
                tb, db, _a = A.stream_bits(g.reshape(-1))
                t_raw += tb
                data_raw += db
                tbl_raw += 16 + _a * 24
                nkeys_tot += 1
        # unmerged grouped + greedy + JS + rans, per channel
        tU = HEADER
        tG = HEADER
        tJ = HEADER
        r0 = HEADER
        r1 = HEADER
        ntabU = ntabG = ntabJ = 0
        ch_gain_G, ch_gain_J, ch_m_G, ch_m_J = [], [], 0, 0
        win_parts = []
        for c in prep:
            res, key = c["res"], c["key"]
            akeys, groups = quantile_groups(res, key, K0)
            nkeys = len(akeys)
            C0 = len(groups)
            # U-GRP
            u = sum(cluster_cost_huff(counts_of(res, key, g)[0])[0]
                    + cluster_cost_huff(counts_of(res, key, g)[0])[2]
                    for g in groups) + map_bits(nkeys, C0)
            # NOTE: counts_of called twice for clarity; cheap vs pair search
            tU += u
            ntabU += C0
            gres = greedy_merge(nkeys, groups, res, key, "huff")
            jres = js_order_merge(nkeys, groups, res, key, "huff")
            tG += gres["total"]
            tJ += jres["total"]
            ntabG += gres["C"]
            ntabJ += jres["C"]
            ch_gain_G += gres["gains"]
            ch_gain_J += jres["gains"]
            ch_m_G += gres["nmerge"]
            ch_m_J += jres["nmerge"]
            # rANS stretch: unmerged + greedy under rans costs
            ru = 0.0
            for g in groups:
                cnt, _n = counts_of(res, key, g)
                dd, _a, tt = cluster_cost_rans(cnt)
                ru += dd + tt
            ru += map_bits(nkeys, C0)
            r0 += ru
            rres = greedy_merge(nkeys, groups, res, key, "rans")
            r1 += rres["total"]
            # winner partition for round-trip (min of G,J under huff)
            win_parts.append(gres["part"] if gres["total"] <= jres["total"]
                             else jres["part"])
        w = min(tG, tJ)
        ntabW = ntabG if tG <= tJ else ntabJ
        rows.append((nm, dn, t_raw / dn, tU / dn, tG / dn, tJ / dn, w / dn,
                     r0 / dn, r1 / dn, nkeys_tot, ntabU, ntabG, ntabJ, ntabW,
                     ch_m_G, ch_m_J))
        sumU += tU
        sumG += tG
        sumJ += tJ
        sumW += w
        sumR0 += r0
        sumR1 += r1
        sumDn += dn
        tbU += ntabU
        tbW += ntabW
        mgG += sum(ch_gain_G)
        mgW += sum((ch_gain_G if tG <= tJ else ch_gain_J))
        nmG += ch_m_G
        nmW += (ch_m_G if tG <= tJ else ch_m_J)
        detail[nm] = {"win_parts": win_parts,
                      "winner": "G" if tG <= tJ else "J"}
        print(f"  {nm}: raw={t_raw/dn:.4f} U={tU/dn:.4f} G={tG/dn:.4f} "
              f"J={tJ/dn:.4f} W={w/dn:.4f} "
              f"rU={r0/dn:.4f} rM={r1/dn:.4f} "
              f"tab {ntabU}->{ntabW} m={ch_m_G}/{ch_m_J} "
              f"win={detail[nm]['winner']}", flush=True)

    print("=" * 100)
    print(f"AVG huff: U-GRP={sumU/sumDn:.4f} G={sumG/sumDn:.4f} "
          f"J={sumJ/sumDn:.4f} WIN={sumW/sumDn:.4f} "
          f"dWIN-U={sumW/sumDn-sumU/sumDn:+.4f} "
          f"({100*(sumW/sumU-1):+.2f}%)")
    print(f"AVG rans-proxy: U={sumR0/sumDn:.4f} M={sumR1/sumDn:.4f} "
          f"d={sumR1/sumDn-sumR0/sumDn:+.4f} "
          f"({100*(sumR1/sumR0-1):+.2f}%)")
    print(f"Tables/image: before={tbU/7:.1f} after={tbW/7:.1f} "
          f"(saved {tbU-tbW} total, {(tbU-tbW)/7:.1f}/image)")
    print(f"Avg merge gain: G {mgG/max(1,nmG)/8:.1f} bits/merge "
          f"({nmG} merges); WIN {mgW/max(1,nmW)/8:.1f} B/merge ({nmW} merges)")
    # table share of unmerged grouped
    # recompute quickly for share note
    print(f"RAW-perkey showcase avg={(sum(r[2]*r[1] for r in rows))/sumDn:.4f} "
          f"(fragmentation ceiling; tables ~0.6bpp there)")

    # ---- round-trip checks ----
    print("-- round-trip: canonical symbol ALL-7 on winner + "
          "literal bitstream on kodim01/kodim07 --")
    ok_all = True
    for path in IMAGES:
        nm = path.split("/")[-1]
        _img, dn, prep = prep_image(path)
        parts = detail[nm]["win_parts"]
        ok = True
        for c, part in zip(prep, parts):
            res, key = c["res"], c["key"]
            # every active key covered exactly once
            flat = sorted([k for g in part for k in g])
            if flat != sorted([int(v) for v in np.unique(key)]):
                ok = False
            for g in part:
                cnt, _n = counts_of(res, key, g)
                if len(cnt) == 0:
                    continue
                L = huffman_lengths(cnt)
                enc, dec = canonical_codes(L)
                # Kraft check
                kr = sum(2.0 ** (-L[s]) for s in L)
                assert abs(kr - 1.0) < 1e-9 or len(L) == 1, (nm, kr)
                mask = np.isin(key, np.asarray(g))
                vals = res[mask].tolist()
                for s in vals:
                    cd, ln = enc[int(s)]
                    assert dec[(cd, ln)] == int(s)
        print(f"  {nm}: symbol-rt {'PASS' if ok else 'FAIL'}", flush=True)
        ok_all = ok_all and ok
    # literal bitstream for 2 images, all groups
    for pick in [IMAGES[0], IMAGES[3]]:
        nm = pick.split("/")[-1]
        _img, dn, prep = prep_image(pick)
        parts = detail[nm]["win_parts"]
        n_sym = n_bit = 0
        for c, part in zip(prep, parts):
            res, key = c["res"], c["key"]
            for g in part:
                mask = np.isin(key, np.asarray(g))
                vals = [int(s) for s in res[mask].tolist()]
                if len(vals) == 0:
                    continue
                cnt, _n = counts_of(res, key, g)
                L = huffman_lengths(cnt)
                enc, dec = canonical_codes(L)
                # pack MSB-first
                buf, nb, out = 0, 0, bytearray()
                for s in vals:
                    cd, ln = enc[s]
                    buf = (buf << ln) | cd
                    nb += ln
                    while nb >= 8:
                        nb -= 8
                        out.append((buf >> nb) & 0xFF)
                if nb:
                    out.append((buf << (8 - nb)) & 0xFF)
                # unpack + decode to N symbols
                bits = np.unpackbits(np.frombuffer(bytes(out), dtype=np.uint8))
                acc, al, got, bi = 0, 0, [], 0
                for b in bits.tolist():
                    acc = (acc << 1) | b
                    al += 1
                    if (acc, al) in dec:
                        got.append(dec[(acc, al)])
                        acc, al = 0, 0
                        if len(got) == len(vals):
                            break
                assert got == vals, f"bitstream mismatch {nm}"
                n_sym += len(vals)
                n_bit += sum(L[s] for s in vals)
        print(f"  {nm}: literal-bitstream PASS ({n_sym} syms, {n_bit} bits)",
              flush=True)
    print(f"TOTAL time {time.time()-t_all:.1f}s; symbol-rt ALL-7 "
          f"{'PASS' if ok_all else 'FAIL'}")
    # stash summary for RESULTS writer
    with open("/tmp/opencode/autocompress/experiments/probe_b12_summary.txt",
              "w") as f:
        f.write(f"AVG0={avg0:.4f}\n")
        f.write(f"CROWNlite={crown_bits/crown_dn:.4f}\n")
        f.write(f"U={sumU/sumDn:.4f} G={sumG/sumDn:.4f} J={sumJ/sumDn:.4f} "
                f"W={sumW/sumDn:.4f}\n")
        f.write(f"RU={sumR0/sumDn:.4f} RM={sumR1/sumDn:.4f}\n")
        f.write(f"TAB {tbU/7:.1f}->{tbW/7:.1f}\n")
        for r in rows:
            f.write("ROW " + " ".join(
                [r[0]] + [f"{x:.4f}" for x in r[2:9]]
                + [str(int(x)) for x in r[9:]]) + "\n")


if __name__ == "__main__":
    main()
