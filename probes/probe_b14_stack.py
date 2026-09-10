"""probe_b14_stack: DECISIVE combined-ceiling probe — stack ALL proven classical mechanisms.

Stacks (each MDL-gated, exact counting, probe-frame bpp=total_bits/(H*W*3)):
 (1) LOCO-365 keys + autoK quantile groups + hill-climb refine (CROWN base, WIDE-K)
 (2) per-group best-of-E16 predictors
 (3) per-group backend choice H/G/R (exact rANS via existing libhapre.so, asserts)
 (4) greedy table merges across groups (proxy-driven shortlisted greedy + exact finalize)
 (5) MA-tree-lite partitions vs quantile groups (per-image choice, 2b path flag)
 (6) threshold-grid dictionary (CROWN2 8-grid family, per-image path choice)
 (7) gated Y-split conditioning (b8 I-GATED ported onto CROWN cells, 1b/atom flags)

Reuses proven modules by import (no existing file modified):
 probe_b4_a (ycocg/stream_bits/huff/IMAGES/HEADER), probe_b4_b (nbhd),
 probe_b4_c (loco_ctx365), probe_b5_c (predictors_X/prepX/E16/MOE6),
 probe_b5_d (golomb_best), probe_b4_h (rans_stream_bits),
 probe_b6_ma (build_features/Rmat5), probe_b6_b (grow_tree_deep),
 dp_refine (refine, from csrc).
numpy+PIL+ctypes only, CPU, no torch.

Decoder-C-compatible: causal recon only (nbhd keys, energy, |rY| of decoded Y,
row/col), static transmitted tables (maps/splits/predids/flags/merge-map/tables).
Planes Y->Co->Cg.
"""
import sys, os, math, heapq, time, json
sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
sys.path.insert(0, "/tmp/opencode/autocompress/csrc")
os.chdir("/tmp/opencode/autocompress/experiments")
import numpy as np
from PIL import Image
import probe_b4_a as A
from probe_b4_b import nbhd as _nbhd, predictors as _predictors
from probe_b5_c import prepX, E16, MOE6
from probe_b5_d import golomb_best
from probe_b4_h import rans_stream_bits
from probe_b6_ma import build_features
import probe_b6_b as B6B
import probe_b6_ma as B6M
from dp_refine import refine

IMAGES = A.IMAGES
HEADER = 64
WIDE = (2, 3, 4, 6, 9, 12, 18, 27, 36, 48, 64)
KSET2 = (2, 3, 4, 6, 9, 12, 18, 27, 36)
GRIDS8 = {
    0: np.array([4, 12, 28, 60, 120], dtype=np.int64),
    1: np.array([2, 6, 16, 40, 100], dtype=np.int64),
    2: np.array([8, 24, 64, 160, 400], dtype=np.int64),
    3: np.array([1, 3, 8, 24, 80], dtype=np.int64),
    4: np.array([3, 9, 20, 48, 110], dtype=np.int64),
    5: np.array([6, 18, 44, 100, 220], dtype=np.int64),
    6: np.array([1, 2, 5, 16, 48], dtype=np.int64),
    7: np.array([5, 15, 36, 80, 160], dtype=np.int64),
}
JXL_E3 = {"kodim01.png": 3.3593, "kodim02.png": 3.0611, "kodim05.png": 3.5120,
          "kodim07.png": 2.7310, "kodim13.png": 3.9141, "kodim19.png": 3.2154,
          "kodim23.png": 2.8110}
TREE_SPLIT_SIDE = 32  # 24b (feat+thr, b6) + 8b topology padding (b6 undercounts topology; corrected here)
YBIN_THR = 1          # |rY|<=1 vs >1 (b8 I-Y2 bound)
SHORTLIST_S = 16      # merge partner shortlist per atom

# ---------------- cost primitives ----------------
def huff_tot(vals):
    vals = np.asarray(vals).reshape(-1)
    if vals.size == 0: return 0, 0, 0
    uv, cn = np.unique(vals, return_counts=True)
    d = A.huff_bits(cn.tolist())
    return d + 16 + len(cn) * 24, d, len(cn)

def gol_tot(vals):
    vals = np.asarray(vals).reshape(-1)
    if vals.size == 0: return 0, 0
    gd, k = golomb_best(vals)
    return gd + 4, k  # data + 4b k-side

def ent_proxy(vals):
    vals = np.asarray(vals).reshape(-1)
    if vals.size == 0: return 0, 0
    uv, cn = np.unique(vals, return_counts=True)
    c = cn.astype(np.float64); p = c / c.sum()
    e = 0.0 if len(c) <= 1 else float((-(p * np.log2(p))).sum() * c.sum())
    return e, len(cn)

def counts_of(vals):
    vals = np.asarray(vals).reshape(-1)
    if vals.size == 0: return {}
    uv, cn = np.unique(vals, return_counts=True)
    return {int(v): int(c) for v, c in zip(uv.tolist(), cn.tolist())}

def huff_of_counts(cnt):
    vals = list(cnt.values())
    if len(vals) == 0: return 0, 0
    return A.huff_bits(vals), len(vals)

def check_rans_range(vals, where):
    m = int(np.abs(np.asarray(vals)).max()) if np.asarray(vals).size else 0
    assert m <= 1024, f"RANS ALPHABET VIOLATION {where}: |r|max={m} (abort, never clip)"

# ---------------- atom helpers ----------------
# atom = dict(idx=flat int array, vals=int array (flipped iff flip), pred=expert,
#             ch=channel, flip=bool, base=int base-atom id, bin=int split bin)
def score_2way(vals, ho):
    """best min(H,G) total + which (vals fixed). ho=True -> H-only."""
    ht, _, _ = huff_tot(vals)
    if ho: return ht, "H", 0
    gt, k = gol_tot(vals)
    if ht <= gt: return ht, "H", 0
    return gt, "G", k

def best_expert_2way(RF, idx, bank, ho=False):
    best = None
    for n in bank:
        v = RF[n][idx]
        ht, _, _ = huff_tot(v)
        if ho:
            c, k = ht, 0
        else:
            gt, k = gol_tot(v)
            c = ht if ht <= gt else gt
        if best is None or c < best[0]:
            best = (c, n, ht, k)
    return best  # (cost, expert, hcost, k)

# ---------------- path builders (2-way level) ----------------
def quantile_groups(key, rfM, K):
    kf = np.asarray(key).reshape(-1)
    rfM = np.asarray(rfM).reshape(-1)
    uk, cn = np.unique(kf, return_counts=True)
    ma = np.array([np.abs(rfM[kf == v]).mean() for v in uk])
    order = np.argsort(ma, kind="stable")
    uks, cns = uk[order], cn[order]
    tot = cns.sum(); tgt = tot / K
    groups, cur, acc = [], [], 0
    for u, c in zip(uks, cns):
        cur.append(u); acc += c
        if acc >= tgt and len(groups) < K - 1:
            groups.append(cur); cur, acc = [], 0
    groups.append(cur)
    return groups

def build_Q(chidx, D, RF, bank, pb, KSET, do_refine, ho=False):
    """returns (search_total, fam_side, atoms). atoms carry idx/vals/pred/keys."""
    key, s = D["key"], D["s"]
    kf = key.reshape(-1)
    rfM = RF["MED"]
    nactive = len(np.unique(key))
    best = None
    for K in KSET:
        G = quantile_groups(kf, rfM, K)
        t = nactive * math.ceil(math.log2(K)) + 4 + 8
        for gkeys in G:
            sel = np.isin(kf, np.array(gkeys))
            idx = np.flatnonzero(sel)
            if idx.size == 0: continue
            c, _, _, _ = best_expert_2way(RF, idx, bank, ho)
            t += c + pb + 1
        if best is None or t < best[0]:
            best = (t, K, G)
    _, K, G = best
    if do_refine:
        G, _ = refine(kf, rfM, G, K)
    atoms = []
    t = nactive * math.ceil(math.log2(K)) + 4 + 8
    for gkeys in G:
        sel = np.isin(kf, np.array(gkeys))
        idx = np.flatnonzero(sel)
        if idx.size == 0: continue
        c, bn, _, _ = best_expert_2way(RF, idx, bank, ho)
        t += c + pb + 1
        atoms.append(dict(idx=idx, vals=np.ascontiguousarray(RF[bn][idx]), pred=bn,
                          ch=chidx, flip=True, base=len(atoms), bin=0,
                          keys=[int(u) for u in gkeys]))
    return t, nactive * math.ceil(math.log2(K)) + 4 + 8, atoms, dict(K=K, ng=len(atoms))

def build_O(chidx, D, RF, bank, pb, ho=False):
    """order-0 fallback (LOO-1): single group per channel."""
    N = D["key"].size
    idx = np.arange(N)
    c, bn, _, _ = best_expert_2way(RF, idx, bank, ho)
    t = c + pb + 1
    atoms = [dict(idx=idx, vals=np.ascontiguousarray(RF[bn][idx]), pred=bn,
                  ch=chidx, flip=True, base=0, bin=0, keys=None)]
    return t, 0, atoms, dict(K=1, ng=1)

def build_D(chidx, D, RU, bank, pb, ho=False):
    a, b, c = D["a"], D["b"], D["c"]
    e = (np.abs(a - c) + np.abs(b - c)).astype(np.int64).reshape(-1)
    best = None
    for gi, thr in GRIDS8.items():
        gg = np.digitize(e, thr, right=True)
        t = 3
        cells = []
        for cc in range(6):
            idx = np.flatnonzero(gg == cc)
            if idx.size == 0: continue
            co, bn, _, _ = best_expert_2way(RU, idx, bank, ho)
            t += co + pb + 1
            cells.append((idx, bn, cc))
        if best is None or t < best[0]:
            best = (t, gi, cells)
    t, gi, cells = best
    atoms = [dict(idx=idx, vals=np.ascontiguousarray(RU[bn][idx]), pred=bn,
                  ch=chidx, flip=False, base=i, bin=0, gid=cc)
             for i, (idx, bn, cc) in enumerate(cells)]
    return t, 3, atoms, dict(grid=gi, ng=len(atoms))

def grow_explicit(Fs, Rmat_s, Ffull, Rmat_full, max_leaves=16, max_depth=5):
    """Explicit-topology port of probe_b6_b.grow_tree_deep (same BANK5 search objective,
    same subsample-search + exact-full gate, same open-leaf pop/append order).
    Returns (root, leaves_dfs, struct). Topology transmittable in 8b/inner (child ids)."""
    MS = B6M.MARGINAL_SPLIT; MINL = B6M.MIN_LEAF
    N = Ffull.shape[0]; Ns = Fs.shape[0]
    pFull, _ = B6M.best_pred_cost(Rmat_full, np.arange(N))
    nid = [1]
    def newleaf(fidx, sidx, depth, pcost):
        nd = dict(kind="leaf", fidx=fidx, sidx=sidx, depth=depth, pcost=pcost, id=nid[0]); nid[0] += 1
        return nd
    root = dict(kind="leaf", fidx=np.arange(N), sidx=np.arange(Ns), depth=0, pcost=pFull, id=0)
    open_leaves = [root]
    struct = []
    nF = Fs.shape[1]
    def search_pcost(sidx):
        if sidx.size == 0: return 0
        bc, _ = B6M.best_pred_cost(Rmat_s, sidx); return bc
    while True:
        if len(open_leaves) >= max_leaves: break
        best_gain_sub = -1; best = None
        for li, lf in enumerate(open_leaves):
            if lf["depth"] >= max_depth: continue
            if lf["fidx"].size < 2 * MINL: continue
            sidx = lf["sidx"]
            if sidx.size < 512: continue
            psub = search_pcost(sidx)
            Fnode = Fs[sidx]
            for f in range(nF):
                col = Fnode[:, f]
                if col.min() == col.max(): continue
                for thr in B6M.cand_thresholds(col):
                    m = col <= thr
                    nl = int(m.sum()); nr = int(m.size - nl)
                    if nl < 256 or nr < 256: continue
                    l_s = sidx[m]; r_s = sidx[~m]
                    lc, _ = B6M.best_pred_cost(Rmat_s, l_s)
                    rc, _ = B6M.best_pred_cost(Rmat_s, r_s)
                    g = psub - (lc + rc + MS)
                    if g > best_gain_sub:
                        best_gain_sub = g; best = (li, f, thr, l_s, r_s)
        if best is None or best_gain_sub <= 0: break
        li, f, thr, l_s, r_s = best
        lf = open_leaves[li]
        colfull = Ffull[lf["fidx"], f]
        mfull = colfull <= thr
        l_f = lf["fidx"][mfull]; r_f = lf["fidx"][~mfull]
        if l_f.size < MINL or r_f.size < MINL:
            lf["depth"] = max_depth
            continue
        lcF, _ = B6M.best_pred_cost(Rmat_full, l_f)
        rcF, _ = B6M.best_pred_cost(Rmat_full, r_f)
        gain_full = lf["pcost"] - (lcF + rcF + MS)
        if gain_full <= 0: break
        struct.append(dict(feat=f, thr=thr, gain_full=int(gain_full)))
        newl = newleaf(l_f, l_s, lf["depth"] + 1, lcF)
        newr = newleaf(r_f, r_s, lf["depth"] + 1, rcF)
        lf["kind"] = "inner"; lf["feat"] = f; lf["thr"] = thr
        lf["left"] = newl; lf["right"] = newr
        open_leaves.pop(li); open_leaves.append(newl); open_leaves.append(newr)
        if len(open_leaves) >= max_leaves: break
    leaves = []
    def dfs(nd):
        if nd["kind"] == "leaf": leaves.append(nd)
        else: dfs(nd["left"]); dfs(nd["right"])
    dfs(root)
    return root, leaves, struct

def build_T(chidx, F, Rmat5, RF, bank, pb, ho=False):
    """tree structure grown on BANK5 (proven b6 search), leaves re-scored under bank.
    stores explicit root for decoder tree-walk."""
    N = F.shape[0]
    Fs = F[::B6B.STRIDE]; Rs = Rmat5[::B6B.STRIDE]
    root, leaves, struct = grow_explicit(Fs, Rs, F, Rmat5, max_leaves=16, max_depth=5)
    atoms = []
    t = len(struct) * TREE_SPLIT_SIDE
    for lf in leaves:
        idx = np.ascontiguousarray(lf["fidx"])
        if idx.size == 0: continue
        lf["leaf_pos"] = len(atoms)
        co, bn, _, _ = best_expert_2way(RF, idx, bank, ho)
        t += co + pb + 1
        atoms.append(dict(idx=idx, vals=np.ascontiguousarray(RF[bn][idx]), pred=bn,
                          ch=chidx, flip=True, base=len(atoms), bin=0))
    return t, len(struct) * TREE_SPLIT_SIDE, atoms, dict(nInner=len(struct), ng=len(atoms), root=root)

# ---------------- Y-split ----------------
def apply_ysplit(atoms, rYabs_flat, enable, ho=False):
    """per chroma base-atom local MDL gate. returns (cells, flagbits)."""
    cells = []
    flags = 0
    for at in atoms:
        if at["ch"] == 0 or not enable:
            cells.append(dict(at))
            continue
        flags += 1  # 1b flag always counted
        v = at["vals"]; yb = rYabs_flat[at["idx"]] > YBIN_THR
        if yb.all() or (~yb).all():
            cells.append(dict(at))
            continue
        c0, _, _ = score_2way(v[~yb], ho)
        c1, _, _ = score_2way(v[yb], ho)
        # compare with consistent sides: whole (pb+1 already counted at path level);
        # split adds one extra (pb+1) for the second cell + flag already counted
        cw, _, _ = score_2way(v, ho)
        if c0 + c1 + 5 < cw:  # extra predid(4)+choice(1) for 2nd cell; flag counted above
            i0 = at["idx"][~yb]; i1 = at["idx"][yb]
            cells.append(dict(idx=i0, vals=np.ascontiguousarray(v[~yb]), pred=at["pred"],
                              ch=at["ch"], flip=at["flip"], base=at["base"], bin=0))
            cells.append(dict(idx=i1, vals=np.ascontiguousarray(v[yb]), pred=at["pred"],
                              ch=at["ch"], flip=at["flip"], base=at["base"], bin=1))
        else:
            cells.append(dict(at))
    return cells, flags

# ---------------- 3-way upgrade + merges ----------------
def upgrade_3way(cells, use_rans, huff_only=False):
    """exact per-cell H/G[/R] on winning expert. fills cell['be'] ('H'/'G'/'R'),
    cell['cost'], cell['k'], cell['cnt']. R checked on 2-way winning expert only (b5)."""
    for cl in cells:
        v = cl["vals"]
        if v.size == 0:
            cl.update(be="H", cost=0, k=0, cnt={})
            continue
        ht, _, _ = huff_tot(v)
        if huff_only:
            cl["be"] = "H"; cl["cost"] = ht
        else:
            gt, k = gol_tot(v)
            if ht <= gt:
                be, cost = "H", ht
            else:
                be, cost, cl_k = "G", gt, k
                cl["k"] = cl_k
            if use_rans:
                check_rans_range(v, "upgrade")
                rt, _ = rans_stream_bits(v)
                if rt < cost:
                    be, cost = "R", rt
            cl["be"] = be
            cl["cost"] = cost
        if "k" not in cl: cl["k"] = 0
        cl["cnt"] = counts_of(v)
    return cells

def proxy_of_counts(cnt):
    hd, A = huff_of_counts(cnt)
    ht = hd + 16 + A * 24
    # golomb exact needs values; proxy from counts: reconstruct multiset approx via symbols?
    # caller passes vals too; this helper only for merged dicts -> golomb skipped in search
    # (search proxy = min(H_exact, R_proxy); G handled at finalize). Documented approx.
    e, Ar = ent_proxy(list(cnt.keys()) and np.repeat(list(cnt.keys()), list(cnt.values())))
    rt = e + 16 + Ar * 32
    return min(ht, rt)

def greedy_merge(cells, do_merge):
    """shortlisted greedy agglomeration (proxy costs, exact-gated accepts).
    returns (clusters, merged_bool, mapbits). clusters: list of member-idx lists."""
    n = len(cells)
    cnts = [c["cnt"] for c in cells]
    prox = []
    for i, c in enumerate(cells):
        v = c["vals"]
        ht, _, _ = huff_tot(v)
        gt, _ = gol_tot(v)
        e, Ar = ent_proxy(v)
        prox.append(min(ht, gt, e + 16 + Ar * 32))
    if not do_merge or n <= 1:
        return [[i] for i in range(n)], False, 0
    import math as _m
    def mmap(C): return 0 if C <= 1 else int(n * _m.ceil(_m.log2(C)))
    # shortlist: S nearest partners by mean distance
    means = np.array([float(np.average(list(cnts[i].keys()), weights=list(cnts[i].values()))) if cnts[i] else 0.0 for i in range(n)])
    alive = set(range(n))
    cl_cnt = dict(enumerate(cnts))
    cl_prox = dict(enumerate(prox))
    members = {i: [i] for i in range(n)}
    C = n
    # candidate cache over shortlisted pairs
    def merged_cnt(a, b):
        d = dict(cl_cnt[a])
        for k, v in cl_cnt[b].items(): d[k] = d.get(k, 0) + v
        return d
    def pair_gain(a, b, Cnow):
        m = merged_cnt(a, b)
        mc = proxy_of_counts(m)
        return (cl_prox[a] + cl_prox[b] + mmap(Cnow)) - (mc + mmap(Cnow - 1)), mc, m
    cand = set()
    order = np.argsort(means, kind="stable")
    pos = {v: i for i, v in enumerate(order)}
    for i in range(n):
        p = pos[i]
        for q in range(max(0, p - SHORTLIST_S // 2), min(n, p + SHORTLIST_S // 2 + 1)):
            j = int(order[q])
            if j != i: cand.add((min(i, j), max(i, j)))
    # optional adjacent pre-pass when huge
    if n > 140:
        for k in range(len(order) - 1):
            a, b = int(order[k]), int(order[k + 1])
            if a in alive and b in alive:
                g, mc, m = pair_gain(a, b, C)
                if g > 0:
                    alive.discard(a); alive.discard(b)
                    nid = max(cl_cnt) + 1
                    cl_cnt[nid] = m; cl_prox[nid] = mc; alive.add(nid)
                    members[nid] = members.pop(a) + members.pop(b)
                    C = len(alive)
        cand = set()
        ids = sorted(alive)
        mm = {i: float(np.average(list(cl_cnt[i].keys()), weights=list(cl_cnt[i].values()))) if cl_cnt[i] else 0.0 for i in ids}
        oo = sorted(ids, key=lambda i: mm[i])
        pp = {v: i for i, v in enumerate(oo)}
        for i in ids:
            p = pp[i]
            for q in range(max(0, p - SHORTLIST_S // 2), min(len(oo), p + SHORTLIST_S // 2 + 1)):
                j = oo[q]
                if j != i: cand.add((min(i, j), max(i, j)))
    cache = {}
    for (a, b) in cand:
        if a in alive and b in alive:
            cache[(a, b)] = pair_gain(a, b, C)
    nmerge = 0
    while True:
        C = len(alive)
        if C <= 1: break
        bk, bg = None, 0.0
        for k, v in cache.items():
            if k[0] in alive and k[1] in alive and v[0] > bg:
                bg, bk = v[0], k
        if bk is None or bg <= 0: break
        a, b = bk
        _, mc, m = cache[bk]
        for k in [k for k in cache if k[0] in (a, b) or k[1] in (a, b)]: del cache[k]
        alive.discard(a); alive.discard(b)
        nid = max(cl_cnt) + 1
        cl_cnt[nid] = m; cl_prox[nid] = mc; alive.add(nid)
        members[nid] = members.pop(a) + members.pop(b)
        C2 = len(alive)
        for o in alive:
            if o == nid: continue
            x, y = (o, nid) if o < nid else (nid, o)
            cache[(x, y)] = pair_gain(x, y, C2)
        nmerge += 1
    if nmerge == 0:
        return [[i] for i in range(n)], False, 0
    return [members[i] for i in sorted(alive)], True, mmap(len(alive)) + 8

def finalize(clusters, cells, use_rans, huff_only):
    """exact H/G[/R] per cluster + choice sides. returns (bits, detail)."""
    tot = 0; nb = {"H": 0, "G": 0, "R": 0}
    for mem in clusters:
        v = np.concatenate([cells[i]["vals"] for i in mem]) if len(mem) > 1 else cells[mem[0]]["vals"]
        if v.size == 0: continue
        if huff_only:
            ht, _, _ = huff_tot(v); tot += ht; nb["H"] += 1
            continue
        ht, _, _ = huff_tot(v)
        gt, k = gol_tot(v)
        be, cost = ("H", ht) if ht <= gt else ("G", gt)
        kk = 0 if be == "H" else k
        if use_rans:
            check_rans_range(v, "finalize")
            rt, _ = rans_stream_bits(v)
            if rt < cost: be, cost, kk = "R", rt, 0
        tot += cost + 2 + (4 if be == "G" else 0)
        nb[be] += 1
    return tot, nb

# ---------------- per-image pipeline ----------------
def process_image(path, cfg, prep_cache):
    """cfg keys: bank(list), KSET|None(O), refine, paths(set of Q/T/D[/O]),
    split, rans, merge, huff_only. returns dict with bpp + detail."""
    nm = path.split("/")[-1]
    img = np.array(Image.open(path).convert("RGB"))
    H, W, _ = img.shape; dn = H * W * 3
    Y, Co, Cg = A.ycocg_fwd(img)
    chs = [Y, Co, Cg]
    bank = cfg["bank"]; pb = 4 if len(bank) == 16 else 3
    ho = cfg["huff_only"]
    P = prep_cache[nm]  # per-channel dicts: D, RF, RU, F, Rmat5
    # candidate paths
    cands = {}
    for chidx in range(3):
        d = P[chidx]
        D, RF, RU = d["D"], d["RF"], d["RU"]
        if "Q" in cfg["paths"]:
            t, side, atoms, info = build_Q(chidx, D, RF, bank, pb, cfg["KSET"], cfg["refine"], ho)
            cands.setdefault("Q", []).append((t, side, atoms, info))
        if "O" in cfg["paths"]:
            t, side, atoms, info = build_O(chidx, D, RF, bank, pb, ho)
            cands.setdefault("O", []).append((t, side, atoms, info))
        if "D" in cfg["paths"]:
            t, side, atoms, info = build_D(chidx, D, RU, bank, pb, ho)
            cands.setdefault("D", []).append((t, side, atoms, info))
        if "T" in cfg["paths"]:
            t, side, atoms, info = build_T(chidx, d["F"], d["Rmat5"], RF, bank, pb, ho)
            cands.setdefault("T", []).append((t, side, atoms, info))
    path_tot = {}
    for p, lst in cands.items():
        path_tot[p] = sum(t for t, _, _, _ in lst)
    order = sorted(path_tot, key=lambda p: path_tot[p])
    win = order[0]
    atoms = []
    famside = 0
    det = {"path_order": {p: round(v / dn, 4) for p, v in path_tot.items()}}
    for t, side, at, info in cands[win]:
        atoms.extend(at); famside += side
    det["win"] = win
    det["chinfo"] = [info for _, _, _, info in cands[win]]
    # Y-split keyed on Y winner residuals
    Yvals = np.zeros(H * W, dtype=np.int32)
    for at in atoms:
        if at["ch"] == 0:
            Yvals[at["idx"]] = np.abs(at["vals"])
    cells, flagbits = apply_ysplit(atoms, Yvals, cfg["split"], ho)
    det["ncells"] = len(cells)
    det["split_flags"] = flagbits
    # 3-way upgrade (exact) then merges
    upgrade_3way(cells, cfg["rans"] and not cfg["huff_only"], ho)
    clusters, merged, mapbits = greedy_merge(cells, cfg["merge"])
    if merged:
        fin, nb = finalize(clusters, cells, cfg["rans"] and not cfg["huff_only"], cfg["huff_only"])
        gated = fin + mapbits
        # gate vs unmerged exact
        un, _ = finalize([[i] for i in range(len(cells))], cells, cfg["rans"] and not cfg["huff_only"], cfg["huff_only"])
        if un <= gated:
            clusters, merged, mapbits = [[i] for i in range(len(cells))], False, 0
            fin, nb = un, nb
    else:
        fin, nb = finalize(clusters, cells, cfg["rans"] and not cfg["huff_only"], cfg["huff_only"])
    # final framing: global + path selector + fam sides + predids + split flags + merge map + clusters
    predside = len(atoms) * pb
    total = HEADER + cfg["pathbits"] + famside + predside + flagbits + mapbits + fin
    det.update(path=win, K=[c.get("K", c.get("grid", c.get("nInner", 0))) for c in det["chinfo"]],
               ng=len(atoms), nCl=len(clusters), merged=merged, back=nb,
               famside=famside, predside=predside, mapbits=mapbits, pay_fin=fin)
    return {"bpp": total / dn, "bits": total, "dn": dn, "detail": det,
            "cells": cells, "clusters": clusters, "atoms": atoms, "img": img,
            "nm": nm, "H": H, "W": W}

def prep_all():
    cache = {}
    for path in IMAGES:
        nm = path.split("/")[-1]
        img = np.array(Image.open(path).convert("RGB"))
        Y, Co, Cg = A.ycocg_fwd(img)
        chs = [Y, Co, Cg]
        # Y MED residual for chroma tree features
        aY, bY, cY, dY, Ww, NNe, NE = _nbhd(Y)
        PY = _predictors(aY, bY, cY, dY, Ww, NNe, NE)
        Yres = (Y - PY["MED"]).astype(np.int32)
        Ymean = ((aY + bY) // 2).astype(np.int32)
        per = []
        for ci, ch in enumerate(chs):
            D = prepX(ch)
            RF = {n: (D["s"] * (ch - D["P"][n]).astype(np.int32)).astype(np.int32).reshape(-1) for n in E16}
            RU = {n: (ch - D["P"][n]).astype(np.int32).reshape(-1) for n in E16}
            for n in E16:
                m = int(np.abs(RF[n]).max())
                assert m <= 1024, f"ALPHABET {nm} ch{ci} {n}: {m}"
            if ci == 0:
                F, names, P5, s5, Rmat5, aux = build_features(ch)
            else:
                F, names, P5, s5, Rmat5, aux = build_features(
                    ch, Yres_abs=np.abs(Yres), Yres=Yres, Ymean=Ymean)
            per.append(dict(D=D, RF=RF, RU=RU, F=F, Rmat5=Rmat5, fnames=names))
        cache[nm] = per
    return cache

# ---------------- recon proof (FULL-stack winning config) ----------------
def scalar_pred(bn, av, bv, cv, dv, Ww, NNe, NE):
    if bn == "MED":
        mx = av if av > bv else bv; mn = bv if av > bv else av
        return mn if cv >= mx else (mx if cv <= mn else av + bv - cv)
    if bn == "TOP": return bv
    if bn == "LEFT": return av
    if bn == "PAETH":
        pp = av + bv - cv; pa = abs(pp - av); pb = abs(pp - bv); pc = abs(pp - cv)
        return av if (pa <= pb and pa <= pc) else (bv if pb <= pc else cv)
    if bn == "GRAD": return (av + bv) // 2 + (bv - cv) // 4
    if bn in ("GAP", "GAP80"):
        gh = abs(av - Ww) + abs(bv - cv) + abs(bv - NE); gv = abs(av - cv) + abs(bv - NNe) + abs(NE - bv)
        return av if gv - gh > 80 else (bv if gv - gh < -80 else (av + bv) // 2 + (NE - cv) // 4)
    if bn == "GAP32":
        gh = abs(av - Ww) + abs(bv - cv) + abs(bv - NE); gv = abs(av - cv) + abs(bv - NNe) + abs(NE - bv)
        return av if gv - gh > 32 else (bv if gv - gh < -32 else (av + bv) // 2 + (NE - cv) // 4)
    if bn == "GAP16":
        gh = abs(av - Ww) + abs(bv - cv) + abs(bv - NE); gv = abs(av - cv) + abs(bv - NNe) + abs(NE - bv)
        return av if gv - gh > 16 else (bv if gv - gh < -16 else (av + bv) // 2 + (NE - cv) // 4)
    if bn == "DG": return (cv + dv) // 2
    if bn == "AVG_AB": return (av + bv) // 2
    if bn == "PLANE": return av + bv - cv
    if bn == "AC": return (av + cv) // 2
    if bn == "C": return cv
    if bn == "BC": return (bv + cv) // 2
    if bn == "D": return dv
    if bn == "AVG3": return (av + bv + cv) // 3
    raise ValueError(bn)

def Qkey_of(av, bv, cv, dv):
    def Q(g): return 0 if g == 0 else (1 if abs(g) <= 2 else (2 if abs(g) <= 7 else (3 if abs(g) <= 21 else 4))) * (1 if g > 0 else -1)
    qq = (Q(bv - cv), Q(cv - av), Q(dv - bv))
    neg = qq[0] < 0 or (qq[0] == 0 and qq[1] < 0) or (qq[0] == 0 and qq[1] == 0 and qq[2] < 0)
    kk = tuple(-v for v in qq) if neg else qq
    return kk[0] * 81 + kk[1] * 9 + kk[2], (1 if not neg else -1)

def recon_prove(res, prep_cache):
    """scalar decoder sim for FULL-stack winning config.
    selections re-derived from recon (+ transmitted sides) exactly as a C decoder
    would: Q via key->base LUT (transmitted map), D via energy->gid (transmitted
    grid id), T via explicit tree-walk on recon features; split bin via decoded
    |rY|; table via transmitted merge map; expert via transmitted predids.
    cluster streams built row-major from encoder cells (order == decode walk)."""
    nm, H, W = res["nm"], res["H"], res["W"]
    img = res["img"]; atoms = res["atoms"]; cells = res["cells"]; clusters = res["clusters"]
    det = res["detail"]
    Y, Co, Cg = A.ycocg_fwd(img); chs = [Y, Co, Cg]
    N = H * W
    clust_of_cell = {}
    for ci, mem in enumerate(clusters):
        for m in mem: clust_of_cell[m] = ci
    # transmitted selection sides per channel
    chatoms = {c: [a for a in atoms if a["ch"] == c] for c in range(3)}
    qlut = {}
    for ci in range(3):
        if det["path"] == "Q":
            lut = np.full(729, -1, dtype=np.int32)
            for bi, at in enumerate(chatoms[ci]):
                for u in at["keys"]: lut[u] = bi
            assert (lut >= 0).all() or True  # inactive keys may stay -1; pixels never hit them
            qlut[ci] = lut
    # partition-cover check: every pixel of every channel in exactly one atom
    for ci in range(3):
        cov = np.zeros(N, dtype=np.int32)
        for at in chatoms[ci]: cov[at["idx"]] += 1
        assert (cov == 1).all(), f"cover FAIL ch{ci}"
    rYabs = np.zeros(N, dtype=np.int32)
    Yres_plane = np.zeros(N, dtype=np.int32)
    Ymean_plane = np.zeros(N, dtype=np.int32)
    for ci in range(3):
        kch = chs[ci]
        recon = np.zeros((H, W), dtype=np.int32)
        # encoder-truth streams per cluster (row-major), values from cells
        cellid = np.full(N, -1)
        for li, cl in enumerate(cells):
            if cl["ch"] == ci: cellid[cl["idx"]] = li
        assert (cellid >= 0).all(), f"cell cover FAIL ch{ci}"
        valof = np.zeros(N, dtype=np.int32)
        for cl in cells:
            if cl["ch"] == ci: valof[cl["idx"]] = cl["vals"]
        streams = {}
        for p in range(N):
            streams.setdefault(clust_of_cell[int(cellid[p])], []).append(int(valof[p]))
        curs = {k: 0 for k in streams}
        root = det["chinfo"][ci].get("root") if det["path"] == "T" else None
        for i in range(H):
            for j in range(W):
                if i == 0 and j == 0: av = bv = cv = dv = 0
                elif i == 0: av = int(recon[i, j - 1]); bv = cv = av; dv = av
                elif j == 0: bv = int(recon[i - 1, 0]); av = cv = bv; dv = int(recon[i - 1, 1]) if W > 1 else bv
                else: av = int(recon[i, j - 1]); bv = int(recon[i - 1, j]); cv = int(recon[i - 1, j - 1]); dv = int(recon[i - 1, j + 1]) if j + 1 < W else bv
                Ww = int(recon[i, j - 2]) if j >= 2 else av
                NNe = int(recon[i - 2, j]) if i >= 2 else bv
                NE = int(recon[i - 1, j + 1]) if (i >= 1 and j + 1 < W) else bv
                if i == 0: Ww = av; NNe = av; NE = av
                p = i * W + j
                # re-derive base atom from recon (decoder side)
                if det["path"] == "Q":
                    kid, sgn = Qkey_of(av, bv, cv, dv)
                    D = prep_cache[nm][ci]["D"]
                    assert kid == int(D["key"][i, j]) and sgn == int(D["s"][i, j]), f"key mismatch {(i,j)} ch{ci}"
                    bi = int(qlut[ci][kid])
                    assert bi >= 0
                elif det["path"] == "D":
                    e = abs(av - cv) + abs(bv - cv)
                    thr = GRIDS8[det["chinfo"][ci]["grid"]]
                    g = int(np.digitize([e], thr, right=True)[0])
                    hits = [bi for bi, at in enumerate(chatoms[ci]) if at["gid"] == g]
                    assert len(hits) == 1, f"D cell miss {(i,j)} ch{ci} g={g}"
                    bi = hits[0]; sgn = 1
                elif det["path"] == "T":
                    f = [abs(bv - cv), abs(av - cv), abs(dv - bv),
                         abs(bv - cv) + abs(av - cv),
                         min((abs(bv - cv) + abs(av - cv)) >> 4, 8),
                         abs(av - bv), abs(av + bv - 2 * cv) // 2,
                         (av + bv) // 2, i, j]
                    if ci > 0:
                        f += [int(rYabs[p]), int(Yres_plane[p]), int(Ymean_plane[p])]
                    nd = root
                    while nd["kind"] == "inner":
                        nd = nd["left"] if f[nd["feat"]] <= nd["thr"] else nd["right"]
                    # leaf id -> base index via DFS order used in build_T
                    bi = nd["leaf_pos"]; sgn = 1
                else:
                    raise ValueError(det["path"])
                bat = chatoms[ci][bi]
                bn = bat["pred"]; flip = bat["flip"]
                if det["path"] == "Q":
                    pass  # sgn from key
                # split bin from decoded Y
                cand = [lii for lii, cl in enumerate(cells) if cl["ch"] == ci and cl["base"] == bi]
                if ci > 0 and len(cand) == 2:
                    want = 0 if rYabs[p] <= YBIN_THR else 1
                    cand = [lii for lii in cand if cells[lii]["bin"] == want]
                assert len(cand) == 1, f"cell ambig {(i,j)} ch{ci} base={bi}"
                lii = cand[0]
                pred = scalar_pred(bn, av, bv, cv, dv, Ww, NNe, NE)
                slist = streams[clust_of_cell[lii]]
                rf = slist[curs[clust_of_cell[lii]]]; curs[clust_of_cell[lii]] += 1
                r = sgn * rf if flip else rf
                recon[i, j] = pred + r
                assert recon[i, j] == kch[i, j], f"recon mismatch ch{ci} {(i,j)}"
                if ci == 0:
                    rYabs[p] = abs(rf)
                    mx = av if av > bv else bv; mn = bv if av > bv else av
                    med = mn if cv >= mx else (mx if cv <= mn else av + bv - cv)
                    Yres_plane[p] = recon[i, j] - med
                    Ymean_plane[p] = (av + bv) // 2
        assert all(curs[k] == len(streams[k]) for k in streams), f"cursors ch{ci}"
    return True

def canon_roundtrip_all(res):
    for ci, mem in enumerate(res["clusters"]):
        v = np.concatenate([res["cells"][i]["vals"] for i in mem]) if len(mem) > 1 else res["cells"][mem[0]]["vals"]
        if v.size == 0: continue
        # canonical lengths + Kraft + encode/decode all symbols
        uv, cn = np.unique(v, return_counts=True)
        cnt = {int(a): int(b) for a, b in zip(uv.tolist(), cn.tolist())}
        syms = list(cnt.keys())
        if len(syms) == 1:
            continue
        import itertools
        ctr = itertools.count()
        h = [(cnt[s], next(ctr), ("leaf", s)) for s in syms]; heapq.heapify(h)
        while len(h) > 1:
            c1, _, n1 = heapq.heappop(h); c2, _, n2 = heapq.heappop(h)
            heapq.heappush(h, (c1 + c2, next(ctr), ("in", n1, n2)))
        lens = {}
        def walk(nd, d):
            if nd[0] == "leaf": lens[nd[1]] = max(1, d)
            else: walk(nd[1], d + 1); walk(nd[2], d + 1)
        walk(h[0][2], 0)
        kr = sum(2.0 ** -lens[s] for s in lens)
        assert abs(kr - 1.0) < 1e-9, f"Kraft {ci}"
        order = sorted(cnt, key=lambda s: (lens[s], s))
        enc = {}; code, prev = 0, 0
        for s in order:
            code <<= (lens[s] - prev); enc[s] = (code, lens[s]); code += 1; prev = lens[s]
        # verify prefix-free decode of full multiset via bitstream
        buf, nb, out = 0, 0, bytearray()
        flat = np.asarray(v).reshape(-1).tolist()
        for s in flat:
            cd, ln = enc[int(s)]; buf = (buf << ln) | cd; nb += ln
            while nb >= 8: nb -= 8; out.append((buf >> nb) & 0xFF)
        if nb: out.append((buf << (8 - nb)) & 0xFF)
        dec = {(c, l): s for s, (c, l) in enc.items()}
        bits = np.unpackbits(np.frombuffer(bytes(out), dtype=np.uint8)).tolist()
        acc, al, got = 0, 0, []
        for b in bits:
            acc = (acc << 1) | b; al += 1
            if (acc, al) in dec:
                got.append(dec[(acc, al)]); acc, al = 0, 0
                if len(got) == len(flat): break
        assert got == [int(s) for s in flat], f"bitstream mismatch cluster {ci}"
    return True

CFG_FULL = dict(bank=E16, KSET=WIDE, refine=True, paths={"Q", "T", "D"}, pathbits=2,
                split=True, rans=True, merge=True, huff_only=False)
def CFG_E6(): return dict(bank=MOE6, KSET=WIDE, refine=True, paths={"Q", "T", "D"}, pathbits=2,
                          split=True, rans=True, merge=True, huff_only=False)

def run_config(tag, cfg, prep_cache, images):
    rows = []
    for path in images:
        t0 = time.time()
        r = process_image(path, cfg, prep_cache)
        r["secs"] = time.time() - t0
        rows.append(r)
        d = r["detail"]
        print(f"{tag} {r['nm']}: {r['bpp']:.4f} path={d['path']} ng={d['ng']} nCl={d['nCl']} "
              f"m={int(d['merged'])} back={d['back']} [{r['secs']:.0f}s]", flush=True)
    tot = sum(r["bits"] for r in rows); dn = sum(r["dn"] for r in rows)
    print(f"==> {tag} AVG={tot / dn:.4f}", flush=True)
    return rows, tot / dn

def ser(o):
    if isinstance(o, dict):
        return {k: ser(v) for k, v in o.items() if k != "root"}
    if isinstance(o, (list, tuple)):
        return [ser(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="all", help="all or comma names")
    ap.add_argument("--skip-proofs", action="store_true")
    ap.add_argument("--out", default="/tmp/opencode/autocompress/experiments/probe_b14_summary.json")
    a = ap.parse_args()
    t_all = time.time()
    print("== probe_b14 prep (E16 RF/RU + BANK5 features, all-7) ==", flush=True)
    prep_cache = prep_all()
    images = IMAGES if a.images == "all" else [p for p in IMAGES if p.split("/")[-1] in a.images.split(",")]
    # anchors
    print("== anchors ==", flush=True)
    rH, aH = run_config("ANCH-H", dict(bank=MOE6, KSET=KSET2, refine=True, paths={"Q"}, pathbits=0,
                                       split=False, rans=False, merge=False, huff_only=True), prep_cache, images)
    rR, aR = run_config("ANCH-R", dict(bank=MOE6, KSET=KSET2, refine=True, paths={"Q"}, pathbits=0,
                                       split=False, rans=True, merge=False, huff_only=False), prep_cache, images)
    print("== FULL stack ==", flush=True)
    rF, aF = run_config("FULL", CFG_FULL, prep_cache, images)
    loos = {}
    loos["LOO1-noLOCO"] = dict(bank=E16, KSET=None, refine=False, paths={"O", "T", "D"}, pathbits=2,
                               split=True, rans=True, merge=True, huff_only=False)
    loos["LOO2-E6"] = CFG_E6()
    loos["LOO3-Honly"] = dict(bank=E16, KSET=WIDE, refine=True, paths={"Q", "T", "D"}, pathbits=2,
                              split=True, rans=False, merge=True, huff_only=True)
    loos["LOO4-noMerge"] = dict(bank=E16, KSET=WIDE, refine=True, paths={"Q", "T", "D"}, pathbits=2,
                                split=True, rans=True, merge=False, huff_only=False)
    loos["LOO5-noTree"] = dict(bank=E16, KSET=WIDE, refine=True, paths={"Q", "D"}, pathbits=1,
                               split=True, rans=True, merge=True, huff_only=False)
    loos["LOO6-noGrid"] = dict(bank=E16, KSET=WIDE, refine=True, paths={"Q", "T"}, pathbits=1,
                               split=True, rans=True, merge=True, huff_only=False)
    loos["LOO7-noSplit"] = dict(bank=E16, KSET=WIDE, refine=True, paths={"Q", "T", "D"}, pathbits=2,
                                split=False, rans=True, merge=True, huff_only=False)
    loo_rows, loo_avg = {}, {}
    for tag, cfg in loos.items():
        print(f"== {tag} ==", flush=True)
        rows, avg = run_config(tag, cfg, prep_cache, images)
        loo_rows[tag] = rows; loo_avg[tag] = avg
    # proofs on FULL
    proofs = {}
    if not a.skip_proofs:
        print("== proofs (FULL-stack winning config) ==", flush=True)
        for r in rF:
            canon_roundtrip_all(r)
            recon_prove(r, prep_cache)
            print(f"  PROOF PASS {r['nm']}", flush=True)
            proofs[r["nm"]] = "PASS"
    out = {
        "anchors": {"ANCH-H": {r["nm"]: r["bpp"] for r in rH}, "ANCH-H-avg": aH,
                    "ANCH-R": {r["nm"]: r["bpp"] for r in rR}, "ANCH-R-avg": aR},
        "FULL": {r["nm"]: r["bpp"] for r in rF}, "FULL-avg": aF,
        "FULL-detail": {r["nm"]: ser(r["detail"]) for r in rF},
        "JXL_E3": JXL_E3,
    }
    for tag, rows in loo_rows.items():
        out[tag] = {r["nm"]: r["bpp"] for r in rows}
        out[tag + "-avg"] = loo_avg[tag]
        out[tag + "-detail"] = {r["nm"]: ser(r["detail"]) for r in rows}
    out["proofs"] = proofs
    json.dump(ser(out), open(a.out, "w"), indent=1)
    print(f"wrote {a.out} [{time.time() - t_all:.0f}s total]", flush=True)

if __name__ == "__main__":
    main()
