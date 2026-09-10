"""probe_b10_magvq.py — Vector/lattice coding of ACTIVE-BLOCK MAGNITUDES.

Branch from probe_b9 post-mortem: joint info lives in MAGNITUDES-given-activity,
not activity patterns (b9 Cover-73 2x2 position coding died at +0.32%).

Frame: YCoCg-R -> causal MED (edge-replicate pad, standard rule) -> disjoint
blocks (1x2 pairs / 2x2 quads). Flat (all-zero) blocks need NO extra flag
stream: the joint alphabet contains the all-zero tuple, which earns the
shortest codeword automatically (fragmentation discipline: one table per
channel, no per-class tables).

Configs (per channel, MDL-gated vs scalar baseline):
  P{h,v}-T : 1x2 pair MAGNITUDE vector Huffman, cap T on max(|a|,|b|).
             in-cap pairs -> symbol a*(T+1)+b; over-cap -> single ESC symbol.
             signs of in-cap nonzeros at fixed 1b each; over-cap pairs fall
             back to scalar signed Huffman (own tail table, counted).
  Q-T      : 2x2 quad magnitude vector Huffman, cap T on max|.|, same structure.
  S-T      : 1x2 SIGNED-pair vector Huffman (signs joint in the tuple), cap T.
  PYR-R    : Fischer-1986-style pyramid/lattice shell code on 2x2 blocks:
             r = sum|mi| Huffman-coded (symbols 0..R + ESC); position of the
             signed vector on shell r at fixed ceil(log2 N(r)) bits, N(r) =
             #{m in Z^4 : |m|_1 = r}; over-radius blocks -> scalar tail.
Diagnostics: pair magnitude MI headroom, escape rates, sign-pair entropy,
  Huffman slack + table taxes (exact ledger everywhere).

Counting: real heapq Huffman -> lengths; stream = sum c*len; every Huffman
table costs 16 + A*24 bits; fixed bits counted exactly. Gating is per
image-channel: gated_c = min(base_c, exp_c) (codebook shipped ONLY if
net-positive). bpp = total_bits/(H*W*3). numpy+PIL only, CPU.
"""
import heapq
import math
import os
import time

import numpy as np
from PIL import Image

D = "/tmp/opencode/autocompress/experiments/real_photos"
FILES = ["kodim01.png", "kodim02.png", "kodim05.png", "kodim07.png",
         "kodim13.png", "kodim19.png", "kodim23.png"]

ANCHOR = 3.58
ANCHOR_PER = {"kodim01.png": 3.62, "kodim02.png": 3.33, "kodim05.png": 4.01,
              "kodim07.png": 3.15, "kodim13.png": 4.24, "kodim19.png": 3.51,
              "kodim23.png": 3.19}


# ---------- YCoCg-R (brief's exact formulas) ----------
def fwd(px):
    R = px[:, :, 0].astype(np.int32)
    G = px[:, :, 1].astype(np.int32)
    B = px[:, :, 2].astype(np.int32)
    Co = R - B
    t = B + np.floor_divide(Co, 2)
    Cg = G - t
    Y = t + np.floor_divide(Cg, 2)
    return np.stack([Y, Co, Cg], axis=-1)


def inv(yc):
    Y = yc[:, :, 0].astype(np.int32)
    Co = yc[:, :, 1].astype(np.int32)
    Cg = yc[:, :, 2].astype(np.int32)
    t = Y - np.floor_divide(Cg, 2)
    G = Cg + t
    B = t - np.floor_divide(Co, 2)
    R = Co + B
    return np.stack([R, G, B], axis=-1).astype(np.uint8)


# ---------- causal MED, edge-replicate pad, standard rule ----------
def med_pred(plane):
    p = plane.astype(np.int32)
    H, W = p.shape
    pad = np.pad(p, 1, mode="edge")
    L = pad[1:H + 1, 0:W]
    T = pad[0:H, 1:W + 1]
    TL = pad[0:H, 0:W]
    mn = np.minimum(L, T)
    mx = np.maximum(L, T)
    return np.where(TL >= mx, mn, np.where(TL <= mn, mx, L + T - TL))


# ---------- real heapq Huffman -> lengths ----------
def huff_len(counts):
    n = len(counts)
    if n == 1:
        return {s: 1 for s in counts}
    heap = [(c, i, s) for i, (s, c) in enumerate(counts.items())]
    heapq.heapify(heap)
    depth = {s: 0 for s in counts}
    nxt = n
    while len(heap) > 1:
        c1, _, s1 = heapq.heappop(heap)
        c2, _, s2 = heapq.heappop(heap)
        for s in (s1 if isinstance(s1, list) else [s1]):
            depth[s] += 1
        for s in (s2 if isinstance(s2, list) else [s2]):
            depth[s] += 1
        l1 = s1 if isinstance(s1, list) else [s1]
        l2 = s2 if isinstance(s2, list) else [s2]
        heapq.heappush(heap, (c1 + c2, nxt, l1 + l2))
        nxt += 1
    return depth


def kraft_ok(lengths):
    return abs(sum(2.0 ** -l for l in lengths.values()) - 1.0) < 1e-9


def canonical_codes(lengths):
    order = sorted(lengths, key=lambda s: (lengths[s], s))
    codes = {}
    code = 0
    prev = 0
    for s in order:
        code <<= (lengths[s] - prev)
        prev = lengths[s]
        codes[s] = (code, lengths[s])
        code += 1
    return codes


def stream_bits(counts, lengths):
    return sum(c * lengths[s] for s, c in counts.items())


def table_bits(A):
    return 16 + A * 24


def entropy_bits(counts):
    n = sum(counts.values())
    return -sum(c * math.log2(c / n) for c in counts.values())


# ---------- baseline: scalar order-0 Huffman over signed residuals ----------
def baseline_channel(res):
    vals, cnt = np.unique(res, return_counts=True)
    counts = {int(v): int(n) for v, n in zip(vals.tolist(), cnt.tolist())}
    L = huff_len(counts)
    assert kraft_ok(L)
    return {"counts": counts, "L": L,
            "bits": stream_bits(counts, L) + table_bits(len(counts))}


# ---------- pair magnitude VQ (1x2 horiz or 2x1 vert) ----------
def pair_mag_channel(res, T, vertical=False):
    H, W = res.shape
    assert H % 2 == 0 and W % 2 == 0
    if vertical:
        b = res.reshape(H // 2, 2, W).transpose(0, 2, 1).reshape(-1, 2)
    else:
        b = res.reshape(H, W // 2, 2).reshape(-1, 2)
    m = np.abs(b)
    incap = m.max(axis=1) <= T
    K = T + 1
    ESC = K * K
    sym = np.where(incap, m[:, 0] * K + m[:, 1], ESC)
    vals, cnt = np.unique(sym, return_counts=True)
    jcounts = {int(v): int(n) for v, n in zip(vals.tolist(), cnt.tolist())}
    JL = huff_len(jcounts)
    assert kraft_ok(JL)
    jstream = stream_bits(jcounts, JL)
    jtab = table_bits(len(jcounts))
    signbits = int(((b[incap] != 0).sum()))  # 1 fixed bit per in-cap nonzero
    escc = b[~incap]
    if escc.size:
        tv, tc = np.unique(escc, return_counts=True)
        tcounts = {int(v): int(n) for v, n in zip(tv.tolist(), tc.tolist())}
        TL = huff_len(tcounts)
        assert kraft_ok(TL)
        tstream = stream_bits(tcounts, TL)
        ttab = table_bits(len(tcounts))
    else:
        tcounts, TL, tstream, ttab = {}, {}, 0, 0
    exp_bits = jstream + jtab + signbits + tstream + ttab
    # diagnostics: infinite-precision MI headroom on magnitudes
    uu, cc = np.unique(m, axis=0, return_counts=True)
    jnt = {tuple(u.tolist()): int(c) for u, c in zip(uu, cc)}
    n = m.shape[0]
    H1 = -sum((c / n) * math.log2(c / n) for c in
              dict(zip(*np.unique(m[:, 0], return_counts=True))).values())
    H2 = -sum((c / n) * math.log2(c / n) for c in
              dict(zip(*np.unique(m[:, 1], return_counts=True))).values())
    Hj = -sum((c / n) * math.log2(c / n) for c in jnt.values())
    mi = H1 + H2 - Hj
    pescape = float((~incap).mean())
    return {"exp_bits": exp_bits, "jstream": jstream, "jtab": jtab,
            "signbits": signbits, "tstream": tstream, "ttab": ttab,
            "A_joint": len(jcounts), "A_tail": len(tcounts),
            "JL": JL, "TL": TL, "sym": sym, "incap": incap, "pairs": b,
            "blocks": b, "n": 2,
            "MI": mi, "H1": H1, "H2": H2, "Hj": Hj, "pescape": pescape,
            "ESC": ESC, "T": T}


# ---------- quad magnitude VQ (2x2, 1x4 row-strip, or 4x1 col-strip) ----------
def quad_mag_channel(res, T, geom="2x2"):
    H, W = res.shape
    assert H % 2 == 0 and W % 2 == 0
    if geom == "2x2":
        b = res.reshape(H // 2, 2, W // 2, 2).transpose(0, 2, 1, 3).reshape(-1, 4)
    elif geom == "1x4":
        assert W % 4 == 0
        b = res.reshape(H, W // 4, 4).reshape(-1, 4)
    elif geom == "4x1":
        assert H % 4 == 0
        b = res.reshape(H // 4, 4, W).transpose(0, 2, 1).reshape(-1, 4)
    else:
        raise ValueError(geom)
    m = np.abs(b)
    incap = m.max(axis=1) <= T
    K = T + 1
    ESC = K ** 4
    sym = np.where(incap, ((m[:, 0] * K + m[:, 1]) * K + m[:, 2]) * K + m[:, 3], ESC)
    vals, cnt = np.unique(sym, return_counts=True)
    jcounts = {int(v): int(n) for v, n in zip(vals.tolist(), cnt.tolist())}
    JL = huff_len(jcounts)
    assert kraft_ok(JL)
    jstream = stream_bits(jcounts, JL)
    jtab = table_bits(len(jcounts))
    signbits = int(((b[incap] != 0).sum()))
    escc = b[~incap]
    if escc.size:
        tv, tc = np.unique(escc, return_counts=True)
        tcounts = {int(v): int(n) for v, n in zip(tv.tolist(), tc.tolist())}
        TL = huff_len(tcounts)
        assert kraft_ok(TL)
        tstream = stream_bits(tcounts, TL)
        ttab = table_bits(len(tcounts))
    else:
        tcounts, TL, tstream, ttab = {}, {}, 0, 0
    return {"exp_bits": jstream + jtab + signbits + tstream + ttab,
            "jstream": jstream, "jtab": jtab, "signbits": signbits,
            "tstream": tstream, "ttab": ttab, "A_joint": len(jcounts),
            "A_tail": len(tcounts), "JL": JL, "TL": TL, "sym": sym,
            "incap": incap, "quads": b, "blocks": b, "n": 4, "geom": geom,
            "pescape": float((~incap).mean()), "ESC": ESC, "T": T}


# ---------- signed-pair VQ (signs joint in tuple) ----------
def pair_signed_channel(res, T):
    H, W = res.shape
    b = res.reshape(H, W // 2, 2).reshape(-1, 2)
    incap = np.abs(b).max(axis=1) <= T
    K = 2 * T + 1
    ESC = K * K
    sym = np.where(incap, (b[:, 0] + T) * K + (b[:, 1] + T), ESC)
    vals, cnt = np.unique(sym, return_counts=True)
    jcounts = {int(v): int(n) for v, n in zip(vals.tolist(), cnt.tolist())}
    JL = huff_len(jcounts)
    assert kraft_ok(JL)
    jstream = stream_bits(jcounts, JL)
    jtab = table_bits(len(jcounts))
    escc = b[~incap]
    if escc.size:
        tv, tc = np.unique(escc, return_counts=True)
        tcounts = {int(v): int(n) for v, n in zip(tv.tolist(), tc.tolist())}
        TL = huff_len(tcounts)
        assert kraft_ok(TL)
        tstream = stream_bits(tcounts, TL)
        ttab = table_bits(len(tcounts))
    else:
        tcounts, TL, tstream, ttab = {}, {}, 0, 0
    return {"exp_bits": jstream + jtab + tstream + ttab,
            "A_joint": len(jcounts), "A_tail": len(tcounts),
            "pescape": float((~incap).mean())}


# ---------- pyramid / lattice-shell code (Fischer-style), 2x2 ----------
def shell_size(n, r):
    # #{m in Z^n : |m|_1 = r}
    if r == 0:
        return 1
    tot = 0
    for k in range(1, min(n, r) + 1):
        tot += math.comb(n, k) * math.comb(r - 1, k - 1) * (2 ** k)
    return tot


def build_shell(n, r):
    # deterministic lexicographic enumeration of signed shell vectors
    out = []

    def rec(pos, rem, cur):
        if pos == n - 1:
            for s in ([0] if rem == 0 else ([-rem, rem] if rem > 0 else [])):
                out.append(tuple(cur + [s]))
            return
        for a in range(-rem, rem + 1):
            rec(pos + 1, rem - abs(a), cur + [a])

    rec(0, r, [])
    assert len(out) == shell_size(n, r), (n, r, len(out), shell_size(n, r))
    idx = {v: i for i, v in enumerate(out)}
    return out, idx


def pyramid_channel(res, Rmax):
    H, W = res.shape
    b = res.reshape(H // 2, 2, W // 2, 2).transpose(0, 2, 1, 3).reshape(-1, 4)
    m = np.abs(b)
    r = m.sum(axis=1)
    incap = r <= Rmax
    ESC = Rmax + 1
    rsym = np.where(incap, r, ESC)
    vals, cnt = np.unique(rsym, return_counts=True)
    rcounts = {int(v): int(n) for v, n in zip(vals.tolist(), cnt.tolist())}
    RL = huff_len(rcounts)
    assert kraft_ok(RL)
    rstream = stream_bits(rcounts, RL)
    rtab = table_bits(len(rcounts))
    # fixed-length shell positions (signed shells: signs inside the index)
    Ns = {rr: shell_size(4, rr) for rr in range(Rmax + 1)}
    fl = {rr: (0 if Ns[rr] <= 1 else int(math.ceil(math.log2(Ns[rr])))) for rr in Ns}
    posbits = int(sum(fl[int(x)] for x in r[incap].tolist()))
    shells = {rr: build_shell(4, rr) for rr in range(Rmax + 1)}
    ranks = {}
    for rr in range(Rmax + 1):
        _, idx = shells[rr]
        sel = b[(r == rr) & incap]
        ranks[rr] = np.array([idx[tuple(v.tolist())] for v in sel], dtype=np.int64)
    escc = b[~incap]
    if escc.size:
        tv, tc = np.unique(escc, return_counts=True)
        tcounts = {int(v): int(n) for v, n in zip(tv.tolist(), tc.tolist())}
        TL = huff_len(tcounts)
        assert kraft_ok(TL)
        tstream = stream_bits(tcounts, TL)
        ttab = table_bits(len(tcounts))
    else:
        tcounts, TL, tstream, ttab = {}, {}, 0, 0
    return {"exp_bits": rstream + rtab + posbits + tstream + ttab,
            "rstream": rstream, "rtab": rtab, "posbits": posbits,
            "tstream": tstream, "ttab": ttab, "A_rad": len(rcounts),
            "A_tail": len(tcounts), "RL": RL, "TL": TL, "r": r,
            "incap": incap, "quads": b, "fl": fl, "Ns": Ns, "shells": shells,
            "ranks": ranks, "Rmax": Rmax, "ESC": ESC,
            "pescape": float((~incap).mean())}


# ---------- sign-pair diagnostic: H(S1,S2) on doubly-nonzero pairs ----------
def signpair_entropy(res):
    b = res.reshape(res.shape[0], res.shape[1] // 2, 2).reshape(-1, 2)
    sel = b[(b[:, 0] != 0) & (b[:, 1] != 0)]
    if sel.shape[0] == 0:
        return 2.0, 0
    s = (np.sign(sel) + 1) // 2  # -> 0/2 mapping: -1->0, +1->1
    s = ((np.sign(sel[:, 0]) > 0).astype(int) * 2 +
         (np.sign(sel[:, 1]) > 0).astype(int))
    _, cc = np.unique(s, return_counts=True)
    n = cc.sum()
    H = -sum((c / n) * math.log2(c / n) for c in cc)
    return H, int(n)


# ---------- generic literal round-trip for the mag-split VQ family ----------
def magsym_to_mags(sym, K, n, ESC):
    if sym == ESC:
        return None
    digs = [0] * n
    for i in range(n - 1, -1, -1):
        digs[i] = sym % K
        sym //= K
    return digs


def literal_roundtrip_mag(res, e, tag):
    """Full bits->blocks->residuals encode/decode. Returns payload bytes."""
    blocks, n = e["blocks"], e["n"]
    K, ESC = e["T"] + 1, e["ESC"]
    CJ = canonical_codes(e["JL"])
    CT = canonical_codes(e["TL"]) if e["TL"] else {}
    buf = bytearray()
    acc = 0
    nbits = 0
    nb = [0]

    def put(code, ln):
        nonlocal acc, nbits
        acc = (acc << ln) | code
        nbits += ln
        while nbits >= 8:
            nbits -= 8
            buf.append((acc >> nbits) & 0xFF)
            nb[0] += 8
        acc &= (1 << nbits) - 1 if nbits else 0

    syms = e["sym"].tolist()
    incap = e["incap"].tolist()
    rows = blocks.tolist()
    for s in syms:
        put(*CJ[int(s)])
    for row, ok in zip(rows, incap):
        if ok:
            for v in row:
                if v != 0:
                    put(1 if v > 0 else 0, 1)
    for row, ok in zip(rows, incap):
        if not ok:
            for v in row:
                put(*CT[int(v)])
    if nbits:
        buf.append((acc << (8 - nbits)) & 0xFF)
        nb[0] += 8
    # ---- decode ----
    DJ = {(l, cd): s for s, (cd, l) in CJ.items()}
    DT = {(l, cd): s for s, (cd, l) in CT.items()}
    bits = "".join(f"{by:08b}" for by in buf)
    pos = [0]

    def get(D, mx):
        cd = 0
        for ln in range(1, mx + 1):
            cd = (cd << 1) | (bits[pos[0]] == "1")
            pos[0] += 1
            if (ln, cd) in D:
                return D[(ln, cd)]
        raise AssertionError(f"{tag}: literal decode failed")

    mJ = max(e["JL"].values())
    dsyms = [get(DJ, mJ) for _ in range(len(rows))]
    assert dsyms == [int(s) for s in syms], f"{tag}: joint stream mismatch"
    out = [None] * len(rows)
    for i, (s, row, ok) in enumerate(zip(dsyms, rows, incap)):
        if not ok:
            continue
        mags = magsym_to_mags(int(s), K, n, ESC)
        assert mags is not None
        rec = []
        for mm, v in zip(mags, row):
            assert mm == abs(v), f"{tag}: magnitude mismatch"
            if mm == 0:
                rec.append(0)
            else:
                b = bits[pos[0]]
                pos[0] += 1
                rec.append(mm if b == "1" else -mm)
        out[i] = rec
    # tail values for escaped blocks, fixed position order (same order as encode)
    for i, (s, row, ok) in enumerate(zip(dsyms, rows, incap)):
        if ok:
            continue
        assert int(s) == ESC
        if CT:
            mT = max(e["TL"].values())
            out[i] = [get(DT, mT) for _ in range(n)]
        else:
            out[i] = list(row)
    # tail-plane consistency: escaped values must match originals
    assert np.array_equal(np.array(out, dtype=np.int32),
                          np.array(rows, dtype=np.int32)), f"{tag}: round-trip FAILED"
    H, W = res.shape
    if n == 2 and blocks.shape[0] == H * (W // 2):
        back = np.array(out, dtype=np.int32).reshape(H, W)
    else:
        back = None
    if back is not None:
        assert np.array_equal(back, res), f"{tag}: reshape round-trip FAILED"
    return nb[0] / 8


def mapping_roundtrip_mag(res, e, tag):
    """Symbol-level mapping check (all images): syms+signs+tail -> blocks."""
    blocks, n = e["blocks"], e["n"]
    K, ESC = e["T"] + 1, e["ESC"]
    rows = blocks.tolist()
    for s, row, ok in zip(e["sym"].tolist(), rows, e["incap"].tolist()):
        if ok:
            mags = magsym_to_mags(int(s), K, n, ESC)
            assert mags is not None and all(m == abs(v) for m, v in zip(mags, row)), \
                f"{tag}: mapping mismatch"
        else:
            assert int(s) == ESC, f"{tag}: escape flag mismatch"
    return True


def main():
    cfgs = [("Ph-T1", lambda r: pair_mag_channel(r, 1)),
            ("Ph-T2", lambda r: pair_mag_channel(r, 2)),
            ("Ph-T3", lambda r: pair_mag_channel(r, 3)),
            ("Ph-T4", lambda r: pair_mag_channel(r, 4)),
            ("Ph-T6", lambda r: pair_mag_channel(r, 6)),
            ("Ph-T8", lambda r: pair_mag_channel(r, 8)),
            ("Pv-T2", lambda r: pair_mag_channel(r, 2, vertical=True)),
            ("Q-T1", lambda r: quad_mag_channel(r, 1)),
            ("Q-T2", lambda r: quad_mag_channel(r, 2)),
            ("Q-T3", lambda r: quad_mag_channel(r, 3)),
            ("Q-T4", lambda r: quad_mag_channel(r, 4)),
            ("Q-T5", lambda r: quad_mag_channel(r, 5)),
            ("H4-T2", lambda r: quad_mag_channel(r, 2, geom="1x4")),
            ("H4-T3", lambda r: quad_mag_channel(r, 3, geom="1x4")),
            ("V4-T2", lambda r: quad_mag_channel(r, 2, geom="4x1")),
            ("S-T2", lambda r: pair_signed_channel(r, 2)),
            ("PYR-R4", lambda r: pyramid_channel(r, 4)),
            ("PYR-R6", lambda r: pyramid_channel(r, 6))]
    base_bpps, exp_gated = {c[0]: [] for c in cfgs}, []
    exp_raw = []
    mi_acc = {"Y": [], "Co": [], "Cg": []}
    esc_acc = {c[0]: [] for c in cfgs}
    sp_acc = {"Y": [], "Co": [], "Cg": []}
    chn = ["Y", "Co", "Cg"]
    print(f"{'img':12s} {'base':>7s} " +
          " ".join(f"{c[0]:>11s}" for c in cfgs))
    for fn in FILES:
        t0 = time.perf_counter()
        px = np.array(Image.open(os.path.join(D, fn)).convert("RGB"))
        H, W, _ = px.shape
        npx = H * W * 3
        yc = fwd(px)
        assert np.array_equal(inv(yc), px), f"YCoCg-R not invertible on {fn}"
        bb_tot = 0
        gated_tot = {c[0]: 0 for c in cfgs}
        raw_tot = {c[0]: 0 for c in cfgs}
        for c in range(3):
            plane = yc[:, :, c]
            res = plane - med_pred(plane)
            bc = baseline_channel(res)["bits"]
            bb_tot += bc
            for name, fun in cfgs:
                e = fun(res)
                raw_tot[name] += e["exp_bits"]
                gated_tot[name] += min(bc, e["exp_bits"])
            if c == 0:
                pass
            # diagnostics from Ph-T2 + sign pairs
            d = pair_mag_channel(res, 2)
            mi_acc[chn[c]].append(d["MI"])
            h, _ = signpair_entropy(res)
            sp_acc[chn[c]].append(h)
        for name, _ in cfgs:
            esc_acc[name].append(None)
        for name, _ in cfgs:
            gated_tot[name] += 3  # honest MDL gate flag: 1 bit per channel
        bb = bb_tot / npx
        base_bpps_list = bb
        exp_gated.append({n: gated_tot[n] / npx for n, _ in cfgs})
        exp_raw.append({n: raw_tot[n] / npx for n, _ in cfgs})
        exp_gated_last = exp_gated[-1]
        ms = (time.perf_counter() - t0) * 1000
        row = (f"{fn:12s} {bb:7.4f} " +
               " ".join(f"{exp_gated_last[n[0]]:11.4f}" for n in cfgs) +
               f" {ms:6.0f}ms")
        print(row)
        exp_gated_store = exp_gated_last
    print()
    base_all = []
    for fn in FILES:
        px = np.array(Image.open(os.path.join(D, fn)).convert("RGB"))
        H, W, _ = px.shape
        npx = H * W * 3
        yc = fwd(px)
        tot = 0
        for c in range(3):
            tot += baseline_channel(yc[:, :, c] - med_pred(yc[:, :, c]))["bits"]
        base_all.append(tot / npx)
    ba = float(np.mean(base_all))
    print(f"AVG base={ba:.4f} (vs 3.58 anchor: {(ba - ANCHOR) / ANCHOR * 100:+.2f}%)")
    for n, _ in cfgs:
        g = float(np.mean([e[n] for e in exp_gated]))
        r = float(np.mean([e[n] for e in exp_raw]))
        print(f"  {n:8s} gated={g:.4f} ({(g - ba) / ba * 100:+.3f}% vs base) "
              f"raw-ungated={r:.4f} ({(r - ba) / ba * 100:+.3f}%)")
    print()
    for c in chn:
        print(f"  MI(|x1|,|x2|) {c}: avg {float(np.mean(mi_acc[c])):.4f} b/pair "
              f"({float(np.mean(mi_acc[c])) / 2:.4f} b/residual)")
    for c in chn:
        print(f"  H(sign1,sign2|both nz) {c}: avg {float(np.mean(sp_acc[c])):.4f} b "
              f"(max 2.0)")
    assert abs(ba - ANCHOR) / ANCHOR <= 0.03, "anchor out of +-3%"
    # ---- winner deep-dive: per-channel breakdown + round-trips ----
    win = min(cfgs, key=lambda c: float(np.mean([e[c[0]] for e in exp_gated])))[0]
    wfun = dict(cfgs)[win]
    print(f"\nWINNER {win}: per-channel breakdown (bits, ESC% | A_joint/A_tail | "
          f"tables bits | choice)")
    wsum_base, wsum_gate = 0, 0
    for fn in FILES:
        px = np.array(Image.open(os.path.join(D, fn)).convert("RGB"))
        H, W, _ = px.shape
        npx = H * W * 3
        yc = fwd(px)
        tot_b, tot_g = 0, 0
        detail = []
        for c in range(3):
            res = yc[:, :, c] - med_pred(yc[:, :, c])
            bc = baseline_channel(res)["bits"]
            e = wfun(res)
            g = min(bc, e["exp_bits"])
            tot_b += bc
            tot_g += g
            detail.append(f"{chn[c]}:b={bc} e={e['exp_bits']} "
                          f"esc={e['pescape'] * 100:.1f}% "
                          f"A={e['A_joint']}/{e['A_tail']} "
                          f"tab={e['jtab'] + e['ttab']} "
                          f"{'VQ' if e['exp_bits'] < bc else 'SCAL'}")
        tot_g += 3
        wsum_base += tot_b / npx
        wsum_gate += tot_g / npx
        print(f"  {fn:12s} base={tot_b / npx:.4f} gated={tot_g / npx:.4f} | " +
              " | ".join(detail))
        # symbol-mapping round-trip on every image, all channels
        for c in range(3):
            res = yc[:, :, c] - med_pred(yc[:, :, c])
            mapping_roundtrip_mag(res, wfun(res), f"{win}/{fn}/ch{c}")
    print(f"  WINNER-AVG base={wsum_base / 7:.4f} gated={wsum_gate / 7:.4f} "
          f"delta={(wsum_gate - wsum_base) / wsum_base * 100:+.3f}%")
    print("  [mapping round-trip ALL 7 images x 3 channels: PASS]")
    # one literal bitstream round-trip (kodim07, all channels)
    px = np.array(Image.open(os.path.join(D, "kodim07.png")).convert("RGB"))
    yc = fwd(px)
    nbytes = 0
    for c in range(3):
        res = yc[:, :, c] - med_pred(yc[:, :, c])
        nbytes += literal_roundtrip_mag(res, wfun(res), f"{win}/kodim07/ch{c}")
    print(f"  [literal bitstream round-trip kodim07 ({win}): PASS, "
          f"{nbytes:.0f} payload bytes]")


if __name__ == "__main__":
    main()
