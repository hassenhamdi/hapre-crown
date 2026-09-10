"""probe_b17_rctw.py — Attack-1 port: per-group RCT bank + Weighted-form predictor.

Branch: takedown review (survey/bosstakedown/BOSS_TAKEDOWN.md) found JXL's
per-group RCT selection + Weighted predictor are the exploitable weapons behind
the Boss-5 (JXL-e3 3.2291) gap. This probe ports both onto our YCoCg-R/MED frame.

JXL RCT catalogue (verified against JXL docs 2026-09-07):
  rct = 7*perm + t, perm in 0..5 (0:RGB 1:GBR 2:BRG 3:RBG 4:GRB 5:BGR),
  t in 0..6: t0 identity; t1 C-=A; t2 B-=A; t3 B-=A,C-=A;
  t4 B-=floor((A+C)/2); t5 t4+C-=A; t6 YCoCg-R.
Fixed bank B=8 (3-bit ids): C6 C27 C13 C12 C0 C20 C34 C3 — covers every takedown
winner (e3-C27 3.2083, C13, C12, C20) + identity + subtract mode.

Configs (all exact bits, real heapq Huffman, every side byte counted):
  A0  anchor: global C6 (YCoCg-R) + causal MED + order-0 Huffman (~3.58 target)
  A0b sanity: same with old edge-replicate border (must agree <0.5%)
  (a0) global best-of-8 RCT + MED + Huffman (+3b image id) — permutation gain
  (a)  per-block (16/32/64, per-image best +2b) best-of-8 RCT, L1 selection on
       MED residuals; mixed planes re-MEDded; global Huffman/plane + 3b/block
  (b)  Weighted-form: per-group (32/64, per-image best +1b) LS-fit 4-tap integer
       weights on stencil {L,T,TL,TR} (JXL Weighted neighbourhood), divisor 16,
       5 bits/weight (range -16..15, LS solution scaled x16 before rounding),
       Laplacian-MDL gate vs MED; global Huffman/plane + side
  (a+b) (a) then (b) re-fit on mixed planes (selection coupling approx, noted)

Decoder-safety: RCT ids always transmitted (never inferred); predictors use
causal recon only (CROWN2 border doctrine: zero border, row0 copies left,
col0 copies top); integer ops throughout; bit-exact RCT round-trips asserted.

Rules: numpy+PIL only, CPU. bpp = total_bits/(H*W*3). No existing files touched.
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
HERE = "/tmp/opencode/autocompress/experiments"

ANCHOR = 3.58
CROWN2_CITED = 3.2686
JXL_E3 = 3.2291
JXL_E3_PER = {"kodim01.png": 3.3593, "kodim02.png": 3.0611, "kodim05.png": 3.5120,
              "kodim07.png": 2.7310, "kodim13.png": 3.9141, "kodim19.png": 3.2154,
              "kodim23.png": 2.8110}

PERMS = [[0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 2, 1], [1, 0, 2], [2, 1, 0]]
BANK = [(0, 6), (3, 6), (1, 6), (1, 5), (0, 0), (2, 6), (4, 6), (0, 3)]
BANK_NAMES = ["C6", "C27", "C13", "C12", "C0", "C20", "C34", "C3"]
NB = len(BANK)
IDBITS = int(math.ceil(math.log2(NB)))


# ---------------- JXL-style RCT bank (integer, reversible) ----------------
def rct_fwd(px, perm, t):
    p = px[:, :, PERMS[perm]].astype(np.int32)
    A, B, C = p[:, :, 0], p[:, :, 1], p[:, :, 2]
    if t == 0:
        X, Y, Z = A, B, C
    elif t == 1:
        X, Y, Z = A, B, C - A
    elif t == 2:
        X, Y, Z = A, B - A, C
    elif t == 3:
        X, Y, Z = A, B - A, C - A
    elif t == 4:
        X, Y, Z = A, B - np.floor_divide(A + C, 2), C
    elif t == 5:
        X, Y, Z = A, B - np.floor_divide(A + C, 2), C - A
    elif t == 6:  # YCoCg-R on (A,B,C) as (R,G,B); b10-verified formulas
        Co = A - C
        tt = C + np.floor_divide(Co, 2)
        Cg = B - tt
        Y = tt + np.floor_divide(Cg, 2)
        X, Y, Z = Y, Co, Cg
    else:
        raise ValueError(t)
    return np.stack([X, Y, Z], axis=-1)


def rct_inv(yc, perm, t):
    X, Y, Z = yc[:, :, 0].astype(np.int32), yc[:, :, 1].astype(np.int32), yc[:, :, 2].astype(np.int32)
    if t == 0:
        A, B, C = X, Y, Z
    elif t == 1:
        A, B, C = X, Y, Z + X
    elif t == 2:
        A, B, C = X, Y + X, Z
    elif t == 3:
        A, B, C = X, Y + X, Z + X
    elif t == 4:
        A, B, C = X, Y + np.floor_divide(X + Z, 2), Z
    elif t == 5:
        A, B, C = X, Y + X + np.floor_divide(Z, 2), Z + X
    elif t == 6:
        t_ = X - np.floor_divide(Z, 2)
        G = Z + t_
        Bc = t_ - np.floor_divide(Y, 2)
        A, B, C = Y + Bc, G, Bc
    else:
        raise ValueError(t)
    q = np.stack([A, B, C], axis=-1)
    unp = [0, 0, 0]
    for i, j in enumerate(PERMS[perm]):
        unp[j] = i
    return q[:, :, unp].astype(np.uint8)


# ---------------- causal MED (CROWN2 border doctrine, decoder-safe) ----------------
def med_pred(plane):
    p = plane.astype(np.int32)
    H, W = p.shape
    L = np.zeros_like(p)
    L[:, 1:] = p[:, :-1]
    L[:, 0] = np.concatenate([[0], p[:-1, 0]])
    T = np.zeros_like(p)
    T[1:, :] = p[:-1, :]
    T[0, :] = np.concatenate([[0], p[0, :-1]])
    TL = np.zeros_like(p)
    TL[1:, 1:] = p[:-1, :-1]
    mn = np.minimum(L, T)
    mx = np.maximum(L, T)
    return np.where(TL >= mx, mn, np.where(TL <= mn, mx, L + T - TL))


def med_pred_old(plane):  # b10 edge-replicate (anchor-sanity only, not decode-safe)
    p = plane.astype(np.int32)
    H, W = p.shape
    pad = np.pad(p, 1, mode="edge")
    L = pad[1:H + 1, 0:W]
    T = pad[0:H, 1:W + 1]
    TL = pad[0:H, 0:W]
    mn = np.minimum(L, T)
    mx = np.maximum(L, T)
    return np.where(TL >= mx, mn, np.where(TL <= mn, mx, L + T - TL))


def nbhd4(plane):
    """Causal {L,T,TL,TR} planes (CROWN2 borders) for Weighted stencil."""
    p = plane.astype(np.int32)
    L = np.zeros_like(p)
    L[:, 1:] = p[:, :-1]
    L[:, 0] = np.concatenate([[0], p[:-1, 0]])
    T = np.zeros_like(p)
    T[1:, :] = p[:-1, :]
    T[0, :] = np.concatenate([[0], p[0, :-1]])
    TL = np.zeros_like(p)
    TL[1:, 1:] = p[:-1, :-1]
    TR = np.zeros_like(p)
    TR[1:, :-1] = p[:-1, 1:]
    return L, T, TL, TR


# ---------------- Huffman utils (b10-exact ledger) ----------------
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


def exact_plane_bits(res):
    assert int(np.abs(res).max()) <= 1024, "alphabet breach (never clip)"
    vals, cnt = np.unique(res, return_counts=True)
    counts = {int(v): int(n) for v, n in zip(vals.tolist(), cnt.tolist())}
    L = huff_len(counts)
    assert kraft_ok(L)
    return stream_bits(counts, L) + table_bits(len(counts)), counts, L


# ---------------- block helpers ----------------
def block_id_map(H, W, S):
    nbh, nbw = math.ceil(H / S), math.ceil(W / S)
    bm = np.zeros((H, W), dtype=np.int32)
    for bi in range(nbh):
        for bj in range(nbw):
            bm[bi * S:min((bi + 1) * S, H), bj * S:min((bj + 1) * S, W)] = bi * nbw + bj
    return bm, nbh * nbw


# ---------------- Weighted-form: per-group LS integer weights ----------------
def wpred_from_w(cols, w):
    num = cols[:, 0] * w[0] + cols[:, 1] * w[1] + cols[:, 2] * w[2] + cols[:, 3] * w[3]
    return np.floor_divide(num + 8, 16)


def weighted_fit(planes, H, W, GS):
    """Single shared Weighted-form implementation (main + decode proof).

    planes: 3 causal transformed planes. Per GS-group per channel: closed-form
    LS 4-tap fit on {L,T,TL,TR}, quantize x16 to [-16,15] (5b/weight, /16),
    Laplacian-MDL gate vs MED (need N*log2(b_med/b_w) > 21 = 1b flag+20b w).
    Decoder-safe: weights transmitted; prediction from causal recon only.
    Returns (residual_planes, use, wall, side_bits, blockmap, ngroups).
    """
    bm, ng = block_id_map(H, W, GS)
    bflat = bm.ravel()
    wside = 0
    out_res, use_all, wall_all = [], [], []
    for c in range(3):
        plane = planes[c]
        L, T, TL, TR = nbhd4(plane)
        cols = np.stack([L.ravel(), T.ravel(), TL.ravel(), TR.ravel()], axis=1).astype(np.int32)
        y = plane.ravel().astype(np.int32)
        med = med_pred(plane).ravel()
        base_r = y - med
        out = base_r.copy()
        uc, wc = [], []
        for g in range(ng):
            idx = np.where(bflat == g)[0]
            yg = y[idx]
            Cg = cols[idx].astype(np.float64)
            b_med = float(np.mean(np.abs(base_r[idx]))) + 1e-9
            sol, *_ = np.linalg.lstsq(Cg, yg.astype(np.float64), rcond=None)
            wq = np.clip(np.round(sol * 16), -16, 15).astype(np.int32)
            ok = False
            if np.any(wq):
                pred = np.floor_divide((Cg * wq).sum(axis=1) + 8, 16).astype(np.int32)
                rw = yg - pred
                b_w = float(np.mean(np.abs(rw))) + 1e-9
                if b_w < b_med and idx.shape[0] * math.log2(b_med / b_w) > 21:
                    out[idx] = rw
                    wside += 20
                    ok = True
            uc.append(ok)
            wc.append(wq if ok else None)
        out_res.append(out.reshape(H, W))
        use_all.append(uc)
        wall_all.append(wc)
    return out_res, use_all, wall_all, ng * 3 * 1 + wside, bm, ng


def main():
    t_all = time.perf_counter()
    # ---- RCT round-trip asserts (all bank entries, all images + synthetic edge) ----
    edge = np.zeros((8, 8, 3), dtype=np.uint8)
    edge[::2, ::2] = [255, 0, 255]
    edge[1::2, 1::2] = [0, 255, 0]
    edge[0, :] = [255, 255, 255]
    for fn in FILES + [None]:
        px = edge if fn is None else np.array(Image.open(os.path.join(D, fn)).convert("RGB"))
        for bi, (perm, t) in enumerate(BANK):
            assert np.array_equal(rct_inv(rct_fwd(px, perm, t), perm, t), px), \
                f"RCT {BANK_NAMES[bi]} not invertible"
    print(f"[RCT round-trip ALL-8 bank x 7 images + edge-case: PASS]")

    res = {}
    for fn in FILES:
        t0 = time.perf_counter()
        px = np.array(Image.open(os.path.join(D, fn)).convert("RGB"))
        H, W, _ = px.shape
        npx = H * W * 3
        R = {"file": fn, "H": H, "W": W}

        # ---- A0 anchor: global C6 + causal MED + order-0 Huffman ----
        yc0 = rct_fwd(px, 0, 6)
        b0 = 0
        for c in range(3):
            bb, _, _ = exact_plane_bits(yc0[:, :, c] - med_pred(yc0[:, :, c]))
            b0 += bb
        R["A0"] = b0 / npx
        # sanity: old border convention
        b0o = 0
        for c in range(3):
            bb, _, _ = exact_plane_bits(yc0[:, :, c] - med_pred_old(yc0[:, :, c]))
            b0o += bb
        R["A0old"] = b0o / npx

        # ---- candidate-global MED residuals for all 8 (selection currency) ----
        cand_res = []   # per bank: 3 planes of residuals
        cand_L1 = []    # per bank: per-block L1 at S=32 (filled later)
        cand_bits = []
        for bi, (perm, t) in enumerate(BANK):
            yc = rct_fwd(px, perm, t)
            rr, tb = [], 0
            for c in range(3):
                r = yc[:, :, c] - med_pred(yc[:, :, c])
                rr.append(r)
                bb, _, _ = exact_plane_bits(r)
                tb += bb
            cand_res.append(rr)
            cand_bits.append(tb / npx)

        # ---- (a0): global best-of-8 + 3b id ----
        gbest = int(np.argmin(cand_bits))
        R["a0"] = cand_bits[gbest] + IDBITS / npx
        R["a0_id"] = BANK_NAMES[gbest]

        # ---- (a): per-block best-of-8 at S in {16,32,64} ----
        R["a_sizes"] = {}
        for S in (16, 32, 64):
            bm, nblk = block_id_map(H, W, S)
            L1 = np.zeros((NB, nblk))
            for bi in range(NB):
                tot = np.zeros(nblk)
                for c in range(3):
                    tot += np.bincount(bm.ravel(),
                                       weights=np.abs(cand_res[bi][c]).ravel(),
                                       minlength=nblk)
                L1[bi] = tot
            pick = np.argmin(L1, axis=0)
            # assemble mixed planes from per-block chosen RCTs
            fwd_all = [rct_fwd(px, *BANK[bi]) for bi in range(NB)]
            mixed = np.zeros_like(yc0)
            for k in range(nblk):
                m = (bm == k)
                mixed[m] = fwd_all[int(pick[k])][m]
            tb = nblk * IDBITS
            for c in range(3):
                bb, _, _ = exact_plane_bits(mixed[:, :, c] - med_pred(mixed[:, :, c]))
                tb += bb
            R["a_sizes"][S] = tb / npx
            if S == 32:
                R["a_pick_hist"] = {BANK_NAMES[bi]: int((pick == bi).sum()) for bi in range(NB)}
                R["a_mixed32"] = mixed
        bestS = min((16, 32, 64), key=lambda s: R["a_sizes"][s])
        R["a"] = R["a_sizes"][bestS] + 2 / npx
        R["a_S"] = bestS

        # ---- (b): Weighted-form on C6 planes; groups 32/64 ----
        R["b_gs"] = {}
        for GS in (32, 64):
            out_res, _, _, side, _, _ = weighted_fit([yc0[:, :, c] for c in range(3)], H, W, GS)
            tb = side + sum(exact_plane_bits(out_res[c])[0] for c in range(3))
            R["b_gs"][GS] = tb / npx
        bestG = min((32, 64), key=lambda g: R["b_gs"][g])
        R["b"] = R["b_gs"][bestG] + 1 / npx
        R["b_G"] = bestG

        # ---- (a+b): Weighted re-fit on (a)-mixed planes (S=32 mixed kept) ----
        mixed = R.pop("a_mixed32")
        R["ab_gs"] = {}
        a_side_blocks = math.ceil(H / R["a_S"]) * math.ceil(W / R["a_S"]) * IDBITS + 2
        for GS in (32, 64):
            out_res, _, _, side, _, _ = weighted_fit([mixed[:, :, c] for c in range(3)], H, W, GS)
            tb = a_side_blocks + side + 1 + sum(exact_plane_bits(out_res[c])[0] for c in range(3))
            R["ab_gs"][GS] = tb / npx
        bestG2 = min((32, 64), key=lambda g: R["ab_gs"][g])
        R["ab"] = R["ab_gs"][bestG2]
        R["ab_G"] = bestG2

        # ---- (a0+b): global best RCT (homogeneous planes) + Weighted re-fit ----
        ycB = rct_fwd(px, *BANK[gbest])
        R["a0b_gs"] = {}
        for GS in (32, 64):
            out_res, _, _, side, _, _ = weighted_fit([ycB[:, :, c] for c in range(3)], H, W, GS)
            tb = IDBITS + side + 1 + sum(exact_plane_bits(out_res[c])[0] for c in range(3))
            R["a0b_gs"][GS] = tb / npx
        bestG3 = min((32, 64), key=lambda g: R["a0b_gs"][g])
        R["a0b"] = R["a0b_gs"][bestG3]
        R["a0b_G"] = bestG3
        R["ms"] = (time.perf_counter() - t0) * 1000
        res[fn] = R
        print(f"{fn}: A0={R['A0']:.4f} A0old={R['A0old']:.4f} a0={R['a0']:.4f}({R['a0_id']}) "
              f"a={R['a']:.4f}(S{R['a_S']}) b={R['b']:.4f}(G{R['b_G']}) ab={R['ab']:.4f}(G{R['ab_G']}) "
              f"a0b={R['a0b']:.4f}(G{R['a0b_G']}) {R['ms']:.0f}ms", flush=True)

    # ---- summary ----
    def avg(k):
        return float(np.mean([res[fn][k] for fn in FILES]))

    print(f"\nAVG A0={avg('A0'):.4f} (vs 3.58: {(avg('A0')-ANCHOR)/ANCHOR*100:+.2f}%) "
          f"A0old={avg('A0old'):.4f}")
    assert abs(avg("A0") - ANCHOR) / ANCHOR <= 0.03, "anchor out of +-3%"
    print(f"AVG a0={avg('a0'):.4f} a={avg('a'):.4f} b={avg('b'):.4f} ab={avg('ab'):.4f} "
          f"a0b={avg('a0b'):.4f}")
    for k in ("a0", "a", "b", "ab", "a0b"):
        d = (avg(k) - avg("A0")) / avg("A0") * 100
        dc = (avg(k) - CROWN2_CITED) / CROWN2_CITED * 100
        de = (avg(k) - JXL_E3) / JXL_E3 * 100
        print(f"  {k}: {d:+.3f}% vs A0 | {dc:+.3f}% vs CROWN2 {CROWN2_CITED} | {de:+.3f}% vs JXL-e3")
    print("global-a0 picks:", {fn: res[fn]["a0_id"] for fn in FILES})
    print("a block histograms:", {fn: res[fn]["a_pick_hist"] for fn in FILES})

    # ---- winning-config full decode proof (kodim07, best of a/b/ab/a0b) ----
    win = min(("a", "b", "ab", "a0b"), key=lambda k: avg(k))
    print(f"\nWINNER for decode proof: ({win})")
    prove_decode(win, res)

    # ---- write RESULTS.md ----
    write_report(res, avg, win)
    print(f"\nTOTAL {(time.perf_counter()-t_all)/60:.1f} min")


def prove_decode(tag, res):
    fn = "kodim07.png"
    px = np.array(Image.open(os.path.join(D, fn)).convert("RGB"))
    H, W, _ = px.shape
    npx = H * W * 3
    # rebuild the winner exactly as in main() via the SAME shared code paths
    yc0 = rct_fwd(px, 0, 6)
    inv_ids = None
    inv_global = None
    if tag in ("a", "ab"):
        S = res[fn]["a_S"]
        bm, nblk = block_id_map(H, W, S)
        cand = []
        for bi in range(NB):
            yc = rct_fwd(px, *BANK[bi])
            rr = [yc[:, :, c] - med_pred(yc[:, :, c]) for c in range(3)]
            cand.append(rr)
        L1 = np.zeros((NB, nblk))
        for bi in range(NB):
            tot = np.zeros(nblk)
            for c in range(3):
                tot += np.bincount(bm.ravel(), weights=np.abs(cand[bi][c]).ravel(), minlength=nblk)
            L1[bi] = tot
        pick = np.argmin(L1, axis=0)
        fwd_all = [rct_fwd(px, *BANK[bi]) for bi in range(NB)]
        mixed = np.zeros_like(yc0)
        for k in range(nblk):
            m = (bm == k)
            mixed[m] = fwd_all[int(pick[k])][m]
        inv_ids = pick
        planes = [mixed[:, :, c].copy() for c in range(3)]
    elif tag == "a0b":
        gi = BANK_NAMES.index(res[fn]["a0_id"])
        mixed = rct_fwd(px, *BANK[gi])
        inv_global = BANK[gi]
        planes = [mixed[:, :, c].copy() for c in range(3)]
    else:
        planes = [yc0[:, :, c].copy() for c in range(3)]
    use = wall = wbm = None
    out_res = None
    if tag in ("b", "ab", "a0b"):
        GS = res[fn]["b_G" if tag == "b" else ("ab_G" if tag == "ab" else "a0b_G")]
        out_res, use, wall, side, wbm, _wng = weighted_fit(planes, H, W, GS)
    # symbol-level mapping + literal bitstream on plane residuals
    nbytes = 0
    dec_planes = []
    if tag in ("a", "ab", "a0b"):
        base_planes = [mixed[:, :, c] for c in range(3)]
    else:
        base_planes = [yc0[:, :, c] for c in range(3)]
    for c in range(3):
        plane = base_planes[c]
        if tag in ("b", "ab", "a0b"):
            r = out_res[c]
        else:
            r = plane - med_pred(plane)
        bb, counts, LL = exact_plane_bits(r)
        CJ = canonical_codes(LL)
        # literal encode
        buf = bytearray()
        acc = nbits = 0
        for v in r.ravel().tolist():
            cd, ln = CJ[int(v)]
            acc = (acc << ln) | cd
            nbits += ln
            while nbits >= 8:
                nbits -= 8
                buf.append((acc >> nbits) & 0xFF)
                acc &= (1 << nbits) - 1 if nbits else 0
        if nbits:
            buf.append((acc << (8 - nbits)) & 0xFF)
        nbytes += len(buf)
        # literal decode
        DJ = {(ln, cd): s for s, (cd, ln) in CJ.items()}
        bits = "".join(f"{by:08b}" for by in buf)
        pos, mL = 0, max(LL.values())
        syms = []
        for _ in range(H * W):
            cd = 0
            for ln in range(1, mL + 1):
                cd = (cd << 1) | (bits[pos] == "1")
                pos += 1
                if (ln, cd) in DJ:
                    syms.append(DJ[(ln, cd)])
                    break
        assert np.array_equal(np.array(syms, dtype=np.int32), r.ravel()), "stream mismatch"
        # sequential causal inverse-MED
        rec = np.zeros((H, W), dtype=np.int32)
        rr = np.array(syms, dtype=np.int32).reshape(H, W)
        for i in range(H):
            for j in range(W):
                lv = rec[i, j - 1] if j > 0 else (rec[i - 1, 0] if i > 0 else 0)
                tv = rec[i - 1, j] if i > 0 else (rec[i, j - 1] if j > 0 else 0)
                tl = rec[i - 1, j - 1] if i > 0 and j > 0 else 0
                mn, mx = (lv, tv) if lv < tv else (tv, lv)
                pr = mn if tl >= mx else (mx if tl <= mn else lv + tv - tl)
                if tag in ("b", "ab", "a0b"):
                    g = int(wbm[i, j])
                    if use[c][g]:
                        w = wall[c][g]
                        tr = rec[i - 1, j + 1] if i > 0 and j + 1 < W else 0
                        num = int(w[0]) * lv + int(w[1]) * tv + int(w[2]) * tl + int(w[3]) * tr
                        pr = math.floor((num + 8) / 16)
                rec[i, j] = rr[i, j] + pr
        assert np.array_equal(rec, base_planes[c]), f"inverse-predictor mismatch ch{c}"
        dec_planes.append(rec)
    dec_mixed = np.stack(dec_planes, axis=-1)
    if tag in ("a", "ab"):
        bm2, _ = block_id_map(H, W, res[fn]["a_S"])
        # per-block inverse RCT (vectorized per block)
        out = np.zeros_like(px)
        for k in range(int(inv_ids.shape[0])):
            m = (bm2 == k)
            blk = np.zeros((m.sum(), 3), dtype=np.int32)
            for c in range(3):
                blk[:, c] = dec_mixed[:, :, c][m]
            out[m] = rct_inv_block(blk, *BANK[int(inv_ids[k])])
    elif tag == "a0b":
        out = rct_inv(dec_mixed.astype(np.int32), *inv_global)
    else:
        out = rct_inv(dec_mixed.astype(np.int32), 0, 6)
    assert np.array_equal(out, px), "FULL DECODE ROUND-TRIP FAILED"
    print(f"[literal bitstream + sequential inverse-MED + inverse-RCT kodim07 ({tag}): PASS, "
          f"{nbytes} payload bytes]")


def rct_inv_block(blk, perm, t):
    H = blk.shape[0]
    yc = blk.reshape(1, H, 3)
    return rct_inv(yc, perm, t).reshape(H, 3)


def write_report(res, avg, win):
    L = []
    A = lambda k: float(np.mean([res[fn][k] for fn in FILES]))
    L.append("# probe_b17 (Attack-1 port: per-group RCT + Weighted-form) — RESULTS")
    L.append("")
    L.append(f"Anchor A0 reproduced: {A('A0'):.4f} vs 3.58 ({(A('A0')-3.58)/3.58*100:+.2f}%, "
             f"{'PASS' if abs(A('A0')-3.58)/3.58 <= 0.03 else 'FAIL'}); old-border sanity "
             f"{A('A0old'):.4f} (delta {(A('A0old')-A('A0'))/A('A0')*100:+.3f}%).")
    L.append("")
    L.append("| img | A0 | (a0)+id | (a)+side | (b)+side | (a+b)+side | (a0+b)+side | JXL-e3 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for fn in FILES:
        r = res[fn]
        L.append(f"| {fn} | {r['A0']:.4f} | {r['a0']:.4f}({r['a0_id']}) | {r['a']:.4f}(S{r['a_S']}) "
                 f"| {r['b']:.4f}(G{r['b_G']}) | {r['ab']:.4f}(G{r['ab_G']}) | "
                 f"{r['a0b']:.4f}(G{r['a0b_G']}) | {JXL_E3_PER[fn]:.4f} |")
    L.append("")
    L.append(f"| AVG | {A('A0'):.4f} | {A('a0'):.4f} | {A('a'):.4f} | {A('b'):.4f} | {A('ab'):.4f} | "
             f"{A('a0b'):.4f} | {JXL_E3:.4f} |")
    L.append("")
    for k in ("a0", "a", "b", "ab", "a0b"):
        L.append(f"- {k}: {(A(k)-A('A0'))/A('A0')*100:+.3f}% vs A0; "
                 f"{(A(k)-CROWN2_CITED)/CROWN2_CITED*100:+.3f}% vs CROWN2 {CROWN2_CITED}; "
                 f"{(A(k)-JXL_E3)/JXL_E3*100:+.3f}% vs JXL-e3.")
    L.append("")
    L.append("See console log for block histograms, sizes, timings, decode proof.")
    open(os.path.join(HERE, "probe_b17_RESULTS.md"), "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()