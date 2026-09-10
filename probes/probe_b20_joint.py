"""probe_b20_joint.py — JOINT (group x activity) backward-adaptive coding (new file only).

Branch: b19 proved global adaptive states win on MED frame (-2.18% A-act, -1.74%
C-gol); CROWN5 build PROVED transfer is ~0% (21/21 static picks, ~100% overlap:
WIDE-K per-group static is strictly finer than 4-bin global adaptive).
TRUE construction: JOINT states — fine LOCO groups (conditioning sharpness)
with ZERO transmitted tables (adaptation learned on the fly, decoder mirrors).

Front end (FIXED for all configs, pure entropy-coding test):
  RGB -> C6 YCoCg-R (B17.rct_fwd perm0 t6) -> causal MED (B17.med_pred,
  CROWN2 border doctrine) -> sign-flip by LOCO-365 (b4_c.loco_ctx365 sign s,
  stored sym = s*res; decoder-visible, CROWN convention) -> per-plane symbols.

Groups: per-plane LOCO-365 quantile groups (b18 greedy equal-count, K sweep).
Map side (counted for grouped configs, static AND adaptive — fair):
  16 + 736 + nActive*ceil(log2 K) bits/plane (b18/CROWN ledger verbatim).

Configs (identical symbols, deltas are pure entropy-coding effects):
  S0  global order-0 Huffman/plane (B17.exact_plane_bits, tables counted).
  SG-K  static grouped: per-group best-of {Huffman(+table), Golomb best-k 0..12
        + best bias d in -4..3 (+4b k, +3b d)} + 1b choice/group + map side.
  CG  global adaptive Golomb (4 act ctx, raster N/A, RESET) on same sym.
  AA  global adaptive binary arithmetic (b19 16-ctx, real coder bytes) on sym.
  JG-K joint (group x act) adaptive Golomb: nctx=K*4 states, raster order,
      init N=1/A=4, RESET halving. ZERO tables (map side only).
  JA-K joint (group x act x pos) adaptive binary arithmetic: nctx=K*16 binary
      states + raw suffix bypass, real BinEnc/BinDec bytes. ZERO tables.

M mapping (probe convention throughout b20): M = 2|r|-(r>0).
Deviation from b18/CROWN static-G mapping (2v/-2v-1) noted in RESULTS;
within-b20 comparisons are self-consistent (same mapping everywhere).

Decoder-safety (explicit): every state (N/A/k per joint ctx; c0/c1 per joint
binary ctx; group LUT from transmitted map; activity bin; MED pred; LOCO
key/sign) is a pure function of already-decoded pixels/symbols (+ transmitted
map + static init/RESET/threshold schedules). Proven by real decodes (§round-trip).

Unit: bpp = total_bits/(H*W*3). numpy+PIL only, CPU, no torch.
"""
import math
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "/tmp/opencode/autocompress/csrc")
sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
os.chdir("/tmp/opencode/autocompress/experiments")
import probe_b17_rctw as B17
import probe_b19_adaptive as AD
from probe_b4_a import huff_bits
from probe_b4_b import nbhd as nbhd_base
from probe_b4_c import loco_ctx365

D = "/tmp/opencode/autocompress/experiments/real_photos"
FILES = ["kodim01.png", "kodim02.png", "kodim05.png", "kodim07.png",
         "kodim13.png", "kodim19.png", "kodim23.png"]
JXL_E3 = {"kodim01.png": 3.3593, "kodim02.png": 3.0611, "kodim05.png": 3.5120,
          "kodim07.png": 2.7310, "kodim13.png": 3.9141, "kodim19.png": 3.2154,
          "kodim23.png": 2.8110}
CROWN4 = {"kodim01.png": 3.2991, "kodim02.png": 2.9854, "kodim05.png": 3.5917,
          "kodim07.png": 2.6790, "kodim13.png": 3.9525, "kodim19.png": 3.1510,
          "kodim23.png": 2.7364}
NKEY = 729
ATHR = (4, 12, 48)


def r_to_M(r):
    r = int(r)
    return 2 * abs(r) - (1 if r > 0 else 0)


def M_to_r(M):
    M = int(M)
    if M == 0:
        return 0
    if M & 1:
        return (M + 1) // 2
    return -(M // 2)


def quantile_groups(key, metric, K):
    uk, cn = np.unique(key, return_counts=True)
    ma = np.array([np.abs(metric[key == v]).mean() for v in uk])
    order = np.argsort(ma, kind="stable")
    uks, cns = uk[order], cn[order]
    tot = int(cns.sum())
    tgt = tot / K
    groups, cur, acc = [], [], 0
    for u, c in zip(uks.tolist(), cns.tolist()):
        cur.append(int(u))
        acc += int(c)
        if acc >= tgt and len(groups) < K - 1:
            groups.append(cur)
            cur, acc = [], 0
    groups.append(cur)
    return groups


def build_lut(groups):
    lut = np.full(NKEY, -1, dtype=np.int32)
    for g, keys in enumerate(groups):
        for k in keys:
            lut[int(k)] = g
    return lut


def map_side_bits(n_active, K):
    gbits = int(math.ceil(math.log2(K))) if K > 1 else 1
    return 16 + 736 + int(n_active) * gbits


def pack_map(groups, K):
    """Real map bytes: 92B active-key bitmask + MSB-first group ids. Returns bytes."""
    active = np.zeros(NKEY, dtype=np.uint8)
    lut = build_lut(groups)
    for k in range(NKEY):
        if lut[k] >= 0:
            active[k] = 1
    mask = np.packbits(active).tobytes()  # 92 bytes (736 bits)
    assert len(mask) == 92
    gbits = int(math.ceil(math.log2(K))) if K > 1 else 1
    acc = 0
    nb = 0
    out = bytearray()
    for k in range(NKEY):
        if active[k]:
            acc = (acc << gbits) | int(lut[k])
            nb += gbits
            while nb >= 8:
                nb -= 8
                out.append((acc >> nb) & 0xFF)
                acc &= ((1 << nb) - 1) if nb else 0
    if nb:
        out.append((acc << (8 - nb)) & 0xFF)
    return bytes(mask) + bytes(out)


def unpack_map(buf, K):
    """Invert pack_map -> LUT. Proves map framing is real/decodable."""
    mask = np.unpackbits(np.frombuffer(buf[:92], dtype=np.uint8))
    active = mask[:NKEY].astype(np.int32)
    gbits = int(math.ceil(math.log2(K))) if K > 1 else 1
    bits = []
    for by in bytearray(buf[92:]):
        for i in range(7, -1, -1):
            bits.append((by >> i) & 1)
    lut = np.full(NKEY, -1, dtype=np.int32)
    p = 0
    for k in range(NKEY):
        if active[k]:
            v = 0
            for _ in range(gbits):
                v = (v << 1) | bits[p]
                p += 1
            lut[k] = v
    return lut


def plane_symbols(ch):
    """Fixed front end per plane. Returns dict of flat/raster arrays + planes."""
    ch = np.asarray(ch, dtype=np.int32)
    assert int(np.abs(ch).max()) <= 32767
    pred = B17.med_pred(ch)
    res = (ch - pred).astype(np.int32)
    assert int(np.abs(res).max()) <= 1024, "alphabet breach (never clip)"
    a, b, c, d, _, _, _ = nbhd_base(ch)
    key, s = loco_ctx365(a, b, c, d)
    sym = (s.astype(np.int32) * res).astype(np.int32)
    e = AD.activity_plane(ch)
    ab = np.digitize(e.ravel(), [5, 13, 49], right=False).astype(np.int32)
    M = (2 * np.abs(sym) - (sym > 0)).astype(np.int64).ravel()
    return {"sym": sym, "symf": sym.ravel().tolist(),
            "absf": np.abs(sym).ravel().astype(np.int64).tolist(),
            "M": M.tolist(), "key": key, "keyf": key.ravel().tolist(),
            "s": s, "ab": ab.tolist(), "act": e,
            "pred": pred, "ch": ch}


# ---------------- static grouped ----------------
def static_grouped_plane_bits(symf, keyf, K, metric2d, H, W):
    """Full static grouped plane bits incl map side + 1b/group choice. Returns dict."""
    key = np.asarray(keyf, dtype=np.int32)
    groups = quantile_groups(np.asarray(keyf, dtype=np.int32).reshape(H, W),
                             np.asarray(metric2d).reshape(H, W), K)
    lut = build_lut(groups)
    n_active = int(sum(1 for g in groups for _ in g))
    ms = map_side_bits(n_active, K)
    sym = np.asarray(symf, dtype=np.int64)
    data = 0
    tbl = 0
    npicks = {"H": 0, "G": 0}
    ngroups = 0
    for gkeys in groups:
        sel = np.isin(key, np.array(gkeys, dtype=np.int32))
        g = sym[sel]
        if g.size == 0:
            continue
        ngroups += 1
        _, cn = np.unique(g, return_counts=True)
        h_data = huff_bits(cn.tolist())
        h_tot = h_data + 16 + len(cn) * 24
        M0 = np.where(g > 0, 2 * g - 1, np.where(g < 0, -2 * g, 0))
        best_g = None
        for k in range(13):
            t = int(np.sum(M0 >> np.int64(k))) + g.size * (1 + k) + 4
            if best_g is None or t < best_g[0]:
                best_g = [t, k, 0]
        for dd in (-4, -3, -2, -1, 1, 2, 3):
            gd = g - dd
            Md = np.where(gd > 0, 2 * gd - 1, np.where(gd < 0, -2 * gd, 0))
            for k in range(13):
                t = int(np.sum(Md >> np.int64(k))) + gd.size * (1 + k) + 4 + 3
                if t < best_g[0]:
                    best_g = [t, k, dd]
        if h_tot <= best_g[0]:
            data += h_tot
            tbl += 16 + len(cn) * 24
            npicks["H"] += 1
        else:
            data += best_g[0]
            npicks["G"] += 1
    data += ngroups  # 1b choice per non-empty group
    return {"bits": data + ms, "data": data, "tables": tbl, "map": ms,
            "picks": npicks, "groups": groups, "lut": lut, "ngroups": ngroups}


# ---------------- joint (group x act) adaptive Golomb ----------------
def jg_analytic(Mlist, absf, ab, gid, K, reset=64, A0=4, N0=1):
    """Exact JG code lengths over mirrored state sequence. Zero tables."""
    nctx = K * 4
    N = [N0] * nctx
    A = [A0] * nctx
    tot = 0
    k_hist = [0] * nctx
    cnt = [0] * nctx
    for M, abv, av, g in zip(Mlist, ab, absf, gid):
        ctx = int(g) * 4 + int(abv)
        Nv = N[ctx]
        Av = A[ctx]
        k = 0
        while (Nv << k) < Av:
            k += 1
        tot += (int(M) >> k) + 1 + k
        k_hist[ctx] += k
        cnt[ctx] += 1
        A[ctx] = Av + int(av)
        Nv += 1
        if Nv >= reset:
            Nv >>= 1
            A[ctx] >>= 1
            if Nv < 1:
                Nv = 1
        N[ctx] = Nv
    return {"bits": tot, "cnt": cnt, "kmean": [k_hist[i] / cnt[i] if cnt[i] else 0.0 for i in range(nctx)]}


def jg_encode_stream(Mlist, absf, ab, gid, K, reset=64):
    """Real bitstream (CROWN polarity zeros+one). Returns (bytes, nbits)."""
    nctx = K * 4
    N = [1] * nctx
    A = [4] * nctx
    acc = 0
    nb = 0
    buf = bytearray()
    tot = 0

    def put(v, n):
        nonlocal acc, nb
        acc = (acc << n) | (v & ((1 << n) - 1))
        nb += n
        while nb >= 8:
            nb -= 8
            buf.append((acc >> nb) & 0xFF)
            acc &= ((1 << nb) - 1) if nb else 0

    for M, abv, av, g in zip(Mlist, ab, absf, gid):
        ctx = int(g) * 4 + int(abv)
        Nv = N[ctx]
        Av = A[ctx]
        k = 0
        while (Nv << k) < Av:
            k += 1
        M = int(M)
        q = M >> k
        for _ in range(q):
            put(0, 1)
        put(1, 1)
        if k:
            put(M & ((1 << k) - 1), k)
        tot += q + 1 + k
        A[ctx] = Av + int(av)
        Nv += 1
        if Nv >= reset:
            Nv >>= 1
            A[ctx] >>= 1
            if Nv < 1:
                Nv = 1
        N[ctx] = Nv
    if nb:
        buf.append((acc << (8 - nb)) & 0xFF)
    return bytes(buf), tot


def jg_decode_stream(buf, ab, gid, K, n, reset=64):
    """Real decode: bits -> M list (caller maps M->sym). Mirrors encoder states."""
    bits = []
    for by in bytearray(buf):
        for i in range(7, -1, -1):
            bits.append((by >> i) & 1)
    nctx = K * 4
    N = [1] * nctx
    A = [4] * nctx
    p = 0
    out = []
    for t in range(n):
        ctx = int(gid[t]) * 4 + int(ab[t])
        Nv = N[ctx]
        Av = A[ctx]
        k = 0
        while (Nv << k) < Av:
            k += 1
        q = 0
        while bits[p] == 0:
            q += 1
            p += 1
        p += 1
        rem = 0
        for _ in range(k):
            rem = (rem << 1) | bits[p]
            p += 1
        M = (q << k) | rem
        out.append(M)
        A[ctx] = Av + abs(M_to_r(M))
        Nv += 1
        if Nv >= reset:
            Nv >>= 1
            A[ctx] >>= 1
            if Nv < 1:
                Nv = 1
        N[ctx] = Nv
    return out


def jg_decode_plane(buf, H, W, lut, K, reset=64):
    """Real raster-order full JG decode from recon only. Returns (sym, rec).

    Recomputes MED pred, LOCO key/sign, activity bin, group from causal recon;
    reads the joint Golomb stream with mirrored (N/A/k) states.
    """
    bits = []
    for by in bytearray(buf):
        for i in range(7, -1, -1):
            bits.append((by >> i) & 1)
    nctx = K * 4
    N = [1] * nctx
    A = [4] * nctx
    p = 0
    rec = np.zeros((H, W), dtype=np.int32)
    sym = np.zeros((H, W), dtype=np.int32)
    for i in range(H):
        for j in range(W):
            Lv = int(rec[i, j - 1]) if j > 0 else (int(rec[i - 1, 0]) if i > 0 else 0)
            Tv = int(rec[i - 1, j]) if i > 0 else (int(rec[i, j - 1]) if j > 0 else 0)
            TLv = int(rec[i - 1, j - 1]) if (i > 0 and j > 0) else 0
            if i == 0 and j > 0:
                bv, cv = Lv, Lv
            elif j == 0 and i > 0:
                bv, cv = Tv, Tv
            elif i == 0 and j == 0:
                bv, cv = 0, 0
            else:
                bv, cv = Tv, TLv
            av = Lv
            if i == 0:
                TRv = av
            elif j == W - 1:
                TRv = bv
            else:
                TRv = int(rec[i - 1, j + 1])
            key, s = scalar_key_sign(av, bv, cv, TRv)
            g = int(lut[key])
            assert g >= 0, "inactive key at decode"
            e = abs(Lv - TLv) + abs(Tv - TLv)
            ab = 0 if e <= 4 else (1 if e <= 12 else (2 if e <= 48 else 3))
            ctx = g * 4 + ab
            Nv = N[ctx]
            Av = A[ctx]
            k = 0
            while (Nv << k) < Av:
                k += 1
            q = 0
            while bits[p] == 0:
                q += 1
                p += 1
            p += 1
            rem = 0
            for _ in range(k):
                rem = (rem << 1) | bits[p]
                p += 1
            M = (q << k) | rem
            r = M_to_r(M)
            mn = Lv if Lv < Tv else Tv
            mx = Tv if Lv < Tv else Lv
            pr = mn if TLv >= mx else (mx if TLv <= mn else Lv + Tv - TLv)
            sym[i, j] = r
            rec[i, j] = int(pr) + int(s) * int(r)
            A[ctx] = Av + abs(int(r))
            Nv += 1
            if Nv >= reset:
                Nv >>= 1
                A[ctx] >>= 1
                if Nv < 1:
                    Nv = 1
            N[ctx] = Nv
    return sym, rec


# ---------------- joint (group x act x pos) adaptive arithmetic ----------------
def ja_encode_plane(symf, ab, gid, K):
    """Real integer range coder, nctx=K*16 binary states + raw bypass. Returns dict."""
    nctx = K * 16
    enc = AD.BinEnc(nctx)
    raw_acc = 0
    raw_nb = 0
    raw_out = bytearray()
    for rv, abv, g in zip(symf, ab, gid):
        base = (int(g) * 4 + int(abv)) * 4
        M = 2 * abs(int(rv)) - (1 if int(rv) > 0 else 0)
        cn = M + 1
        L = cn.bit_length()
        for i in range(L):
            b = 0 if i < L - 1 else 1
            pp = i if i < 4 else 3
            enc.enc(b, base + pp)
        if L > 1:
            rem = cn ^ (1 << (L - 1))
            for j in range(L - 2, -1, -1):
                raw_acc = (raw_acc << 1) | ((rem >> j) & 1)
                raw_nb += 1
                if raw_nb == 8:
                    raw_out.append(raw_acc & 0xFF)
                    raw_acc = 0
                    raw_nb = 0
    arith = enc.finish()
    if raw_nb:
        raw_out.append((raw_acc << (8 - raw_nb)) & 0xFF)
    return {"arith": arith, "raw": bytes(raw_out),
            "bits": len(arith) * 8 + len(raw_out) * 8 + 64}


# scalar causal helpers (must match vectorized frame; asserted in driver)
def loco_q_s(g):
    a = abs(int(g))
    if g == 0:
        return 0
    s = 1 if g > 0 else -1
    if a <= 2:
        return s
    if a <= 7:
        return 2 * s
    if a <= 21:
        return 3 * s
    return 4 * s


def scalar_key_sign(Lv, Tv, TLv, TRv):
    """b4-doctrine scalars -> (key, s). Row0/col0 handling mirrors nbhd_base."""
    q1 = loco_q_s(Tv - TLv)      # b-c with b=T? b4: a=left,b=top,c=diag,d=topright
    q2 = loco_q_s(TLv - Lv)      # c-a
    q3 = loco_q_s(TRv - Tv)      # d-b
    neg = (q1 < 0) or ((q1 == 0) and (q2 < 0)) or ((q1 == 0) and (q2 == 0) and (q3 < 0))
    s = -1 if neg else 1
    m1 = -q1 if neg else q1
    m2 = -q2 if neg else q2
    m3 = -q3 if neg else q3
    return int(m1 * 81 + m2 * 9 + m3), s


def ja_decode_plane(arith, raw, H, W, lut, K):
    """Real raster-order joint decode from recon only. Returns (sym_plane, rec_plane)."""
    nctx = K * 16
    assert int(lut.max()) < K, "group id out of range"
    dec = AD.BinDec(arith, nctx)
    rawbits = []
    for by in bytearray(raw):
        for i in range(7, -1, -1):
            rawbits.append((by >> i) & 1)
    rp = 0
    rec = np.zeros((H, W), dtype=np.int32)
    sym = np.zeros((H, W), dtype=np.int32)
    for i in range(H):
        for j in range(W):
            # B17-doctrine L/T/TL (med_pred borders) + b4-doctrine a,b,c,d
            Lv = int(rec[i, j - 1]) if j > 0 else (int(rec[i - 1, 0]) if i > 0 else 0)
            Tv = int(rec[i - 1, j]) if i > 0 else (int(rec[i, j - 1]) if j > 0 else 0)
            TLv = int(rec[i - 1, j - 1]) if (i > 0 and j > 0) else 0
            # b4 a,b,c,d scalars (= L/T/TL/TR under respective fixes):
            av = Lv
            if i == 0 and j > 0:
                bv = Lv
                cv = Lv
            elif j == 0 and i > 0:
                bv = Tv
                cv = Tv
            elif i == 0 and j == 0:
                bv = 0
                cv = 0
            else:
                bv = Tv
                cv = TLv
            if i == 0:
                TRv = av
            elif j == W - 1:
                TRv = bv
            else:
                TRv = int(rec[i - 1, j + 1])
            key, s = scalar_key_sign(av, bv, cv, TRv)
            g = int(lut[key])
            assert g >= 0, "inactive key at decode"
            e = abs(Lv - TLv) + abs(Tv - TLv)
            ab = 0 if e <= 4 else (1 if e <= 12 else (2 if e <= 48 else 3))
            base = (g * 4 + ab) * 4
            L = 1
            while True:
                pp = (L - 1) if (L - 1) < 4 else 3
                b = dec.dec(base + pp)
                if b == 1:
                    break
                L += 1
                assert L <= 16, "prefix runaway"
            rem = 0
            for _ in range(L - 1):
                rem = (rem << 1) | rawbits[rp]
                rp += 1
            cn = (1 << (L - 1)) | rem
            M = cn - 1
            r = M_to_r(M)
            mn = Lv if Lv < Tv else Tv
            mx = Tv if Lv < Tv else Lv
            pr = mn if TLv >= mx else (mx if TLv <= mn else Lv + Tv - TLv)
            sym[i, j] = r
            rec[i, j] = int(pr) + int(s) * int(r)
    return sym, rec
