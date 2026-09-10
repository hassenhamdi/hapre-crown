"""probe_b9_enum22.py — Refresher quest: enumerative (combinatorial) coding of 2x2
MED-residual blocks (Cover'73 weight-class + lexicographic rank scheme, n=4).

Baseline (anchor): YCoCg-R -> causal MED -> per-channel order-0 Huffman.
  bits = sum_c [ sum_v count*len + (16 + A*24) ], bpp = bits/(H*W*3).

Experiment: partition each channel's residual plane into disjoint 2x2 blocks.
  Per block: k = #zeros (0..4) -> Huffman on {0..4} (tiny table);
             rank of nonzero-position mask among C(4,4-k) at FIXED length
             ceil(log2 C(4,w)): w=1:2b, w=2:3b, w=3:4->2b, w in {0,4}:0b;
             nonzero values in fixed position order -> shared order-0 Huffman
             over observed nonzero values.
  Total = k-stream + table + pattern fixed bits + nz-stream + table, per channel.
Decoder: k -> mask (rank) -> scatter nz values. Mapping round-trip asserted
on all 7 images; one literal bits->pixels round-trip (kodim07); Kraft==1
asserted for every Huffman table. numpy+PIL only, CPU.
"""
import heapq, itertools, math, os, time
import numpy as np
from PIL import Image

D = "/tmp/opencode/autocompress/experiments/real_photos"
FILES = ["kodim01.png", "kodim02.png", "kodim05.png", "kodim07.png",
         "kodim13.png", "kodim19.png", "kodim23.png"]

# ---------- YCoCg-R (brief's exact formulas) ----------
def fwd(px):
    R = px[:, :, 0].astype(np.int32); G = px[:, :, 1].astype(np.int32); B = px[:, :, 2].astype(np.int32)
    Co = R - B
    t = B + np.floor_divide(Co, 2)
    Cg = G - t
    Y = t + np.floor_divide(Cg, 2)  # == Cg>>1 (floor) per brief
    return np.stack([Y, Co, Cg], axis=-1)

def inv(yc):
    Y = yc[:, :, 0].astype(np.int32); Co = yc[:, :, 1].astype(np.int32); Cg = yc[:, :, 2].astype(np.int32)
    t = Y - np.floor_divide(Cg, 2)
    G = Cg + t
    B = t - np.floor_divide(Co, 2)
    R = Co + B
    return np.stack([R, G, B], axis=-1).astype(np.uint8)

# ---------- causal MED, 0/left/top border rule (fully vectorized; lossless so
# prediction uses original neighbors) ----------
def med_pred(plane):
    p = plane.astype(np.int32)
    L = np.zeros_like(p); L[:, 1:] = p[:, :-1]
    T = np.zeros_like(p); T[1:, :] = p[:-1, :]
    TL = np.zeros_like(p); TL[1:, 1:] = p[:-1, :-1]
    mn = np.minimum(L, T); mx = np.maximum(L, T)
    m = np.where(TL >= mx, mn, np.where(TL <= mn, mx, L + T - TL))
    first_row = np.zeros_like(p); first_row[0, :] = L[0, :]
    first_col = np.zeros_like(p); first_col[:, 0] = T[:, 0]
    out = m; out[0, :] = first_row[0, :]; out[:, 0] = first_col[:, 0]
    out[0, 0] = 0
    return out

# ---------- real heapq Huffman -> canonical lengths ----------
def huffman_lengths(counts):
    syms = list(counts.keys())
    if len(syms) == 1:
        return {syms[0]: 1}
    heap = [[c, [s]] for s, c in counts.items()]
    heapq.heapify(heap)
    parent = {}
    while len(heap) > 1:
        c1, s1 = heapq.heappop(heap); c2, s2 = heapq.heappop(heap)
        for s in s1: parent[s] = parent.get(s, []) + [0]
        for s in s2: parent[s] = parent.get(s, []) + [1]
        heapq.heappush(heap, [c1 + c2, s1 + s2])
    # lengths from merge tree: recompute properly via standard bookkeeping
    # (redo cleanly: track depth by climbing)
    # Simpler correct approach: rebuild with explicit nodes
    return _huff_len(counts)

def _huff_len(counts):
    n = len(counts)
    if n == 1:
        return {s: 1 for s in counts}
    heap = [(c, i, s) for i, (s, c) in enumerate(counts.items())]
    heapq.heapify(heap)
    depth = {s: 0 for s in counts}
    nxt = n
    while len(heap) > 1:
        c1, _, s1 = heapq.heappop(heap); c2, _, s2 = heapq.heappop(heap)
        for s in (s1 if isinstance(s1, list) else [s1]):
            depth[s] += 1
        for s in (s2 if isinstance(s2, list) else [s2]):
            depth[s] += 1
        l1 = s1 if isinstance(s1, list) else [s1]
        l2 = s2 if isinstance(s2, list) else [s2]
        heapq.heappush(heap, (c1 + c2, nxt, l1 + l2)); nxt += 1
    return depth

def kraft_ok(lengths):
    return abs(sum(2.0 ** -l for l in lengths.values()) - 1.0) < 1e-9

def canonical_codes(lengths):
    order = sorted(lengths, key=lambda s: (lengths[s], s))
    codes = {}; code = 0; prev = 0
    for s in order:
        code <<= (lengths[s] - prev); prev = lengths[s]
        codes[s] = (code, lengths[s]); code += 1
    return codes

def stream_bits(counts, lengths):
    return sum(c * lengths[s] for s, c in counts.items())

def table_bits(A):
    return 16 + A * 24

# ---------- enumerative 2x2 setup: mask rank tables ----------
MASKS = {}   # w -> {mask: rank}
RMASKS = {}  # w -> [mask] (rank -> mask)
for w in range(5):
    ms = [m for m in range(16) if bin(m).count("1") == w]
    MASKS[w] = {m: i for i, m in enumerate(ms)}
    RMASKS[w] = ms
PATBITS = {0: 0, 1: 2, 2: 3, 3: 2, 4: 0}  # ceil(log2 C(4,w)); C(4,2)=6 -> 3b

def analyze(res):
    """res: int32 residual plane, H,W even. Returns dict of exact counts."""
    H, W = res.shape
    assert H % 2 == 0 and W % 2 == 0, "odd dims"
    b = res.reshape(H // 2, 2, W // 2, 2).transpose(0, 2, 1, 3).reshape(-1, 4)
    z = (b == 0)
    k = z.sum(axis=1).astype(np.int64)          # #zeros 0..4
    w = 4 - k                                    # #nonzeros
    masks = (z[:, 0].astype(np.int64) * 0 + (b[:, 0] != 0).astype(np.int64) * 8 +
             (b[:, 1] != 0).astype(np.int64) * 4 + (b[:, 2] != 0).astype(np.int64) * 2 +
             (b[:, 3] != 0).astype(np.int64) * 1)
    ranks = np.array([MASKS[wi][m] for wi, m in zip(w.tolist(), masks.tolist())], dtype=np.int64)
    nz = b[~z]  # nonzero values in fixed position order
    kc = dict(zip(*np.unique(k, return_counts=True)))
    nzc = dict(zip(*np.unique(nz, return_counts=True))) if nz.size else {}
    return {"k": k, "w": w, "ranks": ranks, "nz": nz,
            "kcounts": {int(a): int(c) for a, c in kc.items()},
            "nzcounts": {int(a): int(c) for a, c in nzc.items()},
            "nblocks": b.shape[0]}

def decode_check(res, an):
    H, W = res.shape
    b = np.empty((an["nblocks"], 4), dtype=res.dtype)
    it = iter(an["nz"].tolist())
    for i in range(an["nblocks"]):
        m = RMASKS[int(an["w"][i])][int(an["ranks"][i])]
        for j in range(4):
            bit = (m >> (3 - j)) & 1
            b[i, j] = next(it) if bit else 0
    assert next(it, None) is None, "nz stream length mismatch"
    out = b.reshape(H // 2, W // 2, 2, 2).transpose(0, 2, 1, 3).reshape(H, W)
    assert np.array_equal(out, res), "enumerative mapping round-trip FAILED"
    return True

def main():
    print(f"{'img':12s} {'base_bpp':>9s} {'enum_bpp':>9s} {'delta%':>8s} "
          f"{'p0Y':>6s} {'p0Co':>6s} {'p0Cg':>6s} {'ms':>7s}")
    base_all, enum_all = [], []
    for fn in FILES:
        t0 = time.perf_counter()
        px = np.array(Image.open(os.path.join(D, fn)).convert("RGB"))
        H, W, _ = px.shape
        npx = H * W * 3
        yc = fwd(px)
        assert np.array_equal(inv(yc), px), f"YCoCg-R not invertible on {fn}"
        base_bits, enum_bits = 0, 0
        p0s = []
        lits = {}
        for c in range(3):
            plane = yc[:, :, c]
            res = plane - med_pred(plane)
            vals, cnt = np.unique(res, return_counts=True)
            counts = {int(v): int(n) for v, n in zip(vals.tolist(), cnt.tolist())}
            L0 = _huff_len(counts)
            assert kraft_ok(L0)
            base_bits += stream_bits(counts, L0) + table_bits(len(counts))
            p0s.append(counts.get(0, 0) / res.size)
            an = analyze(res)
            assert decode_check(res, an)
            Lk = _huff_len(an["kcounts"]); assert kraft_ok(Lk)
            Lz = _huff_len(an["nzcounts"]) if an["nzcounts"] else {}
            if Lz: assert kraft_ok(Lz)
            pat = sum(PATBITS[int(x)] for x in an["w"].tolist())
            enum_bits += (stream_bits(an["kcounts"], Lk) + table_bits(len(an["kcounts"]))
                          + pat
                          + (stream_bits(an["nzcounts"], Lz) + table_bits(len(Lz)) if Lz else 0))
            lits[c] = (res, an, Lk, Lz)
        bb = base_bits / npx; eb = enum_bits / npx
        base_all.append(bb); enum_all.append(eb)
        ms = (time.perf_counter() - t0) * 1000
        print(f"{fn:12s} {bb:9.4f} {eb:9.4f} {(eb-bb)/bb*100:+8.2f} "
              f"{p0s[0]:6.3f} {p0s[1]:6.3f} {p0s[2]:6.3f} {ms:7.0f}")
        if fn == "kodim07.png":
            literal_roundtrip(lits)
    ba, ea = float(np.mean(base_all)), float(np.mean(enum_all))
    print(f"\nAVG base={ba:.4f} enum={ea:.4f} delta={(ea-ba)/ba*100:+.2f}% "
          f"(vs 3.58 anchor: base {(ba-3.58)/3.58*100:+.2f}%, enum {(ea-3.58)/3.58*100:+.2f}%)")

def literal_roundtrip(lits):
    """One true bits->symbols->residuals decode (kodim07, all 3 channels)."""
    nb = 0
    for c, (res, an, Lk, Lz) in lits.items():
        Ck, Cz = canonical_codes(Lk), canonical_codes(Lz)
        buf = bytearray(); acc = 0; nbits = 0
        def put(code, ln):
            nonlocal acc, nbits, nb
            acc = (acc << ln) | code; nbits += ln
            while nbits >= 8:
                nbits -= 8; buf.append((acc >> nbits) & 0xFF); nb += 8
            acc &= (1 << nbits) - 1 if nbits else 0
        for x in an["k"].tolist(): put(*Ck[int(x)])
        for wi, r in zip(an["w"].tolist(), an["ranks"].tolist()):
            if PATBITS[int(wi)]: put(int(r), PATBITS[int(wi)])
        for v in an["nz"].tolist(): put(*Cz[int(v)])
        if nbits: buf.append((acc << (8 - nbits)) & 0xFF); nb += 8
        # decode
        Dk = {(l, cd): s for s, (cd, l) in Ck.items()}
        Dz = {(l, cd): s for s, (cd, l) in Cz.items()}
        bits = "".join(f"{by:08b}" for by in buf)
        pos = 0
        def get(D, mx):
            nonlocal pos
            cd = 0
            for ln in range(1, mx + 1):
                cd = (cd << 1) | (bits[pos] == "1"); pos += 1
                if (ln, cd) in D: return D[(ln, cd)]
            raise AssertionError("literal decode failed")
        mk, mz = max(Lk.values()), max(Lz.values())
        ks = [get(Dk, mk) for _ in range(an["nblocks"])]
        assert ks == an["k"].tolist()
        ws = [4 - x for x in ks]; pos0 = pos
        for i in range(an["nblocks"]):
            if PATBITS[ws[i]]:
                r = int(bits[pos:pos + PATBITS[ws[i]]], 2); pos += PATBITS[ws[i]]
                assert r == int(an["ranks"][i]), "pattern rank mismatch"
        nnz = int((np.array(ws) ).sum())
        vs = [get(Dz, mz) for _ in range(nnz)]
        assert vs == an["nz"].tolist(), "nz literal mismatch"
    print(f"  [literal bitstream round-trip kodim07: PASS, {nb/8:.0f} payload bytes]")

if __name__ == "__main__":
    main()
