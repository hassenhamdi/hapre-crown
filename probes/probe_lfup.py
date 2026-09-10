"""Probe LF-UP (branch 1b): is Squeeze-lite's loss from blocky NN upsampling or structural?

Method (numpy + PIL only, CPU):
 1. YCoCg-R reversible integer transform, invertibility asserted every image.
 2. Baseline per channel: MED residuals (vectorized, edge-replicate pad) +
    EXACT order-0 Huffman bits via heapq code lengths (sum(count*len)+16+A*24).
 3. Squeeze variants (same 4x floor-mean box LF, edge-replicate pad to mult of 4):
    (a) NN upsample (replicate);
    (b) BILINEAR upsample (integer fixed-point, documented below);
    (c) SMOOTHED-NN (3x3 box blur of NN plane, integer floor).
    For each: HF = orig - up; MED+Huffman HF; LF side = Huffman(LF vals)+64 dims bits.
 4. bpp = total_bits/(H*W*3) (correct units).

BILINEAR definition (centered, integer, floor):
 LF sample d[bi,bj] is the floor-mean of the 4x4 block rows bi*4..bi*4+3,
 cols bj*4..bj*4+3, i.e. located at block center (bi*4+2, bj*4+2) in
 full-res coordinates. Output pixel (bi*4+m, bj*4+n), m,n in 0..3, lies
 between neighboring centers spaced 4 apart. 1D linear weights (denom 8):
   m=0: 3/8 prev + 5/8 curr
   m=1: 1/8 prev + 7/8 curr
   m=2: 7/8 curr + 1/8 next
   m=3: 5/8 curr + 3/8 next
 (same for n in columns). 2D weight = row_w * col_w / 64, numerator summed
 in int64 then floor-divided by 64 (numpy // = floor, exact for negatives).
 LF edges replicated (d[-1]=d[0], d[Hd]=d[Hd-1]). Constant planes reproduce
 exactly (weights sum to 64). All integer, deterministic.

SMOOTHED-NN definition: upNN = NN-upsampled plane (HxW); edge-replicate pad
 1px; each output = floor(sum of 3x3 window / 9) via int64 sum // 9.
"""
import heapq
import numpy as np
from PIL import Image

IMAGES = [
    "/tmp/opencode/autocompress/experiments/real_photos/kodim01.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim02.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim05.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim07.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim13.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim19.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim23.png",
]

TABLE_HDR = 16
TABLE_PER_SYM = 24
DIMS_OVERHEAD_BITS = 8 * 8  # 8 bytes dims, counted once globally


def ycocg_r_forward(rgb):
    R = rgb[:, :, 0].astype(np.int32)
    G = rgb[:, :, 1].astype(np.int32)
    B = rgb[:, :, 2].astype(np.int32)
    Co = R - B
    t = B + (Co // 2)  # floor div2 (arithmetic shift)
    Cg = G - t
    Y = t + (Cg // 2)
    return Y, Co, Cg


def ycocg_r_inverse(Y, Co, Cg):
    t = Y - (Cg // 2)
    B = t - (Co // 2)
    G = Cg + t
    R = Co + B
    return R, G, B


def med_residuals(ch):
    p = np.pad(ch, 1, mode="edge")
    a = p[1:-1, :-2]   # left
    b = p[:-2, 1:-1]   # top
    c = p[:-2, :-2]    # topleft
    mx = np.maximum(a, b)
    mn = np.minimum(a, b)
    pred = np.where(c >= mx, mn, np.where(c <= mn, mx, a + b - c))
    return (ch - pred).astype(np.int32)


def huffman_bits(vals):
    """EXACT order-0 Huffman bits: sum(count*len) + 16 + A*24."""
    _, counts = np.unique(vals, return_counts=True)
    A = len(counts)
    if A == 0:
        return 0
    if A == 1:
        return int(counts[0]) * 1 + TABLE_HDR + A * TABLE_PER_SYM
    heap = [int(c) for c in counts]
    heapq.heapify(heap)
    data = 0
    while len(heap) > 1:
        x = heapq.heappop(heap)
        y = heapq.heappop(heap)
        data += x + y
        heapq.heappush(heap, x + y)
    return data + TABLE_HDR + A * TABLE_PER_SYM


def box_down4(x):
    H, W = x.shape
    Hp = (H + 3) // 4 * 4
    Wp = (W + 3) // 4 * 4
    xp = np.pad(x, ((0, Hp - H), (0, Wp - W)), mode="edge")
    blocks = xp.reshape(Hp // 4, 4, Wp // 4, 4)
    s = blocks.sum(axis=(1, 3), dtype=np.int64)
    return (s // 16).astype(np.int32)


def nn_up4(d, H, W):
    return np.repeat(np.repeat(d, 4, axis=0), 4, axis=1)[:H, :W].astype(np.int32)


# 1D phase weights: (w0, w1, off0, off1), denom 8
_ROW_W = {
    0: (3, 5, -1, 0),
    1: (1, 7, -1, 0),
    2: (7, 1, 0, 1),
    3: (5, 3, 0, 1),
}


def bilinear_up4(d, H, W):
    """Centered bilinear 4x upsample, integer floor (see module docstring)."""
    d = d.astype(np.int64)
    Hd, Wd = d.shape
    Hp, Wp = Hd * 4, Wd * 4
    dp = np.pad(d, 1, mode="edge")  # (Hd+2)x(Wd+2), dp[bi+1,bj+1]==d[bi,bj]
    up = np.empty((Hp, Wp), dtype=np.int32)
    for m in range(4):
        rw0, rw1, ro0, ro1 = _ROW_W[m]
        for n in range(4):
            cw0, cw1, co0, co1 = _ROW_W[n]
            v00 = dp[1 + ro0:1 + ro0 + Hd, 1 + co0:1 + co0 + Wd]
            v01 = dp[1 + ro0:1 + ro0 + Hd, 1 + co1:1 + co1 + Wd]
            v10 = dp[1 + ro1:1 + ro1 + Hd, 1 + co1:1 + co1 + Wd] if False else dp[1 + ro1:1 + ro1 + Hd, 1 + co0:1 + co0 + Wd]
            v11 = dp[1 + ro1:1 + ro1 + Hd, 1 + co1:1 + co1 + Wd]
            # NOTE: v10 uses (ro1,co0); written explicitly above for clarity
            num = rw0 * cw0 * v00 + rw0 * cw1 * v01 + rw1 * cw0 * v10 + rw1 * cw1 * v11
            up[m::4, n::4] = (num // 64).astype(np.int32)
    return up[:H, :W]


def smoothed_nn_up4(d, H, W):
    """3x3 box blur (integer floor) of the NN-upsampled plane."""
    up = nn_up4(d, H, W).astype(np.int64)
    p = np.pad(up, 1, mode="edge")
    s = (p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:] +
         p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:] +
         p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:])
    return (s // 9).astype(np.int32)


def main():
    print("bpp = total_bits/(H*W*3). Sanity anchor: baseline avg ~3.5-3.7 bpp.")
    print(f"{'image':12s} {'HxW':>9s} {'base':>7s} {'NN':>7s} {'dNN%':>7s} {'BILIN':>7s} {'dBI%':>7s} {'SMOOTH':>7s} {'dSM%':>7s}")
    rows = []
    for path in IMAGES:
        name = path.rsplit("/", 1)[-1]
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        H, W, _ = rgb.shape

        Y, Co, Cg = ycocg_r_forward(rgb)
        Ri, Gi, Bi = ycocg_r_inverse(Y, Co, Cg)
        assert np.array_equal(Ri, rgb[:, :, 0].astype(np.int32)) \
            and np.array_equal(Gi, rgb[:, :, 1].astype(np.int32)) \
            and np.array_equal(Bi, rgb[:, :, 2].astype(np.int32)), \
            f"YCoCg-R NOT invertible on {name}"

        chans = (Y, Co, Cg)
        denom = H * W * 3

        base_bits = sum(huffman_bits(med_residuals(c)) for c in chans)
        base_bpp = base_bits / denom

        # LF planes (shared across variants)
        lfs = [box_down4(c) for c in chans]
        side_bits = DIMS_OVERHEAD_BITS + sum(huffman_bits(d) for d in lfs)

        totals = {}
        for key, fn in (("nn", nn_up4), ("bilin", bilinear_up4), ("smooth", smoothed_nn_up4)):
            hf_bits = 0
            for c, d in zip(chans, lfs):
                up = fn(d, H, W)
                hf = (c.astype(np.int32) - up.astype(np.int32)).astype(np.int32)
                hf_bits += huffman_bits(med_residuals(hf))
            total = hf_bits + side_bits
            totals[key] = (total, hf_bits, total / denom)

        nn_bpp = totals["nn"][2]
        bi_bpp = totals["bilin"][2]
        sm_bpp = totals["smooth"][2]
        d_nn = (nn_bpp - base_bpp) / base_bpp * 100.0
        d_bi = (bi_bpp - base_bpp) / base_bpp * 100.0
        d_sm = (sm_bpp - base_bpp) / base_bpp * 100.0
        rows.append(dict(name=name, H=H, W=W, base_bits=base_bits, base_bpp=base_bpp,
                         nn_bits=totals["nn"][0], nn_bpp=nn_bpp, d_nn=d_nn,
                         bi_bits=totals["bilin"][0], bi_bpp=bi_bpp, d_bi=d_bi,
                         sm_bits=totals["smooth"][0], sm_bpp=sm_bpp, d_sm=d_sm,
                         side_bits=side_bits))
        print(f"{name:12s} {H}x{W:>4d} {base_bpp:7.4f} {nn_bpp:7.4f} {d_nn:+7.2f}% {bi_bpp:7.4f} {d_bi:+7.2f}% {sm_bpp:7.4f} {d_sm:+7.2f}%", flush=True)

    avg_base = float(np.mean([r["base_bpp"] for r in rows]))
    avg_nn = float(np.mean([r["nn_bpp"] for r in rows]))
    avg_bi = float(np.mean([r["bi_bpp"] for r in rows]))
    avg_sm = float(np.mean([r["sm_bpp"] for r in rows]))
    dnn = (avg_nn - avg_base) / avg_base * 100.0
    dbi = (avg_bi - avg_base) / avg_base * 100.0
    dsm = (avg_sm - avg_base) / avg_base * 100.0
    print(f"\nAVERAGE(7): base={avg_base:.4f} bpp NN={avg_nn:.4f} ({dnn:+.2f}%) "
          f"BILIN={avg_bi:.4f} ({dbi:+.2f}%) SMOOTH={avg_sm:.4f} ({dsm:+.2f}%)")
    # sanity anchor check
    if not (3.5 * 0.85 <= avg_base <= 3.7 * 1.15):
        print(f"WARNING: baseline {avg_base:.4f} outside 15% of 3.5-3.7 anchor — debug before trusting deltas.")
    else:
        print("Sanity anchor PASS: baseline within 15% of 3.5-3.7 bpp.")
    # NN replication check
    print(f"NN replication: expect ~+13.5%; got {dnn:+.2f}%.")


if __name__ == "__main__":
    main()
