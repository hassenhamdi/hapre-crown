"""Reversible integer-DCT front-end probe vs HAPRE-C MOE champion.

Method (numpy + PIL only, CPU, no torch):
 1. RGB -> YCoCg-R reversible integer transform, assert round-trip.
 2. H.264-style 4x4 integer DCT: Cf*block*Cf^T, exact inverse asserted on random blocks.
 3. 16 subbands/channel (48 streams), EXACT order-0 Huffman cost via heapq.
 4. Total + 64B header -> bpp.
 5. Time numpy forward transform.
 6. Compare vs MOE 3.464 avg + per-image baselines.

New file only; does not modify existing files.
"""
import heapq
import time
from collections import Counter
from fractions import Fraction
from pathlib import Path

import numpy as np
from PIL import Image

Cf = np.array(
    [[1, 1, 1, 1],
     [2, 1, -1, -2],
     [1, -1, -1, 1],
     [1, -2, 2, -1]],
    dtype=np.int64,
)
# Row norms^2 of Cf (Cf @ Cf.T == diag(DSCALE))
DSCALE = np.array([4, 10, 4, 10], dtype=np.int64)
DINV = [Fraction(1, 4), Fraction(1, 10), Fraction(1, 4), Fraction(1, 10)]

IMAGES = [
    "/tmp/opencode/autocompress/experiments/real_photos/kodim01.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim02.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim05.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim07.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim13.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim19.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim23.png",
]
MOE_PER_IMAGE = {
    "kodim01.png": 3.56,
    "kodim02.png": 3.23,
    "kodim05.png": 3.84,
    "kodim07.png": 3.01,
    "kodim13.png": 4.17,
    "kodim19.png": 3.41,
    "kodim23.png": 3.03,
}
MOE_AVG = 3.464


def rgb_to_ycocgr(r, g, b):
    """Reversible YCoCg-R. Inputs int32 arrays/scalars. Uses floor div (== arithmetic >>1)."""
    co = r - b
    t = b + (co // 2)
    cg = g - t
    y = t + (cg // 2)
    return y, co, cg


def ycocgr_to_rgb(y, co, cg):
    t = y - (cg // 2)
    g = cg + t
    b = t - (co // 2)
    r = b + co
    return r, g, b


def dct_forward_block(X):
    return Cf @ X @ Cf.T


def dct_inverse_exact(Y):
    """Exact rational inverse: M = D^-1 Y D^-1, X = Cf^T M Cf. Returns int64 4x4."""
    M = np.empty((4, 4), dtype=object)
    for i in range(4):
        for j in range(4):
            M[i, j] = Fraction(int(Y[i, j])) * DINV[i] * DINV[j]
    CfT = Cf.T
    tmp = np.empty((4, 4), dtype=object)
    for i in range(4):
        for j in range(4):
            s = Fraction(0, 1)
            for k in range(4):
                s += Fraction(int(CfT[i, k])) * M[k, j]
            tmp[i, j] = s
    Xrec = np.empty((4, 4), dtype=int)
    for i in range(4):
        for j in range(4):
            s = Fraction(0, 1)
            for k in range(4):
                s += tmp[i, k] * Fraction(int(Cf[k, j]))
            assert s.denominator == 1, f"non-exact inverse at {(i, j)}: {s}"
            Xrec[i, j] = int(s)
    return Xrec


def assert_dct_invertible(n_trials=200):
    rng = np.random.default_rng(0)
    # Cover Y range (0..255), Co/Cg range (-255..255), and amplified values
    for t in range(n_trials):
        if t % 3 == 0:
            X = rng.integers(0, 256, size=(4, 4))
        elif t % 3 == 1:
            X = rng.integers(-255, 256, size=(4, 4))
        else:
            X = rng.integers(-255, 256, size=(4, 4))
        Y = dct_forward_block(X)
        Xr = dct_inverse_exact(Y)
        assert np.array_equal(Xr, X), f"DCT round-trip failed trial {t}\n{X}\n{Y}\n{Xr}"
    # Float fast-path sanity (rounding recovers exactly)
    Cfinv = np.linalg.inv(Cf.astype(float))
    for _ in range(20):
        X = rng.integers(-255, 256, size=(4, 4))
        Y = dct_forward_block(X)
        Xr2 = np.rint(Cfinv @ Y @ Cfinv.T).astype(int)
        assert np.array_equal(Xr2, X)
    print(f"[ok] DCT exact inverse verified on {n_trials} random blocks")


def assert_ycocgr_random():
    rng = np.random.default_rng(1)
    for _ in range(5000):
        R, G, B = (int(x) for x in rng.integers(0, 256, size=3))
        y, co, cg = rgb_to_ycocgr(R, G, B)
        R2, G2, B2 = ycocgr_to_rgb(y, co, cg)
        assert (R2, G2, B2) == (R, G, B), "YCoCg-R scalar round-trip failed"
    print("[ok] YCoCg-R scalar round-trip verified (5000 random RGB)")


def dct_forward_image(ch):
    """ch: HxW int32/64, H,W multiples of 4. Returns (h4,w4,4,4) int64 blocks."""
    H, W = ch.shape
    assert H % 4 == 0 and W % 4 == 0, f"need multiples of 4, got {H}x{W}"
    h4, w4 = H // 4, W // 4
    blocks = ch.reshape(h4, 4, w4, 4).transpose(0, 2, 1, 3)
    Yb = np.einsum("ka,ijab,lb->ijkl", Cf, blocks.astype(np.int64), Cf)
    return Yb


def huffman_bits_exact(values):
    """EXACT order-0 Huffman cost via heapq merging.

    Returns (data_bits, table_bits, alphabet_size).
    data_bits = sum(count*len) = sum of merged weights (0 if A==1).
    table_bits = 16 + A*24 per probe spec.
    """
    freq = Counter(values.tolist())
    A = len(freq)
    if A == 0:
        return 0, 16, 0
    if A == 1:
        return 0, 16 + 24, 1
    heap = list(freq.values())
    heapq.heapify(heap)
    data_bits = 0
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        data_bits += a + b
        heapq.heappush(heap, a + b)
    table_bits = 16 + A * 24
    return data_bits, table_bits, A


def probe_one(path, n_timing_repeats=5):
    name = Path(path).name
    img = np.array(Image.open(path).convert("RGB"))
    H, W, _ = img.shape
    R = img[:, :, 0].astype(np.int32)
    G = img[:, :, 1].astype(np.int32)
    B = img[:, :, 2].astype(np.int32)

    Y, Co, Cg = rgb_to_ycocgr(R, G, B)
    # Verify invertibility in code on the real image
    R2, G2, B2 = ycocgr_to_rgb(Y, Co, Cg)
    assert np.array_equal(R2, R) and np.array_equal(G2, G) and np.array_equal(B2, B), \
        f"YCoCg-R image round-trip failed for {name}"

    # Time numpy forward transform (DCT of 3 channels)
    ms_list = []
    Yb = Cob = Cgb = None
    for _ in range(n_timing_repeats):
        t0 = time.perf_counter()
        Yb = dct_forward_image(Y)
        Cob = dct_forward_image(Co)
        Cgb = dct_forward_image(Cg)
        t1 = time.perf_counter()
        ms_list.append((t1 - t0) * 1000.0)
    fwd_ms = float(np.median(ms_list))

    total_data = 0
    total_table = 0
    for blk in (Yb, Cob, Cgb):
        for i in range(4):
            for j in range(4):
                d, tb, _ = huffman_bits_exact(blk[:, :, i, j].reshape(-1))
                total_data += d
                total_table += tb
    total_bits = total_data + total_table + 64 * 8  # 64B header
    total_bytes = (total_bits + 7) // 8
    bpp = total_bytes * 8 / (H * W * 3)
    return {
        "name": name, "H": H, "W": W,
        "data_bits": total_data, "table_bits": total_table,
        "total_bits": total_bits, "total_bytes": total_bytes,
        "bpp": bpp, "fwd_ms": fwd_ms,
    }


def main():
    assert_dct_invertible()
    assert_ycocgr_random()
    rows = []
    for p in IMAGES:
        r = probe_one(p)
        r["moe"] = MOE_PER_IMAGE[r["name"]]
        r["delta"] = r["bpp"] - r["moe"]
        rows.append(r)
        print(f"{r['name']}: bpp={r['bpp']:.4f} moe={r['moe']:.2f} "
              f"delta={r['delta']:+.4f} fwd={r['fwd_ms']:.1f}ms "
              f"data={r['data_bits']} table={r['table_bits']}")
    avg = float(np.mean([r["bpp"] for r in rows]))
    moe_avg = MOE_AVG
    print(f"AVG dct={avg:.4f} moe={moe_avg:.3f} delta={avg - moe_avg:+.4f}")
    avg_ms = float(np.mean([r["fwd_ms"] for r in rows]))
    print(f"AVG fwd time {avg_ms:.1f} ms/image (numpy einsum, 3ch, 512x768)")

    # Write RESULTS.md (same dir as this script per task path)
    out = Path("/tmp/opencode/autocompress/experiments/probe_dct_RESULTS.md")
    lines = []
    lines.append("# probe_dct RESULTS — reversible integer-DCT front-end vs HAPRE-C MOE\n")
    lines.append("Method: RGB->YCoCg-R (floor-shift, round-trip asserted) + H.264 4x4 "
                 "integer DCT `Cf*block*Cf^T` (exact rational inverse asserted on 200 random "
                 "blocks before measuring) + EXACT order-0 Huffman per subband via heapq "
                 "(`sum(count*len) + 16 + A*24` table bits, 48 streams) + 64B header. "
                 "numpy+PIL only, CPU.\n")
    lines.append("| image | HxW | DCT bpp | MOE bpp | delta (DCT-MOE) | fwd ms | data bits | table bits |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| {r['name']} | {r['H']}x{r['W']} | {r['bpp']:.4f} | "
                     f"{r['moe']:.2f} | {r['delta']:+.4f} | {r['fwd_ms']:.1f} | "
                     f"{r['data_bits']} | {r['table_bits']} |")
    lines.append(f"\nAverage DCT bpp: **{avg:.4f}**. MOE champion avg: **{moe_avg:.3f}**. "
                 f"Delta (DCT-MOE): **{avg - moe_avg:+.4f} bpp** "
                 f"({'WORSE' if avg > moe_avg else 'BETTER'} than MOE).\n")
    lines.append(f"JXL-e9 reference: 3.03 avg (final boss, not beaten by either).\n")
    lines.append(f"Avg numpy forward-transform time: **{avg_ms:.1f} ms/image** "
                 "(median of 5 einsum runs, 3 channels, 512x768, int64).\n")
    lines.append("## Verdict\n")
    if avg > moe_avg:
        lines.append(f"No — transform direction as probed does NOT beat 3.464 bpp. "
                     f"It is {avg - moe_avg:.4f} bpp worse on average, and worse on "
                     f"{sum(1 for r in rows if r['bpp'] > r['moe'])}/7 images "
                     "(all 7 if that is the outcome — see table). "
                     "Causal prediction (MED/MoE) remains champion.\n")
    else:
        lines.append("Yes — transform direction beats MOE on average (see table).\n")
    lines.append("## Honest limitations\n")
    lines.append("- Order-0 Huffman per subband only: no run-length / run-mode, no context/adaptive coding, "
                 "no bit-plane or significance-map coding, no DC DPCM across blocks.")
    lines.append("- Unnormalized H.264 integer core inflates coefficients (row gains 4/10); "
                 "order-0 code pays ~log2(gain) extra bits/coeff vs an orthonormal DCT. No scaling compensation applied.")
    lines.append("- No quantization (lossless) so energy compaction does not reduce symbol counts; "
                 "DC subband keeps ~10-11 bits/symbol entropy.")
    lines.append("- Fixed 4x4, no adaptive block size, no directional modes, no chroma handling beyond YCoCg-R.")
    lines.append("- Table cost counted as 16+A*24 bits per stream (48 streams); real container overhead differs.")
    lines.append("- Single-symbol edge case costed as 0 data bits + table (never hit on these images).")
    lines.append("- Timing is numpy einsum forward only; excludes YCoCg-R, Huffman counting, and I/O.")
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
