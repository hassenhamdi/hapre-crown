"""Probe: horizontal band-adaptive order-0 Huffman vs global tables (net of overhead).
Branch 2 of image-compression campaign. numpy + PIL only, CPU, no torch.
UNIT RULE: bpp = total_bits / (H*W*3).
"""
import heapq
import os
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

DIMS_OVERHEAD_BITS = 8 * 8  # 8 bytes dims


def rgb_to_ycocg_r(r, g, b):
    """Reversible YCoCg-R. Inputs int32 HxW. Uses floor //2 for negative Co/Cg."""
    co = r - b
    t = b + (co // 2)
    cg = g - t
    y = t + (cg // 2)
    return y, co, cg


def ycocg_r_to_rgb(y, co, cg):
    t = y - (cg // 2)
    b = t - (co // 2)
    g = cg + t
    r = co + b
    return r, g, b


def med_residuals(ch):
    """MED residuals, edge-replicate pad, vectorized. ch: int32 HxW."""
    padded = np.pad(ch, ((1, 0), (1, 0)), mode="edge")
    a = padded[1:, :-1]   # left
    b = padded[:-1, 1:]   # top
    c = padded[:-1, :-1]  # diag
    x = padded[1:, 1:]
    mx = np.maximum(a, b)
    mn = np.minimum(a, b)
    p = np.where(c >= mx, mn, np.where(c <= mn, mx, a + b - c))
    return (x - p).astype(np.int32)


def huffman_data_bits(counts):
    """Exact order-0 Huffman data bits = sum of merged weights. counts: list of ints >0."""
    n = len(counts)
    if n == 0:
        return 0
    if n == 1:
        return int(counts[0]) * 1  # single-symbol table: 1 bit/symbol (conservative)
    heap = [int(c) for c in counts]
    heapq.heapify(heap)
    total = 0
    while len(heap) > 1:
        x = heapq.heappop(heap)
        y = heapq.heappop(heap)
        s = x + y
        total += s
        heapq.heappush(heap, s)
    return total


def table_bits_for_alphabet(A):
    return 16 + A * 24


def bits_for_residual_field(resid):
    """resid: int32 array (any shape). Returns (data_bits, A, total_with_table)."""
    _, counts = np.unique(resid, return_counts=True)
    A = len(counts)
    data = huffman_data_bits(counts.tolist())
    return data, A, data + table_bits_for_alphabet(A)


def main():
    assert np is not None
    # no-torch guard
    try:
        import torch  # noqa
        has_torch = True
    except ImportError:
        has_torch = False
    print(f"torch importable: {has_torch} (must NOT be used; this probe uses numpy+PIL only)")
    print(f"numpy {np.__version__}")
    results = []
    for path in IMAGES:
        assert os.path.exists(path), f"missing {path}"
        im = Image.open(path).convert("RGB")
        arr = np.array(im, dtype=np.int32)  # HxWx3
        H, W, _ = arr.shape
        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]

        # 1. YCoCg-R + exact invertibility assert
        y, co, cg = rgb_to_ycocg_r(r, g, b)
        r2, g2, b2 = ycocg_r_to_rgb(y, co, cg)
        assert np.array_equal(r, r2), f"R invert fail {path}"
        assert np.array_equal(g, g2), f"G invert fail {path}"
        assert np.array_equal(b, b2), f"B invert fail {path}"

        # 2. MED residuals per channel
        ry = med_residuals(y)
        rco = med_residuals(co)
        rcg = med_residuals(cg)

        denom = H * W * 3

        # 3. Baseline global order-0 Huffman
        base_total = DIMS_OVERHEAD_BITS
        base_info = {}
        for name, res in (("Y", ry), ("Co", rco), ("Cg", rcg)):
            data, A, tot = bits_for_residual_field(res)
            base_total += tot
            base_info[name] = (data, A)
        base_bpp = base_total / denom

        # 4. Band-adaptive B=2 and B=4
        band_bpps = {}
        band_detail = {}
        for B in (2, 4):
            tot = DIMS_OVERHEAD_BITS
            detail = []
            for name, res in (("Y", ry), ("Co", rco), ("Cg", rcg)):
                bands = np.array_split(res, B, axis=0)
                for bi, band in enumerate(bands):
                    data, A, bt = bits_for_residual_field(band)
                    tot += bt
                    detail.append((name, bi, band.shape, A, data))
            bpp = tot / denom
            band_bpps[B] = bpp
            band_detail[B] = detail

        d2 = band_bpps[2] - base_bpp
        d4 = band_bpps[4] - base_bpp
        results.append({
            "img": os.path.basename(path),
            "HxW": f"{H}x{W}",
            "base": base_bpp,
            "b2": band_bpps[2],
            "b4": band_bpps[4],
            "d2": d2,
            "d4": d4,
            "base_total": base_total,
            "base_info": base_info,
            "denom": denom,
        })
        print(f"{os.path.basename(path)} ({H}x{W}): base={base_bpp:.4f} "
              f"B2={band_bpps[2]:.4f} (d={d2:+.4f}) B4={band_bpps[4]:.4f} (d={d4:+.4f}) "
              f"A(Y/Co/Cg)={[base_info[k][1] for k in ('Y','Co','Cg')]}")

    avg_base = float(np.mean([x["base"] for x in results]))
    avg_b2 = float(np.mean([x["b2"] for x in results]))
    avg_b4 = float(np.mean([x["b4"] for x in results]))
    avg_d2 = avg_b2 - avg_base
    avg_d4 = avg_b4 - avg_base
    print("-" * 70)
    print(f"AVG over {len(results)}: base={avg_base:.4f} B2={avg_b2:.4f} (d={avg_d2:+.4f}) "
          f"B4={avg_b4:.4f} (d={avg_d4:+.4f})")
    if not (4.0 <= avg_base <= 5.0):
        print(f"WARNING: sanity anchor FAILED (base avg {avg_base:.4f} outside 4.0-5.0). DEBUG BEFORE PROCEEDING.")
    else:
        print(f"Sanity anchor OK: base avg {avg_base:.4f} in [4.0,5.0] (expected ~4.4-4.6).")
    if avg_d2 < 0 or avg_d4 < 0:
        print("Band adaptation WINS net (at least one B).")
    else:
        print("Band adaptation LOSES net (both B worse than baseline).")


if __name__ == "__main__":
    main()
