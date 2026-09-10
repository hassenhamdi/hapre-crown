"""Probe LF2: clustered LF side-information conditioning vs order-0 Huffman baseline.

Method (numpy + PIL only, CPU):
 1. YCoCg-R reversible integer transform, invertibility asserted.
 2. MED residuals per channel (vectorized).
 3. Baseline: per-channel EXACT order-0 Huffman bits (heapq), +16+A*24 per stream.
 4. LF: 4x box-average downsample (integer floor) per channel, NN upsample,
    e = orig - upLF, 4 bins from per-channel quartile thresholds (rounded to int16).
 5. Conditioned: per-(channel,bin) EXACT Huffman + side (per-channel order-0
    Huffman of downsampled values) + thresholds counted.
Compares conditioned+side vs baseline per image.

Honesty notes documented in RESULTS.md:
 - Threshold cost counted as 9x16-bit = 18 bytes (3 thresholds x 3 channels).
   Spec text said "6 bytes"; the 12-byte difference (~0.00037 bpp) is reported.
 - Bin assignment is an ORACLE bound: decoder cannot derive e-bins from LF side
   info alone (needs the original), and the 2 bit/px bin map is NOT counted,
   per the probe spec. So a "win" here is necessary-but-not-sufficient for a
   real codec; a "loss" here kills the idea outright.
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

TABLE_HDR = 16      # bits per stream header
TABLE_PER_SYM = 24  # bits per alphabet symbol


def ycocg_r_forward(rgb):
    R = rgb[:, :, 0].astype(np.int32)
    G = rgb[:, :, 1].astype(np.int32)
    B = rgb[:, :, 2].astype(np.int32)
    Co = R - B
    t = B + np.right_shift(Co, 1)          # arithmetic shift (floor div2)
    Cg = G - t
    Y = t + np.right_shift(Cg, 1)
    return Y, Co, Cg


def ycocg_r_inverse(Y, Co, Cg):
    t = Y - np.right_shift(Cg, 1)
    B = t - np.right_shift(Co, 1)
    G = Cg + t
    R = Co + B
    return R, G, B


def med_residuals(ch):
    # ch: HxW int32. Edge-replicate pad 1px.
    p = np.pad(ch, 1, mode="edge")
    a = p[1:-1, :-2]    # left
    b = p[:-2, 1:-1]    # top
    c = p[:-2, :-2]     # topleft
    mx = np.maximum(a, b)
    mn = np.minimum(a, b)
    pred = np.where(c >= mx, mn, np.where(c <= mn, mx, a + b - c))
    return (ch - pred).astype(np.int32)


def huffman_data_bits(counts):
    """Exact Huffman data bits = sum(count*len) via merge sums. Returns (bits, A)."""
    vals = [int(c) for c in counts if c > 0]
    A = len(vals)
    if A == 0:
        return 0, 0
    if A == 1:
        return vals[0] * 1, 1  # single-symbol alphabet: 1 bit/symbol (conservative)
    heapq.heapify(vals)
    total = 0
    while len(vals) > 1:
        x = heapq.heappop(vals)
        y = heapq.heappop(vals)
        total += x + y
        heapq.heappush(vals, x + y)
    return total, A


def stream_bits(symbols):
    """Full honest stream cost: Huffman data bits + 16 + A*24."""
    _, counts = np.unique(symbols, return_counts=True)
    data, A = huffman_data_bits(counts)
    if A == 0:
        return 0
    return data + TABLE_HDR + A * TABLE_PER_SYM


def box_down4(x):
    H, W = x.shape
    Hp = (H + 3) // 4 * 4
    Wp = (W + 3) // 4 * 4
    xp = np.pad(x, ((0, Hp - H), (0, Wp - W)), mode="edge")
    blocks = xp.reshape(Hp // 4, 4, Wp // 4, 4)
    return (blocks.sum(axis=(1, 3)) // 16).astype(np.int32)  # integer floor mean


def nn_up4(d, H, W):
    return np.repeat(np.repeat(d, 4, axis=0), 4, axis=1)[:H, :W]


def main():
    print(f"{'image':12s} {'Npix':>8s} {'base_bpp':>9s} {'cond+side':>9s} {'delta%':>8s}")
    base_bpp_all, cond_bpp_all = [], []
    for path in IMAGES:
        name = path.rsplit("/", 1)[-1]
        rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        H, W, _ = rgb.shape
        N = H * W

        Y, Co, Cg = ycocg_r_forward(rgb)
        # 1. invertibility assert
        Ri, Gi, Bi = ycocg_r_inverse(Y, Co, Cg)
        assert np.array_equal(Ri, rgb[:, :, 0].astype(np.int32)) \
            and np.array_equal(Gi, rgb[:, :, 1].astype(np.int32)) \
            and np.array_equal(Bi, rgb[:, :, 2].astype(np.int32)), \
            f"YCoCg-R NOT invertible on {name}"

        chans = (Y, Co, Cg)
        res = [med_residuals(c) for c in chans]

        # 3. baseline
        base_bits = sum(stream_bits(r) for r in res)
        base_bpp = base_bits / N

        # 4-5. LF conditioned
        cond_bits = 0
        side_bits = 0
        for c, r in zip(chans, res):
            d = box_down4(c)
            up = nn_up4(d, H, W)
            e = (c - up).astype(np.int32)
            qs = np.quantile(e.astype(np.float64), [0.25, 0.50, 0.75])
            thr = np.round(qs).astype(np.int32)  # what is actually transmitted (int16)
            assert np.all(thr >= -32768) and np.all(thr <= 32767), "threshold int16 overflow"
            b = ((e > thr[0]).astype(np.int32) + (e > thr[1]).astype(np.int32)
                 + (e > thr[2]).astype(np.int32))
            for k in range(4):
                rk = r[b == k]
                if rk.size:
                    cond_bits += stream_bits(rk)
                # empty bin -> 0 bits (quartile bins are non-empty in practice)
            side_bits += stream_bits(d)
        thr_bits = 9 * 16  # 3 thresholds x 3 channels, 16-bit each (honest; see docstring)
        total_bits = cond_bits + side_bits + thr_bits
        cond_bpp = total_bits / N

        delta = (cond_bpp - base_bpp) / base_bpp * 100.0
        base_bpp_all.append(base_bpp)
        cond_bpp_all.append(cond_bpp)
        print(f"{name:12s} {N:8d} {base_bpp:9.4f} {cond_bpp:9.4f} {delta:+8.2f}%")

    avg_base = float(np.mean(base_bpp_all))
    avg_cond = float(np.mean(cond_bpp_all))
    avg_delta = (avg_cond - avg_base) / avg_base * 100.0
    print(f"\nAVERAGE over {len(IMAGES)} images: baseline={avg_base:.4f} bpp  "
          f"cond+side={avg_cond:.4f} bpp  delta={avg_delta:+.2f}%")


if __name__ == "__main__":
    main()
