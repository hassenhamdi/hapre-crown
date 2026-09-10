"""Squeeze-style LF-PREDICTION probe (branch 1): predict pixels from transmitted
low-frequency layer, then MED+Huffman-code the high-frequency remainder.

Method per spec: numpy + PIL only, CPU.
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

DIMS_OVERHEAD_BITS = 8 * 8  # 8 bytes dims


def rgb_to_ycocgr(arr):
    R = arr[:, :, 0].astype(np.int32)
    G = arr[:, :, 1].astype(np.int32)
    B = arr[:, :, 2].astype(np.int32)
    Co = R - B
    t = B + (Co // 2)  # arithmetic floor shift
    Cg = G - t
    Y = t + (Cg // 2)
    return Y, Cg, Co


def ycocgr_to_rgb(Y, Cg, Co):
    t = Y - (Cg // 2)
    B = t - (Co // 2)
    G = Cg + t
    R = Co + B
    H, W = Y.shape
    out = np.empty((H, W, 3), dtype=np.int32)
    out[:, :, 0] = R
    out[:, :, 1] = G
    out[:, :, 2] = B
    return out


def med_predict(plane):
    plane = plane.astype(np.int32)
    P = np.pad(plane, 1, mode="edge")
    a = P[1:-1, :-2]  # left
    b = P[:-2, 1:-1]  # top
    c = P[:-2, :-2]  # topleft
    mx = np.maximum(a, b)
    mn = np.minimum(a, b)
    pred = np.where(c >= mx, mn, np.where(c <= mn, mx, a + b - c))
    return pred


def huffman_bits(vals):
    """EXACT order-0 Huffman bits: sum(count*len) + 16 + A*24."""
    _, counts = np.unique(vals, return_counts=True)
    A = len(counts)
    if A == 1:
        data = int(counts[0]) * 1
        return data + 16 + A * 24, data
    heap = counts.tolist()
    heapq.heapify(heap)
    data = 0
    while len(heap) > 1:
        x = heapq.heappop(heap)
        y = heapq.heappop(heap)
        data += x + y
        heapq.heappush(heap, x + y)
    return data + 16 + A * 24, data


def plane_residual_bits(plane):
    pred = med_predict(plane)
    resid = plane.astype(np.int32) - pred
    bits, _ = huffman_bits(resid)
    return bits, resid


def downsample4(plane):
    H, W = plane.shape
    Hp = ((H + 3) // 4) * 4
    Wp = ((W + 3) // 4) * 4
    ph, pw = Hp - H, Wp - W
    if ph or pw:
        p = np.pad(plane, ((0, ph), (0, pw)), mode="edge")
    else:
        p = plane
    # integer floor mean of 4x4 blocks
    blk = p.reshape(Hp // 4, 4, Wp // 4, 4).sum(axis=(1, 3)) // 16
    return blk


def upsample4_nn(lf, H, W):
    return np.repeat(np.repeat(lf, 4, axis=0), 4, axis=1)[:H, :W]


def main():
    rows = []
    for path in IMAGES:
        img = np.array(Image.open(path).convert("RGB"))
        assert img.ndim == 3 and img.shape[2] == 3, path
        H, W, _ = img.shape
        N = H * W

        Y, Cg, Co = rgb_to_ycocgr(img)
        # Assert exact invertibility
        rt = ycocgr_to_rgb(Y, Cg, Co)
        assert rt.shape == img.shape, path
        assert np.array_equal(rt, img.astype(np.int32)), f"YCoCg-R round-trip FAILED: {path}"

        # Baseline: MED residuals + order-0 Huffman per channel
        base_bits = 0
        base_resids = []
        for ch in (Y, Cg, Co):
            b, r = plane_residual_bits(ch)
            base_bits += b
            base_resids.append(r.ravel())
        base_all = np.concatenate(base_resids)
        base_bpp = base_bits / N

        # Squeeze-lite
        hf_bits = 0
        hf_resids = []
        lf_side_bits = DIMS_OVERHEAD_BITS
        for ch in (Y, Cg, Co):
            lf = downsample4(ch.astype(np.int32))
            up = upsample4_nn(lf, H, W)
            hf = ch.astype(np.int32) - up
            b, r = plane_residual_bits(hf)
            hf_bits += b
            hf_resids.append(r.ravel())
            lb, _ = huffman_bits(lf)
            lf_side_bits += lb
        hf_all = np.concatenate(hf_resids)
        total_bits = hf_bits + lf_side_bits
        sq_bpp = total_bits / N
        delta = (sq_bpp - base_bpp) / base_bpp * 100.0
        lf_share = lf_side_bits / total_bits * 100.0
        var_ratio = float(np.var(hf_all)) / float(np.var(base_all)) if float(np.var(base_all)) > 0 else float("nan")

        rows.append(dict(
            name=path.split("/")[-1], H=H, W=W,
            base_bits=base_bits, base_bpp=base_bpp,
            hf_bits=hf_bits, lf_bits=lf_side_bits, total_bits=total_bits,
            sq_bpp=sq_bpp, delta=delta, lf_share=lf_share,
            var_base=float(np.var(base_all)), var_hf=float(np.var(hf_all)),
            var_ratio=var_ratio,
        ))
        print(f"{rows[-1]['name']} {H}x{W}: base={base_bpp:.4f} bpp, "
              f"squeeze={sq_bpp:.4f} bpp, delta={delta:+.2f}%, "
              f"LFshare={lf_share:.1f}%, varRatio={var_ratio:.3f}", flush=True)

    avg_base = float(np.mean([r["base_bpp"] for r in rows]))
    avg_sq = float(np.mean([r["sq_bpp"] for r in rows]))
    avg_delta = (avg_sq - avg_base) / avg_base * 100.0
    avg_lf = float(np.mean([r["lf_share"] for r in rows]))
    avg_vr = float(np.mean([r["var_ratio"] for r in rows]))
    print(f"AVG: base={avg_base:.4f} bpp, squeeze={avg_sq:.4f} bpp, delta={avg_delta:+.2f}%")
    print(f"AVG LF share={avg_lf:.1f}%, AVG varRatio={avg_vr:.3f}")

    md = []
    md.append("# Probe LF-PREDICTION (Squeeze-lite) RESULTS")
    md.append("")
    md.append("Method: YCoCg-R + 4x box-average LF (floor mean, edge-replicate pad), "
              "nearest-neighbor upsample, HF=orig-upsampledLF, MED residuals + EXACT "
              "order-0 Huffman (heapq codelengths, +16+A*24 table bits) on HF planes; "
              "LF side = order-0 Huffman of downsampled LF values + 8 bytes dims. "
              "numpy+PIL only, CPU. YCoCg-R round-trip asserted on all 7 images: PASS.")
    md.append("")
    md.append("| image | HxW | baseline bpp | HF bits | LF side bits | total bits | squeeze bpp | delta% | LF share% | var(base resid) | var(HF resid) | var ratio HF/base |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        md.append(f"| {r['name']} | {r['H']}x{r['W']} | {r['base_bpp']:.4f} | {r['hf_bits']} | "
                  f"{r['lf_bits']} | {r['total_bits']} | {r['sq_bpp']:.4f} | {r['delta']:+.2f} | "
                  f"{r['lf_share']:.1f} | {r['var_base']:.1f} | {r['var_hf']:.1f} | {r['var_ratio']:.3f} |")
    md.append("")
    md.append(f"Average over 7: baseline={avg_base:.4f} bpp, squeeze={avg_sq:.4f} bpp, "
              f"delta={avg_delta:+.2f}%. Avg LF side share={avg_lf:.1f}%. "
              f"Avg HF-vs-baseline residual variance ratio={avg_vr:.3f}.")
    md.append("")
    md.append("## Mechanism")
    md.append(f"LF side stream costs ~{avg_lf:.1f}% of the squeeze total on average; "
              "HF MED residuals vs direct MED residuals variance ratio averages "
              f"{avg_vr:.3f} (per-image above). Ratio <1 means the LF predictor removed "
              "variance before MED; ratio >1 means upsampled-LF subtraction ADDED variance "
              "that MED must then code, on top of the LF side stream.")
    md.append("")
    if avg_delta < 0:
        md.append(f"## Verdict: LF-prediction WINS by {-avg_delta:.2f}% net (after all side costs counted).")
    else:
        md.append(f"## Verdict: LF-prediction LOSES by {avg_delta:.2f}% net (after all side costs counted). "
                  "Direct MED coding beats Squeeze-lite LF-prediction on these 7 images.")
    md.append("")
    md.append("## Recommended follow-up (single)")
    md.append("Bilinear/bicubic-upsample ablation on the same 7 images (keep 4x box LF, "
              "swap NN for smooth upsampling, recount every byte): tests whether the loss "
              "comes from blocky NN prediction rather than from the LF/HF split itself.")
    md.append("")
    with open("/tmp/opencode/autocompress/experiments/probe_lfpred_RESULTS.md", "w") as f:
        f.write("\n".join(md) + "\n")

    # stash raw rows for the final message
    import json
    with open("/tmp/opencode/autocompress/experiments/probe_lfpred_rows.json", "w") as f:
        json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
