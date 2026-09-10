"""probe_b18_selfsim.py — 5-minute patch-feasibility check on 05 (new file).

For sampled 8x8 blocks (tire-band bottom half + control top + 13 samples):
  L1_MED = sum|MED residual| in block (current-prediction cost proxy)
  L1_BESTMATCH = min SAD vs causal 8x8 candidates (stride 2, window 96x64)
Ratio << 1 => patch headroom. Operates on C27-Y plane (luma dominates cost).
numpy+PIL only.
"""
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
from probe_b17_rctw import rct_fwd, med_pred

D = "/tmp/opencode/autocompress/experiments/real_photos"


def sad_best(block, search_img, by, bx, B=8, wy=48, wx=72, stride=2):
    """Best SAD of block vs causal candidates in search_img.

    Causal: candidate top-left (cy,cx) with cy+B<=by (fully above) or
    (cy==by... use cy+B<=by OR (same band cy>by-B and cx+B<=bx)).
    """
    H, W = search_img.shape
    best = None
    cy = 0
    while cy + B <= H:
        cx = 0
        while cx + B <= W:
            ok = (cy + B <= by) or (cy >= by - B + 1 and cx + B <= bx)
            if ok and not (cy == by and cx == bx):
                cand = search_img[cy:cy + B, cx:cx + B].astype(np.int32)
                s = int(np.abs(block.astype(np.int32) - cand).sum())
                if best is None or s < best:
                    best = s
            cx += stride
        cy += stride
    return best


def run(fn, rows, n=120, seed=7):
    px = np.array(Image.open(os.path.join(D, fn)).convert("RGB"))
    yc = rct_fwd(px, 3, 6)
    Y = yc[:, :, 0]
    M = med_pred(Y)
    R = np.abs((Y - M).astype(np.int32))
    H, W = Y.shape
    B = 8
    rng = np.random.default_rng(seed)
    ratios = []
    meds = []
    for _ in range(n):
        by = int(rng.integers(rows[0], rows[1])) * B
        bx = int(rng.integers(0, W // B - 1)) * B
        by = min(by, H - B)
        l1_med = int(R[by:by + B, bx:bx + B].sum())
        b = sad_best(Y[by:by + B, bx:bx + B], Y, by, bx)
        if b is not None and l1_med > 0:
            ratios.append(b / l1_med)
            meds.append(l1_med)
    ratios = np.array(ratios)
    meds = np.array(meds)
    # weight by med cost: total best-match cost / total MED cost
    print(f"{fn} rows{rows}: n={len(ratios)} med avg block-L1={meds.mean():.0f} | "
          f"ratio mean={ratios.mean():.3f} med={np.median(ratios):.3f} | "
          f"frac<0.5={(ratios < 0.5).mean():.2f} frac<0.7={(ratios < 0.7).mean():.2f}",
          flush=True)


if __name__ == "__main__":
    t0 = time.time()
    # 05: tire band = bottom half rows (y 256-512 -> block rows 32-64); top = riders/grass
    run("kodim05.png", (32, 63))
    run("kodim05.png", (4, 28))
    # 13: river band middle, forest top, rocks
    run("kodim13.png", (28, 48))
    run("kodim13.png", (4, 24))
    # control 01
    run("kodim01.png", (16, 48))
    print(f"TOTAL {(time.time() - t0):.0f}s")
