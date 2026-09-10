"""probe_b18_wgran.py — WAVG prediction-granularity screen (new file only).

b17 M3 tested G32-vs-G64 (G32 won 7/7: prediction wants LOCAL). NEVER tested:
G16/G8 (denser objects need smaller blocks). Frame = b17-(b) exact:
C27 planes + WAVG-GS + global order-0 Huffman + side (1b flag/group-ch +
20b/used) + 1b size-sel. Compares GS in {8,16,32,64} + MED baseline.
numpy+PIL only. bpp exact.
"""
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
from probe_b17_rctw import rct_fwd, med_pred, weighted_fit, exact_plane_bits

D = "/tmp/opencode/autocompress/experiments/real_photos"
FILES = ["kodim01.png", "kodim02.png", "kodim05.png", "kodim07.png",
         "kodim13.png", "kodim19.png", "kodim23.png"]
JXL = {"kodim01.png": 3.3593, "kodim02.png": 3.0611, "kodim05.png": 3.5120,
       "kodim07.png": 2.7310, "kodim13.png": 3.9141, "kodim19.png": 3.2154,
       "kodim23.png": 2.8110}


def main():
    t0 = time.time()
    R = {}
    for f in FILES:
        px = np.array(Image.open(os.path.join(D, f)).convert("RGB"))
        H, W, _ = px.shape
        npx = H * W * 3
        yc = rct_fwd(px, 3, 6)
        planes = [yc[:, :, c] for c in range(3)]
        med = sum(exact_plane_bits((pl - med_pred(pl)).astype(np.int32))[0]
                  for pl in planes) + 64
        line = f"{f}: MED={med / npx:.4f}"
        for GS in [8, 16, 32, 64]:
            out_res, use, wall, side, bm, ng = weighted_fit(planes, H, W, GS)
            tb = side + 1 + sum(exact_plane_bits(out_res[c])[0] for c in range(3)) + 64
            R.setdefault(GS, {})[f] = tb / npx
            nu = sum(sum(u) for u in use)
            line += f" G{GS}={tb / npx:.4f}({(tb - med) / npx:+.4f},use={nu}/{ng * 3})"
        print(line + f" JXL={JXL[f]:.4f}", flush=True)
    print("AVG:", {g: round(float(np.mean(list(R[g].values()))), 4) for g in [8, 16, 32, 64]})
    print(f"TOTAL {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
