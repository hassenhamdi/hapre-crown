"""probe_b18_bands.py — WHERE (spatially) does JXL beat us on 05/13? (new file).

Split 05/13 (+control 01) into horizontal bands; per band: JXL-e3 bytes
(cjxl crop encodes) vs our MED-frame exact bits (C27). Ratio comparison
localizes the gap to content (tires vs riders; rapids vs forest).
cjxl via subprocess; MED via b17 imports. Fast (~2 min).
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
from probe_b17_rctw import rct_fwd, med_pred, exact_plane_bits

D = "/tmp/opencode/autocompress/experiments/real_photos"


def jxl_bpp(pil_img):
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        p = f.name
    pil_img.save(p)
    o = p + ".jxl"
    subprocess.run(["cjxl", "-d", "0", "-e", "3", p, o],
                   check=True, capture_output=True)
    n = os.path.getsize(o)
    os.unlink(p)
    os.unlink(o)
    w, h = pil_img.size
    return n * 8 / (h * w * 3)


def our_med_bpp(arr):
    H, W, _ = arr.shape
    yc = rct_fwd(arr, 3, 6)
    tb = 64 + sum(exact_plane_bits((yc[:, :, c] - med_pred(yc[:, :, c])))[0]
                  for c in range(3))
    return tb / (H * W * 3)


def main():
    for fn in ["kodim05.png", "kodim13.png", "kodim01.png"]:
        px = Image.open(os.path.join(D, fn)).convert("RGB")
        W, H = px.size
        print(f"== {fn} ({W}x{H}) ==")
        for i in range(4):
            box = (0, i * H // 4, W, (i + 1) * H // 4)
            crop = px.crop(box)
            arr = np.array(crop)
            jb = jxl_bpp(crop)
            ob = our_med_bpp(arr)
            print(f"  band{i} y[{box[1]}:{box[3]}]: JXL={jb:.4f} OURS-MED={ob:.4f} "
                  f"gap={ob - jb:+.4f} ({(ob - jb) / jb * 100:+.1f}%)", flush=True)
        # full for reference
        arr = np.array(px)
        print(f"  FULL: JXL={jxl_bpp(px):.4f} OURS-MED={our_med_bpp(arr):.4f}", flush=True)


if __name__ == "__main__":
    main()
