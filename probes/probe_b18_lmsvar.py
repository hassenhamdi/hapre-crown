"""probe_b18_lmsvar.py — LMS variant sweep (thr/stencil), b17-frame exact."""
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
from probe_b17_rctw import rct_fwd, med_pred, exact_plane_bits
from probe_b4_b import nbhd as nbhd_base
from probe_b18_stack import lms_pred_plane

D = "/tmp/opencode/autocompress/experiments/real_photos"
FILES = ["kodim05.png", "kodim13.png", "kodim01.png", "kodim07.png"]


def lms_pred_plane4(plane, thr):
    p = plane.astype(np.int32)
    H, W = p.shape
    a, b, c, d, Ww, NNe, NE = nbhd_base(plane)
    L, T, TL, TR = a.ravel(), b.ravel(), c.ravel(), d.ravel()
    y = p.ravel()
    out = np.empty_like(y)
    w = [4, 4, 4, 4]
    for n in range(y.size):
        l, t, tl, tr = int(L[n]), int(T[n]), int(TL[n]), int(TR[n])
        pr = (w[0] * l + w[1] * t + w[2] * tl + w[3] * tr + 8) // 16
        out[n] = pr
        e = int(y[n]) - pr
        if abs(e) > thr:
            se = 1 if e > 0 else -1
            m = (l + t + tl + tr + 2) // 4
            xs = (l, t, tl, tr)
            for i in range(4):
                di = xs[i] - m
                if di > 0:
                    w[i] += se
                elif di < 0:
                    w[i] -= se
            for i in range(4):
                w[i] = min(20, max(-8, w[i]))
            s = sum(w)
            while s > 16:
                j = max(range(4), key=lambda i: (w[i], -i))
                w[j] -= 1
                s -= 1
            while s < 16:
                j = min(range(4), key=lambda i: (w[i], i))
                w[j] += 1
                s += 1
    return out.reshape(H, W)


def main():
    t0 = time.time()
    for fn in FILES:
        px = np.array(Image.open(os.path.join(D, fn)).convert("RGB"))
        H, W, _ = px.shape
        npx = H * W * 3
        yc = rct_fwd(px, 3, 6)
        planes = [yc[:, :, c] for c in range(3)]
        base = (64 + sum(exact_plane_bits((pl - med_pred(pl)).astype(np.int32))[0]
                         for pl in planes)) / npx
        line = f"{fn}: MED={base:.4f}"
        for tag, fn_ in [("L5T0", lambda pl: lms_pred_plane(pl, 0)),
                         ("L5T1", lambda pl: lms_pred_plane(pl, 1)),
                         ("L5T2", lambda pl: lms_pred_plane(pl, 2)),
                         ("L5T3", lambda pl: lms_pred_plane(pl, 3)),
                         ("L5T5", lambda pl: lms_pred_plane(pl, 5)),
                         ("L4T3", lambda pl: lms_pred_plane4(pl, 3))]:
            tot = 64 + sum(exact_plane_bits((pl - fn_(pl)).astype(np.int32))[0]
                           for pl in planes)
            line += f" {tag}={tot / npx:.4f}({(tot / npx) - base:+.4f})"
        print(line, flush=True)
    print(f"TOTAL {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
