"""probe_b18_perkeyk.py — per-KEY static Golomb-k screen (new file).

Texture groups pick Golomb (table-free) but single-k misfits mixed scales.
Per-key k (729 keys x 4b = 0.0025bpp/ch side, static = no adaptation whiplash,
decoder-derivable key): exact min-Rice-k per key vs per-group best-k.
Screen on 05/13 Q-Golomb groups (from dumps) + controls. Fast (no re-encode).
"""
import math
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
os.chdir("/tmp/opencode/autocompress/experiments")
from probe_b5_d import golomb_best

NKEY = 729 * 4


def rice_bits(vals, k):
    tot = 0
    for v in vals:
        M = v * 2 if v >= 0 else -v * 2 - 1
        tot += (M >> k) + 1 + k
    return tot


def main():
    t0 = time.time()
    for fn in ["kodim05.png", "kodim13.png", "kodim01.png", "kodim07.png"]:
        path = f"/tmp/b18_dump_{fn[:7]}.pkl"
        if not os.path.exists(path):
            print(f"missing {path}")
            continue
        recs = []
        with open(path, "rb") as f:
            while True:
                try:
                    recs.append(pickle.load(f))
                except EOFError:
                    break
        if fn in ("kodim01.png", "kodim07.png") and not recs:
            print(f"{fn}: no dump, encoding quickly...")
            from PIL import Image
            import probe_b18_stack as S
            D = "/tmp/opencode/autocompress/experiments/real_photos"
            px = np.array(Image.open(os.path.join(D, fn)).convert("RGB"))
            gs = S.proxy_gs_choice(px)
            S.DUMP_GROUPS = path
            if os.path.exists(path):
                os.unlink(path)
            blob, info = S.encode_image_rct(px, "C27", 3, 6, gs, use_lms=True, use_err4=False)
            with open(path, "rb") as f:
                while True:
                    try:
                        recs.append(pickle.load(f))
                    except EOFError:
                        break
        print(f"== {fn} ==")
        npx = 512 * 768 * 3
        tb = ta = 0
        for ci, r in enumerate(recs):
            symplane, expmap, ng = r["symplane"], r["expmap"], r["ng"]
            wins = r["wins"]
            key = r["key"]
            sb = sa = 0
            for gi in range(ng):
                w = wins[gi]
                if w[2] != 1:
                    continue  # only Golomb groups
                m = (expmap == gi)
                g = symplane[m]
                # current static: recompute exact (sanity vs w[0])
                gd0, k0 = golomb_best(g.astype(np.int64))
                cur = gd0 + 4 + (3 if w[4] != 0 else 0)
                sb += cur
                # per-key k
                kk = key[m]
                uk = np.unique(kk)
                tot = 0
                for u in uk:
                    gu = g[kk == u].astype(np.int64)
                    bk = min(range(13), key=lambda k: rice_bits(gu, k))
                    tot += rice_bits(gu, bk) + 4
                # active-key-only k-table side: len(uk)*4 bits
                tot += len(uk) * 4
                # bias d: reuse group's d (fold then Rice) — approximate: shift by w[4]
                sa += tot + (3 if w[4] != 0 else 0)
            print(f"  ch{ci}: GolombGroups static={sb} perkeyK={sa} d={sa - sb:+.0f} "
                  f"({(sa - sb) / npx:+.4f}bpp)", flush=True)
            tb += sb
            ta += sa
        print(f"  TOTAL Golomb-part: d={(ta - tb) / npx:+.4f} bpp", flush=True)
    print(f"TOTAL {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
