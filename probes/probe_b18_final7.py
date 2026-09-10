"""probe_b18_final7.py — all-7 LMS-only validation, exact per-image GS-min (new file).

Config: GS32/GS16 exact min-over (per image, +1b GS side) x RCT min-over
{C6,C27,C12} (CROWN3 parity) + LMS5_T0/T3 gated experts, no ERR4.
Decode winner per image, assert round-trip. vs CROWN3 refs + JXL-e3 refs.
"""
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
os.chdir("/tmp/opencode/autocompress/experiments")
import probe_b18_stack as S

D = "/tmp/opencode/autocompress/experiments/real_photos"
JXL = {"kodim01.png": 3.3593, "kodim02.png": 3.0611, "kodim05.png": 3.5120,
       "kodim07.png": 2.7310, "kodim13.png": 3.9141, "kodim19.png": 3.2154,
       "kodim23.png": 2.8110}
CR3 = {"kodim01.png": 3.3026, "kodim02.png": 2.9889, "kodim05.png": 3.5984,
       "kodim07.png": 2.6824, "kodim13.png": 3.9542, "kodim19.png": 3.1566,
       "kodim23.png": 2.7448}
OUT = "/tmp/opencode/autocompress/experiments/probe_b18_final7.json"


def run_image(fn):
    px = np.array(Image.open(os.path.join(D, fn)).convert("RGB"))
    H, W, _ = px.shape
    npx = H * W * 3
    t0 = time.time()
    cands = []
    for gs in (32, 16):
        for (rname, perm, t) in S.RCTS:
            blob, info = S.encode_image_rct(px, rname, perm, t, gs,
                                           use_lms=True, use_err4=False)
            bpp = (len(blob) * 8 + 1) / npx  # +1b GS-side
            cands.append((bpp, blob, info, gs))
            print(f"  {fn} {rname}/G{gs}: {bpp:.4f}", flush=True)
    cands.sort(key=lambda t_: t_[0])
    bpp, blob, info, gs = cands[0]
    out, di = S.decode_image(blob)
    assert np.array_equal(np.frombuffer(out, dtype=np.uint8).reshape(H, W, 3), px), \
        f"ROUND-TRIP FAIL {fn}"
    res = {"bpp": round(bpp, 4), "gs": gs, "rct": info["rct"],
           "fams": [c["fam"] for c in info["chs"]],
           "picks": [c.get("picks", {}) for c in info["chs"]],
           "roundtrip": True, "sec": round(time.time() - t0)}
    print(f"==> {fn}: B18={bpp:.4f} (CR3={CR3[fn]:.4f}, JXL={JXL[fn]:.4f}) "
          f"dCR3={(bpp - CR3[fn]) / CR3[fn] * 100:+.2f}% "
          f"dJXL={(bpp - JXL[fn]) / JXL[fn] * 100:+.2f}% "
          f"{'FLIP' if bpp < JXL[fn] else ('HOLD' if bpp < JXL[fn] else 'lose')} "
          f"RT-PASS [{res['sec']}s]", flush=True)
    return res


def main():
    fns = sys.argv[1:] or ["kodim01.png", "kodim02.png", "kodim05.png", "kodim07.png",
                           "kodim13.png", "kodim19.png", "kodim23.png"]
    acc = {}
    if os.path.exists(OUT):
        acc = json.load(open(OUT))
    for fn in fns:
        if fn in acc and acc[fn].get("roundtrip"):
            print(f"{fn}: cached {acc[fn]['bpp']}", flush=True)
            continue
        acc[fn] = run_image(fn)
        json.dump(acc, open(OUT, "w"), indent=1)
    bpp = [acc[f]["bpp"] for f in
           ["kodim01.png", "kodim02.png", "kodim05.png", "kodim07.png",
            "kodim13.png", "kodim19.png", "kodim23.png"] if f in acc]
    if len(bpp) == 7:
        avg = float(np.mean(bpp))
        w = sum(1 for f in acc if acc[f]["bpp"] < JXL[f])
        print(f"AVG={avg:.4f} vs CR3 3.2040 vs JXL 3.2291 | W/L={w}/{7 - w}")
    print(json.dumps(acc, indent=1))


if __name__ == "__main__":
    main()
