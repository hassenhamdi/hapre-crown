"""probe_b18_validate.py — B18 STACK validation vs CROWN3 / JXL-e3 refs (new file).

For each image: proxy GS choice, encode min-over-{C6,C27,C12} (+1b GS side),
decode winner, assert round-trip. Reports per-image bpp vs CROWN3 refs and
JXL-e3 refs + flip verdicts. Writes probe_b18_valid.json incrementally.
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
OUT = "/tmp/opencode/autocompress/experiments/probe_b18_valid.json"


def run_image(fn, rcts=S.RCTS):
    px = np.array(Image.open(os.path.join(D, fn)).convert("RGB"))
    H, W, _ = px.shape
    npx = H * W * 3
    t0 = time.time()
    gs = S.proxy_gs_choice(px)
    cands = []
    for (rname, perm, t) in rcts:
        blob, info = S.encode_image_rct(px, rname, perm, t, gs)
        bpp = (len(blob) * 8 + 1) / npx  # +1b GS-side
        cands.append((bpp, blob, info))
        print(f"  {fn} {rname}/G{gs}: {bpp:.4f} "
              f"({(bpp - CR3[fn]) / CR3[fn] * 100:+.2f}% vs CR3) "
              f"fams={[c['fam'] for c in info['chs']]}", flush=True)
    cands.sort(key=lambda t_: t_[0])
    bpp, blob, info = cands[0]
    out, di = S.decode_image(blob)
    assert np.array_equal(np.frombuffer(out, dtype=np.uint8).reshape(H, W, 3), px), \
        f"ROUND-TRIP FAIL {fn}"
    dt = time.time() - t0
    res = {"bpp": round(bpp, 4), "gs": gs, "rct": info["rct"],
           "fams": [c["fam"] for c in info["chs"]],
           "picks": [c.get("picks", {}) for c in info["chs"]],
           "roundtrip": True, "sec": round(dt)}
    print(f"==> {fn}: B18={bpp:.4f} (CR3={CR3[fn]:.4f}, JXL={JXL[fn]:.4f}) "
          f"dCR3={(bpp - CR3[fn]) / CR3[fn] * 100:+.2f}% "
          f"dJXL={(bpp - JXL[fn]) / JXL[fn] * 100:+.2f}% "
          f"{'FLIP ✓' if bpp < JXL[fn] else 'lose'} RT-PASS [{dt:.0f}s]", flush=True)
    return res


def main():
    targets = sys.argv[1:] or ["kodim05.png", "kodim13.png"]
    acc = {}
    if os.path.exists(OUT):
        acc = json.load(open(OUT))
    for fn in targets:
        acc[fn] = run_image(fn)
        json.dump(acc, open(OUT, "w"), indent=1)
    print(json.dumps(acc, indent=1))


if __name__ == "__main__":
    main()
