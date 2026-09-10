"""probe_b22 run: adaptive-Golomb delta on CROWN6 winning groups (05/C27, 13/C12).

Reads CROWN_DUMPGROUPS .gdat files, verifies static costs vs the driver
(fidelity gate: dumped (k,d) must match driver golomb_cost exactly), then
measures exact adaptive lengths:
  (a1) per-group, 1 ctx      (a4) per-group, 4 act-ctx
  (b1) global pooled raster, 1 ctx   (b4) global pooled raster, 4 act-ctx
Gate: net saving >= gap_bits x 1.2  =>  build the C wire backend.
Gaps (CROWN6 vs JXL-e3, exact): 05 +0.0797bpp, 13 +0.0384bpp (N=1179648 bits
per bpp unit denominator H*W*3).
"""
import sys
import os
import json
import numpy as np
from PIL import Image

sys.path.insert(0, "/tmp/opencode/autocompress/src")
sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
sys.path.insert(0, "/tmp/opencode/autocompress/probes")

import probe_b17_rctw as B17
from probe_b22_adaptive import (read_gdat, static_verify, adaptive_bits,
                                zero_border_activity, act_bin)

NPIX = 768 * 512 * 3  # bpp denominator
GAPS = {"kodim05.png": 0.0797, "kodim13.png": 0.0384}
RCTS = {"C6": (0, 6), "C27": (3, 6), "C12": (1, 5)}
WIN = {"kodim05.png": "C27", "kodim13.png": "C12"}


def run_image(fn, dumpdir, rct):
    print(f"== {fn} [{rct}] ==", flush=True)
    rgb = np.array(Image.open(
        f"/tmp/opencode/autocompress/experiments/real_photos/{fn}").convert("RGB"))
    perm, t = RCTS[rct]
    yc = B17.rct_fwd(rgb, perm, t)
    per_ch = []
    for ch in range(3):
        p = (f"{fn}.{rct}.ch{ch}.gdat")
        g = read_gdat(os.path.join(dumpdir, p))
        plane = yc[:, :, ch].astype(np.int32)
        act = zero_border_activity(plane)
        H, W = plane.shape
        stat = a1 = a4 = 0
        pool_idx, pool_sym, pool_act = [], [], []
        mismatch = 0
        for gr in g["groups"]:
            sym = gr["sym"]
            tot, k, d = static_verify(sym)
            if (k, d) != (gr["k"], gr["d"]):
                mismatch += 1
            stat += tot
            order = np.argsort(gr["idx"], kind="stable")
            s = sym[order]
            ii = gr["idx"][order]
            assert (np.diff(ii) > 0).all(), "dump order must be raster"
            a1 += adaptive_bits(s, np.zeros_like(s), 1)
            ac = np.array([act_bin(act[i // W, i % W]) for i in ii])
            a4 += adaptive_bits(s, ac, 4)
            pool_idx.append(ii)
            pool_sym.append(s)
            pool_act.append(ac)
        if mismatch:
            print(f"  ch{ch}: FIDELITY FAIL {mismatch} k/d mismatches — abort")
            raise SystemExit(1)
        pi = np.concatenate(pool_idx)
        ps = np.concatenate(pool_sym)
        pa = np.concatenate(pool_act)
        o = np.argsort(pi, kind="stable")
        b1 = adaptive_bits(ps[o], np.zeros_like(ps[o]), 1)
        b4 = adaptive_bits(ps[o], pa[o], 4)
        per_ch.append({"ch": ch, "fam": g["family"], "ng": g["ng"],
                       "ngroups": len(g["groups"]),
                       "nsyms": int(sum(len(x["sym"]) for x in g["groups"])),
                       "static": int(stat), "a1": int(a1), "a4": int(a4),
                       "b1": int(b1), "b4": int(b4)})
        print(f"  ch{ch}: fam={g['family']} Ggroups={len(g['groups'])} "
              f"static={stat} a1={a1}({(stat-a1)/stat*100:+.2f}%) "
              f"a4={a4}({(stat-a4)/stat*100:+.2f}%) "
              f"b1={b1}({(stat-b1)/stat*100:+.2f}%) "
              f"b4={b4}({(stat-b4)/stat*100:+.2f}%)", flush=True)
    return per_ch


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="/tmp/b22dump")
    ap.add_argument("--out", default="/tmp/opencode/autocompress/probes/probe_b22_nums.json")
    ap.add_argument("--images", nargs="*", default=["kodim05.png", "kodim13.png"])
    a = ap.parse_args()
    res = {}
    for fn in a.images:
        per_ch = run_image(fn, a.dump, WIN[fn])
        gap_bits = GAPS[fn] * NPIX
        tot = {k: sum(c[k] for c in per_ch) for k in ("static", "a1", "a4", "b1", "b4")}
        print(f"  -- {fn}: gap={gap_bits:.0f} bits "
              + " ".join(f"save_{k}={tot['static']-tot[k]} ({(tot['static']-tot[k])/gap_bits*100:.0f}% of gap)"
                         for k in ("a1", "a4", "b1", "b4")), flush=True)
        ok = {k: (tot["static"] - tot[k]) >= gap_bits * 1.2 for k in ("a1", "a4", "b1", "b4")}
        print(f"  -- gate(x1.2): {ok}", flush=True)
        res[fn] = {"channels": per_ch, "gap_bits": gap_bits,
                   "totals": {k: int(v) for k, v in tot.items()},
                   "saves": {k: int(tot["static"] - tot[k]) for k in ("a1", "a4", "b1", "b4")},
                   "gate": ok}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote " + a.out, flush=True)


if __name__ == "__main__":
    main()
