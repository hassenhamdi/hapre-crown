"""probe_b20_run.py — all-7 joint-adaptive driver (new file only)."""
import json
import math
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, "/tmp/opencode/autocompress/csrc")
sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
os.chdir("/tmp/opencode/autocompress/experiments")
import probe_b17_rctw as B17
import probe_b19_adaptive as AD
import probe_b20_joint as J

D = J.D
FILES = J.FILES
JXL_E3 = J.JXL_E3
CROWN4 = J.CROWN4
JG_KS = [4, 8, 16, 32, 64]
RESETS = [32, 64, 128]


def load_yc(fn):
    px = np.array(Image.open(os.path.join(D, fn)).convert("RGB"))
    H, W, _ = px.shape
    yc = B17.rct_fwd(px, 0, 6)
    assert np.array_equal(B17.rct_inv(yc, 0, 6), px), "RCT FAIL"
    return px, [yc[:, :, c].astype(np.int32) for c in range(3)], H, W


def main():
    t_all = time.perf_counter()
    npx_all = {}
    # per-image per-plane stores
    pp = {fn: {} for fn in FILES}  # pp[fn][cfg] = [b0,b1,b2] plane bits
    S = {}  # image-level bpp rows

    # ---- 0. S0 anchor ----
    print("== S0 anchor ==", flush=True)
    for fn in FILES:
        px, planes, H, W = load_yc(fn)
        npx = H * W * 3
        npx_all[fn] = npx
        tot = 0
        planes_bits = []
        for c in range(3):
            r = (planes[c] - B17.med_pred(planes[c])).astype(np.int32)
            bb, _, _ = B17.exact_plane_bits(r)
            tot += bb
            planes_bits.append(bb)
        pp[fn]["S0"] = planes_bits
        S.setdefault(fn, {})["S0"] = tot / npx
        print(f"  {fn}: S0={tot / npx:.4f}", flush=True)
    avg0 = float(np.mean([S[fn]["S0"] for fn in FILES]))
    print(f"AVG S0={avg0:.4f} vs 3.58 ({(avg0 - 3.58) / 3.58 * 100:+.2f}%) "
          f"{'PASS' if abs(avg0 - 3.58) / 3.58 <= 0.03 else 'FAIL'}", flush=True)

    # ---- front-end symbols (cached) ----
    print("== front end ==", flush=True)
    FE = {}
    for fn in FILES:
        px, planes, H, W = load_yc(fn)
        FE[fn] = {"H": H, "W": W, "planes": []}
        for c in range(3):
            FE[fn]["planes"].append(J.plane_symbols(planes[c]))
        print(f"  {fn} symbols ok", flush=True)

    # ---- 1. static grouped SG-K ----
    for K in (12, 64):
        print(f"== SG-{K} ==", flush=True)
        for fn in FILES:
            H, W = FE[fn]["H"], FE[fn]["W"]
            npx = npx_all[fn]
            tot = 0
            pb = []
            det = []
            for c in range(3):
                P = FE[fn]["planes"][c]
                r = J.static_grouped_plane_bits(P["symf"], P["keyf"], K,
                                                np.abs(P["sym"]), H, W)
                tot += r["bits"]
                pb.append(r["bits"])
                det.append(r)
            pp[fn][f"SG{K}"] = pb
            S[fn][f"SG{K}"] = tot / npx
            if K == 64:
                FE[fn][f"SG{K}"] = det
            print(f"  {fn}: SG{K}={tot / npx:.4f}", flush=True)

    # ---- 2. global controls CG / AA on identical sym ----
    print("== CG / AA ==", flush=True)
    for fn in FILES:
        npx = npx_all[fn]
        tcg = 0
        taa = 0
        pbc, pba = [], []
        for c in range(3):
            P = FE[fn]["planes"][c]
            e = P["act"]
            tcg_c = AD.golomb_adaptive_bits(P["symf"], e.ravel().tolist(),
                                            reset=64, use_act=True)
            tcg += tcg_c
            pbc.append(tcg_c)
            enc = AD.cabac_encode_plane(P["sym"].astype(np.int32),
                                        e.astype(np.int32), use_act=True)
            taa += enc["bits"]
            pba.append(enc["bits"])
        pp[fn]["CG"] = pbc
        pp[fn]["AA"] = pba
        S[fn]["CG"] = tcg / npx
        S[fn]["AA"] = taa / npx
        print(f"  {fn}: CG={tcg / npx:.4f} AA={taa / npx:.4f}", flush=True)

    # ---- 3. JG sweep ----
    for K in JG_KS:
        print(f"== JG-{K} (RESET=64) ==", flush=True)
        for fn in FILES:
            H, W = FE[fn]["H"], FE[fn]["W"]
            npx = npx_all[fn]
            tot = 0
            pb = []
            luts = []
            for c in range(3):
                P = FE[fn]["planes"][c]
                groups = J.quantile_groups(P["key"], np.abs(P["sym"]), K)
                lut = J.build_lut(groups)
                n_active = int((lut >= 0).sum())
                ms = J.map_side_bits(n_active, K)
                gid = lut[P["key"].ravel()].tolist()
                r = J.jg_analytic(P["M"], P["absf"], P["ab"], gid, K, reset=64)
                tot += r["bits"] + ms
                pb.append(r["bits"] + ms)
                luts.append((groups, lut))
            pp[fn][f"JG{K}"] = pb
            S[fn][f"JG{K}"] = tot / npx
            FE[fn][f"JG{K}"] = luts
            print(f"  {fn}: JG{K}={tot / npx:.4f}", flush=True)

    avgs = {k: float(np.mean([S[fn][k] for fn in FILES])) for k in S[FILES[0]]}
    print("--- AVG so far ---", flush=True)
    for k in sorted(avgs):
        print(f"  {k}={avgs[k]:.4f} (dS0={(avgs[k] - avgs['S0']) / avgs['S0'] * 100:+.3f}%)", flush=True)
    bestK = min(JG_KS, key=lambda k: float(np.mean([S[fn][f"JG{k}"] for fn in FILES])))
    print(f"bestK={bestK}", flush=True)

    # ---- 4. RESET sweep on bestK ----
    for rs in RESETS:
        if rs == 64:
            continue
        print(f"== JG-{bestK} RESET={rs} ==", flush=True)
        for fn in FILES:
            npx = npx_all[fn]
            tot = 0
            pb = []
            for c in range(3):
                P = FE[fn]["planes"][c]
                groups, lut = FE[fn][f"JG{bestK}"][c]
                n_active = int((lut >= 0).sum())
                ms = J.map_side_bits(n_active, bestK)
                gid = lut[P["key"].ravel()].tolist()
                r = J.jg_analytic(P["M"], P["absf"], P["ab"], gid, bestK, reset=rs)
                tot += r["bits"] + ms
                pb.append(r["bits"] + ms)
            pp[fn][f"JG{bestK}R{rs}"] = pb
            S[fn][f"JG{bestK}R{rs}"] = tot / npx
            print(f"  {fn}: JG{bestK}R{rs}={tot / npx:.4f}", flush=True)

    # ---- 5. JA joint arithmetic on bestK (+neighbor) ----
    for K in sorted(set([bestK, 16])):
        print(f"== JA-{K} (real coder) ==", flush=True)
        for fn in FILES:
            H, W = FE[fn]["H"], FE[fn]["W"]
            npx = npx_all[fn]
            t0 = time.perf_counter()
            tot = 0
            pb = []
            encs = []
            for c in range(3):
                P = FE[fn]["planes"][c]
                if K == bestK:
                    groups, lut = FE[fn][f"JG{bestK}"][c]
                else:
                    groups = J.quantile_groups(P["key"], np.abs(P["sym"]), K)
                    lut = J.build_lut(groups)
                n_active = int((lut >= 0).sum())
                ms = J.map_side_bits(n_active, K)
                gid = lut[P["key"].ravel()].tolist()
                enc = J.ja_encode_plane(P["symf"], P["ab"], gid, K)
                tot += enc["bits"] + ms
                pb.append(enc["bits"] + ms)
                encs.append((enc, lut))
            pp[fn][f"JA{K}"] = pb
            S[fn][f"JA{K}"] = tot / npx
            FE[fn][f"JA{K}"] = encs
            print(f"  {fn}: JA{K}={tot / npx:.4f} ({(time.perf_counter() - t0):.1f}s)", flush=True)

    avgs = {k: float(np.mean([S[fn][k] for fn in FILES])) for k in S[FILES[0]]}
    print("--- AVG all ---", flush=True)
    for k in sorted(avgs):
        print(f"  {k}={avgs[k]:.4f} (dS0={(avgs[k] - avgs['S0']) / avgs['S0'] * 100:+.3f}%)", flush=True)

    # ---- 6. per-plane best-of (2b choice side/plane) ----
    cands = [f"SG64", f"JG{bestK}", f"JA{bestK}", "AA"]
    cands = [c for c in cands if c in pp[FILES[0]]]
    print(f"== best-of {cands} ==", flush=True)
    for fn in FILES:
        npx = npx_all[fn]
        tot = 0
        for c in range(3):
            tot += min(pp[fn][k][c] for k in cands) + 2
        S[fn]["BEST"] = tot / npx
        print(f"  {fn}: BEST={tot / npx:.4f}", flush=True)

    # ---- 7. winner round-trips ----
    # pick winner among joint configs by avg
    jkeys = [f"JG{bestK}"] + ([f"JG{bestK}R32", f"JG{bestK}R128"] if f"JG{bestK}R32" in pp[FILES[0]] else []) + [f"JA{bestK}"]
    jkeys = [k for k in jkeys if k in pp[FILES[0]]]
    win = min(jkeys, key=lambda k: float(np.mean([S[fn][k] for fn in FILES])))
    print(f"WINNER={win}", flush=True)
    if win.startswith("JG"):
        rs = 64
        if "R32" in win:
            rs = 32
        elif "R128" in win:
            rs = 128
        for fn in FILES:
            px, planes, H, W = load_yc(fn)
            for c in range(3):
                P = FE[fn]["planes"][c]
                groups, lut = FE[fn][f"JG{bestK}"][c]
                # map framing proof
                packed = J.pack_map(groups, bestK)
                lut2 = J.unpack_map(packed, bestK)
                assert np.array_equal(lut, lut2), f"map framing FAIL {fn} ch{c}"
                gid = lut[P["key"].ravel()].tolist()
                buf, nb = J.jg_encode_stream(P["M"], P["absf"], P["ab"], gid, bestK, reset=rs)
                ana = J.jg_analytic(P["M"], P["absf"], P["ab"], gid, bestK, reset=rs)["bits"]
                assert nb == ana, f"stream!=analytic {fn} ch{c}"
                dec_sym, dec_rec = J.jg_decode_plane(buf, H, W, lut2, bestK, reset=rs)
                assert np.array_equal(dec_sym, P["sym"]), f"JG sym mismatch {fn} ch{c}"
                assert np.array_equal(dec_rec, planes[c]), f"JG recon FAIL {fn} ch{c}"
            print(f"  [{fn} JG full-recon round-trip 3/3 PASS]", flush=True)
    else:
        K = bestK
        for fn in FILES:
            px, planes, H, W = load_yc(fn)
            for c in range(3):
                enc, lut = FE[fn][f"JA{K}"][c]
                dec_sym, dec_rec = J.ja_decode_plane(enc["arith"], enc["raw"], H, W, lut, K)
                assert np.array_equal(dec_sym, FE[fn]["planes"][c]["sym"]), f"JA sym mismatch {fn} ch{c}"
                assert np.array_equal(dec_rec, planes[c]), f"JA recon FAIL {fn} ch{c}"
            print(f"  [{fn} JA full-recon round-trip 3/3 PASS]", flush=True)

    # ---- 8. adaptation stats for winner/bestK ----
    stats = {}
    for fn in FILES:
        npx = npx_all[fn]
        # static tables vs zero
        sgt = FE[fn]["SG64"]
        tbl_bits = sum(d["tables"] for d in sgt)
        map_bits = sum(d["map"] for d in sgt)
        # JG ctx occupancy at bestK RESET64
        occ = None
        km = None
        totc = 0
        for c in range(3):
            P = FE[fn]["planes"][c]
            groups, lut = FE[fn][f"JG{bestK}"][c]
            gid = lut[P["key"].ravel()].tolist()
            r = J.jg_analytic(P["M"], P["absf"], P["ab"], gid, bestK, reset=64)
            totc += sum(1 for x in r["cnt"] if x > 0)
            if occ is None:
                occ = r["cnt"]
                km = [x for x in r["kmean"]]
        stats[fn] = {"sg_tables_bpp": tbl_bits / npx, "sg_map_bpp": map_bits / npx,
                     "jg_active_ctx": totc, "jg_nctx": bestK * 4 * 3}

    out = {"rows": {fn: dict(S[fn]) for fn in FILES},
           "avg": {k: float(np.mean([S[fn][k] for fn in FILES])) for k in S[FILES[0]]},
           "bestK": bestK, "winner": win, "stats": stats,
           "JXL": dict(JXL_E3), "CROWN4": dict(CROWN4)}
    json.dump(out, open("/tmp/opencode/autocompress/experiments/probe_b20_nums.json", "w"), indent=1)
    print("--- FINAL AVG ---", flush=True)
    for k in sorted(out["avg"]):
        print(f"  {k}={out['avg'][k]:.4f}", flush=True)
    print(f"TOTAL {(time.perf_counter() - t_all) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
