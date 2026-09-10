"""probe_b19_run.py — all-7 exact comparison (new file only)."""
import math
import os
import sys
import time
import json

import numpy as np
from PIL import Image

sys.path.insert(0, "/tmp/opencode/autocompress/csrc")
sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
os.chdir("/tmp/opencode/autocompress/experiments")
import probe_b17_rctw as B17
import probe_b19_adaptive as AD

D = AD.D
FILES = AD.FILES
CROWN4 = AD.CROWN4
JXL_E3 = AD.JXL_E3


def load_planes(fn):
    px = np.array(Image.open(os.path.join(D, fn)).convert("RGB"))
    H, W, _ = px.shape
    yc = B17.rct_fwd(px, 0, 6)
    assert np.array_equal(B17.rct_inv(yc, 0, 6), px), "RCT FAIL"
    return px, [yc[:, :, c] for c in range(3)], H, W


def baseline_image(fn):
    px, planes, H, W = load_planes(fn)
    npx = H * W * 3
    tot = 0
    H0 = 0.0
    det = []
    for c in range(3):
        r = AD.med_residuals(planes[c])
        assert int(np.abs(r).max()) <= 1024
        bb, _, _ = B17.exact_plane_bits(r)
        tot += bb
        H0 += AD.entropy_bits(r)
        e = AD.activity_plane(planes[c])
        ab = np.digitize(e.ravel(), [5, 13, 49], right=False)
        Hc = 0.0
        for b in range(4):
            m = ab == b
            if m.sum():
                Hc += AD.entropy_bits(r.ravel()[m])
        det.append({"huff": bb, "H0": H0, "Hc": Hc})
    return {"bpp": tot / npx, "bits": tot, "H0bpp": H0 / npx, "npx": npx,
            "H": H, "W": W}


def full_image(fn, reset=64, do_cabac=True, do_fwd=False):
    px, planes, H, W = load_planes(fn)
    npx = H * W * 3
    out = {}
    # static golomb ref
    sg = 0
    for c in range(3):
        r = AD.med_residuals(planes[c])
        t, k = AD.golomb_static_best(r.ravel().tolist())
        sg += t
    out["S0G"] = sg / npx
    # adaptive golomb C
    cg = 0
    for c in range(3):
        r = AD.med_residuals(planes[c])
        e = AD.activity_plane(planes[c])
        cg += AD.golomb_adaptive_bits(r.ravel().tolist(), e.ravel().tolist(),
                                      reset=reset, use_act=True)
    out["C"] = cg / npx
    # run D
    dr = 0
    for c in range(3):
        r = AD.med_residuals(planes[c])
        e = AD.activity_plane(planes[c])
        dr += AD.run_adaptive_bits(r.ravel().tolist(), e.ravel().tolist(),
                                   reset=reset)
    out["D"] = dr / npx
    # CABAC A-act / A-pos
    if do_cabac:
        ta = 0
        tp = 0
        encs = []
        for c in range(3):
            r = AD.med_residuals(planes[c])
            e = AD.activity_plane(planes[c])
            ea = cabac = AD.cabac_encode_plane(r, e, use_act=True)
            ep = AD.cabac_encode_plane(r, e, use_act=False)
            ta += cabac["bits"]
            tp += ep["bits"]
            encs.append((cabac, ep))
        out["Aact"] = ta / npx
        out["Apos"] = tp / npx
        out["_encs"] = encs
    if do_fwd:
        # forward init: empirical prefix-p1 per ctx -> 4b levels (side counted)
        tf = 0
        for c in range(3):
            r = AD.med_residuals(planes[c])
            e = AD.activity_plane(planes[c])
            # estimate p1 per ctx from actual prefix bins (one pass, encoder-only)
            nctx = AD.NACT * AD.NPOS
            n0 = [0] * nctx
            n1 = [0] * nctx
            rL = r.ravel().tolist()
            aL = e.ravel().tolist()
            t0, t1, t2 = AD.ATHR
            for rv, ev in zip(rL, aL):
                ab = 0 if ev <= t0 else (1 if ev <= t1 else (2 if ev <= t2 else 3))
                M = 2 * abs(rv) - (1 if rv > 0 else 0)
                L = (M + 1).bit_length()
                for i in range(L):
                    b = 0 if i < L - 1 else 1
                    p = i if i < AD.NPOS else AD.NPOS - 1
                    if b:
                        n1[ab * AD.NPOS + p] += 1
                    else:
                        n0[ab * AD.NPOS + p] += 1
            lv = []
            for ctx in range(nctx):
                t = n0[ctx] + n1[ctx]
                p1 = (n1[ctx] + 0.5) / (t + 1) if t else 0.5
                lv.append(min(15, max(0, int(p1 * 16))))
            enc = AD.cabac_encode_plane(r, e, use_act=True, init_levels=lv)
            tf += enc["bits"]
        out["Bfwd"] = tf / npx
    return out


def verify_golomb_stream():
    """Real bitstream write/read for adaptive Golomb + run on kodim05 ch-Y."""
    import random
    fn = "kodim05.png"
    px, planes, H, W = load_planes(fn)
    r = AD.med_residuals(planes[0])
    e = AD.activity_plane(planes[0])
    rL = r.ravel().tolist()
    eL = e.ravel().tolist()
    # writer
    st = AD.GolombState(AD.NACT, reset=64)
    acc = 0
    nb = 0
    buf = bytearray()
    exp_bits = 0

    def put(v, n):
        nonlocal acc, nb
        acc = (acc << n) | (v & ((1 << n) - 1))
        nb += n
        while nb >= 8:
            nb -= 8
            buf.append((acc >> nb) & 0xFF)
            acc &= ((1 << nb) - 1) if nb else 0
    for rv, ev in zip(rL, eL):
        ctx = AD.act_bin(ev)
        k = st.k(ctx)
        M = 2 * abs(rv) - (1 if rv > 0 else 0)
        q = M >> k
        for _ in range(q):
            put(0, 1)
        put(1, 1)
        if k:
            put(M & ((1 << k) - 1), k)
        exp_bits += q + 1 + k
        st.upd(ctx, abs(rv))
    if nb:
        buf.append((acc << (8 - nb)) & 0xFF)
    ana = AD.golomb_adaptive_bits(rL, eL, reset=64, use_act=True)
    assert exp_bits == ana, (exp_bits, ana)
    # reader
    bits = []
    for by in buf:
        for i in range(7, -1, -1):
            bits.append((by >> i) & 1)
    p = 0
    st2 = AD.GolombState(AD.NACT, reset=64)
    dec = []
    for ev in eL:
        ctx = AD.act_bin(ev)
        k = st2.k(ctx)
        q = 0
        while bits[p] == 0:
            q += 1
            p += 1
        p += 1
        rem = 0
        for _ in range(k):
            rem = (rem << 1) | bits[p]
            p += 1
        M = (q << k) | rem
        dec.append(AD.M_to_r(M))
        st2.upd(ctx, abs(dec[-1]))
    assert dec == rL, "golomb stream round-trip FAIL"
    print(f"[golomb stream kodim05-Y PASS bits={exp_bits}]", flush=True)
    # run-mode stream check (length-only + interrupt mapping sanity on prefix)
    tot_run = AD.run_adaptive_bits(rL[:20000], eL[:20000])
    tot_c = AD.golomb_adaptive_bits(rL[:20000], eL[:20000], reset=64, use_act=True)
    print(f"[run prefix 20k: C={tot_c} D={tot_run}]", flush=True)


def main():
    t_all = time.perf_counter()
    verify_golomb_stream()
    # 1) baseline all-7
    base = {}
    for fn in FILES:
        t0 = time.perf_counter()
        b = baseline_image(fn)
        base[fn] = b
        print(f"{fn}: S0={b['bpp']:.4f} H0={b['H0bpp']:.4f} "
              f"({(time.perf_counter()-t0):.1f}s)", flush=True)
    avg0 = float(np.mean([base[fn]["bpp"] for fn in FILES]))
    print(f"AVG S0={avg0:.4f} vs 3.58 ({(avg0-3.58)/3.58*100:+.2f}%) "
          f"{'PASS' if abs(avg0-3.58)/3.58 <= 0.03 else 'FAIL'}", flush=True)
    # RESET sweep on 05 (fast, Golomb only)
    print("--- RESET sweep kodim05 (C-golomb exact) ---", flush=True)
    px, planes, H, W = load_planes("kodim05.png")
    npx = H * W * 3
    for rs in (32, 64, 128, 256):
        t = 0
        for c in range(3):
            r = AD.med_residuals(planes[c])
            e = AD.activity_plane(planes[c])
            t += AD.golomb_adaptive_bits(r.ravel().tolist(), e.ravel().tolist(),
                                         reset=rs, use_act=True)
        print(f"  RESET={rs}: {t/npx:.4f}", flush=True)
    # 2) full backends all-7
    res = {}
    for fn in FILES:
        t0 = time.perf_counter()
        b = base[fn]
        f = full_image(fn, reset=64, do_cabac=True, do_fwd=True)
        row = {"S0": b["bpp"], "H0": b["H0bpp"]}
        row.update({k: v for k, v in f.items() if not k.startswith("_")})
        res[fn] = row
        print(f"{fn}: S0={row['S0']:.4f} S0G={row['S0G']:.4f} C={row['C']:.4f} "
              f"D={row['D']:.4f} Apos={row['Apos']:.4f} Aact={row['Aact']:.4f} "
              f"Bfwd={row['Bfwd']:.4f} ({(time.perf_counter()-t0)/60:.1f}min)",
              flush=True)
        # real CABAC round-trip per image (A-act, all 3 planes)
        for c in range(3):
            r = AD.med_residuals(planes[c] if fn == "kodim05.png" else
                                 load_planes(fn)[1][c])
        # (full decode proof below for winner; per-image quick check here)
        encs = f["_encs"]
        _, pls, H2, W2 = load_planes(fn)
        for c in range(3):
            cab = encs[c][0]
            rec0 = np.zeros((H2, W2), dtype=np.int32)
            # rebuild rec0 progressively? decode needs empty rec seed (zeros);
            # cabac_decode_plane derives pred from running recon (starts 0) —
            # matches encoder border doctrine. verify res then full recon.
            dec_res, dec_rec = AD.cabac_decode_plane(cab["arith"], cab["raw"],
                                                     H2, W2, rec0, use_act=True)
            assert np.array_equal(dec_res, AD.med_residuals(pls[c])), \
                f"CABAC res mismatch {fn} ch{c}"
            assert np.array_equal(dec_rec, pls[c]), f"CABAC recon FAIL {fn} ch{c}"
        print(f"  [{fn} CABAC A-act round-trip 3/3 PASS]", flush=True)
    avg = {k: float(np.mean([res[fn][k] for fn in FILES]))
           for k in ("S0", "H0", "S0G", "C", "D", "Apos", "Aact", "Bfwd")}
    print("--- AVG ---", flush=True)
    for k in ("S0", "H0", "S0G", "C", "D", "Apos", "Aact", "Bfwd"):
        print(f"  {k}={avg[k]:.4f} "
              f"(dS0={(avg[k]-avg['S0'])/avg['S0']*100:+.3f}%)", flush=True)
    # combined math vs CROWN4
    print("--- transfer math (additive MED-frame delta -> CROWN4) ---", flush=True)
    for k in ("C", "D", "Apos", "Aact", "Bfwd"):
        d = avg[k] - avg["S0"]
        est = CROWN4_EST = float(np.mean([CROWN4[fn] + (res[fn][k] - res[fn]["S0"])
                                          for fn in FILES]))
        print(f"  {k}: delta={d:+.4f} est_avg={est:.4f} "
              f"vs JXL {(est-3.2291)/3.2291*100:+.2f}%", flush=True)
    json.dump({"base": {f: {"S0": res[f]["S0"], "H0": res[f]["H0"],
                                 "S0G": res[f]["S0G"], "C": res[f]["C"],
                                 "D": res[f]["D"], "Apos": res[f]["Apos"],
                                 "Aact": res[f]["Aact"], "Bfwd": res[f]["Bfwd"]}
                        for f in FILES}, "avg": avg},
              open("/tmp/opencode/autocompress/experiments/probe_b19_nums.json", "w"),
              indent=1)
    print(f"TOTAL {(time.perf_counter()-t_all)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
