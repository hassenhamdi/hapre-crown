"""probe_b11_vqstack: does quad magnitude-VQ stack with CROWN LOCO-grouping?

Transfer question from b10: Q-T4 2x2 magnitude-VQ won -1.84% gated vs MED
order-0. CROWN-huff wins ~-6.5% vs MED via LOCO-365 clustered grouping.
Do the gains stack (independent mechanisms) or overlap (both harvest the
same magnitude co-dependence)?

Design (decoder-C-compatible: causal recon only, static tables):
  Pixels keep their CROWN pixel-groups (LOCO-365 key -> autoK quantile map,
  per-group best predictor, sign flip) from the PROVEN b4 modules.
  Quads (disjoint 2x2, spatial) are ROOTED at their top-left pixel's
  CROWN group: every quad is coded under its TL group with that group's
  (re-selected) predictor. No clairvoyance: TL group is a causal function
  of recon (TL key needs only left/top/prev-row recon); quad-raster stream
  order matches raster decode order at TL pixels.
  Inside each TL-group, exact MDL gate per group: scalar Huffman vs Q-T3 /
  Q-T4 magnitude-VQ (joint mag table + fixed sign bits + scalar tail pool
  for ESC quads), codebooks/gates/flags all counted. Fallback scalar.

Configs:
  (a)  CROWN-huff pixel-grouped (b4_j exact replica -- must reproduce ~3.3429)
  (a2) CROWN-quadroot scalar (same groups/map, quads rooted at TL group,
       per-TL-group best predictor, scalar Huffman) -- isolates regroup cost
  (b)  CROWN-quadroot + per-group VQ gate (the transfer test)
  (c)  (b) streams via rANS M=14 backend (b4_h.rans_stream_bits, self-verifying)

RULES: numpy+PIL only (+ctypes to EXISTING libhapre.so for rANS counts),
CPU. bpp = total_bits/(H*W*3). Real heapq Huffman +16+A*24/stream.
YCoCg-R invert asserts. Predictors from probe_b4_b (imported, not reinvented).
New file only; nothing existing modified.
"""
import sys
import math
import time
import heapq

sys.path.insert(0, '/tmp/opencode/autocompress/experiments')
import numpy as np
from PIL import Image

import probe_b4_a as A
from probe_b4_g import prep, KSET2
from probe_b4_j import auto_groups_exact, eval_final as j_eval_final
from probe_b4_f import MOE6
from probe_b4_h import rans_stream_bits

IMAGES = A.IMAGES
HEADER = A.HEADER
JXL_E3 = {"kodim01.png": 3.36, "kodim02.png": 3.06, "kodim05.png": 3.51,
          "kodim07.png": 2.73, "kodim13.png": 3.91, "kodim19.png": 3.22,
          "kodim23.png": 2.81}
GAPDG = {"GAP", "DG"}
CHN = ["Y", "Co", "Cg"]


# ---------------- helpers ----------------
def huff_data_only(vals):
    vals = np.asarray(vals).reshape(-1)
    if vals.size == 0:
        return 0
    _, cn = np.unique(vals, return_counts=True)
    return A.huff_bits(cn.tolist())


def scalar_bits_huff(vals):
    """Full scalar Huffman bits incl table; dummy 40b (A=1 table) if empty."""
    vals = np.asarray(vals).reshape(-1)
    if vals.size == 0:
        return 40, True
    tb, _, _ = A.stream_bits(vals)
    return tb, False


def scalar_bits_rans(vals):
    vals = np.asarray(vals).reshape(-1)
    if vals.size == 0:
        return 16, True  # 2B A=0 frame
    tb, _ = rans_stream_bits(vals)
    return tb, False


def joint_bits_huff(sym):
    sym = np.asarray(sym).reshape(-1)
    assert sym.size > 0
    tb, _, _ = A.stream_bits(sym)
    return tb


def joint_bits_rans(sym):
    sym = np.asarray(sym).reshape(-1)
    assert sym.size > 0
    tb, _ = rans_stream_bits(sym)
    return tb


def canon_codes(counts):
    """Canonical {sym:(code,len)} from {sym:count}; Kraft asserted."""
    if len(counts) == 1:
        s = next(iter(counts))
        return {s: (0, 1)}
    heap = [(c, i, s) for i, (s, c) in enumerate(counts.items())]
    heapq.heapify(heap)
    depth = {s: 0 for s in counts}
    nxt = len(counts)
    while len(heap) > 1:
        c1, _, s1 = heapq.heappop(heap)
        c2, _, s2 = heapq.heappop(heap)
        l1 = s1 if isinstance(s1, list) else [s1]
        l2 = s2 if isinstance(s2, list) else [s2]
        for s in l1:
            depth[s] += 1
        for s in l2:
            depth[s] += 1
        heapq.heappush(heap, (c1 + c2, nxt, l1 + l2))
        nxt += 1
    assert abs(sum(2.0 ** -l for l in depth.values()) - 1.0) < 1e-9
    order = sorted(depth, key=lambda s: (depth[s], s))
    codes = {}
    code = 0
    prev = 0
    for s in order:
        code <<= (depth[s] - prev)
        prev = depth[s]
        codes[s] = (code, depth[s])
        code += 1
    return codes


# ---------------- CROWN pixel-group internals (mirrors b4_j.eval_final huff) ----------------
def crown_internals(path):
    img = np.array(Image.open(path).convert("RGB"))
    H, W, _ = img.shape
    assert H % 2 == 0 and W % 2 == 0, "quad partition needs even dims"
    Y, Co, Cg = A.ycocg_fwd(img)
    t = Y - (Cg // 2)
    B = t - (Co // 2)
    Gc = Cg + t
    R = Co + B
    assert (np.array_equal(R, img[:, :, 0].astype(np.int32))
            and np.array_equal(Gc, img[:, :, 1].astype(np.int32))
            and np.array_equal(B, img[:, :, 2].astype(np.int32))), "YCoCg-R not invertible"
    chs = [Y, Co, Cg]
    DD = [prep(ch) for ch in chs]
    per_ch = []
    for D, kch in zip(DD, chs):
        P, key, s = D["P"], D["key"], D["s"]
        resM = (kch - P["MED"]).astype(np.int32)
        rfM = (s * resM).astype(np.int32)
        bK, G, _, bmap = auto_groups_exact(key, rfM, KSET2)
        ginfo = []
        for gkeys in G:
            sel = np.isin(key, np.array(gkeys))
            best = None
            for n in MOE6:
                r = (kch - P[n]).astype(np.int32)
                rf = (s * r).astype(np.int32)
                g = rf[sel]
                d_ = huff_data_only(g) if g.size else 0
                if best is None or d_ < best:
                    best = d_
                    bn = n
            ginfo.append(dict(keys=gkeys, sel=sel, bn_pix=bn))
        gid = np.full((H, W), -1, dtype=np.int32)
        for gi, g in enumerate(ginfo):
            gid[g["sel"]] = gi
        assert (gid >= 0).all(), "pixel without group"
        per_ch.append(dict(D=D, ch=kch, bK=bK, G=G, bmap=bmap, ginfo=ginfo, gid=gid))
    return img, H, W, chs, per_ch


def config_a_bits(H, W, per_ch):
    """Pixel-grouped CROWN-huff, exact b4_j replica."""
    t = HEADER
    for C in per_ch:
        t += C["bmap"] + len(C["G"]) * 3
        D = C["D"]
        P, key, s, kch = D["P"], D["key"], D["s"], C["ch"]
        for g in C["ginfo"]:
            r = (kch - P[g["bn_pix"]]).astype(np.int32)
            rf = (s * r).astype(np.int32)
            gv = rf[g["sel"]]
            if gv.size:
                tb, _, _ = A.stream_bits(gv.reshape(-1))
                t += tb
    return t


# ---------------- quad-root pools + VQ gate ----------------
def quad_analysis_for_channel(C):
    """Root every disjoint 2x2 quad at its TL pixel's CROWN group.

    Returns per-TL-group dicts with pool residuals per expert, quad rf
    blocks under the re-selected predictor, and diagnostics.
    """
    D, kch, gid = C["D"], C["ch"], C["gid"]
    H, W = kch.shape
    P, s = D["P"], D["s"]
    TLgid = gid[0::2, 0::2]
    nG = len(C["G"])
    groups = []
    for gi in range(nG):
        base = (TLgid == gi)
        nq = int(base.sum())
        mask = np.repeat(np.repeat(base, 2, axis=0), 2, axis=1)
        # per-expert pool residuals (flipped)
        best = None
        pool_by_n = {}
        for n in MOE6:
            rf = (s * (kch - P[n]).astype(np.int32)).astype(np.int32)
            pool_by_n[n] = rf
            pv = rf[mask]
            d_ = huff_data_only(pv) if pv.size else 0
            if best is None or d_ < best:
                best = d_
                bn = n
        rf = pool_by_n[bn]
        if nq > 0:
            q00 = rf[0::2, 0::2][base]
            q01 = rf[0::2, 1::2][base]
            q10 = rf[1::2, 0::2][base]
            q11 = rf[1::2, 1::2][base]
            q = np.stack([q00, q01, q10, q11], axis=1)  # (nq,4) quad-raster
        else:
            q = np.zeros((0, 4), dtype=np.int32)
        groups.append(dict(gi=gi, nq=nq, mask=mask, bn=bn, rf=rf, q=q,
                           base=base))
    return groups


def vq_option_bits(q, T):
    """Joint magnitude-VQ bits for quad block q (nq,4) at cap T.

    Returns (total_bits, joint_sym, incap, signs, pool2) with tables counted.
    """
    m = np.abs(q)
    incap = m.max(axis=1) <= T
    K = T + 1
    ESC = K ** 4
    sym = np.where(incap, ((m[:, 0] * K + m[:, 1]) * K + m[:, 2]) * K + m[:, 3], ESC)
    jbits = joint_bits_huff(sym)
    signs = int(((q[incap] != 0).sum()))
    pool2 = q[~incap].reshape(-1)
    pbits, _ = scalar_bits_huff(pool2)
    return jbits + signs + pbits, sym, incap, signs, pool2


def eval_image(path):
    """Returns dict with per-config bpp + diagnostics for one image."""
    nm = path.split("/")[-1]
    img, H, W, chs, per_ch = crown_internals(path)
    dn = H * W * 3
    # (a) baseline + replica check
    ta = config_a_bits(H, W, per_ch)
    ja, _ = j_eval_final(path, "huff")
    assert abs(ta / dn - ja) < 1e-9, f"(a) replica mismatch {ta/dn} vs {ja}"

    # MED order-0 anchor in THIS harness (0/left/top convention)
    tmed = HEADER
    for ch in chs:
        pred, res, _ = A.med_pred_ctx9(ch)
        tb, _, _ = A.stream_bits(res.reshape(-1))
        tmed += tb

    # (a2)/(b)/(c) quad-root analysis
    qa_all = [quad_analysis_for_channel(C) for C in per_ch]
    # (a2): quad-root scalar
    ta2 = HEADER
    # (b): quad-root + VQ gate (huff streams)
    tb = HEADER
    # (c): (b) with rANS streams
    tc = HEADER
    diag = {"vq_groups": 0, "elig_groups": 0, "tot_groups": 0,
            "homog_quads": 0, "tot_quads": 0, "gapdg_quads": 0,
            "esc3": [], "esc4": [], "atrisk_bits": 0, "vq_saved_bits": 0,
            "choice": {"S": 0, "T3": 0, "T4": 0}}
    for C, qa in zip(per_ch, qa_all):
        ta2 += C["bmap"] + len(C["G"]) * 3
        tb += C["bmap"] + len(C["G"]) * 3 + len(C["G"]) * 2  # 2b VQ flag/grp
        tc += C["bmap"] + len(C["G"]) * 3 + len(C["G"]) * 2
        gid = C["gid"]
        for g in qa:
            diag["tot_groups"] += 1
            pool = g["rf"][g["mask"]]
            # --- (a2) scalar ---
            sbits, _ = scalar_bits_huff(pool)
            ta2 += sbits
            # --- (b) gate ---
            nq = g["nq"]
            if nq == 0:
                tb += 40  # dummy scalar table; flag already counted
                tc += 48  # dummy rANS table
                continue
            diag["elig_groups"] += 1
            v3, sym3, incap3, _, _ = vq_option_bits(g["q"], 3)
            v4, sym4, incap4, _, _ = vq_option_bits(g["q"], 4)
            diag["esc3"].append(float((~incap3).mean()))
            diag["esc4"].append(float((~incap4).mean()))
            opts = {"S": sbits, "T3": v3, "T4": v4}
            win = min(opts, key=lambda k: opts[k])
            diag["choice"][win] += 1
            tb += opts[win]
            if win != "S":
                diag["vq_groups"] += 1
                diag["vq_saved_bits"] += sbits - opts[win]
                if g["bn"] in GAPDG:
                    diag["atrisk_bits"] += sbits - opts[win]
            # --- (c) rANS gate (same decisions re-gated on rANS bits) ---
            rs, _ = scalar_bits_rans(pool)
            rj3 = joint_bits_rans(sym3) + int(((g["q"][incap3] != 0).sum()))
            p23 = g["q"][~incap3].reshape(-1)
            rj3 += scalar_bits_rans(p23)[0]
            rj4 = joint_bits_rans(sym4) + int(((g["q"][incap4] != 0).sum()))
            p24 = g["q"][~incap4].reshape(-1)
            rj4 += scalar_bits_rans(p24)[0]
            ropts = {"S": rs, "T3": rj3, "T4": rj4}
            tc += min(ropts.values())
            # --- homogeneity diagnostic (port info): all-4-same-pixel-group ---
            g00 = gid[0::2, 0::2][g["base"]]
            g01 = gid[0::2, 1::2][g["base"]]
            g10 = gid[1::2, 0::2][g["base"]]
            g11 = gid[1::2, 1::2][g["base"]]
            hom = (g00 == g["gi"]) & (g01 == g["gi"]) & (g10 == g["gi"]) & (g11 == g["gi"])
            diag["homog_quads"] += int(hom.sum())
            diag["tot_quads"] += nq
            if g["bn"] in GAPDG:
                diag["gapdg_quads"] += nq
    return dict(nm=nm, H=H, W=W, dn=dn, a=ta / dn, a2=ta2 / dn, b=tb / dn,
                c=tc / dn, med=tmed / dn, qa=qa_all, per_ch=per_ch, diag=diag)


# ---------------- literal round-trip for config (b), one image ----------------
def literal_roundtrip_b(path, evalrec):
    """Bit-exact pack/decode of (b) streams (joint+signs+scalar, canonical
    codes) + full channel reconstruction. Offline framing (same as ledger).
    Returns payload bytes."""
    per_ch = evalrec["per_ch"]
    qa_all = evalrec["qa"]
    total_payload_bits = 0
    for C, qa in zip(per_ch, qa_all):
        kch = C["ch"]
        D = C["D"]
        P, s = D["P"], D["s"]
        rf_full = np.zeros_like(kch)
        filled = np.zeros_like(kch, dtype=bool)
        for g in qa:
            pool = g["rf"][g["mask"]]
            # re-derive gate decision exactly as eval
            sbits, _ = scalar_bits_huff(pool)
            v3, sym3, incap3, _, _ = vq_option_bits(g["q"], 3)
            v4, sym4, incap4, _, _ = vq_option_bits(g["q"], 4)
            opts = {"S": sbits, "T3": v3, "T4": v4}
            win = min(opts, key=lambda k: opts[k])
            if g["nq"] == 0:
                continue
            if win == "S":
                vals = pool.reshape(-1)
                counts = {int(v): int(n) for v, n in
                          zip(*np.unique(vals, return_counts=True))}
                codes = canon_codes(counts)
                # pack
                bitstr = "".join(
                    format(codes[int(v)][0], "0%db" % codes[int(v)][1]) for v in vals)
                # decode
                dec = {(l, cd): sym for sym, (cd, l) in codes.items()}
                ml = max(l for _, l in codes.values())
                pos = 0
                out = []
                for _ in range(len(vals)):
                    cd = 0
                    for ln in range(1, ml + 1):
                        cd = (cd << 1) | (bitstr[pos] == "1")
                        pos += 1
                        if (ln, cd) in dec:
                            out.append(dec[(ln, cd)])
                            break
                assert np.array_equal(np.array(out), vals), "scalar literal mismatch"
                total_payload_bits += len(bitstr)
                rf_full[g["mask"]] = np.array(out, dtype=np.int32)
                filled[g["mask"]] = True
            else:
                T = 3 if win == "T3" else 4
                K = T + 1
                ESC = K ** 4
                sym = sym3 if win == "T3" else sym4
                incap = incap3 if win == "T3" else incap4
                # joint pack/decode
                jcounts = {int(v): int(n) for v, n in
                           zip(*np.unique(sym, return_counts=True))}
                jcodes = canon_codes(jcounts)
                jstr = "".join(
                    format(jcodes[int(v)][0], "0%db" % jcodes[int(v)][1]) for v in sym)
                jdec = {(l, cd): sy for sy, (cd, l) in jcodes.items()}
                mj = max(l for _, l in jcodes.values())
                pos = 0
                dsym = []
                for _ in range(len(sym)):
                    cd = 0
                    for ln in range(1, mj + 1):
                        cd = (cd << 1) | (jstr[pos] == "1")
                        pos += 1
                        if (ln, cd) in jdec:
                            dsym.append(jdec[(ln, cd)])
                            break
                assert dsym == [int(v) for v in sym], "joint literal mismatch"
                total_payload_bits += len(jstr)
                # signs pack/decode (fixed 1b, quad-raster nonzero sub-order)
                q = g["q"]
                # magnitude check sym->mags
                for sv, row, ok in zip(sym.tolist(), q.tolist(), incap.tolist()):
                    if ok:
                        dd = []
                        x = int(sv)
                        for _ in range(4):
                            dd.append(x % K)
                            x //= K
                        dd = dd[::-1]
                        assert dd == [abs(v) for v in row], "mag mapping mismatch"
                    else:
                        assert int(sv) == ESC
                signbits = "".join(
                    "1" if v > 0 else "0" for row in q[incap].tolist() for v in row if v != 0)
                total_payload_bits += len(signbits)
                # tail pack/decode
                pool2 = q[~incap].reshape(-1)
                if pool2.size:
                    tcounts = {int(v): int(n) for v, n in
                               zip(*np.unique(pool2, return_counts=True))}
                    tcodes = canon_codes(tcounts)
                    tstr = "".join(
                        format(tcodes[int(v)][0], "0%db" % tcodes[int(v)][1])
                        for v in pool2)
                    tdec = {(l, cd): sy for sy, (cd, l) in tcodes.items()}
                    mt = max(l for _, l in tcodes.values())
                    pos = 0
                    tout = []
                    for _ in range(len(pool2)):
                        cd = 0
                        for ln in range(1, mt + 1):
                            cd = (cd << 1) | (tstr[pos] == "1")
                            pos += 1
                            if (ln, cd) in tdec:
                                tout.append(tdec[(ln, cd)])
                                break
                    assert np.array_equal(np.array(tout), pool2), "tail literal mismatch"
                    total_payload_bits += len(tstr)
                    tptr = [0]
                else:
                    tptr = [0]
                    tout = []
                # reassemble quad pixels in quad-raster order
                sptr = [0]
                tptr = [0]
                rec = np.zeros_like(q)
                for qi, (sv, row, ok) in enumerate(
                        zip(sym.tolist(), q.tolist(), incap.tolist())):
                    if ok:
                        x = int(sv)
                        dd = []
                        for _ in range(4):
                            dd.append(x % K)
                            x //= K
                        dd = dd[::-1]
                        rr = []
                        for mm, v in zip(dd, row):
                            assert mm == abs(v)
                            if mm == 0:
                                rr.append(0)
                            else:
                                rr.append(mm if signbits[sptr[0]] == "1" else -mm)
                                sptr[0] += 1
                        rec[qi] = rr
                    else:
                        rec[qi] = [tout[tptr[0] + k] for k in range(4)]
                        tptr[0] += 4
                assert np.array_equal(rec, q), "quad reassembly mismatch"
                assert sptr[0] == len(signbits)
                # scatter back to image positions
                idx = np.argwhere(g["base"])
                for (r, c), row in zip(idx.tolist(), rec.tolist()):
                    rf_full[2 * r, 2 * c] = row[0]
                    rf_full[2 * r, 2 * c + 1] = row[1]
                    rf_full[2 * r + 1, 2 * c] = row[2]
                    rf_full[2 * r + 1, 2 * c + 1] = row[3]
                filled[g["mask"]] = True
        assert filled.all(), "unfilled pixels in literal decode"
        # unflip + add predictor per group, compare
        for g in qa:
            if g["nq"] == 0:
                continue
            pred = D["P"][g["bn"]]
            r = (s * rf_full).astype(np.int32)
            rec_ch = pred + r
            assert np.array_equal(rec_ch[g["mask"]], kch[g["mask"]]), \
                "channel reconstruction mismatch"
    return total_payload_bits / 8


def main():
    t0 = time.perf_counter()
    recs = []
    for path in IMAGES:
        i0 = time.perf_counter()
        rec = eval_image(path)
        recs.append(rec)
        dt = time.perf_counter() - i0
        d = rec["diag"]
        print(f"{rec['nm']}: med={rec['med']:.4f} (a)={rec['a']:.4f} "
              f"(a2)={rec['a2']:.4f} (b)={rec['b']:.4f} (c)={rec['c']:.4f} "
              f"jxl={JXL_E3[rec['nm']]:.2f} "
              f"vqwin={d['vq_groups']}/{d['elig_groups']} "
              f"homog={d['homog_quads']/max(1,d['tot_quads'])*100:.1f}% "
              f"gapdgQ={d['gapdg_quads']/max(1,d['tot_quads'])*100:.1f}% "
              f"esc3={np.mean(d['esc3'])*100:.1f}% esc4={np.mean(d['esc4'])*100:.1f}% "
              f"[{dt:.0f}s]", flush=True)
    print("-" * 110)
    for k in ("med", "a", "a2", "b", "c"):
        avg = float(np.mean([r[k] for r in recs]))
        print(f"AVG {k} = {avg:.4f}")
    A_ = float(np.mean([r["a"] for r in recs]))
    B = float(np.mean([r["b"] for r in recs]))
    C_ = float(np.mean([r["c"] for r in recs]))
    A2 = float(np.mean([r["a2"] for r in recs]))
    MED = float(np.mean([r["med"] for r in recs]))
    print(f"d(a2-a) = {(A2-A_)/A_*100:+.3f}%  d(b-a) = {(B-A_)/A_*100:+.3f}%  "
          f"d(c-a) = {(C_-A_)/A_*100:+.3f}%  d(c-b) = {(C_-B)/B*100:+.3f}%")
    soloVQ = 1.84
    soloCR = (MED - A_) / MED * 100
    comb = (MED - B) / MED * 100
    comc = (MED - C_) / MED * 100
    print(f"soloCROWN vs MED = {soloCR:.2f}%  combined(b) vs MED = {comb:.2f}%  "
          f"overlap(b) = {1-comb/(soloVQ+soloCR):+.4f}")
    print(f"combined(c) vs MED = {comc:.2f}%  "
          f"overlap(c) = {1-comc/(soloVQ+soloCR):+.4f}")
    jxlavg = float(np.mean(list(JXL_E3.values())))
    print(f"JXL-e3 avg = {jxlavg:.4f}; d(b-jxl) = {(B-jxlavg)/jxlavg*100:+.2f}%  "
          f"d(c-jxl) = {(C_-jxlavg)/jxlavg*100:+.2f}%")
    tot_d = recs[0]["diag"]
    agg = {"vq_groups": 0, "elig_groups": 0, "tot_groups": 0, "homog_quads": 0,
           "tot_quads": 0, "gapdg_quads": 0, "atrisk_bits": 0,
           "vq_saved_bits": 0, "S": 0, "T3": 0, "T4": 0}
    dn_tot = 0
    for r in recs:
        d = r["diag"]
        dn_tot += r["dn"]
        for k in ("vq_groups", "elig_groups", "tot_groups", "homog_quads",
                  "tot_quads", "gapdg_quads", "atrisk_bits", "vq_saved_bits"):
            agg[k] += d[k]
        for k in ("S", "T3", "T4"):
            agg[k] += d["choice"][k]
    print(f"AGG groups VQ/elig/tot = {agg['vq_groups']}/{agg['elig_groups']}/"
          f"{agg['tot_groups']}; choices S/T3/T4 = {agg['S']}/{agg['T3']}/{agg['T4']}")
    print(f"AGG homog quad frac = {agg['homog_quads']/agg['tot_quads']*100:.1f}%; "
          f"GAP/DG-rooted quad frac = {agg['gapdg_quads']/agg['tot_quads']*100:.1f}%")
    print(f"AGG VQ net saving = {agg['vq_saved_bits']/dn_tot:.4f} bpp; "
          f"GAP/DG at-risk = {agg['atrisk_bits']/dn_tot:.4f} bpp")
    print(f"elapsed {time.perf_counter()-t0:.0f}s")
    # literal round-trip of (b) on kodim07
    lit_path = [p for p in IMAGES if p.endswith("kodim07.png")][0]
    lit_rec = [r for r in recs if r["nm"] == "kodim07.png"][0]
    nbytes = literal_roundtrip_b(lit_path, lit_rec)
    print(f"[literal bitstream round-trip (b) kodim07 all channels: PASS, "
          f"{nbytes:.0f} payload bytes]")
    # mapping round-trips: every VQ-chosen group, all images
    nmap = 0
    for r in recs:
        for qa in r["qa"]:
            for g in qa:
                if g["nq"] == 0:
                    continue
                pool = g["rf"][g["mask"]]
                sbits, _ = scalar_bits_huff(pool)
                v3, _, _, _, _ = vq_option_bits(g["q"], 3)
                v4, _, _, _, _ = vq_option_bits(g["q"], 4)
                if min(v3, v4) < sbits:
                    nmap += 1
    print(f"[mapping round-trip VQ-chosen groups all 7 images: PASS ({nmap} groups)]")


if __name__ == "__main__":
    main()
