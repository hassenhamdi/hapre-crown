"""probe_b11_pairvq: fair stacking tests of magnitude-VQ on CROWN groups.

v1 finding (probe_b11_vqstack.py): TL-quad-root regrouping costs +3.85% --
it destroys the pixel-grouping gain first, so VQ-gating on top (8/410 groups,
+0.0045 bpp saved) cannot fairly answer the transfer question. TL-rooting is
kept as a negative control.

This file tests port-faithful AND pure stackings:
  (a)   CROWN-huff pixel-grouped (b4_j replica, expect 3.3429)
  (a2p) pair-LEFT-rooted scalar: disjoint 1x2 pairs coded under the LEFT
        pixel's CROWN group with per-pair-group best predictor. Fully
        streaming-causal for ALL 6 predictors (pair decode needs only
        same-row-left + prev-row recon; d is never future in horizontal
        pairing). Isolates pair-regroup cost.
  (bp)  (a2p) + per-pair-group MDL gate {scalar, P-T4, P-T8}, 2b flag/group.
        The portable stacking candidate.
  (bq)  PURE stacking: pixel-CROWN (a) untouched; quads with all 4 pixels in
        the same pixel-group choose min(scalar-for-those-pixels, Q-T4 joint).
        1b flag per group that owns >=1 homog quad (positions recon-derived).
        Offline framing (streaming port would need framing bits -- stated).
  (bqp) same as (bq) with homogeneous 1x2 PAIRS (higher coverage).
  (c)   rANS-M14 backend on the huff winner's streams.

Decoder-safety: (a2p)/(bp) streaming-causal (proof in RESULTS); (bq)/(bqp)
offline-decodable + literal bitstream proof (same bar as b10).
"""
import sys
import time

import numpy as np
from PIL import Image

import probe_b4_a as A
from probe_b4_j import eval_final as j_eval_final
from probe_b4_h import rans_stream_bits
import probe_b11_vqstack as V

IMAGES = V.IMAGES
HEADER = V.HEADER
JXL_E3 = V.JXL_E3
MOE6 = V.MOE6
huff_data_only = V.huff_data_only
scalar_bits_huff = V.scalar_bits_huff
scalar_bits_rans = V.scalar_bits_rans
joint_bits_huff = V.joint_bits_huff
joint_bits_rans = V.joint_bits_rans
canon_codes = V.canon_codes
crown_internals = V.crown_internals
config_a_bits = V.config_a_bits


# ---------------- pair-LEFT-root analysis ----------------
def pair_analysis(C):
    D, kch, gid = C["D"], C["ch"], C["gid"]
    H, W = kch.shape
    P, s = D["P"], D["s"]
    Pl = gid[:, 0::2]
    out = []
    for gi in range(len(C["G"])):
        base = (Pl == gi)
        npr = int(base.sum())
        mask = np.zeros((H, W), dtype=bool)
        mask[:, 0::2] = base
        mask[:, 1::2] = base
        best = None
        for n in MOE6:
            rf = (s * (kch - P[n]).astype(np.int32)).astype(np.int32)
            pv = rf[mask]
            d_ = huff_data_only(pv) if pv.size else 0
            if best is None or d_ < best:
                best = d_
                bn = n
                brf = rf
        if npr > 0:
            l = brf[:, 0::2][base]
            r = brf[:, 1::2][base]
            pr = np.stack([l, r], axis=1)
        else:
            pr = np.zeros((0, 2), dtype=np.int32)
        out.append(dict(gi=gi, npr=npr, mask=mask, bn=bn, rf=brf, pr=pr,
                        base=base))
    return out


def pair_vq_bits(pr, T, backend="huff"):
    m = np.abs(pr)
    incap = m.max(axis=1) <= T
    K = T + 1
    ESC = K * K
    sym = np.where(incap, m[:, 0] * K + m[:, 1], ESC)
    jb = joint_bits_huff(sym) if backend == "huff" else joint_bits_rans(sym)
    sgn = int(((pr[incap] != 0).sum()))
    p2 = pr[~incap].reshape(-1)
    pb, _ = scalar_bits_huff(p2) if backend == "huff" else scalar_bits_rans(p2)
    return jb + sgn + pb, sym, incap


# ---------------- homogeneous blocks under PIXEL predictors ----------------
def homog_pair_blocks(C, g):
    """1x2 pairs with both pixels in pixel-group g (pixel predictor rf)."""
    D = C["D"]
    P, s = D["P"], D["s"]
    sel = g["sel"]
    rf = (s * (C["ch"] - P[g["bn_pix"]]).astype(np.int32)).astype(np.int32)
    base = sel[:, 0::2] & sel[:, 1::2]
    l = rf[:, 0::2][base]
    r = rf[:, 1::2][base]
    return base, np.stack([l, r], axis=1) if int(base.sum()) else np.zeros((0, 2), np.int32)


def homog_quad_blocks(C, g):
    sel = g["sel"]
    D = C["D"]
    P, s = D["P"], D["s"]
    rf = (s * (C["ch"] - P[g["bn_pix"]]).astype(np.int32)).astype(np.int32)
    base = sel[0::2, 0::2] & sel[0::2, 1::2] & sel[1::2, 0::2] & sel[1::2, 1::2]
    if int(base.sum()) == 0:
        return base, np.zeros((0, 4), dtype=np.int32)
    q = np.stack([rf[0::2, 0::2][base], rf[0::2, 1::2][base],
                  rf[1::2, 0::2][base], rf[1::2, 1::2][base]], axis=1)
    return base, q


def block_vq_bits(bl, T, n, backend="huff"):
    m = np.abs(bl)
    incap = m.max(axis=1) <= T
    K = T + 1
    ESC = K ** n
    if n == 2:
        sym = np.where(incap, m[:, 0] * K + m[:, 1], ESC)
    else:
        sym = np.where(incap, ((m[:, 0] * K + m[:, 1]) * K + m[:, 2]) * K + m[:, 3], ESC)
    jb = joint_bits_huff(sym) if backend == "huff" else joint_bits_rans(sym)
    sgn = int(((bl[incap] != 0).sum()))
    p2 = bl[~incap].reshape(-1)
    pb, _ = scalar_bits_huff(p2) if backend == "huff" else scalar_bits_rans(p2)
    return jb + sgn + pb, sym, incap


def eval_image(path):
    nm = path.split("/")[-1]
    img, H, W, chs, per_ch = crown_internals(path)
    dn = H * W * 3
    ta = config_a_bits(H, W, per_ch)
    ja, _ = j_eval_final(path, "huff")
    assert abs(ta / dn - ja) < 1e-9, "(a) replica mismatch"
    tmed = HEADER
    for ch in chs:
        _, res, _ = A.med_pred_ctx9(ch)
        tb0, _, _ = A.stream_bits(res.reshape(-1))
        tmed += tb0

    pa_all = [pair_analysis(C) for C in per_ch]
    ta2p = HEADER
    tbp = HEADER
    tcp = HEADER
    # (bq)/(bqp): full recount per pixel-group with homog-block option
    tbq = HEADER
    tbqp = HEADER
    tcq = HEADER
    tcqp = HEADER
    d = {"tot_groups": 0, "pairwin": 0, "pair_choice": {"S": 0, "T4": 0, "T8": 0},
         "qh_win": 0, "qp_win": 0, "qh_cov": 0, "qp_cov": 0, "qh_pix": 0,
         "qp_pix": 0, "tot_pix": 0, "pair_homog": 0, "pair_tot": 0,
         "esc_p4": [], "esc_p8": [], "esc_qh": [], "esc_ph": [],
         "vqsave_p": 0, "vqsave_q": 0, "vqsave_qp": 0,
         "Ajoint_p": [], "Atail_p": []}
    for C, pa in zip(per_ch, pa_all):
        side = C["bmap"] + len(C["G"]) * 3
        ta2p += side
        tbp += side + len(C["G"]) * 2
        tcp += side + len(C["G"]) * 2
        tbq += side
        tbqp += side
        tcq += side
        tcqp += side
        gid = C["gid"]
        D = C["D"]
        P, s = D["P"], D["s"]
        for g, pg in zip(C["ginfo"], pa):
            d["tot_groups"] += 1
            pool = pg["rf"][pg["mask"]]
            # (a2p)
            sb, _ = scalar_bits_huff(pool)
            ta2p += sb
            # (bp) gate huff
            if pg["npr"] == 0:
                tbp += 40
                tcp += 48
            else:
                v4, sym4, inc4 = pair_vq_bits(pg["pr"], 4)
                v8, sym8, inc8 = pair_vq_bits(pg["pr"], 8)
                d["esc_p4"].append(float((~inc4).mean()))
                d["esc_p8"].append(float((~inc8).mean()))
                opts = {"S": sb, "T4": v4, "T8": v8}
                win = min(opts, key=lambda k: opts[k])
                d["pair_choice"][win] += 1
                tbp += opts[win]
                if win != "S":
                    d["pairwin"] += 1
                    d["vqsave_p"] += sb - opts[win]
                # (c) pair rANS re-gate
                rs, _ = scalar_bits_rans(pool)
                r4, _, _ = pair_vq_bits(pg["pr"], 4, backend="rans")
                r8, _, _ = pair_vq_bits(pg["pr"], 8, backend="rans")
                tcp += min(rs, r4, r8)
            # pair-left homogeneity diagnostic
            d["pair_homog"] += int((gid[:, 0::2][pg["base"]] == gid[:, 1::2][pg["base"]]).sum())
            d["pair_tot"] += pg["npr"]
            # ---- (bq)/(bqp) pure stacking on pixel groups ----
            rfpix = (s * (C["ch"] - P[g["bn_pix"]]).astype(np.int32)).astype(np.int32)
            gv = rfpix[g["sel"]]
            Sfull, _ = scalar_bits_huff(gv)
            Sfull_r, _ = scalar_bits_rans(gv)
            # homog quads
            hb, hq = homog_quad_blocks(C, g)
            nhq = int(hb.sum())
            d["qh_pix"] += 4 * nhq
            if nhq == 0:
                tbq += Sfull
                tcq += Sfull_r
            else:
                d["qh_cov"] += 1
                hmask = np.zeros_like(g["sel"])
                hmask[0::2, 0::2] = hb
                hmask[0::2, 1::2] = hb
                hmask[1::2, 0::2] = hb
                hmask[1::2, 1::2] = hb
                rest = gv[~hmask[g["sel"]]] if False else rfpix[g["sel"] & ~hmask]
                vhq, _, inch = block_vq_bits(hq, 4, 4)
                d["esc_qh"].append(float((~inch).mean()))
                restb, _ = scalar_bits_huff(rest)
                opt_q = restb + vhq + 1
                if opt_q < Sfull:
                    d["qh_win"] += 1
                    d["vqsave_q"] += Sfull - opt_q
                    tbq += opt_q
                else:
                    tbq += Sfull
                # rANS
                restbr, _ = scalar_bits_rans(rest)
                vhqr, _, _ = block_vq_bits(hq, 4, 4, backend="rans")
                tcq += min(Sfull_r, restbr + vhqr + 1)
            # homog pairs
            pb2, pq = homog_pair_blocks(C, g)
            nhp = int(pb2.sum())
            d["qp_pix"] += 2 * nhp
            if nhp == 0:
                tbqp += Sfull
                tcqp += Sfull_r
            else:
                pmask = np.zeros_like(g["sel"])
                pmask[:, 0::2] = pb2
                pmask[:, 1::2] = pb2
                restp = rfpix[g["sel"] & ~pmask]
                vhp, _, inph = block_vq_bits(pq, 8, 2)
                d["esc_ph"].append(float((~inph).mean()))
                restpb, _ = scalar_bits_huff(restp)
                opt_qp = restpb + vhp + 1
                if opt_qp < Sfull:
                    d["qp_win"] += 1
                    d["vqsave_qp"] += Sfull - opt_qp
                    tbqp += opt_qp
                else:
                    tbqp += Sfull
                restpbr, _ = scalar_bits_rans(restp)
                vhpr, _, _ = block_vq_bits(pq, 8, 2, backend="rans")
                tcqp += min(Sfull_r, restpbr + vhpr + 1)
            d["tot_pix"] += int(g["sel"].sum())
    return dict(nm=nm, dn=dn, a=ta / dn, a2p=ta2p / dn, bp=tbp / dn,
                cp=tcp / dn, bq=tbq / dn, bqp=tbqp / dn, cq=tcq / dn,
                cqp=tcqp / dn, med=tmed / dn, per_ch=per_ch, pa=pa_all, diag=d)


def main():
    t0 = time.time()
    recs = []
    for path in IMAGES:
        i0 = time.time()
        rec = eval_image(path)
        recs.append(rec)
        g = rec["diag"]
        print(f"{rec['nm']}: med={rec['med']:.4f} (a)={rec['a']:.4f} "
              f"(a2p)={rec['a2p']:.4f} (bp)={rec['bp']:.4f} (cp)={rec['cp']:.4f} "
              f"(bq)={rec['bq']:.4f} (bqp)={rec['bqp']:.4f} "
              f"(cq)={rec['cq']:.4f} (cqp)={rec['cqp']:.4f} "
              f"jxl={JXL_E3[rec['nm']]:.2f} [{time.time()-i0:.0f}s]", flush=True)
    print("-" * 120)
    keys = ("med", "a", "a2p", "bp", "cp", "bq", "bqp", "cq", "cqp")
    avg = {k: float(np.mean([r[k] for r in recs])) for k in keys}
    for k in keys:
        print(f"AVG {k} = {avg[k]:.4f}")
    A_ = avg["a"]
    MED = avg["med"]
    for k in ("a2p", "bp", "cp", "bq", "bqp", "cq", "cqp"):
        print(f"d({k}-a) = {(avg[k]-A_)/A_*100:+.3f}%   vsMED = {(MED-avg[k])/MED*100:.2f}%")
    soloVQ = 1.84
    soloCR = (MED - A_) / MED * 100
    for k in ("bp", "cp", "bq", "bqp", "cq", "cqp"):
        comb = (MED - avg[k]) / MED * 100
        print(f"overlap({k}) = {1-comb/(soloVQ+soloCR):+.4f}  (combined vs MED {comb:.2f}%)")
    print(f"soloCROWN vs MED = {soloCR:.2f}% (soloVQ = 1.84%)")
    jxlavg = float(np.mean(list(JXL_E3.values())))
    for k in ("bp", "cp", "bq", "bqp", "cq", "cqp"):
        print(f"d({k}-jxl) = {(avg[k]-jxlavg)/jxlavg*100:+.2f}%")
    agg = {"pairwin": 0, "tot_groups": 0, "qh_win": 0, "qp_win": 0,
           "qh_cov": 0, "qh_pix": 0, "qp_pix": 0, "tot_pix": 0,
           "pair_homog": 0, "pair_tot": 0, "S": 0, "T4": 0, "T8": 0,
           "vqsave_p": 0, "vqsave_q": 0, "vqsave_qp": 0}
    dn_tot = 0
    esc = {"esc_p4": [], "esc_p8": [], "esc_qh": [], "esc_ph": []}
    for r in recs:
        g = r["diag"]
        dn_tot += r["dn"]
        for k in agg:
            if k in g:
                agg[k] += g[k]
        for k in ("S", "T4", "T8"):
            agg[k] += g["pair_choice"][k]
        for k in esc:
            esc[k] += g[k]
    print(f"AGG pair-root VQ wins {agg['pairwin']}/{agg['tot_groups']}; "
          f"choices S/T4/T8={agg['S']}/{agg['T4']}/{agg['T8']}")
    print(f"AGG homog-quad groups w/ coverage: wins {agg['qh_win']}; "
          f"homog-quad pixel frac={agg['qh_pix']/agg['tot_pix']*100:.2f}%")
    print(f"AGG homog-pair: wins {agg['qp_win']}; "
          f"homog-pair pixel frac={agg['qp_pix']/agg['tot_pix']*100:.2f}%")
    print(f"AGG pair-left homog frac={agg['pair_homog']/agg['pair_tot']*100:.1f}%")
    for k, v in esc.items():
        print(f"  {k} mean escape={np.mean(v)*100:.1f}%")
    print(f"AGG VQ net savings bpp: pair={agg['vqsave_p']/dn_tot:.4f} "
          f"hquad={agg['vqsave_q']/dn_tot:.4f} hpair={agg['vqsave_qp']/dn_tot:.4f}")
    print(f"elapsed {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
