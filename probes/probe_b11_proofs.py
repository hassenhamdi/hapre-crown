"""probe_b11_proofs: literal bitstream round-trip for the winning config (bqp).

(bqp) = pixel-CROWN + per-group gated homogeneous-pair magnitude-VQ (T=8).
Framing (offline, recon-derived, static tables): per pixel-group, 1b flag iff
the group owns >=1 homogeneous 1x2 pair (positions derivable from recon);
flag=1 -> joint magnitude stream (pair-raster order) + fixed sign bits +
scalar tail (ESC pairs) + scalar rest (non-homog group pixels, raster order);
flag=0 -> single scalar stream. All Huffman tables canonical, counted
16+A*24. This script packs every stream with canonical codes, decodes,
reassembles residuals, unflips, adds predictors, asserts channel equality
(kodim07, all channels), plus sym->mag mapping checks on all 7 images.
(cqp) rANS streams self-verify inside rans_stream_bits (real C codec
encode+decode asserted on EVERY stream during measurement).
"""
import sys

import numpy as np
from PIL import Image

import probe_b4_a as A
import probe_b11_vqstack as V
import probe_b11_pairvq as P

IMAGES = A.IMAGES


def pack_decode_vals(vals):
    """Canonical pack + full decode of one symbol stream. Returns nbits."""
    vals = np.asarray(vals).reshape(-1).tolist()
    assert len(vals) > 0
    uv, cn = np.unique(np.array(vals), return_counts=True)
    counts = {int(v): int(n) for v, n in zip(uv.tolist(), cn.tolist())}
    codes = V.canon_codes(counts)
    bitstr = "".join(format(codes[v][0], "0%db" % codes[v][1]) for v in vals)
    dec = {(l, cd): sy for sy, (cd, l) in codes.items()}
    ml = max(l for _, l in codes.values())
    pos = 0
    out = []
    for _ in range(len(vals)):
        cd = 0
        for ln in range(1, ml + 1):
            cd = (cd << 1) | (1 if bitstr[pos] == "1" else 0)
            pos += 1
            if (ln, cd) in dec:
                out.append(dec[(ln, cd)])
                break
        else:
            raise AssertionError("decode failed")
    assert out == [int(v) for v in vals], "stream round-trip mismatch"
    assert pos == len(bitstr)
    return len(bitstr)


def literal_bqp_image(path):
    img, H, W, chs, per_ch = V.crown_internals(path)
    dn = H * W * 3
    counted_data = 0
    literal_bits = 0
    nstreams = 0
    for C in per_ch:
        D = C["D"]
        P_, s = D["P"], D["s"]
        kch = C["ch"]
        rf_full = np.zeros_like(kch)
        filled = np.zeros_like(kch, dtype=bool)
        for g in C["ginfo"]:
            rfpix = (s * (kch - P_[g["bn_pix"]]).astype(np.int32)).astype(np.int32)
            gv = rfpix[g["sel"]]
            Sfull, _ = V.scalar_bits_huff(gv)
            pb2, pq = P.homog_pair_blocks(C, g)
            nhp = int(pb2.sum())
            use_vq = False
            if nhp > 0:
                pmask = np.zeros_like(g["sel"])
                pmask[:, 0::2] = pb2
                pmask[:, 1::2] = pb2
                restp = rfpix[g["sel"] & ~pmask]
                vhp, sym, inph = P.block_vq_bits(pq, 8, 2)
                restpb, _ = V.scalar_bits_huff(restp)
                use_vq = (restpb + vhp + 1 < Sfull)
            if not use_vq:
                counted_data += Sfull
                literal_bits += pack_decode_vals(gv)
                nstreams += 1
                rf_full[g["sel"]] = gv
                filled[g["sel"]] = True
            else:
                K, ESC = 9, 81
                assert all(int(v) == ESC if not ok else True
                           for v, ok in zip(sym.tolist(), inph.tolist()))
                for sv, row, ok in zip(sym.tolist(), pq.tolist(), inph.tolist()):
                    if ok:
                        assert [int(sv) // K, int(sv) % K] == [abs(v) for v in row]
                m = np.abs(pq)
                incap = m.max(axis=1) <= 8
                jsym = np.where(incap, m[:, 0] * K + m[:, 1], ESC)
                # joint
                uv, cn = np.unique(jsym, return_counts=True)
                counted_data += V.joint_bits_huff(jsym)
                literal_bits += pack_decode_vals(jsym)
                nstreams += 1
                # signs
                signs = "".join("1" if v > 0 else "0"
                                for row in pq[incap].tolist() for v in row if v != 0)
                literal_bits += len(signs)
                counted_data += len(signs)
                # tail + rest
                pool2 = pq[~incap].reshape(-1)
                if pool2.size:
                    _, cn2 = np.unique(pool2, return_counts=True)
                    tb2, _, _ = A.stream_bits(pool2)
                    counted_data += tb2
                    literal_bits += pack_decode_vals(pool2)
                    nstreams += 1
                else:
                    counted_data += 40
                tptr = [0]
                tout = pool2.tolist()
                if restp.size:
                    tbr, _, _ = A.stream_bits(restp)
                    counted_data += tbr
                    literal_bits += pack_decode_vals(restp)
                    nstreams += 1
                    rptr = [0]
                    rout = restp.tolist()
                else:
                    counted_data += 40
                    rptr = [0]
                    rout = []
                counted_data += 1  # flag
                # reassemble: homog pairs in pair-raster order
                sptr = [0]
                rec = np.zeros_like(pq)
                for qi, (sv, row, ok) in enumerate(
                        zip(jsym.tolist(), pq.tolist(), incap.tolist())):
                    if ok:
                        mm = [int(sv) // K, int(sv) % K]
                        rr = []
                        for a, v in zip(mm, row):
                            assert a == abs(v)
                            if a == 0:
                                rr.append(0)
                            else:
                                rr.append(a if signs[sptr[0]] == "1" else -a)
                                sptr[0] += 1
                        rec[qi] = rr
                    else:
                        rec[qi] = [tout[tptr[0] + k] for k in range(2)]
                        tptr[0] += 2
                assert np.array_equal(rec, pq), "pair reassembly mismatch"
                assert sptr[0] == len(signs)
                idx = np.argwhere(pb2)
                for (r, c), row in zip(idx.tolist(), rec.tolist()):
                    rf_full[r, 2 * c] = row[0]
                    rf_full[r, 2 * c + 1] = row[1]
                    filled[r, 2 * c] = True
                    filled[r, 2 * c + 1] = True
                # rest pixels in raster order of (sel & ~pmask)
                ridx = np.argwhere(g["sel"] & ~pmask)
                for (r, c), v in zip(ridx.tolist(), rout):
                    rf_full[r, c] = v
                    filled[r, c] = True
        assert filled.all(), "unfilled pixels"
        for g in C["ginfo"]:
            pred = D["P"][g["bn_pix"]]
            rec_ch = pred + (s * rf_full).astype(np.int32)
            assert np.array_equal(rec_ch[g["sel"]], kch[g["sel"]]), \
                "channel reconstruction mismatch"
    return literal_bits, counted_data, nstreams


def main():
    # mapping checks: every VQ-winning homog-pair group, all 7 images
    nmap = 0
    for path in IMAGES:
        _, _, _, _, per_ch = V.crown_internals(path)
        for C in per_ch:
            D = C["D"]
            P_, s = D["P"], D["s"]
            for g in C["ginfo"]:
                rfpix = (s * (C["ch"] - P_[g["bn_pix"]]).astype(np.int32)).astype(np.int32)
                gv = rfpix[g["sel"]]
                Sfull, _ = V.scalar_bits_huff(gv)
                pb2, pq = P.homog_pair_blocks(C, g)
                if int(pb2.sum()) == 0:
                    continue
                pmask = np.zeros_like(g["sel"])
                pmask[:, 0::2] = pb2
                pmask[:, 1::2] = pb2
                restp = rfpix[g["sel"] & ~pmask]
                vhp, sym, inph = P.block_vq_bits(pq, 8, 2)
                restpb, _ = V.scalar_bits_huff(restp)
                if restpb + vhp + 1 < Sfull:
                    nmap += 1
                    K, ESC = 9, 81
                    for sv, row, ok in zip(sym.tolist(), pq.tolist(), inph.tolist()):
                        if ok:
                            assert [int(sv) // K, int(sv) % K] == [abs(v) for v in row]
                        else:
                            assert int(sv) == ESC
    print(f"[mapping round-trip (bqp) VQ-winning homog-pair groups, all 7: "
          f"PASS ({nmap} groups)]")
    lit_path = [p for p in IMAGES if p.endswith("kodim07.png")][0]
    lb, cb, ns = literal_bqp_image(lit_path)
    print(f"[literal bitstream round-trip (bqp) kodim07 all channels: PASS, "
          f"{ns} streams, payload {lb}b vs ledger data+tables {cb}b "
          f"(tables+flags {cb-lb}b = {(cb-lb)/ns:.0f}b/stream)]")
    assert lb <= cb, "payload exceeds ledger!"


if __name__ == "__main__":
    main()
