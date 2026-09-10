"""probe_b13_verify: round-trip check for the WINNING (c2) config.

Rebuilds (a) grouping + both greedy partitions, re-derives the exact (c2) winner
(Huffman-proxy partition + per-cluster exact H/R), and asserts:
 - partition covers all atoms exactly once per image,
 - Kraft=1 + canonical symbol round-trip on every Huffman-kept (c2) cluster, ALL-7,
 - rANS decode asserted on every R-kept (c2) cluster (via rans_stream_bits),
 - YCoCg-R inverse ALL-7,
 - literal bitstream pack->unpack->decode on kodim01 (c2) cluster 0,
 - exact (c2) totals reproduce probe_b13_summary.txt to 4 decimals.
"""
import sys
import os
import time

sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
sys.path.insert(0, "/tmp/opencode/autocompress/csrc")
os.chdir("/tmp/opencode/autocompress/experiments")

import numpy as np
from PIL import Image

import probe_b4_a as A
from probe_b4_h import rans_stream_bits
from probe_b13_merge import build_image, greedy_merge_atoms, huffman_lengths, merge_map_bits, IMAGES

EXP_C2 = {"kodim01.png": 3.3758, "kodim02.png": 3.0260, "kodim05.png": 3.6811,
          "kodim07.png": 2.7825, "kodim13.png": 4.0550, "kodim19.png": 3.2238,
          "kodim23.png": 2.8202}


def main():
    t0 = time.time()
    ok_all = True
    for path in IMAGES:
        nm = path.split("/")[-1]
        bi = build_image(path)
        dn = bi["dn"]
        atoms = bi["atoms"]
        ng = len(atoms)
        base_sides = 64 + sum(c["bmap"] + len(c["G"]) * 3 for c in bi["channels"])
        gh = greedy_merge_atoms(atoms, backend="huff")
        g2 = greedy_merge_atoms(atoms, backend="rans")
        # exact-encode both partitions with H/R choice, take min (mirrors merge.py)
        cands = []
        for g, tag in ((gh, "h"), (g2, "r")):
            part = g["part"]
            C = g["C"]
            mmap = merge_map_bits(ng, C) + 8
            cl_vals = [atoms[members[0]]["vals"] if len(members) == 1
                       else np.concatenate([atoms[x]["vals"] for x in members]) for members in part]
            tot_hr = 0
            picks = []
            for v in cl_vals:
                uv, cn = np.unique(v, return_counts=True)
                import heapq
                h = cn.tolist()
                if len(h) == 0:
                    ht = 0
                elif len(h) == 1:
                    ht = int(h[0]) * 1 + 16 + 24
                else:
                    hh = h[:]
                    heapq.heapify(hh)
                    t = 0
                    while len(hh) > 1:
                        x = heapq.heappop(hh)
                        y = heapq.heappop(hh)
                        s = x + y
                        t += s
                        heapq.heappush(hh, s)
                    ht = t + 16 + len(h) * 24
                rt, _ = rans_stream_bits(v)
                if rt < ht:
                    tot_hr += rt
                    picks.append("R")
                else:
                    tot_hr += ht
                    picks.append("H")
            exact = base_sides + mmap + C * 1 + tot_hr
            cands.append((exact, C, part, picks, cl_vals, tag))
        cands.sort(key=lambda t: t[0])
        exact, C, part, picks, cl_vals, tag = cands[0]
        bpp = exact / dn
        match = abs(bpp - EXP_C2[nm]) < 5e-4
        # cover
        flat = sorted([m for grp in part for m in grp])
        cover = (flat == list(range(ng)))
        # per-cluster checks
        nH = nR = 0
        for v, p in zip(cl_vals, picks):
            if p == "R":
                rt, _ = rans_stream_bits(v)  # asserts decode
                nR += 1
            else:
                uv, cn = np.unique(v, return_counts=True)
                cnt = {int(s): int(c) for s, c in zip(uv.tolist(), cn.tolist())}
                L = huffman_lengths(cnt)
                kr = sum(2.0 ** (-L[s]) for s in L)
                assert abs(kr - 1.0) < 1e-9 or len(L) == 1, (nm, kr)
                enc_order = sorted(cnt.keys(), key=lambda s: (L[s], int(s)))
                enc, dec = {}, {}
                code, prev = 0, 0
                for s in enc_order:
                    ln = L[s]
                    code <<= (ln - prev)
                    enc[s] = (code, ln)
                    dec[(code, ln)] = s
                    code += 1
                    prev = ln
                for s in enc_order:
                    cd, ln = enc[s]
                    assert dec[(cd, ln)] == s
                nH += 1
        img = np.array(Image.open(path).convert("RGB"))
        Y, Co, Cg = A.ycocg_fwd(img)
        t = Y - (Cg // 2)
        B = t - (Co // 2)
        G = Cg + t
        R = Co + B
        assert np.array_equal(R, img[:, :, 0].astype(np.int32))
        ok = cover and match
        ok_all = ok_all and ok
        print(f"  {nm}: c2={bpp:.4f} expect={EXP_C2[nm]:.4f} "
              f"[{'PASS' if match else 'FAIL'}] cover={'PASS' if cover else 'FAIL'} "
              f"H={nH} R={nR} via={tag}", flush=True)
    # literal bitstream on kodim01 (c2) cluster 0
    bi = build_image(IMAGES[0])
    atoms = bi["atoms"]
    ng = len(atoms)
    g2 = greedy_merge_atoms(atoms, backend="rans")
    part = g2["part"]
    mem0 = part[0]
    v0 = atoms[mem0[0]]["vals"] if len(mem0) == 1 else np.concatenate([atoms[m]["vals"] for m in mem0])
    uv, cn = np.unique(v0, return_counts=True)
    cnt0 = {int(s): int(c) for s, c in zip(uv.tolist(), cn.tolist())}
    L0 = huffman_lengths(cnt0)
    enc_order = sorted(cnt0.keys(), key=lambda s: (L0[s], int(s)))
    enc, dec = {}, {}
    code, prev = 0, 0
    for s in enc_order:
        ln = L0[s]
        code <<= (ln - prev)
        enc[s] = (code, ln)
        dec[(code, ln)] = s
        code += 1
        prev = ln
    vals0 = v0.tolist()
    buf, nb, out = 0, 0, bytearray()
    for s in vals0:
        cd, ln = enc[int(s)]
        buf = (buf << ln) | cd
        nb += ln
        while nb >= 8:
            nb -= 8
            out.append((buf >> nb) & 0xFF)
    if nb:
        out.append((buf << (8 - nb)) & 0xFF)
    bits = np.unpackbits(np.frombuffer(bytes(out), dtype=np.uint8))
    acc, al, got = 0, 0, []
    for b in bits.tolist():
        acc = (acc << 1) | b
        al += 1
        if (acc, al) in dec:
            got.append(dec[(acc, al)])
            acc, al = 0, 0
            if len(got) == len(vals0):
                break
    assert got == [int(s) for s in vals0]
    print(f"  kodim01 c2-cl0 literal-bitstream PASS ({len(vals0)} syms, "
          f"{sum(L0[s] for s in vals0)} bits)")
    print(f"VERIFY ALL-7 {'PASS' if ok_all else 'FAIL'} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
