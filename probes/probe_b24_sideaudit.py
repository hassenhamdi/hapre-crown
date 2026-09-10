"""probe_b24: side-info audit of CROWN6 winner streams (05/C27, 13/C12).

Single linear walker mirroring driver.decode_image field order/widths
(byte-exact; asserts p==len at end of blob). Attributes BITS per channel to:
chdr, mlp_side, wuse, wweights, map, experts, golombkd, H-tables, H-pay,
G-pay, R-headers, R-pay.
Question: does rANS-header mass (or any side) cover the JXL-e3 gaps
(05 +10,115B, 13 +5,770B)? If yes -> design header-sharing build.
Usage: python3 probes/probe_b24_sideaudit.py
"""
import sys
import math
import numpy as np

WIDE = (2, 3, 4, 6, 9, 12, 18, 27, 36, 48, 64)
GS = 32


def audit(blob):
    H = int.from_bytes(blob[2:4], "little")
    W = int.from_bytes(blob[4:6], "little")
    assert blob[:2] == b"C6" and blob[6] == 1, blob[:8]
    p = 8
    out = []
    for _ in range(3):
        a = {}
        b0, ng, mlp_flag = blob[p], blob[p + 1], blob[p + 2]
        p += 3
        a["chdr"] = 24
        fam = b0 & 1
        grid = (b0 >> 1) & 7
        if mlp_flag:
            p += 64
            nwin = 0
            scid = list(blob[p:p + 9])
            p += 9
            for k in range(9):
                if scid[k] != 255:
                    p += 226
                    nwin += 1
            a["mlp_side"] = (64 + 9 + 226 * nwin) * 8
        else:
            a["mlp_side"] = 0
        nbw = math.ceil(W / GS)
        nbh = math.ceil(H / GS)
        ng32 = nbh * nbw
        nbytes = (ng32 + 7) // 8
        raw = blob[p:p + nbytes]
        p += nbytes
        a["wuse"] = ng32
        # decode wuse bits MSB-first
        acc = nb = qp = 0
        wuse = []
        for g in range(ng32):
            while nb < 1:
                acc = (acc << 8) | raw[qp]
                qp += 1
                nb += 8
            nb -= 1
            wuse.append((acc >> nb) & 1)
            acc &= ((1 << nb) - 1) if nb else 0
        nused = sum(wuse)
        nbytes = (nused * 20 + 7) // 8
        p += nbytes
        a["wweights"] = nused * 20
        cmap = None
        mask = None
        if fam == 0:
            mask = np.unpackbits(np.frombuffer(blob[p:p + 92], dtype=np.uint8))[:729]
            p += 92
            kidx = (b0 >> 4) & 15
            K = WIDE[kidx]
            gbits = math.ceil(math.log2(K))
            na = int(mask.sum())
            nbytes = (na * gbits + 7) // 8
            # rebuild cmap like driver
            rawm = blob[p:p + nbytes]
            p += nbytes
            a["map"] = 92 * 8 + na * gbits
            acc = nb = qp = 0
            uk = np.where(mask)[0]
            cmap = np.zeros(729, dtype=np.uint8)
            for u in uk:
                while nb < gbits:
                    acc = (acc << 8) | rawm[qp]
                    qp += 1
                    nb += 8
                nb -= gbits
                cmap[u] = (acc >> nb) & ((1 << gbits) - 1)
                acc &= ((1 << nb) - 1) if nb else 0
        else:
            occ = blob[p]
            p += 1
            a["map"] = 8
        nbytes = (ng * 7 + 7) // 8
        raw3 = blob[p:p + nbytes]
        p += nbytes
        a["experts"] = ng * 7
        acc = nb = qp = 0
        predid = []
        backs = []
        for gi in range(ng):
            while nb < 7:
                acc = (acc << 8) | raw3[qp]
                qp += 1
                nb += 8
            nb -= 7
            v = (acc >> nb) & 0x7F
            acc &= ((1 << nb) - 1) if nb else 0
            predid.append((v >> 2) & 31)
            backs.append(v & 3)
        nG = sum(1 for x in backs if x == 1)
        nbytes = (nG * 4 + 7) // 8
        p += nbytes
        nbytes = (nG * 3 + 7) // 8
        p += nbytes
        a["golombkd"] = nG * 7
        if fam == 0:
            nonempty = np.zeros(ng, dtype=bool)
            for u in np.where(mask)[0]:
                nonempty[cmap[u]] = True
        else:
            nonempty = None  # GRID uses occ bitmask (a["_occ"]) below
        a["_backs"] = backs
        a["_nonempty"] = nonempty
        a["_occ"] = occ if fam == 1 else None
        # H tables
        htab = 0
        hnon = 0
        for gi in range(ng):
            if fam == 0:
                ne = nonempty[gi]
            else:
                ne = (a["_occ"] >> gi) & 1
            if backs[gi] == 0 and ne:
                A_ = int.from_bytes(blob[p:p + 2], "little")
                p += 2 + A_ * 3
                htab += 16 + A_ * 24
                hnon += 1
        a["H-tables"] = htab
        a["_hnon"] = hnon
        hn = int.from_bytes(blob[p:p + 4], "little")
        p += 4
        p += hn
        a["H-pay"] = hn * 8
        gn_ = int.from_bytes(blob[p:p + 4], "little")
        p += 4
        p += gn_
        a["G-pay"] = gn_ * 8
        rhead = 0
        rpay = 0
        for gi in range(ng):
            if backs[gi] == 2:
                cnt = int.from_bytes(blob[p:p + 4], "little")
                p += 4
                A_ = int.from_bytes(blob[p:p + 2], "little")
                p += 2
                p += A_ * 4
                n_ = int.from_bytes(blob[p:p + 4], "little")
                p += 4
                p += n_
                rhead += 32 + 16 + A_ * 32 + 32
                rpay += n_ * 8
        a["R-headers"] = rhead
        a["R-pay"] = rpay
        a["_ng"] = ng
        a["_nG"] = nG
        out.append(a)
    assert p == len(blob), (p, len(blob))
    return out, (H, W)


def show(path, jxl_bpp):
    blob = open(path, "rb").read()
    chs, (H, W) = audit(blob)
    tot_bits = len(blob) * 8
    print(f"{path}: {H}x{W} bytes={len(blob)} bpp={tot_bits/(H*W*3):.4f} "
          f"vs JXL {jxl_bpp} gap={(tot_bits/(H*W*3)-jxl_bpp)*(H*W*3)/8:+.0f}B")
    cats = ["mlp_side", "wuse", "wweights", "map", "experts", "golombkd",
            "H-tables", "H-pay", "G-pay", "R-headers", "R-pay"]
    print(f"  {'ch':>3} " + " ".join(f"{c:>9}" for c in cats) + f" {'ng':>3}")
    for i, a in enumerate(chs):
        print(f"  ch{i} " + " ".join(f"{a.get(c,0)/8:9.0f}" for c in cats)
              + f" {a['_ng']:3d}")
    tot = {c: sum(a.get(c, 0) for a in chs) for c in cats}
    print("  TOT(B): " + " ".join(f"{c}={tot[c]/8:.0f}" for c in cats))


if __name__ == "__main__":
    import json
    import os
    nums = {}
    for path, jxl in (("/tmp/dump05b.bin", 3.5120), ("/tmp/dump13.bin", 3.9141)):
        blob = open(path, "rb").read()
        chs, (H, W) = audit(blob)
        fn = "kodim05.png" if "05" in path else "kodim13.png"
        nums[fn] = {"bytes": len(blob),
                    "bpp": len(blob) * 8 / (H * W * 3),
                    "jxl_bpp": jxl,
                    "channels": [{k: v for k, v in a.items() if not k.startswith("_")}
                                 for a in chs]}
        show(path, jxl)
    with open("/tmp/opencode/autocompress/probes/probe_b24_nums.json", "w") as f:
        json.dump(nums, f, indent=1)
    print("wrote probes/probe_b24_nums.json")
