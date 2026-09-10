"""HAPRE-C E16 C-port driver: WIDE-K + E16 experts + per-group backend (H/G/R) + C decode.
Encoder decisions via proven probe modules; packing/streaming decode in C."""
import ctypes, math, numpy as np, time, os, sys
from PIL import Image
sys.path.insert(0, "/tmp/opencode/autocompress/csrc")
sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
os.chdir("/tmp/opencode/autocompress/experiments")
from driver import lib
import probe_b4_a as A
from probe_b4_j import auto_groups_exact
from probe_b4_b import nbhd
from probe_b4_h import rans_stream_bits
from probe_b5_d import golomb_best, E16
import sys as _s; _s.path.insert(0,'/tmp/opencode/autocompress/csrc')
from dp_refine import refine
c_u8 = ctypes.c_uint8; c_i16 = ctypes.c_int16; c_u16 = ctypes.c_uint16
lib.e16_residuals.argtypes = [ctypes.POINTER(c_i16), ctypes.c_int, ctypes.c_int, ctypes.POINTER(c_i16)]
lib.crown_keys.argtypes = [ctypes.POINTER(c_i16), ctypes.c_int, ctypes.c_int,ctypes.POINTER(c_i16), ctypes.POINTER(ctypes.c_int8)]
lib.huff_unpack.argtypes = [ctypes.POINTER(c_u8), ctypes.c_size_t, ctypes.c_int,ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(c_u8), ctypes.POINTER(c_i16)]
lib.huff_unpack.restype = ctypes.c_int
lib.golomb_unpack.argtypes = [ctypes.POINTER(c_u8), ctypes.c_size_t,ctypes.POINTER(c_u8), ctypes.POINTER(c_u8), ctypes.c_int, ctypes.POINTER(c_i16)]
lib.golomb_unpack.restype = ctypes.c_int
lib.crown2_assemble.argtypes = [ctypes.POINTER(c_i16), ctypes.POINTER(ctypes.c_int),ctypes.c_int, ctypes.POINTER(c_u16), ctypes.POINTER(c_u8),ctypes.POINTER(c_i16),ctypes.c_int,ctypes.c_int,ctypes.c_int]
lib.crown2_assemble.restype = ctypes.c_int
lib.rans_norm.argtypes = [ctypes.POINTER(ctypes.c_int64), ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_uint16),ctypes.POINTER(ctypes.c_int)]
lib.rans_norm.restype = None
lib.rans_encode.argtypes = [ctypes.POINTER(c_i16), ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_uint16),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(c_u8)]
lib.rans_encode.restype = ctypes.c_size_t
lib.rans_slots.argtypes = [ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_int),ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
lib.rans_decode.argtypes = [ctypes.POINTER(c_u8), ctypes.c_size_t,ctypes.c_int, ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_uint16),ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(c_i16)]
lib.rans_decode.restype = ctypes.c_int
lib.huff_pack.argtypes = [ctypes.POINTER(c_i16), ctypes.c_int, ctypes.POINTER(ctypes.c_uint32),
                          ctypes.POINTER(c_u8), ctypes.POINTER(c_u8), ctypes.c_size_t]
lib.huff_pack.restype = ctypes.c_size_t
lib.golomb_pack.argtypes = [ctypes.POINTER(c_i16), ctypes.POINTER(c_u8), ctypes.POINTER(c_u8),
                            ctypes.c_int, ctypes.POINTER(c_u8), ctypes.c_size_t]
lib.golomb_pack.restype = ctypes.c_size_t

KSET_WIDE = (2, 3, 4, 6, 9, 12, 18, 27, 36, 48, 64)
E2C = {"MED": 0, "TOP": 1, "LEFT": 6, "PAETH": 2, "GRAD": 3, "GAP80": 4, "DG": 5,
       "AVG_AB": 9, "PLANE": 10, "GAP32": 7, "AC": 11, "C": 12, "BC": 13, "D": 14,
       "GAP16": 8, "AVG3": 15, "GAP": 4}
B2I = {"H": 0, "G": 1, "R": 2}


def canon_of_vals(g):
    vals, cn = np.unique(np.asarray(g).reshape(-1), return_counts=True)
    counts = {int(v): int(c) for v, c in zip(vals.tolist(), cn.tolist())}
    if len(counts) == 1:
        s = next(iter(counts))
        return {s + 1024: (0, 1)}, counts
    import heapq
    H = [(c, s) for s, c in counts.items()]; heapq.heapify(H); par = {}; nxt = 1 << 28; H2 = H[:]
    while len(H2) > 1:
        a, sa = heapq.heappop(H2); b, sb = heapq.heappop(H2); nn = nxt; nxt += 1
        par[sa] = (nn, 0); par[sb] = (nn, 1); heapq.heappush(H2, (a + b, nn))
    lens = {}
    for s in counts:
        d = 0; n = s
        while n in par:
            n = par[n][0]; d += 1
        lens[s] = d
    order = sorted(counts, key=lambda s: (lens[s], s))
    codes = {}; code = 0; prev = 0
    for s in order:
        code <<= (lens[s] - prev); codes[s + 1024] = (code, lens[s]); code += 1; prev = lens[s]
    return codes, counts


def rans_pack(g):
    g = np.ascontiguousarray(g.reshape(-1).astype(np.int16)); N = g.size
    uv = np.unique(g)
    syms = np.array([int(v) + 1024 for v in uv.tolist()], dtype=np.int32); Ad = len(syms)
    hist = np.zeros(2049, dtype=np.int64)
    for v in uv.tolist():
        hist[int(v) + 1024] = int((g == v).sum())
    h = hist.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
    sp = syms.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    fq = np.zeros((Ad,), np.uint16); cu = np.zeros((Ad,), np.int32)
    lib.rans_norm(h, Ad, sp, fq.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)), cu.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
    lut = np.full((2049,), -1, np.int32)
    for i, s in enumerate(syms.tolist()):
        lut[s] = i
    cap = N * 3 + 16; buf = (c_u8 * cap)()
    n = lib.rans_encode(g.ctypes.data_as(ctypes.POINTER(c_i16)), N,
                        lut.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                        fq.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
                        cu.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), buf)
    assert n > 0
    return bytes(buf[:n]), syms, fq, cu


def encode_one(rgb, use_rans=True):
    H, W, _ = rgb.shape; N = H * W
    Y, Co, Cg = A.ycocg_fwd(rgb); chs = [Y, Co, Cg]
    yc = np.zeros((3 * N,), dtype=np.int16)
    for k in range(3):
        yc[k * N:(k + 1) * N] = chs[k].reshape(-1).astype(np.int16)
    t0 = time.perf_counter()
    key = np.zeros((3 * N,), dtype=np.int16); sgn = np.zeros((3 * N,), dtype=np.int8)
    lib.crown_keys(yc.ctypes.data_as(ctypes.POINTER(c_i16)), H, W,
                   key.ctypes.data_as(ctypes.POINTER(c_i16)), sgn.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)))
    res16 = np.zeros((3 * 16 * N,), dtype=np.int16)
    lib.e16_residuals(yc.ctypes.data_as(ctypes.POINTER(c_i16)), H, W,
                      res16.ctypes.data_as(ctypes.POINTER(c_i16)))
    out = bytearray()
    for k in range(3):
        kk = key[k * N:(k + 1) * N].reshape(H, W)
        ss = sgn[k * N:(k + 1) * N].reshape(H, W)
        # sign-flipped MED residuals for grouping (match probe)
        rmed = res16[(k * 16 + 0) * N:(k * 16 + 1) * N].reshape(H, W).astype(np.int32)
        rfM = (ss.astype(np.int32) * rmed).astype(np.int32)
        bK, G, _, _ = auto_groups_exact(kk, rfM, KSET_WIDE)
        # per-group best (expert16 x backend)
        uk = np.unique(kk)
        out += int(bK).to_bytes(1, "little") + int(len(uk)).to_bytes(2, "little")
        gmap = np.zeros((729,), dtype=np.uint8)
        for gi, gkeys in enumerate(G):
            for u in gkeys:
                gmap[int(u)] = gi
        for u in sorted(uk.tolist()):
            out += int(u).to_bytes(2, "little") + int(gmap[int(u)]).to_bytes(1, "little")
        out += int(len(G)).to_bytes(1, "little")
        for gi, gkeys in enumerate(G):
            sel = np.isin(kk, np.array(gkeys))
            best = None
            for nme in E16:
                r = res16[(k * 16 + E2C[nme]) * N:(k * 16 + E2C[nme] + 1) * N].reshape(H, W).astype(np.int32)
                g = (ss.astype(np.int32) * r)[sel]
                if g.size == 0:
                    cand = (0, nme, "H", 0); 
                else:
                    codes, _ = canon_of_vals(g)
                    ht = sum(len(g[g == (s - 1024)]) * l for s, (c, l) in codes.items()) + 16 + len(codes) * 24
                    gd, gk = golomb_best(g)
                    cand_h, cand_g = (ht, nme, "H", 0), (gd + 4, nme, "G", gk)
                    cand = cand_h if cand_h[0] <= cand_g[0] else cand_g
                    if use_rans:
                        pay, syms, fq, cu = rans_pack(g)
                        rt = 16 + len(syms) * 32 + len(pay) * 8
                        if rt < cand[0]:
                            cand = (rt, nme, "R", 0)
                if best is None or cand[0] < best[0]:
                    best = cand
            out += int(E2C[best[1]]).to_bytes(1, "little") + int(B2I[best[2]]).to_bytes(1, "little")
            if best[2] == "G":
                out += int(best[3]).to_bytes(1, "little")
            # payload
            r = res16[(k * 16 + E2C[best[1]]) * N:(k * 16 + E2C[best[1]] + 1) * N].reshape(H, W).astype(np.int32)
            g = (ss.astype(np.int32) * r)[sel]
            if best[2] == "H":
                codes, _ = canon_of_vals(g)
                out += int(len(codes)).to_bytes(2, "little")
                cd = np.zeros((2049,), np.uint32); ln = np.zeros((2049,), np.uint8)
                for s, (cc, ll) in codes.items():
                    out += ((s - 1024) & 0xFFFF).to_bytes(2, "little"); out.append(ll)
                    cd[s] = cc; ln[s] = ll
                # ordered pack of THIS group's symbols only (decoder: per-group arrays)
                arr = np.ascontiguousarray(g.reshape(-1).astype(np.int16))
                cap = arr.size * 32 // 8 + 8; buf = (c_u8 * cap)()
                lib.huff_pack(arr.ctypes.data_as(ctypes.POINTER(c_i16)), arr.size,
                              cd.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                              ln.ctypes.data_as(ctypes.POINTER(c_u8)), buf, cap)
                import ctypes as C
                # huff_pack returns size_t; restype default int is enough here (<2GB), recompute n via helper below
                n = _huff_len(arr, cd, ln)
                out += int(g.size).to_bytes(4, "little") + int(n).to_bytes(4, "little") + bytes(buf[:n])
            elif best[2] == "G":
                arr = np.ascontiguousarray(g.reshape(-1).astype(np.int16))
                cap = arr.size * 40 // 8 + 8; buf = (c_u8 * cap)()
                kk0 = np.zeros((arr.size,), np.uint8)
                karr = np.full((1,), best[3], np.uint8)
                lib.golomb_pack(arr.ctypes.data_as(ctypes.POINTER(c_i16)), kk0.ctypes.data_as(ctypes.POINTER(c_u8)),
                                karr.ctypes.data_as(ctypes.POINTER(c_u8)), arr.size, buf, cap)
                n = _golomb_len(arr, best[3])
                out += int(g.size).to_bytes(4, "little") + int(n).to_bytes(4, "little") + bytes(buf[:n])
            else:
                pay, syms, fq, cu = rans_pack(g)
                out += int(g.size).to_bytes(4, "little") + int(len(syms)).to_bytes(2, "little")
                for i, sm in enumerate(syms.tolist()):
                    out += ((sm - 1024) & 0xFFFF).to_bytes(2, "little") + int(fq[i]).to_bytes(2, "little")
                out += int(len(pay)).to_bytes(4, "little") + pay
    return bytes(out), (time.perf_counter() - t0,)


def _huff_len(arr, cd, ln):
    tot = 0
    for v in arr.tolist():
        tot += int(ln[int(v) + 1024])
    return (tot + 7) // 8


def _golomb_len(arr, k):
    tot = 0
    for v in arr.tolist():
        u = 2 * v if v >= 0 else -2 * v - 1
        tot += (u >> k) + 1 + k
    return (tot + 7) // 8


def decode_one(blob, H, W):
    import ctypes as C
    N = H * W; p = 0
    img = np.zeros((3 * N,), dtype=np.int16)
    lib.huff_unpack.restype = ctypes.c_int
    for k in range(3):
        na = int.from_bytes(blob[p:p + 2], "little"); p += 2
        cmap = np.zeros((729,), np.uint16)
        acc = 0; nb = 0; qpos = p
        raw = blob

        def _bits(n):
            nonlocal acc, nb, qpos
            while nb < n:
                acc = (acc << 8) | raw[qpos]; qpos += 1; nb += 8
            nb -= n
            v = (acc >> nb) & ((1 << n) - 1)
            acc &= ((1 << nb) - 1) if nb else 0
            return v

        for _ in range(na):
            e = _bits(16)
            cmap[(e >> 6) & 0x3FF] = e & 0x3F
        p = qpos
        G = int.from_bytes(blob[p:p + 1], "little"); p += 1
        predid = np.zeros((36 + 64,), np.uint8)
        back = np.zeros((36 + 64,), np.uint8)
        ks = np.zeros((36 + 64,), np.uint8)
        groups = []
        for gi in range(G):
            pb = int.from_bytes(blob[p:p + 1], "little"); p += 1
            predid[gi] = (pb >> 2) & 0xF; bb = pb & 0x3
            back[gi] = bb
            if bb == 1:
                ks[gi] = int.from_bytes(blob[p:p + 1], "little"); p += 1
            cnt = None; pay = None; aux = None
            if bb == 0:
                A = int.from_bytes(blob[p:p + 2], "little"); p += 2
                cd = np.zeros((2049,), np.uint32); ln = np.zeros((2049,), np.uint8)
                syms = []
                for _ in range(A):
                    sv = int.from_bytes(blob[p:p + 2], "little"); l = blob[p + 2]; p += 3
                    s = (sv + 1024) & 0xFFFF; ln[s] = l; syms.append(s)
                syms.sort(key=lambda s: (ln[s], s))
                code = 0; prev = 0
                for s in syms:
                    L = int(ln[s]); code <<= (L - prev); cd[s] = code; code += 1; prev = L
                cnt = int.from_bytes(blob[p:p + 4], "little"); p += 4
                n = int.from_bytes(blob[p:p + 4], "little"); p += 4
                pay = bytes(blob[p:p + n]); p += n
                dec = np.zeros((cnt,), dtype=np.int16)
                bb2 = (c_u8 * n)(*pay)
                rc = lib.huff_unpack(bb2, n, cnt, cd.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                                     ln.ctypes.data_as(ctypes.POINTER(c_u8)), dec.ctypes.data_as(ctypes.POINTER(c_i16)))
                assert rc == 0, (k, gi, rc)
            elif bb == 1:
                cnt = int.from_bytes(blob[p:p + 4], "little"); p += 4
                n = int.from_bytes(blob[p:p + 4], "little"); p += 4
                pay = bytes(blob[p:p + n]); p += n
                dec = np.zeros((cnt,), dtype=np.int16)
                kk0 = np.zeros((cnt,), np.uint8)
                karr = np.full((1,), int(ks[gi]), np.uint8)
                bb2 = (c_u8 * n)(*pay)
                rc = lib.golomb_unpack(bb2, n, kk0.ctypes.data_as(ctypes.POINTER(c_u8)),
                                       karr.ctypes.data_as(ctypes.POINTER(c_u8)), cnt,
                                       dec.ctypes.data_as(ctypes.POINTER(c_i16)))
                assert rc == 0, (k, gi, rc)
            else:
                cnt = int.from_bytes(blob[p:p + 4], "little"); p += 4
                A = int.from_bytes(blob[p:p + 2], "little"); p += 2
                syms = np.zeros((A,), np.int32); fq = np.zeros((A,), np.uint16); cu = np.zeros((A,), np.int32)
                for i in range(A):
                    sv = int.from_bytes(blob[p:p + 2], "little"); f = int.from_bytes(blob[p + 2:p + 4], "little"); p += 4
                    syms[i] = (sv + 1024) & 0xFFFF; fq[i] = f
                c = 0
                for i in range(A):
                    cu[i] = c; c += int(fq[i])
                n = int.from_bytes(blob[p:p + 4], "little"); p += 4
                pay = bytes(blob[p:p + n]); p += n
                slots = np.zeros((16384,), np.int32)
                lib.rans_slots(fq.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)), cu.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                               A, slots.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
                dec = np.zeros((cnt,), dtype=np.int16)
                bb2 = (c_u8 * n)(*pay)
                rc = lib.rans_decode(bb2, n, cnt, syms.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                     fq.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)), cu.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                     A, slots.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                     dec.ctypes.data_as(ctypes.POINTER(c_i16)))
                assert rc == 0, (k, gi, rc)
            groups.append(dec)
        # assemble channel via map
        goff = [0]
        for dec in groups:
            goff.append(goff[-1] + len(dec))
        syms_all = np.concatenate(groups).astype(np.int16) if groups else np.zeros((0,), np.int16)
        goff = np.array(goff, dtype=np.int32)
        # map group ids: transmitted groups indexed 0..G-1 already match cmap values
        rc = lib.crown_assemble(syms_all.ctypes.data_as(ctypes.POINTER(c_i16)),
                                goff.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), G,
                                cmap.ctypes.data_as(ctypes.POINTER(c_u16)),
                                predid.ctypes.data_as(ctypes.POINTER(c_u8)),
                                img.ctypes.data_as(ctypes.POINTER(c_i16)), H, W, k)
        assert rc == 0, (k, rc)
    out = bytearray(N * 3)
    lib.ycocg_inv(img.ctypes.data_as(ctypes.POINTER(c_i16)), H, W, (ctypes.c_char * len(out)).from_buffer(out))
    return bytes(out)


if __name__ == "__main__":
    d = "/tmp/opencode/autocompress/experiments/real_photos"
    import sys
    for fn in sys.argv[1:] or ["kodim23.png"]:
        rgb = np.array(Image.open(os.path.join(d, fn)).convert("RGB"))
        H, W, _ = rgb.shape
        blob, (te,) = encode_one(rgb, use_rans=True)
        dec = decode_one(blob, H, W)
        print(fn, "bytes=", len(blob), "bpp=%.4f" % (len(blob) * 8 / (H * W * 3)),
              "enc=%.0fms" % (te * 1000), "ROUNDTRIP=", "PASS" if dec == rgb.tobytes() else "FAIL", flush=True)
