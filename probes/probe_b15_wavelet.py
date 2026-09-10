"""probe_b15: transform-bitplane arc — LeGall 5/3 integer wavelet + EZW/SPIHT-style
context-coded bitplanes + MED-DPCM LL, YCoCg-R front-end. numpy+PIL only.

Scan orders (decoder-causal by construction):
  S1 channels Y,Co,Cg in fixed order; S2 subbands coarse->fine
     (LL_L, then level L..1 triplets [rowHP-colLP, rowLP-colHP, HH]);
  S3 within a detail subband: bitplanes MSB->LSB; inside one plane, raster
     row-major significance pass, then sign bits (raster order of newly-
     significant), then refinement bits (raster order of already-significant).
Contexts use ONLY already-decoded state: coarser-scale parent magnitudes
(fully decoded: coarser levels precede finer) + higher-plane significance
(full planes precede lower planes, any spatial position) + (maxbit,p) which
are transmitted/known. No within-plane feedback -> vectorized + causal.
LL: raster MED-DPCM (left/top/topleft causal), Huffman-coded.
Stream framing: every Huffman stream has decoder-derivable symbol count
(fixed geometry / significance-state counts), so no length side; tables
counted 16+A*24 (A = distinct symbols present; A==1 -> 0 data bits).
Sign/refinement bits are raw (decoder knows counts from significance map).
UNIT: bpp = total_bits/(H*W*3).
"""
import sys, os, math, heapq, time, json
import numpy as np
from PIL import Image

IMGDIR = "/tmp/opencode/autocompress/experiments/real_photos"
IMGS = ["kodim01.png", "kodim02.png", "kodim05.png", "kodim07.png",
        "kodim13.png", "kodim19.png", "kodim23.png"]
JXL_E3 = {"kodim01.png": 3.3593, "kodim02.png": 3.0611, "kodim05.png": 3.5120,
          "kodim07.png": 2.7310, "kodim13.png": 3.9141, "kodim19.png": 3.2154,
          "kodim23.png": 2.8110}
CEIL = 3.2515

# ---------------- YCoCg-R (lossless, floor shifts) ----------------
def rgb_to_ycocg(r, g, b):
    co = r - b
    t = b + (co >> 1)
    cg = g - t
    y = t + (cg >> 1)
    return y, co, cg

def ycocg_to_rgb(y, co, cg):
    t = y - (cg >> 1)
    g = cg + t
    b = t - (co >> 1)
    r = b + co
    return r, g, b

# ---------------- LeGall 5/3 lifting, 1D (whole-sample symmetric ext) ----------------
def fwd53_1d(x):
    n = x.shape[0]
    ne = (n + 1) // 2
    no = n // 2
    ev = x[0::2].copy()
    od = x[1::2].copy()
    if no == 0:
        return ev, np.empty(0, np.int64)
    a = ev[:no]
    if no < ne:
        b = ev[1:no + 1]
    else:  # n even: last odd mirrors x[n]=x[n-2]
        b = np.concatenate([ev[1:], ev[ne - 1:]])
    d = od - ((a + b) // 2)
    # update: s[i] needs d[i-1],d[i] with symmetric mirrors; vectors length ne
    dl = np.empty(ne, np.int64)
    dl[0] = d[0]
    if ne > 1:
        dl[1:] = d[np.minimum(np.arange(1, ne) - 1, no - 1)]
    dr = d[np.minimum(np.arange(ne), no - 1)]
    s = ev + ((dl + dr + 2) // 4)
    return s, d

def inv53_1d(s, d):
    ne = s.shape[0]
    no = d.shape[0]
    n = ne + no
    if no == 0:
        return s.copy()
    dl = np.empty(ne, np.int64)
    dl[0] = d[0]
    if ne > 1:
        dl[1:] = d[np.minimum(np.arange(1, ne) - 1, no - 1)]
    dr = d[np.minimum(np.arange(ne), no - 1)]
    ev = s - ((dl + dr + 2) // 4)
    a = ev[:no]
    if no < ne:
        b = ev[1:no + 1]
    else:
        b = np.concatenate([ev[1:], ev[ne - 1:]])
    od = d + ((a + b) // 2)
    y = np.empty(n, np.int64)
    y[0::2] = ev
    y[1::2] = od
    return y

def fwd53_2d(img, levels):
    h, w = img.shape
    a = img.astype(np.int64)
    det = []  # (level, key, array), level 1 = finest
    for l in range(1, levels + 1):
        h2, w2 = a.shape
        wl, wh = (w2 + 1) // 2, w2 // 2
        lo = np.empty((h2, wl), np.int64)
        hi = np.empty((h2, wh), np.int64)
        for r in range(h2):
            s, d = fwd53_1d(a[r])
            lo[r] = s
            hi[r] = d
        hl, hh = (h2 + 1) // 2, h2 // 2
        LL = np.empty((hl, wl), np.int64)
        LH = np.empty((hh, wl), np.int64)
        for c in range(wl):
            s, d = fwd53_1d(lo[:, c])
            LL[:, c] = s
            LH[:, c] = d
        HL = np.empty((hl, wh), np.int64)
        HH = np.empty((hh, wh), np.int64)
        for c in range(wh):
            s, d = fwd53_1d(hi[:, c])
            HL[:, c] = s
            HH[:, c] = d
        det.append((l, "HL", HL))  # row-HP col-LP
        det.append((l, "LH", LH))  # row-LP col-HP
        det.append((l, "HH", HH))
        a = LL
    return a, det

def inv53_2d(LL, det, levels):
    a = LL
    for l in range(levels, 0, -1):
        tri = {k: v for (ll, k, v) in det if ll == l}
        HL, LH, HH = tri["HL"], tri["LH"], tri["HH"]
        hl, wl = a.shape
        h2 = hl + LH.shape[0]
        lo = np.empty((h2, wl), np.int64)
        for c in range(wl):
            lo[:, c] = inv53_1d(a[:, c], LH[:, c])
        hi = np.empty((h2, HL.shape[1]), np.int64)
        for c in range(HL.shape[1]):
            hi[:, c] = inv53_1d(HL[:, c], HH[:, c])
        w2 = wl + hi.shape[1]
        a = np.empty((h2, w2), np.int64)
        for r in range(h2):
            a[r] = inv53_1d(lo[r], hi[r])
    return a

# ---------------- exact Huffman accounting ----------------
def huff_bits_from_counts(counts):
    counts = [c for c in counts if c > 0]
    if not counts:
        return 0, 0
    A = len(counts)
    side = 16 + A * 24
    if A == 1:
        return 0, side  # single symbol, count derivable -> 0 data bits
    h = list(counts)
    heapq.heapify(h)
    tot = 0
    while len(h) > 1:
        x = heapq.heappop(h)
        y = heapq.heappop(h)
        tot += x + y
        heapq.heappush(h, x + y)
    return tot, side

def huff_stream_bits(syms):
    """syms: 1D int array. Returns exact (data+table) bits."""
    if syms.size == 0:
        return 0
    _, cnt = np.unique(syms, return_counts=True)
    d, s = huff_bits_from_counts(cnt.tolist())
    return d + s

# ---------------- LL MED-DPCM ----------------
def med(a, b, c):
    return sorted([a, b, c])[1]

def ll_dpcm_bits(LL):
    h, w = LL.shape
    res = np.empty((h, w), np.int64)
    for r in range(h):
        for c in range(w):
            if r == 0 and c == 0:
                p = 0
            elif r == 0:
                p = LL[r, c - 1]
            elif c == 0:
                p = LL[r - 1, c]
            else:
                p = med(int(LL[r, c - 1]), int(LL[r - 1, c]), int(LL[r - 1, c - 1]))
            res[r, c] = int(LL[r, c]) - int(p)
    # NOTE: row loop with per-pixel python ops; LL is small (<=192x128). fine.
    return huff_stream_bits(res.reshape(-1))

# ---------------- context bitplane coder (detail subband) ----------------
def detail_bitplane_bits(S, P, nctx=16):
    """S: detail subband int64. P: coarser parent same key or None.
    Returns (bits, dbg). Decoder-derivable counts everywhere."""
    H, W = S.shape
    A = np.abs(S)
    mx = int(A.max()) if A.size else 0
    maxbit = mx.bit_length()
    bits = 4  # maxbit side (0..16 range)
    if maxbit == 0:
        return bits, {"maxbit": 0}
    # parent upsampled abs (nearest); None -> zeros
    if P is not None:
        PA = np.abs(P)
        prow = np.minimum(np.arange(H) // 2, PA.shape[0] - 1)
        pcol = np.minimum(np.arange(W) // 2, PA.shape[1] - 1)
        PUp = PA[prow][:, pcol]
    else:
        PUp = np.zeros((H, W), np.int64)
    sig = np.zeros((H, W), bool)
    streams = [[] for _ in range(nctx)]
    sign_n = 0
    ref_n = 0
    pdense_thr = maxbit - 2  # planes p < thr are the dense class
    for p in range(maxbit - 1, -1, -1):
        thr = np.int64(1) << np.int64(p)
        notsig = ~sig
        if notsig.any():
            psig = (PUp >= thr).astype(np.int64)
            # 8-neighbourhood sum over higher-plane sig state (all decoded)
            pad = np.pad(sig.astype(np.int64), 1)
            nbr = (pad[:-2, :-2] + pad[:-2, 1:-1] + pad[:-2, 2:] +
                   pad[1:-1, :-2] + pad[1:-1, 2:] +
                   pad[2:, :-2] + pad[2:, 1:-1] + pad[2:, 2:])
            bucket = np.minimum(nbr, 3).astype(np.int64)
            pcls = np.int64(1) if p < pdense_thr else np.int64(0)
            if nctx == 16:
                ctx = (psig << 3) | (bucket << 1) | pcls
            else:
                ctx = (psig << 2) | bucket
            sb = ((A >> np.int64(p)) & np.int64(1)).astype(np.int64)
            ns_idx = np.flatnonzero(notsig)
            cvals = ctx.reshape(-1)[ns_idx]
            bvals = sb.reshape(-1)[ns_idx]
            order = np.argsort(cvals, kind="stable")
            cvals = cvals[order]
            bvals = bvals[order]
            # split runs per ctx (raster order preserved within ctx)
            chg = np.flatnonzero(np.diff(cvals) != 0)
            bounds = np.concatenate([[0], chg + 1, [len(cvals)]])
            for i in range(len(bounds) - 1):
                k = int(cvals[bounds[i]])
                streams[k].append(bvals[bounds[i]:bounds[i + 1]])
            new = notsig & (sb == 1)
            sign_n += int(new.sum())
            sig |= new
    # ---- exact sign/refine counts: deterministic replay ----
    # (refinement bits at plane p = #coeffs significant BEFORE plane p)
    sig2 = np.zeros((H, W), bool)
    sign_n = 0
    ref_n = 0
    for p in range(maxbit - 1, -1, -1):
        sb = ((A >> np.int64(p)) & np.int64(1)).astype(bool)
        ref_n += int(sig2.sum())
        new = (~sig2) & sb
        sign_n += int(new.sum())
        sig2 |= new
    assert sign_n == int((A > 0).sum()), "every nonzero coeff gets exactly one sign bit"
    total = bits + sign_n + ref_n  # raw sign + raw refinement
    ntab = 0
    for k in range(nctx):
        if streams[k]:
            arr = np.concatenate(streams[k])
            nb = (len(arr) + 3) // 4
            padn = nb * 4 - len(arr)
            if padn:
                arr = np.concatenate([arr, np.zeros(padn, np.int64)])
            nib = (arr.reshape(nb, 4) * np.int64([1, 4, 16, 64])).sum(axis=1)
            total += huff_stream_bits(nib)
            ntab += 1
    return total, {"maxbit": maxbit, "sign": sign_n, "ref": ref_n, "ntab": ntab}

def order0_subband_bits(S):
    return huff_stream_bits(S.reshape(-1))

# ---------------- config C: raster-causal conditioned Huffman (signed) ----------------
def detail_cond_bits(S, P, Yact=None, nabuck=(0, 2, 8), parbuck=(1, 5)):
    """Position-driven per-group Huffman on SIGNED coefficients.
    Context (all decoder-known in raster order): act=|l|+|t|+|tl|+|tr|
    bucketed x parent-magnitude bucket [x Y-activity bit].
    Side: 12b presence mask + 16+A*24 per nonempty group."""
    H, W = S.shape
    Sp = np.pad(np.abs(S), 1)
    act = Sp[1:-1, :-2] + Sp[:-2, 1:-1] + Sp[:-2, :-2] + Sp[:-2, 2:]
    ab = np.digitize(act, np.int64(nabuck))  # 0..3
    if P is not None:
        PA = np.abs(P)
        prow = np.minimum(np.arange(H) // 2, PA.shape[0] - 1)
        pcol = np.minimum(np.arange(W) // 2, PA.shape[1] - 1)
        PUp = PA[prow][:, pcol]
    else:
        PUp = np.zeros((H, W), np.int64)
    pb = np.digitize(PUp, np.int64(parbuck))  # 0..2
    if Yact is not None:
        yb = (Yact > 0).astype(np.int64)
        grp = (ab * 3 + pb) * 2 + yb
    else:
        grp = ab * 3 + pb
    G = int(grp.max()) + 1
    total = G  # 1 presence bit per group
    flat_s = S.reshape(-1)
    flat_g = grp.reshape(-1)
    for k in range(G):
        m = flat_g == k
        if m.any():
            total += huff_stream_bits(flat_s[m])
    return total, {"G": G}

# ---------------- per-image probe ----------------
def probe_image(path, levels, mode, nctx=16):
    im = np.array(Image.open(path).convert("RGB")).astype(np.int64)
    H, W, _ = im.shape
    r, g, b = im[:, :, 0], im[:, :, 1], im[:, :, 2]
    y, co, cg = rgb_to_ycocg(r, g, b)
    # exact invertibility assert (front end)
    rr, gg, bb = ycocg_to_rgb(y, co, cg)
    assert (rr == r).all() and (gg == g).all() and (bb == b).all(), "YCoCg-R not exact"
    total = 0
    det_info = []
    # transform all channels first (CX cross-channel ctx needs Y decoded first;
    # Y is coded first, so Y subbands are available when coding Co/Cg)
    pack = []
    for ch in (y, co, cg):
        LL, det = fwd53_2d(ch, levels)
        # exact inverse assert (transform)
        ch_rt = inv53_2d(LL, det, levels)
        assert ch_rt.shape == ch.shape and (ch_rt == ch).all(), "5/3 inverse mismatch"
        pack.append((LL, det))
    pmapY = {(l, k): S for (l, k, S) in pack[0][1]}
    for ci, (LL, det) in enumerate(pack):
        if mode == "A0":
            total += order0_subband_bits(LL)
            for (l, k, S) in det:
                total += order0_subband_bits(S)
        elif mode == "A1":
            total += ll_dpcm_bits(LL)
            for (l, k, S) in det:
                total += order0_subband_bits(S)
        elif mode == "B":
            total += ll_dpcm_bits(LL)
            pmap = {(l, k): S for (l, k, S) in det}
            for (l, k, S) in det:
                P = pmap.get((l + 1, k), None)
                tb, dbg = detail_bitplane_bits(S, P, nctx)
                total += tb
                det_info.append((l, k, dbg))
        elif mode in ("C", "CX", "C24"):
            total += ll_dpcm_bits(LL)
            pmap = {(l, k): S for (l, k, S) in det}
            for (l, k, S) in det:
                P = pmap.get((l + 1, k), None)
                Yact = None
                if mode == "CX" and ci > 0:
                    Yact = np.abs(pmapY[(l, k)])  # same geometry by construction
                    assert Yact.shape == S.shape
                if mode == "C24":
                    tb, dbg = detail_cond_bits(S, P, Yact,
                                               nabuck=(0, 1, 2, 4, 8, 16),
                                               parbuck=(0, 1, 2, 5, 12))
                else:
                    tb, dbg = detail_cond_bits(S, P, Yact)
                total += tb
                det_info.append((l, k, dbg))
        else:
            raise ValueError(mode)
    bpp = total / (H * W * 3)
    return bpp, det_info

def selftest():
    rng = np.random.default_rng(0)
    for shape in [(1,), (2,), (3,), (7,), (8,), (64,), (65,), (768,), (512,)]:
        x = rng.integers(-300, 300, size=shape).astype(np.int64)
        assert (inv53_1d(*fwd53_1d(x)) == x).all(), shape
    for shape in [(5, 7), (8, 8), (33, 65), (64, 48), (192, 128)]:
        x = rng.integers(-300, 300, size=shape).astype(np.int64)
        for L in (1, 2, 3):
            LL, det = fwd53_2d(x, L)
            assert (inv53_2d(LL, det, L) == x).all(), (shape, L)
    print("SELFTEST PASS (lifting exact on even+odd lengths, 2D L=1..3)")

if __name__ == "__main__":
    selftest()
    only = sys.argv[1:] or IMGS
    cfgs = [("A0", 2, 16), ("A1", 2, 16), ("B8", 2, 8), ("B16", 2, 16),
            ("C", 2, 16), ("CX", 2, 16), ("C24", 2, 16), ("C-L1", 1, 16),
            ("B16L3", 3, 16), ("CX-L3", 3, 16)]
    res = {}
    for name in only:
        p = os.path.join(IMGDIR, name)
        row = {}
        for cfg, L, nc in cfgs:
            t0 = time.time()
            mode = cfg.split("-")[0]
            if mode.startswith("B"):
                mode = "B"
            bpp, dbg = probe_image(p, L, mode, nc)
            row[cfg] = bpp
            print(f"{name} {cfg}: {bpp:.4f} bpp ({time.time()-t0:.1f}s)", flush=True)
        res[name] = row
    print(json.dumps(res, indent=1))
    avgs = {c: sum(res[n][c] for n in res) / len(res) for c, _, _ in cfgs}
    print("AVGS:", json.dumps(avgs, indent=1))
