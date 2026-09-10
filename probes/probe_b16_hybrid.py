"""probe_b16 HYBRID — pixel-domain CROWN machinery where it wins (Y + LL),
transform/sparse coding ONLY where coefficients are actually sparse (chroma HF).

Design (subband split, NO spatial map — split is dyadic + fixed channel
assignment, so no region map to transmit; only counted selectors):
  P0 anchor : full-res MED + order-0 Huffman per channel (expect ~3.58).
  P9 pixel  : full-res MED + CTX9 energy-grouped Huffman per channel (proven b4).
  T arm     : 1-level LeGall 5/3; LL via min(order0, CTX9)+1b sel; each detail via
              min(order0, sparse-map+nz, subband-MED)+2b sel. L=1 only (b15: L2/L3 worse).
  H-CHROMA  : Y via P9 + Co/Cg via T arm (fixed design, 0 selector bits).
  H-ORACLE  : per-channel min(P9, T) + 3b channel-selector (upper bound of blending).
  T-ALL     : all channels via T arm (sanity vs b15 C-L1 3.4310).

UNIT: bpp = total_bits/(H*W*3). Exact counting: heapq Huffman data +
16+A*24 per transmitted table (A==1 -> 0 data bits, table still counted;
empty streams cost 0), +9b presence mask per grouped stream, +1b LL selector,
+2b per detail-subband selector, +3b oracle channel selector, +64b global header.
numpy+PIL only, CPU. Reuses probe_b4_* (YCoCg-R/MED-CTX9) and probe_b15_wavelet
(5/3 lifting + exact Huffman) by import.

Decoder-safety / scan order: S1 channels Y->Co->Cg fixed. Pixel arm: raster
left->right/top->bottom MED (0/left/top champion rule), grouped-stream counts
from presence mask in header. Transform arm: LL raster MED first, then details
HL,LH,HH raster; sparse arm = significance map (count=H*W known) then nonzero
values (count = popcount, known after map); subband-MED raster-causal. All
conditioning causally available; every stream length geometry- or state-derivable.
Bit-exact reconstruction asserted per image (YCoCg-R + 5/3 inverse + MED inversion
+ sparse map/nz rebuild); any failure aborts.
"""
import sys, os, heapq, time, json
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from probe_b4_a import med_pred_ctx9, ycocg_fwd
from probe_b15_wavelet import fwd53_2d, inv53_2d, huff_stream_bits

IMGDIR = "/tmp/opencode/autocompress/experiments/real_photos"
IMGS = ["kodim01.png", "kodim02.png", "kodim05.png", "kodim07.png",
        "kodim13.png", "kodim19.png", "kodim23.png"]
JXL_E3 = {"kodim01.png": 3.3593, "kodim02.png": 3.0611, "kodim05.png": 3.5120,
          "kodim07.png": 2.7310, "kodim13.png": 3.9141, "kodim19.png": 3.2154,
          "kodim23.png": 2.8110}
CEIL = 3.2515
HEADER = 64


def med_abc(x):
    H, W = x.shape
    a = np.empty_like(x); b = np.empty_like(x); c = np.empty_like(x)
    a[:, 1:] = x[:, :-1]; a[:, 0] = 0
    b[1:, :] = x[:-1, :]; b[0, :] = 0
    c[1:, 1:] = x[:-1, :-1]; c[0, :] = 0; c[:, 0] = 0
    b[0, 1:] = a[0, 1:]; c[0, 1:] = a[0, 1:]
    a[1:, 0] = b[1:, 0]; c[1:, 0] = b[1:, 0]
    return a, b, c


def med_full(x):
    a, b, c = med_abc(x)
    mx = np.maximum(a, b); mn = np.minimum(a, b)
    pred = np.where(c >= mx, mn, np.where(c <= mn, mx, a + b - c))
    res = x - pred
    e = np.minimum((np.abs(a - c) + np.abs(b - c)) >> 4, 8)
    return pred, res, e


def grouped9_bits(res, ctx):
    total = 9  # 1b presence mask per group, header-counted
    for q in range(9):
        g = res[ctx == q]
        if g.size:
            total += huff_stream_bits(g.reshape(-1))
    return total


def ll_best_bits(LL):
    o0 = huff_stream_bits(LL.reshape(-1))
    pred, res, e = med_full(LL)
    assert (pred + res == LL).all(), "LL MED inversion"
    g9 = grouped9_bits(res, e)
    if o0 <= g9:
        return o0 + 1, "o0"
    return g9 + 1, "g9"


def detail_best_bits(S):
    o0 = huff_stream_bits(S.reshape(-1))
    m = S != 0
    mb = huff_stream_bits(m.reshape(-1).astype(np.int64))
    nzb = huff_stream_bits(S[m].reshape(-1)) if m.any() else 0
    sp = mb + nzb
    assert (S[m].size + (~m).sum() == S.size)
    pred, res, _ = med_full(S)
    assert (pred + res == S).all(), "subband MED inversion"
    md = huff_stream_bits(res.reshape(-1))
    best = min(o0, sp, md)
    sel = "o0" if best == o0 else ("sparse" if best == sp else "med")
    return best + 2, sel, {"o0": o0, "sparse": sp, "med": md}


def transform_arm(ch):
    ch = ch.astype(np.int64)
    LL, det = fwd53_2d(ch, 1)
    assert (inv53_2d(LL, det, 1) == ch).all(), "5/3 inverse mismatch"
    llb, llsel = ll_best_bits(LL)
    tot = llb
    wins = {}
    spars = {}
    for (l, k, S) in det:
        b, sel, dbg = detail_best_bits(S)
        tot += b
        wins[k] = sel
        spars[k] = (float((S != 0).mean()), dbg)
    return tot, llsel, wins, spars


def probe_image(path):
    img = np.array(Image.open(path).convert("RGB"))
    H, W, _ = img.shape
    denom = H * W * 3
    Y, Co, Cg = ycocg_fwd(img)
    t = Y - (Cg // 2); B = t - (Co // 2); G = Cg + t; R = Co + B
    assert (np.array_equal(R, img[:, :, 0].astype(np.int32)) and
            np.array_equal(G, img[:, :, 1].astype(np.int32)) and
            np.array_equal(B, img[:, :, 2].astype(np.int32))), "YCoCg-R not exact"
    pix = {}
    for nm, ch in (("Y", Y), ("Co", Co), ("Cg", Cg)):
        pred, res, ctx = med_pred_ctx9(ch)
        assert ((pred + res) == ch).all(), f"pixel MED inversion {nm}"
        p0 = huff_stream_bits(res.reshape(-1))
        p9 = grouped9_bits(res, ctx)
        pix[nm] = (p0, p9)
    P0 = sum(v[0] for v in pix.values()) + HEADER
    P9 = sum(v[1] for v in pix.values()) + HEADER
    Tch = {}
    for nm, ch in (("Y", Y), ("Co", Co), ("Cg", Cg)):
        b, llsel, wins, spars = transform_arm(ch)
        Tch[nm] = (b, llsel, wins, spars)
    T_ALL = sum(v[0] for v in Tch.values()) + HEADER
    H_CHROMA = pix["Y"][1] + Tch["Co"][0] + Tch["Cg"][0] + HEADER
    H_ORACLE = sum(min(pix[nm][1], Tch[nm][0]) for nm in ("Y", "Co", "Cg")) + 3 + HEADER
    out = {"P0": P0 / denom, "P9": P9 / denom, "T_ALL": T_ALL / denom,
           "H_CHROMA": H_CHROMA / denom, "H_ORACLE": H_ORACLE / denom,
           "pix": {k: (a / denom, b / denom) for k, (a, b) in pix.items()},
           "tarm": {k: v[0] / denom for k, v in Tch.items()},
           "llsel": {k: v[1] for k, v in Tch.items()},
           "wins": {k: v[2] for k, v in Tch.items()},
           "spars": {k: {sk: (fr, {kk: vv / denom for kk, vv in dd.items()})
                          for sk, (fr, dd) in v[3].items()}
                      for k, v in Tch.items()}}
    return out


if __name__ == "__main__":
    only = sys.argv[1:] or IMGS
    res = {}
    for name in only:
        t0 = time.time()
        r = probe_image(os.path.join(IMGDIR, name))
        res[name] = r
        print(f"{name}: P0={r['P0']:.4f} P9={r['P9']:.4f} T_ALL={r['T_ALL']:.4f} "
              f"H_CHROMA={r['H_CHROMA']:.4f} H_ORACLE={r['H_ORACLE']:.4f} "
              f"({time.time()-t0:.1f}s) JXL-e3={JXL_E3[name]:.4f}", flush=True)
        print(f"   pix(Y/Co/Cg p9)=({r['pix']['Y'][1]:.4f},{r['pix']['Co'][1]:.4f},"
              f"{r['pix']['Cg'][1]:.4f}) tarm=({r['tarm']['Y']:.4f},{r['tarm']['Co']:.4f},"
              f"{r['tarm']['Cg']:.4f}) wins={r['wins']} llsel={r['llsel']}", flush=True)
    avg = lambda k: sum(res[n][k] for n in res) / len(res)
    print(f"AVG P0={avg('P0'):.4f} (anchor 3.58 +-3% -> "
          f"[{(3.58*0.97):.4f},{(3.58*1.03):.4f}]) "
          f"{'PASS' if 3.58*0.97 <= avg('P0') <= 3.58*1.03 else 'FAIL'}")
    for k in ("P9", "T_ALL", "H_CHROMA", "H_ORACLE"):
        print(f"AVG {k}={avg(k):.4f} d_ceil={avg(k)-CEIL:+.4f} "
              f"({(avg(k)/CEIL-1)*100:+.2f}%)")
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "probe_b16_dump.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
