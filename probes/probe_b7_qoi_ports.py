"""Probe B7: QOI-portable mechanisms — luma-anchored chroma (A) + hash-cache escape (B).

Branch probe for the lossless-codec campaign (boss JXL-e3 3.23 needs ~-1.5% from 3.272).

Method: numpy + PIL only, CPU, no torch. UNIT RULE: bpp = total_bits/(H*W*3).
Exact counting: real heapq Huffman data bits + 16 + A*24 per stream, all side bytes.

YCoCg-R: Co=R-B; t=B+(Co//2) floor; Cg=G-t; Y=t+(Cg//2). Inverse asserted per image.
Baseline: MED residuals (edge-replicate pad) per channel, independent order-0
  Huffman per channel: total = 64 dims bits + sum_c(data_c + 16 + A_c*24).

Mechanism A (luma-anchored chroma), Y stream coded first, all decoder-safe:
  A-joint: single (rCo,rCg) pair alphabet vs two independent streams.
    Honest table cost 16 + A_joint*48 (each entry holds TWO int16 values).
  A-binK (K=2,3,4): partition pixels by |rY| with FIXED global thresholds
    (zero side bytes; decoder recomputes bins from already-decoded Y stream):
      K=2 bounds [1]: {|rY|<=1} vs {>1}
      K=3 bounds [1,4]: <=1 / 2..4 / >4
      K=4 bounds [0,2,7]: 0 / 1..2 / 3..7 / >7
    Co/Cg residuals in each bin get their own Huffman table:
      total = 64 + Y_full + sum_bins(Co_bin + Cg_bin).
  A-binQ4: K=4 equal-mass quantile bins of |rY| per image; thresholds counted
    (+24 bits, conservative — decoder could recompute them from Y for 0).

Mechanism B (distant-repeat hash-cache escape, QOI-INDEX port):
  Causal row-major scan over YCoCg triplets. 64-entry cache, zero-init (QOI-style,
  decoder mirrors exactly). Hash h = (Y*3 + (Co+256)*5 + (Cg+256)*7) & 63
  (all terms non-negative so &63 == %64; mirrors QOI_COLOR_HASH shape).
  Per pixel: hit iff cache[h]==triple -> emit HIT (no residual); else MISS, store.
  Exact counting, all entropy-coded (no assumed "1 bit"):
    flag stream (hit/miss, A<=2) + index stream (hit slot values, A<=64,
      present iff nhit>0) + 3 miss-residual streams (MED residuals of MISS
      pixels only, scan order). Total = 64 + flag + index + sum_c miss_c.
  Decoder mirror: init zeros; reads flags (N decisions), index seq, Y-miss
  residuals -> full Y recon (causal MED, lossless) -> places Co/Cg miss
  residuals at miss positions. Hit pixels: triple = cache[h]. State evolves from
  recon triples only -> identical to encoder. Exact-repeat check means no
  false hits (hash collision -> verified MISS, silent evict like QOI).

Combined A+B: B's cache front-end + miss Co/Cg residuals binned by |rY| of the
  same (miss) pixel with the K=4 fixed bounds. Decoder: Y fully recon first
  (hits from cache, misses from Y-miss stream), bins recomputed per miss pixel,
  then per-bin Co/Cg miss streams decoded. Total = 64 + flag + index + Ymiss
  + sum_bins(Co_miss_bin + Cg_miss_bin).
"""

import heapq
import os
import time

import numpy as np
from PIL import Image

IMAGES = [
    "/tmp/opencode/autocompress/experiments/real_photos/kodim01.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim02.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim05.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim07.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim13.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim19.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim23.png",
]

# Spec anchor: MED + order-0 Huffman avg ~3.58 (01:3.62 02:3.33 05:4.01 07:3.15 13:4.24 19:3.36 23:3.18)
EXPECTED = {
    "kodim01.png": 3.62, "kodim02.png": 3.33, "kodim05.png": 4.01,
    "kodim07.png": 3.15, "kodim13.png": 4.24, "kodim19.png": 3.36,
    "kodim23.png": 3.18,
}
DIMS_BITS = 64  # dims header, counted on every arm

OUT_MD = "/tmp/opencode/autocompress/experiments/probe_b7_RESULTS.md"


def rgb_to_ycocgr(arr):
    R = arr[:, :, 0].astype(np.int32)
    G = arr[:, :, 1].astype(np.int32)
    B = arr[:, :, 2].astype(np.int32)
    Co = R - B
    t = B + (Co // 2)
    Cg = G - t
    Y = t + (Cg // 2)
    return Y, Cg, Co


def ycocgr_to_rgb(Y, Cg, Co):
    t = Y - (Cg // 2)
    B = t - (Co // 2)
    G = Cg + t
    R = Co + B
    out = np.empty(Y.shape + (3,), dtype=np.int32)
    out[:, :, 0] = R
    out[:, :, 1] = G
    out[:, :, 2] = B
    return out


def med_residuals(ch):
    ch = ch.astype(np.int32)
    P = np.pad(ch, ((1, 0), (1, 0)), mode="edge")
    a = P[1:, :-1]
    b = P[:-1, 1:]
    c = P[:-1, :-1]
    mx = np.maximum(a, b)
    mn = np.minimum(a, b)
    pred = np.where(c >= mx, mn, np.where(c <= mn, mx, a + b - c))
    return (ch - pred).astype(np.int32)


def huffman_data_bits(counts):
    n = len(counts)
    if n == 0:
        return 0
    if n == 1:
        return int(counts[0]) * 1  # conservative single-symbol cost
    heap = [int(c) for c in counts]
    heapq.heapify(heap)
    total = 0
    while len(heap) > 1:
        x = heapq.heappop(heap)
        y = heapq.heappop(heap)
        s = x + y
        total += s
        heapq.heappush(heap, s)
    return total


def field_bits(arr, width=24):
    """Exact (data_bits, alphabet_size, total incl. 16+A*width table)."""
    _, counts = np.unique(np.asarray(arr), return_counts=True)
    A = len(counts)
    data = huffman_data_bits(counts.tolist())
    return data, A, data + 16 + A * width


def hash_cache_sim(Y, Co, Cg):
    """QOI-style 64-entry exact-repeat cache over YCoCg triplets.

    Returns (flags uint8 N, hit_index int array (nhit,), missmask bool N).
    Decoder mirrors exactly (zero init, same update rule, recon == orig).
    """
    yf = Y.astype(np.int32).ravel().tolist()
    of = Co.astype(np.int32).ravel().tolist()
    gf = Cg.astype(np.int32).ravel().tolist()
    N = len(yf)
    cY = [0] * 64
    cO = [0] * 64
    cG = [0] * 64
    flags = bytearray(N)
    hit_idx = []
    miss = bytearray(N)
    for n in range(N):
        y = yf[n]
        o = of[n]
        g = gf[n]
        h = (y * 3 + (o + 256) * 5 + (g + 256) * 7) & 63
        if cY[h] == y and cO[h] == o and cG[h] == g:
            flags[n] = 1
            hit_idx.append(h)
        else:
            cY[h] = y
            cO[h] = o
            cG[h] = g
            miss[n] = 1
    return (np.frombuffer(bytes(flags), dtype=np.uint8),
            np.array(hit_idx, dtype=np.int64),
            np.frombuffer(bytes(miss), dtype=np.uint8).astype(bool))


def main():
    try:
        import torch  # noqa
        has_torch = True
    except ImportError:
        has_torch = False
    print(f"torch importable: {has_torch} (NOT used; numpy+PIL only), numpy {np.__version__}")
    t_start = time.time()
    rows = []
    for path in IMAGES:
        name = os.path.basename(path)
        t0 = time.time()
        img = np.array(Image.open(path).convert("RGB"), dtype=np.int32)
        H, W, _ = img.shape
        denom = H * W * 3
        N = H * W
        Y, Cg, Co = rgb_to_ycocgr(img)
        assert np.array_equal(ycocgr_to_rgb(Y, Cg, Co), img), f"YCoCg-R round-trip FAILED {name}"

        rY = med_residuals(Y)
        rCo = med_residuals(Co)
        rCg = med_residuals(Cg)
        arY = np.abs(rY)

        # ---- baseline: independent order-0 Huffman ----
        dY, AY, tY = field_bits(rY)
        dCo, ACo, tCo = field_bits(rCo)
        dCg, ACg, tCg = field_bits(rCg)
        base_total = DIMS_BITS + tY + tCo + tCg
        base_bpp = base_total / denom

        # ---- A-joint: (rCo,rCg) pair alphabet, honest 48-bit entries ----
        keys = (rCo.astype(np.int64) + 1024) * 4096 + (rCg.astype(np.int64) + 1024)
        dJ, AJ, tJ = field_bits(keys, width=48)
        a_joint_total = DIMS_BITS + tY + tJ
        a_joint_bpp = a_joint_total / denom

        # ---- A-binK: fixed-threshold |rY| bins, separate Co/Cg tables ----
        bin_cfgs = {"K2": [1], "K3": [1, 4], "K4": [0, 2, 7]}
        a_bin = {}
        fY = rY.ravel()
        fCo = rCo.ravel()
        fCg = rCg.ravel()
        farY = arY.ravel()
        for cfg, bounds in bin_cfgs.items():
            b = np.digitize(farY, np.array(bounds, dtype=np.int64), right=True)
            tot = DIMS_BITS + tY
            nbins = len(bounds) + 1
            for bi in range(nbins):
                m = b == bi
                _, _, tco = field_bits(fCo[m])
                _, _, tcg = field_bits(fCg[m])
                tot += tco + tcg
            a_bin[cfg] = tot / denom
        # ---- A-binQ4: per-image equal-mass quantile bins (+24 side bits, conservative) ----
        qs = np.quantile(farY.astype(np.float64), [0.25, 0.5, 0.75])
        qb = np.unique(qs.astype(np.int64))
        if len(qb) < 3:  # degenerate fallback to fixed K4 bounds
            qb = np.array([0, 2, 7], dtype=np.int64)
        else:
            qb = qb[:3]
        bq = np.digitize(farY, qb, right=True)
        totq = DIMS_BITS + tY + 24
        for bi in range(len(qb) + 1):
            m = bq == bi
            _, _, tco = field_bits(fCo[m])
            _, _, tcg = field_bits(fCg[m])
            totq += tco + tcg
        a_bin["Q4"] = totq / denom

        # ---- B: hash-cache escape ----
        flags, hit_idx, miss = hash_cache_sim(Y, Co, Cg)
        nhit = int(flags.sum())
        nmiss = N - nhit
        assert nhit + nmiss == N
        assert len(hit_idx) == nhit
        dF, AF, tF = field_bits(flags)
        if nhit > 0:
            dI, AI, tI = field_bits(hit_idx)
        else:
            dI, AI, tI = 0, 0, 0
        dYm, AYm, tYm = field_bits(fY[miss])
        dOm, AOm, tOm = field_bits(fCo[miss])
        dGm, AGm, tGm = field_bits(fCg[miss])
        b_total = DIMS_BITS + tF + tI + tYm + tOm + tGm
        b_bpp = b_total / denom
        hit_rate = nhit / N
        # decomposition: residual-data saved vs escape-side cost
        resid_data_base = dY + dCo + dCg
        resid_data_miss = dYm + dOm + dGm
        escape_cost = tF + tI
        table_delta = (16 + AYm * 24) + (16 + AOm * 24) + (16 + AGm * 24) - (
            (16 + AY * 24) + (16 + ACo * 24) + (16 + ACg * 24))

        # ---- A+B: cache front-end + K4-fixed-binned miss chroma ----
        bK4 = np.array([0, 2, 7], dtype=np.int64)
        tot_ab = DIMS_BITS + tF + tI + tYm
        fCom = fCo[miss]
        fCgm = fCg[miss]
        fArm = farY[miss]
        bbm = np.digitize(fArm, bK4, right=True)
        for bi in range(4):
            m = bbm == bi
            _, _, tco = field_bits(fCom[m])
            _, _, tcg = field_bits(fCgm[m])
            tot_ab += tco + tcg
        ab_total = tot_ab
        ab_bpp = ab_total / denom

        anchor = EXPECTED[name]
        ok = abs(base_bpp - anchor) / anchor <= 0.03
        print(f"{name} {H}x{W}: base={base_bpp:.4f} (exp {anchor:.2f} {'PASS' if ok else 'MISS'}) "
              f"Ajoint={a_joint_bpp:.4f} AK2={a_bin['K2']:.4f} AK3={a_bin['K3']:.4f} "
              f"AK4={a_bin['K4']:.4f} AQ4={a_bin['Q4']:.4f} "
              f"B={b_bpp:.4f} (hit={hit_rate*100:.2f}%) A+B={ab_bpp:.4f} [{time.time()-t0:.1f}s]",
              flush=True)
        rows.append(dict(name=name, H=H, W=W, denom=denom, N=N,
                         base_bpp=base_bpp, base_total=base_total,
                         AY=AY, ACo=ACo, ACg=ACg,
                         AJ=AJ, a_joint_bpp=a_joint_bpp, a_joint_total=a_joint_total,
                         a_bin=dict(a_bin), qb=[int(x) for x in qb],
                         b_bpp=b_bpp, b_total=b_total, nhit=nhit, nmiss=nmiss,
                         hit_rate=hit_rate, AF=AF, dF=dF, tF=tF, AI=AI, dI=dI, tI=tI,
                         AYm=AYm, AOm=AOm, AGm=AGm,
                         resid_data_base=resid_data_base, resid_data_miss=resid_data_miss,
                         escape_cost=escape_cost, table_delta=table_delta,
                         ab_bpp=ab_bpp, ab_total=ab_total))

    def avg(k):
        return float(np.mean([r[k] for r in rows]))

    avg_base = avg("base_bpp")
    avg_joint = avg("a_joint_bpp")
    avg_bin = {c: float(np.mean([r["a_bin"][c] for r in rows])) for c in ("K2", "K3", "K4", "Q4")}
    avg_b = avg("b_bpp")
    avg_ab = avg("ab_bpp")
    tot_base = sum(r["base_total"] for r in rows)
    d_joint = (avg_joint - avg_base) / avg_base * 100
    d_bin = {c: (v - avg_base) / avg_base * 100 for c, v in avg_bin.items()}
    d_b = (avg_b - avg_base) / avg_base * 100
    d_ab = (avg_ab - avg_base) / avg_base * 100
    avg_hit = float(np.mean([r["hit_rate"] for r in rows])) * 100
    print("-" * 80)
    print(f"AVG base={avg_base:.4f} Ajoint={avg_joint:.4f} ({d_joint:+.2f}%) "
          + " ".join(f"A{c}={avg_bin[c]:.4f} ({d_bin[c]:+.2f}%)" for c in ("K2", "K3", "K4", "Q4"))
          + f" B={avg_b:.4f} ({d_b:+.2f}%, hit {avg_hit:.2f}%) A+B={avg_ab:.4f} ({d_ab:+.2f}%)")
    print(f"[{time.time()-t_start:.0f}s total]")

    # ---- write RESULTS.md ----
    L = []
    L.append("# Probe B7 RESULTS — QOI ports: luma-anchored chroma (A) + hash-cache escape (B)")
    L.append("")
    L.append("Branch probe: two cheap QOI-portable mechanisms from SURVEY_genai_qoi.md §2, "
             "probed separately and combined. numpy + PIL only, CPU, no torch. "
             "UNIT RULE throughout: `bpp = total_bits/(H*W*3)`. Exact counting: real heapq "
             "Huffman data bits + `16+A*24` per stream, every table and side byte counted.")
    L.append("")
    L.append("## Baseline + anchors")
    L.append("")
    L.append("MED residuals (edge-replicate pad) in YCoCg-R + independent order-0 Huffman per "
             "channel. Total = 64 dims bits + Σ(data_c + 16 + A_c·24). "
             "YCoCg-R round-trip asserted per image: PASS 7/7.")
    L.append("")
    L.append("| image | HxW | base bpp | expected | anchor ±3% | A(Y/Co/Cg) |")
    L.append("|---|---|---|---|---|---|")
    for r in rows:
        exp = EXPECTED[r["name"]]
        ok = abs(r["base_bpp"] - exp) / exp <= 0.03
        L.append(f"| {r['name']} | {r['H']}x{r['W']} | {r['base_bpp']:.4f} | {exp:.2f} | "
                 f"{'PASS' if ok else 'MISS'} | {r['AY']}/{r['ACo']}/{r['ACg']} |")
    L.append("")
    L.append(f"Baseline avg = **{avg_base:.4f} bpp** (spec ≈3.58). "
             f"{'All 7 anchors PASS ±3%.' if all(abs(r['base_bpp']-EXPECTED[r['name']])/EXPECTED[r['name']]<=0.03 for r in rows) else 'ANCHOR MISS — see trail.'}")
    L.append("")
    L.append("## Mechanism A — luma-anchored chroma (Y stream first, decoder-safe)")
    L.append("")
    L.append("A-joint: one (rCo,rCg) pair alphabet; table cost honest `16+A·48` (two int16/entry). "
             "A-binK: pixels binned by |rY| with FIXED global thresholds (0 side bytes; decoder "
             "recomputes bins from the already-decoded Y stream), separate Co/Cg Huffman tables "
             "per bin. Bounds: K2 [1]; K3 [1,4]; K4 [0,2,7] (upper |rY| bounds). A-binQ4: "
             "per-image equal-mass quantile bins of |rY| (+24 side bits counted, conservative).")
    L.append("")
    L.append("| image | base | A-joint (Δ%) | A-K2 (Δ%) | A-K3 (Δ%) | A-K4 (Δ%) | A-Q4 (Δ%) |")
    L.append("|---|---|---|---|---|---|---|")
    for r in rows:
        def dp(v):
            return (v - r["base_bpp"]) / r["base_bpp"] * 100
        L.append(f"| {r['name']} | {r['base_bpp']:.4f} | {r['a_joint_bpp']:.4f} ({dp(r['a_joint_bpp']):+.2f}) | "
                 f"{r['a_bin']['K2']:.4f} ({dp(r['a_bin']['K2']):+.2f}) | "
                 f"{r['a_bin']['K3']:.4f} ({dp(r['a_bin']['K3']):+.2f}) | "
                 f"{r['a_bin']['K4']:.4f} ({dp(r['a_bin']['K4']):+.2f}) | "
                 f"{r['a_bin']['Q4']:.4f} ({dp(r['a_bin']['Q4']):+.2f}) |")
    L.append("")
    L.append(f"Avg: base **{avg_base:.4f}** | A-joint **{avg_joint:.4f}** ({d_joint:+.2f}%) | "
             + " | ".join(f"A-{c} **{avg_bin[c]:.4f}** ({d_bin[c]:+.2f}%)" for c in ("K2", "K3", "K4", "Q4")) + ".")
    L.append("Joint pair alphabets A_J per image: "
             + ", ".join(f"{r['name']}={r['AJ']}" for r in rows) + ". "
             + "Q4 quantile bounds per image: "
             + ", ".join(f"{r['name']}={r['qb']}" for r in rows) + ".")
    L.append("")
    L.append("## Mechanism B — distant-repeat hash-cache escape (QOI-INDEX port)")
    L.append("")
    L.append("64-entry exact-repeat cache over YCoCg triplets, zero-init, "
             "`h=(Y·3+(Co+256)·5+(Cg+256)·7)&63`, verified-hit else store (QOI eviction semantics, "
             "no false hits). Honest entropy-coded counting: flag stream (hit/miss) + index stream "
             "(hit slot, iff nhit>0) + 3 miss-residual streams (scan order). Decoder mirrors exactly "
             "(see Decoder-safety).")
    L.append("")
    L.append("| image | base | B bpp (Δ%) | hit% | flag bits (A) | index bits (A) | resid-data base→miss | tableΔ miss−full |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        dp = (r["b_bpp"] - r["base_bpp"]) / r["base_bpp"] * 100
        L.append(f"| {r['name']} | {r['base_bpp']:.4f} | {r['b_bpp']:.4f} ({dp:+.2f}) | "
                 f"{r['hit_rate']*100:.2f} | {r['tF']} ({r['AF']}) | {r['tI']} ({r['AI']}) | "
                 f"{r['resid_data_base']}→{r['resid_data_miss']} | {r['table_delta']:+d} |")
    L.append("")
    L.append(f"Avg: base **{avg_base:.4f}** → B **{avg_b:.4f}** ({d_b:+.2f}%), mean hit rate {avg_hit:.2f}%. "
             f"Effective flag cost = {sum(r['tF'] for r in rows)/sum(r['N'] for r in rows):.3f} bits/px; "
             f"index cost = {sum(r['tI'] for r in rows)/max(1,sum(r['nhit'] for r in rows)):.3f} bits/hit.")
    L.append("")
    L.append("## Combined A+B (cache front-end + K4-binned miss chroma)")
    L.append("")
    L.append("| image | base | A+B bpp (Δ%) | Δ vs B alone (pp) |")
    L.append("|---|---|---|---|")
    for r in rows:
        dp = (r["ab_bpp"] - r["base_bpp"]) / r["base_bpp"] * 100
        dpb = (r["ab_bpp"] - r["b_bpp"]) / r["base_bpp"] * 100
        L.append(f"| {r['name']} | {r['base_bpp']:.4f} | {r['ab_bpp']:.4f} ({dp:+.2f}) | {dpb:+.2f} |")
    L.append("")
    L.append(f"Avg: base **{avg_base:.4f}** → A+B **{avg_ab:.4f}** ({d_ab:+.2f}%).")
    L.append("")
    L.append("## Decoder-safety statement")
    L.append("")
    L.append("Stream order: dims → [flag table+data, index table+data] → Y table+data "
             "(miss residuals, scan order) → Co/Cg per-bin tables+data (scan order). "
             "A-only: Y stream decoded fully first (causal MED recon, lossless ⇒ encoder-identical); "
             "bins are a deterministic function of decoded |rY| with fixed global thresholds, so "
             "Co/Cg table selection is available before Co/Cg decode — causal, streaming-compatible "
             "(row-wise Y-first ordering also valid). Q4 variant: thresholds transmitted (+24 bits "
             "counted) or deterministically recomputed — either way available pre-Co/Cg. "
             "B: cache init zeros both sides; hit/miss sequence fully determined by decoded flags; "
             "cache evolves from reconstructed triplets only, which equal the encoder's (lossless), "
             "so decoder state tracks encoder state exactly — hit decision never depends on "
             "not-yet-decoded data. Exact-match verification (not bare hash) means collisions "
             "resolve to MISS on both sides identically. A+B: miss-pixel |rY| known after Y recon, "
             "before Co/Cg decode — bin selection causal. No lookahead required anywhere.")
    L.append("")
    L.append("## Trail (what was tried, exact)")
    L.append("")
    L.append(f"Script `experiments/probe_b7_qoi_ports.py` (numpy {np.__version__} + PIL, torch unused). "
             f"Runtime ~{time.time()-t_start:.0f}s total. One script covers A-joint, A-binK2/K3/K4, "
             "A-binQ4, B, A+B on all 7 images. No runs, no context mixing, no rANS — pure order-0 "
             "Huffman isolation of the two QOI-portable mechanisms. Joint-pair packing "
             "`key=(rCo+1024)*4096+(rCg+1024)` (residual range ±510 fits). Cache loop is plain "
             "Python over ravelled int lists (decoder-mirror clarity over speed).")
    L.append("")
    # verdict computed from numbers
    best_a = min(d_bin, key=lambda c: d_bin[c])
    L.append("## Verdict")
    L.append("")
    L.append(f"A (luma-anchored chroma): best variant A-{best_a} {d_bin[best_a]:+.2f}% avg; "
             f"joint-pair {d_joint:+.2f}% avg. "
             + ("PAYS (net win vs independent baseline)." if min(d_joint, min(d_bin.values())) < -0.1
                else "DOES NOT PAY under exact table counting (fragmentation ≥ conditioning gain)."))
    L.append(f"B (hash-cache escape): {d_b:+.2f}% avg at {avg_hit:.2f}% mean hit rate. "
             + ("PAYS (escape overhead < residual savings)." if d_b < -0.1
                else "DOES NOT PAY on these photos (flat-region hits too rare to cover flag+index+retabling)."))
    L.append(f"Combined A+B: {d_ab:+.2f}% avg. Boss-5 math: JXL-e3 3.23 needs ≈−1.3% "
             f"(task ≈−1.5%) from campaign 3.272; this probe's baseline-frame delta is {d_ab:+.2f}% "
             + ("— within striking distance, port candidate." if d_ab <= -1.0
                else "— NOT near Boss-5 math on top of order-0 Huffman; stack with CROWN/rANS or retire."))
    L.append("")
    L.append("## Follow-up")
    L.append("")
    if d_b < min(d_joint, min(d_bin.values())) and d_b < 0:
        L.append("Port B first: add the 64-entry exact-repeat cache as a front-end bypass to the "
                 "CROWN-huff arm (hits bypass context+residual path; count flag/index tables exactly; "
                 "keep QOI's no-refresh-on-bypass discipline). Re-probe A only inside CROWN groups "
                 "if B ports (per-group Y-binning shares tables better than global bins).")
    elif min(d_joint, min(d_bin.values())) < 0 and min(d_joint, min(d_bin.values())) <= d_b:
        L.append("Port the winning A variant first onto the CROWN-huff arm (per-group chroma tables "
                 "keyed by already-coded group-Y statistics, all tables counted). Re-test B only as "
                 "a bypass for flat-region groups where hit rate justifies its tables.")
    else:
        L.append("Neither port pays under exact counting on Kodak photos — matches the survey's "
                 "caution (QOI ≈30% behind us on photos already) and the cycle-3 LF-conditioning "
                 "retirement (side ~3× savings). Recommended: retire both as ratio plays on photos; "
                 "keep B ONLY as a non-photo/escape fast-path (icons, graphics, flat skies) where "
                 "hit rate clears the flag+index overhead; redirect Boss-e3 effort to IFCE-style "
                 "inter-group conditioning and MDL-gated micro-adapters (survey TOP-5 #4/#5).")
    L.append("")
    with open(OUT_MD, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
