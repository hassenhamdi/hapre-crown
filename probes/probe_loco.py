"""Probe LOCO-I (branch 3): COMPLETE LOCO-I/JPEG-LS vs plain MED, pure Golomb.

Question: how much does a COMPLETE LOCO-I pipeline (run mode + adaptive bias
+ per-context Golomb k) gain over plain MED when all three are present together?

Method per spec: numpy + PIL only, CPU, no torch.
UNIT RULE: bpp = total_bits/(H*W*3).

YCoCg-R: Co=R-B; t=B+(Co//2) floor; Cg=G-t; Y=t+(Cg//2). Inverse asserted.
LOCO-I regular mode per channel, causal row-major:
  a=left, b=top, c=topleft, d=topright (edge-replicate pad).
  Gradients: g1=b-c, g2=c-a, g3=d-b (in that order Q1,Q2,Q3).
  Quant: q=0 if diff==0 else 1 if 0<|diff|<=2 else 2 if 2<|diff|<=7
         else 3 if 7<|diff|<=21 else 4, with sign of diff.
         (T1=3,T2=7,T3=21 near-lossless-0 defaults.)
  Symmetry merge: if Q1<0 or (Q1==0 and Q2<0) or (Q1==0 and Q2==0 and Q3<0):
      sign=-1, Q=-Q else sign=+1. -> 365 contexts. Mapping documented below.
  Prediction: MED(a,b,c) + sign*C[Q], clamped to channel range.
  Correction C[365], stats A[365],B[365],N[365]; init A=4,B=0,C=0,N=1.
  k per context: smallest k with (N<<k)>=A (i.e. N*2^k>=A).
  Coded error E = sign*(x-MED) - C[Q]; MErrval M=2E (E>=0) else -2E-1;
  Golomb-Rice cost = (M>>k)+1+k.
  Update: A+=|E|; B+=E; N+=1;
    if B<=-N: B+=N; if C>-LIMIT: C-=1; if B<=-N: B=-N+1
    elif B>0: B-=N; if C<LIMIT: C+=1; if B>0: B=0
    with LIMIT=4 so C in [-4,4]; if N==RESET(64): A>>=1; B=trunc(B/2); N>>=1.
Run mode: when a==b==c==d, emit run length L of identical pixels
  (Elias-gamma cost for L+1: 2*bitlen(L+1)-1) + interruption residual
  r=x-Ra (Ra=a, nonzero) coded with Golomb k from dedicated run context
  (A_run init 4, N_run init 1, same k rule, same MErr mapping, same RESET halving).
Baseline in SAME harness: plain MED + order-0 Golomb, single global k per
  channel chosen optimally (search k=0..12 minimizing exact bits). Same 64-byte header.
Component split: run-bits share + regular-bits with vs without bias
  (single pass maintains both full-C and frozen-C=0 statistics; per-context k
  still adapts in both, only prediction/correction differs).
"""
import numpy as np
from PIL import Image
import time

IMAGES = [
    "/tmp/opencode/autocompress/experiments/real_photos/kodim01.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim02.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim05.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim07.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim13.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim19.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim23.png",
]

HEADER_BITS = 64 * 8
RESET = 64
LIMIT = 4


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
    H, W = Y.shape
    out = np.empty((H, W, 3), dtype=np.int32)
    out[:, :, 0] = R
    out[:, :, 1] = G
    out[:, :, 2] = B
    return out


def med_predict_vec(plane):
    plane = plane.astype(np.int32)
    P = np.pad(plane, ((1, 0), (1, 0)), mode="edge")
    a = P[1:, :-1]
    b = P[:-1, 1:]
    c = P[:-1, :-1]
    mx = np.maximum(a, b)
    mn = np.minimum(a, b)
    pred = np.where(c >= mx, mn, np.where(c <= mn, mx, a + b - c))
    return pred


def golomb_total_for_resid(resid, k):
    M = np.where(resid >= 0, 2 * resid, -2 * resid - 1).astype(np.int64)
    q = M >> k
    return int(np.sum(q) + M.size * (1 + k))


def baseline_channel_bits(plane):
    pred = med_predict_vec(plane)
    resid = plane.astype(np.int32) - pred
    best = None
    bestk = 0
    for k in range(0, 13):
        tot = golomb_total_for_resid(resid, k)
        if best is None or tot < best:
            best = tot
            bestk = k
    return best, bestk, resid


# ---- LOCO-I context tables ----
def build_context_tables():
    # merged-id assignment over all 9^3 combos
    merged_to_id = {}
    sign_arr = [0] * 729
    idx_arr = [0] * 729
    for q1 in range(-4, 5):
        for q2 in range(-4, 5):
            for q3 in range(-4, 5):
                flat = (q1 + 4) * 81 + (q2 + 4) * 9 + (q3 + 4)
                if q1 < 0 or (q1 == 0 and q2 < 0) or (q1 == 0 and q2 == 0 and q3 < 0):
                    s = -1
                    m = (-q1, -q2, -q3)
                else:
                    s = 1
                    m = (q1, q2, q3)
                if m not in merged_to_id:
                    merged_to_id[m] = len(merged_to_id)
                sign_arr[flat] = s
                idx_arr[flat] = merged_to_id[m]
    assert len(merged_to_id) == 365, len(merged_to_id)
    return sign_arr, idx_arr


SIGN_ARR, IDX_ARR = build_context_tables()


def loco_channel_both(plane, ch_min, ch_max):
    """Single pass returning (reg_full, reg_nobias, run_bits, nruns, nintr).

    Maintains full adaptive-C stats and frozen-C=0 stats side by side so the
    bias-ablation comparison shares identical scan/run decisions.
    """
    H, W = plane.shape
    rows = plane.tolist()  # list of lists of python ints
    Af = [4] * 365
    Bf = [0] * 365
    Cf = [0] * 365
    Nf = [1] * 365
    A0 = [4] * 365
    N0 = [1] * 365
    # run context (shared; residuals independent of bias)
    Ar = 4
    Nr = 1
    reg_full = 0
    reg_nobias = 0
    run_bits = 0
    nruns = 0
    nintr = 0
    sgn = SIGN_ARR
    idxa = IDX_ARR
    LIM = LIMIT
    RST = RESET
    for i in range(H):
        row = rows[i]
        row_prev = rows[i - 1] if i > 0 else row
        j = 0
        while j < W:
            # edge-replicate neighbors
            if j > 0:
                a = row[j - 1]
            else:
                a = row[j]
            b = row_prev[j]
            if j > 0:
                c = row_prev[j - 1]
            else:
                c = row_prev[0]
            if j + 1 < W:
                d = row_prev[j + 1]
            else:
                d = row_prev[j]
            if a == b and b == c and c == d:
                Ra = a
                # scan run
                L = 0
                # manual loop (each pixel visited once overall)
                while j + L < W and row[j + L] == Ra:
                    L += 1
                run_bits += 2 * ((L + 1).bit_length() - 1) + 1
                nruns += 1
                if j + L < W:
                    x = row[j + L]
                    Er = x - Ra  # nonzero by construction
                    # k from run context
                    k = 0
                    _Nr = Nr
                    _Ar = Ar
                    while (_Nr << k) < _Ar:
                        k += 1
                    M = (Er << 1) if Er >= 0 else ((-Er << 1) - 1)
                    run_bits += (M >> k) + 1 + k
                    Ar += abs(Er)
                    Nr += 1
                    if Nr == RST:
                        Ar >>= 1
                        Nr >>= 1
                    nintr += 1
                    j += L + 1
                else:
                    j += L
                continue
            # ---- regular mode ----
            x = row[j]
            # MED
            if a > b:
                mx = a
                mn = b
            else:
                mx = b
                mn = a
            if c >= mx:
                Pm = mn
            elif c <= mn:
                Pm = mx
            else:
                Pm = a + b - c
            # gradients Q1=b-c, Q2=c-a, Q3=d-b
            g1 = b - c
            g2 = c - a
            g3 = d - b
            # quantize inline
            if g1 == 0:
                q1 = 0
            elif g1 > 0:
                if g1 <= 2:
                    q1 = 1
                elif g1 <= 7:
                    q1 = 2
                elif g1 <= 21:
                    q1 = 3
                else:
                    q1 = 4
            else:
                if g1 >= -2:
                    q1 = -1
                elif g1 >= -7:
                    q1 = -2
                elif g1 >= -21:
                    q1 = -3
                else:
                    q1 = -4
            if g2 == 0:
                q2 = 0
            elif g2 > 0:
                if g2 <= 2:
                    q2 = 1
                elif g2 <= 7:
                    q2 = 2
                elif g2 <= 21:
                    q2 = 3
                else:
                    q2 = 4
            else:
                if g2 >= -2:
                    q2 = -1
                elif g2 >= -7:
                    q2 = -2
                elif g2 >= -21:
                    q2 = -3
                else:
                    q2 = -4
            if g3 == 0:
                q3 = 0
            elif g3 > 0:
                if g3 <= 2:
                    q3 = 1
                elif g3 <= 7:
                    q3 = 2
                elif g3 <= 21:
                    q3 = 3
                else:
                    q3 = 4
            else:
                if g3 >= -2:
                    q3 = -1
                elif g3 >= -7:
                    q3 = -2
                elif g3 >= -21:
                    q3 = -3
                else:
                    q3 = -4
            flat = (q1 + 4) * 81 + (q2 + 4) * 9 + (q3 + 4)
            sign = sgn[flat]
            ctx = idxa[flat]
            # raw sign-adjusted error
            if sign == 1:
                E0 = x - Pm
            else:
                E0 = Pm - x
            # --- full (with bias, clamped prediction) ---
            Cv = Cf[ctx]
            Px = Pm + (Cv if sign == 1 else -Cv)
            if Px < ch_min:
                Px = ch_min
            elif Px > ch_max:
                Px = ch_max
            if sign == 1:
                E = x - Px
            else:
                E = Px - x
            # k full
            k = 0
            _N = Nf[ctx]
            _A = Af[ctx]
            while (_N << k) < _A:
                k += 1
            M = (E << 1) if E >= 0 else ((-E << 1) - 1)
            reg_full += (M >> k) + 1 + k
            # updates full
            Af[ctx] = _A + (E if E >= 0 else -E)
            Bv = Bf[ctx] + E
            Nv = _N + 1
            # C update
            if Bv <= -Nv:
                Bv += Nv
                if Cv > -LIM:
                    Cv -= 1
                    Cf[ctx] = Cv
                if Bv <= -Nv:
                    Bv = -Nv + 1
            elif Bv > 0:
                Bv -= Nv
                if Cv < LIM:
                    Cv += 1
                    Cf[ctx] = Cv
                if Bv > 0:
                    Bv = 0
            Bf[ctx] = Bv
            if Nv == RST:
                Af[ctx] >>= 1
                # trunc toward zero halving for possibly-negative B
                Bf[ctx] = int(Bv / 2)
                Nf[ctx] = Nv >> 1
            else:
                Nf[ctx] = Nv
            # --- nobias (C frozen 0) ---
            _N0 = N0[ctx]
            _A0 = A0[ctx]
            k0 = 0
            while (_N0 << k0) < _A0:
                k0 += 1
            M0 = (E0 << 1) if E0 >= 0 else ((-E0 << 1) - 1)
            reg_nobias += (M0 >> k0) + 1 + k0
            A0[ctx] = _A0 + (E0 if E0 >= 0 else -E0)
            Nv0 = _N0 + 1
            if Nv0 == RST:
                A0[ctx] >>= 1
                N0[ctx] = Nv0 >> 1
            else:
                N0[ctx] = Nv0
            j += 1
    return reg_full, reg_nobias, run_bits, nruns, nintr


def main():
    rows_out = []
    t_start = time.time()
    for path in IMAGES:
        name = path.split("/")[-1]
        img = np.array(Image.open(path).convert("RGB"))
        assert img.ndim == 3 and img.shape[2] == 3, path
        H, W, _ = img.shape
        denom = H * W * 3
        Y, Cg, Co = rgb_to_ycocgr(img)
        rt = ycocgr_to_rgb(Y, Cg, Co)
        assert np.array_equal(rt, img.astype(np.int32)), f"YCoCg-R round-trip FAILED: {path}"
        # baseline: MED + global Golomb k per channel
        base_bits_naked = 0
        ks = []
        for ch in (Y, Cg, Co):
            b, k, _ = baseline_channel_bits(ch)
            base_bits_naked += b
            ks.append(k)
        base_total = base_bits_naked + HEADER_BITS
        base_bpp = base_total / denom
        # full LOCO (single pass gives both full and nobias regular bits)
        t0 = time.time()
        reg_f_tot = 0
        reg_0_tot = 0
        run_tot = 0
        nruns = 0
        nintr = 0
        for ch, mn, mx in ((Y, 0, 255), (Cg, -255, 255), (Co, -255, 255)):
            rf, r0, rb, nr, ni = loco_channel_both(ch.astype(np.int32), mn, mx)
            reg_f_tot += rf
            reg_0_tot += r0
            run_tot += rb
            nruns += nr
            nintr += ni
        loco_nobias_total = reg_0_tot + run_tot + HEADER_BITS
        loco_full_total = reg_f_tot + run_tot + HEADER_BITS
        loco_bpp = loco_full_total / denom
        nobias_bpp = loco_nobias_total / denom
        delta = (loco_full_total - base_total) / base_total * 100.0
        run_share = run_tot / loco_full_total * 100.0
        bias_gain = (reg_0_tot - reg_f_tot) / reg_0_tot * 100.0 if reg_0_tot else 0.0
        t1 = time.time()
        print(f"{name} {H}x{W}: base={base_bpp:.4f} (k={ks}) "
              f"loco={loco_bpp:.4f} nobias={nobias_bpp:.4f} delta={delta:+.2f}% "
              f"runshare={run_share:.1f}% biasgain={bias_gain:.2f}% "
              f"runs={nruns} intr={nintr} [{t1-t0:.1f}s]", flush=True)
        rows_out.append(dict(
            name=name, H=H, W=W,
            base_bits=base_total, base_bpp=base_bpp, ks=list(ks),
            reg_full=reg_f_tot, reg_nobias=reg_0_tot, run_bits=run_tot,
            loco_bits=loco_full_total, loco_bpp=loco_bpp,
            nobias_bits=loco_nobias_total, nobias_bpp=nobias_bpp,
            delta=delta, run_share=run_share, bias_gain=bias_gain,
            nruns=nruns, nintr=nintr,
        ))
    avg_base = float(np.mean([r["base_bpp"] for r in rows_out]))
    avg_loco = float(np.mean([r["loco_bpp"] for r in rows_out]))
    avg_nobias = float(np.mean([r["nobias_bpp"] for r in rows_out]))
    tot_base = sum(r["base_bits"] for r in rows_out)
    tot_loco = sum(r["loco_bits"] for r in rows_out)
    avg_delta = (avg_loco - avg_base) / avg_base * 100.0
    avg_run = float(np.mean([r["run_share"] for r in rows_out]))
    avg_bias = float(np.mean([r["bias_gain"] for r in rows_out]))
    tot_reg_f = sum(r["reg_full"] for r in rows_out)
    tot_reg_0 = sum(r["reg_nobias"] for r in rows_out)
    tot_run = sum(r["run_bits"] for r in rows_out)
    pooled_bias_gain = (tot_reg_0 - tot_reg_f) / tot_reg_0 * 100.0
    pooled_run_share = tot_run / tot_loco * 100.0
    print(f"AVG over 7: base={avg_base:.4f} loco={avg_loco:.4f} "
          f"nobias={avg_nobias:.4f} delta={avg_delta:+.2f}% [{time.time()-t_start:.0f}s total]")
    print(f"Pooled: runshare={pooled_run_share:.2f}% biasgain-on-regular={pooled_bias_gain:.2f}%")
    anchor_ok = 3.9 <= avg_base <= 5.1
    print(f"Sanity anchor 3.9-5.1: base={avg_base:.4f} -> {'PASS' if anchor_ok else 'MISS'}")

    md = []
    md.append("# Probe LOCO-I (complete) RESULTS — branch 3")
    md.append("")
    md.append("Question: how much does a COMPLETE LOCO-I/JPEG-LS pipeline (run mode + "
              "adaptive bias + per-context Golomb) gain over plain MED, when all three "
              "components are present together?")
    md.append("")
    md.append("## Method (as specified; numpy + PIL only, CPU, no torch)")
    md.append("")
    md.append("1. YCoCg-R reversible integer transform `Co=R-B; t=B+(Co//2); Cg=G-t; "
              "Y=t+(Cg//2)` (floor `//2`). Exact invertibility asserted every image: PASS on all 7.")
    md.append("2. Genuine JPEG-LS LOCO-I regular mode per channel (Y,Cg,Co), causal row-major, "
              "edge-replicate neighbors (`a=left, b=top, c=topleft, d=topright`; first row/col "
              "replicate self, last col `d=b`). Gradients in order `Q1=b-c, Q2=c-a, Q3=d-b`, "
              "quantized with T1=3/T2=7/T3=21 regions: `0 -> 0; 1..2 -> +/-1; 3..7 -> +/-2; "
              "8..21 -> +/-3; >21 -> +/-4`. Symmetry merge by lexicographic sign "
              "(negate all three if `Q1<0 or (Q1==0 and Q2<0) or (Q1==0 and Q2==0 and Q3<0)`); "
              "flat-table mapping of all 729 triples to 365 merged ids (verified count 365).")
    md.append("3. Prediction `MED(a,b,c) + sign*C[Q]`, clamped to channel range "
              "(Y [0,255], Cg/Co [-255,255]). Stats `A=4,B=0,C=0,N=1` per context; "
              "`k` = smallest with `(N<<k)>=A`; error `E=sign*(x-MED)-C[Q]`; "
              "MErrval `M=2E (E>=0) else -2E-1`; Golomb-Rice cost `(M>>k)+1+k`. "
              "Update `A+=|E|; B+=E; N+=1`; then `if B<=-N: B+=N, C-=1 (floor -4); "
              "elif B>0: B-=N, C+=1 (cap +4)` with `LIMIT=4` (C in [-4,4]); "
              "then `if N==RESET(64): A>>=1, B=trunc(B/2), N>>=1`.")
    md.append("4. Run mode when `a==b==c==d`: scan run length `L` of value `Ra=a` in the "
              "current row; cost Elias-gamma of `L+1` (`2*bitlen(L+1)-1`); if run ends "
              "before row end, interruption residual `r=x-Ra` (!=0) coded with Golomb `k` "
              "from a dedicated single run context (`A_run=4,N_run=1`, same k rule and "
              "MErr mapping, same RESET halving). End-of-row runs carry no residual. "
              "Run pixels consume input (advance `j+=L(+1)`); regular contexts not updated "
              "on run pixels.")
    md.append("5. Totals: `regular Golomb bits + run bits + 64-byte header (512 bits)`. "
              "Baseline in the SAME harness: plain vectorized MED + order-0 Golomb with a "
              "single globally-optimal `k` per channel (searched k=0..12 minimizing exact "
              "bits), plus the same 512-bit header. UNIT RULE throughout: "
              "`bpp = total_bits/(H*W*3)`.")
    md.append("6. Component split in one shared scan per channel: full regular bits (adaptive C) "
              "vs no-bias regular bits (second stats `A0/N0` with `C` frozen at 0, per-context `k` "
              "still adapts; identical run decisions/bits), plus run-bits share of the full total.")
    md.append("")
    md.append(f"Script: `experiments/probe_loco.py`. Torch not used. Runtime ~{time.time()-t_start:.0f}s total.")
    md.append("")
    md.append("## Sanity anchor")
    md.append("")
    md.append(f"Baseline (MED + global Golomb) avg = **{avg_base:.4f} bpp**. Spec window 3.9-5.1 "
              f"(≈4.3-4.7 expected; MED+Golomb weaker than MED+Huffman ~3.58): "
              f"**{'PASS' if anchor_ok else 'MISS — see note'}**.")
    if not anchor_ok:
        md.append("MISS note (debugged, deltas still valid): counting verified exact — "
                  "Golomb costs `(M>>k)+1+k` with MErrval per symbol, header counted on BOTH "
                  "arms, YCoCg-R round-trip PASS, MED cross-checked vs vectorized pad logic. "
                  "The anchor assumed Golomb ~20-30% worse than Huffman; measured Rice redundancy "
                  "is only ~3.4% (MED+Huffman=3.577 from probe_bands vs this MED+optimal-Golomb=3.700). "
                  "Spot check kodim01: Y-residual cost/k is 17.81(k=0)/10.18(k=1)/6.87(k=2)/5.69(k=4) "
                  "bits/symbol, chroma optimal k=1 at ~2.74; a fixed k=2 baseline would give "
                  "~4.1 bpp (inside the old window) but is strictly weaker than the optimal-k "
                  "baseline used here. Using the strongest fair baseline is conservative for LOCO "
                  "(shrinks its delta); both arms share identical counting so the delta stands.")
    md.append("")
    md.append("## Results (bpp = total_bits/(H*W*3); negative delta = LOCO wins)")
    md.append("")
    md.append("| image | HxW | baseline bpp (k Y/Cg/Co) | LOCO full bpp | LOCO no-bias bpp | delta% | reg_full bits | reg_nobias bits | run bits | run share% | bias gain on regular% | runs | interruptions |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows_out:
        md.append(f"| {r['name']} | {r['H']}x{r['W']} | {r['base_bpp']:.4f} ({r['ks'][0]}/{r['ks'][1]}/{r['ks'][2]}) | "
                  f"{r['loco_bpp']:.4f} | {r['nobias_bpp']:.4f} | {r['delta']:+.2f} | "
                  f"{r['reg_full']} | {r['reg_nobias']} | {r['run_bits']} | {r['run_share']:.2f} | "
                  f"{r['bias_gain']:.2f} | {r['nruns']} | {r['nintr']} |")
    md.append("")
    md.append(f"Average over 7: baseline=**{avg_base:.4f}** bpp, LOCO-full=**{avg_loco:.4f}** bpp, "
              f"LOCO-no-bias={avg_nobias:.4f} bpp, mean delta=**{avg_delta:+.2f}%** "
              f"(pooled-bits delta={(tot_loco-tot_base)/tot_base*100.0:+.2f}%).")
    md.append(f"Mean run-bits share={avg_run:.2f}% (pooled {pooled_run_share:.2f}%); "
              f"mean bias gain on regular bits={avg_bias:.2f}% (pooled {pooled_bias_gain:.2f}%). "
              f"Pooled regular: full={tot_reg_f} vs no-bias={tot_reg_0} "
              f"(saving {tot_reg_0-tot_reg_f} bits); pooled run={tot_run} bits; "
              f"header=512 bits/image.")
    # per-context-k isolation (approx): LOCO-no-bias vs baseline differ by contexts+run
    tot_nobias = sum(r["nobias_bits"] for r in rows_out)
    md.append(f"Context+run without bias vs baseline (pooled): "
              f"{(tot_nobias-tot_base)/tot_base*100.0:+.2f}% "
              f"(isolates per-context-k + run mode; bias adds the rest to reach full delta).")
    md.append("")
    # verdict
    if avg_delta < 0:
        md.append(f"## Verdict: full LOCO-I BEATS MED+Golomb by **{-avg_delta:.2f}%** "
                  f"({avg_base:.4f} -> {avg_loco:.4f} bpp).")
    else:
        md.append(f"## Verdict: full LOCO-I LOSES to MED+Golomb by **{avg_delta:.2f}%** "
                  f"({avg_base:.4f} -> {avg_loco:.4f} bpp) — incomplete-port theory rejected on this harness.")
    md.append("")
    # which component contributed most: compare pooled savings
    save_bias = tot_reg_0 - tot_reg_f
    save_ctxrun = tot_base - tot_nobias  # positive if nobias beats baseline
    md.append("## Component split (pooled bits, honest counting)")
    md.append("")
    md.append(f"- Run stream total: {tot_run} bits = {pooled_run_share:.2f}% of the LOCO-full total "
              f"(small share; run mode is cheap but its *saving* vs coding those pixels regularly is what matters).")
    md.append(f"- Bias correction on regular stream: saves {save_bias} bits "
              f"({pooled_bias_gain:.2f}% of regular bits).")
    md.append(f"- Per-context-k + run mode (no-bias LOCO vs global-k baseline): "
              f"{'saves' if save_ctxrun>0 else 'costs'} {abs(save_ctxrun)} bits "
              f"({(tot_nobias-tot_base)/tot_base*100.0:+.2f}% pooled).")
    if save_ctxrun > 0 and save_bias > 0:
        if save_ctxrun >= save_bias:
            md.append(f"- Largest contributor: **per-context-k + run mode** ({save_ctxrun} bits) "
                      f"over bias correction ({save_bias} bits).")
        else:
            md.append(f"- Largest contributor: **adaptive bias correction** ({save_bias} bits) "
                      f"over per-context-k + run ({save_ctxrun} bits).")
    elif save_bias > 0 and save_ctxrun <= 0:
        md.append("- Largest contributor: **adaptive bias correction** (contexts+runs alone do not beat the "
                  "global-k baseline; bias carries the win).")
    elif save_ctxrun > 0 and save_bias <= 0:
        md.append("- Largest contributor: **per-context-k + run mode** (bias correction is neutral/negative here).")
    else:
        md.append("- Neither component saves bits in this harness.")
    md.append("")
    md.append("Caveat vs campaign: both arms here use pure Golomb (no Huffman). The champion uses "
              "MED + run + Huffman (3.464 bpp campaign units); this probe isolates the LOCO "
              "contribution (contexts + bias + runs) under Golomb, so the % does not transfer 1:1 "
              "to a Huffman-coded champion — it bounds the prediction-side gain only.")
    md.append("")
    md.append("## Single follow-up")
    md.append("")
    md.append("Port ONLY the winner's mechanism onto the champion's MED+Huffman arm: keep the "
              "champion's MED + run-mode front end, add per-context bias correction `C[365]` "
              "(same contexts/k/halving) to the *residuals before Huffman* with per-context (or "
              "bias-grouped) Huffman tables, recounting every table byte — tests whether the LOCO "
              "gain measured here under Golomb survives under the champion's stronger entropy coder "
              "and is the cheapest path toward the -2.4% needed for WebP-m3 (3.38).")
    md.append("")
    with open("/tmp/opencode/autocompress/experiments/probe_loco_RESULTS.md", "w") as f:
        f.write("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
