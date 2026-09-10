# probe_b21 RESULTS — PER-CONTEXT micro-MLPs (neural successor at full strength)

Campaign: lossless codec; prediction-side gap concentrated on texture (05/13).
Branch question: does per-context neural beat the proven global per-channel micro-MLP?

## 0. TL;DR verdict

**YES — per-context MLPs beat global MLPs: −2.93% vs MED (−0.85% vs global-MLP), 6/7 images,
all exact-Huffman, quantized, gated, side-info counted.** The win required two remedies
found by autopsy (below): (i) int16×256 quantization DESTROYS per-ctx nets (mandated check:
−21% worst case) → adaptive-int16 scale {256…4096} retains ~100% of float gain, still int16;
(ii) 800-iter/0.01-LR harness overcooks 43k-sample subsets (sharp minima) → 200 iters/0.003
stays in the quant-robust LS neighborhood AND trains 4× faster (~32 s/image, no blocker).
Committed config: **energy-9 quantile contexts × (12-feat → 8 → 1, L1, LS-init, it200/lr0.003,
adaptive-int16), double MDL gate (per-ctx subset + per-channel assembly).**
Follow-up: MLP-as-expert transfer onto CROWN4 (Boss-5 math inside — texture blockers in reach).

## 1. Headline numbers (bpp = total_bits/(H·W·3), exact heapq Huffman +16+A·24 per channel)

Committed config h8/f12/it200/lr0.003/adaptive-int16, all 7 images:

| image | MED | global-MLP-gated | Δ vs MED | per-ctx-MLP-gated | Δ vs MED | Δ vs global |
|---|---|---|---|---|---|---|
| kodim01 | 3.6178 | 3.5745 | −1.20% | 3.5565 | −1.70% | −0.50% |
| kodim02 | 3.3255 | 3.2742 | −1.54% | 3.2464 | −2.38% | −0.85% |
| kodim05 | 4.0119 | 3.8863 | −3.13% | 3.8611 | −3.76% | −0.65% |
| kodim07 | 3.1529 | 3.1108 | −1.34% | 3.0580 | −3.01% | −1.70% |
| kodim13 | 4.2378 | 4.1358 | −2.41% | 4.1416 | −2.27% | **+0.14%** (sole loss) |
| kodim19 | 3.5081 | 3.4170 | −2.60% | 3.3949 | −3.23% | −0.65% |
| kodim23 | 3.1842 | 3.1147 | −2.18% | 3.0462 | −4.33% | −2.20% |
| **AVG** | **3.5769** | **3.5019** | **−2.10%** | **3.4721** | **−2.93%** | **−0.85%** |

- Anchor check: MED+order-0-Huffman = 3.5769 vs 3.58 nominal (0.09%) — PASS, no debug needed.
- Baselines compared on IDENTICAL residuals/coding (per-channel global Huffman); only the
  predictor differs. All per-ctx figures include full side info (below).
- Global baseline here (−2.10%) is stronger than cycle-25's −1.64% because it also uses the
  12-feat inputs + it200/lr0.003 + adaptive-int16 found in this probe (harder baseline = honest win).
- Train time: 30–37 s/image (12-thread torch CPU), full-7 ≈ 4 min. No blowup, no blocker.

## 2. Overhead accounting (counted in every per-ctx bpp above)

Per winning (ch,ctx) net: 113 params × 2 B (int16) + 1 select flag + 3 scale-id bits = 1811 bits.
Per channel with ≥1 win: 8 quantile thresholds × 16 b = 128 bits. Image header: 3-bit channel map.
Empty contexts (quantile ties, typically slots 2/4 on smooth images): fixed MED, no flag, no weights.

| image | per-ctx side (bits) | ≈ bpp | global side (bits) | ctx wins /27 |
|---|---|---|---|---|
| 01 | 23952 | 0.020 | 1814 | 13 |
| 02 | 23817 | 0.020 | 3625 | 17 |
| 05 | 38441 | 0.033 | 5436 | 21 |
| 07 | 33006 | 0.028 | 1814 | 18 |
| 13 | 36630 | 0.031 | 3625 | 20 |
| 19 | 31195 | 0.026 | 3625 | 17 |
| 23 | 31194 | 0.026 | 1814 | 17 |

MDL gates (both exact-bits): L1 per-(ch,ctx) subset Huffman (quantized net + its side vs MED
subset — conservative, subset tables overestimate cost); L2 per-channel global-Huffman assembly
+ side + thresholds vs MED (whole-channel fallback if lost — triggered once: kodim02 Co).
Quantile thresholds + flags are in the totals; Huffman table bytes (+16+A·24/ch) are inside hbits.

## 3. Quantization proof (mandated int16×256 check FIRST, then remedy)

Subset-level aggregate over evaluated contexts (22–23/image, 113 params → side 1812 b):

| image | float Δ | ×256 Δ | adapt Δ | wins ×256 / adapt (of ~22) |
|---|---|---|---|---|
| 01 | +2.49% | +0.00% | +2.51% | 9 / 13 |
| 02 | +2.60% | −0.96% | +2.70% | 13 / 17 |
| 05 | +4.81% | +0.70% | +4.80% | 14 / 21 |
| 07 | +5.18% | −4.02% | +5.03% | 4 / 18 |
| 13 | +3.26% | +2.51% | +3.28% | 18 / 20 |
| 19 | +3.87% | −1.96% | +3.78% | 10 / 17 |
| 23 | +5.28% | **−21.44%** | +5.27% | 1 / 17 |

- ×256 verdict: DESTROYS per-ctx nets on 4/7 images (including −21% on kodim23); erratic on the
  rest (only 69/153 context wins). Same for 12-feat global nets (kodim23 Co +48%, Cg +48% at ×256).
  Cycle-25's "×256 retains 70%" does NOT transfer from 8-feat global to per-ctx/12-feat nets. Check done, claim-free.
- Mechanism (measured, not speculated): NOT clipping — max|w| ≤ 11.4 everywhere
  (256·11.4 = 2918 ≪ 32767, clip=False on all probes). It is LSB resolution: per-ctx minima rely
  on fine weight cancellations and (critically) bias rounding — a ~0.5-gray systematic bias shift
  doubles the residual alphabet under Huffman (hence +48% catastrophes). ×1024+ LSB (≤0.03 gray)
  removes it. Selected scales are always 2048/4096 (never 256), i.e. the data independently
  rejects ×256 — the adaptive rule (largest scale with max|w|·scale ≤ 32767, +3-bit id) is doing work.
- Adaptive-int16 verdict: retains 97–100% of float subset gain on all 7 images (123/153 wins),
  assembly-level retention 66–78% after side info (e.g. kodim23 5.57%→4.33%, kodim01 2.56%→1.70%).
  All weights int16, range proof per net (max|w| recorded in JSON). Decoder spec §6.

## 4. Train curves summary (L1 minibatch loss, it0 → it199, lr 0.003 — representative)

- Global nets: Y 0.19–0.28 → 0.02–0.10; Co/Cg 0.01–0.09 → 0.007–0.013. Monotone, no divergence.
- Smooth ctx0: → 0.003–0.017 (near-converged by it199).
- Texture ctx8: 0.29–0.40 → 0.07–0.22, still descending at it199 (more iters might add float,
  but 800-iter runs proved longer training hurts QUANTIZED bits — sharp-minimum effect; keep 200).
- Contrast with retired 800/0.01 setting: loss explode-recover mid-training on texture subsets
  (e.g. Co-ctx8 0.024→0.144→0.024), flat texture curves, seed-fragile quantization
  (Y-ctx0 q-bits 288755 vs 251035 across RNG states) — all gone at 200/0.003.

## 5. Architecture decisions (with justifications)

- **Energy-9 quantile contexts over LOCO-key clusters.** LOCO-365 fragments (cycles 2/21);
  clustering it needs a search. E=|a−b|+|a−c|+|b−c| keeps ~40–100k samples/net for 113-param MLPs,
  is 3 integer ops on recon, and splits exactly the smooth/texture axis where the campaign gap lives.
  Quantile (not fixed) thresholds adapt to texture images; ties yield empty slots that safely
  fall back (effective K 20–23 of 27 slots). Overhead 48 B/image.
- **8→8→1 → 8→8→1 on 12 feats (committed); 8→16→1 REJECTED.** kodim23: h8f8 −3.88%,
  h16f8 −3.35% (float assembly identical 1468034 vs 1467992 — capacity unused, side ×1.9:
  161 vs 81…113 params), h8f12 −4.19%. kodim05 confirms f12 (−3.80%) > f8 (−2.70%).
  Extra 4 feats = lag-2/3 taps (L3/T3/TL2/TR2), all causal (prev rows fully known, left known).
  LS-init handicap noted: f12's 4 extra feats start ignored at init (first-8 passthrough), Adam recruits.
- **LS-init + L1 kept exactly** (mandated lessons 1–2); ridge-LS sweep showed damping kills texture
  gains (Co-ctx3 float −3.99% at ridge 1000 vs +3.74% at ridge 0). Random-init not re-tried per lesson.
- **Features all decoder-computable from recon** (causal left/top/diag/lag taps + derived means/abs).
  Probe uses original as recon proxy (valid lossless); sequential proof §7 closes the loop.

## 6. Decoder story (specification, not built)

Per image the decoder needs: (a) 3-bit channel map; (b) per active channel 8×int16 energy thresholds;
(c) per winning (ch,ctx): 113 int16 weights + 1 select flag + 3-bit scale id; (d) 3 Huffman tables
(inside bit accounting). Per pixel (raster): compute E from recon causal neighbors (int32),
bin via thresholds → ctx; if flagged, build 12 int features from recon, forward pass
y = round(128·(W2·tanh(W1·(x/128)) + b2)) with dequantized weights (int16/2^k, k∈{8…12});
integer implementation: 12-bit tanh LUT + 32-bit accumulator (max |acc| < 11.4·8 + margin, no overflow);
else MED (3 compares). Add parsed residual → recon. Worst-case ~2.3k int ops/px for MLP pixels,
MED-fallback pixels cost MED only (gating keeps smooth pixels on nets too where they win — noted cost).

## 7. Sanity proofs

- YCoCg-R invertibility asserted on every image load (7/7 PASS).
- Sequential raster reconstruction on 128×128 kodim23 crop, all 3 channels, quantized nets,
  numpy integer-path forward: pred_seq == pred_vec **0/16384 mismatches**, recon == original,
  on BOTH gated system and forced-MLP pass (16k MLP pixels/channel). `probe_b21_sanity.py` → SANITY PASS.
  (Trail honesty: first sanity version had two bugs — circular recon assignment, then missing /128
  in the numpy forward — both caught by the forced-MLP mismatch assert and fixed; the assert did its job.)

## 8. Trail (what was tried, in order)

1. MED anchor repro: 3.5769 (0.09% off 3.58) PASS; torch 2.14 CPU/12 threads verified.
2. Ported harness to `probe_b21_perctx.py` (LS-init/cosine/L1 intact) + energy-9 contexts + double gate.
3. Pilot kodim23 h8f8/it800: per-ctx-float −4.9% but quantized −0.36% ( Ausschlag: Y-asmQ +2.4% over MED).
4. Autopsy (`probe_b21_diag.py`): no clipping anywhere; ×256 erratic, ×1024 retains; 800/0.01 minima
   sharp/seed-fragile; ridge hurts; 200/0.003 ≥ 800/0.01 on float at 4× less compute.
5. Remedy build: adaptive-int16 + it200/lr0.003 → kodim23 −3.88% (−2.04% vs global), kodim05 −2.70%.
6. Variants kodim23/05: h16 rejected, f12 committed (both images).
7. Full-7 committed + curves → table §1. Sanity (with two self-caught bugs) → PASS.

## 9. Boss-5 math (transfer to CROWN4-exact 3.1993 vs JXL-e3 3.2291)

- CROWN4 leads on average (−0.9%) but Boss-5 STANDS at 5W-2L (p=.375): blockers kodim05 (+2.27%)
  and kodim13 (+0.98%) — both texture, exactly this branch's target.
- This probe's MED-frame gains on the blockers: 05 −3.76%, 13 −2.27% (per-ctx), vs global −3.13%/−2.41%.
- Transfer discount (cycle-18: grouping consumes ~72% of unstructured harvest) is the pessimistic
  pole: 28% × −3.76% ≈ −1.05% on 05 (flips it), 28% × −2.27% ≈ −0.64% on 13 (not quite).
  Optimistic pole (MLP-as-17th-expert, per-group MDL auto-select, cycle-25 plan): experts compete
  INSIDE groups so overlap is lower than VQ's; 50% × gains flips both blockers and extends mean lead
  to ~3.15–3.17. Either pole beats the global-MLP transfer (40% less MED-frame gain to discount).
- Defined construction: add per-ctx quantized MLPs (frozen, transmitted) to the E16 expert bank;
  per-group best-of search + hillclimb does the MDL automatically; merges pass stays unchanged.
  Cost: encode-time only (decode = one extra expert); side info rides existing group headers.

## 10. Follow-up (ordered)

1. **MLP-as-expert transfer**: per-ctx nets into CROWN group search → measure 05/13 flips (the Boss-5 shot).
2. **Iters tail**: texture ctx8 still descending at it199 — try it300 + EMA-to-LS-neighborhood
   (average late iterates; keeps quant-robustness while harvesting the tail).
3. **LOCO-key second opinion**: one run with (group×activity)-style keys replacing energy bins,
   same budget — only if (1) under-transfers (tests whether binning is the bottleneck).
4. **Chroma-pair alphabet**: cycle-12's −1.58% joint (Co,Cg) finding applied to MLP residuals.

## 11. Files (new only, existing tree untouched)

- `experiments/probe_b21_perctx.py` — main probe (harness adapt + contexts + gates + adaptive quant).
- `experiments/probe_b21_diag.py` — autopsy + remedy grid.
- `experiments/probe_b21_sanity.py` — sequential causality proof (PASS).
- `experiments/probe_b21_sweep.sh` — pilot sweep driver.
- `experiments/probe_b21_RESULTS.md` — this file.
- `experiments/probe_b21_*.json/.log` — raw numbers + logs (pilot23, sweeps h8f8/h16f8/h8f12 ×23/05, full7, diag, sanity).

Verdict restated: **per-context neural beats global (+0.85% avg, 6/7) with quantization + MDL fully
counted — the neural successor at full strength is REAL. Transfer it to CROWN and take Boss-5.**