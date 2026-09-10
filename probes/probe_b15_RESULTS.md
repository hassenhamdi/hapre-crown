# probe_b15 RESULTS — transform-bitplane arc (LeGall 5/3 + context coding)

Code: `experiments/probe_b15_wavelet.py` (numpy+PIL only, CPU). All configs share the
YCoCg-R front-end (exact inverse asserted) + reversible LeGall 5/3 integer wavelet via
lifting with whole-sample symmetric extension (exact inverse asserted per channel per
image, levels 1–3). UNIT: bpp = total_bits/(H·W·3). Exact counting everywhere:
heapq Huffman data bits + `16+A·24` per transmitted table (A = distinct symbols present;
A==1 → 0 data bits with table still counted; empty streams cost 0 and decoder derives
emptiness from already-decoded state). No torch, no training, no existing files touched.

## Configs

| cfg | levels | LL | details |
|---|---|---|---|
| A0 (anchor) | 2 | order-0 Huffman | order-0 Huffman per subband |
| A1 | 2 | MED-DPCM + Huffman | order-0 Huffman per subband |
| B8 / B16 | 2 | MED-DPCM + Huffman | bitplane loop MSB→LSB; significance bit ctx = parent-sig(1b) + 8-nbr higher-plane-sig bucket(2b) [+ dense-plane class(1b)]; nibble-packed Huffman per (subband,ctx); sign + refinement raw |
| B16L3 | 3 | same as B16 | same as B16 |
| C (best) | 2 | MED-DPCM + Huffman | **raster-causal conditioned Huffman on signed values**: ctx = neighbor-activity bucket(4: \|l\|+\|t\|+\|tl\|+\|tr\|) × parent-magnitude bucket(3) = 12 groups; position-driven decode + 12b presence mask |
| CX / CX-L3 | 2 / 3 | same | C + Y-activity bit (24 groups) |
| C24 | 2 | same | C with 6×5=30 groups (fragmentation ablation) |
| C-L1 | 1 | same | C with 1 level (depth ablation) |

Normalization note (mandatory #1): the 5/3 lifting has DC gain exactly 1 — no core
inflation by construction (unlike the failed unnormalized-DCT probe). Per-subband
dynamic-range differences (LL maxabs ~280 vs Cg-HH maxabs ~13) are absorbed by
separate per-subband models + per-subband bitplane depths, i.e. scaling is accounted
in the code lengths, not in the coefficients. Entropy audit on kodim07 confirms sane
scaling: Y-LL 7.12b, Y details 3.5–5.0b, Co details 1.8–3.8b, Cg details 1.6–3.3b,
Huffman within 1–4% above each marginal entropy (all ≥ entropy ⇒ no undercounting).

## Per-image results (bpp)

| img | A0 | A1 | B8 | B16 | **C** | CX | C24 | C-L1 | JXL-e3 |
|---|---|---|---|---|---|---|---|---|---|
| kodim01 | 3.6064 | 3.5672 | 4.3206 | 4.3290 | **3.5372** | 3.5466 | 3.5648 | 3.6126 | 3.3593 |
| kodim02 | 3.1932 | 3.1582 | 4.0391 | 4.0464 | **3.1314** | 3.1427 | 3.1585 | 3.2077 | 3.0611 |
| kodim05 | 3.7612 | 3.7185 | 4.5443 | 4.5531 | **3.6826** | 3.6973 | 3.7169 | 3.6962 | 3.5120 |
| kodim07 | 3.0782 | 3.0129 | 3.8436 | 3.8519 | **2.9371** | 2.9477 | 2.9625 | 3.0346 | 2.7310 |
| kodim13 | 4.0673 | 4.0248 | 4.8154 | 4.8239 | **4.0253** | 4.0404 | 4.0881 | 4.0194 | 3.9141 |
| kodim19 | 3.5337 | 3.4432 | 4.3144 | 4.3244 | **3.3939** | 3.4074 | 3.4165 | 3.4681 | 3.2154 |
| kodim23 | 3.1061 | 2.9528 | 3.9391 | 3.9467 | **2.9088** | 2.9211 | 2.9354 | 2.9783 | 2.8110 |
| **avg** | 3.4780 | 3.4111 | 4.2595 | 4.2679 | **3.3737** | 3.3862 | 3.4061 | 3.4310 | 3.2291 |

(B16L3 avg 4.3034, CX-L3 avg 3.4262 — deeper pyramids hurt both families; omitted
per-image for brevity, full numbers in console logs.)

## Deltas — best transform build (C, 3.3737 avg)

| img | C − JXL-e3 (abs) | C − JXL-e3 (%) |
|---|---|---|
| 01 | +0.1779 | +5.30% |
| 02 | +0.0703 | +2.30% |
| 05 | +0.1706 | +4.86% |
| 07 | +0.2061 | +7.55% |
| 13 | +0.1112 | +2.84% |
| 19 | +0.1785 | +5.55% |
| 23 | +0.0978 | +3.48% |
| **avg** | **+0.1446 (+4.48%), 0W-7L** | |

- vs classical ceiling 3.2515: **+0.1222 (+3.76%)**. Verdict: transform arc does NOT beat it.
- Boss-5 math: tying JXL-e3 needs **−4.3%** from C; final boss JXL-e9 (3.03) needs **−10.2%**.
  Neither is reachable inside the measured family gradients (best moves: A0→A1 −1.9%,
  A1→C −1.1%, finer groups +0.9% wrong way, 3rd level +1.6% wrong way).

## Anchor audit (mandatory sanity point)

A0 averaged **3.4780**, not inside the suggested 4.5–5.0 band. This is a calibration
miss in the brief, not a transform bug — verified by the entropy audit above (every
stream's Huffman length sits 1–4% above its marginal entropy, none below, so no
undercounting; inverse asserts pass). The band presumably assumed RGB-domain or
full-frame order-0: with YCoCg-R the chroma detail marginals collapse (Cg-L1 detail
entropies 1.6–2.2b) and per-subband tables adapt to each band's range, so 3.48 is the
honest order-0 number. Transform scaling is healthy (no inflation: maxabs bounded,
LL ≈ image entropy, details small).

## Mechanism analysis

1. **Bitplanes lose to joint coding on dense bands (+26% B vs A1).** Natural 5/3 detail
   bands are NOT sparse: nonzero fractions 0.48–0.86 (sensor texture/noise everywhere).
   Zerotree-style significance coding harvests sparsity; here there is almost none, so
   B pays the full representation overhead: a sig-0 "trail" at every higher plane
   (~3.0 sig bits/coeff measured on Y-L1 details vs 3.7 total order-0), a flat raw
   sign bit (1.0) per nonzero with no shaping, and raw refinement. Order-0 Huffman
   codes (magnitude,sign) jointly in one lookup and wins by construction. Lesson:
   bitplane machinery without EBCOT's three missing pieces (current-plane feedback
   contexts, sign/refinement contexts, run-coded cleanup passes) is strictly worse
   than a single joint table. B8 vs B16 identical (+0.2%) ⇒ the plane-class bit is
   dead weight; B16L3 worse ⇒ depth adds tables, not compaction.
2. **The 5/3 already decorrelates; conditioning harvest is thin (−1.1% C vs A1).**
   Neighbor-activity × parent-magnitude conditioning (C) is the correct causal shape
   and beats order-0 on all 7 images, but only by ~0.04 bpp: after lifting there is
   little conditional structure left for 12 Huffman groups to find. Finer groups
   (C24, +0.9%) and cross-channel activity (CX, +0.4%) both fragment tables past the
   MDL break-even — the same fragmentation curve as cycle-5/19, reproduced in the
   transform domain. 1 level (C-L1, +1.7%) under-compacts LL; 3 levels over-fragments.
3. **Where the gap lives.** C's deficit vs JXL-e3 concentrates on easy/smooth images
   (07: +7.5%, 19: +5.6%, 01: +5.3%) — exactly where pixel-domain predictors (MED/GAP,
   MA-trees) exploit smooth gradients that wavelets scatter across scales. Texture
   images (13: +2.8%, 02: +2.3%) are closer. The transform destroys the smooth-field
   predictability that the classical arm's 3.2515 is built on, and its own
   conditioning cannot buy it back.

## Trail

- Built lifting 5/3 fwd/inv (vectorized predict/update, symmetric mirrors); self-test
  PASS on even+odd lengths incl. n=1,2,3 and 2D levels 1–3 on 5 shapes incl. odd.
- Smoke on kodim07 exposed anchor168 B-vs-A0 inversion → entropy audit proved A0
  honest (band miscalibrated) and B structurally lossy → added C/CX per brief's
  "adapt freely" (raster-causal conditioned signed Huffman, presence-masked).
- Full-7 run: C best (3.3737); ablations C24/C-L1/L3 all negative ⇒ family exhausted.

## Verdict

**Transform-bitplane arc (this build) NEGATIVE: best 3.3737 vs 3.2515 ceiling (+3.8%),
0W-7L vs JXL-e3 (+4.5%). Boss 5 STANDS; Boss-5 math needs −4.3% (e3) / −10.2% (e9)
with all measured gradients ≤1.9% and mostly exhausted.** The failure is structural
(dense-band bitplane overhead + smooth-field predictability loss), not implementational
— same species as the three logged transform failures, now with exact numbers.

## Follow-up (ordered)

1. **Retire pure bitplane-for-density work** (EZW/SPIHT shapes assume sparsity Kodak
   doesn't have); any revival needs EBCOT's full trio (feedback + sign/ref ctx + runs).
2. **Hybrid arc (untested, recommended): LL + low-frequency via pixel-domain CROWN
   (keeps smooth-field prediction), finest-HH-only via conditioned Huffman** — concedes
   the bands where wavelets win (sparse chroma HF) without surrendering Y-LL. Est.
   ceiling ≈ blend of arms, +0.5–1% at best; honest expectation: still short of e3.
3. **lf-prediction (HPCM lesson) stays the last structural weapon**: predict HF from
   decoded LF with integer weights instead of conditioning entropy on it — attacks the
   smooth-field loss directly, unlike everything in this probe.
4. Consolidation unchanged: classical exact 3.2515 remains champion; Boss-5/9 need
   learned entropy (CALLIC/HPAC lineage) or MA-tree splits, not transforms.

Bit-exactness: YCoCg-R round-trip + 5/3 inverse == original asserted for every
image × every level (7/7 PASS, L=1,2,3 — any failure aborts with AssertionError;
all runs completed). Decoder-safety: S1 channels Y→Co→Cg fixed; S2 LL_L then
levels L..1 triplets; S3a bitplanes MSB→LSB raster significance → raster signs →
raster refinement (B), S3b single raster pass left→right/top→bottom with
left/TL/top/TR + coarser-level-parent + full-Y contexts (C/CX). All contexts are
decodable-before-use by construction; every stream length is geometry- or
state-derivable (Huffman tables + maxbit nibbles + 12/24b presence masks counted).
