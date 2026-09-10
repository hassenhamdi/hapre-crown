# probe_b17 — Attack-1 port (per-group RCT + Weighted-form predictor): RESULTS

Campaign: lossless codec vs standing Boss-5 JXL-e3 (3.2291 bpp Kodak-7 avg).
Branch brief: port JXL's per-group RCT selection + Weighted predictor onto our frame.
Code: `experiments/probe_b17_rctw.py` (new file only; nothing existing modified).
Rules kept: numpy+PIL, CPU only; bpp = total_bits/(H·W·3); real heapq Huffman with
16+A·24 table bits per stream; every side byte counted; integer ops; causal recon;
RCT ids transmitted, never inferred; bit-exact RCT round-trips asserted.

## 0. Anchors reproduced (all exact, all decode-verified)

| anchor | avg | note |
|---|---|---|
| A0 MED + global C6-YCoCg-R + order-0 Huffman (this probe) | **3.5782** | vs 3.58 target: −0.05% → **PASS** (±3% rule) |
| A0old same, edge-replicate border (b10 convention) | 3.5769 | border-convention delta +0.04% — negligible; probe uses causal CROWN2-doctrine borders (zero + row0-copies-left + col0-copies-top), fully streaming-decodable |
| CROWN2-exact `csrc/driver_crown2.py` `encode_image`/`decode_image`, re-run here | **3.2686** | 7/7 round-trip PASS; matches cited 3.2686 to 4 decimals |
| JXL-e3 `cjxl -d 0 -e 3`, re-verified here | **3.2291** | exact match to takedown table, per-image values identical |
| JXL e3-C27 `cjxl -d 0 -e 3 -C 27`, re-verified here | **3.2084** | matches cited 3.2083 (moving-bar honesty: §5) |

CROWN2-measured per image: 01: 3.3789 · 02: 3.0274 · 05: 3.6594 · 07: 2.7556 ·
13: 4.0245 · 19: 3.2207 · 23: 2.8138. CROWN2 vs e3: 1W-6L, p≈.16 n.s. (reproduces
takedown's p=.156) — Boss-5 STANDS at +1.22% (need −0.0395 bpp from 3.2686).

## 1. Master per-image table (exact bpp, side included)

RCT bank (B=8, 3-bit ids, JXL catalogue `7·perm+t`): C6 C27 C13 C12 C0 C20 C34 C3.
Configs: (a0)=global best-of-8+3b · (a)=per-block best-of-8 (S∈{16,32,64} best+2b,
3b/block) · (b)=Weighted LS-4-tap on C6 planes (G∈{32,64} best+1b, 1b flag/group/ch,
20b per used group) · (a+b)=(a)-mixed planes + Weighted · (a0+b)=global-best RCT
(homogeneous) + Weighted.

| img | A0 | (a0) | (a) | (b) | (a+b) | (a0+b) | CROWN2 | JXL-e3 |
|---|---|---|---|---|---|---|---|---|
| kodim01 | 3.6179 | 3.5521(C27) | 3.5627(S64) | 3.5850(G32) | 3.5791(G32) | 3.5605(G32) | 3.3789 | 3.3593 |
| kodim02 | 3.3255 | 3.2829(C12) | 3.3185(S64) | 3.2692(G32) | 3.3284(G32) | 3.2612(G32) | 3.0274 | 3.0611 |
| kodim05 | 4.0119 | 3.9855(C27) | 3.9903(S64) | 3.9367(G32) | 3.9590(G32) | 3.9321(G32) | 3.6594 | 3.5120 |
| kodim07 | 3.1529 | 3.0754(C27) | 3.0831(S64) | 3.0899(G32) | 3.0657(G32) | 3.0420(G32) | 2.7556 | 2.7310 |
| kodim13 | 4.2378 | 4.1983(C27) | 4.1945(S64) | 4.1688(G32) | 4.1579(G32) | 4.1348(G32) | 4.0245 | 3.9141 |
| kodim19 | 3.5133 | 3.4757(C27) | 3.4765(S64) | 3.3847(G32) | 3.4568(G32) | 3.4250(G32) | 3.2207 | 3.2154 |
| kodim23 | 3.1879 | 3.1234(C27) | 3.1331(S64) | 3.0737(G32) | 3.1049(G32) | 3.0641(G32) | 2.8138 | 2.8110 |
| **AVG** | **3.5782** | **3.5276** | **3.5370** | **3.5012** | **3.5217** | **3.4885** | **3.2686** | **3.2291** |

Average deltas vs A0 (exact Wilcoxon, n=7): (a0) **−1.413%** 7/7 p=.016 ·
(a) **−1.152%** 7/7 p=.016 · (b) **−2.153%** 7/7 p=.016 · (a+b) **−1.579%** 6W-1L
(only kodim02 loses, +0.003) p≈.031 · (a0+b) **−2.505%** 7/7 p=.016.
Vs measured CROWN2-exact: (a0) +7.92% · (a) +8.21% · (b) +7.12% · (a+b) +7.74% ·
(a0+b) +6.73%. Vs JXL-e3: (a0+b) +8.03% (0W-7L — different frame, expected).

## 2. Mechanism

**M1. The RCT gain is the dictionary, not the search (JXL-W3 mirrored).**
Global best-of-8 picks C27 on 6/7 and C12 on kodim02 — the same permuted-YCoCg
entries that beat JXL's own search (C27/C13/C12/C20 family). Our current global C6
is simply the wrong permutation: RBG-ordered YCoCg (C27) decorrelates Kodak better
under MED (−1.41% avg, up to −2.4% on kodim07/23). Zero decoder risk (3 bits/image).

**M2. Per-block RCT is DEAD: global wins 6/7, and size-monotonicity is perfect.**
Exact bits without selectors — global < S64 < S32 < S16 on 6/7 images
(only kodim13 prefers S64 by 0.004); with honest 3b/block side, (a) loses to (a0)
by +0.26pp avg. Block-id histograms concentrate on C27/C12/C20/C3 (C13/C34/C0/C6
near-zero). This is the fragmentation doctrine again (JXL-W1 group-256, our
WIDE-K/merges): adaptation regions must be LARGE; per-32×32 RCT search spends
1152 side bits to re-discover the global answer plus boundary-mixing damage.

**M3. Weighted-form is a REAL second weapon (−2.15%, 7/7).**
Per-64×64-group closed-form LS 4-tap on causal {L,T,TL,TR} (JXL Weighted
neighbourhood), weights ×16 quantized to [−16,15] (5b each, /16, integer
MAC+shift), Laplacian-MDL gate (keep iff N·log2(b_med/b_w) > 21 = flag+weights).
G32 beats G64 on 7/7 (prediction wants LOCAL stationarity — opposite granularity
to M2's entropy-group doctrine; groups serve different masters). Use-rate 35–77%
of groups; side only 0.002–0.015 bpp. Fitted kernels are edge-sharpeners
(mean |w|/16 ≈ [0.65,0.31,0.18,0.21], TL weight typically negative — matches the
5-tap float oracle which puts −0.12..−0.43 on TL). Chroma Cg adopts most
(70–79/96 groups on 07/05); Y least. Biggest wins on textured 19 (−3.7%) and
23 (−3.6%) — complementary to CROWN2's smooth-field strength.

**M4. Homogeneity rule: RCT-mixing poisons predictors — (a0+b) > (b) > (a+b).**
(a+b) refit on per-block-mixed planes loses to (b) on C6 planes by +0.6pp, while
(a0+b) (global RCT, homogeneous planes) is best everywhere (7/7, −2.51%).
Boundary pixels where neighbours come from a different RCT break both MED and
the LS fit. Port rule: exactly ONE RCT per image (or per large homogeneous
region ≥256px), never per-small-block mixing.

**M5. Headroom confirmed by 5-tap oracle (kodim07, float, unquantized).**
LS {L,T,TL,TR,MED} weights: Y [0.34,0.10,−0.12,0.14,**0.54**],
Co [0.66,0.27,−0.43,0.24,0.26], Cg [0.64,0.10,−0.30,0.34,0.22];
MSE −8%/−27%/−28% vs MED; L1 −6.3%/−8.3% on Co/Cg (Y-L1 still MED-best).
JXL's Weighted indeed blends MED-like terms — adding a MED tap (5×5b=25b/group)
is quantified follow-up #2.

## 3. Trail (reproducibility + honesty log)

- RCT round-trips: ALL-8 bank × 7 Kodak + saturated edge-case (8×8 checker
  255/0 + white row): PASS. Residual alphabet |r|≤1024 asserted (never clip);
  Kraft asserted per Huffman table.
- Winning config (a0+b) full decode proof on kodim07: literal MSB-first bitstream
  encode→decode (445601 payload bytes; +tables/side = reported 3.0420 exactly) +
  sequential causal inverse-predictor (MED-or-Weighted per transmitted flags,
  recon only) + global inverse RCT → bit-exact original: PASS.
- Bugs found and fixed (auditable in-file): (i) t5/t6 inverse formula errors
  caught by round-trip asserts before any measurement; (ii) a weight-scale bug
  (LS plain-unit weights rounded without the ×16 pre-scale → all-zero weights →
  first (b) run measured +0.007%, reported here as INVALID, not hidden).
  Re-ran everything after the fix; numbers above are post-fix.
- Side ledger: (a0) 3b/img; (a) 3b/block + 2b size-sel (S64: 290b = 0.00025 bpp);
  (b)/(a0+b) 1b flag per group-channel + 20b per used group + 1b size-sel +
  3b RCT id for (a0+b); 3 Huffman tables 16+A·24 each (dominant side, as expected
  for order-0 global tables).
- Timings: ~9 s/image (LS fits dominate); CROWN2 anchor ~100 s/image.

## 4. Verdict — does Attack-1 close Boss-5 math?

Boss-5 math: CROWN2-exact 3.2686 → JXL-e3 3.2291 needs **−0.0395 bpp (−1.22%)**.
Attack-1 measured on the MED frame: RCT-perm −0.0505 bpp (−1.41%), Weighted
−0.0770 bpp (−2.15%), combined (a0+b) −0.0897 bpp (−2.51%) — **2.3× the required
absolute saving, but on the weaker frame**, so transfer must be discounted for
what CROWN2 already harvests (E16 experts, per-group backends, WIDE-K groups):
- RCT-perm (global C27 swap, 3 bits): front-end, backend-orthogonal, E16
  re-selects under its MDL gate → est. 60–90% transfers ≈ **−0.8..−1.3%**.
- Weighted as gated 17th E16 expert (same groups, same gate, +1 expert-id value):
  overlaps GAP16/directional experts → est. **−0.4..−0.9%** over E16-best.
- Central ≈ −1.5..−1.7% COVERS −1.22%; pessimistic ≈ −1.2% borderline.
- **Moving-bar honesty**: JXL answers with one flag (e3-C27 3.2084, verified) →
  bar becomes −1.83% from CROWN2-exact. Since C27 is our free idea too (adopting
  it neutralizes their move), the race position is unchanged *if we port first*;
  vs the moving bar Attack-1 alone is likely 60–100% of the way, not a KO.
- NOT portable (measured dead): per-block RCT (M2), per-block mixing (M4).

**Verdict: Attack-1 is a STRONG PARTIAL — two significant, decoder-safe, cheap
weapons (both p≤.031, 6-7/7), with a concrete C-port path, but Boss-5 math
closes only under central-transfer + static-bar assumptions. Pair with Attack-2
(cross-channel props, the takedown's insurance) for margin vs the C27 bar.**

## 5. Follow-ups (ranked)

1. **C-port, in this order**: (i) global RCT selector over {C6,C27,C12} (2 bits,
   ~1 day + golden) — expected ≈ −1% on the CROWN2 arm alone; (ii) Weighted-LS
   as gated 17th E16 expert, G32 groups, 5b weights, existing MDL gate (~1 week).
   Re-measure Boss-5 on the exact arm. Explicitly DO NOT port per-block RCT.
2. 5-tap Weighted+MED blend (M5 oracle: −6..8% L1 on chroma pre-quantization).
3. Attack-2 (co-located-Y property groups) — needed if the bar moves to e3-C27.
4. Reconcile group granularities: entropy groups LARGE (M2), prediction groups
   SMALL (M3) — CROWN2's single grouping currently serves both; split them.
