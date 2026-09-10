# probe_b11 RESULTS — does quad magnitude-VQ stack with CROWN grouping?

Transfer question from b10: Q-T4 2x2 magnitude-VQ won **−1.84%** gated vs
MED+order-0. CROWN-huff wins **−6.58%** vs MED via LOCO-365 clustered
grouping. Do the gains stack or overlap?

Files (new only, existing untouched):
`experiments/probe_b11_vqstack.py` (v1: TL-quad-rooted port),
`experiments/probe_b11_pairvq.py` (v2: pair-rooted + pure homogeneous forms),
`experiments/probe_b11_proofs.py` (literal round-trip of winner),
`experiments/probe_b11_log.txt`, `probe_b11b_log.txt`, `probe_b11c_log.txt`.
Proven modules imported, not reinvented: `probe_b4_b.predictors`,
`probe_b4_c.loco_ctx365`, `probe_b4_f.MOE6`, `probe_b4_g.prep`,
`probe_b4_j.auto_groups_exact` (+ replica check vs `eval_final`),
`probe_b4_h.rans_stream_bits` (real `libhapre.so` M=14 codec).

## 1. Baseline reproduced

(a) CROWN-huff pixel-grouped replica is **bit-identical** to
`probe_b4_j.eval_final(path,"huff")` on all 7 images (asserted <1e-9):
**AVG 3.3429** (01:3.4340, 02:3.0931, 05:3.7394, 07:2.8630, 13:4.1010,
19:3.2758, 23:2.8940). In-harness MED order-0 anchor: **3.5782**
(0/left/top convention) ⇒ soloCROWN = −6.58% ≈ brief's −6.6% ✓.

## 2. Per-image results (bpp = total_bits/(H·W·3), exact ledgers)

| img | MED | (a) CROWN | (bqp)+hVQ ★ | (cqp)+rANS ★ | JXL-e3 |
|---|---|---|---|---|---|
| kodim01 | 3.6179 | 3.4340 | 3.4124 (−0.63%) | 3.3850 (+0.74% vs JXL) | 3.36 |
| kodim02 | 3.3256 | 3.0931 | 3.0755 (−0.57%) | 3.0482 (−0.39% WIN) | 3.06 |
| kodim05 | 4.0119 | 3.7394 | 3.7278 (−0.31%) | 3.7103 (+5.71%) | 3.51 |
| kodim07 | 3.1530 | 2.8630 | 2.8309 (−1.12%) | 2.7890 (+2.16%) | 2.73 |
| kodim13 | 4.2378 | 4.1010 | 4.0900 (−0.27%) | 4.0783 (+4.30%) | 3.91 |
| kodim19 | 3.5134 | 3.2758 | 3.2655 (−0.31%) | 3.2444 (+0.76%) | 3.22 |
| kodim23 | 3.1880 | 2.8940 | 2.8763 (−0.61%) | 2.8537 (+1.56%) | 2.81 |
| **AVG** | **3.5782** | **3.3429** | **3.3255 (−0.521%)** | **3.3013 (−1.245%)** | **3.2286** |

% in (bqp) column vs (a); (cqp) column shows gap vs JXL-e3.
(bqp) beats (a) **7/7** (Wilcoxon W=0, p≈.016). (cqp) beats JXL 1/7.

Rooted (port-faithful) forms — regroup cost dominates:

| img | (a2)TLquad | (b)TL+VQ | (a2p)pairL | (bp)pair+VQ | (cp)pair+rANS |
|---|---|---|---|---|---|
| 01 | 3.5411 | 3.5394 | 3.4891 | 3.4773 | 3.4549 |
| 02 | 3.2274 | 3.2248 | 3.1657 | 3.1494 | 3.1332 |
| 05 | 3.8755 | 3.8757 | 3.8074 | 3.7997 | 3.7915 |
| 07 | 3.0173 | 2.9947 | 2.9431 | 2.9118 | 2.8898 |
| 13 | 4.1972 | 4.1973 | 4.1554 | 4.1530 | 4.1469 |
| 19 | 3.3746 | 3.3747 | 3.3406 | 3.3383 | 3.3214 |
| 23 | 3.0676 | 3.0631 | 2.9962 | 2.9833 | 2.9712 |
| **AVG** | **3.4715 (+3.85%)** | **3.4671 (+3.72%)** | **3.4139 (+2.12%)** | **3.4018 (+1.76%)** | **3.3870 (+1.32%)** |

(% vs (a).) Quad-homog form (bq/cq): 3.3348 (−0.24%) / 3.3087 (−1.02%).

## 3. Overlap analysis (brief formula: overlap = 1 − combined/(soloVQ+soloCROWN))

soloVQ = 1.84%, soloCROWN = 6.58%, naive sum = 8.42%.

| config | combined vs MED | overlap fraction | reading |
|---|---|---|---|
| (bqp) pure hpair-VQ | 7.06% | **+0.16** | 84% of naive sum retained |
| (cqp) + rANS | 7.74% | **+0.08** | 92% retained |
| (bq) pure hquad-VQ | 6.80% | +0.19 | coverage-limited (5% px) |
| (bp) pair-rooted | 4.93% | +0.41 | confounded by +2.1% regroup loss |
| (b) TL-rooted | 3.11% | +0.63 | regroup destruction, not VQ overlap |

Marginal lens (same data, VQ's own effect): solo −1.84% → marginal on
CROWN arm −0.52% (huff). CROWN grouping consumes ~72% of what VQ harvests;
the residual within-group magnitude-TC pays only where
coverage × saving > extra tables (VQ wins 70/410 groups hpair, 21 hquad,
76/410 pair-rooted; net savings 0.0174/0.0081/0.0122 bpp).
Both lenses agree: **stacking is real but partial — VQ is a minor term.**

Mechanism diagnostics: homogeneous-pair pixel coverage 20.6% (= pair-left
homog rate), escape 7.1% at T8; homog-quad coverage only 5.0%
(why quads fail: fine LOCO clusters rarely cover 2x2); pair-root choices
S/T4/T8 = 334/55/21; GAP/DG-rooted quad share 19.5% with ≈0 at-risk saving
(port exclusion would cost nothing — VQ never fires there anyway).

## 4. Decoder-safety statement

- **(a2p)/(bp)/(cp) pair-left-rooted: streaming-causal, C-portable as
  specified.** Pair-left group is known at the left pixel (LOCO key needs
  only left/top/prev-row recon); the joint symbol is consumed in pair-raster
  order; both residuals use the pair-group predictor whose neighborhood
  (a,b,c,d) is fully decoded (d is prev-row — never future in horizontal
  pairing), so ALL 6 predictors incl. GAP/DG are safe; signs/tail follow in
  pair order; all side info static. No original-pixel peeks.
- **(bq)/(bqp) homogeneous forms: offline-decodable** (framing purely
  recon-derived, static canonical tables). Proofs: YCoCg-R invert 7/7;
  sym→mag mapping on all 70 VQ-winning groups × 7 images PASS; **literal
  bitstream→residuals→channels decode (bqp) kodim07 all channels PASS**
  (62 streams, payload 3288588b ≤ ledger 3334476b, diff = tables+flags);
  (cqp) rANS: EVERY stream through the real C `rans_encode`+`rans_decode`,
  asserted during measurement. Streaming-port caveat: homogeneous framing
  has a joint-symbol↔framing circularity (framing needs full-quad groups;
  BR key needs same-row-future d under quad-raster decode), so a C port
  needs framing bits (~0.25bpp, kills it) or the pair-rooted form (causal
  but +2.1% regroup). Hence (bqp)/(cqp) are ledger-level — same evidence
  bar as the CROWN-rans Boss-2/3 KOs (probe-level, C build pending).
- (a) replica bit-exact vs b4_j; Kraft=1 asserted for every Huffman table
  built (via `canon_codes` in proofs; measurement uses the same heapq
  construction as the campaign).

## 5. Trail

v1 TL-quad-root: regroup shock +3.85% (only 5% quads homogeneous; forcing 4
pixels onto TL predictor destroys per-pixel adaptation) → VQ gate fires
8/410 groups (+0.0045bpp): cannot fairly test stacking; kept as negative
control. v2 pair-left-root: regroup +2.12%, VQ wins 76 groups (−0.35% vs own
scalar base, 7/7 — VQ works everywhere) but net +1.76% vs (a). v2 pure
homogeneous forms keep pixel-CROWN intact and add VQ only → (bq) −0.24%,
(bqp) −0.52%, both 7/7; rANS backends (cq) −1.02%, (cqp) −1.25%.

## 6. Verdict

**VQ stacks (pure forms win 7/7, p≈.016; overlap 0.16 huff / 0.08 rANS —
84–92% of the naive sum is retained), but the absolute effect (−0.52%
huff, −1.25% w/ rANS) does NOT change Boss-5 math: best (cqp) 3.3013 vs
JXL-e3 3.2286 (+2.25%, 1W-6L), and behind already-built CROWN2-exact
3.2686 (E16 experts).** Grouping granularity dominates block-joint coding:
any block framing that disturbs per-pixel LOCO adaptation (TL +3.85%,
pair-left +2.12%) loses more than VQ can recover. VQ is a real minor
stacking term, not a Boss-5 weapon.

## 7. Follow-up

1. VQ on the CROWN2-E16 arm (finer groups ⇒ less residual TC + more
   fragmentation; expect ≤−0.3%, cheap to test, low priority).
2. Per-channel T / sparse-tail (Rice) tail to push the T frontier — only if
   (1) shows headroom.
3. Do NOT revisit: quad-rooted ports (dead: +3.85%), larger unfactorized
   alphabets (fragmentation wall confirmed at group level too), joint sign
   coding (b10 verdict stands).
