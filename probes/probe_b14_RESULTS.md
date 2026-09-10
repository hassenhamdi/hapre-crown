# probe_b14 RESULTS — DECISIVE combined-ceiling probe: all proven classical mechanisms stacked

**Campaign question:** what is the COMBINED ceiling of all proven classical mechanisms — and does it reach JXL-e3 (3.2291 avg)?
**Answer: NO. Combined ceiling = 3.2515 avg (+0.69% over JXL-e3, 3W-4L). Boss-5 STANDS. Classical arc declared exhausted.**

Harness: `experiments/probe_b14_stack.py` (numpy+PIL+ctypes only, CPU, no torch, ~38 min all-7).
Summary: `experiments/probe_b14_summary.json`. New files only (`probe_b14_*`); nothing existing modified.
Proven modules reused by import: `probe_b4_a` (YCoCg-R/stream_bits/IMAGES/HEADER), `probe_b4_b` (nbhd/predictors),
`probe_b4_c` (loco_ctx365), `probe_b5_c` (predictors_X/prepX/E16/MOE6), `probe_b5_d` (golomb_best),
`probe_b4_h` (rans_stream_bits, exact C rANS M=14 + asserted decodes), `probe_b6_ma` (build_features/leaf costs/thresholds),
`dp_refine` (hill-climb refine, from csrc). Tree growth is a faithful explicit-topology port of
`probe_b6_b.grow_tree_deep` (fidelity-verified: identical leaf multisets on kodim07 Y+Co, §7).

UNIT RULE throughout: `bpp = total_bits/(H*W*3)`. Exact counting everywhere: real heapq Huffman data +
`16+A*24` tables, exact Golomb data + 4b k, measured `rans_encode` payload + `16+A*32` tables
(probe frame, same as b4/b5/b13), every map/flag/selector/header bit (§5).

## 1. Anchors (exact, this harness) — PASS within ±3%

| image | ANCH-H repro (expect CROWN-huff ≈3.343) | ANCH-R repro (expect CROWN-rans ≈3.312) | JXL-e3 ref |
|---|---|---|---|
| kodim01.png | 3.4172 | 3.3809 | 3.3593 |
| kodim02.png | 3.0700 | 3.0269 | 3.0611 |
| kodim05.png | 3.7247 | 3.6829 | 3.5120 |
| kodim07.png | 2.8431 | 2.7737 | 2.7310 |
| kodim13.png | 4.0848 | 4.0419 | 3.9141 |
| kodim19.png | 3.2599 | 3.2262 | 3.2154 |
| kodim23.png | 2.8740 | 2.8264 | 2.8110 |
| **AVG** | **3.3248 (−0.57% vs 3.3429 pin)** | **3.2811 (−0.94% vs 3.3124 pin)** | **3.2291** |

ANCH-H = Q/E6/KSET2+refine/Huffman-only (≈ b13(a) 3.3256, reproduces to 0.001). ANCH-R = same + H/G/R
per-group choice (beats the b4 all-R pin via refine + Golomb option — honest framing difference, stated).
Both PASS ±3%. MED order-0 control (b4 frame): 3.5782.

## 2. Full-stack result (all 7 components, per-image MDL-gated)

| image | FULL | JXL-e3 | Δ (FULL−e3) | W/L | path | K (Y,Co,Cg) | groups→clusters | backends H/G/R |
|---|---|---|---|---|---|---|---|---|
| kodim01.png | 3.3602 | 3.3593 | +0.0009 | L | Q | 64,64,64 | 119→70 | 4/41/25 |
| kodim02.png | 3.0041 | 3.0611 | −0.0570 | **W** | Q | 27,48,64 | 89→44 | 1/13/30 |
| kodim05.png | 3.6483 | 3.5120 | +0.1363 | L | Q | 64,64,64 | 139→84 | 0/65/19 |
| kodim07.png | 2.7334 | 2.7310 | +0.0024 | L | Q | 64,64,64 | 112→55 | 0/27/28 |
| kodim13.png | 4.0163 | 3.9141 | +0.1022 | L | Q | 64,64,64 | 139→78 | 0/63/15 |
| kodim19.png | 3.2036 | 3.2154 | −0.0118 | **W** | Q | 4,64,64 | 70→35 | 2/7/26 |
| kodim23.png | 2.7946 | 2.8110 | −0.0164 | **W** | Q | 64,64,64 | 115→49 | 0/23/26 |
| **AVG** | **3.2515** | **3.2291** | **+0.0224 (+0.69%)** | **3W-4L** | Q 7/7 | — | 773→415 | 7/239/169 |

2-way path-selection margins (Q vs runner-up T): 01:+0.017, 02:+0.002, 05:+0.059, 07:+0.018, 13:+0.041,
19:+0.006, 23:+0.022 (Q won 7/7; D never closer than +0.04). Y-splits adopted/image: 8,6,10,9,3,4,8
(~10% of chroma atoms). Merges accepted 7/7 (415 final clusters from 458 pre-merge cells; map ~450–1050 b/image).

**Boss-5 verdict: STANDS.** Avg fails (3.2515 > 3.2291); majority-wins fails (3W-4L; Wilcoxon W=12, n.s.).
Stack progression: MED-order-0 3.5782 → ANCH-H 3.3248 (−7.1%) → ANCH-R 3.2811 (−1.3%) → FULL 3.2515 (−0.9%).
FULL beats CROWN2-exact 3.2686 by −0.5% (merges + WIDE + splits on top of the CROWN2 arm).

## 3. Ablation: full-stack vs leave-one-out (per-image bpp)

| image | FULL | LOO1 noLOCO | LOO2 E6 | LOO3 H-only | LOO4 noMerge | LOO5 noTree | LOO6 noGrid | LOO7 noSplit |
|---|---|---|---|---|---|---|---|---|
| kodim01 | 3.3602 | 3.3838 | 3.3635 | 3.4004 | 3.3708 | 3.3602 | 3.3602 | 3.3608 |
| kodim02 | 3.0041 | 3.0202 | 3.0219 | 3.0486 | 3.0172 | 3.0041 | 3.0041 | 3.0043 |
| kodim05 | 3.6483 | 3.7174 | 3.6550 | 3.7011 | 3.6631 | 3.6483 | 3.6483 | 3.6471 |
| kodim07 | 2.7334 | 2.7657 | 2.7417 | 2.8137 | 2.7501 | 2.7334 | 2.7334 | 2.7358 |
| kodim13 | 4.0163 | 4.0690 | 4.0229 | 4.0562 | 4.0359 | 4.0163 | 4.0163 | 4.0157 |
| kodim19 | 3.2036 | 3.2148 | 3.2109 | 3.2404 | 3.2129 | 3.2036 | 3.2036 | 3.2036 |
| kodim23 | 2.7946 | 2.8221 | 2.7999 | 2.8485 | 2.8136 | 2.7946 | 2.7946 | 2.7957 |
| **AVG** | **3.2515** | **3.2847** | **3.2594** | **3.3013** | **3.2662** | **3.2515** | **3.2515** | **3.2519** |
| **marginal** | — | **+0.0332 (+1.02%)** | **+0.0079 (+0.24%)** | **+0.0498 (+1.53%)** | **+0.0147 (+0.45%)** | **+0.0000** | **+0.0000** | **+0.0004 (+0.01%)** |

LOO path fallbacks: LOO1 → T on 6/7, D on kodim13; LOO2 → T on kodim02 (rest Q); LOO3 → T on
kodim02/kodim19 (kodim02-T even rejects merges, m=0 — tree leaves already MDL-separated, cf. b6 §5).

## 4. Overlap analysis (which overlap? which are additive?)

1. **Backend choice (H/G/R) is the additive one: +0.0498, largest marginal, 7/7.** Orthogonal to
   partitioning (it prices whatever cells exist); post-merge tables amortize rANS headers so R/G take
   408/415 final clusters. No overlap with anything — the one component that stacks cleanly.
2. **LOCO/autoK/refine overlaps ~heavily with tree+grid — but is still the best partitioner (+0.0332
   marginal with T+D present).** LOO1 falls back to T (6/7) at only +1.02%: the tree rediscovers most of
   the LOCO-quantile separation (energy splits ≈ quantile cuts). Standalone LOCO value (Q vs order-0)
   is far larger (~−7% with predictors); the *marginal* measures only what tree/grid cannot cover.
3. **Tree and grid are fully overlapped in the stack (marginals 0.0000) — subsumed, not useless.**
   Q+WIDE-64+E16+refine beats T on 7/7 at 2-way level (margins 0.002–0.06) and D by ≥0.04. Mechanism:
   64-group quantiles of the 365-key ordering already express the joint sign-patterns the tree splits
   for, with finer granularity (89–139 atoms vs 29–40 tree leaves) at acceptable table cost once merges
   + Golomb (no-table) reprice fragmentation. Standalone tree value (−0.56% vs E6/KSET2 CROWN, b6) and
   grid value (−0.52% vs G6, b8) were measured against weaker bases; against the WIDE-E16-merge arm
   there is nothing left for them to harvest. LOO1 proves the redundancy is functional: remove Q and
   the tree immediately becomes the winner (6/7).
4. **Y-split is absorbed by merges (+0.0004, ~zero; kodim05 slightly negative).** Only ~10% of chroma
   atoms adopt the split, and merge re-clustering largely re-derives the same separations globally.
   kodim05's −0.0012 inversion (splits ON loses) is greedy-interaction noise: splits change merge atoms
   → heuristic shortlisted/proxy search lands in a different local optimum. Non-monotonicity is honest
   stacked behavior, bounded at ±0.001.
5. **E16's marginal (+0.0079) is half its standalone (−0.014, b5):** fine WIDE grouping + merges cover
   half the expert-bank gaps (LEFT/PLANE/AVG_AB/GAP16 matter less once groups are small and tables shared).
6. Marginals do NOT sum (Σ = +0.106 vs any single-arm delta) — the stack lives on overlap; §3 numbers
   are leave-one-out marginals, not independent contributions.

## 5. Decoder-safety statement

Framing order: dims 64b → path selector 2b → per-channel family sides (Q: key→group map
`nactive·⌈log2K⌉` + 4b K-id + 8b header; T: 32b/inner split incl. 8b topology correction to b6's 24b;
D: 3b grid id, occupancy decoder-known from recon energy) → per-atom 4b predids (3b in E6 arms) →
1b Y-split flags per chroma atom (always counted) → merge map `ncells·⌈log2C⌉` + 8b C → per-cluster
2b backend + 4b Golomb-k + winning tables only → payloads. Planes Y→Co→Cg (Y fully decoded before
chroma |rY| bins). Every selector (key/sign/energy/|rY|/row/col/tree-walk) re-derives from causal recon
only — encoder features ≡ decoder features (lossless). Proofs on the FULL-stack config, ALL-7 PASS:
(i) YCoCg-R RGB inverse; (ii) canonical Huffman bitstream pack→unpack→decode of EVERY final cluster;
(iii) Kraft=1 every cluster; (iv) partition-cover exactly-once every channel; (v) `rans_decode`
asserted on every R cluster (inside `rans_stream_bits`); (vi) full scalar decoder simulation —
per-pixel group/cell/table/expert re-derivation from recon with stream-cursor exhaustion asserts.

## 6. Trail (kept / killed / bugs)

- Kept: Q+WIDE+E16+3-way+merge+split stack (3.2515); per-image path gate (Q 7/7); per-atom Y-split gate
  (~10% adoption); shortlisted greedy merges (proxy search, exact-gated accepts, exact 3-way finalize, 7/7).
- Killed (measured zero in-stack): tree path, grid path, Y-split (kept in stack only because gated cost ≈ 0).
- Bugs found & fixed: (a) flat-vs-2D indexing on RF/RU (IndexError); (b) merge pre-pass dropped member
  lists → phantom cluster ids (IndexError; member tracking now spans both passes); (c) H-only arms
  initially optimized K/experts under H/G objective (objective now threaded honestly); (d) result JSON
  needed a numpy sanitizer (root trees excluded from dump; sides already counted).
- Approximations (bounded, stated): path selection at 2-way level (3-way+merges post-hoc; closest call
  kodim02 Q-vs-T 0.002 at 2-way → FULL-avg risk ≤ ~0.0003); tree structure grown for BANK5, leaves
  E16-rescored (conservative for tree; FULL total unaffected since T never selected); merge search is
  shortlisted (S=16) + proxy-driven (accepted merges exact-gated; b12: ordered search captures ~95%);
  Y-split bound [1] fixed from b8 (no tuning).

## 7. Verdict

- **Boss-5 (JXL-e3 3.2291): STANDS.** FULL-stack 3.2515 = +0.69%, 3W-4L (W=12, n.s.). Under 3.2291 avg?
  NO. Majority wins? NO (3/7).
- **Classical ceiling declared at 3.2515 avg** (probe frame, exact-counted, decode-verified).
  Trigger met (>3.24). Unstacked classical residuals are bounded and insufficient: Golomb-bias gate
  (~−0.001 avg est., CROWN2), per-channel (vs per-image) path mixing (~−0.002, more sides),
  DP-optimal cuts (~−0.005 est., b4), joint 3-way K-selection + Co→Cg cross-chroma feature (untested,
  small). Generous combined residual ≈ −0.01 → ~3.24, still short of 3.2291. No per-image win pattern
  suggests a missing classical mechanism: the gap lives on high-frequency texture (kodim05 +0.136,
  kodim13 +0.102), where causal spatial predictors are structurally blind.
- Campaign consequence: **stop stacking spatial-causal classical mechanisms. The remaining gap is
  representational, not allocational** — seven allocation tricks (partitioning, experts, backends,
  sharing, conditioning) now overlap each other and sum to +0.7% short.

## 8. Follow-up (transform/learned, with specific justification)

1. **Transform + bitplane arc (primary).** Justification: the FULL-vs-e3 gap is texture-concentrated
   (05/13 contribute 0.034 of the 0.022 avg gap — i.e., the other five images net −0.012 in our favor).
   Spatial MED/GAP-class predictors cannot compact paper-grain/foliage energy; JXL-e3's edge is exactly
   there. Cycle-3's DCT failure (6.95) was an unnormalized core without DPCM/context/bitplanes — the
   lesson was "EBCOT-like full build," never attempted since. Integer reversible wavelet (5/3, JXL
   VarDCT-family thinking) + LOCO-style context coding of bitplanes reuses the entire b4–b14 backend
   arsenal (E16-equivalent context experts, 3-way backends, merges) on transformed coefficients.
2. **Learned entropy / hyperprior (secondary).** Survey neural ceiling ~2.52 vs our 3.25: ~0.7 bpp of
   headroom is distributional, not allocational. A tiny offline-trained spatial mixer (PAQ-style logistic
   weights — the cycle-1 retry prescription) or hyperprior side-info would attack the same texture
   residuals from the modeling side. Constraint from GenAI survey stands: learn offline, ship integers.
3. **Luma-anchored chroma prediction (CfL-style, cheap classical residual).** Our Y-SPLIT conditioned
   *tables* on |rY|; predicting chroma *values* from decoded Y (rather than conditioning histograms)
   is the untested dual (b3's untested Squeeze lesson applied to chroma). Expected small (~−0.3%);
   worth one probe before leaving classical fully.
4. Retire: further context partitions, expert banks >16, backend additions beyond H/G/R, bias adapters
   under Huffman (proven null, b8 theorem), enumerative/VQ regrouping (b9–b11 arc closed).

Files: `experiments/probe_b14_stack.py`, `experiments/probe_b14_summary.json`, this report.
Anchors: CROWN-huff ≈3.343 ✓ (3.3248), CROWN-rans ≈3.312 ✓ (3.2811), JXL-e3 refs per §1 ✓ (avg 3.2291).
