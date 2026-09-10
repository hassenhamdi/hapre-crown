# Ideation Memory (M_I) — cycle 1 — 2026-09-07 — image compression
## Feasible Directions
- [FEASIBLE] YCoCg-R + causal MED + exact Huffman in streaming C (HAPRE-C): 3.58 Kodak avg, 9.76 MP/s enc,
  bit-exact round-trip 7/7, dominates PNG (p=.016), ties WebP-m0 ratio at 1.6× speed. ELO-equivalent top.
  Evidence: /tmp/opencode/autocompress/RESULTS.md + csrc/driver.py Pareto bench.
- [FEASIBLE] Functionally-lossless τ-quantized residuals (PSNR 47dB, RGB maxerr 3) at 3.26 avg: ties JXL-e3 rate.
  Publishable in near-lossless/FLLIC track with strict labeling. Needs JPEG-LS-near-lossless baseline.
## Unsuccessful Directions
- [IMPLEMENTATION FAILURE] Per-block LPC-3 in YCoCg (16×16): overfits texture, header overhead exceeds gain (−4%).
  Retryable with larger blocks / transmitted-only-on-gain + proper RD check (we used crude energy heuristic).
- [IMPLEMENTATION FAILURE] Crude gradient-rule mixer (MED/GAP/Paeth/LPC select): −3.7% vs MED-only.
  Retryable with trained (not hand-threshold) mixer, e.g. PAQ-style online logistic weights.
- [IMPLEMENTATION FAILURE] GAP + adaptive-bias quick port without run mode + per-ctx k: 5.00 bpp.
  Retryable as full JPEG-LS (run mode + per-ctx Golomb k) — literature says +3-5% over MED.
- [FUNDAMENTAL FAILURE for super-low-bpp goal] zlib as sole entropy back-end for 2D coefficients (wavelet+zlib q=1: 4.78):
  LZ cannot exploit 2D residual structure; EBCOT/context coders needed. Keep zlib only for speed mode.
- [FUNDAMENTAL for MED-class] Zero-order/9-ctx coding at 97% of conditional entropy: predictor wall reached.
  Further ratio needs larger-neighborhood prediction (MA-trees, learned) or transforms — out of tiny-model scope.
# Experimentation Memory (M_E) — cycle 1
## Data Strategies
- Kodak via r0k.us stable; md5-pinned (CHECKSUMS.txt). Synthetic photo_like misleading (too hard); always confirm on real photos.
## Training Strategies
- N/A (no learned weights in frozen codec). LPC per-image overfit lesson: always count header bytes in bpp.
## Architecture Strategies
- Border convention must match EXACTLY between encoder/decoder (0/left/top rule); edge-replicate breaks streaming decode.
- Canonical Huffman with explicit tables: 16b A + A×24b; zero-init C buffers via `(c_u8*n)()` — never `(*([0]*n))`.
- 64-bit buffer bit-packer ~fine; 8-bit LUT decode helps little when reconstruct dominates — profile before optimizing.
## Debugging Strategies
- When Huffman bytes look too good: check bound-vs-exact (entropy estimate vs realized stream). Always assert decode==input on full images.
- When synthetic images fail catastrophically but natural pass: suspect run/dictionary coding absence (constant residuals).
- Python bit loops: vectorize or move to C; numpy row loops with per-row overhead can exceed C-ref encoders.
# Cycle 2 (boss ladder) — 2026-09-07
## Feasible (new)
- [FEASIBLE] µMoE-8x8x6 block-adaptive experts + run + CTX9 Huffman: 3.464 avg, all PASS, KOs WebP-m0 (p=.031).
  L1-proxy selection + exact global Huffman; ids 3b-packed; GAP/DG experts essential. csrc/driver_moe.py.
- [FEASIBLE] Standalone verified LZ77 core (4KB window, greedy, 36ms/786KB, PASS) — REJECTED for residuals (4.18 bpp).
## Unsuccessful (new)
- [FUNDAMENTAL-ish] Linear prediction wall: FIR-12/lag-2/LS all ≈ MED in bits. Lag-2 acf (−0.28) not exploitable linearly.
- [IMPLEMENTATION] Naive 16-bin LF conditioning (−8..−12%): table fragmentation; needs clustering.
- [FUNDAMENTAL mismatch] Byte-LZ77 on int16 residuals (4.18): symbol structure destroyed.
- [BUG-CLASS] Residual alphabet clipping (±256): silent corruption on saturated edges; fixed via ±1024 + proven bounds.
- [BUG-CLASS] Run-framing: L=0 runs must emit tokens (JPEG-LS semantics); undecoded-pixel peeks break causality.
- [BUG-CLASS] rANS precision: M must satisfy 2^M >> A (A=232 needs M=14, not 10).
- [IMPL FAILURE] LOCO-I-lite bias under Huffman (−0.2%): dithering hurts; reverted to golden.
## Boss state
DEAD: PNG-9, JXL-e1, WebP-m0. STANDING: WebP-m3 (we lose p=.031), WebP-m6, JXL-e3, JXL-e9.
# Cycle 3 (subagent probes) — 2026-09-07
- [PROBE-NEG] Integer-DCT + order-0 subband Huffman: 6.95 avg (vs 3.464). Unnormalized-core flaw noted;
  structural lesson: transforms need DPCM/context/bitplanes (EBCOT-like full build). Transform arc = major arc.
- [PROBE-NEG] LF-clustered conditioning (4-bin, subagent, baseline=raw-signal order-0): +7.5% loss;
  side stream ~3x the conditioning savings. LF-CONDITIONING retired. LF-PREDICTION (Squeeze-style,
  predict HF from LF rather than condition entropy) UNTESTED — last structurally-sound Boss-2 weapon.
- Paper draft v0.1 banked: docs/HAPRE-C-paper-draft-v01.md.
# Cycle 4 (sequential branches 1/1b) — 2026-09-07
- [AUDITED PROBE] Branch 1 Squeeze-NN: units were 3x (bits/pixel not /3ch) but delta +13.48% VALID;
  baseline/3 = 3.58/checks vs champion MED. Mechanism: NN block edges poison MED (var ratio 1.073).
- [PROBE-NEG] Branch 1b smooth upsampling (bilinear +10.77%, smoothed-NN +10.54%, baseline anchor 3.577 PASS):
  recovers only ~3pp. Transmitted-4x-LF family RETIRED (side ~0.40bpp structurally > savings).
# Cycle 5 (Branch 2 band-adaptive) — 2026-09-07
- [MARGINAL] Fixed bands: B=2 −0.012, B=4 −0.017 bpp (~0.4%). Table fragmentation confirmed at small scale.
  Subagent diligence note: my 4.4–4.6 anchor was miscalibrated (forgot YCoCg chroma gain); RGB-MED control 4.96 proved impl correct.
- [REPLAN] Boss-2 needs −3.3%: coding tweaks (bands+rANS ≈ −1%) insufficient. Next: full LOCO-I hybrid
  (run+bias+per-ctx-k was never properly built — earlier GAP port lacked run mode and per-ctx k).
# Cycle 6 (Branch 3 full LOCO-I) — 2026-09-07
- [STRONG PROBE] Full LOCO-I (run+bias+per-ctx-k, pure Golomb): 3.333 vs MED+Golomb 3.700 = −9.9%, 7/7.
  Split: per-ctx-k+run ~2x the bias gain. Caveat: Golomb-side % won't transfer 1:1 to Huffman champion.
  Follow-up: port bias-corrected residuals + per-ctx/grouped Huffman onto champion arm.
# Cycle 7 (Branch 4 exploratory CROWN) — 2026-09-07
- [WIN] CROWN-huff 3.3429 (−3.5% vs champ, 7/7 p=.016): sign-flipped LOCO-365 ctx → auto-K quantile
  clustering → per-group best-of-6 predictor → per-group Huffman. No runs/ids. Anchors reproduced.
- [WIN] CROWN-rans stretch 3.3124 (−4.4%): per-image diffs vs PIL-m3 all negative → Boss-2 KO (p=.016).
  CROWN-huff vs PIL-m3 = TIE (1W-6L, p≈.47). Board m3 bar corrected 3.38→3.347 (perimage.py; subagent confirmed 3.345).
- Boss 2 WebP-m3: KO (rANS backend, probe-level proof, C build pending).
# Cycle 8 (CROWN-C exact + Boss-2 KO) — 2026-09-07
- [BUILT] CROWN-C exact codec (driver_crown.py): LOCO-365 keys + autoK quantile groups + per-group
  best-of-6 + Huffman/rANS backends, C streaming decode. Bit-exact vs probe designs.
- [BUGS] cut-semantics off-by-one (bounds use c+1); empty-group rans framing (count+A written, n+pay skipped —
  mirrored); greedy-overshoot (fewer groups than K, normal). Map bit-packed (10b+6b).
- [KO] CROWN-rans-hillclimb 3.309 avg, 7/7 vs WebP-m3, p=.016. BOSS 2 DEAD at exact-codec level.
- Golden snapshots: hapre.c.golden-moe (RUN/MOE era). Current hapre.c = CROWN era (superset).
# Cycle 9 (Branch 5 backend-bit + E16) — 2026-09-07
- [WIN] E16 experts + per-group Huff/Golomb 2-way + WIDE-K(64): 3.2979 probe (−0.34% vs 3.309).
  vs m6: 6W-1L, p=.16 → TIE (not KO). LEFT/PLANE/AVG_AB/GAP16 steal share.
- [WIN/KO] +rANS 3-way: 3.2720 (−1.1% vs 3.309), 7/7 vs m6, p=.016 → BOSS 3 (WebP-m6) DEAD (probe-level,
  decode-verified; exact-C build = CROWN-rans v2, pending).
- Board: DEAD PNG-9, JXL-e1, WebP-m0, WebP-m3, WebP-m6. STANDING: JXL-e3 (3.23), JXL-e9 (3.03).
# Cycle 9 (Branch 5 Boss-3) — 2026-09-07
- [WIN probe-level] E16 experts + Huff/Golomb 2-way + WIDE-K(64): 3.2979 (−0.34% vs 3.309).
  vs m6: 6W-1L but p=.16 → TIE (audited; subagent overclaimed KO).
- [KO probe-level] +rANS 3-way: 3.2720 (−1.12%), 7/7 vs m6, W=0 p=.016 → BOSS 3 DEAD (C port pending).
- Boss board: DEAD: PNG-9, JXL-e1, WebP-m0, WebP-m3, WebP-m6(rANS). STANDING: JXL-e3 (3.23), JXL-e9 (3.03).
# Cycle 9 (Branch 5 backend-bit + Boss-3 KO) — 2026-09-07
- [WIN] E16 experts + per-group Huff/Golomb 2-way + WIDE-K(48,64): 3.2979 non-rANS (ties m6, p=.16).
- [WIN] +rANS 3-way refine: 3.2720 avg, 7/7 vs WebP-m6, W=0 p=.016 (verified independently). BOSS 3 DEAD (probe-level; C-port pending).
- Boss board: DEAD PNG-9, JXL-e1, WebP-m0, WebP-m3, WebP-m6. STANDING: JXL-e3 (3.23), JXL-e9 (3.03).
# Cycle 10 (survey: GenAI/CV + QOI teardown, delegated) — 2026-09-07
- Survey banked: survey/SURVEY_genai_qoi.md (+qoi_notes.md, qoi/ clone). Neural ceiling ~2.52 (CALLIC/HPAC);
  JXL 2.87–3.06; SEEC 2.84 supports multi-distribution thesis.
- TOP-5: MA-tree-lite (−2–4%); luma-anchored chroma (−0.5–1%); distant-repeat hash-cache (−0.3–0.8%);
  IFCE inter-group conditioning (−1–2%); MDL-gated micro-adapters (−1–3%).
- QOI most portable: LUMA green-anchored asymmetric differencing. GenAI verdict: learn-offline/ship-integers only.
# Cycle 11 (MA-tree-lite probe) — 2026-09-07
- [WIN-SMALL] MA-tree-lite (deep ≤16 leaves + sharing): 3.3242 avg, −0.56% vs CROWN-huff, 6/7.
  Shallow trees lose; sharing helps CROWN more than tree (MDL-separated leaves). rANS projection ≈3.254.
- Boss 5 verdict: NOT in reach via tree alone (+2.96% gap remains). Tree ≈ 1/3 of e3 gap.
- Next: QOI-ports (luma-chroma + hash-cache), then IFCE/MDL adapters; stack toward −1.3%.
# Cycle 12 (QOI-ports) — 2026-09-07
- [WIN] A-joint (Co,Cg) pair alphabet: −1.58% order-0 frame (chroma-chroma ≈ 2x luma-chroma). Candidate for CROWN arm.
- [DEAD] B hash-cache escape +2.85% on photos (flag 1.000b + uniform index + selection bias). Keep as non-photo fast path only.
- Estimator note: exact-Huffman MED = 3.51 on kodim19 (my 3.36 was Rice/Huffman-select bound, optimistic). Anchor 3.577 stands.
# Cycle 13 (IFCE+MDL) — 2026-09-07
- [WIN] I-GATED (MDL-gated Y-split): −0.29%, 7/7. M-THR grid dictionary: −0.52%, 7/7 (D0 grid too coarse).
  C-FULL2 stack: −0.65% vs G6, 7/7. Port: grid + gated splits onto CROWN-rans.
- [THEOREM] Bias-shift adapters are Huffman-translation-invariant (Δ=0 proven) — Golomb-only weapon. Retired under Huffman.
- Retired: I-DC2, I-EC2, all Huffman bias adapters.
# Cycle 14 (CROWN2 greenfield exact) — 2026-09-07
- [BUILT] CROWN2 (crown2.c + driver_crown2.py + FORMAT.md, separate files, existing untouched):
  exact 3.2686 avg, ALL-7 PASS, beats probe 3.2720 (−0.10%). Backends H:54 G:521 R:237.
- [RETIRED in-frame] A-joint pairs (+7..+17% under E16 groups), I-GATED (+0.0003, WIDE-K harvests it),
  M-THR grids (lose on Kodak; kept as torture-input fast path). G-bias pays (−0.36% on 13).
- Boss 5 (JXL-e3 3.2291 re-measured): STANDING, +1.22%, p=.156 (1W-6L). Boss-3 margin extended (−1.42%).
# Cycle 15 (quality metrics + RD table) — 2026-09-07
- Metrics module: experiments/quality.py (PSNR/SSIM/maxerr, selftest PASS).
- RD sweep (rd_sweep.py, prototype family, exact Rice bytes): tau0 lossless (PSNR inf);
  tau1 ~2.75-3.23bpp maxerr3 PSNR47.3 SSIM~0.99; tau2 ~2.0-2.6 maxerr4 PSNR43.5 SSIM~0.98.
- JXL lossy reference: d0.5 ~0.8-1.2bpp PSNR~42; d1.0 ~0.4-0.8 PSNR~38-40. Near-lossless NOT
  PSNR-competitive with lossy (expected; its value is maxerr guarantees). Documented, not hidden.
# Cycle 16 (refresher: enumerative 2x2) — 2026-09-07
- [BEAUTIFUL DEAD END] Cover-73 enumerative coding on 2x2 MED-residual blocks: +0.32% loss.
  Mechanism: positions uniform given weight (ranks buy ~0); conditioning on nonzero inflates value
  entropy (cancels savings); infinite-precision ledger caps headroom at +0.5%. Larger blocks worse.
- [OPENED] Monetizable joint info is in MAGNITUDES-given-activity, not activity patterns → magnitude-VQ branch.
- Strategy note: finish probe campaign, then ONE consolidated C port (not incremental).
# Cycle 17 (magnitude-VQ) — 2026-09-07
- [WIN] Q-T4 2x2 magnitude-VQ: −1.84% gated vs MED order-0 (3.577→3.511), 7/7, round-trips pass.
  Mechanism: clipped-magnitude total correlation ~0.079 b/res; joint codebook harvests 84%.
  Pyramid factorization dead (−0.06%, never factorize what you can tabulate). Signs independent (split optimal).
- Transfer to CROWN arm UNMEASURED — next branch. Corrigendum: b9's "170x" was P(all-active) mislabeled.
# Cycle 18 (VQ-transfer) — 2026-09-07
- [PARTIAL] VQ stacks at −0.52% (huff) / −1.25% (rANS→3.3013) over CROWN-huff, but BEHIND CROWN2-exact
  3.2686. Overlap fraction ~0.1-0.2 (grouping consumes ~72% of VQ harvest). Naive regrouping +3.85% (destroys
  per-pixel adaptation). Rule: grouping granularity dominates; never disturb per-pixel adaptation for VQ.
- Boss-5 verdict: best exact 3.2686 vs JXL-e3 3.2291 (+1.2%, p=.156 earlier). Classical arsenal ~1% from ceiling.
  Remaining structural path: transform-bitplane arc (for e3/e9).
# Cycle 19 (histogram sharing) — 2026-09-07
- [WIN] Greedy table merges: −1.20% (MED frame), 7/7, tables 60→18, 5:1 MDL return. JS-order ≈95% cheaper.
  Transfer to CROWN2 est. −0.3..−0.7% (less fat there). rANS stretch −1.53% (proxy-optimistic).
- Next: merge pass ON CROWN2 histograms (probe), then consolidated C port, then Boss-5 verdict.
# Cycle 20 (merge-on-CROWN2) — 2026-09-07
- Merge transfer: (b) 3.3134 (−0.37%), (c) rANS 3.2926, (c2) 3.2806 (−1.35% vs their (a)).
  Boss-5 STANDS (+1.6%, 1W-6L). Merges ≈ 1/3 of e3 gap.
- CONFIRMED: CROWN2-exact (3.2686) vs WebP-m6 = 7/7 wins, p=.016 → Boss-3 DEAD at exact-codec level.
- Board: DEAD PNG-9, JXL-e1, WebP-m0, WebP-m3, WebP-m6 (all exact-codec). STANDING: JXL-e3, JXL-e9.
# Cycle 21 (classical ceiling) — 2026-09-07 — DECISIVE
- Full classical stack: 3.2515 avg vs JXL-e3 3.2291 (+0.69%, 3W-4L n.s.). Boss 5 STANDS vs all-classical.
- Leave-one-out: backend-choice +0.050 (orthogonal!), LOCO-partition +0.033, merges +0.015, E16 +0.008,
  Y-split +0.0004, MA-tree/grid never selected (subsumed). Marginals don't sum (overlap).
- CLASSICAL CEILING DECLARED at 3.2515. Gap texture-concentrated (05+13). Unstacked crumbs ≈ −0.01.
- Next: transform-bitplane arc (primary), learned entropy (secondary). Retire: partitions, banks>16, backends>H/G/R.
# Cycle 22 (transform-bitplane) — 2026-09-07 — NEGATIVE
- Best (2-level 5/3 + LL-DPCM + 12-group conditioned Huffman): 3.3737, +3.8% vs classical ceiling, 0W-7L.
  Mechanism: 5/3 detail bands dense (48-86% nonzero) → zerotree significance costs ~3b/coeff unharvestable;
  wavelets scatter smooth-field predictability (worst gaps on easy images). Family exhausted (1/2/3-level,
  finer groups, cross-channel all negative). Bitplane-B +26% (stripped EBCOT never again).
- Standing: JXL-e3 (+0.7%), JXL-e9 (+7%). Remaining structural idea: hybrid LL+CROWN / sparse-chroma.
# Cycle 23 (hybrid) — 2026-09-07 — NEGATIVE
- Hybrid Y-pixel + chroma-wavelet: 3.3306 (best transform-family, −1.3% vs b15) but +2.4% over ceiling.
  Mechanism: channel asymmetry unanimous (Y-pixel always, Co/Cg-transform always); detail order-0 sweeps 27/27;
  sparsity coding needs sparser bands than 5/3 gives. Hybrid/subband arc RETIRED.
- CAMPAIGN STATE: 5 bosses dead exact-level. Classical+transform+hybrid ceiling ≈ 3.25. e3 (+0.7%) and e9 (+7%)
  require learned scale (no torch/GPU here) or undiscovered structure. Recommend: consolidate C-port + publish.
# Cycle 24 (E16 C-port + Boss-3 exact KO) — 2026-09-07
- [BUILT] driver_crownE.py: WIDE-K + E16 + per-group H/G backend + hillclimb + tight format.
  Exact 3.2850 avg, ALL-7 PASS. vs WebP-m6: 6W-1L (only kodim13 +0.014), p≈.047. BOSS 3 DEAD exact-level.
- Golden: hapre.c.golden-e16 (adds E16 cases 6-15, golomb pack/unpack, b4_nbhd, e16_residuals, crown2_assemble).
- Board: DEAD PNG-9, JXL-e1, WebP-m0, WebP-m3, WebP-m6 (all exact-codec, asserted round-trips).
  STANDING: JXL-e3 (+1.8%), JXL-e9 (+8%).
# Cycle 25 (learned micro-MLP) — 2026-09-07
- torch 2.14 CPU installed (pip --user --break-system-packages; 6GB disk OK).
- TinyMLP 8->8->1 (81 params), L1 loss, LS-init (random-init LOSES badly: init matters more than capacity).
- Gated per-channel (MLP vs MED): −1.64% order-0 frame (Cg always wins +1.5..+8.7%; Y/Co noisy).
- Quantization: int16 x256 retains 70% of gain (viable, 486B/img); int8 destroys (+81%). Gate+overhead counted.
- Next: MLP as 17th expert in per-group search (automatic MDL) + merges + MA-tree → Boss-5 assault (~30-40%).
# Cycle 26 (boss takedown review) — 2026-09-07
- JXL ladder reproduced: e1 3.72 → e3 3.2291 → e6 3.089 → e7 3.0424 → e8 3.0442 (REGRESSION) → e9 3.0324.
- WEAKNESSES: group-256 fragments stats (-g3: e9→2.9898!); e9 overfits (matches upstream #4107/#3888);
  fixed RCT -C27 beats e3 search (3.2083); squeeze-on +15-52%; noise +9%.
- TRUE e9 bar with tuning ≈2.94 (Boss-6 needs learned-micro, not hand contexts). Boss-5 feasible via Attack 1+2.
- Attack 1: per-group RCT + permutation front-end + Weighted expert (−0.7–1.0%).
# Cycle 27 (Attack-1: RCT + Weighted) — 2026-09-07
- [WIN] Global best-RCT (a0): −1.41% (C27 6/7, C12 on 02 — mirrors JXL). Per-block RCT DEAD (dictionary wins).
- [WIN] Weighted per-G32 LS-4tap x16-quant MDL-gated: −2.15%. Kernels = edge-sharpeners (neg TL weight).
- [WIN] Homogeneous (a0+b): −2.51% MED frame, 7/7 p=.016, decode proof PASS.
- Transfer to CROWN2 est. −1.2..−1.7% (need −1.22% for Boss-5). C-port: global RCT {C6,C27,C12} + Weighted-17th.
- Honesty: quantizer scale bug caught by mechanism diagnostics, invalidated + rerun, logged.
# Cycle 28 (CROWN3 exact) — 2026-09-07
- [BUILT] CROWN3 (crown3.c + driver_crown3.py + FORMAT.md, new files only): RCT{C6,C27,C12} + Weighted-17th.
  Exact 3.2040 avg, ALL-7 PASS. Transfer −1.98% ( beat est.). vs CROWN2: 7/7 p=.016.
- Boss-5: LEAD 5W-2L (avg −0.78%) but p=.375 → STANDING under KO rules. Losses: kodim05 +2.46%, kodim13 +1.02% (texture).
- Encode ~5min/image Python (Pareto-enc dead for CROWN3; ratio game only). Decode C ~140ms.
- Deviations logged: G32 fixed; golomb unary polarity vendored (wire-format split from libhapre era — documented).
# Cycle 29 (texture-gap: LMS pair) — 2026-09-07
- [WIN-SMALL] LMS5_T0/T3 gated adaptive-FIR experts (18th/19th): new best 3.1994 (−0.15%), 7/7 gains p=.016,
  no regressions. 05 now +2.27% (was +2.46%), 13 +0.98%. Boss-5 STANDS (5W-2L p=.375).
- [DIAGNOSIS] 05's loss ≈90% entropy-coding overhead (floor 3.500 already beats JXL 3.512!): K=64 fine groups →
  tables too big → Golomb default misfits heavy tails. WEAPON IDENTIFIED: CABAC-class adaptivity.
- Killed 15 (lags, patch-match, bilateral, cells, per-key-k, mixtures...), 2 frame-poisons (GS16, ERR4), merges wounded.
# Cycle 30 (CROWN4 exact: LMS port) — 2026-09-07
- [BUILT] CROWN4 (crown4.c + driver_crown4.py + FORMAT.md, new files only): exact 3.1993 avg, ALL-7 PASS,
  matches probe ±0.00%. 7/7 beat CROWN3 (p=.016). Encode ~5.5min Python; decode C ~230ms.
- Deviations: LMS is 5-tap (4-tap loses); unary polarity vendored+documented (wire-split from libhapre era).
- Boss-5 STANDS (5W-2L p=.375; 05 +2.27%, 13 +0.98%). New best exact. Next: CABAC adaptivity for 05 overhead.
# Cycle 31 (CABAC adaptivity) — 2026-09-07
- [WIN] A-act (16-ctx adaptive range coder, real bytes): −2.18% MED frame, 7/7, round-trips pass.
  C-gol (adaptive-k): −1.74%, beats A-act on blockers. D-run/B-fwd/A-pos killed. Conditioning essential.
- Transfer est vs CROWN4: best-of 3.1138 (7W-0L p=.016) IF ~65%+ transfers. Boss-5 KO path = defined construction.
- Next: CROWN5 greenfield (CROWN4 predictors/groups + global raster-order adaptive states).
# Cycle 32 (CROWN5: adaptive transfer) — 2026-09-07 — NEGATIVE with mechanism
- CROWN5 best-of picks static 21/21; bytes = CROWN4+21B. Transfer est invalidated (+2.75% gap).
- Mechanism: ~100% overlap — WIDE-K per-group static strictly finer than 4-bin global adaptive.
- TRUE construction: JOINT (group x activity) backward-adaptive states (fine groups, zero tables).
  (LOCO-I per-ctx adaptive Golomb was the primitive version; need it with rANS/arithmetic + our groups.)
- Boss-5 STANDS. Next: joint-adaptive branch.
# Cycle 33 (joint adaptivity) — 2026-09-07 — DECISIVE NEGATIVE
- Joint (group x activity) backward-adaptive: +3.36% vs static-grouped, 0W-7L. Mechanism 7:1 against:
  table prize 0.018 vs cold-start/whiplash deficit +0.11. Static oracle picks 21/21 planes.
- ENTROPY-CODING ARC CLOSED: global adaptivity dead, joint adaptivity dead, merges help (+0.4%).
  Static clustered tables + rANS = local optimum. Remaining gap is PREDICTION-side (05/13 texture).
# Cycle 34 (per-context neural) — 2026-09-07
- [WIN] Per-ctx micro-MLP (12f->8->1, L1, LS-init, adaptive-int16, double MDL): −2.93% MED frame,
  6/7 over global-MLP. f12 (lag taps) wins; h16 rejected (side). Fixed-x256 quant FAILS per-ctx (LSB doubling);
  adaptive-int16 (2048/4096) retains 97-100%.
- Transfer est: optimistic ~3.15-3.17 (KO range) / pessimistic flips 05. Next: MLP-17th-expert CROWN integration.
- CAUTION for C-port: torch-tanh vs C-libm 1-ulp divergence risk → cross-check int-rounded equality on all images.
# Cycle 35 (CROWN6 exact: neural integration) — 2026-09-07
- [BUILT] CROWN6 (crown6.c + driver + FORMAT + tanhlut + weights): exact 3.1978 avg, ALL-7 PASS.
  Beats CROWN4 bar (−0.05%). Boss-5: 5W-2L p=.219, STANDS (05 +1.95%, 13 +1.00%).
- [SOLVED] torch-vs-C tanh 1-ulp divergence via frozen LUT (2^-14 step, bit-exact literals, -ffp-contract=off);
  0 mismatches / 17.2M px. Causality bug (edge-replicate peeks) caught by forced-MLP test.
- Honest stop on finer variants (10x short). New best exact. Remaining gap = texture, needs larger-neighborhood ideas.
# Cycle 36 (paper v1.0 draft) — 2026-09-07
- Draft: docs/HAPRE-CROWN-paper-v10.md, 5557 words, 146 [VERIFY] marks (honest transcription policy).
- Weakest: RD section (3-image prototype ranges; needs 7-image exact + JPEG-LS baseline for FLLIC track).
# Cycle 37 (verification audit) — 2026-09-07
- 52 claim-rows CONFIRMED (all boss per-image tables, averages as row-means, 8/9 Wilcoxon rows).
- 1 correction: CROWN-huff vs m3 p .47 -> .30 (still tie; verdict unchanged). All 5 KOs re-verified W=0 p=.016.
- Gap to camera-ready: bank per-image ledgers for PNG-9/WebP-m0/RUN/byte-LZ/LOCO-lite/G-bias (or soften citations).
# Cycle 38 (harness hardening) — 2026-09-07/08
- [BUG-CLASS] Import-time os.chdir side effects in 8 drivers broke any external cwd (found via reproduce.sh
  debugging). Removed; absolute sys.path only. Verified: reproduce.sh green.
- reproduce.sh: checksums + make all + RUN/CROWN-huff round-trips (3.11/3.06, 2.89/2.86) exit 0.
# Cycle 39 (harness hardening, self-performed) — 2026-09-07/08
- README/Makefile/requirements/reproduce.sh/.gitignore/docs-INDEX written. reproduce.sh GREEN.
- REMOVED import-time os.chdir side effects (8 drivers) — broke any external cwd; verified green after.
- Deleted crashprobe.py, *.bak, __pycache__; .jxl repro moved to data-archive/ (gitignored).
- NO moves of code/probes (import-path risk documented in README).
- SELF-CRITIQUE: overwrote RESULTS.md v3 narrative instead of appending — detail survives in memory log +
  probe RESULTS + VERIFY_report, but don't repeat (append-only for ledgers).
# Cycle 40 (reorg: symlinks + dupes) — 2026-09-07/08
- Prior silent reorg committed inside harness add -A (src/+probes/+compat links). Resolved now.
- Deleted 130 dead probes/ copies (experiments/ links were the live path all along — md5 "identical"
  finding was vacuous link-vs-target; lesson: check file types in audits).
- Removed csrc/ compat dir; repointed sys.path/Makefile/reproduce/README/docs to src/.
- crown2 import failure (pre-existing): vendored missing helpers into crown2.c per codebase convention;
  imports OK; encode bytes == banked (414918 kodim23). Roundtrip check inconclusive (driver returns
  tuple; slow encode) — SKIPPED per user order (isolated to crown2).
- Fixed: import-time os.chdir removal stood; reproduce.sh green throughout.
# Cycle 41 (reorg: symlinks + dupes, self-performed) — 2026-09-07/08
- Found 170-compat-symlink silent reorg committed inside harness add -A (src/+probes/ real, csrc/ links).
- NEAR-MISS: deleted probes/ thinking experiments/ canonical — inverted! experiments/*.py are LINKS to
  probes/ files (md5 equality was vacuous). Restored from git. Lesson: check file types in audits.
- Final layout: src/ canonical code, probes/ canonical probe files, experiments/ compat links (load-bearing).
  csrc/ removed; sys.path/Makefile/reproduce/README/docs repointed; probe .so-path refs repointed.
- crown2 import failure (pre-existing, missing vendored helpers): vendored pack_syms/rans/ycocg_inv from
  hapre.c per codebase convention; imports OK; encode bytes == banked (414918 kodim23). Roundtrip check
  inconclusive (tuple-vs-bytes test bug + slow encode) — SKIPPED per user (isolated to crown2).
- Main guards added to pareto.py/perimage.py (executed on import before).
- Verified: make clean+all, reproduce.sh green, 17/17 imports, per-image numbers match banked.
- Committed d432a28 (2A/31D/36M/9R), tree clean.
# Cycle 45 (C++ trainer, pure-C++17, publication-grade) — 2026-09-08
- Built: cpp/crown_train.cpp + cpp/TRAINER.md (methods, verification, ablation,
  reproducibility). Makefile builds crown_train (-O2 -ffp-contract=off -pthread).
  Zero new deps (C++17 + linked C objects); .gitignore covers binary.
- Config: 12→8→1 tanh, L1, LS-init, it200/lr0.003, Adam(0.9,0.999,1e-8)+cosine,
  bs16384, adaptive-int16 {256…4096}, double MDL gate, zero-border causal,
  f64 thresholds, expert-19. Seeds bit-identical to Python (sha256 verified).
- Deviations T1–T5 documented (mt19937 vs Philox order, tanhf vs Sleef,
  normal-eq vs gelsd, f32/f64 casts, no torch). Libtorch option evaluated and
  rejected: same algorithm, heavier link, not faster than hand-rolled f32 Adam
  (memory-bound on 113 params).
- Evidence (same machine, byte-asserted round-trips):
  crop256 smooth 256²: train 5.4s vs Py 9.1s, gating 27/27, bytes 66728=66728,
  bpp 2.7152 Δ+0.0000%, sha identical (4d0a9c…).
  crop05tex texture 256²: train 5.1s vs Py 12.0s (2.35×), gating 26/27 (1
  borderline L1), bytes 97279=97279, bpp 3.9583 Δ+0.0000%, sha identical.
  kodim23 full: train 12.9s, gating 74/81=91.4% vs banked Python weights;
  full encode parity pending (~7 min, background job).
- Bug-classes: singular LS dir → 0 + Adam recovers (0 hits); stale thresholds
  abort; alphabet/framing asserts kept; NPY writer pads to 64B, stored-zip with
  CRC, merges existing entries for --rct subsets.
- Lesson: different qw basins still give identical gating/streams on smooth
  content (quant-robust basin evidence for paper); texture borderline gates are
  the right sensitivity metric, not weight equality.
# Cycle 46 (encode speed 8.1x, bit-exact) — 2026-09-08
- Profile (crop128, env-gated CROWN_TIME timers, thread-local merge):
  Golomb costing 16.1/18.5s (87%), Huff 2.1s, rest noise. Frontend/asm ~0.
- Optimizations (all proven sha-identical: 3 crops + full-7 vs pre-opt binary):
  (a) single-pass fused Golomb (9 bias variants x 13 k in 1 sweep, int64-exact,
  same selection order; 117 sweeps + tmp alloc removed);
  (b) histogram Huff over proven ±1024 alphabet (O(n) vs copy+sort; counts
  sorted ascending, totals tie-break-invariant per existing heap comment);
  (c) dead no-MLP-track elimination in Q-coarse (coarse uses tot_all only —
  same computation, half the calls; GRID/refine need both, untouched);
  (d) 9-way (RCT,channel) parallelism via std::async, fixed-order join +
  strict-< RCT pick (job-count invariant, verified jobs=1/9);
  C fns on path are pure caller-buffer ops (no shared mutable state — checked).
- Results: crop128 18.5→2.1s; kodim23 432→53.6s; full-7 byte counts == banked
  goldens (486346,440219,527981,395158,582929,464631,403508, avg 3.1978),
  round-trips PASS. Target ~1 min/img MET with margin (queue item 2 DONE).
- Lesson: profile before optimizing (87% in one fn); Fargo-style dead-code
  (unused dual track) hides in faithful ports — port exact, then delete dead.
# Cycle 47 (Boss-5 quest: triple-negative, banked + redirected) — 2026-09-08
- Focal Q: flip 05 (+10,115B) and/or 13 (+5,770B) vs JXL-e3 at exact level.
  Gaps recomputed from AUTHORITATIVE bytes (banked goldens vs JXL bpp×147456);
  HANDOVER per-img decimals noted stale for 05 (3.5917 vs bytes-implied 3.5806).
- [KILLED b22] adaptive-Golomb transfer: residual delta ~0 over per-group
  best-k (05: negative; 13: +18% of gap best case). Mechanism: quantile groups
  are scale-homogeneous by construction (static-k near-optimal); N/A/k pays
  cold-start and lacks bias adapter; global pooling mixes experts (b18
  whiplash, -2..-6%). Chosen-G beats H0+real-tables by 5-7% (coding exhausted).
  Infra: CROWN_DUMPGROUPS dump (GGD1) in crown_enc.cpp, behavior-preserving
  (sha-verified); probes/probe_b22_{adaptive,run,RESULTS,nums}.
- [CLOSED b23] it300 + late-avg texture tail: all |Δ|<=0.5%, wins identical
  18/18 units, aggregate ~1-2kb vs -81/-46kb needed. Basin flat in quantized
  space; training axis on this arch DONE. probes/probe_b23_{tail,RESULTS,nums}.
- [INSUFFICIENT b24] side audit (bit-exact walker, p==len): R-headers 2.9KB/4.3KB
  (29%/74% of gaps); 50% sharing recovers 15-37%. H-tables/G-kd dead.
  probes/probe_b24_{sideaudit,RESULTS,nums}.
- Lessons: (1) transfer estimates must use the CURRENT frame's baseline, not
  MED-frame (b19 overlap ~100% surprise); (2) weight-equality != gate-equality
  (b22 wins matched despite different qw); (3) H0+tables counterfactual
  separates coding-exhaustion from prediction-gap in one test.
- Boss-5 STANDS (5W-2L). Redirect per owner: banked + moved on.
# Cycle 48 (publish audit: venues, SOTA, draft) — 2026-09-08
- Venue (verified official, checked 2026-09-08): DCC 2027 full paper, deadline
  Oct 2 2026 (~24d), 10pp TOTAL incl refs/figs, 1-col 12pt, single-blind (names
  on ms), PDF, LaTeX sample ZIP on site, proceedings IEEE Xplore.
- SOTA verified vs opened PDFs: CALLIC 2.54 (AAAI25, MGCF 2.77->2.54 LoRA MDL,
  575K, enc 9.7s); HPAC-FT 2.52 (base 2.73, 677K, 0.89s); SEEC 8.53 total bpp
  (=2.84, DLPR-backbone+2-class masks); P2-LLM 2.83 Kodak (Llama3-8B, dec 273s);
  classics full-24: JPEG-LS 2.82, CALIC 2.87, FLIF 2.72, JXL 2.63-3.06 spread
  (versions+subsets; ours e9 3.03 7-img sits inside). Draft fixes: HPAC-FT vs
  base disambiguation, SEEC unit note, P2-LLM naming+efficiency contrast,
  "~60%" denominator is DLPR-2.86 (60.7% VERIFIED), JXL triple-coordinate rule,
  7-subset skew stated. DualComp-I 2.57 snippet-only (verify before citing).
- Draft audit: contribution conditional-pass (sell efficiency Pareto, not ratio);
  0 figures (F1-F6 specified, F5 Pareto is thesis); tables->booktabs at LaTeX;
  page-budget OVER RISK (cut §5 30%, don't "move to appendix" — counts);
  gaps: FLIF/JPEG-LS/CALIC unmeasured (imagecodecs/FLIF attempt), VERIFY
  Counts ledgers, W=6, multi-comparison footnote, train-data footnotes for
  learned rows, csrc->src/cpp path sweep, seeds/threads note. Humanizer targets
  flagged, rewrite deferred (needs author voice sample).
- Doc: docs/PUBLISH_audit.md (compliance note, SOTA table, audit, F-list,
  evidence ledger, 9-item camera-ready list).
# Cycle 49 (classical baselines measured: FLIF + JPEG-LS) — 2026-09-08
- FLIF v0.4 source build (tag v0.4, g++ -O2, default -e): 2.9699 avg, pixel-PASS
  7/7, kodim23 enc 1.28s/dec 0.35s. Beats CROWN6 0W-7L p=.016 AND JXL-e3 7/7 on
  our seven -> NEW STANDING BOSS (stronger than e3 here). Ledger
  survey/bosstakedown/data/classical7.json. Boss count 5->6 dead still (FLIF
  stands); draft/README/RESULTS/AGENTS/HANDOVER updated (stopping at three).
- JPEG-LS (CharLS 2.4.4 RGB defaults, RT-asserted): 4.5757 avg, DEAD 7W-0L.
  Discrepancy vs literature 2.82 investigated: core works (gradient 1.11bpp,
  flat 83B); gap is color pipeline (standard has no RCT). Ablation
  JPEG-LS-o-YCoCg-R (NOT a codec): 3.3664 -> front end buys 1.21bpp; stack adds
  further -0.17. Literature 2.82 uses undisclosed pipeline (SEEC's own gives
  4.36) -> report measured + spread, never transcribe as baseline.
- CALIC: no reference binary in-box (only unverified Python reimplementation)
  -> literature row retained with explicit "not measured locally".
- VERIFY_report §5 appended (means + Wilcoxons recomputed from ledger).
# Cycle 50 (FLIF teardown, subagent full-read review) — 2026-09-08
- Dispatched general subagent (read-only, full-files-at-once instruction) over
  FLIF v0.4 src (tag v0.4, md5 c9615a4a...; NOT vendored). Report banked:
  survey/flif/FLIF_TEARDOWN.md (trim labeled). Owner spot-checked threshold/
  prune/predictor/YCoCg claims vs source: all PASS.
- Findings: Kodak path = YCoCg->Bounds[+Buckets]->MANIAC (no palette); 3 fixed
  predictors (avg/2 medians, non-causal interlaced); per-plane learned forest
  (cost-gated grow ~64b + count-gated prune 50/30); Y-conditional chroma ranges;
  24-bit binary RAC, single-scale fast states (multiscale compiled OUT);
  bit-plane alphabet with impossible-bit elision.
- Texture-win thesis (3 taxes we pay): group-worst-case alphabets, one histogram
  for smooth+texture pixels per group, causal-only prediction (~7% sum).
- Ranked ports: #1 Y-cond ranges (NEW, cheapest), #2 alphabet elision (NEW),
  #3 xch-activity (corroborates Attack-2), #4 centred predictor (NEW),
  #5 binary adapt (skepticism-high post-b22), #6 micro-tree (conditional).
  Non-rec: Buckets/Palette wholesale. Challenges to takedown: RCT-search prize
  overstated; endgame = global activity-keyed contexts, not bigger tiles;
  Weighted arm unnecessary for this gap; smooth flips are cheaper half of gap.
# Cycle 51 (FLIF quest, one-by-one i: palette DEAD + #1/#2 analytic kill) — 2026-09-08
- [DEAD b25] palette oracle (top-N + ESC-as-symbol, exact Huffman, N-sweep to
  full-U): loses all 7 (02 best case +105KB vs CROWN6, +188KB vs FLIF bar).
  Near-uniform indices over k-symbols + palette side + raw escapes >> residuals
  on photos. Matches FLIF's own 512-cap veto. probes/probe_b25_*.
- [KILLED analytic, no run] teardown #1/#2: Δ≡0 for explicit-occurrence cost
  models (all backend costs depend only on occurring symbols; Y-restriction
  removes zero-count symbols only; search argmins unchanged). Transfers only
  to range-declared bit-plane coders. Caveat logged (grouping second-order: none).
- 02 insight: FLIF wins 02 palette-less -> smooth-field machinery, not color
  counting; sub-1-bit-zero door already closed at exact level (CROWN5 static
  21/21 over A-act/C-gol with real C decoder, src/libcrown5.so).
- Quest state vs FLIF (-0.23 avg): palette 0, #1/#2 0, #5 dead (CROWN5), tail 0
  (b23), headers ≤37% (b24). Untested: #3 xch-activity, #4 centred predictor,
  #6 micro-tree (all small-scale). Recommend: bank + redirect (DCC Oct 2).
# Cycle 52 (QOI measured: 7th KO at -35.0%) — 2026-09-08
- Built survey/qoi/qoiconv.c (+ vendored stb headers in /tmp, NOT repo) ->
  QOI Kodak-7 4.9197 avg (worse than PNG-9 4.75 and literature 4.66; subset
  skew + encoder variance), pixel-PASS 7/7, kodim23 enc 12ms/dec ms-scale.
- CROWN6 vs QOI 7W-0L W=0 p=.016, margin exactly 35.0%. Counted as 7th boss
  (rule is rule); QOI keeps the speed axis honestly (F5 tells both directions:
  QOI ~33MP/s enc vs HAPRE-C 9.76 vs CROWN6 ~0.007).
- Draft/README/RESULTS/AGENTS/HANDOVER to seven dead; VERIFY §5 appended.
  Ledger survey/bosstakedown/data/classical7.json (QOI row).
# Cycle 53 (DCC figure suite F1-F5, data-backed) — 2026-09-08
- Built docs/figures/make_figures.py -> F1 ladder, F1b per-image heatmap,
  F2 lineage, F3 ceiling-LOO, F4 negative atlas, F5 ratio-decode Pareto
  (PDF+300dpi RGB PNG+CSV+manifest.json each). No F6 (no banked RD outputs).
- Measurements for F5: QOI qoibench enc 3.9ms/dec 3.4ms avg; djxl e9 ~0.1s;
  PIL PNG dec 12.7ms; ours CROWN6 kodim23 123ms (range bar 182-348ms drawn).
- Review catches fixed pre-commit: duplicate legend color, 2dp label collision
  (CROWN4/6), F5 label clipping + cluster overlap, F3 '+-0.0000', F4 byte-LZ
  wrong value (+1.84 fabricated reads -> +20.7 vs MOE-champ, baselines column
  added), PNG alpha flatten. All inspected at render size.
# Cycle 54 (SIMD + compiler limits: ~14x total) — 2026-09-08
- Compiler first: `make fast` (-O3 -march=native, separate objects/binary,
  portable untouched, -ffp-contract=off kept): 53.6s -> 28.1s full-23 (1.9x),
  sha-identical. Integer costing exact under vectorization by construction.
- AVX2 i32 fused-Golomb built, measured, REVERTED: 199s -> 192s CPU (~4%).
  Why: autovectorizer already captures the fused loop; hand kernel trades 8
  sweeps for cleaner ALU (memory-bound residual) + dual paths + untestable
  n-guard fallback. d=0-loop skip KEPT (proven dead: gd+4+3 vs best<=gd0+4).
- Pruning shown impossible (any exact argmin needs all costs; bounds vacuous:
  min possible tot n+7 vs typical best ~1.5n rarely prunes).
- Gate: full-7 fast bytes == goldens (486346..403508, avg 3.1978), portable
  sha-identical on crops. Final source re-verified full-7 post-revert at
  ~31s/img (AVX2 build was ~28s; scalar ~31s). 432s -> ~31s = ~14x. Quest DONE
  (target ~60s, beaten ~2x). Lesson: measure the autovec baseline before
  hand-intrinsics; revert fast when <1.3x (complexity budget discipline).
- Process note: a mid-session workdir mix-up (make run from repo root instead
  of cpp/) briefly tested stale binaries; caught by timestamp/md5 check,
  clean-rebuilt from current source and re-gated full-7 before commit. Lesson:
  always `make` in `cpp/`; gate on md5s, not assumptions.
# Cycle 55 (camera-ready sprint i: ledgers + LaTeX + BibTeX) — 2026-09-08
- Ledgers: perimage7.csv (PNG-9/WebP rows filled in Table 4.3; pins confirm:
  4.7457->4.75, 3.5971->3.60, m3 3.347, m6 3.3157 exact), run7.txt (RUN row
  3.5114->3.511 all PASS). 4 secondary-only figures softened to report-level
  with explicit labels (byte-LZ, wavelet+zlib, LOCO-lite, G-bias).
- LaTeX: official dccpaper.cls+IEEEbib.bst (site AuthorKit) in paper/;
  paper.tex compiled clean: 0 errors, 0 Type-3, 1 sub-point overfull, 10pp
  exactly. Fixed en route: booktabs install (user-tree), arXiv @misc entries,
  author-brace tabular bug, JXL col drop, \S->Type3 bitmap (two sizes!),
  section numbering (cls starred variants), title break, table p-cols/sizes,
  refs sloppy-group, F3 -0.0000 snap. Figures embedded as PDF (Type42/CID).
- BibTeX: 10 entries, all from Crossref/arXiv APIs (spot-checked); CALLIC as
  AAAI proceedings; HPAC/SEEC/P2-LLM as arXiv misc.
- Draft deltas: boss table +7.1% FLIF row verified; per-image Table 5 (6-col);
  learned Table 2.2 with compute footnotes; Pareto F5.
# Cycle 56 (float placement + F1b cut) — 2026-09-08
- Fixed refs-split-around-figure (F3 amid refs [1-7]/[8-10]): dropped F1b
  heatmap from paper (duplicates Table 5; kept for slides), shrank all floats
  ~15%, [h]->[ht], \clearpage before refs. Map now clean (Figs 1,2 p4-5;
  Tabs 4-5 p6; Fig3+Conclusion p7; Fig4+Tab6 p8; Tab7 p9; refs p10), 10pp.
- F2 lineage never \included (redundant with Table 3; slides only).
