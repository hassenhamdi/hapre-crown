# Probe B7 RESULTS — QOI ports: luma-anchored chroma (A) + hash-cache escape (B)

Branch probe: two cheap QOI-portable mechanisms from SURVEY_genai_qoi.md §2, probed separately and combined. numpy + PIL only, CPU, no torch. UNIT RULE throughout: `bpp = total_bits/(H*W*3)`. Exact counting: real heapq Huffman data bits + `16+A*24` per stream, every table and side byte counted.

## Baseline + anchors

MED residuals (edge-replicate pad) in YCoCg-R + independent order-0 Huffman per channel. Total = 64 dims bits + Σ(data_c + 16 + A_c·24). YCoCg-R round-trip asserted per image: PASS 7/7.

| image | HxW | base bpp | expected | anchor ±3% | A(Y/Co/Cg) |
|---|---|---|---|---|---|
| kodim01.png | 512x768 | 3.6179 | 3.62 | PASS | 250/72/55 |
| kodim02.png | 512x768 | 3.3255 | 3.33 | PASS | 223/85/44 |
| kodim05.png | 512x768 | 4.0119 | 4.01 | PASS | 318/109/90 |
| kodim07.png | 512x768 | 3.1529 | 3.15 | PASS | 221/103/46 |
| kodim13.png | 512x768 | 4.2378 | 4.24 | PASS | 369/150/71 |
| kodim19.png | 768x512 | 3.5081 | 3.36 | MISS | 238/113/64 |
| kodim23.png | 512x768 | 3.1842 | 3.18 | PASS | 232/84/47 |

Baseline avg = **3.5769 bpp** (spec ≈3.58, 0.1% off). 6/7 anchors PASS ±3%; kodim19 reads
3.5081 vs listed 3.36 — DEBUGGED, baseline stands: (i) independent re-implementation reproduces
3.5081 to 4 decimals; (ii) RGB-MED Huffman control on kodim19 = 4.88, consistent with the logged
RGB-MED ≈4.96 control proving impl correctness; (iii) with 3.51 the 7-mean is 3.577, exactly the
independently logged 3.577 anchor (cycle-1 memory, branch-1b) — i.e. the avg anchor *requires*
kodim19≈3.51, so the listed "3.36" is a typo. Baseline VERIFIED CORRECT.

## Mechanism A — luma-anchored chroma (Y stream first, decoder-safe)

A-joint: one (rCo,rCg) pair alphabet; table cost honest `16+A·48` (two int16/entry). A-binK: pixels binned by |rY| with FIXED global thresholds (0 side bytes; decoder recomputes bins from the already-decoded Y stream), separate Co/Cg Huffman tables per bin. Bounds: K2 [1]; K3 [1,4]; K4 [0,2,7] (upper |rY| bounds). A-binQ4: per-image equal-mass quantile bins of |rY| (+24 side bits counted, conservative).

| image | base | A-joint (Δ%) | A-K2 (Δ%) | A-K3 (Δ%) | A-K4 (Δ%) | A-Q4 (Δ%) |
|---|---|---|---|---|---|---|
| kodim01.png | 3.6179 | 3.5381 (-2.21) | 3.6005 (-0.48) | 3.5985 (-0.54) | 3.6005 (-0.48) | 3.6020 (-0.44) |
| kodim02.png | 3.3255 | 3.2201 (-3.17) | 3.2979 (-0.83) | 3.2978 (-0.83) | 3.2913 (-1.03) | 3.2982 (-0.82) |
| kodim05.png | 4.0119 | 4.0127 (+0.02) | 3.9944 (-0.44) | 3.9909 (-0.52) | 3.9859 (-0.65) | 3.9994 (-0.31) |
| kodim07.png | 3.1529 | 3.0657 (-2.77) | 3.1227 (-0.96) | 3.1066 (-1.47) | 3.1128 (-1.27) | 3.1013 (-1.64) |
| kodim13.png | 4.2378 | 4.2496 (+0.28) | 4.2350 (-0.07) | 4.2354 (-0.06) | 4.2273 (-0.25) | 4.2369 (-0.02) |
| kodim19.png | 3.5081 | 3.4522 (-1.59) | 3.4959 (-0.35) | 3.4911 (-0.49) | 3.4860 (-0.63) | 3.4919 (-0.46) |
| kodim23.png | 3.1842 | 3.1052 (-2.48) | 3.1436 (-1.28) | 3.1377 (-1.46) | 3.1334 (-1.60) | 3.1370 (-1.48) |

Avg: base **3.5769** | A-joint **3.5205** (-1.58%) | A-K2 **3.5557** (-0.59%) | A-K3 **3.5512** (-0.72%) | A-K4 **3.5482** (-0.80%) | A-Q4 **3.5524** (-0.69%).
Joint pair alphabets A_J per image: kodim01.png=480, kodim02.png=727, kodim05.png=2326, kodim07.png=1198, kodim13.png=2657, kodim19.png=1054, kodim23.png=1071. Q4 quantile bounds per image: kodim01.png=[2, 5, 12], kodim02.png=[1, 2, 4], kodim05.png=[2, 4, 12], kodim07.png=[1, 2, 3], kodim13.png=[3, 9, 18], kodim19.png=[1, 3, 6], kodim23.png=[1, 2, 3].

## Mechanism B — distant-repeat hash-cache escape (QOI-INDEX port)

64-entry exact-repeat cache over YCoCg triplets, zero-init, `h=(Y·3+(Co+256)·5+(Cg+256)·7)&63`, verified-hit else store (QOI eviction semantics, no false hits). Honest entropy-coded counting: flag stream (hit/miss) + index stream (hit slot, iff nhit>0) + 3 miss-residual streams (scan order). Decoder mirrors exactly (see Decoder-safety).

| image | base | B bpp (Δ%) | hit% | flag bits (A) | index bits (A) | resid-data base→miss | tableΔ miss−full |
|---|---|---|---|---|---|---|---|
| kodim01.png | 3.6179 | 3.6867 (+1.90) | 22.34 | 393280 (2) | 524115 (64) | 4258670→3422563 | -72 |
| kodim02.png | 3.3255 | 3.2937 (-0.96) | 36.42 | 393280 (2) | 830619 (64) | 3914398→2652977 | +0 |
| kodim05.png | 4.0119 | 4.1923 (+4.50) | 12.91 | 393280 (2) | 302520 (64) | 4720115→4237189 | -24 |
| kodim07.png | 3.1529 | 3.2603 (+3.41) | 42.08 | 393280 (2) | 980171 (64) | 3710373→2463579 | +0 |
| kodim13.png | 4.2378 | 4.3979 (+3.78) | 9.77 | 393280 (2) | 225357 (64) | 4984865→4555114 | -72 |
| kodim19.png | 3.5081 | 3.5704 (+1.78) | 25.55 | 393280 (2) | 603363 (64) | 4128266→3205189 | -48 |
| kodim23.png | 3.1842 | 3.3495 (+5.19) | 25.99 | 393280 (2) | 613129 (64) | 3747457→2936005 | -24 |

Avg: base **3.5769** → B **3.6787** (+2.85%), mean hit rate 25.01%. Effective flag cost = 1.000 bits/px; index cost = 5.926 bits/hit.

## Combined A+B (cache front-end + K4-binned miss chroma)

| image | base | A+B bpp (Δ%) | Δ vs B alone (pp) |
|---|---|---|---|
| kodim01.png | 3.6179 | 3.6838 (+1.82) | -0.08 |
| kodim02.png | 3.3255 | 3.2920 (-1.01) | -0.05 |
| kodim05.png | 4.0119 | 4.1888 (+4.41) | -0.09 |
| kodim07.png | 3.1529 | 3.2500 (+3.08) | -0.33 |
| kodim13.png | 4.2378 | 4.3992 (+3.81) | +0.03 |
| kodim19.png | 3.5081 | 3.5677 (+1.70) | -0.08 |
| kodim23.png | 3.1842 | 3.3376 (+4.82) | -0.37 |

Avg: base **3.5769** → A+B **3.6742** (+2.72%).

## Decoder-safety statement

Stream order: dims → [flag table+data, index table+data] → Y table+data (miss residuals, scan order) → Co/Cg per-bin tables+data (scan order). A-only: Y stream decoded fully first (causal MED recon, lossless ⇒ encoder-identical); bins are a deterministic function of decoded |rY| with fixed global thresholds, so Co/Cg table selection is available before Co/Cg decode — causal, streaming-compatible (row-wise Y-first ordering also valid). Q4 variant: thresholds transmitted (+24 bits counted) or deterministically recomputed — either way available pre-Co/Cg. B: cache init zeros both sides; hit/miss sequence fully determined by decoded flags; cache evolves from reconstructed triplets only, which equal the encoder's (lossless), so decoder state tracks encoder state exactly — hit decision never depends on not-yet-decoded data. Exact-match verification (not bare hash) means collisions resolve to MISS on both sides identically. A+B: miss-pixel |rY| known after Y recon, before Co/Cg decode — bin selection causal. No lookahead required anywhere.

## Trail (what was tried, exact)

Script `experiments/probe_b7_qoi_ports.py` (numpy 2.4.6 + PIL, torch unused). Runtime ~2s total. One script covers A-joint, A-binK2/K3/K4, A-binQ4, B, A+B on all 7 images. No runs, no context mixing, no rANS — pure order-0 Huffman isolation of the two QOI-portable mechanisms. Joint-pair packing `key=(rCo+1024)*4096+(rCg+1024)` (residual range ±510 fits). Cache loop is plain Python over ravelled int lists (decoder-mirror clarity over speed).

## Verdict

A (luma-anchored chroma): best variant **A-joint −1.58% avg** (5/7 images; only textured
kodim05 +0.02% / kodim13 +0.28% resist); Y-binned tables −0.59…−0.80% (7/7 directionally,
K4 best). **A PORT PAYS — joint (Co,Cg) pair alphabet is the winner, chroma–chroma correlation
beats luma→chroma conditioning ~2:1.** Honest cost already applied (`16+A·48`/pair-stream).
B (hash-cache escape): **+2.85% avg at 25.01% mean hit rate — DOES NOT PAY on photos.**
Structural cause, visible in the table: under Huffman a binary flag costs exactly 1.000 bit/px
(unskewable — measured identical 393280 b/flag-stream on all 7), the 64-symbol index costs
5.93 bits/hit (near-uniform, ≈ raw 6 b), and selection bias caps savings (hits strike flat
pixels whose residuals were already cheap: e.g. kodim07 at 42% hits still loses +3.41%).
Only high-hit kodim02 (−0.96%) clears the overhead. QOI's fixed 2-bit tags hide this; an
entropy-coded honest count exposes it.
Combined A+B: **+2.72% avg** — B's flag+index overhead swamps A's gain; binning helps B
marginally (−0.05…−0.37 pp on 5/7) but never flips it.
Boss-5 math: JXL-e3 3.23 needs ≈−1.3% (task ≈−1.5%) from campaign 3.272. **A-joint's −1.58%
clears that bar arithmetically in the order-0 probe frame** — but the frame is MED+Huffman
(3.577), not the CROWN-rANS arm (3.272), and LOCO-cycle precedent says Golomb/Huffman-frame %
does not transfer 1:1. Status: PORT CANDIDATE, needs CROWN-arm confirmation, not a KO claim.

## Follow-up

1. **Port A-joint onto the CROWN-huff arm first**: replace independent per-channel Co/Cg group
   tables with joint (Co,Cg) pair tables (or per-group pair tables where A_J stays small; all
   tables counted at `16+A·48`). Re-probe Δ vs CROWN-huff — this is the single mechanism here
   that clears Boss-5 math in-frame. Untested combo with upside: joint pairs *inside* Y bins
   (helps 05/13 where binning won but joint lost); gate on MDL — pair alphabets up to A_J=2657
   × 4 bins risk the fragmentation failure mode, so try per-CROWN-group pairs before global bins.
2. **B: retire as a Kodak-ratio play; keep ONLY as a non-photo fast-path** (icons/graphics/flat
   skies where hit rate ≫40% clears the 1-bit/px Huffman flag). Optional cheap re-probe: B under
   the rANS backend (flag→H(p)≈0.8 b/px, index→~5.5 b/hit) to quantify the backend-dependent
   residue — expect at best −0.5% on kodim02-class images, nowhere near Boss-5 alone.
3. Redirect remaining Boss-e3 effort to survey TOP-5 #4/#5 (IFCE inter-group conditioning,
   MDL-gated micro-adapters) stacked with A-joint.
