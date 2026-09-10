# probe_b12 RESULTS — histogram SHARING (JXL's core trick), measured honestly

Branch B12 (exploratory). Question: tables cost ~2–16% of the stream depending on
regime — does merging similar distributions into shared tables pay NET after
counting extra data bits + saved tables + the cluster-map side info?
Short answer: **YES — −0.0415 bpp (−1.20%) avg, 7/7 images, tables 59.9→18.4/image.**

Harness: `probe_b12_histshare.py` (numpy+PIL only, CPU, ~52 s). New files only,
nothing existing modified.
UNIT: `bpp = total_bits/(H*W*3)`. Exact counting everywhere:
per-cluster `huff_data + 16+A*24` (Huffman) or `entropy + 16+A*32` (rANS-proxy),
plus per-channel map `nkeys*ceil(log2(C))` bits, plus 64 b global header.

## 1. Anchors (exact, this harness)

| image | MED order-0 Huffman | expected | verdict |
|---|---|---|---|
| kodim01.png | 3.6179 | 3.62 | PASS |
| kodim02.png | 3.3256 | 3.33 | PASS |
| kodim05.png | 4.0119 | 4.01 | PASS |
| kodim07.png | 3.1530 | 3.15 | PASS |
| kodim13.png | 4.2378 | 4.24 | PASS |
| kodim19.png | 3.5134 | 3.51 | PASS |
| kodim23.png | 3.1880 | 3.19 | PASS |
| **AVG** | **3.5782** | **3.58** | **PASS (7/7)** |

LOCO-clustered anchor (3.34): our MED-only quantile arm sits HIGHER by
construction (MED-only K=24 grouped = 3.4479; K=9 = ~3.50 — no predictor
selection). CROWN-lite anchor (LOCO-quant K=9/ch + per-group best-of-7
predictor, same harness): **3.3893 avg** (01:3.4986, 02:3.1563, 05:3.7662,
07:2.9270, 13:4.1291, 19:3.3208, 23:2.9273) — within ±3% of 3.34
(range 3.24–3.44), PASS. Debug note (not a failure): the 3.34 reference needs
per-group predictor selection; the sharing delta below is measured MED-only so
that unmerged-vs-merged is same-residual honest. Transfer math to the CROWN arm
is done in §5 with stated discount.

## 2. Main result — greedy sharing vs unmerged (Huffman, MED + LOCO-365)

Atoms: per-channel LOCO-quantile groups (K0=24/ch; greedy-overshoot gives
55–62 alive tables/image, avg 59.9 — the 60–100 regime; huge flat keys consume
multiple shares in one group, same phenomenon as cycle-8 — normal, not a bug).
Rule G: greedy-best-first all-pairs agglomeration, exact total gates every
merge. Rule J: Jensen–Shannon-distance order, multi-pass, exact-gated.
Winner W = min(G,J) per image.

| image | RAW-perkey | U-GRP (unmerged) | G (greedy) | J (JS-order) | WIN | Δ(W−U) | % | tables b→a | merges G/J | rule |
|---|---|---|---|---|---|---|---|---|---|---|
| kodim01.png | 4.0761 | 3.5348 | 3.4890 | 3.4908 | 3.4890 | −0.0458 | −1.30% | 60→17 | 43/45 | G |
| kodim02.png | 3.6055 | 3.1868 | 3.1604 | 3.1631 | 3.1604 | −0.0264 | −0.83% | 61→20 | 41/43 | G |
| kodim05.png | 4.5685 | 3.8303 | 3.7781 | 3.7801 | 3.7781 | −0.0522 | −1.36% | 62→18 | 44/42 | G |
| kodim07.png | 3.3997 | 2.9615 | 2.9399 | 2.9405 | 2.9399 | −0.0216 | −0.73% | 55→19 | 36/36 | G |
| kodim13.png | 5.0368 | 4.2091 | 4.1237 | 4.1297 | 4.1237 | −0.0854 | −2.03% | 62→16 | 46/46 | G |
| kodim19.png | 3.8943 | 3.4024 | 3.3643 | 3.3657 | 3.3643 | −0.0381 | −1.12% | 61→18 | 43/45 | G |
| kodim23.png | 3.4021 | 3.0102 | 2.9891 | 2.9915 | 2.9891 | −0.0211 | −0.70% | 58→21 | 37/38 | G |
| **AVG** | **3.9976** | **3.4479** | **3.4064** | **3.4088** | **3.4064** | **−0.0415** | **−1.20%** | **59.9→18.4** | **290 tot** | **G 7/7** |

Merge statistics: 290 merges total (41.4 saved tables/image); avg net gain
**~1182 bits (~148 B) per merge**; G beats J on 7/7 but only by ~0.0024 avg —
JS-order captures ~95% of the gain and is the cheaper encoder (recommendation:
ship hybrid = JS shortlist + exact-greedy on top-T).
J alone also beats U on 7/7 (3.4088 vs 3.4479): the verdict does not depend on
the rule.

Where the bits go (avg split, exact):

| arm | data | tables | map |
|---|---|---|---|
| U-GRP | 3.3676 | 0.0760 (2.21%) | 0.0042 |
| WIN (merged) | 3.3775 (+0.0099) | 0.0263 (0.77%) | 0.0025 |
| ledger | extra data **+0.0099** | saved tables+map **+0.0514** | net **−0.0415** |

MDL-rent rule obeyed with 5:1 return: similar contexts truly are similar
(extra data only +0.3%), and the map side is negligible (0.004→0.002 bpp —
the "which table" fear was unfounded at fixed-length coding; JXL's context-map
machinery buys little extra here).
Fragmentation showcase: RAW per-key tables burn **~0.6 bpp (~16%)**;
quantile grouping already harvests most of that (3.9976→3.4479); sharing takes
the second half of the table bill (0.076→0.026).

## 3. Stretch — rANS backend (same merge pass, entropy-proxy data + 16+A*32 tables)

| image | rANS-U | rANS-M | Δ | % |
|---|---|---|---|---|
| kodim01.png | 3.5074 | 3.4471 | −0.0603 | −1.72% |
| kodim02.png | 3.1494 | 3.1184 | −0.0310 | −0.98% |
| kodim05.png | 3.8177 | 3.7513 | −0.0664 | −1.74% |
| kodim07.png | 2.9008 | 2.8773 | −0.0235 | −0.81% |
| kodim13.png | 4.2166 | 4.1018 | −0.1148 | −2.72% |
| kodim19.png | 3.3795 | 3.3312 | −0.0483 | −1.43% |
| kodim23.png | 2.9676 | 2.9452 | −0.0224 | −0.75% |
| **AVG** | **3.4199** | **3.3675** | **−0.0524** | **−1.53%** |

Sharing pays MORE under rANS (−1.53% vs −1.20%) because tables are pricier
(32 b/symbol vs 24 b). Caveat: data = Shannon-entropy proxy, optimistic
~0.5–1% vs a real rANS payload (prior probes used measured `rans_encode`;
treat rANS column as upper bound on the gain, Huffman column as exact).

## 4. Decoder-safety statement

Merge map transmitted per channel (`nkeys*ceil(log2(C))` bits, counted in
every total above; C itself negligible, <0.0001 bpp). Decode: read C → map →
tables; per pixel, LOCO-365 key is re-derived from already-decoded causal
recon (keys depend only on recon, never on the map — no circularity), key→map→
merged table → symbol → MED-add → recon. Checks: every winner partition covers
each channel's active keys exactly once (asserted); all codebooks satisfy Kraft
= 1 (asserted); canonical symbol round-trip PASS 7/7 on winner maps; literal
bitstream pack→unpack→decode PASS on kodim01 (1,179,648 syms, 4,085,294 bits)
and kodim07 (1,179,648 syms, 3,438,391 bits); YCoCg-R inverse asserted all 7.
Decode uses merged tables exclusively — no unmerged state retained.

## 5. Trail + verdict + follow-up

Trail: anchors reproduced (order-0 7/7 PASS; CROWN-lite 3.3893 explains the
3.34 reference within tolerance) → atoms K0=24 land in the 60–100-table regime
(59.9 alive after documented greedy-overshoot) → G and J both win 7/7 vs U →
rANS stretch widens the win → round-trips pass. No dead-ends; one labeling fix
(script prints "bits/merge" where the value is bytes/merge — numbers above are
corrected: ~1182 bits ≈ 148 B/merge).

Verdict: **sharing pays net — YES, −1.20% Huffman-exact (−1.53% rANS-proxy),
7/7, 5:1 MDL return.** Boss-5 math (honest, discounted): JXL-e3 = 3.2291 vs
CROWN2-exact 3.2686 needs −1.22%. Naive transfer (−1.20% × 3.2686 → ~3.229,
exact parity) is OVER-optimistic: CROWN2's WIDE-K hillclimb + 3-way selection
already leave fewer/worse tables than this MED-only arm, so expect a fraction
(~−0.3…−0.7%) — sharing alone does NOT kill Boss-5, but it is a stackable
contributor (rANS-priced tables make it worth more in the exact-C build, where
it counts most). e9 (3.03) still needs the transform-bitplane arc regardless.

Follow-up (ranked):
1. Port the merge pass onto the CROWN2 arm (post-3-way per-group histograms,
   rANS costs driving merges — the backend where sharing pays most), measure
   the discounted transfer exactly.
2. Cross-channel sharing (this probe clusters per-channel; JXL shares across
   all contexts — joint clustering may find Y/Co/Cg twins).
3. Hybrid encoder order (JS shortlist → exact-greedy top-T) + MA-tree-lite
   leaves (cycle-11) as merge atoms instead of quantile groups — the full
   "learned splits + shared histograms" JXL shape.
