# Probe: horizontal band-adaptive order-0 Huffman vs global tables (net of overhead)

Branch 2, self-contained. Question: does splitting the image into horizontal bands with
SEPARATE entropy tables beat global tables (nonstationarity capture), after counting
all extra table overhead?

## Method (as specified; numpy + PIL only, CPU, no torch)

1. Load RGB PNG; cast to int32.
2. YCoCg-R reversible integer transform (floor `//2` for negatives):
   `Co=R-B; t=B+(Co//2); Cg=G-t; Y=t+(Cg//2)`.
   Inverse `t=Y-(Cg//2); B=t-(Co//2); G=Cg+t; R=Co+B`.
   Exact-invertibility asserted every image: PASS on all 7.
3. MED residuals per channel, vectorized numpy, edge-replicate pad
   (`np.pad(ch,((1,0),(1,0)),mode='edge')`; `a=left, b=top, c=diag`;
   `p=min(a,b) if c>=max(a,b); max(a,b) if c<=min(a,b); else a+b-c`; `res=x-p`).
   Vectorized output verified bit-exact vs naive double-loop MED on 32x32 crop (max abs diff 0).
4. Baseline: global order-0 Huffman per channel with REAL heapq code lengths.
   `data_bits = sum of merged weights` (== `sum(count*len)`; toy `[5,2,1]->11` verified).
   Single-symbol table convention: 1 bit/symbol (never triggered; all A>>1).
   Per-table cost `16+A*24` bits; total `= sum_channels(data+16+A*24) + 64` (8 bytes dims).
5. Band-adaptive: `np.array_split(residual, B, axis=0)` per channel (B=2 and B=4 separately);
   separate Huffman table per (channel, band), each paying its own `16+A*24`.
   Total `= sum_{ch,band}(data+16+A*24) + 64`. Band boundaries are deterministic from H,B:
   no extra signalling bits beyond the counted dims bytes.
6. UNIT RULE throughout: `bpp = total_bits/(H*W*3)`.

Script: `experiments/probe_bands.py`. Torch not used (torch not even installed).

## Sanity anchor: MISS (investigated, blocking discrepancy documented)

- Spec anchor: baseline avg should be ~4.4-4.6 bpp (order-0 MED, no contexts); debug if outside 4.0-5.0.
- Observed baseline avg: **3.5769 bpp** — outside window. Debug performed before reporting band deltas:
  - Huffman verified on toys; Huffman-vs-entropy gap on real data is +0.03 (Y) / +0.06-0.08 (Co/Cg),
    exactly the expected Huffman redundancy. No undercount.
  - MED verified vs loop (exact match). YCoCg-R inverts exactly on all 7.
  - Cross-checks: RGB-only MED order-0 (no transform) avg = **4.964 bpp** (edge of window, above anchor
    center); YCoCg-R MED order-0 avg = 3.577. The ~1.39 bpp drop is the cross-channel decorrelation
    gain: typical MED residual entropy Y~5.5, Co~2.6, Cg~2.6 bits (kodim01) vs RGB ~5.5/channel.
    Mean `(5.51+2.56+2.59)/3=3.55` + Huffman/table overhead = 3.62 observed. Internally consistent.
  - PNG file sizes bracket the result: 7-image PNG avg = 4.61 bpp; our 3.58 beats PNG by ~1.0 bpp,
    plausibly explained by YCoCg-R (PNG has no cross-channel decorrelation; its filters ~ MED).
  - Conclusion: implementation is correct; the **anchor appears miscalibrated** (likely estimated
    without the full YCoCg-R chroma gain, since even RGB-MED at 4.96 sits above the anchor center).
    Crucially, the anchor offset does NOT affect the research question: baseline and bands share
    identical counting, so **deltas are valid** regardless of anchor level.

## Results (bpp = total_bits/(H*W*3); deltas vs baseline, negative = band wins)

| image | HxW | baseline | B=2 | B=4 | d(B2-base) | d(B4-base) | A base (Y/Co/Cg) |
|---|---|---|---|---|---|---|---|
| kodim01.png | 512x768 | 3.6179 | 3.6111 | 3.6093 | -0.0068 | -0.0086 | 250/72/55 |
| kodim02.png | 512x768 | 3.3255 | 3.3081 | 3.3077 | -0.0174 | -0.0178 | 223/85/44 |
| kodim05.png | 512x768 | 4.0119 | 4.0062 | 4.0165 | -0.0058 | +0.0046 | 318/109/90 |
| kodim07.png | 512x768 | 3.1529 | 3.1576 | 3.1606 | +0.0047 | +0.0077 | 221/103/46 |
| kodim13.png | 512x768 | 4.2378 | 4.2348 | 4.2305 | -0.0030 | -0.0074 | 369/150/71 |
| kodim19.png | 768x512 | 3.5081 | 3.4592 | 3.4403 | -0.0489 | -0.0678 | 238/113/64 |
| kodim23.png | 512x768 | 3.1842 | 3.1751 | 3.1561 | -0.0092 | -0.0281 | 232/84/47 |
| **AVG (7)** | — | **3.5769** | **3.5646** | **3.5601** | **-0.0123** | **-0.0168** | — |

Gross-vs-overhead (bpp units; `+data_sav` = data-bits reduction, `+extra_tbl` = added table cost):
B=2 always reduces data bits (+0.001 to +0.055) at +0.005 to +0.010 table cost; B=4 reduces data
(+0.008 to +0.082) at +0.011 to +0.027 table cost. Nonstationarity is real (data never increases),
but overhead eats most of it. Wins: B=2 on 5/7 (loses kodim07, and only kodim07 cleanly; kodim05 B2 wins,
B4 loses), B=4 on 5/7 (loses kodim05/kodim07). Outlier kodim19 (portrait, sky/ground horizontal
structure aligned with bands): -0.049/-0.068, ~3-4x the mean effect.

## Verdict

**Yes, band-adaptation wins net of exact table costs — but only barely, so NO as a campaign direction.**
Average net: B=2 **-0.0123 bpp**, B=4 **-0.0168 bpp** (~0.3-0.5% relative). B=4 beats B=2 on average,
confirming more granularity captures more nonstationarity, but per-image losses (kodim07 both B;
kodim05 B=4) confirm the table-fragmentation mechanism from the naive 16-bin experiment at smaller
scale: each extra table costs ~0.005 bpp and gross savings saturate. For context, the gap from this
probe's baseline (3.577) to the champion HAPRE-C MOE (3.464) is 0.113 bpp; B=4 closes only ~15% of it.
Fixed horizontal geometry is not the lever — kill fixed-band scaling here; any follow-up must be
content-adaptive or it is not worth the tables.

## Single follow-up

**One adaptive-cut probe: per-image (or per-channel) DP-optimal single horizontal split.**
Search every candidate row boundary with the exact rate criterion
`data_bits(left)+tbl(left)+data_bits(right)+tbl(right)` (same Huffman+`16+A*24` counting, +~10 bits
to signal the cut row) and keep the split only if it beats global. This isolates whether fixed-half
geometry is the limiter (kodim19 suggests alignment is everything) at near-zero extra overhead,
and directly predicts whether any spatial partition can pay for itself before trying quads/16-bins
on top of the champion's 27 context tables.
