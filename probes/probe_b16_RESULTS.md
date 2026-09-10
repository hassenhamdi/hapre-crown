# probe_b16 RESULTS — HYBRID pixel/transform (Y-pixel + chroma-wavelet)

Code: `experiments/probe_b16_hybrid.py` (numpy+PIL only, CPU). Reuses proven
modules by import: `probe_b4_a` (YCoCg-R, champion-rule MED + CTX9) and
`probe_b15_wavelet` (LeGall 5/3 lifting + exact Huffman). No existing files touched.
UNIT: bpp = total_bits/(H·W·3). Exact counting: heapq Huffman data + `16+A·24`
per table (A==1 → 0 data bits, table still counted; empty streams 0), +9b presence
mask per grouped stream, +1b LL selector, +2b per detail-subband selector, +3b oracle
channel selector, +64b global header. Split is dyadic-subband + fixed channel
assignment → NO spatial/region map to transmit (only the counted selectors above).

## Configs (all exact-counted, decode-verified)

| cfg | Y | Co/Cg |
|---|---|---|
| P0 (anchor) | MED + order-0 Huffman | same |
| P9 (pixel) | MED + CTX9-grouped Huffman | same |
| T-ALL | 1-level 5/3, LL min(o0,CTX9)+1b; details min(o0,sparse,subMED)+2b | same |
| H-CHROMA (hybrid) | P9 pixel | T arm (fixed design, 0 selector bits) |
| H-ORACLE | per-channel min(P9, T) + 3b channel selector (blending upper bound) | — |

`sparse` = significance bitmap (binary Huffman, count=H·W known) + Huffman on
nonzeros (count = popcount, known after map). `subMED` = raster MED inside the
subband + Huffman. L=1 only (b15: L2/L3 over-fragment).

## Per-image results (bpp)

| img | P0 | P9 | T-ALL | H-CHROMA | H-ORACLE | JXL-e3 |
|---|---|---|---|---|---|---|
| kodim01 | 3.6179 | 3.5954 | 3.5001 | 3.4163 | 3.4163 | 3.3593 |
| kodim02 | 3.3256 | 3.2961 | 3.1031 | 3.0527 | 3.0527 | 3.0611 |
| kodim05 | 4.0119 | 3.9236 | 3.7192 | 3.6370 | 3.6370 | 3.5120 |
| kodim07 | 3.1530 | 3.0967 | 2.9585 | 2.8988 | 2.8988 | 2.7310 |
| kodim13 | 4.2378 | 4.2069 | 4.0481 | 4.0000 | 4.0000 | 3.9141 |
| kodim19 | 3.5134 | 3.4791 | 3.4252 | 3.3713 | 3.3713 | 3.2154 |
| kodim23 | 3.1880 | 3.1456 | 2.9440 | 2.9378 | 2.9378 | 2.8110 |
| **avg** | 3.5782 | 3.5348 | 3.3855 | **3.3306** | **3.3306** | 3.2291 |

## Deltas

- vs classical ceiling 3.2515 (avg only; no per-image ceiling published):
  P9 +0.2833 (+8.7%), T-ALL +0.1340 (+4.1%), **H-CHROMA +0.0791 (+2.43%)**.
- H-CHROMA − JXL-e3 per image: 01 +0.0570 (+1.70%), 02 **−0.0084 (−0.27%, sole win)**,
  05 +0.1250 (+3.56%), 07 +0.1678 (+6.14%), 13 +0.0859 (+2.19%), 19 +0.1559 (+4.85%),
  23 +0.1268 (+4.51%). **Avg +0.1015 (+3.14%), 1W-6L.**
- Hybrid vs family: H-CHROMA − P9 = −0.2042 (−5.8%); − T-ALL = −0.0549 (−1.6%);
  − b15-C (3.3737) = −0.0431 (−1.3%); − b15 C-L1 (3.4310) = −0.1004 (−2.9%).
  Best transform-family number measured, still short of the ceiling.
- Boss-5 math: tying JXL-e3 needs **−3.05%** from the hybrid; final boss JXL-e9
  (3.03) needs **−9.0%**. No measured within-family gradient exceeds −1.6%
  (and that one — pixel→hybrid — is already harvested).

## Mechanism

1. **Channel asymmetry is real and unanimous: H-ORACLE == H-CHROMA on 7/7.**
   Y-pixel beats Y-transform on every image (smooth-field predictability:
   Y-pix 1.26–2.08 vs Y-arm 1.27–2.13 per-image-channel bpp); Co/Cg-transform
   beats Co/Cg-pixel on every image (LL energy compaction wins). The oracle's 3b
   selector buys nothing — the assignment is image-independent. b15's diagnosis
   (transform scatters smooth Y predictability) is confirmed constructively:
   keeping Y in the pixel domain recovers ~0.05 bpp vs full-transform.
2. **Inside detail bands, joint order-0 beats everything, 27/27 subbands.**
   Y details are 85–86% nonzero → sparse loses by +0.034–0.037/subband,
   subband-MED loses by +0.047–0.069 (lifting already decorrelated; MED on
   details re-predicts noise). Chroma HL/LH are 52–58% nonzero — the sparsest
   bands Kodak has — yet sparse only *ties* order-0 (+0.0000–0.0022) and never
   wins after its own 2b selector; Cg-HH (78% nonzero) loses by +0.020. Lesson:
   at ≥50% density the significance map costs ~1b/coeff and joint (magnitude,sign)
   Huffman wins by construction — same verdict as b15-B, now with the density
   threshold mapped: sparsity coding needs bands far sparser than 5/3 gives Kodak.
3. **LL likes grouping (g9 wins 21/21 LLs)** — LL is just a small natural image,
   pixel machinery transfers intact. The hybrid's entire gain over T-ALL comes
   from the Y channel choice, not from smarter detail coding (detail best-of
   contributes ~0: o0 sweeps all).

## Trail

- Built channel-asymmetric arms on reused proven modules (b4 MED/CTX9, b15
  lifting/Huffman); vectorized subband-MED + map/nz sparse arms added.
- First run: anchor P0 = 3.5782 vs 3.58 (−0.05%) → PASS, no debug detour.
- Ledger showed o0 sweeping all 27 detail subbands and oracle==fixed-design 7/7 —
  no follow-up configs needed; family gradient exhausted by construction.

## Verdict

**HYBRID NEGATIVE for Boss-5: 3.3306 vs ceiling 3.2515 (+2.4%), vs JXL-e3 +3.1%
(1W-6L).** It is the best transform-family build (beats b15-C by −1.3%) and proves
the channel-asymmetric structure (Y-pixel / chroma-wavelet, unanimous 7/7), but
the remaining gap needs −3.05% with all measured gradients ≤1.6% and harvested.
Classical pixel machinery (3.2515) stays champion; the hybrid's transform half
cannot pay for what the wavelet does to Y — even when Y is exempted, chroma-HF
sparsity (≤58% zeros in the best bands) is too thin to monetize.

## Follow-up (ordered)

1. **Retire the hybrid/subband arc** (this probe + b15 exhaust it: L1/L2/L3,
   order-0/conditioned/bitplane/sparse/subband-MED, pixel/chroma splits — all
   measured, best 3.3306). No untested subband shape remains.
2. **lf-prediction stays the last structural weapon** (predict HF from decoded LF
   with integer weights — attacks smooth-field loss directly; untouched by b15/b16).
3. Otherwise Boss-5/9 need learned entropy (CALLIC/HPAC lineage) or MA-tree splits,
   not transforms. Consolidation target unchanged: classical exact 3.2515.

Bit-exactness: YCoCg-R round-trip + 5/3 fwd/inv per channel + MED pred+res
inversion (full-res, LL, subband arms) + sparse map/nz rebuild asserted every
image (7/7 PASS — completion implies all asserts held). Decoder-safety: S1
Y→Co→Cg; pixel arm raster MED (0/left/top); transform arm LL-raster then
HL,LH,HH raster; sparse map (count from geometry) then values (count from
popcount); selectors in header. All contexts decodable-before-use.
