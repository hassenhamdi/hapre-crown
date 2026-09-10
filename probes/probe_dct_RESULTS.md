# probe_dct RESULTS — reversible integer-DCT front-end vs HAPRE-C MOE

Method: RGB->YCoCg-R (floor-shift, round-trip asserted) + H.264 4x4 integer DCT `Cf*block*Cf^T` (exact rational inverse asserted on 200 random blocks before measuring) + EXACT order-0 Huffman per subband via heapq (`sum(count*len) + 16 + A*24` table bits, 48 streams) + 64B header. numpy+PIL only, CPU.

| image | HxW | DCT bpp | MOE bpp | delta (DCT-MOE) | fwd ms | data bits | table bits |
|---|---|---|---|---|---|---|---|
| kodim01.png | 512x768 | 6.9637 | 3.56 | +3.4037 | 64.8 | 7621335 | 592872 |
| kodim02.png | 512x768 | 6.5820 | 3.23 | +3.3520 | 61.9 | 7279236 | 484728 |
| kodim05.png | 512x768 | 7.5470 | 3.84 | +3.7070 | 61.0 | 8076921 | 825312 |
| kodim07.png | 512x768 | 6.5072 | 3.01 | +3.4972 | 62.1 | 7099086 | 576648 |
| kodim13.png | 512x768 | 7.7089 | 4.17 | +3.5389 | 61.0 | 8313946 | 779280 |
| kodim19.png | 768x512 | 6.7952 | 3.41 | +3.3852 | 62.9 | 7425237 | 590208 |
| kodim23.png | 512x768 | 6.5646 | 3.03 | +3.5346 | 61.7 | 7101391 | 642024 |

Average DCT bpp: **6.9527**. MOE champion avg: **3.464**. Delta (DCT-MOE): **+3.4887 bpp** (WORSE than MOE).

JXL-e9 reference: 3.03 avg (final boss, not beaten by either).

Avg numpy forward-transform time: **62.2 ms/image** (median of 5 einsum runs, 3 channels, 512x768, int64).

## Verdict

No — transform direction as probed does NOT beat 3.464 bpp. It is 3.4887 bpp worse on average, and worse on 7/7 images. Causal prediction (MED/MoE) remains champion.

## Honest limitations

- Order-0 Huffman per subband only: no run-length / run-mode, no context/adaptive coding, no bit-plane or significance-map coding, no DC DPCM across blocks.
- Unnormalized H.264 integer core inflates coefficients (row gains 4/10); order-0 code pays ~log2(gain) extra bits/coeff vs an orthonormal DCT. No scaling compensation applied.
- No quantization (lossless) so energy compaction does not reduce symbol counts; DC subband keeps ~10-11 bits/symbol entropy.
- Fixed 4x4, no adaptive block size, no directional modes, no chroma handling beyond YCoCg-R.
- Table cost counted as 16+A*24 bits per stream (48 streams); real container overhead differs.
- Single-symbol edge case costed as 0 data bits + table (never hit on these images).
- Timing is numpy einsum forward only; excludes YCoCg-R, Huffman counting, and I/O.