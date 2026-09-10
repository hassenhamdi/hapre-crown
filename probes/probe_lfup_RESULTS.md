# Probe LF-UP (branch 1b) RESULTS — NN vs bilinear vs smoothed-NN upsampling

Script: `probe_lfup.py` (numpy + PIL only, CPU, no torch).
Question: was Squeeze-lite's 13.5% loss caused by blocky NN upsampling (fixable)
or by the LF/HF split itself (structural)?

Method:
- YCoCg-R reversible integer transform (Co=R-B; t=B+(Co//2) floor; Cg=G-t;
  Y=t+(Cg//2)); exact round-trip asserted every image: PASS on all 7.
- Baseline per channel: MED residuals (vectorized numpy, edge-replicate pad;
  a=left, b=top, c=topleft; standard MED rule) + EXACT order-0 Huffman bits
  via real heapq code lengths (sum(count*len) + 16 + A*24 table bits per stream).
- LF: 4x floor-mean box downsample per channel (edge-replicate pad to multiples
  of 4; all 7 Kodak images are already multiples of 4 so pad is no-op).
- Variants differ only in upsampler; HF = original - upsampled; HF coded with
  identical MED + EXACT Huffman; LF side = sum over 3 channels of EXACT Huffman
  (same table rule) of downsampled LF values + 64 bits (8 bytes) dims counted
  once globally. Total = HF + side.
- (a) NN: replicate 4x4 blocks, cropped to HxW.
- (b) BILINEAR: centered integer fixed-point (see script docstring). LF sample
  treated as 4x4-block center; 1D phases (denom 8): m=0: 3/8 prev+5/8 curr;
  m=1: 1/8 prev+7/8 curr; m=2: 7/8 curr+1/8 next; m=3: 5/8 curr+3/8 next (same
  for columns); 2D weight = row_w*col_w/64; int64 numerator, floor //64;
  LF edges replicated; constant planes reproduce exactly (verified).
- (c) SMOOTHED-NN: 3x3 box blur of the NN-upsampled plane, edge-replicate pad,
  int64 3x3 sum //9 floor.
- UNIT RULE: bpp = total_bits/(H*W*3). Sanity anchor: baseline avg over the 7
  images expected ≈3.5–3.7 bpp (known MED+Huffman level).

## Per-image numbers (bpp = total_bits/(H*W*3))

| image | baseline bpp | NN total bpp | NN delta% | bilinear total bpp | bilinear delta% | smoothed total bpp | smoothed delta% |
|---|---|---|---|---|---|---|---|
| kodim01.png | 3.6178 | 4.0495 | +11.93% | 4.0184 | +11.07% | 4.0064 | +10.74% |
| kodim02.png | 3.3255 | 3.7506 | +12.78% | 3.6756 | +10.53% | 3.6649 | +10.21% |
| kodim05.png | 4.0119 | 4.5547 | +13.53% | 4.3813 | +9.21% | 4.3762 | +9.08% |
| kodim07.png | 3.1529 | 3.6551 | +15.93% | 3.5329 | +12.05% | 3.5221 | +11.71% |
| kodim13.png | 4.2378 | 4.6969 | +10.84% | 4.6211 | +9.05% | 4.6076 | +8.73% |
| kodim19.png | 3.5081 | 3.9581 | +12.83% | 3.9022 | +11.23% | 3.8939 | +11.00% |
| kodim23.png | 3.1842 | 3.7485 | +17.72% | 3.6034 | +13.17% | 3.6056 | +13.23% |
| **average (7)** | **3.5769** | **4.0591** | **+13.48%** | **3.9621** | **+10.77%** | **3.9538** | **+10.54%** |

Checks:
- YCoCg-R invertibility: PASS, exact on all 7 images.
- Sanity anchor: baseline avg 3.5769 bpp is inside 3.5–3.7 (within 15%): PASS.
- NN replication: +13.48% reproduces branch-1 Squeeze-lite loss (+13.5%): PASS.
  (Baseline 3.5769*3=10.7306 and NN 4.0591*3=12.1772 match branch-1 bit counts
  exactly; only the denominator changed per the corrected unit rule.)
- Bilinear constant-plane exactness: verified (constant in → constant out).
- LF side cost in correct units: 0.3492–0.4645 bpp per image (avg ~0.40 bpp),
  i.e. ~9–12% of NN total — matches branch-1 "LF side ~10%" finding.

## Verdict per variant

- (a) NN: LOSES by +13.48% avg (7/7 images lose, +10.84% to +17.72%). Replicates
  branch 1. Not viable.
- (b) BILINEAR: LOSES by +10.77% avg (7/7 lose, +9.05% to +13.17%). Recovers
  ~2.7 pp vs NN (avg total 4.0591 → 3.9621 bpp, saves ~0.097 bpp) but leaves a
  ~10.8% net loss. Smoother prediction helps directionally, does not rescue.
- (c) SMOOTHED-NN: LOSES by +10.54% avg (7/7 lose, +8.73% to +13.23%). Best of
  the three (beats bilinear on 6/7 images; loses to bilinear only on kodim23 by
  0.0022 bpp). Recovers ~2.9 pp vs NN (saves ~0.105 bpp) but still loses by
  ~10.5% net.

Overall: NO — smooth upsampling does NOT rescue Squeeze-lite. The loss is
STRUCTURAL (the LF/HF split itself), not just blocky NN artifacts. Mechanism:
the LF side stream alone costs avg ~0.40 bpp while the NN→smooth HF saving is
only ~0.10 bpp; even the best smoother cannot pay for the transmitted LF layer,
and HF-after-subtraction still codes worse than direct MED on every image.

## Single recommended follow-up

Retire the transmitted-4x-LF split (stop upsampler ablations — the headroom is
only ~0.1 bpp against a ~0.48 bpp gap); next test a side-free variant that keeps
any LF-conditioning gain with zero side bytes, e.g. condition the existing
MED/Huffman coder on an already-decodable coarse context (causal-neighborhood
mean) instead of a transmitted LF layer, and count every byte.
