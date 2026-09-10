# HAPRE-X Design Doc — 2026-09-07 — Novel Lightweight Lossless/Near-Lossless Image Codec

## Goal
Beat PNG/WebP-lossless and match JPEG-XL-lossless bpp with faster CPU decode and <100KB edge model (numpy-only, no torch), plus scalable near-lossless (l_inf bound 0/1/2/3). Target: publishable lightweight-track paper (DCC / TCSVT / TIP-short).

## Cross-domain transfers (user-approved)
- **Audio (FLAC/LPC, 1970s→now)**: per-block 3-tap adaptive linear predictor fit by least-squares on causal neighborhood (like FLAC's LPC order selection + partitioned Rice). Old idea, enhanced with image-gradient conditioning.
- **Text/LLM (PAQ 2002, FineZip'24, LLMZip)**: tiny PAQ-style logistic mixer of 4 predictor-experts over 16 gradient contexts, online 1-step update per pixel (PAQ's APM but 1000x smaller). BPE-style 8-entry pattern cache for flat runs (from text subword dicts).
- **GenAI (VQ, MaskGIT, Cool-chic overfitted '24)**: per-image overfitted 16-entry bias/scale table (<128 bytes overhead), checkerboard/row-parallel decode.
- **Physics (MaxEnt, ANS/Duda'09, RG)**: Laplace residual model = max-entropy under L1; ANS-heritage adaptive Rice (partitioned, like FLAC); 2-level coarse-to-fine DC plane (RG-inspired) for 16x16 block headers.
- **Old image ideas revived**: CALIC-GAP (1996), LOCO-I-MED+Golomb-Rice (1999), YCoCg-R (2003), Paeth (PNG), FELICS adjusted coding.

## Architecture
```
RGB -> YCoCg-R (reversible, int) -> per 16x16 block:
  fit LPC-3 coeffs (8-bit each, 3B overhead/block) IF variance high else fixed GAP/MED
  per pixel: 4 experts (MED, GAP, Paeth, LPC) -> PAQ-tiny logistic mix (16 ctx from quantized gradients) -> pred
  residual r = x - pred -> Laplace(b_ctx) + bias_ctx -> partitioned Rice(k_ctx adaptive) -> bitstream
  headers: 2b/block mode + optional LPC coeffs + 16x(bias,scale) table
Decode: row-parallel (1-row causal dependency only), vectorized numpy.
Near-lossless: r_q = round(r/(2τ+1)), reconstruct x_hat = pred + r_q*(2τ+1), guarantee |err|<=τ.
```

## Components
- `color.py`: YCoCg-R forward/inverse (exact int).
- `predict.py`: MED, GAP, Paeth, LPC-3 fit/predict.
- `mixer.py`: 16-ctx gradient quantizer + online logistic weights (4 floats/ctx).
- `entropy.py`: Laplace scale estimator, partitioned Rice encode/decode (bit-exact), Shannon bound calculator.
- `codec.py`: block loop, header packing, near-lossless quant.
- `eval.py`: bpp, enc/dec MP/s, maxerr, PSNR, vs PNG/WebP/JPEG-XL (cjxl).

## Data flow / Error handling
Encode: transform→block LPC fit→mixer predict→residual→Rice. Assert decode(encode(x))==x for τ=0; assert maxerr<=τ otherwise. Fallback: if block expands vs raw 8b, store raw flag.

## Testing / Gates (experiment-pipeline)
- S1: PNG/WebP/JPEG-XL baselines reproduce within 2% (same images).
- S2: Rice k init + ctx thresholds stable, variance <5% over 3 runs.
- S3: HAPRE-X bpp < best classical (PNG/WebP) by ≥5%, decode ≥20MP/s-equivalent in numpy (scaled claim: Python prototype speed + asymptotic analysis for optimized C), params <100KB.
- S4: ablate LPC, mixer, bias-table, YCoCg-R, pattern cache (leave-one-out).

## Honesty constraints
- Report real Rice bytes AND Shannon bound (don't conflate).
- Python prototype speed is lower bound; claim edge-runnability via op-count + memory, not fake C speed.
- If S3 gate fails, trigger IVE classification (implementation vs fundamental) per evo-memory.
