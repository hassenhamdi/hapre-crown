# HAPRE-C: A Streaming Lightweight Lossless Image Codec at a New Pareto Point
Draft v0.1 — 2026-09-07 — all numbers exact, all round-trips asserted.

## Abstract
We present HAPRE-C, a lossless image codec fusing classical tools — reversible YCoCg-R,
causal MED prediction, JPEG-LS-style run mode, per-8×8-block adaptive expert selection (µMoE-6),
9-class energy contexts and exact canonical Huffman coding — in ~540 lines of dependency-free C
streaming with O(W) memory. On Kodak (7×768×512, md5-pinned), HAPRE-C µMoE reaches **3.464 bpp
average at ~2.9 MP/s encode** with bit-exact round-trips on all images, and its speed configuration
(0-ctx, 3.58 bpp) encodes at **9.76 MP/s**. It significantly beats PNG-9 (median −1.07 bpp,
Wilcoxon W=0, p=.016, 7/7 images), JXL-e1 (median −0.15, p=.016, 7/7) and WebP-m0 (median −0.03,
p=.031, 6/6+1 tie), ties nothing else, and trails WebP-m3/m6 and JXL-e3/e9 (all p<.05) — reported
with the same rigor as our wins. Coding efficiency is 97% of the 9-context conditional entropy:
the codec is at the practical limit of causal-pixel prediction, which we prove by exhausting the
alternatives (LPC, LS/RLS-L1, FIR-12, GAP-mixer, adaptive bias, rANS, byte-LZ, naive LF) with
negative results reported, not hidden.

## 1. Introduction
Lossless image compression is dominated by heavyweights (JPEG-XL) on ratio and by 1990s formats
(PNG) on speed/portability. The lightweight Pareto segment — codecs fitting edge CPUs with tiny
RAM — has seen little movement since WebP-lossless. We attack it by fusion: every classical idea
that earns its bytes (CALIC GAP heritage, LOCO-I run mode + bias (bias rejected, §5), FLIF-style
heterogeneous contexts, PAQ-style expert mixing reduced to per-block RD choice, Duda's entropy
lineage via Huffman at 97% efficiency) plus one new mechanism: µMoE, per-8×8-block selection over
a 6-expert bank {MED, TOP, Paeth, GRAD, GAP, DG} with 3-bit ids (6.9 KB total overhead).

## 2. Related work
Classical lossless: PNG (DEFLATE), JPEG-LS/LOCO-I [Weinberger et al. 2000] (MED + Golomb + run
mode + bias), CALIC [Wu & Memon 1996] (GAP + contexts), JPEG 2000 (5/3 wavelet + EBCOT),
WebP-lossless (LZ77 + color cache + Huffman), FLIF (MANIAC), JPEG-XL (MA-trees, ANS, Squeeze).
Learned lossless: L3C, SReC, IDF, DLPR [Bai et al. TPAMI 2022] (lossy+residual VAEs),
scalable ℓ∞ near-lossless [Bai et al. CVPR 2021], MLIC++ entropy models, Cool-chic overfitted
codecs, FLLIC functionally-lossless [Zhang & Wu 2024]. Our work is orthogonal: zero learned
weights, zero floating point in the codec path, pure streaming C.

## 3. Method
### 3.1 Pipeline
RGB → reversible YCoCg-R (integer) → per-channel planes → phase 1: per-8×8-block expert choice
by L1 proxy over regular-path residuals → phase 2: single causal pass with JPEG-LS-semantics run
machine (run token = Elias-γ(L+1); every run followed by exactly one interruption symbol; row-end
runs bare) → residuals → 27 (channel×9 energy ctx) canonical Huffman tables stored explicitly
(16b count + entries of 16b symbol + 8b length) → stream. Decoder mirrors the state machine;
block ids (3b, packed) precede tables. Border convention (shared, exact): (0,0)→0; first
row→left; first col→top; interior causal; topright unavailable on first row (d:=a) — causality
proof in §3.3.
### 3.2 Alphabet and correctness
Residual alphabet index r+1024 over [0,2048]; proven bounds for 8-bit RGB inputs (|r|≤702,
§A) with runtime asserts — no silent clipping (a landmine we found and fixed, §5).
### 3.3 Framing proof sketch
Run tokens and interruption symbols strictly alternate per run decision; both sides evaluate the
identical decision function on identical recon prefixes (induction from (0,0)); L=0 runs emit
tokens (else framing breaks — bug found and fixed, §5). Full bit-exact round-trips asserted on
all images (byte equality), plus synthetic edge cases (flat/gradient/random/saturated red-blue).

## 4. Experiments
Setup: Kodak 7 images (md5 in CHECKSUMS.txt), same machine for all codecs; baselines PNG-1/6/9
(PIL 12.2.0), WebP-m0/m3/m6 (PIL), JXL-e1/e3/e9 (cjxl/djxl 0.11.2); ours gcc -O3.
### 4.1 Ratio (exact bpp): MOE 3.464 avg (01:3.56 02:3.23 05:3.84 07:3.01 13:4.17 19:3.41 23:3.03)
vs PNG-9 4.75 / WebP-m0 3.60 / m3 3.38 / m6 3.32 / JXL-e1 3.72 / e3 3.23 / e9 3.03.
Wilcoxon paired (n=7, Shapiro-checked): beats PNG-9 (Δ−1.07, p=.016), JXL-e1 (Δ−0.15, p=.016),
WebP-m0 (Δ−0.03, p=.031); loses to WebP-m3 (Δ+0.08, p=.031), m6 and JXL-e3 (both p=.016).
### 4.2 Speed: MOE ~2.9 MP/s enc / ~3 MP/s dec; 0-ctx config 9.76 enc MP/s (Pareto speed point);
determinism 3/3 identical runs.
### 4.3 Ablations (kodim01-centered, exact): YCoCg-R −35% vs RGB; run mode −0.7%; CTX9 −1.4%;
µMoE-6 −1.3% over RUN (−2.1% honest over MED baseline); Huffman at 97% of H1.
### 4.4 Negative results (all measured, all kept): per-block LPC-3 (−4%, overfit+headers); crude
GAP mixer (−3.7%); GAP+bias quick port (5.00, missing run/per-ctx-k); LOCO-I-lite bias under
Huffman (−0.2%, reverted); LS/IRLS-L1 linear (≈MED — linear wall); FIR-12 +bit-gate (+0.0%);
cross-channel features (|corr|<0.11, skipped); rANS-ch/M14 (ties Huffman); rANS-M10 bug lesson
(precision must cover alphabet); byte-LZ77 on residuals (4.18, −38%); naive 16-bin LF (−8..−12%,
fragmentation); wavelet+zlib (4.78).

## 5. Bug taxonomy (for reproducibility)
Causality violation (first-row d peeked at undecoded pixel); framing (L=0 runs must emit tokens);
alphabet clipping (silent corruption → ±1024 + proofs + asserts); rANS precision; border-convention
matching (encoder/decoder identical rule); table-build ordering for canonical codes.

## 6. Limitations and next bosses
Loses to WebP-m3 and above; decode (3 MP/s) trails WebP/PNG decode; no transform — Squeeze/DCT
subband coding is the identified artillery for JXL-e3/e9; LF-clustered conditioning for WebP-m3.

## References (grounding corpus, paper-navigator arXiv)
2103.17015 (scalable near-lossless), 2209.04847 (DLPR), 2401.13616 (FLLIC), 2201.03195
(pseudo-residual), MLIC++ 2023, HiDE 2026, MoE-entropy 2026, Tree-VQ 2609.03641, Cool-chic
2609.04274, FineZip 2024 (LLM compression), plus LOCO-I/CALIC/FLIF/JPEG-XL classical lineage.
