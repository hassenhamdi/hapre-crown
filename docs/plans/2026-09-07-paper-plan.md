# Paper Plan — HAPRE-C: A Streaming Lightweight Lossless Image Codec at a New Pareto Point
Target: DCC (Data Compression Conference) short / PCS / IEEE TCSVT correspondence / Entropy (MDPI) special issue.
Title: "HAPRE-C: YCoCg-R + Causal MED + Exact Huffman in 120 Lines of C — 3.58 bpp at 9.8 MP/s on Kodak"
Contributions: (1) cross-domain fusion design (audio-LPC lesson→rejected w/ evidence, text-PAQ mixer→rejected w/ evidence,
physics-MaxEnt Huffman at 97% efficiency, CALIC/JPEG-LS heritage); (2) exact small-table streaming format + bit-proven C codec;
(3) same-machine Pareto bench incl. JXL (honest: JXL-e3 dominates overall; record scoped to lightweight class);
(4) ablation + IVE failure taxonomy (LPC/mixer/wavelet negative results reported, not hidden).
Tables: per-image bpp (7 Kodak × 10 codecs), speed table, ablation table, Wilcoxon stats (APA), memory table (working set KB).
Figures: Pareto scatter (bpp vs enc MP/s, log-x) with frontier; ablation waterfall; context-entropy efficiency bar.
Required before submission: CLIC-mobile + DIV2K-val replication, JPEG-LS near-lossless baseline (loco), τ-sweep RD curves (PSNR/MS-SSIM),
working-set measurement (/usr/bin/time -v), decode LUT v2, rANS variant appendix (chase JXL-e3: CTX9 + rANS + run mode → target ≤3.3).
Integrity: no JXL-beating lossless claim; WebP-m0 reported as tie (p=.84); tau results labeled with RGB maxerr, not ℓ∞≤1.
