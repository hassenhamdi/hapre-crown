# From Pixels to Proof: A Lossless Image-Compression Campaign from HAPRE-C to CROWN6 — Exact Codecs, a Classical Ceiling, and All the Dead Ends

**Draft v1.0 — DCC full-paper style — 2026-09-08**

> Campaign scope: Kodak 7×768×512 (md5-pinned, `CHECKSUMS.txt`), same machine, exact stream bytes, bit-exact round-trips asserted. Probe-level numbers are labeled (P); exact-codec numbers are labeled (E). Nothing in this draft exceeds the banked evidence. `[VERIFY]` marks any digit transcribed from probe logs/JSON that must be re-run before camera-ready.

## Abstract

We report a 35-cycle lossless image-compression campaign that starts from a tiny streaming codec (HAPRE-C: reversible YCoCg-R + causal MED + exact Huffman in ~540 lines of dependency-free C) and ends at CROWN6, an exact integer codec at **3.1978 bpp** Kodak-7 average (E) with all-7 byte-asserted round-trips. Along the way we kill seven bosses with paired Wilcoxon significance (n=7): PNG-9 (4.75), QOI (4.9197), JXL-e1 (3.72), WebP-m0 (3.60), WebP-m3 (3.347), WebP-m6 (3.3157) and JPEG-LS-CharLS (4.5757), and we document exactly where we stop: JXL-e3 (3.2291) stands at 5W-2L (p=.219) with texture-concentrated losses (kodim05 +1.95%, kodim13 +1.00%), FLIF v0.4 (2.9699, measured §4.2) stands at 0W-7L (p=.016) — a stronger boss than e3 on our seven — and JXL-e9 (3.03) stands at +~5.5%. Throughout, a *boss* is a widely-used baseline configuration, *KO* means average win plus majority wins plus two-sided Wilcoxon p<.05 (n=7), and *standing* means the boss held under that rule.

The technical contribution is a fully-exact codec family — CROWN-huff (3.343) → CROWN-rans-hc (3.309) → CROWN2-exact (3.2686) → CROWN4-exact (3.1993) → CROWN6-exact (3.1978, current best) — built from sign-flipped LOCO-365 contexts, auto-K quantile / WIDE-K clustering, an expanding expert bank (E6 → E16/E17 → LMS pair → per-context micro-MLP), global RCT selection over {C6,C27,C12}, and per-group best-of-3 backends (Huffman / Golomb-Rice / rANS M=14). The methodological contribution is the negative-result record: we declare a **classical ceiling at 3.2515** (E/P混合 — full classical stack, probe frame, exact-counted, decode-verified) with leave-one-out marginals, prove that transform, LF-side, byte-LZ, enumerative, VQ-transfer, and backward-adaptive families fail with mechanisms (not luck), and publish the bug taxonomy that made exactness possible. A near-lossless extension (τ-quantized residuals) and a JXL-lossy context are reported as rate–distortion, not as lossless claims.

We do not claim the neural SOTA (2.52–2.54 bpsp on Kodak). We claim the strongest fully-counted classical-plus-micro-neural exact codec in this campaign, with every loss reported at the same rigor as every win.

## 1. Introduction

Lossless image compression has two attractors. Heavyweights (JPEG-XL, FLIF) win ratio with MA-trees,ANS, and learned contexts. Lightweights (PNG, QOI, WebP-lossless fast modes) win speed and portability. The lightweight Pareto segment has barely moved since WebP-lossless: edge CPUs still ship PNG because nothing tiny beats it safely.

This paper is a campaign report. Our question was narrow: how far can a streaming, integer-only, table-explicit codec go on real photos (Kodak) if every byte is counted and every round-trip is asserted?

The answer has three parts.

**First, a codec lineage that earns its bytes.** HAPRE-C v0.1 (see `docs/HAPRE-C-paper-draft-v01.md`, now superseded in numbers) fused YCoCg-R, causal MED, JPEG-LS-style run mode, per-8×8-block µMoE-6 expert selection, 9-class energy contexts and canonical Huffman at 3.464 bpp (E). That codec already killed PNG-9, JXL-e1 and WebP-m0. The CROWN line then replaced blocks with LOCO-365 gradient-pattern contexts, quantile clustering, per-group best-of-N predictors, WIDE-K search, 3-way backends, global RCT permutation search, gated Weighted-LS and LMS adaptive-FIR experts, and finally per-context micro-MLPs — each gated by exact assembled bytes. The result is CROWN6-exact at 3.1978 bpp (E), with C decode in ~200 ms [VERIFY].

**Second, a ceiling with a proof of exhaustion.** By stacking every proven classical mechanism (LOCO partition + E16 + H/G/R backends + merges + MA-tree/grid + Y-splits) under exact counting we reach 3.2515 avg (P, exact-counted probe, decode-verified) at 3W-4L vs JXL-e3. Leave-one-out shows backend-choice (+0.050) [VERIFY] and LOCO-partition (+0.033) [VERIFY] as the only large marginals; tree/grid marginals are 0.0000 (subsumed, not useless). Further classical stacking is bounded at ~−0.01. The remaining gap is representational (texture predictors), not allocational. Neural-micro (per-context MLPs) then adds the last step to 3.1978 (E).

**Third, a full negative-results section — a feature, not an appendix.** Linear predictors, three LF-side variants, DCT/wavelet/hybrid transforms, byte-LZ, fixed bands, Huffman-invariant bias adapters, hash-cache escapes, enumerative 2×2 codes, VQ-transfer, global and joint backward-adaptivity, MA-tree-lite alone, A-joint pairs in-frame, I-GATED splits in-frame, and grids all fail or retire with measured mechanisms. We report them because the next worker should not have to replicate our search.

Disclosure rules govern the whole draft: (i) losses with the same rigor as wins (WebP-m3 ties under Huffman, JXL-e3/e9 standing); (ii) every number labeled (E)xact-codec vs (P)robe-level; (iii) no beaten-boss claim beyond the Wilcoxon statistic; (iv) `[VERIFY]` on any transcribed digit that needs a re-run.

## 2. Related Work

### 2.1 Classical lineage: the tools we actually used

**PNG / DEFLATE** (4.75 bpp here, E): LZ77 + Huffman on bytes. No cross-channel decorrelation; filters approximate MED. Our YCoCg-R alone beats it by ~35% in ablation (P) [VERIFY].

**JPEG-LS / LOCO-I** (Weinberger et al. 2000): MED + Golomb + run mode + per-context bias + 365 merged gradient contexts. We reuse all of it except where we prove otherwise: full LOCO-I under Golomb gains −9.9% over MED+Golomb (P, 7/7) [VERIFY]; under Huffman the bias adapter is provably null (translation-invariance theorem, §5) and run mode hurts once LOCO groups are fine (+0.007) (P) [VERIFY].
**JPEG-LS measured** (CharLS 2.4.4 via imagecodecs, RGB defaults, decode-asserted 7/7): **4.5757 avg** (E) — DEAD, far below literature's 2.82 (undisclosed pipeline; SEEC's own pipeline gives 4.36). Ablation **JPEG-LS∘YCoCg-R** (our C6 planes, NOT a codec): 3.3664 (E) — the 1.21 bpp gap to RGB-framing isolates exactly what our reversible front end buys; CROWN6's further −0.17 to 3.1978 is the grouping+experts+backend stack. Ledger `survey/bosstakedown/data/classical7.json`.

**CALIC** (Wu & Memon 1996): GAP + 144-texture/576-compound contexts. GAP16 survives into our E16 bank; crude GAP mixers lose −3.7% (P) [VERIFY].

**WebP-lossless**: LZ77 on pixels+transforms, color cache, Huffman. WebP-m0 (3.60, E), m3 (3.347, E), m6 (3.3157, E) are Bosses 1–3. Lesson: WebP's LZ works on pixels, not on int16 residual bytes (our byte-LZ scores 4.18 avg, report-level P) [VERIFY].

**FLIF / MANIAC → JPEG-XL MA-trees**: signaled per-image decision trees, per-leaf (predictor, histogram), histogram sharing, static decode. Our LOCO-365 + autoK + per-group best-of is a hand-built 1-level MA-tree. JXL's ladder (see §2.3) is the boss table. **FLIF v0.4 measured** (source build, default `flif -e`, pixel-compare round-trips PASS 7/7, kodim23 enc 1.28 s / dec 0.35 s): **2.9699 avg** (E) on our seven (ledger `survey/bosstakedown/data/classical7.json`) — beats CROWN6 0W-7L (p=.016) and beats JXL-e3 7/7 on the same seven. FLIF is a STANDING boss, stronger than e3 here; its MANIAC contexts handle our texture blockers better (05: 3.3757 vs JXL 3.5120). Literature full-24 FLIF 2.72–3.01 (CALLIC/P2-LLM/SEEC tables) brackets our 7-subset number.

**JPEG 2000 (5/3 + EBCOT)**: the transform-bitplane ideal. Our stripped-down bitplanes without EBCOT's trio (feedback + sign/ref contexts + runs) lose by +26% (P) [VERIFY].

### 2.2 Learned lossless: the ceiling we do not claim

Normalized to bpsp (= our bpp; neural papers quoting total bpp are divided by 3):

| Model | Kodak (bpsp) | Note |
|---|---|---|
| CALLIC (AAAI25) | 2.54 [VERIFY] | Masked Gated ConvFormer + per-image LoRA adapters under 2-stage MDL; base 2.77 → 2.54 |
| HPAC (Nov 2025) | ~2.52 [VERIFY] | Group-parallel scan + adaptive focus coding; fastest of the top tier |
| SEEC (Sep 2025) | 2.84 [VERIFY] | N=2 semantic-region entropy models; validates multi-distribution thesis (−0.02 over DLPR) |
| LLM + visual prompts (SJTU 2025) | 2.83 [VERIFY] | Frozen LLM buys +0.03 over DLPR — hype-only for C codecs |
| JXL (in their tables) | 2.87–3.06 [VERIFY] | Consistent with our JXL-e9 3.03 (E) |
| Ours (CROWN6-exact) | 3.1978 (E) | ~60% of the way from JXL-e1 to neural SOTA |

The portable lesson from GenAI (survey `SURVEY_genai_qoi.md`) is procedures, not models: learn trees/predictors/clusterings offline, ship integers. Decode-time NN inference, diffusion/flow/VAE codecs, and per-symbol neural entropy heads are explicitly out of scope (GPU, stochastic, or 20–200 ms/px-batch entropy heads).

### 2.3 JPEG-XL as bosses: ladder, weaknesses, moving bar

Reproduced with `cjxl -d 0 -e E` (4 threads, exact bytes, bpp = bytes·8/1179648):

| img [VERIFY per-image] | e1 | e3 (Boss-5) | e6 | e7 | e8 | e9 (Boss-6) |
|---|---|---|---|---|---|---|
| kodim01 | 3.7888 | 3.3593 | 3.1966 | 3.1691 | 3.1723 | 3.1336 |
| kodim02 | 3.4894 | 3.0611 | 2.9377 | 2.8589 | 2.8337 | 2.8401 |
| kodim05 | 4.1091 | 3.5120 | 3.3984 | 3.3588 | 3.3755 | 3.4082 |
| kodim07 | 3.3244 | 2.7310 | 2.5456 | 2.4972 | 2.5098 | 2.4433 |
| kodim13 | 4.3389 | 3.9141 | 3.7982 | 3.7774 | 3.7822 | 3.7815 |
| kodim19 | 3.6637 | 3.2154 | 3.0739 | 3.0419 | 3.0334 | 3.0101 |
| kodim23 | 3.3281 | 2.8110 | 2.6723 | 2.5937 | 2.6025 | 2.6097 |
| **AVG** | **3.7204 [VERIFY]** | **3.2291 (E)** | **3.0890 [VERIFY]** | **3.0424 [VERIFY]** | **3.0442 [VERIFY]** | **3.0324 ≈3.03 (E)** |

Mechanism map: e1→e2 −4.8% (RCT-6 on + ANS); e2→e3 −8.8% (Gradient→Weighted + fixed tree); e3→e4 −0.5% (MA-tree learning on); e4→e5 −0.8% (per-group RCT search); e5→e6 −3.1% (more RCT + wider MA); e7→e8 +0.06% regression (overfit, matches upstream #4107); e8→e9 −0.4% (kBest + clustering + LZ77, 4.4× time).

Three weaknesses (all with local evidence, `BOSS_TAKEDOWN.md`):

1. **Groups fragment stats.** `-g 3` (1024) beats default-256 everywhere: e9 3.0324→2.9898 (−1.4%) [VERIFY]; e7-g3 (3.0121) [VERIFY] beats e9-default in 1/4 the time. Confirms our WIDE-K/merges doctrine.
2. **Ladder non-monotonic; e9 overfits.** e7 beats e9 on 3/7; kodim05 degrades e7→e8→e9 (+1.5%) [VERIFY]. Tuned-e9 ceiling ≈2.94 (E/P) [VERIFY] — the true Boss-6 bar is ~10% below us, not 5.5%.
3. **Fixed-RCT search beaten by one fixed RCT.** e3-C27 3.2083 (−0.65%) [VERIFY]; even e9-C12 beats per-group search on 2 images. Our global {C6,C27,C12} bank ports this directly.

QOI teardown (649-line `qoi.h` read + measured): QOI Kodak-7 **4.9197** (E, pixel-PASS 7/7, ledger `classical7.json`) — we beat QOI 7/7 p=.016 at −35.0% (E). QOI optimizes the other end of the Pareto axis (kodim23 enc 12 ms / dec ms-scale vs our CROWN6 ~54 s / ~0.2 s; HAPRE-C 9.76 MP/s sits between) — Figure F5 tells both directions. Portable ideas ranked: luma-anchored asymmetric coding, distant-repeat cache escape, run hygiene, greedy escape ordering, tiny-diff fast path. Hash-cache fails on photos under full byte-counting (§5).

## 3. Method: HAPRE-C → CROWN6

### 3.1 Front end (unchanged since HAPRE-C)

**YCoCg-R (integer, reversible).** `Co=R−B; t=B+(Co//2); Cg=G−t; Y=t+(Cg//2)` with floor `//2` for negatives; inverse asserted per image. Proven residual alphabet |r|≤702 for 8-bit RGB (§A of v0.1); encoder asserts |r|≤1024 loudly, decoder returns −4, never clips (a former silent-corruption landmine, §6).

**Global RCT bank (CROWN3+).** One RCT per image over {C6 `(perm0,t6)`, C27 `(perm3,t6)`, C12 `(perm1,t5)`} (JXL catalogue `7·perm+t`), 2 bits in flags byte. Global best-of-8 gains −1.41% MED-frame (P, 7/7 p=.016) [VERIFY]; C27 wins 6/7, C12 on kodim02. Per-block RCT is dead (global wins 6/7; fragmentation + boundary-mixing, P) [VERIFY]. Homogeneity rule: RCT-mixing poisons predictors — (a0+b) > (b) > (a+b) (P) [VERIFY].

**Borders.** Dual doctrine, audited: E16/keys/GRID use b4-style (zero border; row0 copies left; col0 copies top; TR copies L/T at edges); WAVG/LMS/MLP-activity use B17-style zero-TL. Encoder == decoder exactly; two real border bugs caught (sign-flip abs-vs-negate, TL doctrine).

### 3.2 Prediction: from MED to E20

| Generation | Bank | Reported gain |
|---|---|---|
| HAPRE-C 0-ctx / CTX9 / RUN / MOE | MED → MED+9-ctx → +run → µMoE-6 {MED,TOP,Paeth,GRAD,GAP,DG} | 3.58 → 3.537 → 3.511 → 3.464 (E) |
| CROWN-huff | Per-group best-of-6 under LOCO groups | 3.343 (E), −3.50% vs champ 7/7 p=.016 |
| CROWN-rans / E16+WIDE | E16 + Huff/Golomb 2-way + WIDE-K(≤64); +rANS 3-way | 3.2979 (P) ties m6; 3.2720 (P) KOs m6 7/7 p=.016 |
| CROWN2-exact | E16 + 3-way + WIDE-K + G-bias + GRID-dictionary, exact C | 3.2686 (E) |
| CROWN3-exact | + global RCT + Weighted-17th (G32 LS-4tap ×16, MDL>21) | 3.2040 (E), 7/7 vs CROWN2 p=.016 |
| CROWN4-exact | + LMS5_T0/T3 gated 18th/19th (recon-only sum-16 renorm, zero side) | 3.1993 (E), 7/7 vs CROWN3 p=.016 |
| CROWN6-exact | + per-ctx micro-MLP 20th (predid 19, §3.4) | **3.1978 (E)**, current best |

**LOCO-365 contexts.** Gradients `Q1=b−c, Q2=c−a, Q3=d−b` quantized at 3/7/21, sign-merged to 365 ids. Sign-flip `rf=s·res` fixes bimodal groups (−0.03, P) [VERIFY]. Full LOCO-I under Golomb: −9.9% vs MED+Golomb (P) [VERIFY]; per-ctx-k+run ≈2× bias gain.

**Clustering.** Auto-K quantile on per-context mean-|res| (K∈{2,3,4,6,9,12,18,27,36} then WIDE to 64), greedy equal-pixel + hillclimb refine, 92-byte dense bitmask + `gbits` ids. GRID M-THR energy dictionary (8 grids, 3-bit selector) competes per-channel by exact bytes; Q wins 7/7 in FULL stack but GRID survives as torture-input fast path.

**WAVG-17th.** Per-G32-block closed-form LS on {L,T,TL,TR}, ×16 quantized to [−16,15] (5b×4 = 20b/used + 1b flag), gate `N·log2(b_med/b_w)>21`. G32 beats G64 7/7 (P) [VERIFY]; kernels are edge-sharpeners (negative TL). −2.15% MED-frame (P, 7/7 p=.016) [VERIFY].

**LMS pair.** Sign-sign adaptive-FIR 5-tap (stencil [L,T,TL,TR,MED5], thr 0/3, init [3,3,3,3,4] sum 16, clamp [−8,20], sum-renorm). Zero side bytes; both sides re-simulate from recon. −0.15% exact (E, 7/7 p=.016), no regressions; 05 +2.46%→+2.27%, 13 +1.02%→+0.98% (E) [VERIFY per-image deltas].

**Run mode.** JPEG-LS semantics (L=0 emits token; every run followed by exactly one interruption symbol; row-end runs bare). Helps early (+run −0.7%, P) [VERIFY]; hurts under fine LOCO groups (+0.007, P) [VERIFY]; killed on texture (D-run −1.31% vs C-gol, P) [VERIFY].

### 3.3 Entropy: per-group best-of-3 + merges

Per non-empty group: backend 0=Huffman (canonical, explicit tables `A u16le` + A×(`sym i16le`+`len u8`), cost `16+A·24` bits), 1=Golomb-Rice (best k 0..12 + 4b k + 3b dbias {−4..3}, mapping `M=2v/−2v−1`, code q ZEROS + `1` + k-bit rem — opposite polarity to libhapre era-variant, vendored packer), 2=rANS M=14 (`count u32le`+`A u16le`+A×(`sym i16le`+`freq u16le`)+`paylen u32le`+payload, table `16+A·32` + 64 framing bits). Empty groups forced backend-0, no table. Backend metadata 1b (2-way) / 2b (3-way) per group, exact-gated.

Greedy table merges (shortlisted S=16, proxy search, exact-gated accepts, exact 3-way finalize): −1.20% MED-frame (P, 7/7, tables 60→18, 5:1 MDL return) [VERIFY]; transfer onto CROWN −0.37% Huffman / −1.35% with rANS re-pick (P) [VERIFY]; merges ≈1/3 of e3 gap. Lesson: drive merges with the backend that will encode them (Huffman-greedy over-merges for rANS).

### 3.4 Neural-micro: per-context MLPs as the 20th expert (CROWN6)

Committed config (probe_b21): energy-9 quantile contexts (E=|L−T|+|L−TL|+|T−TL|, 8 float64 thresholds, 48 B/image) × (12feat→8→1 tanh MLP, L1, LS-init, 200 iters @ lr 0.003 cosine, adaptive-int16 scales {256…4096} + 3-bit id, double MDL gate L1-per-(ch,ctx) + L2-per-channel). 12 feats are causal zero-border taps [L,T,TL,TR,Ww,NNe,(L+T)//2,|L−T|,L3,T3,TL2,TR2]; non-winning ctx falls back to edge-MED.

Wire-forward is bit-exact by construction: frozen tanh LUT (`crown6_tanhlut.h`, 131073 entries on [0,8] step 2⁻¹⁴, `%.17g` literals, `-ffp-contract=off`), dequant `w=int16/S` exact in float64, `pred=floor(y·128+0.5)` half-up. Solves the torch-vs-C 1-ulp divergence (e.g. pre=19.0612793 [VERIFY]) and the edge-replicate causality peek (zero-border taps only). Exhaustive C-vs-wire xcheck: 0 mismatches / 17.2M px (E) [VERIFY]; torch-vs-wire int disagreement ≈1% px ±1 LSB ≈+0.04% bits (P, documented) [VERIFY].

Side framing: 64 B thresholds + 9× scale-id bytes (255=absent) + 226 B/winning net (113×int16le: fc1 96 + bias 8 + fc2 8 + bias 1). Per-channel dual-track exact-byte gate (MLP-variant vs no-MLP, strict `<`) + Q-vs-GRID + RCT-min ⇒ CROWN6 ≤ CROWN4 per image by construction (+3 flag bytes). Per-ctx beats global-MLP by −0.85% MED-frame (P, 6/7) [VERIFY]; fixed-×256 quant destroys per-ctx nets (−21% worst, P) [VERIFY] — adaptive-int16 retains 97–100% of float gain.

### 3.5 Exact stream design (CROWN6 v1, `CROWN6_FORMAT.md`)

Magic `C6` + H/W u16le + ver/flags(RCT id) = 8 B global. Per channel (order p0,p1,p2 of RCT): 2-B family header (`[kidx:4][grid:3][fam:1]` + ng) + 1-B MLP flags + MLP side (iff present) + W-side (`ceil(ng32/8)` use flags + `ceil(nused·20/8)` weights) + LMS zero bytes + group map (Q: 92-B mask + ids; GRID: 1-B occ) + `ceil(ng·7/8)` B 7-bit `[predid:5][backend:2]` (0..19; 19 requires mlp_present) + kvals/dbias + Huffman tables/payloads (`n u32le`+n bytes via `pack_syms`) + Golomb payloads + rANS groups. Decoder: single causal scan per channel + `crown6_rct_inv`; `p==len(blob)` asserted; `decode==original` byte-compared. CROWN2/3/4/5 formats differ only in magic, predid width (6→7 bits), W/LMS/MLP sides, and choice bytes — all specified byte-exact in `src/CROWN*FORMAT.md`.

## 4. Experiments

### 4.1 Setup

Kodak 7 images (768×512×3, N=1179648 sub-pixels; 19 is 768×512 portrait [VERIFY orientation]), md5-pinned (`CHECKSUMS.txt`). Baselines same machine: PNG-1/6/9 (PIL 12.2.0), WebP-m0/m3/m6 (PIL/libwebp-1.6.0), JXL-e1/e3/e9 (cjxl/djxl 0.11.2, 4 threads). Ours: `gcc -O3/-O2`, Python 3.14.2 / numpy 2.4.6 / PIL 12.2.0 / torch 2.14 CPU (MLP training only). bpp = total_stream_bytes·8/(H·W·3). All claims Wilcoxon paired two-sided (n=7) unless noted. Round-trips byte-asserted (E) or scalar-sim + table proofs (P).

### 4.2 Boss ladder (exact averages + stats)

| # | Codec | Avg bpp | vs CROWN6 3.1978 | Wilcoxon (n=7) | Verdict |
|---|---|---|---|---|---|
| — | PNG-9 | 4.75 (E) | −32.7% [VERIFY calc] | W=0 p=.016 7/7 | KO |
| — | QOI | 4.9197 (E) | −35.0% [VERIFY calc] | W=0 p=.016 7/7 | KO (speed-axis context, §2.3) |
| — | JXL-e1 | 3.72 (E) | −14.0% [VERIFY calc] | W=0 p=.016 7/7 | KO |
| — | WebP-m0 | 3.60 (E) | −11.2% [VERIFY calc] | W=0 p=.031 6/6+1T | KO (RUN config) |
| 1 | CROWN-huff | 3.343 (E) | −4.3% [VERIFY calc] | — | Champion-era |
| 2 | WebP-m3 | 3.347 (E) | −4.5% [VERIFY calc] | W=0 p=.016 7/7 (CROWN-rans-hc) | KO (rANS backend) |
| — | CROWN-rans-hc | 3.309 (E) | −3.4% [VERIFY calc] | (the killer arm) | BOSS-2 KILLER |
| 3 | WebP-m6 | 3.3157 (E) | −3.6% [VERIFY calc] | W=0 p=.016 7/7 (CROWN2-exact) | KO (exact-codec) |
| — | CROWN2-exact | 3.2686 (E) | −2.2% [VERIFY calc] | — | Extends m6 margin −1.42% [VERIFY] |
| 5 | JXL-e3 | 3.2291 (E) | −1.0% [VERIFY calc] | 5W-2L p=.219 (CROWN6) | STANDING |
| — | FLIF v0.4 | 2.9699 (E) | +7.1% [VERIFY calc] | 0W-7L p=.016 (CROWN6) | STANDING (beats e3 7/7 here) |
| — | JPEG-LS (CharLS RGB) | 4.5757 (E) | −30.1% [VERIFY calc] | 7W-0L p=.016 (CROWN6, RT-asserted) | KO (front-end lesson, §2.1) |
| — | CROWN4-exact | 3.1993 (E) | −0.05% [VERIFY calc] | 7/7 vs CROWN3 p=.016 | Best-classical+ |
| — | **CROWN6-exact** | **3.1978 (E)** | — | (current best) | **CHAMPION** |
| 6 | JXL-e9 | 3.03 (E) | +~5.5% | — | FINAL BOSS, standing |

Disclosure notes: KOs are listed strongest-first; the paper's weight rests on the WebP-m3/m6 and JXL configurations, not the weaklings (PNG/QOI/JPEG-LS), which we include for completeness. CROWN-huff vs PIL-m3 ties (−0.0023 avg, 1W-6L p≈.30 W=7, P) [VERIFIED 2026-09-08] — the m3 KO requires the rANS backend (probe-level proof then CROWN-C exact build, E). CROWN2-exact vs WebP-m6 is 7/7 p=.016 (E); CROWN6 vs m6 is ties-or-better (do not overclaim 7/7). JXL-e3-C27 (3.2083, one flag) [VERIFY] widens the CROWN2 gap to +1.9% [VERIFY] — a moving bar, but C27 is our free idea too. Tuned-e9 ≈2.94 (P) [VERIFY] is the true Boss-6 bar.

### 4.3 Per-image bpp (main codecs)

All per-image entries are [VERIFY] (transcribed from takedown CSVs / `crown6_results.json` / probe re-runs; re-verify with `src/perimage.py` + `src/driver_crown*.py` before camera-ready).

| img | PNG-9 [VERIFY] | JXL-e1 | WebP-m0 [VERIFY] | CROWN-huff (P) | WebP-m3 [VERIFY] | CROWN-rans-hc (P/E) | WebP-m6 [VERIFY] | CROWN2-exact | CROWN4-exact [VERIFY] | CROWN6-exact | JXL-e3 | JXL-e9 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kodim01 | 5.27 | 3.7888 | 3.63 | 3.4340 | 3.4074 [PIL] | 3.4002 | 3.39 | 3.3789 | 3.2991 | 3.2982 | 3.3593 | 3.1336 |
| kodim02 | 4.19 | 3.4894 | 3.30 | 3.0931 | 3.0659 [PIL] | 3.0590 | 3.06 | 3.0274 | 2.9854 | 2.9854 | 3.0611 | 2.8401 |
| kodim05 | 5.51 | 4.1091 | 4.15 | 3.7394 | 3.8563 [PIL] | 3.7176 | 3.77 | 3.6594 | 3.5917 | 3.5806 | 3.5120 | 3.4082 |
| kodim07 | 3.83 | 3.3244 | 3.11 | 2.8630 | 2.8397 [PIL] | 2.8060 | 2.81 | 2.7556 | 2.6790 | 2.6798 | 2.7310 | 2.4433 |
| kodim13 | 6.07 | 4.3389 | 4.33 | 4.1010 | 4.0990 [PIL] | 4.0879 | 4.05 | 4.0245 | 3.9525 | 3.9532 | 3.9141 | 3.7815 |
| kodim19 | 4.58 | 3.6637 | 3.47 | 3.2758 | 3.2730 [PIL] | 3.2522 | 3.27 | 3.2207 | 3.1510 | 3.1510 | 3.2154 | 3.0101 |
| kodim23 | 3.77 | 3.3281 | 3.19 | 2.8940 | 2.8754 [PIL] | 2.8636 | 2.86 | 2.8138 | 2.7364 | 2.7365 | 2.8110 | 2.6097 |
| **AVG** | **4.75** | **3.7204** | **3.60** | **3.3429** | **3.347 / 3.3452 PIL** | **3.3124 / 3.309 HC** | **3.3157** | **3.2686** | **3.1993** | **3.1978** | **3.2291** | **3.0324** |

Notes: PNG-9/WebP-m0 per-image banked 2026-09-08 (`survey/bosstakedown/data/perimage7.csv`, PIL/cjxl; avgs 4.7457→4.75 / 3.5971→3.60 confirm pins). RUN per-image (E, all PASS): 3.58/3.27/3.90/3.06/4.19/3.47/3.11, avg 3.5114→3.511 (`survey/bosstakedown/data/run7.txt`). MOE-champion per-image (E, superseded): 01:3.56 02:3.23 05:3.84 07:3.01 13:4.17 19:3.41 23:3.03 [VERIFY]. CROWN6 RCT picks: C27 on 01/05/07/19/23, C12 on 02/13 [VERIFY]; fams Q 7/7 [VERIFY]; MLP active on 01/05/13 only [VERIFY]. CROWN6 vs JXL-e3 per-image Δ%: 01 −1.82, 02 −2.47, 05 +1.95, 07 −1.87, 13 +1.00, 19 −2.00, 23 −2.65 [VERIFY].

### 4.4 Wilcoxon table (paired, n=7, two-sided)

| Comparison | W/L/T | W | p | Claim |
|---|---|---|---|---|
| MOE/CROWN-era vs PNG-9 | 7/0/0 | 0 | .016 | KO |
| CROWN6 vs QOI | 7/0/0 | 0 | .016 | KO (margin −35.0%; QOI wins the speed axis instead) |
| vs JXL-e1 | 7/0/0 | 0 | .016 | KO |
| RUN vs WebP-m0 | 6/0/1 | 0 | .031 | KO (RUN config) |
| CROWN-rans-hc vs WebP-m3 | 7/0/0 | 0 | .016 | KO (exact-codec) |
| CROWN-huff vs PIL-m3 | 1/6/0 | 7 | ≈.30 [VERIFIED 2026-09-08] | TIE (reported, not hidden) |
| CROWN2-exact vs WebP-m6 | 7/0/0 | 0 | .016 | KO (exact-codec) |
| E16+2-way (3.2979, P) vs m6 | 6/1/0 | — | .16 [VERIFY] | TIE (audited; subagent overclaim corrected) |
| E16+3-way (3.2720, P) vs m6 | 7/0/0 | 0 | .016 | KO (probe-level; C-port = CROWN2 line) |
| CROWN3-exact vs CROWN2 | 7/0/0 | 0 | .016 | Win |
| B18/CROWN4 vs CROWN3 | 7/0/0 | 0 | .016 | Win |
| CROWN6 vs JXL-e3 | 5/2/0 | 6 | .219 | STANDS (losses 05/13 texture) |
| CROWN6 vs FLIF v0.4 | 0/7/0 | 0 | .016 | STANDS (new boss; beats e3 7/7 on our seven) |
| CROWN6 vs JPEG-LS RGB | 7/0/0 | 0 | .016 | KO (exact-codec; front-end lesson) |
| FULL-classical (3.2515, P) vs JXL-e3 | 3/4/0 | 12 [VERIFY] | n.s. | STANDS |

KO rules: require avg win + majority wins + p<.05. CROWN3 led JXL-e3 on avg (−0.78%, 5W-2L p=.375) [VERIFY] yet stood — same discipline that makes our KOs credible.

### 4.5 Ablations and the classical ceiling (probe_b14, P, exact-counted)

FULL stack (Q/WIDE-64/E16/3-way/merges/Y-splits, per-image path gate): **3.2515 avg (P)** vs JXL-e3 3.2291 (+0.69%, 3W-4L) [VERIFY]. Progression: MED-order-0 3.5782 → ANCH-H 3.3248 (−7.1%) → ANCH-R 3.2811 (−1.3%) → FULL 3.2515 (−0.9%) (P) [VERIFY]. FULL beats CROWN2-exact by −0.5% (P) [VERIFY].

Leave-one-out (per-image bpp table abridged; full in `probe_b14_RESULTS.md`):

| Arm | Avg (P) | Marginal vs FULL | Reading |
|---|---|---|---|
| FULL | 3.2515 | — | Ceiling |
| LOO1 noLOCO (→T 6/7) | 3.2847 [VERIFY] | +0.0332 (+1.02%) | Best partitioner; tree rediscovers most |
| LOO2 E6 (no E16) | 3.2594 [VERIFY] | +0.0079 (+0.24%) | Half standalone value (grouping covers rest) |
| LOO3 H-only (no G/R) | 3.3013 [VERIFY] | +0.0498 (+1.53%) | Largest, orthogonal, 7/7 — stacks cleanly |
| LOO4 noMerge | 3.2662 [VERIFY] | +0.0147 (+0.45%) | ≈1/3 of e3 gap |
| LOO5 noTree | 3.2515 [VERIFY] | +0.0000 | Subsumed (Q+WIDE already expresses splits) |
| LOO6 noGrid | 3.2515 [VERIFY] | +0.0000 | Subsumed |
| LOO7 noSplit (Y-split) | 3.2519 [VERIFY] | +0.0004 (+0.01%) | Absorbed by merges |

Marginals do not sum (Σ +0.106). Backends H:7/G:239/R:169 groups (P) [VERIFY]; groups→clusters 773→415 (P) [VERIFY]. Unstacked crumbs bounded ≈−0.01 → ~3.24, still short. Gap texture-concentrated (05 +0.136, 13 +0.102; other five net −0.02, P) [VERIFY].

### 4.6 Speed

HAPRE-C 0-ctx: 9.76 MP/s enc (E/P boundary) [VERIFY]; MOE ~2.9 MP/s enc / ~3 MP/s dec [VERIFY]; determinism 3/3 identical [VERIFY]. CROWN3+: encode ~5 min/image Python (ratio work only; encoder-Pareto effort stopped here) [VERIFY]; CROWN6 encode ~600 s/image wall + ~540 s one-time train (cached) (E, `crown6_results.json`) [VERIFY]; decode C ~140 ms (CROWN3) / ~230 ms (CROWN4) / 182–348 ms (CROWN6 per-image) [VERIFY]. rANS M=14; 8-bit LUT decode helps little when reconstruct dominates.

### 4.7 Near-lossless RD + JXL-lossy context (prototype family, exact Rice bytes — P)

`experiments/rd_sweep.py` (τ-quantized residuals, exact Rice+run bytes + header; metrics `quality.py`):

| Operating point | bpp | maxerr (RGB) | PSNR | SSIM | Level |
|---|---|---|---|---|---|
| τ0 lossless | = lossless arm | 0 | inf | 1.0 | E/P |
| τ1 | ~2.75–3.23 [VERIFY] | 3 [VERIFY] | 47.3 [VERIFY] | ~0.99 [VERIFY] | P (prototype) |
| τ2 | ~2.0–2.6 [VERIFY] | 4 [VERIFY] | 43.5 [VERIFY] | ~0.98 [VERIFY] | P (prototype) |
| JXL lossy d0.5 | ~0.8–1.2 [VERIFY] | — | ~42 [VERIFY] | — | Context |
| JXL lossy d1.0 | ~0.4–0.8 [VERIFY] | — | ~38–40 [VERIFY] | — | Context |

Near-lossless is NOT PSNR-competitive with lossy (expected; its value is maxerr guarantees). Functionally-lossless τ-track ties JXL-e3 rate at PSNR 47 dB in cycle-1 notes (P) [VERIFY]; needs JPEG-LS-near-lossless baseline before any FLLIC-track claim. Documented, not hidden.

## 5. Negative Results (full section — read this before extending the codec)

Each entry: config, exact delta, mechanism (measured, not hand-waving), file pointer. All deltas in bpp unless noted; (P)=probe-level exact-counted, (E)=exact-codec.

**Linear wall + LPC.** Per-block LPC-3 (16×16): −4% (overfit + headers) (P) [VERIFY]; flat-LS / IRLS-L1 ≈ MED (±0.5%) (P) [VERIFY]; FIR-12 + bit-gate +0.0% (P) [VERIFY]; lag-2 acf −0.28 real but not linearly exploitable (P) [VERIFY]. Mechanism: L2-optimal Gaussianizes; retry only with larger blocks / transmitted-only-on-gain + RD check, or PAQ-style online logistic mixer (crude gradient mixer −3.7%, P) [VERIFY].

**LF family (3 variants, all retired).** Naive 16-bin LF conditioning −8..−12% (table fragmentation) (P) [VERIFY]; clustered LF +7.50% avg (side ~1.1 bpp ≈3× the ~0.4 bpp conditioning saving; bin map 2b/px not even counted) (P) [VERIFY]; Squeeze-NN LF-prediction +13.48% (P, units audited: 3× bpp vs /3ch confusion fixed, delta valid) [VERIFY]; smooth upsampling (bilinear +10.77%, smoothed-NN +10.54%, anchor 3.577 PASS) recovers only ~3 pp (P) [VERIFY]. Mechanism: transmitted-4×-LF side ~0.40 bpp structurally > savings; NN block edges poison MED (var ratio 1.073, P) [VERIFY]. Weapon parked; LF-PREDICTION from decoded coarse (no side) remains the only untested structural variant (HPCM lesson).

**Transforms: DCT + wavelet + hybrid (density mechanism).** Integer-DCT + order-0 subband Huffman: 6.95 avg vs 3.464 (P) [VERIFY] (unnormalized-core flaw noted). 5/3 wavelet best (2-level + LL-DPCM + 12-group conditioned Huffman): 3.3737, +3.8% vs ceiling, 0W-7L (P) [VERIFY]; bitplane-B +26% (P) [VERIFY]. Hybrid Y-pixel + chroma-wavelet: 3.3306 (best transform-family, −1.3% vs b15) but +2.4% over ceiling (P) [VERIFY]. Mechanism: 5/3 detail bands dense (48–86% nonzero) [VERIFY] → zerotree significance costs ~3b/coeff unharvestable; wavelets scatter smooth-field predictability (worst gaps on easy images 07/19/01); order-0 sweeps 27/27 detail subbands (P) [VERIFY]; channel asymmetry unanimous (Y-pixel always, Co/Cg-transform always). Family exhausted (1/2/3-level, finer groups, cross-channel all negative). We do not pursue stripped EBCOT further.

**Byte-LZ.** LZ77-on-residual-bytes: 4.18 avg (report-level P, no per-image ledger; see VERIFY) [VERIFY] (destroys symbol structure; standalone LZ core 4 KB window greedy 36 ms/786 KB PASS but rejected for residuals). Wavelet+zlib: 4.78 avg (report-level P, no per-image ledger) [VERIFY] (zlib can't use 2D structure; keep zlib for speed mode only).

**Bands.** Fixed B=2 −0.012, B=4 −0.017 bpp (~0.4%, P) [VERIFY]. Mechanism: table fragmentation at small scale; portrait kodim19 outlier (−0.05/−0.07, P) [VERIFY] proves alignment is everything — content-adaptive or nothing.

**Bias-under-Huffman theorem.** LOCO-I-lite bias −0.2% (report-level P, no primary ledger) [VERIFY]; M-GLOB 0/7, M-GROUP 0/126 cells (P) [VERIFY]. Proven: shift preserves symbol-count multiset ⇒ Huffman data + alphabet bit-identical (Δ=0 before side; side loses by construction). Golomb-only weapon (G-bias −0.36% on kodim13, report-level P, untraced figure — bank or drop pre-camera-ready) [VERIFY]. Retired under Huffman.

**Hash-cache.** 64-entry exact-repeat escape +2.85% on photos (P) [VERIFY]; hit 25.0% mean [VERIFY]; flag 1.000 b/px + index ~5.93 b/hit + selection bias (hits strike already-cheap flats). Only kodim02 −0.96% clears overhead (P) [VERIFY]. Keep as non-photo fast path only.

**Enumerative 2×2 (beautiful dead end).** Cover-73 weight+rank on MED blocks: +0.32% (P) [VERIFY]. Mechanism: H(pattern|w)≈logC (ranks buy ~0); conditioning on nonzero inflates value entropy (Hz>H0); infinite-precision headroom caps at +0.5% (P) [VERIFY]; larger blocks worse. Corrigendum: b9's "170×" is P(all-active) mislabeled; true all-zero ~7× overdispersion (P) [VERIFY]. Monetizable joint info is in magnitudes-given-activity → magnitude-VQ branch.

**VQ-partial.** Q-T4 2×2 magnitude-VQ: −1.84% gated vs MED order-0 (P, 7/7, 84% of 0.079 b/res TC harvested, signs independent) [VERIFY]; pyramid factorization −0.06% (never factorize what you can tabulate) (P) [VERIFY]. Transfer stacks at −0.52% Huff / −1.25% rANS→3.3013 (P) [VERIFY] but behind CROWN2-exact 3.2686 (overlap ~0.1–0.2; grouping consumes ~72%) (P) [VERIFY]. Naive regrouping +3.85% quad / +2.12% pair (P) [VERIFY]. Rule: grouping granularity dominates; never disturb per-pixel adaptation for VQ.

**Joint-adaptivity 7:1 + global-adaptivity overlap.** A-act −2.18% / C-gol −1.74% MED-frame (P, 7/7) [VERIFY]; transfer est best-of 3.1138 (7W-0L p=.016, P) [VERIFY] IF ~65% transfers. CROWN5 build: best-of picks static 21/21; bytes = CROWN4+21 B (P) [VERIFY]. Joint (group×activity) backward-adaptive: +3.36% vs static, 0W-7L (P) [VERIFY]. Mechanism 7:1 against: table prize 0.018 vs cold-start/whiplash deficit +0.11 (P) [VERIFY]; WIDE-K static strictly finer than 4-bin global adaptive (~100% overlap). Static oracle picks 21/21 planes. Entropy-coding arc closed; remaining gap prediction-side.

**LF-prediction smooth (see LF family), MA-tree partial, A-joint retired-in-frame, I-GATED retired, grids, M-THR fast path kept.** MA-tree-lite deep ≤16 leaves + sharing: 3.3242, −0.56% vs CROWN-huff, 6/7 (P) [VERIFY]; shallow loses; rANS projection ≈3.254 (P) [VERIFY] — ≈1/3 of e3 gap, not a killer. A-joint (Co,Cg) pairs: −1.58% order-0 frame (P) [VERIFY] but +7..+17% under E16 groups in-frame (fragmentation) — retired in-frame. I-GATED (MDL-gated Y-split): −0.29% 7/7 (P) [VERIFY] but +0.0003 in-frame (WIDE-K harvests it) — retired. M-THR grids: −0.52% 7/7 (P) [VERIFY]; lose on Kodak in-frame, kept as torture-input fast path. Histogram sharing / merges: the one sharing idea that pays (see §4.5).

## 6. Bug Taxonomy (for reproducibility — every bug below bit us)

1. **Causality peeks.** First-row `d` peeked at undecoded pixel; edge-replicate MLP taps peeked at (0,0). Fix: 0/left/top + zero-border causal taps; forced-MLP test catches it.
2. **L=0 framing.** Runs of length 0 must emit tokens (JPEG-LS semantics); empty-group rANS framing (count+A written, n+pay skipped) mirrored; undecoded-pixel peeks break framing.
3. **Alphabet clipping + proven bounds.** ±256 clipping = silent corruption on saturated edges. Fix: ±1024 + proven bounds (|r|≤702) + loud asserts; decoder −4, never clip.
4. **rANS precision.** M must satisfy 2^M >> A (A=232 needs M=14, not 10).
5. **Border matching.** Encoder/decoder must share the exact convention (0/left/top; `d:=a` row0); `(*([0]*n))` zero-init aliasing bug in C buffers.
6. **Unary polarity wire-splits.** libhapre-era Golomb (q ONES + `0`) vs CROWN polarity (q ZEROS + `1`); vendored packer + documented split.
7. **tanh 1-ulp + LUT fix.** torch-tanh vs C-libm differ 1 ulp → int-rounded flips at .5 boundaries. Fix: frozen LUT 2⁻¹⁴, bit-exact literals, `-ffp-contract=off`, 0/17.2M mismatches required.
8. **Quantizer scale bug.** LS weights rounded without ×16 pre-scale → all-zero → invalid +0.007% run; caught by mechanism diagnostics, invalidated + rerun.
9. **Cut-semantics off-by-one.** Bounds use `c+1`; greedy-overshoot (fewer groups than K) normal.
10. **Greedy-overshoot / map framing.** Dense bitmask + `ceil(active·gbits/8)` ids; `na` field saved via mask popcount; `p==len(blob)` asserted.

Additional: Huffman bytes too-good ⇒ check bound-vs-exact; synthetic-vs-natural divergence ⇒ suspect run/dictionary absence; Python bit loops ⇒ vectorize or move to C; numpy row-loop overhead can exceed C-ref encoders — profile before optimizing.

## 7. Limitations & Future Work

**Texture gap.** CROWN6 leads on average yet loses kodim05 (+1.95%) and kodim13 (+1.00%) (E) [VERIFY per-image]. Autopsy: 05's entropy floor 3.500 already beats JXL 3.5120 — loss is ~90% coding overhead (gaps 0.077 + maps 0.008 + WAVG-side 0.013, P) [VERIFY]; K=64 fine groups → tables too big → Golomb default misfits heavy tails. Chroma-texture energy splits wins/losses; Y energy does not. Larger-neighborhood prediction (not finer coding) is the remaining path.

**Learned scale.** Classical ceiling 3.2515 (P) + neural-micro to 3.1978 (E). True e9 bar ≈2.94 (P) [VERIFY] needs learned-MA + per-group RCT + precise entropy at JXL scale — rebuilding JXL around our backend would be a multi-person effort, so we pursue the learned-micro arc (per-group learned predictors + offline-trained trees, "learn the tables, ship the tables"). Finer MLP variants stopped at 10× short of their bar (cycle-35).

**Speed.** Encode is slow-Python (~5–10 min/image, P/E) [VERIFY]; decode is C (~140–350 ms, E) [VERIFY]. Encoder-Pareto work stopped at CROWN3+; ratio game only. Noise expands (+9% over raw 8.0, P) [VERIFY] and old 0-ctx driver hard-fails decode on noise (rc=−2) — robustness TODO before publish. Small-image side (≈2 KB ≈0.04 bpsp on Kodak, P) [VERIFY] affordable only if MDL-gated.

**What we will not do.** Finer MA splits without gates (JXL e8/e9 over-split lesson), squeeze/LF-side streams (retired 4×), transform-bitplane arcs (retired), byte-LZ on residuals (dead twice), bigger expert banks without gating, decode-time NN.

## 8. Conclusion

From HAPRE-C (3.464, E) to CROWN6 (3.1978, E) we climbed the exact-bytes ladder one gated mechanism at a time, killing seven bosses and stopping at three, losses on record. The classical ceiling (3.2515, P) with leave-one-out marginals tells the next worker where hand contexts end; the micro-MLP step (→3.1978, E) tells them where learning begins to pay inside an exact stream. The dead-ends (§5) and bugs (§6) are the other contribution — a map of where not to drill.

All streams are exact, all round-trips asserted, all losses reported. The code (`src/crown*.c` + `src/driver_crown*.py` + `src/CROWN*FORMAT.md`, plus the `cpp/` C++ port), probes (`probes/probe_b*_RESULTS.md`), surveys, and boss takedown are banked for the next cycle.

## Appendix A. Stream-format reference (normative pointers)

CROWN2 (`C2`, Q/GRID, E16, 6-bit `[predid:4][backend:2]`), CROWN3 (`C3`, +RCT{6,27,12} + WAVG-G32, 7-bit `[predid:5][backend:2]` ids 0..16), CROWN4 (`C4`, +LMS5_T0/T3 ids 17/18, zero-side LMS), CROWN5 (`C5`, per-channel best-of-3 adaptive/static — adaptive loses, static 21/21, bytes=CROWN4+21 B), CROWN6 (`C6`, +MLP id 19, thresholds/scales/weights side, LUT wire-forward). Full byte tables in `src/CROWN2_FORMAT.md` … `src/CROWN6_FORMAT.md`. Golden snapshots: `src/hapre.c.golden-moe/run/crown/e16`.

## Appendix B. Disclosure appendix

Probe (P) vs exact (E): E = `driver_crown*.py` assembled bytes + C decode + `decode==original` byte-compare; P = numpy/heapq/`rans_encode` exact-counted ledgers + scalar-sim / table / Kraft / map-cover proofs at the same counting rule. Transfer estimates (e.g. b19 3.1138, P) assume additive MED-frame deltas and are labeled estimates, not codecs. Moving bars noted (e3-C27 3.2083, tuned-e9 ~2.94). No boss claimed beyond its Wilcoxon row (Table §4.4). Near-lossless RD is prototype-family (Rice), not the CROWN backend.

## References

Classical: Weinberger–Seroussi–Sapiro LOCO-I/JPEG-LS (HPL-98-193); Wu & Memon CALIC 1996; JPEG 2000 5/3 + EBCOT; WebP-lossless; FLIF/MANIAC (Sneyers & Wuille 2016) → JXL MA-trees (Alakuijala et al. 2019; spec §5.2.2); Duda ANS; Cover enumerative TIT-19(1) 1973; Fischer PVQ TIT-32(4) 1986; Dai & Zakhor DCC 2003.

Learned (alphaxiv IDs in survey): 2412.17464 CALLIC; 2511.10991 HPAC; 2509.07704 SEEC; 2502.16163 LLM-visual-prompts; 2607.08221 LUMI; 2606.06273 Diffusion-LM; 2604.15472 Chained predictors; 2509.18815 FlashGMM; 2605.02726 Cool-chic 5.0; 2507.19125 LIC-HPCM; 2412.00505 C3/WD-C3; plus DLPR (Bai et al. TPAMI 2022), scalable ℓ∞ (Bai et al. CVPR 2021), MLIC++, FLLIC (Zhang & Wu 2024), HiDE 2026, MoE-entropy 2026, Tree-VQ 2609.03641, Cool-chic 2609.04274, FineZip 2024.

Codecs/systems: cjxl/djxl 0.11.2; PIL 12.2.0; libjxl (web-read audit); qoi.h (phoboslab, 649 lines, `survey/qoi/`); QOIR; Blend2D; Cloudinary modular explainer.

---

*End of draft v1.0. Next: re-run every [VERIFY] with pinned drivers, fill PNG-9/WebP-m0 per-image from `src/perimage.py`, expand RD to 7-image exact-CROWN backend, then camera-ready.*