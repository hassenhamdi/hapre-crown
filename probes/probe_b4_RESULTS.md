# probe_b4 RESULTS — Huffman-arm Boss-2 campaign: sign-flipped LOCO clusters + per-group predictors

Branch B4 (exploratory). numpy + PIL only (rANS byte counts via the EXISTING
`libhapre.so` M=14 coder through ctypes — no existing file modified).
UNIT RULE throughout: `bpp = total_bits/(H*W*3)`. Every byte counted:
per-stream `sum(count*len)` from real heapq lengths `+16+A*24` table bits
(Huffman arms), rANS arms use measured `rans_encode` payload + `16+A*32`
freq-table bits; context-map side, predictor-id side, K-id side, 64-bit header.
`A` = alphabet size of the stream. Anchors reproduced within ±3% (see §1).

## 1. Anchors (exact, this harness)

| image | MED order-0 Huffman (expect ~3.58) | MED CTX9 Huffman (expect ~3.537) | champion MOE/C bytes (pinned) |
|---|---|---|---|
| kodim01.png | 3.6179 | 3.5953 | 3.56 |
| kodim02.png | 3.3256 | 3.2961 | 3.23 |
| kodim05.png | 4.0119 | 3.9236 | 3.84 |
| kodim07.png | 3.1530 | 3.0967 | 3.01 |
| kodim13.png | 4.2378 | 4.2069 | 4.17 |
| kodim19.png | 3.5134 | 3.4790 | 3.41 |
| kodim23.png | 3.1880 | 3.1455 | 3.03 |
| **AVG** | **3.5782 (PASS, +0.0% of 3.58)** | **3.5347 (≈3.537)** | **3.4643** |

Border convention = champion 0/left/top rule (NOT edge pad); YCoCg-R round-trip
asserted every image. Champion bytes re-measured via `csrc/driver_moe.py`: PASS 7/7.

## 2. WINNER — CROWN-huff (best Huffman-arm configuration)

MED + **sign-flipped LOCO-365 gradient contexts** → per-channel **auto-K
quantile clustering** (K ∈ {2,3,4,6,9,12,18,27,36}, sort key = per-context mean
|residual|, equal-pixel partition) → **per-group best predictor** (best of
{MED,TOP,PAETH,GRAD,GAP,DG} by exact Huffman data bits) → per-group Huffman
tables. **No run mode** (ablated: runs HURT by +0.007 under these contexts).
No block ids. Side counted: map `active_ctx*ceil(log2K)` + 4b K-id + 8b header
per channel + 3b predictor-id per group + 64b global.

| image | CROWN-huff bpp | vs champ 3.4643 | per-channel (K, ngroups) |
|---|---|---|---|
| kodim01.png | 3.4340 | −0.1260 | Y(6,6) Co(36,25) Cg(36,24) |
| kodim02.png | 3.0931 | −0.1369 | Y(9,9) Co(36,29) Cg(36,22) |
| kodim05.png | 3.7394 | −0.1006 | Y(6,6) Co(36,29) Cg(36,27) |
| kodim07.png | 2.8630 | −0.1470 | Y(6,6) Co(36,24) Cg(36,21) |
| kodim13.png | 4.1010 | −0.0690 | Y(4,4) Co(18,17) Cg(36,27) |
| kodim19.png | 3.2758 | −0.1342 | Y(6,6) Co(36,26) Cg(36,22) |
| kodim23.png | 2.8940 | −0.1360 | Y(36,30) Co(36,27) Cg(36,23) |
| **AVG** | **3.3429** | **−0.1214 (−3.50%), 7/7, Wilcoxon W=0 p≈.016** | — |

STRETCH (same streams, rANS M=14 backend): **CROWN-rans AVG = 3.3124**
(01:3.4002, 02:3.0590, 05:3.7176, 07:2.8060, 13:4.0879, 19:3.2522, 23:2.8636;
−0.1519 vs champ, −4.38%).

Proofs: decoder-simulation RECON PASS 7/7 (causal key/sign/pred/map/run
re-derivation from recon); canonical-Huffman round-trip asserted on EVERY
stream of all 7; rANS `rans_decode` asserted on EVERY stream of all 7.

## 3. Verdict on Boss 2 (WebP-m3)

- vs **campaign-pinned 3.38**: CROWN-huff **WINS by 0.037 (1.1%)** → **BOSS 2 DEAD**.
  CROWN-rans wins by 0.068 (2.0%).
- Honesty footnote: no `cwebp` binary exists locally; PIL/libwebp-1.6.0 `-m 3`
  measures 3.3452 avg on the same files (01:3.4074, 02:3.0659, 05:3.8563,
  07:2.8397, 13:4.0990, 19:3.2730, 23:2.8754). Against THAT toolchain:
  CROWN-huff ties (+0.0023, parity), CROWN-rans wins (−0.0337, 1.0%, 5/7 images).
  Campaign pins are the official target and are beaten either way by rANS.
- Bonus: CROWN-rans 3.3124 also edges pinned WebP-m6 (3.32) by −0.008 and local
  PIL-m6 (3.3160) by −0.004. JXL-e3 (3.23) still stands.

## 4. Exploration trail (kept / killed, avg bpp, Δ vs MED-CTX9 3.5347)

| idea | avg | Δ | fate |
|---|---|---|---|
| global per-(ch,energy-ctx) mean/median bias | 3.5349 | +0.0002 | KILLED (means already ~0) |
| LOCO-365 global mean bias + CTX9 coding | 3.5366 | +0.0019 | KILLED (Huffman insensitive to Golomb-asymmetry; confirms LOCO-lite −0.2%) |
| inter-channel Cg-from-Co linear (corr Co–Cg +0.17!) | 3.5309 | −0.0039 | KILLED (α tiny, gain noise) |
| finer energy bins >>3/16 tables | 3.5256 | −0.0091 | KILLED (fragmentation, cf. bands) |
| oracle per-(ch,ctx9) best predictor (7-way) | 3.5066 | −0.0282 | folded into winner (strictly weaker) |
| LOCO-365 → K equal-pixel clusters, no flip (K=9) | 3.4224 | −0.1124 | KEPT as stepping stone |
| + JPEG-LS sign flip (rf = s·res) | 3.3889 | −0.1459 | KEPT (flip fixes bimodal groups; um729 variant 3.4115 weaker) |
| deterministic magnitude-tier clusters (ZERO side) | 3.4530 | −0.0818 | fallback (beats champ side-free) |
| entropy sort-key / K=6..18 sweep | 3.42–3.43 | — | kept best (autoK 3.3714 pre-predictor) |
| per-group best predictor (G) | 3.3491 | −0.1856 | KEPT (core of crown, −0.019 over clusters) |
| Lloyd re-cluster round 2 | 3.3478/3.3414 | −0.001 | marginal; DROPPED (complexity > gain) |
| energy sub-split inside groups (2× tables) | 3.3788 | +0.030 vs G | KILLED (fragmentation again) |
| µMoE 8×8 block experts + flip clusters + runs | 3.3803 | +0.012 vs MED version | KILLED (expert gross < 0.047 ids side once contexts are good; gains overlap) |
| run mode on flip clusters | +0.007 | hurts | KILLED (flat pixels already ~1 bit; gamma ≥ saving) |
| rANS M=14 per stream (exact bytes, decoded) | −0.019…−0.031 | — | KEPT as stretch (freq tables eat ~1/3 of Huffman gap) |

Web/academic sources used: LOCO-I/JPEG-LS report (Weinberger–Seroussi–Sapiro
HPL-98-193: bias-from-B/N, 365 merged contexts, sign-flip, Golomb-tuning —
key insight that bias is Golomb-specific, hence global-Huffman mismatch);
CALIC (GAP + 144-texture/576-compound contexts → motivated gradient-pattern
contexts over 1-D energy); JPEG-XL/context-map clustering (forward clustered
maps with transmitted assignment → our map-side model); ANS reviews
(Huffman 1-bit/symbol floor vs rANS near-entropy → motivated exact rANS count).

Why it wins (mechanism): 1-D energy contexts conflate edge orientations with
different residual statistics; 365 gradient-pattern contexts separate them
(data −0.11); sign-flip restores symmetry inside merged groups (further −0.03);
per-group predictor choice harvests the MOE gain without block-id side
(−0.02, MED picked in most groups, GAP/GRAD/PAETH/TOP/DG steal edge groups).

## 5. EXACT C changes to productize CROWN-huff (no C files touched)

1. `hapre.c`: add `loco_key_sign(const int16_t *P,int H,int W,int16_t *key,int8_t *sgn)`
   — champion borders (0/left/top; `d=a` row 0, `d=b` last col), gradients
   `Q1=b-c,Q2=c-a,Q3=d-b`, quant thresholds 3/7/21, lexicographic sign merge to
   ids 0..364. Pure function of recon → callable in encode pass 1 and streaming decode.
2. `hapre.c`: generalize `pack_syms`/`huff_unpack_ctx9` tables from 27 to
   `cd/ln[36*2049]` (max ngroups=36); group index replaces `(ch,ctx)` key.
3. Encoder (new `driver_clus.py` or C `clus_encode`): pass 1 = keys/signs +
   all-6 `pred_expert` residuals; per channel replicate `auto_groups_exact`
   (K∈{2,3,4,6,9,12,18,27,36}, exact `sum(count*len)+16+A*24` + map-side model)
   + per-group argmin over experts; pass 2 = emit header + per-channel
   (K-id 4b, 365-entry map of `ceil(log2K)`-bit group ids, 3b predictor ids,
   canonical tables, `pack_syms` payload). Drop `med_run/moe_run` + gamma paths.
4. Decoder: parse map → per-pixel group from recon key → predictor id →
   `pred_expert(...)` → `recon = pred + sgn*rf`. Mirrors `probe_b4_i.py:prove`.
5. rANS stretch: swap step-3 tables for `rans_norm/rans_encode` per group with
   framing `2B A + A*(2B sym+2B freq)` + payload (4B state included), decode via
   `rans_slots/rans_decode` (all asserted in `probe_b4_h.py:rans_stream_bits`).

## 6. Top follow-up

DP-optimal (not equal-pixel-greedy) partition of meanabs-sorted LOCO contexts
with the exact rate criterion incl. map side — greedy overshoot on the massive
flat context leaves uneven groups (e.g. 25 effective of K=36); DP should recover
~0.005–0.01. Second: per-group Golomb-vs-Huffman backend choice bit (groups with
tiny alphabets may prefer Golomb). Files:
`experiments/probe_b4_{a,b,c,d,e,f,g,h,i,j}.py` (this report:
`experiments/probe_b4_RESULTS.md`).
