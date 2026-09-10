# probe_b19 (CABAC-class adaptivity vs static-table overhead) — RESULTS

Campaign: lossless codec CROWN4-exact 3.1993 bpp Kodak-7 avg vs boss JXL-e3 3.2291.
We lead avg (−0.92%), 5W-2L, p=.375 n.s. Blockers: kodim05 (+2.27%), kodim13 (+0.98%).
b18 autopsy PROVED 05's loss is ~90% entropy-coding overhead (floor 3.500 beats JXL
3.5120; K=64 fine groups → oversized tables → Golomb fallback misfits heavy tails).
Mission: kill the overhead with backward/forward-adaptive coders (CABAC/JPEG-LS
principles, integer-exact, decoder-compatible). No regressions on the other five.

Code (new files only, prefix `probe_b19_`, nothing existing modified — existing
modules used by import): `probe_b19_adaptive.py` (frame + binary arithmetic coder +
adaptive-Golomb + run/interrupt machinery), `probe_b19_run.py` (all-7 driver +
round-trips), `probe_b19_nums.json` (machine-readable numbers).
Rules: numpy+PIL only, CPU, no torch. Unit bpp = total_bits/(H·W·3) throughout.

## 0. Frame + anchors — PASS

Fixed front end for ALL configs (predictor held constant, pure entropy-coding test):
RGB → C6 YCoCg-R (`B17.rct_fwd` perm0 t6, round-trip asserted) → causal MED
(`B17.med_pred`, CROWN2 border doctrine) → int32 residuals (|r|≤1024 asserted).

- S0 anchor MED + C6 + order-0 global Huffman per plane (real tables 16+A·24):
  **3.5782 avg** (−0.05% vs 3.58, PASS ±3%; reproduces b17/b18 anchors to 4 decimals).
- Per-image S0: 01:3.6179, 02:3.3255, 05:4.0119, 07:3.1529, 13:4.2378, 19:3.5133,
  23:3.1879. (C6-frame; CROWN4 refs use min-over-{C6,C27,C12} — transfer caveat §5.)

## 1. Backends (all exact bits, all side counted)

- **S0** static order-0 Huffman/plane (baseline). Side: 16+A·24/plane.
- **S0-G** static best-k Golomb/plane (+4b k). Reference: shows static-k compromise.
- **A-pos** backward-adaptive binary arithmetic, Exp-Golomb binarization of
  interleaved magnitude M=2|r|−(r>0), prefix bins through 4 pos-indexed binary
  contexts, suffix raw (bypass). Uniform init c0=c1=1, cap 128 with halving.
  Side: 2×32b stream lengths/plane. NO activity conditioning (ablation).
- **A-act (MAIN)** same + 4 activity bins (e=|a−c|+|b−c| from causal recon,
  static thresholds (4,12,48)) × 4 pos = **16 binary contexts**. Side same. Zero tables.
- **B-fwd** A-act + transmitted 4b/ctx init levels (16×4b=64b/plane fitting
  empirical prefix-p1). Tests whether forward side pays.
- **C-gol** per-activity-ctx adaptive Golomb-k (JPEG-LS A/N counters, raster order,
  init N=1/A=4, RESET=64): k from (N<<k)≥A, code M with q=M>>k unary + k rem bits.
  Side: none (init/RESET static). Raster order (NOT group-scattered — the b18 kill
  was per-group-scattered whiplash; this is the fix).
- **D-run** activity-gated run/interrupt over C: act==0 → run length L (L=0 allowed,
  JPEG-LS semantics) via global adaptive Golomb run-coder + interrupt M−1
  (M≥1 guaranteed) via flat-ctx Golomb state; act≠0 → C path. Side: none.

Decoder-safety (explicit): every adaptation state (binary c0/c1 per ctx, Golomb
N/A/k per ctx, run coder N/A/k, activity bin, MED pred) is a **pure function of
already-decoded symbols/pixels** (+ B's transmitted init levels + static RESET/
threshold schedules known to both sides). A-act/C/D uniform-init decoders start
from the same constants and apply identical integer updates in identical raster
order — proven by asserted real decodes (§4). B's init is transmitted side (counted).

## 2. All-7 exact numbers (bpp; bold = column best)

| img | S0 | S0-G | C-gol | D-run | A-pos | A-act | B-fwd | CROWN4 | JXL-e3 |
|---|---|---|---|---|---|---|---|---|---|
| kodim01 | 3.6179 | 3.7299 | 3.5957 | 3.5794 | 3.6156 | **3.5822** | 3.5825 | 3.2991 | 3.3593 |
| kodim02 | 3.3255 | 3.4355 | 3.2971 | 3.3024 | 3.3142 | **3.2600** | 3.2603 | 2.9854 | 3.0611 |
| kodim05 | 4.0119 | 4.1610 | **3.8897** | 3.8936 | 4.0165 | 3.8942 | 3.8944 | 3.5917 | 3.5120 |
| kodim07 | 3.1529 | 3.3455 | 3.0741 | 3.0892 | 3.1219 | **3.0283** | 3.0285 | 2.6790 | 2.7310 |
| kodim13 | 4.2378 | 4.3778 | **4.1774** | 4.2193 | 4.2756 | 4.2269 | 4.2273 | 3.9525 | 3.9141 |
| kodim19 | 3.5133 | 3.5912 | 3.4630 | 3.4981 | 3.4747 | **3.4346** | 3.4347 | 3.1510 | 3.2154 |
| kodim23 | 3.1879 | 3.2910 | 3.1148 | 3.1371 | 3.1438 | **3.0762** | 3.0763 | 2.7364 | 2.8110 |
| **AVG** | 3.5782 | 3.7046 | 3.5160 | 3.5313 | 3.5660 | **3.5003** | 3.5006 | 3.1993 | 3.2291 |
| ΔS0 | — | +3.53% | −1.74% | −1.31% | −0.34% | **−2.18%** | −2.17% | — | — |

Per-image ΔS0 extremes: A-act best −3.95% (07), −3.50% (23), −2.93% (05-target);
worst −0.26% (13). C-gol best −3.04% (05), −1.42% (13); C beats A-act on BOTH
texture blockers (05 by 0.0045, 13 by 0.0495) and loses on smooth/peaked images
(07 by 0.0458). Per-plane best-of-{S0,C,A-act} (+6b choice side): 3.4730 avg
(−2.94% vs S0; 05:3.8585, 13:4.1459 — finer selection helps texture most).
RESET sweep on 05 (C-gol exact): 32:3.8904 / 64:3.8897 / 128:3.8889 / 256:3.8887
— insensitive (0.04% range); locked RESET=64 per JPEG-LS convention.

## 3. Overhead-split analysis (bpp; H0 = order-0 entropy, Hc = activity-4 entropy)

| img | H0 | Hc (MI) | S0 (ovh) | A-act (ovh) | C-gol (ovh) |
|---|---|---|---|---|---|
| 01 | 3.5526 | 3.4840 (0.069) | 3.6179 (+0.065) | 3.5822 (+0.030) | 3.5957 (+0.043) |
| 02 | 3.2542 | 3.1604 (0.094) | 3.3255 (+0.071) | 3.2600 (+0.006) | 3.2971 (+0.043) |
| 05 | 3.9719 | 3.7860 (0.186) | 4.0119 (+0.040) | 3.8942 (−0.078) | 3.8897 (−0.082) |
| 07 | 3.1084 | 2.9748 (0.134) | 3.1529 (+0.045) | 3.0283 (−0.080) | 3.0741 (−0.034) |
| 13 | 4.2014 | 4.0922 (0.109) | 4.2378 (+0.036) | 4.2269 (+0.026) | 4.1774 (−0.024) |
| 19 | 3.4501 | 3.3538 (0.096) | 3.5133 (+0.063) | 3.4346 (−0.016) | 3.4630 (+0.013) |
| 23 | 3.1219 | 3.0087 (0.113) | 3.1879 (+0.066) | 3.0762 (−0.046) | 3.1148 (−0.007) |
| AVG | 3.5229 | 3.4086 (0.114) | 3.5782 (+0.055) | 3.5003 (−0.023) | 3.5160 (−0.007) |

Mechanism (two components, both measured):
(i) **Table elimination** ≈0.02–0.04 bpp: static Huffman pays 16+A·24/plane
(avg ovh +0.055); adaptive states are tableless (64b lengths only).
(ii) **Conditioning harvest** ≈0.04–0.08 on texture: A-act/C beat H0 wherever
MI is large (05 MI 0.186, harvests 0.078/0.082 = 42–44%; 07 MI 0.134 → 0.080).
05's gain (−0.118) is 3× its static ovh (0.040) — the win is mostly conditioning
plus Golomb-shape repair, not just rounding. C-gol beats H0 on 05/13/07/23:
adaptive-k tracks local scale where static-k compromises (S0-G is +3.5% — the
static-k mixture penalty; adaptive-k recovers 5.1pp of it).
A-pos (no activity) proves conditioning is the active ingredient: −0.34% avg and
LOSES on both texture images (05 +0.005, 13 +0.038) — unconditioned backward
adaptation whiplashes on texture, replicating b18's adaptive-k kill (+0.16–0.23)
and vindicating the raster-order per-ctx fix (C-gol −1.74% with the same RESET
family, only regrouped by causal activity instead of scattered groups).

## 4. Killed / wounded / winners

- **WINNER A-act (−2.18% avg, 7/7 vs S0)**: real integer range coder, actual bytes
  counted (arith + raw-bypass + 64b lengths; byte-padding counted), round-trip
  PASS 7/7×3 planes (raster-order real decode, recon==input asserted per plane).
- **WINNER C-gol (−1.74% avg, 7/7 vs S0; beats A-act on 05+13)**: exact Golomb code
  lengths (deterministic q+1+k per symbol under mirrored k-sequence); real
  bitstream write/read round-trip PASS on kodim05-Y (2098526 bits, analytic ==
  stream); N/A/k update pure function of decoded residuals.
- **KILLED B-fwd (+0.0003 vs A-act)**: 64b/plane transmitted init buys nothing —
  uniform init + cap-128 halving adapts within ~100 symbols. Forward-init retired
  (round-trip machinery proven PASS on 05-Y, but never worth the side).
- **KILLED D-run (−1.31%, loses to C by 0.43pp)**: helps only the smoothest image
  (01: −0.016 vs C) via long zero-runs, hurts texture (13: +0.042 vs C from L=0
  run tokens + run-coder overhead on sparse flats). Counts are exact Golomb
  lengths over verified shared machinery (prefix-length sanity on 20k-symbol
  prefix: C=80869 vs D=82003, run loses immediately on texture); full D-stream
  decode = C-port follow-up (D is killed, so no exactness debt touches the winner).
- **KILLED A-pos**: unconditioned adaptation insufficient (see §3).

## 5. Transfer math vs CROWN4 + Boss-5 verdict (ESTIMATE, labeled)

Additive transfer est_i = CROWN4_i + (MED-backend_i − MED-S0_i). Front-end caveat:
MED deltas measured C6-only; CROWN4 refs are min-over-{C6,C27,C12} (05=C27,
13=C12/C6). RCT-coding interaction assumed second-order (both are YCoCg-R
variants; entropy-coding effect transfers).

| img | CROWN4 | A-act est (vsJXL) | C-gol est (vsJXL) | best-of est |
|---|---|---|---|---|
| 01 | 3.2991 | 3.2634 (−2.85%) | 3.2769 (−2.45%) | 3.2634 |
| 02 | 2.9854 | 2.9198 (−4.62%) | 2.9570 (−3.40%) | 2.9198 |
| 05 | 3.5917 | 3.4740 (−1.08%) FLIP | 3.4695 (−1.21%) FLIP | 3.4695 FLIP |
| 07 | 2.6790 | 2.5544 (−6.47%) | 2.6002 (−4.79%) | 2.5544 |
| 13 | 3.9525 | 3.9417 (+0.70%) lose | 3.8922 (−0.56%) FLIP | 3.8922 FLIP |
| 19 | 3.1510 | 3.0723 (−4.45%) | 3.1007 (−3.57%) | 3.0723 |
| 23 | 2.7364 | 2.6246 (−6.63%) | 2.6632 (−5.26%) | 2.6246 |
| AVG | 3.1993 | 3.1215 (−3.33%) | 3.1371 (−2.85%) | 3.1138 (−3.58%) |

KO math (est): A-act est diffs (est−JXL) = {−0.0959,−0.1413,−0.0380,−0.1766,
+0.0276,−0.1431,−0.1864} → 6W-1L, W+=1, p=.031 <0.05. C-gol est diffs all negative
→ 7W-0L, p=.016. Best-of → 7W-0L, p=.016. **Under additive transfer, adaptivity
KOs Boss-5 (first method to flip 05 in any frame, and best-of flips both blockers).**
Honesty discount (overlap): CROWN4's LOCO-365 quantile groups already condition on
gradients ≈ our activity, so part of the MED-frame gain is harvested there. Flip
thresholds: 05 needs 68% transfer (A-act) / 65% (C-gol); 13 needs 64% (C-gol).
Plausibility: 05's CROWN-Y groups are all-but-one Golomb (autopsy) — grouping
harvests ~0 there and the loss is pure Golomb shape-misfit, which adaptive-k
repairs orthogonally → high transfer on 05 expected. 13 is mixed-backend → transfer
moderate, 13-flip less certain. Sensitivity: at 50% transfer, avg still beats JXL
(~3.16, −2.1%) but 05 stays +0.5% → KO fails. **Verdict: ADAPTIVITY IS THE WEAPON
(frame-level proof on fixed frame + transfer-KO math), but the exact KO
requires the CROWN4-port build (transfer est ≠ exact codec). Boss-5 STANDS at
exact-codec level tonight; the path to KO is now a defined construction, not a search.**

## 6. Follow-up (ranked)

1. **CROWN4-adaptive port (the KO build)**: keep CROWN4 predictor/groups/RCT verbatim;
   replace per-group static tables with GLOBAL raster-order adaptive states shared
   across groups (predictor-groups stay fine, entropy-states go global — this
   decoupling is the architectural fix that kills table-side without whiplash):
   (a) flat/texture Golomb-groups → C-gol N/A/k with 4-act ctx (raster order);
   (b) Huffman-groups → A-act 16-ctx binary arithmetic on Exp-Golomb prefix +
   raw suffix; per-group backend choice kept (1–2b side) or per-plane best-of.
   Predicted exact ≈3.11–3.16 (transfer range); needs C streaming decode.
2. **Per-plane/per-group best-of-{static, C-gol, A-act}** with 2b choice side
   (probe: −2.94% on MED frame; texture gains most from finer selection).
3. **Suffix-codec refinement**: Exp-Golomb raw suffix is 1b/bin; truncated-binary
   or small-k Golomb suffix for peaked groups may take another −0.2–0.4%.
4. **Do NOT pursue**: transmitted inits (killed), zero-runs on texture (killed),
   unconditioned adaptation (killed), per-group-scattered N/A (b18 kill confirmed).

## 7. Trail (files, method compliance, reproducibility)

New files only: `probe_b19_adaptive.py` (frame, BinEnc/BinDec 16-bit E1/E2/E3,
cabac_encode/decode_plane, GolombState, golomb_adaptive/run bit functions),
`probe_b19_run.py` (driver: baseline, RESET sweep, all-7 exact, per-image CABAC
round-trips, transfer math, `probe_b19_nums.json`), this RESULTS.md. Nothing
existing modified (imports of `probe_b17_rctw` only).
Method rules: numpy+PIL only (no torch), CPU; bpp=total_bits/(H·W·3); adaptive
arithmetic counted from ACTUAL coder bytes + real decode (never entropy
estimates); Golomb/run counted as exact code lengths over mirrored state
sequences with real-stream proof (05-Y) + constructional EOF/L=0 handling;
Huffman via real heapq-equivalent `B17.exact_plane_bits` (tables counted).
Anchors: S0 3.5782 PASS ±3%; RCT round-trip asserted per image; alphabet
assert-loud (no clipping); causal-recon-only conditioning asserted by 21/21
CABAC plane round-trips + Golomb/B-fwd stream proofs. Timings: ~0.1s/image
baseline/Golomb-estimate; ~12s/image CABAC exact+decode (7-image total ≈2.1 min).
Logs: console (this run) + `probe_b19_nums.json`.