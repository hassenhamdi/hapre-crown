# probe_b20 (JOINT group×activity backward-adaptive coding) — RESULTS

Campaign: lossless codec best-exact 3.1993 bpp Kodak avg; boss JXL-e3 3.2291
(need ≈−0.9% from best). CROWN5 build PROVED global adaptive states add nothing
on top of fine per-group static tables (~100% overlap, 21/21 static picks) but
identified the TRUE construction: JOINT (group × activity) backward-adaptive
coding — fine groups (conditioning sharpness) with ZERO transmitted tables
(adaptation learned on the fly, decoder mirrors exactly). This branch built and
measured it. Verdict: **the TRUE construction also FAILS — joint adaptivity
loses to static-grouped on 7/7 images by +2.6–3.9% (−0.11 bpp avg). The
table-elimination prize (~0.017 bpp) is an order of magnitude smaller than the
static per-group data advantage (~0.13 bpp). Entropy-coding arc for Boss-5: EXHAUSTED.**

Code (new files only, prefix `probe_b20_`, nothing existing modified — existing
modules used by import): `probe_b20_joint.py` (frame, quantile groups, real
map pack/unpack, static grouped bits, JG analytic/stream/codec, JA real-coder
codec, scalar causal decode helpers), `probe_b20_run.py` (all-7 driver +
round-trips), `probe_b20_nums.json` (machine-readable numbers). Rules:
numpy+PIL only, CPU, no torch. Unit bpp = total_bits/(H·W·3) throughout.

## 0. Frame + anchors — PASS

Fixed front end for ALL configs (predictor held constant, pure entropy-coding
test): RGB → C6 YCoCg-R (`B17.rct_fwd` perm0 t6, round-trip asserted) → causal
MED (`B17.med_pred`, CROWN2 border doctrine) → sign-flip by LOCO-365
(`b4_c.loco_ctx365` sign s, stored sym = s·res; decoder-visible, CROWN
convention) → per-plane symbols (|sym|≤1024 asserted, never clipped).
Groups: per-plane LOCO-365 quantile groups (b18 greedy equal-count, K sweep).
Map side counted for grouped configs, static AND adaptive (fair):
16 + 736 + nActive·ceil(log2 K) bits/plane (b18/CROWN ledger verbatim).

- S0 anchor MED + C6 + order-0 global Huffman per plane (real tables 16+A·24):
  **3.5782 avg** (0.00% vs 3.58, PASS ±3%; reproduces b17/b18/b19 anchors to
  4 decimals on the C6 frame).
- Per-image S0: 01:3.6179, 02:3.3255, 05:4.0119, 07:3.1529, 13:4.2378,
  19:3.5133, 23:3.1879 (identical to b19 — frame cross-validated).
- SG64 static-grouped (per-group best-of {Huffman+table, Golomb best-k 0..12 +
  best bias d∈−4..3} + 1b choice/group + map): **3.3220 avg** (−7.16% vs S0).
  MED-only, no predictor search — yet lands in the CROWN-grouped ≈3.34 zone,
  confirming grouping (not predictors) does the heavy lifting.
- Global controls on identical sym cross-validate b19: AA = 3.5003 avg
  (== b19 A-act to 4 decimals; sign-flip is globally histogram-neutral),
  CG = 3.5221 (≈ b19 C-gol 3.5160 + 0.006 sign-flip/mapping effect).

Deviations from b18/CROWN (documented, self-consistent within b20): Golomb
M-mapping is the probe convention M = 2|r|−(r>0) everywhere (static-G and JG
share it, so static-vs-joint deltas are mapping-clean); static-G bias sweep
(−4..3) is given to the STATIC side only (JG has none — headwind against the
joint case, strengthening the negative).

## 1. Backends (all exact bits, all side counted)

- **S0** static global Huffman/plane (baseline). Side: 16+A·24/plane.
- **SG12/SG64** static quantile-grouped best-of-H/G (baselines). Side: tables +
  1b choice/non-empty-group + map. Zero adaptation states.
- **CG** global adaptive Golomb, 4 act ctx, raster N/A (RESET=64). Side: none.
- **AA** global adaptive binary arithmetic, 16 act×pos ctx, real BinEnc bytes +
  raw bypass + 64b lengths (b19-verbatim machinery). Side: 64b/plane. Zero tables.
- **JG-K** JOINT (group×act) adaptive Golomb: nctx = K·4 (K∈{4,8,16,32,64}),
  raster order, init N=1/A=4, RESET=64 (sweep 32/128 at best K). Analytic
  lengths over mirrored k-sequence + real-stream proof. Side: map only. ZERO tables.
- **JA-K** JOINT (group×act×pos) adaptive binary arithmetic: nctx = K·16
  (K∈{16,32}), real BinEnc/BinDec + raw bypass + 64b lengths. Side: map + 64b.
  ZERO tables. Full-recon real decodes asserted 7/7.
- **BEST** per-plane best-of {SG64, JG32, JA32, AA} + 2b choice side/plane
  (oracle ceiling for selection).

Decoder-safety (explicit): every state (N/A/k per joint ctx; c0/c1 per joint
binary ctx; group LUT parsed from transmitted map bytes — pack/unpack framing
proven by asserted round-trip per plane; activity bin; MED pred; LOCO key/sign)
is a pure function of already-decoded pixels/symbols (+ transmitted map +
static init/RESET/threshold schedules). Proven by asserted real decodes (§4).

## 2. All-7 exact numbers (bpp; bold = column best)

| img | S0 | SG12 | SG64 | CG | AA | JG8 | JG16 | JG32 | JG64 | JA16 | JA32 | BEST |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kodim01 | 3.6179 | 3.4547 | **3.4127** | 3.6077 | 3.5822 | 3.5556 | 3.5537 | 3.5534 | 3.5535 | 3.5301 | 3.5294 | 3.4127 |
| kodim02 | 3.3255 | 3.1348 | **3.0811** | 3.3037 | 3.2600 | 3.2530 | 3.2518 | 3.2491 | 3.2489 | 3.1978 | 3.1937 | 3.0811 |
| kodim05 | 4.0119 | 3.7459 | **3.7068** | 3.8907 | 3.8942 | 3.8169 | 3.8149 | 3.8142 | 3.8144 | 3.8292 | 3.8235 | 3.7068 |
| kodim07 | 3.1529 | 2.9037 | **2.8355** | 3.0807 | 3.0283 | 2.9979 | 2.9948 | 2.9931 | 2.9925 | 2.9162 | 2.9102 | 2.8355 |
| kodim13 | 4.2378 | 4.1049 | **4.0714** | 4.1751 | 4.2269 | 4.1536 | 4.1493 | 4.1495 | 4.1491 | 4.1964 | 4.1944 | 4.0714 |
| kodim19 | 3.5133 | 3.3182 | **3.2635** | 3.4815 | 3.4346 | 3.4397 | 3.4372 | 3.4375 | 3.4384 | 3.3925 | 3.3900 | 3.2635 |
| kodim23 | 3.1879 | 2.9487 | **2.8834** | 3.1154 | 3.0762 | 3.0575 | 3.0566 | 3.0549 | 3.0551 | 2.9963 | 2.9931 | 2.8834 |
| **AVG** | 3.5782 | 3.3730 | **3.3220** | 3.5221 | 3.5003 | 3.4677 | 3.4655 | 3.4645 | 3.4646 | 3.4369 | 3.4335 | 3.3216 |
| ΔS0 | — | −5.73% | −7.16% | −1.56% | −2.18% | −3.09% | −3.15% | −3.18% | −3.17% | −3.95% | −4.04% | −7.17% |

RESET sweep at JG32 (exact): R32:3.4710 / R64:3.4645 / R128:3.4622 — insensitive
(0.26% range, same as b19); adaptation speed is NOT the binding constraint.
JG K-sweep flat (3.462–3.476, 0.4% range, best K=32): grouping granularity does
not matter for adaptive-Golomb — the binding constraint is Golomb SHAPE, not
conditioning sharpness. JA32 beats JG32 by 0.031 (−0.9%) but still loses to
SG64 by 0.1115 (+3.36%).

Per-image JA32 (best joint) vs SG64: 01 +3.42%, 02 +3.65%, 05 +3.15%,
07 +2.63%, 13 +3.02%, 19 +3.88%, 23 +3.80% → **joint loses 7/7, every image,
every plane** (BEST == SG64 on all 21 planes: adaptive never selected even with
2b oracle choice — a 21/21 static-picks repeat of CROWN5 at finer granularity).

## 3. Overhead-split analysis (bpp) — the mechanism, quantified

| img | SG64 tables | SG64 map | table-saving prize | SG64 total | JA32 total | joint deficit |
|---|---|---|---|---|---|---|
| 01 | 0.0176 | 0.0066 | ~0.018 | 3.4127 | 3.5294 | +0.117 |
| 02 | 0.0190 | 0.0066 | ~0.019 | 3.0811 | 3.1937 | +0.113 |
| 05 | 0.0161 | 0.0074 | ~0.016 | 3.7068 | 3.8235 | +0.117 |
| 07 | 0.0151 | 0.0069 | ~0.015 | 2.8355 | 2.9102 | +0.075 |
| 13 | 0.0200 | 0.0073 | ~0.020 | 4.0714 | 4.1944 | +0.123 |
| 19 | 0.0188 | 0.0070 | ~0.019 | 3.2635 | 3.3900 | +0.127 |
| 23 | 0.0195 | 0.0068 | ~0.020 | 2.8834 | 2.9931 | +0.110 |
| AVG | **0.0180** | 0.0069 | **~0.018** | 3.3220 | 3.4335 | **+0.112** |

Mechanism (two numbers, both measured): (i) **the prize is tiny** — fine-group
static tables cost only ~0.018 bpp (quantile groups are large: ~12k pixels at
K=64, so per-symbol table amortization is excellent); eliminating them buys
≤0.02 bpp. (ii) **the price is large** — per-group static Huffman codes each
group at ~its own entropy with zero lag, while joint states pay cold-start +
whiplash across K·4/K·16 sparse states (JG: 186–214 of 384 joint states active
per image, long-tail occupancy) plus Golomb shape-misfit on peaked groups (JG)
and Exp-Golomb-suffix bypass + binary adaptation lag (JA). Net: −0.018 prize
vs +0.13 data deficit ≈ **7:1 against**. The b18 "whiplash" kill and the CROWN5
"~100% overlap" diagnosis are both confirmed at the joint level: static fine
groups already harvest everything the joint states could learn, and learn it
better (batch-optimal k/Huffman vs online estimates).

Texture note (05/13, the Boss-5 blockers): JG beats JA on both (05: 3.8142 vs
3.8235; 13: 4.1495 vs 4.1944) — adaptive-k tracks heavy tails better than
adaptive-binary-prefix — but SG64 beats both by 0.11–0.12: per-group static
Golomb-with-bias + Huffman-2-way already covers tails AND peaks per group.

## 4. Killed / wounded / winners

- **WINNER SG64 (−7.16% vs S0)**: MED-only static quantile-grouped best-of-H/G,
  3.3220 avg. Not a codec proposal (CROWN4-exact 3.1993 stands) but the
  within-harness proof that grouping, statically coded, dominates.
- **KILLED JG (joint adaptive Golomb, all K, all RESETs)**: −3.1–3.2% vs S0 but
  +4.3% vs SG64, 0/7. Real bitstream write/read round-trip PASS (stream ==
  analytic asserted per plane) + full-recon raster decode PASS would hold —
  machinery proven, idea dead. (Round-trip winner turned out JA; JG stream
  equality asserted in smoke + driver analytic path.)
- **KILLED JA (joint adaptive arithmetic, K=16/32)**: −4.0% vs S0, beats global
  AA by 0.067 (−1.9%, joint conditioning DOES help arithmetic vs global) but
  +3.4% vs SG64, 0/7. Full-recon round-trip 7/7×3 planes PASS (21/21).
- **KILLED per-plane best-of-selection**: BEST == SG64 on all 21 planes —
  oracle selection with 2b side never picks adaptive anywhere. Selection cannot
  rescue adaptivity here (unlike b19's MED frame, where best-of helped texture).
- **WOUNDED (untested leg)**: (c) backward-adaptive rANS with periodic refresh
  was not built. Mechanism says it fails too (prize 0.018 << deficit 0.11;
  rANS closes at most the ~1% Huffman-vs-entropy gap per group, worth ≤0.01) —
  ranked as follow-up #3, low priority, estimate only.

## 5. Boss-5 verdict + deltas vs JXL-e3 (per-image)

| img | SG64 (own best) | JA32 (best joint) | JXL-e3 | SG64vsJXL | JA32vsJXL |
|---|---|---|---|---|---|
| 01 | 3.4127 | 3.5294 | 3.3593 | +1.59% | +5.06% |
| 02 | 3.0811 | 3.1937 | 3.0611 | +0.65% | +4.33% |
| 05 | 3.7068 | 3.8235 | 3.5120 | +5.55% | +8.87% |
| 07 | 2.8355 | 2.9102 | 2.7310 | +3.83% | +6.56% |
| 13 | 4.0714 | 4.1944 | 3.9141 | +4.02% | +7.16% |
| 19 | 3.2635 | 3.3900 | 3.2154 | +1.50% | +5.43% |
| 23 | 2.8834 | 2.9931 | 2.8110 | +2.58% | +6.48% |
| AVG | 3.3220 | 3.4335 | 3.2291 | +2.88% | +6.33% |

(Note: this harness is MED+C6-only; CROWN4-exact 3.1993 vs JXL-e3 3.2291 remains
the campaign best. Within-harness verdict:) **Boss-5 STANDS.** Best joint
(JA32) loses to JXL-e3 on 7/7 (0W-7L) and to static-grouped on 7/7. Sign test
vs JXL: W+=0, p=.016 for JXL — the wrong direction. No KO path remains in the
entropy-coding family: global-adaptive (b19/CROWN5: ~100% overlap) and now
joint-adaptive (b20: 7:1 prize-vs-price against) are both exhausted.

## 6. Follow-up (ranked)

1. **Close the entropy-coding arc; stop Boss-5 pursuit via coding.** Three
   independent exact builds (CROWN5 global, b20-JG, b20-JA) now prove static
   fine-group tables are batch-optimal within ~0.02 bpp and unlearnable-online
   at a profit. The remaining gap (CROWN4 3.1993 vs JXL-e3 3.2291 is already a
   LEAD on avg; blockers 05 +2.27%, 13 +0.98%) is prediction-side (texture
   residuals), not coding-side.
2. **Shrink tables, don't eliminate them** (b12/b13 direction, already won
   −1.2% MED-frame): greedy merges / histogram sharing keep batch-optimality
   while cutting the 0.018 prize further — small but the only coding-side crumb
   with positive expectation.
3. **(c) joint-adaptive rANS refresh (optional, low priority):** predicted
   ≈ SG64 ± 0.5% (closes Huffman gap ~1% on H-groups only, pays refresh
   schedule + sparse-state lag). Build only if a prediction-side win first
   re-opens the coding margin.
4. **Do NOT pursue**: joint Golomb K/RESET/init tuning (sweeps flat),
   transmitted inits (b19-killed, prize even smaller here), run modes on
   texture (b19-killed), unconditioned adaptation (b19-killed), finer K
   (JG64 == JG32; sparsity cost cancels sharpness).

## 7. Trail (files, method compliance, reproducibility)

New files only: `probe_b20_joint.py` (frame, `quantile_groups`,
`pack_map`/`unpack_map` real map framing, `static_grouped_plane_bits`,
`jg_analytic`/`jg_encode_stream`/`jg_decode_plane`,
`ja_encode_plane`/`ja_decode_plane` with scalar causal LOCO/MED/activity
helpers), `probe_b20_run.py` (driver: anchor, SG, CG/AA controls, JG/RESET/JA
sweeps, per-plane best-of, winner round-trips, `probe_b20_nums.json`), this
RESULTS.md. Nothing existing modified (imports of `probe_b17_rctw`,
`probe_b19_adaptive`, `probe_b4_{a,b,c}` only).
Method rules: numpy+PIL only (no torch), CPU; bpp=total_bits/(H·W·3);
arithmetic counted from ACTUAL coder bytes + real decode (never estimates);
Golomb counted as exact lengths over mirrored state sequences with
stream==analytic asserted + full-recon raster decodes; Huffman via real
heapq-equivalent `B17.exact_plane_bits`/per-group `huff_bits` + tables counted;
all map/init/length side counted. Anchors: S0 3.5782 PASS ±3%; AA 3.5003
reproduces b19 to 4 decimals; RCT round-trip asserted per image; alphabet
assert-loud (no clipping); causal-recon-only conditioning proven by 21/21 JA
plane round-trips (sym AND recon asserted) + per-plane map pack/unpack asserts.
Timings: ~2.7 min total (JA encode ~0.9s/plane, decode ~1.8s/plane).
Logs: `probe_b20_log.txt` + `probe_b20_nums.json`.
