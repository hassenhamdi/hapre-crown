# probe_b13 RESULTS — greedy-merge transfer onto the CROWN frame (Boss-5 branch)

Branch B13 (exploratory). Question: b12 banked −1.20% greedy histogram merging on a
MED-only frame (60→18 tables). Does it transfer onto the CROWN frame, where WIDE-K
hill-climbed quantile groups + per-group predictors already leave fewer/fatter tables?
Short answer: **YES but discounted — −0.37% Huffman-exact (7/7), −0.99% with exact
per-cluster rANS upgrade, −1.35% with rANS-proxy-driven merges; best probe 3.2806 avg,
still +1.59% above JXL-e3. Boss-5 STANDS.**

Harness: `probe_b13_merge.py` + `probe_b13_verify.py` (numpy+PIL+ctypes only, CPU,
~70 s + ~81 s). New files only (`probe_b13_*`), nothing existing modified. Proven
modules reused by import: `probe_b4_b` (nbhd/predictors), `probe_b4_c` (loco_ctx365),
`probe_b4_g` (prep/KSET2), `probe_b4_f` (MOE6), `probe_b4_h` (rans_stream_bits, exact C
rANS M=14 bytes + asserted decodes), `dp_refine` (hill-climb), b12 merge algorithm
reimplemented generically over group atoms (cross-channel, image-wide).

UNIT: `bpp = total_bits/(H*W*3)`. Exact counting everywhere (details §2).
Decoder-safety: merged tables + maps transmitted; keys re-derived from causal recon (§5).

## 1. (a) CROWN-huff reproduction — PASS 7/7 (group structure EXACT, totals −0.52% via refine)

Grouping: sign-flipped LOCO-365 keys + auto-K quantile (KSET2) + `dp_refine` hill-climb
+ per-group best-of-MOE6 predictor (3b predid), per-group Huffman. Side per channel:
`active*ceil(log2K)+4b K-id+8b header`; global 64b.

| image | repro (a) | pinned CROWN-huff | Δ | verdict | K (Y,Co,Cg) | nG |
|---|---|---|---|---|---|---|
| kodim01.png | 3.4172 | 3.4340 | −0.0168 (−0.49%) | PASS | (6,36,36) | (6,25,24) |
| kodim02.png | 3.0690 | 3.0931 | −0.0241 (−0.78%) | PASS | (9,36,36) | (9,29,22) |
| kodim05.png | 3.7220 | 3.7394 | −0.0174 (−0.47%) | PASS | (6,36,36) | (6,29,27) |
| kodim07.png | 2.8469 | 2.8630 | −0.0161 (−0.56%) | PASS | (6,36,36) | (6,24,21) |
| kodim13.png | 4.0844 | 4.1010 | −0.0166 (−0.40%) | PASS | (4,18,36) | (4,17,27) |
| kodim19.png | 3.2599 | 3.2758 | −0.0159 (−0.49%) | PASS | (6,36,36) | (6,26,22) |
| kodim23.png | 2.8801 | 2.8940 | −0.0139 (−0.48%) | PASS | (36,36,36) | (30,27,23) |
| **AVG** | **3.3256** | **3.3429** | **−0.0173 (−0.52%)** | **PASS** | — | 58.6 img⁻¹ |

K choices AND effective group counts reproduce the pinned table exactly on all 7
(e.g. 01: Y(6,6) Co(36,25) Cg(36,24)); the systematic −0.5% is the hill-climb refine
gain (pinned CROWN-huff had no refine; CROWN-C hillclimb direction consistent).
(a) split (avg): data 3.2732 + tables 0.0480 (1.44%) + base sides 0.0044 = 3.3256 —
CROWN tables are already ~40% thinner than b12's MED arm (0.076), hence the
discounted transfer below. Predicted b12 §5 range was −0.3…−0.7%: measured −0.37%.

## 2. Main result — (a) vs (b) +merges vs (c) +rANS vs (c2) proxy-driven (exact bits)

Atoms: all per-group histograms image-wide (cross-channel, ng = 48–80, avg 58.6).
Rule: greedy-best-first all-pairs agglomeration, exact total gates every merge.
Cost (b): Huffman data (`heapq` bits) + `16+A*24` tables + merge map
`ng*ceil(log2C)` + 8b C-header, gated per image vs (a) (no-merge fallback, never
triggered — 7/7 MERGED). Cost (c): (b) partition + per-cluster exact H-vs-R choice
(`rans_stream_bits` C bytes, decode-asserted) + 1b choice/cluster, gated vs (b).
Cost (c2, stretch): greedy under rANS-proxy costs (entropy + `16+A*32`, same map),
final clusters exact-encoded with H/R choice + 1b choice, gated vs (c) — won 7/7,
so (c2) is the winning config. JXL-e3 refs are the freshly re-measured values.

| image | (a) | (b) | (c) | (c2) | Δb−a (%) | Δc−a | Δc2−a | JXL-e3 | Δc2−JXL |
|---|---|---|---|---|---|---|---|---|---|
| kodim01.png | 3.4172 | 3.4091 | 3.3856 | 3.3758 | −0.0081 (−0.24%) | −0.0317 | −0.0414 | 3.3593 | +0.0165 |
| kodim02.png | 3.0690 | 3.0607 | 3.0366 | 3.0260 | −0.0083 (−0.27%) | −0.0324 | −0.0430 | 3.0611 | −0.0351 WIN |
| kodim05.png | 3.7220 | 3.7065 | 3.6903 | 3.6811 | −0.0156 (−0.42%) | −0.0317 | −0.0409 | 3.5120 | +0.1691 |
| kodim07.png | 2.8469 | 2.8398 | 2.8130 | 2.7825 | −0.0071 (−0.25%) | −0.0339 | −0.0644 | 2.7310 | +0.0515 |
| kodim13.png | 4.0844 | 4.0707 | 4.0567 | 4.0550 | −0.0136 (−0.33%) | −0.0277 | −0.0294 | 3.9141 | +0.1409 |
| kodim19.png | 3.2599 | 3.2490 | 3.2295 | 3.2238 | −0.0109 (−0.33%) | −0.0304 | −0.0361 | 3.2154 | +0.0084 |
| kodim23.png | 2.8801 | 2.8576 | 2.8362 | 2.8202 | −0.0224 (−0.78%) | −0.0439 | −0.0599 | 2.8110 | +0.0092 |
| **AVG** | **3.3256** | **3.3134** | **3.2926** | **3.2806** | **−0.0123 (−0.37%)** | **−0.0330 (−0.99%)** | **−0.0450 (−1.35%)** | **3.2291** | **+0.0515 (+1.59%)** |

(b) beats (a) 7/7; (c) beats (b) 7/7 (−0.0208 avg); (c2) beats (c) 7/7 (−0.0120 avg).
(c2) beats JXL-e3 on 1/7 (kodim02 only).

Where the bits go, (b) vs (a) (avg, exact):

| arm | data | tables | merge map (+8b C) |
|---|---|---|---|
| (a) unmerged | 3.2732 | 0.0480 (1.44%) | — |
| (b) merged | 3.2807 (+0.0075) | 0.0280 (0.84%) | 0.0003 |
| ledger | extra data **+0.0075** | saved tables **−0.0200** | map **+0.0003** → net **−0.0123** (exact; components 4-dec rounded) |

MDL return ≈ 2.6:1 (b12 MED arm was 5:1 — CROWN groups are fatter and less
redundant, as predicted). Map side negligible again (0.0003 bpp).

## 3. Merge statistics

(b) Huffman partition: tables/image 58.6→24.6 (410→172 total, 238 merges, 7/7 images
merged); avg net gain **436 bits (~54 B)/merge**. Per image (ng→C, merges):
01: 55→22 (33); 02: 60→31 (29); 05: 62→23 (39); 07: 51→26 (25); 13: 48→19 (29);
19: 54→23 (31); 23: 80→28 (52).
(c) backend on (b) partition: R wins 156/172 clusters (91%): 22/22, 29/31, 19/23,
21/26, 18/19, 21/23, 26/28 — merged tables amortize rANS's pricier headers, so rANS
dominates post-merge (vs ~1/3 R share pre-merge in probe_b5's unmerged groups).
(c2) proxy partition merges LESS (221 total: C = 30,33,24,27,18,28,29) yet wins
after exact H/R coding (179/189 clusters R, 95%; kodim01 all-R) — the Huffman-greedy
path over-merges pairs whose entropy gap exceeds their Huffman gap. Lesson: drive
merges with the backend that will encode them.

## 4. Boss-5 verdict — STANDS

Best probe (c2) 3.2806 vs JXL-e3 3.2291 = **+0.0515 (+1.59%), 1W-6L — NO KO**
(majority-wins fails; avg fails). (b) 1W-6L (kodim02 by 0.0004), (c) 1W-6L.
Context: (c2) 3.2806 also trails CROWN2-exact 3.2686 by +0.37% — but that comparison
is NOT apples-to-apples (this harness is E6/KSET2/H-or-R; CROWN2 adds E16 + WIDE-K +
Golomb + GRID, worth ≈−1.7% combined). The transfer onto full CROWN2 is therefore
UNMEASURED here and bounded above by ≈−0.4% Huffman-part (its tables are already
coarser, several Golomb groups carry no table at all) plus a backend-priced kicker.
Stack status vs e3 gap (−1.22% needed from 3.2686): merges ≈ 1/3 of the gap; e9
(3.03) still needs the transform-bitplane arc regardless.

## 5. Decoder-safety statement

Transmitted for the winning (c2) config: global 64b; per-channel Q maps
(`active*ceil(log2K)` + 4b K-id + 8b header) + per-group 3b predictor ids (both kept
from (a)); merge map `ng*ceil(log2C)` + 8b C header + 1b H/R choice per merged
cluster; winning tables only (`16+A*24` Huffman / `16+A*32` + 64b framing rANS).
Decode: read C → merge map → tables; per pixel, the LOCO-365 key/sign is re-derived
from already-decoded causal recon (keys depend only on recon, never on either map —
no circularity), key→per-channel-group→merge-map→table→symbol→predictor-add→recon.
Checks: every winner partition covers all atoms exactly once (asserted 7/7, both
partitions); Kraft=1 on every Huffman-kept cluster (asserted); canonical symbol
round-trip on every Huffman-kept (c2) cluster ALL-7 (`probe_b13_verify.py`);
`rans_decode` asserted on EVERY R cluster (inside `rans_stream_bits`, 179 clusters);
literal bitstream pack→unpack→decode demonstrated on kodim01 cluster-0 data
(61788 syms, 208123 bits); YCoCg-R inverse asserted all 7. Decode uses merged tables
exclusively — no unmerged state retained. (c2) exact totals reproduce to 4 decimals
in the independent verify run (7/7 PASS).

## 6. Trail + follow-up

Trail: (a) grouping reproduces pinned K/nG exactly with refine explaining −0.52%
→ Huffman greedy wins 7/7 but discounted −0.37% (inside b12's predicted −0.3…−0.7%)
→ per-cluster rANS upgrade wins 7/7 (−0.62% further, 91% R) → proxy-driven partition
wins 7/7 again (−0.36% further). No dead-ends; one labeling note: merge.py's console
"tables before→after" counts groups (58.6), consistent with §1 nG sums.

Follow-up (ranked):
1. Port the merge pass onto the FULL CROWN2 arm (post-3-way E16/WIDE/Golomb/GRID
   histograms, proxy-driven search + exact H/R/G coding of final clusters, hybrid
   JS-shortlist → exact-greedy for encoder speed), measure the discounted transfer
   exactly — the only number that can stack toward Boss-5.
2. Test merge × DP-refine interaction (refine-then-merge vs merge-aware cuts).
3. Consolidated C port only after (1) banks; transform-bitplane arc for e9.

Files: `experiments/probe_b13_merge.py` (arms a/b/c/c2 + (b) round-trips),
`experiments/probe_b13_verify.py` ((c2) independent round-trip + repro),
`experiments/probe_b13_summary.txt`, this report.
