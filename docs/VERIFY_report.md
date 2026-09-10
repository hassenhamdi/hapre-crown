# VERIFY Report — HAPRE-CROWN-paper-v10.md load-bearing numbers audit

Date: 2026-09-08. Method: read-only. Recomputation: mean-of-row, pooled-bytes
(total_bytes·8/(7·1179648)), `scipy.stats.wilcoxon` paired two-sided (n=7).
Draft ref: `/tmp/opencode/autocompress/docs/HAPRE-CROWN-paper-v10.md` (v10).

Statuses: CONFIRMED | CORRECTED (draft value wrong) | SECONDARY-ONLY (value matches
a banked secondary source, no primary probe file with per-image breakdown) |
UNBANKED (no source file found; draft honestly marks "—"/todo) |
NOTE (confirmed with precision/denominator caveat).

## 1. Boss-ladder per-image bpp (draft Table §4.3, lines 168–175)

### 1a. CROWN6-exact per-image — source `experiments/crown6_results.json`, `bpp` dict lines 323–331, `avg_bpp` line 332

| claim (draft §4.3) | source | verified value | status |
|---|---|---|---|
| kodim01 3.2982 | crown6_results.json:4,324 | 3.2982 (bytes 486346 → 3.2982449) | CONFIRMED |
| kodim02 2.9854 | crown6_results.json:53,325 | 2.9854 (bytes 440219 → 2.9854262) | CONFIRMED |
| kodim05 3.5806 | crown6_results.json:103,326 | 3.5806 (bytes 527981 → 3.5806003) | CONFIRMED |
| kodim07 2.6798 | crown6_results.json:144,327 | 2.6798 (bytes 395158 → 2.6798367) | CONFIRMED |
| kodim13 3.9532 | crown6_results.json:190,328 | 3.9532 (bytes 582929 → 3.9532403) | CONFIRMED |
| kodim19 3.1510 | crown6_results.json:235,329 | 3.1510 (bytes 464631 → 3.1509806) | CONFIRMED |
| kodim23 2.7365 | crown6_results.json:277,330 | 2.7365 (bytes 403508 → 2.7364638) | CONFIRMED |
| AVG 3.1978 | crown6_results.json:332 | row-mean 3.1978143 → 3.1978; pooled-bytes 3.1978275 → 3.1978 | CONFIRMED |

NOTE on audit brief: the brief's parenthetical "(01:3.2982?, 02:2.9854?, 05:3.5917?,
07:2.6790?, 13:3.9525?, 19:3.1510?, 23:2.7365?)" mixes two rows — 05:3.5917 /
07:2.6790 / 13:3.9525 are the **CROWN4** values (see §2), not CROWN6. The draft
table itself keeps them separate and correct: CROWN6 = 3.5806/2.6798/3.9532,
CROWN4 = 3.5917/2.6790/3.9525. No draft error.
RCT/fam/MLP note (draft line 177): RCT C27 on 01/05/07/19/23 + C12 on 02/13 —
crown6_results.json lines 6,55,105,146,193,237,279 CONFIRMED; fams all-Q lines
7–11 etc. CONFIRMED; MLP active on 01/05/13 only lines 12–16,111–115,198–202
CONFIRMED. CROWN6-vs-JXL-e3 Δ% (draft line 177: −1.82/−2.47/+1.95/−1.87/+1.00/
−2.00/−2.65) recomputed from json lines 334–342: −1.819/−2.473/+1.953/−1.875/
+0.999/−2.003/−2.650 — all CONFIRMED (07 rounds −1.8749→−1.87).

### 1b. JXL-e3 per-image — source `survey/bosstakedown/data/ladder.csv` lines 2–8, `BOSS_TAKEDOWN.md` lines 13–20

| claim | source | verified value | status |
|---|---|---|---|
| kodim01 3.3593 | ladder.csv:2; BOSS_TAKEDOWN.md:13 | 3.3593 | CONFIRMED |
| kodim02 3.0611 | ladder.csv:3 | 3.0611 | CONFIRMED |
| kodim05 3.5120 | ladder.csv:4 | 3.5120 | CONFIRMED |
| kodim07 2.7310 | ladder.csv:5 | 2.7310 | CONFIRMED |
| kodim13 3.9141 | ladder.csv:6 | 3.9141 | CONFIRMED |
| kodim19 3.2154 | ladder.csv:7 | 3.2154 | CONFIRMED |
| kodim23 2.8110 | ladder.csv:8 | 2.8110 | CONFIRMED |
| AVG 3.2291 | ladder.csv (row-mean 3.2291286; pooled-bytes 3.2291347) | 3.2291 | CONFIRMED |

### 1c. WebP-m3 per-image (PIL) — source `experiments/probe_b4_RESULTS.md` lines 61–62

| claim | source | verified value | status |
|---|---|---|---|
| kodim01 3.4074 | probe_b4_RESULTS.md:61 | 3.4074 | CONFIRMED |
| kodim02 3.0659 | probe_b4_RESULTS.md:61 | 3.0659 | CONFIRMED |
| kodim05 3.8563 | probe_b4_RESULTS.md:61 | 3.8563 | CONFIRMED |
| kodim07 2.8397 | probe_b4_RESULTS.md:62 | 2.8397 | CONFIRMED |
| kodim13 4.0990 | probe_b4_RESULTS.md:62 | 4.0990 | CONFIRMED |
| kodim19 3.2730 | probe_b4_RESULTS.md:62 | 3.2730 | CONFIRMED |
| kodim23 2.8754 | probe_b4_RESULTS.md:62 | 2.8754 | CONFIRMED |
| AVG 3.3452 PIL (/ 3.347 campaign pin) | row-mean 3.3452429 → 3.3452; pin `RESULTS.md:18` | 3.3452 PIL; 3.347 pin | CONFIRMED (both toolchains labeled, draft line 175) |

### 1d. WebP-m6 per-image — source `experiments/probe_b5_RESULTS.md` m6-ref columns lines 35–42, 52–59

| claim | source | verified value | status |
|---|---|---|---|
| kodim01 3.39 | probe_b5_RESULTS.md:35,52 | 3.39 (2dp as banked) | CONFIRMED (NOTE: 2dp precision only) |
| kodim02 3.06 | probe_b5_RESULTS.md:36,53 | 3.06 | CONFIRMED (2dp) |
| kodim05 3.77 | probe_b5_RESULTS.md:37,54 | 3.77 | CONFIRMED (2dp) |
| kodim07 2.81 | probe_b5_RESULTS.md:38,55 | 2.81 | CONFIRMED (2dp) |
| kodim13 4.05 | probe_b5_RESULTS.md:39,56 | 4.05 | CONFIRMED (2dp) |
| kodim19 3.27 | probe_b5_RESULTS.md:40,57 | 3.27 | CONFIRMED (2dp) |
| kodim23 2.86 | probe_b5_RESULTS.md:41,58 | 2.86 | CONFIRMED (2dp) |
| AVG 3.3157 | row-mean of 2dp row 3.3157143 → 3.3157 | 3.3157 | CONFIRMED (NOTE: exact PIL-m6 avg cited as 3.3160 in probe_b4_RESULTS.md:66 — 0.0003 mean-of-rounded artifact; draft is transparent, not wrong) |

### 1e. PNG-9 / WebP-m0 per-image — no banked source

| claim | source | verified value | status |
|---|---|---|---|
| PNG-9 per-image "—" (avg 4.75 only) | none found (avg: `RESULTS.md:15`, v01 draft:59) | — | UNBANKED (draft honestly marks "—", line 310 todos perimage.py fill) |
| WebP-m0 per-image "—" (avg 3.60 only) | none found (avg: `RESULTS.md:17`) | — | UNBANKED (same honest marking) |

## 2. Champion averages — recomputed as mean of draft per-image row

| claim | source | verified value | status |
|---|---|---|---|
| CROWN6 3.1978 | crown6_results.json:332 (row-mean 3.1978143) | 3.1978 | CONFIRMED |
| CROWN4 3.1993 | probe_b19_RESULTS.md:58–65 (row 3.2991/2.9854/3.5917/2.6790/3.9525/3.1510/2.7364; mean exactly 3.1993) | 3.1993 | CONFIRMED (NOTE: `probe_b18_final7.json` holds 3.2992/2.9855/3.5918 — 0.0001 higher on 3 imgs, mean 3.199357→3.1994; draft follows the probe_b19 table; inter-artifact rounding only) |
| CROWN2 3.2686 | probe_b17_RESULTS.md:20–21,41 (row 3.3789/3.0274/3.6594/2.7556/4.0245/3.2207/2.8138; mean 3.2686143) | 3.2686 | CONFIRMED |
| CROWN-rans-hc 3.309 | row mean is 3.3124 (probe_b4_RESULTS.md:48–50; probe_b5_RESULTS.md:22); 3.309 = hillclimb champ (probe_b5_RESULTS.md:22 parenthetical; `RESULTS.md:6`; memory/cycle1-memory.md:79) | row 3.3124; HC 3.309 | CONFIRMED-AS-LABELED (draft AVG row lists both "3.3124 / 3.309 HC", line 175 — 3.309 is NOT the row mean; no per-image row banked for the HC variant) |
| MOE 3.464 | probe_b4_RESULTS.md:15–22 (avg 3.4643; 2dp row mean 3.4642857); `RESULTS.md:23`; v01 draft:58 | 3.464 (3dp of 3.4643) | CONFIRMED |
| RUN 3.511 | `RESULTS.md:9` avg only; no per-image row in draft or probes | 3.511 | UNBANKED-AS-ROW (cannot recompute as row mean; consistent w/ campaign report. CAUTION: probe_b10 Q-T4 VQ avg 3.5111, probe_b10_RESULTS.md:71, is a different experiment with a coincidentally equal average — do not cite it as the RUN row) |

## 3. Wilcoxon claims — `scipy.stats.wilcoxon` paired two-sided, n=7, recomputed from per-image pairs above

| claim (draft) | source | verified value | status |
|---|---|---|---|
| Boss-2 KO: CROWN-rans-hc vs WebP-m3 7/7 p=.016 (Table §4.4 line 186; §4.2 line 151) | pairs probe_b4:48–50 vs probe_b4:61–62 | diffs all neg; W=0, p=0.015625 → .016, 7/7 | CONFIRMED |
| E16+2-way (3.2979) vs m6 6/1 p=.16 TIE (Table line 189) | pairs probe_b5:35–41 vs §1d m6 row | 6W-1L; W=5, p=0.15625 → .16 | CONFIRMED |
| E16+3-way (3.2720) vs m6 7/7 p=.016 KO (Table line 190) | pairs probe_b5:52–58 vs §1d | W=0, p=0.015625 → .016, 7/7 | CONFIRMED |
| Boss-3 CROWN2 vs WebP-m6 7/7 p=.016 (Table line 188) | pairs §2-CROWN2 vs §1d m6-2dp | W=0, p=0.015625 → .016, 7/7 (min diff −0.0111, robust to 2dp rounding) | CONFIRMED |
| CROWN-huff vs PIL-m3 1W-6L p≈.47 TIE (line 160, Table line 187) | pairs probe_b4:39–45 vs probe_b4:61–62 | 1W-6L; W=7, p=0.296875 → ≈.30 | **CORRECTED: p≈.30 (W=7), not ≈.47.** Verdict TIE (n.s.) unchanged. (Also: huff avg 3.3429 beats PIL 3.3452 by −0.0023; draft writes "+0.0023" — sign/direction ambiguous, magnitude right.) |
| Boss-5 CROWN6 vs JXL-e3 5W-2L p=.219 STANDS (Table line 193) | pairs crown6_results.json:323–331 vs :343–351 | 5W-2L; W=6, p=0.21875 → .219 (matches json line 353) | CONFIRMED |
| CROWN3 led JXL-e3 5W-2L p=.375 (line 196; −0.78%) | CROWN4-row pairs (CROWN3 per-image row not tabled): 5W-2L; W=8, p=0.375; (3.2291−3.2040)/3.2291=0.777% | 5W-2L p=.375 pattern CONFIRMED on CROWN4 row; CROWN3-row recompute not possible (row unbanked) | CONFIRMED-WITH-CAVEAT (statistic matches the shared 05/13-loss pattern; CROWN3 per-image row itself not in evidence) |
| FULL-classical vs JXL-e3 3W-4L W=12 n.s. (Table line 194) | pairs probe_b14_summary.json FULL vs JXL_E3 | diffs +0.0009/−0.0570/+0.1363/+0.0024/+0.1022/−0.0118/−0.0164 → 3W-4L; W=12, p=0.8125 n.s. | CONFIRMED (W=12 exact) |
| (context) MOE-2dp vs JXL-e1 7/7 p=.016 (Table lines 183–184) | pairs §2-MOE vs ladder.csv e1 | all neg; W=0, p=0.015625 | CONFIRMED |
| Brief item "Boss-3 E16 KO (6W-1L p≈.047)" | no such claim in draft (the 6/1 arm is tabled TIE p=.16) | correct value for the 6/1 arm is p=.16 (W=5) | NOT-FOUND — likely brief typo for either the .16 arm or the huff .47→.30 arm |

## 4. Dead-end delta spot-checks (5 picks)

| claim (draft §5) | source | verified value | status |
|---|---|---|---|
| byte-LZ 4.18 (P) (line 245) | `RESULTS.md:33`; memory/cycle1-memory.md:36,40; v01 draft:70 | 4.18 as transcribed | SECONDARY-ONLY (no probe RESULTS file with per-image breakdown found; standalone-LZ 4KB/36ms/786KB PASS detail only in memory:36 — same secondary level) |
| bitplane-B +26% (P) (line 243) | probe_b15_RESULTS.md:36–43 (B8 avg 4.2595, B16 4.2679, A1 3.4111, C 3.3737); mechanism line 79 | B8-vs-A1 +24.87%, B16-vs-A1 +25.12%; B8-vs-C +26.26%, B16-vs-C +26.51% | CONFIRMED-WITH-NOTE ("+26%" matches B-vs-best-joint-C; probe line 79's own "B vs A1" denominator label is ~1pp high — recomputed vs A1 is +24.9/+25.1%. Draft inherits probe rounding; magnitude/direction right, family-killing conclusion unaffected) |
| hash-cache +2.85% (P) (line 251); hit 25.0% | probe_b7_RESULTS.md:47–57 (avg 3.5769→3.6787; mean hit 25.01%; flag 1.000 b/px; index 5.926 b/hit); verdict lines 87–93 | (3.6787−3.5769)/3.5769=+2.846% → +2.85% | CONFIRMED (all four sub-figures match) |
| LF-NN Squeeze +13.48% (P) (line 241) | probe_lfup_RESULTS.md:41 (avg 3.5769→4.0591) and probe_lfpred_RESULTS.md:15 (10.7306→12.1772) | +13.4809% and +13.4811% → +13.48% | CONFIRMED (both probes coincide at +13.48%; attribution "Squeeze-NN LF-prediction" matches lfpred; lfup upsampling variants +10.77/+10.54 per lfup:41 also consistent w/ draft) |
| bias-Huffman theorem (line 249): LOCO-I-lite −0.2%; M-GLOB 0/7, M-GROUP 0/126; Δ=0 proof; G-bias −0.36% on kodim13 | M-theorem: probe_b8_RESULTS.md:43–55 (M-GLOB 3.4800 +0.00% 0/7; M-GROUP 3.4801 +0.00% 0/126; translation-invariance NOTE line 55) | M-GLOB 0/7, M-GROUP 0/126, Δ=0-before-side | CONFIRMED (theorem part) |
| — LOCO-I-lite −0.2% sub-claim | `RESULTS.md:30`; memory/cycle1-memory.md:44 (corroborated by probe_b4_RESULTS.md:73 "+0.0019 KILLED" context) | −0.2% as transcribed | SECONDARY-ONLY (no primary probe per-image file) |
| — G-bias −0.36% on kodim13 sub-claim | none found (nearest Golomb-bias evidence: probe_loco_RESULTS.md bias-gain column, kodim13 2.87% regular-bits — different metric) | — | UNBANKED (theorem unaffected; figure untraced — either bank the probe or drop the digits) |

## Counts

- CONFIRMED (incl. CONFIRMED-AS-LABELED / WITH-NOTE or CAVEAT): 52 claim-rows
  (§1: 28 per-image + CROWN6 avg/RCT/MLP/Δ% notes; §2: CROWN6, CROWN4, CROWN2, rans-hc-as-labeled, MOE;
  §3: 8 of 9 incl. FULL-W12, both Boss-5 standings, E16 arms, both KOs, MOE-vs-e1;
  §4: hash-cache bundle, LF-NN, bitplane magnitude, M-GLOB/GROUP theorem).
- CORRECTED (draft value wrong): 1 — CROWN-huff vs PIL-m3 p≈.47 → **p≈.30 (W=7)**;
  verdict TIE unchanged. (Plus inherited imprecision: bitplane "+26% vs A1" label is
  +24.9/+25.1% vs A1, +26.3% vs C — no draft edit needed beyond denominator wording.)
- UNBANKED / SECONDARY-ONLY (honestly-marked gaps, draft line 310 already todos perimage.py):
  PNG-9 per-image, WebP-m0 per-image, RUN per-image row (avg 3.511 report-level only),
  byte-LZ 4.18 breakdown, wavelet+zlib 4.78 breakdown, LOCO-lite −0.2% primary,
  G-bias −0.36%-on-13 primary, CROWN3 per-image row (for the p=.375 attribution),
  CROWN-rans-hillclimb-3.309 per-image row.

## Corrections list (actionable)

1. Draft line 160 + Table line 187: `CROWN-huff vs PIL-m3 … p≈.47` →
   **p≈.30 (W=7, 1W-6L, n=7 two-sided)**. Recompute: diffs
   [+0.0266,+0.0272,−0.1169,+0.0233,+0.0020,+0.0028,+0.0186] → ranks of positives
   1+2+3+4+5+6=21, negative 7 → W=7 → exact p=0.296875. Verdict stays TIE.
   Also fix "+0.0023" → "−0.0023 (huff wins avg by 0.0023)" or state direction explicitly.
2. (Non-blocking notes, no value change): bitplane §5 wording "+26% B vs A1" →
   "+25% vs A1 (+26% vs best joint C)"; WebP-m6 AVG footnote "mean of 2dp refs
   (exact PIL avg 3.3160 per probe_b4:66)"; CROWN4 AVG footnote "probe_b19 table;
   b18 json differs by ≤0.0001/img"; e1 AVG 3.7204 footnote "pooled-bytes
   (mean-of-rounded 3.7203)".
3. (Pre-camera-ready gaps, already in draft line 310): bank or drop — PNG-9/WebP-m0
   per-image (run `csrc/perimage.py`), RUN per-image row, byte-LZ/wavelet+zlib
   per-image ledgers, LOCO-lite primary, G-bias −0.36%-on-13 primary, CROWN3
   per-image row, rans-hillclimb-3.309 per-image row.

## Overall verdict

**Numerically sound, not yet camera-ready.** Every load-bearing exact-codec figure
(CROWN6/CROWN4/CROWN2 per-image + averages, JXL-e3 ladder, WebP-m3 PIL row, all
KO/standing Wilcoxons except one) recomputes exactly from banked primary sources.
The single wrong digit found (huff-vs-m3 p .47→.30) does not change any verdict
(TIE stands; all five KOs re-verify at W=0 p=.016; both Boss-5 standings re-verify).
Dead-end spot-checks confirm (hash +2.85%, LF-NN +13.48%, bitplane +26%-class,
bias-theorem zeros). Remaining exposure is precisely the evidence the draft already
flags: PNG-9/WebP-m0/RUN per-image rows and four secondary-only dead-end figures
(byte-LZ 4.18, wavelet+zlib 4.78, LOCO-lite −0.2%, G-bias −0.36%) have no primary
probe artifact — re-run and bank them (or soften to report-level citations) before
camera-ready. The `[VERIFY]` discipline is working as designed.

## 5. Classical baselines appendix (2026-09-08; source `survey/bosstakedown/data/classical7.json`)

| claim (draft §2.1/§4.2/§4.4) | recompute | status |
|---|---|---|
| FLIF v0.4 per-image 3.2252/2.4281/3.3757/2.4190/3.7011/3.0253/2.6146 | bytes 475578/358041/497770/356699/545751/446097/385544 → same 4dp | CONFIRMED |
| FLIF AVG 2.9699 | row-mean 2.9698714 → 2.9699 | CONFIRMED |
| CROWN6 vs FLIF 0W-7L p=.016 (W=0) | diffs all +0.073..+0.5573 → W=0, p=0.015625 | CONFIRMED |
| FLIF beats JXL-e3 7/7 here | per-image FLIF < e3 on 7/7 (max 3.7011 < 3.9141) | CONFIRMED |
| JPEG-LS-CharLS-RGB per-image 5.2462/4.0458/5.1719/3.6369/5.9347/4.4556/3.5391 | bytes → same 4dp | CONFIRMED |
| JPEG-LS-RGB AVG 4.5757 | row-mean 4.5757286 → 4.5757 | CONFIRMED |
| CROWN6 vs JPEG-LS 7W-0L p=.016 | diffs all negative → W=0, p=0.015625 | CONFIRMED |
| JPEG-LS∘YCoCg-R AVG 3.3664 (ablation, not a codec) | row 3.4679/3.1456/3.7666/2.8654/4.0864/3.2931/2.9398 → mean 3.3664 | CONFIRMED |
| vs-us cols: FLIF +7.1%, JLS −30.1% | (3.1978−2.9699)/3.1978=+7.13%; (4.5757−3.1978)/4.5757=−30.11% | CONFIRMED |

Method notes (audited): FLIF v0.4 source build, default `flif -e`, PIL
pixel-compare PASS 7/7; CharLS 2.4.4 RGB defaults, decode-asserted;
YCoCg-R ablation sums 3 per-plane streams (headers ~150B incl, noted).
Literature JPEG-LS 2.82 uses an undisclosed pipeline and is NOT imported —
draft reports the measured number plus the spread (P2-LLM 2.82 vs SEEC 4.36).
CALIC: no reference binary obtainable in-box (only a Python reimplementation
found); literature row retained with "not measured locally".

| QOI per-image 5.0601/4.4618/5.6340/4.1558/5.7280/4.8187/4.5793 | classical7.json bytes 746144/657924/830762/612802/844625/710544/675251 → same 4dp | CONFIRMED |
| QOI AVG 4.9197 | row-mean 4.9196714 → 4.9197 | CONFIRMED |
| CROWN6 vs QOI 7W-0L p=.016 (W=0) | diffs all negative → W=0, p=0.015625 | CONFIRMED |
| Margin 35.0% | (4.9197−3.1978)/4.9197 = 0.3500 | CONFIRMED |
