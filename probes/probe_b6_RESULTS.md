# probe_b6: MA-tree-lite — learned signaled decision tree over context space

**Code:** `experiments/probe_b6_ma.py` (shallow ≤8 leaves/d≤3) + `experiments/probe_b6_b.py` (deep ≤16/d≤5 + sharing).
Summaries: `experiments/probe_b6_summary.json`, `experiments/probe_b6_b_summary.json`.
**Method:** numpy+PIL only, CPU. **Unit:** bpp = total_bits/(H·W·3). Every byte counted.

## 0. Background (mandatory reading — what it told us)

- `memory/cycle1-memory.md`: three dead-ends and WHY — (i) fixed bands (−0.4%: table fragmentation at small scale),
  (ii) naive 16-bin LF conditioning (−8–12%: side stream ~3× the savings), (iii) linear/FIR wall (≈MED: lag-2 acf −0.28
  not linearly exploitable). Lesson: any conditioning must pay MDL-rent; joint patterns > 1-D slices > linear.
- `SURVEY_genai_qoi.md` §1: (1) CALLIC MDL objective `L(φ)+log 1/q` — formalizes "always count header bytes"
  (rank-ablation r=8>r=16 = our LPC-3 overfit); (3) SEEC N=2 optimal, N=8 loses (mask bits eat content savings —
  same fragmentation curve as our bands); (11) FLIF MANIAC → JXL MA-trees: **signaled splits, per-leaf
  predictor+histogram, histogram sharing, static decode**. Our CROWN (LOCO-365 key + quantile groups + per-group
  best predictor) is a 1-level hand-built MA-tree; JXL says learn the SPLITS. That is this probe.
- Classical check (web/spec knowledge, no new code dependency): JXL spec §5.2.2 MA-trees test integer properties
  (neighbor diffs, channel cross-terms, position) at inner nodes; leaves carry (predictor, histogram); tree coded
  with a small fixed context model; enc balances tree cost vs gain. Our design mirrors this with explicit byte counts.

## 1. Harness validation (must pass before trusting deltas)

- MED + order-0 Huffman recomputed in-harness: **3.5782 avg** (per-image 01:3.6179, 02:3.3256, 05:4.0119,
  07:3.1530, 13:4.2378, 19:3.5134, 23:3.1880) → within ±3% of 3.58 anchor → **PASS**.
- CROWN-huff reimplementation (same 5-pred bank, KSET 2..36, map `nKeys·⌈log2K⌉+4+8` + 3b/group ids, tables
  `16+A·24`): **3.3424 avg** vs given external 3.3429 → Δ=0.0005. Harness is calibrated; tree deltas below are
  apples-to-apples (same bank, same table math).

## 2. Design (decoder C-compatible: integer tests, static tables)

- Planar order Y→Co→Cg. Per channel, causal integer features from recon only:
  `|b−c|`, `|a−c|`, `|d−b|`, energy `|b−c|+|a−c|`, `e>>4 cap 8`, `|a−b|`, `|a+b−2c|/2`, neighbor mean `(a+b)/2`,
  `row`, `col` (+ for chroma: `|Yres|`, `Yres`, `Ymean` from the already-decoded Y plane, MED residual).
  Border convention = champion 0/left/top rule via shared `nbhd()` — encoder features ≡ decoder features.
- Residuals are LOCO-sign-flipped (`rf = s·(ch−P)`, same `s` as CROWN) for all 5 predictors
  (MED/TOP/PAETH/GRAD/GAP). Per-leaf cost = min over predictors of (Huffman data + `16+A·24`).
- Greedy best-gain-first growth: at each step try all (feature × threshold) splits
  (thresholds = all cutpoints if ≤16 uniques else 9 quantiles); split kept iff
  `gain = parent − (left+right+27b) > 0` with **EXACT** full-image Huffman math for the accepted split
  (search ranked on stride-4 subsample for speed, gating exact — conservative).
  Side: 24b/inner node (8b feat-id + 16b threshold) + 3b/leaf predictor id. Global 64b header once.
- Stretch: JXL-style histogram sharing — same-predictor leaves may share one table;
  hist-id side `⌈log2(nHist)⌉`b/leaf, greedy merge while net-positive (exact math). Applied to tree AND
  CROWN groups so the comparison stays fair.

## 3. Results — per-image table (headline = deep tree ≤16 leaves/d≤5 + sharing)

| image | deep+sh | CROWN-huff given | Δ vs base | JXL-e3 | Δ vs e3 | shallow ≤8 (ref) | CROWN5 recomputed |
|---|---|---|---|---|---|---|---|
| kodim01 | 3.4173 | 3.4340 | **−0.0167** | 3.36 | +0.0573 | 3.4389 (+0.0049) | 3.4340 |
| kodim02 | 3.0500 | 3.0931 | **−0.0431** | 3.06 | **−0.0100** | 3.0761 (−0.0170) | 3.0939 |
| kodim05 | 3.7510 | 3.7394 | +0.0116 | 3.51 | +0.2410 | 3.7714 (+0.0320) | 3.7391 |
| kodim07 | 2.8288 | 2.8630 | **−0.0342** | 2.73 | +0.0988 | 2.8780 (+0.0150) | 2.8643 |
| kodim13 | 4.0985 | 4.1010 | −0.0025 | 3.91 | +0.1885 | 4.1086 (+0.0076) | 4.0999 |
| kodim19 | 3.2498 | 3.2758 | **−0.0260** | 3.22 | +0.0298 | 3.2735 (−0.0023) | 3.2762 |
| kodim23 | 2.8739 | 2.8940 | **−0.0201** | 2.81 | +0.0639 | 2.9179 (+0.0239) | 2.8897 |
| **avg** | **3.3242** | **3.3429** | **−0.0187 (−0.56%)** | **3.2286** | **+0.0956 (+2.96%)** | 3.3520 (+0.0091) | 3.3424 |

Fair-bank cross-checks (same 5 predictors, same accounting): deep-plain 3.3249 vs CROWN5 3.3424 = **−0.0175**;
deep+sh 3.3242 vs CROWN5+sh 3.3338 = **−0.0096**. Sharing saves CROWN 0.0086 but tree only 0.0007 (see §5).
Shallow (≤8/d≤3) **loses**: 3.3520 vs 3.3424 = +0.0096. Depth budget is load-bearing.

## 4. Learned tree structure summary (deep run)

- Leaf counts: Y (ch0) 4–10 leaves; Co (ch1) 12–16; Cg (ch2) 12–16. Y saturates early (4 leaves on 01/13 —
  further splits don't pay); chroma keeps splitting to the cap → chroma distributions are more splittable
  than Y at this feature set.
- First splits are almost always energy: `|b−c|+|a−c| ≤ 6..10` (Y) or `|b−c| ≤ 0` / `|a−c| ≤ 1` (chroma),
  gains 30–80 kbit on 393kpx images. Second level: `|a−b|`, `|d−b|` refinements (2–20 kbit).
  Threshold 0 on `|b−c|` isolates perfectly-flat neighborhoods — the tree rediscovers run/flat detection.
- Position splits appear where they earn it: kodim19-ch0 `row ≤ 418` (+16.8 kbit) then `row ≤ 639` (+9.9 kbit) —
  portrait 512×768, sky/ground bands differ; kodim19-ch2 `row ≤ 431` (+1.6 kbit). No other image keeps a
  position split → correctly gated, not forced.
- Luma anchoring (`|Yres| ≤ 0`, gains 0.8–3.5 kbit/split) is kept in 6/7 chroma channels but always deep
  (depth ≥2) and small — real, ~0.5–1% of chroma bits, consistent with the survey's −0.5–1% estimate for
  QOI-LUMA-style ports. `Ymean ≤ 137` kept once (kodim13-ch2, +3.4 kbit, bright-sky leaf).
- Per-leaf predictor usage (deep): MED wins 60–80% of leaves; TOP steals 1–5 leaves/image (smooth gradients);
  GRAD/GAP take 1–3 leaves on textured Y (kodim05-ch0: GRAD×2+GAP×1); PAETH takes 1–2 chroma leaves.
  No leaf ever picks a predictor that loses on the full channel — selection is pure win, side (3b) negligible.
- CROWN comparison: CROWN picks K=4–18 on Y but K=36 (22–30 active groups) on chroma. The tree matches/beats
  this with ~half the distributions because its splits are 2-D+ (energy × edge × luma) where CROWN's are
  1-D quantiles of a 365-key ordering. Single-axis depth-3 (≤8 leaves) cannot express the joint
  sign-patterns the 365-key packs (hence the shallow loss, worst on Cg: +5..+25 kbit/channel); depth-5/16
  leaves recover the interactions.

## 5. Exploration trail

1. Shallow tree first (spec sketch: d≤3, ≤8): avg **+0.0096 loss** vs CROWN5. Per-channel autopsy: tree won
   Co (−7..−19 kbit) but lost Y (+2..+25 kbit) and Cg (+6..+25 kbit). Hypothesis: 8 leaves < CROWN's ~25
   chroma groups; joint-pattern deficit.
2. Raised budget to ≤16 leaves/d≤5 (same search, same accounting): avg **−0.0175**, 6/7 images win.
   Confirms hypothesis — the sketch's leaf cap was the binding constraint, not the tree idea.
3. Added histogram sharing: CROWN 3.3424→3.3338 (−0.0086); tree 3.3249→3.3242 (−0.0007). Nearly all tree
   merges rejected (nHist ≈ nLeaves; only kodim02-ch1 12→11, kodim07-ch0 9→8 merge). Interpretation per
   SEEC/CALLIC: CROWN over-fragments (36 groups → tables eat gains); tree leaves are already
   MDL-separated, so sharing has nothing left to fix. Sharing narrows tree's edge (−0.0175→−0.0096) but
   tree still wins both races.
4. kodim05 is the persistent loss (+0.0116 even deep): high-frequency texture everywhere; energy splits
   saturate (ch0 stops at 5 leaves, gains thin). Candidates: distant-repeat/hash-cache (flat patches recur),
   not finer energy splits.

## 6. Verdict

- **Does MA-tree-lite beat 3.3429? YES — with the deep budget: 3.3242 avg (−0.56%), 6/7 images, exact
  byte-counted side info, same predictor bank.** With the as-sketched ≤8-leaf budget: NO (+0.29%).
- **Is Boss 5 (JXL-e3 3.23) in reach? NO.** Gap is +0.0956 (+2.96%); only kodim02 beats e3 (−0.0100).
  Campaign standing is CROWN-rans 3.272; transplanting the tree's −0.56% Huffman gain onto the rANS backend
  projects ≈3.254 — still ~0.7% short of e3. MA-tree-lite is a real contributor (≈1/3 of the e3 gap) but
  not a boss-killer alone. JXL-e9 (3.03) remains ~9% away — needs stacking, not tuning.

## 7. Follow-up (ranked)

1. **Port tree partitions to the rANS backend** (per-leaf rANS streams instead of Huffman): measures the
   true campaign delta (expect the −0.5% to transfer; verifies additivity with the 3-way backend selection).
2. **Offline-global tree + per-image refine**: learn one fixed tree on Kodak-excluded photos, ship in-binary
   (zero side), per-image only leaf predictor ids + optional 1–2 refine splits MDL-gated. Kills the 24b/split
   rent that blocks small leaves.
3. **Stack, don't deepen**: tree + luma-anchored chroma predictors (CfL-style) + 64-entry distant-repeat
   hash-cache front-end (kodim05/13 flat regions) — each attacks residuals the energy tree can't see.
4. **Cross-chroma conditioning**: Cg splits use only Y today; Co residual at same position is decoded before
   Cg and was never offered as a feature — cheapest new split candidate.
5. Retire: deeper-than-5 trees (gains at depth 5 are ≤2 kbit, all rejected at 6+ in pilot), blind histogram
   sharing on tree leaves (proven ~0 here — keep for CROWN's 36-group mode only).

## 8. Decoder-safety statement

Every split test uses **causally-available reconstruction values only**: neighbor recon `a` (left), `b` (top),
`c` (top-left), `d` (top-right), `Ww/NNe/NE` where used by predictors, `row`/`col` position, and — for
Co/Cg leaves only — the fully-decoded Y plane (`Yres`, `Ymean`). No test reads the current pixel, any
future pixel, or any encoder-only statistic (quantile thresholds are constants in the signaled tree).
Leaf decoding is static: read predictor id + histogram id from the header, predict from recon neighbors,
inverse-flip by recomputed `s`, Huffman/rANS-decode. Border rule matches encoder exactly (0/left/top).

## 9. Repro

```
python3 experiments/probe_b6_ma.py   # anchor PASS + shallow tree (~75 s)
python3 experiments/probe_b6_b.py    # deep + sharing (~140 s)
```
