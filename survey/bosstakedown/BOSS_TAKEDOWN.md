# BOSS TAKEDOWN — adversarial dissection of JPEG-XL lossless (the bosses themselves)

Date: 2026-09-07. Analyst: boss-takedown mission. Tools: cjxl/djxl/jxlinfo 0.11.2, web source audit (libjxl, read-only, no clone), GitHub-issue literature survey.
Campaign state: our best exact codec CROWN2 3.2686 bpp Kodak-7 avg (ALL-7 PASS); classical full-stack probe 3.2515. Bosses: JXL-e3 3.2291 (−1.2% needed), JXL-e9 3.03 (−7.4% needed).
All bpp below = bits/(H·W·3) unless noted. Kodak-7: 768×512×3, N=1179648 sub-pixels. Data: `data/` (CSVs + logs + .jxl repro files).

---

## 1. Reproduced boss table (per-effort, `cjxl -d 0 -e E`, 4 threads, exact bytes)

| img | e1 | e2 | e3 (BOSS-5) | e4 | e5 | e6 | e7 | e8 | e9 (BOSS-6) |
|---|---|---|---|---|---|---|---|---|---|
| kodim01 | 3.7888 | 3.5869 | 3.3593 | 3.3406 | 3.3226 | 3.1966 | 3.1691 | 3.1723 | 3.1336 |
| kodim02 | 3.4894 | 3.2605 | 3.0611 | 3.0333 | 2.9866 | 2.9377 | 2.8589 | 2.8337 | 2.8401 |
| kodim05 | 4.1091 | 3.9997 | 3.5120 | 3.5106 | 3.4970 | 3.3984 | 3.3588 | 3.3755 | 3.4082 |
| kodim07 | 3.3244 | 3.0864 | 2.7310 | 2.7203 | 2.6852 | 2.5456 | 2.4972 | 2.5098 | 2.4433 |
| kodim13 | 4.3389 | 4.2606 | 3.9141 | 3.9039 | 3.8711 | 3.7982 | 3.7774 | 3.7822 | 3.7815 |
| kodim19 | 3.6637 | 3.4717 | 3.2154 | 3.1793 | 3.1730 | 3.0739 | 3.0419 | 3.0334 | 3.0101 |
| kodim23 | 3.3281 | 3.1226 | 2.8110 | 2.7951 | 2.7720 | 2.6723 | 2.5937 | 2.6025 | 2.6097 |
| **AVG** | **3.7204** | **3.5412** | **3.2291** | **3.2119** | **3.1868** | **3.0890** | **3.0424** | **3.0442** | **3.0324** |
| enc s/img | 0.019 | 0.040 | 0.060 | 0.230 | 0.384 | 0.558 | 0.973 | 3.164 | 4.303 |
| Δ vs prev | — | −4.82% | **−8.81%** | −0.53% | −0.78% | **−3.07%** | −1.51% | **+0.06%** | −0.39% |

Boss numbers reproduced exactly (e3 3.2291 ✓, e9 3.0324 ≈ 3.03 ✓). e10 spot-check (kodim07): 2.4359, −0.3% vs e9 for 3.5× time — ladder is flat past e7.

### 1.1 Effort-ladder mechanism map (behavioral knock-outs + libjxl source audit)

| step | Δavg | mechanism (what the effort buys) |
|---|---|---|
| e1 | (4.75 PNG → 3.72) | Fixed Gradient predictor, Huffman+RLE-only entropy, NO RCT (`-C` flags all no-op: 4.1091×3), no MA tree, no palette. Decode 12 ms. |
| e1→e2 | −4.8% | Fixed RCT-6 (YCoCg) switched on (`-C 0` costs +39%: 5.5496 vs 3.9996); ANS replaces Huffman (decode 12→33 ms by e3); global channel palette. Predictor/predictor-set flags still no-op (fixed tree). |
| e2→e3 | **−8.8%** | Gradient→Weighted predictor + fixed WP-error MA-tree (static 32-leaf tree; `-I 0`/`-I 100`/`-P`/`-E` ALL no-op at e3 — e3's structure is effort-fixed). Source: `enc_modular.cc:ComputeTree` fixed-tree branch, `HistogramParams` kFast/ANS. Decode slows 2.8× (ANS + more contexts). |
| e3→e4 | −0.5% | **MA-tree learning switches on** (pixel-sampled greedy tree, default `-I≈50`): `-I 0` costs +13% (3.9740 vs 3.5106 on kodim05). Per-group RCT search NOT yet on (global YCoCg). |
| e4→e5 | −0.8% | **Pivot: per-group RCT search on** (`EstimateCost` trials), local palette + patches. |
| e5→e6 | **−3.1%** | More RCT candidates + wider MA properties/thresholds. Best time/ratio step on the ladder (0.56 s for −3.1%). |
| e6→e7 | −1.5% | Squirrel default; more RCTs, props 7/max48. Sweet spot. |
| e7→e8 | **+0.06% REGRESSION** | Precise-ANS + WP-mode search + bigger tables overfit/misfire. Matches upstream issue #4107 (non-monotonic ladder, context-clustering/LZ77 suspect). |
| e8→e9 | −0.4% | kBest entropy + full clustering + LZ77 + all-16-props; 4.4× time for −0.4%. e9 default ≈ I50, group 256, E0, RCT-search. |

Bit destinations at e9 (source audit, photo ~3 bpp): residual tokens + HybridUInt raw bits ~87–92%; histogram/ANS tables 4–8% (`ClusterHistograms`, `EncodeContextMap`, `ANS_TAB_SIZE=4096`); MA-tree signaling 1–3% (6-context tree code); RCT/palette/squeeze/patches <1% combined; framing/TOC ~0.005 bpp.
Component knock-out value at e9 (kodim05 / kodim07): RCT −29.5%/−28.4% → MA-tree −14.0%/−18.5% → predictor-mix −0.3..−0.8% → E-props −0.15%/−0.6% → group-256→1024 −2.1%/−1.0% (a *gain* when changed — §2.1) → palette 0 on photos.

---

## 2. Top-3 weaknesses found (all with local evidence)

### W1. Default 256×256 groups fragment statistics — 1024 groups win everywhere (−1.4% at e9, and e7-g3 BEATS e9-default)
Forcing `-g 3` (1024) beats default on 6–7/7 images at every effort ≥3: e3 3.2291→3.2217 (−0.23%), e7 3.0424→**3.0121** (−1.0%), e9 3.0324→**2.9898** (−1.4%). JXL's strongest *default* boss (e9) is beaten by its own e7 with bigger groups in 1/4 the time. Source audit: 768×512 = 6 groups at 256 (fragmented histograms, `stream_id` tree splits, context-map cost); group prop is even dropped from MA search at e4–e8 (`num_streams<30` guard) and re-added at e9 — the plumbing admits groups are overhead. For us: independent confirmation of our WIDE-K lesson — adaptation regions should be LARGE; our merge pass (415 clusters from 773) is the same force. (`data/group_full7.csv`)

### W2. The ladder is non-monotonic and e9 overfits — e7 beats e9 on 3/7 images, kodim05 degrades monotonically e7→e8→e9
kodim05: 3.3588 → 3.3755 → 3.4082 (+1.5% worse for +4.4× time); kodim02/23 also best at ≤e8. `-I 100` (fits tree on ALL pixels yet raises the split-cost bar) BEATS e9-default on kodim05 (3.3941 < 3.4082): the default I50 tree over-splits. Effort 8 regresses on average (+0.06%). Upstream confirms: issues #4107 (e5<e10 by +4.6% on gray), #3888 (e9 always beats e10/e11 on JPEG path — "smaller threshold ≠ better"), #3323/PR #3337 (pathological e9 predictor default). The e8→e9 precision machinery (Precise ANS, full clustering, LZ77 on Laplacian residuals where repeats are rare) does not pay. For us: (a) never chase e9-default as "optimal" — the true boss is ~1.5–3% lower (tuned e9) but ALSO e9-default is a soft target our lean per-group Huffman/rANS already mimics; (b) any tree/clustering we add needs the MDL gate we already use — JXL is the cautionary tale of ungated growth.

### W3. Fixed-RCT default search is beaten by ONE fixed RCT (−0.6% at e3; even −0.1..−0.3% at e9)
At e3 the `-C -1` "search" always picks RCT-6 (forcing `-C 6` ≡ default on 7/7), yet RCT-27/13/12 (same YCoCg pattern, different channel permutations) beat it on 6/7: e3-C27 avg **3.2083** (−0.65%: 01:3.3402 02:3.0646 05:3.4914 07:2.6896 13:3.8829 19:3.2010 23:2.7886). Even at e9, forcing global `-C 12` beats the per-group search (kodim07 −0.25%, kodim05 −0.12%). Full 24-point RCT sweep saved in §5 table. Two readings, both actionable: (i) JXL's `EstimateCost` RCT heuristic (fixed-uint, maxdiff cutoffs, tree-unaware) misfires — their search is weaker than their dictionary; (ii) for OUR codec, which uses ONE global YCoCg-R: channel-permutation + RCT-type selection is an unharvested −0.5%-class front-end gain (our G-first permutation check: G-first orderings cut chroma energy −5.7%, consistent direction). C-options {7,14,21} catastrophic (4.48) — invalid/degenerate entries accepted silently, another robustness wart.

Runner-up weaknesses (evidence in data/logs, one line each): (a) squeeze/R1 catastrophic on photos (+15–52%: default-off is load-bearing; `-R 0` forced is +0.3..2.4% worse than default ⇒ default keeps some responsive path — flag semantics unclear); (b) pure noise EXPANDS (+9% over raw 8.0: 8.714 — prediction without raw-fallback; same bug class our old 0-ctx driver has, which hard-FAILS decode on noise, rc=−2 — robustness TODO before publish); (c) e3 can't do periodic (0.563) where e7+ does 0.024 (−96%): fixed contexts can't learn periods — but Kodak has little periodic content, so not our gap-closer; (d) palette/patches dead weight on photos (all palette flags no-op) yet searched at encode time.

---

## 3. Where JXL beats us / we beat JXL (content-adaptive picture)

Our FULL-stack probe (3.2515) vs e3 (3.2291) per image: 01: +0.0009 (tie) · 02: **−0.0570 WIN** · 05: +0.1363 (worst loss) · 07: +0.0024 (tie) · 13: +0.1022 (2nd loss) · 19: −0.0118 WIN · 23: −0.0164 WIN.
Image stats (std/mean-grad/sat%/unique-kcolors): 02 = low-grad + HIGH-sat (29.9%) + few-colors (13.5k) → our per-group Huffman + E16 experts love saturated-smooth; 05/13 = highest gradients (25.2/34.0) + most colors → JXL's ANS + rich static contexts model high-entropy texture residuals better than our per-group Huffman. **The 1.2% gap is texture-concentrated (05+13 = +0.239 of the +0.022 avg deficit... i.e. everything else nets −0.02).**
Synthetics (JXL e3→e9): gradient ≈0 (both), graphics/palette ≈0, periodic 0.563→0.024 (JXL strength at e7+, e3 weak), sky 2.90→1.83 (−37%), grain 6.42→6.08, noise 8.84→8.71 (both expand). Our MED+Huffman on gradient: 1.0007 vs JXL 0.004 — 250× relative, ~0.03 bpp-avg absolute: irrelevant to Kodak, do not chase.
e3→e9 gains concentrate on SMOOTH images (kodim07 −10.5%, kodim01 −6.7%) not texture (kodim05 −3.0%, kodim13 −3.4%): e9's machinery (big MA props, WP search, precise entropy) harvests smooth-field structure; texture stays hard for everyone.

---

## 4. TAKEDOWN PLAN (ranked; all внутри C-speed/integer scope)

### Attack 1 (closes Boss-5 alone, −0.7..−1.2% expected): per-group RCT + permutation front-end + Weighted-predictor arm
Mechanism: JXL's own numbers prove the front-end is worth −0.6% (fixed C27 vs C6) and its per-group search another −0.1..−5% (e9 −C 6 KO: kodim07 +5.4%, kodim05 +1.1%). Our codec runs ONE global YCoCg-R with MED/GAP-class experts — no RCT search, no Weighted predictor (JXL's e3 weapon: Gradient→Weighted alone was −8.8%). Cost: encoder tries ~4 RCT/permutation candidates × E16 experts per group under existing MDL gate; decoder = 2–3 b selector per group (already have map plumbing). Expected gain: −0.5% (RCT-27-class permutation) + −0.2..−0.5% (Weighted expert wins smooth groups: kodim01/07 ties flip) ≈ −0.7..−1.0% — exactly the −0.69% FULL needs (−1.22% from CROWN2-exact needs RCT + one of below).
First experiment (probe-only, numpy, no C): `experiments/probe_b17_rct.py` — for each Kodak image, YCoCg-R under 6 channel-permutations × {subtract-only, YCoCg} × E16+Weighted experts on existing LOCO groups; exact Huffman-bit objective incl. 3 b/group selector; report Δ vs CROWN2-exact per image + which images flip. Metrics: per-image bpp, W/L vs e3-3.2291, Wilcoxon. Go/no-go: ≥−0.4% avg with wins on 01/07 → port to C.

### Attack 2 (Boss-5 insurance + Boss-6 path, −0.3..−0.8%): cross-channel MA properties (−E analog) + noise/raw fallback
Mechanism: JXL `-E 3/11` (previous-channel properties in MA tree) gains −0.15%/−0.6% on photos with ZERO decoder-side model — pure context enrichment (co-located Y magnitude/activity as chroma-group selector). Our codec codes Y/Co/Cg independently (cross-channel residual features retired on |corr|<0.11 — but PROPERTIES ≠ residuals; JXL proves the property channel works). Plus: raw-fallback for incompressible blocks (JXL expands noise +9%; our old driver FAILS it) — cheap rate insurance on texture, small but same images (05/13) where we bleed. Cost: +1 context feature per group (existing clustering absorbs it), +1 b/block raw flag gated by exact-bit check.
First experiment: `probe_b17_xch.py` — add co-located-Y-magnitude-bin as second-level group key (Cartesian with existing groups → re-merge pass), exact bits; plus per-16×16-block raw-vs-coded gate on kodim05/13. Metrics: Δbpp on 05/13 specifically (need −0.05 combined there). Go/no-go: wins ≥3/7 with no loss >+0.1% anywhere.

### Attack 3 ("their power against us" inversion, −0.3..−0.6% + strategic): cjxl-as-oracle harness + group-size doctrine
Mechanism: (a) JXL's ladder proves bigger adaptation regions win (−1.4% e9-g3; our merges −0.37%, WIDE-K −0.34% — same force, we under-apply it: 415 clusters still too many). Push grouping coarser (K≤16 super-groups, merge to MDL floor) — zero decoder change. (b) Use `cjxl -d 0 -e 3 -C {6,12,13,27}` (0.05 s/img!) as a 4-way RCT oracle during OUR encoder R&D to label per-image/per-region best-RCT ground truth, then distill to 2 integer rules (energy-based, no JXL at runtime) — JXL's search becomes our training data. (c) Positioning: e7-g3 (3.0121, 1.2 s) vs e9-default (3.0324, 4.3 s) proves even JXL's own users should stop at tuned-e7 — our ms-scale exact codec competes on the Pareto frontier, not the density point.
First experiment: `probe_b17_merge2.py` — continue greedy merges past current stop (allow clusters→~100 total, exact gate) + oracle-label script (`oracle_rct.sh`: 4× e3 encodes, pick best C per image, correlate with our group stats). Metrics: merge-floor Δ; oracle-vs-rule agreement %. Go/no-go: merges find another −0.2% → port.

NOT pursued (evidence-backed): finer MA splits without gates (JXL e8/e9 over-split lesson), squeeze/LF-side streams (retired 4×: +10–13%), transform-bitplane arcs (retired: +2.4..3.8%), byte-LZ on residuals (dead twice), bigger expert banks without selection gating, any decode-time NN.

---

## 5. Verdicts

**Boss-5 feasibility (e3 3.2291, need −1.22% from CROWN2-exact / −0.69% from FULL-3.2515): FEASIBLE with Attack 1 (+2 or 3).** Closest images to flipping: kodim01 (+0.0009), kodim07 (+0.0024) — a Weighted-predictor arm alone likely flips both; kodim05/13 need RCT-permutation + cross-channel props. Required math from CROWN2-exact: −0.040 bpp avg; Attack 1 (−0.02..−0.03) + Attack 2/3 (−0.01..−0.02) covers it with all-exact-bits accounting. Honesty note: JXL can also move — e3-C27 (3.2083, one flag) widens our CROWN2 gap to +1.9%; race, not a static bar. But C27 is OUR free idea too (same dictionary), so net race position unchanged if we port it first.

**Boss-6 honesty check (e9 3.0324, need −7.4% from 3.27; −6.8% from FULL): NOT reachable classically — and the bar is lower than it looks.** Our campaign already proved a classical ceiling at 3.2515 (+0.69% vs e3); e9's remaining −6% over e3 decomposes (measured) into learned-MA (−3..−4% of it), per-group RCT (−1%), precise-entropy/clustering (−0.5..−1%), WP-modes/palette (−0.4%). Harvesting ALL of that inside our architecture = rebuilding JXL's encoder around our backend — man-years, and JXL's tuned ceiling (−g3 −I100 −E11: −3.1% below default-e9 on tests, ≈2.94 est. avg) moves the true bar to ≈−10%. The honest Boss-6 path is the learned-micro track already opened (cycle-25: int16 micro-MLP keeps 70% of −1.64% order-0 gain): per-group learned predictors + offline-trained trees ("learn the tables, ship the tables" — survey TOP-1), NOT more hand contexts. Recommend: kill Boss-5 with Attacks 1–3, consolidate the C port, publish; run learned-micro as the separate Boss-6 arc with its own budget.

---

## Appendix A. RCT sweep at e3 (avg Kodak-7; default −C−1 ≡ C6 = 3.2290)

C1 3.9062 · C2 3.8729 · C3 3.2770 · C4 3.8467 · C5 3.2485 · C6 3.2290 · C7 4.4819 · C8 3.8595 · C9 3.8869 · C10 3.2423 · C11 3.8564 · C12 3.2105 · **C13 3.2092** · C14 4.4819 · C15 3.9017 · C16 3.9075 · C17 3.3069 · C18 3.8375 · C19 3.2367 · C20 3.2198 · C21 4.4819 · **C27 3.2083** · C34 3.2207 · C41 3.2289. (C7/14/21 degenerate 4.4819 — silent acceptance wart.)
e4 fixed-RCT landscape (kodim07: C0 3.5194, C3 2.7260, C5 2.7363, C6 2.7202, C10 2.7195, **C12 2.6792** vs default 2.7203; kodim05: **C12 3.4949** vs default 3.5106).
Stacked JXL ceiling probes: e9−g3−I100−E11 kodim05 3.3031 (−3.1%), kodim07 2.3655 (−3.2%); e9−C12 beats e9-default (−0.12%/−0.25%); −E3/11: −0.15%/−0.63%; palettes all no-op on photos.

## Appendix B. Synthetics (bpp; N/A = not measured — Kodak pattern is the comparator)

gradient e3 0.004→e9 0.001 (ours 1.0007) · graphics 0.004→0.002 · periodic 0.563→0.024 (e3 weak, e7+ 20× better) · sky 2.900→1.832 · grain 6.415→6.081 · noise 8.837→8.714 (EXPANDS vs raw 8.0). Generators: `data/syn_*.png` (script §6).

## Appendix C. Subagent intelligence credits (full texts not stored; key pointers)
- Code audit (libjxl web read-only): tier map e→10−e (Lightning..Glacier/Tectonic); fixed trees e1–e3 (`enc_modular.cc:ComputeTree`, `enc_encoding.cc:MakeFixedTree`); learning from e4 (`CollectPixelSamples/GatherTreeData/LearnTree`, `I=.5` default, threshold=82+14·tier); per-group RCT pivot at e5 (`PrepareStreamParams:EstimateCost`); entropy ladder kFast→kBest (`enc_ans_params.h`, `enc_ans.cc`, `enc_cluster.h`); bit-destination split §1.1; decode-cost implication (Huffman→ANS e1→e2/3). Actionable test matrix (−g/−I/−P/−C/−E/−R) all validated behaviorally in §1–2.
- Literature: encode_effort.md ladder semantics (warns e2>e3 non-photo, e3>e4 photo expected); issues #4107 (non-monotonic), #3441/#4387 (palette fragility, `-Y 0`), #3888 (e9>e10/11 thresholds), #3323/PR#3337 (e9 predictor pathology), #4447/#3729 (fast-lossless/small-image regressions); Cloudinary Pareto + modular explainer (photos ~10bpp noise wall; manga/graphics = target classes); WangXuan95 bench (e3→e8 photo ≈2.6–5% for 32× time); forum max-density recipe `-g3 -E11 -I100` (validated §A); no beating fork (jxl-encoder/libjxlz/fjxl are speed/decode plays).

## Appendix D. Repro commands (all exact; 4 threads; bpp = bytes·8/1179648)
```
cjxl -d 0 -e E IMG.png OUT.jxl [--num_threads=4]            # ladder (E=1..9)
cjxl -d 0 -e 9 -I 0|-C 0|-P 4|-E 0|-R 0|-R 1|-g 0..3 IMG...  # knock-outs (§1.1, data/ablation*.csv)
cjxl -d 0 -e 3 -C 27 IMG.png ...                            # e3-C27 3.2083 boss-strengthener
cjxl -d 0 -e 9 -g 3 -I 100 -E 11 IMG.png ...                # tuned-e9 ceiling probe (~2.94 est avg)
python3 -c "from PIL import Image..."                       # synthetics (see §6 script in session; files data/syn_*.png)
```
Disk used: 78 MB under `survey/bosstakedown/` (mostly repro .jxl). Nothing outside that dir touched; no files under `/tmp/opencode/autocompress/` modified; no clones made (web-read audit instead — disk was fine).

(End of report.)
