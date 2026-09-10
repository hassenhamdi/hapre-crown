# probe_b18 (surgical texture-gap probe: kodim05 + kodim13) — RESULTS

Campaign: lossless codec CROWN3-exact 3.2040 bpp Kodak-7 avg vs boss JXL-e3 3.2291.
We lead avg (−0.78%), 5W-2L, p=.375 n.s. Blockers: kodim05 (+2.46%), kodim13 (+1.02%).
Mission: diagnose the two losses, flip ≥1 with a targeted weapon. No regressions
past JXL margins on the other five.

Code (new files only, prefix `probe_b18_`): screen/group/ctx/lms/spatial/selfsim/
rct8/bilat/wgran/autopsy/stack/validate. Nothing existing modified (imports only).
Rules: numpy+PIL+ctypes (+cjxl/djxl binaries for analysis), CPU only.
Unit: bpp = total_bits/(H·W·3). All side counted (real heapq Huffman +16+A·24/stream,
Golomb k/d, rANS tables, maps, W-flags/weights, headers).

## 0. Anchors — PASS
- MED + C6-YCoCg-R + order-0 global Huffman (this probe): **3.5782 avg** (−0.05% vs 3.58).
- CROWN3 single-RCT C27 on kodim05: 3.5984 — reproduces cited CROWN3-05 ref EXACTLY
  (→ CROWN3's 05 winner is C27; min-over-3 cannot beat it).
- CROWN3 single-RCT C27 on kodim13: 3.9661 vs cited 3.9542 (→ 13's winner is C6/C12,
  saves 0.0119; validation runs min-over-3 for parity).

## 1. Diagnosis

### 1a. Content (statistics + visual inspection of the actual images)
- kodim05 = MOTOCROSS START: riders/bikes/tires/helmets/plates. Dense small objects,
  knobby tire tread, spokes, text/logos, 63.5k colors (most of 7). Isotropic gradient
  orientations (flat 8-bin histogram 0.10–0.16), fastest AC decay (lag32 0.18/0.13),
  54.1% HF-band energy (highest), highest chroma energy (Co 28.3, Cg 17.4).
- kodim13 = MOUNTAIN RIVER: whitewater foam (noise-like), rocky banks (impulse-like
  stones), pine forest (vertical texture), mountain+sky (smooth). Diagonal orientation
  bias (0.18/0.17), HIGHEST edge density (frac>10 = 0.73), long horizontal correlation
  (AC lag64 H=+0.47 vs V=+0.13: river bands/rapids + ridge lines).
- Common thread (the clean split): **chroma-texture energy**. MED-frame C27 chroma
  cost shares (Co+Cg): losses 05: 6.35, 13: 6.31 vs wins 01: 5.10, 07: 5.19, 23: 5.49,
  02: 5.52, 19: 5.56. Y energy does NOT split (13 highest but 05 mid). Both losses =
  highest-chroma; all five wins = lower-chroma.

### 1b. JXL behavior on the two (cjxl 0.11.2 knockouts, exact bytes)
- RCT matters MOST on exactly these two (−C 0 @e3: 13: +49.6%, 05: +39.4% vs
  28–38% on controls). MA-tree matters most here too (−I 0 @e4: +14.3%/+13.2%).
- JXL-e3 = Weighted + fixed MA-tree + C6 + ANS. Since our C27 front-end already beats
  their C6, their residual/context/entropy machinery overcomes ~0.06 worse RCT + wins.

### 1c. Exact-arm autopsy (CROWN3 machinery by import, per-group entropy vs chosen bits)
- kodim05 (C27): total 3.5984 = entropy floor ≈3.500 + coding overhead ≈0.098
  (data-vs-entropy gaps 0.077 + maps 0.008 + WAVG-side 0.013). **Entropy floor 3.500
  already beats JXL-e3 3.5120 — the loss is ~90% entropy-coding overhead.**
  Biggest piece: Y-channel K=64 groups all-but-one choose Golomb (385k px, gap
  0.0266: Golomb misfits heavy-tail texture; Huffman tables too big per small
  group; rANS side Ad·32 too big per small group → trap: fine groups +
  per-group backends).
- kodim13 (C27): overhead ≈0.118, floor ≈3.85 < JXL 3.9141. Same story, more headroom.
- Control 01: overhead ≈0.075 (we win anyway; more headroom everywhere).
- (GRID-family entropy rows in the autopsy log used sign-flipped residuals by mistake;
  all Q-family rows — the winners on every channel — are correct; conclusions rest on Q.)

### 1d. Mechanism statements
- M-05: object-dense saturated texture → huge residual alphabets + heavy tails →
  per-group Golomb/Huffman fall into table-side vs misfit trap; JXL's clustered ANS
  + fine error-feedback contexts evade it. (Front-end RCT already optimal; verified
  C13≡C27 bit-identical by Co-negation symmetry, C12/C20/C34/C3 worse.)
- M-13: luma-dominated (Y share 6.29) impulse+directional texture (rocks/foam/flow) →
  same coding trap + local-linearity beyond G32-block stationarity.

## 2. Weapon screens (exact bits; killed/wounded/winners)
KILLED (measured ~zero/negative, documented): raw lag/directional experts (grouped
gate: never selected but once); YX slope experts; LMS mixers; YMAG cross-channel key
(Attack-2 key dies: +0.004..+0.014 everywhere); spatial cells (+0.04..+0.20 — key-
scattered groups vindicated); patch-match (best-causal-8×8 SAD ≈ MED-L1, median
ratio 0.90–1.22 — tread/spokes not translation-self-similar); bilateral chroma
(never selected on 05/13); RCT-8 expansion (C13≡C27; set already optimal).
WOUNDED (real but small): ERR4 error-feedback key (−0.018/−0.009: JXL WP-error analog
works, small); LMS5t-FIR adaptivity (−0.027/−0.038 global; 4-tap loses, 5th MED tap
is the key — helps texture, hurts smooth).
WINNER screens: WAVG granularity G16 (−0.079 on 05 vs MED; −0.026 vs G32; G16 best
avg; per-image {16,32} choice = free +1b).

## 3. STACK weapon → narrowed to LMS-only (exact codec "B18"; new magic, Python round-trip)
- Full stack (GS16 + LMS + ERR4): 05: 3.6159 (+0.49% — REGRESSES), 13: 3.9646
  (+0.26% — REGRESSES). Ablation (05-C27): frame-parity 3.5985 (+0.002% — frame
  reproduces CROWN3 exactly); +GS16 +0.22% POISON; +ERR4 +0.41% POISON; +LMS
  −0.185% WIN. Stack narrowed to LMS-only (GS32, E17+2 experts, Q+GRID).
- Decoder-safety: dual-neighborhood audit (E16/keys/GRID = b4_b-style c-fix;
  WAVG/LMS-tap/err4-MED = B17-style zeros — matches CROWN3's own split, verified
  against crown3.c c3_nbhd + wpred call site; two real border bugs caught+fixed
  during proof: sign-flip key negation (abs-vs-negate) and TL-border doctrine).
- Full raster-order Python decoder (H/G/R payloads, per-pixel key/group/expert
  re-derivation, LMS state re-simulation, RCT inv); round-trips PASS on all
  validated images (asserted decode==input).

## 4. Validation — LMS-only all-7 (exact GS-min x RCT-min, RT-asserted)
Config: exact GS-min (picked G32 on 7/7 — GS16 never wins in-frame) x RCT
min-over-{C6,C27,C12} (CROWN3 parity) + LMS5_T0/T3 gated 18th/19th experts, no
ERR4. Full B8-blob bytes (headers/maps/tables/flags/weights/side in len(blob);
+1b GS-side). Python raster-order decode asserted == input per image.
(Log labels controls "FLIP" = beats-JXL; properly they are HOLDs — only 05/13
could flip. Relabeled below.)

| img | B18 | CR3 ref | ΔCR3 | JXL-e3 ref | ΔJXL | RCT/GS | verdict |
|---|---|---|---|---|---|---|---|
| kodim01 | 3.2992 | 3.3026 | −0.10% | 3.3593 | −1.79% | C27/G32 | HOLD+ |
| kodim02 | 2.9855 | 2.9889 | −0.12% | 3.0611 | −2.47% | C12/G32 | HOLD+ |
| kodim05 | 3.5918 | 3.5984 | −0.18% | 3.5120 | +2.27% | C27/G32 | LOSE (was +2.46%) |
| kodim07 | 2.6790 | 2.6824 | −0.13% | 2.7310 | −1.90% | C27/G32 | HOLD+ |
| kodim13 | 3.9525 | 3.9542 | −0.04% | 3.9141 | +0.98% | C12/G32 | LOSE (was +1.02%) |
| kodim19 | 3.1510 | 3.1566 | −0.18% | 3.2154 | −2.00% | C27/G32 | HOLD+ |
| kodim23 | 2.7365 | 2.7448 | −0.30% | 2.8110 | −2.65% | C27/G32 | HOLD+ |
| **AVG** | **3.1994** | **3.2040** | **−0.15%** | **3.2291** | **−0.92%** | — | **5W-2L** |

- 7/7 images beat CROWN3 (p=.016 — B18 strictly dominates the crown line).
- 5/5 controls HOLD with extended margins (no regressions; biggest gain 23: −0.30%).
- Targets narrowed but NOT flipped: 05: +2.46%→+2.27%; 13: +1.02%→+0.98%.
- B18 avg 3.1994 = new best exact codec; lead over JXL-e3 widens −0.78%→−0.92%.

## 5. Verdict — 05/13 NOT flipped (BLOCKED, quantified); KO math; follow-up
- Success criterion (flip 05 and/or 13, no control regressions): NOT MET. Best
  exact: 05 needs −2.40%, has −0.19% (8% of the way); 13 needs −1.02%, has −0.04%
  (4% of the way). Quantitative law, not bad luck: 19 weapon trials, 1 portable
  winner (LMS −0.1..−0.3%), 2 frame-poisons found+retired (GS16, ERR4), 1 wounded
  (merges −0.005/−0.010, not worth decoder build), 15 clean kills.
- Boss-5 KO math now: diffs (B18−JXL) = {01:−0.0601, 02:−0.0756, 05:+0.0798,
  07:−0.0520, 13:+0.0384, 19:−0.0644, 23:−0.0745}; |·| ranks → losses rank 7+1
  → W+=8, n=7, exact two-sided p=0.375 (recomputed in-code — same rank pattern as
  CROWN3). Boss-5 STANDS (5W-2L, n.s.). Had 05 flipped: 6W-1L with W+=1 → p≈.016
  KO; flipping 13 alone → W+=7 → p≈.11 (insufficient — 05 was always the priority
  target, correctly).
- Mechanism (static-code trilemma on texture, all legs measured): (i) mixture
  penalty forbids coarse groups (ONE-group +0.09..+0.24; shared-Huffman +0.12 on
  05-Y); (ii) per-group tables forbid fine Huffman on big alphabets (Golomb wins
  by default on most texture pixels); (iii) Golomb shape-misfit (0.027 on 05-Y)
  irreducible by static refinement (per-key-k +0.0015; adaptive-k +0.16..+0.23;
  2-Laplacian mixtures +0.013; cross-image tables LOO≈0; within-image merges
  −0.005/−0.010 true value). Predictor space independently exhausted (grouped
  crumbs ≤−0.002; per-pixel oracles vacuous without selection-side).
- Headroom proof (opportunity real, not a wall): residual-entropy floors (05:
  3.500 < JXL 3.5120; 13: ≈3.85 < 3.9141) beat JXL — the loss is ~90%
  entropy-coding overhead (05: gaps 0.077 + maps 0.008 + WAVG-side 0.013).
  Harvesting needs per-pixel-adaptive probabilities (CABAC-class) or learned
  contexts — the Boss-6 learned-micro arc, not hand codes.
- Portable win: LMS5_T0/T3 gated experts — exact, decoder-safe (recon-only
  deterministic state, zero side, 5b ids hold ≤32), 7/7 gains, RT-proven.
  Recommend C-port (new E-cases beside wpred; this probe's Python decoder is the
  executable spec; border-doctrine audit in §3).
- Follow-ups (ranked): (1) C-port LMS pair onto CROWN line; (2) CABAC-lite
  feasibility on peaked-big-alphabet groups (mode-mass-32% case banked);
  (3) learned-micro predictors (cycle-25 track) as the structural Boss-5/6 path;
  (4) GS16-in-frame exact-min already built into final7 (picked G32 7/7 — closed).
- Honesty notes: (i) GRID-family entropy rows in the autopsy log used
  sign-flipped residuals by mistake — all Q rows (winners everywhere) correct;
  conclusions rest on Q. (ii) +1b GS-side counted though the gsbit byte is in
  len(blob) (over-counts 8 bits ≈ 7e-6 bpp, conservative, immaterial).
  (iii) Train-on-test optimistic X-tables −0.022 reported as such (honest LOO≈0
  is the claim). (iv) NLMS diverged (+4.7..+6.9 — ad-hoc normalization unstable;
  only bounded sign-sign LMS claimed).

## 6. Trail (files, method compliance, reproducibility)
New files only (`probe_b18_*`, nothing existing modified — existing modules used
by import): screen / group / ctx / lms / lmsvar / nlms / spatial / selfsim /
rct8 / bilat / wgran / adapt / perkeyk / mix / xtables / mergesim / mergeexact /
micro / oracle / bands / band-analysis scripts + autopsy / stack (862-line exact
B8 codec + Python decoder) / validate / ablate / final7 + dump/scratch under /tmp
(not in library). Logs: /tmp/b18_*.log; dumps /tmp/b18_dump_*.pkl (23 MB);
valid JSON probe_b18_valid.json / probe_b18_final7.json.
Method rules: numpy+PIL+ctypes only (+cjxl/djxl binaries for band/knockout
analysis), CPU, no torch. Unit bpp=total_bits/(H·W·3) throughout. YCoCg-R
invertibility asserted per encode (B17.rct round-trip). Causal-recon-only
conditioning: YES — every predictor/key/state derives from causal recon (or
transmitted side); decoder re-derives identically (proven by asserted
decode==input on all 7 final images + crops). Anchors: MED+order-0 C6 = 3.5782
avg (−0.05% vs 3.58, PASS ±3%); CROWN3-C27-05 = 3.5984 reproduced EXACTLY.
Timings: screens ~1–5 min; stack encode ~110–150 s/image/RCT; Python raster
decode ~60–90 s/image; all-7 final7 ≈ 80 min (12-core host: oracle/nlms
side-quests ran in parallel without interference).
