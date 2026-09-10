# probe_b5 RESULTS — per-group backend choice + expert-bank expansion (Boss-3 campaign)

Branch B5 (exploratory). numpy + PIL only (rANS byte counts via EXISTING `libhapre.so` M=14 through ctypes — no existing file modified).
UNIT RULE throughout: `bpp = total_bits/(H*W*3)`. Every byte counted:
- Huffman stream: `sum(count*len)` real heapq + `16+A*24` table bits.
- Golomb-Rice stream: exact `sum((M>>k)+1+k)` best k 0..12, `M=2v (v>=0) else -2v-1`, +4b k-side, NO table.
- rANS stream: measured `rans_encode` payload + `16+A*32` freq-table bits, `rans_decode` asserted EVERY stream.
- Side: per-channel map `active_ctx*ceil(log2K)+4b K-id+8b header` + per-group `ceil(log2E)` predid + per-group backend-choice (`1b` 2-way / `2b` 3-way) + 64b global.
Border = champion 0/left/top rule. YCoCg-R round-trip asserted. No runs (CROWN ablative: runs hurt).

## 1. Anchors (exact, this harness — `probe_b5_a.py`)

| image | MED order-0 Huff | CROWN-huff (definitive) | CROWN-rans |
|---|---|---|---|
| kodim01 | 3.6179 | 3.4340 | 3.4002 |
| kodim02 | 3.3256 | 3.0931 | 3.0590 |
| kodim05 | 4.0119 | 3.7394 | 3.7176 |
| kodim07 | 3.1530 | 2.8630 | 2.8060 |
| kodim13 | 4.2378 | 4.1010 | 4.0879 |
| kodim19 | 3.5134 | 3.2758 | 3.2522 |
| kodim23 | 3.1880 | 2.8940 | 2.8636 |
| **AVG** | **3.5782 PASS (±3% of 3.58)** | **3.3429 PASS** | **3.3124 (champ hillclimb 3.309)** |

CROWN-huff groups reproduce task list exactly (Y(6,6) Co(36,25) Cg(36,24) etc.).

## 2. WINNERS

### 2a. BEST non-rANS (productizable, no ANS decoder): E16 + Huff-vs-Golomb 2-way + WIDE-K — `probe_b5_d.py`

E16 bank (16 experts, 4b predid): MED,TOP,LEFT,PAETH,GRAD,GAP80,GAP32,GAP16,DG,AVG_AB,PLANE,AC,C,BC,D,AVG3.
Per-group joint best (expert × backend), 1b backend-choice/group, Golomb k 4b/group. K auto per-channel from WIDE set (2,3,4,6,9,12,18,27,36,48,64) on exact final cost.

| image | bpp | Δ vs 3.309 avg | m6 ref | Δ vs m6 |
|---|---|---|---|---|
| kodim01 | 3.4081 | +0.0991 | 3.39 | +0.0181 |
| kodim02 | 3.0548 | −0.2542 | 3.06 | −0.0052 |
| kodim05 | 3.6856 | +0.3766 | 3.77 | −0.0844 |
| kodim07 | 2.8051 | −0.5039 | 2.81 | −0.0049 |
| kodim13 | 4.0435 | +0.7345 | 4.05 | −0.0065 |
| kodim19 | 3.2441 | −0.0649 | 3.27 | −0.0259 |
| kodim23 | 2.8440 | −0.4650 | 2.86 | −0.0160 |
| **AVG** | **3.2979** | **−0.0111 (−0.34%)** | 3.3157 (pin 3.32) | **−0.0178 (−0.54%) / −0.0221 vs 3.32** |

Groups: 01:(64,56)(64,33)(64,30) 02:(27,25)(48,35)(64,29) 05:(64,56)(64,45)(64,38) 07:(64,48)(64,35)(64,29) 13:(64,56)(64,45)(64,38) 19:(4,4)(64,39)(64,27) 23:(64,48)(64,36)(64,31). 6W-1L vs m6.

### 2b. BEST overall: E16 + 3-way (Huff/Golomb/rANS, 2b choice) on WIDE best-K — `probe_b5_d.py` (rANS decode-verified, recon PASS 7/7)

Same streams as 2a, per-group rANS check on winning expert only (Huffman-rank ≈ rANS-rank; 1 rANS/group, not 16). Honest 2b choice/group.

| image | bpp | Δ vs 3.309 avg | m6 ref | Δ vs m6 |
|---|---|---|---|---|
| kodim01 | 3.3787 | +0.0697 | 3.39 | −0.0113 |
| kodim02 | 3.0271 | −0.2819 | 3.06 | −0.0329 |
| kodim05 | 3.6682 | +0.3592 | 3.77 | −0.1018 |
| kodim07 | 2.7564 | −0.5526 | 2.81 | −0.0536 |
| kodim13 | 4.0353 | +0.7263 | 4.05 | −0.0147 |
| kodim19 | 3.2217 | −0.0873 | 3.27 | −0.0483 |
| kodim23 | 2.8167 | −0.4923 | 2.86 | −0.0433 |
| **AVG** | **3.2720** | **−0.0370 (−1.12%)** | 3.3157 (pin 3.32) | **−0.0437 (−1.32%) / −0.0480 vs 3.32** |

Backends per image (H/G/R groups): 01:5/58/56 02:10/29/50 05:7/91/41 07:9/51/52 13:11/99/29 19:9/17/44 23:10/57/48. 7W-0L vs m6, Wilcoxon W=0 p≈.016. RECON PASS 7/7 (scalar decoder sim with E16 formulas), Huffman round-trip every H-stream, rANS decode every R-stream.

## 3. Exploration trail (avg bpp, Δ vs CROWN-huff 3.3429)

| idea | avg | Δ | fate |
|---|---|---|---|
| E6 joint-K Huff (re-tuned K on exact best-pred cost) | 3.3416 | −0.0013 | KEPT as corrected baseline (`probe_b5_b/c`) |
| (a) E6 + Huff-vs-Golomb 2-way, KSET2 (1b choice+4b k) | 3.3236 | −0.0193 | KEPT — Golomb wins 22–54/74–89 groups/img; tiny-A saves table; K→36 everywhere (no-table favors fine) |
| (b) E12 Huff-only KSET2 (4b predid: +LEFT,AVG_AB,PLANE,GAP32,AC,C) | 3.3297 | −0.0132 | KEPT — LEFT 49, PLANE 30, AVG_AB 25 picks steal from MED (296→225) |
| (b) E16 Huff-only KSET2 (+BC,D,GAP16,AVG3, same 4b) | 3.3287 | −0.0142 | KEPT — GAP16 26 picks, +0.001 over E12 free (same side) |
| (a+b) E16 + 2-way KSET2 | 3.3088 | −0.0341 | KEPT — beats 3.309 by 0.0002; additivity ~holds (−0.019 + −0.014) |
| (a+b) E16 + 2-way WIDE-K (48,64; same 6b map) | 3.2979 | −0.0450 | **WINNER non-rANS** — further −0.011; K=64 picked 11/21 channels (Golomb fragmentation economics) |
| E12 + 2-way WIDE-K | 3.2991 | −0.0438 | killed vs E16 (−0.0012, keep E16, same side) |
| (a+b)+rANS 3-way refine on WIDE (2b choice) | 3.2720 | −0.0709 | **WINNER overall** — further −0.026; R wins 29–56 groups/img |

Breakdown notes (`probe_b5_b.py`): Golomb-winners not only tiny-A (sample A: 14–89); saves 13–118kb/img vs extra on H-groups; net win because table 16+A*24 eliminated. E-bank picks (`probe_b5_c.py` total 401 groups E16): MED 221, LEFT 50, PLANE 29, AVG_AB 25, GAP16 26, TOP 20, rest <10; GRAD ~dead (1), DG 5 — GAP/AVG/PLANE cover edges better.

Killed/not-tried: separate per-stream (not per-group) backend (strictly coarser than per-group, dominated); adaptive Golomb-k per context (same as per-group best-k, already exact); DP-optimal partition (greedy equal-pixel overshoot noted; WIDE-K already harvests most; left as follow-up ~0.005); Lloyd re-cluster (probe_b4: −0.001, dropped).

## 4. Verdict on Boss 3 (WebP-m6 at 3.32)

- **BOSS 3 DEAD both arms.** Non-rANS 3.2979 < 3.32 (−0.67%) with 6W-1L vs m6 per-image (only kodim01 loses by +0.018). rANS 3.2720 < 3.32 (−1.45%) with **7W-0L**, W=0 p≈.016.
- Aligned-margins ask (≈−0.5%): non-rANS −0.54% vs m6-mean, rANS −1.32%. Both meet; rANS doubles it.
- Honesty: all sides counted (map ceil(log2K), predid ceil(log2E), 1b/2b backend choice, 4b Golomb-k, Huff/rANS tables, 64b header); Huffman round-trip + rANS decode + scalar recon proofs all PASS.

## 5. Follow-up (ordered)

1. Productize non-rANS winner in C (`hapre.c` + `driver_crown.py`): add 10 new predictors (LEFT/AVG_AB/PLANE/GAP32/GAP16/AC/C/BC/D/AVG3), widen group tables to 64, per-group 1b backend + 4b k + Rice codec; streaming decode mirrors `probe_b5_d.prove_recon`.
2. DP-optimal equal-cost partition (exact rate incl. map+predid+backend sides) — greedy leaves uneven groups (e.g. 56 eff. of K=64); expect +0.005.
3. True joint 3-way K-selection (currently K picked on 2-way then rANS-refined; rANS tables favor coarser K — re-select per-channel under 3-way cost).
4. Per-group sign-flip on/off bit (currently always flip; flat groups may prefer no-flip + Golomb).

## Files

- `experiments/probe_b5_a.py` — Stage-1 anchors (PASS).
- `experiments/probe_b5_b.py` — backend 2-way + breakdown.
- `experiments/probe_b5_c.py` — E12/E16 bank bake-off (predictors_X/prepX shared by d).
- `experiments/probe_b5_d.py` — COMBINED WIDE 2-way/3-way + `prove_recon` (scalar E16 decoder sim).
- `experiments/probe_b5_e.py` — E12 vs E16 WIDE ablation.
- This report: `experiments/probe_b5_RESULTS.md`.

Sources: probe_b4 §4 trail + §6 follow-up (Golomb-vs-Huffman bit idea, DP-partition idea); LOCO-I/JPEG-LS report HPL-98-193 (Rice mapping, sign-flip, per-ctx k); CALIC GAP thresholds (motivated GAP16/32 variants); no new web search (idea space from campaign memory).
