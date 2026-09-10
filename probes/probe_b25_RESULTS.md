# probe_b25 RESULTS — palette oracle: DEAD on photos (+ analytic #1/#2 kill)

Branch question (FLIF quest, teardown ports one-by-one): kodim02 carries 35%
of the FLIF gap (+0.5573 bpp, 13.5k colors) — does a top-N palette + escapes
front end take it? Second: do teardown ideas #1 (Y-conditional ranges) / #2
(alphabet elision) transfer to our cost model?

## 0. Verdict

**Palette DEAD: oracle loses on all 7 images** (best case 02: +105 KB vs
CROWN6, +188 KB vs the FLIF bar; worst 23: +507 KB). Near-uniform indices
over thousands of symbols (~12 b/px) + N·24b palette side + raw escapes
cannot beat predictive residuals (~9 b/px) on photos. Matches FLIF's own
design (palette vetoes above 512 colors). **#1/#2 KILLED ANALYTICALLY
(Δ≡0, no code run):** every backend cost in our model depends only on
OCCURRING symbols (Huffman tables/codes over occurring counts, Golomb
lengths/k over occurring M, rANS alphabet/freq over occurring) — Y-restriction
removes only zero-count symbols, so tables, codes, k/d, headers AND the
search argmins are all unchanged. #1/#2 assume range-declared bit-plane
coding (FLIF) and do not transfer to explicit-occurrence cost models.

## 1. Oracle numbers (bytes; flag-free ESC-as-symbol design, exact Huffman)

| img | U | bestN | oracle | vs CROWN6 | vs FLIF bar | hit |
|---|---|---|---|---|---|---|
| 01 | 19182 | 8192 | 676635 | +190289 | +201057 | .95 |
| 02 | 13452 | 4096 | 545772 | +105553 | +187731 | .96 |
| 05 | 63558 | 8192 | 864665 | +336684 | +366895 | .70 |
| 07 | 37552 | 8192 | 708101 | +312943 | +351402 | .86 |
| 13 | 39784 | 16384 | 836410 | +253481 | +290659 | .89 |
| 19 | 24807 | 8192 | 725002 | +260371 | +278905 | .90 |
| 23 | 72079 | 16384 | 910237 | +506729 | +524693 | .76 |

Gate (oracle(02) < FLIF-02 358,041): FAIL by 188 KB. Design counted:
palette N·24 + Huffman stream (16+A·24 table, A incl. ESC) + ESC raw 24b +
1b gate flag. Correctness note: early run had an Npx bug (negative bytes);
fixed via return_inverse vectorization; rerun full-7 clean.

## 2. Mechanism + the 02 insight (for the record)

Photos carry 13–72k colors; the index stream's near-uniform thousands-symbol
alphabet costs more than the residuals it replaces. FLIF beats us on 02
WITHOUT its palette (13.5k > its 512 cap) — via smooth-field machinery, not
color counting. The residual 02 analysis points at sub-1-bit zeros
(arithmetic territory), but CROWN5 already tested per-channel best-of
{A-act binary range coder, C-gol} at exact level with a real C decoder
(`src/libcrown5.so`) and static won 21/21 — the coder alone is not the
door; FLIF's combination (partitions + per-bit states + conditional ranges)
is. No cheap path remains on this axis.

## 3. Files

- `probes/probe_b25_palette.py` (oracle + N-sweep), `probes/probe_b25_nums.json`.
  Symlinked in `experiments/`.
