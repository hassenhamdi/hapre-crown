# probe_b9 — Enumerative (combinatorial) coding of 2×2 MED-residual blocks

**Discipline: information theory (Cover'73), never tried in this campaign.**
Files: `experiments/probe_b9_enum22.py` (new, existing files untouched).

## Idea
Replace per-symbol Huffman on MED residuals with a block-joint code on disjoint
2×2 blocks (4 residuals): encode `k` = #zeros per block with a tiny Huffman
table on {0..4}; encode the rank of the nonzero-position mask among C(4,4−k)
at fixed length ⌈log₂C(4,w)⌉ (w=1:2b, w=2:3b, w=3:2b, w∈{0,4}:0b); encode the
nonzero values (fixed position order) with a shared order-0 Huffman table.
This is Cover's enumerative scheme (weight class + lexicographic index, n=4,
Schalkwijk-style binomial indexing), with the value alphabet handled by Huffman.

## Theory paragraph (why it could beat the 97%-efficiency wall)
Per-symbol Huffman is blind to JOINT structure: it pays ΣH(Xᵢ) while the truth
is H(X₁..X₄) = ΣH(Xᵢ) − TC with total correlation TC ≥ 0. MED residuals are
*bursty*: zeros cluster in smooth regions (measured P(k=4)=0.43 on kodim07-Y
vs 0.0025 under independence — 170× overdispersion), so the 4th-order joint
deviates violently from the product distribution. Enumerative coding harvests
exactly this: the weight stream H(K)/4 + rank stream E[logC(4,W)]/4 replaces
4·h₂(p₀) position costs, needing only two extra micro-tables (k: ≤5 symbols;
patterns: zero table, fixed-length) — MDL-clean, zero side info beyond
16+A·24 header bits per stream, fully causal-decodable.

## Exploration trail
- Read cycle1-memory.md + RESULTS.md: per-symbol context game ~exhausted
  (CROWN2 3.2686, MA-tree +0.56%, pair-alphabet dies under grouping,
  bias adapters Huffman-invariant). Joint-across-pixels coding unexplored
  (runs are 1-D; chroma-pair alphabet is value-domain, order-0 frame).
- Web: Cover T.M., "Enumerative source encoding", IEEE TIT-19(1):73–77, 1973
  (verified original: index-by-weight + inverse algorithm; Markov-statistics
  extension) + Ryabko 2024 survey of Cover-code applications (confirms
  weight-class + rank factorization is the canonical use). Dai & Zakhor,
  "Binary combinatorial coding", DCC 2003 (precedent for compression use).
- Considered and rejected: (a) free segmentation — duplicates LOCO-365/CTX9
  conditioning already in CROWN; (c) cross-scale histograms — transmitted-LF
  family retired (side 0.40 bpp structural); (d) decision list — MA-tree-lite
  already measured it (−0.56%, 1/3 of need); (e) adaptive lifting — DCT probe
  6.95 bpp showed transforms need full EBCOT build, out of probe scope.
- Chose (b): cheapest joint-structure probe, exact-countable, MDL-safe.

## Method (exact, numpy+PIL, CPU)
YCoCg-R per brief formulas (invertibility asserted 7/7); whole-image
vectorized causal MED with 0/left/top borders; real heapq Huffman →
canonical lengths, bits = Σcount·len, table = 16+A·24 per stream;
bpp = total_bits/(H·W·3). Decode proofs: enumerative mapping round-trip
asserted 7/7; Kraft=1 asserted for all 42 Huffman tables; one literal
bitstream→residuals decode (kodim07, all streams) PASS.

## Results — anchor reproduced, experiment honest
Baseline anchor check: avg **3.5782** vs 3.58 (−0.05%, ±3% PASS); per-image
3.62/3.33/4.01/3.15/4.24/3.51/3.19 all match brief.

| img | base bpp | enum bpp | Δ% | p₀ Y/Co/Cg |
|---|---|---|---|---|
| kodim01 | 3.6179 | 3.6173 | −0.02 | 0.104/0.387/0.356 |
| kodim02 | 3.3255 | 3.3112 | −0.43 | 0.176/0.353/0.356 |
| kodim05 | 4.0119 | 4.0359 | +0.60 | 0.108/0.297/0.298 |
| kodim07 | 3.1529 | 3.1421 | −0.34 | 0.223/0.415/0.415 |
| kodim13 | 4.2378 | 4.2926 | +1.29 | 0.058/0.277/0.260 |
| kodim19 | 3.5133 | 3.5436 | +0.86 | 0.117/0.327/0.321 |
| kodim23 | 3.1879 | 3.1854 | −0.08 | 0.205/0.373/0.361 |
| **AVG** | **3.5782** | **3.5897** | **+0.32 (LOSS)** | — |

**Avg delta vs 3.58 anchor: enum +0.27% (vs own baseline +0.32%).**

## Why it died (mechanism, measured — not hand-waving)
1. **Positions are uniform given weight.** Empirical H(pattern|w) =
   1.997/2.583/2.000 vs log₂C(4,w) = 2.0/2.585/2.0 — ranks carry ~zero
   residual structure; the factorization's only harvest is raw sparsity.
2. **"Given nonzero" inflates value cost.** Removing zeros raises value
   entropy (kodim07-Y: Hz=4.131 > H₀=3.974); the position savings and the
   value inflation nearly cancel by construction.
3. **Ideal headroom is ~0.5%, taxes eat it.** Infinite-precision ledger
   (empirical H(K), H(pattern|W), H(V|nz)): headroom only +0.18..+1.02%
   per image, avg +0.53%. Realization taxes — ⌈·⌉ ceil-tax on w=2 ranks
   (3b vs 2.585b, most-frequent mixed class), Huffman integer-length slack
   ×2 tables, extra k-table headers — total ~0.85pp, flipping +0.5% ideal
   to +0.32% measured loss. Even a perfect implementation caps at ~0.5%:
   2.6× short of the Boss-5 gap (+1.22% needed vs JXL-e3 3.2291).
4. Correlated failure mode: worst on textured images (13: +1.29, 19: +0.86)
   where p₀ is lowest — the scheme's harvest vanishes exactly where bits live.

## Verdict: BEAUTIFUL DEAD END (retire the family)
The joint structure is real (170× zero-clustering) but its *monetizable*
fraction under any weight+rank factorization is ≤1%: the information lives
in value-magnitudes given activity, not in activity patterns. No larger block
(n=9/16) fixes this — rank alphabet 2ⁿ explodes while H(pattern|w) stays
≈logC(n,w); the ceil-tax and the Hz>H₀ inflation both grow with n.

## Follow-up (if camp ever revisits joint coding)
- Don't code *positions* of activity — code *magnitudes jointly*: the open
  question is E[log P(v₁..v₄|all active)] vs Σlog P(vᵢ|active); needs
  vector-Huffman/lattice VQ with MDL-gated codebooks (bigger arc).
- Bitplane-significance route (EBCOT-style: separate sparse significance
  from ~50/50 refinement) attacks the same burstiness from the magnitude
  side; this probe's H(pattern|w)≈logC measurement says significance maps
  are near-uniform too — budget skepticism accordingly.
- Reusable artifact: `probe_b9_enum22.py`'s exact bit-ledger + literal
  round-trip harness for any future residual-recoding probe.
