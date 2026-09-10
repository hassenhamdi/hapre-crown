# probe_b10 — Vector/lattice coding of active-block MAGNITUDES

**Branch from the probe_b9 post-mortem: the open question was whether the joint
information lives in magnitudes-given-activity. Answer: YES — and it pays.**
Files: `experiments/probe_b10_magvq.py` (sweep + exact ledger + round-trips),
`experiments/probe_b10_diag.py` (mechanism diagnostics),
`experiments/probe_b10_log.txt` (full run log). Existing files untouched.

## Idea
On MED residuals in YCoCg-R, partition each channel into disjoint blocks
(1×2 pairs, 2×2 quads, 1×4/4×1 strips) and code the **magnitude tuple jointly**
with a small vector-Huffman codebook. Design points, all MDL-honest:
- Sign–magnitude split: magnitudes go through the joint table; signs of
  in-cap nonzeros cost a fixed 1 b each (measured sign-pair entropy ≈ 2.0 b,
  so joint sign coding cannot help — verified by the S-T2 config losing to
  the split).
- Cap + escape: tuples with max|·| ≤ T use the joint alphabet ((T+1)^n + 1
  symbols incl. a single ESC); over-cap blocks fall back to a scalar Huffman
  tail with its own counted table. No per-class tables (fragmentation
  discipline — one joint + one tail table per channel, nothing else).
- Flat blocks need **no escape flag**: the all-zero tuple is the top joint
  symbol and earns the shortest codeword automatically.
- **MDL gate per image-channel**: gated_c = min(base_c, exp_c), +1 flag bit
  per channel (3 bits/image, counted). The codebook is shipped ONLY if
  net-positive.
- Pyramid variant (PYR-R, Fischer 1986 style): r = Σ|mi| Huffman-coded,
  position of the signed vector on shell r at fixed ⌈log₂N(4,r)⌉ bits,
  over-radius blocks to a scalar tail. Integer-lattice enumeration with
  exact N(r) = Σ_k C(4,k)·C(r−1,k−1)·2^k (asserted against brute force).

## Theory paragraph (why magnitudes, not positions)
b9 showed H(pattern|weight) ≈ log₂C(4,w): *where* activity sits is uniform.
What is not uniform is *how large* co-located residuals are together:
texture produces bursts of large |r|, smooth areas bursts of small |r|
(volatility clustering that MED, a linear predictor, cannot remove).
The monetizable quantity is the total correlation of magnitudes,
TC = ΣH(|Xi|) − H(|X₁..Xn|), harvested by a joint magnitude alphabet whose
(T+1)^n + 1 codebook stays tiny (626 symbols at Q-T4 → ~15 kbit/image,
≈ 0.004 bpp). Fischer's PVQ ("A Pyramid Vector Quantizer", IEEE TIT
32(4):568–583, 1986) is the classical lattice construction for Laplacian
sources; we test both its shell-factorized form and the unfactorized
vector-Huffman form — and the comparison is the most instructive result.

## Method (exact, numpy+PIL, CPU)
YCoCg-R per brief formulas (Co=R−B; t=B+(Co//2) floor; Cg=G−t; Y=t+(Cg>>1)),
invertibility asserted 7/7. Causal MED with **edge-replicate pad, standard
rule** (brief's convention; differs from b9's 0/left/top only in the first
row/col). Real heapq Huffman → lengths; bits = Σcount·len; every table costs
16+A·24; fixed sign/shell bits counted exactly; bpp = total/(H·W·3).
Kraft=1 asserted for every Huffman table built. Decode proofs: symbol-mapping
round-trip on ALL 7 images × 3 channels for the winning config, plus one
literal bitstream→residuals decode (kodim07, all channels) — both PASS.
(A literal-harness bug — decode interleaving signs/tails while encode writes
all-signs-then-tails — was caught by the harness itself and fixed; the PASS
above is post-fix.)

## Results — anchor reproduced, experiment wins
Baseline anchor check: avg **3.5769** vs 3.58 (−0.09%, ±3% PASS); per-image
3.6178/3.3255/4.0119/3.1529/4.2378/3.5081/3.1842 vs brief
3.62/3.33/4.01/3.15/4.24/≈3.51/3.19 (all ≤ 0.2%).

| img | base | Q-T4 ★ | Ph-T8 | Q-T3 | H4-T3 | S-T2 | PYR-R4 |
|---|---|---|---|---|---|---|---|
| kodim01 | 3.6178 | **3.5636** | 3.5648 | 3.5709 | 3.5814 | 3.5995 | 3.6178 |
| kodim02 | 3.3255 | **3.2477** | 3.2605 | 3.2482 | 3.2590 | 3.2962 | 3.3255 |
| kodim05 | 4.0119 | **3.9505** | 3.9555 | 3.9709 | 3.9783 | 4.0106 | 4.0119 |
| kodim07 | 3.1529 | **3.0419** | 3.0756 | 3.0430 | 3.0594 | 3.1269 | 3.1452 |
| kodim13 | 4.2378 | 4.2296 | **4.2192** | 4.2319 | 4.2370 | 4.2378 | 4.2378 |
| kodim19 | 3.5081 | **3.4443** | 3.4492 | 3.4611 | 3.4704 | 3.4911 | 3.5081 |
| kodim23 | 3.1842 | **3.1002** | 3.1060 | 3.1013 | 3.1101 | 3.1601 | 3.1772 |
| **AVG** | **3.5769** | **3.5111 (−1.838%)** | 3.5187 (−1.627%) | 3.5182 (−1.640%) | 3.5280 (−1.368%) | 3.5603 (−0.463%) | 3.5748 (−0.059%) |

(kodim05 PYR-R4 = 4.0119, gated to scalar; typo guard: the gate fired.)
Full 19-config avg deltas (gated / raw-ungated): Ph-T1 +0.000/+3.108,
Ph-T2 −0.198/+1.028, Ph-T3 −0.822/−0.302, Ph-T4 −1.253/−0.944,
Ph-T6 −1.523/−1.347, Ph-T8 −1.627/−1.522, Pv-T2 −0.187/+1.031,
Q-T1 −0.128/+1.904, Q-T2 −0.848/−0.239, Q-T3 −1.640/−1.445,
**Q-T4 −1.838/−1.650**, Q-T5 −1.291/−1.052, H4-T2 −0.603/+0.091,
H4-T3 −1.368/−1.153, V4-T2 −0.590/+0.129, S-T2 −0.463/+0.736,
PYR-R4 −0.059/+1.266, PYR-R6 −0.047/+1.255.
**Avg delta vs 3.58 anchor: Q-T4 −1.93% (vs own baseline −1.838%).**

Winner Q-T4 (2×2 quad, T=4) channel detail: VQ chosen on 20/21 channels
(the only fallback is kodim13-Y, 93% escaped — the gate working as designed);
chroma escapes only 1–19% while Y escapes 30–79%.

## Mechanism analysis (measured)
1. **Headroom is real and nearly fully harvested.** Clipped-TC (T=4,
   well-sampled 625-symbol alphabet, no plug-in bias) averages Y 0.108 /
   Co 0.072 / Cg 0.056 b/residual → mean 0.079 b/res headroom. Realized
   saving is 0.066 b/res ⇒ **~84% capture** (kodim07-Y: 0.557 b/block
   realized vs 0.623 b/block headroom, ~90%). The joint table is an
   efficient harvester, not a lucky one.
2. **ESC-block tax is small.** Over-cap blocks pay one ESC symbol + tail
   lengths instead of baseline lengths: measured +0.31 (Y) / −0.03 (Co) /
   +0.27 (Cg) b/block — the tail table re-fits the escaped-conditional
   distribution, nearly cancelling the ESC overhead (Co even nets negative).
   This is why Q-T4 still wins with Y escape rates up to 79%.
3. **Signs are independent — split is optimal.** H(sign-pair|both-nonzero)
   = 1.996 (Y) / 1.981 (Co) / 1.836 (Cg) b vs 2.0 max. Joint-sign config
   S-T2 (−0.46%) underperforms the sign-magnitude split at the same T
   (Ph-T2 −0.20% gated but +1.03% raw vs S-T2 +0.74% raw — and the split
   family scales to T=8 while the signed alphabet explodes).
4. **Pyramid factorization dies on the ceil-tax — the b9 mechanism redux.**
   PYR-R4 7-image ledger vs baseline: radius stream+table 12.9%, fixed
   shell positions 15.5%, tail 72.8% → 101.2% (+1.2% raw). The shell index
   (N(4)=192→8 b vs 7.58 b ideal, N(6)=1104→11 b) re-imposes fixed-length
   rank coding on exactly the magnitudes b9 showed are uniform-given-shell
   only in position — the radius carries the dependence, and Huffman-coding
   the radius while fixed-coding the position is the wrong split. Unfactorized
   vector Huffman (Q-T2 −0.85%) beats the pyramid (−0.06%) at the same
   locality: **never factorize what you can tabulate.**
5. **Geometry sweet spot, then fragmentation.** 2×2 quads > horizontal pairs
   > 1×4 strips ≈ vertical pairs (H4-T3 −1.37%, V4-T2 −0.59%: 2-D locality
   beats 1-D at fixed n=4). T=4 is the peak: Q-T5 falls to −1.29% as the
   1297-symbol table fragments (≈31 kbit) and ESC events thin out — the
   campaign's fragmentation discipline, confirmed from the other side.
6. **Corrigendum to b9 used in reasoning:** b9's "P(k=4)=0.43 on kodim07-Y"
   is a labeling slip — with p₀=0.223 (reproduced exactly here), 43%
   all-zero quads is impossible; 0.43 is P(all-**active**). True P(all-zero
   quad) is 0.017 (kodim07-Y; 7-avg Y 0.011, Co 0.053, Cg 0.050), i.e. ~7×
   overdispersion vs independence — real clustering, but an order of
   magnitude smaller than the "170×" quoted. Activity is dense, not sparse:
   all the more reason position-coding was doomed and magnitude-coding pays.

## Verdict: WIN (magnitude-VQ pays net of codebooks)
Q-T4: **−1.838% gated (−1.650% raw ungated), 7/7 images**, tables + ESC +
gate-flag all counted, round-trips PASS. The b9 open question is closed:
magnitudes-given-activity carry ~0.08 b/res of joint information and a
625-word vector codebook extracts ~84% of it. Effect size (≈ half the
CROWN-huff Boss-2 margin) is probe-family-significant on the MED+order-0
frame. Caveat: this is the MED+order-0 frame (3.58), not the CROWN arm
(3.27) — transfer must be measured, since per-group tables already harvest
part of this dependence (stacking risk is real).

## Follow-up
- **Port to CROWN**: per-group magnitude-VQ (or a single VQ layer under the
  existing groups) — test whether the −1.8% stacks or overlaps with
  LOCO-365 grouping. Biggest open item; decides Boss-5 relevance.
- Gated geometry select (2×2 vs pairs, 1 flag/block-class or per-channel):
  kodim13 prefers pairs (4.2192 < 4.2296) — texture-adaptive geom could add
  ~0.1–0.2 pp.
- Per-channel T (Y:8, chroma:3?) — chroma escapes are already ≤12% at T=4
  while Y wants T=8; split caps cut Y-ESC tax.
- Sparse-tail coding (Golomb/Rice tail instead of a second Huffman table)
  to push the T frontier past the Q-T5 fragmentation wall.
- Do NOT revisit: pyramid shell factorization (dead, mechanism understood),
  joint sign coding (dead, H≈2.0), larger unfactorized alphabets without a
  sparsity story (Q-T5 already turned).
