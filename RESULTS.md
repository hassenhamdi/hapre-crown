# HAPRE-C Campaign Report v3 — 2026-09-07 — Boss Ladder + µMoE Champion
Env: Python 3.14.2 / PIL 12.2.0 / numpy 2.4.6 / cjxl-djxl 0.11.2 / gcc -O3. Kodak 7×768×512 md5-pinned.
All bpp = EXACT stream bytes. All claims Wilcoxon paired (n=7). Round-trips byte-asserted.

## Champions (exact, round-trip PASS 7/7)
| CROWN-rans-hc (LOCO365+autoK-hillclimb+per-group-best6+rANS) | 3.309 | ~0.04* | BOSS-2 KILLER |
| codec | avg bpp | enc MP/s | status |
| HAPRE-C MOE (µMoE-8×8×6 + run + CTX9 Huff) | 3.464 | ~2.9 | CURRENT CHAMPION |
| HAPRE-C RUN (MED + run + CTX9 Huff) | 3.511 | ~7 | Pareto speed axis |
| HAPRE-C CTX9 (MED + CTX9 Huff) | 3.537 | ~8 | — |
| HAPRE-C 0-ctx (MED + Huff) | 3.58 | 9.76 | Pareto speed record |

## Boss board
| Boss | bpp | vs MOE 3.464 | stats | verdict |
| PNG-9 (4.75) | 4.75 | −27% | W=0 p=.016 7/7 | ☠ DEAD |
| JXL-e1 (3.72) | 3.72 | −7% | W=0 p=.016 7/7 | ☠ DEAD |
| WebP-m0 (3.60) | 3.60 | −3.8% | W=0 p=.031 6/6+1T | ☠ DEAD (RUN config) |
| WebP-m3 (3.347) | 3.347 | −1.1% | W=0 p=.016, 7/7 (CROWN-rans-hc 3.309) | ☠ DEAD |
| WebP-m6 (3.32) | 3.32 | −0.3% | W=10 p=.58 (3W-4L, tie) | STANDING (tie) |
| JXL-e3 (3.23) | 3.23 | +2.6% | W=1 p=.031 (we lose) | STANDING |
| JXL-e9 (3.03) | 3.03 | +9% | — | FINAL BOSS |

## MOE per-image (exact): 01:3.56 02:3.23 05:3.84 07:3.01 13:4.17 19:3.41 23:3.03
GAP/diagonal experts earn their keep (selection shares 15-20% each). ids 3b-packed (6.9KB).

## Dead-ends log (with evidence)
- LPC-3/flat-LS/I RLS-L1 linear prediction: tie MED (±0.5%) — linear wall; L2-optimal HURTS bits (Gaussianization).
- Lag-2 autocorrelation (−0.28) is real but not linearly exploitable (FIR-12 +bit-gate: +0.0%).
- rANS-ch (M=14): ties Huffman (both order-0). M=10 lesson: precision must cover alphabet (232 syms need M≥14).
- Adaptive bias (LOCO-I-lite): −0.2% (dithering adds entropy under Huffman) — reverted.
- Cross-channel residual features: |corr|<0.11 — skipped by probe.
- Wavelet+zlib: 4.78 (zlib can't use 2D structure).
- LZ77-on-residual-bytes: 4.18 (destroys symbol structure; WebP's LZ works on pixels+transforms, not residual bytes).
- Naive 16-bin LF conditioning: −8..−12% (table fragmentation) — needs clustering; weapon parked.
- Residual clipping ±256: silent-corruption landmine found+fixed (alphabet ±1024, proven bounds for 8-bit RGB).

## Next weapons vs Boss 2 (needs −3.3%), ranked by probed EV
1. LF-clustered conditioning (+1–3%?): 4-bin LF with shared/clustered tables — probe was naive; proper design pending.
2. Band-adaptive tables (+0.5–1%?): 4–8 row-bands, nonstationarity capture.
3. rANS fine (+0.3–0.5%): recover Huffman integer-length overhead.
4. Transform-domain (Boss 5/6 territory): integer DCT/Squeeze subbands + context coding — the actual JXL weapons; new arc.