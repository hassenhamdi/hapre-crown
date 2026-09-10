---
name: eagle-eye-debugging
description: Use when a codec, compressor, or bit-exact pipeline produces valid output with wrong size, passes round-trips yet mismatches a reference, or behaves differently on easy vs hard inputs
---

# Eagle-Eye Debugging

## Overview
Valid output is not correct output. This method hunts bugs that survive round-trips,
small tests, and code review — using arithmetic impossibility, forced-variant testing,
and distrust of coincidence. Distilled from a real 18-turn hunt (predictions packed as
residuals, masked four ways).

## The core loop
1. **Do the math first.** Convert the symptom to required numbers (e.g. 57 KB over 6144
   symbols at k≤12 and |v|≤1024 ⇒ avg|v|≈73 — so VALUES are wrong, not tables/k/counts).
   Kill theories by arithmetic, not by reading more code.
2. **Instrument for measurement, not logging.** Env-gated counters (section bytes,
   per-group k/count, value magnitudes) with zero behavior change. One rebuild, one run.
3. **Force every variant through decode on HARD inputs.** Unselected code paths
   (losing codec variants, fallback arms) are never validated — force-select each and
   decode it. Flat/small inputs make all experts coincide and prove nothing.
4. **Bisect by construction.** Force-MED-only (or equivalent trivial mode): if output
   turns sane, the bug is in selection/assignment; if still broken, it's in values/packing.
5. **Distrust self-consistency.** Encoder/decoder sharing a flaw passes round-trips.
   Cross-check against an INDEPENDENT implementation (reference encoder, numpy model).
6. **Distrust coincidence.** Same bytes on easy input ≠ same logic (flat content hides
   expert/border/predictor bugs). Verify on texture; verify every expert engages.

## Red flags — stop theorizing, measure
| Thought | Reality (from the case file) |
|---|---|
| "Round-trip passes, so it's correct" | Shared flaws pass; only proves self-consistency |
| "Matches reference on the small test" | Flat inputs make all experts coincide |
| "Same function, so same values" | Check call sites: search-vs-pack duality (cost on X, pack on Y) |
| "k/tables/counts look sane" | Then values are huge — measure magnitudes directly |
| "Decoder mirrors encoder" | Verify: forced-variant decode of YOUR bytes, not just final output |
| "One more code read will find it" | After 3 failed reads, instrument instead |

## Bug taxonomy (this codebase's greatest hits)
causality peeks at undecoded pixels · L=0 framing tokens · alphabet clipping
(±1024 assert, never clip) · rANS precision vs alphabet size · border-convention
mismatch · unary polarity wire-splits · tanh 1-ulp divergence (frozen LUT) ·
predictions-packed-as-residuals (the classic: search subtracts, assembly forgets) ·
search-vs-pack array mismatch · stale planes across loop iterations.

## Bring-up checklist (new codec/arm)
- [ ] Every variant forced-selected AND decoded on textured (never flat-only) input
- [ ] Byte-compare vs independent reference, not just round-trip
- [ ] Magnitude audit: packed values match predicted distribution scale
- [ ] k/table/counts cross-checked between search-time and pack-time
- [ ] Framing asserts per channel/stream; loud asserts on ranges, never silent clips
