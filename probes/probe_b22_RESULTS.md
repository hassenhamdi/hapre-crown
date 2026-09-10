# probe_b22 RESULTS — adaptive-Golomb transfer on the CROWN6 frame: KILLED

Branch question (Boss-5 quest): b19 proved backward-adaptive coders (−2.18%
MED-frame) with transfer math predicting a JXL-e3 KO, but its baseline was
per-PLANE static-k. CROWN6 already uses per-GROUP best-k + G-bias. How much
transferable delta remains? Measured on the encoder's own winning groups
(`CROWN_DUMPGROUPS` dumps, fidelity-gated) for both blockers.

## 0. Verdict

**KILLED — residual adaptive delta ≈ 0 (05) to +0.4% of gap (13).**
Per-group N/A/k adaptation LOSES to static best-k on 05 (−0.2..−1.1%:
cold-start + missing bias adapter outweigh tracking); global pooling loses
badly everywhere (−2..−6.6%, b18 whiplash re-confirmed). Chosen static-Golomb
additionally BEATS the H0-with-real-tables counterfactual by 117kb (05-Y) /
181kb (13-Y): entropy coding on texture G-groups is EXHAUSTED. The b19 transfer
estimate's overlap with per-group best-k is ~100%. Boss-5 must come from the
PREDICTION axis (or stand). No C wire-format change is warranted.

## 1. Numbers (exact bits; static verified k/d-identical to encoder, 0 mismatches)

Gaps are authoritative BYTE gaps vs JXL-e3 sizes (JXL bpp×147456):
05 +10,115 B (80,920 b; ours 527,981 vs JXL 517,866), 13 +5,770 B (46,160 b).

kodim05 [C27] (all-Q winner; Y ch0: 55/56 groups Golomb, 385k syms):

| ch | static-G | (a1) per-grp 1ctx | (a4) per-grp 4ctx | (b1) global 1ctx | (b4) global 4ctx |
|---|---|---|---|---|---|
| 0 | 1,970,463 | 1,974,976 (−0.23%) | 1,972,948 (−0.13%) | 2,100,799 (−6.61%) | 2,011,190 (−2.07%) |
| 1 | 394,921 | 398,022 (−0.79%) | 397,825 (−0.74%) | 410,430 (−3.93%) | 401,923 (−1.77%) |
| 2 | 557,574 | 563,458 (−1.06%) | 562,755 (−0.93%) | 580,045 (−4.03%) | 568,248 (−1.91%) |
| Σ save vs gap 80,920b | — | −13,498 (−14%) | −10,570 (−11%) | −168,316 | −58,403 |

kodim13 [C12]:

| ch | static-G | (a1) | (a4) | (b1) | (b4) |
|---|---|---|---|---|---|
| 0 | 2,302,517 | 2,290,471 (+0.52%) | 2,288,803 (+0.60%) | 2,314,035 (−0.50%) | 2,295,888 (+0.29%) |
| 1 | 629,251 | 634,817 (−0.88%) | 633,925 (−0.74%) | 661,303 (−5.09%) | 641,213 (−1.90%) |
| 2 | 280,658 | 281,817 (−0.41%) | 281,400 (−0.26%) | 287,205 (−2.33%) | 283,755 (−1.10%) |
| Σ save vs gap 46,160b | — | +5,321 (12%) | +8,298 (18%) | −50,117 | −8,430 |

Gate (≥gap×1.2): all FAIL. Best case (13-a4) reaches 18% of gap.

## 2. Mechanism (measured, not speculated)

1. **Quantile groups are scale-homogeneous by construction** (sorted by
   mean|res|): within-group static-k is already near-optimal; N/A/k tracking
   adds cold-start (~100-symbol transient per group × 55 groups) and has no
   G-bias equivalent (static d∈−4..3 wins often on skewed groups).
2. **Global pooling mixes experts/scales** (MLP vs WAVG vs MED groups in one
   state) → whiplash, exactly the b18 kill (+0.16–0.23 there, −2..−6% here
   because grouping harvest is larger now).
3. **H0 test**: chosen-G beats order-0-entropy+real-tables by 5–7% on Y
   (heavy-tail parametric fit) — nothing left for any order-0 method.
4. Wire note (for the record): backend value 3 is free in the 2b field and the
   G payload is already raster-ordered with per-pixel keys, so per-group AND
   global adaptive decode were both C-feasible — feasibility was never the
   blocker; absent gain is.

## 3. Consequences

- DO NOT build adaptive-G wire backend; DO NOT pursue table-side on texture
  (tables ≤756 B/ch on 05 — measured small); DO NOT retry A-joint pairs
  (killed cycle-14, +7..+17% in-frame).
- Boss-5 axis is now PREDICTION on texture Y: b21#2 (it300+EMA tail),
  b21#3 (LOCO-key MLP contexts), or new texture experts. Estimated need:
  −10 KB on 05 (−1.9% of image) or −5.8 KB on 13 (−1.0%).
- Revisit triggers: a predictor that cuts 05-Y residuals ≥2%; new evidence
  that R-group (rANS) headers dominate some channel; JXL-e3 re-measurement.

## 4. Files (new only)

- `cpp/crown_enc.cpp`: `CROWN_DUMPGROUPS=<dir>` dump (winning-variant
  G-groups: expert/k/d + member (idx,sym); `GGD1` binary; behavior-preserving,
  sha-verified) + `dump_ggroups` helper. General bridge for Python-side
  measurement probes (user-blessed pattern; shared-lib if call-level needed).
- `probes/probe_b22_adaptive.py` (gdat reader + CROWN-zigzag adaptive simulator
  + zero-border activity), `probes/probe_b22_run.py` (driver + gate),
  `probes/probe_b22_nums.json`, this file. Symlinked in `experiments/`.
- Dumps: `/tmp/b22dump/` (regenerable, ~50 s/image).

## 5. Session log (scientific-brainstorming, abridged)

- Focal Q: flip 05 and/or 13 at exact-codec level with C-decodable construction.
- Independent ideas (8): adaptive-G transfer (P1), C-feasibility (P2),
  byte-gap KO math (P3), overlap skepticism — per-group best-k already adapts
  k per group (P4, DECISIVE), table-side (P5), chroma-pair (P6, pre-killed
  cycle-14), suffix refine (P7), MLP-context tail (P8).
- Criteria: flip potential (bpp math) > C-feasibility > build size >
  no-regression risk. Adversarial review passed only the measurement-gated
  variant of P1 (measure residual delta BEFORE any wire build).
- Evidence check: encoder SECT/GRP dumps (G 66% of groups; Y-Gpay 98% of ch0)
  motivated the probe; probe outcome killed P1/P5, preserved P8 + prediction
  pivots. Reopened round produced the H0-exhaustion test (double-kill).
- Decision: kill adaptivity transfer; pivot Boss-5 quest to prediction axis;
  owner: user (pending).
