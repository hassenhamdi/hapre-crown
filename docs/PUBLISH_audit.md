# PUBLISH Audit — venue standards, SOTA comparison, draft audit (2026-09-08)

Skills driving this audit: `venue-templates` (compliance), `literature-review`
(search + synthesis), `paper-lookup` (identifier/provenance discipline),
`scientific-writing` (evidence binding, no fabrication), `paper-review`
(5-aspect self-review + figure/table checks), `humanizer` (targets flagged,
rewrite deferred to LaTeX stage with author voice sample).

## 0. Compliance note (venue-templates §3)

```text
Target: DCC 2027 full paper, initial submission
Official source: https://datacompressionconference.org/information-for-authors/
Checked: 2026-09-08
Deadline: October 2, 2026, 11:59pm US Pacific (~24 days)
Main-text limit: 10 pages TOTAL incl. references, figures, tables, appendices
Format: single-column, 12pt, 1" top / 1.25" left margins, 9x6" text area, PDF
Anonymity: single-blind (author names + affiliations ON manuscript)
Official template: LaTeX sample ZIP + Word .doc + template PDF (site links)
Proceedings: IEEE Xplore. Notify: late December.
```

## 1. SOTA comparison (verified; evidence ledger §5)

bpsp = bits/sub-pixel. Our scale: CROWN6 **3.1978** (exact, 7-img hard subset).

| Codec | Kodak (bpsp) | Source / setting | Status |
|---|---|---|---|
| PNG | 3.93–4.48 | P2-LLM Tab.2 3.93; CALLIC Tab.1 4.35; SEEC Tab.I 4.48 (=13.44/3) | verified (spread = versions) |
| JPEG-LS | 2.82–2.99 | P2-LLM 2.82; P2-LLM text 2.99 | verified; NOT measured locally |
| CALIC | 2.87–3.07 | P2-LLM 2.87; P2-LLM text 3.07 | verified; NOT measured locally |
| JPEG 2000 | 2.93–3.19 | P2-LLM 2.93; CALLIC 3.19; HPAC Tab.1 3.19; SEEC 3.19 (=9.58/3) | verified |
| WebP-lossless | 2.90–3.11 | P2-LLM 2.90; ours m6 3.3157 (7-img, PIL) | verified (subset differs) |
| FLIF | 2.72–3.01 | P2-LLM 2.72; CALLIC 2.90; SEEC 3.01 (=9.04/3) | verified; NOT measured locally |
| JPEG-XL | 2.63–3.06 | P2-LLM 2.63; CALLIC 2.87; SEEC 3.06 (=9.18/3); ours e9 3.0324 (7-img, cjxl 0.11.2) | verified (version spread real) |
| L3C / RC | 2.94–3.26 / 2.93–3.38 | CALLIC Tab.1; P2-LLM Tab.2 | verified |
| iVPF / iFlow | 2.54 / 2.44–2.68 | P2-LLM Tab.2 (Kodak 2.54/2.44) | verified |
| DLPR (TPAMI24) | 2.38–2.86 | P2-LLM 2.86; CALLIC 2.86; SEEC 2.86 (=8.58/3); 22.2M params | verified |
| ArIB-BPS | 2.42–2.78 | P2-LLM —; CALLIC 2.78; HPAC Tab.1 2.78; 146.6M params | verified |
| MGCF / HPAC base | 2.33–2.77 / 2.73 | CALLIC Tab.1–2 (575K); HPAC Tab.1 (677K, 0.89s enc) | verified |
| CALLIC (AAAI25) | **2.54** | Tab.1 Kodak; MGCF 2.77→2.54 via LoRA-RPFT MDL; 575K+25K; enc 9.7s dec 1.7s | verified (primary PDF pp.6–7) |
| HPAC-FT (2025) | **2.52** | Tab.1 Kodak (base HPAC 2.73); 677K; SARP-FT T=50 | verified (primary PDF p.9) |
| SEEC (2026) | 2.84 | Tab.I 8.53 total bpp ÷3; DLPR-backbone + 2-class masks (+0.02 bpp) | verified (unit conversion noted) |
| P2-LLM (2025) | 2.83 | Tab.2 Kodak (CLIC.m 2.08); Llama3-8B LoRA; 8×A800; dec ~273 s/img | verified (primary PDF p.7) |
| DualComp-I 96M | 2.57 | 96M params, CPU 317 KB/s | snippet-only (not PDF-verified) |
| **CROWN6-exact (ours)** | **3.1978** | exact bytes, 7-img subset, C dec ~0.1–0.35 s, CPU-only, zero-dep | banked (crown6_results.json) |

Reading: CROWN6 beats PNG / WebP-lossless / JPEG 2000 / L3C-class on
overlapping images, sits near CALIC on a hard 7-subset, below FLIF / JXL /
learned (2.4–2.9). The draft's "~60% of the way from JXL-e1 to neural SOTA"
recomputes to **60.7% with DLPR-2.86 as denominator**
((3.72−3.1978)/(3.72−2.86)) — VERIFIED, recommend stating denominator.
Unit hygiene: SEEC reports total bpp (÷3 done); CALLIC/P2-LLM/HPAC report
bpsp natively. JXL spread (2.63–3.06) is version/settings + 24-img vs 7-img
subset — ours (3.03, cjxl 0.11.2 -e 9, 7-img) sits inside it; state all three
coordinates with every JXL number. Our 7-subset skews hard (05/13/19 in,
many easy ones out) — say so explicitly; do NOT present 3.1978 as full-24.

## 2. Draft audit (paper-review 5 aspects)

**Contribution sufficiency — CONDITIONAL PASS.** Ratio leadership absent
(3.1978 vs FLIF ~2.9 / JXL ~2.6–2.9 / learned ~2.5). Carrying contributions:
(a) exact-bytes campaign w/ 5 Wilcoxon KOs, (b) classical ceiling + LOO proof,
(c) negative atlas w/ mechanisms, (d) EFFICIENCY Pareto — currently
under-sold: C dec 0.1–0.35 s CPU vs DLPR 1.8 s GPU / CALLIC 1.7 s /
ArIB-BPS 7 s / P2-LLM 273 s. Recommend a ratio-vs-decode-time Pareto figure
as the paper's visual thesis (DCC-valued: real decoders).
**Writing clarity — PASS with targets.** Reproducibility unusually strong
(code, checksums, pinned versions, seeds via unit_seed). Gaps: torch thread
count determinism note (12 threads — state seed+threads fixed or verify
bit-stability); `csrc/` vs `src/` path drift (draft §3.5/§8 say `csrc/`, repo
moved to `src/` + `cpp/` — FIX before camera-ready); CROWN5 mentioned only
in appendices (fine) but §4.2 skips #4 numbering (cosmetic).
**Results quality — PASS.** 52/52 banked rows re-verify (VERIFY_report);
single wrong digit found already (huff-vs-m3 p .47→.30, verdict unchanged).
**Testing completeness — GAP LIST (ordered):** (1) FLIF v0.4 MEASURED 2026-09-08
(2.9699 avg, pixel-PASS 7/7, ledger `survey/bosstakedown/data/classical7.json`;
new standing boss, beats e3 7/7 here) + JPEG-LS measured two ways (CharLS RGB
4.5757 DEAD; YCoCg-R ablation 3.3664). CALIC still literature-only (no reference
binary in-box; reimplementation found is unverified Python — not used).
(2) VERIFY_report §Counts gaps stand: PNG-9/WebP-m0/RUN per-image rows,
byte-LZ/wavelet+zlib/LOCO-lite/G-bias primaries, CROWN3 row, HC-3.309 row.
(3) Missing W=6 on Table line 193 (Boss-5 row; VERIFY has it).
(4) Multiple-comparison footnote for 5 sequential KOs (distinct
pre-registered pairwise hypotheses; n=7 floor p=.016 — state it before a
reviewer does). (5) Learned rows need train-data/compute footnotes
(CALLIC DIV2K+Flickr2K/2M steps; HPAC 90k imgs/2M steps; P2-LLM 4×A800) —
otherwise "575K params" misleads (pretraining excluded).
**Method design — PASS.** Scope discipline explicit (no decode-time NN).
Note fragility honestly: old 0-ctx driver hard-fails noise decode (rc=−2);
frame as robustness TODO (already §7) + gate new claims on it.

## 3. Tables / figures (paper-review checklist + DCC budget)

Draft has **zero figures** — required before submission:
F1 teaser (boss ladder: ours vs JXL ladder vs learned, with subset note);
F2 codec-lineage pipeline (HAPRE→CROWN6 blocks, gate symbols);
F3 ceiling/LOO bar chart (§4.5); F4 negative-atlas schematic (families ×
mechanism, one glance); F5 Pareto ratio-vs-decode-time (the efficiency
thesis); F6 RD τ-plot (§4.7, marked prototype). Tables: convert to booktabs
at LaTeX stage, captions above, metric arrows (bpp ↓), bold best, keep
E/P labels + [VERIFY]→resolved values. Page budget (12pt single-col ≈
500 wds/pg): ~4,500-word draft + 6 figs + 8 tabs ≈ 9–11 pp — OVER RISK.
Plan: cut §5 prose 30% (table-ify mechanisms), move LOO full table +
FORMAT details to supplementary (DCC allows appendices inside the 10 pp —
no, TOTAL incl. appendices: cut, don't move).

## 4. Humanizer targets (deferred rewrite, needs author voice sample)

Flagged (not yet rewritten): Abstr. "a feature, not an appendix" (§1 staging);
§1 "written for DCC because DCC values…" (fine, keep — venue fit is a real
claim); §7 "What we will not do" list (triad-stacking + closers — keep items,
drop send-off rhythm); repeated "honest/honestly" (~6× — vary or cut 3);
"man-years" (§7, informal — quantify or cut); "never again" (§5, chatty —
cut). Functional contrasts to KEEP (correct reader beliefs): "do not claim
neural SOTA / claim strongest exact classical-plus-micro" (§13). Rule: every
kept contrast must correct a belief a DCC reviewer actually holds.

## 5. Evidence ledger (scientific-writing binding; H = human-verified vs opened source)

| Row | Source | Locator | Accessed | H |
|---|---|---|---|---|
| CALLIC 2.54 (+table) | arXiv:2412.17464v1 | pp.6–7 Tab.1–2 | 2026-09-08 | H (PDF text above) |
| HPAC-FT 2.52 / base 2.73 | arXiv:2511.10991v1 | p.9 Tab.1, p.10 Tab.3 | 2026-09-08 | H (PDF text above) |
| SEEC 8.53 total (÷3=2.84) | arXiv:2509.07704v2 | p.5 Tab.I | 2026-09-08 | H (PDF text above) |
| P2-LLM 2.83 Kodak / classics | arXiv:2411.12448v2 | p.7 Tab.2, p.8 Tab.5 | 2026-09-08 | H (PDF text above) |
| JXL spread 2.63–3.06 | three tables above | as listed | 2026-09-08 | H |
| DCC rules/deadline | datacompressionconference.org | /information-for-authors, /paper-submission | 2026-09-08 | H (fetched text) |
| DualComp-I 2.57 | web snippet (DualComp paper) | search highlight | 2026-09-08 | snippet-only — verify before citing |
| HPAC "Nov 2025 / fastest" | 2511.10991 d.2025-11-14; Tab.3 0.89s | discovery + PDF | 2026-09-08 | H |
| Ours 3.1978 + Wilcoxons | crown6_results.json; VERIFY_report | banked | 2026-09-08 | H (repo) |

## 6. Camera-ready task list (deadline-ordered, ~24 days)

1. [VERIFY] p .47→.30 + sign fix (10 min).
2. Classical baselines: imagecodecs-JPEGLS/FLIF attempt on the 7 (2 h; else caveat rows).
3. Per-image ledger gaps (VERIFY §Counts) — bank or soften (1–2 d).
4. `csrc/`→`src/`+`cpp/` path sweep in draft (30 min).
5. Figures F1–F6 (2–3 d; F5 Pareto is the thesis figure).
6. LaTeX on official DCC sample: booktabs, captions, full BibTeX (expand §Refs shorthand), page-budget cut to 10 pp (2–3 d).
7. Humanizer pass w/ author sample on Abstr/§1/§7/§8 (half-day, post-layout).
8. Seeds/threads determinism note + multiple-comparison footnote (1 h).
9. Fresh `./reproduce.sh` + frozen `CHECKSUMS.txt` + code snapshot Appendix (half-day).
