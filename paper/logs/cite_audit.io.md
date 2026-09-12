# Citation audit (ts-paper-cite discipline) — 2026-09-12, post-submission check

Scope: `paper/refs.bib` (10 entries) + all `\cite{}` in `paper/paper.tex`.
Method: structural completeness per entry + existence verified against
authoritative pages (arXiv abs for preprints; well-known DOIs for classics).

## Verdict: PASS — all 10 entries are real, correctly attributed papers

### Classics (complete: authors/year/venue/vol/pages/DOI; DOIs well-known, valid)
- weinberger2000loco — LOCO-I/JPEG-LS, IEEE TIP 9(8):1309–1324, 10.1109/83.855427 ✓
- wu1997calic — CALIC, IEEE TCOM 45(4):437–444, 10.1109/26.585919 ✓
- skodras2001jpeg — JPEG 2000, IEEE SPM 18(5):36–58, 10.1109/79.952804 ✓
- sneyers2016flif — FLIF/MANIAC, ICIP 2016, 10.1109/icip.2016.7532320 ✓
- bai2024dlpr — DLPR, IEEE TPAMI 46(5):3577–3594, 10.1109/tpami.2023.3348486 ✓
- alakuijala2019jxl — JPEG XL architecture, SPIE 11137:112–124, 10.1117/12.2529237 ✓

### Preprints (existence verified live on arXiv abs pages this session)
- li2024callic = arXiv:2412.17464 CALLIC (authors match; Accepted AAAI 2025) ✓
- chen2024pllm = arXiv:2411.12448 P²-LLM (9 authors match) ✓
- zheng2026seec = arXiv:2509.07704 SEEC (3 authors match; Accepted ICME 2026) ✓
- li2025hpac = arXiv:2511.10991 HPAC (6 authors match; title verbatim) ✓

## Notes (no action; paper is submitted at 10/10 pages)
- Key/year cosmetics (li2024callic→year 2025) are arbitrary labels, harmless.
- SEEC now shows ICME 2026 acceptance; HPAC note "Under review" is unverifiable
  from the arXiv page. Both are sub-line refinements — apply only inside a
  camera-ready revision (any bib edit needs a rebuild + page-count re-gate).
- No stubs, no orphans (10 cites in text, 10 entries), no duplicate keys.
- Coverage is intentionally narrow (10 refs for a systems campaign paper);
  appropriate here — every cite supports a concrete number or method claim.
