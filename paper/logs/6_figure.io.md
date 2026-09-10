# 6_figure.io.md — figure finalization (scientific-visualization + venue-templates)

## INPUT (evidence → destination)
- Destination: DCC 2027 full paper, initial submission; single-column 12pt,
  9×6in text area; official source https://datacompressionconference.org/information-for-authors/
  (checked 2026-09-10); 10 pages TOTAL incl. refs/figs; single-blind.
- Placeholders in paper.tex: F1_ladder, F5_pareto, F3_ceiling_loo,
  F4_negative_atlas (all `\includegraphics[width=\figurewidth]{Figures/...}`).
- Engine: matplotlib data plots (all four are real measured-data figures;
  no image-model figures needed; no TS_FIG credentials required).
- Generator: docs/figures/make_figures.py (Agg, Okabe-Ito + markers/hatching
  redundant cues, white bg, constrained layout, DejaVu Type42 in PDF).
  Every coordinate cites its ledger; per-figure CSVs + manifest.json emitted.

## DECISIONS
- F1_ladder: as-generated; vision-checked (bars from 0, hatched literature
  + legend redundant, values match banked row). No change.
- F2_lineage: as-generated (not referenced by paper; kept in docs/figures).
- F3_ceiling_loo: data file probe_b14_summary.json was absent → reconstructed
  from probe_b14_RESULTS.md AVG row (FULL 3.2515; arms +0.0332/+0.0079/+0.0498/
  +0.0147/0/0/+0.0004) with `_provenance` key in the JSON. Marginals match
  paper caption. Vision-checked.
- F4_negative_atlas: generator CRASHED (3-tuple rows vs 4-tuple unpack — malformed
  merge). Fixed by restoring baseline labels AND removing 5 rows with no
  traceable probe-local baseline (byte-LZ +1.84, GAP +3.70, LPC-3 +4.00,
  MA-tree-lite −0.56, I-GATED −0.29); 11 fully-ledgered rows kept (see
  docs/ATLAS_BASELINES.md). No directional bias (removed wins and losses).
  Vision-checked (diverging bars from 0, polarity explained in title).
- F5_pareto: 4 critique rounds on renders (PDF print-proofs via pdftoppm):
  r1 overlapping mid-stack labels → manual offsets; r2 P2-LLM overflow →
  right-anchor; r3 E16/CROWN4 collision → leader line; r4 full mid-stack
  leader lines + shortened axis labels (detail in caption/CSV). x log scale
  labeled with base; learned points marked (GPU) + caption non-comparability
  note. Final: all labels legible, 6.0×3.9in page, Type42 embedded.
- Installed to paper/Figures/: F1_ladder, F5_pareto, F3_ceiling_loo,
  F4_negative_atlas (PDF vector; PNG kept in docs/figures).
- Official kit: dccpaper.cls + IEEEbib.bst downloaded from the DCC AuthorKit
  (2026-09-10) into paper/, UNCHANGED.

## OUTPUT
- paper/Figures/*.pdf (4, vector, embedded DejaVuSans Type42).
- docs/figures/: 6×(pdf+png+csv) + manifest.json (F1/F1b/F2/F3/F4/F5).
- experiments/probe_b14_summary.json reconstructed (provenance inside).
- paper.pdf: 10/10 pages, 0 LaTeX errors, 0 undefined refs, fonts embedded
  (validate_format: page-count pass). Print-proof pages 1,4 inspected.
- OPEN (author): author names/affiliations still TBD (single-blind allows
  names ON manuscript — fill before Oct 2); portal preview review.

## ts-figure-svg check (2026-09-10; skill loaded and evaluated)
- Verdict: redraw pipeline NOT applicable — paper contains zero free-form
  schematics (all 4 floats are real-data matplotlib plots; matplotlib path
  skips step 5b by design). No PaperBanana run, no SVG redraw, no credentials
  needed. Its applicable vector gates were run instead, all pass:
  pdfimages 0 raster rows in all 4 figure PDFs; pdffonts emb+sub+uni (DejaVuSans
  Type42); smallest word in densest figure (F5) 6.52pt ≥ 5.0 floor, median 9.32;
  paper.pdf 0 raster rows on all 10 pages; captions located (Fig1 ladder,
  Fig2 pareto, ceiling LOO).
- docs/figures/: 6×(pdf+png+csv) + manifest.json (F1/F1b/F2/F3/F4/F5).
- experiments/probe_b14_summary.json reconstructed (provenance inside).
- paper.pdf: 10/10 pages, 0 LaTeX errors, 0 undefined refs, fonts embedded
  (validate_format: page-count pass). Print-proof pages 1,4 inspected.
- OPEN (author): author names/affiliations still TBD (single-blind allows
  names ON manuscript — fill before Oct 2); portal preview review.

## ts-paper-figure attempt (2026-09-10; user-directed, skill followed to gates)
- Placeholder sweep: 0 `\fbox{\rule}` (grep exit 1; control 5 `\includegraphics`
  match, all resolve to vector PDFs); 0 FIGURE-SPEC. Only commented float is
  the intentionally dropped F1b (duplicates Table 5).
- Credential gate: TS_FIG_MODEL/API_KEY/BASE_URL unset, no .env anywhere;
  gen_image.py hard-returns `unset env` → per skill, STOP + ask user.
- User decision: accept — no placeholders; all floats are matplotlib data plots
  that skip the image-model path by the skill's own routing. No PaperBanana
  run, no schematic commissioned.
