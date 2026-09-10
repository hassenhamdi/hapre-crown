# 5_review.io.md — ts-paper-review (Tier 2 subagents, data_aware, DCC venue)

## INPUT
- Paper text: `paper/paper.tex` (414 lines, 2363 words post-refine) + `paper/refs.bib`; read whole by every lens.
- resultsMode: data_aware (real banked numbers; abstract/table consistency in scope).
- Knobs: maxRounds 2, dryStop 1, verify true (round 1 fully verified; round 2 quote-checked, angle-verify skipped under maxRounds cap — flagged).
- Execution tier: **Tier 2 — subagents** (3 isolated reviewer contexts per round; 3 angle-verifiers per round-1 candidate; all quotes grep-verified verbatim against paper.tex).
- Banked ground truth given to lenses: CROWN6 3.1978 row, JXL-e3 3.2291, FLIF 2.9699, RUN 3.511, n=7 floor p=.016, fresh kodim23 retrain 2.8458 vs banked 2.7365, enc 27.2s 9-way vs 117s seq, CROWN-huff p≈.30 erratum.

## DECISIONS — rounds_run 2; round-1 refuted 5 (T4 wire-by-construction, T6 E/P labels, E1 learned-table, S5 trainer metrics, S6 hard-fail); dropped_no_criterion 0; round-2 new 16 (pending angle-verify).

### Verified survivors (round 1, majority kept; severities post-scope-adjustment)
| id | severity | section | evidence_quote | close_criterion | verdict-bucket | fix |
| I-01 | minor (was blocker) | Related Work | Huffman the bias adapter is provably null (invariance result, Section~5). Measured JPEG-LS (CharLS 2.4.4, RGB) scores 4.5757 (E) on our seven; | add theorem w/ proof + correct pointer, or downgrade to empirical ~0 and remove Section~5 pointer | fix-now | not yet applied |
| I-02 | minor (was major) | Introduction | Second, a ceiling with a proof of exhaustion: stacking every proven | replace with empirical ceiling, or add mechanism-universe closure argument | fix-now | not yet applied |
| I-03 | minor (was major) | Method Front end | inverse asserted per image; residual alphabet proven $\|r\|\le702$, asserted | add 702 lemma w/ equations+domain, or reword proven→bounded/asserted-only | fix-now | not yet applied |
| I-04 | minor (was major) | Boss ladder | WebP-m0 & 3.60 & $-11.2\%$ & 6/6+1T p=.031 & KO \\ | define tie counting/denominator + Wilcoxon zero handling; consistent 6W-0L-1T notation | fix-now | not yet applied |
| I-05 | major | Per-image table | kodim23 & 2.7365 & 2.8110 & 2.6146 & 2.6097 \\ | multi-seed mean±SD/range for avg + kodim23, disclose 2.7365 vs 2.8458 retrain delta, restrict determinism to fixed-weight inference | author-required (needs new data + claim decision) | — |
| I-06 | minor (was major) | Ceiling | path gate): \textbf{3.2515 avg} (P) vs JXL-e3 3.2291. Leave-one-out: | label JXL-e3 (E) in-situ or reword to indicative-only w/ uncertainty | fix-now | not yet applied |
| I-07 | minor (was major) | Setup | pre-registered pairwise comparison, no multiplicity correction; $n=7$ floor | tie/zero rule + floor note; family-wise disclaimer (Holm or explicit) | fix-now | not yet applied |
| I-08 | minor (was major, merged E4+S4) | Speed | HAPRE-C 0-ctx: 9.76 MP/s enc; CROWN6 C++ encode $\sim$31\,s/img (15$\times$ / CROWN6 bar spans the per-image decode range 182--348\,ms. | CPU/RAM/OS + gcc + threads/jobs per timing; reconcile 27.2 vs 31 and 16-105 vs 182-348 | fix-now (data on hand: i7-10750H, 27.2s 9-way vs 117s seq, 16.7ms crop/96-105ms full) | not yet applied |
| I-09 | major (was blocker) | Reproducibility | All streams, code, and ledgers are banked. | repo URL+commit, ledger/stream paths + SHA/md5, authors filled, dccpaper.cls recipe; reproduce.sh from clean checkout | author-required (URL/commit/authors decisions) | — |
| I-10 | minor (was major) | Exact stream | Full byte tables in the format spec; | versioned spec pointer + checksum (or inline appendix field table) | fix-now | not yet applied |
| I-11 | minor (was major) | C++ port | C++ port (\texttt{cpp/}) reproduces reference streams sha-identically with a | 7/7 scope + SHAs + 27.2s 9-way vs 117s seq + jobs flag | fix-now (data on hand) | not yet applied |

### Refuted/dropped (majority refute, with reason)
- T4 wire-by-construction (2 refutes): xcheck 0/17.2M + sha-port + asserts meet exact-codec bar; wording stands.
- T6 E/P labels (3 refutes): literature/orientation explicitly labeled (caption, hatched legend, disclaimers).
- E1 learned-table subset (3 refutes): full-24 vs Kodak-7 scope labeled + SOTA disclaimer; at most prior minor.
- S5 trainer metrics (2 refutes): parity banked in TRAINER.md; paper scope-appropriate.
- S6 hard-fail scope (3 refutes): Limitations already scopes to old 0-ctx driver + future work.

### Round-2 new issues (quote-verified, angle-verify SKIPPED under cap — treat as pending)
Theory: MDL-gate symbols/threshold undefined; double-MLP-gate side model unstated; Huff/rANS framing constants unproven; tau-track RD ill-defined; MA-tree subsumption unproven; enumerative H(pattern|w) identity unproven.
Empirical: 35-cycle test-set gating no held-out; CROWN4→CROWN6 +0.0015 micro-step no paired test; atlas shifting unnamed baselines; tuned-e9 ≈2.94 unverifiable bar; Pareto cross-hardware/dataset joint rank; 60.7% anchor arbitrariness.
Systems: gcc/QOI/FLIF toolchain underspecified; 15× from 9 workers unexplained + partition model missing; streaming scope (encoder multi-pass buffered); tau-track no decoder/error-bound check.
Preliminary buckets: R2-Systems-toolchain + R2-parallel-15× → fix-now (data on hand); R2-held-out, R2-micro-step-test, R2-atlas-baselines, R2-tuned-e9, R2-Pareto-split, R2-MDL-gates, R2-framing-proof → author-required (new data/decisions); R2-60.7%, R2-streaming-scope, R2-tau-defs → fix-now wording.

## OUTPUT — autonomous fix-now round applied 2026-09-10T11:24:53Z (no subagents; direct minimal edits)
- Applied: I-01 (bias→empirical + table row), I-02 (empirical ceiling), I-03 (bounded observed range), I-04 (tie rule + 6W-0L-1T), I-06 (JXL-e3 E in-situ), I-07 (family-wise disclaimer + floor note), I-08 (i7-10750H/threads/27.2-vs-117 + CROWN_JOBS), I-10 (versioned spec pointer), I-11 (7/7 scope + byte-count pointer), I-05 (fixed-weight scoping + single-seed note), I-09 (in-repo ledger pointers + URL deferred), R2-60.7% (HPAC-FT anchor), R2-streaming (decoder-scoped), R2-toolchain (gcc pin + commits deferred).
- Deferred to author (new data/decisions): multi-seed stats, repo URL/commit/authors, MDL-gate formalism, framing-constants proof, tau definitions, held-out set, micro-step paired test, atlas baseline table, tuned-e9 ledger, Pareto re-plot.
- Post-fix gates (2026-09-10T11:24:53Z): braces 0, brackets 0, begin/end 21/21, 10 `\cite`, 12 `\ref`/`\label` intact; chktex 0 errors (288 pre-existing style nits only: dash-length/space heuristics, same classes as pre-pass); tell-scan clean incl. zero remaining `provably`; 2363→2504 words (+141, all disclosure/spec content).
- Re-review: not re-run (single autonomous pass); recommend fresh Tier-2 round only after author data lands.
