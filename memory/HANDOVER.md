# HANDOVER — session state for a fresh agent (2026-09-08)

## One-paragraph state
7 bosses dead at exact-codec level with Wilcoxon proofs (PNG-9, QOI, JXL-e1, WebP-m0/m3/m6, JPEG-LS-CharLS);
best exact codec CROWN6 at 3.1978 bpp Kodak avg; C++ port byte-identical 7/7 after fixing
a predictions-as-residuals bug in Q assembly; C++ trainer done (pure C++17, no torch,
0.00% BPP parity bit-identical on smooth+texture crops, 12.9 s full-image train,
gating 91% vs banked); standing: FLIF v0.4 (0W-7L, 2.9699), JXL-e3 (+0.97%, 5W-2L p=.219),
JXL-e9 (+5.5%). Classical+neural-micro ceiling ≈3.20 with proof. Harness hardened
(README/Makefile/reproduce/git, 4 commits). Paper v1.0 drafted + number-audited.

## Numbers to trust (exact bytes, asserted round-trips)
CROWN6 3.1978 (per-img 01:3.2982 02:2.9854 05:3.5917 07:2.6790 13:3.9525 19:3.1510 23:2.7365)
vs JXL-e3 (01:3.3593 02:3.0611 05:3.5120 07:2.7310 13:3.9141 19:3.2154 23:2.8110).
C++ encode ~7.2 min/img (was Python 38 min); decode 100–350 ms. Full detail: RESULTS.md.

## Queue (in order)
1. **C++ trainer DONE** (committed `2b4c20d`) — pure C++17, 0.00% parity
   bit-identical on 2 crops + full-23 (−0.0000%), full-23 train 12.9 s.
2. **Encode speed DONE** — ~14× bit-exact (fused Golomb, histogram Huff,
   dead-track cut, 9-way channels, `make fast`); ~31 s/img, full-7 sha-identical (cycle 46+54).
3. **Boss 5 STANDS** — triple-negative banked (cycle 47): b22 adaptivity ~0
   (overlap ~100%), b23 tail ~0, b24 headers ≤37% of gaps. Needs prediction
   mass (−10 KB on 05); no queued weapon left at scale.
3. **Boss 5** — needs −1.2% from 3.1978; texture (05/13) is the gap; CABAC-adaptivity was the
   identified weapon (probe_b19: −2.18% MED-frame, transfer unbuilt).
4. **Boss 6** — needs learned scale; torch 2.14+cpu ready; disk tight.
5. **Publish** — DCC2027 target (deadline Oct 2 2026, 10pp total, single-blind);
   audit banked (`docs/PUBLISH_audit.md`): SOTA verified vs PDFs, 0 figures
   (F1–F6 specified), 9-item camera-ready list. Next: classical baselines +
   ledgers, then LaTeX.

## Gotchas (paid for in blood)
- Flat inputs hide ALL expert bugs (verify on texture, force every variant through decode).
- `experiments/*.py` are symlinks → `probes/` (never copy over them).
- Drivers must not `os.chdir` at import; unary polarities differ by codec era; rANS needs M=14;
  tanh needs frozen LUT; int8 weights destroy nets (int16-adaptive only); L1 loss + LS-init mandatory.
- Subagent dispatch hits rate limits intermittently — wait ~90 s and retry once; user can resume manually.

## Key files
AGENTS.md, README.md, Makefile, reproduce.sh, RESULTS.md, CHECKSUMS.txt,
memory/cycle1-memory.md, docs/ (paper v10, VERIFY_report, INDEX, NAVIGATION),
src/CROWN*_FORMAT.md, cpp/README.md, cpp/TRAINER.md, survey/ (SURVEY_genai_qoi, BOSS_TAKEDOWN),
experiments/rd_sweep.py + quality.py (metrics), experiments/train_mlp.py.