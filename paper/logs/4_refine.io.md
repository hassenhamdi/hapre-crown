# 4_refine.io.md — ts-paper-refine pass on DCC single-file draft

Scope note: repo paper is a single-file DCC 2027 draft (`paper/paper.tex`,
2353 words pre-pass), not a ts-paper pipeline workdir (no `template.json` /
`sections/*.tex`). Applied the refine pass principles adapted to venue:
DCC conference conventions respected (no journal re-sizing); prose-only
edits; all `\cite` / equations / `\ref` / table-figure labels preserved;
zero number changes except one lineage-label correction bound to the
lineage table + RESULTS.md.

## INPUT
- `paper/paper.tex`: 2353 words; 10 `\cite`, 12 `\ref`/`\label`; chktex: only
  pre-existing style nits (command-space, dash-length heuristics); braces 0,
  brackets 0, begin/end 21/21.
- Canned-phrase scan: clean (no tapestry/testament/plays-a-crucial-role/
  it-is-worth-noting/delve/realm/paradigm-shift/in-order-to).
- Banked numbers (VERIFY_report): CROWN6 3.1978 + per-image row CONFIRMED;
  fresh RUN full-7 pooled 3.5114 matches banked 3.511.

## DECISIONS (4 minimal prose edits, each logic self-checked)
1. Introduction ¶1: merged two choppy sentences with "while" (flow; no
   terminology switch, no claim change).
2. Related Work learned-table lead-in: "(...)" → "--- ..." (colon/em-dash
   connection per de-AI comma-soup rule; meaning identical).
3. JXL mechanism map closer: fragment "Full ladder, ...: takedown ledger."
   → full sentence "The full ladder, ... are in the takedown ledger."
   (grammar; same pointer, no new claim).
4. Conclusion lineage: "From HAPRE-C (3.464, E)" → "From HAPRE-C (3.58, E)
   through MOE (3.464, E)" (honesty: 3.464 is the MOE row per lineage
   Table 4 + RESULTS.md; HAPRE-C 0-ctx is 3.58).

## OUTPUT
- `paper/paper.tex`: 2363 words (+10, all function words from edits 1/3/4).
- Preserved: 10 `\cite`, 12 `\ref`/`\label`, all equations/tables/figures.
- Tells re-scan: clean. chktex: no new warnings on edited lines (remaining
  dash warnings are pre-existing file-wide heuristic noise).
- PDF gate: blocked externally (`dccpaper.cls` not in repo) — structural
  gates (braces/brackets/begin-end/chktex) pass instead.
- Deletions: none. No redundancy removed beyond sentence-level (draft is
  v10-audited; no repeated derivations found).
