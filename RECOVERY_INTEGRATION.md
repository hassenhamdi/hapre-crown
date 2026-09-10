# Integration (2026-09-10, time-ordered merge)
- Base A: /tmp/opencode_recovery/autocompress (176 files, fork+original image compression, 418 applied/21 skipped/40 non-blocking conflicts, py 117 OK/1 FAIL probe_xch U+201D)
- Overlay B: /tmp/opencode_recovery_crown6_main/crown6 (46 files+13 symlinks, cpp trainer ses_f7eac6c1 family, 197 edits+40 replaces+19 appends/9 skips/21 conflicts, py 17 OK, crown_enc FAIL documented cascade)
- Rule: B wins on 29 overlaps (trainer session 1788876-1788953 later than fork 1788852-1788876). A supplies unique probes/builds/docs. Pre-existing survey/paper retained additively.
- Result: 195 files +13 symlinks. Validation: py 128 OK/1 FAIL (csrc/probe_xch.py:72 U+201D, faithful), no conflict markers, crown_enc.cpp 33943B (known wf/l0/l3/mlp1 cascade, see crown6 REPORT), 4 *_nums.json symlinks dangle (runtime-generated, no trace base).
- Backup: /tmp/backup_autocompress_20260910. Repro scripts: /tmp/opencode_recovery/recover.py, /tmp/opencode_recovery_crown6_main/recover_crown6.py. Manifests: /tmp/opencode_recovery/MANIFEST.json, /tmp/opencode_recovery_crown6_main/MANIFEST.json.
