# Repository navigation (structured map)

```
autocompress/
├── AGENTS.md               # agent operating manual (read first)
├── README.md               # human entry: mission, boss board, quickstart
├── Makefile                # all/clean/check/help  |  reproduce.sh  # checksums→build→round-trips
├── requirements.txt        # pinned env (torch optional)
├── RESULTS.md              # results ledger (current truth)
├── CHECKSUMS.txt           # Kodak md5 pins
├── src/                    # CANONICAL CODE: hapre.c + crown{2..6}.c + driver_*.py
│   ├── *_FORMAT.md         # byte-level stream specs per generation (read before touching streams)
│   ├── hapre.c.golden-*    # era snapshots (read-only)
│   ├── dp_refine.py fit_nlls.py pareto.py perimage.py test_rans.py rans_sanity.py
│   └── *.so (build artifacts, git-ignored)
├── cpp/                    # C++17 port (crown_enc/crown_dec, byte-identical 7/7): codec.* crown_enc/dec.cpp
├── probes/                 # CANONICAL probe files: probe_b{1..21}_*.py + RESULTS + logs/jsons
├── experiments/            # real_photos/ + quality.py rd_sweep.py train_mlp.py + crown6_weights/
│                           # + compat symlinks → ../probes/ (LOAD-BEARING, do not copy over)
├── docs/                   # paper drafts (v10 current), VERIFY_report.md, INDEX.md, plans/
├── survey/                 # literature survey, QOI teardown (+qoi/ clone, untracked),
│                           # JXL boss-takedown (+bosstakedown/, heavy .jxl git-ignored in data-archive/)
└── memory/                 # cycle1-memory.md (full log) + HANDOVER.md + evolution-reports/ + kg.*
```

## Wayfinding by task
| Task | Start here |
|---|---|
| Reproduce results | `./reproduce.sh` |
| New compression idea | `memory/cycle1-memory.md` dead-ends first, then `probes/probe_b*_RESULTS.md` patterns |
| Touch a stream format | `src/CROWN*_FORMAT.md` + `AGENTS.md` rules 5,6,11,12,13 |
| Beat JXL-e3/e9 | `survey/bosstakedown/BOSS_TAKEDOWN.md` + HANDOVER queue |
| Train MLPs | `experiments/train_mlp.py` (LS-init/L1/int16 lessons in memory cycle 25) |
| C++ work | `cpp/README.md`, byte-identity vs `ref_bins` goldens (colab bundle) |
| Write up | `docs/HAPRE-CROWN-paper-v10.md` + `docs/VERIFY_report.md` |
