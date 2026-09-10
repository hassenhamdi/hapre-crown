# AGENTS.md — operating manual for agents working this repo

Read this FIRST in any fresh session, then `memory/HANDOVER.md`, then `README.md`.

## Mission
Lossless image compression campaign: exact bit-level codecs beating JPEG/WebP/JXL/JPEG-LS/QOI
configurations one by one (Wilcoxon paired n=7, p≤.05 to declare a KO), all claims
verified by byte-exact round-trips. Current best exact: CROWN6 3.1978 bpp Kodak avg.
Standing bosses: FLIF v0.4 (0W-7L), JXL-e3 (+0.97%), JXL-e9 (+5.5%).

## Non-negotiable rules
1. **Evidence before claims.** No completion claim without fresh verification output in hand.
   Round-trip = `decode(encode(x)) == x` byte equality, asserted, not eyeballed.
2. **Probe-level ≠ exact-codec.** numpy estimates never substitute for stream bytes.
   Label which level every number lives at.
3. **Losses reported like wins.** Dead ends go in `memory/` with mechanisms, never deleted.
4. **Unit rule.** `bpp = total_bits/(H*W*3)`. Every side byte counted (tables, maps, flags, headers).
5. **Decoder-safety.** Every conditioning signal must be causally available at decode;
   state scan order explicitly. Border conventions must match bit-exactly both sides.
6. **Alphabet bounds.** Residuals proven within ±1024 for 8-bit RGB; assert loudly, never clip silently.
7. **No `os.chdir` at import.** Use absolute `sys.path`. (Burned once — cycle 38.)
8. **Single canonical locations.** `src/` = code, `probes/` = probe files, `experiments/` holds
   compat symlinks → `../probes/` (load-bearing for imports; never replace links with copies).
   Never add compat shims; never duplicate files (check `md5sum` + file types, not names).
9. **rANS precision.** M=14 minimum for 200+ symbol alphabets (M=10 starves tails).
10. **Float discipline.** `-ffp-contract=off` for crown6 builds; frozen tanh LUT for wire forward;
    numpy/torch/C-libm tanh differ by 1 ulp — cross-check int-rounded equality, never assume.
11. **Unary polarities are wire format.** Document per codec (variants exist across generations).
12. **Round-trip passing proves consistency, never quality.** Force EVERY variant/arm through
    decode + byte-compare on HARD inputs (flat crops hide all expert bugs — cycle 43 lesson).
13. **Framing asserts everywhere** (`p == len`), per channel and per stream.

## Build / verify (see Makefile, reproduce.sh)
- `make all` — six .so libs (libhapre -O3; libcrown* -O2; crown6 + `-ffp-contract=off`).
- `./reproduce.sh` — checksums → build → byte-exact round-trips + bpp table. Exit nonzero = fail.
- Python 3.14 needs `python3 -m pip` (system `pip` binary is broken); torch 2.14+cpu installed
  `--user --break-system-packages` (MLP training only). Disk is tight (~6 GB free): no datasets,
  shallow clones only, clean core dumps.
- Subagent delegation: sequential per experimental branch (shared state); parallel only for
  independent probes with SEPARATE output files. New files only unless branch owns the file.

## Skills (repo-local, load when triggered)
- `skills/eagle-eye-debugging/` — sniper debugging for valid-but-wrong codecs (distilled cycle-43).

## Key paths
Test images: `experiments/real_photos/kodim0{1,2,5,7,13,19,23}.png` (md5: `CHECKSUMS.txt`).
Live codec: `src/hapre.c` (+`crown*.c`); goldens: `src/hapre.c.golden-*` (read-only).
C++ port: `cpp/` (byte-identical to Python refs on 7/7). Weights: `experiments/crown6_weights/`.
Colab bundle: `/tmp/opencode/colab/crown6_colab.tar.gz`. Full map: `docs/NAVIGATION.md`.

## Current queue (see HANDOVER for detail)
C++ trainer → deeper encode-speed work → Boss 5/6 → publish.