# CROWN6 C++ port (`cpp/` — all files NEW, existing tree untouched)

Self-contained C++17 deterministic encoder + decoder for the champion
lossless codec (CROWN6, 3.1978 bpp Kodak avg). Goal: **bit-identical**
streams to the reference Python encoders (`src/driver_crown6.py`) on all
7 Kodak images (byte equality via sha256, not just bpp parity).

## Files (you own these; nothing else modified)

- `codec.h` / `codec.cpp` — shared port: E16 experts, b4 neighborhoods,
  LOCO-365 keys/signs, RCT bank, weighted-fit (incl. least-squares),
  LMS5 pair, causal MLP features + wire forward, numpy-linear quantile,
  Huffman via exact CPython-heapq port, Golomb best-k, rANS costing,
  stored-zip NPZ loader, PNG via vendored `stb_image.h`.
- `crown_enc.cpp` — encoder: YCoCg-R→RCT shortlist → autoK quantile groups
  (+ exact top-3 rANS refine) / GRID search → per-group best-of E20 experts
  (+ frozen-MLP from `.npz`) → per-group Huffman/Golomb/rANS → wire bytes.
- `crown_dec.cpp` — decoder: owns scan/format logic; calls the linked
  read-only C objects for rANS decode, the causal channel scan and RCT
  inverse. Asserts exact framing (`p == len`).
- `Makefile` — builds `crown_enc` + `crown_dec` + `crown_train` (C++17, `-O2`,
  `-ffp-contract=off` everywhere for the frozen-tanh-LUT contract).
- `crown_train.cpp` — pure-C++17 MLP trainer (probe_b21 committed config, no
  torch/Python at runtime). See `TRAINER.md` for methods + verification.
- `stb_image.h` — vendored single-header PNG reader (choice: vendored
  stb instead of libpng: zero link dependencies, decodes our 8-bit RGB
  Kodak PNGs deterministically).

## Build

```bash
make            # builds crown_enc crown_dec crown_train (+ hapre_c.o, crown6_c.o)
```

Speed: `crown_enc` runs the 9 (RCT,channel) search pipelines in parallel
(`std::async`, 12-core wall ~54 s/img vs ~7.2 min single-threaded, 8.1×;
`CROWN_JOBS=1` forces sequential for debugging). `make fast` builds
`crown_enc_fast` (`-O3 -march=native`, separate objects, portable binaries
untouched) at ~31 s/img (~14× total); its bytes must reproduce the portable
binary sha-identically (verified full-7). Join order is fixed and the
RCT pick uses strict `<`, so bytes are identical at any job count (verified
jobs=1/9 + pre-threading binary, sha-identical full-7). `CROWN_TIME=1` prints
a phase breakdown (timers are thread-local, behavior-preserving).

Links (read-only, never modified) `../src/hapre.c` (`pack_syms`, `rans_*`)
and `../src/crown6.c` (`crown6_golomb_pack`, `crown6_decode_ch`,
`crown6_rct_inv`) plus `../src/crown6_tanhlut.h` (frozen tanh LUT).
Needs only a C++17 compiler + g++ (`-ffp-contract=off`).

## Usage

```bash
./crown_enc <in.png> <weights_dir> <out.bin>   # weights: dir with crown6_<fn>.npz
./crown_dec <in.bin> <out.rgb>                 # raw RGB bytes, H*W*3
./crown_train <in.png> <weights_dir> [--rct C6|C27|C12|all]  # train MLP nets → .npz
```

- Encoder == reference driver with `train=False`: it **loads** frozen MLP
  weights from `<weights_dir>/crown6_<img>.npz` (all 3 RCTs × 3 channels);
  train them with `./crown_train` (pure C++, no torch) or the Python driver
  (`--train-only`). Missing/stale `.npz` aborts loudly — never silently
  retrains or falls back.
- Decoder needs no weights file (weights live in-stream per
  `src/CROWN6_FORMAT.md`, §MLP side info).
- Format spec: `src/CROWN6_FORMAT.md` (+ `src/CROWN4_FORMAT.md` for the
  shared sections). Magic `C6`, version 1.
- Deviations from reference: **none intended** — no hillclimb exists in the
  CROWN6 reference driver (only WIDE-K shortlist + exact top-3 refine),
  replicated as-is. Any byte mismatch is a bug, reported with image +
  byte offset + diagnosis (usual suspects: border rules, floor-division,
  Huffman tie-breaks, map packing order, predid mappings).
- Optimizations (all bit-exact, sha-proven full-7): single-pass fused Golomb
  costing (1 sweep vs 117, int64-exact), histogram Huff (O(n) vs sort,
  tie-break-invariant totals), dead no-MLP-track elimination in Q-coarse,
  9-way channel parallelism. See `memory/cycle1-memory.md` cycle 46.

## Correctness bar

- Encoder output **sha256-identical** to reference `.bin` on all 7 Kodak
  images; decode == original byte equality through **this** decoder.
- Loud asserts (`FATAL file:line`, abort) on: alphabet |r| > 1024, bad
  magic/version/RCT/predid/backend/ng, mlp_present violations, `p != len`,
  rANS round-trip, stale `.npz` thresholds, singular least-squares block.

## Bug-classes avoided (per `memory/cycle1-memory.md`)

Causality (zero-border causal MLP taps, never edge-replicate peeks);
framing incl. empty-group backend-0; alphabet ±1024 asserts; per-path
border doctrines (b4 vs B17 vs causal-zero); unary polarity q-ZEROs+`1`
(vendored semantics); rANS M=14 via linked objects; tanh 1-ulp via frozen
LUT include; int16-quant power-of-two scales (exact dequant); floor
division everywhere; stable sorts + strict `<` matching Python.

## Scope control

MLP training is in (`crown_train`, pure C++). No torch/Python at runtime
anywhere in `cpp/`. Provenance of every mechanism: see `src/CROWN6_FORMAT.md`
+ `TRAINER.md`.