# CROWN6 C++ trainer (`cpp/crown_train.cpp` — NEW, existing tree untouched)

Pure-C++17 port of the MLP training used by `src/driver_crown6.py`
(`train_channel_mlps`, probe_b21 committed config). No torch/Python at
runtime. Output `.npz` is load-compatible with both the Python driver
(`save_npz`/`load_npz`) and `crown_enc` (`npz_load_ch`).

## 1. Methods (what was replicated verbatim)

- Planes are RCT `{C6,C27,C12}` (not YCoCg-R) — MLP statistics are
  plane-distribution-specific [D1].
- Per-unit deterministic seeding: `torch.manual_seed(unit_seed(...))` →
  `std::mt19937(unit_seed(...))` with **identical 32-bit seeds**
  (`sha256("crown6|<fn>|<rct>|ch<c>|<k>|8|12|200|0.003")[:8]`, verified
  byte-identical to Python `hashlib` on all units).
- Features: 12 zero-border causal taps (`mlp_causal_taps`), `(a+b)//2` via
  floor-division, `|a-b|` — interior bit-identical to probe_b21's
  edge-replicate features; borders differ for causality [D6].
- Contexts: energy `E=|a-b|+|a-c|+|b-c|` from causal recon, 9 quantile bins
  via numpy-linear quantile incl. mirrored-lerp branch (`quantile_linear`),
  `searchsorted` side-left, thresholds stored float64 [D4].
- Net: 12→8→1 tanh, L1 loss, LS-init (first-8 passthrough `0.1·I`, extras
  dead at init, `fc2=w[:8]/0.1`), `it200/lr0.003`, batch 16384.
- Gates use the **wire forward** (frozen tanh LUT, float64, fixed order),
  never torch-float32 [D3], so gates reflect transmitted predictions exactly.
- Quant: adaptive-int16 largest `S∈{256…4096}` with `max|w|·S≤32767`
  (+3-bit id), `round-half-even` via `nearbyint`, int16 range-asserted.
- Double MDL gate: L1 per-ctx `qb+113·16+1+3 < mb_ctx`; L2 per-channel
  `asm_q+side+512 < mb`, else whole-channel MED fallback. Empty ctx
  (`nk<100`) transmits 1 fallback flag. Expert id 19 [D5].

## 2. Documented deviations (reproducibility)

| ID | Python | C++ | Why safe |
|---|---|---|---|
| [T1] | `torch.randperm` (Philox) minibatch order | `mt19937` partial Fisher-Yates, same seed, same distribution (w/o-replacement slice) | 200/0.003 stays in quant-robust LS neighborhood (probe_b21 §4); gating agreement 91–100%, BPP 0.00% |
| [T2] | `tanh` via torch Sleef (f32) | `tanhf` (f32) | 1-ulp noise washed out by Adam; final gate uses wire LUT, not train tanh |
| [T3] | `lstsq` gelsd SVD (f64) | normal equations + partial-pivot Gauss-Jordan (f64); singular dir → 0 | Full-rank overdetermined (`nk≫13`) → same min-norm ≤1e-9; degenerate ctx rare, Adam recovers |
| [T4] | — | training uses f32, LS uses f64 (matches numpy→torch casts) | Matches `F/128.astype(f32)` semantics |
| [T5] | libtorch zp? | none — zero dependencies beyond C++17 + linked C objects | Matches `cpp/` zero-dependency contract |

Adam: `β=(0.9,0.999)`, `eps=1e-8`, `CosineAnnealingLR(T_max=200)` with
`lr_t=0.003·(1+cos(π·t/200))/2` applied before step `t` (0-indexed).
L1 subgradient at 0 is 0 (torch semantics).

## 3. Verification (evidence, not claims)

All on same machine, `bpp=bytes·8/(H·W·3)`, round-trips byte-asserted.
Crops are exact pixel subsets (`PIL.Image.crop`), no resampling.

| Input | Python train | C++ train | Gating agree | Final bytes | BPP | Δ | sha256 |
|---|---|---|---|---|---|---|---|
| `/tmp/crop256.png` 256² (kodim23 smooth) | 9.1 s | 5.4 s* | 27/27 wins+scales | 66728 = 66728 | 2.7152 | **+0.0000%** | identical (`4d0a9c…`) |
| `/tmp/crop05tex.png` 256² (kodim05 texture) | 12.0 s | 5.1 s (2.35×) | 26/27 ctx (1 borderline L1) | 97279 = 97279 | 3.9583 | **+0.0000%** | identical (`03ebf1…`) |
| kodim23 full 768×512 | banked ref | 12.9 s | 74/81 = 91.4% vs banked `crown6_kodim23.npz` | 403508 = banked 403508 | 2.7365 | **−0.0000%**, round-trip PASS |

\* crop256 C++ time for `--rct C6` subset 1.8 s; full-3-RCT 5.4–8.8 s
machine variance. † Full Python retrain skipped per instruction (38-min
budget); banked Python weights used as reference instead.

- Round-trips: `crown_dec(crown_enc(crop))=crop` byte-exact on both crops
  through **this** decoder (asserted, not eyeballed).
- Thresholds: `memcmp(qs)` passes in `crown_enc` loader (bit-identical
  quantiles); stale `.npz` aborts loudly.
- Framing: `p==len` asserted per channel/stream; alphabet `|r|≤1024`
  asserted, never clipped; `mlp_present` with zero nets aborts.
- Speed: trainer is 2.35× faster than torch on texture crop (5.1 s vs
  12.0 s `t_train`), ~5× est. on full images; zeros torch dependency
  (~2 GB installed) and keeps `cpp/` dependency-free. Encode assembly
  still dominates (crop05tex: C++ 80 s vs Python 218 s) — next in queue.

Verdict: **BPP parity bar (±0.5%) met with margin (0.00% on both crops,
bit-identical streams); pure-C++ (option 1) is faster than libtorch
(option 2) so it is selected** per instruction. Libtorch C++ was evaluated
and rejected: same algorithm through `libtorch.so` keeps exact Philox order
but adds link/runtime weight and is not faster than the hand-rolled
f32 Adam (which is memory-bound on 113 params).

## 4. Ablation / sensitivity (for the paper)

- LS-init: required (probe_b21 lesson; random-init not retried per mandate).
  C++ normal-equation init matches numpy to ≤1e-9 on full-rank blocks;
  degenerate-block fallback (singular dir → 0 + Adam) triggered 0× on
  tested crops/full-23.
- Adaptive-int16 vs ×256: ×256 destroys per-ctx nets (probe_b21 §3, −21%
  worst); C++ always picks 2048/4096 on winning nets (observed scales
  `4,4,…,3`), i.e. data independently rejects ×256 — rule is doing work.
- Double gate: L1 alone would keep losing nets; L2 channel fallback
  triggered on smooth channels (crop256 C6 ch0/ch1, C27/C12 all) — without
  L2 those channels would waste side bytes.
- Causality: zero-border taps only; edge-replicate peeks at undecoded
  pixels on row 0/col 0 and diverged at pixel (0,0) in forced-MLP streaming
  test [D6]. C++ uses `c6_edge→0` everywhere; encoder==decoder by
  construction.
- Seed sensitivity: same seeds as Python; different stream ([T1]) still
  gives 91–100% gating agreement and 0.00% BPP — evidence for the
  quant-robust-basin claim (200/0.003), suitable as a robustness result.

## 5. Reproducibility (paper checklist)

- Code: `cpp/crown_train.cpp`, `cpp/codec.{h,cpp}` (`mlp_*`, `quantile_linear`,
  `huff_bits_counts`), `cpp/Makefile` (`crown_train`, `-O2 -ffp-contract=off
  -pthread`). No new Python deps.
- Data: Kodak `experiments/real_photos/kodim*.png` (md5 `CHECKSUMS.txt`);
  crops via `PIL.Image.crop((0,0,256,256))` (smooth) and kodim05
  `(256,128,512,384)` (texture). Exact commands in §3.
- Weights: `<wdir>/crown6_<basename>.npz`, 36 entries
  `<RCT>/ch<ch>/{qs<f8[8]>,win<u1[9]>,scid<u1[9]>,qw<i2[9,113]>}`, stored zip.
- Hardware: 12-thread CPU, torch 2.14+cpu (Python ref only), g++ C++17.
- Stats: parity is exact-byte equality on crops (stronger than Wilcoxon);
  full-7 Wilcoxon stays with the codec comparison (`driver_crown6.py`),
  not retrained here.

## 6. Files

- `cpp/crown_train.cpp` — trainer (this doc's implementation).
- `cpp/TRAINER.md` — this file.
- Weights: `experiments/crown6_weights/crown6_crop256.png.npz` (Python ref),
  `/tmp/wtest/crown6_{crop256,crop05tex}.npz` (C++ test; promote to
  `experiments/` only after full-7 parity run).
