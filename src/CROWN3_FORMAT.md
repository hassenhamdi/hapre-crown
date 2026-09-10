# CROWN3 Stream Format (v1) — byte-level spec

Single exact bit-level codec file pair: `csrc/crown3.c` (streaming channel
decoder + RCT inverse + Golomb packer) + `csrc/driver_crown3.py` (Python
encoder, assembly, C-decode calls). Linked read-only: `csrc/libhapre.so`
(`pack_syms`, `rans_norm/encode/slots/decode` — all verified wire-compatible;
its `golomb_pack` unary variant is NOT used, see below). `libcrown3.so` is
built from `crown3.c` alone (`gcc -O2 -fPIC -shared`). No existing file modified.

CROWN3 = CROWN2 line + Attack-1 winners (`experiments/probe_b17_rctw.py`):
global RCT over {C6,C27,C12} + Weighted-4tap 17th expert on fixed G32 groups.
Per-block RCT explicitly dead (probe M2/M4) — not in this format.

All integers little-endian unless noted. Bit-packed fields are MSB-first
(`BitWriter.put`: accumulator shifted left, flushed high-bit-first — identical
to `driver_crown2.py` / `driver_crown.py` map packing and `canon_tables` code
emission).

## Constraints

- Image: H ≥ 2, W ≥ 2 (inherited; Kodak scope unaffected).
- Residual alphabet: |r| ≤ 1024 everywhere; encoder asserts loudly, decoder
  returns -4. Never clipped.
- Entropy groups per channel ng ≤ 64 (WIDE-K max 64; GRID fixed 6).
- Weighted spatial groups: GS = 32 fixed (no size selector; probe M3: G32 beats
  G64 7/7). ng32 = ceil(H/32)·ceil(W/32); block index b = (i>>5)·nbw + (j>>5),
  nbw = ceil(W/32), row-major.

## Global header (8 bytes)

| bytes | content |
|---|---|
| 0–1 | magic `C` `3` |
| 2–3 | H u16le |
| 4–5 | W u16le |
| 6 | version u8 (=1) |
| 7 | flags u8, bits[1:0] = RCT id (0=C6 `(perm0,t6)`, 1=C27 `(perm3,t6)`, 2=C12 `(perm1,t5)`; JXL catalogue `7·perm+t`); bits[7:2] reserved 0 |

## Channel blob (×3, order p0, p1, p2 of the selected RCT)

### Channel header (2 bytes, CROWN2-identical)

- byte0: `[kidx:4][grid:3][fam:1]` — fam 0 = quantile (Q), 1 = energy-grid (GRID);
  kidx = index into WIDE=(2,3,4,6,9,12,18,27,36,48,64), 15 if GRID;
  grid = M-THR selector 0..7 (valid iff GRID).
- byte1: ng u8 (1..64; 6 for GRID).

### W-side (Weighted groups; NEW vs CROWN2, placed here)

- `ceil(ng32/8)` bytes: use flags, 1 bit per G32 block in index order,
  MSB-first (`BitWriter.put(use,1)`; tail pad bits zero).
- `ceil(nused·20/8)` bytes: weights for used blocks in index order, 4 taps
  w0..w3 (stencil {L,T,TL,TR}), 5 bits each MSB-first (stored `w&31`, signed as
  `v≥16 ? v−32 : v`, range [−16,15]). 20 bits per used block — matches the
  probe ledger (1b flag + 20b/used; no size-sel bit since GS is fixed).
- Encoder gate (probe_b17 verbatim): per-G32-block closed-form LS fit on causal
  {L,T,TL,TR} from the originals (= recon, lossless), ×16 rounded, clipped to
  [−16,15]; keep iff any weight nonzero and `N·log2(b_med/b_w) > 21`
  (= 1b flag + 20b weights MDL). Gated-off blocks fall back to MED on both
  sides (C: `predid==16 && !wuse → pred_e16(0,…)`), so the WAVG residual column
  equals the MED column there and E16 ties keep MED (strict `<` selection,
  WAVG searched last).

### Group map (CROWN2-identical)

- Q: 92-byte dense bitmask (`np.packbits` of 729-bit active-key vector) +
  `ceil(active·gbits/8)` bytes of group ids in ascending key order,
  `gbits = ceil(log2 K)` bits each, MSB-first.
- GRID: 1-byte occupancy mask `occ` (bit g = group g non-empty). Group
  recomputed by decoder from recon: `e = |a−c|+|b−c|`, `g = #{thr[i] < e}`.

### Per-group metadata (group order 0..ng−1; WIDTH CHANGED vs CROWN2)

1. `ceil(ng·7/8)` bytes, 7 bits/group MSB-first: `[predid:5][backend:2]`.
   predid = E17 index (0 MED … 15 AVG3, same order as CROWN2 E16; 16 = WAVG);
   backend 0=Huffman, 1=Golomb, 2=rANS. (CROWN2 used 6 bits: 4+2.)
   Empty groups forced to backend 0 with no table/payload.
2. kvals: 4 bits per G-group (backend==1), group order, MSB-first. k ∈ 0..12.
3. dbias: 3 bits per G-group, two's-complement {−4..3} (stored `v&7`), group
   order, MSB-first. Transmitted value `t = v−d`; decoder adds back.

### Huffman tables / payloads, Golomb payload, rANS groups (CROWN2-identical)

- Per H non-empty group: `A u16le` + A × (`sym i16le` + `len u8`); cost
  `16+A·24` bits. Single-symbol groups use code (0, len 1).
- H payload: `n u32le` + n bytes, scan-order symbols packed with per-pixel
  group tables via `pack_syms` (libhapre.so, verified bit-identical to the
  libcrown2.so copy on random data).
- G payload: `n u32le` + n bytes, transmitted `t = v−d` packed via
  `crown3_golomb_pack` (vendored crown2.c-verbatim: mapping
  `M = v≥0 ? 2v : −2v−1`, code = q zeros + `1` + k-bit remainder).
  Rationale: libhapre.so's `golomb_pack` emits q ones + zero (era variant,
  verified divergent same-length/different-bits 2026-09-07) and is NOT wire
  compatible with the CROWN reader — hence vendored, not linked.
- rANS groups (M=14): `count u32le` + `A u16le` + A × (`sym i16le` +
  `freq u16le`) + `paylen u32le` + payload `[bytes…][4B final state LE]`,
  via libhapre.so `rans_*` (encode and decode use the same library in this
  driver; per-group round-trip asserted at encode time).

## Decoder data flow (per channel)

Parse headers → W-side (`wuse[nb32]`, `ww[nb32][4]`) → `cmap[729]` (Q) or
`grid_thr[5]` (GRID) → predid/backend/kvals/dbias/hasn arrays → Huffman
canonical codes rebuilt from (len,sym) order rule → rANS per-group decode into
`rsyms` + `goff` → ONE C call `crown3_decode_ch` (single causal scan:
neighbors → key/sign or energy → group → expert/backend → symbol → recon;
Weighted stencil L,T,TL,TR re-derived from recon with probe_b17 borders) →
`crown3_rct_inv` (exact port of probe_b17 `rct_inv` for the 3 carried entries;
returns nonzero on out-of-range, asserted 0). Final `p == len(blob)` asserted
(exact framing); `decode == original` byte-compared by the driver.

## Border conventions (encoder == decoder)

- E16 neighborhoods: zero border; row0 copies left; col0 copies top
  (`c3_nbhd`, crown2.c-verbatim; NEVER edge-replicate).
- Weighted stencil (probe_b17 `nbhd4`-exact):
  L = (j>0)?P[i,j−1] : ((i>0)?P[i−1,0] : 0);
  T = (i>0)?P[i−1,j] : ((j>0)?P[i,j−1] : 0);
  TL = (i>0&&j>0)?P[i−1,j−1] : 0; TR = (i>0&&j+1<W)?P[i−1,j+1] : 0.
- Weighted eval: `pred = floor((w0·L+w1·T+w2·TL+w3·TR+8)/16)` (C `c3_fdiv`
  matches numpy `//`; verified by full-image round-trips).
- RCT forward (encoder, probe_b17 `rct_fwd` verbatim) / inverse (C,
  `crown3_rct_inv`): cross-checked C-vs-numpy on random + saturated-edge
  inputs for all 3 ids before any measurement.

## Provenance of every mechanism

- Q grouping/sign-flip/LOCO-365, E16 experts + floor-division, 3-way backends,
  WIDE-K, G-bias, GRID dictionary: CROWN2 line, unchanged (see
  `CROWN2_FORMAT.md` + driver docstring).
- Global RCT {C6,C27,C12} + homogeneous-planes rule: probe_b17 M1/M4
  (−1.41% MED-frame, 6/7 C27 + C12-on-02; per-block RCT dead, not built).
- Weighted-4tap G32 LS ×16 MDL-gated 17th expert: probe_b17 M3 (−2.15%
  MED-frame, G32 7/7, edge-sharpener kernels). Deviations: GS fixed 32 (no
  1b size-sel — G64 never won); gate threshold kept at 21 (marginally
  conservative without the size-sel bit — honest, documented); weights packed
  5b×4 (probe ledger) not byte-padded.
