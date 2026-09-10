# CROWN2 Stream Format (v1) — byte-level spec

Single exact bit-level codec file pair: `csrc/crown2.c` (streaming decoders +
bit packers) + `csrc/driver_crown2.py` (Python encoder, assembly, C-decode calls).
Linked read-only: `csrc/hapre.c` (ycocg, canonical-Huffman `pack_syms`, rANS M=14).
`libcrown2.so` is built from `crown2.c + hapre.c` without modifying either source.

All integers little-endian unless noted. Bit-packed fields are MSB-first
(`BitWriter.put`: accumulator shifted left, flushed high-bit-first — identical to
`driver_crown.py` map packing and `canon_tables` code emission).

## Constraints

- Image: H ≥ 2, W ≥ 2 (inherited from probe `nbhd`: 1-wide/1-tall degenerate inputs
  index out of range; Kodak scope unaffected).
- Residual alphabet: |r| ≤ 1024 everywhere; encoder asserts loudly, decoder
  returns -4. Never clipped.
- Groups per channel ng ≤ 64 (WIDE-K max 64; GRID fixed 6). Expanded groups
  (IG sub-cells, currently always identity) nX ≤ 128.

## Global header (8 bytes = 64 bits, matches campaign HEADER budget)

| bytes | content |
|---|---|
| 0–1 | magic `C` `2` |
| 2–3 | H u16le |
| 4–5 | W u16le |
| 6 | version u8 (=1) |
| 7 | flags u8 (=0; bit0 reserved for joint-chroma, unused in v1) |

## Channel blob (×3, order Y, Co, Cg — Y first so future |rY| splits are causal)

### Channel header (2 bytes)

- byte0: `[kidx:4][grid:3][fam:1]` — fam 0 = quantile (Q), 1 = energy-grid (GRID);
  kidx = index into WIDE=(2,3,4,6,9,12,18,27,36,48,64), 15 if GRID;
  grid = M-THR selector 0..7 (valid iff GRID).
- byte1: ng u8 (1..64; 6 for GRID).

### Group map

- Q: 92-byte dense bitmask (`np.packbits` of 729-bit active-key vector, 7 pad bits
  zero) + `ceil(active·gbits/8)` bytes of group ids in ascending key order,
  `gbits = ceil(log2 K)` bits each, MSB-first. No `na` field — decoder uses
  mask popcount (saves 16b/channel vs naive framing).
- GRID: 1-byte occupancy mask `occ` (bit g = group g non-empty). Group of a pixel
  is recomputed by the decoder from recon: `e = |a−c|+|b−c|`,
  `g = #{thr[i] < e}` (== `np.digitize(e, thr, right=True)`), zero side bytes.

### Per-group metadata (in expanded-group order 0..ng−1)

1. `ceil(ng·6/8)` bytes, 6 bits/group MSB-first: `[predid:4][backend:2]`.
   predid = E16 index (0 MED … 15 AVG3); backend 0=Huffman, 1=Golomb, 2=rANS.
   Empty groups (no pixels) are forced to backend 0 with no table/payload.
2. kvals: 4 bits per G-group (backend==1), in group order, MSB-first.
   Golomb k ∈ 0..12.
3. dbias: 3 bits per G-group, two's-complement in {−4..3} (stored `v&7`),
   in group order, MSB-first. Transmitted value `t = v − d`; decoder adds back.

### Huffman tables (backend-0 non-empty groups, group order)

Per group: `A u16le` + A × (`sym i16le` as residual value −1024..1024 + `len u8`).
Table cost exactly `16+A·24` bits (matches probe accounting). Single-symbol
groups use code (0, len 1). Empty/G/R groups carry no table (decoder knows
occupancy: Q from mask+cmap, GRID from `occ`).

### Huffman payload

`n u32le` + n bytes. Scan-order (row-major) symbols of all H-group pixels packed
with per-pixel group tables via C `pack_syms`. `n = 0` legal (no H pixels).

### Golomb payload

`n u32le` + n bytes. Scan-order transmitted values `t = v−d` of all G-group
pixels packed via C `golomb_pack` (per-pixel k from group). Mapping
`M = v≥0 ? 2v : −2v−1`, code = q zeros + `1` + k-bit remainder. `n = 0` legal.

### rANS groups (backend-2 groups, group order)

Per group: `count u32le` + `A u16le` + A × (`sym i16le` + `freq u16le`, M=14
normalized) + `paylen u32le` + payload. Table cost `16+A·32` bits + 64 framing
bits (count+paylen, honest overhead vs probe estimates). Payload layout
`[bytes…][4B final state LE]`, last-to-first encode (`hapre.c` verbatim).

## Decoder data flow (per channel)

Parse headers → `cmap[729]` (Q) or `grid_thr[5]` (GRID) → predid/backend/kvals/
dbias/hasn arrays → Huffman canonical codes rebuilt from (len,sym) order rule →
rANS per-group `rans_slots` + `rans_decode` into concatenated `rsyms` + `goff` →
ONE C call `crown2_decode_ch` (H/G bitstreams + R symbols, single causal scan:
neighbors → key/sign or energy → group → expert/backend → symbol → recon) →
`ycocg_inv`. Final `p == len(blob)` asserted (exact framing); `decode == original`
byte-compared by the driver.

## Provenance of every mechanism

- Q grouping/sign-flip/LOCO-365: `probe_b4_{b,c,g,j}` + `dp_refine` semantics
  (greedy equal-pixel quantile, stable-meanabs order).
- E16 experts + floor-division: `probe_b5_c.predictors_X` (C `pred_e16`
  verified bit-exact on 3 images × 3 channels × 16 experts, 20k samples each).
- 3-way backend choice + WIDE-K: `probe_b5_d` (2-way K-scan, top-3 3-way
  re-select — strict improvement on probe's 2-way-then-refine).
- G-bias shift adapters: B8 NOTE re-probe under Golomb (Huffman/rANS-invariant,
  Golomb-visible; per-group 3b gate, adopted iff `min_d cost+3 < cost`).
- GRID dictionary: `probe_b8` M-THR (D0..D3 verbatim + D4..D7 geometric variants,
  3b selector, per-channel MDL gate vs Q on exact assembled bytes).
- A-joint pairs / I-GATED: measured in-framework and RETIRED (see driver
  docstring + campaign report): pairs +7…+17% (fragmentation), gated splits
  −0.03…−0.07% (below port threshold; WIDE-K granularity already harvests it).
