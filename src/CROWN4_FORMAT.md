# CROWN4 Stream Format (v1) — byte-level spec

Single exact bit-level codec file pair: `src/crown4.c` (streaming channel
decoder + RCT inverse + Golomb packer) + `src/driver_crown4.py` (Python
encoder, assembly, C-decode calls). Linked read-only: `src/libhapre.so`
(`pack_syms`, `rans_norm/encode/slots/decode` — all verified wire-compatible;
its `golomb_pack` unary variant is NOT used, see below). `libcrown4.so` is
built from `crown4.c` alone (`gcc -O2 -fPIC -shared`). No existing file modified.

CROWN4 = CROWN3 line + B18 winners (`experiments/probe_b18_stack.py`):
LMS5_T0 / LMS5_T3 per-pixel sign-sign adaptive-FIR 5-tap experts as 18th/19th
candidates (5-bit ids already hold ≤32; ZERO side bytes; recon-only
deterministic state). GRID family is CROWN3-verbatim (E17, LMS-free); LMS
experts are Q-family-only (B18-stack parity). Per-channel Q-vs-GRID by exact
assembled bytes; global RCT min-over-{C6,C27,C12} like CROWN3.

All integers little-endian unless noted. Bit-packed fields are MSB-first
(`BitWriter.put`: accumulator shifted left, flushed high-bit-first — identical
to `driver_crown3.py` / `driver_crown2.py` map packing and `canon_tables` code
emission).

## Constraints

- Image: H ≥ 2, W ≥ 2 (inherited; Kodak scope unaffected).
- Residual alphabet: |r| ≤ 1024 everywhere; encoder asserts loudly, decoder
  returns -4. Never clipped.
- Entropy groups per channel ng ≤ 64 (WIDE-K max 64; GRID fixed 6).
- Weighted spatial groups: GS = 32 fixed (no size selector; B18-final7 picked
  G32 on 7/7). ng32 = ceil(H/32)·ceil(W/32); block index b = (i>>5)·nbw + (j>>5),
  nbw = ceil(W/32), row-major.
- Predictor ids 0..18 fit the 5-bit field (values 19..31 never emitted;
  decoder rejects predid > 18 with -5).

## Global header (8 bytes, MAGIC DIFFERS from CROWN3)

| bytes | content |
|---|---|
| 0–1 | magic `C` `4` (CROWN3 used `C3`) |
| 2–3 | H u16le |
| 4–5 | W u16le |
| 6 | version u8 (=1) |
| 7 | flags u8, bits[1:0] = RCT id (0=C6 `(perm0,t6)`, 1=C27 `(perm3,t6)`, 2=C12 `(perm1,t5)`; JXL catalogue `7·perm+t`); bits[7:2] reserved 0 |

## Channel blob (×3, order p0, p1, p2 of the selected RCT)

### Channel header (2 bytes, CROWN3-identical)

- byte0: `[kidx:4][grid:3][fam:1]` — fam 0 = quantile (Q), 1 = energy-grid (GRID);
  kidx = index into WIDE=(2,3,4,6,9,12,18,27,36,48,64), 15 if GRID;
  grid = M-THR selector 0..7 (valid iff GRID).
- byte1: ng u8 (1..64; 6 for GRID).

### W-side (Weighted groups; CROWN3-identical, placed here)

- `ceil(ng32/8)` bytes: use flags, 1 bit per G32 block in index order,
  MSB-first (`BitWriter.put(use,1)`; tail pad bits zero).
- `ceil(nused·20/8)` bytes: weights for used blocks in index order, 4 taps
  w0..w3 (stencil {L,T,TL,TR}), 5 bits each MSB-first (stored `w&31`, signed as
  `v≥16 ? v−32 : v`, range [−16,15]). 20 bits per used block.
- Encoder gate (probe_b17 verbatim): per-G32-block closed-form LS fit on causal
  {L,T,TL,TR} from the originals (= recon, lossless), ×16 rounded, clipped to
  [−16,15]; keep iff any weight nonzero and `N·log2(b_med/b_w) > 21`.
  Gated-off blocks fall back to MED on both sides, so the WAVG residual column
  equals the MED column there and ties keep MED (strict `<` selection,
  WAVG searched before LMS, LMS only wins on strict improvement).

### LMS-side (NEW vs CROWN3: none — ZERO bytes)

- No flags, no weights, no per-image header. Both LMS states init to
  [3,3,3,3,4] (sum 16) per channel and re-simulate deterministically from
  causal recon on both sides. Group selection among E19 is decoder-visible via
  the per-group predid field below (ids 17/18); the LMS trajectories themselves
  need no transmission.

### Group map (CROWN3-identical)

- Q: 92-byte dense bitmask (`np.packbits` of 729-bit active-key vector) +
  `ceil(active·gbits/8)` bytes of group ids in ascending key order,
  `gbits = ceil(log2 K)` bits each, MSB-first.
- GRID: 1-byte occupancy mask `occ` (bit g = group g non-empty). Group
  recomputed by decoder from recon: `e = |a−c|+|b−c|`, `g = #{thr[i] < e}`.

### Per-group metadata (group order 0..ng−1; SAME WIDTH as CROWN3)

1. `ceil(ng·7/8)` bytes, 7 bits/group MSB-first: `[predid:5][backend:2]`.
   predid = E19 index (0 MED … 15 AVG3, same order as CROWN2/CROWN3 E16;
   16 = WAVG; **17 = LMS5_T0; 18 = LMS5_T3**); backend 0=Huffman, 1=Golomb,
   2=rANS. (CROWN3 used 0..16; CROWN4 extends to 0..18 — the only metadata
   change besides the magic.)
   Empty groups forced to backend 0 with no table/payload.
2. kvals: 4 bits per G-group (backend==1), group order, MSB-first. k ∈ 0..12.
3. dbias: 3 bits per G-group, two's-complement {−4..3} (stored `v&7`), group
   order, MSB-first. Transmitted value `t = v−d`; decoder adds back.

### Huffman tables / payloads, Golomb payload, rANS groups (CROWN3-identical)

- Per H non-empty group: `A u16le` + A × (`sym i16le` + `len u8`); cost
  `16+A·24` bits. Single-symbol groups use code (0, len 1).
- H payload: `n u32le` + n bytes, scan-order symbols packed with per-pixel
  group tables via `pack_syms` (libhapre.so, verified bit-identical).
- G payload: `n u32le` + n bytes, transmitted `t = v−d` packed via
  `crown4_golomb_pack` (crown2/crown3-verbatim; see UNARY POLARITY below).
- rANS groups (M=14): `count u32le` + `A u16le` + A × (`sym i16le` +
  `freq u16le`) + `paylen u32le` + payload `[bytes…][4B final state LE]`,
  via libhapre.so `rans_*` (per-group round-trip asserted at encode time).

## UNARY POLARITY (wire — explicit, differs from libhapre era-variant)

- CROWN4 Golomb-Rice code (all G groups, both families):
  mapping `M = v≥0 ? 2v : −2v−1`; code = **q ZERO bits, one `1`, then the
  k-bit remainder MSB-first** (total  q+1+k bits/symbol).
  Decoder (`crown4_decode_ch` G-branch): counts `0` bits until the first `1`
  (`while bit==0 q++; expect 1`), then reads k remainder bits MSB-first,
  `M=(q<<k)|rem`, `v = (M&1) ? −((M+1)>>1) : (M>>1)`, plus `dbias`.
- libhapre.so's `golomb_pack` emits the opposite polarity (**q ONES + a `0`**)
  and is NOT wire-compatible (verified divergent same-length/different-bits
  2026-09-07). It is linked for H/rANS only and NEVER called for G payloads —
  hence the vendored `crown4_golomb_pack`, not a link. Packer chunking
  (24-zero bounded shifts) is an implementation detail; bits are identical.

## Decoder data flow (per channel)

Parse headers → W-side (`wuse[nb32]`, `ww[nb32][4]`) → `cmap[729]` (Q) or
`grid_thr[5]` (GRID) → predid/backend/kvals/dbias/hasn arrays (predid ≤ 18
asserted; >18 returns -5) → Huffman canonical codes rebuilt from (len,sym)
order rule → rANS per-group decode into `rsyms` + `goff` → ONE C call
`crown4_decode_ch` (single causal scan: neighbors → key/sign or energy →
group → expert/backend → symbol → recon; WAVG stencil + LMS state re-derived
from recon) → `crown4_rct_inv` (exact port of probe_b17 `rct_inv` for the 3
carried entries; returns nonzero on out-of-range, asserted 0). Final
`p == len(blob)` asserted (exact framing); `decode == original` byte-compared
by the driver.

## Border conventions (encoder == decoder; dual-neighborhood audit)

- E16 neighborhoods / LOCO-365 keys / GRID energy: b4_b-style via `c4_nbhd`
  (crown3-verbatim; NEVER edge-replicate):
  zero border; row0 copies left (`B=A,C=A`); col0 copies top (`A=B,C=B`);
  `TR` copies `L` on row0 and copies `T` in last column; `Ww` copies `L` for
  j<2; `NNe` copies `T` for i<2; `NE == TR`.
- Weighted stencil (probe_b17 `nbhd4`-exact) and LMS 5th-tap MED: B17-style:
  `L = (j>0)?P[i,j−1] : ((i>0)?P[i−1,0] : 0)`;
  `T = (i>0)?P[i−1,j] : ((j>0)?P[i,j−1] : 0)`;
  `TL0 = (i>0&&j>0)?P[i−1,j−1] : 0`; `TR0 = (i>0&&j+1<W)?P[i−1,j+1] : 0`.
  `L/T` are numerically identical in both doctrines; ONLY `TL/TR` split
  (the wire-split: E16/keys use copy-L, WAVG/LMS-tap use zero).
- LMS stencil: `xs = [L,T,TL,TR (b4), MED5 (B17)]` with
  `MED5 = MED(L,T,TL0)`; `m = floor((L+T+TL+TR+MED5+2)/5)`;
  `pred = floor((w·xs+8)/16)` (C `c4_fdiv` matches numpy `//`; verified by
  crop + full-image round-trips with LMS-selected groups).
- RCT forward (encoder, probe_b17 `rct_fwd` verbatim) / inverse (C,
  `crown4_rct_inv`): cross-checked C-vs-numpy on random + saturated-edge
  inputs for all 3 ids before any measurement.
- Audit trail (B18 proof): two real border bugs caught+fixed — (1) sign-flip
  key negation must NEGATE (`−q`), not `abs()`; (2) TL-border doctrine above.
  CROWN4 preserves the fixed decoder exactly as the Python spec does.

## LMS5 exact semantics (ZERO-side adaptive-FIR)

- Per channel, two states `w0` (thr 0) / `w3` (thr 3), init `[3,3,3,3,4]`.
- Per pixel in raster order: predict with current state (if selected), then
  BOTH states advance using the true recon value (whether or not selected):
  `e = y−pred`; if `|e|>thr`: `se=sign(e)`;
  `w[i] += se` if `xs[i]>m`, `w[i] −= se` if `xs[i]<m`;
  clamp `[-8,20]`; sum-renorm to 16 (decrement current max / increment current
  min, lowest-index tie-break — matches Python `max(key=(w,−i))` /
  `min(key=(w,i))`).
- Encoder (`probe_b18_stack.lms_pred_plane` verbatim, imported not
  reimplemented) predicts whole planes then groups; decoder re-simulates after
  each recon. Any 1-LSB divergence avalanches — asserted round-trips on crops
  with LMS-selected groups plus all-7 full images prove bit-exactness.

## Provenance of every mechanism

- Q grouping/sign-flip/LOCO-365, E16 experts + floor-division, 3-way backends,
  WIDE-K, G-bias, GRID dictionary, W-side, RCT bank, Golomb polarity: CROWN3
  line, unchanged (see `CROWN3_FORMAT.md` + driver docstring).
- LMS5_T0/T3 gated 18th/19th experts: probe_b18 (`probe_b18_stack.py`
  `lms_pred_plane`/`lms_step`/`decode_channel`, `probe_b18_final7.py` config:
  GS-min×RCT-min, LMS-only stack, 7/7 gains p=.016, no regressions).
- Deviations from CROWN3: magic `C4` (framing disambiguation); predid range
  extended 16→18 (same 7-bit field, no width change); LMS-side zero bytes;
  GS fixed 32 (B18-final7 picked G32 7/7 — no GS selector bit, unlike the
  probe's +1b GS-side ledger entry); GRID candidates stay E17 (LMS-free).
  Probe B8 framing (`B8` magic, `Q` tag + `nkf` byte, 365B ERR4 masks) is NOT
  used — CROWN4 reuses the tighter CROWN3 framing, so CROWN4 bytes run
  ~7B/image SMALLER than the probe ledger at identical decisions (conservative
  direction; within the ±1% bar by construction).