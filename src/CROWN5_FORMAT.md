# CROWN5 Stream Format (v1) — byte-level spec

Single exact bit-level codec file pair: `csrc/crown5.c` (streaming adaptive
channel decoder + RCT inverse) + `csrc/driver_crown5.py` (Python encoder,
assembly, C-decode calls). Linked read-only: `csrc/libcrown4.so`
(`crown4_decode_ch` for static channels — verified wire-compatible CROWN4
suffix) + `csrc/libhapre.so` (`rans_norm/encode/slots/decode` via the CROWN4
module handle — static rANS groups only). `libcrown5.so` is built from
`crown5.c` alone (`gcc -O2 -fPIC -shared`). No existing file modified.

CROWN5 = CROWN4 predictor/group front end VERBATIM + per-channel best-of-3
entropy (probe_b19 follow-up #2: per-plane best-of-{static, C-gol, A-act}):

- Front end (CROWN4-identical, reused by import not reimplementation):
  GLOBAL best-RCT over {C6,C27,C12} (2b flags, homogeneous); E19 bank
  (E16 + WAVG G32 LS-4tap x16 MDL>21 + LMS5_T0/T3 recon-only sum-16 renorm);
  Q-family sign-flipped LOCO-365 + WIDE-K quantile vs GRID M-THR energy;
  per-channel (family x entropy) joint selection by EXACT assembled bytes.
- Entropy per channel (choice byte, 2b of 8b used, rest reserved 0):
  choice 0 = A-act GLOBAL raster-order adaptive (probe-verbatim states);
  choice 1 = C-gol GLOBAL raster-order adaptive (probe-verbatim states);
  choice 2 = static CROWN4-verbatim (per-group H/G/R + kvals/dbias + tables).
  Best-of guarantees `len(CROWN5-ch) <= len(CROWN4-ch)+1B` (choice byte);
  image best-of over RCT likewise. No transmitted init (B-fwd killed);
  no runs (D-run killed); no unconditioned adaptation (A-pos killed).

All integers little-endian unless noted. Bit-packed fields MSB-first
(`BitWriter.put`: accumulator shifted left, flushed high-bit-first — identical
to `driver_crown4.py` map packing and `canon_tables` code emission; adaptive
Golomb unary uses the same CROWN polarity: q ZEROS + one 1 + k-bit rem).

## Constraints

- Image: H >= 2, W >= 2 (inherited; Kodak scope unaffected).
- Residual/symbol alphabet: |r| <= 1024 everywhere; encoder asserts loudly,
  both decoders return -4. Never clipped.
- Entropy groups per channel ng <= 64 (WIDE-K max 64; GRID fixed 6).
- Weighted spatial groups: GS = 32 fixed (no size selector). ng32 =
  ceil(H/32)*ceil(W/32); block index b = (i>>5)*nbw + (j>>5), nbw = ceil(W/32).
- Predictor ids 0..18 fit the 5-bit field (values 19..31 never emitted;
  both decoders reject predid > 18 with -5). Static backend ids 0..2
  (H/G/R); adaptive acodec 0..1; choice 0..2 (else -5).

## Global header (8 bytes, MAGIC DIFFERS from CROWN4)

| bytes | content |
|---|---|
| 0–1 | magic `C` `5` (CROWN4 used `C4`) |
| 2–3 | H u16le |
| 4–5 | W u16le |
| 6 | version u8 (=1) |
| 7 | flags u8, bits[1:0] = RCT id (0=C6 `(perm0,t6)`, 1=C27 `(perm3,t6)`, 2=C12 `(perm1,t5)`); bits[7:2] reserved 0 |

## Channel blob (x3, order p0, p1, p2 of the selected RCT)

### Channel header (3 bytes; byte2 NEW vs CROWN4)

- byte0: `[kidx:4][grid:3][fam:1]` — fam 0 = quantile (Q), 1 = energy-grid
  (GRID); kidx = index into WIDE=(2,3,4,6,9,12,18,27,36,48,64), 15 if GRID;
  grid = M-THR selector 0..7 (valid iff GRID). CROWN4-identical.
- byte1: ng u8 (1..64; 6 for GRID). CROWN4-identical.
- byte2: choice u8 (0=A-act, 1=C-gol, 2=static-CROWN4). NEW. Decoder rejects
  >2 with -5. Cost 8b/channel (the only overhead vs CROWN4 on static channels).

### W-side (Weighted groups; CROWN4-identical, shared both paths)

- `ceil(ng32/8)` bytes: use flags, 1 bit per G32 block in index order,
  MSB-first (tail pad bits zero).
- `ceil(nused*20/8)` bytes: weights for used blocks in index order, 4 taps
  w0..w3 (stencil {L,T,TL,TR}), 5 bits each MSB-first (stored `w&31`, signed
  as `v>=16 ? v-32 : v`, range [-16,15]). 20 bits per used block.
- Encoder gate (probe_b17 verbatim): per-G32-block closed-form LS fit on causal
  {L,T,TL,TR} from the originals (= recon, lossless), x16 rounded, clipped to
  [-16,15]; keep iff any weight nonzero and `N*log2(b_med/b_w) > 21`.
  Gated-off blocks fall back to MED on both sides (strict `<` selection).

### LMS-side (ZERO bytes; CROWN4-identical)

- No flags/weights/header. Both LMS states init to [3,3,3,3,4] (sum 16) per
  channel and re-simulate deterministically from causal recon on both sides.
  Group selection among E19 decoder-visible via per-group predid.

### Group map (CROWN4-identical, shared both paths)

- Q: 92-byte dense bitmask (`np.packbits` of 729-bit active-key vector) +
  `ceil(active*gbits/8)` bytes of group ids in ascending key order,
  `gbits = ceil(log2 K)` bits each, MSB-first.
- GRID: 1-byte occupancy mask `occ` (bit g = group g non-empty). Group
  recomputed by decoder from recon: `e = |a-c|+|b-c|`, `g = #{thr[i] < e}`.

### Per-group metadata + payloads (branch on choice)

- choice 0/1 (adaptive): `ceil(ng*5/8)` bytes, 5 bits/group MSB-first predid
  0..18 (E19 index: 0 MED .. 15 AVG3, 16 WAVG, 17 LMS5_T0, 18 LMS5_T3).
  No backend/kvals/dbias/tables. Then:
  - acodec 0 (A-act): `alen u32le` + alen arith bytes + `rlen u32le` + rlen
    raw bytes. Arith = 16-ctx adaptive binary range coder bytes (see below);
    raw = Exp-Golomb suffix bits MSB-first, tail-padded with zeros.
    Framing counted exactly (probe: 64b lengths; here 2x u32le).
  - acodec 1 (C-gol): `paylen u32le` + paylen Golomb bytes (single raster-order
    bitstream, q ZEROS + 1 + k rem MSB-first per symbol, tail-padded zeros).
- choice 2 (static): CROWN4-verbatim suffix starting immediately after byte2:
  `ceil(ng*7/8)` bytes 7 bits/group `[predid:5][backend:2]` MSB-first
  (backend 0=H,1=G,2=R; empty groups forced backend 0, no table/payload);
  kvals 4b/G-group + dbias 3b/G-group (two's-complement {-4..3}, `v&7`);
  per H non-empty group `A u16le` + A x (`sym i16le` + `len u8`);
  H payload `n u32le` + n bytes via `pack_syms`; G payload `n u32le` + n bytes
  via CROWN-polarity `crown4_golomb_pack` (q ZEROS + 1 + k rem; libhapre
  era-variant NOT used); rANS groups (M=14) `count u32le` + `A u16le` + A x
  (`sym i16le` + `freq u16le`) + `paylen u32le` + payload + 4B final state.

## Adaptive states (probe_b19_adaptive.py-verbatim; pure function of decoded data)

- Activity: `e = |lv-tl|+|tv-tl|` from causal recon only, B17-style borders
  (lv = left else above-col0 else 0; tv = top else left else 0; tl = diag else
  0); bins `ab = 0/1/2/3` by static thresholds ATHR=(4,12,48). Known both sides.
- A-act: 16 binary contexts `ctx = ab*4 + pos` (pos = prefix index 0..2, 3+
  clamped; NPOS=4). Per-ctx (c0,c1) init (1,1); update `c_bit++`, halving to
  `(c+1)//2` (floor 1) when `c0+c1 >= 128` (MAXT). Binarization: `M =
  2|r|-(r>0)` (r=0->0, 1->1, -1->2, ...), `cn = M+1`, `L = cn.bit_length()`;
  prefix L bins (L-1 zeros with pos ctx, then 1); suffix `rem = cn ^
  (1<<(L-1))` as L-1 raw bits MSB-first. Encoder `BinEnc` / decoder `BinDec`
  16-bit E1/E2/E3 (`rng = high-low+1`, `split = (rng*c0)//tot` clamp
  1..rng-1, follow bits, `finish: follow+=1, out(0) iff low<0x4000`,
  decoder `_read` past EOF = 0, `L<=16` else -6). States updated identically
  per bin in raster order — proven by asserted real decodes.
- C-gol: 4 contexts `ctx = ab`. Per-ctx (N,A) init (1,4); `k` from `(N<<k)>=A`
  loop; code `M` with `q = M>>k` zeros + 1 + k rem bits MSB-first (CROWN
  polarity); update `A += |r|`, `N += 1`, halving `N>>=1,A>>=1` (floor N>=1)
  when `N >= 64` (RESET). Raster order, init/RESET static. Analytic lengths ==
  emitted bits asserted at encode time (`golomb_adaptive_encode` cross-check).
- Sign-flip (Q only): LOCO `sg = -1` iff `(q1<0||(q1==0&&q2<0)||(q1==0&&q2==0&&
  q3<0))` with `qi = loco_q1(gradients)` (negation, NOT abs — B18 bug #1);
  stored symbol `sym = s*res`, reconstructed `yv = p + sg*sym` (GRID sg=+1).
  `sg` recomputed from recon on both sides (decoder-visible).
- LMS/WAVG recon-only simulation identical to CROWN4 (states advance EVERY
  pixel in raster order regardless of selection; any 1-LSB divergence
  avalanches — asserted round-trips prove bit-exactness).

## UNARY POLARITY (wire — explicit)

- All Golomb payloads (static G via `crown4_golomb_pack`, adaptive C-gol via
  `golomb_adaptive_encode`): mapping `M = 2|r|-(r>0)` for adaptive
  (probe convention: 0->0, 1->1, -1->2, 2->3, -2->4) and `M = v>=0 ? 2v :
  -2v-1` for static-G (CROWN convention — a bijection differing only in
  odd/even assignment; each path self-consistent, documented here);
  code = **q ZERO bits, one `1`, then k-bit remainder MSB-first**.
  libhapre.so's `golomb_pack` (q ONES + 0) is NEVER used for either.

## Decoder data flow (per channel, branch on choice)

Parse headers -> W-side -> cmap[729] (Q) or grid_thr[5] (GRID) -> choice:
- adaptive (0/1): 5b predids -> ONE C call `crown5_decode_ch` (single causal
  scan: neighbors -> key/sign or energy -> group -> expert pred + activity bin
  -> adaptive symbol -> recon; WAVG/LMS/adaptive states re-derived) with
  arith/raw or gbuf payloads.
- static (2): 7b predid/backend + kvals/dbias + H tables + H/G pays + rANS
  pre-decode via libhapre (identical to driver_crown4) -> ONE C call
  `crown4_decode_ch` (libcrown4.so, read-only link) with same recon/pred/LMS
  semantics as CROWN4.
Then `crown5_rct_inv` (crown4-exact port for the 3 carried ids; nonzero =
out-of-range, asserted 0). Final `p == len(blob)` asserted (exact framing);
`decode == original` byte-compared by the driver (loud asserts, never silent).

## Border conventions (encoder == decoder; CROWN4-verbatim)

- E16 neighborhoods / LOCO-365 keys / GRID energy: b4_b-style via `c5_nbhd`
  (crown4 `c4_nbhd`-verbatim): zero border; row0 copies left; col0 copies top;
  `TR` copies `L` on row0 and copies `T` in last column; `Ww` copies `L` for
  j<2; `NNe` copies `T` for i<2; `NE == TR`.
- Weighted stencil / LMS 5th-tap MED / LMS m / activity lv/tv/tl: B17-style
  (`L = (j>0)?P[i,j-1] : ((i>0)?P[i-1,0] : 0)`,
  `T = (i>0)?P[i-1,j] : ((j>0)?P[i,j-1] : 0)`,
  `TL0/TR0 = (i>0&&j>0)?P[i-1,j-1] : 0` else 0; activity tl likewise).
  `L/T` numerically identical in both doctrines; ONLY `TL/TR` split.
- LMS stencil `xs = [L,T,TL,TR (b4), MED5 (B17)]`, `m =
  floor((L+T+TL+TR+MED5+2)/5)`, `pred = floor((w.xs+8)/16)` via `c5_fdiv`
  (matches numpy `//`; verified by crop + full-image round-trips).
- RCT forward (probe_b17 `rct_fwd` verbatim) / inverse (`crown5_rct_inv`,
  cross-checked C-vs-numpy on random + saturated-edge inputs).

## Provenance and deviations (explicit)

- Predictors/groups/RCT/W-side/LMS/borders/static suffix: CROWN4 line
  unchanged (see `CROWN4_FORMAT.md` + driver docstring); reused by importing
  `driver_crown4` (decisions + static assembly) and linking `libcrown4.so`
  (static-channel decode) read-only. Nothing existing modified.
- Adaptive states/arithmetic: `experiments/probe_b19_adaptive.py` verbatim
  (`BinEnc/BinDec`, `GolombState`, `cabac_encode_plane` binarization,
  `ATHR/NPOS/MAXT/RESET/init`, CROWN-polarity real-stream writer proven by
  `golomb_adaptive_encode` analytic==stream assert + 21/21 plane round-trips
  in probe + CROWN5 crop/full-image round-trips here). Killed designs NOT
  shipped (B-fwd/D-run/A-pos, see driver docstring).
- Deviation (documented, reasoned): pure-global-adaptive (no static fallback)
  was built first and LOSES to per-group static on all Kodak-7 channels
  tested (e.g. kodim07 2.8935 vs 2.6790 +8%; kodim05/13 all-channels static
  wins by 2–9%) because 4-bin global conditioning is coarser than WIDE-K
  64-group + per-group H/G/R + G-bias static conditioning, and table
  elimination (~0.02–0.04bpp) cannot cover the gap on CROWN4's tight
  residuals. Hence per-channel best-of-3 with a static fallback (probe
  follow-up #2 verbatim: "per-plane best-of with 2b choice side"), so
  CROWN5 >= CROWN4 is impossible by construction (modulo the 8b/channel
  choice byte, counted). The probe's 3.11–3.16 transfer estimate assumed
  additive MED-frame deltas onto CROWN4 symbols; the build PROVES overlap is
  ~100% (grouping already harvests the conditioning), invalidating the
  additive assumption — diagnosed explicitly in the run report, not hidden.
- Wire-split preserved: static-G M-mapping (CROWN `2v/-2v-1`) vs adaptive M
  (probe `2|r|-(r>0)`) — each self-consistent with its decoder (`M_to_r`
  inverts probe mapping; static branch inverts CROWN mapping); polarities
  identical (zeros+one). No cross-path symbol reuse.
