# FLIF v0.4 Teardown — static review (lossless path)

Date: 2026-09-08. Method: read-only subagent review (full-file reads), owner
spot-checked load-bearing claims (all PASS, see Verification footer).
Provenance: `FLIF-hub/FLIF` tag **v0.4**
(https://github.com/FLIF-hub/FLIF/releases/tag/v0.4, source-only release),
tarball md5 `c9615a4a525ecd39b27317ceb8365652`, reviewed at
`/tmp/flifb4/FLIF-0.4` (NOT vendored — regenerable from tag; build artifacts
excluded). Measured binary from same source: 2.9699 bpp Kodak-7
(`survey/bosstakedown/data/classical7.json`), pixel-PASS 7/7.
Reason: FLIF stands 0W-7L over CROWN6 and beats JXL-e3 7/7 on our seven —
this teardown hunts the mechanism and portable ideas (QOI-teardown pattern).

Queue mapping for our codec (owner assessment, not subagent's):
- #3 cross-channel activity ≈ takedown Attack-2 (already planned) — corroborated.
- #1 Y-conditional ranges, #2 alphabet elision, #4 centred predictor — NEW, candidates.
- #5 binary adaptation — skepticism HIGH post-b22/CROWN5 (measure-gated only).
- #6 micro-tree — conditional on #3 saturating; copies grow+prune BOTH halves.

---

*Subagent report banked below (lightly trimmed for length: redundant phrasing
removed; every technical claim and file:line pointer preserved).*
---
# FLIF v0.4 Static Review — Lossless Path Only (subagent report)

> READ-ONLY. Sources under `/tmp/flifb4/FLIF-0.4/src/`. Binary not executed, nothing built/modified. Background: `BOSS_TAKEDOWN.md:1-119` §§1–3 only. All `src/` paths below are relative to `/tmp/flifb4/FLIF-0.4/src/`.

## 1) Encode pipeline overview

### 1a. Container / header / transform signalling

1. Magic `FLIF` + 1 type byte (`16*encoding+numPlanes`, `+32` if animated) + bit-depth byte + varint W/H + metadata + `0` version byte: `flif-enc.cpp:799-845`.
2. `UniformSymbolCoder` meta layer writes: bit-depths, alpha-zero flag, cutoff/alpha override flag, later checksum flag: `flif-enc.cpp:848-888,1047-1055`.
3. Transforms are tried **in candidate order**, each either accepted (`rac.write_bit(true)` + `write_name()` + `trans->save()` + `meta()` + `data()`) or skipped (`rac.write_bit(false)` terminator): `flif-enc.cpp:904-944`; name coder is `write_int(0,MAX_TRANSFORM,nb)`: `flif-enc.cpp:38-47`; `MAX_TRANSFORM 13`: `common.hpp:48`; name table: `common.cpp:22-26`.
4. Default still-photo candidate order is fixed in code — **not searched**: `flif-enc.hpp:11-12`, built in `flif.cpp:289-332`: `Channel_Compact → YCoCg → PermutePlanes → Bounds → Palette_Alpha → Palette → Color_Buckets [+animation transforms]`. Each `init()/process()` can veto; e.g. YCoCg vetoes if `<3` planes, negative mins, or flat channel: `transform/ycocg.hpp:183-191`.
5. Practical consequence for photos: `Channel_Compact` usually rejects, `Palette[_Alpha]` rejects (`>max_palette_size`, default `512`: `config.h:66`), `Bounds` usually accepts but near-trivial, `Color_Buckets` accepts only on large images (`>10000` px). So the Kodak-7 win path is almost always **`YCoCg → Bounds [+Color_Buckets] → MANIAC`**, no palette.

### 1b. Two encodings; photos use interlaced

1. Method defaults to `interlaced` when `nb_pixels*frames>=10000`, else `nonInterlaced`: `flif.cpp:321-324`. Kodak 768×512 is interlaced.
2. Interlaced path (`FLIF2`, `flif-enc.cpp:194-316`): codes plane×zoomlevel units in priority order from `plane_zoomlevel()`: `common.cpp:168-214`, with lag caps `max_behind={0,2,4,0,0}` (`Y` leads, `Co` ≤2 behind, `Cg` ≤4 behind): `common.cpp:177`.
3. Within one `(p,z)`: even `z` = code odd rows left→right; odd `z` = code odd columns top→bottom, stepping 2: `flif-enc.cpp:221-279`. Key property: at `z=0` every coded pixel already knows top **and** bottom (even `z`) or left **and** right (odd `z`) neighbours at same zoomlevel, plus coarser levels. Prediction/context is therefore **non-causal**, unlike scanline MED/GAP.

### 1c. Two-pass MANIAC + tree signalling

1. `flif_encode_main<bits>()`: `flif-enc.cpp:682-746`: Pass 0 codes coarse zoomlevels with empty forest (no learning); `roughZL` floor 0; `NB_NOLEARN_ZOOMS 12` (`config.h:83-84`, ≈64×64 thumbnail before tree exists). Pass 1 = **learn**: `learn_repeats` iterations with `RacDummy` + growing forest: `flif-enc.cpp:716-723`; default `TREE_LEARN_REPEATS 2`: `config.h:64`. Tree signalling via `MetaPropertySymbolCoder::write_tree()`: `maniac/compound_enc.hpp:373-378`. Pass 2 = **final** real encode with `FinalPropertySymbolCoder`, growth divisors zeroed: `flif-enc.cpp:733-744`.
2. Per-plane forest: `std::vector<Tree> forest(numPlanes)`: `flif-enc.cpp:698`; decode rebuilds identical forest via `read_tree()`: `flif-dec.cpp:860-877`.

## 2) Predictor set + context-property details

### 2a. Predictor set: exactly 3, fixed per `(plane,zoomlevel)` (`common.hpp:260-264,288-292`)

| `predictor` | horizontal (even `z`) | vertical (odd `z`) |
|---|---|---|
| 0 (`avg`) | `(top+bottom)>>1` | `(left+right)>>1` |
| 1 (`median_grad`) | `median3(avg, left+top-topleft, left+bottom-bottomleft)` | `median3(avg, left+top-topleft, right+top-topright)` |
| 2 (`median_nb`) | `median3(top,bottom,left)` | `median3(top,left,right)` |

`MAX_PREDICTOR 2` (`common.hpp:49`) — no weighted/GAP predictor exists. Selection per `(p,z)` via subsampled heuristic cost `sum ilog2(|curr-guess|)+(resid!=0?1:0)` (`flif-enc.cpp:133-191`), signalled per `(p,z)` (`flif-enc.cpp:208-218`); guess range-snapped before coding (`common.hpp:265,293`).

### 2b. Properties (interlaced photo path; `NB_PROPERTIES={8,10,9,8,8}`: `common.cpp:75-76`)

1. Cross-plane causals (`Y` if `p>0`, `Co` if `p>1`): `common.hpp:229-233`.
2. `which ∈{0,1,2}` — which median input won: `common.hpp:253-256`.
3. Luma-miss **only for `p==1,2`**: `common.hpp:257-259`.
4. Four local-activity signed diffs + 5. `guess` itself: `common.hpp:266-269,300`.
6. Far gradients except `p==2`: `common.hpp:311-316`.

### 2c. Tree: cost-gated growth + count-gated prune

- Node `PropertyDecisionNode{property, count, splitval, childID, leafID}` (`maniac/compound.hpp:35-48`); no explicit max depth — growth cost-gated, pruning count-gated.
- Learning leaf tracks per-property virtual hypothesis pairs vs real estimator; best = smallest `virtSize` beating `realSize` (`maniac/compound_enc.hpp:22-49,69-87,247-256`).
- Split at `splitval=div_down(virtPropSum,count)` when `realSize>virtSize[best]+split_threshold` (`maniac/compound_enc.hpp:197-245`); default threshold `5461*8*8` ≈ 64 internal bits (`config.h:69`).
- Post-learn prune: subtree pruned if count-sum `<min_size` (default `50`: `config.h:72`); counts divided by `divisor` (default `30`: `config.h:71`), clamped `[1,512]` (`config.h:90-91`).
- Signalling preorder: `(property+1, count∈[1,512], splitval∈[oldmin,oldmax-1])` (`maniac/compound_enc.hpp:348-378`).

## 3) Color handling — byte-exact reversible integer YCoCg-R variant

Forward (`transform/ycocg.hpp:202-226`): `Y=(((R+B)>>1)+G)>>1; Co=R-B; Cg=G-((R+B)>>1)` (floor `>>`, arithmetic-shift assert). Compression-relevant: **conditional range tightening** `Co|(Y)`, `Cg|(Y,Co)` (`transform/ycocg.hpp:74-134,167-172`) — decoder-side chroma alphabet shrinkage on dark/bright pixels for free (Y,Co already coded; plane order Y→Co→Cg). No RCT search (fixed order `Channel_Compact → YCoCg → PermutePlanes → Bounds → Palette → Color_Buckets`); alternatives only if YCoCg vetoes.

## 4) Entropy coder mechanics — 24-bit binary RAC, per-bit adaptive states

- RAC core 24-bit (`maniac/rac.hpp:31-48,69-97`); encode delayed-byte/carry-safe (`maniac/rac_enc.hpp:32-68`); learning passes use no-op `RacDummy`.
- Alphabet: bit-plane decomposition of snapped `d=curr-guess` — BIT_ZERO, BIT_SIGN (iff straddling), unary exponent, MSB→LSB mantissa with impossible-bit elision (`minabs1>amax⇒0 / maxabs0<amin⇒1`, no bit emitted: `maniac/symbol_enc.hpp:95-109`).
- Chance: single `uint16` 12-bit state per bit-position per leaf, table-driven exponential decay + floor (`cut=2`: `config.h:178`; `maniac/chance.hpp:106-111`, `maniac/chance.cpp:23-57`). **Build ships `FAST_BUT_WORSE_COMPRESSION=1` (`config.h:87`) — multiscale mixing compiled out; single fast-adapting table suffices for the shipped 2.9699.**
- Bit-cost estimator `estim()` scaled by 5461 (`maniac/chance.hpp:113-115`) doubles as exact MDL-gate currency.

## 5) Why FLIF plausibly wins on texture vs per-group Huffman/Golomb/rANS

1. **Per-pixel alphabet shrinkage** (snapped `[min-guess,max-guess]`, impossible mantissa bits never emitted) vs per-group worst-case alphabets — structural on texture.
2. **Activity-partitioned contexts, not location tiles** — smooth/texture pixels sharing a group get different distributions under one global forest; no 256×256-group fragmentation.
3. **Per-bit binary adaptation with floor** vs integer-bit static prefix codes + per-group tables; tracks pixel-to-pixel variance.
4. **Non-causal prediction + `guess`-as-property** (interlaced avg sees opposite side); `guess`/miss-magnitudes remain tree properties, quarantining texture leaves.
5. **Cross-channel conditioning at the pixel** (co-located Y + luma-miss for Co/Cg) + Y-conditional chroma ranges — harvests per-group-RCT-like gains with bytes of tree cost.

Net: causal predictor → fixed groups → one histogram per group pays three texture taxes FLIF avoids: (a) group-worst-case alphabet, (b) one distribution for smooth+texture pixels in a group, (c) causal-only prediction. The 0.23 bpp gap (≈7%) is plausibly their sum.

## 6) Ranked portable ideas (integer-exact C codec)

- **#1 Y-conditional chroma range narrowing** (cheapest): restrict `Co|(Y)`, `Cg|(Y,Co)` via precomputed per-Y `[lo,hi]` tables; decoder = table lookups. (`transform/ycocg.hpp:66-134,150-172`.)
- **#2 Per-pixel min/max-snapped residual alphabet + impossible-bit elision** (`maniac/symbol_enc.hpp:95-109` pattern); gains on bounded channels.
- **#3 One cross-channel activity property as second-level context key** (co-located luma miss/activity, 2–4 bins, Cartesian + MDL-merge; Y decoded first anyway). (`common.cpp:92`.)
- **#4 Centred `(top+bottom)>>1` predictor arm**, per-group/region exact-bit gate. (`common.hpp:260-264`.)
- **#5 Fast single-scale binary adaptation with floor** for residual bit-planes; `estim()` as MDL-gate currency. (`maniac/chance.hpp:106-119`; `config.h:178`.)
- **#6 Two-split MDL micro-tree on activity** (grow threshold ≈64 bits + hard prune count/divisor) — only if #3 saturates; copy BOTH halves.
- Non-recommendation: wholesale `Color_Buckets`/`Palette` ports (veto on texture by design).

## 7) Takedown-assumption challenges (abridged)

per-group RCT search overstated (FLIF: none, yet wins); "bigger groups" stops short (FLIF: no groups, global forest); Weighted arm not necessary for this gap (FLIF: avg+2 medians only); texture-vs-smooth attribution needs revisiting (FLIF toolkit helps both); over-split caution confirmed but FLIF's gate is loose-grow + hard-prune (copy both); alphabet-coder sophistication secondary to partitioning/adaptation; palette/noise notes consistent.

## 8) File:line index — see §1–§6 inline pointers (all `src/`-relative to `/tmp/flifb4/FLIF-0.4/src/`).

---

## Owner verification footer (2026-09-08)

Spot-checked vs source, all PASS: `config.h:69` threshold `5461*8*8`,
`:71-72` divisor 30 / min-size 50, `:83` nolearn 12, `:87` fast-but-worse 1;
`common.hpp` predictor/property block matches (§2a/§2b forms); YCoCg forward
formula matches (`transform/ycocg.hpp:202-226` incl. commented-out alternative).
Tarball md5 recorded above. Binary not executed by reviewer; measured numbers
come from owner's banked ledger (`survey/bosstakedown/data/classical7.json`).