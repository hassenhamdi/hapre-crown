# Per-group framing field list — src/CROWN6_FORMAT.md (factual extraction only)

Source: `/home/hassenhamdi/autocompress/src/CROWN6_FORMAT.md` (195 lines). Global header 8B (`C6`+H/W u16le+ver/flags, FORMAT:38-46); channel header 3B (`[kidx:4][grid:3][fam:1]`+ng+MLP flags, FORMAT:50-58); MLP side 64B+9B+226B/nwin (FORMAT:60-70); W-side `ceil(ng32/8)`+`ceil(nused·20/8)` (FORMAT:72-76); LMS zero bytes (FORMAT:78); group map Q 92B+ids / GRID 1B occ (FORMAT:80-84); per-group metadata 7b `[predid:5][backend:2]` (FORMAT:86-91) + kvals 4b + dbias 3b (FORMAT:92-94).

## Huffman (16+A·24) — spec lines 96-99
> `98: - Per H non-empty group: 'A u16le' + A × ('sym i16le' + 'len u8').`
> `99: - H payload: 'n u32le' + n bytes via 'pack_syms' (libhapre.so).`

Bit width: table `16 + A·24` bits (A=u16 count, each entry 16b sym + 8b len = 24b); payload `32 + 8·n` bits (n=u32le byte count). Empty groups forced backend 0 with no table/payload (FORMAT:91). Same width as CROWN4 (FORMAT:72-102 header `CROWN4-identical`).

## rANS M=14 (16+A·32+64) — spec lines 101-102
> `101: - rANS groups (M=14): 'count u32le' + 'A u16le' + A × ('sym i16le' + 'freq u16le') + 'paylen u32le' + payload, via libhapre.so 'rans_*'.`

Bit widths: table `16 + A·32` bits (A=u16, each 16b sym + 16b freq); framing `64` bits = `count u32le` (32b: alphabet offset/symbol count for decode) + `paylen u32le` (32b payload bytes); total per non-empty R group header `16+A·32+64` bits + payload bytes. M=14 minimum for 200+ alphabets per AGENTS.md:9 (`A=232 needs M=14, not 10`).

## Golomb + metadata widths (for context, same section)
- G payload: `n u32le` + n bytes via `crown6_golomb_pack` (FORMAT:100); per-G kvals 4b (`k∈0..12`, FORMAT:92) + dbias 3b (`{−4..3}` two's-complement, FORMAT:93-94); backend choice 2b/group in the 7b field (FORMAT:88-90).
- Unary polarity wire (FORMAT:104-108): `M=v≥0?2v:−2v−1; code = q ZEROS + 1 + k-bit remainder MSB-first` (libhapre opposite polarity NEVER used).
- Framing asserts: `p==len(blob)` per channel + `decode==original` (FORMAT:162-163).

No new math claims; widths are counted in `huff_cost`/`golomb_cost`/`rans_cost` (`src/driver_crown6.py:418-473`).
