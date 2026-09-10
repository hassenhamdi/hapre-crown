# probe_b24 RESULTS — side-info audit: headers cover 29–74% of gaps, insufficient

Branch question (Boss-5 quest): after b22 (adaptivity ~0) and b23 (tail ~0),
is there KB-scale side-info mass — specifically per-group rANS headers
(16+Ad·32+64 b) — whose sharing could flip a blocker?

## 0. Verdict

**INSUFFICIENT ALONE.** Bit-exact walker (mirrors driver.decode_image,
p==len asserted) over winner streams: R-headers total 2,938 B (05, 29% of the
+10,116 B gap) and 4,270 B (13, 74% of the +5,771 B gap). Even 50% recovery
via header sharing (optimistic) yields 15–37% of gap. H-tables ≤1.3 KB,
Golomb k/d side ≤77 B — dead. Combined coding-side ceiling (b22+b24):
≤2–3 KB vs gaps of 10.1/5.8 KB. Boss-5 needs PREDICTION mass (05-Y G-pay alone
is 246 KB) or stands.

## 1. Full side tables (bytes; framing-asserted)

kodim05 [C27] bytes=527,981 bpp=3.5806 vs JXL 3.5120 (gap +10,116 B):

| ch | mlp_side | wuse | wweights | map | experts | gkd | H-tab | H-pay | G-pay | R-head | R-pay | ng |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1881 | 48 | 588 | 366 | 49 | 48 | 101 | 2304 | 246267 | 0 | 0 | 56 |
| 1 | 1429 | 48 | 672 | 364 | 35 | 14 | 563 | 20295 | 49353 | 1710 | 61363 | 40 |
| 2 | 1655 | 48 | 542 | 365 | 30 | 15 | 340 | 17526 | 69685 | 1228 | 49003 | 34 |

kodim13 [C12] bytes=582,929 bpp=3.9532 vs JXL 3.9141 (gap +5,771 B):

| ch | mlp_side | wuse | wweights | map | experts | gkd | H-tab | H-pay | G-pay | R-head | R-pay | ng |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 48 | 762 | 366 | 49 | 47 | 212 | 2730 | 287777 | 270 | 486 | 56 |
| 1 | 1203 | 48 | 540 | 363 | 25 | 13 | 462 | 13368 | 78646 | 1938 | 57310 | 29 |
| 2 | 0 | 48 | 820 | 305 | 30 | 9 | 647 | 26797 | 35076 | 2062 | 70427 | 34 |

Note: mlp_side/wweights/map buy prediction (not dead weight); R-headers are
the only sharable side at scale. 13-ch1/ch2 carry the largest R-heads.

## 2. Files

- `probes/probe_b24_sideaudit.py` (walker + JSON), `probes/probe_b24_nums.json`.
  Reusable for any future side-sharing design (parse once, evaluate exactly).
