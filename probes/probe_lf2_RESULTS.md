# Probe LF2 RESULTS — clustered LF side-information conditioning vs order-0 baseline

Script: `probe_lf2.py` (numpy + PIL only, CPU). All bytes counted: per-stream
`sum(count*len)` via real heapq code lengths + `16 + A*24` table overhead per
stream, side stream Huffman, 9×16-bit quartile thresholds (18 bytes).

## Per-image numbers (bpp = total bits / pixels, 3 channels)

| image | baseline bpp | conditioned+side bpp | delta |
|---|---|---|---|
| kodim01.png | 10.8535 | 11.6045 | +6.92% |
| kodim02.png | 9.9764 | 10.5825 | +6.07% |
| kodim05.png | 12.0356 | 12.9248 | +7.39% |
| kodim07.png | 9.4587 | 10.2672 | +8.55% |
| kodim13.png | 12.7133 | 13.6098 | +7.05% |
| kodim19.png | 10.5242 | 11.2645 | +7.03% |
| kodim23.png | 9.5526 | 10.4915 | +9.83% |
| **average (7)** | **10.7306** | **11.5350** | **+7.50%** |

YCoCg-R invertibility asserted exactly on all 7 images (forward→inverse == input).

## Verdict: NO — clustered-LF does NOT beat the baseline. Net margin −7.5% (loss) on average, losses on all 7/7 images (+6.1% … +9.8%).

(Note: this probe's baseline ≈10.73 bpp is the weak order-0 Huffman-on-MED
reference, not the 3.464 bpp champion — which already removes most of this
redundancy via context/run/adaptive coding. The probe tests whether LF
conditioning adds anything *on top of decorrelation*; it does not, so it
cannot close the −2.4% gap to WebP-m3.)

## Why it failed (measured breakdown, kodim01 / kodim23, per-channel bpp)

- Conditioning *works* directionally: per-bin splits cut data bits
  (e.g. kodim01 Y: 5.537 → 5.351 bpp, saves ~0.19 bpp; total data saving
  across 3 channels ≈ 0.4 bpp).
- Extra table headers from 12 tables vs 3 are negligible (~+0.04 bpp).
- **The killer is the side stream: ~1.1 bpp total** (~0.3–0.45 bpp/channel of
  LF-downsampled data bits + its own tables). The 4× downsampled planes still
  carry ~full local-mean information and order-0 Huffman has no spatial model,
  so transmitting them costs ~3× more than the ~0.4 bpp the conditioning saves.
- Oracle bonus ignored: bin assignment `e = orig − upLF` needs the original,
  so a real decoder could not derive bins from the side info alone — yet the
  2 bit/px bin map was NOT counted. The probe loses even with this free gift,
  so the idea fails as stated, not just as implemented.
- Counting note: thresholds were counted honestly as 9×16-bit = 18 bytes
  (3 quartile thresholds × 3 channels); the spec's "6 bytes" undercounts by
  12 bytes ≈ 0.0004 bpp — immaterial to the verdict either way.

## Single recommended follow-up

**Drop the transmitted-LF design; condition on already-decodable context
instead** — e.g. split residuals by a coarse function of the causal
already-decoded neighborhood mean (no side stream at all, zero side bytes,
bins decodable by construction). That keeps the ~0.4 bpp data-bit win while
deleting the ~1.1 bpp side cost that caused this failure; it ports directly
into the champion's existing context-Huffman machinery as one extra context
bit. If that context-conditioned variant still loses, retire the LF family.
