# probe_b23 RESULTS — texture-context training tail: NO HARVEST

Branch question (Boss-5 quest, b21 follow-up #2): texture-ctx loss still
descending at it199 — does it300 + late-iterate averaging harvest the tail
while keeping quant-robustness?

## 0. Verdict

**NO HARVEST — all deltas within ±0.5%, wins identical across arms.**
it300-final and it300-late-avg (mean of iters 250–290 checkpoints) both
scatter ±0.2% around the it200 control with no systematic direction;
win/loss gates flip on 0/18 units (13-ch1-ctx8 loses in ALL arms —
consistent). Aggregate savings are ~1–2 kb vs the −81 kb (05) / −46 kb (13)
needed: two orders of magnitude short. The quant-robust basin claim (b21 §4)
extends to it300: longer training neither helps nor (with averaging) hurts
quantized bits. b21#2 CLOSED.

## 1. Numbers (quantized WIRE bits per (ch, texture-ctx); all arms gated)

kodim05 [C27], ctx {6,7,8} (9 units):

| unit | med | A it200 | B it300 (Δ) | C it300 (Δ) | C-avg (Δ) |
|---|---|---|---|---|---|
| ch0-6 | 269208 | 258186 | 258146 (−0.02%) | 258137 (−0.02%) | 258137 (−0.02%) |
| ch0-7 | 276730 | 263315 | 263429 (+0.04%) | 263439 (+0.05%) | 263439 (+0.05%) |
| ch0-8 | 314207 | 303931 | 304051 (+0.04%) | 303911 (−0.01%) | 303923 (−0.00%) |
| ch1-6 | 180109 | 169069 | 169045 (−0.01%) | 169281 (+0.13%) | 169151 (+0.05%) |
| ch1-7 | 124574 | 116611 | 116600 (−0.01%) | 116597 (−0.01%) | 116608 (−0.00%) |
| ch1-8 | 179254 | 166784 | 167120 (+0.20%) | 167207 (+0.25%) | 167104 (+0.19%) |
| ch2-6 | 144463 | 136661 | 137300 (+0.47%) | 137301 (+0.47%) | 137301 (+0.47%) |
| ch2-7 | 141831 | 130758 | 130543 (−0.16%) | 130543 (−0.16%) | 130495 (−0.20%) |
| ch2-8 | 178592 | 171354 | 170995 (−0.21%) | 171108 (−0.14%) | 171115 (−0.14%) |

kodim13 [C12], ctx {6,7,8} (9 units): same picture, all |Δ|≤0.26%,
worst B +0.08% / best C-avg −0.20%; ch1-ctx8 loses in all arms.

No arm wins systematically (B better on 8/18, C-avg on 7/18, A on 3/18 —
coin flip). Late-averaging does not rescue it300's scatter (C≈B).

## 2. Mechanism

Training-loss tail (still descending at it199) is real but lives in
directions the int16 grid cannot resolve: extra iters move weights within a
flat float valley whose quantized projections are equivalent (deltas ±0.2%
≈ rounding noise across 113 params × ~40k samples). Sharp-minimum risk does
not materialize at it300 either (no blowups; worst +0.47%) — the 200/0.003
basin is simply flat in quantized space. Training-axis harvest on this
architecture is DONE.

## 3. Files

- `probes/probe_b23_tail.py` (harness copy of B21.train_net + late-average
  checkpoints + driver-verbatim quant/wire/gate eval), `probes/probe_b23_nums.json`
  (18 units × arms). Symlinked in `experiments/`. Torch CPU, ~20 min/image.
