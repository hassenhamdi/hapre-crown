"""probe_b22: adaptive-Golomb transfer measurement on the REAL CROWN6 frame.

Question: CROWN6's texture loss (05 +0.0797bpp, 13 +0.0384bpp vs JXL-e3) sits
~98% in per-group static-Golomb payload (cycle-46 evidence). Would a
decoder-side adaptive Golomb-k (JPEG-LS N/A counters, RESET=64) flip the
blockers? b19 proved adaptivity on the MED frame; this probe measures the
RESIDUAL delta over CROWN6's per-group best-k (the overlap question b19 left
open) using the encoder's own winning groups (CROWN_DUMPGROUPS dumps).

Method (numpy-only, no C build): read .gdat (winning variant G-groups with
encoder (expert,k,d) + member (idx,sym) pairs); verify static costs against
driver golomb_cost (must match k/d exactly — fidelity gate); simulate:
  (a) per-group adaptive (state per group, raster order, 1 ctx and 4 act-ctx);
  (b) global adaptive (all G symbols pooled in raster order, 1 ctx / 4 act-ctx).
Lengths use CROWN zigzag mapping M=r>=0?2r:-2r-1, len=(M>>k)+1+k (NOTE: differs
from b19's M sign polarity — self-test below asserts length parity with
driver golomb_best on random vectors before measuring). States init N=1/A=4,
RESET=64 (b19 convention).
Activity for act-ctx: zero-border causal e=|L-TL|+|T-TL|, bins (4,12,48).

Wire honesty: adaptive drops static +4b k / +3b bias sides (init static);
backend value 3 fits the existing 2b field (+0b). Gate margin x1.2 covers it.

Usage:
  python3 probes/probe_b22_adaptive.py --dump /tmp/b22dump --images kodim05.png kodim13.png
"""
import sys
import os
import struct
import numpy as np

sys.path.insert(0, "/tmp/opencode/autocompress/src")
sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
sys.path.insert(0, "/tmp/opencode/autocompress/probes")

import driver_crown6 as D6

ATHR = (4, 12, 48)


def act_bin(e):
    if e <= ATHR[0]:
        return 0
    if e <= ATHR[1]:
        return 1
    if e <= ATHR[2]:
        return 2
    return 3


def read_gdat(path):
    with open(path, "rb") as f:
        buf = f.read()
    assert buf[:4] == b"GGD1", path
    fam, kg, ng, nG = struct.unpack_from("<BBHI", buf, 4)
    off = 12
    groups = []
    for _ in range(nG):
        ex, kk, dd, pad, cnt = struct.unpack_from("<BBbBI", buf, off)
        off += 8
        idx = np.frombuffer(buf, dtype="<i4", count=cnt, offset=off).astype(np.int64)
        off += 4 * cnt
        sym = np.frombuffer(buf, dtype="<i4", count=cnt, offset=off).astype(np.int64)
        off += 4 * cnt
        groups.append({"expert": ex, "k": kk, "d": dd, "idx": idx, "sym": sym})
    assert off == len(buf), (path, off, len(buf))
    return {"family": fam, "k_or_grid": kg, "ng": ng, "groups": groups}


class AState:
    """JPEG-LS style N/A/k counters, one set per ctx (b19 GolombState)."""

    def __init__(self, nctx, reset=64):
        self.N = [1] * nctx
        self.A = [4] * nctx
        self.reset = reset

    def k(self, ctx):
        N, A = self.N[ctx], self.A[ctx]
        k = 0
        while (N << k) < A:
            k += 1
        return k

    def upd(self, ctx, abr):
        self.A[ctx] += int(abr)
        self.N[ctx] += 1
        if self.N[ctx] >= self.reset:
            self.N[ctx] >>= 1
            self.A[ctx] >>= 1
            if self.N[ctx] < 1:
                self.N[ctx] = 1


def crown_len(sym, k):
    M = 2 * sym if sym >= 0 else -2 * sym - 1
    return (M >> k) + 1 + k


def adaptive_bits(syms, ctxs, nctx, reset=64):
    """Exact adaptive length over symbol sequence with per-symbol ctx ids."""
    st = AState(nctx, reset)
    tot = 0
    for s, c in zip(syms.tolist(), ctxs.tolist()):
        k = st.k(int(c))
        tot += crown_len(int(s), k)
        st.upd(int(c), abs(int(s)))
    return tot


def zero_border_activity(plane):
    """e=|L-TL|+|T-TL| with zero outside (decoder-computable causal taps)."""
    P = np.asarray(plane, dtype=np.int64)
    L = np.zeros_like(P)
    L[:, 1:] = P[:, :-1]
    T = np.zeros_like(P)
    T[1:, :] = P[:-1, :]
    TL = np.zeros_like(P)
    TL[1:, 1:] = P[:-1, :-1]
    return np.abs(L - TL) + np.abs(T - TL)


def static_verify(syms):
    """driver golomb_cost (exact, incl +4/+3 sides). Returns (tot,k,d)."""
    tot, k, d, _ = D6.golomb_cost(np.asarray(syms, dtype=np.int32))
    return tot, k, d


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", default="/tmp/b22dump")
    ap.add_argument("--gdat", nargs="*", default=None)
    a = ap.parse_args()
    paths = a.gdat or sorted(os.path.join(a.dump, f) for f in os.listdir(a.dump)
                             if f.endswith(".gdat"))
    for p in paths:
        g = read_gdat(p)
        n = sum(len(x["sym"]) for x in g["groups"])
        print(f"{os.path.basename(p)}: fam={g['family']} ng={g['ng']} "
              f"Ggroups={len(g['groups'])} Gsyms={n}")
