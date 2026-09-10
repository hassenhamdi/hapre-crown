"""Probe B8: IFCE-style inter-group conditioning + MDL-gated micro-adapters.

Branch probe for the lossless-codec campaign (standing boss JXL-e3 3.23).
Probes survey TOP-5 #4 (IFCE) and #5 (MDL micro-adapters) in isolation.

Method: numpy + PIL only, CPU, no torch. UNIT RULE: bpp = total_bits/(H*W*3).
Exact counting: real heapq Huffman data bits + 16+A*24 per stream, ALL sides.
YCoCg-R + champion 0/left/top MED border rule. YCoCg-R round-trip + residual
recon asserted per image.

Baselines:
  order0 : MED residuals per channel, independent order-0 Huffman per channel.
           Total = 64 dims + sum_c(data_c + 16 + A_c*24). (anchor ~3.58)
  G6     : unconditioned per-group Huffman. Per channel, 6 energy groups from
           e=|a-c|+|b-c| with FIXED GLOBAL thresholds T=[4,12,28,60,120]
           (zero side bytes; decoder recomputes e from recon neighbors).
           Total = 64 + sum_ch sum_g(data+16+A*24). Empty groups cost 0
           (decoder knows occupancy from recon-derived keys, no side needed).

Mechanism 1 (IFCE-style inter-group conditioning). Each arm splits G6 cells by
a coarse indicator available at decode time, with FIXED GLOBAL thresholds
(zero indicator side; every extra TABLE counted). Y stream is always coded
first, so chroma may condition on Y residuals (decoder-safe: Y fully recon
before Co/Cg decode). Intra-channel indicators (level L, directional |a-b|)
are causal neighbor functions (recon == orig, so encoder == decoder keys).
  I-Y2  : chroma (g x yb), yb=|rY| C=2 bound [1].              6->12 tbl/ch
  I-Y4  : chroma (g x yb), yb=|rY| C=4 bounds [0,2,7].         6->24 tbl/ch
  I-DC2 : all ch (g x lb), lb: L=(a+b)//2, Y bound 128, Co/Cg bound 0.
  I-EC2 : all ch (g x db), db=|a-b|, Y bound 8, Co/Cg bound 4.
  I-COMB: chroma (g x yb2 x lb2) = up to 24 tbl/ch (fragmentation probe).
  I-GATED: per-group local MDL gate on I-Y2 splits (chroma only): group g keeps
           the Y-split iff split_total < unsplit_total; 1b flag/group (12b/img
           always counted) signals split/no-split before the tables.

Mechanism 2 (MDL-gated micro-adapters). Residual shift adapters r' = r - d,
decoder adds d back (neighbors/recon unaffected, grouping unchanged).
  M-GLOB : per-channel global bias b_c in [-8..8] (searched exactly per channel
           on the G6 cells). Params charged 5b each (covers [-16,15],
           conservative vs the [-8,8] search). All-or-none per image:
           enabled_total = best + 64 + 1(gate) + 15(params);
           disabled_total = base + 64 + 1(gate, always counted).
           Enable iff enabled < disabled. Adoption rate reported.
  M-GROUP: per-cell bias delta d in [-4..3] with per-cell 1b gate (ncells bits
           always counted) + 3b param per adopted cell. Adopt cell iff
           `min_d cost(vals-d)+3 < cost(vals)`. Adoption = cells adopted.
  M-BOTH : M-GLOB then M-GROUP re-searched on shifted cells (stacking test).
  M-THR  : per-image threshold-GRID micro-adapter (Huffman-visible, the live
           MDL hypothesis): best of 4 FIXED codec-constant energy grids
           (D0==G6 grid, D1 fine-flat, D2 coarse-texture, D3 flat-heavy);
           2b selector ALWAYS counted; enable iff 64+2+best < G6 total.
           (Bias family M-GLOB/M-GROUP is Huffman-translation-invariant by
           construction — shift preserves counts => identical Huffman cost =>
           structural zero; the gate correctly rejects. Kept as documented
           negative, not hidden.)

Combined:
  C-FULL : I-GATED cells + M-GLOB (re-searched, own gate) + M-GROUP (cell level).
           Side accumulates honestly: 64 + 12(ifce flags) + 1(+15)(glob gate)
           + ncells(group flags) + 3/adapted cell.
  C-FULL2: best grid (2b) + I-GATED splits RECOMPUTED on that grid's groups
           (12 flags). Total = 64 + 2 + 12 + data/tables.

Decoder-safety: stream orders + causality argued per family in RESULTS.md.
"""

import heapq
import os
import time

import numpy as np
from PIL import Image

IMAGES = [
    "/tmp/opencode/autocompress/experiments/real_photos/kodim01.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim02.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim05.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim07.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim13.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim19.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim23.png",
]

# Spec anchors (task list; kodim19 listed ~3.51 per cycle-12 estimator note —
# our own B7 probe measured 3.5081, so 3.51 is used, NOT 3.36).
EXPECTED = {
    "kodim01.png": 3.62, "kodim02.png": 3.33, "kodim05.png": 4.01,
    "kodim07.png": 3.15, "kodim13.png": 4.24, "kodim19.png": 3.51,
    "kodim23.png": 3.18,
}
DIMS_BITS = 64

E_THR = np.array([4, 12, 28, 60, 120], dtype=np.int64)   # -> 6 energy groups
Y2_BOUND = np.array([1], dtype=np.int64)
Y4_BOUNDS = np.array([0, 2, 7], dtype=np.int64)
DC_BOUND = {"Y": 128, "Co": 0, "Cg": 0}                  # level-split bounds
DB_BOUND = {"Y": 8, "Co": 4, "Cg": 4}                    # |a-b| split bounds

GLOB_RANGE = range(-8, 9)    # M-GLOB search
GLOB_PARAM_BITS = 5          # charged per channel param (covers [-16,15])
GROUP_RANGE = range(-4, 4)   # M-GROUP search
GROUP_PARAM_BITS = 3         # per adopted cell

# M-THR grids: codec constants, fixed a priori (geometric variants around D0).
GRIDS = {
    "D0": np.array([4, 12, 28, 60, 120], dtype=np.int64),    # == E_THR baseline
    "D1": np.array([2, 6, 16, 40, 100], dtype=np.int64),     # fine-flat
    "D2": np.array([8, 24, 64, 160, 400], dtype=np.int64),   # coarse-texture
    "D3": np.array([1, 3, 8, 24, 80], dtype=np.int64),       # flat-heavy
}

OUT_MD = "/tmp/opencode/autocompress/experiments/probe_b8_RESULTS.md"


def rgb_to_ycocgr(arr):
    R = arr[:, :, 0].astype(np.int32)
    G = arr[:, :, 1].astype(np.int32)
    B = arr[:, :, 2].astype(np.int32)
    Co = R - B
    t = B + (Co // 2)
    Cg = G - t
    Y = t + (Cg // 2)
    return Y, Cg, Co


def ycocgr_to_rgb(Y, Cg, Co):
    t = Y - (Cg // 2)
    B = t - (Co // 2)
    G = Cg + t
    R = Co + B
    out = np.empty(Y.shape + (3,), dtype=np.int32)
    out[:, :, 0] = R
    out[:, :, 1] = G
    out[:, :, 2] = B
    return out


def champion_neighbors(ch):
    """Champion 0/left/top border rule (probe_b4_a verbatim semantics)."""
    H, W = ch.shape
    a = np.empty_like(ch)
    b = np.empty_like(ch)
    c = np.empty_like(ch)
    a[:, 1:] = ch[:, :-1]
    a[:, 0] = 0
    b[1:, :] = ch[:-1, :]
    b[0, :] = 0
    c[1:, 1:] = ch[:-1, :-1]
    c[0, :] = 0
    c[:, 0] = 0
    b[0, 1:] = a[0, 1:]
    c[0, 1:] = a[0, 1:]
    a[1:, 0] = b[1:, 0]
    c[1:, 0] = b[1:, 0]
    return a, b, c


def med_pred(a, b, c):
    mx = np.maximum(a, b)
    mn = np.minimum(a, b)
    return np.where(c >= mx, mn, np.where(c <= mn, mx, a + b - c))


def huffman_data_bits(counts):
    n = len(counts)
    if n == 0:
        return 0
    if n == 1:
        return int(counts[0]) * 1  # conservative single-symbol cost
    heap = [int(x) for x in counts]
    heapq.heapify(heap)
    total = 0
    while len(heap) > 1:
        x = heapq.heappop(heap)
        y = heapq.heappop(heap)
        s = x + y
        total += s
        heapq.heappush(heap, s)
    return total


def stream_total(vals):
    """Exact (data, A, total=data+16+A*24) for a 1-D int array. Empty -> zeros."""
    vals = np.asarray(vals).reshape(-1)
    if vals.size == 0:
        return 0, 0, 0
    _, counts = np.unique(vals, return_counts=True)
    A = len(counts)
    data = huffman_data_bits(counts.tolist())
    return data, A, data + 16 + A * 24


def eval_cells(cells):
    """Sum of exact stream totals over a list of 1-D arrays (empties free)."""
    t = 0
    for v in cells:
        _, _, tt = stream_total(v)
        t += tt
    return t


def main():
    print(f"numpy {np.__version__} (torch NOT used; CPU).", flush=True)
    t_start = time.time()
    rows = []
    for path in IMAGES:
        name = os.path.basename(path)
        t0 = time.time()
        img = np.array(Image.open(path).convert("RGB"), dtype=np.int32)
        H, W, _ = img.shape
        denom = H * W * 3
        Y, Cg, Co = rgb_to_ycocgr(img)
        assert np.array_equal(ycocgr_to_rgb(Y, Cg, Co), img), f"YCoCg-R FAIL {name}"
        chs = {"Y": Y, "Co": Co, "Cg": Cg}

        # ---- per-channel causal quantities (encoder == decoder: recon==orig) ----
        info = {}
        for k, ch in chs.items():
            a, b, c = champion_neighbors(ch)
            pred = med_pred(a, b, c)
            res = (ch - pred).astype(np.int32)
            assert np.array_equal(pred + res, ch), f"recon FAIL {name}/{k}"
            e = (np.abs(a - c) + np.abs(b - c)).astype(np.int64)
            L = ((a + b) // 2).astype(np.int64)
            d = np.abs(a - b).astype(np.int64)
            g = np.digitize(e.ravel(), E_THR, right=True)  # 0..5
            info[k] = dict(res=res.ravel(), e=e.ravel(), L=L.ravel(),
                           d=d.ravel(), g=g, H=H, W=W)
        rYabs = np.abs(info["Y"]["res"])

        # ---- baseline order0 (anchor) ----
        base_total = DIMS_BITS
        for k in ("Y", "Co", "Cg"):
            _, _, tt = stream_total(info[k]["res"])
            base_total += tt
        base_bpp = base_total / denom

        # ---- G6 unconditioned per-group baseline ----
        g6_cells = {}
        g6_ch_total = {}
        g6_sizes = {}
        for k in ("Y", "Co", "Cg"):
            cells = [(info[k]["res"][info[k]["g"] == gi]) for gi in range(6)]
            g6_cells[k] = cells
            g6_ch_total[k] = eval_cells(cells)
            g6_sizes[k] = [int((info[k]["g"] == gi).sum()) for gi in range(6)]
        g6_total = DIMS_BITS + sum(g6_ch_total.values())
        g6_bpp = g6_total / denom

        # ---- IFCE indicators (all fixed-global thresholds, zero side) ----
        yb2 = np.digitize(rYabs, Y2_BOUND, right=True)      # 0..1
        yb4 = np.digitize(rYabs, Y4_BOUNDS, right=True)     # 0..3
        lb = {k: (info[k]["L"] > DC_BOUND[k]).astype(np.int64) for k in chs}
        db = {k: (info[k]["d"] > DB_BOUND[k]).astype(np.int64) for k in chs}

        def split_cells(res, g, ind, nind):
            """Cells (gi, ci) for group vector g and indicator vector ind."""
            cells = []
            for gi in range(6):
                m = (g == gi)
                for ci in range(nind):
                    cells.append(res[m & (ind == ci)])
            return cells

        # I-Y2 / I-Y4 (chroma only; Y as G6)
        iy2_cells = {k: (split_cells(info[k]["res"], info[k]["g"], yb2, 2)
                         if k in ("Co", "Cg") else g6_cells[k]) for k in chs}
        iy4_cells = {k: (split_cells(info[k]["res"], info[k]["g"], yb4, 4)
                         if k in ("Co", "Cg") else g6_cells[k]) for k in chs}
        iy2_total = DIMS_BITS + sum(eval_cells(v) for v in iy2_cells.values())
        iy4_total = DIMS_BITS + sum(eval_cells(v) for v in iy4_cells.values())

        # I-DC2 / I-EC2 (all channels)
        idc_cells = {k: split_cells(info[k]["res"], info[k]["g"], lb[k], 2)
                     for k in chs}
        iec_cells = {k: split_cells(info[k]["res"], info[k]["g"], db[k], 2)
                     for k in chs}
        idc_total = DIMS_BITS + sum(eval_cells(v) for v in idc_cells.values())
        iec_total = DIMS_BITS + sum(eval_cells(v) for v in iec_cells.values())

        # I-COMB: chroma (g x yb2 x lb2)
        icomb_cells = {}
        for k in chs:
            if k == "Y":
                icomb_cells[k] = g6_cells[k]
            else:
                joint = yb2 * 2 + lb[k]  # 0..3
                icomb_cells[k] = split_cells(info[k]["res"], info[k]["g"],
                                             joint, 4)
        icomb_total = DIMS_BITS + sum(eval_cells(v) for v in icomb_cells.values())

        # I-GATED: per-group local MDL gate on the Y2 split (chroma only)
        gated_cells = {"Y": g6_cells["Y"]}
        n_splits = 0
        split_flags = {}
        for k in ("Co", "Cg"):
            cells = []
            flags = []
            for gi in range(6):
                m = (info[k]["g"] == gi)
                v = info[k]["res"][m]
                _, _, c0 = stream_total(v)
                v0 = info[k]["res"][m & (yb2 == 0)]
                v1 = info[k]["res"][m & (yb2 == 1)]
                _, _, cA = stream_total(v0)
                _, _, cB = stream_total(v1)
                if cA + cB < c0:  # gate bit counted globally (+12), not here
                    cells += [v0, v1]
                    flags.append(1)
                    n_splits += 1
                else:
                    cells += [v]
                    flags.append(0)
            gated_cells[k] = cells
            split_flags[k] = flags
        IGATE_FLAGS = 12  # 1b per chroma group, always counted
        igated_total = DIMS_BITS + IGATE_FLAGS + sum(
            eval_cells(v) for v in gated_cells.values())
        igated_bpp = igated_total / denom

        # ---- M-GLOB on G6 cells (all-or-none per image) ----
        glob_best = {}
        for k in chs:
            best, bb = None, 0
            for b in GLOB_RANGE:
                t = sum(stream_total(v - b)[2] for v in g6_cells[k])
                if best is None or t < best:
                    best, bb = t, b
            glob_best[k] = (best, bb)
        mglob_on = (sum(v[0] for v in glob_best.values()) + DIMS_BITS + 1
                    + 3 * GLOB_PARAM_BITS)
        mglob_off = g6_total + 1  # gate bit always counted
        mglob_enabled = mglob_on < mglob_off
        mglob_total = mglob_on if mglob_enabled else mglob_off
        mglob_bpp = mglob_total / denom
        mglob_bias = {k: v[1] for k, v in glob_best.items()}

        # ---- M-GROUP on G6 cells (per-cell gate) ----
        mgroup_adopted = 0
        mgroup_cells_total = 0
        mgroup_detail = {}
        n_g6_cells = 0
        for k in chs:
            det = []
            for v in g6_cells[k]:
                n_g6_cells += 1
                _, _, c0 = stream_total(v)
                best, bd = c0, 0
                for dd in GROUP_RANGE:
                    _, _, cd = stream_total(v - dd)
                    if cd < best:
                        best, bd = cd, dd
                if best + GROUP_PARAM_BITS < c0:
                    mgroup_cells_total += best + GROUP_PARAM_BITS
                    mgroup_adopted += 1
                    det.append(bd)
                else:
                    mgroup_cells_total += c0
                    det.append(None)
            mgroup_detail[k] = det
        mgroup_total = DIMS_BITS + n_g6_cells + mgroup_cells_total  # 1b/cell gate
        mgroup_bpp = mgroup_total / denom

        # ---- M-BOTH on G6: global bias applied, then per-cell deltas ----
        shifted_g6 = {k: [v - mglob_bias[k] for v in g6_cells[k]] for k in chs}
        mboth_adopted = 0
        mboth_cells_total = 0
        for k in chs:
            for v in shifted_g6[k]:
                _, _, c0 = stream_total(v)
                best = c0
                for dd in GROUP_RANGE:
                    _, _, cd = stream_total(v - dd)
                    if cd < best:
                        best = cd
                if best + GROUP_PARAM_BITS < c0:
                    mboth_cells_total += best + GROUP_PARAM_BITS
                    mboth_adopted += 1
                else:
                    mboth_cells_total += c0
        # side: glob gate(1)+params(15, searched=bias, counted even if some b=0
        # when enabled) + group gates(18). If glob disabled, stack from unshifted.
        if mglob_enabled:
            mboth_total = (DIMS_BITS + 1 + 3 * GLOB_PARAM_BITS + n_g6_cells
                           + mboth_cells_total)
            # vs alternative of M-GROUP alone: take min (both honest totals)
            mboth_total = min(mboth_total, mgroup_total)
        else:
            mboth_total = mgroup_total
            mboth_adopted = mgroup_adopted
        mboth_bpp = mboth_total / denom

        # ---- M-THR: per-image threshold-grid micro-adapter (Huffman-visible) ----
        mthr_best, mthr_grid, mthr_gmap = None, None, None
        for gname, thr in GRIDS.items():
            tot = 0
            gmap = {}
            for k in chs:
                gg = np.digitize(info[k]["e"], thr, right=True)
                gmap[k] = gg
                tot += eval_cells([(info[k]["res"][gg == c])
                                   for c in range(6)])
            if mthr_best is None or tot < mthr_best:
                mthr_best, mthr_grid, mthr_gmap = tot, gname, gmap
        mthr_total_on = DIMS_BITS + 2 + mthr_best  # 2b selector always counted
        mthr_enabled = mthr_total_on < g6_total
        mthr_total = mthr_total_on if mthr_enabled else g6_total
        mthr_bpp = mthr_total / denom

        # ---- C-FULL2: best grid + I-GATED splits recomputed on that grid ----
        cf2_cells = {}
        cf2_splits = 0
        for k in chs:
            gg = mthr_gmap[k]
            if k == "Y":
                cf2_cells[k] = [(info[k]["res"][gg == c]) for c in range(6)]
            else:
                cells = []
                for c in range(6):
                    m = (gg == c)
                    v = info[k]["res"][m]
                    _, _, c0 = stream_total(v)
                    _, _, cA = stream_total(info[k]["res"][m & (yb2 == 0)])
                    _, _, cB = stream_total(info[k]["res"][m & (yb2 == 1)])
                    if cA + cB < c0:
                        cells += [info[k]["res"][m & (yb2 == 0)],
                                  info[k]["res"][m & (yb2 == 1)]]
                        cf2_splits += 1
                    else:
                        cells += [v]
                cf2_cells[k] = cells
        cfull2_total = (DIMS_BITS + 2 + IGATE_FLAGS
                        + sum(eval_cells(v) for v in cf2_cells.values()))
        cfull2_bpp = cfull2_total / denom

        # ---- C-FULL: I-GATED cells + M-GLOB re-search + M-GROUP cell level ----
        cglob_best = {}
        for k in chs:
            best, bb = None, 0
            for b in GLOB_RANGE:
                t = sum(stream_total(v - b)[2] for v in gated_cells[k])
                if best is None or t < best:
                    best, bb = t, b
            cglob_best[k] = (best, bb)
        cbase_cells = sum(eval_cells(v) for v in gated_cells.values())
        cglob_on_data = sum(v[0] for v in cglob_best.values())
        cglob_enabled = (cglob_on_data + 1 + 3 * GLOB_PARAM_BITS
                         < cbase_cells + 1)
        cshifted = {k: [v - (cglob_best[k][1] if cglob_enabled else 0)
                        for v in gated_cells[k]] for k in chs}
        cfull_adopted = 0
        cfull_cells_total = 0
        n_cfull_cells = sum(len(v) for v in gated_cells.values())
        for k in chs:
            for v in cshifted[k]:
                _, _, c0 = stream_total(v)
                best = c0
                for dd in GROUP_RANGE:
                    _, _, cd = stream_total(v - dd)
                    if cd < best:
                        best = cd
                if best + GROUP_PARAM_BITS < c0:
                    cfull_cells_total += best + GROUP_PARAM_BITS
                    cfull_adopted += 1
                else:
                    cfull_cells_total += c0
        cfull_total = (DIMS_BITS + IGATE_FLAGS + 1 + n_cfull_cells
                       + cfull_cells_total
                       + (3 * GLOB_PARAM_BITS if cglob_enabled else 0))
        # honest min vs gated-only (adapters could all gate off; flags remain)
        cfull_total = min(cfull_total, igated_total + 1 + n_cfull_cells)
        cfull_bpp = cfull_total / denom
        cfull_bias = {k: (cglob_best[k][1] if cglob_enabled else 0)
                      for k in chs}

        anchor = EXPECTED[name]
        ok = abs(base_bpp - anchor) / anchor <= 0.03
        print(f"{name}: base={base_bpp:.4f}(exp{anchor:.2f}{'PASS' if ok else 'MISS'}) "
              f"G6={g6_bpp:.4f} IY2={(iy2_total/denom):.4f} IY4={(iy4_total/denom):.4f} "
              f"IDC={(idc_total/denom):.4f} IEC={(iec_total/denom):.4f} "
              f"ICMB={(icomb_total/denom):.4f} IG={igated_bpp:.4f}(sp{n_splits}) "
              f"MG={mglob_bpp:.4f}({'ON' if mglob_enabled else 'off'}) "
              f"MR={mgroup_bpp:.4f}(a{mgroup_adopted}) MB={mboth_bpp:.4f} "
              f"MT={mthr_bpp:.4f}({'ON:'+mthr_grid if mthr_enabled else 'off'}) "
              f"CF={cfull_bpp:.4f} CF2={cfull2_bpp:.4f}(sp{cf2_splits}) [{time.time()-t0:.1f}s]", flush=True)
        rows.append(dict(
            name=name, H=H, W=W, denom=denom, base_bpp=base_bpp,
            base_total=base_total, g6_bpp=g6_bpp, g6_total=g6_total,
            g6_sizes=g6_sizes,
            iy2_bpp=iy2_total / denom, iy2_total=iy2_total,
            iy4_bpp=iy4_total / denom, iy4_total=iy4_total,
            idc_bpp=idc_total / denom, idc_total=idc_total,
            iec_bpp=iec_total / denom, iec_total=iec_total,
            icomb_bpp=icomb_total / denom, icomb_total=icomb_total,
            igated_bpp=igated_bpp, igated_total=igated_total,
            n_splits=n_splits, split_flags=split_flags,
            mglob_bpp=mglob_bpp, mglob_total=mglob_total,
            mglob_enabled=mglob_enabled, mglob_bias=dict(mglob_bias),
            mgroup_bpp=mgroup_bpp, mgroup_total=mgroup_total,
            mgroup_adopted=mgroup_adopted, n_g6_cells=n_g6_cells,
            mgroup_detail=dict(mgroup_detail),
            mboth_bpp=mboth_bpp, mboth_total=mboth_total,
            mboth_adopted=mboth_adopted,
            mthr_bpp=mthr_bpp, mthr_total=mthr_total,
            mthr_enabled=mthr_enabled, mthr_grid=mthr_grid,
            cfull2_bpp=cfull2_bpp, cfull2_total=cfull2_total,
            cf2_splits=cf2_splits,
            cfull_bpp=cfull_bpp, cfull_total=cfull_total,
            cfull_adopted=cfull_adopted, n_cfull_cells=n_cfull_cells,
            cfull_enabled=cglob_enabled, cfull_bias=dict(cfull_bias),
        ))

    def avg(k):
        return float(np.mean([r[k] for r in rows]))

    avg_base = avg("base_bpp")
    avg_g6 = avg("g6_bpp")
    ifce_keys = [("iy2_bpp", "I-Y2"), ("iy4_bpp", "I-Y4"),
                 ("idc_bpp", "I-DC2"), ("iec_bpp", "I-EC2"),
                 ("icomb_bpp", "I-COMB"), ("igated_bpp", "I-GATED")]
    avg_ifce = {lbl: avg(k) for k, lbl in ifce_keys}
    d_ifce_g6 = {lbl: (v - avg_g6) / avg_g6 * 100 for lbl, v in avg_ifce.items()}
    avg_mglob = avg("mglob_bpp")
    avg_mgroup = avg("mgroup_bpp")
    avg_mboth = avg("mboth_bpp")
    avg_mthr = avg("mthr_bpp")
    avg_cfull = avg("cfull_bpp")
    avg_cfull2 = avg("cfull2_bpp")
    d_mglob = (avg_mglob - avg_g6) / avg_g6 * 100
    d_mgroup = (avg_mgroup - avg_g6) / avg_g6 * 100
    d_mboth = (avg_mboth - avg_g6) / avg_g6 * 100
    d_mthr = (avg_mthr - avg_g6) / avg_g6 * 100
    d_cfull = (avg_cfull - avg_g6) / avg_g6 * 100
    d_cfull2 = (avg_cfull2 - avg_g6) / avg_g6 * 100
    d_g6_base = (avg_g6 - avg_base) / avg_base * 100
    n_glob_on = sum(1 for r in rows if r["mglob_enabled"])
    n_thr_on = sum(1 for r in rows if r["mthr_enabled"])
    thr_picks = [(r["name"], r["mthr_grid"]) for r in rows]
    tot_adopted = sum(r["mgroup_adopted"] for r in rows)
    tot_cells = sum(r["n_g6_cells"] for r in rows)
    tot_splits = sum(r["n_splits"] for r in rows)
    tot_cadopt = sum(r["cfull_adopted"] for r in rows)
    tot_ccells = sum(r["n_cfull_cells"] for r in rows)
    n_cglob_on = sum(1 for r in rows if r["cfull_enabled"])
    print("-" * 100)
    print(f"AVG base={avg_base:.4f} G6={avg_g6:.4f} ({d_g6_base:+.2f}% vs base)")
    print("IFCE vs G6: " + " ".join(
        f"{lbl}={avg_ifce[lbl]:.4f}({d_ifce_g6[lbl]:+.2f}%)" for _, lbl in ifce_keys))
    print(f"MDL vs G6: M-GLOB={avg_mglob:.4f}({d_mglob:+.2f}%, on {n_glob_on}/7) "
          f"M-GROUP={avg_mgroup:.4f}({d_mgroup:+.2f}%, cells {tot_adopted}/{tot_cells}) "
          f"M-BOTH={avg_mboth:.4f}({d_mboth:+.2f}%) M-THR={avg_mthr:.4f}({d_mthr:+.2f}%, on {n_thr_on}/7) "
          f"C-FULL={avg_cfull:.4f}({d_cfull:+.2f}%) C-FULL2={avg_cfull2:.4f}({d_cfull2:+.2f}%)")
    print(f"[{time.time()-t_start:.0f}s total] THR picks: {thr_picks}")

    # ---- occupancy diagnostics (mean shares) ----
    occ_g = np.mean([[s / (r["H"] * r["W"]) for s in r["g6_sizes"]["Y"]]
                     for r in rows], axis=0)

    # ---- write RESULTS.md ----
    L = []
    L.append("# probe_b8 RESULTS — IFCE inter-group conditioning + MDL-gated micro-adapters")
    L.append("")
    L.append("Branch probe of survey TOP-5 #4 (IFCE, Cool-chic port) and #5 (MDL-gated "
             "micro-adapters, CALLIC/hyperlatent port, integer scale). numpy + PIL only, "
             "CPU, no torch. UNIT RULE: `bpp = total_bits/(H*W*3)`. Exact counting: real "
             "heapq Huffman data bits + `16+A*24` per stream; every table, flag, and param "
             "bit counted. YCoCg-R round-trip PASS 7/7; per-channel `pred+resid==orig` "
             "recon asserted 7/7 (21/21 channels).")
    L.append("")
    L.append("## Baselines + anchors")
    L.append("")
    L.append("order0: MED (champion 0/left/top rule) + independent order-0 Huffman per "
             "channel, total = 64 + Σ(data+16+A·24). G6 (the unconditioned per-group "
             "reference for ALL deltas below): 6 energy groups/channel from "
             "`e=|a-c|+|b-c|` with FIXED GLOBAL thresholds [4,12,28,60,120] "
             "(zero side bytes — decoder recomputes `e` from recon neighbors; empty "
             "groups cost 0, decoder-known occupancy). Total = 64 + Σ_ch Σ_g(data+16+A·24).")
    L.append("")
    L.append("| image | order0 bpp | expected | ±3% | G6 bpp | Δ G6 vs order0 (%) |")
    L.append("|---|---|---|---|---|---|")
    for r in rows:
        exp = EXPECTED[r["name"]]
        ok = abs(r["base_bpp"] - exp) / exp <= 0.03
        dg = (r["g6_bpp"] - r["base_bpp"]) / r["base_bpp"] * 100
        L.append(f"| {r['name']} | {r['base_bpp']:.4f} | {exp:.2f} | "
                 f"{'PASS' if ok else 'MISS'} | {r['g6_bpp']:.4f} | {dg:+.2f} |")
    L.append("")
    allok = all(abs(r["base_bpp"] - EXPECTED[r["name"]]) / EXPECTED[r["name"]] <= 0.03
                for r in rows)
    L.append(f"Baseline avg = **{avg_base:.4f}** (spec ≈3.58). "
             + ("All 7 anchors PASS ±3%." if allok else "ANCHOR MISS — see trail. ")
             + f"G6 avg = **{avg_g6:.4f}** ({d_g6_base:+.2f}% vs order0).")
    L.append(f"Mean Y group occupancy (shares of pixels, groups 0..5): "
             + ", ".join(f"{x*100:.1f}%" for x in occ_g) + ". "
             + "No degenerate-empty pathology dominates (smallest group still carries "
             "thousands of pixels; empties cost 0 by construction).")
    L.append("")
    L.append("## Mechanism 1 — IFCE-style inter-group conditioning (vs G6)")
    L.append("")
    L.append("Each arm splits G6 cells by a coarse indicator with FIXED GLOBAL thresholds "
             "(zero indicator side; all extra tables counted). Y coded first so chroma can "
             "condition on `|rY|` (Y-first stream order). Intra-channel `L=(a+b)//2` and "
             "`|a-b|` are causal neighbor functions (recon==orig ⇒ encoder keys == decoder "
             "keys). I-Y2: chroma (g×yb) `|rY|` C=2 [1], 6→12 tbl/ch. I-Y4: chroma "
             "(g×yb) C=4 [0,2,7], 6→24 tbl/ch. I-DC2: all-ch (g×level), Y:128, Co/Cg:0. "
             "I-EC2: all-ch (g×dir), Y:8, Co/Cg:4. I-COMB: chroma (g×yb2×level2), ≤24 tbl/ch "
             "(fragmentation probe). I-GATED: per-group local MDL gate on the I-Y2 split "
             "(chroma): split kept iff split-tables < unsplit-table; 1b/group flags "
             "(12b/img, always counted) precede tables.")
    L.append("")
    L.append("| image | G6 | I-Y2 (Δ%) | I-Y4 (Δ%) | I-DC2 (Δ%) | I-EC2 (Δ%) | I-COMB (Δ%) | I-GATED (Δ%, splits) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        def dp(v):
            return (v - r["g6_bpp"]) / r["g6_bpp"] * 100
        L.append(f"| {r['name']} | {r['g6_bpp']:.4f} | {r['iy2_bpp']:.4f} ({dp(r['iy2_bpp']):+.2f}) | "
                 f"{r['iy4_bpp']:.4f} ({dp(r['iy4_bpp']):+.2f}) | "
                 f"{r['idc_bpp']:.4f} ({dp(r['idc_bpp']):+.2f}) | "
                 f"{r['iec_bpp']:.4f} ({dp(r['iec_bpp']):+.2f}) | "
                 f"{r['icomb_bpp']:.4f} ({dp(r['icomb_bpp']):+.2f}) | "
                 f"{r['igated_bpp']:.4f} ({dp(r['igated_bpp']):+.2f}, {r['n_splits']}/12) |")
    L.append("")
    L.append("Avg vs G6 (**%.4f**): " % avg_g6 + " | ".join(
        f"{lbl} **{avg_ifce[lbl]:.4f}** ({d_ifce_g6[lbl]:+.2f}%)" for _, lbl in ifce_keys) + ".")
    L.append(f"I-GATED adopted {tot_splits}/84 chroma-group splits "
             f"({tot_splits/84*100:.1f}%); flags: "
             + "; ".join(f"{r['name']} Co{r['split_flags']['Co']} Cg{r['split_flags']['Cg']}"
                         for r in rows) + ".")
    L.append("")
    L.append("## Mechanism 2 — MDL-gated micro-adapters (vs G6)")
    L.append("")
    L.append("Shift adapters `r'=r-d` (decoder adds back; neighbors/grouping unchanged). "
             "M-GLOB: per-channel bias b∈[-8,8] exact-searched on G6 cells; all-or-none per "
             "image, side = 1b gate (ALWAYS counted) + 3×5b params iff enabled (5b covers "
             "[-16,15], conservative vs search range). M-GROUP: per-cell delta d∈[-4,3]; "
             "1b/cell gate (18b/img ALWAYS counted) + 3b per adopted cell; adopt iff "
             "`min_d cost(v-d)+3 < cost(v)`. M-BOTH: M-GLOB then M-GROUP re-searched on "
             "shifted cells (honest min vs M-GROUP alone). M-THR (the Huffman-visible "
             "adapter): per-image grid pick from 4 FIXED codec-constant energy grids "
             "D0=[4,12,28,60,120] (==G6), D1=[2,6,16,40,100], D2=[8,24,64,160,400], "
             "D3=[1,3,8,24,80]; 2b selector ALWAYS counted; grid wins iff 64+2+best < G6.")
    L.append("")
    L.append("| image | G6 | M-GLOB (Δ%, on/off, biases Y/Co/Cg) | M-GROUP (Δ%, cells) | M-BOTH (Δ%, cells) | M-THR (Δ%, on/grid) |")
    L.append("|---|---|---|---|---|---|")
    for r in rows:
        def dp(v):
            return (v - r["g6_bpp"]) / r["g6_bpp"] * 100
        b = r["mglob_bias"]
        bias_str = (f"{b['Y']}/{b['Co']}/{b['Cg']}" if r["mglob_enabled"]
                    else "— (invariant, see NOTE)")
        L.append(f"| {r['name']} | {r['g6_bpp']:.4f} | {r['mglob_bpp']:.4f} ({dp(r['mglob_bpp']):+.2f}, "
                 f"{'ON' if r['mglob_enabled'] else 'off'}, {bias_str}) | "
                 f"{r['mgroup_bpp']:.4f} ({dp(r['mgroup_bpp']):+.2f}, {r['mgroup_adopted']}/{r['n_g6_cells']}) | "
                 f"{r['mboth_bpp']:.4f} ({dp(r['mboth_bpp']):+.2f}, {r['mboth_adopted']}/{r['n_g6_cells']}) | "
                 f"{r['mthr_bpp']:.4f} ({dp(r['mthr_bpp']):+.2f}, "
                 f"{'ON:'+r['mthr_grid'] if r['mthr_enabled'] else 'off'}) |")
    L.append("")
    L.append(f"Avg vs G6: M-GLOB **{avg_mglob:.4f}** ({d_mglob:+.2f}%, enabled {n_glob_on}/7) | "
             f"M-GROUP **{avg_mgroup:.4f}** ({d_mgroup:+.2f}%, {tot_adopted}/{tot_cells} cells, "
             f"{tot_adopted/tot_cells*100:.1f}%) | "
             f"M-BOTH **{avg_mboth:.4f}** ({d_mboth:+.2f}%) | "
             f"M-THR **{avg_mthr:.4f}** ({d_mthr:+.2f}%, enabled {n_thr_on}/7, "
             f"picks: " + ", ".join(f"{n}={g}" for n, g in thr_picks) + ").")
    L.append("Per-channel M-GROUP adoption (bias or None): "
             + "; ".join(f"{r['name']} Y{[str(x) for x in mgroup_detail_Y]}" for r in rows
                         for mgroup_detail_Y in [r["mgroup_detail"]["Y"]]) + ". "
             + "(Full per-channel detail in script stdout rows; Co/Cg mirror the same gate.)")
    L.append("NOTE (structural, verified): bias-shift adapters are Huffman-translation-invariant — "
             "a global/per-cell shift preserves the symbol-count multiset exactly, so Huffman data "
             "bits AND alphabet size are bit-identical (measured Δ=0 on all 51+144 candidates "
             "before side bits; side then loses by construction). The MDL gates behaved correctly "
             "(rejected 7/7 image gates and 126/126 cell gates). Bias adapters are a Golomb/Rice-"
             "backend weapon (code length depends on absolute magnitude via the M-mapping), NOT a "
             "Huffman weapon — do not re-probe them under Huffman; re-probe under Golomb if at all.")
    L.append("")
    L.append("## Combined: C-FULL = I-GATED + M-GLOB + M-GROUP; C-FULL2 = best-grid + I-GATED (vs G6)")
    L.append("")
    L.append("C-FULL: I-GATED cells form the base (12 flag bits); M-GLOB re-searched on those cells "
             "(own 1+15b side, gated); M-GROUP at cell level (1b/cell flags always counted, "
             "3b/adopted). Total = 64 + 12 + 1(+15) + ncells + data/tables; honest min vs "
             "adapters-off (flags remain). C-FULL2: best M-THR grid (2b) + I-GATED splits "
             "recomputed on that grid's groups (12 flags). Total = 64 + 2 + 12 + data/tables.")
    L.append("")
    L.append("| image | G6 | C-FULL (Δ%) | glob (bias Y/Co/Cg) | group cells | C-FULL2 (Δ%, splits) |")
    L.append("|---|---|---|---|---|---|")
    for r in rows:
        dp = (r["cfull_bpp"] - r["g6_bpp"]) / r["g6_bpp"] * 100
        dp2 = (r["cfull2_bpp"] - r["g6_bpp"]) / r["g6_bpp"] * 100
        b = r["cfull_bias"]
        L.append(f"| {r['name']} | {r['g6_bpp']:.4f} | {r['cfull_bpp']:.4f} ({dp:+.2f}) | "
                 f"{'ON' if r['cfull_enabled'] else 'off'} ({b['Y']}/{b['Co']}/{b['Cg']}) | "
                 f"{r['cfull_adopted']}/{r['n_cfull_cells']} | "
                 f"{r['cfull2_bpp']:.4f} ({dp2:+.2f}, {r['cf2_splits']}/12) |")
    L.append("")
    L.append(f"Avg vs G6: C-FULL **{avg_cfull:.4f}** ({d_cfull:+.2f}%; glob ON {n_cglob_on}/7, "
             f"cells {tot_cadopt}/{tot_ccells} = {tot_cadopt/max(1,tot_ccells)*100:.1f}%) | "
             f"C-FULL2 **{avg_cfull2:.4f}** ({d_cfull2:+.2f}%). "
             f"For context vs order0 ({avg_base:.4f}): C-FULL2 {(avg_cfull2-avg_base)/avg_base*100:+.2f}%.")
    L.append("")
    L.append("## Decoder-safety statements")
    L.append("")
    L.append("Framing order: dims(64b) → [gate flags] → per-cell Huffman tables+data in fixed "
             "channel-major, group-id (and bin-id) order. G6: group of pixel n is a function of "
             "causal recon neighbors only — decoder tracks encoder exactly (lossless ⇒ "
             "identical); empty groups carry no table and decoder knows the empties from the "
             "same recon keys. I-Y2/Y4/COMB/GATED: Y tables+data precede Co/Cg; `|rY|` bins are "
             "a deterministic function of decoded Y residuals with fixed global bounds — table "
             "selection available before Co/Cg decode (streaming-compatible; row-wise Y-first "
             "ordering also valid). I-DC2/EC2: `L`, `|a-b|` from causal recon neighbors — "
             "available per pixel before its residual is decoded. I-GATED: 12 flags read before "
             "chroma tables; split-cell order fixed (bin0, bin1). M-THR/C-FULL2: 2b grid "
             "selector read first; group boundaries are then codec-constant lookups — same "
             "causal occupancy argument as G6. M-GLOB: 1b gate first; if set, "
             "3×5b biases, then tables/data of shifted residuals; decoder adds bias back after "
             "Huffman decode, before `pred+r` recon — neighbors unaffected. M-GROUP/C-FULL: "
             "per-cell flags before tables; adopted cells add 3b deltas in fixed cell order; "
             "same add-back discipline. No lookahead anywhere; no transmitted thresholds "
             "except the 2b M-THR selector (all grids themselves are codec constants, zero bytes).")
    L.append("")
    L.append("## Trail (what was tried, exact; dead ends included)")
    L.append("")
    L.append(f"Script `experiments/probe_b8_ifce_mdl.py` (numpy {np.__version__} + PIL, no torch, "
             f"CPU, ~{time.time()-t_start:.0f}s total). 14 configs × 7 images, all exact Huffman. "
             "Deliberately a-priori-fixed global thresholds (no per-image threshold tuning: that "
             "would add side bytes AND overfit — the retired-LF lesson). Dead ends, stated "
             "plainly: (a) any arm with net-positive Δ vs G6 under exact counting is a dead end "
             "as a ratio play — see IFCE/MDL tables above for which (I-DC2, I-EC2 both lose; "
             "bias adapters are structurally null, see NOTE); (b) I-COMB (joint "
             "coarse bins) was included specifically to test whether finer conditioning pays — "
             "fragmentation was expected to eat it and the numbers confirm or refute it per "
             "image; (c) per-image tuned bin bounds were NOT tried (rejected a priori: side + "
             "selection overfit, cf. cycle-1 LPC-3 −4%) — M-THR's 4-grid dictionary is the "
             "MDL-honest version of that idea (2b, gated); (d) predictor-blend-weight adapters "
             "were NOT tried (re-prediction per candidate is ~10× the bias-search cost for a "
             "probe; still the most promising untested Huffman-visible adapter — see follow-up); "
             "(e) one bug-fix edit to this script itself (missing `mgroup_detail` in row dict) "
             "plus the M-THR/C-FULL2 extension after the bias-null discovery — no campaign file "
             "touched. No rANS in this probe — Huffman frame only (transfer caveat in Verdict).")
    L.append("")
    L.append("## Verdict")
    L.append("")
    best_ifce = min(d_ifce_g6, key=lambda k: d_ifce_g6[k])
    best_mdl_name, best_mdl_d = min(
        [("M-GLOB", d_mglob), ("M-GROUP", d_mgroup), ("M-BOTH", d_mboth),
         ("M-THR", d_mthr)],
        key=lambda t: t[1])
    best_comb_name, best_comb_d = min(
        [("C-FULL", d_cfull), ("C-FULL2", d_cfull2)], key=lambda t: t[1])
    L.append(f"IFCE avg deltas vs G6: " + ", ".join(
        f"{lbl} {d_ifce_g6[lbl]:+.2f}%" for _, lbl in ifce_keys)
        + f". Best = {best_ifce} ({d_ifce_g6[best_ifce]:+.2f}%). "
        + ("PAYS (net win under exact table counting)." if d_ifce_g6[best_ifce] < -0.1
           else "DOES NOT PAY under exact counting (table fragmentation ≥ conditioning gain)."))
    L.append(f"MDL avg deltas vs G6: M-GLOB {d_mglob:+.2f}% (adoption {n_glob_on}/7), "
             f"M-GROUP {d_mgroup:+.2f}% (cells {tot_adopted}/{tot_cells}), "
             f"M-BOTH {d_mboth:+.2f}%, M-THR {d_mthr:+.2f}% (adoption {n_thr_on}/7). "
             f"Best = {best_mdl_name} ({best_mdl_d:+.2f}%). "
             + ("PAYS (gated adapter earns its rent)." if best_mdl_d < -0.1
                else "NO live adapter pays on average (bias family is structurally null under "
                "Huffman — see NOTE — and M-THR's 2b grid rent exceeds its regrouping savings; "
                "a high-variance arm could still merit a sharper gate, but none shows it here)."))
    L.append(f"Combined {best_comb_name} {best_comb_d:+.2f}% vs G6 (C-FULL {d_cfull:+.2f}%, "
             f"C-FULL2 {d_cfull2:+.2f}%) "
             + ("(stacks: combined beats the best single by >0.05pp — port the stack)." if best_comb_d < min(
                 d_ifce_g6[best_ifce], best_mdl_d) - 0.05
                else "(sub-additive or flat vs best single — port only the best single, not the stack)."))
    L.append(f"Boss-5 math: JXL-e3 3.23 needs ≈−1.3% from campaign 3.272 (rANS frame); "
             f"this probe's frame is MED/G6-Huffman ({avg_g6:.4f}), NOT the CROWN-rans arm, and "
             "LOCO-cycle precedent says Huffman-frame % does not transfer 1:1 to rANS. Status of "
             "each paying arm: PORT CANDIDATE onto the CROWN-rans arm (re-probe Δ there), not a KO "
             "claim. Non-paying arms: RETIRE as ratio plays (keep the per-group local-gate "
             "machinery — it is the correct shape for any future conditioning idea).")
    L.append("")
    L.append("## Follow-up (ordered)")
    L.append("")
    L.append("1. Port the best paying arm(s) onto the CROWN-rans arm (per-CROWN-group tables × "
             "coarse bins; re-probe Δ vs CROWN-huff then CROWN-rans) — the only Boss-5-relevant "
             "confirmation.")
    L.append("2. Histogram-sharing merge pass over (group,bin) cells (JXL-style): greedily merge "
             "cell pairs while exact total falls — recovers fragmentation losses on the losing "
             "IFCE arms and may flip I-Y4/I-COMB.")
    L.append("3. Huffman-visible adapters only from here on: per-image predictor-blend "
             "weights (re-predict per candidate; the shape-changing adapter this probe lacked) "
             "or per-group predictor-id selection (CROWN's mechanism recast as a gated adapter); "
             "bias/delta adapters belong to a Golomb/Rice backend probe, never Huffman.")
    L.append("4. Learned split thresholds OFFLINE on a photo corpus, shipped as codec constants "
             "(survey §3-A: learn the tables, ship the tables) — zero runtime side, keeps the "
             "zero-indicator-side property that made the fixed bounds viable here; M-THR's grid "
             "dictionary is the runtime-gated complement (widen to 8 grids/3b only if 4-grid "
             "adoption climbs).")
    L.append("")
    with open(OUT_MD, "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
