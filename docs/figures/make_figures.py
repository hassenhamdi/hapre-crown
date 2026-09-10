"""DCC figure suite (data-backed only). Reads banked ledgers, writes PDF+PNG.

Evidence -> destination: DCC 2027 full paper, single-column 12pt (6in text
width). Figures sized for 6in width. Every coordinate below cites its ledger;
see manifest.json + per-figure CSVs. No RD figure (no banked RD outputs).
Style: Okabe-Ito + markers/linestyles (redundant cues), white background,
constrained layout, DejaVu (Type42 in PDF).
Usage: python3 docs/figures/make_figures.py
"""
import csv
import json
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/tmp/opencode/autocompress/docs/figures"
os.makedirs(OUT, exist_ok=True)

# Okabe-Ito (colorblind-safe qualitative)
OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "red": "#D55E00",
      "purple": "#CC79A7", "gray": "#999999", "black": "#000000"}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

MANIFEST = {"figures": [], "notes": []}


def save(fig, name, alt, sources):
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), **kw)
    MANIFEST["figures"].append({"name": name, "alt": alt, "sources": sources,
                                "files": [f"{name}.pdf", f"{name}.png"]})
    plt.close(fig)


def csv_out(name, header, rows):
    with open(os.path.join(OUT, f"{name}.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ---------------- data (banked) ----------------
IMGS = ["kodim01", "kodim02", "kodim05", "kodim07", "kodim13", "kodim19", "kodim23"]
CROWN6 = [3.2982, 2.9854, 3.5806, 2.6798, 3.9532, 3.1510, 2.7365]
JXLE3 = [3.3593, 3.0611, 3.5120, 2.7310, 3.9141, 3.2154, 2.8110]
FLIF = [3.2252, 2.4281, 3.3757, 2.4190, 3.7011, 3.0253, 2.6146]

# F1: avg ladder (7-avg exact unless marked *)
F1_ROWS = [
    ("QOI", 4.9197, "boss-dead"),
    ("PNG-9", 4.75, "boss-dead"),
    ("JPEG-LS CharLS", 4.5757, "boss-dead"),
    ("JXL-e1", 3.72, "boss-dead"),
    ("WebP-m0", 3.60, "boss-dead"),
    ("CROWN-huff", 3.343, "ours"),
    ("WebP-m3", 3.347, "boss-dead"),
    ("CROWN-rans-hc", 3.309, "ours"),
    ("WebP-m6", 3.3157, "boss-dead"),
    ("CROWN2", 3.2686, "ours"),
    ("JXL-e3", 3.2291, "boss-stands"),
    ("CROWN4", 3.1993, "ours"),
    ("CROWN6", 3.1978, "ours-best"),
    ("FLIF v0.4", 2.9699, "boss-stands"),
    ("JXL-e9", 3.0324, "boss-stands"),
    ("DLPR*", 2.86, "learned"),
    ("CALLIC*", 2.54, "learned"),
]
F1_STATUS = {"boss-dead": ("#56B4E9", " KO'd (7/7, p=.016)"),
             "ours": (OI["green"], " CROWN lineage"),
             "ours-best": (OI["blue"], " CROWN6 champion"),
             "boss-stands": (OI["red"], " standing"),
             "learned": (OI["purple"], " learned (full-24, GPU)")}  # hatched

# ---------------- F1: boss ladder (horizontal bars from 0) ----------------
def fig_f1():
    rows = sorted(F1_ROWS, key=lambda r: -r[1])
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    cols = [F1_STATUS[r[2]][0] for r in rows]
    fig, ax = plt.subplots(figsize=(6.0, 4.2), layout="constrained")
    y = np.arange(len(rows))
    bars = ax.barh(y, vals, color=cols, edgecolor="black", linewidth=0.5)
    for i, r in enumerate(rows):
        if r[2] == "learned":
            bars[i].set_hatch("///")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.invert_yaxis()
    ax.set_xlabel("Kodak average (bpp, lower is better)")
    ax.set_title("Boss ladder: exact bytes on Kodak-7 (hatched* = literature full-24)")
    for i, v in enumerate(vals):
        ax.text(v + 0.03, i, f"{v:.3f}", va="center", fontsize=7)
    from matplotlib.patches import Patch
    seen = []
    for r in rows:
        if r[2] not in seen:
            seen.append(r[2])
    handles = [Patch(facecolor=F1_STATUS[s][0], edgecolor="black",
                     hatch="///" if s == "learned" else None,
                     label=F1_STATUS[s][1]) for s in seen]
    ax.legend(handles=handles, loc="lower right", fontsize=7)
    fig.savefig("/tmp/_f1check.png")
    return fig


# ---------------- F1b: per-image heatmap + row winners ----------------
def fig_f1b():
    data = np.array([CROWN6, JXLE3, FLIF])
    fig, ax = plt.subplots(figsize=(6.0, 2.3), layout="constrained")
    im = ax.imshow(data, cmap="Oranges", aspect="auto", interpolation="nearest")
    ax.set_xticks(range(7))
    ax.set_xticklabels([s.replace("kodim", "k") for s in IMGS])
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["CROWN6 3.1978", "JXL-e3 3.2291", "FLIF 2.9699"])
    winners = np.argmin(data, axis=0)
    for j in range(7):
        for i in range(3):
            mark = " ★" if winners[j] == i else ""
            ax.text(j, i, f"{data[i, j]:.2f}{mark}", ha="center", va="center",
                    fontsize=7, color="black" if data[i, j] < 3.6 else "white")
    ax.set_title("Per-image bpp (★ = row winner; FLIF sweeps all rows)")
    fig.colorbar(im, ax=ax, label="bpp")
    return fig


# ---------------- F2: lineage progression ----------------
F2_POINTS = [
    ("HAPRE-C", 3.58), ("CTX9", 3.537), ("RUN", 3.511), ("MOE", 3.464),
    ("CROWN-huff", 3.343), ("CROWN-rans-hc", 3.309), ("CROWN2", 3.2686),
    ("CROWN3", 3.2040), ("CROWN4", 3.1993), ("CROWN6", 3.1978),
]


def fig_f2():
    names = [p[0] for p in F2_POINTS]
    vals = [p[1] for p in F2_POINTS]
    fig, ax = plt.subplots(figsize=(6.0, 3.0), layout="constrained")
    ax.plot(names, vals, marker="o", color=OI["blue"], linestyle="-",
            linewidth=1.5, markersize=5, label="champion avg (exact)")
    for ref, v, ls, c in [("JXL-e3", 3.2291, "--", OI["red"]),
                          ("JXL-e9", 3.0324, ":", OI["red"]),
                          ("FLIF", 2.9699, "-.", OI["orange"])]:
        ax.axhline(v, linestyle=ls, color=c, linewidth=1, label=f"{ref} {v}")
    ax.set_ylabel("Kodak-7 avg bpp")
    ax.set_title("Champion lineage (each step exact-gated)")
    ax.tick_params(axis="x", rotation=30, labelsize=7)
    ax.legend(fontsize=7, loc="upper right")
    ax.set_ylim(2.85, 3.65)
    return fig


# ---------------- F3: ceiling LOO marginals ----------------
def fig_f3():
    import json
    d = json.load(open("/tmp/opencode/autocompress/experiments/probe_b14_summary.json"))
    base = d["FULL-avg"]
    arms = [("noLOCO", d["LOO1-noLOCO-avg"] - base),
            ("E6 (no E16)", d["LOO2-E6-avg"] - base),
            ("H-only", d["LOO3-Honly-avg"] - base),
            ("noMerge", d["LOO4-noMerge-avg"] - base),
            ("noTree", d["LOO5-noTree-avg"] - base),
            ("noGrid", d["LOO6-noGrid-avg"] - base),
            ("noSplit", d["LOO7-noSplit-avg"] - base)]
    arms.sort(key=lambda t: -t[1])
    arms = [(n, 0.0 if abs(v) < 5e-5 else v) for n, v in arms]
    fig, ax = plt.subplots(figsize=(6.0, 2.4), layout="constrained")
    y = np.arange(len(arms))
    ax.barh(y, [a[1] for a in arms], color=OI["blue"], edgecolor="black",
            linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([a[0] for a in arms])
    ax.invert_yaxis()
    ax.set_xlabel("Marginal vs FULL 3.2515 (bpp, probe-level exact-counted)")
    ax.set_title("Classical ceiling leave-one-out (tree/grid subsumed at 0.0000)")
    for i, a in enumerate(arms):
        lab = f"{a[1]:+.4f}" if a[1] != 0 else "0.0000"
        ax.text(a[1] + 0.001, i, lab, va="center", fontsize=7)
    return fig

# ---------------- F4: negative atlas (signed bars from 0) ----------------
# Each bar vs its probe-local baseline (see §5); exact deltas banked.
# 4-tuples: (family, delta_pct, level, baseline). Rows without a traceable
# baseline in the ledgers are EXCLUDED (2026-09-10 audit: byte-LZ +1.84, GAP
# crude mixer +3.70, LPC-3 +4.00, MA-tree-lite -0.56, I-GATED -0.29 have no
# probe-local baseline on disk; see docs/ATLAS_BASELINES.md unresolved set).
F4_ROWS = [
    ("bitplane-B vs joint-C", +26.3, "P", "joint-C 3.3737"),
    ("Squeeze-NN LF-pred", +13.48, "P", "lfpred 10.7306"),
    ("smooth upsample (bilin)", +10.77, "P", "lfup 3.5769"),
    ("LF-cluster cond", +7.50, "P", "lf2 10.7306"),
    ("naive regroup quad", +3.85, "P", "MED 3.5782"),
    ("joint group×act adapt", +3.36, "P", "S0 3.5782"),
    ("hash-cache escape", +2.85, "P", "MED-order-0 3.5769"),
    ("regroup pair", +2.12, "P", "MED 3.5782"),
    ("VQ transfer stack", -1.25, "P", "CROWN2 3.2686"),
    ("A-joint pairs (frame)", -1.58, "P", "MED 3.5769"),
    ("LOCO-I full (Golomb)", -9.90, "P", "MED+Golomb"),
]


def fig_f4():
    rows = sorted(F4_ROWS, key=lambda r: -r[1])
    fig, ax = plt.subplots(figsize=(6.0, 4.2), layout="constrained")
    y = np.arange(len(rows))
    cols = [OI["red"] if v > 0 else OI["green"] for _, v, _, _ in rows]
    ax.barh(y, [r[1] for r in rows], color=cols, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=7)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Delta vs probe-local baseline (% bpp; + worse, − better)")
    ax.set_title("Negative atlas (selected; each bar own baseline — see §5)")
    return fig


# ---------------- F5: Pareto ratio vs decode time ----------------
# decode seconds: kodim23-class single-image measured (†) unless noted avg.
F5_POINTS = [
    # (label, bpp-avg, dec-s, family, marker)
    ("PNG-9", 4.75, 0.013, "classic", "o"),
    ("QOI", 4.9197, 0.0034, "classic", "o"),
    ("HAPRE-C", 3.58, 0.120, "ours", "s"),
    ("RUN", 3.511, 0.120, "ours", "s"),
    ("MOE", 3.464, 0.135, "ours", "s"),
    ("CROWN-huff", 3.359, 0.130, "ours", "s"),
    ("CROWN-rans-hc", 3.309, 0.130, "ours", "s"),
    ("CROWN-E16", 3.285, 0.140, "ours", "s"),
    ("CROWN4", 3.1993, 0.230, "ours", "s"),
    ("CROWN6", 3.1978, 0.265, "ours-best", "D"),
    ("JXL-e1", 3.72, 0.012, "classic", "o"),
    ("JXL-e3", 3.2291, 0.033, "classic", "o"),
    ("JXL-e9", 3.0324, 0.100, "classic", "o"),
    ("FLIF", 2.9699, 0.350, "classic", "o"),
    ("DLPR (GPU)", 2.86, 1.80, "learned", "^"),
    ("CALLIC (GPU)", 2.54, 1.70, "learned", "^"),
    ("HPAC (GPU)", 2.73, 0.90, "learned", "^"),
    ("ArIB-BPS (GPU)", 2.78, 7.0, "learned", "^"),
    ("P2-LLM (8xA800)", 2.83, 273.0, "learned", "^"),
]
F5_COL = {"classic": OI["gray"], "ours": OI["sky"], "ours-best": OI["blue"],
          "learned": OI["purple"]}


def fig_f5():
    fig, ax = plt.subplots(figsize=(6.0, 3.9), layout="constrained")
    # leader-line targets for the tight mid-stack (data coords, empty zone)
    LEAD = {"CROWN-huff": (0.55, 3.375), "CROWN-rans-hc": (0.55, 3.325),
            "CROWN-E16": (0.55, 3.26), "CROWN4": (0.55, 3.205)}
    OFF = {"HAPRE-C": (4, 7), "RUN": (4, -7), "MOE": (4, 3),
           "CROWN6": (4, -10), "P2-LLM (8xA800)": (-4, 0)}
    RIGHT_OF = {"P2-LLM (8xA800)"}
    for fam, mk in (("classic", "o"), ("ours", "s"), ("ours-best", "D"),
                    ("learned", "^")):
        xs = [p[2] for p in F5_POINTS if p[3] == fam]
        ys = [p[1] for p in F5_POINTS if p[3] == fam]
        ls = [p[0] for p in F5_POINTS if p[3] == fam]
        ax.scatter(xs, ys, c=F5_COL[fam], marker=mk, s=45, edgecolors="black",
                   linewidths=0.5, label=fam, zorder=3)
        for x, y, s in zip(xs, ys, ls):
            if s in LEAD:
                # crowded mid-stack: leader line into the empty zone
                ax.annotate(s, (x, y), xytext=LEAD[s],
                            fontsize=6, va="center", ha="left",
                            arrowprops=dict(arrowstyle="-", lw=0.5,
                                            color="black"))
                continue
            dx, dy = OFF.get(s, (4, 0))
            ha = "right" if s in RIGHT_OF else "left"
            ax.annotate(s, (x, y), xytext=(dx, dy), textcoords="offset points",
                        fontsize=6, va="center", ha=ha)
    # CROWN6 decode interval 182-348ms as explicit range bar
    ax.hlines(3.1978, 0.182, 0.348, colors=OI["blue"], linewidths=3, zorder=2)
    ax.set_xscale("log")
    ax.set_xlabel("decode time per image, s (log scale; CPU-exact vs GPU-learned, see caption)")
    ax.set_ylabel("Kodak avg bpp (exact 7-avg vs literature full-24)")
    ax.set_title("Ratio–decode Pareto (CPU exact cluster vs GPU learned cluster)")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, which="major", axis="x", linestyle=":", linewidth=0.5)
    return fig


if __name__ == "__main__":
    import matplotlib
    print("matplotlib", matplotlib.__version__)
    f = fig_f1()
    save(f, "F1_ladder",
         "Horizontal bar chart of Kodak average bpp for 17 codecs: ours from 4.75 down to learned 2.54.",
         ["RESULTS.md boss board", "crown6_results.json",
          "survey/bosstakedown/data/classical7.json",
          "CALLIC Tab.1 (2412.17464 p.6)", "P2-LLM Tab.2 (2411.12448 p.7)"])
    csv_out("F1_ladder", ["codec", "bpp_avg", "status"],
            [[r[0], r[1], r[2]] for r in F1_ROWS])
    f = fig_f1b()
    save(f, "F1b_perimage",
         "Heatmap of per-image bpp for CROWN6, JXL-e3, FLIF across 7 Kodak images with row winners starred.",
         ["crown6_results.json", "survey/bosstakedown/data/ladder.csv",
          "survey/bosstakedown/data/classical7.json"])
    csv_out("F1b_perimage", ["image"] + IMGS,
            [["CROWN6"] + CROWN6, ["JXL-e3"] + JXLE3, ["FLIF"] + FLIF])
    f = fig_f2()
    save(f, "F2_lineage",
         "Line chart of champion average bpp across 10 generations from HAPRE-C 3.58 to CROWN6 3.1978 with boss reference lines.",
         ["RESULTS.md exact-codecs table", "draft Table lineage"])
    csv_out("F2_lineage", ["generation", "bpp_avg"],
            [[p[0], p[1]] for p in F2_POINTS])
    f = fig_f3()
    save(f, "F3_ceiling_loo",
         "Horizontal bars of leave-one-out marginals over the 3.2515 classical ceiling.",
         ["experiments/probe_b14_summary.json"])
    f = fig_f4()
    save(f, "F4_negative_atlas",
         "Signed horizontal bars of dead-end deltas, each versus its probe-local baseline.",
         ["draft §5 with per-probe RESULTS files"])
    csv_out("F4_negative_atlas", ["family", "delta_pct", "level", "baseline"],
            [[r[0], r[1], r[2], r[3]] for r in F4_ROWS])
    f = fig_f5()
    save(f, "F5_pareto",
         "Scatter of average bpp versus decode seconds (log) for CPU exact codecs and GPU learned codecs.",
         ["RESULTS.md decode column", "qoibench (QOI 3.4ms)",
          "djxl/PIL timings (measured)", "CALLIC/HPAC/P2-LLM/DLPR/ArIB papers"])
    csv_out("F5_pareto", ["codec", "bpp_avg", "decode_s", "family"],
            [[p[0], p[1], p[2], p[3]] for p in F5_POINTS])
    MANIFEST["notes"] = [
        "No F6 RD figure: no banked RD sweep outputs (rd_sweep.py exists, outputs not banked).",
        "CROWN2 decode omitted from F5 (unbanked).",
        "Learned points: full-24 Kodak avgs, paper-reported GPU times; ours/classic: 7-avg exact ratios.",
        "F5 CROWN6 xric: explicit interval bar 182-348ms (per-image range), point at midpoint.",
    ]
    with open(os.path.join(OUT, "manifest.json"), "w") as fh:
        json.dump(MANIFEST, fh, indent=1)
    print("wrote figures + manifest to", OUT)
