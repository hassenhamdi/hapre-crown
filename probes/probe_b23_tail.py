"""probe_b23: texture-context training tail (b21 follow-up #2).

b21 §4: texture ctx (e.g. ctx8) loss still descending at it199, but 800-iter
runs proved longer training hurts QUANTIZED bits (sharp minima); 200/0.003
keeps the quant-robust LS neighborhood. This probe tests whether it300 +
late-iterate averaging (EMA-to-LS-neighborhood: mean of late checkpoints)
harvests the tail while keeping quant-robustness.

Arms per (image, channel, texture-ctx) unit, identical harness otherwise
(LS-init, Adam 0.003, cosine T_max=iters, bs16384, L1):
  A control : it200, final iterate      (committed config)
  B long    : it300, final iterate      (isolates more-iters effect)
  C longavg : it300, mean of late checkpoints (iters 250,260,...,300)
Eval (all arms): adaptive-int16 quantize + WIRE forward (driver, frozen LUT)
+ L1 per-ctx gate (qb + 113*16+1+3 < mb_ctx), exactly as driver gates.
Metric: quantized gated win indicator + qb delta vs control per unit; assembly
projection per channel. Success bar (Boss-5): texture wins scaling to >=10KB
on 05 / >=5.8KB on 13 at assembly level.

Subjects: kodim05 [C27] + kodim13 [C12] (winning RCTs), all 3 channels,
texture ctx {6,7,8} (top energy bins). Deterministic per-unit seeds
("probe_b23" namespace; single seed — seed-fragility noted per b21).

Usage:
  python3 probes/probe_b23_tail.py --images kodim05.png kodim13.png [--out JSON]
"""
import sys
import os
import time
import json
import argparse
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, "/tmp/opencode/autocompress/src")
sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
sys.path.insert(0, "/tmp/opencode/autocompress/probes")

import probe_b17_rctw as B17
import probe_b21_perctx as B21
import driver_crown6 as D6

torch.set_num_threads(12)

RCTS = {"C6": (0, 6), "C27": (3, 6), "C12": (1, 5)}
WINRCT = {"kodim05.png": "C27", "kodim13.png": "C12"}
TCTX = (6, 7, 8)
SIDE = 113 * 16 + 1 + 3


def train_unit(Fk, yk, fin=12, h=8, iters=200, lr=0.003, seed=0,
               late_avg=False):
    """probe-local copy of B21.train_net harness + optional late averaging.

    Returns dict with final net and (if late_avg) averaged-state net.
    """
    torch.manual_seed(seed)
    Fn = torch.from_numpy((Fk / 128.0).astype(np.float32))
    yn = torch.from_numpy((yk / 128.0).astype(np.float32))
    net = B21.TinyMLP(fin, h)
    with torch.no_grad():
        A = np.concatenate([Fk / 128.0, np.ones((Fk.shape[0], 1))], axis=1)
        Afull, _, _, _ = np.linalg.lstsq(A, yk / 128.0, rcond=None)
        w, b = Afull[:fin], Afull[fin]
        k = min(h, fin)
        W1 = torch.zeros(h, fin)
        W1[:k, :k] = torch.eye(k) * 0.1
        net.fc1.weight.copy_(W1)
        net.fc1.bias.zero_()
        W2 = torch.zeros(1, h)
        W2[0, :k] = torch.from_numpy((w[:k] / 0.1).astype(np.float32))
        net.fc2.weight.copy_(W2)
        net.fc2.bias.copy_(torch.tensor([float(b)], dtype=torch.float32))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    N = Fn.shape[0]
    bse = int(min(16384, N))
    ckpts = []
    net.train()
    for it in range(iters):
        idx = torch.randperm(N)[:bse]
        opt.zero_grad()
        loss = (net(Fn[idx]) - yn[idx]).abs().mean()
        loss.backward()
        opt.step()
        sched.step()
        if late_avg and it >= 250 and it % 10 == 0:
            ckpts.append({n: v.detach().clone()
                          for n, v in net.named_parameters()})
    out = {"final": net}
    if late_avg:
        assert len(ckpts) == 5, len(ckpts)  # iters 250..290 step 10
        avg = {}
        for n in ckpts[0]:
            avg[n] = sum(c[n] for c in ckpts) / len(ckpts)
        import copy
        neta = copy.deepcopy(net)
        with torch.no_grad():
            for n, v in neta.named_parameters():
                v.copy_(avg[n])
        out["avg"] = neta
    return out


def eval_quant(Fk_sub, Psub, net):
    """adaptive-quant + wire forward + hbits, driver-verbatim. Returns dict."""
    neta, scale = B21.adaptive_quantize(net)
    with torch.no_grad():
        p = {n: v.numpy() for n, v in neta.named_parameters()}
    qi = np.concatenate([p["fc1.weight"].reshape(-1), p["fc1.bias"].reshape(-1),
                         p["fc2.weight"].reshape(-1), p["fc2.bias"].reshape(-1)])
    qi16 = np.round(qi * scale).astype(np.int64)
    assert qi16.max() <= 32767 and qi16.min() >= -32768
    q = D6.dequant_net(qi16.astype(np.int16), scale)
    pv = D6.mlp_forward_wire(Fk_sub, q)
    qb = B21.hbits(Psub.astype(np.int32) - pv)
    ma = B21.maxabs_of(net)
    return {"qb": int(qb), "scale": int(scale), "maxw": float(ma)}


def seed_of(*parts):
    import hashlib
    h = hashlib.sha256(("|".join(map(str, parts))).encode()).hexdigest()[:8]
    return int(h, 16)


def run_unit(Fk, yk, Psub, med_bits, tag):
    res = {}
    for arm, iters, late in (("A", 200, False), ("B", 300, False),
                             ("C", 300, True)):
        t0 = time.perf_counter()
        nets = train_unit(Fk, yk, iters=iters,
                          seed=seed_of("probe_b23", tag, arm), late_avg=late)
        dt = time.perf_counter() - t0
        ev = eval_quant(Fk, Psub, nets["final"])
        win = (ev["qb"] + SIDE) < med_bits
        r = {"qb": ev["qb"], "scale": ev["scale"], "win": bool(win),
             "train_s": round(dt, 1)}
        if late:
            eva = eval_quant(Fk, Psub, nets["avg"])
            wina = (eva["qb"] + SIDE) < med_bits
            r["avg"] = {"qb": eva["qb"], "scale": eva["scale"],
                        "win": bool(wina)}
        res[arm] = r
    return res


def run_image(fn):
    t0 = time.perf_counter()
    rct = WINRCT[fn]
    perm, t = RCTS[rct]
    rgb = np.array(Image.open(
        f"/tmp/opencode/autocompress/experiments/real_photos/{fn}").convert("RGB"))
    yc = B17.rct_fwd(rgb, perm, t)
    out = {"file": fn, "rct": rct, "units": []}
    for ci in range(3):
        P = yc[:, :, ci].astype(np.int32)
        F, y, _ = D6.c6_features_targets(P, 12)
        Ff = F.reshape(-1, 12)
        ctx, qs, E = D6.c6_ctx_of_plane(P)
        cf = ctx.reshape(-1)
        med_plane = D6.c6_med_plane(P)
        med_res = (P - med_plane).reshape(-1)
        for k in TCTX:
            m = (cf == k)
            nk = int(m.sum())
            if nk < 100:
                continue
            mb = B21.hbits(med_res[m])
            r = run_unit(Ff[m], y[m], P.reshape(-1)[m], mb,
                         f"{fn}|{rct}|ch{ci}|ctx{k}")
            r.update({"ch": ci, "ctx": k, "n": nk, "med": int(mb)})
            out["units"].append(r)
            da = r["A"]["qb"]
            print(f"{fn} ch{ci} ctx{k} n={nk} med={mb} "
                  f"A:qb={da} win={r['A']['win']} "
                  f"B:qb={r['B']['qb']}({(r['B']['qb']-da)/da*100:+.2f}%) win={r['B']['win']} "
                  f"C:qb={r['C']['qb']}({(r['C']['qb']-da)/da*100:+.2f}%) "
                  f"Cavg:qb={r['C']['avg']['qb']}({(r['C']['avg']['qb']-da)/da*100:+.2f}%) "
                  f"win={r['C']['avg']['win']}", flush=True)
    out["train_s"] = round(time.perf_counter() - t0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="*", default=["kodim05.png", "kodim13.png"])
    ap.add_argument("--out", default="/tmp/opencode/autocompress/probes/probe_b23_nums.json")
    a = ap.parse_args()
    res = [run_image(fn) for fn in a.images]
    with open(a.out, "w") as f:
        json.dump(res, f)
    print("wrote " + a.out, flush=True)


if __name__ == "__main__":
    main()
