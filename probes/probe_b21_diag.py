"""probe_b21_diag.py — per-context failure autopsy + remedy sweep (new file).

Qs: (1) max|w| per ctx net (clip suspect)? (2) ridge LS-init fix? (3) lr/iters for
subsets? (4) per-net adaptive int16 scale {256,1024,4096} retain gain?
Runs on kodim23 Y (ctx0 smooth / ctx8 texture) + Co ctx3. No full-image trains.
"""
import sys, os, heapq, time
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
from probe_b21_perctx import (hbits, med_pred_plane, med_bits, features_targets,
                              TinyMLP, nparams_of, predict_plane, ctx_of_plane)
sys.path.insert(0, "/tmp/opencode/autocompress/experiments/stage3_method")
from codec import rgb_to_ycocg_r, ycocg_r_to_rgb

torch.manual_seed(0)
torch.set_num_threads(12)
D = "/tmp/opencode/autocompress/experiments/real_photos"

def maxabs(net):
    with torch.no_grad():
        return max(float(p.abs().max()) for p in net.parameters())

def train_ctx(Fk, yk, fin=8, h=8, iters=200, lr=0.003, ridge=0.0, bs=16384):
    Fn = torch.from_numpy((Fk / 128.0).astype(np.float32))
    yn = torch.from_numpy((yk / 128.0).astype(np.float32))
    net = TinyMLP(fin, h)
    with torch.no_grad():
        A = np.concatenate([Fk / 128.0, np.ones((Fk.shape[0], 1))], axis=1)
        if ridge > 0:
            G = A.T @ A + ridge * np.eye(A.shape[1])
            wfull = np.linalg.solve(G, A.T @ (yk / 128.0))
        else:
            wfull, _, _, _ = np.linalg.lstsq(A, yk / 128.0, rcond=None)
        w, b = wfull[:fin], wfull[fin]
        net.fc1.weight.copy_(torch.eye(h, fin) * 0.1)
        net.fc1.bias.zero_()
        net.fc2.weight.copy_(torch.from_numpy((w / 0.1).astype(np.float32)).unsqueeze(0))
        net.fc2.bias.copy_(torch.tensor([float(b)], dtype=torch.float32))
    init_ma = maxabs(net)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    N = Fn.shape[0]; bse = int(min(bs, N))
    net.train()
    l0 = None; lN = None
    for it in range(iters):
        idx = torch.randperm(N)[:bse]
        opt.zero_grad()
        loss = (net(Fn[idx]) - yn[idx]).abs().mean()
        if it == 0: l0 = float(loss.detach())
        if it == iters - 1: lN = float(loss.detach())
        loss.backward(); opt.step(); sched.step()
    return net, init_ma, maxabs(net), l0, lN

def qbits(net, Fk, tgt, scale):
    import copy
    qn = copy.deepcopy(net)
    with torch.no_grad():
        clip = False
        for p in qn.parameters():
            q = torch.round(p.data * scale)
            if float(q.abs().max()) > 32767: clip = True
            q = q.clamp(-32768, 32767)
            p.data.copy_(q / scale)
    pred = predict_plane(qn, Fk)
    return hbits(tgt - pred), clip

rgb = np.array(Image.open(os.path.join(D, "kodim23.png")).convert("RGB"))
yc = rgb_to_ycocg_r(rgb)
assert np.array_equal(ycocg_r_to_rgb(yc), rgb)
targets = [("Y", 0), ("Y", 8), ("Co", 3)]
planes = {}
for nm, _ in targets:
    ch = {"Y": 0, "Co": 1, "Cg": 2}[nm]
    P = yc[:, :, ch].astype(np.int32)
    F, y, _ = features_targets(P, 8)
    ctx, qs, E = ctx_of_plane(P)
    planes[nm] = (P, F.reshape(-1, 8), y, ctx.reshape(-1))

print("== A: vanilla harness (iters=800, lr=0.01, ridge=0) weight stats + multi-scale quant ==")
for nm, k in targets:
    P, Ff, y, cf = planes[nm]
    m = cf == k
    Fk, yk = Ff[m], y[m]
    tgt = P.reshape(-1)[m].astype(np.int32)
    mb = hbits((tgt - med_pred_plane(P).reshape(-1)[m]))
    net, ima, fma, l0, lN = train_ctx(Fk, yk, 8, 8, 800, 0.01, 0.0)
    pred = predict_plane(net, Fk)
    fb = hbits(tgt - pred)
    row = []
    for s in (256, 1024, 4096):
        qb, clip = qbits(net, Fk, tgt, s)
        row.append(f"x{s}={qb}(clip={clip})")
    print(f"{nm} ctx{k} n={m.sum()}: med={mb} float={fb} ({(mb-fb)/mb*100:+.2f}%) "
          f"init_max|w|={ima:.1f} final_max|w|={fma:.1f} L {l0:.4f}->{lN:.4f} " + " ".join(row), flush=True)

print("== B: remedy grid (ridge x lr/iters) on Y-ctx0 (smooth, ill-cond suspect) ==")
P, Ff, y, cf = planes["Y"]; m = cf == 0
Fk, yk = Ff[m], y[m]; tgt = P.reshape(-1)[m].astype(np.int32)
mb = hbits(tgt - med_pred_plane(P).reshape(-1)[m])
for ridge in (0.0, 10.0, 1000.0):
    for iters, lr in ((200, 0.003), (400, 0.003), (200, 0.01)):
        net, ima, fma, l0, lN = train_ctx(Fk, yk, 8, 8, iters, lr, ridge)
        pred = predict_plane(net, Fk)
        fb = hbits(tgt - pred)
        qb, clip = qbits(net, Fk, tgt, 256)
        qb1, clip1 = qbits(net, Fk, tgt, 1024)
        print(f"ridge={ridge} it={iters} lr={lr}: med={mb} float={fb} ({(mb-fb)/mb*100:+.2f}%) "
              f"q256={qb}({'CLIP' if clip else 'ok'}) q1024={qb1}({'CLIP' if clip1 else 'ok'}) "
              f"maxw={ima:.1f}->{fma:.1f} L{l0:.4f}->{lN:.4f}", flush=True)

print("== C: remedy spot-check on texture ctx (Y-ctx8, Co-ctx3) with best ridge ==")
for nm, k in [("Y", 8), ("Co", 3)]:
    P, Ff, y, cf = planes[nm]; m = cf == k
    Fk, yk = Ff[m], y[m]; tgt = P.reshape(-1)[m].astype(np.int32)
    mb = hbits(tgt - med_pred_plane(P).reshape(-1)[m])
    for ridge in (0.0, 1000.0):
        net, ima, fma, l0, lN = train_ctx(Fk, yk, 8, 8, 200, 0.003, ridge)
        pred = predict_plane(net, Fk)
        fb = hbits(tgt - pred)
        qb, clip = qbits(net, Fk, tgt, 256)
        qb1, clip1 = qbits(net, Fk, tgt, 1024)
        print(f"{nm} ctx{k} ridge={ridge}: med={mb} float={fb} ({(mb-fb)/mb*100:+.2f}%) "
              f"q256={qb}({'CLIP' if clip else 'ok'}) q1024={qb1}({'CLIP' if clip1 else 'ok'}) "
              f"maxw={ima:.1f}->{fma:.1f} L{l0:.4f}->{lN:.4f}", flush=True)
