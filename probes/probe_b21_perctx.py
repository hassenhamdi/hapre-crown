"""probe_b21: PER-CONTEXT micro-MLPs — neural successor at full strength (new file, no mods).

Branch: per (channel x energy-context 9 classes) micro-MLPs vs MED and vs global-MLP.
Adapts proven harness from train_mlp.py (LS-init, cosine, minibatch L1) — DO NOT modify that file.

Contexts: energy E=|a-b|+|a-c|+|b-c| from causal recon neighbors, 9 quantile bins per
(image,channel). Justification (in RESULTS): LOCO-365 fragments (cycle2/21 lessons);
energy-9 keeps ~43k samples/net for 81-param MLPs, decoder-trivial, no search.
All features decoder-computable from recon (causal only); probe uses original as recon
proxy (valid lossless: recon==original, proven by sequential sanity on 1 image crop).

Accounting: EXACT Huffman bits (heapq +16+A*24, same fn as harness) per channel globally.
Gate L1: per (ch,ctx) subset-hbits MLP-float vs MED (conservative: subset tables).
Gate L2: per channel global-hbits gated-assembly vs MED.
Quant: int16x256 per winning net, re-gate quantized vs MED. Side info counted:
  winning weights 2B/param + 27 selector flags + 48B quantile thresholds.
bpp = total_bits/(H*W*3). YCoCg-R with invertibility asserts.

Usage:
  python3 probe_b21_perctx.py --mode global --images kodim23.png
  python3 probe_b21_perctx.py --mode perctx --images kodim23.png kodim05.png --hidden 8 --nfeat 8 --iters 800
  python3 probe_b21_perctx.py --mode full7 --hidden 8 --nfeat 8 --iters 800
"""
import sys, os, heapq, time, argparse, json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
sys.path.insert(0, "/tmp/opencode/autocompress/experiments/stage3_method")
from codec import rgb_to_ycocg_r, ycocg_r_to_rgb

torch.manual_seed(0)
torch.set_num_threads(12)

D = "/tmp/opencode/autocompress/experiments/real_photos"
NCTX = 9

# ---------- exact bits (identical to train_mlp.py) ----------
def hbits(res):
    vals, cn = np.unique(np.asarray(res).reshape(-1), return_counts=True)
    counts = {int(v): int(c) for v, c in zip(vals.tolist(), cn.tolist())}
    if len(counts) == 1:
        return len(res) * 1 + 16 + 24
    H = [(c, s) for s, c in counts.items()]; heapq.heapify(H); par = {}; nxt = 1 << 28; H2 = H[:]
    while len(H2) > 1:
        a, sa = heapq.heappop(H2); b, sb = heapq.heappop(H2); nn = nxt; nxt += 1
        par[sa] = (nn, 0); par[sb] = (nn, 1); heapq.heappush(H2, (a + b, nn))
    tot = 0
    for s, c in counts.items():
        dd = 0; n = s
        while n in par:
            n = par[n][0]; dd += 1
        tot += c * dd
    return tot + 16 + len(counts) * 24

def med_pred_plane(P):
    Pp = np.pad(P, ((1, 0), (1, 0)), mode="edge").astype(np.int32)
    a = Pp[1:, :-1]; b = Pp[:-1, 1:]; c = Pp[:-1, :-1]
    return np.where(c >= np.maximum(a, b), np.minimum(a, b),
                    np.where(c <= np.minimum(a, b), np.maximum(a, b), a + b - c))

def med_bits(P):
    return hbits((P - med_pred_plane(P)).reshape(-1))

# ---------- features ----------
def causal_planes(P, pad=3):
    """Return causal neighbor planes (int32). All decoder-computable from recon.
    a=L,b=T,c=TL,d=TR(NE),Ww=L2,NNe=T2,L3,T3,TL2,TR2."""
    Pp = np.pad(P, ((pad, 0), (pad, 2)), mode="edge").astype(np.int32)
    H, W = P.shape
    sl = lambda di, dj: Pp[pad + di:pad + di + H, pad + dj:pad + dj + W]
    a = sl(0, -1); b = sl(-1, 0); c = sl(-1, -1); d = sl(-1, +1)
    Ww = sl(0, -2); NNe = sl(-2, 0)
    L3 = sl(0, -3); T3 = sl(-3, 0); TL2 = sl(-2, -2); TR2 = sl(-2, +2)
    # right-edge TR/TR2 replicate (causal: prev row fully known, j+1/j+2 clip)
    return a, b, c, d, Ww, NNe, L3, T3, TL2, TR2

def features_targets(P, nfeat=8):
    a, b, c, d, Ww, NNe, L3, T3, TL2, TR2 = causal_planes(P)
    base = [a, b, c, d, Ww, NNe, (a + b) // 2, np.abs(a - b)]
    if nfeat == 8:
        F = np.stack(base, -1)
    elif nfeat == 12:
        F = np.stack(base + [L3, T3, TL2, TR2], -1)
    else:
        raise ValueError(nfeat)
    return F, P.reshape(-1).astype(np.float32), (a, b, c)

class TinyMLP(nn.Module):
    def __init__(self, fin=8, h=8):
        super().__init__()
        self.fc1 = nn.Linear(fin, h)
        self.fc2 = nn.Linear(h, 1)
    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x))).squeeze(-1)

def nparams_of(fin, h):
    return fin * h + h + h * 1 + 1

# ---------- training (proven harness pattern: LS-init, cosine, minibatch L1) ----------
def train_net(F, y, fin=8, h=8, iters=800, lr=0.01, bs=16384, log_curve=False, ridge=0.0):
    Fn = torch.from_numpy((F / 128.0).astype(np.float32))
    yn = torch.from_numpy((y / 128.0).astype(np.float32))
    net = TinyMLP(fin, h)
    with torch.no_grad():
        A = np.concatenate([F / 128.0, np.ones((F.shape[0], 1))], axis=1)
        if ridge > 0:
            G = A.T @ A + ridge * np.eye(A.shape[1])
            Afull = np.linalg.solve(G, A.T @ (y / 128.0))
        else:
            Afull, _, _, _ = np.linalg.lstsq(A, y / 128.0, rcond=None)
        w, b = Afull[:fin], Afull[fin]
        # LS-init generalized to h!=fin (h==fin==8 is the exact proven case):
        # fc1 = 0.1*I on the leading min(h,fin) block (tanh near-linear at start),
        # fc2 = w/0.1 on the same block, extras start dead (Adam recruits them).
        # NOTE h8f12: lag-3 feats start ignored at init (documented handicap for that variant).
        k = min(h, fin)
        W1 = torch.zeros(h, fin); W1[:k, :k] = torch.eye(k) * 0.1
        net.fc1.weight.copy_(W1)
        net.fc1.bias.zero_()
        W2 = torch.zeros(1, h)
        W2[0, :k] = torch.from_numpy((w[:k] / 0.1).astype(np.float32))
        net.fc2.weight.copy_(W2)
        net.fc2.bias.copy_(torch.tensor([float(b)], dtype=torch.float32))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    N = Fn.shape[0]
    bse = int(min(bs, N))
    curve = []
    net.train()
    for it in range(iters):
        idx = torch.randperm(N)[:bse]
        opt.zero_grad()
        loss = (net(Fn[idx]) - yn[idx]).abs().mean()
        if log_curve and (it % 100 == 0 or it == iters - 1):
            curve.append((it, float(loss.detach())))
        loss.backward()
        opt.step(); sched.step()
    return net, curve

@torch.no_grad()
def predict_plane(net, F):
    Fn = torch.from_numpy((F / 128.0).astype(np.float32))
    out = []
    bs = 65536
    net.eval()
    for i in range(0, Fn.shape[0], bs):
        out.append(net(Fn[i:i + bs]).numpy() * 128.0)
    return np.concatenate(out).round().astype(np.int32)

SCALES = (256, 512, 1024, 2048, 4096)  # int16 family; 3-bit scale id

def maxabs_of(net):
    with torch.no_grad():
        return max(float(p.abs().max()) for p in net.parameters())

def quantize_net(net, scale=256):
    """int16 x scale quantize all params on a copy; return (qnet, clipped_bool)."""
    import copy
    qn = copy.deepcopy(net)
    clipped = False
    with torch.no_grad():
        for p in list(qn.parameters()):
            q = torch.round(p.data * scale)
            if float(q.abs().max()) > 32767:
                clipped = True
            q = q.clamp(-32768, 32767)
            p.data.copy_(q / scale)
    return qn, clipped

def adaptive_quantize(net):
    """Largest scale with no int16 clip + 3-bit id. Returns (qnet, scale, idbits=3)."""
    ma = maxabs_of(net)
    best = 256
    for s in SCALES:
        if ma * s <= 32767.0:
            best = s
    return quantize_net(net, best)[0], best

# ---------- contexts ----------

def ctx_of_plane(P):
    """Energy E=|a-b|+|a-c|+|b-c| (recon-computable ints) + 9 quantile bins per plane.
    Returns (ctxmap HxW int, thresholds int8 array, E)."""
    a, b, c, *_ = causal_planes(P)
    E = np.abs(a - b) + np.abs(a - c) + np.abs(b - c)
    qs = np.quantile(E.reshape(-1).astype(np.float64), np.linspace(0, 1, NCTX + 1)[1:-1])
    thr = np.unique(qs.astype(np.int32))
    # digitize with unique thresholds; map to 0..NCTX-1 via searchsorted then rebalance not needed
    ctx = np.searchsorted(qs, E.reshape(-1)).reshape(E.shape).astype(np.int32)
    ctx = np.clip(ctx, 0, NCTX - 1)
    return ctx, qs.astype(np.float64), E

# ---------- per-image run ----------
def run_image(fn, hidden=8, nfeat=8, iters=200, lr=0.003, do_global=True, log_curves=False):
    t0 = time.perf_counter()
    rgb = np.array(Image.open(os.path.join(D, fn)).convert("RGB"))
    yc = rgb_to_ycocg_r(rgb)
    assert np.array_equal(ycocg_r_to_rgb(yc), rgb), fn + " YCoCg-R invert fail"
    H, W, _ = rgb.shape
    N = H * W * 3
    out = {"file": fn, "HxW": [H, W], "channels": [], "hidden": hidden, "nfeat": nfeat,
           "iters": iters, "lr": lr, "nparams": nparams_of(nfeat, hidden)}
    tot_med = 0
    tot_glob = 0
    tot_ctx_float = 0   # gated float assembly (pre-quant, no side info yet)
    tot_ctx_quant = 0
    side_quant = 0  # side-info bits for quantized per-ctx system
    side_glob = 0
    for ch, nm in enumerate(["Y", "Co", "Cg"]):
        P = yc[:, :, ch].astype(np.int32)
        mb = med_bits(P)
        med_res = (P - med_pred_plane(P)).reshape(-1)
        F, y, _ = features_targets(P, nfeat)
        # global MLP (mandated x256 check + adaptive-int16 claim, gated)
        gb = None; gwin = False
        if do_global:
            gnet, gcurve = train_net(F.reshape(-1, nfeat), y, nfeat, hidden, iters, lr,
                                     log_curve=log_curves)
            gpred = predict_plane(gnet, F.reshape(-1, nfeat))
            gres = P.reshape(-1).astype(np.int32) - gpred
            gb_float = hbits(gres)
            gnet256, _ = quantize_net(gnet, 256)
            gb_256 = hbits(P.reshape(-1).astype(np.int32) - predict_plane(gnet256, F.reshape(-1, nfeat)))
            gneta, gscale = adaptive_quantize(gnet)
            gresq = P.reshape(-1).astype(np.int32) - predict_plane(gneta, F.reshape(-1, nfeat))
            gb_quant = hbits(gresq)
            wp = nparams_of(nfeat, hidden) * 16
            if gb_quant + wp + 1 + 3 < mb:
                gwin = True
                tot_glob += gb_quant + wp + 1 + 3
                side_glob += wp + 1 + 3
            else:
                tot_glob += mb + 1  # 1 flag to signal MED fallback
                side_glob += 1
                gb_quant = mb
            gb = {"med": mb, "float": gb_float, "q256": gb_256, "quant": gb_quant,
                  "scale": gscale, "maxw": maxabs_of(gnet), "win": gwin, "curve": gcurve}
        # per-context MLPs
        ctx, qs, E = ctx_of_plane(P)
        cf = ctx.reshape(-1)
        float_res = np.empty_like(P.reshape(-1))
        quant_res = np.empty_like(P.reshape(-1))
        ch_side_q = 0
        ch_ctx_info = []
        for k in range(NCTX):
            m = cf == k
            nk = int(m.sum())
            mr = med_res[m]
            mb_ctx = hbits(mr) if nk else 0
            if nk < 100:
                float_res[m] = P.reshape(-1)[m] - med_pred_plane(P).reshape(-1)[m]
                quant_res[m] = med_res[m]
                ch_ctx_info.append({"ctx": k, "n": nk, "med": mb_ctx, "skip": True,
                                    "win_f": False, "win_q": False})
                continue
            Fk = F.reshape(-1, nfeat)[m]; yk = y[m]
            net, curve = train_net(Fk, yk, nfeat, hidden, iters, lr,
                                   log_curve=log_curves and k in (0, NCTX - 1))
            pk = predict_plane(net, Fk)
            rk = P.reshape(-1)[m].astype(np.int32) - pk
            fb = hbits(rk)
            win_f = fb < mb_ctx
            net256, _ = quantize_net(net, 256)
            qb256 = hbits(P.reshape(-1)[m].astype(np.int32) - predict_plane(net256, Fk))
            neta, ascale = adaptive_quantize(net)
            rkq = P.reshape(-1)[m].astype(np.int32) - predict_plane(neta, Fk)
            qb = hbits(rkq)
            wp = nparams_of(nfeat, hidden) * 16
            win_q = (qb + wp + 1 + 3) < mb_ctx
            if win_q:
                quant_res[m] = rkq
                ch_side_q += wp + 1 + 3
            else:
                quant_res[m] = mr
                ch_side_q += 1  # fallback flag still transmitted
            if win_f:
                float_res[m] = rk
            else:
                float_res[m] = mr
            ch_ctx_info.append({"ctx": k, "n": nk, "med": mb_ctx, "float": fb,
                                "q256": qb256, "quant": qb, "scale": ascale,
                                "maxw": maxabs_of(net),
                                "win_f": bool(win_f), "win_q": bool(win_q),
                                "curve": curve if (log_curves and k in (0, NCTX - 1)) else []})
        # channel assemblies with global Huffman (actual reported coding)
        asm_f = hbits(float_res)
        asm_q = hbits(quant_res)
        # per-channel L2 gate for quantized: assembly+side vs MED(+1 ch flag already in side? keep explicit)
        # side counted per-context above (flags+weights); add threshold cost 8*16b per channel
        thr_bits = 8 * 16
        tot_ctx_float += asm_f
        # quantized channel gate: if assembly+side+thr worse than MED, whole channel falls back
        ch_q_total = asm_q + ch_side_q + thr_bits
        ch_fallback = (ch_q_total >= mb)
        if ch_fallback:
            tot_ctx_quant += mb
            ch_side_q = 0
            thr_held = 0
        else:
            tot_ctx_quant += ch_q_total
            thr_held = thr_bits
        side_quant += ch_side_q + thr_held
        tot_med += mb
        out["channels"].append({"name": nm, "med": mb, "global": gb,
                                "ctx_asm_float": asm_f, "ctx_asm_quant": asm_q,
                                "ctx_side_q": ch_side_q, "ctx_thr_held": thr_held,
                                "ctx_fallback": ch_fallback, "ctx": ch_ctx_info,
                                "qs": qs.tolist()})
    # image-level: +3-bit channel map (which channels use per-ctx nets).
    # Global system flags already counted per channel.
    out["tot_med"] = tot_med
    out["tot_glob_gated"] = tot_glob
    out["tot_ctx_float_noside"] = tot_ctx_float
    out["tot_ctx_quant_gated"] = tot_ctx_quant + 3
    out["side_glob"] = side_glob
    out["side_quant"] = side_quant + 3
    out["bpp_med"] = tot_med / N
    out["bpp_glob"] = tot_glob / N
    out["bpp_ctx"] = out["tot_ctx_quant_gated"] / N
    out["train_s"] = time.perf_counter() - t0
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="perctx", choices=["global", "perctx", "full7"])
    ap.add_argument("--images", nargs="*", default=["kodim23.png"])
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--nfeat", type=int, default=8)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--out", default="")
    ap.add_argument("--curves", action="store_true")
    a = ap.parse_args()
    files = a.images
    if a.mode == "full7":
        files = ["kodim01.png", "kodim02.png", "kodim05.png", "kodim07.png",
                 "kodim13.png", "kodim19.png", "kodim23.png"]
    res = []
    for fn in files:
        r = run_image(fn, a.hidden, a.nfeat, a.iters, a.lr, True, a.curves)
        res.append(r)
        print(f"{fn} h={a.hidden} f={a.nfeat}: MED={r['bpp_med']:.4f} "
              f"GLOB={r['bpp_glob']:.4f} ({(r['bpp_glob']-r['bpp_med'])/r['bpp_med']*100:+.2f}%) "
              f"CTXq={r['bpp_ctx']:.4f} ({(r['bpp_ctx']-r['bpp_med'])/r['bpp_med']*100:+.2f}%) "
              f"vsGLOB={(r['bpp_ctx']-r['bpp_glob'])/r['bpp_glob']*100:+.2f}% "
              f"t={r['train_s']:.0f}s", flush=True)
        for c in r["channels"]:
            w = sum(1 for x in c["ctx"] if x.get("win_q"))
            sc = sorted(set(x.get("scale", 0) for x in c["ctx"] if x.get("win_q")))
            print(f"   {c['name']}: med={c['med']} asmF={c['ctx_asm_float']} asmQ={c['ctx_asm_quant']} "
                  f"side={c['ctx_side_q']}+thr{c['ctx_thr_held']} fallback={c['ctx_fallback']} "
                  f"qwins={w}/9 scales={sc} glob_win={c['global']['win'] if c['global'] else '-'} "
                  f"gscale={c['global'].get('scale') if c['global'] else '-'}", flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f)
        print("wrote " + a.out, flush=True)

if __name__ == "__main__":
    main()