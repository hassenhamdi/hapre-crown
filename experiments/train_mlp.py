"""Learned attempt: tiny per-(image,channel) MLP predictor (8 causal feats -> pixel), L1 loss, torch CPU."""
import numpy as np, time, sys, os, heapq
import torch
import torch.nn as nn
from PIL import Image
sys.path.insert(0, "/tmp/opencode/autocompress/experiments/stage3_method")
from codec import rgb_to_ycocg_r

torch.manual_seed(0)
torch.set_num_threads(12)


def features_targets(P):
    Pp = np.pad(P, ((1, 0), (1, 0)), mode="edge").astype(np.float32)
    H, W = P.shape
    a = Pp[1:, :-1]; b = Pp[:-1, 1:]; c = Pp[:-1, :-1]
    d = np.zeros_like(a); d[:, :-1] = Pp[:-1, 2:]; d[:, -1] = b[:, -1]
    Ww = np.zeros_like(a); Ww[:, 1:] = Pp[1:, :-2]; Ww[:, 0] = a[:, 0]
    NNe = np.zeros_like(a); NNe[1:, :] = Pp[:-2, 1:]; NNe[0, :] = b[0, :]
    F = np.stack([a, b, c, d, Ww, NNe, (a + b) / 2, np.abs(a - b)], -1)
    return F.reshape(-1, 8), P.reshape(-1).astype(np.float32)


class TinyMLP(nn.Module):
    def __init__(self, h=8):
        super().__init__()
        self.fc1 = nn.Linear(8, h)
        self.fc2 = nn.Linear(h, 1)

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x))).squeeze(-1)


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


def med_bits(P):
    Pp = np.pad(P, ((1, 0), (1, 0)), mode="edge").astype(np.int32)
    a = Pp[1:, :-1]; b = Pp[:-1, 1:]; c = Pp[:-1, :-1]
    pred = np.where(c >= np.maximum(a, b), np.minimum(a, b), np.where(c <= np.minimum(a, b), np.maximum(a, b), a + b - c))
    return hbits((P - pred).reshape(-1))


def train_channel(P, iters=800, h=8, lr=0.01):
    F, y = features_targets(P)
    Fn = torch.from_numpy(F / 128.0)
    yn = torch.from_numpy(y / 128.0)
    net = TinyMLP(h)
    # LS init: solve linear first, load into fc2 with fc1 ~ identity passthrough
    with torch.no_grad():
        A, _, _, _ = np.linalg.lstsq(
            np.concatenate([F / 128.0, np.ones((F.shape[0], 1))], axis=1), y / 128.0, rcond=None)
        w, b = A[:8], A[8]
        # fc1 = identity-ish (8->8): weight eye*0.5, bias 0; fold linear into fc2 is nonlinear... instead:
        # set fc1 to pass through scaled inputs, fc2 to LS weights with tanh linearized (tanh(z)~z near 0):
        net.fc1.weight.copy_(torch.eye(h, 8) * 0.1)
        net.fc1.bias.zero_()
        # tanh(0.5x) ≈ 0.5x - ... ; compensate: fc2 = w / 0.5 * correction for tanh slope at operating point
        net.fc2.weight.copy_(torch.from_numpy((w / 0.1).astype(np.float32)).unsqueeze(0))
        net.fc2.bias.copy_(torch.tensor([float(b)], dtype=torch.float32))
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    N = Fn.shape[0]
    bs = 16384
    net.train()
    for it in range(iters):
        idx = torch.randperm(N)[:bs]
        opt.zero_grad()
        loss = (net(Fn[idx]) - yn[idx]).abs().mean()
        loss.backward()
        opt.step(); sched.step()
    net.eval()
    with torch.no_grad():
        pred = (net(Fn).numpy() * 128.0).round().astype(np.int32)
    res = P.reshape(-1).astype(np.int32) - pred
    return hbits(res), net


if __name__ == "__main__":
    d = "/tmp/opencode/autocompress/experiments/real_photos"
    for fn in sys.argv[1:] or ["kodim23.png"]:
        rgb = np.array(Image.open(os.path.join(d, fn)).convert("RGB"))
        yc = rgb_to_ycocg_r(rgb)
        for k, nm in enumerate(["Y", "Co", "Cg"]):
            P = yc[:, :, k].astype(np.int32)
            mb = med_bits(P)
            t0 = time.perf_counter()
            lb, net = train_channel(P)
            nparams = sum(p.numel() for p in net.parameters())
            print(f"{fn} {nm}: MEDbits={mb} MLPbits={lb} d={(mb-lb)/mb*100:+.2f}% params={nparams} train={time.perf_counter()-t0:.0f}s", flush=True)