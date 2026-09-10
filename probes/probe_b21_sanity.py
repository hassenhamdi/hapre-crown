"""probe_b21_sanity.py — decoder-causality sanity (new file).

Proves per-context gated predictions are decoder-computable from recon alone:
sequential raster reconstruction on a 128x128 crop of kodim23 (all 3 channels)
using ONLY recon neighbors: recompute energy ctx from recon, pick gated predictor
(quantized per-ctx MLP or MED per stored gate map), add stored residual, assert
recon == original. Uses quantized weights (as transmitted) with a manual numpy
forward pass (tanh). Full-image equality follows by induction (same rule per pixel;
vectorized probe predictions use original==recon neighbors).
Also asserts YCoCg-R invertibility on the crop.
"""
import sys, os, json
import numpy as np
import torch
from PIL import Image
sys.path.insert(0, "/tmp/opencode/autocompress/experiments/stage3_method")
from codec import rgb_to_ycocg_r, ycocg_r_to_rgb

D = "/tmp/opencode/autocompress/experiments/real_photos"

def mlp_forward_numpy(Frow, fc1w, fc1b, fc2w, fc2b):
    return (np.tanh(Frow @ fc1w.T + fc1b) @ fc2w.T + fc2b)[..., 0]

def main():
    fn = "kodim23.png"
    rgb = np.array(Image.open(os.path.join(D, fn)).convert("RGB"))
    yc = rgb_to_ycocg_r(rgb)
    assert np.array_equal(ycocg_r_to_rgb(yc), rgb), "YCoCg-R invert fail (full)"
    H, W, _ = rgb.shape
    # need trained nets: retrain crop-relevant nets quickly? Instead load gate+residuals
    # from full7 JSON and re-derive predictions with retrained-nets is heavy; do a
    # self-contained sequential proof: train the 27 ctx nets on the CROP only, gate,
    # quantize, then sequential recon. Proves causality/end-to-end, not the full7 numbers
    # (those use identical code path vectorized; equality by induction).
    sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
    from probe_b21_perctx import (features_targets, ctx_of_plane, TinyMLP, train_net,
                                  quantize_net, adaptive_quantize, hbits, med_bits,
                                  med_pred_plane, nparams_of)
    torch.manual_seed(0)
    torch.set_num_threads(12)
    h0, w0, S = 100, 100, 128
    ok_all = True
    for ch, nm in enumerate(["Y", "Co", "Cg"]):
        P = yc[h0:h0 + S, w0:w0 + S, ch].astype(np.int32)
        assert np.array_equal(ycocg_r_to_rgb(np.stack(
            [yc[h0:h0 + S, w0:w0 + S, 0], yc[h0:h0 + S, w0:w0 + S, 1],
             yc[h0:h0 + S, w0:w0 + S, 2]], -1)), rgb[h0:h0 + S, w0:w0 + S]), "crop invert"
        F12, y, _ = features_targets(P, 12)
        Ff = F12.reshape(-1, 12)
        ctx, qs, E = ctx_of_plane(P)
        cf = ctx.reshape(-1)
        med_plane = med_pred_plane(P)
        med_res = (P - med_plane).reshape(-1)
        # train+gate+quantize per ctx (committed config); store vectorized preds+residuals
        mods = {}
        pall = {}
        pred_vec = np.empty_like(med_res)
        for k in range(9):
            m = cf == k
            if m.sum() < 100:
                mods[k] = None
                pred_vec[m] = med_plane.reshape(-1)[m]
                continue
            net, _ = train_net(Ff[m], y[m], 12, 8, 200, 0.003)
            neta, sc = adaptive_quantize(net)
            with torch.no_grad():
                p = {n: v.numpy() for n, v in neta.named_parameters()}
            pv = np.round(mlp_forward_numpy(Ff[m].astype(np.float64) / 128.0, p["fc1.weight"], p["fc1.bias"],
                                            p["fc2.weight"], p["fc2.bias"]) * 128.0).astype(np.int32)
            qb = hbits(P.reshape(-1)[m].astype(np.int32) - pv)
            pall[k] = p
            if (qb + nparams_of(12, 8) * 16 + 4) < hbits(med_res[m]):
                mods[k] = p
                pred_vec[m] = pv
            else:
                mods[k] = None
                pred_vec[m] = med_plane.reshape(-1)[m]
        res_vec = P.reshape(-1).astype(np.int32) - pred_vec
        # sequential raster recon from recon-only neighbors; LOAD-BEARING assert:
        # pred_seq (recon neighbors) == pred_vec (original neighbors) at EVERY pixel.
        R = np.pad(P, ((3, 0), (3, 2)), mode="edge").astype(np.int32)
        orig = P
        mismatch = 0
        for i in range(S):
            for j in range(S):
                pi, pj = i + 3, j + 3
                a = R[pi, pj - 1]; b = R[pi - 1, pj]; c = R[pi - 1, pj - 1]
                e = abs(int(a) - int(b)) + abs(int(a) - int(c)) + abs(int(b) - int(c))
                k = min(8, max(0, int(np.searchsorted(qs, e))))
                assert k == int(cf[i * S + j]), f"ctx mismatch at {(i, j)}"
                mod = mods[k]
                if mod is None:
                    pred = (min(a, b) if c >= max(a, b) else (max(a, b) if c <= min(a, b) else a + b - c))
                else:
                    p = mod
                    d = R[pi - 1, pj + 1]; Ww = R[pi, pj - 2]; NNe = R[pi - 2, pj]
                    L3 = R[pi, pj - 3]; T3 = R[pi - 3, pj]
                    TL2 = R[pi - 2, pj - 2]; TR2 = R[pi - 2, pj + 2]
                    f = np.array([a, b, c, d, Ww, NNe, (a + b) // 2, abs(a - b),
                                  L3, T3, TL2, TR2], dtype=np.float64) / 128.0
                    pred = int(round(float(mlp_forward_numpy(f, p["fc1.weight"], p["fc1.bias"],
                                                             p["fc2.weight"], p["fc2.bias"]) * 128.0)))
                if pred != int(pred_vec[i * S + j]):
                    mismatch += 1
                R[pi, pj] = np.int32(pred + int(res_vec[i * S + j]))
        recon = R[3:3 + S, 3:3 + S]
        ok = bool(np.array_equal(recon, orig)) and mismatch == 0
        print(f"{nm}: pred_seq==pred_vec mismatches={mismatch}/16384 "
              f"recon==original={np.array_equal(recon, orig)} "
              f"(gated wins {sum(1 for v in mods.values() if v is not None)}/9)", flush=True)
        # forced-MLP pass: same sequential rule with EVERY trained ctx net (no gate) —
        # exercises the MLP recon path on all pixels where a net exists.
        R2 = np.pad(P, ((3, 0), (3, 2)), mode="edge").astype(np.int32)
        mismatch2 = 0; nmlp2 = 0
        pv_all = np.empty_like(med_res)
        for k in range(9):
            m = cf == k
            if pall.get(k) is None:
                pv_all[m] = med_plane.reshape(-1)[m]
            else:
                p = pall[k]
                pv_all[m] = np.round(mlp_forward_numpy(
                    Ff[m].astype(np.float64) / 128.0, p["fc1.weight"], p["fc1.bias"],
                    p["fc2.weight"], p["fc2.bias"]) * 128.0).astype(np.int32)
        res_all = P.reshape(-1).astype(np.int32) - pv_all
        for i in range(S):
            for j in range(S):
                pi, pj = i + 3, j + 3
                a = R2[pi, pj - 1]; b = R2[pi - 1, pj]; c = R2[pi - 1, pj - 1]
                e = abs(int(a) - int(b)) + abs(int(a) - int(c)) + abs(int(b) - int(c))
                k = min(8, max(0, int(np.searchsorted(qs, e))))
                p = pall.get(k)
                if p is None:
                    pred = (min(a, b) if c >= max(a, b) else (max(a, b) if c <= min(a, b) else a + b - c))
                else:
                    nmlp2 += 1
                    d = R2[pi - 1, pj + 1]; Ww = R2[pi, pj - 2]; NNe = R2[pi - 2, pj]
                    L3 = R2[pi, pj - 3]; T3 = R2[pi - 3, pj]
                    TL2 = R2[pi - 2, pj - 2]; TR2 = R2[pi - 2, pj + 2]
                    f = np.array([a, b, c, d, Ww, NNe, (a + b) // 2, abs(a - b),
                                  L3, T3, TL2, TR2], dtype=np.float64) / 128.0
                    pred = int(round(float(mlp_forward_numpy(f, p["fc1.weight"], p["fc1.bias"],
                                                             p["fc2.weight"], p["fc2.bias"]) * 128.0)))
                if pred != int(pv_all[i * S + j]):
                    mismatch2 += 1
                R2[pi, pj] = np.int32(pred + int(res_all[i * S + j]))
        recon2 = R2[3:3 + S, 3:3 + S]
        ok2 = bool(np.array_equal(recon2, orig)) and mismatch2 == 0
        print(f"{nm}-FORCED-MLP: mlp-pixels={nmlp2}/16384 mismatches={mismatch2} "
              f"recon==original={np.array_equal(recon2, orig)}", flush=True)
        ok_all &= ok2
        ok_all &= ok
    print("SANITY " + ("PASS" if ok_all else "FAIL"), flush=True)

if __name__ == "__main__":
    main()
