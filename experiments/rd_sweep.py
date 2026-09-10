"""RD sweep: prototype near-lossless family (exact Rice bytes + recon metrics) vs JXL lossy points."""
import sys, os, math, subprocess, time
import numpy as np
from PIL import Image
sys.path.insert(0, "/tmp/opencode/autocompress/experiments/stage3_method")
sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
from codec import rgb_to_ycocg_r, ycocg_r_to_rgb, med, paeth, gap
from codec_v2 import elias_gamma_bits
from quality import psnr, mssim_rgb, maxerr

BLOCK = 16


def run_tau(rgb, tau):
    H, W, _ = rgb.shape
    yc = rgb_to_ycocg_r(rgb)
    P = np.pad(yc, ((2, 0), (2, 0), (0, 0)), mode="edge").astype(np.int16)
    R = P.copy()
    res = np.zeros_like(yc)
    for i in range(H):
        pi = i + 2
        for j in range(W):
            pj = j + 2
            for ch in range(3):
                a = int(R[pi, pj - 1, ch]); b = int(R[pi - 1, pj, ch]); c = int(R[pi - 1, pj - 1, ch])
                Wv = int(R[pi, pj - 2, ch]); NE = int(R[pi - 1, pj + 1, ch]) if pj + 1 < W + 2 else b
                NNE = int(R[pi - 2, pj, ch])
                m = med(a, b, c); g = gap(Wv, Wv, b, c, NE, NNE); p = paeth(a, b, c)
                gh = abs(a - Wv) + abs(b - c) + abs(b - NE); gv = abs(a - c) + abs(b - NNE) + abs(NE - NNE)
                pred = m if gv - gh > 40 else (g if gv - gh < -40 else (m + g) // 2)
                x = int(P[pi, pj, ch]); r = x - pred
                if tau > 0:
                    rq = int(round(r / (2 * tau + 1)))
                    R[pi, pj, ch] = np.int16(pred + rq * (2 * tau + 1)); res[i, j, ch] = np.int16(rq)
                else:
                    R[pi, pj, ch] = np.int16(x); res[i, j, ch] = np.int16(r)
    rec = np.zeros((H, W, 3), dtype=np.int16)
    for i in range(H):
        for j in range(W):
            rec[i, j] = R[i + 2, j + 2]
    rec_rgb = ycocg_r_to_rgb(rec)
    # exact Rice+run bytes over quantized indices
    tb = 0
    for ch in range(3):
        flat = res[:, :, ch].reshape(-1)
        nz = flat[flat != 0]
        m = float(np.mean(np.abs(nz.astype(float)))) + 1e-9 if len(nz) else 1.0
        k = int(max(0, min(8, math.floor(math.log2(m)))))
        i = 0; N = len(flat); b = 4
        while i < N:
            if flat[i] == 0:
                j = i
                while j < N and flat[j] == 0:
                    j += 1
                b += 1 + elias_gamma_bits(j - i); i = j
            else:
                v = int(flat[i]); u = 2 * v if v >= 0 else -2 * v - 1
                b += 1 + (u >> k) + 1 + k; i += 1
        tb += b
    nb = (H // BLOCK + 1) * (W // BLOCK + 1); hdr = nb * 2 + 48 * 8 + 48
    total = (tb + hdr + 7) // 8
    return total, rec_rgb


def jxl_point(im_path, distance):
    jxl = "/tmp/_q_rd.jxl"; png = "/tmp/_q_rd.png"
    subprocess.run(["cjxl", im_path, jxl, "-d", str(distance), "-e", "9", "--quiet"], capture_output=True)
    subprocess.run(["djxl", jxl, png, "--quiet"], capture_output=True)
    return np.array(Image.open(png).convert("RGB")), os.path.getsize(jxl)


if __name__ == "__main__":
    d = "/tmp/opencode/autocompress/experiments/real_photos"
    print("img,tau,bytes,bpp,maxerr,psnr,ssim", flush=True)
    for fn in ["kodim01.png", "kodim07.png", "kodim23.png"]:
        rgb = np.array(Image.open(os.path.join(d, fn)).convert("RGB"))
        H, W, _ = rgb.shape
        for tau in [0, 1, 2]:
            t0 = time.perf_counter()
            by, rec = run_tau(rgb, tau)
            print(f"{fn},tau{tau},{by},{by*8/(H*W*3):.4f},{maxerr(rgb, rec)},{psnr(rgb, rec):.2f},{mssim_rgb(rgb, rec):.5f}  # {time.perf_counter()-t0:.0f}s", flush=True)
        for dist in ["0.5", "1.0", "2.0"]:
            dec, by = jxl_point(os.path.join(d, fn), dist)
            print(f"{fn},jxl-d{dist},{by},{by*8/(H*W*3):.4f},{maxerr(rgb, dec)},{psnr(rgb, dec):.2f},{mssim_rgb(rgb, dec):.5f}", flush=True)
