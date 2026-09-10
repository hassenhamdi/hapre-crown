"""Quality metrics: PSNR / SSIM / maxerr (numpy + scipy, no skimage)."""
import numpy as np
from scipy.ndimage import gaussian_filter


def psnr(a, b, peak=255.0):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10(peak * peak / mse)


def _ssim_channel(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    K1, K2, L = 0.01, 0.03, 255.0
    C1, C2 = (K1 * L) ** 2, (K2 * L) ** 2
    mu_x = gaussian_filter(x, 1.5)
    mu_y = gaussian_filter(y, 1.5)
    s_xx = gaussian_filter(x * x, 1.5) - mu_x * mu_x
    s_yy = gaussian_filter(y * y, 1.5) - mu_y * mu_y
    s_xy = gaussian_filter(x * y, 1.5) - mu_x * mu_y
    num = (2 * mu_x * mu_y + C1) * (2 * s_xy + C2)
    den = (mu_x * mu_x + mu_y * mu_y + C1) * (s_xx + s_yy + C2)
    return float(np.mean(num / den))


def mssim_rgb(a, b):
    return float(np.mean([_ssim_channel(a[:, :, k], b[:, :, k]) for k in range(3)]))


def maxerr(a, b):
    return int(np.abs(np.asarray(a, dtype=np.int32) - np.asarray(b, dtype=np.int32)).max())


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
    assert psnr(x, x) == float("inf")
    assert abs(mssim_rgb(x, x) - 1.0) < 1e-9
    assert maxerr(x, x) == 0
    y = np.clip(x.astype(int) + 1, 0, 255).astype(np.uint8)
    print("selftest: psnr(+1)=%.2f ssim(+1)=%.5f maxerr(+1)=%d" % (psnr(x, y), mssim_rgb(x, y), maxerr(x, y)))
