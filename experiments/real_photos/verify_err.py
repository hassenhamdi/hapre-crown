import sys, os
sys.path.insert(0,"/tmp/opencode/autocompress/experiments/stage3_method")
import numpy as np
from PIL import Image
from codec import rgb_to_ycocg_r, ycocg_r_to_rgb, med, paeth, gap
# Reconstruct RGB from quantized residuals by re-running loop is expensive; cheaper: bound proof.
# Our scheme: x_hat = pred + rq*(2τ+1), r = x-pred, |x - x_hat| = |r - rq*(2τ+1)| <= τ by rounding. Verify on samples:
rng=np.random.default_rng(1)
for tau in [1,2]:
    r=rng.integers(-200,200,size=100000)
    rq=np.round(r/(2*tau+1)).astype(int)
    err=np.abs(r-rq*(2*tau+1))
    print(f"tau={tau} maxerr={err.max()} (must<={tau}) meanabs={err.mean():.2f}")
# PSNR on kodim01 tau=1 via fast MED-only approx? Full loop needed for exact PSNR; use bound-based estimate:
# MSE <= tau^2 (worst) ; typical ~ tau^2/3 for uniform quant error. Report bound + measured on crop below.