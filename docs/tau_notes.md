# Tau / maxerr / PSNR definitions — probes+experiments grep (factual extraction only)

Probe listing hits for `tau` (case-insensitive): `experiments/rd_sweep.py`, `experiments/real_photos/eval_tau1_full.py`, `experiments/real_photos/eval_nearlossless.py`, `experiments/real_photos/verify_err.py`, `experiments/real_photos/eval_kodak.py`, `experiments/stage3_method/codec{,_v2}.py`, `experiments/stage4_ablation/exact_tau1.py` (plus PNG binaries). No `probes/probe_*_RESULTS.md` tau ledger; near-lossless lives in `experiments/` + paper §4.7.

## tau (quantizer step) — `experiments/rd_sweep.py:32-34`, `experiments/stage3_method/codec.py:108-109`
> `rd_sweep.py:32:                    rq = int(round(r / (2 * tau + 1)))`
> `rd_sweep.py:33-34:                    R[pi,pj,ch] = np.int16(pred + rq * (2*tau+1)); res[i,j,ch] = np.int16(rq)`
> `codec.py:109:                    rq=int(round(r/(2*tau+1))); xh=pred+rq*(2*tau+1)`

`tau=0` lossless (`R=x`, `res=r`, `rd_sweep.py:35-36`, `codec.py:137`); `tau>0` uniform scalar quantization of the prediction residual with step `2·tau+1`, reconstruction `pred+rq·step` (decoder mirrors from recon only). Exact Rice+run bytes over quantized indices `res` (`rd_sweep.py:42-61`, `exact_tau1.py:exact_bits`). Sweep points `tau∈{0,1,2}` (`rd_sweep.py:78`); JXL-lossy context `cjxl -d {0.5,1.0,2.0} -e 9` (`rd_sweep.py:67,82-84`).

## maxerr (RGB) — `experiments/quality.py:34-35`
> `quality.py:34: def maxerr(a, b):`
> `quality.py:35:     return int(np.abs(np.asarray(a, dtype=np.int32) - np.asarray(b, dtype=np.int32)).max())`

Maximum absolute error over all H·W·3 RGB uint8 values; `maxerr(x,x)==0` (selftest `quality.py:43`). Paper §4.7 reports `tau1 maxerr 3, tau2 maxerr 4` (prototype values as banked; step formula would give 1/2 on residuals — reported maxerr is end-to-end RGB after YCoCg-R + MED loop, not the step itself).

## PSNR — `experiments/quality.py:6-12`
> `quality.py:6:  def psnr(a, b, peak=255.0):`
> `quality.py:9:      mse = float(np.mean((a - b) ** 2))`
> `quality.py:10-11:     if mse == 0: return float("inf")`
> `quality.py:12:     return 10.0 * np.log10(peak * peak / mse)`

RGB PSNR (mean over all pixels/channels, peak 255); lossless `tau0 → inf`, prototype `tau1 ~47.3dB, tau2 ~43.5dB`, JXL-d0.5 ~42dB / d1.0 ~38-40dB (paper §4.7 table, `[VERIFY]`). SSIM companion `mssim_rgb` = mean of per-channel Gaussian (`sigma=1.5, K1=0.01,K2=0.03`) SSIM (`quality.py:15-31`); `tau0 → 1.0`.

No new math claims; near-lossless is prototype-family (Rice), not the CROWN backend (paper App.B).
