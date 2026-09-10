"""ULTRA-FAST path: row-vectorized predictors (numpy C-speed) + zlib (C-speed). No Python pixel loops."""
import numpy as np, zlib, time
from PIL import Image
import os
def saturate8(a): return a
d="/tmp/opencode/autocompress/experiments/real_photos"
def ycocg(rgb):
    R=rgb[:,:,0].astype(np.int16); G=rgb[:,:,1].astype(np.int16); B=rgb[:,:,2].astype(np.int16)
    Co=R-B; t=B+(Co>>1); Cg=G-t; Y=t+(Cg>>1)
    return {"Y":Y,"Co":Co,"Cg":Cg}
def predict_top(P):  # P padded (H+1,W+1); row-vectorized: pred[i,j]=P[i,j+1](top)
    return P[:-1,1:]
def residuals_top(arr):
    P=np.pad(arr,((1,0),(1,0)),mode='edge').astype(np.int16)
    return (arr-predict_top(P)).astype(np.int16)
def residuals_avg(arr):
    P=np.pad(arr,((1,0),(1,0)),mode='edge').astype(np.int16)
    pred=((P[:-1,1:].astype(np.int32)+P[:-1,:-1].astype(np.int32))//2).astype(np.int16)
    return (arr-pred).astype(np.int16)
def residuals_gradcorr(arr):
    P=np.pad(arr,((1,0),(1,1)),mode='edge').astype(np.int16)
    top=P[:-1,1:-1].astype(np.int32); tl=P[:-1,:-2].astype(np.int32); tr=P[:-1,2:].astype(np.int32)
    pred=(top+((tl-tr)//2)).astype(np.int16)
    return (arr-pred).astype(np.int16)
def zz_bytes(r):
    u=np.where(r>=0,2*r,-2*r-1).astype(np.int32)
    # store as 16-bit LE bytes (two byte planes: low, high) - high plane mostly zero -> zlib loves it
    return u.astype(np.uint16).tobytes()
for fn in ["kodim01.png","kodim07.png","kodim23.png"]:
    rgb=np.array(Image.open(os.path.join(d,fn)).convert("RGB")); H,W,_=rgb.shape
    Y=ycocg(rgb)
    for name,f in [("raw",None),("top",residuals_top),("avg",residuals_avg),("gradcorr",residuals_gradcorr)]:
        t0=time.perf_counter()
        if f is None:
            pay=b''.join(v.astype(np.int16).tobytes() for v in Y.values())
        else:
            pay=b''.join(zz_bytes(f(v)) for v in Y.values())
        n=len(zlib.compress(pay,9))+24
        t=time.perf_counter()-t0
        print(f"{fn} {name:9s} bpp={n*8/(H*W*3):.2f} enc={t*1000:.0f}ms ({H*W/1e6/t:.1f}MP/s)",flush=True)