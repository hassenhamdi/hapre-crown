"""Vectorized LeGall 5/3 integer wavelet + zlib (C backends) = ultra-fast lossless/near-lossless."""
import numpy as np, zlib, time
from PIL import Image
import os
def fwd53_1d(x):
    x=x.astype(np.int32); n=len(x)
    ev=x[0::2].copy(); od=x[1::2].copy()
    # high: od[i] - (ev[i]+ev[i+1])//2 ; last odd uses symmetric ev[-1]
    ev_ext=np.concatenate([ev,ev[-1:]])
    H=od-(ev_ext[:-1]+ev_ext[1:])//2
    # low: ev[i] + (H[i-1]+H[i]+2)//4
    Hp=np.concatenate([H[:1],H])
    L=ev+(Hp[:-1]+Hp[1:]+2)//4
    return L,H
def inv53_1d(L,H):
    Hp=np.concatenate([H[:1],H])
    ev=L-(Hp[:-1]+Hp[1:]+2)//4
    ev_ext=np.concatenate([ev,ev[-1:]])
    od=H+(ev_ext[:-1]+ev_ext[1:])//2
    n=len(L)+len(H); x=np.empty(n,dtype=np.int32)
    x[0::2]=ev; x[1::2]=od
    return x
def fwd2d(a):
    Lr=np.zeros((a.shape[0],(a.shape[1]+1)//2),np.int32); Hr=np.zeros((a.shape[0],a.shape[1]//2),np.int32)
    for i in range(a.shape[0]): Lr[i],Hr[i]=fwd53_1d(a[i])
    Lc=np.zeros(((Lr.shape[0]+1)//2,Lr.shape[1]),np.int32); Hc=np.zeros((Lr.shape[0]//2,Lr.shape[1]),np.int32)
    for j in range(Lr.shape[1]):
        Lc[:,j],Hc[:,j]=fwd53_1d(Lr[:,j])
    Lc2=np.zeros(((Hr.shape[0]+1)//2,Hr.shape[1]),np.int32); Hc2=np.zeros((Hr.shape[0]//2,Hr.shape[1]),np.int32)
    for j in range(Hr.shape[1]):
        Lc2[:,j],Hc2[:,j]=fwd53_1d(Hr[:,j])
    return Lc,Hc,Lc2,Hc2
def rgb_to_ycocg(arr):
    R=arr[:,:,0].astype(np.int32); G=arr[:,:,1].astype(np.int32); B=arr[:,:,2].astype(np.int32)
    Co=R-B; t=B+(Co>>1); Cg=G-t; Y=t+(Cg>>1)
    return Y,Co,Cg
def enc_channel(Y, qstep=1):
    LL,LH,HL,HH=fwd2d(Y)
    parts=[]
    for sb,q in [(LL,1),(LH,qstep),(HL,qstep),(HH,qstep)]:
        if q>1: sb=np.round(sb/q).astype(np.int32)
        parts.append(sb.astype(np.int16).tobytes())
    return zlib.compress(b''.join(parts),9), (LL.shape,LL.nbytes)
d="/tmp/opencode/autocompress/experiments/real_photos"
for fn in ["kodim01.png","kodim07.png","kodim23.png"]:
    rgb=np.array(Image.open(os.path.join(d,fn)).convert("RGB")); H,W,_=rgb.shape
    for q in [1,3,5]:
        t0=time.perf_counter()
        Y,Co,Cg=rgb_to_ycocg(rgb)
        bY,_=enc_channel(Y,q); bCo,_=enc_channel(Co,q); bCg,_=enc_channel(Cg,q)
        t=time.perf_counter()-t0
        tot=len(bY)+len(bCo)+len(bCg)+24
        print(f"{fn} q={q} bytes={tot} bpp={tot*8/(H*W*3):.2f} enc={t*1000:.0f}ms",flush=True)