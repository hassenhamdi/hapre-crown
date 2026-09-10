"""GAP + JPEG-LS-style per-context adaptive bias correction + Golomb exact bits (single-variable change vs MED)."""
import sys, numpy as np, math
sys.path.insert(0,"/tmp/opencode/autocompress/experiments/stage3_method")
from codec import rgb_to_ycocg_r, gap
from PIL import Image
import os, time
def golomb_bits(v, k):
    u=2*v if v>=0 else -2*v-1
    return int((u>>k)+1+k)
def run_gap_bias(arr):
    H,W,_=arr.shape
    P=np.pad(arr,((2,0),(2,0),(0,0)),mode='edge').astype(np.int16)
    B=np.zeros((9,3),dtype=float); C=np.zeros((9,3)); Nn=np.zeros((9,3))  # JPEG-LS bias vars
    tot=0; res=np.zeros_like(arr)
    for i in range(H):
        for j in range(W):
            for ch in range(3):
                Wv=int(P[i+2,j,ch]); Nv=int(P[i+1,j+1,ch]); NW=int(P[i+1,j,ch]); NE=int(P[i+1,j+2,ch]) if j+2<W+1 else Nv; NNE=int(P[i,j+1,ch])
                WW=int(P[i+2,max(0,j-1),ch])
                g=gap(Wv,WW,Nv,NW,NE,NNE)
                e=min(8,(abs(Wv-NW)+abs(Nv-NW))//16)
                pred=int(g)
                if Nn[e,ch]>0 and B[e,ch]<=-Nn[e,ch]: pred-=1; B[e,ch]+=Nn[e,ch]
                elif Nn[e,ch]>0 and B[e,ch]>0: pred+=1; B[e,ch]-=Nn[e,ch]
                x=int(P[i+2,j+1,ch]); r=x-pred
                res[i,j,ch]=r
                # update (LIMIT=4, RESET=64 simplified: C clip)
                B[e,ch]+=r; Nn[e,ch]+=1
                if Nn[e,ch]>=64: B[e,ch]//=2; Nn[e,ch]//=2
    # exact Golomb per ctx with k from mean|corrected residual|
    tb=0
    for ch in range(3):
        for e in range(9):
            pass
    # per-channel k (simpler, bias already per-ctx): compute exact via loop over stored res with ctx recompute? approximate with global k per ch
    for ch in range(3):
        flat=res[:,:,ch].reshape(-1)
        nz=flat[flat!=0]
        m=float(np.mean(np.abs(nz.astype(float))))+1e-9 if len(nz) else 1
        k=int(max(0,min(8,math.floor(math.log2(m)))))
        # zero-run gamma + golomb nonzero (same token format as Rice)
        from codec_v2 import elias_gamma_bits
        i=0;N=len(flat)
        tb+=4
        while i<N:
            if flat[i]==0:
                j=i
                while j<N and flat[j]==0: j+=1
                from codec_v2 import elias_gamma_bits as eg
                tb+=1+eg(j-i); i=j
            else: tb+=1+golomb_bits(int(flat[i]),k); i+=1
    return res,tb
d="/tmp/opencode/autocompress/experiments/real_photos"
for fn in ["kodim01.png","kodim02.png","kodim05.png","kodim07.png","kodim13.png","kodim19.png","kodim23.png"]:
    rgb=np.array(Image.open(os.path.join(d,fn)).convert("RGB")); H,W,_=rgb.shape
    t0=time.perf_counter(); _,tb=run_gap_bias(rgb_to_ycocg_r(rgb))
    tot=(tb+48+7)//8; print(f"{fn} GAPbias-Golomb bpp={tot*8/(H*W*3):.2f} t={time.perf_counter()-t0:.1f}s",flush=True)