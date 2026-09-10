"""Probe FIR-12 (lag-1 + lag-2 causal neighbors) with IRLS-L1 per energy ctx + Huffman-bit gate."""
import numpy as np, heapq, sys, os
from PIL import Image
sys.path.insert(0,"/tmp/opencode/autocompress/experiments/stage3_method")
from codec import rgb_to_ycocg_r
def hbits(res):
    vals,cn=np.unique(res,return_counts=True)
    counts={int(v):int(c) for v,c in zip(vals,cn)}
    if len(counts)==1: return len(res)*1+16+24
    H=[(c,s) for s,c in counts.items()]; heapq.heapify(H); par={}; nxt=1<<28; H2=H[:]
    while len(H2)>1:
        a,sa=heapq.heappop(H2); b,sb=heapq.heappop(H2); nn=nxt; nxt+=1
        par[sa]=(nn,0); par[sb]=(nn,1); heapq.heappush(H2,(a+b,nn))
    tot=0
    for s,c in counts.items():
        dd=0;n=s
        while n in par: n=par[n][0];dd+=1
        tot+=c*dd
    return tot+16+len(counts)*24
d="/tmp/opencode/autocompress/experiments/real_photos"
for fn in sys.argv[1:] or ["kodim23.png"]:
    rgb=np.array(Image.open(os.path.join(d,fn)).convert("RGB")); H,W,_=rgb.shape
    yc=rgb_to_ycocg_r(rgb).astype(np.int32)
    tot_m=0; tot_f=0
    for k in range(3):
        P=np.pad(yc[:,:,k],((2,0),(2,0)),mode='edge')
        C=P[2:,2:]; a=P[2:,1:-1]; b=P[1:-1,2:]; c=P[1:-1,1:-1]
        a2=P[2:,:-2]; b2=P[:-2,2:]; c2=P[:-2,:-2]
        d2tr=P[:-2,3:] if W>1 else b2  # top-right lag (approx, edge ok for probe)
        m=np.where(c>=np.maximum(a,b),np.minimum(a,b),np.where(c<=np.minimum(a,b),np.maximum(a,b),a+b-c))
        rm=(C-m); tot_m+=hbits(rm.reshape(-1))
        e=np.minimum(8,(np.abs(a-c)+np.abs(b-c))//16)
        F=np.stack([np.ones_like(a),a,b,c,a2,b2,c2,np.abs(a-b),(a+b-2*c)],-1).astype(np.float64)
        yv=C.astype(np.float64); pred=np.zeros_like(C)
        for ctx in range(9):
            mm=e==ctx
            if mm.sum()<100:
                pred[mm]=m[mm]; continue
            A=F[mm]; y2=yv[mm]
            w,_,_,_=np.linalg.lstsq(A,y2,rcond=None)
            for _ in range(3):
                rr=np.abs(A@w-y2); rw=1.0/np.maximum(1.0,rr); s=np.sqrt(rw)
                w,_,_,_=np.linalg.lstsq(A*s[:,None],y2*s,rcond=None)
            cand=np.round(A@w)
            base=m[mm].astype(np.float64)
            # bit-gate on full-ctx residuals
            if hbits(np.round(cand).astype(np.int32)) < hbits((y2-base).astype(np.int32)):
                pred[mm]=cand.astype(np.int32)
            else:
                pred[mm]=base.astype(np.int32)
        tot_f+=hbits((C-pred).reshape(-1))
    print(fn, "MEDbits=",tot_med:=tot_m, "FIR12bits=",tot_f, "gain%=",round((tot_m-tot_f)/tot_m*100,2), flush=True)