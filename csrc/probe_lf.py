"""Probe: does a 4x box-average LF plane (decoder-available side info) reduce MED-residual Huffman bits?"""
import numpy as np, heapq, sys, os
sys.path.insert(0,"/tmp/opencode/autocompress/experiments/stage3_method")
from codec import rgb_to_ycocg_r
from PIL import Image
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
        d=0;n=s
        while n in par: n=par[n][0];d+=1
        tot+=c*d
    return tot+16+len(counts)*24
d="/tmp/opencode/autocompress/experiments/real_photos"
for fn in sys.argv[1:] or ["kodim23.png","kodim07.png","kodim01.png"]:
    rgb=np.array(Image.open(os.path.join(d,fn)).convert("RGB")); H,W,_=rgb.shape
    yc=rgb_to_ycocg_r(rgb).astype(np.int32)
    tot_med=0; tot_cond=0
    for k in range(3):
        P=yc[:,:,k]
        Pp=np.pad(P,((1,0),(1,0)),mode='edge')
        a=Pp[1:,:-1]; b=Pp[:-1,1:]; c=Pp[:-1,:-1]
        pred=np.where(c>=np.maximum(a,b),np.minimum(a,b),np.where(c<=np.minimum(a,b),np.maximum(a,b),a+b-c))
        r=(P-pred)
        tot_med+=hbits(r.reshape(-1))
        # LF: 4x box average, nearest-upsampled (decoder can reproduce from LF stream)
        h4=(H+3)//4; w4=(W+3)//4
        pad=np.pad(P,((0,-H%4 if H%4 else 0),(0,-W%4 if W%4 else 0)),mode='edge')
        lf=pad.reshape(h4,4,w4,4).mean(axis=(1,3))
        up=np.repeat(np.repeat(lf,4,axis=0),4,axis=1)[:H,:W].astype(np.int32)
        # condition: split residuals by LF-pred error? Proper test: H(r | up) via per-LFbin tables is complex;
        # cheap proxy: residual of (P - up) i.e. how well LF alone predicts, then H of MED-res given LF-error bins
        # Simplest meaningful: bits = H(LF-quantized coarsely) + H(r | qbin). Use 16 LF-error bins.
        le=(P-up)
        q=np.clip(le//16+8,0,15)
        cond=0; tot_sym=0
        for bin in range(16):
            g=r[q==bin]
            if len(g)==0: continue
            cond+=hbits(g)
            tot_sym+=len(g)
        # LF side cost: order-0 Huffman of lf values
        side=hbits(lf.reshape(-1).astype(np.int32))
        tot_cond+=cond+side
    print(fn, "MEDest=",tot_med, "LFcond+side=",tot_cond, "delta%=",round((tot_med-tot_cond)/tot_med*100,2), flush=True)