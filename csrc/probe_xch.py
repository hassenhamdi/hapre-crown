"""Stage A: vectorized NL-XCh residual probe (no C changes). Y:6 feats; Co/Cg:6+Yres. Bit-gate per table."""
import numpy as np, heapq, sys, os, time
sys.path.insert(0,"/tmp/opencode/autocompress/experiments/stage3_method")
sys.path.insert(0,"/tmp/opencode/autocompress/csrc")
from codec import rgb_to_ycocg_r, med
from fit_nlls import fit_image
from PIL import Image
def hlens(counts):
    if len(counts)==1: return {s:1 for s in counts}
    H=[(c,s) for s,c in counts.items()]; heapq.heapify(H); par={}; nxt=1<<28; H2=H[:]
    while len(H2)>1:
        a,sa=heapq.heappop(H2); b,sb=heapq.heappop(H2); nn=nxt; nxt+=1
        par[sa]=(nn,0); par[sb]=(nn,1); heapq.heappush(H2,(a+b,nn))
    return {s:sum(1 for _ in iter(lambda n=n: par[n][0] if n in par else None,)) for s in counts}
def hbits(res):
    # exact Huffman data+table bits for residual array (values any int)
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
def med_res_vec(P):
    Pp=np.pad(P,((1,0),(1,0)),mode='edge').astype(np.int32)
    a=Pp[1:,:-1]; b=Pp[:-1,1:]; c=Pp[:-1,:-1]
    pred=np.where(c>=np.maximum(a,b),np.minimum(a,b),np.where(c<=np.minimum(a,b),np.maximum(a,b),a+b-c))
    return (P-pred)
d="/tmp/opencode/autocompress/experiments/real_photos"
for fn in sys.argv[1:] or ["kodim23.png","kodim07.png"]:
    rgb=np.array(Image.open(os.path.join(d,fn)).convert("RGB")); H,W,_=rgb.shape
    yc=rgb_to_ycocg_r(rgb)
    chs=[yc[:,:,k].astype(np.int32) for k in range(3)]
    # MED baseline bits per channel
    med_bits=[]
    mres=[]
    for P in chs:
        r=med_res_vec(P); mres.append(r)
        med_bits.append(hbits(r.reshape(-1)))
    # NL fit on subsampled originals (reuse fit_image), then vectorized NL residuals full-image
    Y0=np.zeros((3*H*W,),dtype=np.int16)
    for k in range(3): Y0[k*H*W:(k+1)*H*W]=chs[k].reshape(-1).astype(np.int16)
    coef,medflag=fit_image(Y0,H,W)
    # vectorized NL residuals with 6 feats
    nl_bits=0; detail=[]
    Yres=None
    for k in range(3):
        P=chs[k]
        Pp=np.pad(P,((1,0),(1,0)),mode='edge').astype(np.int32)
        a=Pp[1:,:-1]; b=Pp[:-1,1:]; c=Pp[:-1,:-1]
        e=np.minimum(8,(np.abs(a-c)+np.abs(b-c))//16)
        F=np.stack([np.ones_like(a),a,b,c,np.abs(a-b),(a+b-2*c)],-1)
        pred=np.zeros_like(P)
        for ctx in range(9):
            m=e==ctx
            if medflag[k*9+ctx]:
                mm=np.where(c>=np.maximum(a,b),np.minimum(a,b),np.where(c<=np.minimum(a,b),np.maximum(a,b),a+b-c))
                pred[m]=(F[m]@np.array([0,256,0,0,0,0])/256).astype(np.int32) if False else mm[m]
            else:
                w=coef[k*9+ctx].astype(np.float64)/256
                pred[m]=np.round(F[m]@w).astype(np.int32)
        r=(P-pred)
        if k==0: Yres=r
        bb=hbits(r.reshape(-1)); nl_bits+=bb
        detail.append(f"ch{k}: med={med_bits[k]} nl={bb}")
    print(f"{fn} MEDtot={sum(med_bits)} NL6tot={nl_bits} gain={(sum(med_bits)-nl_bits)/sum(med_bits)*100:.1f}% | "+" | ".join(detail), flush=True)