"""Probe µMoE: per-16x16-block RD expert selection {MED, TOP, PAETH, GRAD} with 2b/block header counted."""
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
def paeth(a,b,c):
    p=a+b-c; pa=abs(p-a); pb=abs(p-b); pc=abs(p-c)
    return np.where((pa<=pb)&(pa<=pc),a,np.where(pb<=pc,b,c))
d="/tmp/opencode/autocompress/experiments/real_photos"
for fn in sys.argv[1:] or ["kodim23.png"]:
    rgb=np.array(Image.open(os.path.join(d,fn)).convert("RGB")); H,W,_=rgb.shape
    yc=rgb_to_ycocg_r(rgb).astype(np.int32)
    tot=0; totmed=0; selcount=np.zeros(6,int)
    for k in range(3):
        P=np.pad(yc[:,:,k],((1,0),(1,0)),mode='edge')
        C=P[1:,1:]; a=P[1:,:-1]; b=P[:-1,1:]; c=P[:-1,:-1]
        m=np.where(c>=np.maximum(a,b),np.minimum(a,b),np.where(c<=np.minimum(a,b),np.maximum(a,b),a+b-c))
        t=b; p=paeth(a,b,c); g=((a+b)//2+(b-c)//4)
        E=[C-m,C-t,C-p,C-g]
        totmed+=hbits(E[0].reshape(-1))
        # per-block RD (bits estimated GLOBALLY per expert? No: proper = joint. Proxy: per-block L1 + global code approx.
        # Correct-ish cheap: choose per block the expert minimizing L1 (Huffman proxy), count 2b header.
        # Then final = global Huffman over chosen residuals + headers (measured exactly below).
        B=8; ch=np.zeros_like(C,dtype=np.uint8)
        for bi in range(0,H,B):
            for bj in range(0,W,B):
                sl=(slice(bi,min(bi+B,H)),slice(bj,min(bj+B,W)))
                l1=[np.abs(e[sl]).sum() for e in E]
                ch[sl]=int(np.argmin(l1))
        R=np.zeros_like(C)
        for x in range(6):
            R[ch==x]=[m,t,p,g,gap,dg][x][ch==x]
            selcount[x]+=(ch==x).sum()
        tot+=hbits((C-R).reshape(-1))
    nb=((H+15)//16)*((W+15)//16)*3
    print(fn,"MEDest=",totmed,"MOEest=",tot,"+hdrBits3b=",nb*3,"net gain%=",round((totmed-tot-nb*2)/totmed*100,2),"sel=",selcount.tolist(),flush=True)