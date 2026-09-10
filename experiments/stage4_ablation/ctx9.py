import sys, numpy as np, heapq
sys.path.insert(0,"/tmp/opencode/autocompress/experiments/stage3_method")
from codec import rgb_to_ycocg_r, med
from huffman_exact import huffman_lens
from PIL import Image
import os, time
def res_ctx9(arr):
    H,W,_=arr.shape
    P=np.pad(arr,((1,0),(1,0),(0,0)),mode='edge').astype(np.int16)
    res=np.zeros_like(arr); ctx=np.zeros_like(arr,dtype=np.uint8)
    for i in range(H):
        for j in range(W):
            for ch in range(3):
                a=int(P[i+1,j,ch]); b=int(P[i,j+1,ch]); c=int(P[i,j,ch])
                e=min(8,(abs(a-c)+abs(b-c))//16)  # 0..8 energy class (JPEG-LS/CALIC style)
                ctx[i,j,ch]=e
                res[i,j,ch]=int(P[i+1,j+1,ch])-med(a,b,c)
    return res,ctx
d="/tmp/opencode/autocompress/experiments/real_photos"
for fn in ["kodim01.png","kodim02.png","kodim05.png","kodim07.png","kodim13.png","kodim19.png","kodim23.png"]:
    rgb=np.array(Image.open(os.path.join(d,fn)).convert("RGB")); H,W,_=rgb.shape
    t0=time.perf_counter()
    r,c=res_ctx9(rgb_to_ycocg_r(rgb))
    tb=0; H0=0; H1=0; N=H*W
    for ch in range(3):
        flat=r[:,:,ch].reshape(-1)
        vals,cn=np.unique(flat,return_counts=True)
        p=cn/cn.sum(); H0+=-(p*np.log2(p)).sum()*N
        for e in range(9):
            g=flat[c[:,:,ch].reshape(-1)==e]
            if len(g)==0: continue
            vals,cn=np.unique(g,return_counts=True)
            counts={int(v):int(n) for v,n in zip(vals,cn)}
            lens=huffman_lens(counts)
            tb+=sum(counts[s]*lens[s] for s in counts)+16+len(counts)*24
            p2=cn/cn.sum(); H1+=-(p2*np.log2(p2)).sum()*len(g)
    tot=(tb+48+7)//8; bpp=tot*8/(H*W*3)
    print(f"{fn} CTX9-HUFF bpp={bpp:.2f} H0={H0/(3*N):.2f} H1={H1/(3*N):.2f} eff_vs_H1={H1/tb:.3f} t={time.perf_counter()-t0:.1f}s",flush=True)