"""probe_b4_a: anchors + fast vectorized tests (numpy+PIL only).
UNIT: bpp = total_bits/(H*W*3). Tables: sum(count*len)+16+A*24 per stream, +64b header.
Border: champion 0/left/top rule (NOT edge pad). CTX9: e=(|a-c|+|b-c|)>>4 cap 8.
"""
import heapq, os, time, math
import numpy as np
from PIL import Image

IMAGES = [
    "/tmp/opencode/autocompress/experiments/real_photos/kodim01.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim02.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim05.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim07.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim13.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim19.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim23.png",
]
CHAMP = {"kodim01.png":3.56,"kodim02.png":3.23,"kodim05.png":3.84,"kodim07.png":3.01,
         "kodim13.png":4.17,"kodim19.png":3.41,"kodim23.png":3.03}
HEADER = 64

def ycocg_fwd(arr):
    R=arr[:,:,0].astype(np.int32); G=arr[:,:,1].astype(np.int32); B=arr[:,:,2].astype(np.int32)
    Co=R-B; t=B+(Co//2); Cg=G-t; Y=t+(Cg//2)
    return Y,Co,Cg

def med_pred_ctx9(ch):
    """Champion-convention MED + energy ctx. ch int32 HxW. Returns pred, resid, ctx."""
    H,W=ch.shape
    P=np.zeros((H+1,W+1),dtype=np.int32); P[1:,1:]=ch
    # first row: P[0,1:]=0? champion: i==0 -> a=P[j-1](recon, 0 at j==0?), b=c=a.
    # With zero border: row0 above = 0, col0 left = 0 at (0,0); for i==0,j>0: a=recon[j-1], b=c=a.
    # Implement via explicit padded recon where top row and left col are 0 beyond image,
    # but causal recon == original (lossless). So:
    # Padu top row zeros, left col zeros.
    a=np.zeros_like(ch); b=np.zeros_like(ch); c=np.zeros_like(ch)
    # vectorized with shifts of ch, handling borders per rule:
    # a = left recon (0 if j==0 and i==0? no: j==0,i>0 -> a=c=b=top)
    # General champion rule:
    # (0,0): a=b=c=0
    # i==0,j>0: a=ch[i,j-1], b=c=a
    # j==0,i>0: b=ch[i-1,j], a=c=b
    # else: a=ch[i,j-1], b=ch[i-1,j], c=ch[i-1,j-1]
    a[:,1:]=ch[:,:-1]; a[:,0]=np.where(np.arange(H)==0,0,ch[:,0]*0+np.concatenate([[0],np.zeros(H-1,dtype=np.int32)])*0)  # placeholder
    # do it explicitly but vectorized:
    a=np.empty_like(ch); b=np.empty_like(ch); c=np.empty_like(ch)
    a[:,1:]=ch[:,:-1]
    a[:,0]=0  # will fix below for i>0
    b[1:,:]=ch[:-1,:]
    b[0,:]=0  # fix below for j>0
    c[1:,1:]=ch[:-1,:-1]
    c[0,:]=0; c[:,0]=0
    # fix first row (i==0,j>0): b=c=a
    b[0,1:]=a[0,1:]; c[0,1:]=a[0,1:]
    # fix first col (j==0,i>0): a=c=b
    a[1:,0]=b[1:,0]; c[1:,0]=b[1:,0]
    # (0,0) already 0
    mx=np.maximum(a,b); mn=np.minimum(a,b)
    pred=np.where(c>=mx,mn,np.where(c<=mn,mx,a+b-c))
    e=(np.abs(a-c)+np.abs(b-c))>>4; e=np.minimum(e,8).astype(np.uint8)
    return pred,(ch-pred).astype(np.int32),e

def huff_bits(counts):
    n=len(counts)
    if n==0: return 0
    if n==1: return int(counts[0])*1
    h=list(map(int,counts)); heapq.heapify(h); t=0
    while len(h)>1:
        x=heapq.heappop(h); y=heapq.heappop(h); s=x+y; t+=s; heapq.heappush(h,s)
    return t

def stream_bits(vals):
    _,cn=np.unique(vals,return_counts=True)
    A=len(cn); return huff_bits(cn.tolist())+16+A*24, huff_bits(cn.tolist()), A

def entropy_bits(vals):
    _,cn=np.unique(vals,return_counts=True)
    p=cn.astype(np.float64)/cn.sum()
    return float(-(p*np.log2(p)).sum()*cn.sum())

def total_order0(Y,Co,Cg):
    denom=Y.size*3
    t=HEADER; d=0; e=0
    for ch in (Y,Co,Cg):
        pred,res,_=med_pred_ctx9(ch)
        tb,db,A=stream_bits(res.reshape(-1)); t+=tb; d+=db
        e+=entropy_bits(res.reshape(-1))
    return t/denom, d/denom, e/denom

def total_ctx9(Y,Co,Cg, res_list=None):
    denom=Y.size*3
    t=HEADER; d=0; e=0
    chs=(Y,Co,Cg)
    for k,ch in enumerate(chs):
        if res_list is None:
            _,res,ctx=med_pred_ctx9(ch)
        else:
            res,ctx=res_list[k]
        for q in range(9):
            g=res[ctx==q]
            if g.size==0: continue
            tb,db,A=stream_bits(g.reshape(-1)); t+=tb; d+=db
            e+=entropy_bits(g.reshape(-1))
    return t/denom, d/denom, e/denom

def main():
    print("anchors: MED order-0 (expect ~3.58) and MED CTX9 (expect ~3.537)")
    av0=[]; av9=[]
    for path in IMAGES:
        nm=path.split("/")[-1]
        img=np.array(Image.open(path).convert("RGB"))
        H,W,_=img.shape; Y,Co,Cg=ycocg_fwd(img)
        # roundtrip check
        t=Y-(Cg//2); B=t-(Co//2); G=Cg+t; R=Co+B
        assert np.array_equal(R,img[:,:,0].astype(np.int32)) and np.array_equal(G,img[:,:,1].astype(np.int32)) and np.array_equal(B,img[:,:,2].astype(np.int32)), nm
        b0,_,e0=total_order0(Y,Co,Cg)
        b9,_,e9=total_ctx9(Y,Co,Cg)
        av0.append(b0); av9.append(b9)
        print(f"{nm}: order0={b0:.4f} (champMOE {CHAMP[nm]:.2f}) ctx9={b9:.4f} ent0={e0:.4f} ent9={e9:.4f}",flush=True)
    print(f"AVG order0={np.mean(av0):.4f} (anchor 3.58, tol +-3% -> [{3.58*0.97:.4f},{3.58*1.03:.4f}]) {'PASS' if 3.58*0.97<=np.mean(av0)<=3.58*1.03 else 'FAIL'}")
    print(f"AVG ctx9  ={np.mean(av9):.4f} (anchor 3.537)")

if __name__=="__main__":
    main()
