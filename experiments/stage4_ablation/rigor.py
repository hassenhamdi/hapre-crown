import sys, numpy as np, math
sys.path.insert(0,"/tmp/opencode/autocompress/experiments/stage3_method")
from codec import rgb_to_ycocg_r, med
from codec_v2 import channel_bytes_v2, elias_gamma_bits
from PIL import Image
import os
d="/tmp/opencode/autocompress/experiments/real_photos"
def res_medonly(arr):
    H,W,_=arr.shape
    P=np.pad(arr,((1,0),(1,0),(0,0)),mode='edge').astype(np.int16)
    res=np.zeros_like(arr)
    for i in range(H):
        for j in range(W):
            for ch in range(3):
                res[i,j,ch]=int(P[i+1,j+1,ch])-med(int(P[i+1,j,ch]),int(P[i,j+1,ch]),int(P[i,j,ch]))
    return res
def exact_rice_run_bits(flat, k):
    # exact token stream bits: per symbol flag1 + unary(q)+k, zero-runs flag0+gamma(L)
    # compute exactly by scanning
    bits=4  # k header
    i=0; N=len(flat)
    while i<N:
        if flat[i]==0:
            j=i
            while j<N and flat[j]==0: j+=1
            bits+=1+elias_gamma_bits(j-i); i=j
        else:
            v=int(flat[i]); u=2*v if v>=0 else -2*v-1
            bits+=1+(u>>k)+1+k; i+=1
    return bits
fn="kodim01.png"
rgb=np.array(Image.open(os.path.join(d,fn)).convert("RGB")); H,W,_=rgb.shape
yc=rgb_to_ycocg_r(rgb)
outs=[]
for run in range(3):
    r=res_medonly(yc)
    tb=0; modes=[]
    for ch in range(3):
        cb=channel_bytes_v2(r[:,:,ch].reshape(-1)); tb+=cb["best"]; modes.append(cb["mode"])
    tot=(tb+48*8+48+7)//8; bpp=tot*8/(H*W*3)
    outs.append((tot,bpp))
    print(f"run{run}: bytes={tot} bpp={bpp:.4f} modes={modes}",flush=True)
print("DETERMINISM:", "PASS" if len(set(o[0] for o in outs))==1 else "FAIL")
# exact Rice+run primary table for all 7
print("=== EXACT Rice+run primary (fully realizable, no Huffman estimates) ===")
for fn in sorted(os.listdir(d)):
    if not fn.startswith("kodim"): continue
    rgb=np.array(Image.open(os.path.join(d,fn)).convert("RGB")); H,W,_=rgb.shape
    r=res_medonly(rgb_to_ycocg_r(rgb))
    tb=0
    for ch in range(3):
        flat=r[:,:,ch].reshape(-1)
        nz=flat[flat!=0]
        m=float(np.mean(np.abs(nz.astype(float))))+1e-9 if len(nz) else 1
        k=int(max(0,min(8,math.floor(math.log2(m)))))
        tb+=exact_rice_run_bits(flat,k)
    tot=(tb+48*8+48+7)//8; bpp=tot*8/(H*W*3)
    print(f"{fn} exactRiceRun bpp={bpp:.2f} bytes={tot}",flush=True)