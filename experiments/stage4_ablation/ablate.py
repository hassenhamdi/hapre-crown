import sys, numpy as np, math
sys.path.insert(0,"/tmp/opencode/autocompress/experiments/stage3_method")
from codec import rgb_to_ycocg_r, med
from codec_v2 import channel_bytes_v2
from PIL import Image
import os

def res_medonly_arr(arr):
    H,W,_=arr.shape
    P=np.pad(arr,((1,0),(1,0),(0,0)),mode='edge').astype(np.int16)
    res=np.zeros_like(arr)
    for i in range(H):
        for j in range(W):
            for ch in range(3):
                res[i,j,ch]=int(P[i+1,j+1,ch])-med(int(P[i+1,j,ch]),int(P[i,j+1,ch]),int(P[i,j,ch]))
    return res
def bpp_of(res,H,W,hdr_bits=0):
    tb=sum(channel_bytes_v2(res[:,:,ch].reshape(-1))["best"] for ch in range(3))
    return (tb+hdr_bits+7)//8*8/(H*W*3)

rgb=np.array(Image.open("/tmp/opencode/autocompress/experiments/real_photos/kodim01.png").convert("RGB"))
H,W,_=rgb.shape
yc=rgb_to_ycocg_r(rgb)
# A: RGB MED-only (no YCoCg, no mixer, no LPC)
r_rgb=res_medonly_arr(rgb.astype(np.int16))
print(f"A_noYCoCg_MEDonly bpp={bpp_of(r_rgb,H,W):.2f}",flush=True)
# B: YCoCg MED-only (isolates color transform gain)
r_yc_med=res_medonly_arr(yc)
print(f"B_YCoCg_MEDonly bpp={bpp_of(r_yc_med,H,W):.2f}",flush=True)
# Full v2 reference 3.65 (mixer+GAP+LPC+Huffman). D: v1 plain Rice 4.36 known.