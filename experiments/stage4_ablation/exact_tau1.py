import sys, numpy as np, math
sys.path.insert(0,"/tmp/opencode/autocompress/experiments/stage3_method")
from codec import encode_image
from codec_v2 import elias_gamma_bits
from PIL import Image
import os
def exact_bits(flat,k):
    bits=4; i=0; N=len(flat)
    while i<N:
        if flat[i]==0:
            j=i
            while j<N and flat[j]==0: j+=1
            bits+=1+elias_gamma_bits(j-i); i=j
        else:
            v=int(flat[i]); u=2*v if v>=0 else -2*v-1
            bits+=1+(u>>k)+1+k; i+=1
    return bits
d="/tmp/opencode/autocompress/experiments/real_photos"
for fn in ["kodim01.png","kodim02.png","kodim05.png","kodim07.png","kodim13.png","kodim19.png","kodim23.png"]:
    rgb=np.array(Image.open(os.path.join(d,fn)).convert("RGB")); H,W,_=rgb.shape
    r=encode_image(rgb,tau=1)
    tb=0
    for ch in range(3):
        flat=r["res"][:,:,ch].reshape(-1)
        nz=flat[flat!=0]
        m=float(np.mean(np.abs(nz.astype(float))))+1e-9 if len(nz) else 1
        k=int(max(0,min(8,math.floor(math.log2(m)))))
        tb+=exact_bits(flat,k)
    nb=(H//16+1)*(W//16+1); hdr=nb*2+r["nlpc"]*3*8+48*8+48
    tot=(tb+hdr+7)//8; bpp=tot*8/(H*W*3)
    print(f"{fn} tau1-EXACT bpp={bpp:.2f} bytes={tot}",flush=True)