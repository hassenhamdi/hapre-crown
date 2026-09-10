import sys
sys.path.insert(0,"/tmp/opencode/autocompress/experiments/stage3_method")
import numpy as np
from PIL import Image
import os
from codec import rgb_to_ycocg_r, ycocg_r_to_rgb, med, paeth, gap
# 128x128 crop of kodim01, tau=1 full loop with recon capture
rgb=np.array(Image.open("/tmp/opencode/autocompress/experiments/real_photos/kodim01.png").convert("RGB"))[:128,:128]
H,W,_=rgb.shape; block=16; tau=1
yc=rgb_to_ycocg_r(rgb)
P=np.pad(yc,((2,0),(2,0),(0,0)),mode='edge').astype(np.int16)
R=P.copy()
for i in range(H):
    pi=i+2
    for j in range(W):
        pj=j+2
        for ch in range(3):
            a=int(R[pi,pj-1,ch]); b=int(R[pi-1,pj,ch]); c=int(R[pi-1,pj-1,ch])
            Wv=int(R[pi,pj-2,ch]); NE=int(R[pi-1,pj+1,ch]) if pj+1<W+2 else b; NNE=int(R[pi-2,pj,ch])
            if a==b==c==Wv: pred=a
            else:
                m=med(a,b,c); g=gap(Wv,Wv,b,c,NE,NNE); p=paeth(a,b,c)
                gh=abs(a-Wv)+abs(b-c)+abs(b-NE); gv=abs(a-c)+abs(b-NNE)+abs(NE-NNE)
                pred=m if gv-gh>40 else (g if gv-gh<-40 else (m+g)//2)
            x=int(P[pi,pj,ch]); r=x-pred
            rq=int(round(r/(2*tau+1))); R[pi,pj,ch]=np.int16(pred+rq*(2*tau+1))
rec_yc=np.zeros((H,W,3),dtype=np.int16)
for i in range(H):
    for j in range(W): rec_yc[i,j]=R[i+2,j+2]
rec=ycocg_r_to_rgb(rec_yc)
err=np.abs(rgb.astype(int)-rec.astype(int))
mse=np.mean((rgb.astype(float)-rec.astype(float))**2)
psnr=10*np.log10(255*255/mse)
print(f"crop128 tau=1 maxerr={err.max()} mse={mse:.2f} psnr={psnr:.2f}dB")