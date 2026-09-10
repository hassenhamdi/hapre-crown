import sys, os
sys.path.insert(0,"/tmp/opencode/autocompress/experiments/stage3_method")
from codec import encode_image
from codec_v2 import channel_bytes_v2
from PIL import Image
import numpy as np
d="/tmp/opencode/autocompress/experiments/real_photos"
# JXL-LL reference bpp from earlier: 01:3.13 07:2.44 23:2.61
for fn in ["kodim01.png","kodim07.png"]:
    rgb=np.array(Image.open(os.path.join(d,fn)).convert("RGB"))
    H,W,_=rgb.shape
    for tau in [1,2]:
        r=encode_image(rgb,tau=tau)
        tb=0
        for ch in range(3):
            # residuals here are quantized indices; byte model same
            cb=channel_bytes_v2(r["res"][:,:,ch].reshape(-1))
            tb+=cb["best"]
        nb=(H//16+1)*(W//16+1); hdr=nb*2+r["nlpc"]*3*8+48*8+48
        tot=(tb+hdr+7)//8; bpp=tot*8/(H*W*3)
        print(f"{fn} tau={tau} bpp={bpp:.2f} bytes={tot}",flush=True)