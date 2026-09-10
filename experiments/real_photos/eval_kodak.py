import sys, os, time
sys.path.insert(0,"/tmp/opencode/autocompress/experiments/stage3_method")
from codec import encode_image
from codec_v2 import channel_bytes_v2
from PIL import Image
import numpy as np
d="/tmp/opencode/autocompress/experiments/real_photos"
for fn in ["kodim01.png","kodim07.png","kodim23.png"]:
    rgb=np.array(Image.open(os.path.join(d,fn)).convert("RGB"))
    H,W,_=rgb.shape
    r=encode_image(rgb,tau=0)
    tb=0; modes=[]
    for ch in range(3):
        cb=channel_bytes_v2(r["res"][:,:,ch].reshape(-1))
        tb+=cb["best"]; modes.append(cb["mode"])
    nb=(H//16+1)*(W//16+1); hdr=nb*2+r["nlpc"]*3*8+48*8+48
    tot=(tb+hdr+7)//8; bpp=tot*8/(H*W*3)
    print(f"{fn} V1bpp={r['bpp']:.2f} V2bpp={bpp:.2f} V2bytes={tot} modes={modes} nlpc={r['nlpc']} ok={r['lossless_ok']}",flush=True)