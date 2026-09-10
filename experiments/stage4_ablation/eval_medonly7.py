import sys, numpy as np
sys.path.insert(0,"/tmp/opencode/autocompress/experiments/stage3_method")
sys.path.insert(0,"/tmp/opencode/autocompress/experiments/stage4_ablation")
from ablate import res_medonly_arr, bpp_of
from codec import rgb_to_ycocg_r
from PIL import Image
import os, time
d="/tmp/opencode/autocompress/experiments/real_photos"
for fn in ["kodim01.png","kodim02.png","kodim05.png","kodim07.png","kodim13.png","kodim19.png","kodim23.png"]:
    rgb=np.array(Image.open(os.path.join(d,fn)).convert("RGB"))
    H,W,_=rgb.shape; yc=rgb_to_ycocg_r(rgb)
    t0=time.perf_counter(); r=res_medonly_arr(yc); t=time.perf_counter()-t0
    print(f"{fn} MEDonlyV2 bpp={bpp_of(r,H,W,hdr_bits=48*8+24):.2f} enc_loop={t:.1f}s",flush=True)