import os, time, subprocess, io
from PIL import Image
d="/tmp/opencode/autocompress/experiments/real_photos"
for fn in sorted(os.listdir(d)):
    if not fn.endswith(".png") or not fn.startswith("kodim"): continue
    p=os.path.join(d,fn)
    im=Image.open(p).convert("RGB"); import numpy as np
    arr=np.array(im); H,W,_=arr.shape; npx=H*W*3
    t0=time.perf_counter(); buf=io.BytesIO(); im.save(buf,format="PNG",optimize=True,compress_level=9); png=len(buf.getvalue()); tpng=time.perf_counter()-t0
    t0=time.perf_counter(); buf=io.BytesIO(); im.save(buf,format="WEBP",lossless=True,quality=100,method=6); webp=len(buf.getvalue()); twebp=time.perf_counter()-t0
    raw="/tmp/_r.ppm"; jxl="/tmp/_r.jxl"; im.save(raw)
    t0=time.perf_counter(); subprocess.run(["cjxl",raw,jxl,"-d","0","-e","9","--quiet"],capture_output=True); tjxl=time.perf_counter()-t0
    jxlb=os.path.getsize(jxl)
    for f,b,t in [("PNG",png,tpng),("WebP-LL",webp,twebp),("JXL-LL",jxlb,tjxl)]:
        print(f"{fn} {f:8s} bytes={b:7d} bpp={b*8/npx:5.2f} enc={t*1000:7.0f}ms")