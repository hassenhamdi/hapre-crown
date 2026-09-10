import os, time, subprocess, io
from PIL import Image
import numpy as np
d="/tmp/opencode/autocompress/experiments/stage1_baseline/imgs"
for fn in sorted(os.listdir(d)):
    p=os.path.join(d,fn)
    im=Image.open(p); arr=np.array(im); H,W,_=arr.shape; npx=H*W*3
    # PNG max
    t0=time.perf_counter(); buf=io.BytesIO(); im.save(buf,format="PNG",optimize=True,compress_level=9); png=len(buf.getvalue()); tpng=time.perf_counter()-t0
    # WebP lossless max
    t0=time.perf_counter(); buf=io.BytesIO(); im.save(buf,format="WEBP",lossless=True,quality=100,method=6); webp=len(buf.getvalue()); twebp=time.perf_counter()-t0
    # JPEG-XL lossless via cjxl
    raw="/tmp/_t.ppm"; jxl="/tmp/_t.jxl"
    im.save(raw)
    t0=time.perf_counter()
    r=subprocess.run(["cjxl",raw,jxl,"-d","0","-e","9","--quiet"],capture_output=True)
    tjxl=time.perf_counter()-t0
    jxlb=os.path.getsize(jxl) if os.path.exists(jxl) else -1
    # decode check cjxl->png via djxl?
    for f,b,t in [("PNG",png,tpng),("WebP-LL",webp,twebp),("JXL-LL",jxlb,tjxl)]:
        bpp=b*8/npx if b>0 else -1
        print(f"{fn:15s} {f:8s} bytes={b:7d} bpp={bpp:5.2f} enc={t*1000:7.1f}ms")