import io, os, subprocess, sys, time
import numpy as np
from PIL import Image
sys.path.insert(0,"/tmp/opencode/autocompress/csrc")
from driver import encode_image
d="/tmp/opencode/autocompress/experiments/real_photos"
print("img,HAPRE,PNG1,PNG6,PNG9,Wm0,Wm3,Wm6,Je1,Je3,Je9")
for fn in ["kodim01.png","kodim02.png","kodim05.png","kodim07.png","kodim13.png","kodim19.png","kodim23.png"]:
    p=os.path.join(d,fn); im=Image.open(p).convert("RGB"); a=np.array(im); H,W,_=a.shape; n=H*W*3
    blob,_,_,_=encode_image(a); r=[fn,f"{len(blob)*8/n:.2f}"]
    for lv in [1,6,9]:
        b=io.BytesIO(); im.save(b,format="PNG",compress_level=lv); r.append(f"{len(b.getvalue())*8/n:.2f}")
    for m in [0,3,6]:
        b=io.BytesIO(); im.save(b,format="WEBP",lossless=True,quality=100,method=m); r.append(f"{len(b.getvalue())*8/n:.2f}")
    raw="/tmp/_q.ppm"; im.save(raw)
    for e in [1,3,9]:
        j="/tmp/_q.jxl"; subprocess.run(["cjxl",raw,j,"-d","0","-e",str(e),"--quiet"],capture_output=True)
        r.append(f"{os.path.getsize(j)*8/n:.2f}")
    print(",".join(r),flush=True)