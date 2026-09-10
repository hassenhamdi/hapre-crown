import time, io, os, subprocess, sys
import numpy as np
from PIL import Image
sys.path.insert(0,"/tmp/opencode/autocompress/csrc")
from driver import encode_image, decode_image
d="/tmp/opencode/autocompress/experiments/real_photos"
files=[f"kodim{n}.png" for n in ["01","02","05","07","13","19","23"]]
agg={}
def add(codec,bpp,emp,dmp):
    a=agg.setdefault(codec,[0,0,0,0]); a[0]+=bpp; a[1]+=emp; a[2]+=dmp; a[3]+=1
for fn in files:
    p=os.path.join(d,fn); im=Image.open(p).convert("RGB"); a=np.array(im); H,W,_=a.shape; npx=H*W
    # OURS
    t0=time.perf_counter(); blob,_,_,_=encode_image(a); te=time.perf_counter()-t0
    t0=time.perf_counter(); dec,_=decode_image(blob,H,W); td=time.perf_counter()-t0
    assert dec==a.tobytes()
    add("HAPRE-C",len(blob)*8/(npx*3),npx/1e6/te,npx/1e6/td)
    # PNG 1/6/9
    for lv in [1,6,9]:
        t0=time.perf_counter(); b=io.BytesIO(); im.save(b,format="PNG",compress_level=lv); te=time.perf_counter()-t0
        by=b.getvalue()
        t0=time.perf_counter(); Image.open(io.BytesIO(by)).load(); td=time.perf_counter()-t0
        add(f"PNG-{lv}",len(by)*8/(npx*3),npx/1e6/te,npx/1e6/td)
    # WebP 0/3/6
    for m in [0,3,6]:
        t0=time.perf_counter(); b=io.BytesIO(); im.save(b,format="WEBP",lossless=True,quality=100,method=m); te=time.perf_counter()-t0
        by=b.getvalue()
        t0=time.perf_counter(); Image.open(io.BytesIO(by)).load(); td=time.perf_counter()-t0
        add(f"WebP-m{m}",len(by)*8/(npx*3),npx/1e6/te,npx/1e6/td)
    # JXL e1/e3 (e9 have bpp; time it too)
    raw="/tmp/_p.ppm"
    im.save(raw)
    for e in [1,3,9]:
        j="/tmp/_p.e.jxl"
        t0=time.perf_counter(); subprocess.run(["cjxl",raw,j,"-d","0","-e",str(e),"--quiet"],capture_output=True); te=time.perf_counter()-t0
        by=os.path.getsize(j)
        t0=time.perf_counter(); subprocess.run(["djxl",j,"/tmp/_p.d.png","--quiet"],capture_output=True); td=time.perf_counter()-t0
        add(f"JXL-e{e}",by*8/(npx*3),npx/1e6/te,npx/1e6/td)
    print("done",fn,flush=True)
print(f"\n{'codec':10s} {'bpp':>6s} {'encMP/s':>8s} {'decMP/s':>8s}")
for k,v in sorted(agg.items(),key=lambda kv:kv[1][0]/kv[1][3]):
    print(f"{k:10s} {v[0]/v[3]:6.2f} {v[1]/v[3]:8.2f} {v[2]/v[3]:8.2f}")