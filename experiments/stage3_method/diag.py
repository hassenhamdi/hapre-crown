import numpy as np, sys
sys.path.insert(0,"/tmp/opencode/autocompress/experiments/stage3_method")
from codec import rgb_to_ycocg_r
from PIL import Image
import os
d="/tmp/opencode/autocompress/experiments/stage1_baseline/imgs"
for fn in ["gradient.png","graphics.png","photo_like.png"]:
    rgb=np.array(Image.open(os.path.join(d,fn)).convert("RGB"))
    yc=rgb_to_ycocg_r(rgb)
    # naive MED residual quick estimate using numpy shifts (interior only)
    for ch,nm in enumerate(["Y","Co","Cg"]):
        X=yc[1:,1:,ch].astype(int); a=yc[1:,:-1,ch].astype(int); b=yc[:-1,1:,ch].astype(int); c=yc[:-1,:-1,ch].astype(int)
        pred=np.where(c>=np.maximum(a,b),np.minimum(a,b),np.where(c<=np.minimum(a,b),np.maximum(a,b),a+b-c))
        r=(X-pred)
        vals,cn=np.unique(r,return_counts=True)
        top=sorted(zip(vals,cn),key=lambda t:-t[1])[:5]
        print(fn,nm,f"unique={len(vals)} top={top} meanabs={np.mean(np.abs(r)):.2f}")