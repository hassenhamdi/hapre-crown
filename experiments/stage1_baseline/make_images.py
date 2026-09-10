import numpy as np
from PIL import Image
import os
out="/tmp/opencode/autocompress/experiments/stage1_baseline/imgs"
os.makedirs(out, exist_ok=True)
rng=np.random.default_rng(0)
H=W=256
# 1. smooth gradient (easy)
x=np.tile(np.arange(W,dtype=np.uint8),(H,1))
img=np.stack([x, np.tile(np.arange(H,dtype=np.uint8)[:,None],(1,W)), np.full((H,W),128,dtype=np.uint8)],axis=-1)
Image.fromarray(img).save(f"{out}/gradient.png")
# 2. photo-like: correlated noise via box blur of uniform
base=rng.integers(0,256,size=(H,W,3)).astype(np.float32)
# simple 3x3 blur iterations for spatial correlation
for _ in range(3):
    base=(base+np.roll(base,1,0)+np.roll(base,-1,0)+np.roll(base,1,1)+np.roll(base,-1,1))/5
base=np.clip(base,0,255).astype(np.uint8)
Image.fromarray(base).save(f"{out}/photo_like.png")
# 3. edges/text-like: white bg with black bars + text blocks
e=np.full((H,W,3),255,dtype=np.uint8)
e[::32,:,:]=0; e[:,::32,:]=0
e[64:96,64:192,:]=np.stack([rng.integers(0,256,size=(32,128))]*3,axis=-1).astype(np.uint8)
Image.fromarray(e).save(f"{out}/graphics.png")
# 4. noisy (hard)
n=rng.integers(0,256,size=(H,W,3),dtype=np.uint8)
Image.fromarray(n).save(f"{out}/noise.png")
# 5. natural-ish: horizontal bands + gradient + noise
v=(np.tile(np.linspace(0,255,W),(H,1))+rng.normal(0,8,size=(H,W))).clip(0,255).astype(np.uint8)
nat=np.stack([v, np.roll(v,40,0), np.roll(v,80,1)],axis=-1)
Image.fromarray(nat).save(f"{out}/bands.png")
print("wrote", os.listdir(out))