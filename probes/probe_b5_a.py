"""probe_b5_a: Stage 1 — reproduce anchors + CROWN-huff/CROWN-rans baselines (exact).
numpy+PIL only (+ctypes libhapre.so for rANS). Imports probe_b4 helpers, no existing files modified.
UNIT: bpp = total_bits/(H*W*3).
Anchors: MED order-0 Huffman ~3.58; CROWN-huff ~3.3429 (per-image list in task).
"""
import sys, math, time
import numpy as np
from PIL import Image
sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
import probe_b4_a as A
from probe_b4_g import prep, KSET2
from probe_b4_f import MOE6
from probe_b4_j import auto_groups_exact, eval_final

IMAGES = [
    "/tmp/opencode/autocompress/experiments/real_photos/kodim01.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim02.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim05.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim07.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim13.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim19.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim23.png",
]
CROWN_HUFF_EXPECT = {"kodim01.png":3.4340,"kodim02.png":3.0931,"kodim05.png":3.7394,
    "kodim07.png":2.8630,"kodim13.png":4.1010,"kodim19.png":3.2758,"kodim23.png":2.8940}

def main():
    print("== anchor 1: MED order-0 Huffman (expect ~3.58) ==")
    av0=[]
    for path in IMAGES:
        nm=path.split("/")[-1]
        img=np.array(Image.open(path).convert("RGB"))
        Y,Co,Cg=A.ycocg_fwd(img)
        b0,_,_=A.total_order0(Y,Co,Cg)
        av0.append(b0*Y.size*3)
        print(f"{nm}: order0={b0:.4f}",flush=True)
    dn=sum(np.array(Image.open(p).convert("RGB")).size for p in IMAGES)
    avg0=sum(av0)/dn
    print(f"AVG order0={avg0:.4f} expect 3.58 tol+-3% [{3.58*0.97:.4f},{3.58*1.03:.4f}] {'PASS' if 3.58*0.97<=avg0<=3.58*1.03 else 'FAIL'}")
    print("== anchor 2: CROWN-huff exact (expect ~3.3429) ==")
    tot=0; dnt=0
    for path in IMAGES:
        nm=path.split("/")[-1]
        bpp,det=eval_final(path,"huff")
        img=np.array(Image.open(path).convert("RGB")); H,W,_=img.shape; d=H*W*3
        dnt+=d; tot+=bpp*d
        exp=CROWN_HUFF_EXPECT[nm]
        print(f"{nm}: crown-huff={bpp:.4f} expect={exp:.4f} d={bpp-exp:+.4f} groups={det}",flush=True)
    avg=tot/dnt
    print(f"AVG crown-huff={avg:.4f} expect 3.3429 d={avg-3.3429:+.4f} {'PASS' if abs(avg-3.3429)/3.3429<=0.03 else 'FAIL'}")
    print("== anchor 3: CROWN-rans (expect ~3.3124 probe-level; champion hillclimb 3.309) ==")
    from probe_b4_h import rans_stream_bits
    tot=0; dnt=0
    for path in IMAGES:
        nm=path.split("/")[-1]
        bpp,det=eval_final(path,"rans")
        img=np.array(Image.open(path).convert("RGB")); H,W,_=img.shape; d=H*W*3
        dnt+=d; tot+=bpp*d
        print(f"{nm}: crown-rans={bpp:.4f} groups={det}",flush=True)
    avg=tot/dnt
    print(f"AVG crown-rans={avg:.4f} (probe 3.3124, champ 3.309)")

if __name__=="__main__":
    main()
