"""probe_b5_e: ablation — E12 vs E16 under WIDE 2-way + pure-rANS on E16WIDE groups for reference.
Fast (no rANS refine except last reference)."""
import sys, time
import numpy as np
from PIL import Image
sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
from probe_b5_d import eval_combined, KSET_WIDE
from probe_b5_c import E12, E16
from probe_b4_h import rans_stream_bits
import probe_b4_a as A

IMAGES = [
    "/tmp/opencode/autocompress/experiments/real_photos/kodim01.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim02.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim05.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim07.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim13.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim19.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim23.png",
]

def main():
    for bank,tag in [(E12,"E12-WIDE-2way"),(E16,"E16-WIDE-2way")]:
        t0=time.time(); tot=0; dnt=0
        for path in IMAGES:
            nm=path.split("/")[-1]
            bpp2,det2,_,_,_=eval_combined(path,bank,KSET_WIDE,False)
            img=np.array(Image.open(path).convert("RGB")); H,W,_=img.shape; d=H*W*3
            dnt+=d; tot+=bpp2*d
            print(f"{tag} {nm}: {bpp2:.4f} {det2}",flush=True)
        print(f"==> {tag} AVG={tot/dnt:.4f} [{time.time()-t0:.0f}s]",flush=True)

if __name__=="__main__":
    main()
