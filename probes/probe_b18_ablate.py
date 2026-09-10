"""probe_b18_ablate.py — ablation of STACK weapons on 05-C27 (new file)."""
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
os.chdir("/tmp/opencode/autocompress/experiments")
import probe_b18_stack as S

D = "/tmp/opencode/autocompress/experiments/real_photos"


def main():
    px = np.array(Image.open(os.path.join(D, "kodim05.png")).convert("RGB"))
    H, W, _ = px.shape
    npx = H * W * 3
    cfgs = [
        ("frame-parity GS32/noLMS/noERR4", 32, False, False),
        ("+GS16 only", 16, False, False),
        ("+LMS only (GS32)", 32, True, False),
        ("+ERR4 only (GS32)", 32, False, True),
        ("FULL GS16+LMS+ERR4", 16, True, True),
    ]
    for tag, gs, ul, ue in cfgs:
        t0 = time.time()
        blob, info = S.encode_image_rct(px, "C27", 3, 6, gs, use_lms=ul, use_err4=ue)
        bpp = len(blob) * 8 / npx
        out, _ = S.decode_image(blob)
        rt = np.array_equal(np.frombuffer(out, dtype=np.uint8).reshape(H, W, 3), px)
        print(f"{tag}: {bpp:.4f} d vs CR3-3.5984={(bpp - 3.5984) / 3.5984 * 100:+.3f}% "
              f"fams={[c['fam'] for c in info['chs']]} RT={'PASS' if rt else 'FAIL'} "
              f"[{time.time() - t0:.0f}s]", flush=True)
        assert rt


if __name__ == "__main__":
    main()
