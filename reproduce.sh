#!/bin/bash
# reproduce.sh — checksums -> build -> byte-exact round-trips + bpp table. Exits nonzero on failure.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
if [ ! -d "$ROOT/experiments" ]; then ROOT="/tmp/opencode/autocompress"; fi
cd "$ROOT"
[ -d experiments/real_photos ] || { echo "FATAL: wrong cwd $(pwd)"; exit 1; }
echo "== 1. checksums =="
md5sum -c CHECKSUMS.txt
echo "== 2. build =="
make all
echo "== 3. round-trips (kodim23 + kodim07, fast exact codecs) =="
python3 - <<'PY'
import os
ROOT = "/tmp/opencode/autocompress"
os.chdir(ROOT)
import sys
sys.path.insert(0, "csrc")
from driver_run import encode_run, decode_run
from driver_crown import encode_crown, decode_crown
from PIL import Image
import numpy as np
D = "experiments/real_photos"
print(f"{'img':14s} {'RUN':>7s} {'CROWN-huff':>10s}")
for fn in ["kodim23.png", "kodim07.png"]:
    rgb = np.array(Image.open(f"{D}/{fn}").convert("RGB")); H, W, _ = rgb.shape
    b1, _ = encode_run(rgb); assert decode_run(b1, H, W) == rgb.tobytes(), fn
    b2, _ = encode_crown(rgb); assert decode_crown(b2, H, W) == rgb.tobytes(), fn
    print(f"{fn:14s} {len(b1)*8/(H*W*3):7.2f} {len(b2)*8/(H*W*3):10.2f}")
print("ALL ROUND-TRIPS PASS")
PY
echo "== reproduce OK =="
