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
echo "== 3. round-trips: RUN (Python) =="
python3 - <<'PY'
import os
ROOT = os.environ.get("AUTOCOMPRESS_ROOT", "/tmp/opencode/autocompress")
import sys
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "probes"))
sys.path.insert(0, os.path.join(ROOT, "experiments"))
from driver_run import encode_run, decode_run
from PIL import Image
import numpy as np
D = os.path.join(ROOT, "experiments/real_photos")
print(f"{'img':14s} {'RUN':>7s}")
for fn in ["kodim23.png"]:
    rgb = np.array(Image.open(os.path.join(D, fn)).convert("RGB")); H, W, _ = rgb.shape
    b1, _ = encode_run(rgb); assert decode_run(b1, H, W) == rgb.tobytes(), fn
    print(f"{fn:14s} {len(b1)*8/(H*W*3):7.2f} PASS")
print("RUN ROUND-TRIPS PASS")
PY
echo "== 4. CROWN6 C++ round-trip (256px crop: train -> enc -> dec) =="
python3 -c "from PIL import Image; Image.open('experiments/real_photos/kodim23.png').crop((0,0,256,256)).save('/tmp/repro_crop256.png')"
mkdir -p /tmp/repro_w
./cpp/crown_train_fast /tmp/repro_crop256.png /tmp/repro_w --rct all
./cpp/crown_enc_fast /tmp/repro_crop256.png /tmp/repro_w /tmp/repro_crop256.bin
./cpp/crown_dec_fast /tmp/repro_crop256.bin /tmp/repro_crop256.rgb
python3 -c "import numpy as np; from PIL import Image; a=np.array(Image.open('/tmp/repro_crop256.png').convert('RGB')).tobytes(); assert open('/tmp/repro_crop256.rgb','rb').read()==a; print('CROWN6 C++ ROUND-TRIP PASS')"
echo "== reproduce OK =="
