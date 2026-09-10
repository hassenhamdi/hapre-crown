#!/bin/bash
cd /tmp/opencode/autocompress/experiments
python3 probe_b21_perctx.py --images kodim23.png --hidden 8 --nfeat 8 --iters 200 --lr 0.003 --curves --out probe_b21_sweep_23h8f8.json
python3 probe_b21_perctx.py --images kodim23.png --hidden 16 --nfeat 8 --iters 200 --lr 0.003 --out probe_b21_sweep_23h16f8.json
python3 probe_b21_perctx.py --images kodim23.png --hidden 8 --nfeat 12 --iters 200 --lr 0.003 --out probe_b21_sweep_23h8f12.json
python3 probe_b21_perctx.py --images kodim05.png --hidden 8 --nfeat 8 --iters 200 --lr 0.003 --out probe_b21_sweep_05h8f8.json