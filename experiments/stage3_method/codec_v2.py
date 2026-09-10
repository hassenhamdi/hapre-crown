"""HAPRE-X v2: v1 residuals + run-length zero coding + per-ctx Rice + Huffman-or-Rice selection + outlier escape."""
import numpy as np, time, sys, os
sys.path.insert(0,"/tmp/opencode/autocompress/experiments/stage3_method")
from codec import encode_image as v1_encode  # reuse v1 residuals? No - recompute properly below
from codec import rgb_to_ycocg_r, ycocg_r_to_rgb, med, paeth, gap
from collections import Counter
import math

def elias_gamma_bits(n):  # n>=1
    return 2*(n.bit_length()-1)+1

def channel_bytes_v2(rflat):
    """Exact-ish realizable byte count for one channel's residual array."""
    N=len(rflat)
    # zero runs
    # find runs of zeros
    nz = rflat!=0
    # run lengths of zeros between nonzeros
    runs=[]; cur=0
    for v in rflat:
        if v==0: cur+=1
        else:
            if cur>0: runs.append(cur); cur=0
    if cur>0: runs.append(cur)
    run_bits=sum(elias_gamma_bits(L+1) for L in runs) + 8  # +EOL marker
    nonz=rflat[nz]
    if len(nonz)==0:
        rice_bits=0; huff_bits=0; best=run_bits
        return {"run":run_bits,"rice":0,"huff":0,"best":best,"mode":"run","k":0}
    # per-16-ctx would need ctx ids; approximate gain: use 4 magnitude-classes partitioned k (decoder-reproducible via causal energy - like FLAC)
    # classes by |causal avg|? Without causal array here, approximate by splitting nonzeros into small (|r|<=4) and large
    small=nonz[np.abs(nonz)<=4]; large=nonz[np.abs(nonz)>4]
    def rice_bits(arr):
        if len(arr)==0: return 0,0
        u=np.where(arr>=0,2*arr,-2*arr-1).astype(np.int64)
        m=float(np.mean(np.abs(arr.astype(np.float64))))+1e-9
        k=int(max(0,min(8,math.floor(math.log2(m)))))
        return int(np.sum((u>>k)+1+k)),k
    bs,k1=rice_bits(small); bl,k2=rice_bits(large)
    # outlier escape: values |r|>64 as raw 10b instead of long unary
    out=np.abs(nonz)>64
    esc_save=int(np.sum([max(0,((2*abs(int(v))>> (k1 if abs(int(v))<=4 else k2))+1)-10) for v in nonz[out]])) if np.any(out) else 0
    rice_bits_tot=run_bits+bs+bl-esc_save+16  # 16b for 2 k values
    # Huffman alternative on full channel alphabet
    cnt=Counter(rflat.tolist()); probs=np.array(list(cnt.values()))/N
    ent=float(-np.sum(probs*np.log2(probs)))
    huff_bits_tot=int(ent*N)+256*8+run_bits*0  # Huffman handles zeros via short code, no separate runs; +code table
    # raw fallback
    raw_bits=N*9  # 9b for int16 range
    cands={"rice+run":rice_bits_tot,"huff":int(huff_bits_tot),"raw":raw_bits}
    mode=min(cands,key=cands.get)
    return {"run":run_bits,"rice":rice_bits_tot,"huff":int(huff_bits_tot),"best":int(cands[mode]),"mode":mode,"k":(k1,k2)}

def evaluate_v2():
    from PIL import Image
    d="/tmp/opencode/autocompress/experiments/stage1_baseline/imgs"
    print(f"{'img':15s} {'ch':8s} {'mode':10s} {'bits':>10s} {'bpp_ch':>7s}")
    tot_bpp={}
    for fn in sorted(os.listdir(d)):
        rgb=np.array(Image.open(os.path.join(d,fn)).convert("RGB"))
        r=v1_encode(rgb,tau=0)  # reuse v1 residual engine (proven lossless), re-estimate bytes with v2 entropy
        res=r["res"]; H,W,_=res.shape
        tb=0
        for ch,nm in enumerate(["Y","Co","Cg"]):
            cb=channel_bytes_v2(res[:,:,ch].reshape(-1))
            bpp=cb["best"]/(H*W)
            tb+=cb["best"]
            print(f"{fn:15s} {nm:8s} {cb['mode']:10s} {cb['best']:10d} {bpp:7.2f}")
        # headers: modes+LPC+bias as v1 (~2-4KB worst) + 3B channel-mode header
        hdr=r["bytes"]*8 - int(sum([0]))  # v1 bytes include old Rice; recompute: headers only
        # v1 headers ≈ modes(2b/block*nblocks)+lpc+HDR const; recompute quickly:
        nb=(H//16+1)*(W//16+1); hdr_bits=nb*2 + r["nlpc"]*3*8 + 48*8 + 24 + 24
        tot_bytes=(tb+hdr_bits+7)//8
        bpp=tot_bytes*8/(H*W*3)
        tot_bpp[fn]=bpp
        print(f"  => {fn} V2 bytes={tot_bytes} bpp={bpp:.2f} (V1 bpp={r['bpp']:.2f}) lossless_ok={r['lossless_ok']}")
    return tot_bpp

if __name__=="__main__":
    evaluate_v2()