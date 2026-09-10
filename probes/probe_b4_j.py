"""probe_b4_j: DEFINITIVE final numbers with exact side accounting.
Side per channel: active_ctx*ceil(log2(K)) map bits + 4b K-id + 8b header.
Configs: CROWN-huff (MED/6exp per-group, autoK, norun, Huffman) and CROWN-rans (same streams, rANS M=14).
"""
import sys, math; sys.path.insert(0,'experiments')
import numpy as np
from PIL import Image
import probe_b4_a as A
from probe_b4_g import prep, KSET2
from probe_b4_f import MOE6
from probe_b4_h import rans_stream_bits
IMAGES=A.IMAGES; HEADER=A.HEADER
CHAMP={"kodim01.png":3.56,"kodim02.png":3.23,"kodim05.png":3.84,"kodim07.png":3.01,
       "kodim13.png":4.17,"kodim19.png":3.41,"kodim23.png":3.03}

def auto_groups_exact(key, rf, KSET):
    best=None
    for K in KSET:
        uk,cn=np.unique(key,return_counts=True)
        ma=np.array([np.abs(rf[key==v]).mean() for v in uk])
        order=np.argsort(ma,kind="stable")
        uks=uk[order]; cns=cn[order]
        tot=cns.sum(); tgt=tot/K; groups=[]; cur=[]; acc=0
        for u,c in zip(uks,cns):
            cur.append(u); acc+=c
            if acc>=tgt and len(groups)<K-1: groups.append(cur); cur=[]; acc=0
        groups.append(cur)
        mapbits=len(uk)*math.ceil(math.log2(K))+4+8
        tt=mapbits
        for gkeys in groups:
            g=rf[np.isin(key,np.array(gkeys))]
            if g.size: tb,_,_=A.stream_bits(g.reshape(-1)); tt+=tb
        if best is None or tt<best: best=tt; bK=K; bG=groups; bmap=mapbits
    return bK,bG,best,bmap

def eval_final(path, coder):
    img=np.array(Image.open(path).convert("RGB"))
    H,W,_=img.shape; dn=H*W*3
    Y,Co,Cg=A.ycocg_fwd(img); chs=[Y,Co,Cg]
    DD=[prep(ch) for ch in chs]
    t=HEADER; detail=[]
    for D,kch in zip(DD,chs):
        P=D["P"]; key=D["key"]; s=D["s"]
        resM=(kch-P["MED"]).astype(np.int32); rfM=(s*resM).astype(np.int32)
        bK,G,_,bmap=auto_groups_exact(key,rfM,KSET2)
        t+=bmap+len(G)*3
        for gkeys in G:
            sel=np.isin(key,np.array(gkeys))
            best=None
            for n in MOE6:
                r=(kch-P[n]).astype(np.int32); rf=(s*r).astype(np.int32)
                g=rf[sel]
                d_=A.huff_bits(np.unique(g,return_counts=True)[1].tolist()) if g.size else 0
                if best is None or d_<best: best=d_; bn=n
            r=(kch-P[bn]).astype(np.int32); rf=(s*r).astype(np.int32)
            g=rf[sel]
            if g.size:
                if coder=="huff": tb,_,_=A.stream_bits(g.reshape(-1)); t+=tb
                else: tb,_=rans_stream_bits(g); t+=tb
        detail.append((bK,len(G)))
    return t/dn, detail

def main():
    for coder in ("huff","rans"):
        tot=0; dn_tot=0
        for path in IMAGES:
            nm=path.split("/")[-1]
            bpp,det=eval_final(path,coder)
            img=np.array(Image.open(path).convert("RGB")); H,W,_=img.shape; dn=H*W*3
            dn_tot+=dn; tot+=bpp*dn
            print(f"{coder} {nm}: {bpp:.4f} d_champ={bpp-CHAMP[nm]:+.4f} groups={det}",flush=True)
        avg=tot/dn_tot
        print(f"==> FINAL-{coder} AVG={avg:.4f} d_champ={avg-3.4643:+.4f} d_m3pin={avg-3.38:+.4f}",flush=True)

if __name__=="__main__":
    main()
