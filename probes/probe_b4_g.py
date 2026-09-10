"""probe_b4_g: push winner further — per-group predictors, wider K, energy sub-split, exact rANS.
numpy+PIL (+ctypes to EXISTING libhapre.so for rANS only). No existing files modified.
"""
import numpy as np, ctypes
from PIL import Image
import probe_b4_a as A
from probe_b4_b import nbhd, predictors
from probe_b4_c import loco_ctx365
from probe_b4_f import run_pass, gamma_cost, block_experts, huff_roundtrip, recon_proof, med_of, MOE6

IMAGES=A.IMAGES; HEADER=A.HEADER
CHAMP={"kodim01.png":3.56,"kodim02.png":3.23,"kodim05.png":3.84,"kodim07.png":3.01,
       "kodim13.png":4.17,"kodim19.png":3.41,"kodim23.png":3.03}
KSET2=(2,3,4,6,9,12,18,27,36)

def prep(ch):
    a,b,c,d,Ww,NNe,NE=nbhd(ch)
    P=predictors(a,b,c,d,Ww,NNe,NE)
    key,s=loco_ctx365(a,b,c,d)
    return dict(a=a,b=b,c=c,d=d,P=P,key=key,s=s)

def auto_groups(key, rf, main, KSET):
    km=key[main]; rm=rf[main]
    best=None
    for K in KSET:
        uk,cn=np.unique(km,return_counts=True)
        ma=np.array([np.abs(rm[km==v]).mean() for v in uk])
        order=np.argsort(ma,kind="stable")
        uks=uk[order]; cns=cn[order]
        tot=cns.sum(); tgt=tot/K; groups=[]; cur=[]; acc=0
        for u,c in zip(uks,cns):
            cur.append(u); acc+=c
            if acc>=tgt and len(groups)<K-1: groups.append(cur); cur=[]; acc=0
        groups.append(cur)
        tt=len(uk)*4+8+3
        for gkeys in groups:
            g=rm[np.isin(km,np.array(gkeys))]
            if g.size: tb,_,_=A.stream_bits(g.reshape(-1)); tt+=tb
        if best is None or tt<best: best=tt; bK=K; bG=groups
    return bK,bG,best

def eval_img(path, mode):
    """mode: base | G(bestpred) | Kwide | Esplit"""
    img=np.array(Image.open(path).convert("RGB"))
    H,W,_=img.shape; dn=H*W*3
    Y,Co,Cg=A.ycocg_fwd(img); chs=[Y,Co,Cg]
    DD=[prep(ch) for ch in chs]
    KS = KSET2 if mode in ("Kwide","G","Esplit","FULL") else (4,6,9,12,18,27)
    total=HEADER
    for D,kch in zip(DD,chs):
        P=D["P"]; key=D["key"]; s=D["s"]
        main,runs=run_pass(kch,H,W)
        if mode in ("G","FULL"):
            # cluster on MED flip residuals, then per-group best predictor
            resM=(kch-P["MED"]).astype(np.int32); rfM=(s*resM).astype(np.int32)
            _,G,_=auto_groups(key,rfM,main,KS)
            for gkeys in G:
                sel=main & np.isin(key,np.array(gkeys))
                # candidates: flipped residuals per expert
                best=None
                for n in MOE6:
                    r=(kch-P[n]).astype(np.int32); rf=(s*r).astype(np.int32)
                    g=rf[sel]
                    if g.size==0: d_=0
                    else:
                        _,cn=np.unique(g,return_counts=True)
                        d_=A.huff_bits(cn.tolist())
                    if best is None or d_<best: best=d_; bn=n
                r=(kch-P[bn]).astype(np.int32); rf=(s*r).astype(np.int32)
                g=rf[sel]
                if g.size: tb,_,_=A.stream_bits(g.reshape(-1)); total+=tb
            uk=np.unique(key[main]); total+=len(uk)*4+8+3+len(G)*3
        elif mode=="Esplit":
            resM=(kch-P["MED"]).astype(np.int32); rfM=(s*resM).astype(np.int32)
            _,G,_=auto_groups(key,rfM,main,KS)
            e9=np.minimum((np.abs(D["a"]-D["c"])+np.abs(D["b"]-D["c"]))>>4,8)
            for gkeys in G:
                sel=main & np.isin(key,np.array(gkeys))
                # split group by energy low/high (median energy of group)
                ge=e9[sel]
                thr=int(np.median(ge)) if ge.size else 0
                for part in (ge<=thr,ge>thr):
                    g=rfM[sel][part[sel] if False else part]
                    if g.size: tb,_,_=A.stream_bits(g.reshape(-1)); total+=tb
            uk=np.unique(key[main]); total+=len(uk)*4+8+3+len(G)*8
        else:
            resM=(kch-P["MED"]).astype(np.int32); rfM=(s*resM).astype(np.int32)
            _,G,_=auto_groups(key,rfM,main,KS)
            for gkeys in G:
                g=rfM[main & np.isin(key,np.array(gkeys))]
                if g.size: tb,_,_=A.stream_bits(g.reshape(-1)); total+=tb
            uk=np.unique(key[main]); total+=len(uk)*4+8+3
        total+=sum(gamma_cost(L) for L in runs)
    total+=64
    return total/dn

def main():
    for mode in ("base","Kwide","G","Esplit"):
        tot=0; dn_tot=0
        for path in IMAGES:
            nm=path.split("/")[-1]
            bpp=eval_img(path,mode)
            img=np.array(Image.open(path).convert("RGB")); H,W,_=img.shape; dn=H*W*3
            dn_tot+=dn; tot+=bpp*dn
            print(f"{mode} {nm}: {bpp:.4f} champ={CHAMP[nm]:.2f} d={bpp-CHAMP[nm]:+.4f}",flush=True)
        avg=tot/dn_tot
        print(f"==> {mode} AVG={avg:.4f} d_champ={avg-3.4643:+.4f} d_m3={avg-3.38:+.4f}",flush=True)

if __name__=="__main__":
    main()
