"""probe_b4_c: oracle per-ctx predictor, LOCO365 bias, clustered LOCO coding, xch decor, rANS gap.
numpy+PIL only.
"""
import numpy as np, heapq
from PIL import Image
import probe_b4_a as A
from probe_b4_b import nbhd, predictors, ctx9_from, bits_ctx9_of_resids, bits_order0_of_resids

IMAGES=A.IMAGES; HEADER=A.HEADER

def loco_q(g):
    a=np.abs(g)
    q=np.zeros_like(g,dtype=np.int8)
    q=np.where(g==0,0,np.where(a<=2,np.sign(g)*1,np.where(a<=7,np.sign(g)*2,np.where(a<=21,np.sign(g)*3,np.sign(g)*4))))
    return q

def loco_ctx365(a,b,c,d):
    q1=loco_q(b-c); q2=loco_q(c-a); q3=loco_q(d-b)
    neg=(q1<0)|((q1==0)&(q2<0))|((q1==0)&(q2==0)&(q3<0))
    s=np.where(neg,-1,1)
    m1=np.where(neg,-q1,q1).astype(np.int64); m2=np.where(neg,-q2,q2).astype(np.int64); m3=np.where(neg,-q3,q3).astype(np.int64)
    key=(m1*81+m2*9+m3)  # 0..728 but only 365 reachable merged ids; compress via unique map per image? use full 729 range (fine)
    return key.astype(np.int32), s.astype(np.int8)

def huff_data_only(vals):
    _,cn=np.unique(vals,return_counts=True)
    return A.huff_bits(cn.tolist()), len(cn)

def main():
    # accumulators (bits, pooled)
    base9=0; orac9=0; locobias9=0; clus9n={9:0,18:0,27:0}; xch=0; denom=0
    rans_save=0; huff_data_tot=0; ent_tot=0; tbl_tot=0
    for path in IMAGES:
        nm=path.split("/")[-1]
        img=np.array(Image.open(path).convert("RGB"))
        H,W,_=img.shape; dn=H*W*3; denom+=dn
        Y,Co,Cg=A.ycocg_fwd(img); chs=[Y,Co,Cg]
        NB=[]
        for ch in chs:
            a,b,c,d,Ww,NNe,NE=nbhd(ch)
            NB.append(dict(a=a,b=b,c=c,d=d,Ww=Ww,NNe=NNe,NE=NE,ctx9=ctx9_from(a,b,c),
                           pred=predictors(a,b,c,d,Ww,NNe,NE),loco=loco_ctx365(a,b,c,d)))
        # --- base CTX9 MED ---
        t=HEADER
        for k in range(3):
            res=(chs[k]-NB[k]["pred"]["MED"]).astype(np.int32); ctx=NB[k]["ctx9"]
            for q in range(9):
                g=res[ctx==q]
                if g.size: tb,_,_=A.stream_bits(g.reshape(-1)); t+=tb
        base9+=t
        # --- oracle per-(ch,ctx9) best predictor among 7 (side 3b*27) ---
        t=HEADER+27*3
        for k in range(3):
            ctx=NB[k]["ctx9"]
            PN=["MED","TOP","LEFT","PAETH","GRAD","GAP","DG"]
            RR={n:(chs[k]-NB[k]["pred"][n]).astype(np.int32) for n in PN}
            for q in range(9):
                best=None
                for n in PN:
                    g=RR[n][ctx==q]
                    if g.size==0: d_=0
                    else: d_,_=huff_data_only(g.reshape(-1))
                    if best is None or d_<best: best=d_
                # table cost of best group (alphabet of best predictor's group)
                # find which predictor won to count its alphabet
                bn=None
                for n in PN:
                    g=RR[n][ctx==q]
                    d_ = huff_data_only(g.reshape(-1))[0] if g.size else 0
                    if d_==best: bn=n; break
                g=RR[bn][ctx==q]
                if g.size: tb,_,_=A.stream_bits(g.reshape(-1)); t+=tb
                else: t+=16
        orac9+=t
        # --- LOCO365 global mean bias + CTX9 coding ---
        t=HEADER
        nactive=0
        for k in range(3):
            res=(chs[k]-NB[k]["pred"]["MED"]).astype(np.int32)
            key,_=NB[k]["loco"]; ctx=NB[k]["ctx9"]
            rc=res.copy()
            uk=np.unique(key)
            for v in uk:
                m=res[key==v]
                mu=int(np.round(m.mean()))
                if mu!=0: rc[key==v]=m-mu; nactive+=1
            for q in range(9):
                g=rc[ctx==q]
                if g.size: tb,_,_=A.stream_bits(g.reshape(-1)); t+=tb
        t+=nactive*4  # side: 4b per active bias (range small)
        locobias9+=t
        # --- clustered LOCO coding: sort ctx by meanabs, equal-pixel K groups, exact Huffman ---
        for K in (9,18,27):
            t=HEADER
            for k in range(3):
                res=(chs[k]-NB[k]["pred"]["MED"]).astype(np.int32)
                key,_=NB[k]["loco"]
                uk,cn=np.unique(key,return_counts=True)
                ma=np.array([np.abs(res[key==v]).mean() for v in uk])
                order=np.argsort(ma)
                # equal-pixel partition
                tot=cn.sum(); tgt=tot/K; groups=[]; cur=[]; acc=0
                for idx in order:
                    cur.append(uk[idx]); acc+=cn[list(uk).index(uk[idx])]
                    if acc>=tgt and len(groups)<K-1: groups.append(cur); cur=[]; acc=0
                groups.append(cur)
                for gkeys in groups:
                    mask=np.isin(key,np.array(gkeys))
                    g=res[mask]
                    if g.size==0: continue
                    tb,_,_=A.stream_bits(g.reshape(-1)); t+=tb
            clus9n[K]+=t
        # --- xch: Cg decor from Co residual (per-image alpha), Y from nothing ---
        rY=(chs[0]-NB[0]["pred"]["MED"]).astype(np.float64)
        rCo=(chs[1]-NB[1]["pred"]["MED"]).astype(np.float64)
        rCg=(chs[2]-NB[2]["pred"]["MED"]).astype(np.float64)
        # optimal alpha = cov/var
        for (src,tgtname) in ((rCo,"Cg"),):
            al=float((src*rCg).sum()/max(1.0,(src*src).sum()))
            alq=int(np.round(al*16))  # Q4 side 8b
            r2=np.round(rCg-alq/16*src).astype(np.int32)
            ctx=NB[2]["ctx9"]
            t=HEADER  # only Cg part + side; compare Cg-only vs base Cg-only
        # full xch total: Y,Co as base ctx9 + Cg decorrelated
        t=HEADER+8
        for k in (0,1):
            res=(chs[k]-NB[k]["pred"]["MED"]).astype(np.int32); ctx=NB[k]["ctx9"]
            for q in range(9):
                g=res[ctx==q]
                if g.size: tb,_,_=A.stream_bits(g.reshape(-1)); t+=tb
        al=float((rCo*rCg).sum()/max(1.0,(rCo*rCo).sum()))
        alq=int(np.round(al*16))
        r2=np.round(rCg-alq/16*rCo).astype(np.int32); ctx=NB[2]["ctx9"]
        for q in range(9):
            g=r2[ctx==q]
            if g.size: tb,_,_=A.stream_bits(g.reshape(-1)); t+=tb
        xch+=t
        # --- rANS gap on base CTX9 streams ---
        for k in range(3):
            res=(chs[k]-NB[k]["pred"]["MED"]).astype(np.int32); ctx=NB[k]["ctx9"]
            for q in range(9):
                g=res[ctx==q]
                if g.size==0: continue
                _,cn=np.unique(g,return_counts=True)
                hd=A.huff_bits(cn.tolist()); Ac=len(cn)
                p=cn.astype(np.float64)/cn.sum(); ent=float(-(p*np.log2(p)).sum()*cn.sum())
                huff_data_tot+=hd; ent_tot+=ent; tbl_tot+=16+Ac*24
        print(f"{nm}: base9={t if False else base9/denom if denom else 0:.4f} (running) oracle_PN done, locobias running, al_CgfromCo={alq/16:+.4f}",flush=True)
    print("="*90)
    print(f"BASE MED-CTX9      ={base9/denom:.4f}")
    print(f"ORACLE perctx-pred ={orac9/denom:.4f}  d={orac9/denom-base9/denom:+.4f}")
    print(f"LOCO365 mean-bias  ={locobias9/denom:.4f}  d={locobias9/denom-base9/denom:+.4f}")
    for K in (9,18,27):
        print(f"CLUS-LOCO K={K}/ch  ={clus9n[K]/denom:.4f}  d={clus9n[K]/denom-base9/denom:+.4f}")
    print(f"XCH Cg-from-Co     ={xch/denom:.4f}  d={xch/denom-base9/denom:+.4f}")
    print(f"HUFF data={huff_data_tot/denom:.4f} ENT={ent_tot/denom:.4f} gap={huff_data_tot/denom-ent_tot/denom:.4f} tables={tbl_tot/denom:.4f}")

if __name__=="__main__":
    main()
