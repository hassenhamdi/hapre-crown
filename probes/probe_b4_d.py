"""probe_b4_d: clustering bake-off with HONEST side info + per-image reporting.
Tests (all MED residuals, YCoCg-R, champion borders):
 D1 adaptive meanabs-quantile K=6,9,12/ch (side: active_ctx*4b + 8b K header per ch)
 D2 deterministic magnitude tiers, zero side (tier=|q1|+|q2|+|q3| -> fixed bins)
 D3 sign handling: noflip/merged vs flip/merged vs nounmerge729
 D4 sort-key: meanabs vs entropy
Side conventions: map entry 4b/ctx (K<=16); thresholds 8b each.
"""
import numpy as np
from PIL import Image
import probe_b4_a as A
from probe_b4_b import nbhd, predictors, ctx9_from
from probe_b4_c import loco_ctx365, loco_q

IMAGES=A.IMAGES; HEADER=A.HEADER
CHAMP={"kodim01.png":3.56,"kodim02.png":3.23,"kodim05.png":3.84,"kodim07.png":3.01,
       "kodim13.png":4.17,"kodim19.png":3.41,"kodim23.png":3.03}

def group_bits(res, groups):
    t=0; d=0
    for gkeys_mask in groups:
        g=res[gkeys_mask]
        if g.size==0: continue
        tb,db,_=A.stream_bits(g.reshape(-1)); t+=tb; d+=db
    return t,d

def aquant_groups(key, res, K):
    uk,cn=np.unique(key,return_counts=True)
    ma=np.array([np.abs(res[key==v]).mean() for v in uk])
    order=np.argsort(ma,kind="stable")
    uks=uk[order]; cns=cn[order]
    tot=cns.sum(); tgt=tot/K; groups=[]; cur=[]; acc=0
    for u,c in zip(uks,cns):
        cur.append(u); acc+=c
        if acc>=tgt and len(groups)<K-1: groups.append(cur); cur=[]; acc=0
    groups.append(cur)
    masks=[np.isin(key,np.array(g)) for g in groups]
    side=len(uk)*4+8  # 4b per active ctx + header
    return masks,side

def entropy_of(v):
    _,cn=np.unique(v,return_counts=True)
    p=cn.astype(np.float64)/cn.sum()
    return float(-(p*np.log2(p)).sum())

def aquant_groups_ent(key,res,K):
    uk,cn=np.unique(key,return_counts=True)
    en=np.array([entropy_of(res[key==v]) for v in uk])
    order=np.argsort(en,kind="stable")
    uks=uk[order]; cns=cn[order]
    tot=cns.sum(); tgt=tot/K; groups=[]; cur=[]; acc=0
    for u,c in zip(uks,cns):
        cur.append(u); acc+=c
        if acc>=tgt and len(groups)<K-1: groups.append(cur); cur=[]; acc=0
    groups.append(cur)
    masks=[np.isin(key,np.array(g)) for g in groups]
    return masks,len(uk)*4+8

def tier_groups(a,b,c,d,K=9):
    q1=np.abs(loco_q(b-c)); q2=np.abs(loco_q(c-a)); q3=np.abs(loco_q(d-b))
    m=(q1+q2+q3).astype(np.int32)  # 0..12
    # fixed thresholds to 9 bins: [0,1,2,3,4,5,6,7,8+] edges
    edges=[1,2,3,4,5,6,7,8]
    t=np.digitize(m,edges)  # 0..8
    return [(t==v) for v in range(K)],0

def main():
    import collections
    agg=collections.defaultdict(int); dn_tot=0
    perimg={}
    for path in IMAGES:
        nm=path.split("/")[-1]
        img=np.array(Image.open(path).convert("RGB"))
        H,W,_=img.shape; dn=H*W*3; dn_tot+=dn
        Y,Co,Cg=A.ycocg_fwd(img); chs=[Y,Co,Cg]
        row={}
        # precompute neighbors + residuals + loco keys (+sign)
        DATA=[]
        for ch in chs:
            a,b,c,d,Ww,NNe,NE=nbhd(ch)
            mx=np.maximum(a,b); mn=np.minimum(a,b)
            med=np.where(c>=mx,mn,np.where(c<=mn,mx,a+b-c))
            res=(ch-med).astype(np.int32)
            key,s=loco_ctx365(a,b,c,d)
            DATA.append(dict(res=res,key=key,s=s,a=a,b=b,c=c,d=d))
        # V0 base energy ctx9
        t=HEADER
        for D in DATA:
            ctx=ctx9_from(D["a"],D["b"],D["c"])
            for q in range(9):
                g=D["res"][ctx==q]
                if g.size: tb,_,_=A.stream_bits(g.reshape(-1)); t+=tb
        row["base9"]=t/dn
        # V1 adaptive K=6,9,12 meanabs
        for K in (6,9,12):
            t=HEADER
            for D in DATA:
                m,side=aquant_groups(D["key"],D["res"],K)
                tt,_=group_bits(D["res"],m); t+=tt+side
            row[f"ad{K}"]=t/dn
        # V2 deterministic tiers
        t=HEADER
        for D in DATA:
            m,side=tier_groups(D["a"],D["b"],D["c"],D["d"])
            tt,_=group_bits(D["res"],m); t+=tt+side
        row["tier9"]=t/dn
        # V3 sign-flip + adaptive K=9
        t=HEADER
        for D in DATA:
            rf=(D["s"]*D["res"]).astype(np.int32)  # JPEG-LS style flip
            m,side=aquant_groups(D["key"],rf,9)
            tt,_=group_bits(rf,m); t+=tt+side
        row["flip9"]=t/dn
        # V4 unmerged729 adaptive K=9 (build unmerged key)
        t=HEADER
        for D in DATA:
            q1=loco_q(D["b"]-D["c"]).astype(np.int32); q2=loco_q(D["c"]-D["a"]).astype(np.int32); q3=loco_q(D["d"]-D["b"]).astype(np.int32)
            um=(q1+4)*81+(q2+4)*9+(q3+4)
            m,side=aquant_groups(um,D["res"],9)
            tt,_=group_bits(D["res"],m); t+=tt+side
        row["um9"]=t/dn
        # V5 entropy sort-key K=9
        t=HEADER
        for D in DATA:
            m,side=aquant_groups_ent(D["key"],D["res"],9)
            tt,_=group_bits(D["res"],m); t+=tt+side
        row["ent9"]=t/dn
        perimg[nm]=row
        for k,v in row.items(): agg[k]+=v*dn
        print(nm+" "+" ".join(f"{k}={v:.4f}" for k,v in row.items())+f" champ={CHAMP[nm]:.2f}",flush=True)
    print("="*110)
    for k in ["base9","ad6","ad9","ad12","tier9","flip9","um9","ent9"]:
        avg=agg[k]/dn_tot
        print(f"{k}: avg={avg:.4f} d_vs_base={avg-agg['base9']/dn_tot:+.4f} d_vs_champ={avg-3.4643:+.4f}")

if __name__=="__main__":
    main()
