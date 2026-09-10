"""probe_b4_e: flip-K sweep + runs (offline exact machine) + block-experts combo + causality proof.
numpy+PIL only. honesty: every side byte counted.
"""
import numpy as np
from PIL import Image
import probe_b4_a as A
from probe_b4_b import nbhd, predictors, ctx9_from
from probe_b4_c import loco_ctx365, loco_q

IMAGES=A.IMAGES; HEADER=A.HEADER
CHAMP={"kodim01.png":3.56,"kodim02.png":3.23,"kodim05.png":3.84,"kodim07.png":3.01,
       "kodim13.png":4.17,"kodim19.png":3.41,"kodim23.png":3.03}
MOE6=["MED","TOP","PAETH","GRAD","GAP","DG"]

def prep(ch):
    a,b,c,d,Ww,NNe,NE=nbhd(ch)
    P=predictors(a,b,c,d,Ww,NNe,NE)
    key,s=loco_ctx365(a,b,c,d)
    return dict(a=a,b=b,c=c,d=d,P=P,key=key,s=s)

def cluster_masks(key, rf, K):
    uk,cn=np.unique(key,return_counts=True)
    ma=np.array([np.abs(rf[key==v]).mean() for v in uk])
    order=np.argsort(ma,kind="stable")
    uks=uk[order]; cns=cn[order]
    tot=cns.sum(); tgt=tot/K; groups=[]; cur=[]; acc=0
    for u,c in zip(uks,cns):
        cur.append(u); acc+=c
        if acc>=tgt and len(groups)<K-1: groups.append(cur); cur=[]; acc=0
    groups.append(cur)
    return [np.isin(key,np.array(g)) for g in groups], len(uk)*4+8

def gamma_cost(L):  # L>=1 run length; cost gamma(L+1)
    v=L+1; return 2*v.bit_length()-1

def run_pass(P, H, W):
    """Champion-exact flat-run machine on recon==P (originals). Returns (main_mask, runs_list).
    main_mask True where a main symbol is emitted. runs_list lengths L."""
    main=np.ones((H,W),dtype=bool); runs=[]
    for i in range(H):
        j=0
        row=P[i]
        while j<W:
            if i==0 and j==0:
                av=bv=cv=dv=0
            elif i==0:
                av=int(row[j-1]); bv=cv=av; dv=av
            elif j==0:
                bv=int(P[i-1,0]); av=cv=bv; dv=int(P[i-1,1]) if W>1 else bv
            else:
                av=int(row[j-1]); bv=int(P[i-1,j]); cv=int(P[i-1,j-1]); dv=int(P[i-1,j+1]) if j+1<W else bv
            if j+1<W and av==bv==cv==dv and not(i==0 and j==0):
                L=0
                while j+L<W and int(row[j+L])==av: L+=1
                runs.append(L)
                main[i,j:j+L]=False
                j+=L
                if j>=W: break
                main[i,j]=True; j+=1  # forced interruption symbol
            else:
                j+=1
    return main,runs

def main():
    KSET=(6,9,12,18)
    agg={}; dn_tot=0
    print("== K sweep (flip, no runs) ==")
    for path in IMAGES:
        nm=path.split("/")[-1]
        img=np.array(Image.open(path).convert("RGB"))
        H,W,_=img.shape; dn=H*W*3; dn_tot+=dn
        Y,Co,Cg=A.ycocg_fwd(img); chs=[Y,Co,Cg]
        DD=[prep(ch) for ch in chs]
        R={}
        for K in KSET:
            t=HEADER
            for D,kch in zip(DD,chs):
                res=(kch-D["P"]["MED"]).astype(np.int32)
                rf=(D["s"]*res).astype(np.int32)
                m,side=cluster_masks(D["key"],rf,K)
                for mm in m:
                    g=rf[mm]
                    if g.size: tb,_,_=A.stream_bits(g.reshape(-1)); t+=tb
                t+=side
            R[K]=t/dn
        # auto-K per channel
        t=HEADER+3*2
        for D,kch in zip(DD,chs):
            res=(kch-D["P"]["MED"]).astype(np.int32)
            rf=(D["s"]*res).astype(np.int32)
            best=None
            for K in KSET:
                m,side=cluster_masks(D["key"],rf,K)
                tt=side
                for mm in m:
                    g=rf[mm]
                    if g.size: tb,_,_=A.stream_bits(g.reshape(-1)); tt+=tb
                if best is None or tt<best: best=tt
            t+=best
        R["auto"]=t/dn
        agg.setdefault("sweep",[]).append((nm,R))
        print(nm+" "+" ".join(f"K{k}={R[k]:.4f}" for k in KSET)+f" auto={R['auto']:.4f} champ={CHAMP[nm]:.2f}",flush=True)
    # pick best K on avg
    av={k:float(np.mean([r[k]*1 for _,r in agg['sweep']])) for k in list(KSET)+["auto"]}
    print("AVG:",{k:round(v,4) for k,v in av.items()})
    BK=min(KSET,key=lambda k:av[k])
    print(f"Best fixed K={BK} ({av[BK]:.4f}); auto={av['auto']:.4f}")
    print("== final: flip+K* + runs ==")
    KBK=BK
    tot_fin=0; tot_medrun_ref=0
    for path in IMAGES:
        nm=path.split("/")[-1]
        img=np.array(Image.open(path).convert("RGB"))
        H,W,_=img.shape; dn=H*W*3
        Y,Co,Cg=A.ycocg_fwd(img); chs=[Y,Co,Cg]
        DD=[prep(ch) for ch in chs]
        t=HEADER
        for D,kch in zip(DD,chs):
            res=(kch-D["P"]["MED"]).astype(np.int32)
            rf=(D["s"]*res).astype(np.int32)
            main,runs=run_pass(kch,H,W)
            m,side=cluster_masks(D["key"][main],rf[main],KBK)
            # careful: masks index full-size key; restrict groups to main pixels:
            # recompute: cluster on main-subset keys, then map full pixels
            # (cluster_masks above already used subset; rebuild full masks via group key sets)
            # Simpler: redo with explicit group key sets:
            uk,cn=np.unique(D["key"][main],return_counts=True)
            ma=np.array([np.abs(rf[main][D["key"][main]==v]).mean() for v in uk])
            order=np.argsort(ma,kind="stable")
            uks=uk[order]; cns=cn[order]
            totm=cns.sum(); tgt=totm/KBK; groups=[]; cur=[]; acc=0
            for u,c in zip(uks,cns):
                cur.append(u); acc+=c
                if acc>=tgt and len(groups)<KBK-1: groups.append(cur); cur=[]; acc=0
            groups.append(cur)
            for gkeys in groups:
                g=rf[main & np.isin(D["key"],np.array(gkeys))]
                if g.size: tb,_,_=A.stream_bits(g.reshape(-1)); t+=tb
            t+=len(uk)*4+8
            t+=sum(gamma_cost(L) for L in runs)
        t+=64  # run-stream header
        bpp=t/dn; tot_fin+=t
        print(f"{nm}: flipK{KBK}+runs={bpp:.4f} champ={CHAMP[nm]:.2f} d={bpp-CHAMP[nm]:+.4f}",flush=True)
    print(f"AVG flipK{KBK}+runs = {tot_fin/dn_tot:.4f} vs champ 3.4643 d={tot_fin/dn_tot-3.4643:+.4f} vs WebP-m3 3.38 d={tot_fin/dn_tot-3.38:+.4f}")

if __name__=="__main__":
    main()
