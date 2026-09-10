"""probe_b6_b: stretch — deeper trees (<=16 leaves, depth<=5) + JXL-style histogram sharing.
Reuses probe_b6_ma machinery. Same unit/side rules. Sharing: same-predictor leaves
may share one Huffman histogram; per-leaf hist-id side = ceil(log2(nHist)) bits.
Greedy pairwise merge while net-positive (exact data+table math).
Also reports per-leaf predictor usage + leaf-count scaling.
"""
import sys, math, time
sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
import numpy as np
from PIL import Image
import probe_b4_a as A
from probe_b6_ma import (build_features, crown_baseline_channel, leaf_cost_from_vals,
    best_pred_cost, cand_thresholds, BANK5, SPLIT_SIDE, LEAF_PRED_SIDE, MARGINAL_SPLIT,
    MIN_LEAF, STRIDE, IMAGES, GIVEN_CROWN, GIVEN_JXLE3, HEADER, KSET2)
from probe_b4_c import loco_ctx365
from probe_b4_b import nbhd, predictors
import probe_b6_ma as M

def grow_tree_deep(Fs, Rmat_s, Ffull, Rmat_full, max_leaves=16, max_depth=5):
    N=Ffull.shape[0]; Ns=Fs.shape[0]
    full_idx=np.arange(N)
    pFull,_=best_pred_cost(Rmat_full,full_idx)
    leaves=[dict(fidx=full_idx,sidx=np.arange(Ns),depth=0,pcost_full=pFull)]
    def search_pcost(sidx):
        if sidx.size==0: return 0
        bc,_=best_pred_cost(Rmat_s,sidx)
        return bc
    struct=[]; nF=Fs.shape[1]
    while len(leaves)<max_leaves:
        best_gain_sub=-1; best=None
        for li,lf in enumerate(leaves):
            if lf["depth"]>=max_depth: continue
            if lf["fidx"].size<2*MIN_LEAF: continue
            sidx=lf["sidx"]
            if sidx.size<512: continue
            psub=search_pcost(sidx)
            Fnode=Fs[sidx]
            for f in range(nF):
                col=Fnode[:,f]
                if col.min()==col.max(): continue
                for thr in cand_thresholds(col):
                    m=col<=thr
                    nl=int(m.sum()); nr=int(m.size-nl)
                    if nl<256 or nr<256: continue
                    l_s=sidx[m]; r_s=sidx[~m]
                    lc,_=best_pred_cost(Rmat_s,l_s)
                    rc,_=best_pred_cost(Rmat_s,r_s)
                    g=psub-(lc+rc+MARGINAL_SPLIT)
                    if g>best_gain_sub:
                        best_gain_sub=g; best=(li,f,thr,l_s,r_s)
        if best is None or best_gain_sub<=0: break
        li,f,thr,l_s,r_s=best
        lf=leaves[li]
        colfull=Ffull[lf["fidx"],f]
        mfull=colfull<=thr
        l_f=lf["fidx"][mfull]; r_f=lf["fidx"][~mfull]
        if l_f.size<MIN_LEAF or r_f.size<MIN_LEAF:
            leaves[li]=dict(fidx=lf["fidx"],sidx=lf["sidx"],depth=max_depth,pcost_full=lf["pcost_full"])
            continue
        pF=lf["pcost_full"]
        lcF,_=best_pred_cost(Rmat_full,l_f)
        rcF,_=best_pred_cost(Rmat_full,r_f)
        if pF-(lcF+rcF+MARGINAL_SPLIT)<=0: break
        struct.append(dict(feat=f,thr=thr,gain_full=int(pF-(lcF+rcF+MARGINAL_SPLIT)),depth=int(lf["depth"]),
                           n=int(lf["fidx"].size),nl=int(l_f.size),nr=int(r_f.size)))
        leaves.pop(li)
        leaves.append(dict(fidx=l_f,sidx=l_s,depth=lf["depth"]+1,pcost_full=lcF))
        leaves.append(dict(fidx=r_f,sidx=r_s,depth=lf["depth"]+1,pcost_full=rcF))
    return leaves, struct

def shared_total(Rmat, leaf_idx_list, leaf_pred):
    """Greedy same-predictor histogram sharing. Returns (total_data_table, nHist, assign).
    total includes data+tables only (no tree/pred/hist-id side); caller adds side."""
    import math as _m
    # start: each leaf own histogram
    clusters=[dict(members=[i], pred=leaf_pred[i], idx=leaf_idx_list[i]) for i in range(len(leaf_idx_list))]
    def clust_cost(cl):
        p=cl["pred"]
        vals=np.concatenate([Rmat[leaf_idx_list[i],p] for i in cl["members"]])
        c,_,_=leaf_cost_from_vals(vals)
        return c
    costs=[clust_cost(c) for c in clusters]
    # greedy merge same-pred pairs
    while True:
        best_save=0; best_pair=None; best_cost=None
        for i in range(len(clusters)):
            for j in range(i+1,len(clusters)):
                if clusters[i]["pred"]!=clusters[j]["pred"]: continue
                vals=np.concatenate([Rmat[leaf_idx_list[k],clusters[i]["pred"]] for k in clusters[i]["members"]+clusters[j]["members"]])
                cc,_,_=leaf_cost_from_vals(vals)
                save=costs[i]+costs[j]-cc
                if save>best_save: best_save=save; best_pair=(i,j); best_cost=cc
        if best_pair is None or best_save<=0: break
        i,j=best_pair
        # hist-id side will be added later; require save > marginal hist-id cost?
        # approximate: merged adds ~ (members)*1b extra id side vs separate 0. Use exact global check below instead.
        # To keep honest, only merge if save exceeds added hist-id side upper bound (nLeaves*1b).
        # Defer: accept here, final total adds exact hist-id side; loop continues while data+table saves exist.
        new=dict(members=clusters[i]["members"]+clusters[j]["members"],pred=clusters[i]["pred"],idx=None)
        # remove j,i add new
        keep=[c for k,c in enumerate(clusters) if k!=i and k!=j]
        keep.append(new)
        clusters=keep
        costs=[clust_cost(c) for c in clusters]
    data_table=sum(costs)
    nH=len(clusters)
    nL=len(leaf_idx_list)
    hid_side = nL*math.ceil(math.log2(nH)) if nH>1 else 0
    return data_table+hid_side, nH, clusters, data_table

def eval_image(path, max_leaves=16, max_depth=5):
    img=np.array(Image.open(path).convert("RGB"))
    H,W,_=img.shape; dn=H*W*3
    Y,Co,Cg=A.ycocg_fwd(img); chs=[Y,Co,Cg]
    aY,bY,cY,dY,Ww,NNe,NE=nbhd(Y); PY=predictors(aY,bY,cY,dY,Ww,NNe,NE)
    Yres=(Y-PY["MED"]).astype(np.int32); Ymean=((aY+bY)//2).astype(np.int32)
    t_tree=HEADER; t_deep_share=HEADER; t_crown_share=HEADER; t_crown=HEADER
    info=[]
    for ci,ch in enumerate(chs):
        if ci==0: F,names,P,s,Rmat,aux=build_features(ch)
        else: F,names,P,s,Rmat,aux=build_features(ch,Yres_abs=np.abs(Yres),Yres=Yres,Ymean=Ymean)
        a,b,c,d=aux
        key,_=loco_ctx365(a,b,c,d)
        cb,bK,bG,bmap=crown_baseline_channel(key,Rmat)
        t_crown+=cb
        # crown groups -> sharing
        kf=key.reshape(-1)
        g_idx=[]; g_pred=[]
        for gkeys in bG:
            idx=np.flatnonzero(np.isin(kf,np.array(gkeys)))
            if idx.size==0: continue
            _,bp=best_pred_cost(Rmat,idx)
            g_idx.append(idx); g_pred.append(bp)
        if g_idx:
            sh_tot,nH,_,_=shared_total(Rmat,g_idx,g_pred)
            t_crown_share+=sh_tot+bmap+len(g_idx)*LEAF_PRED_SIDE
        else: t_crown_share+=bmap
        # deep tree
        N=H*W; Fs=F[::STRIDE]; Rmat_s=Rmat[::STRIDE]
        leaves,struct=grow_tree_deep(Fs,Rmat_s,F,Rmat,max_leaves,max_depth)
        leaf_idx=[lf["fidx"] for lf in leaves]
        leaf_pred=[best_pred_cost(Rmat,idx)[1] for idx in leaf_idx]
        plain=sum(best_pred_cost(Rmat,idx)[0] for idx in leaf_idx)+len(struct)*SPLIT_SIDE+len(leaves)*LEAF_PRED_SIDE
        t_tree+=plain
        sh_tot,nH,_,dt_only=shared_total(Rmat,leaf_idx,leaf_pred)
        shtot=dt_only+len(struct)*SPLIT_SIDE+len(leaves)*LEAF_PRED_SIDE+(len(leaves)*math.ceil(math.log2(nH)) if nH>1 else 0)
        t_deep_share+=shtot
        from collections import Counter
        pc=Counter(BANK5[p] for p in leaf_pred)
        info.append(dict(n_inner=len(struct),n_leaves=len(leaves),plain=plain,shared=shtot,
                         crown_plain=cb,pred_use=dict(pc),nH=nH,
                         struct=[(names[s["feat"]],s["thr"],s["gain_full"]) for s in struct]))
    return t_tree/dn,t_deep_share/dn,t_crown/dn,t_crown_share/dn,info

def main():
    print("== deep tree (<=16 leaves, d<=5) + sharing ==",flush=True)
    rows=[]
    for path in IMAGES:
        nm=path.split("/")[-1]
        t0=time.time()
        pt,ps,pc,pcs,info=eval_image(path)
        dt=time.time()-t0
        rows.append((nm,pt,ps,pc,pcs,info))
        print(f"{nm}: deep={pt:.4f} deep+sh={ps:.4f} crown5={pc:.4f} crown5+sh={pcs:.4f} d_deepsh-crownsh={ps-pcs:+.4f} d_deepsh-given={ps-GIVEN_CROWN[nm]:+.4f} [{dt:.0f}s]",flush=True)
        for ci,inf in enumerate(info):
            print(f"   ch{ci}: leaves={inf['n_leaves']} nH={inf['nH']} pred={inf['pred_use']} splits={inf['struct']}",flush=True)
    dnt=0;aT=aS=aC=aCS=aG=0
    for nm,pt,ps,pc,pcs,_ in rows:
        img=np.array(Image.open("/tmp/opencode/autocompress/experiments/real_photos/"+nm).convert("RGB"))
        H,W,_=img.shape; dn=H*W*3; dnt+=dn; aT+=pt*dn; aS+=ps*dn; aC+=pc*dn; aCS+=pcs*dn; aG+=GIVEN_CROWN[nm]*dn
    print(f"==> AVG deep={aT/dnt:.4f} deep+sh={aS/dnt:.4f} crown5={aC/dnt:.4f} crown5+sh={aCS/dnt:.4f} given={aG/dnt:.4f}",flush=True)
    print(f"==> d_deepsh-crownsh={aS/dnt-aCS/dnt:+.4f} d_deepsh-given={aS/dnt-aG/dnt:+.4f}",flush=True)
    import json
    json.dump(dict(rows=[dict(nm=r[0],deep=r[1],deep_sh=r[2],crown5=r[3],crown5_sh=r[4],info=r[5]) for r in rows],
              avg_deep=aT/dnt,avg_deep_sh=aS/dnt,avg_crown5=aC/dnt,avg_crown5_sh=aCS/dnt),
              open("/tmp/opencode/autocompress/experiments/probe_b6_b_summary.json","w"),indent=1)
    print("wrote probe_b6_b_summary.json",flush=True)

if __name__=="__main__":
    main()
