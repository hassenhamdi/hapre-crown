"""probe_b6_ma: MA-tree-lite — learned signaled per-image decision tree over context space.
Per channel: causal integer features, greedy depth<=3 / <=8 leaves CART with EXACT
Huffman-bit objective, per-leaf best predictor, honest side info.
numpy+PIL only. No existing files modified.

Decoder-safety: every split test uses causally-available recon values only
(neighbor recon a,b,c,d + row/col position + already-decoded Y plane for chroma).
Lossless recon==orig so encoder features == decoder features under identical
border convention (champion 0/left/top rule via imported nbhd()).

Unit: bpp = total_bits/(H*W*3). Tables per leaf stream: 16+A*24 + Huffman data.
Tree side: 24b per inner node (8b feature id + 16b threshold) + 3b per leaf predictor id.
Global header 64b once.
"""
import sys, math, time, heapq
sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
import numpy as np
from PIL import Image
import probe_b4_a as A
from probe_b4_b import nbhd, predictors
from probe_b4_c import loco_ctx365

IMAGES = [
    "/tmp/opencode/autocompress/experiments/real_photos/kodim01.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim02.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim05.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim07.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim13.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim19.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim23.png",
]
HEADER = 64
BANK5 = ["MED", "TOP", "PAETH", "GRAD", "GAP"]
# given external anchors
GIVEN_CROWN = {"kodim01.png":3.4340,"kodim02.png":3.0931,"kodim05.png":3.7394,
               "kodim07.png":2.8630,"kodim13.png":4.1010,"kodim19.png":3.2758,"kodim23.png":2.8940}
GIVEN_JXLE3 = {"kodim01.png":3.36,"kodim02.png":3.06,"kodim05.png":3.51,
               "kodim07.png":2.73,"kodim13.png":3.91,"kodim19.png":3.22,"kodim23.png":2.81}
KSET2 = (2,3,4,6,9,12,18,27,36)

SPLIT_SIDE = 24      # per inner node: 8b feat + 16b thr
LEAF_PRED_SIDE = 3   # per leaf predictor id (ceil(log2(5))=3)
MARGINAL_SPLIT = SPLIT_SIDE + LEAF_PRED_SIDE  # +3 net (1 leaf->2 leaves)
MIN_LEAF = 2048
STRIDE = 4           # search subsample stride for speed; final accounting exact on full
MAX_DEPTH = 3
MAX_LEAVES = 8

OFFSET = 2048
SPAN = 4096

def huff_bits_counts(counts):
    if len(counts)==0: return 0
    if len(counts)==1: return int(counts[0])*1
    h=list(map(int,counts)); heapq.heapify(h); t=0
    while len(h)>1:
        x=heapq.heappop(h); y=heapq.heappop(h); s=x+y; t+=s; heapq.heappush(h,s)
    return t

def leaf_cost_from_vals(vals):
    """exact cost data+table for one leaf residual vector (1-D int array). Returns (total, data, A)."""
    if vals.size==0:
        return 0,0,0
    v = np.asarray(vals).reshape(-1)
    # shift + bincount fast
    s = v + OFFSET
    # residuals occasionally outside +-2048? clip check -> fall back to unique
    if s.min() < 0 or s.max() >= SPAN:
        uv,cn = np.unique(v, return_counts=True)
        A_ = len(cn); d = huff_bits_counts(cn.tolist())
        return d + 16 + A_*24, d, A_
    cn = np.bincount(s, minlength=SPAN)
    nz = cn[cn>0]
    A_ = int(nz.size)
    d = huff_bits_counts(nz.tolist())
    return d + 16 + A_*24, d, A_

def best_pred_cost(Rmat, idx):
    """Rmat: (N,nP) flipped residuals int32. idx: 1-D index array. Returns (bestCost,bestP)."""
    best=None; bp=0
    for p in range(Rmat.shape[1]):
        c,_,_=leaf_cost_from_vals(Rmat[idx,p])
        if best is None or c<best: best=c; bp=p
    return best,bp

def build_features(ch, Yres_abs=None, Yres=None, Ymean=None):
    """All causal integer features. ch: HxW int32 (recon==orig). Returns F (N,nF), names, plus P dict, s, Rmat."""
    H,W = ch.shape
    a,b,c,d,Ww,NNe,NE = nbhd(ch)
    P = predictors(a,b,c,d,Ww,NNe,NE)
    key,s = loco_ctx365(a,b,c,d)
    f0 = np.abs(b-c)
    f1 = np.abs(a-c)
    f2 = np.abs(d-b)
    f3 = f0+f1
    f4 = np.minimum((f0+f1)>>4,8)
    f5 = np.abs(a-b)
    # edge strength |a+b-2c|//2
    f6 = np.abs(a+b-2*c)//2
    # neighbor mean level (causal: from recon neighbors)
    f7 = (a+b)//2
    # position
    ii = np.broadcast_to(np.arange(H,dtype=np.int32)[:,None],(H,W))
    jj = np.broadcast_to(np.arange(W,dtype=np.int32)[None,:],(H,W))
    f8 = ii; f9 = jj
    feats = [f0,f1,f2,f3,f4,f5,f6,f7,f8,f9]
    names = ["|b-c|","|a-c|","|d-b|","|b-c|+|a-c|","e>>4cap8","|a-b|","|a+b-2c|/2","(a+b)/2","row","col"]
    if Yres_abs is not None:
        feats += [Yres_abs, Yres, Ymean]
        names += ["|Yres|","Yres","Ymean"]
    F = np.stack([f.reshape(-1) for f in feats],axis=1).astype(np.int32)
    # residual matrix flipped by s
    N = H*W
    sv = s.reshape(-1).astype(np.int32)
    Rmat = np.stack([(ch.reshape(-1)-P[n].reshape(-1)).astype(np.int32)*sv if False else ((ch-P[n]).astype(np.int32).reshape(-1)*sv) for n in BANK5],axis=1)
    return F, names, P, s, Rmat, (a,b,c,d)

def crown_baseline_channel(key, Rmat, KSET=KSET2):
    """CROWN-huff-style 1-D quantile clustering on LOCO key sorted by mean|rf(MED)|.
    Uses same BANK5 + same table accounting for fair comparison.
    key: HxW int32 LOCO key; Rmat: (N,5) flipped residuals (MED is col0)."""
    N = Rmat.shape[0]
    kf = key.reshape(-1)
    rfM = Rmat[:,0]
    uk,cn = np.unique(kf,return_counts=True)
    # mean abs per key using MED flipped residual
    ma = np.array([np.abs(rfM[kf==v]).mean() for v in uk])
    order = np.argsort(ma,kind="stable")
    uks=uk[order]; cns=cn[order]
    best=None
    for K in KSET:
        tot=cns.sum(); tgt=tot/K; groups=[]; cur=[]; acc=0
        for u,c in zip(uks,cns):
            cur.append(u); acc+=c
            if acc>=tgt and len(groups)<K-1: groups.append(cur); cur=[]; acc=0
        groups.append(cur)
        mapbits=len(uk)*math.ceil(math.log2(K))+4+8
        tt=mapbits+len(groups)*LEAF_PRED_SIDE
        for gkeys in groups:
            sel=np.isin(kf,np.array(gkeys))
            idx=np.flatnonzero(sel)
            if idx.size==0: continue
            bc,_=best_pred_cost(Rmat,idx)
            tt+=bc
        if best is None or tt<best: best=tt; bK=K; bG=groups; bmap=mapbits
    return best,bK,bG,bmap

def cand_thresholds(vals):
    u=np.unique(vals)
    if len(u)<=1: return []
    if len(u)<=16:
        return [int(x) for x in u[:-1]]
    qs=np.percentile(vals,[10,20,30,40,50,60,70,80,90])
    th=sorted(set(int(q) for q in qs))
    th=[t for t in th if t>=int(u.min()) and t<int(u.max())]
    return th

def grow_tree(Fs, Rmat_s, Ffull, Rmat_full):
    """Greedy best-gain-first growth.
    Fs/Rmat_s: subsampled search arrays (Ns,). Ffull/Rmat_full: full (N,).
    Returns leaves: list of dict(idx_full, depth), inners count, structure list.
    idx arrays are index-into-full? We keep separate: search indices + full indices.
    Simplest: nodes hold full idx + search idx (search = subset of full via stride mask)."""
    N=Ffull.shape[0]; Ns=Fs.shape[0]
    # map: search row k corresponds to full index k*STRIDE? Only if we subsample by arange(0,N,STRIDE).
    # Build that way in caller.
    full_idx=np.arange(N)
    search_idx=np.arange(0,N,STRIDE)
    # parent full cost
    pFull,_=best_pred_cost(Rmat_full,full_idx)
    # node list: each node dict with fidx (full indices), sidx (search indices), depth
    # search indices are positions into Fs (0..Ns-1) corresponding to full search_idx
    leaves=[dict(fidx=full_idx,sidx=np.arange(Ns),depth=0,pcost_full=pFull)]
    # cache parent search cost per leaf for ranking
    def search_pcost(sidx):
        if sidx.size==0: return 0
        bc,_=best_pred_cost(Rmat_s,sidx)
        # scale to full? No — rank on subsample scale directly (relative gains comparable)
        return bc
    struct=[]
    nF=Fs.shape[1]
    while len(leaves)<MAX_LEAVES:
        best_gain_sub=-1; best=None  # (leaf_pos, f, thr, l_s, r_s)
        for li,lf in enumerate(leaves):
            if lf["depth"]>=MAX_DEPTH: continue
            if lf["fidx"].size<2*MIN_LEAF: continue
            sidx=lf["sidx"]
            if sidx.size<512: continue
            psub=search_pcost(sidx)
            Fnode=Fs[sidx]
            for f in range(nF):
                col=Fnode[:,f]
                # skip constant
                if col.min()==col.max(): continue
                # skip Y-features that are all zero (Y channel): min==max==0 handled above
                for thr in cand_thresholds(col):
                    m=col<=thr
                    nl=int(m.sum()); nr=int(m.size-nl)
                    if nl<256 or nr<256: continue
                    # min leaf on full scale approx: require scaled
                    # (skip strict check here; verify on full later)
                    l_s=sidx[m]; r_s=sidx[~m]
                    lc,_=best_pred_cost(Rmat_s,l_s)
                    rc,_=best_pred_cost(Rmat_s,r_s)
                    g=psub-(lc+rc+MARGINAL_SPLIT)
                    if g>best_gain_sub:
                        best_gain_sub=g; best=(li,f,thr,l_s,r_s)
        if best is None or best_gain_sub<=0:
            break
        # verify exact gain on full before accepting
        li,f,thr,l_s,r_s=best
        lf=leaves[li]
        colfull=Ffull[lf["fidx"],f]
        mfull=colfull<=thr
        l_f=lf["fidx"][mfull]; r_f=lf["fidx"][~mfull]
        if l_f.size<MIN_LEAF or r_f.size<MIN_LEAF:
            # mark this leaf exhausted to avoid infinite loop: bump depth to block
            # instead, forbid this split by breaking? simplest: block leaf
            leaves[li]=dict(fidx=lf["fidx"],sidx=lf["sidx"],depth=MAX_DEPTH,pcost_full=lf["pcost_full"])
            continue
        pFull=lf["pcost_full"]
        lcF,_=best_pred_cost(Rmat_full,l_f)
        rcF,_=best_pred_cost(Rmat_full,r_f)
        gain_full=pFull-(lcF+rcF+MARGINAL_SPLIT)
        if gain_full<=0:
            # no exact-paying split found from best subsample candidate; stop (conservative)
            break
        # accept split
        struct.append(dict(feat=f,thr=thr,gain_sub=int(best_gain_sub),gain_full=int(gain_full),
                           n=int(lf["fidx"].size),nl=int(l_f.size),nr=int(r_f.size),depth=int(lf["depth"])))
        lcF2,_=best_pred_cost(Rmat_full,l_f)  # recompute not needed
        # build new leaves; search sidx split must be consistent: l_s,r_s already
        newl=dict(fidx=l_f,sidx=l_s,depth=lf["depth"]+1,pcost_full=lcF)
        newr=dict(fidx=r_f,sidx=r_s,depth=lf["depth"]+1,pcost_full=rcF)
        leaves.pop(li)
        leaves.append(newl); leaves.append(newr)
    return leaves, struct

def eval_image_tree(path, verbose=False):
    img=np.array(Image.open(path).convert("RGB"))
    H,W,_=img.shape; dn=H*W*3
    Y,Co,Cg=A.ycocg_fwd(img)
    chs=[Y,Co,Cg]
    # Y features first (need Yres for chroma)
    # compute Y MED residual for chroma conditioning: MED of Y via nbhd
    aY,bY,cY,dY,Ww,NNe,NE = nbhd(Y)
    PY = predictors(aY,bY,cY,dY,Ww,NNe,NE)
    Yres = (Y-PY["MED"]).astype(np.int32)
    Ymean = ((aY+bY)//2).astype(np.int32)
    total_tree=HEADER; total_crown=HEADER
    per_ch=[]
    for ci,ch in enumerate(chs):
        if ci==0:
            F,names,P,s,Rmat,aux = build_features(ch)
        else:
            F,names,P,s,Rmat,aux = build_features(ch, Yres_abs=np.abs(Yres), Yres=Yres, Ymean=Ymean)
        N=H*W
        key,_=loco_ctx365(*aux[:4]) if False else (None,None)
        # recompute key for crown baseline
        a,b,c,d = aux
        key,s2 = loco_ctx365(a,b,c,d)
        # crown baseline (same bank5)
        cb,bK,bG,bmap = crown_baseline_channel(key,Rmat)
        total_crown+=cb
        # tree
        Ns=(N+STRIDE-1)//STRIDE
        Fs=F[::STRIDE]
        Rmat_s=Rmat[::STRIDE]
        t0=time.time()
        leaves,struct=grow_tree(Fs,Rmat_s,F,Rmat)
        # exact leaf totals + predictor ids
        leaf_bits=0; leaf_detail=[]
        for lf in leaves:
            bc,bp=best_pred_cost(Rmat,lf["fidx"])
            leaf_bits+=bc
            # which predictor won (name)
            leaf_detail.append((int(lf["fidx"].size),BANK5[bp],int(bc)))
        n_inner=len(struct); n_leaves=len(leaves)
        ch_tree=leaf_bits+n_inner*SPLIT_SIDE+n_leaves*LEAF_PRED_SIDE
        total_tree+=ch_tree
        per_ch.append(dict(names=names,struct=[dict(feat=names[s["feat"]],thr=s["thr"],
                      gain_full=s["gain_full"],depth=s["depth"],n=s["n"],nl=s["nl"],nr=s["nr"]) for s in struct],
                      leaves=leaf_detail,n_inner=n_inner,n_leaves=n_leaves,
                      ch_tree=ch_tree,ch_crown=cb,K=bK,ng=len(bG)))
        if verbose:
            print(f"  ch{ci}: crownK={bK} ng={len(bG)} cbits={cb} | tree inl={n_inner} lv={n_leaves} tbits={ch_tree} d={ch_tree-cb:+.0f} struct={per_ch[-1]['struct']}",flush=True)
    return total_tree/dn, total_crown/dn, per_ch

def anchor_med_order0():
    tot=0; dnt=0; rows=[]
    for path in IMAGES:
        img=np.array(Image.open(path).convert("RGB"))
        H,W,_=img.shape; dn=H*W*3; dnt+=dn
        Y,Co,Cg=A.ycocg_fwd(img)
        t=HEADER
        for ch in (Y,Co,Cg):
            _,res,_=A.med_pred_ctx9(ch)
            tb,_,_=A.stream_bits(res.reshape(-1)); t+=tb
        rows.append((path.split("/")[-1],t/dn)); tot+=t
    return tot/dnt, rows

def main():
    print("== anchor MED order-0 ==")
    avg,rows=anchor_med_order0()
    for nm,v in rows: print(f"  {nm}: {v:.4f}")
    print(f"  AVG={avg:.4f} (expect ~3.58, tol 3% [{3.58*0.97:.4f},{3.58*1.03:.4f}]) {'PASS' if 3.58*0.97<=avg<=3.58*1.03 else 'FAIL'}",flush=True)
    print("== MA-tree-lite vs CROWN-huff(5pred, same accounting) ==",flush=True)
    trows=[]
    for path in IMAGES:
        nm=path.split("/")[-1]
        t0=time.time()
        bppT,bppC,per_ch=eval_image_tree(path,verbose=True)
        dt=t0 and (time.time()-t0)
        trows.append((nm,bppT,bppC,per_ch,dt))
        print(f"{nm}: TREE={bppT:.4f} CROWN5={bppC:.4f} d_tree-crown5={bppT-bppC:+.4f} d_tree-given335={bppT-GIVEN_CROWN[nm]:+.4f} d_tree-jxle3={bppT-GIVEN_JXLE3[nm]:+.4f} [{dt:.0f}s]",flush=True)
    dn_tot=0; tT=0; tC=0; tG=0; tJ=0
    for nm,bppT,bppC,_,_ in trows:
        img=np.array(Image.open(nm and ("/tmp/opencode/autocompress/experiments/real_photos/"+nm)).convert("RGB"))
        H,W,_=img.shape; dn=H*W*3; dn_tot+=dn; tT+=bppT*dn; tC+=bppC*dn; tG+=GIVEN_CROWN[nm]*dn; tJ+=GIVEN_JXLE3[nm]*dn
    print(f"==> AVG TREE={tT/dn_tot:.4f} CROWN5={tC/dn_tot:.4f} GIVEN-CROWN={tG/dn_tot:.4f} JXLE3={tJ/dn_tot:.4f}",flush=True)
    print(f"==> d_tree-crown5={tT/dn_tot-tC/dn_tot:+.4f} d_tree-given={tT/dn_tot-tG/dn_tot:+.4f} d_tree-jxle3={tT/dn_tot-tJ/dn_tot:+.4f}",flush=True)
    # dump machine-readable summary for report writer
    import json
    out=dict(rows=[dict(nm=nm,tree=bppT,crown5=bppC,given=GIVEN_CROWN[nm],jxle3=GIVEN_JXLE3[nm],
               per_ch=per_ch) for nm,bppT,bppC,per_ch,_ in trows],
               avg_tree=tT/dn_tot,avg_crown5=tC/dn_tot,avg_given=tG/dn_tot,avg_jxle3=tJ/dn_tot,
               anchor=avg)
    open("/tmp/opencode/autocompress/experiments/probe_b6_summary.json","w").write(json.dumps(out,indent=1))
    print("wrote probe_b6_summary.json",flush=True)

if __name__=="__main__":
    main()
