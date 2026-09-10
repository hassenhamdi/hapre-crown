"""probe_b5_d: COMBINED best — E16 bank + per-group Huff-vs-Golomb (2-way) + rANS 3-way refinement.
Wide KSET incl 48,64. Exact sides: map active*ceil(log2K)+4+8/ch, predid ceil(log2E)/group,
backend choice 1b (2-way) / 2b (3-way), Golomb k 4b/group, Huff 16+A*24, rANS 16+A*32+payload (decode-verified).
Also proofs: huff roundtrip every stream, rANS decode every rANS stream, recon simulation.
"""
import sys, math, time
import numpy as np
from PIL import Image
sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
import probe_b4_a as A
from probe_b4_b import nbhd
from probe_b4_c import loco_ctx365
from probe_b4_f import huff_roundtrip
from probe_b4_h import rans_stream_bits
from probe_b5_c import predictors_X, prepX, E16, E12, MOE6

IMAGES = [
    "/tmp/opencode/autocompress/experiments/real_photos/kodim01.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim02.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim05.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim07.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim13.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim19.png",
    "/tmp/opencode/autocompress/experiments/real_photos/kodim23.png",
]
HEADER = A.HEADER
KSET_WIDE = (2,3,4,6,9,12,18,27,36,48,64)
KSET2 = (2,3,4,6,9,12,18,27,36)

def golomb_best(g):
    g = np.asarray(g).reshape(-1).astype(np.int64)
    N = g.size
    if N == 0: return 0, 0
    M = np.where(g >= 0, 2*g, -2*g-1).astype(np.int64)
    best = None
    for k in range(13):
        tot = int(np.sum(M >> np.int64(k)) + N*(1+k))
        if best is None or tot < best[0]: best = (tot, k)
    return best

def huff_total(g):
    uv, cn = np.unique(g, return_counts=True)
    d = A.huff_bits(cn.tolist())
    return d + 16 + len(cn)*24, d, len(cn)

def eval_combined(path, bank, KSET, use_rans_refine=False):
    """Returns (bpp2way, detail2, bpp3way, detail3, streams_info).
    streams_info: per-channel list of (groups, gpred, gback, gk, gvals) for proofs/recon.
    2-way: Huff vs Golomb (1b choice). 3-way: refine 2-way winner's expert with rANS (2b choice)."""
    pred_bits = math.ceil(math.log2(len(bank)))
    img = np.array(Image.open(path).convert("RGB"))
    H,W,_ = img.shape; dn = H*W*3
    Y,Co,Cg = A.ycocg_fwd(img); chs=[Y,Co,Cg]
    DD=[prepX(ch) for ch in chs]
    t2 = HEADER
    ch_info = []  # for 2-way best
    for D,kch in zip(DD,chs):
        P=D["P"]; key=D["key"]; s=D["s"]
        RF={n:(s*(kch-P[n]).astype(np.int32)).astype(np.int32) for n in bank}
        rfM = RF["MED"]
        best=None
        for K in KSET:
            uk,cn=np.unique(key,return_counts=True)
            ma=np.array([np.abs(rfM[key==v]).mean() for v in uk])
            order=np.argsort(ma,kind="stable")
            uks=uk[order]; cns=cn[order]
            tot=cns.sum(); tgt=tot/K; groups=[]; cur=[]; acc=0
            for u,c in zip(uks,cns):
                cur.append(u); acc+=c
                if acc>=tgt and len(groups)<K-1: groups.append(cur); cur=[]; acc=0
            groups.append(cur)
            mapbits=len(uk)*math.ceil(math.log2(K))+4+8
            tt=mapbits+len(groups)*(pred_bits+1)
            win=[]  # per-group (expert, backend, k, cost)
            for gkeys in groups:
                sel=np.isin(key,np.array(gkeys))
                b=None
                for n in bank:
                    g=RF[n][sel]
                    if g.size==0: ht=0; gt=0; gk=0
                    else:
                        ht,_,_=huff_total(g.reshape(-1)); gd,gk=golomb_best(g); gt=gd+4
                    if b is None or ht<b[0]: b=(ht,n,"H",gk)
                    if gt<b[0]: b=(gt,n,"G",gk)
                tt+=b[0]; win.append(b)
            if best is None or tt<best[0]: best=(tt,K,groups,win)
        t2+=best[0]
        ch_info.append(dict(K=best[1],groups=best[2],win=best[3],key=key,s=s,P=P,ch=kch,RF=RF))
    bpp2 = t2/dn
    if not use_rans_refine:
        return bpp2, [(c["K"],len(c["groups"])) for c in ch_info], None, None, ch_info
    # 3-way refinement: for each group, compute rANS bits for winning expert only
    t3 = HEADER
    ch_info3=[]
    for c in ch_info:
        K=c["K"]; groups=c["groups"]; key=c["key"]
        mapbits=len(np.unique(key))*math.ceil(math.log2(K))+4+8
        # choice now 2b/group
        tt=mapbits+len(groups)*(pred_bits+2)
        win3=[]
        for gkeys,w in zip(groups,c["win"]):
            _,bn,bb,gk = w
            g=c["RF"][bn][np.isin(key,np.array(gkeys))]
            if g.size==0:
                win3.append((0,bn,"H",0,None)); continue
            ht,_,_=huff_total(g.reshape(-1)); gd,gk2=golomb_best(g); gt=gd+4
            # rANS exact (decode-verified)
            rt,_=rans_stream_bits(g)
            huff_roundtrip(g.reshape(-1))
            costs={"H":ht,"G":gt,"R":rt}
            b=min(costs,key=lambda k:costs[k])
            tt+=costs[b]
            win3.append((costs[b],bn,b,gk2 if b=="G" else 0, g if True else None))
        t3+=tt
        ch_info3.append(dict(K=K,groups=groups,win=win3,key=c["key"],s=c["s"],P=c["P"],ch=c["ch"],RF=c["RF"]))
    bpp3=t3/dn
    det2=[(c["K"],len(c["groups"])) for c in ch_info]
    det3=[(c["K"],len(c["groups"])) for c in ch_info3]
    return bpp2, det2, bpp3, det3, ch_info3

def prove_recon(path, ch_info):
    """Scalar decoder simulation for E16 bank + backend choice. Re-derives key/sign/map/pred from recon."""
    img=np.array(Image.open(path).convert("RGB"))
    H,W,_=img.shape
    Y,Co,Cg=A.ycocg_fwd(img); chs=[Y,Co,Cg]
    for ci,(kch,c) in enumerate(zip(chs,ch_info)):
        key=c["key"]; s=c["s"]; groups=c["groups"]; win=c["win"]
        # win entries: (cost,bn,bb,gk[,g]) — extract bn
        gpred=[w[1] for w in win]
        k2g={}
        for gi,gkeys in enumerate(groups):
            for v in gkeys: k2g[int(v)]=gi
        # build per-group streams in pixel order (row-major main==all)
        # encoder streams: per group values of RF[bn][sel] in row-major order
        # decoder walks pixels row-major, picks group, takes next symbol of that group's stream
        # verify multiset + order: since both row-major, order matches if we slice in same order
        streams=[]
        for gi,gkeys in enumerate(groups):
            bn=gpred[gi]
            sel=np.isin(key,np.array(gkeys))
            # row-major order values
            g=c["RF"][bn][sel]  # np boolean indexing is row-major order — matches decode walk
            streams.append(np.asarray(g).reshape(-1))
            huff_roundtrip(np.asarray(g).reshape(-1))
        curs=[0]*len(groups)
        recon=np.zeros_like(kch)
        for i in range(H):
            for j in range(W):
                if i==0 and j==0: av=bv=cv=dv=0
                elif i==0: av=int(recon[i,j-1]); bv=cv=av; dv=av
                elif j==0: bv=int(recon[i-1,0]); av=cv=bv; dv=int(recon[i-1,1]) if W>1 else bv
                else: av=int(recon[i,j-1]); bv=int(recon[i-1,j]); cv=int(recon[i-1,j-1]); dv=int(recon[i-1,j+1]) if j+1<W else bv
                Ww=int(recon[i,j-2]) if j>=2 else av
                NNe=int(recon[i-2,j]) if i>=2 else bv
                NE=int(recon[i-1,j+1]) if (i>=1 and j+1<W) else bv
                if i==0: Ww=av; NNe=av; NE=av
                q1=bv-cv; q2=cv-av; q3=dv-bv
                def Q(gg):
                    ag=abs(gg)
                    return 0 if gg==0 else (1 if ag<=2 else (2 if ag<=7 else (3 if ag<=21 else 4)))*(1 if gg>0 else -1)
                qq=(Q(q1),Q(q2),Q(q3))
                neg=qq[0]<0 or (qq[0]==0 and qq[1]<0) or (qq[0]==0 and qq[1]==0 and qq[2]<0)
                kk=tuple(-v for v in qq) if neg else qq
                kid=kk[0]*81+kk[1]*9+kk[2]
                assert kid==int(key[i,j]) and (1 if not neg else -1)==int(s[i,j]), f"ctx mismatch {(i,j)}"
                gi=k2g[kid]; bn=gpred[gi]
                if bn=="MED":
                    mx=av if av>bv else bv; mn=bv if av>bv else av
                    p=mn if cv>=mx else (mx if cv<=mn else av+bv-cv)
                elif bn=="TOP": p=bv
                elif bn=="LEFT": p=av
                elif bn=="PAETH":
                    pp=av+bv-cv; pa=abs(pp-av); pb=abs(pp-bv); pc=abs(pp-cv)
                    p=av if (pa<=pb and pa<=pc) else (bv if pb<=pc else cv)
                elif bn in ("GRAD",): p=(av+bv)//2+(bv-cv)//4
                elif bn in ("GAP","GAP80",):
                    gh=abs(av-Ww)+abs(bv-cv)+abs(bv-NE); gv=abs(av-cv)+abs(bv-NNe)+abs(NE-bv)
                    p=av if gv-gh>80 else (bv if gv-gh<-80 else (av+bv)//2+(NE-cv)//4)
                elif bn=="GAP32":
                    gh=abs(av-Ww)+abs(bv-cv)+abs(bv-NE); gv=abs(av-cv)+abs(bv-NNe)+abs(NE-bv)
                    p=av if gv-gh>32 else (bv if gv-gh<-32 else (av+bv)//2+(NE-cv)//4)
                elif bn=="GAP16":
                    gh=abs(av-Ww)+abs(bv-cv)+abs(bv-NE); gv=abs(av-cv)+abs(bv-NNe)+abs(NE-bv)
                    p=av if gv-gh>16 else (bv if gv-gh<-16 else (av+bv)//2+(NE-cv)//4)
                elif bn=="DG": p=(cv+dv)//2
                elif bn=="AVG_AB": p=(av+bv)//2
                elif bn=="PLANE": p=av+bv-cv
                elif bn=="AC": p=(av+cv)//2
                elif bn=="C": p=cv
                elif bn=="BC": p=(bv+cv)//2
                elif bn=="D": p=dv
                elif bn=="AVG3": p=(av+bv+cv)//3
                else: raise ValueError(bn)
                rf=int(streams[gi][curs[gi]]); curs[gi]+=1
                r=(1 if not neg else -1)*rf
                recon[i,j]=p+r
                assert recon[i,j]==kch[i,j], f"recon mismatch ch{ci} {(i,j)} {recon[i,j]} vs {kch[i,j]}"
        assert all(cc==len(streams[gi]) for gi,cc in enumerate(curs)), "cursor mismatch"
    return True

def main():
    print("== COMBINED E16 + Huff/Golomb 2-way, KSET2 vs WIDE ==", flush=True)
    for KSET,tag in [(KSET2,"KSET2"),(KSET_WIDE,"WIDE")]:
        t0=time.time()
        tot=0; dnt=0
        for path in IMAGES:
            nm=path.split("/")[-1]
            bpp2,det2,_,_,_=eval_combined(path,E16,KSET,False)
            img=np.array(Image.open(path).convert("RGB")); H,W,_=img.shape; d=H*W*3
            dnt+=d; tot+=bpp2*d
            print(f"{tag} 2-way {nm}: {bpp2:.4f} groups={det2}",flush=True)
        print(f"==> {tag} 2-way AVG={tot/dnt:.4f} [{time.time()-t0:.0f}s]",flush=True)
    print("== 3-way rANS refinement on WIDE 2-way best (decode-verified) ==", flush=True)
    t0=time.time(); tot2=0; tot3=0; dnt=0
    for path in IMAGES:
        nm=path.split("/")[-1]
        bpp2,det2,bpp3,det3,chinfo=eval_combined(path,E16,KSET_WIDE,True)
        img=np.array(Image.open(path).convert("RGB")); H,W,_=img.shape; d=H*W*3
        dnt+=d; tot2+=bpp2*d; tot3+=bpp3*d
        # backend counts
        nb={"H":0,"G":0,"R":0}
        for c in chinfo:
            for w in c["win"]: nb[w[2]]+=1
        print(f"3-way {nm}: 2-way={bpp2:.4f} 3-way={bpp3:.4f} groups={det3} backs={nb}",flush=True)
        prove_recon(path,chinfo)
        print(f"  RECON PASS {nm}",flush=True)
    print(f"==> 2-way AVG={tot2/dnt:.4f} 3-way AVG={tot3/dnt:.4f} [{time.time()-t0:.0f}s]",flush=True)

if __name__=="__main__":
    main()
