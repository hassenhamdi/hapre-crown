"""probe_b5_c: expert-bank expansion — E=6 vs 12 vs 16, Huffman-only, honest predid side.
E6 (MOE6): MED,TOP,PAETH,GRAD,GAP80,DG (3b/group)
E12: +LEFT,AVG_AB,PLANE,GAP32,AC,C (4b/group)
E16: +BC,D,GAP16,AVG3 (4b/group)
Clusters on MED flip meanabs, per-group best-of-E by exact Huffman data bits.
numpy+PIL only.
"""
import sys, math
import numpy as np
from PIL import Image
sys.path.insert(0, "/tmp/opencode/autocompress/experiments")
import probe_b4_a as A
from probe_b4_b import nbhd
from probe_b4_c import loco_ctx365
from probe_b4_g import KSET2

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
MOE6 = ["MED","TOP","PAETH","GRAD","GAP80","DG"]
E12 = ["MED","TOP","LEFT","PAETH","GRAD","GAP80","DG","AVG_AB","PLANE","GAP32","AC","C"]
E16 = E12 + ["BC","D","GAP16","AVG3"]

def predictors_X(a,b,c,d,Ww,NNe,NE):
    med = np.where(c>=np.maximum(a,b),np.minimum(a,b),np.where(c<=np.minimum(a,b),np.maximum(a,b),a+b-c))
    top = b
    left = a
    p = a+b-c
    paeth = np.where((np.abs(p-a)<=np.abs(p-b))&(np.abs(p-a)<=np.abs(p-c)),a,np.where(np.abs(p-b)<=np.abs(p-c),b,c))
    grd = (a+b)//2+(b-c)//4
    gh = np.abs(a-Ww)+np.abs(b-c)+np.abs(b-NE); gv = np.abs(a-c)+np.abs(b-NNe)+np.abs(NE-b)
    gap80 = np.where(gv-gh>80,a,np.where(gv-gh<-80,b,(a+b)//2+(NE-c)//4))
    gap32 = np.where(gv-gh>32,a,np.where(gv-gh<-32,b,(a+b)//2+(NE-c)//4))
    gap16 = np.where(gv-gh>16,a,np.where(gv-gh<-16,b,(a+b)//2+(NE-c)//4))
    dg = (c+d)//2
    avg_ab = (a+b)//2
    plane = a+b-c
    ac = (a+c)//2
    cc = c
    bc = (b+c)//2
    dd = d
    avg3 = (a+b+c)//3
    return {"MED":med,"TOP":top,"LEFT":left,"PAETH":paeth,"GRAD":grd,"GAP80":gap80,"GAP32":gap32,
            "GAP16":gap16,"DG":dg,"AVG_AB":avg_ab,"PLANE":plane,"AC":ac,"C":cc,"BC":bc,"D":dd,
            "AVG3":avg3,"GAP":gap80}

def prepX(ch):
    a,b,c,d,Ww,NNe,NE = nbhd(ch)
    P = predictors_X(a,b,c,d,Ww,NNe,NE)
    key,s = loco_ctx365(a,b,c,d)
    return dict(a=a,b=b,c=c,d=d,P=P,key=key,s=s)

def eval_bank(path, bank):
    pred_bits = math.ceil(math.log2(len(bank)))
    img = np.array(Image.open(path).convert("RGB"))
    H,W,_ = img.shape; dn = H*W*3
    Y,Co,Cg = A.ycocg_fwd(img); chs=[Y,Co,Cg]
    DD=[prepX(ch) for ch in chs]
    t=HEADER; detail=[]
    pick_counts={}
    for D,kch in zip(DD,chs):
        P=D["P"]; key=D["key"]; s=D["s"]
        RF={n:(s*(kch-P[n]).astype(np.int32)).astype(np.int32) for n in bank}
        rfM=(s*(kch-P["MED"]).astype(np.int32)).astype(np.int32)
        best=None
        for K in KSET2:
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
            tt=mapbits+len(groups)*pred_bits
            for gkeys in groups:
                sel=np.isin(key,np.array(gkeys))
                be=None
                for n in bank:
                    g=RF[n][sel]
                    d_=A.huff_bits(np.unique(g,return_counts=True)[1].tolist()) if g.size else 0
                    if be is None or d_<be[0]: be=(d_,n)
                g=RF[be[1]][sel]
                if g.size: tb,_,_=A.stream_bits(g.reshape(-1)); tt+=tb
            if best is None or tt<best[0]: best=(tt,K,groups)
        t+=best[0]
        detail.append((best[1],len(best[2])))
        # count picks for best K
        _,_,groups=best
        for gkeys in groups:
            sel=np.isin(key,np.array(gkeys))
            be=None
            for n in bank:
                g=RF[n][sel]
                d_=A.huff_bits(np.unique(g,return_counts=True)[1].tolist()) if g.size else 0
                if be is None or d_<be[0]: be=(d_,n)
            pick_counts[be[1]]=pick_counts.get(be[1],0)+1
    return t/dn, detail, pick_counts

def main():
    for bank,tag in [(MOE6,"E6"),(E12,"E12"),(E16,"E16")]:
        import time; t0=time.time()
        tot=0; dnt=0; agg={}
        print(f"== bank {tag} ({len(bank)} experts, {math.ceil(math.log2(len(bank)))}b/group) ==")
        for path in IMAGES:
            nm=path.split("/")[-1]
            bpp,det,pc=eval_bank(path,bank)
            img=np.array(Image.open(path).convert("RGB")); H,W,_=img.shape; d=H*W*3
            dnt+=d; tot+=bpp*d
            for k,v in pc.items(): agg[k]=agg.get(k,0)+v
            print(f"{tag} {nm}: {bpp:.4f} groups={det} picks={pc}",flush=True)
        print(f"==> {tag} AVG={tot/dnt:.4f} [{time.time()-t0:.0f}s] total_picks={agg}",flush=True)

if __name__=="__main__":
    main()
