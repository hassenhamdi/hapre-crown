"""probe_b4_f: FINAL candidate — MED/MOE + flip-LOCO autoK clusters + runs + proofs.
numpy+PIL only. All bytes honest. Includes recon-simulation + Huffman round-trip proofs.
"""
import numpy as np, heapq
from PIL import Image
import probe_b4_a as A
from probe_b4_b import nbhd, predictors
from probe_b4_c import loco_ctx365

IMAGES=A.IMAGES; HEADER=A.HEADER
CHAMP={"kodim01.png":3.56,"kodim02.png":3.23,"kodim05.png":3.84,"kodim07.png":3.01,
       "kodim13.png":4.17,"kodim19.png":3.41,"kodim23.png":3.03}
MOE6=["MED","TOP","PAETH","GRAD","GAP","DG"]
KSET=(4,6,9,12,18,27)

def prep(ch):
    a,b,c,d,Ww,NNe,NE=nbhd(ch)
    P=predictors(a,b,c,d,Ww,NNe,NE)
    key,s=loco_ctx365(a,b,c,d)
    return dict(a=a,b=b,c=c,d=d,P=P,key=key,s=s)

def bestK_groups(key, rf, main):
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
        tt=len(uk)*4+8+3  # map + K-id side
        for gkeys in groups:
            g=rm[np.isin(km,np.array(gkeys))]
            if g.size: tb,_,_=A.stream_bits(g.reshape(-1)); tt+=tb
        if best is None or tt<best: best=tt; bK=K; bG=groups
    return bK,bG,best

def gamma_cost(L):
    v=L+1; return 2*v.bit_length()-1

def run_pass(P,H,W):
    main=np.ones((H,W),dtype=bool); runs=[]
    for i in range(H):
        j=0; row=P[i]
        while j<W:
            if i==0 and j==0: av=bv=cv=dv=0
            elif i==0: av=int(row[j-1]); bv=cv=av; dv=av
            elif j==0: bv=int(P[i-1,0]); av=cv=bv; dv=int(P[i-1,1]) if W>1 else bv
            else: av=int(row[j-1]); bv=int(P[i-1,j]); cv=int(P[i-1,j-1]); dv=int(P[i-1,j+1]) if j+1<W else bv
            if j+1<W and av==bv==cv==dv and not(i==0 and j==0):
                L=0
                while j+L<W and int(row[j+L])==av: L+=1
                runs.append(L); j+=L
                if j>=W: break
                j+=1
            else: j+=1
    # rebuild main mask from runs by re-walking (simpler: mark during walk)
    main=np.ones((H,W),dtype=bool); runs2=[]; 
    for i in range(H):
        j=0; row=P[i]
        while j<W:
            if i==0 and j==0: av=bv=cv=dv=0
            elif i==0: av=int(row[j-1]); bv=cv=av; dv=av
            elif j==0: bv=int(P[i-1,0]); av=cv=bv; dv=int(P[i-1,1]) if W>1 else bv
            else: av=int(row[j-1]); bv=int(P[i-1,j]); cv=int(P[i-1,j-1]); dv=int(P[i-1,j+1]) if j+1<W else bv
            if j+1<W and av==bv==cv==dv and not(i==0 and j==0):
                L=0
                while j+L<W and int(row[j+L])==av: L+=1
                runs2.append(L)
                if L>0: main[i,j:j+L]=False
                j+=L
                if j>=W: break
                j+=1
            else: j+=1
    assert runs==runs2
    return main,runs

def block_experts(ch, P, B=8):
    H,W=ch.shape
    nbh=(H+B-1)//B; nbw=(W+B-1)//B
    Hp=nbh*B; Wp=nbw*B
    L1s=[]
    RR={}
    for n in MOE6:
        r=(ch-P[n]).astype(np.int32)
        rp=np.zeros((Hp,Wp),dtype=np.int64); rp[:H,:W]=np.abs(r.astype(np.int64))
        L1s.append(rp.reshape(nbh,B,nbw,B).sum(axis=(1,3)))
        RR[n]=r
    L1=np.stack(L1s)  # 6,nbh,nbw
    ids=np.argmin(L1,axis=0).astype(np.uint8)
    pred=np.zeros_like(ch)
    for x,n in enumerate(MOE6):
        m=np.zeros((Hp,Wp),dtype=bool); m[:H,:W]=(ids==x)[:H,:W] if False else False
    # build mask per expert via kron
    for x,n in enumerate(MOE6):
        blk=(ids==x)
        full=np.kron(blk,np.ones((B,B),dtype=bool))[:H,:W]
        pred[full]=P[n][full]
    return pred,ids

def canon_lens(counts):
    syms=list(counts.keys())
    if len(syms)==1: return {syms[0]:1}
    H=[(counts[s],s) for s in syms]; heapq.heapify(H); par={}; nxt=1<<28; H2=H[:]
    while len(H2)>1:
        a,sa=heapq.heappop(H2); b,sb=heapq.heappop(H2); nn=nxt; nxt+=1
        par[sa]=(nn,0); par[sb]=(nn,1); heapq.heappush(H2,(a+b,nn))
    lens={}
    for s in syms:
        d=0;n=s
        while n in par: n=par[n][0];d+=1
        lens[s]=d
    return lens

def huff_roundtrip(vals):
    vals=np.asarray(vals).reshape(-1)
    uv,cn=np.unique(vals,return_counts=True)
    counts={int(v):int(c) for v,c in zip(uv,cn)}
    lens=canon_lens(counts)
    order=sorted(counts,key=lambda s:(lens[s],s))
    codes={}; code=0; prev=0
    for s in order: code<<=(lens[s]-prev); codes[s]=(code,lens[s]); code+=1; prev=lens[s]
    # encode
    acc=0; nb=0; out=bytearray()
    for v in vals:
        c,l=codes[int(v)]; acc=(acc<<l)|c; nb+=l
        while nb>=8: nb-=8; out.append((acc>>nb)&0xFF); acc&=((1<<nb)-1) if nb else 0
    if nb: out.append((acc<<(8-nb))&0xFF)
    # decode via tree
    root={}; 
    for s,(c,l) in codes.items():
        n=root
        for bpos in range(l-1,-1,-1):
            b=(c>>bpos)&1
            n=n.setdefault(b,{})
        n["$"]=int(s)
    res=np.empty_like(vals); bitpos=0; raw=bytes(out); totbits=len(raw)*8
    for idx in range(len(vals)):
        n=root
        while "$" not in n:
            byte=raw[bitpos>>3]; bit=(byte>>(7-(bitpos&7)))&1; bitpos+=1; n=n[bit]
        res[idx]=n["$"]
    assert np.array_equal(res,vals), "huff roundtrip FAIL"
    return True

def med_of(a,b,c):
    return np.where(c>=np.maximum(a,b),np.minimum(a,b),np.where(c<=np.minimum(a,b),np.maximum(a,b),a+b-c))

def recon_proof(ch, pred, key, s, main, runs, res_main_ordered, B=None, ids=None, Pdict=None):
    """Simulate decoder: planes decoded with run machine + stored main residuals; assert == ch."""
    H,W=ch.shape
    recon=np.zeros_like(ch)
    rp=list(runs); ri=0; syms=list(res_main_ordered); si=0
    for i in range(H):
        j=0
        while j<W:
            if i==0 and j==0: av=bv=cv=dv=0
            elif i==0: av=int(recon[i,j-1]); bv=cv=av; dv=av
            elif j==0: bv=int(recon[i-1,0]); av=cv=bv; dv=int(recon[i-1,1]) if W>1 else bv
            else: av=int(recon[i,j-1]); bv=int(recon[i-1,j]); cv=int(recon[i-1,j-1]); dv=int(recon[i-1,j+1]) if j+1<W else bv
            if j+1<W and av==bv==cv==dv and not(i==0 and j==0):
                L=rp[ri]; ri+=1
                for t2 in range(L): assert j<W and int(ch[i,j])==av, "run value mismatch"; recon[i,j]=av; j+=1
                if j>=W: break
                # forced symbol below
                if i==0 and j==0: av=bv=cv=dv=0
                elif i==0: av=int(recon[i,j-1]); bv=cv=av; dv=av
                elif j==0: bv=int(recon[i-1,0]); av=cv=bv; dv=int(recon[i-1,1]) if W>1 else bv
                else: av=int(recon[i,j-1]); bv=int(recon[i-1,j]); cv=int(recon[i-1,j-1]); dv=int(recon[i-1,j+1]) if j+1<W else bv
            # regular symbol
            e=min(8,(abs(av-cv)+abs(bv-cv))>>4)
            if B is None: p=int(med_of(np.array(av),np.array(bv),np.array(cv)))
            else:
                blk=ids[(i//B),(j//B)]; n=MOE6[int(blk)]
                # recompute full nbhd for expert from recon
                Ww=recon[i,j-2] if j>=2 else av
                NNe=recon[i-2,j] if i>=2 else bv
                NE=recon[i-1,j+1] if (i>=1 and j+1<W) else bv
                if i==0: Ww=av; NNe=av; NE=av
                if j==0 and i>0: Ww=av
                aav,bvv,cvv,dvv=av,bv,cv,dv
                if n=="MED": p=int(med_of(np.array(aav),np.array(bvv),np.array(cvv)))
                elif n=="TOP": p=bvv
                elif n=="PAETH":
                    pp=aav+bvv-cvv; pa=abs(pp-aav); pb=abs(pp-bvv); pc=abs(pp-cvv)
                    p=aav if (pa<=pb and pa<=pc) else (bvv if pb<=pc else cvv)
                elif n=="GRAD": p=(aav+bvv)//2+(bvv-cvv)//4  # python // floors like fdiv for denom>0
                elif n=="GAP":
                    gh=abs(aav-Ww)+abs(bvv-cvv)+abs(bvv-NE); gv=abs(aav-cvv)+abs(bvv-NNe)+abs(NE-bvv)
                    p=aav if gv-gh>80 else (bvv if gv-gh<-80 else (aav+bvv)//2+(NE-cvv)//4)
                else: p=(cvv+dvv)//2
            # context check: recomputed key/sign must match encoder's
            q1,q2,q3=bv-cv,cv-av,dv-bv
            def Q(g):
                ag=abs(g)
                return 0 if g==0 else (1 if ag<=2 else (2 if ag<=7 else (3 if ag<=21 else 4)))*(1 if g>0 else -1)
            qq=(Q(q1),Q(q2),Q(q3))
            neg=qq[0]<0 or (qq[0]==0 and qq[1]<0) or (qq[0]==0 and qq[1]==0 and qq[2]<0)
            kk=tuple(-v for v in qq) if neg else qq
            kid=(kk[0]*81+kk[1]*9+kk[2])
            assert kid==int(key[i,j]), f"ctx mismatch {(i,j)}"
            assert (1 if not neg else -1)==int(s[i,j])
            rf=syms[si]; si+=1
            r=int(s[i,j])*int(rf)
            recon[i,j]=p+r
            assert recon[i,j]==ch[i,j], f"recon mismatch {(i,j)}"
            j+=1
    assert si==len(syms) and ri==len(rp)
    return True

def eval_config(path, use_moe, B=8):
    img=np.array(Image.open(path).convert("RGB"))
    H,W,_=img.shape; dn=H*W*3
    Y,Co,Cg=A.ycocg_fwd(img); chs=[Y,Co,Cg]
    DD=[prep(ch) for ch in chs]
    total=HEADER; nids=0
    stored=[]  # per channel: (main residuals in decode order grouped? store flat main rf + group assignment)
    for D,kch in zip(DD,chs):
        if use_moe:
            pred,ids=block_experts(kch,D["P"],B)
            nids+=ids.size
        else: pred=D["P"]["MED"]; ids=None
        res=(kch-pred).astype(np.int32)
        rf=(D["s"]*res).astype(np.int32)
        main,runs=run_pass(kch,H,W)
        Km,Gm,best=bestK_groups(D["key"],rf,main)
        # bits: main grouped + map side + runs + (ids counted globally)
        for gkeys in Gm:
            g=rf[main & np.isin(D["key"],np.array(gkeys))]
            if g.size:
                tb,_,_=A.stream_bits(g.reshape(-1)); total+=tb
                huff_roundtrip(g.reshape(-1))
        # NOTE: bestK_groups' `best` already includes data+side; data added above, so add side only:
        uk=np.unique(D["key"][main])
        total+=len(uk)*4+8+3
        total+=sum(gamma_cost(L) for L in runs)
        stored.append(dict(pred=pred,ids=ids,res=res,rf=rf,main=main,runs=runs,D=D,ch=kch))
    if use_moe:
        total+= (nids*3+7)//8*8 + 64
    total+=64
    # recon proof per channel
    for st in stored:
        D=st["D"]
        # main residuals in decode order: walk machine order and collect rf[main pixels] in order
        order_rf=st["rf"][st["main"]]  # row-major order == decode order? decode walks rows, main pixels in row-major → yes
        recon_proof(st["ch"],st["pred"],D["key"],D["s"],st["main"],st["runs"],order_rf,
                    B=B if use_moe else None, ids=st["ids"], Pdict=D["P"])
    return total/dn, total

def main():
    for use_moe in (False,True):
        tot=0; dn_tot=0; rows=[]
        for path in IMAGES:
            nm=path.split("/")[-1]
            bpp,_=eval_config(path,use_moe)
            img=np.array(Image.open(path).convert("RGB")); H,W,_=img.shape; dn=H*W*3; dn_tot+=dn; tot+=bpp*dn
            rows.append((nm,bpp))
            print(f"{'MOE' if use_moe else 'MED'} {nm}: {bpp:.4f} champ={CHAMP[nm]:.2f} d={bpp-CHAMP[nm]:+.4f}",flush=True)
        avg=tot/dn_tot
        print(f"==> {'MOE8' if use_moe else 'MED'}-flipAUTO+runs AVG={avg:.4f} vs champ 3.4643 d={avg-3.4643:+.4f} vs WebP-m3 3.38 d={avg-3.38:+.4f}  PROOFS PASS",flush=True)

if __name__=="__main__":
    main()
