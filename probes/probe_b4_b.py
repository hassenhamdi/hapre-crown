"""probe_b4_b: fast vectorized tests — global bias, predictor sweep, finer ctx, xch.
numpy+PIL only. Same conventions as probe_b4_a (import it).
"""
import numpy as np, heapq
from PIL import Image
import probe_b4_a as A

IMAGES=A.IMAGES; HEADER=A.HEADER

def predictors(a,b,c,d,Ww,NNe,NE):
    med=np.where(c>=np.maximum(a,b),np.minimum(a,b),np.where(c<=np.minimum(a,b),np.maximum(a,b),a+b-c))
    top=b; left=a
    p=a+b-c
    paeth=np.where((np.abs(p-a)<=np.abs(p-b))&(np.abs(p-a)<=np.abs(p-c)),a,np.where(np.abs(p-b)<=np.abs(p-c),b,c))
    grad=(a//2+b//2 if False else 0)  # placeholder
    import math
    # floor div for possibly-negative: use // (numpy floors) — matches fdiv
    grd=(a+b)//2+(b-c)//4
    gh=np.abs(a-Ww)+np.abs(b-c)+np.abs(b-NE); gv=np.abs(a-c)+np.abs(b-NNe)+np.abs(NE-b)
    gap=np.where(gv-gh>80,a,np.where(gv-gh<-80,b,(a+b)//2+(NE-c)//4))
    dg=(c+d)//2
    return {"MED":med,"TOP":top,"LEFT":left,"PAETH":paeth,"GRAD":grd,"GAP":gap,"DG":dg}

def nbhd(ch):
    H,W=ch.shape
    a=np.empty_like(ch); b=np.empty_like(ch); c=np.empty_like(ch); d=np.empty_like(ch)
    Ww=np.empty_like(ch); NNe=np.empty_like(ch); NE=np.empty_like(ch)
    a[:,1:]=ch[:,:-1]; a[:,0]=0
    b[1:,:]=ch[:-1,:]; b[0,:]=0
    c[1:,1:]=ch[:-1,:-1]; c[0,:]=0; c[:,0]=0
    b[0,1:]=a[0,1:]; c[0,1:]=a[0,1:]
    a[1:,0]=b[1:,0]; c[1:,0]=b[1:,0]
    # d = topright recon; i==0 -> d=a; last col -> d=b
    d[:,:-1]=0
    d[1:,:-1]=ch[:-1,1:]; d[0,:-1]=a[0,:-1]
    d[:,W-1]=b[:,W-1]
    # Ww = left-left; NNe = top-top; NE = top-right(=d except row0 handling already)
    Ww[:,2:]=ch[:,:-2]; Ww[:,1]=a[:,1]; Ww[:,0]=a[:,0]
    # fix Ww row0/col per causality (recon==orig): row0: Ww=a for j<2
    Ww[0,:2]=a[0,:2][:2] if W>1 else a[0,:1]
    NNe[2:,:]=ch[:-2,:]; NNe[1,:]=b[1,:]; NNe[0,:]=b[0,:]
    NE=d.copy()
    return a,b,c,d,Ww,NNe,NE

def ctx9_from(a,b,c):
    return np.minimum((np.abs(a-c)+np.abs(b-c))>>4,8).astype(np.uint8)

def bits_ctx9_of_resids(res_ctx_list):
    t=HEADER; d=0
    for res,ctx in res_ctx_list:
        for q in range(9):
            g=res[ctx==q]
            if g.size==0: continue
            tb,db,_=A.stream_bits(g.reshape(-1)); t+=tb; d+=db
    return t,d

def bits_order0_of_resids(reslist):
    t=HEADER; d=0
    for res in reslist:
        tb,db,_=A.stream_bits(res.reshape(-1)); t+=tb; d+=db
    return t,d

def main():
    # per-image loop, keep res for each predictor
    names=["MED","TOP","LEFT","PAETH","GRAD","GAP","DG"]
    tot0={n:0 for n in names}; tot9={n:0 for n in names}; denom=0
    gbias9_tot=0; medbias9_tot=0; fine16_tot=0; base9_tot=0; base0_tot=0
    xcorr_num=[0,0,0]; xcorr_den1=[0,0,0]; xcorr_den2=[0,0,0]
    perimg={}
    for path in IMAGES:
        nm=path.split("/")[-1]
        img=np.array(Image.open(path).convert("RGB"))
        H,W,_=img.shape; dn=H*W*3; denom+=dn
        Y,Co,Cg=A.ycocg_fwd(img)
        chs=[Y,Co,Cg]
        N=nbhd_all(chs)
        # predictor sweep bits
        row={}
        for n in names:
            rl=[]; r9=[]
            for k in range(3):
                P=N[k]["pred"][n]; ch=chs[k]
                res=(ch-P).astype(np.int32); ctx=N[k]["ctx9"]
                rl.append(res); r9.append((res,ctx))
            t0,_=bits_order0_of_resids(rl); t9,_=bits_ctx9_of_resids(r9)
            tot0[n]+=t0; tot9[n]+=t9
            row[n]=(t0/dn,t9/dn)
        base0_tot+=tot0["MED"]*0  # accumulated globally; per-img below
        perimg[nm]=row
        # global per-(ch,ctx9) mean/median bias, MED residuals
        t_corr_mean=HEADER; t_corr_med=HEADER
        t_fine=HEADER
        for k in range(3):
            res=(chs[k]-N[k]["pred"]["MED"]).astype(np.int32); ctx=N[k]["ctx9"]
            rc=res.copy()
            for q in range(9):
                m=res[ctx==q]
                if m.size==0: continue
                mu=int(np.round(m.mean())); rc[ctx==q]=m-mu
            for q in range(9):
                g=rc[ctx==q]
                if g.size==0: continue
                tb,_,_=A.stream_bits(g.reshape(-1)); t_corr_mean+=tb
            t_corr_mean+=8*9  # side: 27 int8 means ~216b total; add per-channel share here (72b/ch)
            rc2=res.copy()
            for q in range(9):
                m=res[ctx==q]
                if m.size==0: continue
                md=int(np.median(m)); rc2[ctx==q]=m-md
            for q in range(9):
                g=rc2[ctx==q]
                if g.size==0: continue
                tb,_,_=A.stream_bits(g.reshape(-1)); t_corr_med+=tb
            t_corr_med+=8*9
            # finer energy bins >>3 cap15 (16 tables/ch)
            a,b,c=N[k]["a"],N[k]["b"],N[k]["c"]
            cf=np.minimum((np.abs(a-c)+np.abs(b-c))>>3,15).astype(np.uint8)
            for q in range(16):
                g=res[cf==q]
                if g.size==0: continue
                tb,_,_=A.stream_bits(g.reshape(-1)); t_fine+=tb
        gbias9_tot+=t_corr_mean; medbias9_tot+=t_corr_med; fine16_tot+=t_fine
        # xchannel corr of MED residuals
        r0=[(chs[k]-N[k]["pred"]["MED"]).astype(np.float64).reshape(-1) for k in range(3)]
        pairs=[(0,1),(0,2),(1,2)]
        for pi,(i,j) in enumerate(pairs):
            xcorr_num[pi]+=float((r0[i]*r0[j]).sum())
            xcorr_den1[pi]+=float((r0[i]**2).sum()); xcorr_den2[pi]+=float((r0[j]**2).sum())
        print(f"{nm}: "+" ".join(f"{n}0={row[n][0]:.3f}/{n}9={row[n][1]:.3f}" for n in names)+f" | meanBias9={t_corr_mean/dn:.4f} medBias9={t_corr_med/dn:.4f} fine16={t_fine/dn:.4f}",flush=True)
    print("-"*100)
    for n in names:
        print(f"{n}: order0={tot0[n]/denom:.4f} ctx9={tot9[n]/denom:.4f}")
    print(f"meanBias9={gbias9_tot/denom:.4f} medBias9={medbias9_tot/denom:.4f} fine16={fine16_tot/denom:.4f}")
    print(f"MED ctx9 baseline={tot9['MED']/denom:.4f} (expect ~3.5347)")
    for pi,(i,j) in enumerate([(0,1),(0,2),(1,2)]):
        print(f"corr ch{i}-ch{j} resid = {xcorr_num[pi]/np.sqrt(xcorr_den1[pi]*xcorr_den2[pi]):+.4f}")

def nbhd_all(chs):
    out=[]
    for ch in chs:
        a,b,c,d,Ww,NNe,NE=nbhd(ch)
        out.append({"a":a,"b":b,"c":c,"ctx9":ctx9_from(a,b,c),
                    "pred":predictors(a,b,c,d,Ww,NNe,NE)})
    return out

if __name__=="__main__":
    main()