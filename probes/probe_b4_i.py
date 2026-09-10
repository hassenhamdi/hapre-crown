"""probe_b4_i: decoder-simulation (recon) proof for G/Lloyd winners on all 7.
Recomputes everything from recon + transmitted side (group map, group->pred, runs).
numpy+PIL only.
"""
import numpy as np
from PIL import Image
import probe_b4_a as A
from probe_b4_g import prep
from probe_b4_f import run_pass
from probe_b4_h import build_G
from probe_b4_f import MOE6

IMAGES=A.IMAGES

def prove(path, lloyd):
    img=np.array(Image.open(path).convert("RGB"))
    H,W,_=img.shape
    Y,Co,Cg=A.ycocg_fwd(img); chs=[Y,Co,Cg]
    streams,side,runbits,meta,(H,W)=build_G(path,lloyd)
    for ci,(D,kch,chstr) in enumerate(zip([prep(ch) for ch in chs],chs,streams)):
        key=D["key"]; s=D["s"]; P=D["P"]
        main,runs=run_pass(kch,H,W)
        assert np.array_equal(main,meta[ci]["main"]) and runs==meta[ci]["runs"]
        # rebuild key->group->pred maps as decoder would
        G=meta[ci]["G"]; gpred=meta[ci]["gpred"]
        k2g={}; 
        for gi,gkeys in enumerate(G):
            for v in gkeys: k2g[int(v)]=gi
        # flatten main-stream values in decode order: regroup chstr per group -> need order!
        # Encoder wrote streams grouped BY GROUP (not in pixel order). Decoder reads per-group
        # streams using per-group cursors: at pixel with group gi, take next symbol of stream gi.
        curs=[0]*len(G)
        # verify multiset equality first
        for gi,gkeys in enumerate(G):
            sel=main & np.isin(key,np.array(gkeys))
            bn=gpred[gi]
            r=(kch-P[bn]).astype(np.int32); rf=(s*r).astype(np.int32)
            assert np.array_equal(np.sort(rf[sel].reshape(-1)),np.sort(np.asarray(chstr[gi]).reshape(-1))),f"stream {gi} mismatch"
        recon=np.zeros_like(kch)
        rp=list(runs); ri=0
        # local refs
        for i in range(H):
            Ri=recon[i]; Ci=kch[i]
            j=0
            while j<W:
                if i==0 and j==0: av=bv=cv=dv=0
                elif i==0: av=int(Ri[j-1]); bv=cv=av; dv=av
                elif j==0: bv=int(recon[i-1,0]); av=cv=bv; dv=int(recon[i-1,1]) if W>1 else bv
                else: av=int(Ri[j-1]); bv=int(recon[i-1,j]); cv=int(recon[i-1,j-1]); dv=int(recon[i-1,j+1]) if j+1<W else bv
                if j+1<W and av==bv==cv==dv and not(i==0 and j==0):
                    L=rp[ri]; ri+=1
                    for t2 in range(L):
                        assert int(Ci[j])==av; Ri[j]=av; j+=1
                    if j>=W: break
                    if i==0: av=int(Ri[j-1]); bv=cv=av; dv=av
                    elif j==0: bv=int(recon[i-1,0]); av=cv=bv; dv=int(recon[i-1,1]) if W>1 else bv
                    else: av=int(Ri[j-1]); bv=int(recon[i-1,j]); cv=int(recon[i-1,j-1]); dv=int(recon[i-1,j+1]) if j+1<W else bv
                # LOCO key/sign from recon
                q1=bv-cv; q2=cv-av; q3=dv-bv
                def Q(g):
                    ag=abs(g)
                    return 0 if g==0 else (1 if ag<=2 else (2 if ag<=7 else (3 if ag<=21 else 4)))*(1 if g>0 else -1)
                qq=(Q(q1),Q(q2),Q(q3))
                neg=qq[0]<0 or (qq[0]==0 and qq[1]<0) or (qq[0]==0 and qq[1]==0 and qq[2]<0)
                kk=tuple(-v for v in qq) if neg else qq
                kid=kk[0]*81+kk[1]*9+kk[2]
                assert kid==int(key[i,j]) and (1 if not neg else -1)==int(s[i,j])
                gi=k2g[kid]; bn=gpred[gi]
                Ww=Ri[j-2] if j>=2 else av
                NNe=recon[i-2,j] if i>=2 else bv
                NE=recon[i-1,j+1] if (i>=1 and j+1<W) else bv
                if i==0: Ww=av; NNe=av; NE=av
                if bn=="MED":
                    mx=av if av>bv else bv; mn=bv if av>bv else av
                    p=mn if cv>=mx else (mx if cv<=mn else av+bv-cv)
                elif bn=="TOP": p=bv
                elif bn=="PAETH":
                    pp=av+bv-cv; pa=abs(pp-av); pb=abs(pp-bv); pc=abs(pp-cv)
                    p=av if (pa<=pb and pa<=pc) else (bv if pb<=pc else cv)
                elif bn=="GRAD": p=(av+bv)//2+(bv-cv)//4
                elif bn=="GAP":
                    gh=abs(av-Ww)+abs(bv-cv)+abs(bv-NE); gv=abs(av-cv)+abs(bv-NNe)+abs(NE-bv)
                    p=av if gv-gh>80 else (bv if gv-gh<-80 else (av+bv)//2+(NE-cv)//4)
                else: p=(cv+dv)//2
                rf=int(chstr[gi][curs[gi]]); curs[gi]+=1
                r=(1 if not neg else -1)*rf
                Ri[j]=p+r
                assert Ri[j]==Ci[j],f"recon mismatch ch{ci} {(i,j)}"
                j+=1
        assert all(c==len(chstr[gi]) for gi,c in enumerate(curs)), "stream cursor mismatch"
    # YCoCg invertibility
    return True

def main():
    import time
    for lloyd in (False,True):
        t0=time.time()
        for path in IMAGES:
            nm=path.split("/")[-1]
            prove(path,lloyd)
            print(f"L{int(lloyd)} {nm}: RECON PASS",flush=True)
        print(f"L{int(lloyd)} all-7 RECON PASS [{time.time()-t0:.0f}s]",flush=True)

if __name__=="__main__":
    main()
