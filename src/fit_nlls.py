"""Fit per-(ch,energy-ctx) nonlinear least-squares predictor on originals (FLAC-style)."""
import numpy as np
Q=8  # fixed-point shift; coeffs int16 = round(w*256)
def fit_image(yc, H, W, cap=60000):
    # Full-grid fit per (ch,ctx) with MED RD-gate: returns icoeff (27,6) int16 + medflag (27,) uint8.
    # medflag=1 -> C loop uses MED for that table (fallback when LS loses on fitting sample).
    P3=[]
    for k in range(3):
        P=np.pad(yc[k*H*W:(k+1)*H*W].reshape(H,W),((1,0),(1,0)),mode='edge').astype(np.int32)
        P3.append(P)
    coef=np.zeros((27,6),dtype=np.int16); medflag=np.zeros((27,),dtype=np.uint8)
    for k in range(3):
        P=P3[k]
        I,J=np.meshgrid(np.arange(H),np.arange(W),indexing='ij')
        Io=I+1; Jo=J+1
        a=P[Io,Jo-1]; b=P[Io-1,Jo]; c=P[Io-1,Jo-1]
        y=P[Io,Jo].astype(np.float64)
        e=np.minimum(8,(np.abs(a-c)+np.abs(b-c))//16)
        F=np.stack([np.ones_like(a),a,b,c,np.abs(a-b),(a+b-2*c)],axis=-1).astype(np.float64)
        m_med=np.minimum(a,b,(c>=np.maximum(a,b)) if False else a)  # placeholder replaced below
        # MED prediction (vectorized)
        m_med=np.where(c>=np.maximum(a,b),np.minimum(a,b),np.where(c<=np.minimum(a,b),np.maximum(a,b),a+b-c))
        for ctx in range(9):
            mm=e==ctx
            n=mm.sum()
            if n<50:
                coef[(k*9+ctx)]=[0,256,0,0,0,0]; medflag[k*9+ctx]=1; continue
            step=max(1,n//cap)
            sel=np.zeros_like(mm,bool); sel[::step,::step]=True; m2=mm&sel
            A=F[m2]; yv=y[m2]
            # IRLS for L1-optimal (entropy/Huffman-matched) fit: 4 iters reweighted LS
            w,_1,_2,_3=np.linalg.lstsq(A,yv,rcond=None)
            for _ in range(4):
                r=np.abs(A@w-yv); rw=1.0/np.maximum(1.0,r)
                s=np.sqrt(rw)
                w,_,_,_=np.linalg.lstsq(A*s[:,None],yv*s,rcond=None)
            # RD gate on L1 (entropy proxy), not MSE
            pred_ls=A@w
            pred_med=m_med[m2].astype(np.float64)
            if np.mean(np.abs(yv-pred_ls)) >= np.mean(np.abs(yv-pred_med)):
                coef[(k*9+ctx)]=[0,256,0,0,0,0]; medflag[k*9+ctx]=1; continue
            q=np.clip(np.round(w*256),-32768,32767).astype(np.int16)
            coef[(k*9+ctx)]=q
    return coef, medflag
if __name__=="__main__":
    import sys, os
    sys.path.insert(0,"/tmp/opencode/autocompress/csrc")
    from driver import lib
    import ctypes
    from PIL import Image
    d="/tmp/opencode/autocompress/experiments/real_photos"
    fn=sys.argv[1] if len(sys.argv)>1 else "kodim23.png"
    rgb=np.array(Image.open(os.path.join(d,fn)).convert("RGB")); H,W,_=rgb.shape
    yc=np.zeros((3*H*W,),dtype=np.int16)
    lib.ycocg_fwd(rgb.tobytes(),H,W,yc.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)))
    import time; t0=time.perf_counter()
    c=fit_image(yc,H,W)
    print("fit ms:",(time.perf_counter()-t0)*1000)
    print("Y ctx0 w:",c[0]/256)
    # report train MSE vs MED proxy (rough): skip