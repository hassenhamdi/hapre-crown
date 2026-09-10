"""HAPRE-C NL-LS driver: Boss-2/3 weapon. Coeffs (27,6) int16 transmitted in stream."""
import ctypes, numpy as np, time, os, sys
from PIL import Image
sys.path.insert(0,"/tmp/opencode/autocompress/csrc")
from driver import lib, canon_tables
from fit_nlls import fit_image
c_u8=ctypes.c_uint8; c_i16=ctypes.c_int16
lib.nlls_res_ctx9.argtypes=[ctypes.POINTER(c_i16),ctypes.c_int,ctypes.c_int,ctypes.POINTER(c_i16),ctypes.POINTER(c_u8),ctypes.POINTER(c_i16),ctypes.POINTER(c_u8),ctypes.POINTER(ctypes.c_int64)]
lib.nlls_unpack.argtypes=[ctypes.POINTER(c_u8),ctypes.c_size_t,ctypes.c_int,ctypes.POINTER(c_i16),ctypes.POINTER(c_u8),ctypes.POINTER(ctypes.c_uint32),ctypes.POINTER(c_u8),ctypes.POINTER(c_i16),ctypes.POINTER(c_i16),ctypes.c_int,ctypes.c_int]
lib.nlls_unpack.restype=ctypes.c_int

def encode_nlls(rgb):
    H,W,_=rgb.shape
    yc=np.zeros((3*H*W,),dtype=np.int16)
    lib.ycocg_fwd(rgb.tobytes(),H,W,yc.ctypes.data_as(ctypes.POINTER(c_i16)))
    t0=time.perf_counter(); coef,medflag=fit_image(yc,H,W); t_fit=time.perf_counter()-t0
    res=np.zeros_like(yc); ctx=np.zeros((3*H*W,),dtype=np.uint8)
    hist=np.zeros((3*9*513,),dtype=np.int64)
    t0=time.perf_counter()
    lib.nlls_res_ctx9(yc.ctypes.data_as(ctypes.POINTER(c_i16)),H,W,coef.ctypes.data_as(ctypes.POINTER(c_i16)),medflag.ctypes.data_as(ctypes.POINTER(c_u8)),res.ctypes.data_as(ctypes.POINTER(c_i16)),ctx.ctypes.data_as(ctypes.POINTER(c_u8)),hist.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)))
    t_res=time.perf_counter()-t0
    N=H*W
    cd=np.zeros((27*513,),np.uint32); ln=np.zeros((27*513,),np.uint8)
    out=bytearray(); out+=coef.tobytes()+medflag.tobytes()  # 324B coeff + 27B flags
    t1=time.perf_counter()
    for k in range(3):
        for e in range(9):
            h=hist[(k*9+e)*513:(k*9+e+1)*513]
            codes=canon_tables(h) if h.sum()>0 else {}
            out+=len(codes).to_bytes(2,'little')
            for s in sorted(codes): out+=((s-256)&0xFFFF).to_bytes(2,'little'); out.append(codes[s][1])
            for s,(c,l) in codes.items(): cd[(k*9+e)*513+s]=c; ln[(k*9+e)*513+s]=l
    cap=(N*3*32)//8+1024
    buf=(c_u8*cap)()
    # reuse ctx9 packer (same layout: res planes + ctx planes); single call
    lib.huff_pack_ctx9.restype=ctypes.c_size_t
    n=lib.huff_pack_ctx9(res.ctypes.data_as(ctypes.POINTER(c_i16)),ctx.ctypes.data_as(ctypes.POINTER(c_u8)),N,cd.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),ln.ctypes.data_as(ctypes.POINTER(c_u8)),buf,cap)
    out+=int(n).to_bytes(4,'little')+bytes(buf[:n])
    t_pack=time.perf_counter()-t1
    return bytes(out),(t_fit,t_res,t_pack)

def decode_nlls(blob,H,W):
    N=H*W; p=0
    coef=np.frombuffer(blob[p:p+324],dtype=np.int16).copy(); p+=324
    medflag=np.frombuffer(blob[p:p+27],dtype=np.uint8).copy(); p+=27
    cd=np.zeros((27*513,),np.uint32); ln=np.zeros((27*513,),np.uint8)
    for k in range(3):
        for e in range(9):
            A=int.from_bytes(blob[p:p+2],'little'); p+=2
            syms=[]
            for _ in range(A):
                sv=int.from_bytes(blob[p:p+2],'little'); l=blob[p+2]; p+=3
                s=(sv+256)&0xFFFF; ln[(k*9+e)*513+s]=l; syms.append(s)
            syms.sort(key=lambda s:(ln[(k*9+e)*513+s],s))
            code=0;prev=0
            for s in syms:
                L=int(ln[(k*9+e)*513+s]); code<<=(L-prev); cd[(k*9+e)*513+s]=code; code+=1; prev=L
    n=int.from_bytes(blob[p:p+4],'little'); p+=4
    buf=(c_u8*n)(*blob[p:p+n]); p+=n
    res=np.zeros((3*N,),dtype=np.int16)
    img=np.zeros((3*N,),dtype=np.int16)
    rc=lib.nlls_unpack(buf,n,N,coef.ctypes.data_as(ctypes.POINTER(c_i16)),medflag.ctypes.data_as(ctypes.POINTER(c_u8)),cd.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),ln.ctypes.data_as(ctypes.POINTER(c_u8)),res.ctypes.data_as(ctypes.POINTER(c_i16)),img.ctypes.data_as(ctypes.POINTER(c_i16)),H,W)
    assert rc==0, rc
    out=bytearray(N*3)
    lib.ycocg_inv(img.ctypes.data_as(ctypes.POINTER(c_i16)),H,W,(ctypes.c_char*len(out)).from_buffer(out))
    return bytes(out)

if __name__=="__main__":
    d="/tmp/opencode/autocompress/experiments/real_photos"
    for fn in sys.argv[1:] or ["kodim23.png"]:
        rgb=np.array(Image.open(os.path.join(d,fn)).convert("RGB")); H,W,_=rgb.shape
        t0=time.perf_counter(); blob,tt=encode_nlls(rgb); te=time.perf_counter()-t0
        H2,W2,_=rgb.shape
        t0=time.perf_counter()
        # decode needs H,W + blob parse: reuse decode_nlls
        dec=decode_nlls(blob,H2,W2); td=time.perf_counter()-t0
        print(f"{fn} NLLS bytes={len(blob)} bpp={len(blob)*8/(H*W*3):.2f} enc={te*1000:.0f}ms dec={td*1000:.0f}ms ROUNDTRIP={'PASS' if dec==rgb.tobytes() else 'FAIL'}",flush=True)
