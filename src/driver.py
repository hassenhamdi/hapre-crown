"""HAPRE-C driver: YCoCg-R + MED (C) + canonical Huffman (C pack/unpack), full round-trip proof + Pareto bench."""
import ctypes, numpy as np, heapq, time, os, io, subprocess
from PIL import Image
lib = ctypes.CDLL("/tmp/opencode/autocompress/src/libhapre.so")
c_u8 = ctypes.c_uint8; c_i16 = ctypes.c_int16; c_i64 = ctypes.c_int64
lib.ycocg_fwd.argtypes=[ctypes.c_char_p,ctypes.c_int,ctypes.c_int,ctypes.POINTER(c_i16)]
lib.ycocg_inv.argtypes=[ctypes.POINTER(c_i16),ctypes.c_int,ctypes.c_int,ctypes.c_char_p]
lib.med_residuals_exact.argtypes=[ctypes.POINTER(c_i16),ctypes.c_int,ctypes.c_int,ctypes.POINTER(c_i16),ctypes.POINTER(c_i64)]
lib.huff_pack.argtypes=[ctypes.POINTER(c_i16),ctypes.c_int,ctypes.POINTER(ctypes.c_uint32),ctypes.POINTER(c_u8),ctypes.POINTER(c_u8),ctypes.c_size_t]
lib.huff_pack.restype=ctypes.c_size_t
lib.huff_unpack.argtypes=[ctypes.POINTER(c_u8),ctypes.c_size_t,ctypes.c_int,ctypes.POINTER(ctypes.c_uint32),ctypes.POINTER(c_u8),ctypes.POINTER(c_i16)]
lib.huff_unpack.restype=ctypes.c_int
lib.med_reconstruct.argtypes=[ctypes.POINTER(c_i16),ctypes.c_int,ctypes.c_int,ctypes.POINTER(c_i16)]

def canon_tables(hist):
    syms=[i for i in range(2049) if hist[i]>0]
    if len(syms)==1: return {syms[0]:(0,1)}
    import heapq
    H=[(hist[s],s) for s in syms]; heapq.heapify(H); nxt=1<<28; par={}
    H2=H[:]
    while len(H2)>1:
        a,_sa=heapq.heappop(H2); b,_sb=heapq.heappop(H2)
        nn=nxt; nxt+=1; par[_sa]=(nn,0); par[_sb]=(nn,1); heapq.heappush(H2,(a+b,nn))
    lens={}
    for s in syms:
        d=0;n=s
        while n in par: n=par[n][0];d+=1
        lens[s]=d
    # canonical codes
    order=sorted(syms,key=lambda s:(lens[s],s))
    codes={}; code=0; prev=0
    for s in order:
        code<<=(lens[s]-prev); codes[s]=(code,lens[s]); code+=1; prev=lens[s]
    return codes

def encode_image(rgb):
    H,W,_=rgb.shape
    yc=np.zeros((3*H*W,),dtype=np.int16)
    lib.ycocg_fwd(rgb.tobytes(),H,W,yc.ctypes.data_as(ctypes.POINTER(c_i16)))
    res=np.zeros_like(yc)
    hist=np.zeros((3*2049,),dtype=np.int64)
    t0=time.perf_counter()
    lib.med_residuals_exact(yc.ctypes.data_as(ctypes.POINTER(c_i16)),H,W,res.ctypes.data_as(ctypes.POINTER(c_i16)),hist.ctypes.data_as(ctypes.POINTER(c_i64)))
    t_res=time.perf_counter()-t0
    out=bytearray(); total_syms=0
    t1=time.perf_counter()
    for k in range(3):
        h=hist[k*2049:(k+1)*2049]
        codes=canon_tables(h)
        cd=np.zeros((2049,),np.uint32); ln=np.zeros((2049,),np.uint8)
        for s,(c,l) in codes.items(): cd[s]=c; ln[s]=l
        A=len(codes)
        out+=A.to_bytes(2,'little')
        for s in sorted(codes): out+=((s-1024)&0xFFFF).to_bytes(2,'little',signed=False); out.append(ln[s])
        r=res[k*H*W:(k+1)*H*W].copy()
        cap=(r.size*32+7)//8+8
        buf=(c_u8*cap)()
        n=lib.huff_pack(r.ctypes.data_as(ctypes.POINTER(c_i16)),r.size,cd.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),ln.ctypes.data_as(ctypes.POINTER(c_u8)),buf,cap)
        out+=int(n).to_bytes(4,'little')+bytes(buf[:n])
        total_syms+=r.size
    t_pack=time.perf_counter()-t1
    return bytes(out), res, hist, (t_res,t_pack)

def decode_image(blob,H,W):
    res=np.zeros((3*H*W,),dtype=np.int16); p=0
    for k in range(3):
        A=int.from_bytes(blob[p:p+2],'little'); p+=2
        cd=np.zeros((2049,),np.uint32); ln=np.zeros((2049,),np.uint8)
        for _ in range(A):
            sv=int.from_bytes(blob[p:p+2],'little',signed=False); l=blob[p+2]; p+=3
            s=(sv+1024)&0xFFFF
            ln[s]=l
        # rebuild canonical codes (same order rule)
        syms=sorted([s for s in range(2049) if ln[s]>0],key=lambda s:(ln[s],s))
        code=0;prev=0
        for s in syms: code<<=(int(ln[s])-prev); cd[s]=code; code+=1; prev=int(ln[s])
        n=int.from_bytes(blob[p:p+4],'little'); p+=4
        buf=(c_u8*n)(*blob[p:p+n]); p+=n
        r=res[k*H*W:(k+1)*H*W]
        rc=lib.huff_unpack(buf,n,H*W,cd.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),ln.ctypes.data_as(ctypes.POINTER(c_u8)),r.ctypes.data_as(ctypes.POINTER(c_i16)))
        assert rc==0, rc
    yc=np.zeros_like(res)
    lib.med_reconstruct(res.ctypes.data_as(ctypes.POINTER(c_i16)),H,W,yc.ctypes.data_as(ctypes.POINTER(c_i16)))
    out=bytearray(H*W*3)
    lib.ycocg_inv(yc.ctypes.data_as(ctypes.POINTER(c_i16)),H,W,(ctypes.c_char*len(out)).from_buffer(out))
    return bytes(out), res

if __name__=="__main__":
    d="/tmp/opencode/autocompress/experiments/real_photos"
    import sys
    files=[f for f in sorted(os.listdir(d)) if f.startswith("kodim")] if len(sys.argv)<2 else sys.argv[1:]
    for fn in files:
        rgb=np.array(Image.open(os.path.join(d,fn)).convert("RGB")); H,W,_=rgb.shape
        t0=time.perf_counter(); blob,res,hist,(t_res,t_pack)=encode_image(rgb); t_enc=time.perf_counter()-t0
        t0=time.perf_counter(); dec,_=decode_image(blob,H,W); t_dec=time.perf_counter()-t0
        ok = dec==rgb.tobytes()
        bpp=len(blob)*8/(H*W*3)
        print(f"{fn} C-HAPRE bytes={len(blob)} bpp={bpp:.2f} enc={t_enc*1000:.0f}ms(res {t_res*1000:.0f}+pack {t_pack*1000:.0f}) dec={t_dec*1000:.0f}ms ROUNDTRIP={'PASS' if ok else 'FAIL'}",flush=True)