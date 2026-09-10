"""HAPRE-C µMoE driver: Boss-2 weapon. ids 3b-packed."""
import ctypes, numpy as np, time, os, sys
from PIL import Image
sys.path.insert(0,"/tmp/opencode/autocompress/csrc")
from driver import lib, canon_tables
c_u8=ctypes.c_uint8; c_i16=ctypes.c_int16
lib.moe_run_encode.argtypes=[ctypes.POINTER(c_i16),ctypes.c_int,ctypes.c_int,ctypes.POINTER(c_u8),ctypes.POINTER(c_i16),ctypes.POINTER(c_u8),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int64)]
lib.moe_run_unpack.argtypes=[ctypes.POINTER(c_u8),ctypes.c_size_t,ctypes.POINTER(c_u8),ctypes.c_size_t,ctypes.POINTER(ctypes.c_uint32),ctypes.POINTER(c_u8),ctypes.POINTER(c_u8),ctypes.POINTER(c_i16),ctypes.c_int,ctypes.c_int]
lib.moe_run_unpack.restype=ctypes.c_int
B=8
def pack3(ids):
    out=bytearray(); acc=0; nb=0
    for v in ids:
        acc=(acc<<3)|int(v); nb+=3
        while nb>=8: nb-=8; out.append((acc>>nb)&0xFF); acc&=((1<<nb)-1) if nb else 0
    if nb: out.append((acc<<(8-nb))&0xFF)
    return bytes(out)
def unpack3(buf,n):
    ids=np.zeros((n,),np.uint8); acc=0; nb=0; p=0; o=0
    for byte in buf:
        acc=(acc<<8)|byte; nb+=8
        while nb>=3 and o<n:
            nb-=3; ids[o]=(acc>>nb)&7; acc&=((1<<nb)-1) if nb else 0; o+=1
    return ids
def encode_moe(rgb):
    H,W,_=rgb.shape; N=H*W
    nbw=(W+B-1)//B; nbh=(H+B-1)//B; ni=3*nbw*nbh
    yc=np.zeros((3*N,),dtype=np.int16)
    lib.ycocg_fwd(rgb.tobytes(),H,W,yc.ctypes.data_as(ctypes.POINTER(c_i16)))
    ids=np.zeros((ni,),dtype=np.uint8)
    msym=np.zeros((3*N,),dtype=np.int16); mctx=np.zeros((3*N,),dtype=np.uint8)
    runs=np.zeros((3*N,),dtype=np.int32)
    n_main=(ctypes.c_int*1)(); n_runs=(ctypes.c_int*1)()
    hist=np.zeros((3*9*2049,),dtype=np.int64)
    t0=time.perf_counter()
    lib.moe_run_encode(yc.ctypes.data_as(ctypes.POINTER(c_i16)),H,W,ids.ctypes.data_as(ctypes.POINTER(c_u8)),msym.ctypes.data_as(ctypes.POINTER(c_i16)),mctx.ctypes.data_as(ctypes.POINTER(c_u8)),runs.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),n_main,n_runs,hist.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)))
    t_res=time.perf_counter()-t0
    nm=n_main[0]; nr=n_runs[0]
    cd=np.zeros((27*2049,),np.uint32); ln=np.zeros((27*2049,),np.uint8)
    out=bytearray()
    idb=pack3(ids)
    out+=int(ni).to_bytes(4,'little')+int(len(idb)).to_bytes(4,'little')+idb
    for t in range(27):
        h=hist[t*2049:(t+1)*2049]
        codes=canon_tables(h) if h.sum()>0 else {}
        out+=len(codes).to_bytes(2,'little')
        for s in sorted(codes): out+=((s-1024)&0xFFFF).to_bytes(2,'little'); out.append(codes[s][1])
        for s,(c,l) in codes.items(): cd[t*2049+s]=c; ln[t*2049+s]=l
    cap=(nm*32)//8+1024; buf=(c_u8*cap)()
    lib.pack_syms.restype=ctypes.c_size_t
    lib.pack_syms.argtypes=[ctypes.POINTER(c_i16),ctypes.POINTER(c_u8),ctypes.c_int,ctypes.POINTER(ctypes.c_uint32),ctypes.POINTER(c_u8),ctypes.POINTER(c_u8),ctypes.c_size_t]
    n=lib.pack_syms(msym.ctypes.data_as(ctypes.POINTER(c_i16)),mctx.ctypes.data_as(ctypes.POINTER(c_u8)),nm,cd.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),ln.ctypes.data_as(ctypes.POINTER(c_u8)),buf,cap)
    out+=int(nm).to_bytes(4,'little')+int(n).to_bytes(4,'little')+bytes(buf[:n])
    rcap=(nr*32)//8+1024; rbuf=(c_u8*rcap)()
    rn=lib.runs_pack(runs.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),nr,rbuf,rcap)
    out+=int(nr).to_bytes(4,'little')+int(rn).to_bytes(4,'little')+bytes(rbuf[:rn])
    return bytes(out),(t_res,nm,nr)
def decode_moe(blob,H,W):
    N=H*W; p=0
    nbw=(W+B-1)//B; nbh=(H+B-1)//B
    ni=int.from_bytes(blob[p:p+4],'little'); p+=4
    ilen=int.from_bytes(blob[p:p+4],'little'); p+=4
    ids=unpack3(blob[p:p+ilen],ni); p+=ilen
    cd=np.zeros((27*2049,),np.uint32); ln=np.zeros((27*2049,),np.uint8)
    for t in range(27):
        A=int.from_bytes(blob[p:p+2],'little'); p+=2
        syms=[]
        for _ in range(A):
            sv=int.from_bytes(blob[p:p+2],'little'); l=blob[p+2]; p+=3
            s=(sv+1024)&0xFFFF; ln[t*2049+s]=l; syms.append(s)
        syms.sort(key=lambda s:(ln[t*2049+s],s))
        code=0;prev=0
        for s in syms:
            L=int(ln[t*2049+s]); code<<=(L-prev); cd[t*2049+s]=code; code+=1; prev=L
    nm=int.from_bytes(blob[p:p+4],'little'); p+=4
    n=int.from_bytes(blob[p:p+4],'little'); p+=4
    main=(c_u8*n)(*blob[p:p+n]); p+=n
    nr=int.from_bytes(blob[p:p+4],'little'); p+=4
    rn=int.from_bytes(blob[p:p+4],'little'); p+=4
    runb=(c_u8*rn)(*blob[p:p+rn]); p+=rn
    img=np.zeros((3*N,),dtype=np.int16)
    rc=lib.moe_run_unpack(main,n,runb,rn,cd.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),ln.ctypes.data_as(ctypes.POINTER(c_u8)),ids.ctypes.data_as(ctypes.POINTER(c_u8)),img.ctypes.data_as(ctypes.POINTER(c_i16)),H,W)
    assert rc==0, rc
    out=bytearray(N*3)
    lib.ycocg_inv(img.ctypes.data_as(ctypes.POINTER(c_i16)),H,W,(ctypes.c_char*len(out)).from_buffer(out))
    return bytes(out)
if __name__=="__main__":
    d="/tmp/opencode/autocompress/experiments/real_photos"
    for fn in sys.argv[1:] or ["kodim23.png"]:
        rgb=np.array(Image.open(os.path.join(d,fn)).convert("RGB")); H,W,_=rgb.shape
        t0=time.perf_counter(); blob,tt=encode_moe(rgb); te=time.perf_counter()-t0
        t0=time.perf_counter(); dec=decode_moe(blob,H,W); td=time.perf_counter()-t0
        print(f"{fn} MOE bytes={len(blob)} bpp={len(blob)*8/(H*W*3):.2f} enc={te*1000:.0f}ms dec={td*1000:.0f}ms ROUNDTRIP={'PASS' if dec==rgb.tobytes() else 'FAIL'}",flush=True)
