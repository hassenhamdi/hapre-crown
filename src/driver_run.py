"""HAPRE-C RUN driver: Boss-1 KO attempt. Two-stream (main Huffman + gamma runs)."""
import ctypes, numpy as np, time, os, sys
from PIL import Image
sys.path.insert(0,"/tmp/opencode/autocompress/csrc")
from driver import lib, canon_tables
c_u8=ctypes.c_uint8; c_i16=ctypes.c_int16
lib.med_run_encode.argtypes=[ctypes.POINTER(c_i16),ctypes.c_int,ctypes.c_int,ctypes.POINTER(c_i16),ctypes.POINTER(c_u8),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int64)]
lib.pack_syms.argtypes=[ctypes.POINTER(c_i16),ctypes.POINTER(c_u8),ctypes.c_int,ctypes.POINTER(ctypes.c_uint32),ctypes.POINTER(c_u8),ctypes.POINTER(c_u8),ctypes.c_size_t]
lib.pack_syms.restype=ctypes.c_size_t
lib.runs_pack.argtypes=[ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.POINTER(c_u8),ctypes.c_size_t]
lib.runs_pack.restype=ctypes.c_size_t
lib.med_run_unpack.argtypes=[ctypes.POINTER(c_u8),ctypes.c_size_t,ctypes.POINTER(c_u8),ctypes.c_size_t,ctypes.POINTER(ctypes.c_uint32),ctypes.POINTER(c_u8),ctypes.POINTER(c_i16),ctypes.c_int,ctypes.c_int]
lib.med_run_unpack.restype=ctypes.c_int

def encode_run(rgb):
    H,W,_=rgb.shape; N=H*W
    yc=np.zeros((3*N,),dtype=np.int16)
    lib.ycocg_fwd(rgb.tobytes(),H,W,yc.ctypes.data_as(ctypes.POINTER(c_i16)))
    msym=np.zeros((3*N,),dtype=np.int16); mctx=np.zeros((3*N,),dtype=np.uint8)
    runs=np.zeros((3*N,),dtype=np.int32)
    n_main=(ctypes.c_int*1)(); n_runs=(ctypes.c_int*1)()
    hist=np.zeros((3*9*2049,),dtype=np.int64)
    t0=time.perf_counter()
    lib.med_run_encode(yc.ctypes.data_as(ctypes.POINTER(c_i16)),H,W,msym.ctypes.data_as(ctypes.POINTER(c_i16)),mctx.ctypes.data_as(ctypes.POINTER(c_u8)),runs.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),n_main,n_runs,hist.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)))
    t_res=time.perf_counter()-t0
    nm=n_main[0]; nr=n_runs[0]
    cd=np.zeros((27*2049,),np.uint32); ln=np.zeros((27*2049,),np.uint8)
    out=bytearray()
    for t in range(27):
        h=hist[t*2049:(t+1)*2049]
        codes=canon_tables(h) if h.sum()>0 else {}
        out+=len(codes).to_bytes(2,'little')
        for s in sorted(codes): out+=((s-256)&0xFFFF).to_bytes(2,'little'); out.append(codes[s][1])
        for s,(c,l) in codes.items(): cd[t*2049+s]=c; ln[t*2049+s]=l
    cap=(nm*32)//8+1024; buf=(c_u8*cap)()
    n=lib.pack_syms(msym.ctypes.data_as(ctypes.POINTER(c_i16)),mctx.ctypes.data_as(ctypes.POINTER(c_u8)),nm,cd.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),ln.ctypes.data_as(ctypes.POINTER(c_u8)),buf,cap)
    out+=int(nm).to_bytes(4,'little')+int(n).to_bytes(4,'little')+bytes(buf[:n])
    rcap=(nr*32)//8+1024; rbuf=(c_u8*rcap)()
    rn=lib.runs_pack(runs.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),nr,rbuf,rcap)
    out+=int(nr).to_bytes(4,'little')+int(rn).to_bytes(4,'little')+bytes(rbuf[:rn])
    return bytes(out),(t_res,nm,nr)

def decode_run(blob,H,W):
    p=0
    cd=np.zeros((27*2049,),np.uint32); ln=np.zeros((27*2049,),np.uint8)
    for t in range(27):
        A=int.from_bytes(blob[p:p+2],'little'); p+=2
        syms=[]
        for _ in range(A):
            sv=int.from_bytes(blob[p:p+2],'little'); l=blob[p+2]; p+=3
            s=(sv+256)&0xFFFF; ln[t*2049+s]=l; syms.append(s)
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
    img=np.zeros((3*H*W,),dtype=np.int16)
    rc=lib.med_run_unpack(main,n,runb,rn,cd.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),ln.ctypes.data_as(ctypes.POINTER(c_u8)),img.ctypes.data_as(ctypes.POINTER(c_i16)),H,W)
    assert rc==0, rc
    out=bytearray(H*W*3)
    lib.ycocg_inv(img.ctypes.data_as(ctypes.POINTER(c_i16)),H,W,(ctypes.c_char*len(out)).from_buffer(out))
    return bytes(out)

if __name__=="__main__":
    d="/tmp/opencode/autocompress/experiments/real_photos"
    for fn in sys.argv[1:] or ["kodim23.png"]:
        rgb=np.array(Image.open(os.path.join(d,fn)).convert("RGB")); H,W,_=rgb.shape
        t0=time.perf_counter(); blob,(tr,nm,nr)=encode_run(rgb); te=time.perf_counter()-t0
        t0=time.perf_counter(); dec=decode_run(blob,H,W); td=time.perf_counter()-t0
        runpx=0
        print(f"{fn} RUN bytes={len(blob)} bpp={len(blob)*8/(H*W*3):.2f} enc={te*1000:.0f}ms dec={td*1000:.0f}ms main={nm} runs={nr} ROUNDTRIP={'PASS' if dec==rgb.tobytes() else 'FAIL'}",flush=True)
