"""HAPRE-C CROWN driver: LOCO-365 + autoK quantile groups + per-group best expert + per-group Huffman.
Reuses proven probe modules for encoder decisions; C for packing/streaming decode."""
import ctypes, numpy as np, time, os, sys
from PIL import Image
sys.path.insert(0,"/tmp/opencode/autocompress/csrc")
sys.path.insert(0,"/tmp/opencode/autocompress/experiments")
os.chdir("/tmp/opencode/autocompress/experiments")
from driver import lib, canon_tables
import probe_b4_a as A
from probe_b4_g import prep, KSET2
from probe_b4_j import auto_groups_exact
from dp_refine import refine
from dp_refine import refine
from probe_b4_e import MOE6
c_u8=ctypes.c_uint8; c_i16=ctypes.c_int16; c_u16=ctypes.c_uint16
lib.pack_syms.argtypes=[ctypes.POINTER(c_i16),ctypes.POINTER(c_u8),ctypes.c_int,ctypes.POINTER(ctypes.c_uint32),ctypes.POINTER(c_u8),ctypes.POINTER(c_u8),ctypes.c_size_t]
lib.pack_syms.restype=ctypes.c_size_t
lib.crown_unpack_ch.argtypes=[ctypes.POINTER(c_u8),ctypes.c_size_t,ctypes.c_int,ctypes.POINTER(ctypes.c_uint32),ctypes.POINTER(c_u8),ctypes.POINTER(c_u8),ctypes.POINTER(c_u16),ctypes.POINTER(c_i16),ctypes.c_int,ctypes.c_int]
lib.crown_unpack_ch.restype=ctypes.c_int
EXPID={n:i for i,n in enumerate(MOE6)}

def encode_crown(rgb):
    H,W,_=rgb.shape; N=H*W
    Y,Co,Cg=A.ycocg_fwd(rgb); chs=[Y,Co,Cg]
    t0=time.perf_counter()
    out=bytearray()
    payloads=[]; tablestore=[]
    for ch in chs:
        D=prep(ch); P=D["P"]; key=D["key"]; s=D["s"]
        resM=(ch-P["MED"]).astype(np.int32); rfM=(s*resM).astype(np.int32)
        bK,G,_,bmap=auto_groups_exact(key,rfM,KSET2)
        G,_=refine(key,rfM,G,bK)
        # per-group best expert by exact Huff data bits
        gpred=[]
        for gkeys in G:
            sel=np.isin(key,np.array(gkeys))
            best=None
            for n in MOE6:
                r=(ch-P[n]).astype(np.int32); rf=(s*r).astype(np.int32)
                g=rf[sel]
                d_=A.huff_bits(np.unique(g,return_counts=True)[1].tolist()) if g.size else 0
                if best is None or d_<best: best=d_; bn=n
            gpred.append(bn)
        # group id per pixel + chosen residuals (sign-flipped)
        gmap=np.zeros((729,),dtype=np.uint8)
        uk=np.unique(key)
        for gi,gkeys in enumerate(G):
            for u in gkeys: gmap[int(u)]=gi
        # stream channel header: K + active map + G + predids
        out+=int(bK).to_bytes(1,'little')+int(len(uk)).to_bytes(2,'little')
        for u in sorted(uk.tolist()):
            out+=int(u).to_bytes(2,'little')+int(gmap[int(u)]).to_bytes(1,'little')
        out+=int(len(G)).to_bytes(1,'little')
        for bn in gpred: out+=int(EXPID[bn]).to_bytes(1,'little')
        # symbols + group keys in scan order
        syms=np.zeros((N,),dtype=np.int16); gkeys=np.zeros((N,),dtype=np.uint8)
        cd=np.zeros((len(G)*2049,),np.uint32); ln=np.zeros((len(G)*2049,),np.uint8)
        flat_key=key.reshape(-1); flat_s=s.reshape(-1)
        for gi,(gkeys_list,bn) in enumerate(zip(G,gpred)):
            sel=np.isin(key,np.array(gkeys_list))
            r=(ch-P[bn]).astype(np.int32); rf=(s*r).astype(np.int32)
            syms[sel.reshape(-1)]=rf[sel].astype(np.int16)
            gkeys[sel.reshape(-1)]=gi
            g=rf[sel]
            if g.size:
                vals,cn=np.unique(g,return_counts=True)
                harr=np.zeros((2049,),dtype=np.int64)
                for v,c in zip(vals.tolist(),cn.tolist()): harr[int(v)+1024]=int(c)
                codes=canon_tables(harr)
                out+=len(codes).to_bytes(2,'little')
                for sym in sorted(codes): out+=((sym-1024)&0xFFFF).to_bytes(2,'little'); out.append(codes[sym][1])
                for sym,(cc,ll) in codes.items(): cd[gi*2049+sym]=cc; ln[gi*2049+sym]=ll
            else:
                out+=(0).to_bytes(2,'little')
        cap=(N*32)//8+1024; buf=(c_u8*cap)()
        n=lib.pack_syms(syms.ctypes.data_as(ctypes.POINTER(c_i16)),gkeys.ctypes.data_as(ctypes.POINTER(c_u8)),N,cd.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),ln.ctypes.data_as(ctypes.POINTER(c_u8)),buf,cap)
        out+=int(n).to_bytes(4,'little')+bytes(buf[:n])
        payloads.append((len(G),cd,ln))
    t_enc=time.perf_counter()-t0
    return bytes(out),(t_enc,payloads)

def decode_crown(blob,H,W):
    N=H*W; p=0
    cd=np.zeros((108*2049,),np.uint32); ln=np.zeros((108*2049,),np.uint8)
    predid=np.zeros((108,),np.uint8); cmap=np.zeros((3*729,),np.uint16)
    for k in range(3):
        bK=int.from_bytes(blob[p:p+1],'little'); p+=1
        na=int.from_bytes(blob[p:p+2],'little'); p+=2
        for _ in range(na):
            u=int.from_bytes(blob[p:p+2],'little'); g=blob[p+2]; p+=3
            cmap[k*729+u]=g
        G=int(bK)  # groups count not directly stored; infer: predids count = ? stored next as count
        # NOTE: encoder must store G count; we stored predids without count -> need count. See encode fix below.
        raise AssertionError("placeholder")
    return None

if __name__=="__main__":
    print("driver scaffold ok")

# ---------- rANS backend variant ----------
lib.crown_assemble.argtypes=[ctypes.POINTER(c_i16),ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.POINTER(c_u16),ctypes.POINTER(c_u8),ctypes.POINTER(c_i16),ctypes.c_int,ctypes.c_int,ctypes.c_int]
lib.crown_assemble.restype=ctypes.c_int
lib.rans_norm.argtypes=[ctypes.POINTER(ctypes.c_int64),ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_uint16),ctypes.POINTER(ctypes.c_int)]
lib.rans_norm.restype=None
lib.rans_encode.argtypes=[ctypes.POINTER(c_i16),ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_uint16),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(c_u8)]
lib.rans_encode.restype=ctypes.c_size_t

def _rans_stream(g):
    g=np.ascontiguousarray(g.reshape(-1).astype(np.int16)); N=g.size
    uv=np.unique(g)
    syms=np.array([int(v)+1024 for v in uv],dtype=np.int32); A=len(syms)
    hist=np.zeros(2049,dtype=np.int64)
    for v in uv: hist[int(v)+1024]=int((g==v).sum())
    h=hist.ctypes.data_as(ctypes.POINTER(ctypes.c_int64))
    sp=syms.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    fq=np.zeros((A,),np.uint16); cu=np.zeros((A,),np.int32)
    lib.rans_norm(h,A,sp,fq.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),cu.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
    lut=np.full((2049,),-1,np.int32)
    for i,s in enumerate(syms): lut[s]=i
    cap=N*3+16; buf=(c_u8*cap)()
    n=lib.rans_encode(g.ctypes.data_as(ctypes.POINTER(c_i16)),N,lut.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),fq.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),cu.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),buf)
    assert n>0
    return bytes(buf[:n]), syms, fq, cu

def encode_crown_rans(rgb):
    import time
    H,W,_=rgb.shape; N=H*W
    Y,Co,Cg=A.ycocg_fwd(rgb); chs=[Y,Co,Cg]
    t0=time.perf_counter()
    out=bytearray()
    for ch in chs:
        D=prep(ch); P=D["P"]; key=D["key"]; s=D["s"]
        resM=(ch-P["MED"]).astype(np.int32); rfM=(s*resM).astype(np.int32)
        bK,G,_,bmap=auto_groups_exact(key,rfM,KSET2)
        G,_=refine(key,rfM,G,bK)
        gpred=[]
        for gkeys in G:
            sel=np.isin(key,np.array(gkeys))
            best=None
            for n in MOE6:
                r=(ch-P[n]).astype(np.int32); rf=(s*r).astype(np.int32)
                g=rf[sel]
                d_=A.huff_bits(np.unique(g,return_counts=True)[1].tolist()) if g.size else 0
                if best is None or d_<best: best=d_; bn=n
            gpred.append(bn)
        gmap=np.zeros((729,),dtype=np.uint8)
        uk=np.unique(key)
        for gi,gkeys in enumerate(G):
            for u in gkeys: gmap[int(u)]=gi
        out+=int(bK).to_bytes(1,'little')+int(len(uk)).to_bytes(2,'little')
        for u in sorted(uk.tolist()):
            out+=int(u).to_bytes(2,'little')+int(gmap[int(u)]).to_bytes(1,'little')
        out+=int(len(G)).to_bytes(1,'little')
        for bn in gpred: out+=int(EXPID[bn]).to_bytes(1,'little')
        for gi,(gkeys,bn) in enumerate(zip(G,gpred)):
            sel=np.isin(key,np.array(gkeys))
            r=(ch-P[bn]).astype(np.int32); rf=(s*r).astype(np.int32)
            g=rf[sel]
            if g.size==0:
                out+=(0).to_bytes(4,'little')+(0).to_bytes(2,'little'); continue
            pay,syms,fq,cu=_rans_stream(g)
            out+=int(len(syms)).to_bytes(2,'little')
            for i,sm in enumerate(syms.tolist()):
                out+=((sm-1024)&0xFFFF).to_bytes(2,'little')+int(fq[i]).to_bytes(2,'little')
            out+=int(len(pay)).to_bytes(4,'little')+pay
    return bytes(out),(time.perf_counter()-t0,)

def decode_crown_rans(blob,H,W):
    import ctypes as C
    N=H*W; p=0
    img=np.zeros((3*N,),dtype=np.int16)
    lib.rans_slots.argtypes=[ctypes.POINTER(ctypes.c_uint16),ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.POINTER(ctypes.c_int)]
    lib.rans_decode.argtypes=[ctypes.POINTER(c_u8),ctypes.c_size_t,ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_uint16),ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(c_i16)]
    lib.rans_decode.restype=ctypes.c_int
    for k in range(3):
        bK=int.from_bytes(blob[p:p+1],'little'); p+=1
        na=int.from_bytes(blob[p:p+2],'little'); p+=2
        cmap=np.zeros((729,),np.uint16)
        for _ in range(na):
            u=int.from_bytes(blob[p:p+2],'little'); g=blob[p+2]; p+=3
            cmap[u]=g
        G=int.from_bytes(blob[p:p+1],'little'); p+=1
        predid=np.zeros((36,),np.uint8)
        for gi in range(G):
            predid[gi]=int.from_bytes(blob[p:p+1],'little'); p+=1
        streams=[]; goff=[0]
        for gi in range(G):
            A=int.from_bytes(blob[p:p+2],'little'); p+=2
            syms=np.zeros((A,),np.int32); fq=np.zeros((A,),np.uint16); cu=np.zeros((A,),np.int32)
            for i in range(A):
                sv=int.from_bytes(blob[p:p+2],'little'); f=int.from_bytes(blob[p+2:p+4],'little'); p+=4
                syms[i]=(sv+1024)&0xFFFF; fq[i]=f
            c=0
            for i in range(A): cu[i]=c; c+=int(fq[i])
            n=int.from_bytes(blob[p:p+4],'little'); p+=4
            pay=bytes(blob[p:p+n]); p+=n
            # rans decode full stream
            lut=np.full((2049,),-1,np.int32)
            for i,sm in enumerate(syms.tolist()): lut[sm]=i
            # count symbols: unknown until decoded! decode bound: use payload-driven loop? rans_decode needs N.
            # N per group unknown at parse -> store count in stream! (encoder fix needed below)
            streams.append((pay,syms,fq,cu))
        raise AssertionError("need per-group counts in stream")
    return None