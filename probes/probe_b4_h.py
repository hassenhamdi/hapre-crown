"""probe_b4_h: G + Lloyd re-cluster + EXACT rANS bytes via existing libhapre.so + final proofs.
"""
import numpy as np, ctypes
from PIL import Image
import probe_b4_a as A
from probe_b4_b import nbhd, predictors
from probe_b4_c import loco_ctx365
from probe_b4_f import run_pass, gamma_cost, huff_roundtrip, MOE6
from probe_b4_g import prep, auto_groups, KSET2

IMAGES=A.IMAGES; HEADER=A.HEADER
CHAMP={"kodim01.png":3.56,"kodim02.png":3.23,"kodim05.png":3.84,"kodim07.png":3.01,
       "kodim13.png":4.17,"kodim19.png":3.41,"kodim23.png":3.03}

lib=ctypes.CDLL("/tmp/opencode/autocompress/csrc/libhapre.so")
c_u8=ctypes.c_uint8; c_i16=ctypes.c_int16; c_i32=ctypes.c_int32; c_u16=ctypes.c_uint16; c_i64=ctypes.c_int64
lib.rans_norm.argtypes=[ctypes.POINTER(c_i64),c_i32,ctypes.POINTER(c_i32),ctypes.POINTER(c_u16),ctypes.POINTER(c_i32)]
lib.rans_encode.argtypes=[ctypes.POINTER(c_i16),c_i32,ctypes.POINTER(c_i32),ctypes.POINTER(c_u16),ctypes.POINTER(c_i32),ctypes.POINTER(c_u8)]
lib.rans_encode.restype=ctypes.c_size_t
lib.rans_slots.argtypes=[ctypes.POINTER(c_u16),ctypes.POINTER(c_i32),c_i32,ctypes.POINTER(c_i32)]
lib.rans_decode.argtypes=[ctypes.POINTER(c_u8),ctypes.c_size_t,c_i32,ctypes.POINTER(c_i32),ctypes.POINTER(c_u16),ctypes.POINTER(c_i32),c_i32,ctypes.POINTER(c_i32),ctypes.POINTER(c_i16)]
lib.rans_decode.restype=c_i32

def rans_stream_bits(g):
    """Exact rANS byte count for int array g (M=14 C impl). Returns (total_bits, payload_bytes).
    Table format counted: 2B A + A*(2B sym + 2B freq) + 4B state-in-payload + payload."""
    g=np.ascontiguousarray(g.reshape(-1).astype(np.int16))
    N=g.size
    uv=np.unique(g)
    syms=np.array([int(v)+1024 for v in uv],dtype=np.int32)
    Ad=len(syms)
    hist=np.zeros(2049,dtype=np.int64); 
    for v in uv: hist[int(v)+1024]=int((g==v).sum())
    h=hist.ctypes.data_as(ctypes.POINTER(c_i64))
    sp=syms.ctypes.data_as(ctypes.POINTER(c_i32))
    freq=np.zeros(Ad,dtype=np.uint16); cum=np.zeros(Ad,dtype=np.int32)
    lib.rans_norm(h,Ad,sp,freq.ctypes.data_as(ctypes.POINTER(c_u16)),cum.ctypes.data_as(ctypes.POINTER(c_i32)))
    lut=np.full(2049,-1,dtype=np.int32)
    for i,s in enumerate(syms): lut[s]=i
    out=np.zeros(N+1024,dtype=np.uint8)
    n=lib.rans_encode(g.ctypes.data_as(ctypes.POINTER(c_i16)),N,
                      lut.ctypes.data_as(ctypes.POINTER(c_i32)),
                      freq.ctypes.data_as(ctypes.POINTER(c_u16)),
                      cum.ctypes.data_as(ctypes.POINTER(c_i32)),
                      out.ctypes.data_as(ctypes.POINTER(c_u8)))
    assert n>0
    # verify decode
    slot=np.zeros(1<<14,dtype=np.int32)
    lib.rans_slots(freq.ctypes.data_as(ctypes.POINTER(c_u16)),cum.ctypes.data_as(ctypes.POINTER(c_i32)),Ad,slot.ctypes.data_as(ctypes.POINTER(c_i32)))
    dec=np.zeros(N,dtype=np.int16)
    rc=lib.rans_decode(out.ctypes.data_as(ctypes.POINTER(c_u8)),n,N,sp,
                       freq.ctypes.data_as(ctypes.POINTER(c_u16)),cum.ctypes.data_as(ctypes.POINTER(c_i32)),Ad,
                       slot.ctypes.data_as(ctypes.POINTER(c_i32)),dec.ctypes.data_as(ctypes.POINTER(c_i16)))
    assert rc==0 and np.array_equal(dec,g), "rANS roundtrip FAIL"
    table_bits=16+Ad*32  # 2B A + 4B/symbol freq entry
    return table_bits+n*8, n

def build_G(path, lloyd=False):
    """Returns per-channel stream list [(values, group_id)] + side bits + runs + meta for proofs."""
    img=np.array(Image.open(path).convert("RGB"))
    H,W,_=img.shape
    Y,Co,Cg=A.ycocg_fwd(img); chs=[Y,Co,Cg]
    DD=[prep(ch) for ch in chs]
    streams=[]; side=HEADER+64; runbits=0
    meta=[]
    for D,kch in zip(DD,chs):
        P=D["P"]; key=D["key"]; s=D["s"]
        main,runs=run_pass(kch,H,W)
        runbits+=sum(gamma_cost(L) for L in runs)
        resM=(kch-P["MED"]).astype(np.int32); rfM=(s*resM).astype(np.int32)
        _,G,_=auto_groups(key,rfM,main,KSET2)
        # per-group predictor
        gpred=[]
        for gkeys in G:
            sel=main & np.isin(key,np.array(gkeys))
            best=None
            for n in MOE6:
                r=(kch-P[n]).astype(np.int32); rf=(s*r).astype(np.int32)
                g=rf[sel]
                d_=A.huff_bits(np.unique(g,return_counts=True)[1].tolist()) if g.size else 0
                if best is None or d_<best: best=d_; bn=n
            gpred.append(bn)
        if lloyd:
            # re-cluster on chosen-predictor residuals
            rfmix=np.zeros_like(rfM); 
            for gkeys,bn in zip(G,gpred):
                sel=np.isin(key,np.array(gkeys))
                r=(kch-P[bn]).astype(np.int32)
                rfmix[sel]=(s[sel]*r[sel]).astype(np.int32)
            _,G,_=auto_groups(key,rfmix,main,KSET2)
            gpred=[]
            for gkeys in G:
                sel=main & np.isin(key,np.array(gkeys))
                best=None
                for n in MOE6:
                    r=(kch-P[n]).astype(np.int32); rf=(s*r).astype(np.int32)
                    g=rf[sel]
                    d_=A.huff_bits(np.unique(g,return_counts=True)[1].tolist()) if g.size else 0
                    if best is None or d_<best: best=d_; bn=n
                gpred.append(bn)
        uk=np.unique(key[main]); side+=len(uk)*4+8+3+len(G)*3
        chstreams=[]
        for gkeys,bn in zip(G,gpred):
            sel=main & np.isin(key,np.array(gkeys))
            r=(kch-P[bn]).astype(np.int32); rf=(s*r).astype(np.int32)
            g=rf[sel]
            if g.size: chstreams.append(g); huff_roundtrip(g)
        streams.append(chstreams)
        meta.append(dict(key=key,s=s,main=main,runs=runs,G=G,gpred=gpred,P=P,ch=kch))
    return streams,side,runbits,meta,(H,W)

def totals(path, lloyd=False, coder="huff"):
    streams,side,runbits,meta,(H,W)=build_G(path,lloyd)
    dn=H*W*3
    if coder=="huff":
        t=side+runbits
        for chs in streams:
            for g in chs:
                tb,_,_=A.stream_bits(g.reshape(-1)); t+=tb
    else:
        t=side+runbits
        for chs in streams:
            for g in chs:
                tb,_=rans_stream_bits(g); t+=tb
    return t/dn

def main():
    for lloyd in (False,True):
        for coder in ("huff","rans"):
            tot=0; dn_tot=0
            for path in IMAGES:
                nm=path.split("/")[-1]
                bpp=totals(path,lloyd,coder)
                img=np.array(Image.open(path).convert("RGB")); H,W,_=img.shape; dn=H*W*3
                dn_tot+=dn; tot+=bpp*dn
                print(f"L{int(lloyd)} {coder} {nm}: {bpp:.4f} d_champ={bpp-CHAMP[nm]:+.4f}",flush=True)
            avg=tot/dn_tot
            print(f"==> L{int(lloyd)} {coder} AVG={avg:.4f} d_champ={avg-3.4643:+.4f} d_m3={avg-3.38:+.4f} d_m6={avg-3.32:+.4f}",flush=True)

if __name__=="__main__":
    main()
