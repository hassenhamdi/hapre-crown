"""rANS per-channel test on CTX9-MED residuals."""
import ctypes, numpy as np, time, os, sys
from PIL import Image
sys.path.insert(0,"/tmp/opencode/autocompress/csrc")
from driver import lib
c_u8=ctypes.c_uint8; c_i16=ctypes.c_int16
import ctypes as C
# prototypes set inside test() below
from driver import lib as L
L.med_res_ctx9.argtypes=[ctypes.POINTER(c_i16),ctypes.c_int,ctypes.c_int,ctypes.POINTER(c_i16),ctypes.POINTER(c_u8),ctypes.POINTER(ctypes.c_int64)]

def test(fn):
    from driver import lib as Lb
    d="/tmp/opencode/autocompress/experiments/real_photos"
    rgb=np.array(Image.open(os.path.join(d,fn)).convert("RGB")); H,W,_=rgb.shape; N=H*W
    yc=np.zeros((3*N,),dtype=np.int16)
    Lb.ycocg_fwd(rgb.tobytes(),H,W,yc.ctypes.data_as(ctypes.POINTER(c_i16)))
    res=np.zeros_like(yc); ctx=np.zeros((3*N,),dtype=np.uint8); hist=np.zeros((3*9*513,),dtype=np.int64)
    Lb.med_res_ctx9(yc.ctypes.data_as(ctypes.POINTER(c_i16)),H,W,res.ctypes.data_as(ctypes.POINTER(c_i16)),ctx.ctypes.data_as(ctypes.POINTER(c_u8)),hist.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)))
    lib.rans_norm.argtypes=[ctypes.POINTER(ctypes.c_int64),ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_uint16),ctypes.POINTER(ctypes.c_int)]
    lib.rans_norm.restype=None
    lib.rans_encode.restype=ctypes.c_size_t
    lib.rans_decode.restype=ctypes.c_int
    lib.rans_encode.argtypes=[ctypes.POINTER(c_i16),ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_uint16),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(c_u8)]
    lib.rans_slots.argtypes=[ctypes.POINTER(ctypes.c_uint16),ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.POINTER(ctypes.c_int)]
    lib.rans_decode.argtypes=[ctypes.POINTER(c_u8),ctypes.c_size_t,ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_uint16),ctypes.POINTER(ctypes.c_int),ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(c_i16)]
    out=bytearray(); tot_t=0
    planes=[]
    for k in range(3):
        h=hist[k*9*513:(k+1)*9*513].reshape(9,513).sum(axis=0)
        syms=np.where(h>0)[0].tolist(); A=len(syms)
        harr=np.ascontiguousarray(h,dtype=np.int64)
        symarr=(ctypes.c_int*A)(*syms)
        fq=np.zeros((A,),dtype=np.uint16); cu=np.zeros((A,),dtype=np.int32)
        lib.rans_norm(harr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),A,symarr,fq.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),cu.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
        lut=np.full((513,),-1,dtype=np.int32)
        for i,s in enumerate(syms): lut[s]=i
        r=res[k*N:(k+1)*N].copy()
        cap=N*3+16
        buf=(c_u8*cap)()
        t0=time.perf_counter()
        n=lib.rans_encode(r.ctypes.data_as(ctypes.POINTER(c_i16)),N,lut.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),fq.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),cu.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),buf)
        tot_t+=time.perf_counter()-t0
        payload=bytes(buf[:n])
        # decode check
        slots=np.zeros((16384,),dtype=np.int32)
        lib.rans_slots(fq.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),cu.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),A,slots.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
        symarr2=(ctypes.c_int*A)(*syms)
        dec=np.zeros((N,),dtype=np.int16)
        bb=(c_u8*len(payload))(*payload)
        rc=lib.rans_decode(bb,n,N,symarr2,fq.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),cu.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),A,slots.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),dec.ctypes.data_as(ctypes.POINTER(c_i16)))
        assert rc==0 and np.array_equal(dec,r), (fn,k,rc)
        out+=A.to_bytes(2,'little')
        for i,s in enumerate(syms): out+=(s-1024&0xFFFF).to_bytes(2,'little')+int(fq[i]).to_bytes(2,'little')
        out+=int(n).to_bytes(4,'little')+payload
        planes.append((A,n))
    bpp=len(out)*8/(N*3)
    print(f"{fn} RANS-perCH bpp={bpp:.2f} bytes={len(out)} enctime={tot_t*1000:.0f}ms tables={planes} ROUNDTRIP=PASS",flush=True)
if __name__=="__main__":
    for fn in sys.argv[1:] or ["kodim23.png"]:
        test(fn)
