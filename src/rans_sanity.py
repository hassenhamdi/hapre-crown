import ctypes, numpy as np
lib=ctypes.CDLL("/tmp/opencode/autocompress/csrc/libhapre.so")
lib.rans_norm.argtypes=[ctypes.POINTER(ctypes.c_int64),ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_uint16),ctypes.POINTER(ctypes.c_int)]
lib.rans_norm.restype=None
lib.rans_encode.argtypes=[ctypes.POINTER(ctypes.c_int16),ctypes.c_int,ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_uint16),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_uint8)]
lib.rans_encode.restype=ctypes.c_size_t
rng=np.random.default_rng(0)
# Laplacian-ish source over alphabet 0..20
p=np.exp(-0.5*np.arange(21)); p/=p.sum()
N=200000
draw=rng.choice(21,size=N,p=p).astype(np.int16)
hist=np.zeros(513,np.int64)
for v in draw: hist[v+256]+=1
syms=np.where(hist>0)[0].tolist(); A=len(syms)
harr=np.ascontiguousarray(hist,dtype=np.int64)
symarr=(ctypes.c_int*A)(*syms)
fq=np.zeros((A,),np.uint16); cu=np.zeros((A,),np.int32)
lib.rans_norm(harr.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),A,symarr,fq.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),cu.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
lut=np.full((513,),-1,np.int32)
for i,s in enumerate(syms): lut[s]=i
cap=N*2+16; buf=(ctypes.c_uint8*cap)()
n=lib.rans_encode(draw.ctypes.data_as(ctypes.POINTER(ctypes.c_int16)),N,lut.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),fq.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),cu.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),buf)
ent=-(p*np.log2(p)).sum()
print(f"rans bits/sym={n*8/N:.3f} entropy={ent:.3f} overhead={(n*8/N-ent):.3f}")