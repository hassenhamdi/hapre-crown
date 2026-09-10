"""Bit-exact Rice+run stream: flag0=zero-run(gamma L), flag1=Rice nonzero. Round-trip proof on 64x64 crop."""
import numpy as np
from PIL import Image
import sys
sys.path.insert(0,"/tmp/opencode/autocompress/experiments/stage3_method")
from codec import rgb_to_ycocg_r, ycocg_r_to_rgb, med, paeth, gap

class BW:
    def __init__(self): self.b=bytearray(); self.acc=0; self.n=0
    def bit(self,v):
        self.acc=(self.acc<<1)|v; self.n+=1
        if self.n==8: self.b.append(self.acc); self.acc=0; self.n=0
    def gamma(self,L):  # L>=1
        b=L.bit_length()-1
        for _ in range(b): self.bit(0)
        for i in range(b,-1,-1): self.bit((L>>i)&1)
    def unary(self,q):
        for _ in range(q): self.bit(1)
        self.bit(0)
    def bits(self,v,k):
        for i in range(k-1,-1,-1): self.bit((v>>i)&1)
    def flush(self):
        if self.n: self.b.append(self.acc<<(8-self.n))
        return bytes(self.b)
class BR:
    def __init__(self,buf): self.buf=buf; self.p=0
    def bit(self):
        byte=self.buf[self.p//8]; b=(byte>>(7-(self.p%8)))&1; self.p+=1; return b
    def gamma(self):
        b=0
        while self.bit()==0: b+=1
        v=1
        for _ in range(b): v=(v<<1)|self.bit()
        return v
    def unary(self):
        q=0
        while self.bit()==1: q+=1
        return q
    def bits(self,k):
        v=0
        for _ in range(k): v=(v<<1)|self.bit()
        return v

def residuals_medonly(yc):
    H,W,_=yc.shape
    P=np.pad(yc,((1,0),(1,0),(0,0)),mode='edge').astype(np.int16)
    res=np.zeros_like(yc)
    for i in range(H):
        for j in range(W):
            for ch in range(3):
                a=int(P[i+1,j,ch]); b=int(P[i,j+1,ch]); c=int(P[i,j,ch])
                m=med(a,b,c)
                res[i,j,ch]=int(P[i+1,j+1,ch])-m
    return res

rgb=np.array(Image.open("/tmp/opencode/autocompress/experiments/real_photos/kodim01.png").convert("RGB"))[:64,:64]
yc=rgb_to_ycocg_r(rgb)
res=residuals_medonly(yc)
# per-channel k from nonzeros
import math
streams={}
for ch,nm in enumerate(["Y","Co","Cg"]):
    flat=res[:,:,ch].reshape(-1)
    nz=flat[flat!=0]
    m=float(np.mean(np.abs(nz.astype(float))))+1e-9 if len(nz) else 1
    k=int(max(0,min(8,math.floor(math.log2(m)))))
    w=BW()
    w.bits(k,4)  # header k
    i=0; N=len(flat)
    while i<N:
        if flat[i]==0:
            j=i
            while j<N and flat[j]==0: j+=1
            w.bit(0); w.gamma(j-i); i=j
        else:
            v=int(flat[i]); u=2*v if v>=0 else -2*v-1
            w.bit(1); w.unary(u>>k); w.bits(u&((1<<k)-1) if k else 0,k); i+=1
    streams[nm]=(w.flush(),k)
    print(nm, "bytes=",len(streams[nm][0]),"k=",k)
# decode + verify
ok=True
for ch,nm in enumerate(["Y","Co","Cg"]):
    buf,k=streams[nm]; r=BR(buf); k2=r.bits(4); assert k2==k
    out=np.zeros(64*64,dtype=int); i=0
    while i<64*64:
        f=r.bit()
        if f==0: L=r.gamma(); out[i:i+L]=0; i+=L
        else:
            q=r.unary(); rem=r.bits(k) if k else 0; u=(q<<k)|rem
            out[i]=u//2 if u%2==0 else -(u+1)//2; i+=1
    if not np.array_equal(out,res[:,:,ch].reshape(-1)): ok=False; print(nm,"MISMATCH")
print("ROUNDTRIP_EXACT:", "PASS" if ok else "FAIL", "total_bytes=",sum(len(v[0]) for v in streams.values()))